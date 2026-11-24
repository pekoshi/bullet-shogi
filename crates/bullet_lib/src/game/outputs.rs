use std::{
    cell::RefCell,
    path::Path,
    sync::{
        OnceLock,
        atomic::{AtomicBool, AtomicI32, Ordering},
    },
};

use bulletformat::{ChessBoard, chess::MarlinFormat};

use crate::shogi::{
    BonaPiece, Color, PackedSfenValue, Piece,
    bona_piece::FE_OLD_END,
    types::{BOARD_PIECE_TYPES, HAND_PIECE_TYPES},
};
pub trait OutputBuckets<T>: Send + Sync + Copy + Default + 'static {
    const BUCKETS: usize;

    fn bucket(&self, pos: &T) -> u8;
}

// S2-B simplified: progress fractional alpha based stochastic bucket selection.
// Enable: `BUCKET_SOFT_PROB=<0.0-1.0>` env var.
// Effect: bucket = lo with prob (1 - soft_prob * alpha), else lo+1 (clamped 0..7).
// soft_prob=0.0 (default) → original floor behaviour, no jitter.
// soft_prob=1.0 → expected bucket = lo + alpha (true soft mixture in expectation).

/// Local xorshift PRNG (inline copy of `crate::value::loader::rng::SimpleRand`).
struct BucketRng(u64);

impl BucketRng {
    fn with_seed() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Guaranteed increasing.")
            .as_micros() as u64
            & 0xFFFF_FFFF;
        Self(seed.max(1))
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn read_bucket_soft_prob() -> f32 {
    static CACHE: OnceLock<f32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("BUCKET_SOFT_PROB").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0).clamp(0.0, 1.0)
    })
}

thread_local! {
    static BUCKET_RNG: RefCell<BucketRng> = RefCell::new(BucketRng::with_seed());
}

/// Stochastic bucket selection with fractional alpha (= S2-B simplified soft mixture).
///
/// `raw` is the unclamped floor(progress * NUM_BUCKETS), `alpha` is the fractional part.
/// Returns `raw.clamp(0, max_bucket)` with probability (1 - soft_prob * alpha),
/// otherwise (raw + 1).clamp(0, max_bucket).
pub(crate) fn apply_soft_bucket(raw: i32, alpha: f32, max_bucket: i32) -> u8 {
    let soft_prob = read_bucket_soft_prob();
    if soft_prob <= 0.0 || alpha <= 0.0 {
        return raw.clamp(0, max_bucket) as u8;
    }
    let r = BUCKET_RNG.with(|rng| {
        let v = rng.borrow_mut().next();
        (v as f32 / u64::MAX as f32).clamp(0.0, 1.0)
    });
    if r < soft_prob * alpha {
        (raw + 1).clamp(0, max_bucket) as u8
    } else {
        raw.clamp(0, max_bucket) as u8
    }
}

#[deprecated(note = "You do not need to specify this anymore, it is the default!")]
#[derive(Clone, Copy, Default)]
pub struct Single;

#[allow(deprecated)]
impl<T: 'static> OutputBuckets<T> for Single {
    const BUCKETS: usize = 1;

    fn bucket(&self, _: &T) -> u8 {
        0
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount<const N: usize>;
impl<const N: usize> OutputBuckets<ChessBoard> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

impl<const N: usize> OutputBuckets<MarlinFormat> for MaterialCount<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &MarlinFormat) -> u8 {
        let divisor = 32usize.div_ceil(N);
        (pos.occ().count_ones() as u8 - 2) / divisor as u8
    }
}

/// 将棋 LayerStacks 用出力バケット
///
/// 両玉の相対段に基づいて N バケットに分類する。
/// rshogi の `compute_bucket_index` / `compute_king_ranks` と同一ロジック。
///
/// 標準では N=9 (3×3 マトリクス):
/// ```text
///        e_rank 0-2  e_rank 3-5  e_rank 6-8
/// f_rank 0-2:    0        1            2
/// f_rank 3-5:    3        4            5
/// f_rank 6-8:    6        7            8
/// ```
#[derive(Clone, Copy, Default)]
pub struct ShogiKingRankBucket<const N: usize>;

impl<const N: usize> OutputBuckets<PackedSfenValue> for ShogiKingRankBucket<N> {
    const BUCKETS: usize = N;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let board = pos.decode();

        let side_to_move = board.side_to_move;
        let f_king = board.king_square(side_to_move);
        let e_king = board.king_square(side_to_move.opponent());

        // 味方玉の段（味方から見た相対段）
        let f_rank = match side_to_move {
            crate::shogi::Color::Black => f_king.rank() as usize,
            crate::shogi::Color::White => 8 - f_king.rank() as usize,
        };

        // 相手玉の段（相手から見た相対段）
        let e_rank = match side_to_move {
            crate::shogi::Color::Black => 8 - e_king.rank() as usize,
            crate::shogi::Color::White => e_king.rank() as usize,
        };

        const F_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
        const E_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];

        let bucket = F_TO_INDEX[f_rank.min(8)] + E_TO_INDEX[e_rank.min(8)];
        bucket.min(N - 1) as u8
    }
}

/// Default boundaries for shogi ply-based 9-bucket split.
///
/// bucket0: <=30, bucket1: <=44, ..., bucket7: <=138, bucket8: >=139
pub const SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS: [u16; 8] = [30, 44, 58, 72, 86, 100, 116, 138];

/// Number of features used by progress-based bucket model.
pub const SHOGI_PROGRESS8_NUM_FEATURES: usize = 6;

/// Feature order for progress-based bucket model (coeff_v1).
pub const SHOGI_PROGRESS8_FEATURE_ORDER: [&str; SHOGI_PROGRESS8_NUM_FEATURES] = [
    "x_board_non_king",
    "x_hand_total",
    "x_major_board",
    "x_promoted_board",
    "x_stm_king_rank_rel",
    "x_ntm_king_rank_rel",
];

/// Number of buckets for progress8.
pub const SHOGI_PROGRESS8_NUM_BUCKETS: usize = 8;

/// Number of KP-absolute weights: `81 * FE_OLD_END`.
pub const SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS: usize = 81 * FE_OLD_END;

// ===== LayerStack「進行度4区分 × 持駒量2区分」モード (P4H2) =====
//
// 有効時、`ShogiProgressKPAbs::bucket` は以下で LayerStack index を返す:
//   stack = progress_bucket4 * 2 + hand_bucket2                       (0 <= stack < 8)
//   progress_bucket4 = progress_bucket8 / 2                           (0..3)
//   hand_bucket2     = (weighted_hand_total >= HAND_VALUE_THRESHOLDS[progress_bucket4]) ? 1 : 0
//
// ★案B(進行度別閾値): 進行度と持駒量は相関するため、進行度ごとに中央値付近で分割する
//   閾値配列を採用(教師分布 novel/base 1M サンプルで確定, 2026-07-18)。
// ★推論側 (C++ p2nnue.cpp: P2_HAND_PIECE_WEIGHT / P2_HAND_VALUE_THRESHOLD[4] /
//   get_layer_stack_index) と持駒重み・閾値配列・比較演算子(>=)・stack 番号を完全一致させること。

/// 持駒量2区分の駒重み。index は `Hand::counts` と同順
/// (= [歩, 香, 桂, 銀, 金, 角, 飛] = HAND_PIECE_TYPES 順)。
/// ★C++ `P2_HAND_PIECE_WEIGHT`(PieceType index 順)と同じ駒→重み対応。
pub const HAND_PIECE_WEIGHTS: [i32; 7] = [1, 2, 2, 3, 4, 6, 7];

/// 持駒量2区分の進行度別閾値(index = progress_bucket4 = 0..3)。
/// `weighted_hand_total >= HAND_VALUE_THRESHOLDS[prog4]` で hand_bucket=1(多)。
/// ★C++ `P2_HAND_VALUE_THRESHOLD[4]` と一致必須(唯一の集約点)。arch marker
///   "LSMODE=P4H2:6-12-20-24," にも反映され、閾値配列違いの net は別ネット扱いとなる。
pub const HAND_VALUE_THRESHOLDS: [i32; 4] = [6, 12, 20, 24];

static LAYERSTACK_P4H2: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// LayerStack を P4H2(進行度4区分×持駒量2区分)モードに切り替える。
///
/// プロセス全体で共有。訓練開始前に一度だけ設定する
/// (`ShogiProgressKPAbs::bucket` と全 input feature の bucket 算出がこれに従う)。
pub fn set_layerstack_p4h2(enabled: bool) {
    LAYERSTACK_P4H2.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

/// P4H2 モードが有効かを返す。
pub fn layerstack_p4h2_enabled() -> bool {
    LAYERSTACK_P4H2.load(std::sync::atomic::Ordering::Relaxed)
}

/// `Hand::counts`(= [歩,香,桂,銀,金,角,飛])から重み付き持駒量を計算する純粋関数。
///
/// `total = Σ_i (black.counts[i] + white.counts[i]) * HAND_PIECE_WEIGHTS[i]`。
pub fn weighted_hand_total_from_counts(black: &[u8; 7], white: &[u8; 7]) -> i32 {
    let mut total = 0i32;
    for (i, &w) in HAND_PIECE_WEIGHTS.iter().enumerate() {
        total += (black[i] as i32 + white[i] as i32) * w;
    }
    total
}

/// 両者合計の重み付き持駒量。`total = weighted(BLACK) + weighted(WHITE)`。
///
/// ★C++ `weighted_hand_total` と完全一致。
pub fn weighted_hand_total(pos: &PackedSfenValue) -> i32 {
    let board = pos.decode();
    weighted_hand_total_from_counts(&board.black_hand.counts, &board.white_hand.counts)
}

/// 持駒量2区分。進行度別閾値以上=1(多)、未満=0(少)。★C++ `hand_bucket2` と一致。
///
/// `progress_bucket4` は 0..=3。範囲外は端にクランプ(0<=prog4<4 は呼出側で保証)。
pub fn hand_bucket2(pos: &PackedSfenValue, progress_bucket4: usize) -> u8 {
    let thr = HAND_VALUE_THRESHOLDS[progress_bucket4.min(3)];
    if weighted_hand_total(pos) >= thr { 1 } else { 0 }
}

static SHOGI_PROGRESS_KP_ABS_WEIGHTS: OnceLock<Box<[f32]>> = OnceLock::new();
static SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS: [f32; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS] =
    [0.0; SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS];

/// Progress-based 8 bucket assignment (logistic regression).
///
/// `p = sigmoid(bias + Σ(w_i * ((x_i - mean_i) / std_i)))`
/// `bucket = min(7, floor(p * 8.0))`
#[derive(Clone, Copy)]
pub struct ShogiProgressBucket8 {
    pub mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub std: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
    pub bias: f32,
    pub z_clip: [f32; 2],
}

impl ShogiProgressBucket8 {
    pub const fn new(
        mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        std: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES],
        bias: f32,
        z_clip: [f32; 2],
    ) -> Self {
        Self { mean, std, weights, bias, z_clip }
    }

    /// Extract raw progress-model features in coeff_v1 order.
    pub fn extract_features(pos: &PackedSfenValue) -> [f32; SHOGI_PROGRESS8_NUM_FEATURES] {
        let board = pos.decode();

        let board_non_king = board
            .board
            .iter()
            .filter(|p| p.piece_type != crate::shogi::PieceType::None && p.piece_type != crate::shogi::PieceType::King)
            .count() as f32;

        let hand_total =
            board.black_hand.counts.iter().chain(board.white_hand.counts.iter()).map(|&v| v as f32).sum::<f32>();

        let major_board = board
            .board
            .iter()
            .filter(|p| {
                matches!(
                    p.piece_type,
                    crate::shogi::PieceType::Bishop
                        | crate::shogi::PieceType::Rook
                        | crate::shogi::PieceType::Horse
                        | crate::shogi::PieceType::Dragon
                )
            })
            .count() as f32;

        let promoted_board = board.board.iter().filter(|p| p.piece_type.is_promoted()).count() as f32;

        let stm = board.side_to_move;
        let f_king = board.king_square(stm);
        let e_king = board.king_square(stm.opponent());

        let stm_king_rank_rel = match stm {
            crate::shogi::Color::Black => f_king.rank() as f32,
            crate::shogi::Color::White => (8 - f_king.rank()) as f32,
        };

        let ntm_king_rank_rel = match stm {
            crate::shogi::Color::Black => (8 - e_king.rank()) as f32,
            crate::shogi::Color::White => e_king.rank() as f32,
        };

        [board_non_king, hand_total, major_board, promoted_board, stm_king_rank_rel, ntm_king_rank_rel]
    }

    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let x = Self::extract_features(pos);

        let mut z = self.bias;
        for (i, &x_i) in x.iter().enumerate() {
            let std = if self.std[i] > 0.0 { self.std[i] } else { 1.0 };
            let x_norm = (x_i - self.mean[i]) / std;
            z += self.weights[i] * x_norm;
        }

        let z_min = self.z_clip[0].min(self.z_clip[1]);
        let z_max = self.z_clip[0].max(self.z_clip[1]);
        let z_clamped = z.clamp(z_min, z_max);
        let p = 1.0 / (1.0 + (-z_clamped).exp());
        p.clamp(0.0, 1.0)
    }
}

impl Default for ShogiProgressBucket8 {
    fn default() -> Self {
        // docs/progress-bucket-coeff-script-spec-v1.md の JSON 例を既定値として採用。
        Self {
            mean: [30.12, 8.45, 2.18, 1.63, 6.71, 6.24],
            std: [3.77, 4.02, 0.66, 1.40, 1.31, 1.27],
            weights: [-0.81, 0.56, -0.32, 0.48, 0.11, -0.09],
            bias: -0.15,
            z_clip: [-8.0, 8.0],
        }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressBucket8 {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as u8
    }
}

/// Progress-based 8 bucket assignment using YaneuraOu/tanuki- style KP-absolute features.
///
/// Weights are process-global so this type stays `Copy` and can be embedded in `OutputBuckets`.
#[derive(Clone, Copy, Default)]
pub struct ShogiProgressKPAbs;

impl ShogiProgressKPAbs {
    fn weights() -> &'static [f32] {
        SHOGI_PROGRESS_KP_ABS_WEIGHTS.get().map_or(&SHOGI_PROGRESS_KP_ABS_ZERO_WEIGHTS, |weights| weights.as_ref())
    }

    /// Enumerates all active KP-absolute feature indices for the position.
    ///
    /// This is the exact feature expansion used by `progress8kpabs`.
    pub fn for_each_active_index(pos: &PackedSfenValue, mut f: impl FnMut(usize)) {
        let board = pos.decode();
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            return;
        }

        let sq_bk = board.black_king_sq.index();
        let sq_wk = board.white_king_sq.inverse().index();

        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);

                    let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }

        for owner in [Color::Black, Color::White] {
            let hand = if owner == Color::Black { board.black_hand } else { board.white_hand };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for c in 1..=count {
                    let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                    if bp_b != BonaPiece::ZERO {
                        f(sq_bk * FE_OLD_END + bp_b.value() as usize);
                    }

                    let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                    if bp_w != BonaPiece::ZERO {
                        f(sq_wk * FE_OLD_END + bp_w.value() as usize);
                    }
                }
            }
        }
    }

    /// Collects all active KP-absolute feature indices into `out`.
    pub fn collect_active_indices(pos: &PackedSfenValue, out: &mut Vec<usize>) {
        out.clear();
        Self::for_each_active_index(pos, |idx| out.push(idx));
    }

    /// Loads KP-absolute weights from a YaneuraOu-compatible `progress.bin`.
    ///
    /// Only one KP-absolute model can be loaded per process.
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * std::mem::size_of::<f64>();
        if bytes.len() != expected {
            return Err(format!("progress.bin size mismatch: got {} bytes, expected {}", bytes.len(), expected));
        }

        let weights: Vec<f32> = bytes
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("chunk size is checked")) as f32)
            .collect();

        SHOGI_PROGRESS_KP_ABS_WEIGHTS
            .set(weights.into_boxed_slice())
            .map_err(|_| "KP-absolute progress weights are already loaded in this process".to_string())?;

        Ok(Self)
    }

    /// Estimates progress in `0.0..=1.0`.
    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let weights = Self::weights();
        let mut sum = 0.0f32;
        Self::for_each_active_index(pos, |idx| sum += weights[idx]);

        let p = 1.0 / (1.0 + (-sum).exp());
        p.clamp(0.0, 1.0)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressKPAbs {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let p = self.progress(pos);
        let scaled = p * 8.0;
        let raw = scaled.floor() as i32;
        let alpha = scaled - scaled.floor();
        let bucket8 = apply_soft_bucket(raw, alpha, 7); // 0..7
        if layerstack_p4h2_enabled() {
            // P4H2: 進行度8bucket を /2 で4区分へ圧縮 + 進行度別閾値の持駒量2区分。
            // ★C++ get_layer_stack_index と完全一致。
            let progress_bucket4 = (bucket8 / 2) as usize; // 0..3
            let hand2 = hand_bucket2(pos, progress_bucket4) as usize;
            (progress_bucket4 * 2 + hand2) as u8
        } else {
            bucket8
        }
    }
}

// ===== LayerStack「進行度8区分 × 持駒量2区分」モード (P8H2) =====
//
// P4H2(progress4×hand2=8）を進行度8区分に拡張した 16 bucket モード。
//   stack = progress_bucket8 * 2 + hand_bucket2                       (0 <= stack < 16)
//   progress_bucket8 = progress8kpabs と同一の進行度8区分                 (0..7)
//   hand_bucket2     = P4H2 と同じ持駒量2区分。閾値 index は
//                      progress_bucket4 = progress_bucket8 / 2 (0..3) で
//                      既存 HAND_VALUE_THRESHOLDS[4] をそのまま再利用する。
//
// P4H2 と異なり進行度を /2 で圧縮しないため出力 bucket 数が 16 になる。したがって
// LayerStack を 16 head で構築する専用 net が必要で、`ShogiProgressKPAbs`(8 bucket)とは
// 別型として扱う(`const BUCKETS = 16`)。arch marker は "LSMODE=P8H2:<t0>-<t1>-<t2>-<t3>,"。
// ★推論側 (C++ p2nnue.cpp: -DP2_LSMODE_P8H2, get_layer_stack_index, LSMODE=P8H2 marker) と
//   進行度区分・持駒重み・閾値配列・比較演算子(>=)・`*2 + hand` の順序を完全一致させること。

// ===== 案 B (Regime LayerStack)。 2026-10-07。 =====
//   事前登録: runs/prereg/PREREG-20261007-REGIME-B.md
//   16 bucket の第 2 軸を「持駒量」から「regime (玉周りが攻められているか)」に差し替える。
//     regime = (bucket8 >= REGIME_MIN_PB) && (V1 >= REGIME_V1_TH)
//     V1 = 両玉 8 近傍の空升への敵の利き数の和 (= C++ の P2Volm::value)。
//          **step + ray** で数える (2026-10-07 に engine 400/400 一致で確定)。
//   🚨REGIME_MIN_PB で序盤を割らない: 実測で ply 1-40 の regime 1 は 0.8% しか無く、
//     全 bucket を割ると前半の regime 1 側に勾配が来ない (★threat_group_starved_b01)。
//   🚨新しい型を作らず既存の 16 bucket 型の中で切り替える (generic の二重 monomorphize を避ける)。
static REGIME_ENABLED: AtomicBool = AtomicBool::new(false);
static REGIME_V1_TH: AtomicI32 = AtomicI32::new(3);
static REGIME_MIN_PB: AtomicI32 = AtomicI32::new(4);

/// 案 B の regime モードを有効にする (CLI から 1 度だけ呼ぶ)。
pub fn set_regime_mode(enabled: bool, v1_th: i32, min_pb: i32) {
    REGIME_V1_TH.store(v1_th, Ordering::Relaxed);
    REGIME_MIN_PB.store(min_pb, Ordering::Relaxed);
    REGIME_ENABLED.store(enabled, Ordering::Relaxed);
}

/// regime モードが有効なら (v1_th, min_pb) を返す。
pub fn regime_mode() -> Option<(i32, i32)> {
    if REGIME_ENABLED.load(Ordering::Relaxed) {
        Some((REGIME_V1_TH.load(Ordering::Relaxed), REGIME_MIN_PB.load(Ordering::Relaxed)))
    } else {
        None
    }
}

/// P8H2(進行度8区分 × 持駒量2区分)の出力 bucket 数。
pub const SHOGI_PROGRESS8_HAND2_NUM_BUCKETS: usize = 16;

/// LayerStack を「進行度8区分 × 持駒量2区分 = 16」で選ぶ OutputBuckets 実装。
///
/// `ShogiProgressKPAbs` と同じ progress.bin(共有 OnceLock)を使い、bucket 算出のみ
/// 16 区分へ拡張する。`ShogiProgressKPAbs` の runtime グローバル(P4H2)には依存しない
/// 自己完結型。
#[derive(Clone, Copy, Default)]
pub struct ShogiProgress8Hand2KPAbs;

impl ShogiProgress8Hand2KPAbs {
    /// progress.bin をロードする(`ShogiProgressKPAbs::load_from_bin` と同一の共有
    /// OnceLock を使うため、プロセス内で1度だけ呼べる)。
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        ShogiProgressKPAbs::load_from_bin(path)?;
        Ok(Self)
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgress8Hand2KPAbs {
    const BUCKETS: usize = SHOGI_PROGRESS8_HAND2_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        // 進行度8区分は progress8kpabs と bit-identical に算出する。
        let p = ShogiProgressKPAbs.progress(pos);
        let scaled = p * 8.0;
        let raw = scaled.floor() as i32;
        let alpha = scaled - scaled.floor();
        let bucket8 = apply_soft_bucket(raw, alpha, 7); // 0..7
        // 持駒量2区分の閾値は P4H2 と同じく progress_bucket4 (= bucket8/2, 0..3) で引く。
        // ★C++ get_layer_stack_index(P8H2) と完全一致: stack = bucket8*2 + hand2。
        // 案 B (2026-10-07): regime モードなら第 2 軸を regime にする。
        //   🚨&& の短絡で、 序盤 (bucket8 < min_pb) では V1 を**計算しない**。
        if let Some((v1_th, min_pb)) = regime_mode() {
            let r = if (bucket8 as i32) >= min_pb {
                let board = crate::shogi::ShogiBoard::from_packed_sfen(pos);
                let v1 = crate::game::inputs::board_effect_v1(&board);
                if v1 >= v1_th { 1usize } else { 0usize }
            } else {
                0usize
            };
            return (bucket8 as usize * 2 + r) as u8; // 0..15
        }
        let progress_bucket4 = (bucket8 / 2) as usize; // 0..3(閾値 index のみ)
        let hand2 = hand_bucket2(pos, progress_bucket4) as usize;
        (bucket8 as usize * 2 + hand2) as u8 // 0..15
    }
}

// ===== OPNARCH 案 A (2026-11-14): king bucket k9k9z (81) と k9k9z × 進行度 3 (243) =====
//   設計書: C:/project/p2shogi/runs/prereg/DESIGN-20261114-OPNARCH-FABLE.md 2.2 章 (YO の k9k9z と同じ)。
//   升番号は YO と同じ sq = file*9 + rank (file 0 = 1 筋 / rank 0 = 1 段)。 Inv(sq) = 80 - sq。
//     own(sq, 持ち主) = 持ち主が先手 ? sq : Inv(sq)       (その玉の持ち主から見た座標。 自陣が 7〜9 段目)
//     z9(sq) = r<3 → 0 / r<6 → 1 / r==6 → 2 / それ以外 → 3 + (r-7)*3 + f/3      (0..8)
//     kz     = z9(own(手番側の玉)) * 9 + z9(own(非手番側の玉))                   (0..80)
//     head   = kz * 3 + pb  (pb = progress8kpabs の bucket。 0..2 だけを使う)    (0..242)
//   添字の順は YO の LayerStack と同じ「玉 → 進行度」(idx = king; idx = idx*progress数 + progress)。
//   🚨python 側 (runs/opnarch/kb_density.py の norm / z9、 opn_static2.py) と同じ添字であること。
//   🚨243 は序盤帯 (進行度 bucket 0-2) 専用。 学習器は `--bucket-filter` (上限 2 以下) を必須にしている。
//     範囲外 (pb >= 3) の局面は pb = 2 の head に寄せる (学習器では filter で来ない / 推論側の目安)。

/// k9k9z の 1 玉あたりの区分数。
pub const OPN_K9Z_ZONES: usize = 9;
/// k9k9z の bucket 数 (両玉 9 × 9)。
pub const OPN_K9Z_NUM_BUCKETS: usize = OPN_K9Z_ZONES * OPN_K9Z_ZONES;

/// 玉の升 (その玉の持ち主から見た座標) を 9 区分にする (YO の k9k9z と同じ)。
#[inline]
pub fn opn_z9(own_sq: crate::shogi::Square) -> usize {
    let f = own_sq.file() as usize;
    let r = own_sq.rank() as usize;
    if r < 3 {
        0
    } else if r < 6 {
        1
    } else if r == 6 {
        2
    } else {
        3 + (r - 7) * 3 + f / 3
    }
}

/// k9k9z の bucket (0..80) を手番と両玉の升 (先手視点の絶対座標) から計算する。
///
/// 玉の升が無効 (81 以上) なら 0 を返す (学習データには現れない前提。 panic させない)。
#[inline]
pub fn opn_kz(side_to_move: Color, black_king_sq: crate::shogi::Square, white_king_sq: crate::shogi::Square) -> usize {
    if !black_king_sq.is_valid() || !white_king_sq.is_valid() {
        return 0;
    }
    // その玉の持ち主から見た座標: 先手玉はそのまま、 後手玉は 180 度回転。
    let own_black = opn_z9(black_king_sq);
    let own_white = opn_z9(white_king_sq.inverse());
    match side_to_move {
        Color::Black => own_black * OPN_K9Z_ZONES + own_white,
        Color::White => own_white * OPN_K9Z_ZONES + own_black,
    }
}

/// progress8kpabs と bit 単位で同じ進行度 8 区分 (0..7)。 `--bucket-filter` の鍵にも使う。
///
/// `ShogiProgressKPAbs::bucket` の P4H2 を通さない値 (= `progress8kpabs` の bucket)。
#[inline]
pub fn opn_progress_bucket8(pos: &PackedSfenValue) -> u8 {
    let p = ShogiProgressKPAbs.progress(pos);
    let scaled = p * 8.0;
    let raw = scaled.floor() as i32;
    let alpha = scaled - scaled.floor();
    apply_soft_bucket(raw, alpha, 7)
}

/// LayerStack を「k9k9z (× 進行度 PROG 区分)」で選ぶ OutputBuckets 実装 (OPN 専用)。
///
/// - `PROG = 1`: k9k9z のみ = 81 bucket (`--bucket-mode opnk9z`、 arch marker `LSMODE=K9Z,`)。
/// - `PROG = 3`: k9k9z × 進行度 3 = 243 bucket (`--bucket-mode opnk9zp3`、 arch marker `LSMODE=K9ZP3,`)。
///
/// 進行度は `ShogiProgressKPAbs` と同じ progress.bin (共有 OnceLock) を使う。
#[derive(Clone, Copy, Default)]
pub struct ShogiOpnK9z<const PROG: usize>;

impl<const PROG: usize> ShogiOpnK9z<PROG> {
    /// progress.bin をロードする (`ShogiProgressKPAbs::load_from_bin` と同じ共有 OnceLock)。
    pub fn load_from_bin(path: &Path) -> Result<Self, String> {
        ShogiProgressKPAbs::load_from_bin(path)?;
        Ok(Self)
    }

    /// head 番号 = kz * PROG + min(pb8, PROG - 1)。 PROG = 1 なら kz。
    #[inline]
    pub fn head(kz: usize, progress_bucket8: u8) -> u8 {
        debug_assert!(PROG == 1 || PROG == 3, "ShogiOpnK9z は PROG = 1 / 3 のみ");
        (kz * PROG + (progress_bucket8 as usize).min(PROG - 1)) as u8
    }
}

impl<const PROG: usize> OutputBuckets<PackedSfenValue> for ShogiOpnK9z<PROG> {
    const BUCKETS: usize = OPN_K9Z_NUM_BUCKETS * PROG;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let board = pos.decode();
        let kz = opn_kz(board.side_to_move, board.black_king_sq, board.white_king_sq);
        let pb = if PROG == 1 { 0 } else { opn_progress_bucket8(pos) };
        Self::head(kz, pb)
    }
}

/// Number of features used by Gikou-lite progress model.
pub const SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES: usize = 34;

/// Feature order for Gikou-lite progress model (coeff_v2).
pub const SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER: [&str; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] = [
    "x_board_non_king",
    "x_hand_total",
    "x_major_board",
    "x_promoted_board",
    "x_stm_king_rank_rel",
    "x_ntm_king_rank_rel",
    "x_stm_all_to_own_king_d1",
    "x_stm_all_to_own_king_d2",
    "x_stm_all_to_own_king_d3p",
    "x_stm_all_to_opp_king_d1",
    "x_stm_all_to_opp_king_d2",
    "x_stm_all_to_opp_king_d3p",
    "x_ntm_all_to_own_king_d1",
    "x_ntm_all_to_own_king_d2",
    "x_ntm_all_to_own_king_d3p",
    "x_ntm_all_to_opp_king_d1",
    "x_ntm_all_to_opp_king_d2",
    "x_ntm_all_to_opp_king_d3p",
    "x_stm_major_to_own_king_d1",
    "x_stm_major_to_own_king_d2",
    "x_stm_major_to_own_king_d3p",
    "x_stm_major_to_opp_king_d1",
    "x_stm_major_to_opp_king_d2",
    "x_stm_major_to_opp_king_d3p",
    "x_ntm_major_to_own_king_d1",
    "x_ntm_major_to_own_king_d2",
    "x_ntm_major_to_own_king_d3p",
    "x_ntm_major_to_opp_king_d1",
    "x_ntm_major_to_opp_king_d2",
    "x_ntm_major_to_opp_king_d3p",
    "x_stm_hand_total",
    "x_ntm_hand_total",
    "x_stm_hand_major",
    "x_ntm_hand_major",
];

#[inline]
fn chebyshev_distance(a: crate::shogi::Square, b: crate::shogi::Square) -> u8 {
    let df = a.file().abs_diff(b.file());
    let dr = a.rank().abs_diff(b.rank());
    df.max(dr)
}

#[inline]
fn distance_bin(d: u8) -> usize {
    if d <= 1 {
        0
    } else if d == 2 {
        1
    } else {
        2
    }
}

#[inline]
fn is_major_piece(pt: crate::shogi::PieceType) -> bool {
    matches!(
        pt,
        crate::shogi::PieceType::Bishop
            | crate::shogi::PieceType::Rook
            | crate::shogi::PieceType::Horse
            | crate::shogi::PieceType::Dragon
    )
}

/// Progress-based 8 bucket assignment (Gikou-lite logistic regression).
///
/// `p = sigmoid(bias + Σ(w_i * ((x_i - mean_i) / std_i)))`
/// `bucket = min(7, floor(p * 8.0))`
#[derive(Clone, Copy)]
pub struct ShogiProgressBucket8GikouLite {
    pub mean: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
    pub std: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
    pub weights: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
    pub bias: f32,
    pub z_clip: [f32; 2],
}

impl ShogiProgressBucket8GikouLite {
    pub const fn new(
        mean: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
        std: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
        weights: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
        bias: f32,
        z_clip: [f32; 2],
    ) -> Self {
        Self { mean, std, weights, bias, z_clip }
    }

    /// Extract raw progress-model features in coeff_v2 order.
    pub fn extract_features(pos: &PackedSfenValue) -> [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] {
        let board = pos.decode();
        let stm = board.side_to_move;
        let stm_king = board.king_square(stm);
        let ntm_king = board.king_square(stm.opponent());

        let mut out = [0.0f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES];
        let baseline = ShogiProgressBucket8::extract_features(pos);
        out[..SHOGI_PROGRESS8_NUM_FEATURES].copy_from_slice(&baseline);

        for (idx, piece) in board.board.iter().enumerate() {
            let pt = piece.piece_type;
            if matches!(pt, crate::shogi::PieceType::None | crate::shogi::PieceType::King) {
                continue;
            }

            let is_stm_piece = piece.color == stm;
            let side_offset = if is_stm_piece { 6usize } else { 12usize };
            let major_offset = if is_stm_piece { 18usize } else { 24usize };

            let sq = crate::shogi::Square::from_index(idx);
            let own_king = if is_stm_piece { stm_king } else { ntm_king };
            let opp_king = if is_stm_piece { ntm_king } else { stm_king };

            let own_bin = distance_bin(chebyshev_distance(sq, own_king));
            let opp_bin = distance_bin(chebyshev_distance(sq, opp_king));

            out[side_offset + own_bin] += 1.0;
            out[side_offset + 3 + opp_bin] += 1.0;

            if is_major_piece(pt) {
                out[major_offset + own_bin] += 1.0;
                out[major_offset + 3 + opp_bin] += 1.0;
            }
        }

        let stm_hand = if stm == crate::shogi::Color::Black { board.black_hand } else { board.white_hand };
        let ntm_hand = if stm == crate::shogi::Color::Black { board.white_hand } else { board.black_hand };

        out[30] = stm_hand.counts.iter().map(|&v| v as f32).sum::<f32>();
        out[31] = ntm_hand.counts.iter().map(|&v| v as f32).sum::<f32>();
        // bishop + rook in hand
        out[32] = (stm_hand.counts[5] + stm_hand.counts[6]) as f32;
        out[33] = (ntm_hand.counts[5] + ntm_hand.counts[6]) as f32;

        out
    }

    pub fn progress(&self, pos: &PackedSfenValue) -> f32 {
        let x = Self::extract_features(pos);

        let mut z = self.bias;
        for (i, &x_i) in x.iter().enumerate() {
            let std = if self.std[i] > 0.0 { self.std[i] } else { 1.0 };
            let x_norm = (x_i - self.mean[i]) / std;
            z += self.weights[i] * x_norm;
        }

        let z_min = self.z_clip[0].min(self.z_clip[1]);
        let z_max = self.z_clip[0].max(self.z_clip[1]);
        let z_clamped = z.clamp(z_min, z_max);
        let p = 1.0 / (1.0 + (-z_clamped).exp());
        p.clamp(0.0, 1.0)
    }
}

impl Default for ShogiProgressBucket8GikouLite {
    fn default() -> Self {
        // coeff_v2 を指定しない場合の最小既定値（ほぼ中立）。
        Self {
            mean: [0.0; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
            std: [1.0; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
            weights: [0.0; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES],
            bias: 0.0,
            z_clip: [-8.0, 8.0],
        }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiProgressBucket8GikouLite {
    const BUCKETS: usize = SHOGI_PROGRESS8_NUM_BUCKETS;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let p = self.progress(pos);
        let raw = (p * 8.0).floor() as i32;
        raw.clamp(0, 7) as u8
    }
}

/// 将棋 LayerStacks 用の手数ベース 9 バケット。
///
/// `game_ply` を固定境界で 9 分割する:
/// - b0: <= bounds[0]
/// - ...
/// - b7: <= bounds[7]
/// - b8: > bounds[7]
#[derive(Clone, Copy)]
pub struct ShogiPlyBucket9 {
    pub bounds: [u16; 8],
}

impl Default for ShogiPlyBucket9 {
    fn default() -> Self {
        Self { bounds: SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS }
    }
}

impl OutputBuckets<PackedSfenValue> for ShogiPlyBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let ply = pos.game_ply();
        for (i, &bound) in self.bounds.iter().enumerate() {
            if ply <= bound {
                return i as u8;
            }
        }
        8
    }
}

/// Runtime-selectable 9-bucket mode for shogi LayerStacks.
///
/// This enum stays `Copy` because `OutputBuckets` requires it, so boxing the
/// larger progress variants is not an option here.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Default)]
pub enum ShogiLayerStackBucket9 {
    #[default]
    KingRank9,
    Ply9([u16; 8]),
    Progress8(ShogiProgressBucket8),
    Progress8GikouLite(ShogiProgressBucket8GikouLite),
    Progress8KPAbs(ShogiProgressKPAbs),
}

impl OutputBuckets<PackedSfenValue> for ShogiLayerStackBucket9 {
    const BUCKETS: usize = 9;

    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        match self {
            Self::KingRank9 => ShogiKingRankBucket::<9>.bucket(pos),
            Self::Ply9(bounds) => {
                let ply = pos.game_ply();
                for (i, &bound) in bounds.iter().enumerate() {
                    if ply <= bound {
                        return i as u8;
                    }
                }
                8
            }
            // 9bucket互換モード:
            // progress8 は bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8(progress) => progress.bucket(pos),
            // 9bucket互換モード:
            // progress8-gikou-lite も bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8GikouLite(progress) => progress.bucket(pos),
            // 9bucket互換モード:
            // KP-absolute も bucket 0..7 を使用し、bucket 8 は未使用となる。
            Self::Progress8KPAbs(progress) => progress.bucket(pos),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn psv_with_ply(ply: u16) -> PackedSfenValue {
        let mut psv = PackedSfenValue::default();
        let bytes = psv.as_bytes_mut();
        let le = ply.to_le_bytes();
        bytes[36] = le[0];
        bytes[37] = le[1];
        psv
    }

    #[test]
    fn test_shogi_ply_bucket9_default_bounds() {
        let bucket = ShogiPlyBucket9::default();
        assert_eq!(bucket.bucket(&psv_with_ply(0)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(30)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(31)), 1);
        assert_eq!(bucket.bucket(&psv_with_ply(138)), 7);
        assert_eq!(bucket.bucket(&psv_with_ply(139)), 8);
        assert_eq!(bucket.bucket(&psv_with_ply(400)), 8);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_ply_mode() {
        let bucket = ShogiLayerStackBucket9::Ply9([10, 20, 30, 40, 50, 60, 70, 80]);
        assert_eq!(bucket.bucket(&psv_with_ply(10)), 0);
        assert_eq!(bucket.bucket(&psv_with_ply(21)), 2);
        assert_eq!(bucket.bucket(&psv_with_ply(80)), 7);
        assert_eq!(bucket.bucket(&psv_with_ply(81)), 8);
    }

    // ===== P4H2 (進行度4区分 × 持駒量2区分) =====

    /// 持駒重み・進行度別閾値の契約(★推論側 C++ と一致必須)を固定する。
    #[test]
    fn test_p4h2_hand_weight_contract() {
        // [歩,香,桂,銀,金,角,飛] の順。
        assert_eq!(HAND_PIECE_WEIGHTS, [1, 2, 2, 3, 4, 6, 7]);
        // 進行度別閾値(prog4=0..3)。C++ P2_HAND_VALUE_THRESHOLD[4] と一致。
        assert_eq!(HAND_VALUE_THRESHOLDS, [6, 12, 20, 24]);
    }

    /// 仕様の持駒重み例。counts index = [歩,香,桂,銀,金,角,飛]。
    #[test]
    fn test_p4h2_weighted_hand_examples() {
        let z = [0u8; 7];
        // 両者無し → 0 → 少
        assert_eq!(weighted_hand_total_from_counts(&z, &z), 0);
        // 歩4 → 4 → 少
        assert_eq!(weighted_hand_total_from_counts(&[4, 0, 0, 0, 0, 0, 0], &z), 4);
        // 銀2 → 6 → 少
        assert_eq!(weighted_hand_total_from_counts(&[0, 0, 0, 2, 0, 0, 0], &z), 6);
        // 角1 + 歩4 → 6 + 4 = 10 → 多
        assert_eq!(weighted_hand_total_from_counts(&[4, 0, 0, 0, 0, 1, 0], &z), 10);
        // 飛1 + 銀1 → 7 + 3 = 10 → 多
        assert_eq!(weighted_hand_total_from_counts(&[0, 0, 0, 1, 0, 0, 1], &z), 10);
    }

    /// 閾値境界(各進行度別): threshold-1 → 少, threshold → 多, threshold+1 → 多。
    #[test]
    fn test_p4h2_threshold_boundary() {
        for &thr in &HAND_VALUE_THRESHOLDS {
            let hb = |total: i32| -> u8 { if total >= thr { 1 } else { 0 } };
            assert_eq!(hb(thr - 1), 0, "thr={thr}");
            assert_eq!(hb(thr), 1, "thr={thr}");
            assert_eq!(hb(thr + 1), 1, "thr={thr}");
        }
    }

    /// 先後対称: 黒白の持駒を入れ替えても total(=hand_bucket)は不変。
    #[test]
    fn test_p4h2_hand_symmetry() {
        let b = [1u8, 0, 2, 0, 1, 0, 1];
        let w = [0u8, 3, 0, 1, 0, 1, 0];
        assert_eq!(weighted_hand_total_from_counts(&b, &w), weighted_hand_total_from_counts(&w, &b));
    }

    /// stack = progress_bucket4*2 + hand_bucket2 は常に 0..=7。
    #[test]
    fn test_p4h2_stack_range() {
        for pb8 in 0u8..8 {
            for hand2 in 0i32..2 {
                let stack = (pb8 / 2) as i32 * 2 + hand2;
                assert!((0..8).contains(&stack), "stack out of range: {}", stack);
            }
        }
    }

    /// P8H2: stack = progress_bucket8*2 + hand_bucket2 は常に 0..=15、かつ
    /// BUCKETS 定数(16)と全域が一致する。
    #[test]
    fn test_p8h2_stack_range() {
        assert_eq!(SHOGI_PROGRESS8_HAND2_NUM_BUCKETS, 16);
        let mut seen = [false; SHOGI_PROGRESS8_HAND2_NUM_BUCKETS];
        for pb8 in 0i32..8 {
            for hand2 in 0i32..2 {
                let stack = pb8 * 2 + hand2;
                assert!((0..16).contains(&stack), "stack out of range: {}", stack);
                seen[stack as usize] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "P8H2 は 0..=15 の全 bucket を張るべき");
    }

    #[test]
    fn test_shogi_progress_bucket8_range() {
        let bucket = ShogiProgressBucket8::default();
        for ply in [1u16, 30, 60, 100, 150, 220, 300] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8(ShogiProgressBucket8::default());
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-in-9 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_progress_bucket8_gikou_lite_range() {
        let bucket = ShogiProgressBucket8GikouLite::default();
        for ply in [1u16, 30, 60, 100, 150, 220, 300] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-gikou-lite bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_progress_bucket8_gikou_lite_prefix_matches_v1_features() {
        let psv = psv_with_ply(60);
        let v1 = ShogiProgressBucket8::extract_features(&psv);
        let v2 = ShogiProgressBucket8GikouLite::extract_features(&psv);
        assert_eq!(&v2[..SHOGI_PROGRESS8_NUM_FEATURES], &v1);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_gikou_lite_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8GikouLite(ShogiProgressBucket8GikouLite::default());
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-gikou-lite-in-9 bucket must be in 0..=7, got {}", b);
        }
    }

    #[test]
    fn test_shogi_progress_kp_abs_default_is_neutral_on_invalid_position() {
        let bucket = ShogiProgressKPAbs;
        let psv = psv_with_ply(60);
        assert_eq!(bucket.progress(&psv), 0.5);
        assert_eq!(bucket.bucket(&psv), 4);
    }

    #[test]
    fn test_shogi_layerstack_bucket9_progress_kp_abs_mode_range() {
        let bucket = ShogiLayerStackBucket9::Progress8KPAbs(ShogiProgressKPAbs);
        for ply in [1u16, 40, 80, 120, 200, 400] {
            let b = bucket.bucket(&psv_with_ply(ply));
            assert!(b <= 7, "progress8-kpabs-in-9 bucket must be in 0..=7, got {}", b);
        }
    }

    // ===== OPNARCH k9k9z (2026-11-14)。 値は設計書 2.2 の式を手で計算した固定値 =====
    //   (python の runs/opnarch/kb_density.py の norm / z9 でも同じ値になる)。
    fn sq(file1: u8, rank1: u8) -> crate::shogi::Square {
        // 引数は将棋の表記 (筋 1..9 / 段 1..9)。 YO の sq = (筋-1)*9 + (段-1)。
        crate::shogi::Square::new(file1 - 1, rank1 - 1)
    }

    #[test]
    fn test_opn_z9_table() {
        // 先手から見た座標: 1〜3 段 → 0、 4〜6 段 → 1、 7 段 → 2、 8 段 → 3/4/5、 9 段 → 6/7/8 (1〜3 筋 / 4〜6 筋 / 7〜9 筋)。
        assert_eq!(opn_z9(sq(5, 1)), 0);
        assert_eq!(opn_z9(sq(9, 3)), 0);
        assert_eq!(opn_z9(sq(1, 4)), 1);
        assert_eq!(opn_z9(sq(5, 6)), 1);
        assert_eq!(opn_z9(sq(7, 7)), 2);
        assert_eq!(opn_z9(sq(2, 8)), 3);
        assert_eq!(opn_z9(sq(5, 8)), 4);
        assert_eq!(opn_z9(sq(8, 8)), 5);
        assert_eq!(opn_z9(sq(1, 9)), 6);
        assert_eq!(opn_z9(sq(5, 9)), 7);
        assert_eq!(opn_z9(sq(9, 9)), 8);
    }

    #[test]
    fn test_opn_kz_fixed_positions() {
        // (1) 平手の初期配置: 先手玉 5 九、 後手玉 5 一 → どちらも z9 = 7 → 7*9+7 = 70 (手番によらない)。
        assert_eq!(opn_kz(Color::Black, sq(5, 9), sq(5, 1)), 70);
        assert_eq!(opn_kz(Color::White, sq(5, 9), sq(5, 1)), 70);
        // (2) 先手玉 2 八 (z9 3)、 後手玉 8 二 (後手から見て 2 八 = z9 3) → 30。
        assert_eq!(opn_kz(Color::Black, sq(2, 8), sq(8, 2)), 30);
        // (3) 先手玉 7 八 (z9 5)、 後手玉 3 二 (後手から見て 7 八 = z9 5) → 50。
        assert_eq!(opn_kz(Color::Black, sq(7, 8), sq(3, 2)), 50);
        // (4) 後手番: 先手玉 7 九 (z9 8)、 後手玉 4 三 (後手から見て 6 七 = z9 2)。
        //     手番側 (後手) が上位桁 → 2*9+8 = 26。 先手番なら 8*9+2 = 74。
        assert_eq!(opn_kz(Color::White, sq(7, 9), sq(4, 3)), 26);
        assert_eq!(opn_kz(Color::Black, sq(7, 9), sq(4, 3)), 74);
        // (5) 相手玉が自陣深く (入玉): 先手玉 1 一 (z9 0)、 後手玉 9 九 (後手から見て 1 一 = z9 0) → 0。
        assert_eq!(opn_kz(Color::Black, sq(1, 1), sq(9, 9)), 0);
        // 無効な升は 0 (panic しない)。
        assert_eq!(opn_kz(Color::Black, crate::shogi::Square::NONE, sq(5, 1)), 0);
    }

    #[test]
    fn test_opn_head_order_king_then_progress() {
        // YO と同じ「玉 → 進行度」の順: head = kz*3 + pb。
        assert_eq!(<ShogiOpnK9z<3> as OutputBuckets<PackedSfenValue>>::BUCKETS, 243);
        assert_eq!(<ShogiOpnK9z<1> as OutputBuckets<PackedSfenValue>>::BUCKETS, 81);
        assert_eq!(ShogiOpnK9z::<3>::head(70, 0), 210);
        assert_eq!(ShogiOpnK9z::<3>::head(70, 2), 212);
        assert_eq!(ShogiOpnK9z::<3>::head(80, 2), 242);
        // 範囲外の進行度は pb = 2 に寄せる。
        assert_eq!(ShogiOpnK9z::<3>::head(80, 7), 242);
        assert_eq!(ShogiOpnK9z::<1>::head(26, 5), 26);
    }
}
