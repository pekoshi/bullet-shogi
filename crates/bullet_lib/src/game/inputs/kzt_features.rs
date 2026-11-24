//! King Zone Threat / 持駒打ち込み (KZT_HAND) sparse head 用 feature 計算
//! (= 2026-06-30、 2026-07-01 factored v1 化)。
//!
//! 推論側 `rshogi-core::nnue::kzt` (= `collect_hand_features`) を **直接呼び出す** ことで、
//! 訓練側 feature emit と engine gather の bit-exact 一致を保証する (= gather 不変)。
//!
//! # factored v1 (= dense 716,800 → 5 テーブル 10,336)
//!
//! dense 版 (= 6 軸直積 716,800 weight) は過学習で eval 極値のブレが大きいため、 同じ active
//! feature を 5 つの低ランクテーブルの和で近似する (= 平滑化)。 各 active drop feature は
//! (persp,bucket,king_region,hand_piece,target_rel,drop_state) の 6 軸を持ち、 これを下記 5
//! テーブルへ射影して **5 つの sparse index** に展開する (= 1 drop が 5 sparse input を叩く)。
//!
//! ```text
//! W_core  [hand_piece][target_rel][drop_state]   = 2,800
//! W_region[king_region][target_rel][drop_state]  = 6,400
//! W_bucket[bucket][hand_piece][drop_state]       = 896
//! W_persp [persp][hand_piece][drop_state]         = 224
//! W_bias  [drop_state]                            = 16
//! ```
//!
//! index 式は推論側 `rshogi-core::nnue::kzt::{kzt_core_index, ...}` を **直接借用** するため、
//! 訓練が展開する index 集合と engine factored evaluate が引く index は完全一致する。
//!
//! # 訓練/推論の符号合成
//!
//! engine eval = `(Σ_{collect(stm)} score - Σ_{collect(nstm)} score) × OUTPUT_SCALE`。
//! 各 table の weight tensor は persp 非依存で共有 (= W_persp のみ persp 軸を index に焼き込む)、
//! pairshape と同様に `net_output += W·black - W·white` (= 常に black - white) で反対称合成する。
//! disk section P2KZT (= version 2) は core → region → bucket → persp → bias の固定順で
//! flat i16 連結し、 engine の `KztFactoredWeights::from_flat` と完全一致する。

use crate::shogi::packed_sfen::ShogiBoard;

use rshogi_core::nnue::kzt::{
    KZT_BIAS_COUNT, KZT_BUCKET_COUNT, KZT_CORE_COUNT, KZT_FACTORED_COUNT, KZT_HAND_MAX_ACTIVE, KZT_PERSP_COUNT,
    KZT_REGION_COUNT, KztHandFeature, collect_hand_features, kzt_bias_index, kzt_bucket_index, kzt_core_index,
    kzt_hand_decode, kzt_persp_index, kzt_region_index,
};
use rshogi_core::position::Position;
use rshogi_core::types::Color as CoreColor;

// =============================================================================
// 定数 (= 推論側 rshogi-core::nnue::kzt と完全一致)
// =============================================================================

/// factored 5 テーブル合計 i16 entry 数 (= disk total = 10,336)。
pub const KZT_TOTAL_COUNT: u32 = KZT_FACTORED_COUNT as u32;

/// 各 factored テーブルの要素数 (= 推論側と一致)。
pub const KZT_CORE_DIM: usize = KZT_CORE_COUNT;
pub const KZT_REGION_DIM: usize = KZT_REGION_COUNT;
pub const KZT_BUCKET_DIM: usize = KZT_BUCKET_COUNT;
pub const KZT_PERSP_DIM: usize = KZT_PERSP_COUNT;
pub const KZT_BIAS_DIM: usize = KZT_BIAS_COUNT;

/// factored テーブル数 (= core/region/bucket/persp/bias)。
pub const KZT_NUM_TABLES: usize = 5;

/// disk 固定順 (= core → region → bucket → persp → bias) の graph weight tensor 名。
pub const KZT_TABLE_WEIGHT_NAMES: [&str; KZT_NUM_TABLES] =
    ["kzt_w_core", "kzt_w_region", "kzt_w_bucket", "kzt_w_persp", "kzt_w_bias"];

/// 各テーブルの黒視点 sparse input 名。
pub const KZT_TABLE_BLACK_NAMES: [&str; KZT_NUM_TABLES] =
    ["kzt_core_black", "kzt_region_black", "kzt_bucket_black", "kzt_persp_black", "kzt_bias_black"];

/// 各テーブルの白視点 sparse input 名。
pub const KZT_TABLE_WHITE_NAMES: [&str; KZT_NUM_TABLES] =
    ["kzt_core_white", "kzt_region_white", "kzt_bucket_white", "kzt_persp_white", "kzt_bias_white"];

/// 各テーブルの要素数 (= sparse input / weight tensor の列数)。
pub const KZT_TABLE_DIMS: [usize; KZT_NUM_TABLES] =
    [KZT_CORE_DIM, KZT_REGION_DIM, KZT_BUCKET_DIM, KZT_PERSP_DIM, KZT_BIAS_DIM];

/// per-table 相対 learning rate (= Codex 指定: core/region/bucket/persp/bias)。
/// `--kzt-hand-lr-mult` を core 基準として、 各 table の lr_scale = mult × 下記。
pub const KZT_TABLE_LR_REL: [f32; KZT_NUM_TABLES] = [1.0, 0.7, 0.5, 0.25, 0.25];

/// per-table 相対 weight decay (= Codex 指定)。 各 table の decay = base_wd × 下記。
pub const KZT_TABLE_WD_REL: [f32; KZT_NUM_TABLES] = [1.0, 1.2, 1.5, 2.0, 2.0];

/// 1 perspective あたりの max active (= 各テーブルに feature 1 index を emit、 175)。
pub const KZT_MAX_ACTIVE_PER_PERSP: usize = KZT_HAND_MAX_ACTIVE;

/// ckpt section sub_magic ("P2KZT\0\0\0")。 rshogi-core `KZT_MAGIC` と一致。
pub const KZT_SUB_MAGIC: &[u8; 8] = b"P2KZT\0\0\0";

/// ckpt section format version (= 2 = factored)。 rshogi-core `KZT_VERSION` と一致。
pub const KZT_SECTION_VERSION: u32 = 2;

/// QA = 127 (= pairshape / king_danger / attack_dev / slider_xray と同 convention)。
pub const KZT_QA: f64 = 127.0;

/// architecture marker (= ckpt log 用)。
pub const KZT_ARCH_MARKER: &str = "KZT=2fac";

// disk total が factored の 5 テーブル和に一致することの静的検証。
const _: () = {
    assert!(KZT_FACTORED_COUNT == KZT_CORE_DIM + KZT_REGION_DIM + KZT_BUCKET_DIM + KZT_PERSP_DIM + KZT_BIAS_DIM);
};

// =============================================================================
// Feature 計算 (= rshogi-core 借用 wrap + factored 展開)
// =============================================================================

/// 1 perspective の active feature を、 5 テーブルそれぞれの sparse index に展開した結果。
///
/// 各配列 `xxx[..n]` が有効。 `core[i]` / `region[i]` / `bucket[i]` / `persp[i]` / `bias[i]` は
/// **同一 drop feature `i`** を各テーブルへ射影した index (= 1 drop が 5 index を叩く)。
#[derive(Clone)]
pub struct KztFactoredActive {
    pub core: [u32; KZT_MAX_ACTIVE_PER_PERSP],
    pub region: [u32; KZT_MAX_ACTIVE_PER_PERSP],
    pub bucket: [u32; KZT_MAX_ACTIVE_PER_PERSP],
    pub persp: [u32; KZT_MAX_ACTIVE_PER_PERSP],
    pub bias: [u32; KZT_MAX_ACTIVE_PER_PERSP],
    pub n: usize,
}

impl KztFactoredActive {
    /// 空 (= n == 0)。
    fn empty() -> Self {
        Self {
            core: [0u32; KZT_MAX_ACTIVE_PER_PERSP],
            region: [0u32; KZT_MAX_ACTIVE_PER_PERSP],
            bucket: [0u32; KZT_MAX_ACTIVE_PER_PERSP],
            persp: [0u32; KZT_MAX_ACTIVE_PER_PERSP],
            bias: [0u32; KZT_MAX_ACTIVE_PER_PERSP],
            n: 0,
        }
    }

    /// table index (= 0:core 1:region 2:bucket 3:persp 4:bias) の有効 index slice を返す。
    pub fn table(&self, t: usize) -> &[u32] {
        match t {
            0 => &self.core[..self.n],
            1 => &self.region[..self.n],
            2 => &self.bucket[..self.n],
            3 => &self.persp[..self.n],
            4 => &self.bias[..self.n],
            _ => &self.core[..0],
        }
    }
}

/// `ShogiBoard` + `bucket` から、 `perspective` 攻め視点の KZT_HAND active feature を集め、
/// 各 feature を 5 テーブルへ射影した sparse index を返す。
///
/// 推論側 `collect_hand_features` の dense flat index を `kzt_hand_decode` で 6 軸に分解し、
/// `kzt_core_index` 等 (= 推論側と同一式) で各テーブル index を算出する (= bit-exact)。
///
/// # Panics
/// SFEN 変換が rshogi-core::Position に受理されない場合 panic (= データ破損検出)。
pub fn collect_kzt_factored_for_side(board: &ShogiBoard, perspective: CoreColor, bucket: usize) -> KztFactoredActive {
    let sfen = super::shogi_board_to_sfen(board);
    let mut pos = Position::new();
    pos.set_sfen_with_pass_rights(&sfen, 0, 0)
        .expect("ShogiBoard → SFEN 変換が rshogi-core::Position::set_sfen で受理されること");

    let mut feats = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
    let n = collect_hand_features(&pos, perspective, bucket, &mut feats);

    let mut out = KztFactoredActive::empty();
    let mut written = 0usize;
    for f in feats.iter().take(n) {
        if written >= KZT_MAX_ACTIVE_PER_PERSP {
            break;
        }
        let (persp, bk, king_region, hand_piece, target_rel, drop_state) = kzt_hand_decode(f.index);
        debug_assert_eq!(persp, perspective.index(), "persp = attacker 絶対色");
        debug_assert_eq!(bk, bucket, "bucket が焼き込みと不一致");
        out.core[written] = kzt_core_index(hand_piece, target_rel, drop_state) as u32;
        out.region[written] = kzt_region_index(king_region, target_rel, drop_state) as u32;
        out.bucket[written] = kzt_bucket_index(bk, hand_piece, drop_state) as u32;
        out.persp[written] = kzt_persp_index(persp, hand_piece, drop_state) as u32;
        out.bias[written] = kzt_bias_index(drop_state) as u32;
        written += 1;
    }
    out.n = written;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kzt_constants_match_rshogi_core() {
        assert_eq!(KZT_SUB_MAGIC, rshogi_core::nnue::kzt::KZT_MAGIC);
        assert_eq!(KZT_SECTION_VERSION, rshogi_core::nnue::kzt::KZT_VERSION);
        assert_eq!(KZT_TOTAL_COUNT as usize, rshogi_core::nnue::kzt::KZT_FACTORED_COUNT);
        assert_eq!(KZT_TOTAL_COUNT, 10_336);
        assert_eq!(KZT_CORE_DIM, 2_800);
        assert_eq!(KZT_REGION_DIM, 6_400);
        assert_eq!(KZT_BUCKET_DIM, 896);
        assert_eq!(KZT_PERSP_DIM, 224);
        assert_eq!(KZT_BIAS_DIM, 16);
        assert_eq!(KZT_MAX_ACTIVE_PER_PERSP, 175);
        // disk 固定順 offset (= from_flat と一致)。
        assert_eq!(KZT_TABLE_DIMS.iter().sum::<usize>(), KZT_FACTORED_COUNT);
    }
}
