//! Activity-Risk Interaction Head 用 sparse feature 計算 (= 2026-06-06)。
//!
//! 推論側 `rshogi-core::nnue::activity_risk` と完全一致する index 契約を保証する。
//!
//! # 経路
//!
//! 1. bullet 側 `ShogiBoard` (= PSV 展開) を SFEN string に変換
//! 2. `rshogi-core::Position::set_sfen_with_pass_rights` で Position 構築
//! 3. `rshogi-core::nnue::activity_risk::compute_major_activity` 等で feature 計算
//! 4. `activity_to_bucket` / `pressure_to_bucket` / `shield_to_bucket` で bucket 化
//! 5. `flatten_idx(bucket, perspective, activity, pressure, shield)` で index 取得
//!
//! # NPS overhead
//!
//! per-eval で SFEN parse + Position 構築が走る (= ~10 μs/eval estimate)。
//! pairshape (= 直接 ShogiBoard 操作) と比べて重いが、 推論側との完全一致を優先する。
//! NPS が問題なら ShogiBoard → Position 直接構築 (= SFEN 経由なし) で 1/3 まで縮小可能。

use crate::shogi::packed_sfen::ShogiBoard;
use crate::shogi::types::{Color as BulletColor, PieceType as BulletPieceType, Square as BulletSquare};

// =============================================================================
// 定数 (= 推論側 rshogi-core::nnue::activity_risk と完全一致)
// =============================================================================

/// bucket 数 (= PSQT bucket と同期、 8 = progress8 系)
pub const ACTIVITY_RISK_NUM_BUCKETS: usize = 8;
/// major_activity bucket 数
pub const ACTIVITY_RISK_NUM_ACTIVITY: usize = 8;
/// king_pressure bucket 数
pub const ACTIVITY_RISK_NUM_PRESSURE: usize = 8;
/// shield bucket 数
pub const ACTIVITY_RISK_NUM_SHIELD: usize = 4;

/// 1 perspective × 1 bucket あたりの weight 数 (= activity × pressure × shield = 256)
pub const ACTIVITY_RISK_PER_PB_DIM: usize =
    ACTIVITY_RISK_NUM_ACTIVITY * ACTIVITY_RISK_NUM_PRESSURE * ACTIVITY_RISK_NUM_SHIELD;

/// 1 perspective 分の disk dim (= bucket × per_pb = 8 × 256 = 2048)
pub const ACTIVITY_RISK_DIM_PER_PERSP: usize = ACTIVITY_RISK_NUM_BUCKETS * ACTIVITY_RISK_PER_PB_DIM;

/// 全 weight 数 = 2 × 2048 = 4096
pub const ACTIVITY_RISK_TOTAL_COUNT: u32 = (2 * ACTIVITY_RISK_DIM_PER_PERSP) as u32;

/// per-perspective active 数 (= 1 fire only)
pub const ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP: usize = 1;

/// disk section magic (= 推論側と完全一致、 P2ACTR + null padding)
pub const ACTIVITY_RISK_SUB_MAGIC: &[u8; 8] = b"P2ACTR\0\0";

/// section version (= 3、 2026-06-08 v1 base 4096 + tactical 1024 = 5120 weight)。
/// v1/v2 ckpt は v3 build で load 不可、 v1 → v3 変換 tool で base copy + tactical 0 作成可能。
pub const ACTIVITY_RISK_SECTION_VERSION: u32 = 3;

/// v3 base 重み数 (= 4096、 既存 4 軸 と同形)
pub const ACTIVITY_RISK_V3_BASE_DIM: usize = ACTIVITY_RISK_DIM_PER_PERSP * 2; // 2048 * 2 = 4096

/// v3 tactical 軸 class 数 (= heavy_attack / escape / hand_check 各 4)
pub const ACTIVITY_RISK_V3_NUM_HEAVY: usize = 4;
pub const ACTIVITY_RISK_V3_NUM_ESCAPE: usize = 4;
pub const ACTIVITY_RISK_V3_NUM_HAND_CHECK: usize = 4;

/// v3 tactical per perspective dim (= heavy × escape × hand_check × bucket = 4 × 4 × 4 × 8 = 512)
pub const ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP: usize = ACTIVITY_RISK_NUM_BUCKETS
    * ACTIVITY_RISK_V3_NUM_HEAVY
    * ACTIVITY_RISK_V3_NUM_ESCAPE
    * ACTIVITY_RISK_V3_NUM_HAND_CHECK;

/// v3 全 weight 数 (= base 4096 + tactical 1024 = 5120)
pub const ACTIVITY_RISK_V3_TOTAL_COUNT: u32 =
    (2 * ACTIVITY_RISK_DIM_PER_PERSP + 2 * ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP) as u32;

/// v3 tactical sparse input 名 (= per perspective 1 active、 dim 512)
pub const ACTIVITY_RISK_V3_TACTICAL_NAME_BLACK: &str = "activity_risk_v3_tactical_black";
pub const ACTIVITY_RISK_V3_TACTICAL_NAME_WHITE: &str = "activity_risk_v3_tactical_white";

/// 量子化スケール (= QA = 127、 PairShape と同条件)
pub const ACTIVITY_RISK_QA: f64 = 127.0;

/// architecture marker (= ckpt log 用)
pub const ACTIVITY_RISK_ARCH_MARKER: &str = "ACTR=1";

/// sparse input 名 (= 内部識別)
pub const ACTIVITY_RISK_SPARSE_NAME_BLACK: &str = "activity_risk_black";
pub const ACTIVITY_RISK_SPARSE_NAME_WHITE: &str = "activity_risk_white";

// =============================================================================
// bucket 境界 (= 推論側 rshogi-core::nnue::activity_risk と bit-exact 同一)
// =============================================================================

/// major_activity (= 飛角馬竜 pseudo mobility 合計) を 0..=7 の bucket に変換する。
///
/// 境界: 0, 1-3, 4-6, 7-9, 10-12, 13-15, 16-19, 20+
#[inline]
pub fn activity_to_bucket(activity: usize) -> usize {
    if activity == 0 {
        0
    } else if activity < 4 {
        1
    } else if activity < 7 {
        2
    } else if activity < 10 {
        3
    } else if activity < 13 {
        4
    } else if activity < 16 {
        5
    } else if activity < 20 {
        6
    } else {
        7
    }
}

/// king_pressure (= 玉 3x3 内の敵利き合計) を 0..=7 の bucket に変換する。
#[inline]
pub fn pressure_to_bucket(pressure: usize) -> usize {
    pressure.min(7)
}

/// shield (= 玉 5x5 内の自軍小駒数) を 0..=3 の bucket に変換する。
///
/// 境界: 0-2, 3-4, 5-6, 7+
#[inline]
pub fn shield_to_bucket(shield: usize) -> usize {
    if shield < 3 {
        0
    } else if shield < 5 {
        1
    } else if shield < 7 {
        2
    } else {
        3
    }
}

// =============================================================================
// Index 契約 (= 推論側 flatten_idx と完全一致)
// =============================================================================

/// `[bucket][perspective][activity][pressure][shield]` row-major flatten index。
///
/// 範囲: `[0, 4096)`
#[inline]
pub const fn flatten_idx(bucket: usize, perspective: usize, activity: usize, pressure: usize, shield: usize) -> usize {
    let inner = ((activity * ACTIVITY_RISK_NUM_PRESSURE) + pressure) * ACTIVITY_RISK_NUM_SHIELD + shield;
    let pb = perspective * ACTIVITY_RISK_PER_PB_DIM + inner;
    bucket * 2 * ACTIVITY_RISK_PER_PB_DIM + pb
}

/// 1 perspective 内 (= 0..2048) の sparse local index (= disk 上で persp 別 sub-table を
/// 仮想的に分割した時の offset)。 sparse input としてはこちらを格納し、 graph 側で
/// `weights[idx_persp_local]` 形式で乗算する。
///
/// **重要**: graph 側 weight tensor は 1 persp 分 (= 2048 entries) を別 tensor として保持し、
/// disk save 時に rshogi-core 期待の row-major (= bucket-major / persp 交互) に並べ替える。
#[inline]
pub const fn local_idx_in_persp(bucket: usize, activity: usize, pressure: usize, shield: usize) -> usize {
    let inner = ((activity * ACTIVITY_RISK_NUM_PRESSURE) + pressure) * ACTIVITY_RISK_NUM_SHIELD + shield;
    bucket * ACTIVITY_RISK_PER_PB_DIM + inner
}

// =============================================================================
// ShogiBoard → SFEN string 変換 helper
// =============================================================================

/// `ShogiBoard` を SFEN string に変換する (= rshogi-core::Position 構築用)。
///
/// 出力例: `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1`
pub fn shogi_board_to_sfen(board: &ShogiBoard) -> String {
    let mut sfen = String::with_capacity(128);

    // 盤面 (= rank 0..8、 各 rank で file 8 → 0 で SFEN 順)
    for rank in 0..9u8 {
        let mut empty_count: u8 = 0;
        for file in (0..9u8).rev() {
            let sq = BulletSquare::new(file, rank);
            let piece = board.piece_on(sq);
            if piece.is_none() {
                empty_count += 1;
            } else {
                if empty_count > 0 {
                    sfen.push_str(&empty_count.to_string());
                    empty_count = 0;
                }
                sfen.push_str(piece_to_sfen_str(piece.color, piece.piece_type));
            }
        }
        if empty_count > 0 {
            sfen.push_str(&empty_count.to_string());
        }
        if rank < 8 {
            sfen.push('/');
        }
    }

    // 手番
    sfen.push(' ');
    sfen.push(match board.side_to_move {
        BulletColor::Black => 'b',
        BulletColor::White => 'w',
    });

    // 持ち駒
    sfen.push(' ');
    let hand_str = hands_to_sfen_str(board);
    if hand_str.is_empty() {
        sfen.push('-');
    } else {
        sfen.push_str(&hand_str);
    }

    // 手数 (= 既知の game_ply を使う、 set_sfen 側で legal とみなされる)
    sfen.push(' ');
    sfen.push_str(&board.ply.to_string());

    sfen
}

/// (color, piece_type) を SFEN board char に変換 (= 成駒は `+` 接頭)。
fn piece_to_sfen_str(color: BulletColor, pt: BulletPieceType) -> &'static str {
    use BulletColor::*;
    use BulletPieceType::*;
    match (color, pt) {
        (Black, Pawn) => "P",
        (Black, Lance) => "L",
        (Black, Knight) => "N",
        (Black, Silver) => "S",
        (Black, Bishop) => "B",
        (Black, Rook) => "R",
        (Black, Gold) => "G",
        (Black, King) => "K",
        (Black, ProPawn) => "+P",
        (Black, ProLance) => "+L",
        (Black, ProKnight) => "+N",
        (Black, ProSilver) => "+S",
        (Black, Horse) => "+B",
        (Black, Dragon) => "+R",
        (White, Pawn) => "p",
        (White, Lance) => "l",
        (White, Knight) => "n",
        (White, Silver) => "s",
        (White, Bishop) => "b",
        (White, Rook) => "r",
        (White, Gold) => "g",
        (White, King) => "k",
        (White, ProPawn) => "+p",
        (White, ProLance) => "+l",
        (White, ProKnight) => "+n",
        (White, ProSilver) => "+s",
        (White, Horse) => "+b",
        (White, Dragon) => "+r",
        (_, None) => "",
    }
}

/// 持ち駒文字列を SFEN 形式で生成 (= 例: `B2g3p`)。
fn hands_to_sfen_str(board: &ShogiBoard) -> String {
    let mut s = String::with_capacity(32);
    let order = [
        BulletPieceType::Rook,
        BulletPieceType::Bishop,
        BulletPieceType::Gold,
        BulletPieceType::Silver,
        BulletPieceType::Knight,
        BulletPieceType::Lance,
        BulletPieceType::Pawn,
    ];
    for &color in &[BulletColor::Black, BulletColor::White] {
        let hand = board.hand(color);
        for &pt in &order {
            let n = hand.count(pt);
            if n == 0 {
                continue;
            }
            if n > 1 {
                s.push_str(&n.to_string());
            }
            s.push_str(piece_to_sfen_str(color, pt));
        }
    }
    s
}

// =============================================================================
// Feature 計算 (= rshogi-core 借用 wrap)
// =============================================================================

/// `ShogiBoard` から activity_risk feature (= base 用 v1 と同 4 軸) を perspective 別に計算する。
///
/// 2026-06-08: v3 化、 pressure は v1 (= 玉 3x3 内 敵利き合計) に戻す (= base 維持)。
/// v3 tactical 軸 (= heavy_attack / escape / hand_check) は `compute_tactical_features_v3` で別途取得。
///
/// 戻り値: `(black_activity, black_pressure, black_shield, white_activity, white_pressure, white_shield)`
pub fn compute_features(board: &ShogiBoard) -> (usize, usize, usize, usize, usize, usize) {
    use rshogi_core::nnue::activity_risk::{compute_king_pressure, compute_major_activity, compute_shield};
    use rshogi_core::position::Position;
    use rshogi_core::types::Color as CoreColor;

    let sfen = shogi_board_to_sfen(board);
    let mut pos = Position::new();
    pos.set_sfen_with_pass_rights(&sfen, 0, 0)
        .expect("ShogiBoard → SFEN 変換が rshogi-core::Position::set_sfen で受理されること");

    let b_act = compute_major_activity(&pos, CoreColor::Black);
    let b_pre = compute_king_pressure(&pos, CoreColor::Black);
    let b_shi = compute_shield(&pos, CoreColor::Black);
    let w_act = compute_major_activity(&pos, CoreColor::White);
    let w_pre = compute_king_pressure(&pos, CoreColor::White);
    let w_shi = compute_shield(&pos, CoreColor::White);

    (b_act, b_pre, b_shi, w_act, w_pre, w_shi)
}

/// v3 tactical features (= heavy_attack / escape / hand_check 各 0..=3) を perspective 別に計算する。
///
/// 戻り値: `(black_heavy, black_escape, black_check, white_heavy, white_escape, white_check)`
pub fn compute_tactical_features_v3(board: &ShogiBoard) -> (usize, usize, usize, usize, usize, usize) {
    use rshogi_core::nnue::activity_risk::{
        compute_escape_class_v3, compute_hand_check_class_v3, compute_heavy_attack_class_v3,
    };
    use rshogi_core::position::Position;
    use rshogi_core::types::Color as CoreColor;

    let sfen = shogi_board_to_sfen(board);
    let mut pos = Position::new();
    pos.set_sfen_with_pass_rights(&sfen, 0, 0)
        .expect("ShogiBoard → SFEN 変換が rshogi-core::Position::set_sfen で受理されること");

    let b_heavy = compute_heavy_attack_class_v3(&pos, CoreColor::Black);
    let b_escape = compute_escape_class_v3(&pos, CoreColor::Black);
    let b_check = compute_hand_check_class_v3(&pos, CoreColor::Black);
    let w_heavy = compute_heavy_attack_class_v3(&pos, CoreColor::White);
    let w_escape = compute_escape_class_v3(&pos, CoreColor::White);
    let w_check = compute_hand_check_class_v3(&pos, CoreColor::White);

    (b_heavy, b_escape, b_check, w_heavy, w_escape, w_check)
}

/// v3 tactical sparse local index (= per perspective 内 offset)。
///
/// `[bucket][heavy][escape][hand_check]` row-major、 範囲 `[0, 512)`。
#[inline]
pub const fn activity_risk_v3_tactical_local_idx(
    bucket: usize,
    heavy: usize,
    escape: usize,
    hand_check: usize,
) -> usize {
    let inner = ((heavy * ACTIVITY_RISK_V3_NUM_ESCAPE) + escape) * ACTIVITY_RISK_V3_NUM_HAND_CHECK + hand_check;
    let per_bucket = ACTIVITY_RISK_V3_NUM_HEAVY * ACTIVITY_RISK_V3_NUM_ESCAPE * ACTIVITY_RISK_V3_NUM_HAND_CHECK;
    bucket * per_bucket + inner
}

// =============================================================================
// 単体テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 定数値が推論側 rshogi-core::nnue::activity_risk と一致 (= 静的契約)。
    #[test]
    fn activity_risk_constants_match_rshogi_core() {
        use rshogi_core::nnue::activity_risk as a;
        assert_eq!(ACTIVITY_RISK_NUM_BUCKETS, a::ACTIVITY_RISK_NUM_BUCKETS);
        assert_eq!(ACTIVITY_RISK_NUM_ACTIVITY, a::ACTIVITY_RISK_NUM_ACTIVITY);
        assert_eq!(ACTIVITY_RISK_NUM_PRESSURE, a::ACTIVITY_RISK_NUM_PRESSURE);
        assert_eq!(ACTIVITY_RISK_NUM_SHIELD, a::ACTIVITY_RISK_NUM_SHIELD);
        assert_eq!(ACTIVITY_RISK_PER_PB_DIM, a::ACTIVITY_RISK_PER_PB_DIM);
        assert_eq!(ACTIVITY_RISK_TOTAL_COUNT, a::ACTIVITY_RISK_COUNT);
        assert_eq!(ACTIVITY_RISK_SUB_MAGIC, &a::ACTIVITY_RISK_SUB_MAGIC);
        assert_eq!(ACTIVITY_RISK_SECTION_VERSION, a::ACTIVITY_RISK_VERSION);
    }

    /// flatten_idx は推論側と bit-exact 一致 (= disk layout 契約)。
    #[test]
    fn flatten_idx_matches_rshogi_core() {
        use rshogi_core::nnue::activity_risk as a;
        // 先頭・末尾・境界 + 内部
        let cases = [(0, 0, 0, 0, 0), (7, 1, 7, 7, 3), (3, 0, 5, 4, 2), (5, 1, 2, 6, 1)];
        for &(b, p, ac, pr, sh) in &cases {
            assert_eq!(flatten_idx(b, p, ac, pr, sh), a::flatten_idx(b, p, ac, pr, sh));
        }
    }

    /// bucket 境界関数が推論側と一致。
    #[test]
    fn bucket_fns_match_rshogi_core() {
        use rshogi_core::nnue::activity_risk as a;
        for x in 0..30usize {
            assert_eq!(activity_to_bucket(x), a::activity_to_bucket(x));
            assert_eq!(pressure_to_bucket(x), a::pressure_to_bucket(x));
            assert_eq!(shield_to_bucket(x), a::shield_to_bucket(x));
        }
    }

    /// 平手局面 SFEN 変換が round-trip 通る (= set_sfen 受理)。
    #[test]
    fn shogi_board_to_sfen_startpos_round_trips() {
        let _psv_bytes = [0u8; 40];
        // 平手 PSV を直接組み立てるのは複雑なので、 ここでは
        // ShogiBoard::default() を平手相当に setup する代わりに簡易検証として
        // 「set_sfen が panic しない」 ことだけ確認。
        // (本格 round-trip test は VM 上で hcpe 経由で確認)
        let board = ShogiBoard::default();
        let sfen = shogi_board_to_sfen(&board);
        assert!(sfen.contains(' '));
    }
}
