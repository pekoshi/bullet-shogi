//! Dual PSQT King-Attacker 補助入力 (rshogi-core `nnue::dual_psqt` と数値一致)。
//!
//! 1 駒が **敵 king zone (= 敵 king + 8 近傍 = 9 sq)** に効きを持つかで attack
//! feature を発火、 **自 king zone** に効きを持つかで defend feature を発火。
//! 各 perspective 独立に計算し、 (attack_active, defend_active) を sparse feature
//! index 集合として bullet 訓練 graph に渡す。
//!
//! # 仕様 (rshogi-core SoT 準拠)
//!
//! - dim per perspective: `2 head × 8 bucket × 81 sq × 14 piece = 18,144`
//! - 2 perspective 合計 = 36,288 i16 weight
//! - 正規化: Black=identity、 White=`sq -> 80 - sq` (Half-Mirror なし)
//! - piece_type 0..=13: Pawn..=Dragon (= `PieceType as u8 - 1`)
//! - king zone = `king sq + 8 近傍` (= 最大 9 sq、 端では 4-6 sq)
//! - feature index = `head * 8*81*14 + bucket * 81*14 + sq * 14 + piece`
//! - 利き計算は **occupancy-aware** (= slider は遮蔽あり ray walk)
//!
//! # bullet ↔ rshogi-core 対応
//!
//! | bullet                                       | rshogi-core                                       |
//! |----------------------------------------------|---------------------------------------------------|
//! | `DUAL_DIMENSIONS_PER_PERSPECTIVE = 18144`    | `DUAL_DIMENSIONS_PER_PERSPECTIVE`                 |
//! | `NUM_DUAL_PSQT_HEADS = 2`                    | `NUM_DUAL_PSQT_HEADS`                             |
//! | `NUM_DUAL_PSQT_BUCKETS = 8`                  | `NUM_DUAL_PSQT_BUCKETS`                           |
//! | `DUAL_SQUARES = 81`                          | `DUAL_SQUARES`                                    |
//! | `DUAL_PIECE_TYPES = 14`                      | `DUAL_PIECE_TYPES`                                |
//! | `dual_feature_index(head,bucket,sq,piece)`   | `dual_feature_index(head,bucket,sq,piece)`        |
//! | `collect_dual_active_for_side`               | `compute_dual_king_attacker` 内 active 集合       |
//! | `DUAL_SUB_MAGIC = b"P2DUAL2\0"`              | `DUAL_PSQT_SUB_MAGIC`                             |
//! | `DUAL_ARCH_MARKER = "DUAL=1"`                | `DUAL_PSQT_ARCH_MARKER`                           |
//!
//! Phase 3a 範囲: feature 計算関数 + ShogiBoard 用利き計算 + 内部一貫性 test。
//! graph 統合 (= sparse input として train) は Phase 3b 以降。
//!
//! # 重要 (= rshogi-core との bit-exact 一致前提)
//!
//! - PieceType の番号 (1..=14) は両 crate で同一順序。 `piece_type_to_dual_idx` で
//!   `(pt as u8 - 1)` で 0..=13 にする計算は両側で同一。
//! - Square index は `file * 9 + rank` で両 crate 同一。
//! - perspective swap (Black=identity / White=`80-sq`) は両側同一。
//! - 利き計算は rshogi-core `piece_attacks_from` と同 logic を ray-walk で再現
//!   (bullet 側は bitboard module 非保有のため `[bool; 81]` を使う)。
//! - king zone は king sq + 8 近傍 (= king_effect と等価) を計算。

use super::SparseInputType;
use super::shogi_halfka_hm_pin_only::ShogiHalfKaHmPin;
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, Piece, PieceType, Square},
};

// =============================================================================
// 定数 (rshogi-core::nnue::dual_psqt と完全一致)
// =============================================================================

/// dual head の数 (= 2: attack head + defend head)。
pub const NUM_DUAL_PSQT_HEADS: usize = 2;

/// bucket 数。 LayerStacks の progress8kpabs と共有。
pub const NUM_DUAL_PSQT_BUCKETS: usize = 8;

/// 盤面マス数 (81)。
pub const DUAL_SQUARES: usize = 81;

/// piece_type 数 (Pawn..=Dragon、 King 含む 14 種)。
pub const DUAL_PIECE_TYPES: usize = 14;

/// 1 perspective あたりの dim 数 = 2 head × 8 bucket × 81 sq × 14 piece = 18,144。
pub const DUAL_DIMENSIONS_PER_PERSPECTIVE: usize =
    NUM_DUAL_PSQT_HEADS * NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES;

const _: () = {
    assert!(DUAL_DIMENSIONS_PER_PERSPECTIVE == 18_144);
};

/// ckpt section の sub_magic (`P2DUAL2\0` = 8 bytes、 King-Attacker format)。
pub const DUAL_SUB_MAGIC: &[u8; 8] = b"P2DUAL2\0";

/// ckpt section format version (現在 1)。
pub const DUAL_SECTION_VERSION: u32 = 1;

/// 2 perspective 合計 i16 entry 数 = 36,288。
pub const DUAL_TOTAL_COUNT: u32 = (2 * DUAL_DIMENSIONS_PER_PERSPECTIVE) as u32;

const _: () = {
    assert!(DUAL_TOTAL_COUNT == 36_288);
};

/// arch_str に埋め込む marker 文字列 (rshogi-core と完全一致)。
pub const DUAL_ARCH_MARKER: &str = "DUAL=1";

/// auxiliary dense 入力名 (黒視点、 legacy dense path)。
pub const DUAL_AUX_NAME_BLACK: &str = "dual_king_attacker_black";

/// auxiliary dense 入力名 (白視点、 legacy dense path)。
pub const DUAL_AUX_NAME_WHITE: &str = "dual_king_attacker_white";

/// auxiliary sparse 入力名 (黒視点 attack head)。
pub const DUAL_SPARSE_NAME_ATTACK_BLACK: &str = "dual_attack_black";

/// auxiliary sparse 入力名 (白視点 attack head)。
pub const DUAL_SPARSE_NAME_ATTACK_WHITE: &str = "dual_attack_white";

/// auxiliary sparse 入力名 (黒視点 defend head)。
pub const DUAL_SPARSE_NAME_DEFEND_BLACK: &str = "dual_defend_black";

/// auxiliary sparse 入力名 (白視点 defend head)。
pub const DUAL_SPARSE_NAME_DEFEND_WHITE: &str = "dual_defend_white";

/// 1 head あたりの dim (= 8 bucket × 81 sq × 14 piece = 9,072)。
/// dense 経路の DUAL_DIMENSIONS_PER_PERSPECTIVE (= 18,144) を head 次元で 2 等分したもの。
pub const DUAL_DIM_PER_HEAD: usize = NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES;

const _: () = {
    assert!(DUAL_DIM_PER_HEAD == 9_072);
};

/// 1 駒あたりの最大 active feature 数 (= attack + defend 両 active)。
pub const DUAL_MAX_PER_PIECE: usize = 2;

/// 1 perspective あたりの max active = (盤上駒数 ~40 駒 / 2 perspective) × 2 head = 上限 50。
///
/// 実用上は 駒数 (= ~20 駒) × 2 head = 40 entry を超えないが、 安全側で 50 にしておく。
pub const DUAL_MAX_ACTIVE_PER_SIDE: usize = 50;

/// 1 sparse input あたりの max active (= 1 head あたり)。
/// dense 経路の DUAL_MAX_ACTIVE_PER_SIDE は attack+defend 合算上限だが、 sparse 経路では
/// head ごとに別 input なので head 単独の上限 (= 25 が実用上限、 安全側で 30)。
pub const DUAL_MAX_ACTIVE_PER_HEAD: usize = 30;

// =============================================================================
// D2 拡張: piece_threat_relation / piece_defended_by の cross feature head
// =============================================================================

/// cross feature 用の sparse input 名 (= D2 追加 4 input、 perspective × {threat, defended})。
pub const DUAL_CROSS_THREAT_BLACK: &str = "dual_cross_threat_black";
pub const DUAL_CROSS_THREAT_WHITE: &str = "dual_cross_threat_white";
pub const DUAL_CROSS_DEFENDED_BLACK: &str = "dual_cross_defended_black";
pub const DUAL_CROSS_DEFENDED_WHITE: &str = "dual_cross_defended_white";

/// cross feature の bucket-aware 1 head dim (= sq 81 × own_piece 14 × target_piece 14 = 15,876)。
pub const DUAL_CROSS_DIM_PER_BUCKET: usize = DUAL_SQUARES * DUAL_PIECE_TYPES * DUAL_PIECE_TYPES;

/// cross feature の total per-head dim (= 8 bucket × 15,876 = 127,008)。
pub const DUAL_CROSS_DIM_PER_HEAD: usize = NUM_DUAL_PSQT_BUCKETS * DUAL_CROSS_DIM_PER_BUCKET;

const _: () = {
    assert!(DUAL_CROSS_DIM_PER_BUCKET == 15_876);
    assert!(DUAL_CROSS_DIM_PER_HEAD == 127_008);
};

/// cross feature の max active per side (= 自軍駒最大 ~20 × target 平均 ~3 = 安全側で 80)。
pub const DUAL_CROSS_MAX_ACTIVE_PER_HEAD: usize = 80;
pub const DUAL_CROSS_GEOMETRY_BLACK: &str = "dual_cross_geometry_black";
pub const DUAL_CROSS_GEOMETRY_WHITE: &str = "dual_cross_geometry_white";
pub const DUAL_CROSS_GEOMETRY_CLASSES: usize = 32;
pub const DUAL_CROSS_GEOMETRY_DIM_PER_PERSP: usize =
    NUM_DUAL_PSQT_BUCKETS * DUAL_PIECE_TYPES * DUAL_PIECE_TYPES * DUAL_CROSS_GEOMETRY_CLASSES;

/// cross feature の flat index。 disk layout `[bucket][own_sq][own_piece][target_piece]`。
#[inline]
pub const fn dual_cross_index(bucket: usize, own_sq: usize, own_piece: usize, target_piece: usize) -> usize {
    bucket * DUAL_CROSS_DIM_PER_BUCKET
        + own_sq * (DUAL_PIECE_TYPES * DUAL_PIECE_TYPES)
        + own_piece * DUAL_PIECE_TYPES
        + target_piece
}

// =============================================================================
// piece_type indexing (rshogi-core::nnue::dual_psqt::piece_type_to_dual_idx と同等)
// =============================================================================

/// `PieceType` enum (1..=14) を dual feature の 0-indexed (0..=13) に変換。
///
/// rshogi-core `piece_type_to_dual_idx` と完全同 mapping:
/// `pt as u8 - 1` ⇒ Pawn(1)→0, Lance→1, Knight→2, Silver→3, Bishop→4, Rook→5,
/// Gold→6, King→7, ProPawn→8, ProLance→9, ProKnight→10, ProSilver→11, Horse→12, Dragon→13。
///
/// PieceType::None (= 0) を渡した場合は `None`。
#[inline]
const fn piece_type_to_dual_idx(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::None => None,
        _ => Some((pt as u8 - 1) as usize),
    }
}

// =============================================================================
// 正規化
// =============================================================================

/// perspective ベースの square index 正規化。
///
/// Black: identity (`sq.index()`)
/// White: `80 - sq.index()` (color swap、 Half-Mirror なし)
#[inline]
const fn oriented_sq_idx(sq_index: usize, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq_index,
        Color::White => 80 - sq_index,
    }
}

/// dual feature の flat index。 rshogi-core `dual_feature_index` と同式。
///
/// `flat = head × 8 × 81 × 14 + bucket × 81 × 14 + sq × 14 + piece`。
///
/// 範囲: `0..DUAL_DIMENSIONS_PER_PERSPECTIVE` (= 0..18,144)。
#[inline]
pub const fn dual_feature_index(head: usize, bucket: usize, sq_idx: usize, piece_idx: usize) -> usize {
    head * (NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES)
        + bucket * (DUAL_SQUARES * DUAL_PIECE_TYPES)
        + sq_idx * DUAL_PIECE_TYPES
        + piece_idx
}

/// sparse 経路用の head 抜き flat index (= head ごとに別 sparse input)。
///
/// `flat = bucket × 81 × 14 + sq × 14 + piece` で範囲は `0..DUAL_DIM_PER_HEAD` (= 0..9,072)。
/// 4 sparse input (attack_b/attack_w/defend_b/defend_w) で head 次元を input 別に分離する設計。
#[inline]
pub const fn dual_sparse_index(bucket: usize, sq_idx: usize, piece_idx: usize) -> usize {
    bucket * (DUAL_SQUARES * DUAL_PIECE_TYPES) + sq_idx * DUAL_PIECE_TYPES + piece_idx
}

// =============================================================================
// king zone (= king + 8 近傍 = 9 sq) 計算
// =============================================================================

/// king sq → king zone (= king 自身 + 8 近傍) を `[bool; 81]` mask で返す。
///
/// rshogi-core `king_zone_mask` の bitboard 等価表現。
fn king_zone_mask(king_sq: Square) -> [bool; 81] {
    let mut mask = [false; 81];
    if !king_sq.is_valid() {
        return mask;
    }
    mask[king_sq.index()] = true;
    let kf = king_sq.file() as i8;
    let kr = king_sq.rank() as i8;
    for df in -1i8..=1 {
        for dr in -1i8..=1 {
            if df == 0 && dr == 0 {
                continue;
            }
            let nf = kf + df;
            let nr = kr + dr;
            if (0..9).contains(&nf) && (0..9).contains(&nr) {
                let sq = Square::new(nf as u8, nr as u8);
                mask[sq.index()] = true;
            }
        }
    }
    mask
}

// =============================================================================
// 駒の利き計算 (occupancy-aware、 rshogi-core `piece_attacks_from` 等価)
// =============================================================================

/// `[bool; 81]` で表現された occupancy mask。
type OccMask = [bool; 81];

/// piece が sq に居るときの利き (occupancy `occ` を考慮) を `[bool; 81]` で返す。
///
/// rshogi-core `piece_attacks_from(piece, sq, occupied)` と等価:
/// - 近接駒 (Pawn/Knight/Silver/Gold-like/King): 該当 dx,dy delta のみ
/// - slider (Lance/Bishop/Rook): 方向 ray walk、 occupancy で停止
/// - Horse = Bishop ray + king moves、 Dragon = Rook ray + king moves
fn piece_attacks_from(piece: Piece, sq: Square, occ: &OccMask) -> [bool; 81] {
    let mut out = [false; 81];
    let color = piece.color;
    let file = sq.file() as i8;
    let rank = sq.rank() as i8;

    /// 1 マス delta (df, dr) で着地点を out に立てる。 範囲外は無視。
    #[inline]
    fn step(out: &mut [bool; 81], file: i8, rank: i8, df: i8, dr: i8) {
        let nf = file + df;
        let nr = rank + dr;
        if (0..9).contains(&nf) && (0..9).contains(&nr) {
            let s = Square::new(nf as u8, nr as u8);
            out[s.index()] = true;
        }
    }

    /// ray を (df, dr) 方向に走らせ、 occupancy に当たるまで (含む) 立てる。
    #[inline]
    fn ray(out: &mut [bool; 81], occ: &OccMask, file: i8, rank: i8, df: i8, dr: i8) {
        let mut nf = file + df;
        let mut nr = rank + dr;
        while (0..9).contains(&nf) && (0..9).contains(&nr) {
            let s = Square::new(nf as u8, nr as u8);
            out[s.index()] = true;
            if occ[s.index()] {
                break;
            }
            nf += df;
            nr += dr;
        }
    }

    match piece.piece_type {
        PieceType::Pawn => {
            let dr = if color == Color::Black { -1 } else { 1 };
            step(&mut out, file, rank, 0, dr);
        }
        PieceType::Lance => {
            let dr = if color == Color::Black { -1 } else { 1 };
            ray(&mut out, occ, file, rank, 0, dr);
        }
        PieceType::Knight => {
            let dr = if color == Color::Black { -2 } else { 2 };
            for df in [-1i8, 1] {
                step(&mut out, file, rank, df, dr);
            }
        }
        PieceType::Silver => {
            let forward = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 5] = [(-1, forward), (0, forward), (1, forward), (-1, -forward), (1, -forward)];
            for (df, dr) in deltas {
                step(&mut out, file, rank, df, dr);
            }
        }
        PieceType::Gold | PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            let forward = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 6] = [(-1, forward), (0, forward), (1, forward), (-1, 0), (1, 0), (0, -forward)];
            for (df, dr) in deltas {
                step(&mut out, file, rank, df, dr);
            }
        }
        PieceType::Bishop => {
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                ray(&mut out, occ, file, rank, df, dr);
            }
        }
        PieceType::Rook => {
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                ray(&mut out, occ, file, rank, df, dr);
            }
        }
        PieceType::Horse => {
            // 角 ray + 上下左右 1 マス
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                ray(&mut out, occ, file, rank, df, dr);
            }
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                step(&mut out, file, rank, df, dr);
            }
        }
        PieceType::Dragon => {
            // 飛 ray + 斜め 1 マス
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                ray(&mut out, occ, file, rank, df, dr);
            }
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                step(&mut out, file, rank, df, dr);
            }
        }
        PieceType::King => {
            for df in -1i8..=1 {
                for dr in -1i8..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    step(&mut out, file, rank, df, dr);
                }
            }
        }
        PieceType::None => {}
    }

    out
}

/// occupancy mask を board から計算する。
fn occupied_mask(board: &ShogiBoard) -> OccMask {
    let mut occ = [false; 81];
    for (i, p) in board.board.iter().enumerate() {
        if p.is_some() {
            occ[i] = true;
        }
    }
    occ
}

/// 2 mask の交差判定 (any bit common)。
#[inline]
fn mask_intersects(a: &[bool; 81], b: &[bool; 81]) -> bool {
    for i in 0..81 {
        if a[i] && b[i] {
            return true;
        }
    }
    false
}

// =============================================================================
// dual feature 計算 (rshogi-core::compute_dual_king_attacker と同 algorithm)
// =============================================================================

/// `perspective` 視点で active な dual feature index を列挙する。
///
/// 戻り値 `(attack_set, defend_set, n_attack, n_defend)`:
/// - `attack_set[..n_attack]` = attack head で active な feature index (= 0..18144 範囲)
/// - `defend_set[..n_defend]` = defend head で active な feature index
///
/// rshogi-core `compute_dual_king_attacker` 内の `dual_feature_index(0/1, ...)`
/// の active 集合と **bit-exact 同値** (= 同 SFEN/同 bucket で同 index 集合)。
///
/// # アルゴリズム
///
/// 1. 自 king zone (= king + 8 近傍) と 敵 king zone を mask で取得
/// 2. perspective 側自軍駒を全 iter (board scan)
/// 3. 各駒の利き mask を occupancy-aware で計算
/// 4. 敵 king zone と交差 → attack head index emit
/// 5. 自 king zone と交差 → defend head index emit (両 active 可能)
pub fn collect_dual_active_for_side(
    board: &ShogiBoard,
    perspective: Color,
    bucket: usize,
) -> ([u32; DUAL_MAX_ACTIVE_PER_SIDE], [u32; DUAL_MAX_ACTIVE_PER_SIDE], usize, usize) {
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    let mut attack_buf = [0u32; DUAL_MAX_ACTIVE_PER_SIDE];
    let mut defend_buf = [0u32; DUAL_MAX_ACTIVE_PER_SIDE];
    let mut n_attack = 0usize;
    let mut n_defend = 0usize;

    let own_king = board.king_square(perspective);
    let enemy_king = board.king_square(perspective.opponent());
    if !own_king.is_valid() || !enemy_king.is_valid() {
        return (attack_buf, defend_buf, n_attack, n_defend);
    }

    let own_zone = king_zone_mask(own_king);
    let enemy_zone = king_zone_mask(enemy_king);
    let occ = occupied_mask(board);

    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let piece = board.piece_on(sq);
        if !piece.is_some() || piece.color != perspective {
            continue;
        }
        let Some(pt_idx) = piece_type_to_dual_idx(piece.piece_type) else {
            continue;
        };

        let attacks = piece_attacks_from(piece, sq, &occ);
        let sq_idx = oriented_sq_idx(sq.index(), perspective);

        if mask_intersects(&attacks, &enemy_zone) {
            let idx = dual_feature_index(0, bucket, sq_idx, pt_idx) as u32;
            if n_attack < DUAL_MAX_ACTIVE_PER_SIDE {
                attack_buf[n_attack] = idx;
                n_attack += 1;
            }
        }
        if mask_intersects(&attacks, &own_zone) {
            let idx = dual_feature_index(1, bucket, sq_idx, pt_idx) as u32;
            if n_defend < DUAL_MAX_ACTIVE_PER_SIDE {
                defend_buf[n_defend] = idx;
                n_defend += 1;
            }
        }
    }

    (attack_buf, defend_buf, n_attack, n_defend)
}

/// sparse 経路用の head 抜き active 集合 (= attack / defend を別 buffer に bucket-only index)。
///
/// 既存 `collect_dual_active_for_side` の sparse 版。 戻り値 index は **DUAL_DIM_PER_HEAD (9072)
/// 範囲** に正規化済 (= head 0/1 区別は呼び出し側で「どの sparse input に書くか」 で表現)。
///
/// max_active は head 単独で `DUAL_MAX_ACTIVE_PER_HEAD` (= 30)。
pub fn collect_dual_active_sparse_for_side(
    board: &ShogiBoard,
    perspective: Color,
    bucket: usize,
) -> ([u32; DUAL_MAX_ACTIVE_PER_HEAD], [u32; DUAL_MAX_ACTIVE_PER_HEAD], usize, usize) {
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    let mut attack_buf = [0u32; DUAL_MAX_ACTIVE_PER_HEAD];
    let mut defend_buf = [0u32; DUAL_MAX_ACTIVE_PER_HEAD];
    let mut n_attack = 0usize;
    let mut n_defend = 0usize;

    let own_king = board.king_square(perspective);
    let enemy_king = board.king_square(perspective.opponent());
    if !own_king.is_valid() || !enemy_king.is_valid() {
        return (attack_buf, defend_buf, n_attack, n_defend);
    }

    let own_zone = king_zone_mask(own_king);
    let enemy_zone = king_zone_mask(enemy_king);
    let occ = occupied_mask(board);

    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let piece = board.piece_on(sq);
        if !piece.is_some() || piece.color != perspective {
            continue;
        }
        let Some(pt_idx) = piece_type_to_dual_idx(piece.piece_type) else {
            continue;
        };

        let attacks = piece_attacks_from(piece, sq, &occ);
        let sq_idx = oriented_sq_idx(sq.index(), perspective);

        if mask_intersects(&attacks, &enemy_zone) {
            let idx = dual_sparse_index(bucket, sq_idx, pt_idx) as u32;
            if n_attack < DUAL_MAX_ACTIVE_PER_HEAD {
                attack_buf[n_attack] = idx;
                n_attack += 1;
            }
        }
        if mask_intersects(&attacks, &own_zone) {
            let idx = dual_sparse_index(bucket, sq_idx, pt_idx) as u32;
            if n_defend < DUAL_MAX_ACTIVE_PER_HEAD {
                defend_buf[n_defend] = idx;
                n_defend += 1;
            }
        }
    }

    (attack_buf, defend_buf, n_attack, n_defend)
}

/// D2 cross feature: piece_threat_relation + piece_defended_by の active 集合を 1 回の
/// board scan で同時に収集する。
///
/// **piece_threat_relation** (= 自軍駒 attacker → 相手駒 target):
/// - 自軍駒 own_p が own_sq に居て、 own_p の利き先 dst に相手駒 target_p が居る → emit
/// - index = dual_cross_index(bucket, oriented(own_sq), own_p_idx, target_p_idx)
///
/// **piece_defended_by** (= 自軍駒 own → 守ってる 自軍駒 defender):
/// - 自軍駒 own_p が own_sq に居て、 別の自軍駒 defender_p が defender_sq から own_sq を攻撃 → emit
/// - 実装最適化: 各駒の利き計算結果を 1 回保存し、 own_sq への利きを post-process で集約
/// - index = dual_cross_index(bucket, oriented(own_sq), own_p_idx, defender_p_idx)
///
/// 戻り値: (threat_buf, defended_buf, n_threat, n_defended)。 max は
/// `DUAL_CROSS_MAX_ACTIVE_PER_HEAD` (= 80)。
pub fn collect_dual_cross_for_side(
    board: &ShogiBoard,
    perspective: Color,
    bucket: usize,
) -> ([u32; DUAL_CROSS_MAX_ACTIVE_PER_HEAD], [u32; DUAL_CROSS_MAX_ACTIVE_PER_HEAD], usize, usize) {
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    let mut threat_buf = [0u32; DUAL_CROSS_MAX_ACTIVE_PER_HEAD];
    let mut defended_buf = [0u32; DUAL_CROSS_MAX_ACTIVE_PER_HEAD];
    let mut n_threat = 0usize;
    let mut n_defended = 0usize;

    let occ = occupied_mask(board);

    // 1 pass: 自軍駒の (own_sq, own_p, attacks) を集める
    // ここで attacks は board scan で 1 回計算、 後段で再利用する
    let mut own_pieces: [(u8, Piece, [bool; 81]); 40] = unsafe { core::mem::zeroed() };
    let mut n_own = 0usize;

    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let piece = board.piece_on(sq);
        if !piece.is_some() || piece.color != perspective {
            continue;
        }
        if piece_type_to_dual_idx(piece.piece_type).is_none() {
            continue;
        }
        if n_own < 40 {
            let attacks = piece_attacks_from(piece, sq, &occ);
            own_pieces[n_own] = (sq_raw, piece, attacks);
            n_own += 1;
        }
    }

    // piece_threat_relation: own → target (= board scan、 各 own の attacks 内で相手駒検出)
    for k in 0..n_own {
        let (own_sq_raw, own_piece, attacks) = own_pieces[k];
        let own_sq = Square(own_sq_raw);
        let own_pt_idx = piece_type_to_dual_idx(own_piece.piece_type).unwrap();
        let own_sq_oriented = oriented_sq_idx(own_sq.index(), perspective);

        for dst_raw in 0..81u8 {
            if !attacks[dst_raw as usize] {
                continue;
            }
            let dst = Square(dst_raw);
            let target = board.piece_on(dst);
            if !target.is_some() || target.color == perspective {
                continue; // 相手駒のみ
            }
            let Some(target_pt_idx) = piece_type_to_dual_idx(target.piece_type) else {
                continue;
            };
            let idx = dual_cross_index(bucket, own_sq_oriented, own_pt_idx, target_pt_idx) as u32;
            if n_threat < DUAL_CROSS_MAX_ACTIVE_PER_HEAD {
                threat_buf[n_threat] = idx;
                n_threat += 1;
            }
        }
    }

    // piece_defended_by: own (own_sq) → defender_p (= 別 own 駒、 own_sq を攻撃)
    // own_pieces のうち、 各 own_sq に対して 他の own 駒の attacks をチェック
    for i in 0..n_own {
        let (own_sq_raw_i, own_piece_i, _) = own_pieces[i];
        let own_sq_i = Square(own_sq_raw_i);
        let own_pt_idx_i = piece_type_to_dual_idx(own_piece_i.piece_type).unwrap();
        let own_sq_oriented_i = oriented_sq_idx(own_sq_i.index(), perspective);

        for j in 0..n_own {
            if i == j {
                continue;
            }
            let (_def_sq_raw, def_piece, def_attacks) = own_pieces[j];
            if !def_attacks[own_sq_raw_i as usize] {
                continue;
            }
            let def_pt_idx = piece_type_to_dual_idx(def_piece.piece_type).unwrap();
            let idx = dual_cross_index(bucket, own_sq_oriented_i, own_pt_idx_i, def_pt_idx) as u32;
            if n_defended < DUAL_CROSS_MAX_ACTIVE_PER_HEAD {
                defended_buf[n_defended] = idx;
                n_defended += 1;
            }
        }
    }

    (threat_buf, defended_buf, n_threat, n_defended)
}

// =============================================================================
// dense fill (training-side: sparse -> dense per perspective, 18,144 dim、 legacy path)
// =============================================================================

/// `perspective` 視点で dual feature を dense vector (= 18,144 dim) に展開する。
///
/// 各 active feature index に +1.0 を立てる (= multi-hot)。 同 index に複数発火は
/// 物理的に不可能 (= 同 (head, sq, piece) は 1 駒に 1 回しか発火しない) なので
/// 衝突は起きない。
///
/// out.len() == DUAL_DIMENSIONS_PER_PERSPECTIVE 必須。
pub fn fill_dual_features_for_side(pos: &PackedSfenValue, side: Color, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        DUAL_DIMENSIONS_PER_PERSPECTIVE,
        "dual feature 出力スロットは {} 次元",
        DUAL_DIMENSIONS_PER_PERSPECTIVE,
    );
    // ★ NPS 最適化 (2026-05-26): bullet loader (= loader.rs::prepare) は batch ごとに
    // `vec![0.0; dims * batch_size]` で auxiliary dense buffer を新規確保し、 既に zero
    // 初期化済の状態で `fill_auxiliary_dense` を呼ぶ。 ここで再度 zero clear すると
    // 18,144 × 8192 = 149M float の二重 memset が走り、 NPS が ~6x 低下する。
    // 既存 zero 前提で active index のみ書き込む。

    let board = ShogiBoard::from_packed_sfen(pos);

    // Phase 3b-graph: sample の actual bucket (= progress8kpabs) を計算して、 active
    // feature index に bucket を encode する。 各 sample は 1 bucket のみ active 化、
    // 他 bucket の weight entry は gradient 0 (= weight 不変)。
    //
    // bucket は `ShogiProgressKPAbs` の global 静的 weights が load 済前提
    // (= shogi_layerstack.rs で `--progress-coeff progress.bin` を指定するときに load)。
    // weight 未 load 時は zero weights が返るため bucket = 0 となり 旧挙動と等価。
    let bucket = compute_dual_bucket(pos) as usize;
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    // attack head の active index は +1.0、 defend head の active index は -1.0 を
    // 立てる。 これにより graph forward `w · input` が `sum(w_attack) - sum(w_defend)`
    // となり、 rshogi-core `compute_dual_king_attacker` の `(attack - defend)` と
    // 一致する (= 両 perspective を `dual_b_out + dual_w_out` で合算する設計)。
    let (att, def, n_a, n_d) = collect_dual_active_for_side(&board, side, bucket);
    for &idx in att.iter().take(n_a) {
        out[idx as usize] = 1.0;
    }
    for &idx in def.iter().take(n_d) {
        out[idx as usize] = -1.0;
    }
}

/// dual feature を emit する際に使う bucket index。
///
/// progress8kpabs (= LayerStacks の bucket 算出 = `ShogiProgressKPAbs::bucket`) と同じ
/// 値を返す。 `ShogiProgressKPAbs` は ZST + 静的 weight 参照のため呼出 cost は無視可。
///
/// 訓練 entry point (= shogi_layerstack.rs) で progress.bin を load しない場合は
/// 0 (= 序盤 bucket) を返す (= 旧挙動と一致)。
#[inline]
fn compute_dual_bucket(pos: &PackedSfenValue) -> u8 {
    use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
    let bucket_impl = ShogiProgressKPAbs;
    OutputBuckets::<PackedSfenValue>::bucket(&bucket_impl, pos)
}

// =============================================================================
// pub helpers for example/training-side glue
// =============================================================================

/// `auxiliary_dense_dims` で追加すべき dual 関連 dense input 定義。
///
/// 戻り値 `[(name, dims); 2]`。 `ShogiHalfKaHmKingAttackerDual` の
/// `auxiliary_dense_dims` から呼び出される。
pub fn dual_aux_entries() -> [(&'static str, usize); 2] {
    [(DUAL_AUX_NAME_BLACK, DUAL_DIMENSIONS_PER_PERSPECTIVE), (DUAL_AUX_NAME_WHITE, DUAL_DIMENSIONS_PER_PERSPECTIVE)]
}

/// dispatch helper for `fill_auxiliary_dense`。
///
/// `name == DUAL_AUX_NAME_BLACK / WHITE` の場合に dual fill を実行し、 `true` を返す。
pub fn try_fill_dual_aux(name: &str, pos: &PackedSfenValue, out: &mut [f32]) -> bool {
    match name {
        DUAL_AUX_NAME_BLACK => {
            fill_dual_features_for_side(pos, Color::Black, out);
            true
        }
        DUAL_AUX_NAME_WHITE => {
            fill_dual_features_for_side(pos, Color::White, out);
            true
        }
        _ => false,
    }
}

// =============================================================================
// SparseInputType wrapper: HalfKA_hm + HandCount + Pin (任意) + Dual King-Attacker
// =============================================================================

/// HalfKA_hm + HandCount + Pin (任意) + Dual King-Attacker dense 補助入力 wrapper。
///
/// sparse 部 (= HalfKA_hm 73,305 dim) は inner `ShogiHalfKaHmPin` に delegate。
/// dual King-Attacker は `dual_enabled = true` のとき `auxiliary_dense_dims` に
/// `dual_king_attacker_{black,white}` (各 18,144 dim) を追加し、 fill 時に
/// `fill_dual_features_for_side` で dense vector を生成する。
#[derive(Clone, Debug)]
pub struct ShogiHalfKaHmKingAttackerDual {
    inner: ShogiHalfKaHmPin,
    /// Dual head (= cross feature D2) を有効化するか。
    dual_enabled: bool,
    /// king_attacker head (= attack/defend 4 sparse input) を有効化するか。
    /// default = true (= 後方互換)。
    /// arm 5 設計: PSQT で代替可能なため削除可能 (= false に設定)。
    enable_king_attacker: bool,
    /// D2 cross feature の **head 1 (= piece_defended_by / defended)** を有効化するか。
    /// default = true (= 後方互換)。 arm 6 (= 2026-05-27): cosine similarity
    /// 解析で head 1 は無方向 random walk + weight decay 縮小 (= 訓練資源の
    /// 無駄) と判明、 rshogi-core 推論側でも production path から削除済。
    /// false にすると `dual_cross_defended_{black,white}` 2 sparse input を
    /// 削除し、 graph 側 `dual_w_defended_*` weight tensor も追加されない。
    enable_dual_cross_defended: bool,
    /// arm 11 (= 2026-05-28): KingRing Threat PSQT scalar head を有効化するか。
    ///
    /// `true` で:
    ///   - `auxiliary_sparse_dims` に `kingring_black` / `kingring_white` 2 entry 追加
    ///     (各 dim = 9,984、 max_active = 24)
    ///   - graph 側 `kingring_w_black` / `kingring_w_white` 2 weight tensor (= `(1, 9984)`)
    ///     を別途追加し、 末端で `score += kingring_b_out + kingring_w_out` 加算合成
    ///   - quantised.bin 末尾に `P2KRING\0` section を append (= 19,968 i16 = 39,952 byte)
    ///
    /// 旧 ckpt から resume する場合は kingring weight が optimiser state に存在しないため
    /// random init (= bullet の default = Zeroed) で start。 後方互換: default false で
    /// 旧挙動 (= kingring 完全 OFF)。
    enable_kingring: bool,
    /// arm 12 (= 2026-05-29): PairShape (= Piece-Pair Shape Table) scalar head を有効化するか。
    ///
    /// `true` で:
    ///   - `auxiliary_sparse_dims` に `pairshape_black` / `pairshape_white` 2 entry 追加
    ///     (各 dim = 1,536 (= 8 bucket × 192 flat)、 max_active = 256)
    ///   - graph 側 `pairshape_w_black` / `pairshape_w_white` 2 weight tensor (= `(1, 1536)`)
    ///     を別途追加し、 末端で `score += pairshape_b_out - pairshape_w_out` 反対称加算合成
    ///   - quantised.bin 末尾に `P2PAIR\0\0` section を append (= 3,072 i16 = 6,160 byte)
    ///
    /// 後方互換: default false で arm 5/6/7/8/11 と一致。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/pairshape.rs` 参照。
    enable_pairshape: bool,
    /// 2026-06-06 追加: Activity-Risk Interaction Head を有効化するか。
    ///
    /// `true` で:
    ///   - `auxiliary_sparse_dims` に `activity_risk_black` / `activity_risk_white` 2 entry 追加
    ///     (各 dim = 2,048 (= 8 bucket × 256 flat)、 max_active = 1)
    ///   - graph 側 `activity_risk_w_black` / `activity_risk_w_white` 2 weight tensor
    ///     (= `(1, 2048)`) を別途追加し、 末端で `score += ar_b_out - ar_w_out` 反対称加算合成
    ///   - quantised.bin 末尾に `P2ACTR\0\0` section を append (= 4,096 i16 = 8,208 byte)
    ///
    /// 後方互換: default false で arm 12 base ckpt と一致。 sb 240 から resume で head
    /// zeroed init + lr_mult x10 で head 育成。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/activity_risk.rs` 参照。
    enable_activity_risk: bool,

    /// 2026-06-09 King Escape Direction (KED) head の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `king_escape_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な 玉脱出方向 class × bucket flat 値を埋める。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/king_escape.rs` 参照。
    enable_king_escape: bool,

    /// 2026-06-15 King Danger (KDGR) head の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `king_danger_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な 自玉危険 class × bucket flat 値を埋める。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/king_danger.rs` 参照。
    enable_king_danger: bool,

    /// 2026-06-16 Attack Development (ADEV) head の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `attack_dev_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な 敵 slider 照準 class × bucket flat 値を埋める。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/attack_dev.rs` 参照。
    enable_attack_dev: bool,

    /// 2026-06-28 Slider X-ray (SXRY) head の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `slider_xray_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な 敵 slider 潜在照準 (= between 1 枚) class
    /// × bucket flat 値を埋める。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/slider_xray.rs` 参照。
    enable_slider_xray: bool,

    /// 2026-06-16 King Mobility (KMOB) head の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `king_mobility_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な 自玉の安全逃げ場 class (= 0..=8) × bucket
    /// flat 値を埋める。 守備系特徴 (= ADEV / ARS1 と直交)。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/king_mobility.rs` 参照。
    enable_king_mobility: bool,

    /// 2026-06-16 Drop Pressure (DROP) head の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `drop_pressure_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な 相手持駒投下脅威 class (= 0..=7) × bucket
    /// flat 値を埋める。 入力源は相手持駒 (= 盤上駒のみの既存 head と直交)。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/drop_pressure.rs` 参照。
    enable_drop_pressure: bool,
    /// Endgame Tempo (EGT) 結合 head の sparse input dispatch。
    enable_endgame_tempo: bool,

    /// 2026-06-09 Attack Relay Head (ARH) の sparse input dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `attack_relay_{black,white}` の 2 entry が追加され、
    /// fill 経路で per-perspective 1 active な (contact, reserve, relay) × bucket flat 値を埋める。
    ///
    /// 詳細は p2shogi 側 `docs/attack_relay_head_design.md` および
    /// `crates/rshogi-core/src/nnue/attack_relay.rs`。
    enable_attack_relay: bool,

    /// 2026-06-10 ARH v2 (= A5 仕様、 attackers / overlap / supported) の sparse input
    /// dispatch を ON にする。 v1 (= contact / reserve / relay) と並列、 別 section magic
    /// `P2AREL2`、 dim 同 1024。
    enable_attack_relay_v2: bool,
    /// Full Threat scalar residual head。
    enable_scalar_threat: bool,
    /// Scalar ThreatとDual Crossを同時に入力する。
    enable_scalar_threat_coexist_dual_cross: bool,
    enable_dual_cross_geometry: bool,

    /// 2026-06-30 King Zone Threat / 持駒打ち込み (KZT_HAND) sparse head の dispatch を ON にする。
    ///
    /// ON 時 `auxiliary_sparse_dims` に `kzt_{black,white}` の 2 entry が追加され、 fill 経路で
    /// 推論側 `rshogi-core::nnue::kzt::collect_hand_features` を直接呼び active feature (= 敵玉
    /// 5x5 zone への合法持駒打ち) を local index 化して emit する。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/kzt.rs` 参照。
    enable_kzt: bool,
}

impl Default for ShogiHalfKaHmKingAttackerDual {
    fn default() -> Self {
        Self {
            inner: ShogiHalfKaHmPin::default(),
            dual_enabled: false,
            // default = true: 既存 ckpt + 旧 arm 2/3/4 互換のため king_attacker は ON
            enable_king_attacker: true,
            // default = true: 旧 arm 5 ckpt (= P2DUALX section) との後方互換。
            // arm 6 以降は `--dual-disable-cross-defended` で OFF 推奨。
            enable_dual_cross_defended: true,
            // default = false: 旧 ckpt 後方互換 (= kingring section なし)。 arm 11 で
            // `--enable-kingring` ON にすると有効化。
            enable_kingring: false,
            // default = false: 旧 ckpt 後方互換 (= P2PAIR section なし)。 arm 12 で
            // `--enable-pairshape` ON にすると有効化。
            enable_pairshape: false,
            // default = false: 旧 ckpt 後方互換 (= P2ACTR section なし)。 sb 240 から
            // `--enable-activity-risk` ON にすると有効化。
            enable_activity_risk: false,
            enable_king_escape: false,
            enable_king_danger: false,
            enable_attack_dev: false,
            enable_slider_xray: false,
            enable_king_mobility: false,
            enable_drop_pressure: false,
            enable_endgame_tempo: false,
            enable_attack_relay: false,
            enable_attack_relay_v2: false,
            enable_scalar_threat: false,
            enable_scalar_threat_coexist_dual_cross: false,
            enable_dual_cross_geometry: false,
            // default = false: 旧 ckpt 後方互換 (= P2KZT section なし)。 2026-06-30 で
            // `--enable-kzt-hand` ON にすると有効化。
            enable_kzt: false,
        }
    }
}

impl ShogiHalfKaHmKingAttackerDual {
    /// Full Threat scalar residual headをON/OFFする。
    pub fn with_scalar_threat_enabled(mut self, enabled: bool) -> Self {
        self.enable_scalar_threat = enabled;
        self
    }

    /// Scalar Threat有効時もDual Crossを残す。
    pub fn with_scalar_threat_coexist_dual_cross(mut self, enabled: bool) -> Self {
        self.enable_scalar_threat_coexist_dual_cross = enabled;
        self
    }

    /// Dual Cross Geometry残差入力をON/OFFする。
    pub fn with_dual_cross_geometry_enabled(mut self, enabled: bool) -> Self {
        self.enable_dual_cross_geometry = enabled;
        self
    }

    /// king_attacker head を OFF にする builder (= arm 5、 cross のみ採用)。
    pub fn with_king_attacker_enabled(mut self, enabled: bool) -> Self {
        self.enable_king_attacker = enabled;
        self
    }

    /// king_attacker enabled の状態を返す。
    pub fn king_attacker_enabled(&self) -> bool {
        self.enable_king_attacker
    }

    /// D2 cross feature の head 1 (= defended) を OFF にする builder (= arm 6 以降)。
    pub fn with_dual_cross_defended_enabled(mut self, enabled: bool) -> Self {
        self.enable_dual_cross_defended = enabled;
        self
    }

    /// D2 cross defended enabled の状態を返す。
    pub fn dual_cross_defended_enabled(&self) -> bool {
        self.enable_dual_cross_defended
    }

    /// arm 11 KingRing scalar head を ON/OFF にする builder。
    pub fn with_kingring_enabled(mut self, enabled: bool) -> Self {
        self.enable_kingring = enabled;
        self
    }

    /// KingRing enabled の状態を返す。
    pub fn kingring_enabled(&self) -> bool {
        self.enable_kingring
    }

    /// arm 12 PairShape scalar head を ON/OFF にする builder。
    pub fn with_pairshape_enabled(mut self, enabled: bool) -> Self {
        self.enable_pairshape = enabled;
        self
    }

    /// PairShape enabled の状態を返す。
    pub fn pairshape_enabled(&self) -> bool {
        self.enable_pairshape
    }

    /// 2026-06-06 Activity-Risk Interaction Head を ON/OFF にする builder。
    pub fn with_activity_risk_enabled(mut self, enabled: bool) -> Self {
        self.enable_activity_risk = enabled;
        self
    }

    /// Activity-Risk enabled の状態を返す。
    pub fn activity_risk_enabled(&self) -> bool {
        self.enable_activity_risk
    }

    /// 2026-06-09 King Escape Direction (KED) head を ON/OFF にする builder。
    pub fn with_king_escape_enabled(mut self, enabled: bool) -> Self {
        self.enable_king_escape = enabled;
        self
    }

    /// KED enabled の状態を返す。
    pub fn king_escape_enabled(&self) -> bool {
        self.enable_king_escape
    }

    /// 2026-06-15 King Danger (KDGR) head を ON/OFF にする builder。
    pub fn with_king_danger_enabled(mut self, enabled: bool) -> Self {
        self.enable_king_danger = enabled;
        self
    }

    /// KDGR enabled の状態を返す。
    pub fn king_danger_enabled(&self) -> bool {
        self.enable_king_danger
    }

    /// 2026-06-16 Attack Development (ADEV) head を ON/OFF にする builder。
    pub fn with_attack_dev_enabled(mut self, enabled: bool) -> Self {
        self.enable_attack_dev = enabled;
        self
    }

    /// ADEV enabled の状態を返す。
    pub fn attack_dev_enabled(&self) -> bool {
        self.enable_attack_dev
    }

    /// 2026-06-28 Slider X-ray (SXRY) head を ON/OFF にする builder。
    pub fn with_slider_xray_enabled(mut self, enabled: bool) -> Self {
        self.enable_slider_xray = enabled;
        self
    }

    /// SXRY enabled の状態を返す。
    pub fn slider_xray_enabled(&self) -> bool {
        self.enable_slider_xray
    }

    /// 2026-06-30 King Zone Threat (KZT_HAND) head を ON/OFF にする builder。
    pub fn with_kzt_enabled(mut self, enabled: bool) -> Self {
        self.enable_kzt = enabled;
        self
    }

    /// KZT_HAND enabled の状態を返す。
    pub fn kzt_enabled(&self) -> bool {
        self.enable_kzt
    }

    /// 2026-06-16 King Mobility (KMOB) head を ON/OFF にする builder。
    pub fn with_king_mobility_enabled(mut self, enabled: bool) -> Self {
        self.enable_king_mobility = enabled;
        self
    }

    /// KMOB enabled の状態を返す。
    pub fn king_mobility_enabled(&self) -> bool {
        self.enable_king_mobility
    }

    /// 2026-06-16 Drop Pressure (DROP) head を ON/OFF にする builder。
    pub fn with_drop_pressure_enabled(mut self, enabled: bool) -> Self {
        self.enable_drop_pressure = enabled;
        self
    }

    /// DROP enabled の状態を返す。
    pub fn drop_pressure_enabled(&self) -> bool {
        self.enable_drop_pressure
    }

    /// Endgame Tempo (EGT) head を ON/OFF にする builder。
    pub fn with_endgame_tempo_enabled(mut self, enabled: bool) -> Self {
        self.enable_endgame_tempo = enabled;
        self
    }

    /// EGT enabled の状態を返す。
    pub fn endgame_tempo_enabled(&self) -> bool {
        self.enable_endgame_tempo
    }

    /// 2026-06-09 Attack Relay Head (ARH) を ON/OFF にする builder。
    pub fn with_attack_relay_enabled(mut self, enabled: bool) -> Self {
        self.enable_attack_relay = enabled;
        self
    }

    /// ARH enabled の状態を返す。
    pub fn attack_relay_enabled(&self) -> bool {
        self.enable_attack_relay
    }

    /// 2026-06-10 ARH v2 (= A5 仕様) を ON/OFF にする builder。
    pub fn with_attack_relay_v2_enabled(mut self, enabled: bool) -> Self {
        self.enable_attack_relay_v2 = enabled;
        self
    }

    /// ARH v2 enabled の状態を返す。
    pub fn attack_relay_v2_enabled(&self) -> bool {
        self.enable_attack_relay_v2
    }
}

impl ShogiHalfKaHmKingAttackerDual {
    /// dual_enabled off (= inner と同等) で構築。
    pub fn new() -> Self {
        Self::default()
    }

    /// `pin_aware` を inner に伝播する builder。
    pub fn with_pin_aware(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_pin_aware(enabled);
        self
    }

    /// dual King-Attacker を有効化する builder。
    pub fn with_dual_enabled(mut self, enabled: bool) -> Self {
        self.dual_enabled = enabled;
        self
    }

    /// dual_enabled の状態を返す。
    pub fn dual_enabled(&self) -> bool {
        self.dual_enabled
    }

    /// pin_aware の状態を返す (inner に delegate)。
    pub fn pin_aware(&self) -> bool {
        self.inner.pin_aware()
    }
}

impl SparseInputType for ShogiHalfKaHmKingAttackerDual {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn max_active(&self) -> usize {
        self.inner.max_active()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.inner.map_features(pos, f);
    }

    fn shorthand(&self) -> String {
        let dual_part = if self.dual_enabled { "+dual" } else { "" };
        format!("{}{}", self.inner.shorthand(), dual_part)
    }

    fn description(&self) -> String {
        format!(
            "{}{}",
            self.inner.description(),
            if self.dual_enabled { " + Dual King-Attacker (18,144 × 2 perspective)" } else { "" }
        )
    }

    fn hand_count_dims(&self) -> Option<usize> {
        self.inner.hand_count_dims()
    }

    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        self.inner.fill_hand_count(pos, out);
    }

    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        // dual は sparse 経路 (= auxiliary_sparse_dims) で渡すため、 dense では追加しない。
        // hand_count + pin 系のみ。 inner から継承。
        self.inner.auxiliary_dense_dims()
    }

    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        // dual の dense fill は廃止 (= sparse 経路へ移行)、 inner にのみ delegate。
        self.inner.fill_auxiliary_dense(name, pos, out);
    }

    fn auxiliary_sparse_dims(&self) -> Vec<(String, usize, usize)> {
        let mut dims: Vec<(String, usize, usize)> = self.inner.auxiliary_sparse_dims();
        if self.dual_enabled {
            if self.enable_king_attacker {
                // sparse 経路 (= 4 input 分離): attack/defend head を別 sparse input にして、
                // graph 側で `score = w_a · attack - w_d · defend` で符号を分離する。
                // 各 input は dim = DUAL_DIM_PER_HEAD (9072)、 max_active = head 単独で 30。
                dims.push((DUAL_SPARSE_NAME_ATTACK_BLACK.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
                dims.push((DUAL_SPARSE_NAME_DEFEND_BLACK.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
                dims.push((DUAL_SPARSE_NAME_ATTACK_WHITE.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
                dims.push((DUAL_SPARSE_NAME_DEFEND_WHITE.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
            }
            // D2 cross feature: piece_threat_relation (+ piece_defended_by if enabled)
            // × 2 perspective = 2 or 4 input。
            // 各 input は dim = DUAL_CROSS_DIM_PER_HEAD (127,008)、 max_active = 80。
            // arm 6 以降 (= 2026-05-27): defended head (= head 1) は無方向 random walk
            // と判明、 enable_dual_cross_defended = false で skip (= 2 input のみ追加)。
            if !self.enable_scalar_threat || self.enable_scalar_threat_coexist_dual_cross {
                dims.push((
                    DUAL_CROSS_THREAT_BLACK.to_string(),
                    DUAL_CROSS_DIM_PER_HEAD,
                    DUAL_CROSS_MAX_ACTIVE_PER_HEAD,
                ));
                if self.enable_dual_cross_defended {
                    dims.push((
                        DUAL_CROSS_DEFENDED_BLACK.to_string(),
                        DUAL_CROSS_DIM_PER_HEAD,
                        DUAL_CROSS_MAX_ACTIVE_PER_HEAD,
                    ));
                }
            }
            dims.push((DUAL_CROSS_THREAT_WHITE.to_string(), DUAL_CROSS_DIM_PER_HEAD, DUAL_CROSS_MAX_ACTIVE_PER_HEAD));
            if self.enable_dual_cross_defended {
                dims.push((
                    DUAL_CROSS_DEFENDED_WHITE.to_string(),
                    DUAL_CROSS_DIM_PER_HEAD,
                    DUAL_CROSS_MAX_ACTIVE_PER_HEAD,
                ));
            }
        }
        if self.enable_scalar_threat {
            dims.push((
                super::SCALAR_THREAT_SPARSE_NAME_BLACK.to_string(),
                super::SCALAR_THREAT_WEIGHT_COUNT,
                super::MAX_ACTIVE_THREAT_FEATURES,
            ));
            dims.push((
                super::SCALAR_THREAT_SPARSE_NAME_WHITE.to_string(),
                super::SCALAR_THREAT_WEIGHT_COUNT,
                super::MAX_ACTIVE_THREAT_FEATURES,
            ));
        }
        if self.enable_dual_cross_geometry {
            dims.push((
                DUAL_CROSS_GEOMETRY_BLACK.to_string(),
                DUAL_CROSS_GEOMETRY_DIM_PER_PERSP,
                DUAL_CROSS_MAX_ACTIVE_PER_HEAD,
            ));
            dims.push((
                DUAL_CROSS_GEOMETRY_WHITE.to_string(),
                DUAL_CROSS_GEOMETRY_DIM_PER_PERSP,
                DUAL_CROSS_MAX_ACTIVE_PER_HEAD,
            ));
        }
        // arm 11 KingRing scalar head: 2 sparse input (black / white)、 各 dim 9984、
        // max_active 24。 dual_enabled に依存せず単独 ON 可能 (= arm 11 = arm 5 + kingring 想定)。
        if self.enable_kingring {
            dims.push((KINGRING_SPARSE_NAME_BLACK.to_string(), KINGRING_DIM_PER_PERSP, KINGRING_MAX_ACTIVE_PER_PERSP));
            dims.push((KINGRING_SPARSE_NAME_WHITE.to_string(), KINGRING_DIM_PER_PERSP, KINGRING_MAX_ACTIVE_PER_PERSP));
        }
        // arm 12 PairShape scalar head: 2 sparse input (black / white)、 各 dim 1536、
        // max_active 256 (= multiset 重複保持で最大 active = 256 件)。
        // dual_enabled に依存せず単独 ON 可能。
        if self.enable_pairshape {
            dims.push((
                PAIRSHAPE_SPARSE_NAME_BLACK.to_string(),
                PAIRSHAPE_DIM_PER_PERSP,
                PAIRSHAPE_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                PAIRSHAPE_SPARSE_NAME_WHITE.to_string(),
                PAIRSHAPE_DIM_PER_PERSP,
                PAIRSHAPE_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-06 Activity-Risk Interaction Head: 2 sparse input (black / white)、
        // 各 dim 2048 (= 8 bucket × 256)、 max_active 1 (= per perspective で 1 bucket
        // × 1 (activity, pressure, shield) tuple のみ発火)。
        if self.enable_activity_risk {
            dims.push((
                super::ACTIVITY_RISK_SPARSE_NAME_BLACK.to_string(),
                super::ACTIVITY_RISK_DIM_PER_PERSP,
                super::ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::ACTIVITY_RISK_SPARSE_NAME_WHITE.to_string(),
                super::ACTIVITY_RISK_DIM_PER_PERSP,
                super::ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP,
            ));
            // 2026-06-08 v3: tactical sparse input 2 軸 (= per persp 1 active、 dim 512)
            dims.push((
                super::ACTIVITY_RISK_V3_TACTICAL_NAME_BLACK.to_string(),
                super::ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP,
                1,
            ));
            dims.push((
                super::ACTIVITY_RISK_V3_TACTICAL_NAME_WHITE.to_string(),
                super::ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP,
                1,
            ));
        }
        // 2026-06-09 King Escape Direction (KED): 2 sparse input (black / white)、
        // 各 dim 32 (= 8 bucket × 4 class)、 max_active 1 (= per perspective で 1 entry)
        if self.enable_king_escape {
            dims.push((
                super::KING_ESCAPE_SPARSE_NAME_BLACK.to_string(),
                super::KING_ESCAPE_DIM_PER_PERSP,
                super::KING_ESCAPE_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::KING_ESCAPE_SPARSE_NAME_WHITE.to_string(),
                super::KING_ESCAPE_DIM_PER_PERSP,
                super::KING_ESCAPE_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-15 King Danger (KDGR): 2 sparse input (black / white)、
        // 各 dim 64 (= 8 bucket × 8 class)、 max_active 1 (= per perspective で 1 entry)
        if self.enable_king_danger {
            dims.push((
                super::KING_DANGER_SPARSE_NAME_BLACK.to_string(),
                super::KING_DANGER_DIM_PER_PERSP,
                super::KING_DANGER_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::KING_DANGER_SPARSE_NAME_WHITE.to_string(),
                super::KING_DANGER_DIM_PER_PERSP,
                super::KING_DANGER_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-16 Attack Development (ADEV): 2 sparse input (black / white)、
        // 各 dim 64 (= 8 bucket × 8 class)、 max_active 1 (= per perspective で 1 entry)
        if self.enable_attack_dev {
            dims.push((
                super::ATTACK_DEV_SPARSE_NAME_BLACK.to_string(),
                super::ATTACK_DEV_DIM_PER_PERSP,
                super::ATTACK_DEV_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::ATTACK_DEV_SPARSE_NAME_WHITE.to_string(),
                super::ATTACK_DEV_DIM_PER_PERSP,
                super::ATTACK_DEV_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-28 Slider X-ray (SXRY): 2 sparse input (black / white)、
        // 各 dim 64 (= 8 bucket × 8 class)、 max_active 1 (= per perspective で 1 entry)
        if self.enable_slider_xray {
            dims.push((
                super::SLIDER_XRAY_SPARSE_NAME_BLACK.to_string(),
                super::SLIDER_XRAY_DIM_PER_PERSP,
                super::SLIDER_XRAY_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::SLIDER_XRAY_SPARSE_NAME_WHITE.to_string(),
                super::SLIDER_XRAY_DIM_PER_PERSP,
                super::SLIDER_XRAY_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-07-01 King Zone Threat (KZT_HAND) factored v1: 5 テーブル × 2 persp = 10 sparse
        // input。 各 table (core/region/bucket/persp/bias) は dim が異なる (= KZT_TABLE_DIMS)、
        // max_active はいずれも 175 (= 各 feature が全 table に 1 index を emit)。
        if self.enable_kzt {
            for t in 0..super::KZT_NUM_TABLES {
                dims.push((
                    super::KZT_TABLE_BLACK_NAMES[t].to_string(),
                    super::KZT_TABLE_DIMS[t],
                    super::KZT_MAX_ACTIVE_PER_PERSP,
                ));
                dims.push((
                    super::KZT_TABLE_WHITE_NAMES[t].to_string(),
                    super::KZT_TABLE_DIMS[t],
                    super::KZT_MAX_ACTIVE_PER_PERSP,
                ));
            }
        }
        // 2026-06-16 King Mobility (KMOB): 2 sparse input (black / white)、
        // 各 dim 72 (= 8 bucket × 9 class)、 max_active 1 (= per perspective で 1 entry)
        if self.enable_king_mobility {
            dims.push((
                super::KING_MOBILITY_SPARSE_NAME_BLACK.to_string(),
                super::KING_MOBILITY_DIM_PER_PERSP,
                super::KING_MOBILITY_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::KING_MOBILITY_SPARSE_NAME_WHITE.to_string(),
                super::KING_MOBILITY_DIM_PER_PERSP,
                super::KING_MOBILITY_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-16 Drop Pressure (DROP): 2 sparse input (black / white)、
        // 各 dim 64 (= 8 bucket × 8 class)、 max_active 1 (= per perspective で 1 entry)
        if self.enable_drop_pressure {
            dims.push((
                super::DROP_PRESSURE_SPARSE_NAME_BLACK.to_string(),
                super::DROP_PRESSURE_DIM_PER_PERSP,
                super::DROP_PRESSURE_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::DROP_PRESSURE_SPARSE_NAME_WHITE.to_string(),
                super::DROP_PRESSURE_DIM_PER_PERSP,
                super::DROP_PRESSURE_MAX_ACTIVE_PER_PERSP,
            ));
        }
        if self.enable_endgame_tempo {
            dims.push((
                super::ENDGAME_TEMPO_SPARSE_NAME_BLACK.to_string(),
                super::ENDGAME_TEMPO_DIM_PER_PERSP,
                super::ENDGAME_TEMPO_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::ENDGAME_TEMPO_SPARSE_NAME_WHITE.to_string(),
                super::ENDGAME_TEMPO_DIM_PER_PERSP,
                super::ENDGAME_TEMPO_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-09 Attack Relay Head (ARH): 2 sparse input (black / white)、
        // 各 dim 512 (= 8 bucket × 4 contact × 4 reserve × 4 relay)、 max_active 1
        if self.enable_attack_relay {
            dims.push((
                super::ARH_SPARSE_NAME_BLACK.to_string(),
                super::ARH_DIM_PER_PERSP,
                super::ARH_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::ARH_SPARSE_NAME_WHITE.to_string(),
                super::ARH_DIM_PER_PERSP,
                super::ARH_MAX_ACTIVE_PER_PERSP,
            ));
        }
        // 2026-06-10 ARH v2 (= 構造化 980 weight、 4 テーブル因子分解 + 先後共有):
        // 2 sparse input (black / white)、 各 dim 980 (= disk 全体)、 max_active 4
        // (= W_base + W_alpha + W_phase + W_alpha_shape、 基準スライスは除外で可変)
        if self.enable_attack_relay_v2 {
            dims.push((
                super::ARH_V2_SPARSE_NAME_BLACK.to_string(),
                super::ARH_V2_WEIGHT_COUNT,
                super::ARH_V2_MAX_ACTIVE_PER_PERSP,
            ));
            dims.push((
                super::ARH_V2_SPARSE_NAME_WHITE.to_string(),
                super::ARH_V2_WEIGHT_COUNT,
                super::ARH_V2_MAX_ACTIVE_PER_PERSP,
            ));
        }
        dims
    }

    fn fill_auxiliary_sparse(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [i32]) -> usize {
        if self.enable_scalar_threat
            && matches!(name, super::SCALAR_THREAT_SPARSE_NAME_BLACK | super::SCALAR_THREAT_SPARSE_NAME_WHITE)
        {
            use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
            let bucket = OutputBuckets::<PackedSfenValue>::bucket(&ShogiProgressKPAbs, pos) as usize;
            let perspective = if name == super::SCALAR_THREAT_SPARSE_NAME_BLACK { Color::Black } else { Color::White };
            return super::fill_scalar_threat_for_perspective(pos, perspective, bucket, out);
        }
        if self.enable_dual_cross_geometry && matches!(name, DUAL_CROSS_GEOMETRY_BLACK | DUAL_CROSS_GEOMETRY_WHITE) {
            let side = if name == DUAL_CROSS_GEOMETRY_BLACK { Color::Black } else { Color::White };
            return fill_dual_cross_geometry(pos, side, out);
        }
        if self.dual_enabled {
            match name {
                DUAL_SPARSE_NAME_ATTACK_BLACK => {
                    return fill_dual_sparse_head(pos, Color::Black, DualHead::Attack, out);
                }
                DUAL_SPARSE_NAME_DEFEND_BLACK => {
                    return fill_dual_sparse_head(pos, Color::Black, DualHead::Defend, out);
                }
                DUAL_SPARSE_NAME_ATTACK_WHITE => {
                    return fill_dual_sparse_head(pos, Color::White, DualHead::Attack, out);
                }
                DUAL_SPARSE_NAME_DEFEND_WHITE => {
                    return fill_dual_sparse_head(pos, Color::White, DualHead::Defend, out);
                }
                DUAL_CROSS_THREAT_BLACK
                    if !self.enable_scalar_threat || self.enable_scalar_threat_coexist_dual_cross =>
                {
                    return fill_dual_cross_head(pos, Color::Black, CrossHead::Threat, out);
                }
                DUAL_CROSS_DEFENDED_BLACK
                    if !self.enable_scalar_threat || self.enable_scalar_threat_coexist_dual_cross =>
                {
                    return fill_dual_cross_head(pos, Color::Black, CrossHead::Defended, out);
                }
                DUAL_CROSS_THREAT_WHITE
                    if !self.enable_scalar_threat || self.enable_scalar_threat_coexist_dual_cross =>
                {
                    return fill_dual_cross_head(pos, Color::White, CrossHead::Threat, out);
                }
                DUAL_CROSS_DEFENDED_WHITE
                    if !self.enable_scalar_threat || self.enable_scalar_threat_coexist_dual_cross =>
                {
                    return fill_dual_cross_head(pos, Color::White, CrossHead::Defended, out);
                }
                _ => {}
            }
        }
        // arm 11 KingRing: dual_enabled に依存せず単独 dispatch (= arm 11 で dual + kingring 同時 ON、
        // または kingring 単独 ON の両方をサポート)。
        if self.enable_kingring {
            match name {
                KINGRING_SPARSE_NAME_BLACK => {
                    return fill_kingring_sparse_head(pos, Color::Black, out);
                }
                KINGRING_SPARSE_NAME_WHITE => {
                    return fill_kingring_sparse_head(pos, Color::White, out);
                }
                _ => {}
            }
        }
        // arm 12 PairShape: dual_enabled に依存せず単独 dispatch。
        if self.enable_pairshape {
            match name {
                PAIRSHAPE_SPARSE_NAME_BLACK => {
                    return fill_pairshape_sparse_head(pos, Color::Black, out);
                }
                PAIRSHAPE_SPARSE_NAME_WHITE => {
                    return fill_pairshape_sparse_head(pos, Color::White, out);
                }
                _ => {}
            }
        }
        // 2026-06-06 Activity-Risk: 1 active index を local_idx_in_persp で計算。
        if self.enable_activity_risk {
            if name == super::ACTIVITY_RISK_SPARSE_NAME_BLACK {
                return fill_activity_risk_sparse_head(pos, Color::Black, out);
            }
            if name == super::ACTIVITY_RISK_SPARSE_NAME_WHITE {
                return fill_activity_risk_sparse_head(pos, Color::White, out);
            }
            // 2026-06-08 v3 tactical sparse input
            if name == super::ACTIVITY_RISK_V3_TACTICAL_NAME_BLACK {
                return fill_activity_risk_v3_tactical_sparse_head(pos, Color::Black, out);
            }
            if name == super::ACTIVITY_RISK_V3_TACTICAL_NAME_WHITE {
                return fill_activity_risk_v3_tactical_sparse_head(pos, Color::White, out);
            }
        }
        // 2026-06-09 King Escape Direction (KED): per persp 1 active = bucket * 4 + class。
        if self.enable_king_escape {
            if name == super::KING_ESCAPE_SPARSE_NAME_BLACK {
                return fill_king_escape_sparse_head(pos, true, out);
            }
            if name == super::KING_ESCAPE_SPARSE_NAME_WHITE {
                return fill_king_escape_sparse_head(pos, false, out);
            }
        }
        // 2026-06-15 King Danger (KDGR): per persp 1 active = bucket * 8 + class。
        if self.enable_king_danger {
            if name == super::KING_DANGER_SPARSE_NAME_BLACK {
                return fill_king_danger_sparse_head(pos, true, out);
            }
            if name == super::KING_DANGER_SPARSE_NAME_WHITE {
                return fill_king_danger_sparse_head(pos, false, out);
            }
        }
        // 2026-06-16 Attack Development (ADEV): per persp 1 active = bucket * 8 + class。
        if self.enable_attack_dev {
            if name == super::ATTACK_DEV_SPARSE_NAME_BLACK {
                return fill_attack_dev_sparse_head(pos, true, out);
            }
            if name == super::ATTACK_DEV_SPARSE_NAME_WHITE {
                return fill_attack_dev_sparse_head(pos, false, out);
            }
        }
        // 2026-06-28 Slider X-ray (SXRY): per persp 1 active = bucket * 8 + class。
        if self.enable_slider_xray {
            if name == super::SLIDER_XRAY_SPARSE_NAME_BLACK {
                return fill_slider_xray_sparse_head(pos, true, out);
            }
            if name == super::SLIDER_XRAY_SPARSE_NAME_WHITE {
                return fill_slider_xray_sparse_head(pos, false, out);
            }
        }
        // 2026-07-01 King Zone Threat (KZT_HAND) factored v1: per persp 多 active、 5 table に
        // 展開。 name = kzt_{core,region,bucket,persp,bias}_{black,white} で table/persp を dispatch。
        if self.enable_kzt {
            for t in 0..super::KZT_NUM_TABLES {
                if name == super::KZT_TABLE_BLACK_NAMES[t] {
                    return fill_kzt_sparse_head(pos, rshogi_core::types::Color::Black, t, out);
                }
                if name == super::KZT_TABLE_WHITE_NAMES[t] {
                    return fill_kzt_sparse_head(pos, rshogi_core::types::Color::White, t, out);
                }
            }
        }
        // 2026-06-16 King Mobility (KMOB): per persp 1 active = bucket * 9 + class。
        if self.enable_king_mobility {
            if name == super::KING_MOBILITY_SPARSE_NAME_BLACK {
                return fill_king_mobility_sparse_head(pos, true, out);
            }
            if name == super::KING_MOBILITY_SPARSE_NAME_WHITE {
                return fill_king_mobility_sparse_head(pos, false, out);
            }
        }
        // 2026-06-16 Drop Pressure (DROP): per persp 1 active = bucket * 8 + class。
        if self.enable_drop_pressure {
            if name == super::DROP_PRESSURE_SPARSE_NAME_BLACK {
                return fill_drop_pressure_sparse_head(pos, true, out);
            }
            if name == super::DROP_PRESSURE_SPARSE_NAME_WHITE {
                return fill_drop_pressure_sparse_head(pos, false, out);
            }
        }
        if self.enable_endgame_tempo {
            if name == super::ENDGAME_TEMPO_SPARSE_NAME_BLACK {
                return fill_endgame_tempo_sparse_head(pos, true, out);
            }
            if name == super::ENDGAME_TEMPO_SPARSE_NAME_WHITE {
                return fill_endgame_tempo_sparse_head(pos, false, out);
            }
        }
        // 2026-06-09 Attack Relay Head (ARH): per persp 1 active
        // = bucket * 64 + contact * 16 + reserve * 4 + relay。
        if self.enable_attack_relay {
            if name == super::ARH_SPARSE_NAME_BLACK {
                return fill_attack_relay_sparse_head(pos, true, out);
            }
            if name == super::ARH_SPARSE_NAME_WHITE {
                return fill_attack_relay_sparse_head(pos, false, out);
            }
        }
        // 2026-06-10 ARH v2 (= 構造化 980 weight): per persp 最大 4 active
        // = W_base + (W_alpha if alpha != 0) + (W_phase if phase != 0) + (W_alpha_shape if alpha != 0)
        if self.enable_attack_relay_v2 {
            if name == super::ARH_V2_SPARSE_NAME_BLACK {
                return fill_attack_relay_v2_sparse_head(pos, true, out);
            }
            if name == super::ARH_V2_SPARSE_NAME_WHITE {
                return fill_attack_relay_v2_sparse_head(pos, false, out);
            }
        }
        self.inner.fill_auxiliary_sparse(name, pos, out)
    }
}

/// KED sparse fill (= per persp 1 active idx = bucket * 4 + class)。
///
/// bucket は eval 側 progress bucket と整合させるため、 PSV `eval`/`material`/`game_ply`
/// から推論側 `progress8kpabs` と同じ値を取得する必要があるが、 訓練 path では
/// `bucket_eval` を bullet 側で別途付与する慣行。 ここでは bucket 0 固定で encode し、
/// graph 上で bucket gating が走るのを期待する。
fn fill_king_escape_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::king_escape_features::{KING_ESCAPE_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    // 訓練 path では bucket 0 固定 (= graph 側の bucket gating で適切な bucket weight に届く)。
    // 推論側は `pos.bucket_index` から正しい bucket を直接 lookup する。
    let bucket: usize = 0;
    let idx = local_idx(bucket, class);
    debug_assert!(idx < KING_ESCAPE_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// KDGR sparse fill (= per persp 1 active idx = bucket * 8 + class)。
///
/// ARH と同じ実 bucket pattern を使う (= KED の bucket 0 固定は非採用、 全 bucket 育成のため)。
/// 推論側 `evaluate_king_danger` は実 bucket_index で lookup するため、 訓練も実 bucket を
/// `ShogiProgressKPAbs` 経由で計算し全 bucket row を育成する必要がある (= ARH で確立した修正 pattern)。
fn fill_king_danger_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::king_danger_features::{KING_DANGER_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, class);
    debug_assert!(idx < KING_DANGER_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// ADEV sparse fill (= per persp 1 active idx = bucket * 8 + class)。
///
/// KDGR と同じ実 bucket pattern を使う (= 推論側 `evaluate_attack_dev` は実 bucket_index で
/// lookup するため、 訓練も実 bucket を `ShogiProgressKPAbs` 経由で計算し全 bucket row を育成する)。
fn fill_attack_dev_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::attack_dev_features::{ATTACK_DEV_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, class);
    debug_assert!(idx < ATTACK_DEV_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// SXRY sparse fill (= per persp 1 active idx = bucket * 8 + class)。
///
/// ADEV / KDGR と同じ実 bucket pattern を使う (= 推論側 `evaluate_slider_xray` は実
/// bucket_index で lookup するため、 訓練も実 bucket を `ShogiProgressKPAbs` 経由で計算し
/// 全 bucket row を育成する)。 class は 8 (= 0..=7、 ADEV / KDGR と同形)。
fn fill_slider_xray_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::slider_xray_features::{SLIDER_XRAY_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, class);
    debug_assert!(idx < SLIDER_XRAY_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// KZT_HAND factored sparse fill (= per persp 多 active を table `t` の index に展開)。
///
/// 推論側 `rshogi-core::nnue::kzt::collect_hand_features` を直接呼ぶため bit-exact。 `t` は
/// table index (= 0:core 1:region 2:bucket 3:persp 4:bias)。 実 bucket を `ShogiProgressKPAbs`
/// 経由で計算し index に焼き込む (= 推論側 `evaluate_kzt_hand` と同 bucket)。
fn fill_kzt_sparse_head(
    pos: &PackedSfenValue,
    perspective: rshogi_core::types::Color,
    table: usize,
    out: &mut [i32],
) -> usize {
    let board = pos.board();
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };

    let active = super::collect_kzt_factored_for_side(&board, perspective, bucket);
    let idxs = active.table(table);
    let max_total = out.len();
    let mut n_written = 0usize;
    for &idx in idxs.iter() {
        if n_written >= max_total {
            break;
        }
        out[n_written] = idx as i32;
        n_written += 1;
    }
    n_written
}

/// KMOB sparse fill (= per persp 1 active idx = bucket * 9 + class)。
///
/// ADEV / KDGR と同じ実 bucket pattern を使う (= 推論側 `evaluate_king_mobility` は実
/// bucket_index で lookup するため、 訓練も実 bucket を `ShogiProgressKPAbs` 経由で計算し
/// 全 bucket row を育成する)。 class は 9 (= 0..=8) である点が ADEV / KDGR と異なる。
fn fill_king_mobility_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::king_mobility_features::{KING_MOBILITY_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, class);
    debug_assert!(idx < KING_MOBILITY_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// DROP sparse fill (= per persp 1 active idx = bucket * 8 + class)。
///
/// ADEV / KDGR と同じ実 bucket pattern を使う (= 推論側 `evaluate_drop_pressure` は実
/// bucket_index で lookup するため、 訓練も実 bucket を `ShogiProgressKPAbs` 経由で計算し
/// 全 bucket row を育成する)。 class は 8 (= 0..=7、 ADEV / KDGR と同形)。
fn fill_drop_pressure_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::drop_pressure_features::{DROP_PRESSURE_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, class);
    debug_assert!(idx < DROP_PRESSURE_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// EGT sparse fill (= per persp 1 active idx = bucket * 64 + class)。
fn fill_endgame_tempo_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::endgame_tempo_features::{ENDGAME_TEMPO_DIM_PER_PERSP, compute_class_for_persp, local_idx};

    let board = pos.board();
    let class = compute_class_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, class);
    debug_assert!(idx < ENDGAME_TEMPO_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// ARH sparse fill (= per persp 1 active idx = bucket * 64 + contact * 16 + reserve * 4 + relay)。
///
/// 2026-06-10 修正 (= task #92): bucket 0 固定 (= 既存 KED と同 pattern) は中盤発火局面の
/// cell が訓練で育成されず、 ply 24 多 SFEN で ARH 寄与 0 が判明。 修正後は実 bucket を
/// `ShogiProgressKPAbs` 経由で計算し、 全 bucket cell を実 bucket 別に育成する。
/// 推論側 `evaluate_arh` も同 bucket index で同 cell を引くため index 契約一致。
fn fill_attack_relay_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::attack_relay_features::{ARH_DIM_PER_PERSP, compute_classes_for_persp, local_idx};

    let board = pos.board();
    let (contact, reserve, relay) = compute_classes_for_persp(&board, persp_is_black);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let idx = local_idx(bucket, contact, reserve, relay);
    debug_assert!(idx < ARH_DIM_PER_PERSP);
    if out.is_empty() {
        return 0;
    }
    out[0] = idx as i32;
    1
}

/// ARH v2 (= 構造化 980 weight、 2026-06-10 改訂) sparse fill。
///
/// 推論側 `rshogi-core::nnue::attack_relay_v2` と完全一致する disk index を返す。
/// 最大 4 active (= W_base / W_alpha / W_phase / W_alpha_shape)、 基準スライス
/// (= `alpha == 0` または `phase == 0`) は activate せず学習対象から除外する。
fn fill_attack_relay_v2_sparse_head(pos: &PackedSfenValue, persp_is_black: bool, out: &mut [i32]) -> usize {
    use super::attack_relay_v2_features::{ARH_V2_MAX_ACTIVE_PER_PERSP, ARH_V2_WEIGHT_COUNT, intrude_active};

    let board = pos.board();
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    let mut buf = [0u32; ARH_V2_MAX_ACTIVE_PER_PERSP];
    let n = intrude_active(&board, persp_is_black, bucket, &mut buf);
    let cap = out.len().min(n);
    for (slot, &idx) in out.iter_mut().take(cap).zip(buf.iter().take(cap)) {
        debug_assert!((idx as usize) < ARH_V2_WEIGHT_COUNT);
        *slot = idx as i32;
    }
    cap
}

/// sparse fill 用の head 識別。
#[derive(Clone, Copy, Debug)]
pub(crate) enum DualHead {
    Attack,
    Defend,
}

/// 1 head 分の active index を out 先頭から書き込み、 active 数を返す (= 末尾は呼出側 -1 padding 前提)。
pub(crate) fn fill_dual_sparse_head(pos: &PackedSfenValue, side: Color, head: DualHead, out: &mut [i32]) -> usize {
    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    let (att, def, n_a, n_d) = collect_dual_active_sparse_for_side(&board, side, bucket);
    let (src, n) = match head {
        DualHead::Attack => (&att, n_a),
        DualHead::Defend => (&def, n_d),
    };

    let max_total = out.len();
    let mut n_written = 0usize;
    for &idx in src.iter().take(n) {
        if n_written >= max_total {
            break;
        }
        out[n_written] = idx as i32;
        n_written += 1;
    }
    n_written
}

/// D2 cross feature head 識別子。
#[derive(Clone, Copy, Debug)]
pub(crate) enum CrossHead {
    /// piece_threat_relation (= 自軍駒 → 攻撃中の相手駒)
    Threat,
    /// piece_defended_by (= 自軍駒 → 守ってる自軍駒)
    Defended,
}

/// 1 cross head 分の active index を out に書き込み、 active 数を返す。
pub(crate) fn fill_dual_cross_head(pos: &PackedSfenValue, side: Color, head: CrossHead, out: &mut [i32]) -> usize {
    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    debug_assert!(bucket < NUM_DUAL_PSQT_BUCKETS);

    let (threat, defended, n_t, n_d) = collect_dual_cross_for_side(&board, side, bucket);
    let (src, n) = match head {
        CrossHead::Threat => (&threat, n_t),
        CrossHead::Defended => (&defended, n_d),
    };

    let max_total = out.len();
    let mut n_written = 0usize;
    for &idx in src.iter().take(n) {
        if n_written >= max_total {
            break;
        }
        out[n_written] = idx as i32;
        n_written += 1;
    }
    n_written
}

#[inline]
fn dual_cross_geometry_class(own: Square, target: Square, perspective: Color) -> usize {
    let own = oriented_sq_idx(own.index(), perspective);
    let target = oriented_sq_idx(target.index(), perspective);
    let (of, or) = ((own / 9) as i32, (own % 9) as i32);
    let (tf, tr) = ((target / 9) as i32, (target % 9) as i32);
    let direction = match ((tf - of).signum(), (tr - or).signum()) {
        (-1, -1) => 0,
        (0, -1) => 1,
        (1, -1) => 2,
        (-1, 0) => 3,
        (1, 0) => 4,
        (-1, 1) => 5,
        (0, 1) => 6,
        (1, 1) => 7,
        _ => unreachable!("attacker and target cannot share a square"),
    };
    let distance = (tf - of).abs().max((tr - or).abs());
    let distance_class = match distance {
        1 => 0,
        2 => 1,
        3 | 4 => 2,
        _ => 3,
    };
    direction * 4 + distance_class
}

fn fill_dual_cross_geometry(pos: &PackedSfenValue, side: Color, out: &mut [i32]) -> usize {
    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        OutputBuckets::<PackedSfenValue>::bucket(&ShogiProgressKPAbs, pos) as usize
    };
    let occupied = occupied_mask(&board);
    let mut len = 0;
    'outer: for own_raw in 0..81u8 {
        let own_sq = Square(own_raw);
        let own_piece = board.piece_on(own_sq);
        if !own_piece.is_some() || own_piece.color != side {
            continue;
        }
        let Some(own_pt) = piece_type_to_dual_idx(own_piece.piece_type) else {
            continue;
        };
        let attacks = piece_attacks_from(own_piece, own_sq, &occupied);
        for target_raw in 0..81u8 {
            if !attacks[target_raw as usize] {
                continue;
            }
            let target_sq = Square(target_raw);
            let target_piece = board.piece_on(target_sq);
            if !target_piece.is_some() || target_piece.color == side {
                continue;
            }
            let Some(target_pt) = piece_type_to_dual_idx(target_piece.piece_type) else {
                continue;
            };
            if len == out.len().min(DUAL_CROSS_MAX_ACTIVE_PER_HEAD) {
                break 'outer;
            }
            let pair = own_pt * DUAL_PIECE_TYPES + target_pt;
            let geometry = dual_cross_geometry_class(own_sq, target_sq, side);
            out[len] = (bucket * DUAL_PIECE_TYPES * DUAL_PIECE_TYPES * DUAL_CROSS_GEOMETRY_CLASSES
                + pair * DUAL_CROSS_GEOMETRY_CLASSES
                + geometry) as i32;
            len += 1;
        }
    }
    len
}

// =============================================================================
// KingRing Threat PSQT scalar head (arm 11、 2026-05-28)
//
// rshogi-core `nnue::kingring` と数値一致。 玉周辺 24 sq × 13 piece × 4 state × 8 bucket ×
// 2 persp = 19,968 i16 weight (~40 KB) の scalar PSQT head。 L1 accumulator には混ぜず、
// 末端で sparse_dot(weight, active_sq×piece×state) を加算合成する。
//
// disk layout: `[bucket][persp][sq][piece][state] -> i16`、 sub_magic = "P2KRING\0"。
//
// bullet 訓練側は 2 sparse input (= black / white perspective) として渡す。 sample 内の
// active 集合は bucket × ring (24) × piece (13) × state (4) を encode した flat index で、
// dim = 8 * 1248 = 9984 (= bucket-major flat、 sample は 1 bucket 分のみ active)。
// =============================================================================

/// bucket 数 (= progress8kpabs と共有、 DUAL と同じ)。
pub const KINGRING_NUM_BUCKETS: usize = 8;

/// ring 内 sq 数 (= 5×5 zone から king 自身を除く)。
pub const KINGRING_NUM_SQ: usize = 24;

/// attacker piece_type 数 (= Pawn..Dragon、 King 除く 13 種)。
pub const KINGRING_NUM_PIECE: usize = 13;

/// target_state 数 (= 0=empty_own_attack / 1=own_defended / 2=enemy_attacked / 3=empty_enemy_attack)。
pub const KINGRING_NUM_STATE: usize = 4;

/// 1 perspective × 1 bucket あたりの flat dim (= 24 × 13 × 4 = 1248)。
pub const KINGRING_PER_BUCKET_DIM: usize = KINGRING_NUM_SQ * KINGRING_NUM_PIECE * KINGRING_NUM_STATE;

const _: () = {
    assert!(KINGRING_PER_BUCKET_DIM == 1248);
};

/// 1 perspective あたりの sparse input dim (= bucket-major flat = 8 × 1248 = 9984)。
pub const KINGRING_DIM_PER_PERSP: usize = KINGRING_NUM_BUCKETS * KINGRING_PER_BUCKET_DIM;

const _: () = {
    assert!(KINGRING_DIM_PER_PERSP == 9_984);
};

/// 2 perspective 合算の i16 entry 数 (= disk total = 8 × 2 × 1248 = 19,968)。
pub const KINGRING_TOTAL_COUNT: u32 = (2 * KINGRING_DIM_PER_PERSP) as u32;

const _: () = {
    assert!(KINGRING_TOTAL_COUNT == 19_968);
};

/// 1 perspective あたりの max active (= ring sq 数の上限、 ≤ 24)。
pub const KINGRING_MAX_ACTIVE_PER_PERSP: usize = KINGRING_NUM_SQ;

/// ckpt section sub_magic ("P2KRING\0")。 rshogi-core `KINGRING_SUB_MAGIC` と一致。
pub const KINGRING_SUB_MAGIC: &[u8; 8] = b"P2KRING\0";

/// ckpt section format version。 rshogi-core `KINGRING_VERSION` と一致。
pub const KINGRING_SECTION_VERSION: u32 = 1;

/// QA = 127 (= dual_psqt / dual_cross と同 convention)。
///
/// 過去 QA=1 で書き出すと f32 weight (abs_max ~0.10-0.12) が round で全 0 になる
/// バグがあった (= 2026-05-29 fix)。 rshogi-core 側 `KINGRING_OUTPUT_SCALE = 1` (=
/// i16 sum を i32 のまま eval 加算前 PSQT scalar に足す) と整合させるため、
/// dual_psqt と同じ QA=127 で量子化する。 i16 range は概ね ±15 (= 127 × 0.12)、
/// 24 active × 8 bucket × (us-them) sum で ±360 程度、 fv_scale=20 後 ±18 cp。
pub const KINGRING_QA: f64 = 127.0;

/// arch_str に埋め込む marker。 rshogi-core loader は `DUAL=1` で section probe するため、
/// kingring は独立 marker 不要 (= dual_psqt_king_attacker と共存)。 ここでは保持のみ。
pub const KINGRING_ARCH_MARKER: &str = "KRING=1";

/// auxiliary sparse input 名 (黒 / 白視点)。
pub const KINGRING_SPARSE_NAME_BLACK: &str = "kingring_black";
pub const KINGRING_SPARSE_NAME_WHITE: &str = "kingring_white";

/// `PieceType` (1..=14) を kingring の attacker idx (0..=12) に変換する (= King 除く)。
///
/// rshogi-core `piece_type_to_kingring_idx` と完全同 mapping:
/// `1=Pawn→0、 2=Lance→1、 ..、 7=Gold→6、 (8=King skip)、 9=ProPawn→7、 ..、 14=Dragon→12`。
#[inline]
const fn piece_type_to_kingring_idx(pt: PieceType) -> Option<usize> {
    let raw = pt as u8;
    if raw == 0 || raw == PieceType::King as u8 {
        // None (= 0) または King は除外
        return None;
    }
    let king_raw = PieceType::King as u8;
    let idx = if raw < king_raw { (raw - 1) as usize } else { (raw - 2) as usize };
    debug_assert!(idx < KINGRING_NUM_PIECE);
    Some(idx)
}

/// king 位置 + perspective から ring 24 sq の `[Option<Square>; 24]` を構築。
///
/// rshogi-core `build_ring_for_king` と等価 (= (df, dr) ∈ {-2..=2}² \ {(0,0)} の 24 組を
/// (df+2)*5+(dr+2) で 0..=24 に map、 idx 12 (= king 自身) を skip、 残り 0..23 を順次)。
/// White perspective では (df, dr) → (-df, -dr) で 180° 回転。
fn kingring_build_ring(king_sq: Square, perspective: Color) -> [Option<Square>; KINGRING_NUM_SQ] {
    let mut out: [Option<Square>; KINGRING_NUM_SQ] = [None; KINGRING_NUM_SQ];
    if !king_sq.is_valid() {
        return out;
    }
    let kf = king_sq.file() as i8;
    let kr = king_sq.rank() as i8;
    let sign: i8 = if perspective == Color::Black { 1 } else { -1 };

    let mut idx: usize = 0;
    for df in -2..=2i8 {
        for dr in -2..=2i8 {
            if df == 0 && dr == 0 {
                continue;
            }
            let nf = kf + sign * df;
            let nr = kr + sign * dr;
            if (0..9).contains(&nf) && (0..9).contains(&nr) {
                out[idx] = Some(Square::new(nf as u8, nr as u8));
            }
            idx += 1;
        }
    }
    debug_assert!(idx == KINGRING_NUM_SQ);
    out
}

/// `attacker_color` 側で `target_sq` を攻撃する駒の最小 piece_type を返す。
///
/// rshogi-core `smallest_attacker` 等価。 PieceType 昇順 (= Pawn → Dragon、 King 除く) で
/// 最初に target_sq を攻撃する駒の piece_type を返す。 該当無しなら None。
fn kingring_smallest_attacker(
    board: &ShogiBoard,
    target_sq: Square,
    attacker_color: Color,
    occ: &OccMask,
) -> Option<PieceType> {
    const ORDER: [PieceType; 13] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Gold,
        PieceType::ProPawn,
        PieceType::ProLance,
        PieceType::ProKnight,
        PieceType::ProSilver,
        PieceType::Horse,
        PieceType::Dragon,
    ];
    let target_idx = target_sq.index();
    for pt in ORDER {
        for sq_raw in 0..81u8 {
            let sq = Square(sq_raw);
            let piece = board.piece_on(sq);
            if !piece.is_some() || piece.color != attacker_color || piece.piece_type != pt {
                continue;
            }
            let attacks = piece_attacks_from(piece, sq, occ);
            if attacks[target_idx] {
                return Some(pt);
            }
        }
    }
    None
}

/// 1 sq の (piece_idx, state_idx) を判定する。
///
/// rshogi-core `classify_sq` と完全同 logic:
/// 1. own piece on sq → state 1、 piece = sq の駒 type
/// 2. enemy on sq → state 2、 piece = sq の駒 type
/// 3. empty + own attack → state 0、 piece = own smallest attacker piece_type
/// 4. empty + enemy attack → state 3、 piece = enemy smallest attacker piece_type
/// 5. 該当なし → None
fn kingring_classify_sq(
    board: &ShogiBoard,
    perspective: Color,
    ring_sq: Square,
    occ: &OccMask,
) -> Option<(usize, usize)> {
    let piece = board.piece_on(ring_sq);
    if piece.is_some() {
        let pt = piece.piece_type;
        let piece_color = piece.color;
        let piece_idx = piece_type_to_kingring_idx(pt)?;
        let state = if piece_color == perspective { 1 } else { 2 };
        return Some((piece_idx, state));
    }
    // empty: own → enemy の順で smallest attacker をチェック
    if let Some(attacker_pt) = kingring_smallest_attacker(board, ring_sq, perspective, occ) {
        let piece_idx = piece_type_to_kingring_idx(attacker_pt)?;
        return Some((piece_idx, 0));
    }
    if let Some(attacker_pt) = kingring_smallest_attacker(board, ring_sq, perspective.opponent(), occ) {
        let piece_idx = piece_type_to_kingring_idx(attacker_pt)?;
        return Some((piece_idx, 3));
    }
    None
}

/// (ring_idx, piece_idx, state_idx) を bucket-local flat index (= 0..1248) に encode。
#[inline]
pub const fn kingring_encode_local(ring_idx: usize, piece_idx: usize, state_idx: usize) -> usize {
    ring_idx * (KINGRING_NUM_PIECE * KINGRING_NUM_STATE) + piece_idx * KINGRING_NUM_STATE + state_idx
}

/// `perspective` 視点で active な kingring sparse index を集める。
///
/// 戻り値 `(buf, n)`: `buf[..n]` が有効、 各 entry は bucket-major flat index
/// (= `bucket * 1248 + local_idx`)、 範囲 `0..KINGRING_DIM_PER_PERSP` (= 0..9984)。
///
/// rshogi-core `collect_active_indices` と同 algorithm + bucket offset 加算。
pub fn collect_kingring_active_for_side(
    board: &ShogiBoard,
    perspective: Color,
    bucket: usize,
) -> ([u32; KINGRING_MAX_ACTIVE_PER_PERSP], usize) {
    debug_assert!(bucket < KINGRING_NUM_BUCKETS);

    let mut buf = [0u32; KINGRING_MAX_ACTIVE_PER_PERSP];
    let mut n: usize = 0;

    let king_sq = board.king_square(perspective);
    if !king_sq.is_valid() {
        return (buf, n);
    }

    let occ = occupied_mask(board);
    let ring = kingring_build_ring(king_sq, perspective);

    for (ring_idx, maybe_sq) in ring.iter().enumerate() {
        let Some(sq) = *maybe_sq else { continue };
        let Some((piece_idx, state_idx)) = kingring_classify_sq(board, perspective, sq, &occ) else { continue };
        let local = kingring_encode_local(ring_idx, piece_idx, state_idx);
        let flat = bucket * KINGRING_PER_BUCKET_DIM + local;
        debug_assert!(flat < KINGRING_DIM_PER_PERSP);
        if n < KINGRING_MAX_ACTIVE_PER_PERSP {
            buf[n] = flat as u32;
            n += 1;
        }
    }
    (buf, n)
}

/// 1 kingring perspective 分の active index を out に書き込み、 active 数を返す。
fn fill_kingring_sparse_head(pos: &PackedSfenValue, side: Color, out: &mut [i32]) -> usize {
    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    debug_assert!(bucket < KINGRING_NUM_BUCKETS);

    let (buf, n) = collect_kingring_active_for_side(&board, side, bucket);
    let max_total = out.len();
    let mut n_written = 0usize;
    for &idx in buf.iter().take(n) {
        if n_written >= max_total {
            break;
        }
        out[n_written] = idx as i32;
        n_written += 1;
    }
    n_written
}

// =============================================================================
// PairShape (= Piece-Pair Shape Table) scalar head (arm 12、 2026-05-29)
//
// rshogi-core `nnue::pairshape` と数値一致。 12 固定 pair_type × 16 shape × 8 bucket ×
// 2 persp = 3,072 i16 weight (~6 KB) の scalar PSQT head。 L1 accumulator には混ぜず、
// 末端で sparse_dot(weight, active_pair_shape) を加算合成する。
//
// disk layout: `[bucket][persp][pair_type 12][shape 16] -> i16`、 sub_magic = "P2PAIR\0\0"。
//
// bullet 訓練側は 2 sparse input (= black / white perspective) として渡す。 sample 内の
// active 集合は bucket × pair_type (12) × shape (16) を encode した flat index で、
// dim = 8 * 192 = 1536 (= bucket-major flat、 sample は 1 bucket 分のみ active)。
// active は multiset (= 重複保持)、 max_active = 256 (= rshogi-core PAIRSHAPE_MAX_ACTIVE)。
// =============================================================================

/// bucket 数 (= progress8kpabs と共有)。
pub const PAIRSHAPE_NUM_BUCKETS: usize = 8;

/// pair_type 数 (= 12 種、 固定)。
pub const PAIRSHAPE_NUM_PAIR_TYPES: usize = 12;

/// shape class 数 (= 16 種、 排他)。
pub const PAIRSHAPE_NUM_SHAPES: usize = 16;

/// 1 perspective × 1 bucket あたりの flat dim (= 12 × 16 = 192)。
pub const PAIRSHAPE_PER_BUCKET_DIM: usize = PAIRSHAPE_NUM_PAIR_TYPES * PAIRSHAPE_NUM_SHAPES;

const _: () = {
    assert!(PAIRSHAPE_PER_BUCKET_DIM == 192);
};

/// 1 perspective あたりの sparse input dim (= bucket-major flat = 8 × 192 = 1536)。
pub const PAIRSHAPE_DIM_PER_PERSP: usize = PAIRSHAPE_NUM_BUCKETS * PAIRSHAPE_PER_BUCKET_DIM;

const _: () = {
    assert!(PAIRSHAPE_DIM_PER_PERSP == 1_536);
};

/// 2 perspective 合算の i16 entry 数 (= disk total = 8 × 2 × 192 = 3,072)。
pub const PAIRSHAPE_TOTAL_COUNT: u32 = (2 * PAIRSHAPE_DIM_PER_PERSP) as u32;

const _: () = {
    assert!(PAIRSHAPE_TOTAL_COUNT == 3_072);
};

/// 1 perspective あたりの max active (= rshogi-core PAIRSHAPE_MAX_ACTIVE と一致、 multiset 上限)。
pub const PAIRSHAPE_MAX_ACTIVE_PER_PERSP: usize = 256;

/// ckpt section sub_magic ("P2PAIR\0\0")。 rshogi-core `PAIRSHAPE_SUB_MAGIC` と一致。
pub const PAIRSHAPE_SUB_MAGIC: &[u8; 8] = b"P2PAIR\0\0";

/// ckpt section format version。 rshogi-core `PAIRSHAPE_VERSION` と一致。
pub const PAIRSHAPE_SECTION_VERSION: u32 = 1;

/// QA = 127 (= dual_psqt / kingring と同 convention)。
///
/// rshogi-core 側 `PAIRSHAPE_OUTPUT_SCALE = 1`、 dual_psqt と同じ QA=127 で量子化する。
/// **重要**: QA=1 だと f32 weight (abs_max ~0.05-0.10) が round で全 0 になるバグの再発
/// 防止 (= 2026-05-29 fix の kingring 前例参照)。
pub const PAIRSHAPE_QA: f64 = 127.0;

/// arch_str に埋め込む marker (= reservation のみ、 ckpt section probe では P2PAIR magic で識別)。
pub const PAIRSHAPE_ARCH_MARKER: &str = "PAIR=1";

/// auxiliary sparse input 名 (黒 / 白視点)。
pub const PAIRSHAPE_SPARSE_NAME_BLACK: &str = "pairshape_black";
pub const PAIRSHAPE_SPARSE_NAME_WHITE: &str = "pairshape_white";

/// 1 pair_type の (piece_a, piece_b) 順序対。 rshogi-core `PairTypeDef` と完全一致。
#[derive(Debug, Clone, Copy)]
struct PairTypeDef {
    a: PieceType,
    b: PieceType,
}

/// 12 種の固定 pair_type 定義 (= rshogi-core `PAIR_TYPES` と完全一致、 順序保持)。
const PAIR_TYPES: [PairTypeDef; PAIRSHAPE_NUM_PAIR_TYPES] = [
    // 1. Gold-Silver
    PairTypeDef { a: PieceType::Gold, b: PieceType::Silver },
    // 2. Gold-Gold
    PairTypeDef { a: PieceType::Gold, b: PieceType::Gold },
    // 3. Silver-Silver
    PairTypeDef { a: PieceType::Silver, b: PieceType::Silver },
    // 4. Silver-Knight
    PairTypeDef { a: PieceType::Silver, b: PieceType::Knight },
    // 5. Rook-Pawn
    PairTypeDef { a: PieceType::Rook, b: PieceType::Pawn },
    // 6. Bishop-Silver
    PairTypeDef { a: PieceType::Bishop, b: PieceType::Silver },
    // 7. Tokin-Gold (= ProPawn-Gold)
    PairTypeDef { a: PieceType::ProPawn, b: PieceType::Gold },
    // 8. Horse-Gold
    PairTypeDef { a: PieceType::Horse, b: PieceType::Gold },
    // 9. Dragon-Gold
    PairTypeDef { a: PieceType::Dragon, b: PieceType::Gold },
    // 10. Pawn-Knight
    PairTypeDef { a: PieceType::Pawn, b: PieceType::Knight },
    // 11. Silver-Bishop
    PairTypeDef { a: PieceType::Silver, b: PieceType::Bishop },
    // 12. Rook-Gold
    PairTypeDef { a: PieceType::Rook, b: PieceType::Gold },
];

/// 16 種の shape class (= 排他、 優先順位順)。 rshogi-core `PairShape` enum と完全一致。
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PairShape {
    SameFile = 0,
    SameRank = 1,
    KnightShape = 2,
    DiagonalAdjacent = 3,
    AdjacentFile = 4,
    AdjacentRank = 5,
    Distance1 = 6,
    Distance2 = 7,
    Distance3Plus = 8,
    Ahead = 9,
    Behind = 10,
    SameFlankLeft = 11,
    SameFlankRight = 12,
    CrossFlank = 13,
    Centered = 14,
    Other = 15,
}

/// (a_sq, b_sq) を perspective 視点で classify。 rshogi-core `classify_pair_shape` と完全一致。
///
/// 利き計算なし、 file / rank / abs diff のみ。 white の場合 rank 軸を flip して
/// 「ahead = 自陣→敵陣方向」 を perspective に対し一貫させる (= 左右反転なし)。
#[inline]
fn classify_pair_shape(persp: Color, a_sq: Square, b_sq: Square) -> PairShape {
    // file 軸 (= 左右反転なし)
    let fa = a_sq.file() as i8;
    let fb = b_sq.file() as i8;
    // rank 軸 (= perspective に応じて flip)
    let ra_raw = a_sq.rank() as i8;
    let rb_raw = b_sq.rank() as i8;
    let (ra, rb) = if persp == Color::Black { (ra_raw, rb_raw) } else { (8 - ra_raw, 8 - rb_raw) };
    let df_signed = fa - fb;
    let dr_signed = ra - rb;
    let df = df_signed.unsigned_abs();
    let dr = dr_signed.unsigned_abs();

    // 1. SameFile / SameRank / KnightShape (= 排他、 最優先)
    if df == 0 {
        return PairShape::SameFile;
    }
    if dr == 0 {
        return PairShape::SameRank;
    }
    if (df == 1 && dr == 2) || (df == 2 && dr == 1) {
        return PairShape::KnightShape;
    }

    // 2. DiagonalAdjacent
    if df == 1 && dr == 1 {
        return PairShape::DiagonalAdjacent;
    }

    // 3. AdjacentFile / AdjacentRank
    if df == 1 {
        return PairShape::AdjacentFile;
    }
    if dr == 1 {
        return PairShape::AdjacentRank;
    }

    // 4. Distance (= Chebyshev)
    let cheby = df.max(dr);
    if cheby == 1 {
        return PairShape::Distance1;
    }
    if cheby == 2 {
        return PairShape::Distance2;
    }

    // 5. Ahead / Behind (= perspective r 軸方向、 |dr| が支配的、 |df|<|dr| の時)
    if dr_signed < 0 && dr > df {
        return PairShape::Ahead;
    }
    if dr_signed > 0 && dr > df {
        return PairShape::Behind;
    }

    // 6. Flank 系 + Centered
    let fa_left = fa <= 2;
    let fa_right = fa >= 6;
    let fa_center = (3..=5).contains(&fa);
    let fb_left = fb <= 2;
    let fb_right = fb >= 6;
    let fb_center = (3..=5).contains(&fb);
    if fa_left && fb_left {
        return PairShape::SameFlankLeft;
    }
    if fa_right && fb_right {
        return PairShape::SameFlankRight;
    }
    if (fa_left && fb_right) || (fa_right && fb_left) {
        return PairShape::CrossFlank;
    }
    if fa_center && fb_center {
        return PairShape::Centered;
    }

    // 7. Distance3Plus or Other fallback
    if cheby >= 3 {
        return PairShape::Distance3Plus;
    }
    PairShape::Other
}

/// (pair_type_idx, shape_idx) を bucket-local flat index (= 0..192) に encode。
#[inline]
const fn pairshape_encode_local(pair_type_idx: usize, shape_idx: usize) -> usize {
    pair_type_idx * PAIRSHAPE_NUM_SHAPES + shape_idx
}

/// `perspective` 視点で active な pairshape sparse index を集める。
///
/// 戻り値: `(buf, n)`、 `buf[..n]` が有効、 各 entry は bucket-major flat index
/// (= `bucket * 192 + local_idx`)、 範囲 `0..PAIRSHAPE_DIM_PER_PERSP` (= 0..1536)。
///
/// active は **multiset** (= 重複保持)、 順序不問 (= graph 側 sparse_dot で sum)。
/// rshogi-core `collect_active_indices_sorted` と同 algorithm + bucket offset 加算。
fn collect_pairshape_active_for_side(
    board: &ShogiBoard,
    perspective: Color,
    bucket: usize,
) -> ([u32; PAIRSHAPE_MAX_ACTIVE_PER_PERSP], usize) {
    debug_assert!(bucket < PAIRSHAPE_NUM_BUCKETS);

    let mut buf = [0u32; PAIRSHAPE_MAX_ACTIVE_PER_PERSP];
    let mut n: usize = 0;
    let bucket_offset = bucket * PAIRSHAPE_PER_BUCKET_DIM;

    // 全 81 sq を 1 回 scan して、 piece_type → own sq list を bucketize する。
    // pair_type は 12 種で各 piece_type は最大 2 種類 (= Gold が pair_type 1/2/7/8/9/12 等で
    // 複数 pair に登場) しかないため、 駒種別の own sq list を作って cross product する方が
    // 81×81 全 cross より高速。
    let mut own_sqs: [[Square; 18]; 15] = [[Square::NONE; 18]; 15];
    let mut own_lens: [usize; 15] = [0; 15];
    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let piece = board.piece_on(sq);
        if !piece.is_some() || piece.color != perspective {
            continue;
        }
        let pt_raw = piece.piece_type as usize;
        if pt_raw < 15 && own_lens[pt_raw] < 18 {
            own_sqs[pt_raw][own_lens[pt_raw]] = sq;
            own_lens[pt_raw] += 1;
        }
    }

    // 各 pair_type を順に列挙、 a / b の own 駒位置 list の cross product
    for (pt_idx, def) in PAIR_TYPES.iter().enumerate() {
        let a_pt = def.a as usize;
        let b_pt = def.b as usize;
        let a_len = own_lens[a_pt];
        let b_len = own_lens[b_pt];
        if a_len == 0 || b_len == 0 {
            continue;
        }

        if def.a as u8 == def.b as u8 {
            // 同 piece_type pair: unordered で a_sq < b_sq のみ採用
            for i in 0..a_len {
                let a_sq = own_sqs[a_pt][i];
                for j in 0..b_len {
                    let b_sq = own_sqs[b_pt][j];
                    if a_sq.0 >= b_sq.0 {
                        continue;
                    }
                    let sh = classify_pair_shape(perspective, a_sq, b_sq);
                    let local = pairshape_encode_local(pt_idx, sh as usize);
                    let flat = bucket_offset + local;
                    debug_assert!(flat < PAIRSHAPE_DIM_PER_PERSP);
                    if n < PAIRSHAPE_MAX_ACTIVE_PER_PERSP {
                        buf[n] = flat as u32;
                        n += 1;
                    }
                }
            }
        } else {
            // 異 piece_type pair: a × b の cross product
            for i in 0..a_len {
                let a_sq = own_sqs[a_pt][i];
                for j in 0..b_len {
                    let b_sq = own_sqs[b_pt][j];
                    let sh = classify_pair_shape(perspective, a_sq, b_sq);
                    let local = pairshape_encode_local(pt_idx, sh as usize);
                    let flat = bucket_offset + local;
                    debug_assert!(flat < PAIRSHAPE_DIM_PER_PERSP);
                    if n < PAIRSHAPE_MAX_ACTIVE_PER_PERSP {
                        buf[n] = flat as u32;
                        n += 1;
                    }
                }
            }
        }
    }
    (buf, n)
}

/// 2026-06-06 Activity-Risk Interaction Head 用 sparse fill (= 1 active per perspective)。
///
/// `ShogiBoard` → SFEN → `rshogi-core::Position` 変換後、 推論側と完全一致する
/// (activity, pressure, shield) bucket 計算を経て `local_idx_in_persp` で 1 entry 発火。
///
/// graph 側 weight tensor は 1 perspective 分 (= 2048 entries) を別 tensor として保持する
/// 設計のため、 sparse index は local (= persp 内 offset) を返す。 disk save 時に
/// row-major (= bucket-major / persp 交互) に並べ替える ([[ACTIVITY_RISK_SUB_MAGIC]] section)。
/// 2026-06-08 v3 tactical sparse input fill (= 1 active per perspective、 local idx in `[0, 512)`).
///
/// `compute_tactical_features_v3` で heavy_attack / escape / hand_check (= 各 0..=3) 計算、
/// `activity_risk_v3_tactical_local_idx(bucket, heavy, escape, hand_check)` で 1 active index 出力。
fn fill_activity_risk_v3_tactical_sparse_head(pos: &PackedSfenValue, side: Color, out: &mut [i32]) -> usize {
    use super::{
        ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP, activity_risk_v3_tactical_local_idx, compute_tactical_features_v3,
    };

    if out.is_empty() {
        return 0;
    }

    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };

    let (b_h, b_e, b_c, w_h, w_e, w_c) = compute_tactical_features_v3(&board);
    let (h, e, c) = match side {
        Color::Black => (b_h, b_e, b_c),
        Color::White => (w_h, w_e, w_c),
    };

    let local = activity_risk_v3_tactical_local_idx(bucket, h, e, c);
    debug_assert!(local < ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP);

    out[0] = local as i32;
    1
}

fn fill_activity_risk_sparse_head(pos: &PackedSfenValue, side: Color, out: &mut [i32]) -> usize {
    use super::{
        ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP, activity_risk_local_idx, activity_to_bucket,
        compute_activity_risk_features, pressure_to_bucket, shield_to_bucket,
    };

    if out.is_empty() {
        return 0;
    }

    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    debug_assert!(bucket < super::ACTIVITY_RISK_NUM_BUCKETS);

    let (b_act, b_pre, b_shi, w_act, w_pre, w_shi) = compute_activity_risk_features(&board);
    let (act, pre, shi) = match side {
        Color::Black => (b_act, b_pre, b_shi),
        Color::White => (w_act, w_pre, w_shi),
    };
    let act_b = activity_to_bucket(act);
    let pre_b = pressure_to_bucket(pre);
    let shi_b = shield_to_bucket(shi);

    let local = activity_risk_local_idx(bucket, act_b, pre_b, shi_b);
    debug_assert!(local < super::ACTIVITY_RISK_DIM_PER_PERSP);

    out[0] = local as i32;
    // max_active = 1 を契約。
    debug_assert!(ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP == 1);
    1
}

/// 1 pairshape perspective 分の active index を out に書き込み、 active 数を返す。
fn fill_pairshape_sparse_head(pos: &PackedSfenValue, side: Color, out: &mut [i32]) -> usize {
    let board = ShogiBoard::from_packed_sfen(pos);
    let bucket = {
        use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
        let b = ShogiProgressKPAbs;
        OutputBuckets::<PackedSfenValue>::bucket(&b, pos) as usize
    };
    debug_assert!(bucket < PAIRSHAPE_NUM_BUCKETS);

    let (buf, n) = collect_pairshape_active_for_side(&board, side, bucket);
    let max_total = out.len();
    let mut n_written = 0usize;
    for &idx in buf.iter().take(n) {
        if n_written >= max_total {
            break;
        }
        out[n_written] = idx as i32;
        n_written += 1;
    }
    n_written
}

#[cfg(test)]
mod tests {
    use super::*;

    /// feature index 範囲 (= 0..18144) + 1 byte magic + count = 36288。
    #[test]
    fn dual_feature_index_range() {
        let idx0 = dual_feature_index(0, 0, 0, 0);
        let idx_max = dual_feature_index(1, 7, 80, 13);
        assert_eq!(idx0, 0);
        assert_eq!(idx_max, DUAL_DIMENSIONS_PER_PERSPECTIVE - 1);
    }

    /// 定数値が rshogi-core と一致 (= 静的契約)。
    #[test]
    fn dual_constants_match_rshogi_core() {
        assert_eq!(DUAL_DIMENSIONS_PER_PERSPECTIVE, 18_144);
        assert_eq!(DUAL_TOTAL_COUNT, 36_288);
        assert_eq!(NUM_DUAL_PSQT_HEADS, 2);
        assert_eq!(NUM_DUAL_PSQT_BUCKETS, 8);
        assert_eq!(DUAL_SQUARES, 81);
        assert_eq!(DUAL_PIECE_TYPES, 14);
        assert_eq!(DUAL_SUB_MAGIC, b"P2DUAL2\0");
        assert_eq!(DUAL_ARCH_MARKER, "DUAL=1");
    }

    /// king_zone_mask: 中央 (5五) で 9 sq、 角 (1一) で 4 sq。
    #[test]
    fn king_zone_mask_central_corner() {
        let center = Square::new(4, 4); // 5五
        let mask_c = king_zone_mask(center);
        let count_c: usize = mask_c.iter().filter(|&&b| b).count();
        assert_eq!(count_c, 9, "center king zone should have 9 sq");

        let corner = Square::new(0, 0); // 1一
        let mask_a = king_zone_mask(corner);
        let count_a: usize = mask_a.iter().filter(|&&b| b).count();
        assert_eq!(count_a, 4, "corner king zone should have 4 sq");
    }

    /// piece_type_to_dual_idx: rshogi-core と同 mapping (Pawn=0..Dragon=13)。
    #[test]
    fn piece_type_to_dual_idx_mapping() {
        assert_eq!(piece_type_to_dual_idx(PieceType::Pawn), Some(0));
        assert_eq!(piece_type_to_dual_idx(PieceType::Lance), Some(1));
        assert_eq!(piece_type_to_dual_idx(PieceType::Knight), Some(2));
        assert_eq!(piece_type_to_dual_idx(PieceType::Silver), Some(3));
        assert_eq!(piece_type_to_dual_idx(PieceType::Bishop), Some(4));
        assert_eq!(piece_type_to_dual_idx(PieceType::Rook), Some(5));
        assert_eq!(piece_type_to_dual_idx(PieceType::Gold), Some(6));
        assert_eq!(piece_type_to_dual_idx(PieceType::King), Some(7));
        assert_eq!(piece_type_to_dual_idx(PieceType::ProPawn), Some(8));
        assert_eq!(piece_type_to_dual_idx(PieceType::ProLance), Some(9));
        assert_eq!(piece_type_to_dual_idx(PieceType::ProKnight), Some(10));
        assert_eq!(piece_type_to_dual_idx(PieceType::ProSilver), Some(11));
        assert_eq!(piece_type_to_dual_idx(PieceType::Horse), Some(12));
        assert_eq!(piece_type_to_dual_idx(PieceType::Dragon), Some(13));
        assert_eq!(piece_type_to_dual_idx(PieceType::None), None);
    }

    /// 空盤面 (= 王のみ): attack 0、 defend は王自身が自玉 zone に効くため 1 駒分。
    ///
    /// ★ ParityCheck Case 1 (= rshogi-core `compute_dual_zero_weight_returns_zero` 同 SFEN)
    /// 初期局面 (= startpos) で defend が立つことを rshogi-core 側 weight=1 test で確認。
    #[test]
    fn collect_dual_empty_board_kings_only() {
        // 黒玉 5九 + 白玉 5一、 他駒なし。
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);

        // Black 視点: 黒玉自身が自 king zone (= 5九 + 8 近傍) に効く (= king_effect は 8 近傍)。
        // king は zone 内に自分自身がいるので defend は king_effect & own_zone =
        // 8 近傍 ∩ own_zone (= 5九 + 8 近傍) = 8 近傍 != 空 → defend active。
        // attack: 5九玉の利き (= 8 近傍) は 5一白玉 zone (= 5一 + 8 近傍 = 4一-6二) と
        // 交差しない (黒玉位置 5九 はほぼ反対側) → attack = 0。
        let (_a, _d, n_a_b, n_d_b) = collect_dual_active_for_side(&board, Color::Black, 0);
        assert_eq!(n_a_b, 0, "black king attack should be 0 (no enemy king zone reach)");
        assert_eq!(n_d_b, 1, "black king defends own king zone (= 8 近傍 ∩ own_zone)");

        // White 視点: 同様に。
        let (_a, _d, n_a_w, n_d_w) = collect_dual_active_for_side(&board, Color::White, 0);
        assert_eq!(n_a_w, 0, "white king attack should be 0");
        assert_eq!(n_d_w, 1, "white king defends own king zone");
    }

    /// 飛車 attack: 5五黒飛車 → 5一白玉 zone (= 5一 + 8 近傍 内 5二) に縦に届く。
    ///
    /// ★ ParityCheck Case 2 (= rshogi-core `compute_dual_attack_head_smoke` 同 SFEN)
    /// SFEN "4k4/9/9/9/4R4/9/9/9/4K4 b - 1" 等価:
    /// - 白玉 5一 (file=4, rank=0)
    /// - 黒飛車 5五 (file=4, rank=4)
    /// - 黒玉 5九 (file=4, rank=8)
    #[test]
    fn collect_dual_rook_attack_smoke() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0); // 5一
        let rb = Square::new(4, 4); // 5五
        let bk = Square::new(4, 8); // 5九
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        // Black 視点:
        //   - 黒飛車 5五 → enemy zone (= 5一 + 8 近傍 = 5二/4二/6二 等) に縦 ray が届く
        //     → attack active
        //   - 黒飛車 5五 → own zone (= 5九 + 8 近傍 = 5八/4八/6八 等) には届かないが、
        //     縦 ray 下方向で 5八 まで届く (= rank 5..8 までは黒玉 5九が遮蔽前) → own zone 5八 に届く
        //     → defend active
        //   - 黒玉 5九 → enemy zone (= 5一 zone) に届かない → attack inactive
        //   - 黒玉 5九 → own zone (= 自分の周囲) に届く → defend active
        // → black persp 合計 (n_a, n_d) = (1, 2)
        let (_a, _d, n_a_b, n_d_b) = collect_dual_active_for_side(&board, Color::Black, 0);
        assert_eq!(n_a_b, 1, "rook should attack enemy king zone (= 1 attacker), got {}", n_a_b);
        assert_eq!(n_d_b, 2, "black rook + king defend own zone (= 2 defenders), got {}", n_d_b);

        // White 視点: 白玉のみ → own zone defend のみ。
        let (_a, _d, n_a_w, n_d_w) = collect_dual_active_for_side(&board, Color::White, 0);
        assert_eq!(n_a_w, 0, "white has no attacker (only king)");
        assert_eq!(n_d_w, 1, "white king defends own zone");

        // ★ ParityCheck と一致: rshogi-core 側 weight=1 で
        //    attack_sum = n_a_b + n_a_w = 1
        //    defend_sum = n_d_b + n_d_w = 3
        // (= parity_check_rook_attack_board test の assert と一致)
        assert_eq!(n_a_b + n_a_w, 1, "parity: total attack count");
        assert_eq!(n_d_b + n_d_w, 3, "parity: total defend count");
    }

    /// feature index が正しい構造を持つ:
    /// 飛車 attack case で active index = `dual_feature_index(0, bucket, sq_idx_5五_persp_black, piece_idx_Rook=5)`
    /// であることを sample。
    #[test]
    fn collect_dual_active_index_decomposition() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0);
        let rb = Square::new(4, 4);
        let bk = Square::new(4, 8);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        let (att, _def, n_a, _n_d) = collect_dual_active_for_side(&board, Color::Black, 3);
        // 飛車 5五 (= sq.index() = 4*9+4 = 40) の attack idx
        // perspective=Black なら sq_idx = 40 (identity)
        let expected_rook_idx = dual_feature_index(0, 3, 40, 5 /* Rook */) as u32;
        let found = att.iter().take(n_a).any(|&v| v == expected_rook_idx);
        assert!(found, "expected rook 5五 attack idx {} not found in {:?}", expected_rook_idx, &att[..n_a]);
    }

    /// bucket=7 (終盤) でも index 計算が一貫していることを sanity check。
    #[test]
    fn collect_dual_bucket_offset() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0);
        let rb = Square::new(4, 4);
        let bk = Square::new(4, 8);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        let (att0, _, n0, _) = collect_dual_active_for_side(&board, Color::Black, 0);
        let (att7, _, n7, _) = collect_dual_active_for_side(&board, Color::Black, 7);
        // bucket だけ違うので active 数は同じ、 index は bucket * 81*14 = bucket * 1134 シフト。
        assert_eq!(n0, n7);
        for i in 0..n0 {
            let diff = att7[i] as i64 - att0[i] as i64;
            assert_eq!(diff, 7 * (81 * 14) as i64, "bucket offset mismatch");
        }
    }

    /// Wrapper struct: dual_enabled off で aux dims 不変 (= inner と同等)。
    #[test]
    fn wrapper_default_no_dual() {
        let input = ShogiHalfKaHmKingAttackerDual::default();
        assert!(!input.dual_enabled());
        let dims = input.auxiliary_dense_dims();
        // dual off + pin off (= inner default) → "hand_count" 1 entry のみ
        assert_eq!(dims.len(), 1);
        assert_eq!(dims[0].0, "hand_count");
    }

    /// Wrapper struct: dual_enabled on で **sparse** aux dims に 4 entry 追加。
    ///
    /// Phase 3b-sparse 以降: dense path は廃止、 dual は 4 sparse input で渡す。
    #[test]
    fn wrapper_with_dual_adds_two_entries() {
        let input = ShogiHalfKaHmKingAttackerDual::new().with_dual_enabled(true);
        // dense aux: hand_count のみ (= dual は dense から削除済)
        let dense_dims = input.auxiliary_dense_dims();
        assert_eq!(dense_dims.len(), 1);
        assert_eq!(dense_dims[0].0, "hand_count");

        // sparse aux: 4 king-attacker + 4 cross = 8 sparse input
        let sparse_dims = input.auxiliary_sparse_dims();
        assert_eq!(sparse_dims.len(), 8);
        assert_eq!(sparse_dims[0].0, DUAL_SPARSE_NAME_ATTACK_BLACK);
        assert_eq!(sparse_dims[0].1, DUAL_DIM_PER_HEAD);
        assert_eq!(sparse_dims[0].2, DUAL_MAX_ACTIVE_PER_HEAD);
        assert_eq!(sparse_dims[1].0, DUAL_SPARSE_NAME_DEFEND_BLACK);
        assert_eq!(sparse_dims[2].0, DUAL_SPARSE_NAME_ATTACK_WHITE);
        assert_eq!(sparse_dims[3].0, DUAL_SPARSE_NAME_DEFEND_WHITE);
        assert_eq!(sparse_dims[4].0, DUAL_CROSS_THREAT_BLACK);
        assert_eq!(sparse_dims[4].1, DUAL_CROSS_DIM_PER_HEAD);
        assert_eq!(sparse_dims[4].2, DUAL_CROSS_MAX_ACTIVE_PER_HEAD);
        assert_eq!(sparse_dims[5].0, DUAL_CROSS_DEFENDED_BLACK);
        assert_eq!(sparse_dims[6].0, DUAL_CROSS_THREAT_WHITE);
        assert_eq!(sparse_dims[7].0, DUAL_CROSS_DEFENDED_WHITE);
    }

    /// Wrapper struct: dual + pin 同時有効で dense aux = hand_count + pin × 2 = 3 entry、
    /// sparse aux = dual × 8 entry (= 4 king-attacker + 4 cross)。
    #[test]
    fn wrapper_with_dual_and_pin() {
        let input = ShogiHalfKaHmKingAttackerDual::new().with_pin_aware(true).with_dual_enabled(true);
        // dense aux: hand_count + pin_black + pin_white = 3 entry (= dual は sparse 経路で別途)
        let dense_dims = input.auxiliary_dense_dims();
        assert_eq!(dense_dims.len(), 3);
        assert_eq!(dense_dims[0].0, "hand_count");
        assert_eq!(dense_dims[1].0, "pin_features_black");
        assert_eq!(dense_dims[2].0, "pin_features_white");

        // sparse aux: 4 king-attacker + 4 cross = 8 sparse input
        let sparse_dims = input.auxiliary_sparse_dims();
        assert_eq!(sparse_dims.len(), 8);
    }

    /// arm 6 (= 2026-05-27): `with_dual_cross_defended_enabled(false)` で D2 cross
    /// head 1 (= defended) の sparse input を 2 個削除する。
    /// - default = 8 sparse input (= 4 KA + 4 cross)
    /// - defended OFF = 6 sparse input (= 4 KA + 2 cross threat)
    /// - defended OFF + KA OFF = 2 sparse input (= 2 cross threat のみ)
    #[test]
    fn wrapper_dual_cross_defended_disabled() {
        // ケース 1: defended OFF (KA は ON のまま)
        let input =
            ShogiHalfKaHmKingAttackerDual::new().with_dual_enabled(true).with_dual_cross_defended_enabled(false);
        let sparse_dims = input.auxiliary_sparse_dims();
        // 4 KA + 2 cross threat (black/white) = 6 sparse input
        assert_eq!(sparse_dims.len(), 6);
        assert_eq!(sparse_dims[0].0, DUAL_SPARSE_NAME_ATTACK_BLACK);
        assert_eq!(sparse_dims[1].0, DUAL_SPARSE_NAME_DEFEND_BLACK);
        assert_eq!(sparse_dims[2].0, DUAL_SPARSE_NAME_ATTACK_WHITE);
        assert_eq!(sparse_dims[3].0, DUAL_SPARSE_NAME_DEFEND_WHITE);
        assert_eq!(sparse_dims[4].0, DUAL_CROSS_THREAT_BLACK);
        assert_eq!(sparse_dims[5].0, DUAL_CROSS_THREAT_WHITE);
        // DUAL_CROSS_DEFENDED_BLACK / DUAL_CROSS_DEFENDED_WHITE が消えていることを確認
        for (name, _, _) in sparse_dims.iter() {
            assert!(
                name != DUAL_CROSS_DEFENDED_BLACK && name != DUAL_CROSS_DEFENDED_WHITE,
                "defended sparse input must be removed: {}",
                name
            );
        }
        assert!(!input.dual_cross_defended_enabled());

        // ケース 2: defended OFF + KA OFF (arm 6 production 想定)
        let input2 = ShogiHalfKaHmKingAttackerDual::new()
            .with_dual_enabled(true)
            .with_king_attacker_enabled(false)
            .with_dual_cross_defended_enabled(false);
        let sparse_dims2 = input2.auxiliary_sparse_dims();
        // 2 cross threat (black/white) のみ
        assert_eq!(sparse_dims2.len(), 2);
        assert_eq!(sparse_dims2[0].0, DUAL_CROSS_THREAT_BLACK);
        assert_eq!(sparse_dims2[1].0, DUAL_CROSS_THREAT_WHITE);
    }

    /// dual_cross_index の境界値 check。
    #[test]
    fn dual_cross_index_range() {
        let idx0 = dual_cross_index(0, 0, 0, 0);
        let idx_max =
            dual_cross_index(NUM_DUAL_PSQT_BUCKETS - 1, DUAL_SQUARES - 1, DUAL_PIECE_TYPES - 1, DUAL_PIECE_TYPES - 1);
        assert_eq!(idx0, 0);
        assert_eq!(idx_max, DUAL_CROSS_DIM_PER_HEAD - 1);
        assert_eq!(DUAL_CROSS_DIM_PER_HEAD, 127_008);
    }

    /// 初期局面 (= startpos に近い 玉だけ盤面) で cross feature 集合は空 (= 自軍駒のみだと相手駒なし)。
    #[test]
    fn collect_dual_cross_kings_only() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);

        // 黒視点: threat = 0 (相手玉 wk への利き無し)、 defended = 0 (自軍駒 1 のみ)
        let (_t, _d, n_t, n_d) = collect_dual_cross_for_side(&board, Color::Black, 0);
        assert_eq!(n_t, 0, "no enemy piece in king attack range");
        assert_eq!(n_d, 0, "only own king present, no other own piece to defend");
    }

    /// 飛車が相手歩を攻撃 + 自軍金が自軍飛を守る pattern で threat=1 + defended=1。
    #[test]
    fn collect_dual_cross_rook_threat_and_defended() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8); // 5九
        let wk = Square::new(4, 0); // 5一
        let rb = Square::new(4, 4); // 5五 黒飛車
        let wp = Square::new(4, 2); // 5三 白歩 (= 飛車の line 上)
        let bg = Square::new(3, 4); // 4五 黒金 (= 飛車を横から守る)
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[wp.index()] = Piece::new(Color::White, PieceType::Pawn);
        board.board[bg.index()] = Piece::new(Color::Black, PieceType::Gold);

        // 黒視点: threat = 飛車→歩 (= 1)、 defended = 飛車←金 (= 1) + 金←飛 = 検査
        let (_t, _d, n_t, n_d) = collect_dual_cross_for_side(&board, Color::Black, 0);
        // 飛車 5五 から見て 5三 (相手歩) は line 上 (= attack)
        // 金 4五 から見て 5五 (= 飛車) は隣接 = 守備
        // 飛車 5五 から見て 4五 (= 金) は隣接 line 上 = 守備
        // つまり defended: 飛車 from 金 + 金 from 飛車 = 2 件 (+ king related があれば +α)
        assert!(n_t >= 1, "rook should threaten white pawn, got {}", n_t);
        assert!(n_d >= 2, "defended chain rook<->gold, got {}", n_d);
    }

    /// shorthand に dual marker が反映される。
    #[test]
    fn wrapper_shorthand_reflects_dual() {
        let s_off = ShogiHalfKaHmKingAttackerDual::new().shorthand();
        assert!(!s_off.contains("+dual"));
        let s_on = ShogiHalfKaHmKingAttackerDual::new().with_dual_enabled(true).shorthand();
        assert!(s_on.contains("+dual"));
    }

    // =========================================================================
    // ParityCheck-weight 拡張 (Phase 3b-graph、 2026-05-26)
    // =========================================================================
    //
    // bullet 側 sparse-emit-on-dense と rshogi-core `compute_dual_king_attacker` の
    // 完全一致を検証する。 両 implementation は **同 algorithm** (= 同 piece scan、
    // 同 occupancy-aware ray walk、 同 king_zone_mask、 同 dual_feature_index 式) を
    // 共有しているため、 active 集合 + weight 値が同じなら i32 sum も完全一致する。
    //
    // ★ rshogi-core 側 test (= `parity_check_*` group) と同 SFEN/同 weight pattern を
    //   使う前提。 物理的に同一 binary 上では実行できないため、 bullet 側で **再現**
    //   できる: 同 board state を ShogiBoard で構築 + 同 weight pattern を local Vec
    //   に作り、 active 集合の dot-product を計算して既知期待値と比較する。

    /// 黒側盤面 (盤上 黒玉 + 黒飛車 + 白玉) で weight pattern を仕込み、
    /// rshogi-core `compute_dual_king_attacker` と等価な sum を bullet 側で計算する。
    ///
    /// rshogi-core パターン (= `parity_check_pattern_weights_rook_attack` 同等):
    /// - weight[head=0, bucket=3, sq=任意, piece=Rook(5)] = 1234
    /// - 他 = 0
    /// - 期待 attack_sum: 5五黒飛車 active (= 1 駒) → 1234 を 1 回 lookup → sum = 1234
    /// - 期待 defend_sum: 0 (= head=1 weight は全 0)
    #[test]
    fn parity_check_weight_pattern_rook_attack() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0); // 5一
        let rb = Square::new(4, 4); // 5五
        let bk = Square::new(4, 8); // 5九
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        // weight tensor: 黒視点用 (Box で heap allocate、 stack 18,144 i16 = 36 KB を回避)
        let mut weight_black: Vec<i16> = vec![0i16; DUAL_DIMENSIONS_PER_PERSPECTIVE];
        let weight_white: Vec<i16> = vec![0i16; DUAL_DIMENSIONS_PER_PERSPECTIVE];

        // 全 sq に対し head=0 (= attack), bucket=3, piece=Rook(5) を 1234 に設定。
        // 飛車が active sq には何度も lookup しない (sq は 1 か所のみ)、 これで
        // 1234 が attack 側に 1 回 contribute する。
        for sq in 0..DUAL_SQUARES {
            let idx = dual_feature_index(0, 3, sq, 5);
            weight_black[idx] = 1234;
        }

        // bullet 側 active index 集合を bucket=3 で取得
        let (att_b, def_b, n_a_b, n_d_b) = collect_dual_active_for_side(&board, Color::Black, 3);
        let (_att_w, _def_w, n_a_w, n_d_w) = collect_dual_active_for_side(&board, Color::White, 3);

        // bullet 側で i32 sum 計算 (= rshogi-core `compute_dual_king_attacker` 等価)
        let mut attack_sum: i32 = 0;
        let mut defend_sum: i32 = 0;
        for &idx in att_b.iter().take(n_a_b) {
            attack_sum += weight_black[idx as usize] as i32;
        }
        for &idx in def_b.iter().take(n_d_b) {
            defend_sum += weight_black[idx as usize] as i32;
        }
        // 白側は head=0 attack に乗らない (= 白側 attack 0 expected)
        assert_eq!(n_a_w, 0, "white has no attacker");
        // 白玉 own defend active は期待値 1
        assert_eq!(n_d_w, 1, "white king defends own zone");
        // 白視点 weight は全 0 → 白側 contribution = 0
        let _ = weight_white;

        // n_a_b = 1 (飛車 attack)、 n_d_b = 2 (飛車 defend + 黒玉 defend)
        assert_eq!(n_a_b, 1, "expected 1 black attacker");
        assert_eq!(n_d_b, 2, "expected 2 black defenders");

        // attack_sum: 飛車 (1234) のみ contribute、 黒玉は head=0 attack に乗らない (敵 zone に届かない)
        assert_eq!(attack_sum, 1234, "rook attack should sum to 1234");
        // defend_sum: head=1 weight は全 0 なので 0
        assert_eq!(defend_sum, 0, "defend head weights are zero → defend_sum = 0");
    }

    /// 防御 head に weight を仕込んで defend_sum が立つことを確認。
    /// rshogi-core `parity_check_pattern_weights_rook_defend` 等価。
    #[test]
    fn parity_check_weight_pattern_rook_defend() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0); // 5一
        let rb = Square::new(4, 4); // 5五
        let bk = Square::new(4, 8); // 5九
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        // head=1 (defend) Rook weight に -567 を設定 (= 全 bucket × 全 sq、 active 1 駒のみ count)
        let mut weight_black: Vec<i16> = vec![0i16; DUAL_DIMENSIONS_PER_PERSPECTIVE];
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            for sq in 0..DUAL_SQUARES {
                let idx = dual_feature_index(1, bucket, sq, 5);
                weight_black[idx] = -567;
            }
        }
        // 黒玉 (= piece=King=7) の defend にも -100 を設定
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            for sq in 0..DUAL_SQUARES {
                let idx = dual_feature_index(1, bucket, sq, 7);
                weight_black[idx] = -100;
            }
        }

        let (att_b, def_b, n_a_b, n_d_b) = collect_dual_active_for_side(&board, Color::Black, 5);

        let mut attack_sum: i32 = 0;
        let mut defend_sum: i32 = 0;
        for &idx in att_b.iter().take(n_a_b) {
            attack_sum += weight_black[idx as usize] as i32;
        }
        for &idx in def_b.iter().take(n_d_b) {
            defend_sum += weight_black[idx as usize] as i32;
        }

        assert_eq!(n_a_b, 1);
        assert_eq!(n_d_b, 2); // 飛車 defend + 黒玉 defend

        // attack_sum: head=0 weight 0 なので 0
        assert_eq!(attack_sum, 0);
        // defend_sum: 飛車 defend (-567) + 黒玉 defend (-100) = -667
        assert_eq!(defend_sum, -667, "defend = rook(-567) + king(-100)");
    }

    /// 両 perspective + 両 head の sum を一括検証 (= rshogi-core
    /// `compute_dual_king_attacker` 完全模倣)。
    ///
    /// 黒視点 attack に Rook=1000、 白視点 defend に King=-500 を仕込み、
    /// `(attack - defend) = (1000+0) - (0+(-500)) = 1500` を得る。
    #[test]
    fn parity_check_weight_both_perspectives() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0); // 5一
        let rb = Square::new(4, 4); // 5五
        let bk = Square::new(4, 8); // 5九
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        // 黒視点 weight: attack Rook = 1000
        let mut w_black: Vec<i16> = vec![0i16; DUAL_DIMENSIONS_PER_PERSPECTIVE];
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            for sq in 0..DUAL_SQUARES {
                w_black[dual_feature_index(0, bucket, sq, 5)] = 1000;
            }
        }
        // 白視点 weight: defend King = -500
        let mut w_white: Vec<i16> = vec![0i16; DUAL_DIMENSIONS_PER_PERSPECTIVE];
        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
            for sq in 0..DUAL_SQUARES {
                w_white[dual_feature_index(1, bucket, sq, 7)] = -500;
            }
        }

        let bucket = 2;
        let (att_b, def_b, n_a_b, n_d_b) = collect_dual_active_for_side(&board, Color::Black, bucket);
        let (att_w, def_w, n_a_w, n_d_w) = collect_dual_active_for_side(&board, Color::White, bucket);

        let mut attack_sum: i32 = 0;
        let mut defend_sum: i32 = 0;
        for &idx in att_b.iter().take(n_a_b) {
            attack_sum += w_black[idx as usize] as i32;
        }
        for &idx in att_w.iter().take(n_a_w) {
            attack_sum += w_white[idx as usize] as i32;
        }
        for &idx in def_b.iter().take(n_d_b) {
            defend_sum += w_black[idx as usize] as i32;
        }
        for &idx in def_w.iter().take(n_d_w) {
            defend_sum += w_white[idx as usize] as i32;
        }

        // attack: 黒飛車 (1000) のみ、 白側 attack なし → 1000
        // defend: 黒側 defend Rook/King (= weight 0) + 白側 defend King (-500) → -500
        assert_eq!(attack_sum, 1000, "attack from black rook only");
        assert_eq!(defend_sum, -500, "defend from white king only");
        // 最終 dual score
        assert_eq!(attack_sum - defend_sum, 1500);
    }

    /// graph forward に渡す dense vector の sign 規則を直接確認する:
    /// attack head の active index には +1.0、 defend head の active index には -1.0。
    ///
    /// `fill_dual_features_for_side` の内部 logic を再現する形で、 dense vector の
    /// 既知 (head, sq, piece) entry に対し sign を 1 駒 case (= rook attack smoke) で
    /// verify する。 ★ fill が PackedSfenValue 経由のため、 SFEN encode 不要な
    /// `collect_dual_active_for_side` 直呼び + 手動 fill で sign 規則を test する。
    #[test]
    fn parity_check_fill_sign_convention() {
        let mut board = ShogiBoard::default();
        let wk = Square::new(4, 0);
        let rb = Square::new(4, 4);
        let bk = Square::new(4, 8);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);

        let bucket = 0;
        let (att, def, n_a, n_d) = collect_dual_active_for_side(&board, Color::Black, bucket);

        // `fill_dual_features_for_side` の sign 規則を再現:
        // attack head index → +1.0、 defend head index → -1.0
        let mut out = vec![0.0f32; DUAL_DIMENSIONS_PER_PERSPECTIVE];
        for &idx in att.iter().take(n_a) {
            out[idx as usize] = 1.0;
        }
        for &idx in def.iter().take(n_d) {
            out[idx as usize] = -1.0;
        }

        // 非 0 entry の sign を確認: attack(+1.0) と defend(-1.0) 以外あってはならない。
        let mut count_pos = 0;
        let mut count_neg = 0;
        for &v in out.iter() {
            if v == 0.0 {
                continue;
            }
            if v == 1.0 {
                count_pos += 1;
            } else if v == -1.0 {
                count_neg += 1;
            } else {
                panic!("unexpected fill value {} (must be 0/+1/-1)", v);
            }
        }
        assert_eq!(count_pos, n_a, "attack count should match +1 entries");
        assert_eq!(count_neg, n_d, "defend count should match -1 entries");
        assert_eq!(count_pos, 1, "rook attack: 1 active");
        assert_eq!(count_neg, 2, "rook defend + king defend: 2 active");
    }

    /// 詰む直前の盤面で active feature 数の上限を確認。
    /// 8 駒 attack + 8 駒 defend (上限 50/side) を超えないこと。
    #[test]
    fn parity_check_active_count_upper_bound() {
        // 黒玉 5九 + 白玉 5一 + 黒側 8 駒 + 白側 8 駒 (= 攻め駒密集) 想定。
        // 上限 50/side を superseded しないかの sanity。
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);

        // 黒側 8 駒を白玉周辺 (4二〜6二) に配置
        let attackers_b = [
            (Square::new(3, 1), PieceType::Gold),
            (Square::new(4, 1), PieceType::Silver),
            (Square::new(5, 1), PieceType::Gold),
            (Square::new(3, 2), PieceType::Knight),
            (Square::new(4, 2), PieceType::Rook),
            (Square::new(5, 2), PieceType::Knight),
            (Square::new(2, 3), PieceType::Bishop),
            (Square::new(6, 3), PieceType::Bishop),
        ];
        for (sq, pt) in attackers_b.iter() {
            board.board[sq.index()] = Piece::new(Color::Black, *pt);
        }

        let (_att, _def, n_a, n_d) = collect_dual_active_for_side(&board, Color::Black, 0);

        // 上限 50/side を超えないこと
        assert!(n_a + n_d <= DUAL_MAX_ACTIVE_PER_SIDE * 2);
        assert!(n_a <= DUAL_MAX_ACTIVE_PER_SIDE);
        assert!(n_d <= DUAL_MAX_ACTIVE_PER_SIDE);
        // 少なくとも数 attacker は active
        assert!(n_a >= 4, "expected several attackers in dense position");
    }

    /// bucket=3 と bucket=5 で同 board の active index 集合が **bucket offset 分のみ**
    /// 異なることを 10 局面サンプリングで確認。 これにより graph weight tensor の
    /// `[bucket]` 次元アクセスが線形 + 連続 (= 1134 offset 単位) であることを保証。
    #[test]
    fn parity_check_bucket_offset_invariance() {
        // 5 board state を構築して bucket 3 と 5 を比較
        let positions: [(u8, u8); 5] = [
            (Square::new(0, 0).0, Square::new(8, 8).0),
            (Square::new(4, 4).0, Square::new(4, 0).0),
            (Square::new(4, 8).0, Square::new(4, 0).0),
            (Square::new(2, 6).0, Square::new(6, 2).0),
            (Square::new(0, 8).0, Square::new(8, 0).0),
        ];

        for (bk_idx, wk_idx) in positions.iter() {
            let mut board = ShogiBoard::default();
            let bk = Square(*bk_idx);
            let wk = Square(*wk_idx);
            board.black_king_sq = bk;
            board.white_king_sq = wk;
            board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
            board.board[wk.index()] = Piece::new(Color::White, PieceType::King);

            // 飛車を中央に置く
            let rb = Square::new(4, 4);
            if rb.index() != bk.index() && rb.index() != wk.index() {
                board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
            }

            let (att3, def3, n_a3, n_d3) = collect_dual_active_for_side(&board, Color::Black, 3);
            let (att5, def5, n_a5, n_d5) = collect_dual_active_for_side(&board, Color::Black, 5);

            assert_eq!(n_a3, n_a5);
            assert_eq!(n_d3, n_d5);
            // bucket diff = 5 - 3 = 2、 offset = 2 × (81 × 14) = 2268
            let expected_offset = 2i64 * (DUAL_SQUARES * DUAL_PIECE_TYPES) as i64;
            for i in 0..n_a3 {
                assert_eq!(att5[i] as i64 - att3[i] as i64, expected_offset);
            }
            for i in 0..n_d3 {
                assert_eq!(def5[i] as i64 - def3[i] as i64, expected_offset);
            }
        }
    }

    // =========================================================================
    // KingRing tests (arm 11、 2026-05-28)
    // =========================================================================

    /// kingring 定数が rshogi-core spec と一致 (= 静的契約)。
    #[test]
    fn kingring_constants_match_rshogi_core() {
        assert_eq!(KINGRING_NUM_BUCKETS, 8);
        assert_eq!(KINGRING_NUM_SQ, 24);
        assert_eq!(KINGRING_NUM_PIECE, 13);
        assert_eq!(KINGRING_NUM_STATE, 4);
        assert_eq!(KINGRING_PER_BUCKET_DIM, 1_248);
        assert_eq!(KINGRING_DIM_PER_PERSP, 9_984);
        assert_eq!(KINGRING_TOTAL_COUNT, 19_968);
        assert_eq!(KINGRING_SUB_MAGIC, b"P2KRING\0");
        assert_eq!(KINGRING_SECTION_VERSION, 1);
        assert_eq!(KINGRING_MAX_ACTIVE_PER_PERSP, 24);
    }

    /// piece_type → kingring idx の mapping が rshogi-core と完全一致。
    #[test]
    fn kingring_piece_type_idx_mapping() {
        assert_eq!(piece_type_to_kingring_idx(PieceType::Pawn), Some(0));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Lance), Some(1));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Knight), Some(2));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Silver), Some(3));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Bishop), Some(4));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Rook), Some(5));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Gold), Some(6));
        assert_eq!(piece_type_to_kingring_idx(PieceType::King), None); // King は除外
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProPawn), Some(7));
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProLance), Some(8));
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProKnight), Some(9));
        assert_eq!(piece_type_to_kingring_idx(PieceType::ProSilver), Some(10));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Horse), Some(11));
        assert_eq!(piece_type_to_kingring_idx(PieceType::Dragon), Some(12));
        assert_eq!(piece_type_to_kingring_idx(PieceType::None), None);
    }

    /// encode_local の境界値: idx 0 と idx 1247 が full 範囲を carry する。
    #[test]
    fn kingring_encode_local_range() {
        assert_eq!(kingring_encode_local(0, 0, 0), 0);
        assert_eq!(
            kingring_encode_local(KINGRING_NUM_SQ - 1, KINGRING_NUM_PIECE - 1, KINGRING_NUM_STATE - 1),
            KINGRING_PER_BUCKET_DIM - 1
        );
    }

    /// build_ring: 中央 (= 5五) で 24 sq 全て inside、 角 (= 1一) で一部 outside。
    #[test]
    fn kingring_build_ring_center_corner() {
        let center = Square::new(4, 4); // 5五
        let ring_c = kingring_build_ring(center, Color::Black);
        let inside_c: usize = ring_c.iter().filter(|s| s.is_some()).count();
        assert_eq!(inside_c, 24, "center ring all inside board");

        let corner = Square::new(0, 0); // 1一
        let ring_a = kingring_build_ring(corner, Color::Black);
        let inside_a: usize = ring_a.iter().filter(|s| s.is_some()).count();
        assert!(inside_a < 24, "corner ring should drop some outside-board sq");
        assert!(inside_a >= 5, "corner ring should still have a few inside");
    }

    /// build_ring: White perspective は 180° 回転で同じ ring。 中央局面の自玉 black 5九 と
    /// White 5一 で、 自玉視点 ring entry が **同一** であることを sanity check (= 鏡像対称)。
    #[test]
    fn kingring_build_ring_white_perspective_symmetric() {
        // 黒玉 5九 (file=4, rank=8)、 white 視点で見たときは 玉の周囲 24 sq だが、 180 度回転で
        // 入射方向が flip する。 ring の **inside 数** は同じであるべき。
        let bk = Square::new(4, 8);
        let ring_b = kingring_build_ring(bk, Color::Black);
        let inside_b: usize = ring_b.iter().filter(|s| s.is_some()).count();

        let wk = Square::new(4, 0); // 白玉 5一 = 180 度反転位置
        let ring_w = kingring_build_ring(wk, Color::White);
        let inside_w: usize = ring_w.iter().filter(|s| s.is_some()).count();

        // 5九 黒玉 vs 5一 白玉 (180° 関係) では同 ring sq 数になるはず
        assert_eq!(inside_b, inside_w, "180-rotated king positions should have same inside count");
    }

    /// collect_active: 空盤面 (= 玉のみ) で kingring active は 0 (= 自玉自身を除いた ring 24 sq に
    /// 何の駒もない + 利き計算上の attacker もないため empty inactive)。
    #[test]
    fn kingring_collect_kings_only_empty_active() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);

        let (_buf, n) = collect_kingring_active_for_side(&board, Color::Black, 0);
        // 黒玉 5九 周辺 ring 24 sq に何の駒も attacker もない (= 黒玉自身は ring 外、 白玉 5一 は
        // 利きの届く範囲外、 5九 周辺 ring 内の empty sq は own attacker も enemy attacker もなし)。
        assert_eq!(n, 0, "no piece/attacker in ring → all inactive");
    }

    /// collect_active: 飛車を 5五 に置くと 黒玉 5九 周辺 ring 内 5八 (= 飛車の縦 ray 上) で
    /// state 0 (own controlled) で active 化、 さらに飛車自身が ring 外なら他には影響なし。
    /// 5八 は 5九 から見て (df=0, dr=-1) → ring idx (0+2)*5 + (-1+2)=2+11 のような位置。
    #[test]
    fn kingring_collect_rook_in_ring_smoke() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8); // 5九 黒玉
        let wk = Square::new(4, 0); // 5一 白玉
        let rb = Square::new(4, 4); // 5五 黒飛車 (= ring 外、 玉から rank 4 = 4 sq 離れている)
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);

        let (buf, n) = collect_kingring_active_for_side(&board, Color::Black, 0);
        // 飛車は 5五 (= ring 外、 玉から rank 距離 4)、 飛車の縦 ray は 5五 → 5六 → 5七 → 5八 → 5九
        // (= 5九 で黒玉と衝突)。 ring 内 sq は 5七 (rank=6) と 5八 (rank=7) が 5九 から rank 距離 ≤ 2
        // の範囲 (= rank 6/7) に入る。
        //   5七 (rank 6): 玉から (df=0, dr=-2)、 ring 内、 empty + 黒飛車 attack → state 0、
        //                 piece = Rook (idx 5)
        //   5八 (rank 7): 玉から (df=0, dr=-1)、 ring 内、 empty + 黒飛車 attack → state 0、
        //                 piece = Rook (idx 5)
        // よって少なくとも 2 entry active (= 飛車以外 ring 内に駒 / 攻撃なし想定)。
        assert!(n >= 2, "rook ray should activate at least 5七 and 5八 in own king ring, got {n}");

        // 全 active が bucket 0 範囲 (= 0..1248) 内にあることを確認
        for i in 0..n {
            assert!(buf[i] < KINGRING_PER_BUCKET_DIM as u32, "bucket 0 active idx out of range: {}", buf[i]);
        }
    }

    /// bucket offset の sanity: 同 board で bucket 0 と bucket 5 を比較すると flat idx は
    /// 完全に `5 * 1248 = 6240` の定数オフセット分シフトする。
    #[test]
    fn kingring_collect_bucket_offset_consistent() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        let rb = Square::new(4, 4);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);

        let (buf0, n0) = collect_kingring_active_for_side(&board, Color::Black, 0);
        let (buf5, n5) = collect_kingring_active_for_side(&board, Color::Black, 5);
        assert_eq!(n0, n5, "bucket だけ違うので active 数は同じ");
        let expected_offset = 5 * KINGRING_PER_BUCKET_DIM as i64;
        for i in 0..n0 {
            let diff = buf5[i] as i64 - buf0[i] as i64;
            assert_eq!(diff, expected_offset, "bucket 5 should shift by exactly 5*1248 = 6240");
        }
    }

    /// auxiliary_sparse_dims: kingring 単独 ON で 2 entry 追加 (= black / white、 dim 9984、
    /// max_active 24)。 dual も同時に ON すると合計が増える。
    #[test]
    fn wrapper_with_kingring_only_adds_two_entries() {
        let input = ShogiHalfKaHmKingAttackerDual::new().with_dual_enabled(false).with_kingring_enabled(true);
        let sparse_dims = input.auxiliary_sparse_dims();
        // dual OFF + kingring ON → kingring 2 entry のみ
        assert_eq!(sparse_dims.len(), 2);
        assert_eq!(sparse_dims[0].0, KINGRING_SPARSE_NAME_BLACK);
        assert_eq!(sparse_dims[0].1, KINGRING_DIM_PER_PERSP);
        assert_eq!(sparse_dims[0].2, KINGRING_MAX_ACTIVE_PER_PERSP);
        assert_eq!(sparse_dims[1].0, KINGRING_SPARSE_NAME_WHITE);
        assert_eq!(sparse_dims[1].1, KINGRING_DIM_PER_PERSP);
        assert_eq!(sparse_dims[1].2, KINGRING_MAX_ACTIVE_PER_PERSP);
        assert!(input.kingring_enabled());
    }

    /// auxiliary_sparse_dims: dual + kingring 同時 ON で kingring 2 entry が末尾に追加される
    /// (= arm 11 = arm 5 + kingring 想定)。 arm 5 構成 (= king_attacker OFF + defended OFF)
    /// で sparse = 2 cross threat + 2 kingring = 4 entry。
    #[test]
    fn wrapper_arm11_dual_plus_kingring_sparse_dims() {
        let input = ShogiHalfKaHmKingAttackerDual::new()
            .with_dual_enabled(true)
            .with_king_attacker_enabled(false)
            .with_dual_cross_defended_enabled(false)
            .with_kingring_enabled(true);
        let sparse_dims = input.auxiliary_sparse_dims();
        // 2 cross threat (black/white) + 2 kingring (black/white) = 4 entry
        assert_eq!(sparse_dims.len(), 4);
        assert_eq!(sparse_dims[0].0, DUAL_CROSS_THREAT_BLACK);
        assert_eq!(sparse_dims[1].0, DUAL_CROSS_THREAT_WHITE);
        assert_eq!(sparse_dims[2].0, KINGRING_SPARSE_NAME_BLACK);
        assert_eq!(sparse_dims[3].0, KINGRING_SPARSE_NAME_WHITE);
    }

    // ---------------------------------------------------------------------
    // PairShape tests (arm 12、 2026-05-29)
    // ---------------------------------------------------------------------

    /// pairshape 定数が rshogi-core spec と一致 (= 静的契約)。
    #[test]
    fn pairshape_constants_match_rshogi_core() {
        assert_eq!(PAIRSHAPE_NUM_BUCKETS, 8);
        assert_eq!(PAIRSHAPE_NUM_PAIR_TYPES, 12);
        assert_eq!(PAIRSHAPE_NUM_SHAPES, 16);
        assert_eq!(PAIRSHAPE_PER_BUCKET_DIM, 192);
        assert_eq!(PAIRSHAPE_DIM_PER_PERSP, 1_536);
        assert_eq!(PAIRSHAPE_TOTAL_COUNT, 3_072);
        assert_eq!(PAIRSHAPE_MAX_ACTIVE_PER_PERSP, 256);
        assert_eq!(PAIRSHAPE_SUB_MAGIC, b"P2PAIR\0\0");
        assert_eq!(PAIRSHAPE_SECTION_VERSION, 1);
        assert!((PAIRSHAPE_QA - 127.0).abs() < 1e-9, "PAIRSHAPE_QA must be 127.0 (= 1.0 バグ厳禁)");
    }

    /// encode_local の range は 0..192。
    #[test]
    fn pairshape_encode_local_range() {
        assert_eq!(pairshape_encode_local(0, 0), 0);
        assert_eq!(
            pairshape_encode_local(PAIRSHAPE_NUM_PAIR_TYPES - 1, PAIRSHAPE_NUM_SHAPES - 1),
            PAIRSHAPE_PER_BUCKET_DIM - 1
        );
    }

    /// shape classifier: SameFile / KnightShape / DiagonalAdjacent の sanity。
    #[test]
    fn pairshape_classifier_sanity() {
        // a=5五 (file=4, rank=4)、 b=5四 (file=4, rank=3) → df=0 → SameFile
        let a = Square::new(4, 4);
        let b = Square::new(4, 3);
        assert_eq!(classify_pair_shape(Color::Black, a, b), PairShape::SameFile);
        // 桂跳び形: a=5五、 b=6三 (file=5, rank=2) → df=1, dr=2 → KnightShape
        let bk = Square::new(5, 2);
        assert_eq!(classify_pair_shape(Color::Black, a, bk), PairShape::KnightShape);
        // 対角隣接: a=5五、 b=6四 (file=5, rank=3) → df=1, dr=1 → DiagonalAdjacent
        let bd = Square::new(5, 3);
        assert_eq!(classify_pair_shape(Color::Black, a, bd), PairShape::DiagonalAdjacent);
    }

    /// collect_active: 玉のみ (= 自分の駒無し) → active 0。
    #[test]
    fn pairshape_collect_kings_only_empty_active() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);

        let (_buf, n) = collect_pairshape_active_for_side(&board, Color::Black, 0);
        assert_eq!(n, 0, "玉のみで pair member 駒無し → active 0");
    }

    /// collect_active: 玉 + 金 + 銀 で Gold-Silver pair 1 件 active。
    #[test]
    fn pairshape_collect_gold_silver_pair() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        let g = Square::new(4, 7); // 5二 黒金
        let s = Square::new(3, 7); // 4二 黒銀
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[g.index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[s.index()] = Piece::new(Color::Black, PieceType::Silver);

        let (buf, n) = collect_pairshape_active_for_side(&board, Color::Black, 0);
        // Gold-Silver は a=Gold × b=Silver で 1 × 1 = 1 pair active (= pt_idx 0)
        assert_eq!(n, 1, "Gold-Silver 1 pair → active 1、 got {n}");
        // 全 active が bucket 0 範囲 (= 0..192) 内
        assert!(buf[0] < PAIRSHAPE_PER_BUCKET_DIM as u32, "active idx out of range: {}", buf[0]);
    }

    /// bucket offset consistency: 同 board で bucket 0 と bucket 5 で active 数同じ、 idx
    /// は `5 * 192 = 960` の定数オフセット。
    #[test]
    fn pairshape_collect_bucket_offset_consistent() {
        let mut board = ShogiBoard::default();
        let bk = Square::new(4, 8);
        let wk = Square::new(4, 0);
        let g = Square::new(4, 7);
        let s = Square::new(3, 7);
        board.black_king_sq = bk;
        board.white_king_sq = wk;
        board.board[bk.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[wk.index()] = Piece::new(Color::White, PieceType::King);
        board.board[g.index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[s.index()] = Piece::new(Color::Black, PieceType::Silver);

        let (buf0, n0) = collect_pairshape_active_for_side(&board, Color::Black, 0);
        let (buf5, n5) = collect_pairshape_active_for_side(&board, Color::Black, 5);
        assert_eq!(n0, n5);
        let expected_offset = 5 * PAIRSHAPE_PER_BUCKET_DIM as i64;
        for i in 0..n0 {
            let diff = buf5[i] as i64 - buf0[i] as i64;
            assert_eq!(diff, expected_offset, "bucket 5 should shift by exactly 5*192=960");
        }
    }

    /// auxiliary_sparse_dims: pairshape 単独 ON で 2 entry (= black / white、 dim 1536、 max_active 256)。
    #[test]
    fn wrapper_with_pairshape_only_adds_two_entries() {
        let input = ShogiHalfKaHmKingAttackerDual::new().with_dual_enabled(false).with_pairshape_enabled(true);
        let sparse_dims = input.auxiliary_sparse_dims();
        assert_eq!(sparse_dims.len(), 2);
        assert_eq!(sparse_dims[0].0, PAIRSHAPE_SPARSE_NAME_BLACK);
        assert_eq!(sparse_dims[0].1, PAIRSHAPE_DIM_PER_PERSP);
        assert_eq!(sparse_dims[0].2, PAIRSHAPE_MAX_ACTIVE_PER_PERSP);
        assert_eq!(sparse_dims[1].0, PAIRSHAPE_SPARSE_NAME_WHITE);
        assert_eq!(sparse_dims[1].1, PAIRSHAPE_DIM_PER_PERSP);
        assert_eq!(sparse_dims[1].2, PAIRSHAPE_MAX_ACTIVE_PER_PERSP);
        assert!(input.pairshape_enabled());
    }

    /// auxiliary_sparse_dims: dual + kingring + pairshape 三連結 (= arm 12 全部入り想定) で 6 entry。
    #[test]
    fn wrapper_dual_plus_kingring_plus_pairshape_sparse_dims() {
        let input = ShogiHalfKaHmKingAttackerDual::new()
            .with_dual_enabled(true)
            .with_king_attacker_enabled(false)
            .with_dual_cross_defended_enabled(false)
            .with_kingring_enabled(true)
            .with_pairshape_enabled(true);
        let sparse_dims = input.auxiliary_sparse_dims();
        // 2 cross threat + 2 kingring + 2 pairshape = 6 entry
        assert_eq!(sparse_dims.len(), 6);
        assert_eq!(sparse_dims[4].0, PAIRSHAPE_SPARSE_NAME_BLACK);
        assert_eq!(sparse_dims[5].0, PAIRSHAPE_SPARSE_NAME_WHITE);
    }

    #[test]
    fn scalar_threat_replaces_cross_but_keeps_king_attacker() {
        let input = ShogiHalfKaHmKingAttackerDual::new().with_dual_enabled(true).with_scalar_threat_enabled(true);
        let dims = input.auxiliary_sparse_dims();
        assert!(dims.iter().any(|entry| entry.0 == DUAL_SPARSE_NAME_ATTACK_BLACK));
        assert!(!dims.iter().any(|entry| entry.0 == DUAL_CROSS_THREAT_BLACK));
        assert!(dims.iter().any(|entry| entry.0 == crate::game::inputs::SCALAR_THREAT_SPARSE_NAME_BLACK));
        assert!(dims.iter().any(|entry| entry.1 == crate::game::inputs::SCALAR_THREAT_WEIGHT_COUNT));
    }

    #[test]
    fn scalar_threat_can_coexist_with_cross() {
        let input = ShogiHalfKaHmKingAttackerDual::new()
            .with_dual_enabled(true)
            .with_scalar_threat_enabled(true)
            .with_scalar_threat_coexist_dual_cross(true);
        let dims = input.auxiliary_sparse_dims();

        assert!(dims.iter().any(|entry| entry.0 == DUAL_CROSS_THREAT_BLACK));
        assert!(dims.iter().any(|entry| entry.0 == DUAL_CROSS_THREAT_WHITE));
        assert!(dims.iter().any(|entry| entry.0 == crate::game::inputs::SCALAR_THREAT_SPARSE_NAME_BLACK));
        assert!(dims.iter().any(|entry| entry.0 == crate::game::inputs::SCALAR_THREAT_SPARSE_NAME_WHITE));
    }
}
