//! HalfKA_hm + Threat 連結特徴量
//!
//! HalfKA_hm (73,305 次元) と Threat (216,720 次元) を連結した
//! sparse input 型を提供する。
//!
//! ## 仕様
//!
//! - HalfKA_hm: 既存の `ShogiHalfKA_hm` と同一ロジック
//! - Threat: rshogi `threat_features.rs` と同一 index 計算
//! - 仕様メモ: `docs/threat_spec.md` (rshogi リポジトリ)

use std::sync::LazyLock;

use super::SparseInputType;
use super::shogi_halfka::{
    HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, halfka_index, is_hm_mirror, king_bonapiece, king_bucket, pack_bonapiece,
};
use super::shogi_halfka_hm_king_attacker_dual::{
    CrossHead, DUAL_CROSS_DEFENDED_BLACK, DUAL_CROSS_DEFENDED_WHITE, DUAL_CROSS_DIM_PER_HEAD,
    DUAL_CROSS_MAX_ACTIVE_PER_HEAD, DUAL_CROSS_THREAT_BLACK, DUAL_CROSS_THREAT_WHITE, DUAL_DIM_PER_HEAD,
    DUAL_MAX_ACTIVE_PER_HEAD, DUAL_SPARSE_NAME_ATTACK_BLACK, DUAL_SPARSE_NAME_ATTACK_WHITE,
    DUAL_SPARSE_NAME_DEFEND_BLACK, DUAL_SPARSE_NAME_DEFEND_WHITE, DualHead, fill_dual_cross_head,
    fill_dual_sparse_head,
};
use super::shogi_halfka_hm_pin::{pin_aux_entries, try_fill_pin_aux};
use super::shogi_threat_exclusion::ThreatProfile;
use crate::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    bona_piece::BonaPiece,
    types::{BOARD_PIECE_TYPES, Color, HAND_PIECE_TYPES, Piece, PieceType, Square},
};

// =============================================================================
// Threat 定数
// =============================================================================

/// ThreatClass の数（King 除外）
pub(super) const NUM_THREAT_CLASSES: usize = 9;

/// active threat features の最大数（安全側の上限）
pub const MAX_ACTIVE_THREAT_FEATURES: usize = 320;

/// Threat 特徴を主 sparse 入力から**分離**して流すときの入力名 (手番側)。
///
/// 主入力に連結したままだと l0w が `(halfka_dim + threat_dims, ft_out)` の 1 枚になり、
/// threat 行だけ出力幅を絞れない (= VRAM も減らない)。 分離すると threat 重みを
/// `(K, threat_dims)` の独立テンソルにでき、 エンジン側 `P2_THREAT_OUT=K` と対応する。
pub const THREAT_FT_SPARSE_STM: &str = "threat_ft_stm";
/// [`THREAT_FT_SPARSE_STM`] の非手番側。
pub const THREAT_FT_SPARSE_NSTM: &str = "threat_ft_nstm";
/// 2026-11-17 K256 案 A (`--threat-ext-out N`): 追加列 `threat_ext_w` 用の threat 疎入力 (手番側)。
///
/// 中身は [`THREAT_FT_SPARSE_STM`] と同じ特徴だが、 **群の offset を足さない** (追加列は 1 群で全 bucket 共有)。
pub const THREAT_EXT_SPARSE_STM: &str = "threat_ext_stm";
/// [`THREAT_EXT_SPARSE_STM`] の非手番側。
pub const THREAT_EXT_SPARSE_NSTM: &str = "threat_ext_nstm";
/// [`THREAT_EXT_SPARSE_STM`] 用の勾配マスク (dense、 次元 = N)。 係数は [`THREAT_GMASK_NAME`] と同じ表から引く。
pub const THREAT_EXT_GMASK_NAME: &str = "threat_ext_gmask";
pub const SCALAR_THREAT_DIMENSIONS: usize = 216_720;
pub const SCALAR_THREAT_BUCKETS: usize = 8;
pub const SCALAR_THREAT_WEIGHT_COUNT: usize = SCALAR_THREAT_DIMENSIONS * SCALAR_THREAT_BUCKETS;
pub const SCALAR_THREAT_SPARSE_NAME_BLACK: &str = "scalar_threat_black";
pub const SCALAR_THREAT_SPARSE_NAME_WHITE: &str = "scalar_threat_white";

/// 最大アクティブ特徴数の合計（HalfKA_hm 40 + Threat 320）
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + MAX_ACTIVE_THREAT_FEATURES;
/// profile 12で各視点へ残すThreat特徴行数。C++側と同一。
const REPRESENTATIVE_THREAT_ROWS: usize = 8;
const REPRESENTATIVE_CLASS_VALUE: [u8; NUM_THREAT_CLASSES] = [1, 2, 2, 3, 4, 7, 8, 9, 10];
#[inline]
fn representative_priority(attacker: ThreatClass, target: ThreatClass) -> u8 {
    REPRESENTATIVE_CLASS_VALUE[target as usize] * 16 + REPRESENTATIVE_CLASS_VALUE[attacker as usize]
}
#[inline]
fn representative_better(left_prio: u8, left_idx: usize, right_prio: u8, right_idx: usize) -> bool {
    left_prio > right_prio || (left_prio == right_prio && left_idx < right_idx)
}
#[inline]
fn offer_representative(
    rows: &mut [(usize, u8); REPRESENTATIVE_THREAT_ROWS],
    count: &mut usize,
    index: usize,
    priority: u8,
) {
    if *count < REPRESENTATIVE_THREAT_ROWS {
        rows[*count] = (index, priority);
        *count += 1;
        return;
    }
    let mut worst = 0usize;
    for i in 1..REPRESENTATIVE_THREAT_ROWS {
        if representative_better(rows[worst].1, rows[worst].0, rows[i].1, rows[i].0) {
            worst = i;
        }
    }
    if representative_better(priority, index, rows[worst].1, rows[worst].0) {
        rows[worst] = (index, priority);
    }
}

// =============================================================================
// ThreatClass
// =============================================================================

/// Threat 駒種分類（King 除外、9 family）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum ThreatClass {
    Pawn = 0,
    Lance = 1,
    Knight = 2,
    Silver = 3,
    GoldLike = 4,
    Bishop = 5,
    Rook = 6,
    Horse = 7,
    Dragon = 8,
}

impl ThreatClass {
    /// PieceType から ThreatClass への変換。King は None。
    #[inline]
    pub(super) fn from_piece_type(pt: PieceType) -> Option<Self> {
        match pt {
            PieceType::Pawn => Some(Self::Pawn),
            PieceType::Lance => Some(Self::Lance),
            PieceType::Knight => Some(Self::Knight),
            PieceType::Silver => Some(Self::Silver),
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => Some(Self::GoldLike),
            PieceType::Bishop => Some(Self::Bishop),
            PieceType::Rook => Some(Self::Rook),
            PieceType::Horse => Some(Self::Horse),
            PieceType::Dragon => Some(Self::Dragon),
            PieceType::King | PieceType::None => None,
        }
    }
}

// =============================================================================
// 各クラスの空盤面利き数 (per color)
// =============================================================================

const ATTACKS_PER_COLOR: [usize; NUM_THREAT_CLASSES] = [
    72,   // Pawn
    324,  // Lance
    112,  // Knight
    328,  // Silver
    416,  // GoldLike
    816,  // Bishop
    1296, // Rook
    1104, // Horse
    1552, // Dragon
];

// =============================================================================
// pair_base テーブル
// =============================================================================

const NUM_PAIRS: usize = 2 * NUM_THREAT_CLASSES * 2 * NUM_THREAT_CLASSES; // 324

/// 除外された pair の sentinel 値
const EXCLUDED_PAIR_BASE: usize = usize::MAX;

/// 指定された profile で pair_base テーブルと THREAT_DIMENSIONS を構築 (runtime)
fn build_pair_base(profile: ThreatProfile) -> ([usize; NUM_PAIRS], usize) {
    let mut table = [0usize; NUM_PAIRS];
    let mut cumulative = 0usize;
    for attacker_side in 0..2 {
        for (ac, &attacks) in ATTACKS_PER_COLOR.iter().enumerate().take(NUM_THREAT_CLASSES) {
            for ds in 0..2 {
                for dc in 0..NUM_THREAT_CLASSES {
                    let idx = attacker_side * 162 + ac * 18 + ds * 9 + dc;
                    if profile.is_excluded(attacker_side, ac, ds, dc) {
                        table[idx] = EXCLUDED_PAIR_BASE;
                    } else {
                        table[idx] = cumulative;
                        cumulative += attacks;
                    }
                }
            }
        }
    }
    (table, cumulative)
}

/// 指定 profile の threat pair レイアウトを返す (重み移植ツール用)。
///
/// 返り値の各要素は `(pair_index, base, len)`:
/// - `pair_index` = `attacker_side * 162 + ac * 18 + attacked_side * 9 + dc`
/// - `base` = その pair の threat feature 開始 index
/// - `len` = その pair が占める feature 数 (= 空盤面利き数)
///
/// 除外された pair は含まれない。要素は `base` 昇順。
pub fn threat_pair_layout(profile: ThreatProfile) -> Vec<(usize, usize, usize)> {
    let (table, _) = build_pair_base(profile);
    let mut out = Vec::new();
    for (idx, &base) in table.iter().enumerate() {
        if base != EXCLUDED_PAIR_BASE {
            let ac = (idx % 162) / 18;
            out.push((idx, base, ATTACKS_PER_COLOR[ac]));
        }
    }
    out.sort_by_key(|&(_, base, _)| base);
    out
}

/// pair_base テーブルから値を取得。除外された pair は None を返す。
#[inline]
fn lookup_pair_base(
    pair_base: &[usize; NUM_PAIRS],
    attacker_side: usize,
    ac: ThreatClass,
    attacked_side: usize,
    dc: ThreatClass,
) -> Option<usize> {
    let idx = attacker_side * 162 + (ac as usize) * 18 + attacked_side * 9 + dc as usize;
    let base = pair_base[idx];
    if base == EXCLUDED_PAIR_BASE { None } else { Some(base) }
}

// =============================================================================
// Attack pattern / from_offset / attack_order（色別 LUT）
// =============================================================================

/// Attack pattern の総数: 9(Black) + 5(White の方向性駒)
pub(super) const NUM_ATTACK_PATTERNS: usize = 14;

/// 方向性駒かどうか
#[inline]
pub(super) fn is_directional(class: ThreatClass) -> bool {
    matches!(
        class,
        ThreatClass::Pawn | ThreatClass::Lance | ThreatClass::Knight | ThreatClass::Silver | ThreatClass::GoldLike
    )
}

/// attack_pattern_id: 方向性駒は色別、非方向性駒は色不問
#[inline]
pub(super) fn attack_pattern_id(class: ThreatClass, oriented_color: Color) -> usize {
    if oriented_color == Color::White && is_directional(class) {
        NUM_THREAT_CLASSES + class as usize // 9..13
    } else {
        class as usize // 0..8
    }
}

// =============================================================================
// 空盤面利き計算（Bitboard を使わず、座標ベースで列挙）
// =============================================================================

/// 空盤面上の攻撃先マスを raw 値昇順で返す。
/// 返り値: (マスの配列, マス数)
pub(super) fn attacks_empty_board(class: ThreatClass, color: Color, from: Square) -> ([u8; 36], usize) {
    let mut targets = [0u8; 36];
    let mut count = 0;
    let file = from.file() as i8;
    let rank = from.rank() as i8;

    match class {
        ThreatClass::Pawn => {
            // 先手: 1マス前 (rank-1)、後手: 1マス後 (rank+1)
            let dr: i8 = if color == Color::Black { -1 } else { 1 };
            let r = rank + dr;
            if (0..9).contains(&r) {
                targets[count] = Square::new(file as u8, r as u8).0;
                count += 1;
            }
        }
        ThreatClass::Lance => {
            // 先手: rank-1, rank-2, ... 0 まで。後手: rank+1, ... 8 まで。
            // 空盤面なので遮蔽なし。
            if color == Color::Black {
                for r in (0..rank).rev() {
                    targets[count] = Square::new(file as u8, r as u8).0;
                    count += 1;
                }
            } else {
                for r in (rank + 1)..9 {
                    targets[count] = Square::new(file as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Knight => {
            // 先手: (file-1, rank-2), (file+1, rank-2)
            // 後手: (file-1, rank+2), (file+1, rank+2)
            let dr: i8 = if color == Color::Black { -2 } else { 2 };
            for df in [-1i8, 1] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Silver => {
            // 先手: (-1,-1),(0,-1),(1,-1),(-1,1),(1,1)
            // 後手: (-1,1),(0,1),(1,1),(-1,-1),(1,-1)
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 5] = [(-1, forward), (0, forward), (1, forward), (-1, -forward), (1, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::GoldLike => {
            // 先手: (-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(0,1)
            // 後手: (-1,1),(0,1),(1,1),(-1,0),(1,0),(0,-1)
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 6] = [(-1, forward), (0, forward), (1, forward), (-1, 0), (1, 0), (0, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Bishop => {
            // 4方向斜め、空盤面なので盤端まで
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
        }
        ThreatClass::Rook => {
            // 4方向直線、空盤面なので盤端まで
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
        }
        ThreatClass::Horse => {
            // 角 + 上下左右1マス (King moves)
            // 斜め（空盤面）
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
            // 上下左右1マス
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
        ThreatClass::Dragon => {
            // 飛 + 斜め1マス
            // 直線（空盤面）
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                    f += df;
                    r += dr;
                }
            }
            // 斜め1マス
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    targets[count] = Square::new(f as u8, r as u8).0;
                    count += 1;
                }
            }
        }
    }

    // raw 値昇順でソート（挿入ソート: count は最大 ~20 程度）
    for i in 1..count {
        let key = targets[i];
        let mut j = i;
        while j > 0 && targets[j - 1] > key {
            targets[j] = targets[j - 1];
            j -= 1;
        }
        targets[j] = key;
    }

    (targets, count)
}

// =============================================================================
// from_offset テーブル
// =============================================================================

/// 全 attack pattern の from_offset テーブル
pub(super) struct FromOffsetTable {
    data: [[usize; 81]; NUM_ATTACK_PATTERNS],
}

impl FromOffsetTable {
    pub(super) fn new() -> Self {
        let all_classes: [ThreatClass; NUM_THREAT_CLASSES] = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];

        let mut data = [[0usize; 81]; NUM_ATTACK_PATTERNS];

        for &class in &all_classes {
            // Black (先手) の from_offset
            {
                let pattern = class as usize;
                let mut cumulative = 0usize;
                for sq_raw in 0..81u8 {
                    data[pattern][sq_raw as usize] = cumulative;
                    let (_, cnt) = attacks_empty_board(class, Color::Black, Square(sq_raw));
                    cumulative += cnt;
                }
            }
            // White (後手) の方向性駒は別エントリ
            if is_directional(class) {
                let pattern = NUM_THREAT_CLASSES + class as usize;
                let mut cumulative = 0usize;
                for sq_raw in 0..81u8 {
                    data[pattern][sq_raw as usize] = cumulative;
                    let (_, cnt) = attacks_empty_board(class, Color::White, Square(sq_raw));
                    cumulative += cnt;
                }
            }
        }

        Self { data }
    }

    #[inline]
    pub(super) fn get(&self, pattern: usize, sq_n: Square) -> usize {
        self.data[pattern][sq_n.index()]
    }
}

/// 遅延初期化された `FromOffsetTable` シングルトン
pub(super) static FROM_OFFSET_TABLE: LazyLock<FromOffsetTable> = LazyLock::new(FromOffsetTable::new);

// =============================================================================
// attack_order LUT (O(1) lookup)
// =============================================================================

/// attack_order の事前計算 LUT (rshogi の ATTACK_ORDER_TABLE と同設計)
///
/// `data[attack_pattern][from_sq][to_sq]` = 空盤面上で from_sq の駒が to_sq を攻撃する際の
/// 0-indexed 順位 (raw 昇順)。攻撃しない (from, to) ペアは `INVALID` (u8::MAX)。
///
/// サイズ: 14 × 81 × 81 = 91,854 bytes ≈ 90 KiB
///
/// LazyLock で初回アクセス時に 1 度だけ構築。以降は `get()` が O(1) lookup。
pub(super) struct AttackOrderTable {
    data: [[[u8; 81]; 81]; NUM_ATTACK_PATTERNS],
}

impl AttackOrderTable {
    pub(super) const INVALID: u8 = u8::MAX;

    pub(super) fn new() -> Self {
        let all_classes: [ThreatClass; NUM_THREAT_CLASSES] = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];

        let mut data = [[[Self::INVALID; 81]; 81]; NUM_ATTACK_PATTERNS];
        for &class in &all_classes {
            // Black (先手) pattern
            {
                let pattern = class as usize;
                Self::fill_pattern(&mut data[pattern], class, Color::Black);
            }
            // White (後手) 方向性駒は別エントリ
            if is_directional(class) {
                let pattern = NUM_THREAT_CLASSES + class as usize;
                Self::fill_pattern(&mut data[pattern], class, Color::White);
            }
        }
        Self { data }
    }

    fn fill_pattern(table: &mut [[u8; 81]; 81], class: ThreatClass, color: Color) {
        for from_raw in 0..81u8 {
            let (targets, count) = attacks_empty_board(class, color, Square(from_raw));
            for (order, &to_raw) in targets.iter().take(count).enumerate() {
                table[from_raw as usize][to_raw as usize] = order as u8;
            }
        }
    }

    /// O(1) で attack_order を取得
    #[inline]
    pub(super) fn get(&self, pattern: usize, from_sq: Square, to_sq: Square) -> u8 {
        self.data[pattern][from_sq.0 as usize][to_sq.0 as usize]
    }
}

/// 遅延初期化された `AttackOrderTable` シングルトン
pub(super) static ATTACK_ORDER_TABLE: LazyLock<AttackOrderTable> = LazyLock::new(AttackOrderTable::new);

// =============================================================================
// attack_order 計算 (legacy、テスト互換用)
// =============================================================================

/// 空盤面上で from_sq の駒が to_sq を攻撃するときの、raw 昇順での順位
///
/// **注意**: ホットパスでは `ATTACK_ORDER_TABLE.get(...)` を直接使うこと。
/// この関数は毎回 `attacks_empty_board` を再計算するため遅い。
/// 既存のテストコードと `threat_index` 内の legacy path との互換維持のためだけに残している。
#[allow(dead_code)]
pub(super) fn compute_attack_order(class: ThreatClass, color: Color, from_sq: Square, to_sq: Square) -> usize {
    let (targets, count) = attacks_empty_board(class, color, from_sq);
    let to_raw = to_sq.0;
    for (i, &target) in targets.iter().enumerate().take(count) {
        if target == to_raw {
            return i;
        }
    }
    panic!("attack_order: to_sq {} is not attacked by {:?} ({:?}) at {}", to_sq.0, class, color, from_sq.0);
}

// =============================================================================
// 実盤面上の利き計算
// =============================================================================

/// occupied bitset: 81 マス分のビットマップ
pub(super) struct Occupied {
    bits: [u64; 2], // bits[0]: sq 0..63, bits[1]: sq 64..80
}

impl Occupied {
    pub(super) fn from_board(board: &ShogiBoard) -> Self {
        let mut bits = [0u64; 2];
        for sq in 0..81u8 {
            if board.board[sq as usize].is_some() {
                if sq < 64 {
                    bits[0] |= 1u64 << sq;
                } else {
                    bits[1] |= 1u64 << (sq - 64);
                }
            }
        }
        Self { bits }
    }

    #[inline]
    pub(super) fn is_occupied(&self, sq: u8) -> bool {
        if sq < 64 { (self.bits[0] >> sq) & 1 != 0 } else { (self.bits[1] >> (sq - 64)) & 1 != 0 }
    }
}

/// 両玉の 8 近傍のうち**駒が乗っていない升**への、 その玉の敵の利き数の和 (= V1)。
///
/// C++ 推論側の `P2Volm::value` (source/p2volm.h) と**完全に同じ値**を返すこと。
/// 案 B (Regime LayerStack) の bucket の鍵なので、 1 でも違うと net が黙って壊れる。
///
/// 利きの数え方は **step + ray** (長い利きを含む、 ブロックあり)。
/// 2026-10-07 に engine 400 局面と突き合わせて確定: step+ray が 400/400 一致、
/// step のみは 283/400 (70.8%) しか一致しなかった。
///
/// 🚨`for_each_attack` は `PieceType::King => {}` で**玉を列挙しない**ので、
///   玉の 8 近傍をここで手で足す (C++ の `board_effect` は kingEffect 込み)。
pub(crate) fn board_effect_v1(board: &ShogiBoard) -> i32 {
    // eff[color][sq] = color の駒が sq に利かせている数
    let mut eff = [[0u8; 81]; 2];
    let occ = Occupied::from_board(board);
    for sq_raw in 0..81u8 {
        let from_sq = Square(sq_raw);
        let pc = board.piece_on(from_sq);
        if pc.is_none() {
            continue;
        }
        let ci = pc.color as usize;
        if pc.piece_type == PieceType::King {
            // 🚨玉は for_each_attack が列挙しないので手で足す (8 近傍)
            let f = from_sq.file() as i8;
            let r = from_sq.rank() as i8;
            for df in -1i8..=1 {
                for dr in -1i8..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let (tf, tr) = (f + df, r + dr);
                    if (0..9).contains(&tf) && (0..9).contains(&tr) {
                        let t = Square::new(tf as u8, tr as u8);
                        eff[ci][t.0 as usize] = eff[ci][t.0 as usize].saturating_add(1);
                    }
                }
            }
            continue;
        }
        for_each_attack(pc.piece_type, pc.color, from_sq, &occ, |to| {
            eff[ci][to.0 as usize] = eff[ci][to.0 as usize].saturating_add(1);
        });
    }
    // 両玉の 8 近傍の**空升**について、 その玉の敵の利き数を足す
    let mut v: i32 = 0;
    for kc in [Color::Black, Color::White] {
        let ksq = board.king_square(kc);
        if ksq == Square::NONE {
            continue;
        }
        let en = 1 - (kc as usize);
        let f = ksq.file() as i8;
        let r = ksq.rank() as i8;
        for df in -1i8..=1 {
            for dr in -1i8..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let (tf, tr) = (f + df, r + dr);
                if !((0..9).contains(&tf) && (0..9).contains(&tr)) {
                    continue;
                }
                let t = Square::new(tf as u8, tr as u8);
                if !board.piece_on(t).is_none() {
                    continue; // 空升のみ
                }
                v += eff[en][t.0 as usize] as i32;
            }
        }
    }
    v
}

/// 実盤面上の攻撃先マスを列挙し、コールバックを呼ぶ
pub(super) fn for_each_attack<F: FnMut(Square)>(
    pt: PieceType,
    color: Color,
    from: Square,
    occ: &Occupied,
    mut callback: F,
) {
    let file = from.file() as i8;
    let rank = from.rank() as i8;

    match pt {
        PieceType::Pawn => {
            let dr: i8 = if color == Color::Black { -1 } else { 1 };
            let r = rank + dr;
            if (0..9).contains(&r) {
                callback(Square::new(file as u8, r as u8));
            }
        }
        PieceType::Lance => {
            if color == Color::Black {
                for r in (0..rank).rev() {
                    let sq = Square::new(file as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                }
            } else {
                for r in (rank + 1)..9 {
                    let sq = Square::new(file as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                }
            }
        }
        PieceType::Knight => {
            let dr: i8 = if color == Color::Black { -2 } else { 2 };
            for df in [-1i8, 1] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Silver => {
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 5] = [(-1, forward), (0, forward), (1, forward), (-1, -forward), (1, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Gold | PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            let forward: i8 = if color == Color::Black { -1 } else { 1 };
            let deltas: [(i8, i8); 6] = [(-1, forward), (0, forward), (1, forward), (-1, 0), (1, 0), (0, -forward)];
            for (df, dr) in deltas {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Bishop => {
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
        }
        PieceType::Rook => {
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
        }
        PieceType::Horse => {
            // 角行き（スライダー）
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
            // 上下左右1マス
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        PieceType::Dragon => {
            // 飛行き（スライダー）
            for (df, dr) in [(-1i8, 0i8), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    let sq = Square::new(f as u8, r as u8);
                    callback(sq);
                    if occ.is_occupied(sq.0) {
                        break;
                    }
                    f += df;
                    r += dr;
                }
            }
            // 斜め1マス
            for (df, dr) in [(-1i8, -1i8), (-1, 1), (1, -1), (1, 1)] {
                let f = file + df;
                let r = rank + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    callback(Square::new(f as u8, r as u8));
                }
            }
        }
        // King と None は threat に含まない
        PieceType::King | PieceType::None => {}
    }
}

// =============================================================================
// マス正規化
// =============================================================================

/// マスを perspective 基準 + HM mirror で正規化
#[inline]
pub(super) fn normalize_sq(sq: Square, perspective: Color, hm_mirror: bool) -> Square {
    let sq_n = if perspective == Color::Black { sq } else { sq.inverse() };
    if hm_mirror { sq_n.mirror_file() } else { sq_n }
}

// =============================================================================
// Threat index 計算
// =============================================================================

/// Threat index 計算用のパラメータ
struct ThreatParams {
    attacker_side: usize,
    attacker_class: ThreatClass,
    oriented_color: Color,
    attacked_side: usize,
    attacked_class: ThreatClass,
    from_sq_n: Square,
    to_sq_n: Square,
}

/// Threat index を計算する。除外された pair は None を返す。
///
/// `attack_order` は O(1) LUT lookup (`ATTACK_ORDER_TABLE`) を使用する。
#[inline]
fn threat_index(
    params: &ThreatParams,
    pair_base_table: &[usize; NUM_PAIRS],
    from_offset_table: &FromOffsetTable,
) -> Option<usize> {
    let base = lookup_pair_base(
        pair_base_table,
        params.attacker_side,
        params.attacker_class,
        params.attacked_side,
        params.attacked_class,
    )?;
    let pattern = attack_pattern_id(params.attacker_class, params.oriented_color);
    let from_off = from_offset_table.get(pattern, params.from_sq_n);
    let attack_ord = ATTACK_ORDER_TABLE.get(pattern, params.from_sq_n, params.to_sq_n);
    debug_assert_ne!(
        attack_ord,
        AttackOrderTable::INVALID,
        "attack_order: to_sq {} not attacked by pattern {pattern} at {}",
        params.to_sq_n.0,
        params.from_sq_n.0
    );
    Some(base + from_off + attack_ord as usize)
}

// =============================================================================
// ShogiHalfKaHmThreat
// =============================================================================

/// threat 特徴 f → pair 番号 の表 (threat factorizer 用)。 各 pair は `[base, base + ATTACKS_PER_COLOR[ac])` を占める。
fn build_feature_to_pair(pair_base: &[usize; NUM_PAIRS], threat_dims: usize) -> Box<[u16]> {
    const UNSET: u16 = u16::MAX;
    let mut table = vec![UNSET; threat_dims];
    for (idx, &base) in pair_base.iter().enumerate() {
        if base == EXCLUDED_PAIR_BASE {
            continue;
        }
        let ac = (idx / 18) % NUM_THREAT_CLASSES;
        for f in base..base + ATTACKS_PER_COLOR[ac] {
            assert_eq!(table[f], UNSET, "threat factor: 特徴 {f} が 2 つの pair に属する");
            table[f] = idx as u16;
        }
    }
    assert!(table.iter().all(|&p| p != UNSET), "threat factor: pair に属さない特徴がある");
    table.into_boxed_slice()
}

/// HalfKA_hm + Threat 連結特徴量
///
/// `SparseInputType` を実装し、HalfKA_hm 特徴量と Threat 特徴量を
/// 連結した sparse input として提供する。
///
/// `ThreatProfile` で除外 pair を runtime 選択する。
#[allow(non_camel_case_types)]
#[derive(Clone, Debug)]
pub struct ShogiHalfKaHmThreat {
    profile: ThreatProfile,
    pair_base: Box<[usize; NUM_PAIRS]>,
    threat_dims: usize,
    /// Pin-aware 補助入力を有効化するか。
    ///
    /// `true` のとき `auxiliary_dense_dims` に `pin_features_black/white` (各 2,268 dim)
    /// を追加し、 fill 時に `pin_features::fill_pin_features_for_side` で dense vector
    /// を生成する。 Threat + Pin combined input をサポート。
    pin_aware: bool,
    /// Dual PSQT king-attacker用のauxiliary sparse入力を供給するか。
    dual_psqt_enabled: bool,
    /// king-attackerのattack/defend 4 headを供給するか。
    dual_king_attacker_enabled: bool,
    /// D2 crossのdefended 2 headを供給するか。
    dual_cross_defended_enabled: bool,
    /// Threat 特徴を主 sparse 入力に連結せず、 専用 sparse 入力
    /// ([`THREAT_FT_SPARSE_STM`] / [`THREAT_FT_SPARSE_NSTM`]) で流すか。
    threat_ft_sparse: bool,
    /// 2026-09-12: threat 表を LayerStack bucket の「群」ごとに独立させる (`--threat-bucket-groups`)。
    ///
    /// `Some(map)` のとき `map[bucket]` = 群番号 g とし、 専用 sparse 入力の index を
    /// `f + g * threat_dims` にずらす (= 群ごとに別の列を引く)。 bucket は
    /// [`ShogiProgressKPAbs`] (progress8kpabs / progress4hand2kpabs の出力 bucket と同一) で決める。
    /// `None` (既定) では index は従来と完全に同じ。 `threat_ft_sparse` が on のときだけ意味を持つ。
    threat_bucket_groups: Option<Box<[u8]>>,
    /// 2026-09-16: 学習時だけの threat factorizer (`--threat-factorizer`、 runs/threat_factor/design.md)。
    ///
    /// `Some(table)` のとき `table[f]` = 特徴 f の pair 番号 (as*162 + ac*18 + ds*9 + dc、 0..324)。
    /// 専用 sparse 入力は実特徴 `f + g * threat_dims` の直後に factor 特徴 `threat_dims * 群数 + pair` を**対で**出す
    /// (factor 行は全群で共有)。 export で各実特徴の行へ畳み込むのでエンジンは無変更。
    threat_factor: Option<Box<[u16]>>,
    /// 2026-09-16: threat 勾配の局面別マスク (`--threat-grad-mask`)。
    ///
    /// `Some((table, k))` のとき dense aux 入力 [`THREAT_GMASK_NAME`] (次元 k = --threat-out K) に
    /// 係数 w(|教師値|) を k 個並べて流す。 graph 側は forward を変えずに勾配だけ w 倍する
    /// (`thr_d + (thr - thr_d) * m`、 `thr_d` は stop-grad)。 `table` は (閾値 cp, 係数) の昇順で、
    /// 最後の要素が閾値超えに適用される。
    threat_gmask: Option<(Box<[(i32, f32)]>, usize)>,
    /// 2026-09-24: profile 16 (hanging-split)。 `Some(table)` のとき
    /// `table[f]` = base feature f に対応する delta 行の局所 index (同色 pair は -1)。
    /// 被攻撃駒にひもが無い pair だけ、 実特徴 `f + g * threat_dims` と対で
    /// delta 特徴 `threat_dims * 群数 + table[f]` を出す (delta 行は全群で共有)。
    /// export で各 hang 行へ `base[g] + delta` を畳み込む。
    threat_hang_of: Option<Box<[i32]>>,
    /// 2026-09-29: B6 可動域 (`--mob`)。
    ///
    /// `true` のとき dense aux 入力 [`MOB_NAME`] (1 次元) に
    /// `mob_diff(board) / MOB_INPUT_SCALE` を流す。 反対称スカラー 1 本なので
    /// graph 側は PSQT と同じ「dense 入力 x bucket 別 affine」で受ける。
    ///
    /// 🚨定義は `shogi_mob.rs`。 Python 正典 / C++ エンジン (`p2mob`) と
    ///   4,010 局面でゼロ許容一致を確認済み (2026-09-29)。
    mob: bool,
    /// 2026-11-07 (T2): **持駒 14 元 dense を L1 入力に concat するか**。
    ///
    /// 🚨本番の入力型は**この型**。 `inputs.rs:29-30` が `#[path]` でこのファイルを
    ///   `shogi_halfka_hm_threat` という名前の module にしているので、 同名の
    ///   `shogi_halfka_hm_threat.rs` は**コンパイルされない死んだファイル**。
    ///   ここが `hand_count_dims` を返さないと `--hand-count-dense` が黙って無効になる。
    ///   中身は `ShogiHalfKaHmHandCount` に委譲する (目盛り k*c/127 はそちらで 1 箇所だけ定義)。
    hand_count_dense: bool,
    /// 2026-11-12: KH ブロック (`--kh`、 `shogi_kh.rs`)。 `Some((parts, null))` のとき
    /// 専用 sparse 入力 [`super::shogi_kh::KH_SPARSE_STM`] / `KH_SPARSE_NSTM` を供給する。
    /// `None` (既定) では aux 入力の並びも中身も従来と同一。
    kh: Option<(super::shogi_kh::KhParts, bool)>,
    /// 2026-11-17 K256 案 A: 追加列の幅 N (`--threat-ext-out`)。 0 (既定) なら aux 入力の並びも中身も従来と同一。
    ///
    /// N > 0 のとき [`THREAT_EXT_SPARSE_STM`] / [`THREAT_EXT_SPARSE_NSTM`] (次元 = threat_dims、 群 offset なし) を供給し、
    /// 勾配マスクが有効なら [`THREAT_EXT_GMASK_NAME`] (次元 N) も供給する。
    threat_ext: usize,
}

/// B6 可動域の dense aux 入力名。
pub const MOB_NAME: &str = "mob";

/// 可動域を dense 入力へ渡すときの目盛り。 `mob_diff` は ±67 まで振れるので割る。
/// 🚨export で 1/`MOB_INPUT_SCALE` を重みへ畳み込むので、 エンジン側は素の `mob_diff` のまま。
pub const MOB_INPUT_SCALE: f32 = 16.0;

/// P2MOB セクションの magic。
// ---- 案1 ブロック射影 (2026-09-30、 fable 設計) ----
//
// threat accumulator (K=64) は今 FT 1024 次元のうち **64 次元** (前半 32 + 後半 32) に
// しか着地していない。 pairwise_mul の積 512 個のうち threat が入るのは 32 個だけ。
// 射影 p[j] (j = 0..FT) を学習し、 全次元へ `ft[j] += p[j] * thr[j % K]` で足す。
// ランク (threat の 64 方向) は増えないので **threat 表も列挙コストも不変** = NPS 税は据え置き。

/// quantised.bin 末尾の射影 section magic。
pub const THREAT_PROJ_SUB_MAGIC: &[u8; 8] = b"P2TPRJ\0\0";
/// section の版。
pub const THREAT_PROJ_SECTION_VERSION: u32 = 1;
/// arch marker。 C++ loader はこれを見て末尾 section を読む。
pub const THREAT_PROJ_ARCH_MARKER: &str = "ThreatProj";
/// 射影係数の量子化目盛り。 Q15 = `_mm256_mulhrs_epi16` がそのまま使える形。
///
/// 🚨学習器側は `clip_pass_through_grad(-1,1)` で ±1 に収めているので、
///   書き出しでも同じ clamp をかけてから量子化する (かけ忘れると engine と食い違う)。
pub const THREAT_PROJ_QA: f64 = 32768.0;

pub const MOB_SUB_MAGIC: &[u8; 8] = b"P2MOB\0\0\0";

/// P2MOB セクションの版。
pub const MOB_SECTION_VERSION: u32 = 1;

/// arch_str の marker。 C++ `load_eval` がこれを見て P2MOB を読む。
pub const MOB_ARCH_MARKER: &str = "MOB=1";

/// 重みの量子化目盛り。 **PSQT head と同じ QA*QB = 8128**
/// (mob head は psqt_diff と同じ `net_output_final` に足されるため)。
pub const MOB_QA: f64 = 8128.0;

/// エンジン側のシフト量。 `MOB_INPUT_SCALE = 16 = 1 << 4` と対応する。
pub const MOB_INPUT_SHIFT: u32 = 4;

/// 重みの本数 (= LayerStack bucket 数)。
pub const MOB_WEIGHT_COUNT: usize = 8;

/// threat 勾配マスクの dense aux 入力名。
pub const THREAT_GMASK_NAME: &str = "threat_gmask";

/// threat factor の行数 (= pair 数 324)。
pub const THREAT_FACTOR_ROWS: usize = NUM_PAIRS;

/// profile 16 (hanging-split) の delta 行数 = 敵味方をまたぐ pair の feature 数。
/// = 2 (cross の (as,ds) 組) x 9 (dc) x 6020 (Σ ATTACKS_PER_COLOR) = 108,360。
/// C++ 側 `Threat::THREAT_HANG_DIM` と一致すること。
pub const THREAT_HANG_ROWS: usize = 108_360;

/// hang 局所 index -> base feature index の対応表 (export で `base[g] + delta` を畳み込むのに使う)。
/// `build_hang_of` の逆写像。 長さ = THREAT_HANG_ROWS。
pub fn threat_hang_base_map(profile: ThreatProfile) -> Vec<u32> {
    let (hang_of, rows) = build_hang_of(profile);
    let mut out = vec![u32::MAX; rows];
    for (base, &h) in hang_of.iter().enumerate() {
        if h >= 0 {
            out[h as usize] = base as u32;
        }
    }
    debug_assert!(out.iter().all(|&v| v != u32::MAX), "hang base map has holes");
    out
}

/// base feature index -> hang 局所 index の対応表を作る (同色 pair は -1)。
/// エンジンの `pair_base2[1]` と同じ入れ子順で cross pair だけを詰める。
fn build_hang_of(profile: ThreatProfile) -> (Box<[i32]>, usize) {
    let (pair_base, dims) = build_pair_base(profile);
    let mut hang_of = vec![-1i32; dims].into_boxed_slice();
    let mut hang_cumulative = 0usize;
    for attacker_side in 0..2 {
        for (ac, &attacks) in ATTACKS_PER_COLOR.iter().enumerate().take(NUM_THREAT_CLASSES) {
            for ds in 0..2 {
                for dc in 0..NUM_THREAT_CLASSES {
                    let idx = attacker_side * 162 + ac * 18 + ds * 9 + dc;
                    if profile.is_excluded(attacker_side, ac, ds, dc) || attacker_side == ds {
                        continue; // 同色 pair は攻め駒自身がひも = h=0 に縮退
                    }
                    let base = pair_base[idx];
                    for k in 0..attacks {
                        hang_of[base + k] = (hang_cumulative + k) as i32;
                    }
                    hang_cumulative += attacks;
                }
            }
        }
    }
    (hang_of, hang_cumulative)
}

impl ShogiHalfKaHmThreat {
    /// 指定 profile で構築 (pin_aware off)。
    pub fn new(profile: ThreatProfile) -> Self {
        let (pair_base, threat_dims) = build_pair_base(profile);
        Self {
            profile,
            pair_base: Box::new(pair_base),
            threat_dims,
            pin_aware: false,
            mob: false,
            dual_psqt_enabled: false,
            dual_king_attacker_enabled: false,
            dual_cross_defended_enabled: false,
            threat_ft_sparse: false,
            threat_bucket_groups: None,
            threat_factor: None,
            threat_gmask: None,
            threat_hang_of: if profile.is_hang_split() {
                let (table, rows) = build_hang_of(profile);
                assert_eq!(rows, THREAT_HANG_ROWS, "hang rows mismatch");
                Some(table)
            } else {
                None
            },
            hand_count_dense: false,
            kh: None,
            threat_ext: 0,
        }
    }

    /// 2026-11-17 K256 案 A: 追加列 (幅 `n`、 0 = 無効 = 従来と同一) 用の疎入力を有効にする builder。
    ///
    /// 追加列は群なし・factor / hanging-split なしの素の threat 特徴だけを受けるので、
    /// それらと同時に指定されたら panic する (学習器の引数検査で先に止めている)。
    pub fn with_threat_ext(mut self, n: usize) -> Self {
        assert!(
            n == 0 || (self.threat_ft_sparse && self.threat_factor.is_none() && self.threat_hang_of.is_none()),
            "threat_ext は threat_ft_sparse と併用し、 factorizer / hanging-split とは併用しないこと"
        );
        self.threat_ext = n;
        self
    }

    /// 2026-11-12: KH ブロックを有効にする builder (`None` = 無効 = 従来と同一)。
    /// `null = true` は特徴シャッフルの対照 (直前の別局面の特徴を流す)。
    pub fn with_kh(mut self, kh: Option<(super::shogi_kh::KhParts, bool)>) -> Self {
        self.kh = kh;
        self
    }

    /// threat 勾配マスクを設定する (`table` = (閾値 |教師値| cp, 係数) 昇順、 `k` = --threat-out K)。
    pub fn with_threat_grad_mask(mut self, table: Option<Vec<(i32, f32)>>, k: usize) -> Self {
        self.threat_gmask = table.map(|t| (t.into_boxed_slice(), k));
        self
    }

    /// 教師値 `score` に対する threat 勾配の係数 (マスク無効なら 1.0)。
    pub fn threat_grad_weight(&self, score: i16) -> f32 {
        match &self.threat_gmask {
            None => 1.0,
            Some((table, _)) => {
                let a = (score as i32).abs();
                for &(th, w) in table.iter() {
                    if a <= th {
                        return w;
                    }
                }
                table.last().map_or(1.0, |&(_, w)| w)
            }
        }
    }

    /// threat factorizer を切り替える (`threat_ft_sparse` と併用すること)。
    ///
    /// 特徴 → pair の表を pair_base から作り、 全特徴がちょうど 1 つの pair に属することを assert する。
    pub fn with_threat_factor(mut self, enabled: bool) -> Self {
        self.threat_factor =
            if enabled { Some(build_feature_to_pair(&self.pair_base, self.threat_dims)) } else { None };
        self
    }

    /// threat factorizer が有効か。
    pub fn threat_factor_enabled(&self) -> bool {
        self.threat_factor.is_some()
    }

    /// 特徴 f の pair 番号 (factorizer 無効なら None)。
    pub fn threat_feature_pair(&self, f: usize) -> Option<usize> {
        self.threat_factor.as_ref().map(|t| t[f] as usize)
    }

    /// threat 表の bucket 群化を設定する (`None` = 従来どおり 1 表)。
    ///
    /// `groups[bucket]` が群番号。 群数は `max + 1`。 専用 sparse 入力の次元は
    /// `threat_dims * 群数` になり、 index は `f + 群 * threat_dims` にずれる。
    /// `threat_ft_sparse` と併用すること (連結入力には効かない)。
    pub fn with_threat_bucket_groups(mut self, groups: Option<Vec<u8>>) -> Self {
        self.threat_bucket_groups = groups.map(Vec::into_boxed_slice);
        self
    }

    /// threat 表の群数 (群化していなければ 1)。
    pub fn threat_group_count(&self) -> usize {
        self.threat_bucket_groups.as_ref().map_or(1, |g| g.iter().copied().max().unwrap_or(0) as usize + 1)
    }

    /// Threat を専用 sparse 入力で流すモードを切り替える。
    ///
    /// on のとき主 sparse 入力は HalfKA_hm (73,305) だけになり、 Threat は
    /// 視点ごとの専用入力になる。 グラフ側は `(K, threat_dims)` の重みを持ち、
    /// その出力を FT の先頭 K 行へ加算する。
    pub fn with_threat_ft_sparse(mut self, enabled: bool) -> Self {
        self.threat_ft_sparse = enabled;
        self
    }

    /// Threat 専用 sparse 入力モードが有効か。
    pub fn threat_ft_sparse(&self) -> bool {
        self.threat_ft_sparse
    }

    /// pin_aware on で構築する builder。
    /// B6 可動域 (2026-09-29)。 有効にすると dense aux 入力 [`MOB_NAME`] (1 次元) を供給する。
    pub fn with_mob(mut self, enabled: bool) -> Self {
        self.mob = enabled;
        self
    }

    /// 2026-11-07 (T2): 持駒 14 元 dense を有効にする builder。
    ///
    /// `false` のとき `hand_count_dims()` は `None` を返し、 graph / arch / 保存形式は
    /// **1 バイトも変わらない** (検算 (a0) で md5 一致を確かめる)。
    pub fn with_hand_count_dense(mut self, enabled: bool) -> Self {
        self.hand_count_dense = enabled;
        self
    }

    pub fn with_pin_aware(mut self, enabled: bool) -> Self {
        self.pin_aware = enabled;
        self
    }

    /// pin_aware の有効状態を返す。
    pub fn pin_aware(&self) -> bool {
        self.pin_aware
    }

    /// Dual PSQT king-attackerのauxiliary sparse入力を有効化する。
    ///
    /// Threat本体のFT sparse入力に加え、グラフが要求するattack/defendとD2 crossの
    /// sparse入力を同じPackedSfenValueから供給する。これにより、未充填sparse入力の
    /// batch=Noneと主出力batch=Some(N)が加算されることを防ぐ。
    pub fn with_dual_psqt(mut self, enabled: bool, king_attacker_enabled: bool, cross_defended_enabled: bool) -> Self {
        self.dual_psqt_enabled = enabled;
        self.dual_king_attacker_enabled = enabled && king_attacker_enabled;
        self.dual_cross_defended_enabled = enabled && cross_defended_enabled;
        self
    }

    /// Threat 特徴量の次元数
    pub fn threat_dimensions(&self) -> usize {
        self.threat_dims
    }

    /// ThreatProfile を取得
    pub fn profile(&self) -> ThreatProfile {
        self.profile
    }
}

/// 局面の「戦術的な当たり」を数える (net 非依存、 threat 特化データ作成用)。
///
/// threat の**ペア数**は実測で単峰かつ狭く (floodgate 実戦で平均 30.1、 p10=21 / p90=39)、
/// 上位 10% を取っても中央の 1.3 倍にしかならないので特化の指標にならない。
/// 30 ペアの大半は「味方に紐が付いている」「歩同士が当たっている」等の自明な関係で、
/// threat が eval を動かすのは**浮き駒**と**安い駒が高い駒に当たっている**局面。
/// そこを直接数える。
///
/// 返り値 `(hanging, unfavorable, pairs)`
/// - `hanging`      … 敵の利きがあり味方の利きが無い駒の数 (玉を除く)
/// - `unfavorable`  … 攻撃側の駒価値 < 被攻撃側の駒価値 である敵対ペアの数
/// - `pairs`        … 全 threat ペア数 (参考)
pub fn threat_tactical_stats(pos: &PackedSfenValue) -> (u32, u32, u32) {
    /// 駒価値の序列 (threat の有利不利判定にだけ使う相対値)。
    fn value_of(pt: PieceType) -> u32 {
        match pt {
            PieceType::Pawn => 1,
            PieceType::Lance => 2,
            PieceType::Knight => 3,
            PieceType::Silver => 5,
            PieceType::Gold
            | PieceType::ProPawn
            | PieceType::ProLance
            | PieceType::ProKnight
            | PieceType::ProSilver => 6,
            PieceType::Bishop => 8,
            PieceType::Rook => 10,
            PieceType::Horse => 12,
            PieceType::Dragon => 14,
            _ => 0, // King / None
        }
    }

    let board = ShogiBoard::from_packed_sfen(pos);
    let occ = Occupied::from_board(&board);
    let mut att = [[0u8; 2]; 81]; // [square][color] = その色が利かせている駒数 (玉を除く)
    let mut pairs = 0u32;
    let mut unfavorable = 0u32;

    for sq_raw in 0..81u8 {
        let from_sq = Square(sq_raw);
        let pc = board.piece_on(from_sq);
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let acol = pc.color;
        let av = value_of(pc.piece_type);
        for_each_attack(pc.piece_type, acol, from_sq, &occ, |to_sq| {
            let tp = board.piece_on(to_sq);
            if tp.is_none() || tp.piece_type == PieceType::King {
                return;
            }
            pairs += 1;
            att[to_sq.index()][acol as usize] += 1;
            if tp.color != acol && av < value_of(tp.piece_type) {
                unfavorable += 1;
            }
        });
    }

    let mut hanging = 0u32;
    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let pc = board.piece_on(sq);
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let own = att[sq.index()][pc.color as usize];
        let enemy = att[sq.index()][pc.color.opponent() as usize];
        if enemy > 0 && own == 0 {
            hanging += 1;
        }
    }
    (hanging, unfavorable, pairs)
}

/// Full profile Threatを絶対色perspectiveで列挙し、feature-major bucket indexを返す。
pub fn fill_scalar_threat_for_perspective(
    pos: &PackedSfenValue,
    perspective: Color,
    bucket: usize,
    out: &mut [i32],
) -> usize {
    debug_assert!(bucket < SCALAR_THREAT_BUCKETS);
    let board = ShogiBoard::from_packed_sfen(pos);
    let stm = board.side_to_move;
    static INPUT: LazyLock<ShogiHalfKaHmThreat> = LazyLock::new(|| ShogiHalfKaHmThreat::new(ThreatProfile::Full));
    let mut written = 0usize;
    INPUT.map_features(pos, |stm_idx, nstm_idx| {
        let combined = if perspective == stm { stm_idx } else { nstm_idx };
        if combined >= HALFKA_HM_DIMENSIONS && written < out.len() {
            let feature = combined - HALFKA_HM_DIMENSIONS;
            out[written] = (feature * SCALAR_THREAT_BUCKETS + bucket) as i32;
            written += 1;
        }
    });
    written
}

impl SparseInputType for ShogiHalfKaHmThreat {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        if self.threat_ft_sparse { HALFKA_HM_DIMENSIONS } else { HALFKA_HM_DIMENSIONS + self.threat_dims }
    }

    fn max_active(&self) -> usize {
        if self.threat_ft_sparse { MAX_ACTIVE_FEATURES } else { MAX_ACTIVE_TOTAL }
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        if self.threat_ft_sparse {
            // Threat は専用 sparse 入力で流すので、 主入力は HalfKA_hm だけ。
            // Part 1 は必ず両視点 Some を返すので unwrap 相当で潰せる。
            self.map_halfka_pairs_split(&board, |stm_idx, nstm_idx| {
                if let (Some(s), Some(n)) = (stm_idx, nstm_idx) {
                    f(s, n);
                }
            });
        } else {
            self.map_threat_features(&board, f);
        }
    }
    /// EnemyOnly(profile 11)ではSTM/NSTMのどちらか片側だけにThreatが存在する。
    /// loaderはsplit APIを優先するため、Noneを使って正しい片側のfeatureだけをemitする。
    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, f: F) {
        let board = ShogiBoard::from_packed_sfen(pos);
        if self.threat_ft_sparse {
            self.map_halfka_pairs_split(&board, f);
        } else {
            self.map_threat_features_split(&board, f);
        }
    }

    /// 2026-11-07 (T2): 持駒 14 元 dense。 有効時だけ `Some(14)` を返す。
    fn hand_count_dims(&self) -> Option<usize> {
        if self.hand_count_dense {
            super::shogi_halfka_hm_hand_count::ShogiHalfKaHmHandCount.hand_count_dims()
        } else {
            None
        }
    }

    /// 2026-11-07 (T2): 中身は `ShogiHalfKaHmHandCount` に委譲する
    ///   (目盛り `k*c/127` を 1 箇所だけで定義するため。 2 箇所に書くと必ずずれる)。
    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        super::shogi_halfka_hm_hand_count::ShogiHalfKaHmHandCount.fill_hand_count(pos, out);
    }

    fn shorthand(&self) -> String {
        let pin_part = if self.pin_aware { "+pin" } else { "" };
        format!("shogi-{}x45hm+threat{}", HALFKA_HM_DIMENSIONS + self.threat_dims, pin_part)
    }

    fn description(&self) -> String {
        let pin_part = if self.pin_aware { " + Pin-aware (2,268 × 2 perspective)" } else { "" };
        format!(
            "Shogi HalfKA_hm ({}) + Threat ({}, profile={}) concatenated{}",
            HALFKA_HM_DIMENSIONS, self.threat_dims, self.profile, pin_part
        )
    }

    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        let mut dims: Vec<(String, usize)> = Vec::new();
        if let Some((_, k)) = &self.threat_gmask {
            dims.push((THREAT_GMASK_NAME.to_string(), *k));
            // 2026-11-17 K256 案 A: 追加列にも同じ係数のマスク (次元 N)。
            if self.threat_ext > 0 {
                dims.push((THREAT_EXT_GMASK_NAME.to_string(), self.threat_ext));
            }
        }
        if self.pin_aware {
            for (name, d) in pin_aux_entries().iter() {
                dims.push(((*name).to_string(), *d));
            }
        }
        if self.mob {
            dims.push((MOB_NAME.to_string(), 1));
        }
        dims
    }

    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        if name == THREAT_GMASK_NAME || name == THREAT_EXT_GMASK_NAME {
            let w = self.threat_grad_weight(pos.score());
            out.fill(w);
            return;
        }
        if self.mob && name == MOB_NAME {
            // 🚨board を decode するので、 mob が無効なときは通らないようにしてある。
            let board = ShogiBoard::from_packed_sfen(pos);
            out[0] = super::shogi_mob::mob_diff(&board) as f32 / MOB_INPUT_SCALE;
            return;
        }
        if self.pin_aware {
            let _ = try_fill_pin_aux(name, pos, out);
        }
    }

    fn auxiliary_sparse_dims(&self) -> Vec<(String, usize, usize)> {
        let mut dims = Vec::new();
        if self.threat_ft_sparse {
            // 群化時は群ごとに別の列を引くので次元が群数倍になる (群化しなければ従来と同じ)。
            // threat factorizer 時は末尾に factor 行 (324) が付き、 1 特徴あたり 2 index になる。
            let factor = self.threat_factor.is_some();
            // profile 16 (hanging-split): 末尾に群共有の delta 行 (108,360) が付き、
            // ひもの無い pair だけ 1 特徴あたり 2 index になる。
            let hang = self.threat_hang_of.is_some();
            let extra_rows = if factor { THREAT_FACTOR_ROWS } else { 0 } + if hang { THREAT_HANG_ROWS } else { 0 };
            let dims_total = self.threat_dims * self.threat_group_count() + extra_rows;
            let nnz = MAX_ACTIVE_THREAT_FEATURES * if factor || hang { 2 } else { 1 };
            dims.push((THREAT_FT_SPARSE_STM.to_string(), dims_total, nnz));
            dims.push((THREAT_FT_SPARSE_NSTM.to_string(), dims_total, nnz));
        }
        if let Some((parts, _)) = self.kh {
            // 2026-11-12: 次元と nnz は K 部の表現 (`--kh-kmode`) で変わる (kmode 1 = 従来の 142 / 54)。
            use super::shogi_kh::{KH_SPARSE_NSTM, KH_SPARSE_STM};
            dims.push((KH_SPARSE_STM.to_string(), parts.dims(), parts.max_active()));
            dims.push((KH_SPARSE_NSTM.to_string(), parts.dims(), parts.max_active()));
        }
        if self.threat_ext > 0 {
            // 2026-11-17 K256 案 A: 追加列は 1 群なので次元は threat_dims のまま (群数倍しない)。
            dims.push((THREAT_EXT_SPARSE_STM.to_string(), self.threat_dims, MAX_ACTIVE_THREAT_FEATURES));
            dims.push((THREAT_EXT_SPARSE_NSTM.to_string(), self.threat_dims, MAX_ACTIVE_THREAT_FEATURES));
        }
        if !self.dual_psqt_enabled {
            return dims;
        }
        if self.dual_king_attacker_enabled {
            dims.push((DUAL_SPARSE_NAME_ATTACK_BLACK.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
            dims.push((DUAL_SPARSE_NAME_DEFEND_BLACK.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
            dims.push((DUAL_SPARSE_NAME_ATTACK_WHITE.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
            dims.push((DUAL_SPARSE_NAME_DEFEND_WHITE.to_string(), DUAL_DIM_PER_HEAD, DUAL_MAX_ACTIVE_PER_HEAD));
        }
        dims.push((DUAL_CROSS_THREAT_BLACK.to_string(), DUAL_CROSS_DIM_PER_HEAD, DUAL_CROSS_MAX_ACTIVE_PER_HEAD));
        if self.dual_cross_defended_enabled {
            dims.push((DUAL_CROSS_DEFENDED_BLACK.to_string(), DUAL_CROSS_DIM_PER_HEAD, DUAL_CROSS_MAX_ACTIVE_PER_HEAD));
        }
        dims.push((DUAL_CROSS_THREAT_WHITE.to_string(), DUAL_CROSS_DIM_PER_HEAD, DUAL_CROSS_MAX_ACTIVE_PER_HEAD));
        if self.dual_cross_defended_enabled {
            dims.push((DUAL_CROSS_DEFENDED_WHITE.to_string(), DUAL_CROSS_DIM_PER_HEAD, DUAL_CROSS_MAX_ACTIVE_PER_HEAD));
        }
        dims
    }

    fn fill_auxiliary_sparse(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [i32]) -> usize {
        if self.threat_ft_sparse && (name == THREAT_FT_SPARSE_STM || name == THREAT_FT_SPARSE_NSTM) {
            return self.fill_threat_ft_sparse(pos, name == THREAT_FT_SPARSE_STM, out);
        }
        if self.threat_ext > 0 && (name == THREAT_EXT_SPARSE_STM || name == THREAT_EXT_SPARSE_NSTM) {
            return self.fill_threat_ext_sparse(pos, name == THREAT_EXT_SPARSE_STM, out);
        }
        if let Some((parts, null)) = self.kh {
            use super::shogi_kh::{KH_SPARSE_NSTM, KH_SPARSE_STM, fill_kh_sparse};
            if name == KH_SPARSE_STM || name == KH_SPARSE_NSTM {
                return fill_kh_sparse(pos, name == KH_SPARSE_STM, parts, null, out);
            }
        }
        if !self.dual_psqt_enabled {
            return 0;
        }
        match name {
            DUAL_SPARSE_NAME_ATTACK_BLACK => fill_dual_sparse_head(pos, Color::Black, DualHead::Attack, out),
            DUAL_SPARSE_NAME_DEFEND_BLACK => fill_dual_sparse_head(pos, Color::Black, DualHead::Defend, out),
            DUAL_SPARSE_NAME_ATTACK_WHITE => fill_dual_sparse_head(pos, Color::White, DualHead::Attack, out),
            DUAL_SPARSE_NAME_DEFEND_WHITE => fill_dual_sparse_head(pos, Color::White, DualHead::Defend, out),
            DUAL_CROSS_THREAT_BLACK => fill_dual_cross_head(pos, Color::Black, CrossHead::Threat, out),
            DUAL_CROSS_DEFENDED_BLACK => fill_dual_cross_head(pos, Color::Black, CrossHead::Defended, out),
            DUAL_CROSS_THREAT_WHITE => fill_dual_cross_head(pos, Color::White, CrossHead::Threat, out),
            DUAL_CROSS_DEFENDED_WHITE => fill_dual_cross_head(pos, Color::White, CrossHead::Defended, out),
            _ => 0,
        }
    }
}

// =============================================================================
// 特徴量列挙
// =============================================================================

impl ShogiHalfKaHmThreat {
    /// HalfKA_hm + Threat の特徴量を列挙する
    fn map_threat_features<F: FnMut(usize, usize)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king_sq = board.king_square(stm);
        let nstm_king_sq = board.king_square(nstm);
        if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
            return;
        }

        // -------------------------------------------------------
        // Part 1: HalfKA_hm 特徴量（既存ロジック複製）
        // -------------------------------------------------------

        let stm_kb = king_bucket(stm_king_sq, stm);
        let stm_hm = is_hm_mirror(stm_king_sq, stm);
        let nstm_kb = king_bucket(nstm_king_sq, nstm);
        let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);

        // 盤上の駒（王以外）
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);
                    let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                    let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                    let stm_idx = halfka_index(stm_kb, stm_packed);

                    let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                    let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                    let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                    f(stm_idx, nstm_idx);
                }
            }
        }

        // 両方の玉の特徴量
        {
            let stm_king_sq_idx = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
            let stm_friend_king_bp = king_bonapiece(stm_king_sq_idx, true);
            let stm_friend_packed = pack_bonapiece(stm_friend_king_bp, stm_hm);
            let stm_friend_idx = halfka_index(stm_kb, stm_friend_packed);

            let nstm_king_sq_for_stm =
                if stm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
            let stm_enemy_king_bp = king_bonapiece(nstm_king_sq_for_stm, false);
            let stm_enemy_packed = pack_bonapiece(stm_enemy_king_bp, stm_hm);
            let stm_enemy_idx = halfka_index(stm_kb, stm_enemy_packed);

            let nstm_king_sq_idx =
                if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
            let nstm_friend_king_bp = king_bonapiece(nstm_king_sq_idx, true);
            let nstm_friend_packed = pack_bonapiece(nstm_friend_king_bp, nstm_hm);
            let nstm_friend_idx = halfka_index(nstm_kb, nstm_friend_packed);

            let stm_king_sq_for_nstm =
                if nstm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
            let nstm_enemy_king_bp = king_bonapiece(stm_king_sq_for_nstm, false);
            let nstm_enemy_packed = pack_bonapiece(nstm_enemy_king_bp, nstm_hm);
            let nstm_enemy_idx = halfka_index(nstm_kb, nstm_enemy_packed);

            f(stm_friend_idx, nstm_friend_idx);
            f(stm_enemy_idx, nstm_enemy_idx);
        }

        // 手駒の特徴量
        for owner in [Color::Black, Color::White] {
            for &pt in &HAND_PIECE_TYPES {
                let count = board.hand(owner).count(pt);
                if count == 0 {
                    continue;
                }
                for i in 1..=count {
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                    if stm_bp != BonaPiece::ZERO {
                        let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                        let stm_idx = halfka_index(stm_kb, stm_packed);

                        let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                        let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                        let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                        f(stm_idx, nstm_idx);
                    }
                }
            }
        }

        // -------------------------------------------------------
        // Part 2: Threat 特徴量
        // -------------------------------------------------------

        let from_offset_table = &*FROM_OFFSET_TABLE;
        let occ = Occupied::from_board(board);

        // STM perspective
        let stm_friend = stm;

        // NSTM perspective
        let nstm_friend = nstm;

        // 全盤上駒を列挙して threat pair を生成
        for sq_raw in 0..81u8 {
            let from_sq = Square(sq_raw);
            let pc = board.piece_on(from_sq);
            if pc.is_none() {
                continue;
            }
            let pt = pc.piece_type;
            let attacker_color = pc.color;

            // King は除外
            if pt == PieceType::King {
                continue;
            }

            let attacker_class = match ThreatClass::from_piece_type(pt) {
                Some(c) => c,
                None => continue,
            };

            // 実盤面上の攻撃先を列挙
            for_each_attack(pt, attacker_color, from_sq, &occ, |to_sq| {
                let target_pc = board.piece_on(to_sq);
                if target_pc.is_none() {
                    return;
                }
                let target_pt = target_pc.piece_type;
                let target_color = target_pc.color;

                // King は除外
                if target_pt == PieceType::King {
                    return;
                }

                let attacked_class = match ThreatClass::from_piece_type(target_pt) {
                    Some(c) => c,
                    None => return,
                };

                // --- STM perspective ---
                let stm_attacker_side = if attacker_color == stm_friend { 0 } else { 1 };
                let stm_attacked_side = if target_color == stm_friend { 0 } else { 1 };
                let stm_from_n = normalize_sq(from_sq, stm, stm_hm);
                let stm_to_n = normalize_sq(to_sq, stm, stm_hm);
                let stm_oriented_color = if stm == Color::Black { attacker_color } else { attacker_color.opponent() };
                let stm_threat_idx = threat_index(
                    &ThreatParams {
                        attacker_side: stm_attacker_side,
                        attacker_class,
                        oriented_color: stm_oriented_color,
                        attacked_side: stm_attacked_side,
                        attacked_class,
                        from_sq_n: stm_from_n,
                        to_sq_n: stm_to_n,
                    },
                    &self.pair_base,
                    from_offset_table,
                );
                let Some(stm_threat_idx) = stm_threat_idx else {
                    return; // excluded pair
                };
                debug_assert!(stm_threat_idx < self.threat_dims);
                let stm_idx = HALFKA_HM_DIMENSIONS + stm_threat_idx;

                // --- NSTM perspective ---
                let nstm_attacker_side = if attacker_color == nstm_friend { 0 } else { 1 };
                let nstm_attacked_side = if target_color == nstm_friend { 0 } else { 1 };
                let nstm_from_n = normalize_sq(from_sq, nstm, nstm_hm);
                let nstm_to_n = normalize_sq(to_sq, nstm, nstm_hm);
                let nstm_oriented_color = if nstm == Color::Black { attacker_color } else { attacker_color.opponent() };
                let nstm_threat_idx = threat_index(
                    &ThreatParams {
                        attacker_side: nstm_attacker_side,
                        attacker_class,
                        oriented_color: nstm_oriented_color,
                        attacked_side: nstm_attacked_side,
                        attacked_class,
                        from_sq_n: nstm_from_n,
                        to_sq_n: nstm_to_n,
                    },
                    &self.pair_base,
                    from_offset_table,
                );
                let Some(nstm_threat_idx) = nstm_threat_idx else {
                    return; // excluded pair
                };
                debug_assert!(nstm_threat_idx < self.threat_dims);
                let nstm_idx = HALFKA_HM_DIMENSIONS + nstm_threat_idx;

                f(stm_idx, nstm_idx);
            });
        }
    }

    fn map_threat_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, board: &ShogiBoard, mut f: F) {
        self.map_halfka_pairs_split(board, &mut f);
        // hanging フラグはこの経路 (主 sparse への連結) では使わない。 profile 16 は
        // threat_ft_sparse 経由でしか使えない (起動時 guard で保証する)。
        self.map_threat_pairs_split(board, |a, b, _hang| f(a, b));
    }

    /// 専用 sparse 入力へ流す Threat 特徴 index を書き出す。
    ///
    /// `stm_side = true` で手番側視点、 false で非手番側視点。 index は
    /// `HALFKA_HM_DIMENSIONS` を引いた **threat ローカル**の値 (0..threat_dims)。
    /// profile 11 系は片側にしか出ない pair があるため `Option` を素直に落とす。
    fn fill_threat_ft_sparse(&self, pos: &PackedSfenValue, stm_side: bool, out: &mut [i32]) -> usize {
        let board = ShogiBoard::from_packed_sfen(pos);
        // 群化時: この局面の LayerStack bucket が属する群の表を引く (index を群 * threat_dims ずらす)。
        // 手番側・非手番側とも同じ局面なので同じ群になる (エンジンも両視点で同じ群の表を使う)。
        let group_offset = match &self.threat_bucket_groups {
            Some(groups) => groups[ShogiProgressKPAbs.bucket(pos) as usize] as usize * self.threat_dims,
            None => 0,
        };
        let mut written = 0usize;
        let factor_base = self.threat_dims * self.threat_group_count();
        let factor = self.threat_factor.as_deref();
        let hang_of = self.threat_hang_of.as_deref();
        self.map_threat_pairs_split(&board, |stm_idx, nstm_idx, hanging| {
            let Some(combined) = (if stm_side { stm_idx } else { nstm_idx }) else {
                return;
            };
            debug_assert!(combined >= HALFKA_HM_DIMENSIONS);
            let f = combined - HALFKA_HM_DIMENSIONS;
            // profile 16: ひもが無い pair は 実特徴と delta 特徴を**対で**出す。
            // 片方だけ落ちると学習と export の関数が食い違うので、 2 個入らないなら両方捨てる。
            if let Some(table) = hang_of {
                let delta = if hanging { table[f] } else { -1 };
                if delta >= 0 {
                    if written + 2 <= out.len() {
                        out[written] = (f + group_offset) as i32;
                        out[written + 1] = (factor_base + delta as usize) as i32;
                        written += 2;
                    }
                } else if written < out.len() {
                    out[written] = (f + group_offset) as i32;
                    written += 1;
                }
                return;
            }
            match factor {
                None => {
                    if written < out.len() {
                        out[written] = (f + group_offset) as i32;
                        written += 1;
                    }
                }
                Some(table) => {
                    // 実特徴と factor 特徴は対で出す (片方だけ落ちると学習と export の関数が食い違う)。
                    if written + 2 <= out.len() {
                        let fi = factor_base + table[f] as usize;
                        debug_assert!(fi >= factor_base && f + group_offset < factor_base);
                        out[written] = (f + group_offset) as i32;
                        out[written + 1] = fi as i32;
                        written += 2;
                    }
                }
            }
        });
        written
    }

    /// 2026-11-17 K256 案 A: 追加列用の threat 疎入力。 [`Self::fill_threat_ft_sparse`] の factor / hanging-split なし・
    /// **群 offset = 0** の版 (index = 特徴 f そのもの)。 打ち切り (`out.len()` 超) の規則も同じ。
    fn fill_threat_ext_sparse(&self, pos: &PackedSfenValue, stm_side: bool, out: &mut [i32]) -> usize {
        debug_assert!(self.threat_factor.is_none() && self.threat_hang_of.is_none());
        let board = ShogiBoard::from_packed_sfen(pos);
        let mut written = 0usize;
        self.map_threat_pairs_split(&board, |stm_idx, nstm_idx, _hanging| {
            let Some(combined) = (if stm_side { stm_idx } else { nstm_idx }) else {
                return;
            };
            debug_assert!(combined >= HALFKA_HM_DIMENSIONS);
            if written < out.len() {
                out[written] = (combined - HALFKA_HM_DIMENSIONS) as i32;
                written += 1;
            }
        });
        written
    }

    /// Part 1 (HalfKA_hm 特徴量) だけを列挙する。
    ///
    /// threat を専用 sparse 入力に分離したときの主入力はこれだけになる。
    fn map_halfka_pairs_split<F: FnMut(Option<usize>, Option<usize>)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king_sq = board.king_square(stm);
        let nstm_king_sq = board.king_square(nstm);
        if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
            return;
        }

        // -------------------------------------------------------
        // Part 1: HalfKA_hm 特徴量（既存ロジック複製）
        // -------------------------------------------------------

        let stm_kb = king_bucket(stm_king_sq, stm);
        let stm_hm = is_hm_mirror(stm_king_sq, stm);
        let nstm_kb = king_bucket(nstm_king_sq, nstm);
        let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);

        // 盤上の駒（王以外）
        for &pt in &BOARD_PIECE_TYPES {
            for color in [Color::Black, Color::White] {
                for sq in board.pieces(color, pt) {
                    let piece = Piece::new(color, pt);
                    let stm_bp = BonaPiece::from_piece_square(piece, sq, stm);
                    let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                    let stm_idx = halfka_index(stm_kb, stm_packed);

                    let nstm_bp = BonaPiece::from_piece_square(piece, sq, nstm);
                    let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                    let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                    f(Some(stm_idx), Some(nstm_idx));
                }
            }
        }

        // 両方の玉の特徴量
        {
            let stm_king_sq_idx = if stm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
            let stm_friend_king_bp = king_bonapiece(stm_king_sq_idx, true);
            let stm_friend_packed = pack_bonapiece(stm_friend_king_bp, stm_hm);
            let stm_friend_idx = halfka_index(stm_kb, stm_friend_packed);

            let nstm_king_sq_for_stm =
                if stm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
            let stm_enemy_king_bp = king_bonapiece(nstm_king_sq_for_stm, false);
            let stm_enemy_packed = pack_bonapiece(stm_enemy_king_bp, stm_hm);
            let stm_enemy_idx = halfka_index(stm_kb, stm_enemy_packed);

            let nstm_king_sq_idx =
                if nstm == Color::Black { nstm_king_sq.index() } else { nstm_king_sq.inverse().index() };
            let nstm_friend_king_bp = king_bonapiece(nstm_king_sq_idx, true);
            let nstm_friend_packed = pack_bonapiece(nstm_friend_king_bp, nstm_hm);
            let nstm_friend_idx = halfka_index(nstm_kb, nstm_friend_packed);

            let stm_king_sq_for_nstm =
                if nstm == Color::Black { stm_king_sq.index() } else { stm_king_sq.inverse().index() };
            let nstm_enemy_king_bp = king_bonapiece(stm_king_sq_for_nstm, false);
            let nstm_enemy_packed = pack_bonapiece(nstm_enemy_king_bp, nstm_hm);
            let nstm_enemy_idx = halfka_index(nstm_kb, nstm_enemy_packed);

            f(Some(stm_friend_idx), Some(nstm_friend_idx));
            f(Some(stm_enemy_idx), Some(nstm_enemy_idx));
        }

        // 手駒の特徴量
        for owner in [Color::Black, Color::White] {
            for &pt in &HAND_PIECE_TYPES {
                let count = board.hand(owner).count(pt);
                if count == 0 {
                    continue;
                }
                for i in 1..=count {
                    let stm_bp = BonaPiece::from_hand_piece(stm, owner, pt, i);
                    if stm_bp != BonaPiece::ZERO {
                        let stm_packed = pack_bonapiece(stm_bp, stm_hm);
                        let stm_idx = halfka_index(stm_kb, stm_packed);

                        let nstm_bp = BonaPiece::from_hand_piece(nstm, owner, pt, i);
                        let nstm_packed = pack_bonapiece(nstm_bp, nstm_hm);
                        let nstm_idx = halfka_index(nstm_kb, nstm_packed);

                        f(Some(stm_idx), Some(nstm_idx));
                    }
                }
            }
        }
    }

    /// Part 2 (Threat 特徴量) だけを列挙する。
    ///
    /// 返す index は `map_threat_features_split` と同じ**連結後**の値
    /// (= `HALFKA_HM_DIMENSIONS + threat_idx`)。 専用 sparse 入力に流すときは
    /// 呼び出し側で `HALFKA_HM_DIMENSIONS` を引くこと。
    /// `f(stm_idx, nstm_idx, hanging)`。 `hanging` は「被攻撃駒にひもが無い」(profile 16 でのみ true になりうる)。
    fn map_threat_pairs_split<F: FnMut(Option<usize>, Option<usize>, bool)>(&self, board: &ShogiBoard, mut f: F) {
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        let stm_king_sq = board.king_square(stm);
        let nstm_king_sq = board.king_square(nstm);
        if !stm_king_sq.is_valid() || !nstm_king_sq.is_valid() {
            return;
        }

        let stm_hm = is_hm_mirror(stm_king_sq, stm);
        let nstm_hm = is_hm_mirror(nstm_king_sq, nstm);
        // -------------------------------------------------------
        // Part 2: Threat 特徴量
        // -------------------------------------------------------

        let use_representatives = self.profile == ThreatProfile::EnemyOnlyRep8;
        let mut stm_representatives = [(0usize, 0u8); REPRESENTATIVE_THREAT_ROWS];
        let mut nstm_representatives = [(0usize, 0u8); REPRESENTATIVE_THREAT_ROWS];
        let mut stm_representative_count = 0usize;
        let mut nstm_representative_count = 0usize;
        let from_offset_table = &*FROM_OFFSET_TABLE;
        let occ = Occupied::from_board(board);
        // profile 16: 各色の「その升に自分の駒の利きがあるか」(玉・歩を含む)。
        // C++ の board_effect[c].e[sq] != 0 と同じ定義。 Direct15 P5 の defended_map と同じ関数を使う。
        let mut defended_by = [[false; 81]; 2];
        if self.threat_hang_of.is_some() {
            for raw in 0..81u8 {
                let from = Square(raw);
                let piece = board.piece_on(from);
                if piece.is_none() {
                    continue;
                }
                let ci = if piece.color == Color::Black { 0 } else { 1 };
                if piece.piece_type == PieceType::King {
                    // 玉の 8 近傍 (盤外は弾く)
                    let file = (from.0 / 9) as i32;
                    let rank = (from.0 % 9) as i32;
                    for df in -1..=1i32 {
                        for dr in -1..=1i32 {
                            if df == 0 && dr == 0 {
                                continue;
                            }
                            let (nf, nr) = (file + df, rank + dr);
                            if (0..9).contains(&nf) && (0..9).contains(&nr) {
                                defended_by[ci][(nf * 9 + nr) as usize] = true;
                            }
                        }
                    }
                } else {
                    for_each_attack(piece.piece_type, piece.color, from, &occ, |to| {
                        defended_by[ci][to.0 as usize] = true;
                    });
                }
            }
        }

        // STM perspective
        let stm_friend = stm;

        // NSTM perspective
        let nstm_friend = nstm;

        // 全盤上駒を列挙して threat pair を生成
        for sq_raw in 0..81u8 {
            let from_sq = Square(sq_raw);
            let pc = board.piece_on(from_sq);
            if pc.is_none() {
                continue;
            }
            let pt = pc.piece_type;
            let attacker_color = pc.color;

            // King は除外
            if pt == PieceType::King {
                continue;
            }

            let attacker_class = match ThreatClass::from_piece_type(pt) {
                Some(c) => c,
                None => continue,
            };

            // 実盤面上の攻撃先を列挙
            for_each_attack(pt, attacker_color, from_sq, &occ, |to_sq| {
                let target_pc = board.piece_on(to_sq);
                if target_pc.is_none() {
                    return;
                }
                let target_pt = target_pc.piece_type;
                let target_color = target_pc.color;

                // King は除外
                if target_pt == PieceType::King {
                    return;
                }

                let attacked_class = match ThreatClass::from_piece_type(target_pt) {
                    Some(c) => c,
                    None => return,
                };

                // --- STM perspective ---
                let stm_attacker_side = if attacker_color == stm_friend { 0 } else { 1 };
                let stm_attacked_side = if target_color == stm_friend { 0 } else { 1 };
                let stm_from_n = normalize_sq(from_sq, stm, stm_hm);
                let stm_to_n = normalize_sq(to_sq, stm, stm_hm);
                let stm_oriented_color = if stm == Color::Black { attacker_color } else { attacker_color.opponent() };
                let stm_threat_idx = threat_index(
                    &ThreatParams {
                        attacker_side: stm_attacker_side,
                        attacker_class,
                        oriented_color: stm_oriented_color,
                        attacked_side: stm_attacked_side,
                        attacked_class,
                        from_sq_n: stm_from_n,
                        to_sq_n: stm_to_n,
                    },
                    &self.pair_base,
                    from_offset_table,
                );
                let stm_idx = stm_threat_idx.map(|idx| {
                    debug_assert!(idx < self.threat_dims);
                    HALFKA_HM_DIMENSIONS + idx
                });
                // --- NSTM perspective ---
                let nstm_attacker_side = if attacker_color == nstm_friend { 0 } else { 1 };
                let nstm_attacked_side = if target_color == nstm_friend { 0 } else { 1 };
                let nstm_from_n = normalize_sq(from_sq, nstm, nstm_hm);
                let nstm_to_n = normalize_sq(to_sq, nstm, nstm_hm);
                let nstm_oriented_color = if nstm == Color::Black { attacker_color } else { attacker_color.opponent() };
                let nstm_threat_idx = threat_index(
                    &ThreatParams {
                        attacker_side: nstm_attacker_side,
                        attacker_class,
                        oriented_color: nstm_oriented_color,
                        attacked_side: nstm_attacked_side,
                        attacked_class,
                        from_sq_n: nstm_from_n,
                        to_sq_n: nstm_to_n,
                    },
                    &self.pair_base,
                    from_offset_table,
                );
                let nstm_idx = nstm_threat_idx.map(|idx| {
                    debug_assert!(idx < self.threat_dims);
                    HALFKA_HM_DIMENSIONS + idx
                });
                if use_representatives {
                    let priority = representative_priority(attacker_class, attacked_class);
                    if let Some(index) = stm_idx {
                        offer_representative(&mut stm_representatives, &mut stm_representative_count, index, priority);
                    }
                    if let Some(index) = nstm_idx {
                        offer_representative(
                            &mut nstm_representatives,
                            &mut nstm_representative_count,
                            index,
                            priority,
                        );
                    }
                } else if stm_idx.is_some() || nstm_idx.is_some() {
                    // ひも = 被攻撃駒と同じ色の駒がその升に利いている。 視点に依らない盤上の事実。
                    let hanging = if self.threat_hang_of.is_some() {
                        let ci = if target_color == Color::Black { 0 } else { 1 };
                        !defended_by[ci][to_sq.0 as usize]
                    } else {
                        false
                    };
                    f(stm_idx, nstm_idx, hanging);
                }
            });
        }

        if use_representatives {
            // 代表行圧縮 (profile 12) は profile 16 と併用しない (起動時 guard)。
            for &(index, _) in stm_representatives.iter().take(stm_representative_count) {
                f(Some(index), None, false);
            }
            for &(index, _) in nstm_representatives.iter().take(nstm_representative_count) {
                f(None, Some(index), false);
            }
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::PieceType;

    fn full_input() -> ShogiHalfKaHmThreat {
        ShogiHalfKaHmThreat::new(ThreatProfile::Full)
    }

    #[test]
    fn test_total_dimensions() {
        let input = full_input();
        assert_eq!(input.num_inputs(), HALFKA_HM_DIMENSIONS + 216_720);
    }

    #[test]
    fn test_cross_side_dimensions() {
        let input = ShogiHalfKaHmThreat::new(ThreatProfile::CrossSide);
        assert_eq!(input.threat_dimensions(), 96_320);
    }

    #[test]
    fn test_no_pawn_target_dimensions() {
        let input = ShogiHalfKaHmThreat::new(ThreatProfile::NoPawnTarget);
        // dc == 0 (Pawn target) の pair を全除外 = (as, ds, dc) 36 通り -> 32 通り
        // sum(ATTACKS_PER_COLOR) = 6020 -> 6020 * 32 = 192_640
        assert_eq!(input.threat_dimensions(), 192_640);
    }

    /// NoPawnTarget profile で残る pair は Full の対応 pair と同じ長さのブロックであり、
    /// Full -> NoPawnTarget の重み移植が「連続ブロックの並べ替え」で表現できることを確認する。
    #[test]
    fn test_no_pawn_target_layout_is_block_permutation_of_full() {
        let full = threat_pair_layout(ThreatProfile::Full);
        let npt = threat_pair_layout(ThreatProfile::NoPawnTarget);
        assert_eq!(full.len(), 324);
        assert_eq!(npt.len(), 288);

        let mut expected_base = 0usize;
        for &(pair_idx, base, len) in &npt {
            assert_eq!(base, expected_base);
            expected_base += len;
            let dc = pair_idx % 9;
            assert_ne!(dc, 0, "pawn target pair must be excluded");
            let (_, _, full_len) = full[full.iter().position(|&(i, _, _)| i == pair_idx).unwrap()];
            assert_eq!(len, full_len);
        }
        assert_eq!(expected_base, 192_640);
    }

    #[test]
    fn test_max_active() {
        let input = full_input();
        assert_eq!(input.max_active(), 64 + 320);
    }

    #[test]
    fn test_pair_base_dimensions() {
        let input = full_input();
        // 最後の non-excluded pair の base + attacks_per_color == threat_dims
        let mut last_base = 0usize;
        let mut last_ac = 0usize;
        for (i, &base) in input.pair_base.iter().enumerate() {
            if base != EXCLUDED_PAIR_BASE && base >= last_base {
                last_base = base;
                last_ac = (i % 162) / 18;
            }
        }
        assert_eq!(last_base + ATTACKS_PER_COLOR[last_ac], input.threat_dimensions());
    }

    #[test]
    fn test_attacks_per_color_totals() {
        let all_classes = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];
        for (i, &class) in all_classes.iter().enumerate() {
            let total: usize = (0..81u8)
                .map(|sq| {
                    let (_, cnt) = attacks_empty_board(class, Color::Black, Square(sq));
                    cnt
                })
                .sum();
            assert_eq!(total, ATTACKS_PER_COLOR[i], "{:?}: expected {}, got {}", class, ATTACKS_PER_COLOR[i], total);
        }
    }

    #[test]
    fn test_from_offset_pawn() {
        let table = FromOffsetTable::new();
        let pattern = attack_pattern_id(ThreatClass::Pawn, Color::Black);
        // sq=0 (file=0, rank=0): offset=0
        assert_eq!(table.get(pattern, Square(0)), 0);
        // sq=1 (file=0, rank=1): offset=0 (sq=0 has 0 attacks for Black pawn at rank=0)
        assert_eq!(table.get(pattern, Square(1)), 0);
        // sq=2 (file=0, rank=2): offset=1 (sq=1 has 1 attack)
        assert_eq!(table.get(pattern, Square(2)), 1);
    }

    #[test]
    fn test_from_offset_rook() {
        let table = FromOffsetTable::new();
        let pattern = attack_pattern_id(ThreatClass::Rook, Color::Black);
        // Rook: 全マスで attacks=16
        for sq_raw in 0..81u8 {
            assert_eq!(table.get(pattern, Square(sq_raw)), 16 * sq_raw as usize);
        }
    }

    #[test]
    fn test_attack_order_rook_center() {
        // Rook at sq=40 (5五): 16 攻撃先
        let (targets, count) = attacks_empty_board(ThreatClass::Rook, Color::Black, Square(40));
        assert_eq!(count, 16);

        // 最初の攻撃先は order=0
        let first_sq = Square(targets[0]);
        assert_eq!(compute_attack_order(ThreatClass::Rook, Color::Black, Square(40), first_sq), 0);
    }

    #[test]
    fn test_threat_index_range() {
        let input = full_input();
        let from_offset_table = FromOffsetTable::new();
        let all_classes = [
            ThreatClass::Pawn,
            ThreatClass::Lance,
            ThreatClass::Knight,
            ThreatClass::Silver,
            ThreatClass::GoldLike,
            ThreatClass::Bishop,
            ThreatClass::Rook,
            ThreatClass::Horse,
            ThreatClass::Dragon,
        ];

        for &class in &all_classes {
            for &oriented_color in &[Color::Black, Color::White] {
                for sq_raw in 0..81u8 {
                    let sq = Square(sq_raw);
                    let (targets, count) = attacks_empty_board(class, oriented_color, sq);
                    for &target_raw in targets.iter().take(count) {
                        let to = Square(target_raw);
                        for as_ in 0..2usize {
                            for ds in 0..2usize {
                                for &dc_class in &all_classes {
                                    let idx = threat_index(
                                        &ThreatParams {
                                            attacker_side: as_,
                                            attacker_class: class,
                                            oriented_color,
                                            attacked_side: ds,
                                            attacked_class: dc_class,
                                            from_sq_n: sq,
                                            to_sq_n: to,
                                        },
                                        &input.pair_base,
                                        &from_offset_table,
                                    );
                                    if let Some(idx) = idx {
                                        assert!(
                                            idx < input.threat_dimensions(),
                                            "index {} out of range for class={:?} color={:?} sq={} to={}",
                                            idx,
                                            class,
                                            oriented_color,
                                            sq.0,
                                            to.0
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// 専用 sparse 入力モード用に切り出した Part1 / Part2 が、
    /// 連結版 `map_threat_features_split` と**完全に同じ列**を出すことを保証する。
    /// (2026-09-01 の「tail 特徴に勾配が来ずゼロのまま」級の静かな不一致を防ぐ)
    fn split_parts_board() -> ShogiBoard {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }
        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop);
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop);
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook);
        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        // 2八飛の前の歩を除けて、 飛車が 2三の後手歩に当たるようにする。
        // enemy-only (profile 11) は「敵の攻撃駒 -> 味方の被攻撃駒」かつ同種除外なので、
        // 初期局面のままだと threat が 1 つも立たずテストにならない。
        board.board[Square::new(1, 6).index()] = Piece::default();
        board
    }

    #[test]
    fn split_parts_reproduce_concatenated_stream() {
        for profile in [ThreatProfile::Full, ThreatProfile::EnemyOnly] {
            let input = ShogiHalfKaHmThreat::new(profile);
            let board = split_parts_board();

            let mut combined = Vec::new();
            input.map_threat_features_split(&board, |a, b| combined.push((a, b)));

            let mut parts = Vec::new();
            input.map_halfka_pairs_split(&board, |a, b| parts.push((a, b)));
            let halfka_len = parts.len();
            input.map_threat_pairs_split(&board, |a, b, _hang| parts.push((a, b)));

            assert_eq!(combined, parts, "profile={profile:?}: 分割前後で列が一致しない");
            assert!(halfka_len > 0, "profile={profile:?}: HalfKA 特徴が空");
            assert!(parts.len() > halfka_len, "profile={profile:?}: threat 特徴が空");
            for (a, b) in &parts[..halfka_len] {
                assert!(a.unwrap() < HALFKA_HM_DIMENSIONS && b.unwrap() < HALFKA_HM_DIMENSIONS);
            }
            for (a, b) in &parts[halfka_len..] {
                for idx in [a, b].into_iter().flatten() {
                    assert!(*idx >= HALFKA_HM_DIMENSIONS, "threat index が HalfKA 域にある");
                    assert!(*idx - HALFKA_HM_DIMENSIONS < input.threat_dimensions());
                }
            }
        }
    }

    #[test]
    fn threat_ft_sparse_changes_main_input_shape_only() {
        let plain = ShogiHalfKaHmThreat::new(ThreatProfile::Full);
        let split = ShogiHalfKaHmThreat::new(ThreatProfile::Full).with_threat_ft_sparse(true);

        assert_eq!(plain.num_inputs(), HALFKA_HM_DIMENSIONS + plain.threat_dimensions());
        assert_eq!(split.num_inputs(), HALFKA_HM_DIMENSIONS);
        assert_eq!(split.threat_dimensions(), plain.threat_dimensions());
        assert!(split.max_active() < plain.max_active());

        let names: Vec<String> = split.auxiliary_sparse_dims().into_iter().map(|(n, _, _)| n).collect();
        assert_eq!(names, vec![THREAT_FT_SPARSE_STM.to_string(), THREAT_FT_SPARSE_NSTM.to_string()]);
        assert!(plain.auxiliary_sparse_dims().is_empty());
    }

    /// `--threat-bucket-groups`: 専用 sparse 入力の index が `群 * threat_dims` だけずれ、 次元が群数倍になる。
    #[test]
    fn threat_bucket_groups_shift_threat_ft_sparse_indices() {
        // washiun_mix12.psv の 123,457 番目のレコード (threat が立つ実戦局面)。
        const RECORD: [u8; 40] = [
            0x8d, 0x9c, 0x91, 0x14, 0x0c, 0x24, 0xfc, 0xf0, 0x4a, 0x01, 0x02, 0xbe, 0x9d, 0x67, 0x3a, 0x1e, 0x90, 0x02,
            0xac, 0x29, 0x8e, 0x87, 0x5f, 0x52, 0x2c, 0x23, 0x29, 0x18, 0x1e, 0x87, 0x64, 0xad, 0x9e, 0xff, 0x00, 0x00,
            0x45, 0x00, 0x01, 0x00,
        ];
        let mut pos = PackedSfenValue::default();
        pos.as_bytes_mut().copy_from_slice(&RECORD);

        let plain = ShogiHalfKaHmThreat::new(ThreatProfile::Full).with_threat_ft_sparse(true);
        let groups = vec![0u8, 1, 2, 3, 4, 5, 6, 7];
        let grouped = ShogiHalfKaHmThreat::new(ThreatProfile::Full)
            .with_threat_ft_sparse(true)
            .with_threat_bucket_groups(Some(groups.clone()));
        let dims = plain.threat_dimensions();
        assert_eq!(plain.threat_group_count(), 1);
        assert_eq!(grouped.threat_group_count(), 8);
        let aux: Vec<usize> = grouped.auxiliary_sparse_dims().into_iter().map(|(_, d, _)| d).collect();
        assert_eq!(aux, vec![8 * dims, 8 * dims]);

        let offset = groups[ShogiProgressKPAbs.bucket(&pos) as usize] as usize * dims;
        for name in [THREAT_FT_SPARSE_STM, THREAT_FT_SPARSE_NSTM] {
            let mut a = vec![-1i32; MAX_ACTIVE_THREAT_FEATURES];
            let mut b = vec![-1i32; MAX_ACTIVE_THREAT_FEATURES];
            let na = plain.fill_auxiliary_sparse(name, &pos, &mut a);
            let nb = grouped.fill_auxiliary_sparse(name, &pos, &mut b);
            assert!(na > 0, "{name}: threat 特徴が空");
            assert_eq!(na, nb);
            for i in 0..na {
                assert_eq!(b[i] as usize, a[i] as usize + offset, "{name}: {i} 番目の index がずれていない");
            }
        }
    }

    /// 2026-11-17 K256 案 A: 追加列の疎入力は「群化した主入力の index - 群 offset」と同じ列で、 次元は 1 群分。
    /// `threat_ext = 0` (既定) では aux 入力の並びが従来と同一。
    #[test]
    fn threat_ext_sparse_is_ungrouped_copy_of_threat_ft_sparse() {
        const RECORD: [u8; 40] = [
            0x8d, 0x9c, 0x91, 0x14, 0x0c, 0x24, 0xfc, 0xf0, 0x4a, 0x01, 0x02, 0xbe, 0x9d, 0x67, 0x3a, 0x1e, 0x90, 0x02,
            0xac, 0x29, 0x8e, 0x87, 0x5f, 0x52, 0x2c, 0x23, 0x29, 0x18, 0x1e, 0x87, 0x64, 0xad, 0x9e, 0xff, 0x00, 0x00,
            0x45, 0x00, 0x01, 0x00,
        ];
        let mut pos = PackedSfenValue::default();
        pos.as_bytes_mut().copy_from_slice(&RECORD);
        let groups = vec![0u8, 1, 1, 2, 2, 3, 3, 3];
        let mk = || {
            ShogiHalfKaHmThreat::new(ThreatProfile::Full)
                .with_threat_ft_sparse(true)
                .with_threat_bucket_groups(Some(groups.clone()))
                .with_threat_grad_mask(Some(vec![(600, 1.0), (1000, 0.5), (i32::MAX, 0.25)]), 64)
        };
        let base = mk();
        let ext = mk().with_threat_ext(192);
        let dims = base.threat_dimensions();

        // 既存の aux 入力は並び・次元とも不変で、 末尾に追加列の 2 本が付く。
        let a_base = base.auxiliary_sparse_dims();
        let a_ext = ext.auxiliary_sparse_dims();
        assert_eq!(&a_ext[..a_base.len()], &a_base[..]);
        assert_eq!(
            &a_ext[a_base.len()..],
            &[
                (THREAT_EXT_SPARSE_STM.to_string(), dims, MAX_ACTIVE_THREAT_FEATURES),
                (THREAT_EXT_SPARSE_NSTM.to_string(), dims, MAX_ACTIVE_THREAT_FEATURES)
            ]
        );
        let d_base = base.auxiliary_dense_dims();
        let d_ext = ext.auxiliary_dense_dims();
        assert_eq!(d_base, vec![(THREAT_GMASK_NAME.to_string(), 64)]);
        assert_eq!(d_ext, vec![(THREAT_GMASK_NAME.to_string(), 64), (THREAT_EXT_GMASK_NAME.to_string(), 192)]);

        let offset = groups[ShogiProgressKPAbs.bucket(&pos) as usize] as usize * dims;
        for (main, extn) in
            [(THREAT_FT_SPARSE_STM, THREAT_EXT_SPARSE_STM), (THREAT_FT_SPARSE_NSTM, THREAT_EXT_SPARSE_NSTM)]
        {
            let mut a = vec![-1i32; MAX_ACTIVE_THREAT_FEATURES];
            let mut b = vec![-1i32; MAX_ACTIVE_THREAT_FEATURES];
            let na = ext.fill_auxiliary_sparse(main, &pos, &mut a);
            let nb = ext.fill_auxiliary_sparse(extn, &pos, &mut b);
            assert!(na > 0, "{main}: threat 特徴が空");
            assert_eq!(na, nb);
            for i in 0..na {
                assert_eq!(a[i] as usize, b[i] as usize + offset, "{extn}: {i} 番目");
                assert!((b[i] as usize) < dims);
            }
        }
        let mut m = vec![0f32; 192];
        ext.fill_auxiliary_dense(THREAT_EXT_GMASK_NAME, &pos, &mut m);
        let mut m0 = vec![0f32; 64];
        ext.fill_auxiliary_dense(THREAT_GMASK_NAME, &pos, &mut m0);
        assert!(m.iter().all(|&v| v == m0[0]));
    }

    #[test]
    fn test_map_features_startpos() {
        // 初期局面を手動で構築
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8), // 5九
            white_king_sq: Square::new(4, 0), // 5一
            ..Default::default()
        };

        // 玉
        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        // 先手の歩 (7段)
        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
        }
        // 後手の歩 (3段)
        for file in 0..9u8 {
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        // 角・飛
        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop); // 8八角
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook); // 2八飛
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop); // 2二角
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook); // 8二飛

        // 香
        board.board[Square::new(0, 8).index()] = Piece::new(Color::Black, PieceType::Lance); // 1九香
        board.board[Square::new(8, 8).index()] = Piece::new(Color::Black, PieceType::Lance); // 9九香
        board.board[Square::new(0, 0).index()] = Piece::new(Color::White, PieceType::Lance); // 1一香
        board.board[Square::new(8, 0).index()] = Piece::new(Color::White, PieceType::Lance); // 9一香

        // 桂
        board.board[Square::new(1, 8).index()] = Piece::new(Color::Black, PieceType::Knight); // 2九桂
        board.board[Square::new(7, 8).index()] = Piece::new(Color::Black, PieceType::Knight); // 8九桂
        board.board[Square::new(1, 0).index()] = Piece::new(Color::White, PieceType::Knight); // 2一桂
        board.board[Square::new(7, 0).index()] = Piece::new(Color::White, PieceType::Knight); // 8一桂

        // 銀
        board.board[Square::new(2, 8).index()] = Piece::new(Color::Black, PieceType::Silver); // 3九銀
        board.board[Square::new(6, 8).index()] = Piece::new(Color::Black, PieceType::Silver); // 7九銀
        board.board[Square::new(2, 0).index()] = Piece::new(Color::White, PieceType::Silver); // 3一銀
        board.board[Square::new(6, 0).index()] = Piece::new(Color::White, PieceType::Silver); // 7一銀

        // 金
        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold); // 4九金
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold); // 6九金
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold); // 4一金
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold); // 6一金

        let mut halfka_count = 0usize;
        let mut threat_count = 0usize;

        let input = full_input();
        let total_dims = input.num_inputs();
        input.map_threat_features(&board, |stm_idx, nstm_idx| {
            if stm_idx < HALFKA_HM_DIMENSIONS {
                halfka_count += 1;
            } else {
                threat_count += 1;
                assert!(stm_idx < total_dims, "stm threat index out of range: {stm_idx}");
            }
            if nstm_idx < HALFKA_HM_DIMENSIONS {
                // HalfKA 部分
            } else {
                assert!(nstm_idx < total_dims, "nstm threat index out of range: {nstm_idx}");
            }
        });

        // HalfKA: 38駒(王以外) + 2(両玉) = 40
        assert_eq!(halfka_count, 40);

        // Threat: 初期局面では threat pair が存在する（歩 vs 歩の対面等）
        assert!(threat_count > 0, "threat features should be non-empty in startpos");
    }

    #[test]
    fn test_map_features_sq_nb_guard() {
        // 片玉データはスキップされる
        let board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::NONE,
            ..Default::default()
        };

        let mut count = 0;
        let input = full_input();
        input.map_threat_features(&board, |_, _| count += 1);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_shorthand() {
        let input = full_input();
        let expected = format!("shogi-{}x45hm+threat", input.num_inputs());
        assert_eq!(input.shorthand(), expected);
    }

    #[test]
    fn test_normalize_sq() {
        // Black perspective, no mirror: そのまま
        let sq = Square::new(4, 4); // 5五
        assert_eq!(normalize_sq(sq, Color::Black, false), sq);

        // Black perspective, mirror: file 反転
        assert_eq!(normalize_sq(sq, Color::Black, true), sq.mirror_file());

        // White perspective, no mirror: inverse
        assert_eq!(normalize_sq(sq, Color::White, false), sq.inverse());

        // White perspective, mirror: inverse + mirror
        assert_eq!(normalize_sq(sq, Color::White, true), sq.inverse().mirror_file());
    }

    /// Canonical test vector: rshogi の threat_features.rs と同一の初期局面 threat index を検証
    /// この値が変わったら rshogi 側も同時に更新すること。
    /// Profile 0 (full) のみ有効（他の profile では index が変わる）。
    #[test]
    #[cfg(not(any(feature = "threat-profile-same-class", feature = "threat-profile-same-class-major-pawn",)))]
    fn test_canonical_startpos_threat_indices() {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };

        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop);
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop);
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook);

        board.board[Square::new(0, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(8, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(0, 0).index()] = Piece::new(Color::White, PieceType::Lance);
        board.board[Square::new(8, 0).index()] = Piece::new(Color::White, PieceType::Lance);

        board.board[Square::new(1, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(7, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(1, 0).index()] = Piece::new(Color::White, PieceType::Knight);
        board.board[Square::new(7, 0).index()] = Piece::new(Color::White, PieceType::Knight);

        board.board[Square::new(2, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(6, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(2, 0).index()] = Piece::new(Color::White, PieceType::Silver);
        board.board[Square::new(6, 0).index()] = Piece::new(Color::White, PieceType::Silver);

        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold);

        let mut stm_threat_indices = Vec::new();
        let input = full_input();
        input.map_threat_features(&board, |stm_idx, _nstm_idx| {
            if stm_idx >= HALFKA_HM_DIMENSIONS {
                stm_threat_indices.push(stm_idx - HALFKA_HM_DIMENSIONS);
            }
        });
        stm_threat_indices.sort();

        #[rustfmt::skip]
        let expected: Vec<usize> = vec![
            1330, 1618, 7147, 7148, 7231, 7232, 11047, 11213,
            16475, 16578, 23268, 23270, 24087, 25717, 37487, 40080,
            43974, 112573, 112861, 116503, 116504, 116587, 116588,
            122160, 122650, 128533, 128636, 138321, 138323, 139136,
            140770, 158280, 160871, 164753,
        ];

        assert_eq!(
            stm_threat_indices,
            expected,
            "Canonical mismatch with rshogi! count: bullet={} vs expected={}",
            stm_threat_indices.len(),
            expected.len()
        );
    }

    /// Canonical test vector (Profile 3 = no-pawn-target): 平手初期局面の threat index。
    /// C++ / rshogi 側 no-pawn-target 実装との bit-exact 照合用。
    #[test]
    fn test_canonical_startpos_threat_indices_no_pawn_target() {
        let mut board = ShogiBoard {
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 0),
            ..Default::default()
        };

        board.board[board.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[board.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);

        for file in 0..9u8 {
            board.board[Square::new(file, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            board.board[Square::new(file, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }

        board.board[Square::new(7, 7).index()] = Piece::new(Color::Black, PieceType::Bishop);
        board.board[Square::new(1, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        board.board[Square::new(1, 1).index()] = Piece::new(Color::White, PieceType::Bishop);
        board.board[Square::new(7, 1).index()] = Piece::new(Color::White, PieceType::Rook);

        board.board[Square::new(0, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(8, 8).index()] = Piece::new(Color::Black, PieceType::Lance);
        board.board[Square::new(0, 0).index()] = Piece::new(Color::White, PieceType::Lance);
        board.board[Square::new(8, 0).index()] = Piece::new(Color::White, PieceType::Lance);

        board.board[Square::new(1, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(7, 8).index()] = Piece::new(Color::Black, PieceType::Knight);
        board.board[Square::new(1, 0).index()] = Piece::new(Color::White, PieceType::Knight);
        board.board[Square::new(7, 0).index()] = Piece::new(Color::White, PieceType::Knight);

        board.board[Square::new(2, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(6, 8).index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[Square::new(2, 0).index()] = Piece::new(Color::White, PieceType::Silver);
        board.board[Square::new(6, 0).index()] = Piece::new(Color::White, PieceType::Silver);

        board.board[Square::new(3, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(5, 8).index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[Square::new(3, 0).index()] = Piece::new(Color::White, PieceType::Gold);
        board.board[Square::new(5, 0).index()] = Piece::new(Color::White, PieceType::Gold);

        let mut stm_threat_indices = Vec::new();
        let input = ShogiHalfKaHmThreat::new(ThreatProfile::NoPawnTarget);
        input.map_threat_features(&board, |stm_idx, _nstm_idx| {
            if stm_idx >= HALFKA_HM_DIMENSIONS {
                stm_threat_indices.push(stm_idx - HALFKA_HM_DIMENSIONS);
            }
        });
        stm_threat_indices.sort();

        #[rustfmt::skip]
        let expected: Vec<usize> = vec![
            9703, 9869, 14387, 14490, 20767, 22397, 34648, 38542,
            108448, 108938, 113989, 114092, 122960, 124594, 142103, 145985,
        ];

        assert_eq!(
            stm_threat_indices,
            expected,
            "Canonical mismatch (no-pawn-target)! count: bullet={} vs expected={}",
            stm_threat_indices.len(),
            expected.len()
        );
    }
}

// =============================================================================
// Profile 13: HalfKA_hm main input + compact factorized Threat auxiliary input
// =============================================================================

pub const THREAT_FACTOR_DIM: usize = 16;
pub const THREAT_FACTOR_SPARSE_NAME_STM: &str = "threat_factor_stm";
pub const THREAT_FACTOR_SPARSE_NAME_NSTM: &str = "threat_factor_nstm";

#[derive(Clone, Debug)]
pub struct ShogiHalfKaHmThreatFactor16 {
    threat: ShogiHalfKaHmThreat,
}

impl ShogiHalfKaHmThreatFactor16 {
    pub fn new() -> Self {
        Self { threat: ShogiHalfKaHmThreat::new(ThreatProfile::EnemyOnlyFactor16) }
    }

    /// Delegates Dual PSQT auxiliary sparse configuration to the base Threat input.
    /// Profile 13 keeps the factorized Threat rows in the main input but must also
    /// supply these four heads whenever the Dual PSQT graph branch is enabled.
    pub fn with_dual_psqt(mut self, enabled: bool, king_attacker_enabled: bool, cross_defended_enabled: bool) -> Self {
        self.threat = self.threat.with_dual_psqt(enabled, king_attacker_enabled, cross_defended_enabled);
        self
    }

    pub fn threat_dimensions(&self) -> usize {
        self.threat.threat_dimensions()
    }
}

impl Default for ShogiHalfKaHmThreatFactor16 {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseInputType for ShogiHalfKaHmThreatFactor16 {
    type RequiredDataType = PackedSfenValue;

    // A single main sparse input appends EnemyOnly Threat rows after HalfKA_hm.
    // This avoids a separate sparse batch and feeds factor_code through the same batch.
    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS + self.threat.threat_dimensions()
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_TOTAL
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.threat.map_features(pos, f);
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.threat.map_features_split(pos, f);
    }

    fn shorthand(&self) -> String {
        format!("shogi-{}x45hm+threat-factor16-main", HALFKA_HM_DIMENSIONS + self.threat.threat_dimensions())
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({}) + EnemyOnly Threat main input ({} rows) for a {}D factor branch",
            HALFKA_HM_DIMENSIONS,
            self.threat.threat_dimensions(),
            THREAT_FACTOR_DIM
        )
    }

    fn auxiliary_sparse_dims(&self) -> Vec<(String, usize, usize)> {
        self.threat.auxiliary_sparse_dims()
    }

    fn fill_auxiliary_sparse(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [i32]) -> usize {
        self.threat.fill_auxiliary_sparse(name, pos, out)
    }
}
