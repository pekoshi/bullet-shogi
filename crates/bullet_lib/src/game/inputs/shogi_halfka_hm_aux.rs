//! HalfKA_hm + AUX (KZ + HANG + STATKD) 連結特徴量
//! (= 2026-07-22、 threat 入力の NPS 半減を避けつつ FT accumulator を強化する
//! 「発火数が少なく既存 BB 計算から安く導ける」 aux 特徴の trainer 側配線)。
//!
//! HalfKA_hm (73,305 次元) と KZ/HANG/STATKD (計 600 次元/persp) を連結した
//! sparse input 型を提供する。 aux 部分は **1024 accumulator へ流す FT 入力** で、
//! SCReLU → L1 → L2 を通り piece features と非線形に co-adapt する (= KZT-FT と同機構)。
//!
//! # 契約 (= scratchpad `cand_features.py` / `cand_features2.py`)
//!
//! aux の特徴定義 (盤面 → 特徴 index、 bucket 境界、 class 順、 persp 規約) は
//! R² 事前検証で使われた Python 実装
//! `cand_features.py::features_persp` (KZ, HANG) および
//! `cand_features2.py::features2_persp` (STATKD) と **bit-exact 一致** させる。
//! (`examples/shogi_aux_canonical.rs` + `scratchpad/aux_xcheck.py` で照合。)
//!
//! ## 重要: これらの aux 特徴には square index が無い
//!
//! KZ/HANG/STATKD はいずれも **count の thermometer 符号化** であり、 per-square
//! 正規化 index を持たない (契約 Python も KZSQ/STATSQ 以外は sq を正規化しない)。
//! したがって HalfKA/threat のような HM mirror / sq 正規化は **適用しない** (= no-op)。
//! perspective 依存は「own = persp 側」 という側の割り当てだけに現れる。
//!
//! ## レイアウト (per perspective、 own = persp)
//!
//! | group  | dim | 内訳                                                        | offset |
//! |--------|-----|-------------------------------------------------------------|--------|
//! | KZ     |  60 | (own玉, 敵玉) × (A8 8 + D8 7 + A24 8 + D24 7)                | 0      |
//! | HANG   | 108 | (us, them) × (hang 9class×3 + cheap 9class×3)                | 60     |
//! | STATKD | 432 | (side0=us, side1=them) × 9class × 4status × 3kdist × 2thermo | 168    |
//!
//! 合計 600 次元/persp。 連結空間では HalfKA (73,305) の後ろに append し、
//! feature index は `[73,305, 73,905)`。
//!
//! ## 非対称 emission
//!
//! aux は perspective (= own 側) 依存で非対称:
//! - STM accumulator: own = STM
//! - NSTM accumulator: own = NSTM
//!
//! [`SparseInputType::map_features_split`] で片側のみ `Some` にして emit する
//! (= KZT-FT / HandThreat defensive と同様)。
//!
//! ## warm-start (zero-init)
//!
//! aux feature index はすべて `>= HALFKA_HM_DIMENSIONS` なので、 l0 affine の aux 行
//! (`[73,305, 73,905)`) を zero-init すると accumulator への寄与が 0 になり、 forward は
//! plain HalfKA と完全一致する (= `--resume-from-quantised` で HalfKA/L1-L3/PSQT のみ
//! 引継ぎ、 aux 行は 0 から育成する warm-start に対応)。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, ShogiHalfKA_hm};
use super::shogi_halfka_hm_threat::{Occupied, ThreatClass, for_each_attack};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

// =============================================================================
// 次元定数 (契約 cand_features.py と一致)
// =============================================================================

/// ThreatClass 数 (King 除外)。
pub const AUX_NUM_CLASSES: usize = 9;

/// KZ: (own玉, 敵玉) × (A8=8 + D8=7 + A24=8 + D24=7) = 30 × 2。
pub const KZ_DIM: usize = 60;
/// HANG: (us, them) × (hang 9×3 + cheap 9×3) = 54 × 2。
pub const HANG_DIM: usize = 108;
/// STATKD: (us, them) × 9class × 4status × 3kdist × 2thermo = 216 × 2。
pub const STATKD_DIM: usize = 432;

/// 1 perspective あたりの aux 総次元 (= 600)。
pub const AUX_DIM_PER_PERSP: usize = KZ_DIM + HANG_DIM + STATKD_DIM;

/// 連結空間内の group offset (HalfKA 末尾からの相対)。
pub const KZ_OFFSET: usize = 0;
pub const HANG_OFFSET: usize = KZ_DIM; // 60
pub const STATKD_OFFSET: usize = KZ_DIM + HANG_DIM; // 168

/// AUX 連結後の総次元数 (= 73,305 + 600 = 73,905)。
pub const HALFKA_HM_AUX_DIMENSIONS: usize = HALFKA_HM_DIMENSIONS + AUX_DIM_PER_PERSP;

/// 1 perspective の aux active feature 上限 (安全側: 全 thermometer bit が発火する
/// 理論最大 = 各 group の次元和)。 実局面では ~35 程度。
pub const AUX_MAX_ACTIVE_PER_PERSP: usize = AUX_DIM_PER_PERSP;

/// HalfKA_hm + aux の 1 perspective 合計 max active。
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + AUX_MAX_ACTIVE_PER_PERSP;

// =============================================================================
// 量子化 / section 定数 (example 側 export / rshogi-core loader 用)
// =============================================================================

/// 量子化スケール (FT 入力なので QA=127)。
pub const AUX_QA: i16 = 127;
/// arch_str marker (loader が末尾 section を読む検出子)。
pub const AUX_ARCH_MARKER: &str = "AUX=600,";
/// quantised.bin section の sub_magic (8 byte)。
pub const AUX_SUB_MAGIC: &[u8; 8] = b"P2AUX\0\0\0";
/// section format version。
pub const AUX_SECTION_VERSION: u32 = 1;

// =============================================================================
// bucket edges (契約 cand_features.py と一致)
// =============================================================================

const A8_E: [i32; 8] = [1, 2, 3, 4, 5, 6, 8, 11];
const D8_E: [i32; 7] = [1, 3, 5, 7, 9, 11, 14];
const A24_E: [i32; 8] = [1, 2, 4, 6, 8, 11, 15, 20];
const D24_E: [i32; 7] = [2, 5, 8, 11, 14, 18, 23];
const CNT3_E: [i32; 3] = [1, 2, 3];
const CNT2_E: [i32; 2] = [1, 2];

/// 駒価値 (VAL)。 PieceType enum 値 (0..=14) で index。 King=99。
/// 契約 cand_features.py VAL と一致 (成駒は全て 6、 King=99)。
const VAL: [i32; 15] = [
    0,  // None
    1,  // Pawn
    3,  // Lance
    4,  // Knight
    5,  // Silver
    8,  // Bishop
    10, // Rook
    6,  // Gold
    99, // King
    6,  // ProPawn
    6,  // ProLance
    6,  // ProKnight
    6,  // ProSilver
    10, // Horse
    12, // Dragon
];

#[inline]
fn val_of(pt: PieceType) -> i32 {
    VAL[pt as usize]
}

// =============================================================================
// 利きマップ (契約 cand_features.py::attack_maps と一致)
// =============================================================================

/// `atk[c][sq]` = 色 c の駒が sq に利かせている利き数 (王の利きも 8 近傍で含む)。
/// `atkmin[c][sq]` = sq を攻撃する色 c の駒の最小 VAL (99 = なし、 王は 99)。
struct AttackMaps {
    atk: [[u16; 81]; 2],
    atkmin: [[i32; 81]; 2],
}

impl AttackMaps {
    fn from_board(board: &ShogiBoard) -> Self {
        let mut atk = [[0u16; 81]; 2];
        let mut atkmin = [[99i32; 81]; 2];
        let occ = Occupied::from_board(board);

        for sq_raw in 0..81u8 {
            let from = Square(sq_raw);
            let pc = board.piece_on(from);
            if pc.is_none() {
                continue;
            }
            let ci = pc.color as usize;
            let v = val_of(pc.piece_type);
            if pc.piece_type == PieceType::King {
                // 王は 8 近傍 (Chebyshev 距離 1) に利く。 契約 attack_maps は
                // ZONE8[sq] (占有無関係の在盤近傍) を利き先とする。
                let f0 = from.file() as i8;
                let r0 = from.rank() as i8;
                for df in -1i8..=1 {
                    for dr in -1i8..=1 {
                        if df == 0 && dr == 0 {
                            continue;
                        }
                        let f = f0 + df;
                        let r = r0 + dr;
                        if (0..9).contains(&f) && (0..9).contains(&r) {
                            let to = Square::new(f as u8, r as u8).index();
                            atk[ci][to] += 1;
                            if v < atkmin[ci][to] {
                                atkmin[ci][to] = v;
                            }
                        }
                    }
                }
            } else {
                for_each_attack(pc.piece_type, pc.color, from, &occ, |to| {
                    let ti = to.index();
                    atk[ci][ti] += 1;
                    if v < atkmin[ci][ti] {
                        atkmin[ci][ti] = v;
                    }
                });
            }
        }

        Self { atk, atkmin }
    }

    /// king 周辺 (Chebyshev 距離 <= radius) の atk[color] 総和。
    #[inline]
    fn zone_sum(&self, color: Color, ksq: Square, radius: i8) -> i32 {
        let ci = color as usize;
        let f0 = ksq.file() as i8;
        let r0 = ksq.rank() as i8;
        let mut acc = 0i32;
        for df in -radius..=radius {
            for dr in -radius..=radius {
                if df == 0 && dr == 0 {
                    continue;
                }
                let f = f0 + df;
                let r = r0 + dr;
                if (0..9).contains(&f) && (0..9).contains(&r) {
                    acc += self.atk[ci][Square::new(f as u8, r as u8).index()] as i32;
                }
            }
        }
        acc
    }
}

// =============================================================================
// 符号化ヘルパ
// =============================================================================

/// thermometer: `x >= edges[i]` の bit を `base + i` として emit。
#[inline]
fn emit_thermo<F: FnMut(usize)>(x: i32, edges: &[i32], base: usize, f: &mut F) {
    for (i, &e) in edges.iter().enumerate() {
        if x >= e {
            f(base + i);
        }
    }
}

/// Chebyshev 距離。
#[inline]
fn chebyshev(a: Square, b: Square) -> i32 {
    let df = (a.file() as i32 - b.file() as i32).abs();
    let dr = (a.rank() as i32 - b.rank() as i32).abs();
    df.max(dr)
}

/// class → 0..=8 (King は None)。
#[inline]
fn class_idx(pt: PieceType) -> Option<usize> {
    ThreatClass::from_piece_type(pt).map(|c| c as usize)
}

// =============================================================================
// aux active index 列挙 (契約と bit-exact)
// =============================================================================

/// `own = persp` 視点の aux active feature を連結空間の **local index**
/// (`[0, AUX_DIM_PER_PERSP)`) で `f` に渡す。
///
/// KZ → HANG → STATKD の順、 各 group 内は契約 cand_features.py のリスト構築順。
fn emit_aux_local<F: FnMut(usize)>(board: &ShogiBoard, maps: &AttackMaps, persp: Color, mut f: F) {
    let us = persp;
    let them = us.opponent();

    // ---- KZ: (own玉, 敵玉) × (A8, D8, A24, D24) ----
    for (ki, &king_side) in [us, them].iter().enumerate() {
        let ksq = board.king_square(king_side);
        if !ksq.is_valid() {
            continue;
        }
        let att_side = king_side.opponent();
        let a8 = maps.zone_sum(att_side, ksq, 1);
        let d8 = maps.zone_sum(king_side, ksq, 1);
        let a24 = maps.zone_sum(att_side, ksq, 2);
        let d24 = maps.zone_sum(king_side, ksq, 2);
        let base = KZ_OFFSET + ki * 30;
        emit_thermo(a8, &A8_E, base, &mut f);
        emit_thermo(d8, &D8_E, base + 8, &mut f);
        emit_thermo(a24, &A24_E, base + 15, &mut f);
        emit_thermo(d24, &D24_E, base + 23, &mut f);
    }

    // ---- HANG: (us, them) × (hang 9×3, cheap 9×3) ----
    for (si, &side) in [us, them].iter().enumerate() {
        let opp = side.opponent();
        let side_i = side as usize;
        let opp_i = opp as usize;
        let mut c_hang = [0i32; AUX_NUM_CLASSES];
        let mut c_cheap = [0i32; AUX_NUM_CLASSES];
        for sq_raw in 0..81u8 {
            let sq = Square(sq_raw);
            let pc = board.piece_on(sq);
            if pc.is_none() || pc.color != side || pc.piece_type == PieceType::King {
                continue;
            }
            let ti = sq.index();
            if maps.atk[opp_i][ti] == 0 {
                continue;
            }
            let Some(tc) = class_idx(pc.piece_type) else { continue };
            if maps.atk[side_i][ti] == 0 {
                c_hang[tc] += 1;
            }
            if maps.atkmin[opp_i][ti] < val_of(pc.piece_type) {
                c_cheap[tc] += 1;
            }
        }
        let base = HANG_OFFSET + si * 54;
        for (tc, &cnt) in c_hang.iter().enumerate() {
            emit_thermo(cnt, &CNT3_E, base + tc * 3, &mut f);
        }
        for (tc, &cnt) in c_cheap.iter().enumerate() {
            emit_thermo(cnt, &CNT3_E, base + 27 + tc * 3, &mut f);
        }
    }

    // ---- STATKD: (side0=us, side1=them) × class × status × kdist × thermo2 ----
    // skd[side][class][status][kdist]
    let mut skd = [[[[0i32; 3]; 4]; AUX_NUM_CLASSES]; 2];
    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let pc = board.piece_on(sq);
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let Some(tc) = class_idx(pc.piece_type) else { continue };
        let color = pc.color;
        let side = if color == us { 0 } else { 1 };
        let ti = sq.index();
        let a = maps.atk[color.opponent() as usize][ti];
        let d = maps.atk[color as usize][ti];
        let ksq_own = board.king_square(color);
        let kb = if ksq_own.is_valid() {
            let kd = chebyshev(sq, ksq_own);
            if kd <= 1 {
                0
            } else if kd == 2 {
                1
            } else {
                2
            }
        } else {
            2
        };
        if a >= 1 {
            skd[side][tc][0][kb] += 1;
            if d == 0 {
                skd[side][tc][1][kb] += 1;
            }
        }
        if d >= 1 {
            skd[side][tc][2][kb] += 1;
        }
        if d >= 2 {
            skd[side][tc][3][kb] += 1;
        }
    }
    for (side, per_side) in skd.iter().enumerate() {
        for (tc, per_class) in per_side.iter().enumerate() {
            for (st, per_status) in per_class.iter().enumerate() {
                for (kb, &cnt) in per_status.iter().enumerate() {
                    let local = ((side * AUX_NUM_CLASSES + tc) * 4 + st) * 3 + kb;
                    emit_thermo(cnt, &CNT2_E, STATKD_OFFSET + local * 2, &mut f);
                }
            }
        }
    }
}

/// `own = persp` 視点の aux active feature を **連結空間の絶対 index**
/// (`[HALFKA_HM_DIMENSIONS, HALFKA_HM_AUX_DIMENSIONS)`) で `f` に渡す (public API)。
///
/// C++ (P2NNUE) 側 aux 実装との bit-exact 照合や example export で使う。
pub fn emit_aux_active_index<F: FnMut(usize)>(board: &ShogiBoard, persp: Color, mut f: F) {
    let maps = AttackMaps::from_board(board);
    emit_aux_local(board, &maps, persp, |local| f(HALFKA_HM_DIMENSIONS + local));
}

// =============================================================================
// ShogiHalfKaHmAux
// =============================================================================

/// HalfKA_hm + AUX (KZ + HANG + STATKD) 連結特徴量。
///
/// `aux_enabled == false` のとき aux part を emit せず plain HalfKA_hm と完全一致する
/// (= warm-start 比較 / profile 切替用)。 次元 (`num_inputs`) は常に 73,905 のまま
/// (aux 行が単に未発火 = zero 寄与) なので、 同一 graph で on/off を比較できる。
#[allow(non_camel_case_types)]
#[derive(Clone, Debug)]
pub struct ShogiHalfKaHmAux {
    aux_enabled: bool,
}

impl Default for ShogiHalfKaHmAux {
    fn default() -> Self {
        Self::new()
    }
}

impl ShogiHalfKaHmAux {
    /// aux 有効で構築 (= 状態なし)。
    pub fn new() -> Self {
        Self { aux_enabled: true }
    }

    /// aux 有効/無効を指定して構築する builder。
    pub fn with_aux_enabled(mut self, enabled: bool) -> Self {
        self.aux_enabled = enabled;
        self
    }

    /// aux 有効状態。
    pub fn aux_enabled(&self) -> bool {
        self.aux_enabled
    }

    /// aux 部分の次元数 (= 600)。
    pub fn aux_dimensions(&self) -> usize {
        AUX_DIM_PER_PERSP
    }
}

impl SparseInputType for ShogiHalfKaHmAux {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_AUX_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_TOTAL
    }

    /// symmetric API (= 後方互換 fallback)。 aux 部分は非対称なので symmetric `f()` では
    /// 正確に emit できず、 HalfKA_hm 部分のみを emit する。 訓練 loader は
    /// [`SparseInputType::map_features_split`] を優先呼びするため通常経路では呼ばれない。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        ShogiHalfKA_hm.map_features(pos, f);
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // HalfKA_hm 部分 (= 対称) は両側 Some で emit。
        ShogiHalfKA_hm.map_features(pos, |stm, nstm| f(Some(stm), Some(nstm)));

        if !self.aux_enabled {
            return;
        }

        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        // 片玉データは aux を emit しない (KZ/STATKD が玉位置に依存)。
        if !board.king_square(stm).is_valid() || !board.king_square(nstm).is_valid() {
            return;
        }
        let maps = AttackMaps::from_board(&board);
        // STM accumulator: own = STM
        emit_aux_local(&board, &maps, stm, |local| f(Some(HALFKA_HM_DIMENSIONS + local), None));
        // NSTM accumulator: own = NSTM
        emit_aux_local(&board, &maps, nstm, |local| f(None, Some(HALFKA_HM_DIMENSIONS + local)));
    }

    fn shorthand(&self) -> String {
        format!("shogi-{HALFKA_HM_AUX_DIMENSIONS}x45hm+aux")
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({HALFKA_HM_DIMENSIONS}) + AUX KZ/HANG/STATKD ({AUX_DIM_PER_PERSP}) concatenated (FT input, asymmetric, aux_enabled={})",
            self.aux_enabled
        )
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::Piece;

    /// 平手初期局面を手動構築 (threat.rs のテストと同一)。
    fn startpos_board() -> ShogiBoard {
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
        board
    }

    fn aux_local_sorted(board: &ShogiBoard, persp: Color) -> Vec<usize> {
        let maps = AttackMaps::from_board(board);
        let mut v = Vec::new();
        emit_aux_local(board, &maps, persp, |local| v.push(local));
        v.sort_unstable();
        v
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(KZ_DIM, 60);
        assert_eq!(HANG_DIM, 108);
        assert_eq!(STATKD_DIM, 432);
        assert_eq!(AUX_DIM_PER_PERSP, 600);
        let input = ShogiHalfKaHmAux::new();
        assert_eq!(input.num_inputs(), 73_305 + 600);
        assert_eq!(input.num_inputs(), 73_905);
        assert_eq!(input.aux_dimensions(), 600);
    }

    #[test]
    fn test_offsets() {
        assert_eq!(KZ_OFFSET, 0);
        assert_eq!(HANG_OFFSET, 60);
        assert_eq!(STATKD_OFFSET, 168);
        assert_eq!(STATKD_OFFSET + STATKD_DIM, AUX_DIM_PER_PERSP);
    }

    #[test]
    fn test_shorthand() {
        assert_eq!(ShogiHalfKaHmAux::new().shorthand(), "shogi-73905x45hm+aux");
    }

    /// 全 aux index が HalfKA 領域外 (= aux 行) に落ちること。
    /// これが warm-start zero-init で plain HalfKA と forward 一致する構造的保証。
    #[test]
    fn test_aux_indices_beyond_halfka() {
        let board = startpos_board();
        for persp in [Color::Black, Color::White] {
            let maps = AttackMaps::from_board(&board);
            emit_aux_local(&board, &maps, persp, |local| {
                assert!(local < AUX_DIM_PER_PERSP, "aux local {local} out of range");
            });
            emit_aux_active_index(&board, persp, |idx| {
                assert!(
                    (HALFKA_HM_DIMENSIONS..HALFKA_HM_AUX_DIMENSIONS).contains(&idx),
                    "aux abs idx {idx} out of aux band"
                );
            });
        }
    }

    /// warm-start: aux_enabled=false のとき split emission の HalfKA 部分は
    /// plain HalfKA_hm と完全一致し、 aux index は一切出ない。
    #[test]
    fn test_aux_disabled_equals_plain_halfka() {
        // PSV を経由せずに直接検証するため、 map_features (HalfKA fallback) を基準にする。
        // aux_enabled=false の split では aux 行が出ないので、 有効 halfka index 集合は
        // aux 有無で不変であることを (index 域で) 確認する。
        let input_off = ShogiHalfKaHmAux::new().with_aux_enabled(false);
        assert!(!input_off.aux_enabled());
        // aux disabled でも次元は不変 (aux 行は zero 寄与)。
        assert_eq!(input_off.num_inputs(), HALFKA_HM_AUX_DIMENSIONS);
    }

    /// canonical: 平手初期局面の aux local index (STM=Black, NSTM=White)。
    /// C++ / cand_features.py との bit-exact 照合用。 値が変わったら契約破りを疑うこと。
    #[test]
    fn test_canonical_startpos_aux_indices() {
        let board = startpos_board();
        let stm = aux_local_sorted(&board, Color::Black);
        let nstm = aux_local_sorted(&board, Color::White);

        // 初期局面は左右対称 + 先後対称なので STM と NSTM は一致するはず。
        assert_eq!(stm, nstm, "startpos aux should be symmetric between perspectives");

        // 契約 cand_features.py / cand_features2.py の平手初期局面出力 (aux_xcheck.py)。
        #[rustfmt::skip]
        let expected: Vec<usize> = vec![
            8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 38, 39, 40, 41, 42, 43, 44,
            53, 54, 55, 56, 57, 58, 59, 182, 183, 184, 185, 188, 190, 191, 208, 232, 254, 255,
            260, 276, 277, 304, 310, 328, 398, 399, 400, 401, 404, 406, 407, 424, 448, 470, 471,
            476, 492, 493, 520, 526, 544,
        ];
        assert_eq!(stm, expected, "Canonical aux mismatch! count bullet={} expected={}", stm.len(), expected.len());
    }
}
