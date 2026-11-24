//! HalfKA_hm + AUXCHEAP (KZ2x + HG6 + OUT6 + STATKD-kd2) 連結特徴量 (2026-07-23)。
//!
//! threat 全列挙 (NPS 半減) を避けた「安いスキャン」aux 特徴の trainer 側配線。
//! C++ 推論側は board_effect (LONG_EFFECT 差分利き) の SIMD 抽出で **PSQT-only**
//! に寄与する (FT accumulator へは流さない)。 trainer 側もそれを鏡像で再現する:
//!
//! - **psqtw の aux 行 [73,305, 73,697) を訓練** (= aux の唯一の寄与経路)
//! - **l0w の aux 行は run_with_zeroed_rows で常時 0 に凍結** (= FT へは流さない)
//!
//! # 契約 (= C++ 実装が正)
//!
//! 特徴定義 (発火 index、 bucket 境界、 グループ分け、 persp 規約) は
//! `E:/shogi/p2shogi_cplus/source/eval/p2nnue/p2nnue_auxcheap.h`
//! (`AuxCheap::build_raw` + `AuxCheap::for_each`) と **bit-exact 一致** させる。
//! 照合は USI `p2auxcheap [sfen]` の dump (STM/NSTM 発火 index) と
//! `examples/shogi_auxcheap_canonical.rs` の出力比較で行う。
//! (定義補助: `scratchpad/cand_features4.py` / `scratchpad/statkd_bbverify.py`。)
//!
//! ## レイアウト (per perspective、 own = persp、 C++ と同一)
//!
//! | group      | dim | 内訳                                                           | offset |
//! |------------|-----|----------------------------------------------------------------|--------|
//! | KZ2x       |  70 | king(own, enemy) × cross-cell(ta 6level × td 6level, 非発火除く 35) | 0      |
//! | HG6        |  16 | side(us, them) × {hang(g0+g1) oh3, hang(g2) oh3, hang(g3) oh2} | 70     |
//! | OUT6       |  18 | side(us, them) × {out(g0+g1) oh4, out(g2) oh3, out(g3) oh2}    | 86     |
//! | STATKD-kd2 |  72 | side(us, them) × 9class × 2status(att,def1) × 2kdist × count-lin | 104    |
//!
//! グループ: g0=歩, g1=香桂, g2=金銀成駒, g3=飛角馬龍。
//! hang = a>0 && d==0、 out = a>d (a=敵利き数, d=自利き数、 玉は集計対象外)。
//! STATKD-kd2 は **count-linear 符号化** (2026-07-25): 各 cell(side,tc,st',kb) は
//! 生カウント v を持ち、 発火 index を **v 回 emit** (multiplicity 加算) する。
//! C1 削減 (status4→2 = hang/def2 drop): 次元 144→72、 aux ΔR²(C0-C1)=+0.0023、 実 NPS ~+5%。
//! 合計 176 次元/persp。 連結空間では HalfKA (73,305) の後ろに append し、
//! feature index は `[73,305, 73,481)`。
//!
//! ## 非対称 emission
//!
//! aux は perspective (= own 側) 依存で非対称。 [`SparseInputType::map_features_split`]
//! で STM accumulator へ own=STM、 NSTM accumulator へ own=NSTM を emit する
//! (= ShogiHalfKaHmAux / KZT-FT と同機構)。
//!
//! ## warm-start / PSQT-only 訓練 (zero-init)
//!
//! aux feature index はすべて `>= HALFKA_HM_DIMENSIONS` なので、 warm-start
//! (`--resume-from-quantised`) で l0w / psqtw の aux 行を zero-init すると step0
//! forward は plain HalfKA と完全一致する。 その後 l0w aux 行のみ
//! `run_with_zeroed_rows` で毎 step 0 固定し、 psqtw aux 行だけが学習される。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, ShogiHalfKA_hm};
use super::shogi_halfka_hm_threat::{Occupied, ThreatClass, for_each_attack};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

// =============================================================================
// 次元定数 (契約 p2nnue_auxcheap.h と一致)
// =============================================================================

/// クラス数 (King 除外、 = C++ SKD_NTC)。
pub const AUXCHEAP_NUM_CLASSES: usize = 9;
/// STATKD status 数 (C1: att, def1。 旧 status4 {att,hang,def1,def2} から hang/def2 を drop)。
pub const AUXCHEAP_NUM_STATUS: usize = 2;
/// STATKD kdist bucket 数 (Chebyshev <=2, >=3)。
pub const AUXCHEAP_NUM_KDIST: usize = 2;

/// KZ2x: king(own, enemy) × cross-cell 35 = 70。
pub const AUXCHEAP_KZ_DIM: usize = 70;
/// HG6: side(us, them) × (3 + 3 + 2) = 16。
pub const AUXCHEAP_HG_DIM: usize = 16;
/// OUT6: side(us, them) × (4 + 3 + 2) = 18。
pub const AUXCHEAP_OUT_DIM: usize = 18;
/// STATKD-kd2 C1: side(us, them) × 9class × 2status(att,def1) × 2kdist × count-linear = 72。
/// (2026-07-25: status4→2 削減 = hang/def2 drop、 aux ΔR²(C0-C1)=+0.0023、 実 NPS ~+5%)
pub const AUXCHEAP_STATKD_DIM: usize = 2 * AUXCHEAP_NUM_CLASSES * AUXCHEAP_NUM_STATUS * AUXCHEAP_NUM_KDIST;

/// group offset (連結空間内、 HalfKA 末尾からの相対。 C++ KZ_OFF/HG_OFF/OUT_OFF/STATKD_OFF)。
pub const AUXCHEAP_KZ_OFFSET: usize = 0;
pub const AUXCHEAP_HG_OFFSET: usize = AUXCHEAP_KZ_DIM; // 70
pub const AUXCHEAP_OUT_OFFSET: usize = AUXCHEAP_KZ_DIM + AUXCHEAP_HG_DIM; // 86
pub const AUXCHEAP_STATKD_OFFSET: usize = AUXCHEAP_KZ_DIM + AUXCHEAP_HG_DIM + AUXCHEAP_OUT_DIM; // 104

/// 1 perspective あたりの auxcheap 総次元 (= 176、 C++ DIM_PER_PERSP、 C1 status4→2)。
pub const AUXCHEAP_DIM_PER_PERSP: usize = AUXCHEAP_STATKD_OFFSET + AUXCHEAP_STATKD_DIM;

/// AUXCHEAP 連結後の総次元数 (= 73,305 + 176 = 73,481)。
pub const HALFKA_HM_AUXCHEAP_DIMENSIONS: usize = HALFKA_HM_DIMENSIONS + AUXCHEAP_DIM_PER_PERSP;

/// 1 perspective の auxcheap active feature 上限 (安全側 = 全次元。 実局面では ~30)。
pub const AUXCHEAP_MAX_ACTIVE_PER_PERSP: usize = AUXCHEAP_DIM_PER_PERSP;

/// HalfKA_hm + auxcheap の 1 perspective 合計 max active。
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + AUXCHEAP_MAX_ACTIVE_PER_PERSP;

// =============================================================================
// section 定数 (example 側 export / C++ loader 用)
// =============================================================================

/// arch_str marker (C++ loader が末尾 P2AUXC section を読む検出子)。
pub const AUXCHEAP_ARCH_MARKER: &str = "AUXCHEAP=176,";
/// quantised.bin section の sub_magic (8 byte)。
pub const AUXCHEAP_SUB_MAGIC: &[u8; 8] = b"P2AUXC\0\0";
/// section format version。
pub const AUXCHEAP_SECTION_VERSION: u32 = 1;

// =============================================================================
// 量子化 edge (契約 p2nnue_auxcheap.h TA_E/TD_E/OH3_E/OH4_E/CNT2_E と一致)
// =============================================================================

const TA_E: [i32; 5] = [4, 8, 12, 18, 26];
const TD_E: [i32; 5] = [4, 8, 12, 16, 22];
const OH3_E: [i32; 3] = [1, 2, 3];
const OH4_E: [i32; 4] = [1, 2, 3, 4];
const CNT2_E: [i32; 2] = [1, 2];

// =============================================================================
// 利き数マップ (= C++ board_effect[c].e[sq] と同値の fresh 構築)
// =============================================================================

/// `atk[c][sq]` = 色 c の駒が sq に利かせている利き数 (玉の 8 近傍利きも含む)。
/// C++ 側は LONG_EFFECT の差分 board_effect を読むが、 値は全駒の effects_from の
/// 和と同値 (long_effect.cpp init: 全駒について effects_from を加算) なので
/// fresh 構築で bit-exact 一致する。
struct EffectMaps {
    atk: [[u16; 81]; 2],
}

impl EffectMaps {
    fn from_board(board: &ShogiBoard) -> Self {
        let mut atk = [[0u16; 81]; 2];
        let occ = Occupied::from_board(board);

        for sq_raw in 0..81u8 {
            let from = Square(sq_raw);
            let pc = board.piece_on(from);
            if pc.is_none() {
                continue;
            }
            let ci = pc.color as usize;
            if pc.piece_type == PieceType::King {
                // 玉は 8 近傍 (Chebyshev 距離 1) に利く (占有無関係)。
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
                            atk[ci][Square::new(f as u8, r as u8).index()] += 1;
                        }
                    }
                }
            } else {
                for_each_attack(pc.piece_type, pc.color, from, &occ, |to| {
                    atk[ci][to.index()] += 1;
                });
            }
        }

        Self { atk }
    }

    /// king 周辺 (Chebyshev 距離 <= radius、 玉マス除く) の atk[color] 総和 (= C++ zone_sum)。
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

    /// zone8 (玉 8 近傍、 在盤のみ) で atk[color] >= 2 のマス数
    /// (= C++ around8_greater_than_one & board_mask の popcount)。
    #[inline]
    fn zone8_ge2_count(&self, color: Color, ksq: Square) -> i32 {
        let ci = color as usize;
        let f0 = ksq.file() as i8;
        let r0 = ksq.rank() as i8;
        let mut cnt = 0i32;
        for df in -1i8..=1 {
            for dr in -1i8..=1 {
                if df == 0 && dr == 0 {
                    continue;
                }
                let f = f0 + df;
                let r = r0 + dr;
                if (0..9).contains(&f)
                    && (0..9).contains(&r)
                    && self.atk[ci][Square::new(f as u8, r as u8).index()] >= 2
                {
                    cnt += 1;
                }
            }
        }
        cnt
    }
}

// =============================================================================
// 符号化ヘルパ (契約 p2nnue_auxcheap.h emit_oh と一致)
// =============================================================================

/// one-hot(level): level = count(v >= edges[i])。 level > 0 なら base + (level-1) を emit。
#[inline]
fn emit_oh<F: FnMut(usize)>(v: i32, edges: &[i32], base: usize, f: &mut F) {
    let level = edges.iter().filter(|&&e| v >= e).count();
    if level > 0 {
        f(base + level - 1);
    }
}

/// Chebyshev 距離。
#[inline]
fn chebyshev(a: Square, b: Square) -> i32 {
    let df = (a.file() as i32 - b.file() as i32).abs();
    let dr = (a.rank() as i32 - b.rank() as i32).abs();
    df.max(dr)
}

/// HG/OUT 用グループ (= C++ GRP)。 King / None は None。
#[inline]
fn grp_of(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Lance | PieceType::Knight => Some(1),
        PieceType::Silver
        | PieceType::Gold
        | PieceType::ProPawn
        | PieceType::ProLance
        | PieceType::ProKnight
        | PieceType::ProSilver => Some(2),
        PieceType::Bishop | PieceType::Rook | PieceType::Horse | PieceType::Dragon => Some(3),
        PieceType::King | PieceType::None => None,
    }
}

// =============================================================================
// Raw 集計 (= C++ build_raw、 視点非依存の絶対色量)
// =============================================================================

/// board_effect 由来の視点非依存 raw 量 (= C++ AuxCheap::Raw)。
struct AuxCheapRaw {
    /// KZ: 玉 (index = king color) の zone 利き量。
    a8: [i32; 2],
    d8: [i32; 2],
    a24: [i32; 2],
    d24: [i32; 2],
    ge2: [i32; 2],
    /// HG/OUT: side (絶対色) ごとのグループ別カウント。
    hang: [[i32; 4]; 2],
    outc: [[i32; 4]; 2],
    /// STATKD-kd2: 絶対色カウント cnt[color][tc][st][kb]。
    statkd: [[[[i32; AUXCHEAP_NUM_KDIST]; AUXCHEAP_NUM_STATUS]; AUXCHEAP_NUM_CLASSES]; 2],
}

fn build_raw(board: &ShogiBoard, maps: &EffectMaps) -> AuxCheapRaw {
    let mut r = AuxCheapRaw {
        a8: [0; 2],
        d8: [0; 2],
        a24: [0; 2],
        d24: [0; 2],
        ge2: [0; 2],
        hang: [[0; 4]; 2],
        outc: [[0; 4]; 2],
        statkd: [[[[0; AUXCHEAP_NUM_KDIST]; AUXCHEAP_NUM_STATUS]; AUXCHEAP_NUM_CLASSES]; 2],
    };

    // ---- KZ zone 量 (玉ごと) ----
    for (kci, &kc) in [Color::Black, Color::White].iter().enumerate() {
        let ksq = board.king_square(kc);
        let att = kc.opponent();
        r.a8[kci] = maps.zone_sum(att, ksq, 1);
        r.d8[kci] = maps.zone_sum(kc, ksq, 1);
        r.a24[kci] = maps.zone_sum(att, ksq, 2);
        r.d24[kci] = maps.zone_sum(kc, ksq, 2);
        r.ge2[kci] = maps.zone8_ge2_count(att, ksq);
    }

    // ---- HG / OUT / STATKD (盤上駒の単一走査、 玉は対象外) ----
    for sq_raw in 0..81u8 {
        let sq = Square(sq_raw);
        let pc = board.piece_on(sq);
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let c = pc.color;
        let ci = c as usize;
        let ti = sq.index();
        let a = maps.atk[c.opponent() as usize][ti] as i32;
        let d = maps.atk[ci][ti] as i32;

        // HG/OUT グループカウント (hang = a>0 && d==0、 out = a>d)。
        if let Some(g) = grp_of(pc.piece_type) {
            if a > 0 && d == 0 {
                r.hang[ci][g] += 1;
            }
            if a > d {
                r.outc[ci][g] += 1;
            }
        }

        // STATKD-kd2 C1 (att / def1 × kdist。 hang/def2 は drop = status4→2)。
        if let Some(tc) = ThreatClass::from_piece_type(pc.piece_type) {
            let tc = tc as usize;
            let kb = if chebyshev(sq, board.king_square(c)) <= 2 { 0 } else { 1 };
            if a >= 1 {
                r.statkd[ci][tc][0][kb] += 1; // st'0 = att (a>=1)
            }
            if d >= 1 {
                r.statkd[ci][tc][1][kb] += 1; // st'1 = def1 (d>=1)
            }
        }
    }

    r
}

// =============================================================================
// active index 列挙 (= C++ for_each、 契約と bit-exact)
// =============================================================================

/// `own = persp` 視点の auxcheap active feature を連結空間の **local index**
/// (`[0, AUXCHEAP_DIM_PER_PERSP)`) で `f` に渡す。
fn emit_auxcheap_local<F: FnMut(usize)>(raw: &AuxCheapRaw, persp: Color, mut f: F) {
    let us = persp as usize;
    let them = persp.opponent() as usize;
    let sides = [us, them];

    // ---- KZ2x: king(own=us, enemy=them) ----
    for (ki, &kc) in sides.iter().enumerate() {
        let ta = 2 * raw.a8[kc] + raw.a24[kc] + 2 * raw.ge2[kc];
        let td = 2 * raw.d8[kc] + raw.d24[kc];
        let la = TA_E.iter().filter(|&&e| ta >= e).count();
        let lb = TD_E.iter().filter(|&&e| td >= e).count();
        let cell = la * 6 + lb; // 0..35
        if cell > 0 {
            f(AUXCHEAP_KZ_OFFSET + ki * 35 + (cell - 1));
        }
    }

    // ---- HG6: side(us, them) ----
    for (si, &s) in sides.iter().enumerate() {
        let base = AUXCHEAP_HG_OFFSET + si * 8;
        emit_oh(raw.hang[s][0] + raw.hang[s][1], &OH3_E, base, &mut f);
        emit_oh(raw.hang[s][2], &OH3_E, base + 3, &mut f);
        emit_oh(raw.hang[s][3], &CNT2_E, base + 6, &mut f);
    }

    // ---- OUT6: side(us, them) ----
    for (si, &s) in sides.iter().enumerate() {
        let base = AUXCHEAP_OUT_OFFSET + si * 9;
        emit_oh(raw.outc[s][0] + raw.outc[s][1], &OH4_E, base, &mut f);
        emit_oh(raw.outc[s][2], &OH3_E, base + 4, &mut f);
        emit_oh(raw.outc[s][3], &CNT2_E, base + 7, &mut f);
    }

    // ---- STATKD-kd2 C1: side(us, them) × 9tc × 2st'(att,def1) × 2kb × count-linear ----
    // (2026-07-25) 各 cell の生カウント v を multiplicity で emit (v 回)。
    // 訓練 backend が重複 index を加算 (scatter-add) するため、 cell の実効寄与 = v × w[cell]。
    // C++ 推論は per-piece LUT 2 加算で同値 (bit-exact)。 局面あたり総 emit 数は
    // Σ(piece,status firing) ≤ ~110 で AUXCHEAP_MAX_ACTIVE_PER_PERSP (248) 未満。
    for (si, &c) in sides.iter().enumerate() {
        let sbase = AUXCHEAP_STATKD_OFFSET + si * AUXCHEAP_NUM_CLASSES * AUXCHEAP_NUM_STATUS * AUXCHEAP_NUM_KDIST;
        for tc in 0..AUXCHEAP_NUM_CLASSES {
            for st in 0..AUXCHEAP_NUM_STATUS {
                for kb in 0..AUXCHEAP_NUM_KDIST {
                    let v = raw.statkd[c][tc][st][kb];
                    let cell = sbase + (tc * AUXCHEAP_NUM_STATUS + st) * AUXCHEAP_NUM_KDIST + kb;
                    for _ in 0..v {
                        f(cell); // count-linear: multiplicity
                    }
                }
            }
        }
    }
}

/// `own = persp` 視点の auxcheap active feature を **連結空間の絶対 index**
/// (`[HALFKA_HM_DIMENSIONS, HALFKA_HM_AUXCHEAP_DIMENSIONS)`) で `f` に渡す (public API)。
///
/// C++ (P2NNUE AuxCheap) の USI `p2auxcheap` dump との bit-exact 照合や
/// example export で使う。
pub fn emit_auxcheap_active_index<F: FnMut(usize)>(board: &ShogiBoard, persp: Color, mut f: F) {
    let maps = EffectMaps::from_board(board);
    let raw = build_raw(board, &maps);
    emit_auxcheap_local(&raw, persp, |local| f(HALFKA_HM_DIMENSIONS + local));
}

// =============================================================================
// ShogiHalfKaHmAuxCheap
// =============================================================================

/// HalfKA_hm + AUXCHEAP (KZ2x + HG6 + OUT6 + STATKD-kd2) 連結特徴量。
///
/// aux 部分は PSQT-only 訓練用 (psqtw aux 行のみ学習、 l0w aux 行は呼び出し側が
/// `run_with_zeroed_rows` で 0 固定する)。
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHmAuxCheap;

impl ShogiHalfKaHmAuxCheap {
    /// 状態なしで構築。
    pub fn new() -> Self {
        Self
    }

    /// aux 部分の次元数 (= 176、 C1 status4→2)。
    pub fn aux_dimensions(&self) -> usize {
        AUXCHEAP_DIM_PER_PERSP
    }
}

impl SparseInputType for ShogiHalfKaHmAuxCheap {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_AUXCHEAP_DIMENSIONS
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

        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        // 片玉データは aux を emit しない (KZ/STATKD が玉位置に依存)。
        if !board.king_square(stm).is_valid() || !board.king_square(nstm).is_valid() {
            return;
        }
        let maps = EffectMaps::from_board(&board);
        let raw = build_raw(&board, &maps);
        // STM accumulator: own = STM
        emit_auxcheap_local(&raw, stm, |local| f(Some(HALFKA_HM_DIMENSIONS + local), None));
        // NSTM accumulator: own = NSTM
        emit_auxcheap_local(&raw, nstm, |local| f(None, Some(HALFKA_HM_DIMENSIONS + local)));
    }

    fn shorthand(&self) -> String {
        format!("shogi-{HALFKA_HM_AUXCHEAP_DIMENSIONS}x45hm+auxcheap")
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({HALFKA_HM_DIMENSIONS}) + AUXCHEAP KZ2x/HG6/OUT6/STATKD-kd2 ({AUXCHEAP_DIM_PER_PERSP}) concatenated (PSQT-only, asymmetric)"
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

    /// 平手初期局面を手動構築 (shogi_halfka_hm_aux のテストと同一)。
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

    fn auxcheap_local_sorted(board: &ShogiBoard, persp: Color) -> Vec<usize> {
        let mut v = Vec::new();
        emit_auxcheap_active_index(board, persp, |idx| v.push(idx - HALFKA_HM_DIMENSIONS));
        v.sort_unstable();
        v
    }

    #[test]
    fn test_dimensions() {
        assert_eq!(AUXCHEAP_KZ_DIM, 70);
        assert_eq!(AUXCHEAP_HG_DIM, 16);
        assert_eq!(AUXCHEAP_OUT_DIM, 18);
        assert_eq!(AUXCHEAP_STATKD_DIM, 72);
        assert_eq!(AUXCHEAP_DIM_PER_PERSP, 176);
        assert_eq!(AUXCHEAP_HG_OFFSET, 70);
        assert_eq!(AUXCHEAP_OUT_OFFSET, 86);
        assert_eq!(AUXCHEAP_STATKD_OFFSET, 104);
        let input = ShogiHalfKaHmAuxCheap::new();
        assert_eq!(input.num_inputs(), 73_305 + 176);
        assert_eq!(input.num_inputs(), 73_481);
        assert_eq!(input.aux_dimensions(), 176);
    }

    #[test]
    fn test_arch_marker_consistency() {
        assert_eq!(AUXCHEAP_ARCH_MARKER, format!("AUXCHEAP={AUXCHEAP_DIM_PER_PERSP},"));
        assert_eq!(AUXCHEAP_SUB_MAGIC.len(), 8);
    }

    /// 全 auxcheap index が HalfKA 領域外 (= aux 行) に落ちること。
    /// これが warm-start zero-init で plain HalfKA と forward 一致する構造的保証。
    #[test]
    fn test_auxcheap_indices_beyond_halfka() {
        let board = startpos_board();
        for persp in [Color::Black, Color::White] {
            emit_auxcheap_active_index(&board, persp, |idx| {
                assert!(
                    (HALFKA_HM_DIMENSIONS..HALFKA_HM_AUXCHEAP_DIMENSIONS).contains(&idx),
                    "auxcheap abs idx {idx} out of aux band"
                );
            });
        }
    }

    /// canonical: 平手初期局面の auxcheap local index の構造的性質 (count-linear、 2026-07-25)。
    /// count-linear では STATKD が multiplicity emit のため index が重複しうる。
    /// 厳密な bit-exact 値照合は C++ `p2auxcheap` dump ↔ `examples/shogi_auxcheap_canonical`
    /// の外部比較で行う (startpos/mid1/mid2 の 3 局面)。 ここでは符号化不変な構造を検証:
    ///   (1) 先後対称局面ゆえ STM と NSTM の multiset が一致
    ///   (2) 全 index が連結空間 aux 帯 [0, DIM_PER_PERSP) 内
    ///   (3) KZ/HG/OUT (local < STATKD_OFF) は one-hot ゆえ重複しない
    #[test]
    fn test_canonical_startpos_auxcheap_structure() {
        let board = startpos_board();
        let stm = auxcheap_local_sorted(&board, Color::Black);
        let nstm = auxcheap_local_sorted(&board, Color::White);

        // (1) 初期局面は左右対称 + 先後対称なので STM と NSTM の multiset は一致する。
        assert_eq!(stm, nstm, "startpos auxcheap should be symmetric between perspectives");

        // (2) 全 index が aux 帯内。
        for &i in &stm {
            assert!(i < AUXCHEAP_DIM_PER_PERSP, "local {i} out of aux band {AUXCHEAP_DIM_PER_PERSP}");
        }

        // (3) KZ/HG/OUT 部 (local < STATKD_OFFSET) は one-hot ゆえ distinct。
        let kzhgout: Vec<usize> = stm.iter().copied().filter(|&i| i < AUXCHEAP_STATKD_OFFSET).collect();
        let mut uniq = kzhgout.clone();
        uniq.dedup();
        assert_eq!(kzhgout, uniq, "KZ/HG/OUT part must be one-hot (no duplicate index)");

        // STATKD 部は count-linear ゆえ重複可。 少なくとも 1 個は発火する (初期局面)。
        assert!(stm.iter().any(|&i| i >= AUXCHEAP_STATKD_OFFSET), "startpos should fire STATKD");
    }
}
