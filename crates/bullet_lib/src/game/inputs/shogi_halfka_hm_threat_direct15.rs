//! Profile 15: HalfKA_hm + 10-class one-sided ThreatDirect sparse input.
//!
//! The Direct evaluator computes, per perspective, enemy-attacker -> friendly-target
//! relations and subtracts the two perspective sums.  This input therefore overrides
//! `map_features_split` so the two active sets may have different lengths.

use super::SparseInputType;
use super::shogi_halfka::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, ShogiHalfKA_hm, is_hm_mirror};
use super::shogi_halfka_hm_threat::{Occupied, for_each_attack, normalize_sq};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

/// P2THRD2 / Profile 15 direct dimensions: 10 target classes times the sum of
/// all 10 attacker-class empty-board attack counts.
pub const THREAT_DIRECT15_DIMENSIONS: usize = 65_640;
pub const THREAT_DIRECT15_CLASS_NB: usize = 10;
pub const THREAT_DIRECT15_KING_CLASS: usize = 9;
pub const MAX_ACTIVE_DIRECT15_FEATURES: usize = 360;
/// P5 (`--direct15-defended`): 被攻撃駒が守られているか (味方の実利きが 1 つ以上あるか)
/// で次元を x2 にした Direct15 の次元数。 index は `[守備なし DIM | 守備あり DIM]`
/// (C++ `threat_direct15_value_defended` と同じ並び)。
pub const THREAT_DIRECT15_DEFENDED_DIMENSIONS: usize = 2 * THREAT_DIRECT15_DIMENSIONS;

/// Direct15 専用 sparse 入力の名前 (手番側 / 非手番側)。
///
/// 2026-09-01: 旧実装は Direct15 の 65,640 次元を HalfKA_hm(73,305) の後ろへ
/// append した「tail 特徴」として主 sparse 入力に流していた。 しかし学習器では
/// index >= 73,305 の tail 特徴に forward/backward のどちらも到達せず
/// (`l0w` / `psqtw` / `threat_direct_w` のいずれも tail が恒久的に 0)、
/// Direct15 head が一度も学習されないという不具合になっていた。
/// 実績のある `auxiliary_sparse_dims` 方式 (Dual PSQT / PairShape と同じ経路) の
/// 専用 sparse 入力へ移行する。 書き出す P2THRD2 payload の形式は不変。
pub const THREAT_DIRECT15_SPARSE_STM: &str = "threat_direct15_stm";
/// [`THREAT_DIRECT15_SPARSE_STM`] の非手番側。
pub const THREAT_DIRECT15_SPARSE_NSTM: &str = "threat_direct15_nstm";

/// attacker class ごとの空盤面利き数 (全 81 升合計)。 C++ `Direct15::ATTACKS_PER_COLOR` と同一。
pub const THREAT_DIRECT15_ATTACKS_PER_CLASS: [usize; THREAT_DIRECT15_CLASS_NB] =
    [72, 324, 112, 328, 416, 816, 1296, 1104, 1552, 544];
const ATTACKS_PER_CLASS: [usize; THREAT_DIRECT15_CLASS_NB] = THREAT_DIRECT15_ATTACKS_PER_CLASS;

/// Direct15 特徴 index (0-based、 defended 半分は `feature % DIM` で畳んでから渡す) が属する
/// (attacker class, target class) の pair id (`ac * 10 + dc`) を返す。
/// P4 factorizer (`--direct15-factorise`) の書き出し畳み込みで使う。
pub fn direct15_pair_of_feature(feature: usize) -> usize {
    let local = feature % THREAT_DIRECT15_DIMENSIONS;
    let mut base = 0usize;
    for ac in 0..THREAT_DIRECT15_CLASS_NB {
        let block = THREAT_DIRECT15_CLASS_NB * ATTACKS_PER_CLASS[ac];
        if local < base + block {
            let dc = (local - base) / ATTACKS_PER_CLASS[ac];
            return ac * THREAT_DIRECT15_CLASS_NB + dc;
        }
        base += block;
    }
    unreachable!("Direct15 feature index out of range: {feature}")
}

/// Direct15 head (`threat_direct_w` / `threat_direct_b_cancels`) の graph 上の形状。
///
/// graph の `threat_direct_w` は column-major `Shape(rows, dims)` で、 要素 (row, feature) は
/// `feature * rows + row` に置かれる (書き出し側 `SavedFormat` の P2THRD2 変換や
/// `load_p2thrd2_direct_if_present` と同じ規約)。
/// - 行: `persp == 2` (P2 `--direct15-persp-split`) なら `2 * bucket + 視点` (視点 0 = stm)、
///   `persp == 1` なら `bucket`。
/// - 列: `halves == 2` (P5 `--direct15-defended`) なら `守備ビット * DIM + local`、
///   `halves == 1` なら `local` (`DIM` = [`THREAT_DIRECT15_DIMENSIONS`])。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Direct15HeadShape {
    /// 出力 bucket 数 (学習器の `NUM_BUCKETS`)。
    pub buckets: usize,
    /// 視点テーブル数 (1 = 両視点共通、 2 = P2 persp split)。
    pub persp: usize,
    /// 守備ビット分割数 (1 = 従来、 2 = P5 defended)。
    pub halves: usize,
}

impl Direct15HeadShape {
    /// 学習器のフラグから形状を作る。
    pub fn new(buckets: usize, persp_split: bool, defended: bool) -> Self {
        Self { buckets, persp: if persp_split { 2 } else { 1 }, halves: if defended { 2 } else { 1 } }
    }

    /// `threat_direct_w` の行数 (= `threat_direct_b_cancels` の要素数)。
    pub fn rows(self) -> usize {
        self.persp * self.buckets
    }

    /// `threat_direct_w` の列数 (= Direct15 sparse 入力の次元)。
    pub fn dims(self) -> usize {
        self.halves * THREAT_DIRECT15_DIMENSIONS
    }

    /// `threat_direct_w` の要素数。
    pub fn weight_len(self) -> usize {
        self.rows() * self.dims()
    }

    /// checkpoint 中の `threat_direct_w` / `threat_direct_b_cancels` の要素数から形状を推定する。
    ///
    /// 要素数 1,050,240 は `[8][131,280]` (P5) と `[16][65,640]` (P2) で一致し weight 単独では
    /// 判別できないため、 bias の要素数 (= rows) で行数を確定してから列数を割り出す。
    /// 既知の 4 形状 (rows ∈ {buckets, 2*buckets}、 dims ∈ {DIM, 2*DIM}) 以外は `None`。
    pub fn infer(buckets: usize, weight_len: usize, bias_len: usize) -> Option<Self> {
        let persp = match bias_len {
            r if r == buckets => 1,
            r if r == 2 * buckets => 2,
            _ => return None,
        };
        let rows = persp * buckets;
        if rows == 0 || weight_len % rows != 0 {
            return None;
        }
        let halves = match weight_len / rows {
            d if d == THREAT_DIRECT15_DIMENSIONS => 1,
            d if d == THREAT_DIRECT15_DEFENDED_DIMENSIONS => 2,
            _ => return None,
        };
        Some(Self { buckets, persp, halves })
    }

    /// (bucket, 視点) が置かれる行。 `persp == 1` のときは視点を無視する。
    fn row(self, bucket: usize, persp: usize) -> usize {
        if self.persp == 2 { 2 * bucket + persp } else { bucket }
    }

    /// `from` から `to` へ複製展開できるか (bucket 数が同じで、 視点・守備ビットとも縮小しない)。
    fn can_expand_to(self, to: Self) -> bool {
        self.buckets == to.buckets && self.persp <= to.persp && self.halves <= to.halves
    }
}

/// `threat_direct_w` (および同形の optimiser state) を `from` 形状から `to` 形状へ複製展開する。
///
/// 展開直後の forward 出力が展開前と同一になるように、 従来 1 テーブルだった軸を複製する:
/// - 視点 1 → 2 (P2): bucket `b` の行を stm 行 `2b` と nstm 行 `2b+1` の両方へ複製する。
///   両視点で同じ重みを引くので `Σ_stm − Σ_nstm` は不変。
/// - 守備ビット 1 → 2 (P5): local 特徴 `f` の列を `f` (守備なし) と `DIM + f` (守備あり) の
///   両方へ複製する。 どちらの守備ビットでも同じ重みを引くので出力は不変。
/// - 両方 (P5+P2): 上 2 つの合成。
///
/// 単純なバッファの繰り返し (tiling) は bucket ごとのブロック構造を跨いでレイアウトを壊すため、
/// 必ず `feature * rows + row` の column-major 規約で要素ごとに写像する。
/// `from == to` なら複製 (恒等)。 縮小 / bucket 数違い / 要素数不整合は `None`。
pub fn expand_direct15_head_weights(src: &[f32], from: Direct15HeadShape, to: Direct15HeadShape) -> Option<Vec<f32>> {
    if !from.can_expand_to(to) || src.len() != from.weight_len() {
        return None;
    }
    let (rows_from, rows_to) = (from.rows(), to.rows());
    let mut dst = vec![0.0f32; to.weight_len()];
    for feature_dst in 0..to.dims() {
        let half_dst = feature_dst / THREAT_DIRECT15_DIMENSIONS;
        let local = feature_dst % THREAT_DIRECT15_DIMENSIONS;
        let half_src = if from.halves == 2 { half_dst } else { 0 };
        let feature_src = half_src * THREAT_DIRECT15_DIMENSIONS + local;
        let col_src = &src[feature_src * rows_from..(feature_src + 1) * rows_from];
        let col_dst = &mut dst[feature_dst * rows_to..(feature_dst + 1) * rows_to];
        for (row_dst, cell) in col_dst.iter_mut().enumerate() {
            let (bucket, persp_dst) = if to.persp == 2 { (row_dst / 2, row_dst % 2) } else { (row_dst, 0) };
            let persp_src = if from.persp == 2 { persp_dst } else { 0 };
            *cell = col_src[from.row(bucket, persp_src)];
        }
    }
    Some(dst)
}

/// `threat_direct_b_cancels` (行ベクトル、 要素数 = rows) を `from` から `to` へ複製展開する。
///
/// 視点 1 → 2 では bucket `b` の値を行 `2b` / `2b+1` の両方へ複製する (stm − nstm で相殺するので
/// 出力不変)。 守備ビットは bias に無関係。 縮小 / 要素数不整合は `None`。
pub fn expand_direct15_head_bias(src: &[f32], from: Direct15HeadShape, to: Direct15HeadShape) -> Option<Vec<f32>> {
    if !from.can_expand_to(to) || src.len() != from.rows() {
        return None;
    }
    let dst = (0..to.rows())
        .map(|row_dst| {
            let (bucket, persp_dst) = if to.persp == 2 { (row_dst / 2, row_dst % 2) } else { (row_dst, 0) };
            let persp_src = if from.persp == 2 { persp_dst } else { 0 };
            src[from.row(bucket, persp_src)]
        })
        .collect();
    Some(dst)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHmThreatDirect15 {
    /// P5: 被攻撃駒が守られているかで特徴を分ける (次元 x2 = 131,280)。 default false。
    defended: bool,
}

#[inline]
fn class_from_piece_type(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Lance => Some(1),
        PieceType::Knight => Some(2),
        PieceType::Silver => Some(3),
        PieceType::Gold | PieceType::ProPawn | PieceType::ProLance | PieceType::ProKnight | PieceType::ProSilver => {
            Some(4)
        }
        PieceType::Bishop => Some(5),
        PieceType::Rook => Some(6),
        PieceType::Horse => Some(7),
        PieceType::Dragon => Some(8),
        PieceType::King => Some(THREAT_DIRECT15_KING_CLASS),
        PieceType::None => None,
    }
}

/// The C++ Profile 15 table groups rows by attacker class and then target class.
#[inline]
fn pair_base(attacker_class: usize, target_class: usize) -> usize {
    debug_assert!(attacker_class < THREAT_DIRECT15_CLASS_NB);
    debug_assert!(target_class < THREAT_DIRECT15_CLASS_NB);
    let base: usize = ATTACKS_PER_CLASS[..attacker_class].iter().map(|n| THREAT_DIRECT15_CLASS_NB * n).sum();
    base + target_class * ATTACKS_PER_CLASS[attacker_class]
}

/// Number of legal empty-board moves from all lower square IDs for one normalized
/// white-oriented attacker.  C++ sorts target square IDs before assigning the row.
#[inline]
fn row_offset(attacker_class: usize, from: Square, to: Square) -> usize {
    let from_raw = from.0 as usize;
    let to_raw = to.0;
    let mut offset = 0usize;
    for raw in 0..from_raw {
        let sq = Square(raw as u8);
        let count = attack_targets(attacker_class, sq, None, None, &mut []);
        offset += count;
    }
    let mut targets = [Square(0); 36];
    let count = attack_targets(attacker_class, from, Some(Color::White), None, &mut targets);
    for (ordinal, target) in targets[..count].iter().enumerate() {
        if target.0 == to_raw {
            return offset + ordinal;
        }
    }
    panic!("Profile15 Direct target is not in the normalized attack set");
}

/// Enumerate empty-board legal target squares in the same raw-square ascending order
/// as the C++ Direct15 table.  This does not inspect occupation; real attacks are
/// obtained separately through `for_each_attack`.
fn attack_targets(
    attacker_class: usize,
    from: Square,
    oriented: Option<Color>,
    _unused: Option<()>,
    out: &mut [Square],
) -> usize {
    let color = oriented.unwrap_or(Color::White);
    let file = (from.0 / 9) as i8;
    let rank = (from.0 % 9) as i8;
    let mut count = 0usize;
    let mut push = |f: i8, r: i8| {
        if (0..9).contains(&f) && (0..9).contains(&r) {
            if count < out.len() {
                out[count] = Square((f as u8) * 9 + r as u8);
            }
            count += 1;
        }
    };
    let fwd = if color == Color::Black { -1 } else { 1 };
    match attacker_class {
        0 => push(file, rank + fwd),
        1 => {
            let mut r = rank + fwd;
            while (0..9).contains(&r) {
                push(file, r);
                r += fwd;
            }
        }
        2 => {
            push(file - 1, rank + 2 * fwd);
            push(file + 1, rank + 2 * fwd);
        }
        3 => {
            for (df, dr) in [(-1, fwd), (0, fwd), (1, fwd), (-1, -fwd), (1, -fwd)] {
                push(file + df, rank + dr);
            }
        }
        4 => {
            for (df, dr) in [(-1, fwd), (0, fwd), (1, fwd), (-1, 0), (1, 0), (0, -fwd)] {
                push(file + df, rank + dr);
            }
        }
        5 | 7 => {
            for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    push(f, r);
                    f += df;
                    r += dr;
                }
            }
            if attacker_class == 7 {
                for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    push(file + df, rank + dr);
                }
            }
        }
        6 | 8 => {
            for (df, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let mut f = file + df;
                let mut r = rank + dr;
                while (0..9).contains(&f) && (0..9).contains(&r) {
                    push(f, r);
                    f += df;
                    r += dr;
                }
            }
            if attacker_class == 8 {
                for (df, dr) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    push(file + df, rank + dr);
                }
            }
        }
        THREAT_DIRECT15_KING_CLASS => {
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df != 0 || dr != 0 {
                        push(file + df, rank + dr);
                    }
                }
            }
        }
        _ => unreachable!(),
    }
    // C++ uses insertion sort by raw square ID.
    let stored = count.min(out.len());
    for i in 1..stored {
        let key = out[i];
        let mut j = i;
        while j > 0 && out[j - 1].0 > key.0 {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[j] = key;
    }
    count
}

/// 玉の利き (8 近傍)。 盤上の駒配置には依存しない。
///
/// 共有ヘルパ `for_each_attack` は legacy threat profile (King 除外) の呼び出し元が
/// King no-op に依存しているため変更できない。 Profile15 は玉を attacker class 9 と
/// して含める仕様 (C++ `threat_direct15_value` は玉を attackers から除外しない) なので、
/// Direct15 専用にここで列挙する。 生成集合は空盤面 King パターン
/// (`attack_targets` の class 9) と一致する (test で検証)。
#[inline]
fn for_each_king_attack<F: FnMut(Square)>(from: Square, mut callback: F) {
    let file = from.file() as i8;
    let rank = from.rank() as i8;
    for df in -1i8..=1 {
        for dr in -1i8..=1 {
            if df == 0 && dr == 0 {
                continue;
            }
            let f = file + df;
            let r = rank + dr;
            if (0..9).contains(&f) && (0..9).contains(&r) {
                callback(Square::new(f as u8, r as u8));
            }
        }
    }
}

impl ShogiHalfKaHmThreatDirect15 {
    pub const fn new() -> Self {
        Self { defended: false }
    }

    /// P5 (`--direct15-defended`) の on/off を指定する。
    pub const fn with_defended(self, defended: bool) -> Self {
        Self { defended }
    }

    pub const fn defended(&self) -> bool {
        self.defended
    }

    /// Direct15 専用 sparse 入力の次元 (65,640、 defended 時は 131,280)。
    pub const fn direct_dims(&self) -> usize {
        if self.defended { THREAT_DIRECT15_DEFENDED_DIMENSIONS } else { THREAT_DIRECT15_DIMENSIONS }
    }

    /// 指定 perspective (`for_stm` = true なら手番側) から見た
    /// 「敵 attacker → 自 target」の Direct15 特徴 index を **0-based**
    /// (HalfKA offset を含まない) で `out` 先頭から書き込み、 個数を返す。
    ///
    /// `out.len()` (= [`MAX_ACTIVE_DIRECT15_FEATURES`]) を超える分は捨てる。
    /// 容量は `profile15_real_psv_direct_activity_fits_capacity` で検証している。
    pub fn direct_indices(&self, pos: &PackedSfenValue, for_stm: bool, out: &mut [i32]) -> usize {
        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        // 旧 map_split と同じく、 どちらかの玉が無い局面では列挙しない。
        if !board.king_square(stm).is_valid() || !board.king_square(nstm).is_valid() {
            return 0;
        }
        let persp = if for_stm { stm } else { nstm };
        let hm = is_hm_mirror(board.king_square(persp), persp);
        let occ = Occupied::from_board(&board);
        let mut count = 0usize;

        // P5: persp 側の全駒の実利き (占有考慮、 歩・玉を含む) の和集合。 target がこの集合に
        // 入っていれば「守られている」として index を +DIM する。 利きの定義は下の pair 列挙と
        // 同じ関数 (for_each_attack / for_each_king_attack) なので C++ 側
        // (threat_direct15_value_defended の att_map) と一致する。
        let mut defended_map = [false; 81];
        if self.defended {
            for raw in 0..81u8 {
                let from = Square(raw);
                let piece = board.piece_on(from);
                if piece.is_none() || piece.color != persp {
                    continue;
                }
                let mark = |to: Square| defended_map[to.0 as usize] = true;
                if piece.piece_type == PieceType::King {
                    for_each_king_attack(from, mark);
                } else {
                    for_each_attack(piece.piece_type, piece.color, from, &occ, mark);
                }
            }
        }
        let defended_offset =
            |to: Square| -> usize { if defended_map[to.0 as usize] { THREAT_DIRECT15_DIMENSIONS } else { 0 } };

        for raw in 0..81u8 {
            let from = Square(raw);
            let attacker = board.piece_on(from);
            if attacker.is_none() {
                continue;
            }
            let Some(ac) = class_from_piece_type(attacker.piece_type) else {
                continue;
            };
            let attacker_color = attacker.color;
            if attacker_color == persp {
                continue;
            }
            let mut emit = |to: Square| {
                let target = board.piece_on(to);
                if target.is_none() || target.color != persp {
                    return;
                }
                let Some(dc) = class_from_piece_type(target.piece_type) else {
                    return;
                };
                if count >= out.len() {
                    return;
                }
                let from_n = normalize_sq(from, persp, hm);
                let to_n = normalize_sq(to, persp, hm);
                out[count] = (pair_base(ac, dc) + row_offset(ac, from_n, to_n) + defended_offset(to)) as i32;
                count += 1;
            };
            if attacker.piece_type == PieceType::King {
                // 2026-09-01 修正: 玉 attacker (class 9) は `for_each_attack` が
                // King no-op のため一度も列挙されず、 C++ 推論が参照する
                // attacker class 9 の全行 (544×10 dim/bucket) が学習で恒久 0 の
                // ままだった (推論値は 0 加算で一致するが、 King as tenth class が
                // attacker 側で機能しない)。 専用ヘルパで 8 近傍を列挙する。
                for_each_king_attack(from, &mut emit);
            } else {
                for_each_attack(attacker.piece_type, attacker_color, from, &occ, emit);
            }
        }

        count
    }
}

impl SparseInputType for ShogiHalfKaHmThreatDirect15 {
    type RequiredDataType = PackedSfenValue;
    /// 主 sparse 入力は HalfKA_hm のみ。 Direct15 は専用 sparse 入力
    /// ([`THREAT_DIRECT15_SPARSE_STM`] / [`THREAT_DIRECT15_SPARSE_NSTM`]) で流す。
    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS
    }
    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        ShogiHalfKA_hm.map_features(pos, f);
    }
    fn auxiliary_sparse_dims(&self) -> Vec<(String, usize, usize)> {
        vec![
            (THREAT_DIRECT15_SPARSE_STM.to_string(), self.direct_dims(), MAX_ACTIVE_DIRECT15_FEATURES),
            (THREAT_DIRECT15_SPARSE_NSTM.to_string(), self.direct_dims(), MAX_ACTIVE_DIRECT15_FEATURES),
        ]
    }
    fn fill_auxiliary_sparse(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [i32]) -> usize {
        match name {
            THREAT_DIRECT15_SPARSE_STM => self.direct_indices(pos, true, out),
            THREAT_DIRECT15_SPARSE_NSTM => self.direct_indices(pos, false, out),
            _ => 0,
        }
    }
    fn shorthand(&self) -> String {
        format!("shogi-{}x45hm+threatdirect15{}", self.num_inputs(), if self.defended { "def" } else { "" })
    }
    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({}) + one-sided 10-class ThreatDirect Profile 15 ({}{})",
            HALFKA_HM_DIMENSIONS,
            self.direct_dims(),
            if self.defended { ", defended bit" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile15_dimension_matches_contract() {
        assert_eq!(ATTACKS_PER_CLASS.iter().sum::<usize>() * THREAT_DIRECT15_CLASS_NB, THREAT_DIRECT15_DIMENSIONS);
    }

    /// `direct15_pair_of_feature` が pair_base の逆写像になっていること (defended 半分も同じ pair)。
    #[test]
    fn profile15_pair_of_feature_inverts_pair_base() {
        for ac in 0..THREAT_DIRECT15_CLASS_NB {
            for dc in 0..THREAT_DIRECT15_CLASS_NB {
                let base = pair_base(ac, dc);
                for r in [0usize, ATTACKS_PER_CLASS[ac] - 1] {
                    let f = base + r;
                    assert_eq!(direct15_pair_of_feature(f), ac * THREAT_DIRECT15_CLASS_NB + dc, "f={f}");
                    assert_eq!(
                        direct15_pair_of_feature(f + THREAT_DIRECT15_DIMENSIONS),
                        ac * THREAT_DIRECT15_CLASS_NB + dc
                    );
                }
            }
        }
    }

    /// C++ Direct15 LUT縺ｮ莉墓ｧ假ｼ・ttacker-major / target-minor縲仝hite蜷代″縺ｮ
    /// directional pattern縲〉aw-square譏・・ｼ峨→King隨ｬ10繧ｯ繝ｩ繧ｹ繧堤・蜷医☆繧九・
    #[test]
    fn profile15_cpp_lut_contract_including_king() {
        let mut expected_pair_base = 0usize;
        for attacker_class in 0..THREAT_DIRECT15_CLASS_NB {
            for target_class in 0..THREAT_DIRECT15_CLASS_NB {
                assert_eq!(pair_base(attacker_class, target_class), expected_pair_base);
                expected_pair_base += ATTACKS_PER_CLASS[attacker_class];
            }
        }
        assert_eq!(expected_pair_base, THREAT_DIRECT15_DIMENSIONS);
        assert_eq!(pair_base(THREAT_DIRECT15_KING_CLASS, THREAT_DIRECT15_KING_CLASS), 65_096);

        for attacker_class in 0..THREAT_DIRECT15_CLASS_NB {
            let mut expected_row = 0usize;
            for raw_from in 0..81u8 {
                let from = Square(raw_from);
                let mut targets = [Square(0); 36];
                let count = attack_targets(attacker_class, from, Some(Color::White), None, &mut targets);
                for (ordinal, target) in targets[..count].iter().enumerate() {
                    assert_eq!(
                        row_offset(attacker_class, from, *target),
                        expected_row + ordinal,
                        "attacker_class={attacker_class}, from={raw_from}, to={}",
                        target.0,
                    );
                }
                expected_row += count;
            }
            assert_eq!(expected_row, ATTACKS_PER_CLASS[attacker_class]);
        }
        assert_eq!(
            pair_base(THREAT_DIRECT15_KING_CLASS, THREAT_DIRECT15_KING_CLASS)
                + ATTACKS_PER_CLASS[THREAT_DIRECT15_KING_CLASS]
                - 1,
            THREAT_DIRECT15_DIMENSIONS - 1
        );
    }

    /// 螳蘖SV縺九ｉDirect15縺ｮ迚・・豢ｻ諤ｧ謨ｰ繧呈ｸｬ繧翫∝崋螳嘖parse螳ｹ驥・60縺ｫ蜿弱∪繧九％縺ｨ繧堤｢ｺ隱阪☆繧九・    /// 蜻ｼ蜃ｺ縺玲凾縺ｫ `PROFILE15_PSV_SAMPLE`縲∽ｻｻ諢上〒 `PROFILE15_PSV_RECORDS` 繧定ｨｭ螳壹☆繧九・
    /// 玉 attacker の列挙 (`for_each_king_attack`) が空盤面 King パターン
    /// (`attack_targets` class 9) と全マスで一致することを検証する。
    /// (玉の利きは駒配置・色に依存しないので、 実利き = 空盤面パターン。)
    #[test]
    fn profile15_king_attack_enumeration_matches_empty_board_pattern() {
        for raw in 0..81u8 {
            let from = Square(raw);
            let mut expected = [Square(0); 36];
            let n = attack_targets(THREAT_DIRECT15_KING_CLASS, from, Some(Color::White), None, &mut expected);
            let expected: Vec<u8> = expected[..n].iter().map(|sq| sq.0).collect();
            let mut actual: Vec<u8> = Vec::new();
            for_each_king_attack(from, |sq| actual.push(sq.0));
            actual.sort_unstable();
            assert_eq!(actual, expected, "from={raw}");
        }
    }

    #[test]
    fn profile15_direct_payload_activity_from_env() {
        use std::{env, fs::File, io::Read};
        let psv_path = match env::var("PROFILE15_PSV_SAMPLE") {
            Ok(path) => path,
            Err(_) => return,
        };
        let quantised_path = match env::var("PROFILE15_QUANTISED") {
            Ok(path) => path,
            Err(_) => return,
        };
        let requested =
            env::var("PROFILE15_PSV_RECORDS").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(100_000);
        let data = std::fs::read(&quantised_path).expect("read PROFILE15_QUANTISED");
        let magic = b"P2THRD2\0";
        let off = data.windows(magic.len()).position(|window| window == magic).expect("P2THRD2");
        let dims = u32::from_le_bytes(data[off + 16..off + 20].try_into().unwrap()) as usize;
        let buckets = u32::from_le_bytes(data[off + 20..off + 24].try_into().unwrap()) as usize;
        assert_eq!(dims, THREAT_DIRECT15_DIMENSIONS);
        assert_eq!(buckets, 8);
        let payload = off + 24;
        let mut weights = vec![0i32; dims];
        for local in 0..dims {
            weights[local] =
                i32::from_le_bytes(data[payload + 4 * local..payload + 4 * (local + 1)].try_into().unwrap());
        }
        let input = ShogiHalfKaHmThreatDirect15::new();
        let mut psv = PackedSfenValue::default();
        let mut record = [0u8; 40];
        let mut file = File::open(&psv_path).expect("open PROFILE15_PSV_SAMPLE");
        let mut scanned = 0usize;
        let mut stm_active = 0usize;
        let mut nstm_active = 0usize;
        let mut stm_weight_nonzero = 0usize;
        let mut nstm_weight_nonzero = 0usize;
        let mut zero_delta = 0usize;
        let mut nonzero_delta = 0usize;
        let mut sum_abs_delta = 0i64;
        let mut max_abs_delta = 0i64;
        while scanned < requested {
            match file.read_exact(&mut record) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => panic!("read PSV record {scanned}: {error}"),
            }
            psv.as_bytes_mut().copy_from_slice(&record);
            let mut stm_sum = 0i64;
            let mut nstm_sum = 0i64;
            let mut buf = [0i32; MAX_ACTIVE_DIRECT15_FEATURES];
            let n_stm = input.direct_indices(&psv, true, &mut buf);
            for &local in &buf[..n_stm] {
                let value = weights[local as usize];
                stm_active += 1;
                if value != 0 {
                    stm_weight_nonzero += 1;
                }
                stm_sum += i64::from(value);
            }
            let n_nstm = input.direct_indices(&psv, false, &mut buf);
            for &local in &buf[..n_nstm] {
                let value = weights[local as usize];
                nstm_active += 1;
                if value != 0 {
                    nstm_weight_nonzero += 1;
                }
                nstm_sum += i64::from(value);
            }
            let delta = stm_sum - nstm_sum;
            if delta == 0 {
                zero_delta += 1;
            } else {
                nonzero_delta += 1;
            }
            let abs = delta.abs();
            sum_abs_delta += abs;
            max_abs_delta = max_abs_delta.max(abs);
            scanned += 1;
        }
        assert!(scanned > 0);
        eprintln!(
            "Profile15 P2THRD2 activity: records={scanned} stm_active={stm_active} nstm_active={nstm_active} stm_weight_nonzero={stm_weight_nonzero} nstm_weight_nonzero={nstm_weight_nonzero} zero_delta={zero_delta} nonzero_delta={nonzero_delta} mean_abs_delta={:.3} max_abs_delta={max_abs_delta}",
            sum_abs_delta as f64 / scanned as f64,
        );
    }
    #[test]
    fn profile15_real_psv_direct_activity_fits_capacity() {
        use std::{env, fs::File, io::Read};

        // 同ファイル内の他の env 駆動テストと同じく、 PSV が指定されていなければ skip する
        // (旧実装は expect で panic していたため env 無しの `cargo test` が常に失敗していた)。
        let Ok(path) = env::var("PROFILE15_PSV_SAMPLE") else {
            return;
        };
        let requested =
            env::var("PROFILE15_PSV_RECORDS").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(100_000);
        assert!(requested > 0, "PROFILE15_PSV_RECORDS must be positive");

        let mut file = File::open(&path).expect("open PROFILE15_PSV_SAMPLE");
        let input = ShogiHalfKaHmThreatDirect15::new();
        let mut psv = PackedSfenValue::default();
        let mut record = [0u8; 40];
        let mut scanned = 0usize;
        let mut max_stm_direct = 0usize;
        let mut max_nstm_direct = 0usize;

        while scanned < requested {
            match file.read_exact(&mut record) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(error) => panic!("read PSV record {scanned}: {error}"),
            }
            psv.as_bytes_mut().copy_from_slice(&record);
            let mut buf = [0i32; MAX_ACTIVE_DIRECT15_FEATURES];
            let stm_direct = input.direct_indices(&psv, true, &mut buf);
            let nstm_direct = input.direct_indices(&psv, false, &mut buf);
            max_stm_direct = max_stm_direct.max(stm_direct);
            max_nstm_direct = max_nstm_direct.max(nstm_direct);
            assert!(
                stm_direct <= MAX_ACTIVE_DIRECT15_FEATURES,
                "STM Direct features exceeded sparse capacity at record {scanned}: {stm_direct}"
            );
            assert!(
                nstm_direct <= MAX_ACTIVE_DIRECT15_FEATURES,
                "NSTM Direct features exceeded sparse capacity at record {scanned}: {nstm_direct}"
            );
            scanned += 1;
        }

        assert!(scanned > 0, "PSV sample contains no complete records");
        eprintln!(
            "Profile15 Direct activity: scanned={scanned}, max_stm={max_stm_direct}, max_nstm={max_nstm_direct}, capacity={MAX_ACTIVE_DIRECT15_FEATURES}"
        );
    }
    #[test]
    fn profile15_direct_payload_single_record_from_env() {
        use std::{env, fs::File, io::Read};
        let psv_path = match env::var("PROFILE15_PSV_SAMPLE") {
            Ok(path) => path,
            Err(_) => return,
        };
        let quantised_path = match env::var("PROFILE15_QUANTISED") {
            Ok(path) => path,
            Err(_) => return,
        };
        let record_index =
            env::var("PROFILE15_PSV_RECORD_INDEX").ok().and_then(|value| value.parse::<usize>().ok()).unwrap_or(0);
        let data = std::fs::read(&quantised_path).expect("read PROFILE15_QUANTISED");
        let magic = b"P2THRD2\0";
        let off = data.windows(magic.len()).position(|window| window == magic).expect("P2THRD2");
        let dims = u32::from_le_bytes(data[off + 16..off + 20].try_into().unwrap()) as usize;
        let buckets = u32::from_le_bytes(data[off + 20..off + 24].try_into().unwrap()) as usize;
        assert_eq!(dims, THREAT_DIRECT15_DIMENSIONS);
        assert_eq!(buckets, 8);
        let payload = off + 24;
        let mut file = File::open(&psv_path).expect("open PROFILE15_PSV_SAMPLE");
        let mut psv = PackedSfenValue::default();
        let mut record = [0u8; 40];
        for index in 0..=record_index {
            file.read_exact(&mut record).unwrap_or_else(|error| panic!("read PSV record {index}: {error}"));
        }
        psv.as_bytes_mut().copy_from_slice(&record);
        let input = ShogiHalfKaHmThreatDirect15::new();
        let mut stm = [0i64; 8];
        let mut nstm = [0i64; 8];
        let mut stm_locals = Vec::new();
        let mut nstm_locals = Vec::new();
        let mut buf = [0i32; MAX_ACTIVE_DIRECT15_FEATURES];
        let n_stm = input.direct_indices(&psv, true, &mut buf);
        for &raw in &buf[..n_stm] {
            let local = raw as usize;
            stm_locals.push(local);
            for bucket in 0..buckets {
                let begin = payload + 4 * (bucket * dims + local);
                stm[bucket] += i64::from(i32::from_le_bytes(data[begin..begin + 4].try_into().unwrap()));
            }
        }
        let n_nstm = input.direct_indices(&psv, false, &mut buf);
        for &raw in &buf[..n_nstm] {
            let local = raw as usize;
            nstm_locals.push(local);
            for bucket in 0..buckets {
                let begin = payload + 4 * (bucket * dims + local);
                nstm[bucket] += i64::from(i32::from_le_bytes(data[begin..begin + 4].try_into().unwrap()));
            }
        }
        let delta: Vec<i64> = (0..buckets).map(|bucket| stm[bucket] - nstm[bucket]).collect();
        eprintln!(
            "Profile15 P2THRD2 single: record={record_index} sfen={} stm_locals={stm_locals:?} nstm_locals={nstm_locals:?} stm_sum={stm:?} nstm_sum={nstm:?} delta={delta:?}",
            psv.board().to_sfen(),
        );
    }

    const DIM: usize = THREAT_DIRECT15_DIMENSIONS;
    const B: usize = 8;

    fn shape(persp_split: bool, defended: bool) -> Direct15HeadShape {
        Direct15HeadShape::new(B, persp_split, defended)
    }

    /// 要素ごとに一意な値 (index) を持つ重み。 f32 で正確に表せる範囲 (< 2^24)。
    fn unique_weights(s: Direct15HeadShape) -> Vec<f32> {
        (0..s.weight_len()).map(|i| i as f32).collect()
    }

    /// 展開前の graph 要素 (bucket, local) の値を直接読む (column-major)。
    fn plain_at(w: &[f32], bucket: usize, local: usize) -> f32 {
        w[local * B + bucket]
    }

    /// 展開後の graph 要素 (bucket, persp, half, local) の値を、 展開先の意味づけで直接読む。
    fn expanded_at(w: &[f32], to: Direct15HeadShape, bucket: usize, persp: usize, half: usize, local: usize) -> f32 {
        let row = if to.persp == 2 { 2 * bucket + persp } else { bucket };
        let feature = if to.halves == 2 { half * DIM + local } else { local };
        w[feature * to.rows() + row]
    }

    #[test]
    fn direct15_shape_infer_disambiguates_by_bias_len() {
        assert_eq!(Direct15HeadShape::infer(B, 525_120, 8), Some(shape(false, false)));
        assert_eq!(Direct15HeadShape::infer(B, 1_050_240, 8), Some(shape(false, true)));
        assert_eq!(Direct15HeadShape::infer(B, 1_050_240, 16), Some(shape(true, false)));
        assert_eq!(Direct15HeadShape::infer(B, 2_100_480, 16), Some(shape(true, true)));
        assert_eq!(Direct15HeadShape::infer(B, 1_050_240, 12), None);
        assert_eq!(Direct15HeadShape::infer(B, 1_050_241, 8), None);
        assert_eq!(shape(false, false).weight_len(), 525_120);
        assert_eq!(shape(true, true).weight_len(), 2_100_480);
    }

    /// 展開後の全要素が、 (bucket, persp, half, local) の意味づけどおり展開前の (bucket, local) を指すこと。
    #[test]
    fn direct15_expand_weights_maps_every_cell_to_source() {
        let from = shape(false, false);
        let src = unique_weights(from);
        for (ps, df) in [(false, true), (true, false), (true, true), (false, false)] {
            let to = shape(ps, df);
            let dst = expand_direct15_head_weights(&src, from, to).expect("expand");
            assert_eq!(dst.len(), to.weight_len());
            for bucket in 0..B {
                for persp in 0..to.persp {
                    for half in 0..to.halves {
                        // 全 local を舐めると 4 形状 × 65,640 × 16 で重いので境界 + 刻みで確認。
                        for local in (0..DIM).step_by(97).chain([DIM - 1]) {
                            assert_eq!(
                                expanded_at(&dst, to, bucket, persp, half, local),
                                plain_at(&src, bucket, local),
                                "bucket={bucket} persp={persp} half={half} local={local} to={to:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// 展開直後の forward (sparse 入力の重み和) が展開前と同一であること。
    #[test]
    fn direct15_expand_weights_keeps_forward_output() {
        let from = shape(false, false);
        // 適当な決定的擬似乱数で重みを埋める (一意性は不要)。
        let mut seed = 0x9E37_79B9u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % 2001) as f32 / 1000.0 - 1.0
        };
        let src: Vec<f32> = (0..from.weight_len()).map(|_| next()).collect();
        // 局面 = (bucket, stm 側 active [(half, local)], nstm 側 active [(half, local)])
        let actives: Vec<(usize, usize)> = (0..40).map(|i| ((i * 7) % 2, (i * 1_777 + 13) % DIM)).collect();
        let (stm, nstm) = actives.split_at(24);
        for bucket in [0usize, 3, 7] {
            let old = |set: &[(usize, usize)]| set.iter().map(|&(_, local)| plain_at(&src, bucket, local)).sum::<f32>();
            let expected = old(stm) - old(nstm);
            for (ps, df) in [(false, true), (true, false), (true, true)] {
                let to = shape(ps, df);
                let dst = expand_direct15_head_weights(&src, from, to).expect("expand");
                let new = |set: &[(usize, usize)], persp: usize| {
                    set.iter().map(|&(half, local)| expanded_at(&dst, to, bucket, persp, half, local)).sum::<f32>()
                };
                // stm 視点は行 2b (persp 0)、 nstm 視点は行 2b+1 (persp 1)。 persp 1 形状では両方 0。
                let nstm_persp = to.persp - 1;
                assert_eq!(new(stm, 0) - new(nstm, nstm_persp), expected, "bucket={bucket} to={to:?}");
            }
        }
    }

    /// P5 → P5+P2、 P2 → P5+P2 の段階展開が plain → P5+P2 の一括展開と一致すること。
    #[test]
    fn direct15_expand_weights_composes() {
        let from = shape(false, false);
        let src = unique_weights(from);
        let both = shape(true, true);
        let direct = expand_direct15_head_weights(&src, from, both).expect("direct");
        for mid in [shape(false, true), shape(true, false)] {
            let step1 = expand_direct15_head_weights(&src, from, mid).expect("step1");
            let step2 = expand_direct15_head_weights(&step1, mid, both).expect("step2");
            assert_eq!(step2, direct, "mid={mid:?}");
        }
        // 恒等展開はコピー。
        assert_eq!(expand_direct15_head_weights(&src, from, from).expect("identity"), src);
    }

    #[test]
    fn direct15_expand_bias_duplicates_per_bucket() {
        let src: Vec<f32> = (0..B).map(|b| b as f32 * 0.5 + 0.25).collect();
        let dst = expand_direct15_head_bias(&src, shape(false, false), shape(true, true)).expect("bias");
        assert_eq!(dst.len(), 2 * B);
        for b in 0..B {
            assert_eq!(dst[2 * b], src[b]);
            assert_eq!(dst[2 * b + 1], src[b]);
        }
        // 守備ビットだけの展開は行数不変 (= コピー)。
        assert_eq!(expand_direct15_head_bias(&src, shape(false, false), shape(false, true)).expect("p5"), src);
        // P2 → P2+P5 は行ごとのコピー。
        let src16: Vec<f32> = (0..2 * B).map(|r| r as f32).collect();
        assert_eq!(expand_direct15_head_bias(&src16, shape(true, false), shape(true, true)).expect("p2->both"), src16);
    }

    #[test]
    fn direct15_expand_rejects_contraction_and_bad_len() {
        let plain = shape(false, false);
        let p5 = shape(false, true);
        let src = unique_weights(p5);
        assert!(expand_direct15_head_weights(&src, p5, plain).is_none(), "contraction");
        assert!(expand_direct15_head_weights(&src[..10], plain, p5).is_none(), "bad len");
        let other_buckets = Direct15HeadShape::new(16, false, false);
        assert!(expand_direct15_head_weights(&unique_weights(plain), plain, other_buckets).is_none(), "buckets");
        assert!(expand_direct15_head_bias(&[0.0; 16], shape(true, false), plain).is_none(), "bias contraction");
        assert!(expand_direct15_head_bias(&[0.0; 7], plain, p5).is_none(), "bias bad len");
    }
}
