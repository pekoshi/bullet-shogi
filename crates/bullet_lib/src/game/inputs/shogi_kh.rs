//! KH ブロック (2026-11-12): 「玉の 8 近傍の空升 × 敵の利き数」(K) と「浮き駒」(H) の疎入力。
//!
//! 背景: 兄弟差分 D1 (`runs/sibling/d1_paired_top3q_v31.log`) で、 現職 net が v3.1 より
//! `zaE` (玉 8 近傍の空升への敵の利きの数) と `hang` (浮き駒の数) を 15〜27σ 楽観的に
//! 評価していると分かった。 その 2 量を one-hot の疎入力にして FT 出力へ足し込む。
//!
//! 数え方の正本は `C:/project/p2shogi/runs/sibling/sib_feats.py` の `analyse()`:
//! - 利き `eff[c][sq]` = 色 c の駒 (**玉を含む**) が sq に利いている数。 slider は 1 枚目で止まる
//!   (x-ray は数えない)。 = `board_effect_v1` と同じ数え方。
//! - zaE[c] = 色 c の玉の 8 近傍のうち盤内かつ**空升**について `eff[1-c]` の和
//! - hang[c] = 色 c の駒 (玉以外) のうち `eff[1-c] > 0` かつ `eff[c] == 0` のものの数
//!
//! 特徴 index (視点 P ごと、 0..[`KH_DIMS`]):
//! - K 部 `[0, 64)`: `side * 32 + dir * 4 + level`
//!   - side 0 = P の玉 / 1 = P の相手の玉
//!   - dir = P から見た向き (df, dr) の番号 (sib_feats の NEIGH と同じ並び。 P が後手なら盤を 180 度回す)
//!   - level = min(その空升への、 その玉の敵の利き数, 3)。 盤外・駒のある升は特徴を出さない
//! - H 部 `[64, 142)`: `64 + side * 39 + pt * 3 + zone`
//!   - side 0 = P の駒 / 1 = P の相手の駒 (浮き駒の持ち主)
//!   - pt = 駒種 13 種 (歩 香 桂 銀 角 飛 金 と 成香 成桂 成銀 馬 龍 の順)
//!   - zone = P から見た段 / 3 (0 = P から見た 1-3 段目 = P の敵陣 / 1 = 4-6 段 / 2 = 7-9 段 = P の自陣)
//!
//! ## K 部の第 2 表現 (`--kh-kmode 2`、 2026-11-12)
//!
//! D3 で K 部 (上の 64 次元) は split-half cos +0.21〜0.26 (null 0.08〜0.11) で不通過。 仮説 =
//! 「同じ空升への利きでも玉の位置 (隅・端・中央 / 囲いの中か居玉か) で意味が変わる」。
//! そこで K 部を**玉の位置の区分で条件付け**した表現に差し替える (H 部はそのまま、 後ろにずれるだけ)。
//!
//! - 座標系: 玉ごとに**その玉の持ち主から見た向き** (先手玉はそのまま / 後手玉は 180 度回す)。
//!   さらに HalfKA_hm と同じく、 持ち主から見た筋 index が 5 以上 (= 6〜9 筋) なら**左右反転**する。
//!   反転後の向き d は「端に近い側 / 中央側」の意味で揃う (sib_feats の NEIGH の df を反転)。
//! - 区分 region = 筋 3 区分 x 段 3 区分 = 9 (`region = 筋区分 * 3 + 段区分`)
//!   - 筋 (反転後の端からの距離): 0 = 1-2 筋 (9-8 筋) / 1 = 3-4 筋 (7-6 筋) / 2 = 5 筋
//!   - 段 (持ち主から見た段): 0 = 9 段 / 1 = 8 段 / 2 = 7 段以上 (1〜7 段)
//!   - 段を 1-3 / 4-6 / 7-9 で切らない理由: fgb_tr 2 万局面 (玉 4 万個) で 7-9 段が **95%**
//!     (9 段 29% / 8 段 57% / 7 段 9% / 6 段以上 5%)。 居玉 (5九) と囲い (8八・2八) を分けるのは
//!     9 段 / 8 段の区別なので、 自陣の中を割る方が情報が多い。 最少区分 (5 筋 x 7 段以上) でも 1.7%。
//! - K2 方向部 `[0, 576)`: `((side * 9 + region) * 8 + dir) * 4 + level` (level は従来と同じ 0..3+)
//! - K2 集計部 `[576, 720)`: `576 + (side * 9 + region) * 8 + min(zaE, 7)`
//!   (zaE = その玉の 8 近傍の盤内の空升への敵の利き数の和。 空升が無くても 0 として 1 本立つ)
//! - H 部 `[720, 798)`: 従来と同じ並び。 1 視点の上限は 2 玉 x (8 + 1) + 38 = 56 本
//!
//! ## X 部 (`--kh-feats hx` / `hxh`、 2026-11-18 KT)
//!
//! 「損な当たり」の大きい方から 1 番目・2 番目を値の段で one-hot にする (H 部の後ろ、 kmode 1 なら `[142, 170)`)。
//! 正本 = `C:/project/p2shogi/runs/tempo_material_20261118/ktprobe4.py` の `top2()`。
//! - 損な額 (玉以外の盤上の駒 t、 持ち主 c、 敵の利き (玉を含む) がある駒だけ):
//!   - 味方の利きが無い (浮き駒) -> 駒価値 `KT_PV[t]`
//!   - 味方の利きがある -> 玉以外の敵の利き駒のうち t より安いものがあれば `KT_PV[t] - その最小値` (無ければ損なし)
//!   - `hxh` (浮き駒だけ) は前者だけを数える (利き駒の値を見ないので利き計算が増えない)
//! - 色ごとに損な額を降順に並べ、 上位 2 本 (rank 0 / 1) を値の段 bin (0..7) で立てる。 額 0 の順位は立てない
//! - index = `x_off + (side * 2 + rank) * 7 + bin` (side 0 = P の駒の損 / 1 = P の相手の駒の損)
//! - bin の境界 (以下): 100 / 330 / 420 / 560 / 900 / 1000 / それ以上 (ktprobe4.py の `BIN`)
//! - 重みはゼロ初期化の列を足すだけなので、 学習前は X 部なしの net と同じ関数 (resume 時は adapter が 0 列を足す)

use std::cell::RefCell;

use super::shogi_halfka_hm_threat::{Occupied, for_each_attack};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, PieceType, Square},
};

/// K 部の次元 (2 玉 x 8 方向 x 4 段)。
pub const KH_K_DIMS: usize = 64;
/// H 部の次元 (2 色 x 13 駒種 x 3 区域)。
pub const KH_H_DIMS: usize = 78;
/// KH ブロックの入力次元 (視点ごと)。
pub const KH_DIMS: usize = KH_K_DIMS + KH_H_DIMS;
/// 1 視点で立つ特徴数の上限 (K は 2 玉 x 8 = 16、 H は玉以外の盤上の駒 38)。
pub const KH_MAX_ACTIVE: usize = 16 + 38;
/// 手番側視点の疎入力名。
pub const KH_SPARSE_STM: &str = "kh_stm";
/// 非手番側視点の疎入力名。
pub const KH_SPARSE_NSTM: &str = "kh_nstm";

/// `--kh-kmode 2` の K 部: 区分の数 (筋 3 x 段 3)。
pub const KH2_REGIONS: usize = 9;
/// `--kh-kmode 2` の K 部のうち方向部の次元 (2 玉 x 9 区分 x 8 方向 x 4 段)。
pub const KH2_K_DIR_DIMS: usize = 2 * KH2_REGIONS * 8 * 4;
/// `--kh-kmode 2` の K 部のうち集計部の次元 (2 玉 x 9 区分 x zaE 0..7+)。
pub const KH2_K_AGG_DIMS: usize = 2 * KH2_REGIONS * 8;
/// `--kh-kmode 2` の K 部の次元。
pub const KH2_K_DIMS: usize = KH2_K_DIR_DIMS + KH2_K_AGG_DIMS;
/// `--kh-kmode 2` の 1 視点で立つ特徴数の上限 (K は 2 玉 x (8 方向 + 集計 1) = 18、 H は 38)。
pub const KH2_MAX_ACTIVE: usize = 18 + 38;
/// X 部の値の段の数 (ktprobe4.py の `BIN` の長さ)。
pub const KH_X_BINS: usize = 7;
/// X 部の次元 (2 色 x 順位 2 x 値 7 段)。
pub const KH_X_DIMS: usize = 2 * 2 * KH_X_BINS;
/// X 部で 1 視点に立つ特徴数の上限 (2 色 x 順位 2)。
const KH_X_MAX_ACTIVE: usize = 4;
/// 内部バッファの長さ (全モードの上限の最大)。
const KH_BUF: usize = KH2_MAX_ACTIVE + KH_X_MAX_ACTIVE;
/// X 部の駒価値 (cshogi / [`PieceType`] の番号で引く。 ktprobe4.py の common.PV と同じ値)。
const KT_PV: [u16; 15] = [0, 90, 315, 405, 495, 855, 990, 540, 0, 540, 540, 540, 540, 945, 1395];
/// X 部の値の段の上端 (この値以下ならその段。 最後の段は上限なし)。
const KT_BIN_UPPER: [u16; KH_X_BINS - 1] = [100, 330, 420, 560, 900, 1000];

/// sib_feats.py の `NEIGH` と同じ並び (df 外側、 dr 内側、 (0,0) を除く)。
const NEIGH: [(i8, i8); 8] = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)];

/// K 部の表現 (`--kh-kmode`)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KhKMode {
    /// 1 (既定): 玉 x 8 方向 x 利き数 (64 次元)。 2026-11-12 の初版と同一。
    Plain,
    /// 2: 玉の位置の区分 9 で条件付け (方向部 576 + 集計部 144 = 720 次元)。
    Region,
}

/// X 部 (損な当たりの 1 番目・2 番目) の表現。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KhXMode {
    /// X 部なし (従来と同一。 `kh_ft_w` の列も増えない)。
    Off,
    /// 浮き駒 + 安い駒に当てられたひも付きの駒の差額 (`--kh-feats hx`)。
    Loss,
    /// 浮き駒だけ (`--kh-feats hxh`)。
    Hang,
}

/// どの部分を出すか。 `--kh-feats` / `--kh-kmode` で選ぶ (`k` / `h` の切り替えでは index の配置は
/// 変わらない = checkpoint の形は同じ。 `kmode` は K 部の次元を変える)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KhParts {
    /// K 部 (玉近傍の空升 x 敵の利き数) を出す。
    pub k: bool,
    /// H 部 (浮き駒) を出す。
    pub h: bool,
    /// K 部の表現。
    pub kmode: KhKMode,
    /// X 部の表現 (`Off` なら列を持たない)。
    pub x: KhXMode,
}

impl KhParts {
    /// K 部の次元 (H 部の index はこの後ろから始まる)。
    pub fn k_dims(&self) -> usize {
        match self.kmode {
            KhKMode::Plain => KH_K_DIMS,
            KhKMode::Region => KH2_K_DIMS,
        }
    }

    /// X 部の次元 (`Off` なら 0)。
    pub fn x_dims(&self) -> usize {
        if self.x == KhXMode::Off { 0 } else { KH_X_DIMS }
    }

    /// X 部の先頭 index (= K 部 + H 部の次元)。
    pub fn x_offset(&self) -> usize {
        self.k_dims() + KH_H_DIMS
    }

    /// 入力次元 (視点ごと) = `kh_ft_w` の列数。
    pub fn dims(&self) -> usize {
        self.k_dims() + KH_H_DIMS + self.x_dims()
    }

    /// K 部 + H 部で 1 視点に立つ特徴数の上限。
    fn base_max_active(&self) -> usize {
        match self.kmode {
            KhKMode::Plain => KH_MAX_ACTIVE,
            KhKMode::Region => KH2_MAX_ACTIVE,
        }
    }

    /// 1 視点で立つ特徴数の上限 (疎入力の nnz)。
    pub fn max_active(&self) -> usize {
        if self.x == KhXMode::Off {
            self.base_max_active()
        } else {
            self.base_max_active() + KH_X_MAX_ACTIVE
        }
    }

    /// K 部の内訳 `(名前, 開始, 終了)` (dump の json に書き、 D3 の cos を部分ごとに切る)。
    pub fn k_sub(&self) -> Vec<(&'static str, usize, usize)> {
        match self.kmode {
            KhKMode::Plain => vec![("K 方向", 0, KH_K_DIMS)],
            KhKMode::Region => vec![("K2 方向", 0, KH2_K_DIR_DIMS), ("K2 集計", KH2_K_DIR_DIMS, KH2_K_DIMS)],
        }
    }
}

/// 色ごとの利きの数 (玉の利きを含む / x-ray は含まない)。 `eff[color][sq]`。
pub fn effect_counts(board: &ShogiBoard) -> [[u8; 81]; 2] {
    let mut eff = [[0u8; 81]; 2];
    let occ = Occupied::from_board(board);
    for sq_raw in 0..81u8 {
        let from = Square(sq_raw);
        let pc = board.piece_on(from);
        if pc.is_none() {
            continue;
        }
        let ci = pc.color as usize;
        if pc.piece_type == PieceType::King {
            // for_each_attack は玉を列挙しないので 8 近傍を手で足す (board_effect_v1 と同じ)。
            let (f, r) = (from.file() as i8, from.rank() as i8);
            for (df, dr) in NEIGH {
                let (tf, tr) = (f + df, r + dr);
                if (0..9).contains(&tf) && (0..9).contains(&tr) {
                    let t = Square::new(tf as u8, tr as u8).index();
                    eff[ci][t] = eff[ci][t].saturating_add(1);
                }
            }
            continue;
        }
        for_each_attack(pc.piece_type, pc.color, from, &occ, |to| {
            eff[ci][to.index()] = eff[ci][to.index()].saturating_add(1);
        });
    }
    eff
}

/// 検算用の生の数 (sib_feats.py の `zaE` / `hang` と同じ定義。 添字は色 0 = 先手 / 1 = 後手)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KhCounts {
    /// 色 c の玉の 8 近傍の空升への、 敵の利き数の和。
    pub za_e: [u32; 2],
    /// 色 c の浮き駒 (玉以外 / 敵の利きあり / 味方の利きなし) の数。
    pub hang: [u32; 2],
}

/// 生の数を返す (検算用)。
pub fn kh_counts(board: &ShogiBoard) -> KhCounts {
    let eff = effect_counts(board);
    let mut out = KhCounts::default();
    for c in [Color::Black, Color::White] {
        let ci = c as usize;
        let k = board.king_square(c);
        if k != Square::NONE {
            let (f, r) = (k.file() as i8, k.rank() as i8);
            for (df, dr) in NEIGH {
                let (tf, tr) = (f + df, r + dr);
                if !((0..9).contains(&tf) && (0..9).contains(&tr)) {
                    continue;
                }
                let t = Square::new(tf as u8, tr as u8);
                if board.piece_on(t).is_none() {
                    out.za_e[ci] += eff[1 - ci][t.index()] as u32;
                }
            }
        }
    }
    for sq in 0..81usize {
        let pc = board.board[sq];
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let ci = pc.color as usize;
        if eff[1 - ci][sq] > 0 && eff[ci][sq] == 0 {
            out.hang[ci] += 1;
        }
    }
    out
}

/// 玉以外の駒種を 0..13 に詰める (歩..金 = 0..6、 と..龍 = 7..12)。
fn piece_index(pt: PieceType) -> Option<usize> {
    let v = pt as usize;
    match v {
        1..=7 => Some(v - 1),
        9..=14 => Some(v - 2),
        _ => None,
    }
}

/// 視点 P の座標へ写す (後手視点は 180 度回転)。 自己逆写像。
#[inline]
fn orient(sq: Square, p: Color) -> Square {
    if p == Color::Black { sq } else { sq.inverse() }
}

/// `--kh-kmode 2` の玉の区分 (0..9) と左右反転の有無。 `kc` = 玉の持ち主、 `k` = 玉の升 (盤の座標)。
fn kh2_region(k: Square, kc: Color) -> (usize, bool) {
    let ko = orient(k, kc);
    let (f, r) = (ko.file() as usize, ko.rank() as usize);
    let mirror = f >= 5; // HalfKA_hm の is_hm_mirror と同じ境界
    let fold = if mirror { 8 - f } else { f };
    let fb = match fold {
        0 | 1 => 0,
        2 | 3 => 1,
        _ => 2,
    };
    let rb = match r {
        8 => 0,
        7 => 1,
        _ => 2,
    };
    (fb * 3 + rb, mirror)
}

/// `--kh-kmode 2` の K 部を視点 `p` について `buf[*n..]` に書く。
fn push_k2(board: &ShogiBoard, eff: &[[u8; 81]; 2], p: Color, buf: &mut [i32; KH_BUF], n: &mut usize) {
    for (side, kc) in [(0usize, p), (1usize, p.opponent())] {
        let k = board.king_square(kc);
        if k == Square::NONE {
            continue;
        }
        let enemy = kc.opponent() as usize;
        let (region, mirror) = kh2_region(k, kc);
        let ko = orient(k, kc);
        let (f, r) = (ko.file() as i8, ko.rank() as i8);
        let base = side * KH2_REGIONS + region;
        let mut za = 0usize;
        for (d, (df, dr)) in NEIGH.iter().enumerate() {
            let df = if mirror { -df } else { *df };
            let (tf, tr) = (f + df, r + dr);
            if !((0..9).contains(&tf) && (0..9).contains(&tr)) {
                continue;
            }
            let t = orient(Square::new(tf as u8, tr as u8), kc);
            if !board.piece_on(t).is_none() {
                continue;
            }
            let e = eff[enemy][t.index()] as usize;
            za += e;
            buf[*n] = ((base * 8 + d) * 4 + e.min(3)) as i32;
            *n += 1;
        }
        buf[*n] = (KH2_K_DIR_DIMS + base * 8 + za.min(7)) as i32;
        *n += 1;
    }
}

/// X 部の値の段 (0..[`KH_X_BINS`])。
fn kt_bin(v: u16) -> usize {
    KT_BIN_UPPER.iter().position(|&t| v <= t).unwrap_or(KH_X_BINS - 1)
}

/// 色ごとの、 玉以外の利き駒の最小の駒価値 (利きが無ければ `u16::MAX`)。 `min_att[color][sq]`。
fn min_attacker_values(board: &ShogiBoard) -> [[u16; 81]; 2] {
    let mut m = [[u16::MAX; 81]; 2];
    let occ = Occupied::from_board(board);
    for sq_raw in 0..81u8 {
        let from = Square(sq_raw);
        let pc = board.piece_on(from);
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let ci = pc.color as usize;
        let v = KT_PV[pc.piece_type as usize];
        for_each_attack(pc.piece_type, pc.color, from, &occ, |to| {
            let t = &mut m[ci][to.index()];
            *t = (*t).min(v);
        });
    }
    m
}

/// X 部の生の値: 色ごとの損な額の上位 2 本 (降順、 無ければ 0)。 `[color][rank]`。 ktprobe4.py の `top2()`。
pub fn kt_top2(board: &ShogiBoard, mode: KhXMode) -> [[u16; 2]; 2] {
    let mut out = [[0u16; 2]; 2];
    if mode == KhXMode::Off {
        return out;
    }
    let eff = effect_counts(board);
    let min_att = if mode == KhXMode::Loss { Some(min_attacker_values(board)) } else { None };
    for sq in 0..81usize {
        let pc = board.board[sq];
        if pc.is_none() || pc.piece_type == PieceType::King {
            continue;
        }
        let ci = pc.color as usize;
        if eff[1 - ci][sq] == 0 {
            continue;
        }
        let v = KT_PV[pc.piece_type as usize];
        let loss = if eff[ci][sq] == 0 {
            v
        } else if let Some(m) = &min_att {
            let a = m[1 - ci][sq];
            v.saturating_sub(a)
        } else {
            0
        };
        let top = &mut out[ci];
        if loss > top[0] {
            top[1] = top[0];
            top[0] = loss;
        } else if loss > top[1] {
            top[1] = loss;
        }
    }
    out
}

/// 1 視点分の特徴。 `n_base` = K 部 + H 部の本数 (その後ろ `[n_base, n)` が X 部)。
#[derive(Clone, Copy)]
struct KhFeat {
    buf: [i32; KH_BUF],
    n: usize,
    n_base: usize,
}

const KH_FEAT_EMPTY: KhFeat = KhFeat { buf: [0i32; KH_BUF], n: 0, n_base: 0 };

/// 両視点 (先手視点 / 後手視点) の特徴を一度に計算する。 戻り値は色で引く。
fn kh_features_both(board: &ShogiBoard, parts: KhParts) -> [KhFeat; 2] {
    let eff = effect_counts(board);
    let k_dims = parts.k_dims();
    let base_max = parts.base_max_active();
    let top2 = kt_top2(board, parts.x);
    let mut res = [KH_FEAT_EMPTY; 2];
    for p in [Color::Black, Color::White] {
        let KhFeat { buf, n, n_base } = &mut res[p as usize];
        if parts.k && parts.kmode == KhKMode::Region {
            push_k2(board, &eff, p, buf, n);
        }
        if parts.k && parts.kmode == KhKMode::Plain {
            for (side, kc) in [(0usize, p), (1usize, p.opponent())] {
                let k = board.king_square(kc);
                if k == Square::NONE {
                    continue;
                }
                let enemy = kc.opponent() as usize;
                let ko = orient(k, p);
                let (f, r) = (ko.file() as i8, ko.rank() as i8);
                for (d, (df, dr)) in NEIGH.iter().enumerate() {
                    let (tf, tr) = (f + df, r + dr);
                    if !((0..9).contains(&tf) && (0..9).contains(&tr)) {
                        continue;
                    }
                    let t = orient(Square::new(tf as u8, tr as u8), p);
                    if !board.piece_on(t).is_none() {
                        continue;
                    }
                    let level = (eff[enemy][t.index()] as usize).min(3);
                    buf[*n] = (side * 32 + d * 4 + level) as i32;
                    *n += 1;
                }
            }
        }
        if parts.h {
            for sq in 0..81usize {
                let pc = board.board[sq];
                if pc.is_none() {
                    continue;
                }
                let Some(pt) = piece_index(pc.piece_type) else { continue };
                let ci = pc.color as usize;
                if !(eff[1 - ci][sq] > 0 && eff[ci][sq] == 0) {
                    continue;
                }
                let side = usize::from(pc.color != p);
                let zone = orient(Square(sq as u8), p).rank() as usize / 3;
                if *n < base_max {
                    buf[*n] = (k_dims + side * 39 + pt * 3 + zone) as i32;
                    *n += 1;
                }
            }
        }
        *n_base = *n;
        if parts.x != KhXMode::Off {
            let x_off = parts.x_offset();
            for (side, c) in [(0usize, p), (1usize, p.opponent())] {
                for (rank, &v) in top2[c as usize].iter().enumerate() {
                    if v > 0 {
                        buf[*n] = (x_off + (side * 2 + rank) * KH_X_BINS + kt_bin(v)) as i32;
                        *n += 1;
                    }
                }
            }
        }
    }
    res
}

/// 視点 `p` の特徴 index を `out` の先頭に書き、 個数を返す。
pub fn kh_features(board: &ShogiBoard, p: Color, parts: KhParts, out: &mut [i32]) -> usize {
    let all = kh_features_both(board, parts);
    let f = &all[p as usize];
    let n = f.n.min(out.len());
    out[..n].copy_from_slice(&f.buf[..n]);
    n
}

/// 局面の 32 byte の盤面部分から作る 64bit hash (FNV-1a + 最終混合)。 split-half と null の鍵に使う。
pub fn kh_position_hash(pos: &PackedSfenValue) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in &pos.as_bytes()[..32] {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    // splitmix64 の最終混合 (下位ビットの偏りを消す)
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// `--kh-split` の半分の番号 (0 / 1)。 同じ盤面は必ず同じ半分に入る。
pub fn kh_split_half(pos: &PackedSfenValue) -> u8 {
    (kh_position_hash(pos) >> 63) as u8
}

/// null (特徴シャッフル) 用の状態。 loader は「入力名ごと x chunk ごと」に scoped thread を立て、
/// chunk 内の局面を順に処理するので、 thread-local に「直前の別局面」の特徴を持てば
/// 手番側・非手番側の 2 回の pass で同じ組み合わせになる (chunk 先頭の局面は特徴なし)。
struct NullState {
    key: u64,
    valid: bool,
    /// 現在の局面の (手番側, 非手番側) の特徴。
    cur: [KhFeat; 2],
    /// 直前の別局面の (手番側, 非手番側) の特徴。
    prev: Option<[KhFeat; 2]>,
}

thread_local! {
    static KH_NULL_STATE: RefCell<NullState> = const {
        RefCell::new(NullState {
            key: 0,
            valid: false,
            cur: [KH_FEAT_EMPTY; 2],
            prev: None,
        })
    };
}

/// loader から呼ぶ入口。 `null = true` なら直前の別局面の特徴を返す (局面と教師の対応を壊した対照)。
/// X 部があるときの null は **X 部だけ**を直前の別局面のものに差し替え、 K 部・H 部は現在の局面のまま
/// (学習済みの H 部を壊さずに X 部の勾配の向きだけを対照と比べるため。 2026-11-18 KT の D3)。
pub fn fill_kh_sparse(pos: &PackedSfenValue, stm_side: bool, parts: KhParts, null: bool, out: &mut [i32]) -> usize {
    let board = ShogiBoard::from_packed_sfen(pos);
    let p = if stm_side { board.side_to_move } else { board.side_to_move.opponent() };
    if !null {
        return kh_features(&board, p, parts, out);
    }
    let key = kh_position_hash(pos);
    KH_NULL_STATE.with(|cell| {
        let mut st = cell.borrow_mut();
        if !st.valid || st.key != key {
            let all = kh_features_both(&board, parts);
            let stm = board.side_to_move as usize;
            let cur = [all[stm], all[1 - stm]];
            st.prev = if st.valid { Some(st.cur) } else { None };
            st.cur = cur;
            st.key = key;
            st.valid = true;
        }
        let side = usize::from(!stm_side);
        if parts.x != KhXMode::Off {
            // K 部・H 部 = 現在の局面 / X 部 = 直前の別局面 (chunk 先頭なら X 部なし)。
            let cur = &st.cur[side];
            let mut n = cur.n_base.min(out.len());
            out[..n].copy_from_slice(&cur.buf[..n]);
            if let Some(prev) = &st.prev {
                let pf = &prev[side];
                for &i in &pf.buf[pf.n_base..pf.n] {
                    if n < out.len() {
                        out[n] = i;
                        n += 1;
                    }
                }
            }
            return n;
        }
        let Some(prev) = st.prev else { return 0 };
        let f = &prev[side];
        let n = f.n.min(out.len());
        out[..n].copy_from_slice(&f.buf[..n]);
        n
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_layout() {
        assert_eq!(KH_DIMS, 142);
        assert_eq!(piece_index(PieceType::Pawn), Some(0));
        assert_eq!(piece_index(PieceType::Gold), Some(6));
        assert_eq!(piece_index(PieceType::King), None);
        assert_eq!(piece_index(PieceType::ProPawn), Some(7));
        assert_eq!(piece_index(PieceType::Dragon), Some(12));
    }

    /// 180 度回転で対称な盤 (玉 / 金 / 歩 / 角 / 飛)。
    fn symmetric_board() -> ShogiBoard {
        use crate::shogi::types::Piece;
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
        board
    }

    #[test]
    fn symmetric_board_features_match_between_perspectives() {
        let board = symmetric_board();
        let c = kh_counts(&board);
        assert_eq!(c.hang, [0, 0]);
        assert_eq!(c.za_e[0], c.za_e[1]);
        let mut out = [0i32; KH_MAX_ACTIVE];
        let parts = KhParts { k: true, h: true, kmode: KhKMode::Plain, x: KhXMode::Off };
        let nb = kh_features(&board, Color::Black, parts, &mut out);
        let fb: Vec<i32> = out[..nb].to_vec();
        let nw = kh_features(&board, Color::White, parts, &mut out);
        let fw: Vec<i32> = out[..nw].to_vec();
        // 180 度対称なので両視点の特徴は一致する (向きの写像が正しいことの検査)。
        assert_eq!(fb, fw);
        assert!(!fb.is_empty());
        assert!(fb.iter().all(|&i| (i as usize) < KH_K_DIMS));
    }

    #[test]
    fn kmode2_symmetric_and_layout() {
        let board = symmetric_board();
        let parts = KhParts { k: true, h: true, kmode: KhKMode::Region, x: KhXMode::Off };
        assert_eq!(parts.dims(), 720 + 78);
        let mut out = [0i32; KH2_MAX_ACTIVE];
        let nb = kh_features(&board, Color::Black, parts, &mut out);
        let fb: Vec<i32> = out[..nb].to_vec();
        let nw = kh_features(&board, Color::White, parts, &mut out);
        let fw: Vec<i32> = out[..nw].to_vec();
        assert_eq!(fb, fw);
        // 5九 / 5一 の居玉 = 筋区分 2 (5 筋) x 段区分 0 (9 段) = region 6。 集計部は玉ごとに 1 本。
        let agg: Vec<usize> =
            fb.iter().map(|&i| i as usize).filter(|&i| (KH2_K_DIR_DIMS..KH2_K_DIMS).contains(&i)).collect();
        assert_eq!(agg.len(), 2);
        assert_eq!((agg[0] - KH2_K_DIR_DIMS) / 8, 6);
        assert_eq!((agg[1] - KH2_K_DIR_DIMS) / 8, KH2_REGIONS + 6);
        // 集計部の zaE は kh_counts と一致する。
        let c = kh_counts(&board);
        assert_eq!((agg[0] - KH2_K_DIR_DIMS) % 8, (c.za_e[0] as usize).min(7));
    }

    #[test]
    fn kmode2_mirror_folds_files() {
        // 先手玉 2八 (筋 index 1) と 8八 (筋 index 7) は左右反転で同じ区分・同じ方向集合になる。
        use crate::shogi::types::Piece;
        let mk = |kf: u8| {
            let mut b = ShogiBoard {
                side_to_move: Color::Black,
                black_king_sq: Square::new(kf, 7),
                white_king_sq: Square::new(4, 0),
                ..Default::default()
            };
            b.board[b.black_king_sq.index()] = Piece::new(Color::Black, PieceType::King);
            b.board[b.white_king_sq.index()] = Piece::new(Color::White, PieceType::King);
            b
        };
        let parts = KhParts { k: true, h: false, kmode: KhKMode::Region, x: KhXMode::Off };
        let mut out = [0i32; KH2_MAX_ACTIVE];
        let n1 = kh_features(&mk(1), Color::Black, parts, &mut out);
        let mut a: Vec<i32> = out[..n1].to_vec();
        let n7 = kh_features(&mk(7), Color::Black, parts, &mut out);
        let mut b: Vec<i32> = out[..n7].to_vec();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        assert_eq!(kh2_region(Square::new(1, 7), Color::Black), (1, false));
        assert_eq!(kh2_region(Square::new(7, 7), Color::Black), (1, true));
        // 後手玉は持ち主から見る: 盤の (7, 1) = 後手から見て (1, 7) 相当。
        assert_eq!(kh2_region(Square::new(7, 1), Color::White), (1, false));
    }

    #[test]
    fn x_part_loss_and_hang() {
        use crate::shogi::types::Piece;
        // 後手の歩 (2,4) は浮き駒 (損 90)。 先手の金 (2,5) は歩 (2,6) のひもがあるが歩に当たっている (損 540 - 90 = 450)。
        let mut board = symmetric_board();
        board.board[Square::new(2, 4).index()] = Piece::new(Color::White, PieceType::Pawn);
        board.board[Square::new(2, 5).index()] = Piece::new(Color::Black, PieceType::Gold);
        assert_eq!(kt_top2(&board, KhXMode::Loss), [[450, 0], [90, 0]]);
        assert_eq!(kt_top2(&board, KhXMode::Hang), [[0, 0], [90, 0]]);
        assert_eq!(kt_top2(&board, KhXMode::Off), [[0, 0], [0, 0]]);
        let parts = KhParts { k: false, h: true, kmode: KhKMode::Plain, x: KhXMode::Loss };
        assert_eq!(parts.dims(), 170);
        assert_eq!(parts.max_active(), KH_MAX_ACTIVE + 4);
        let mut out = [0i32; KH_BUF];
        let n = kh_features(&board, Color::Black, parts, &mut out);
        let x: Vec<i32> = out[..n].iter().copied().filter(|&i| i >= 142).collect();
        // 先手視点: 自分の 1 番目 450 = bin 3 / 相手の 1 番目 90 = bin 0。
        assert_eq!(x, vec![142 + 3, 142 + 2 * 7]);
        let n = kh_features(&board, Color::White, parts, &mut out);
        let x: Vec<i32> = out[..n].iter().copied().filter(|&i| i >= 142).collect();
        assert_eq!(x, vec![142, 142 + 2 * 7 + 3]);
        // bin の境界 (以下でその段)。
        assert_eq!([kt_bin(90), kt_bin(100), kt_bin(101), kt_bin(900), kt_bin(1000), kt_bin(1395)], [0, 0, 1, 4, 5, 6]);
    }

    #[test]
    fn hanging_pawns_are_counted() {
        use crate::shogi::types::Piece;
        let mut board = symmetric_board();
        // file 0 の rank 5 に後手の歩: 先手の歩 (rank 6) と互いに当たり、 どちらもひもが無い。
        board.board[Square::new(0, 5).index()] = Piece::new(Color::White, PieceType::Pawn);
        let c = kh_counts(&board);
        assert_eq!(c.hang, [1, 1]);
        let mut out = [0i32; KH_MAX_ACTIVE];
        let n = kh_features(
            &board,
            Color::Black,
            KhParts { k: false, h: true, kmode: KhKMode::Plain, x: KhXMode::Off },
            &mut out,
        );
        let mut f: Vec<i32> = out[..n].to_vec();
        f.sort();
        // 先手視点: 自分の歩 (side 0, pt 0) は rank 6 = zone 2、 相手の歩 (side 1, pt 0) は rank 5 = zone 1。
        assert_eq!(f, vec![(KH_K_DIMS + 2) as i32, (KH_K_DIMS + 39 + 1) as i32]);
    }
}
