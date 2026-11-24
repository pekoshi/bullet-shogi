//! ShogiHalfKaHmPolicy — HalfKA_hm sparse features + `policy_target` 2187-dim
//! one-hot dense auxiliary。
//!
//! Phase 1 (one-hot teacher、 既存 .pack/.spk 流用) の multi-head 訓練用 input
//! type。 sparse 部分は [`ShogiHalfKA_hm`] に完全 delegate (= value 訓練時と
//! bit-exact 互換)。 auxiliary に `policy_target` (2187 dim one-hot) を追加。
//!
//! ## policy index encoding (dlshogi `make_move_label` 互換)
//!
//! - `2187 = 81 × 27`
//!   - 81: 移動先マス (stm 視点、 white 番では 180 度回転)
//!   - 27: direction × promote
//!     - `0..10`: 盤上 10 方向 (Up / UpLeft / UpRight / Left / Right / Down /
//!       DownLeft / DownRight / Up2Left / Up2Right) 不成
//!     - `10..20`: 同方向 成
//!     - `20..27`: 持駒 7 種 (Pawn / Lance / Knight / Silver / Bishop / Rook /
//!       Gold) の打ち
//!
//! dlshogi-style 訓練で広く使われるエンコーディング。 rshogi の
//! `tools::dlshogi_features::make_move_label` と完全一致するよう実装した。
//!
//! ## 教師シグナル
//!
//! PackedSfenValue.move16 を bestmove として読み、 該当 label を 1.0 に
//! それ以外を 0.0 にする one-hot 教師。 visit dist 教師 (dlshogi hcpe3 系列)
//! は Phase 2 で扱う。
//!
//! ## RequiredDataType
//!
//! [`ShogiHalfKA_hm`] と同じ [`PackedSfenValue`]。 ShogiBoard を介さず PSV から
//! move16 と stm を直接取り出す (PSV byte 0 bit 0 = side_to_move、
//! bytes 34..36 = move16)。

use super::SparseInputType;
use super::shogi_halfka::ShogiHalfKA_hm;
use crate::shogi::PackedSfenValue;
use crate::shogi::types::Color;

/// policy label の総数 (dlshogi `make_move_label` の MAX_MOVE_LABEL_NUM)。
pub const POLICY_TARGET_DIMS: usize = 2187;

/// auxiliary dense input 名 (graph builder 側の `new_dense_input("policy_target", ...)`
/// と一致させる)。
pub const POLICY_TARGET_NAME: &str = "policy_target";

/// 移動方向数 (盤上 10 方向 × 不成/成 2 = 20)。
const MOVE_DIRECTION_NUM: u32 = 20;
/// マス数。
const SQUARE_NUM: usize = 81;
/// 持駒の種類数 (Pawn..Rook + Gold = 7)。
const HAND_PIECE_NUM: u32 = 7;

const _: () = {
    // 2187 = 81 × 27 のサニティチェック。
    let n = SQUARE_NUM * (MOVE_DIRECTION_NUM as usize + HAND_PIECE_NUM as usize);
    assert!(n == POLICY_TARGET_DIMS, "POLICY_TARGET_DIMS dimension check failed");
};

/// HalfKA_hm sparse + 2187-dim one-hot policy auxiliary。
///
/// 既存 [`ShogiHalfKA_hm`] と完全に同じ sparse 出力を返す。 追加で
/// `policy_target` という名前の dense auxiliary を 2187 dim で公開する。
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHmPolicy;

impl SparseInputType for ShogiHalfKaHmPolicy {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        ShogiHalfKA_hm.num_inputs()
    }

    /// 標準は 40 (合法局面の駒数固定)、 ただし combined_v17 / mekitto 等 .spk には
    /// 詰将棋・編集局面など >40 active を含む anomaly あり (loader panic 回避のため
    /// 60 に拡張、 memory 増加 50%)。
    fn max_active(&self) -> usize {
        60
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        ShogiHalfKA_hm.map_features(pos, f)
    }

    fn shorthand(&self) -> String {
        "halfka_hm_policy".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKA_hm sparse + 2187 one-hot policy auxiliary (Phase 1)".to_string()
    }

    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        vec![(POLICY_TARGET_NAME.to_string(), POLICY_TARGET_DIMS)]
    }

    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        if name != POLICY_TARGET_NAME {
            // 未知の auxiliary は dataloader 側で 0 で初期化済 (loader.rs:200)。
            return;
        }
        // out は呼び出し前に dataloader が 0 で埋めている (loader.rs:199)。
        // ここでは該当 label に 1.0 を立てるだけで良い。
        debug_assert_eq!(out.len(), POLICY_TARGET_DIMS);

        let mv = pos.move16();
        if mv == 0 {
            // bestmove 不在 (PSV 末尾局面 等) は均一 0 のまま (loss は mask 設計で扱う)。
            return;
        }
        let stm = side_to_move_from_psv(pos);
        if let Some(idx) = make_move_label(mv, stm) {
            // 範囲外は無視 (理論上発生しない、 万一の異常 record 対策)。
            if idx < POLICY_TARGET_DIMS {
                out[idx] = 1.0;
            }
        }
    }
}

/// PSV の packed_sfen バイト 0 の最下位 bit が side_to_move (`0`=先手 / `1`=後手)。
#[inline]
fn side_to_move_from_psv(pos: &PackedSfenValue) -> Color {
    let stm_bit = pos.sfen().data[0] & 1;
    if stm_bit == 0 { Color::Black } else { Color::White }
}

/// move16 (やねうら王 / Bullet shogipack 仕様) を 2187 政策ラベルに変換する。
///
/// 仕様詳細はモジュールドックの "policy index encoding" を参照。
/// rshogi `tools::dlshogi_features::make_move_label` と数値完全一致。
/// 不正 move (to/from 範囲外、 piece type 範囲外) は `None`。
pub fn make_move_label(move16: u16, color: Color) -> Option<usize> {
    let to = (move16 & 0x7F) as u8;
    let from_or_pt = ((move16 >> 7) & 0x7F) as u8;
    let promote = (move16 & 0x4000) != 0;

    let is_white = color == Color::White;

    if from_or_pt >= 81 {
        // 持駒の打ち。
        // move16 仕様: from_or_pt = 81 + pt_index、 pt_index ∈ 1..=7 (cshogi)
        let pt_index = from_or_pt - 81;
        if !(1..=7).contains(&pt_index) {
            return None;
        }
        // dlshogi label: hand_piece = pt-1 (Pawn=0 .. Rook=5、 Gold=6)
        let hand_piece = if pt_index <= 6 { pt_index - 1 } else { 6 } as u32;

        if to >= 81 {
            return None;
        }
        let to_sq = if is_white { 80 - to as usize } else { to as usize };

        let direction = MOVE_DIRECTION_NUM + hand_piece;
        Some(SQUARE_NUM * direction as usize + to_sq)
    } else {
        // 盤上移動。
        let from = from_or_pt;
        if to >= 81 || from >= 81 {
            return None;
        }
        let (to_sq, from_sq) =
            if is_white { (80 - to as usize, 80 - from as usize) } else { (to as usize, from as usize) };
        let to_file = to_sq / 9;
        let to_rank = to_sq % 9;
        let from_file = from_sq / 9;
        let from_rank = from_sq % 9;
        let dir_x = from_file as i32 - to_file as i32;
        let dir_y = to_rank as i32 - from_rank as i32;

        let mut direction = get_move_direction(dir_x, dir_y);
        if promote {
            direction += 10;
        }
        Some(SQUARE_NUM * direction as usize + to_sq)
    }
}

/// dlshogi `MoveDirection` 互換の方向 index を返す。
///
/// `dir_x = from_file - to_file`、 `dir_y = to_rank - from_rank` (cshogi と同じ符号)。
/// 戻り値:
/// - 0: Up
/// - 1: UpLeft
/// - 2: UpRight
/// - 3: Left
/// - 4: Right
/// - 5: Down
/// - 6: DownLeft
/// - 7: DownRight
/// - 8: Up2Left (桂)
/// - 9: Up2Right (桂)
fn get_move_direction(dir_x: i32, dir_y: i32) -> u32 {
    if dir_y < 0 && dir_x == 0 {
        0 // Up
    } else if dir_y == -2 && dir_x == -1 {
        8 // Up2Left
    } else if dir_y == -2 && dir_x == 1 {
        9 // Up2Right
    } else if dir_y < 0 && dir_x < 0 {
        1 // UpLeft
    } else if dir_y < 0 && dir_x > 0 {
        2 // UpRight
    } else if dir_y == 0 && dir_x < 0 {
        3 // Left
    } else if dir_y == 0 && dir_x > 0 {
        4 // Right
    } else if dir_y > 0 && dir_x == 0 {
        5 // Down
    } else if dir_y > 0 && dir_x < 0 {
        6 // DownLeft
    } else {
        7 // DownRight (dir_y > 0 && dir_x > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_dims_consistency() {
        assert_eq!(POLICY_TARGET_DIMS, 2187);
        let total = SQUARE_NUM * (MOVE_DIRECTION_NUM as usize + HAND_PIECE_NUM as usize);
        assert_eq!(total, POLICY_TARGET_DIMS);
    }

    /// 7g7f (Black、 from=60 file=6/rank=6、 to=59 file=6/rank=5) → direction=Up=0、 to_sq=59、 label = 59
    /// (USI 7g7f を cshogi 形式に直訳すると from=60(=6*9+6)、 to=59(=6*9+5) になる)
    #[test]
    fn label_pawn_up_black() {
        let move16: u16 = 59 | (60 << 7);
        let label = make_move_label(move16, Color::Black).expect("should encode");
        // direction = 0 (Up)、 to_sq = 59
        assert_eq!(label, 59);
    }

    /// White の正規パウンプッシュ (White 3c3d 相当、 White 側の歩を 1 マス
    /// 前進、 つまり absolute sq で言えば file=2 rank=2 → file=2 rank=3) を
    /// 想定する。 dlshogi 規約では Color::White の move16 は absolute 座標で
    /// 与えられ、 ラベル encoding 内部で 80 − sq へ反転して STM 視点に正規化
    /// する。
    ///
    /// from=20 (file=2, rank=2)、 to=21 (file=2, rank=3)、 不成。
    /// 反転後: to_sq = 80-21 = 59、 from_sq = 80-20 = 60。
    /// to_file=6, to_rank=5, from_file=6, from_rank=6 → dir_x=0, dir_y=-1
    /// → direction = Up = 0。 label = 81*0 + 59 = 59。
    ///
    /// つまり「White が前進する歩」 と 「Black が前進する歩」 は同一ラベル 59 に
    /// 落ちる (STM 視点で正規化されているため)。
    #[test]
    fn label_pawn_forward_white_normalized_to_up() {
        let from: u16 = 20;
        let to: u16 = 21;
        let move16: u16 = to | (from << 7);
        let label = make_move_label(move16, Color::White).expect("should encode");
        assert_eq!(label, 59);
    }

    /// 歩の打ち (P*5e、 to=40、 piece_idx=1) は direction = MOVE_DIRECTION_NUM + 0 = 20。
    /// label = 81*20 + 40 = 1660。
    #[test]
    fn label_pawn_drop_black() {
        let from_or_pt = 81 + 1; // Pawn drop
        let move16: u16 = 40 | ((from_or_pt as u16) << 7);
        let label = make_move_label(move16, Color::Black).expect("should encode");
        assert_eq!(label, 81 * 20 + 40);
    }

    /// 成りフラグ立て (direction += 10)。 from=60→to=59 を成りで encode → direction=10、 label=81*10+59 = 869。
    #[test]
    fn label_promote_adds_10() {
        let mut move16: u16 = 59 | (60 << 7);
        move16 |= 0x4000;
        let label = make_move_label(move16, Color::Black).expect("should encode");
        assert_eq!(label, 81 * 10 + 59);
    }

    /// move16=0 (bestmove なし) は `None` ではなく `Some(...)` を返す可能性があるため、
    /// 呼び側 (`fill_auxiliary_dense`) で move16==0 early return する設計を確認する。
    /// ここでは make_move_label 自体が `Some` を返しても OK な範囲を示す: from=0/to=0
    /// → direction=Down(=5、 dir_y=0 dir_x=0 は DownRight 7 だが厳密には origin の差で 7)、
    /// 詳細は呼び側で 0 ガード。
    #[test]
    fn label_invalid_out_of_range_returns_none() {
        // to=82 (out of range) はエンコード失敗。
        let move16: u16 = 82 | (50 << 7);
        assert!(make_move_label(move16, Color::Black).is_none());
    }
}
