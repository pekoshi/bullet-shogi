//! ShogiHalfKaHmHandCount — HalfKA_hm + HandCount Dense 補助入力
//!
//! 盤・駒・王の情報は従来の HalfKA_hm と完全に同じ sparse 表現で提供し、
//! 追加で 14 元の dense vector (= stm 持ち駒 7 種 + nstm 持ち駒 7 種) を
//! `"hand_count"` という名前の dense 入力として公開する。
//!
//! L1 層の入力に `[FT_output (1536), hand_count (14)]` という形で concat して
//! 使うことを想定している。
//!
//! レイアウト順は以下の 14 元:
//!
//! ```text
//! index 0..6: stm 視点の持ち駒本数 (歩, 香, 桂, 銀, 金, 角, 飛)
//! index 7..13: nstm 視点の持ち駒本数 (歩, 香, 桂, 銀, 金, 角, 飛)
//! ```
//!
//! 値は `u8` の駒本数を `f32` にキャストしたもの（0.0〜最大 18.0）。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES};
use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, HAND_PIECE_TYPES},
};

/// HandCount dense 補助入力の次元数。
///
/// stm/nstm 視点 × 7 駒種（歩香桂銀金角飛）= 14。
pub const HAND_COUNT_DIMS: usize = 14;

/// ShogiHalfKA_hm + HandCount Dense 補助入力を提供する SparseInputType。
///
/// sparse feature は HalfKA_hm と完全互換（73,305 次元、max_active 40）。
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHmHandCount;

impl SparseInputType for ShogiHalfKaHmHandCount {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_FEATURES
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        // sparse 部分は HalfKA_hm と完全に同一なので delegate。
        super::ShogiHalfKA_hm.map_features(pos, f);
    }

    fn shorthand(&self) -> String {
        "shogi-73305x45hm+handcount14".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKA_hm + HandCount Dense (14 dims, stm/nstm hand piece counts)".to_string()
    }

    fn hand_count_dims(&self) -> Option<usize> {
        Some(HAND_COUNT_DIMS)
    }

    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        assert_eq!(out.len(), HAND_COUNT_DIMS, "hand_count 出力スロットは 14 次元");

        let board = ShogiBoard::from_packed_sfen(pos);
        let stm = board.side_to_move;
        let nstm = stm.opponent();

        // 2026-11-07 (T1 / fable 設計 DESIGN-20261107-HANDFEAT-FABLE.md §3 候補 A):
        //   **素の枚数 (0.0〜18.0) をそのまま出してはいけない**。
        //   エンジンの fc0 は u8 x i8 の `maddubs` で、 u8 lane は **126 以下**でなければ
        //   飽和不変条件 (a <= 126 かつ |w| <= 127 で 2 項の和 <= 32,004) が壊れる。
        //   「1.0 = 127」の目盛りなので、 枚数 c を素で送ると 127c で即破綻する。
        //
        //   そこで駒種ごとの係数 k を掛け、 **u8 の値が k*c (<= 126)** になるようにする:
        //     歩 7 (最大 18 枚 -> 126) / 香 桂 銀 金 31 (最大 4 枚 -> 124)
        //     角 飛 63 (最大 2 枚 -> 126)
        //   学習器は `k*c / 127` を出し、 エンジンは `u8 = k*c` を送る -> 目盛りが一致する。
        //   🚨HAND_PIECE_TYPES の順は **歩 香 桂 銀 金 角 飛**。 YO の PieceType 順
        //     (歩 香 桂 銀 **角 飛 金**) とは違うので、 エンジン側 (E2) は必ずこの順で埋める。
        const HC_SCALE: [f32; 7] = [7.0, 31.0, 31.0, 31.0, 31.0, 63.0, 63.0];
        let write = |out: &mut [f32], offset: usize, color: Color, board: &ShogiBoard| {
            for (i, &pt) in HAND_PIECE_TYPES.iter().enumerate() {
                out[offset + i] = HC_SCALE[i] * f32::from(board.hand(color).count(pt)) / 127.0;
            }
        };

        write(out, 0, stm, &board);
        write(out, 7, nstm, &board);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::PieceType;

    fn dummy_pos_with_hands(black_counts: [u8; 7], white_counts: [u8; 7], stm: Color) -> PackedSfenValue {
        // packed から board を作るのは重いので、ShogiBoard を直接構築して
        // fill_hand_count に渡せる形を作る簡易テスト。
        // ここでは存在するコードパスを呼びたいだけなので board は直接操作。
        let _ = (black_counts, white_counts, stm);
        PackedSfenValue::default()
    }

    #[test]
    fn dimensions_match_halfka_hm() {
        let input = ShogiHalfKaHmHandCount;
        assert_eq!(input.num_inputs(), HALFKA_HM_DIMENSIONS);
        assert_eq!(input.max_active(), MAX_ACTIVE_FEATURES);
        assert_eq!(input.hand_count_dims(), Some(14));
    }

    #[test]
    fn fill_hand_count_writes_14_elements() {
        let input = ShogiHalfKaHmHandCount;
        let pos = dummy_pos_with_hands([0; 7], [0; 7], Color::Black);
        let mut out = [f32::NAN; 14];
        input.fill_hand_count(&pos, &mut out);
        // 全値 0 or 有効な f32。NaN が残っていない。
        for v in out.iter() {
            assert!(!v.is_nan(), "hand_count が未書き込み");
            assert!(*v >= 0.0, "hand_count は非負");
        }
    }

    #[test]
    fn hand_piece_types_len_matches_half() {
        // HAND_PIECE_TYPES は 7 種（歩香桂銀金角飛）。
        let pts: Vec<PieceType> = HAND_PIECE_TYPES.to_vec();
        assert_eq!(pts.len(), HAND_COUNT_DIMS / 2);
    }
}
