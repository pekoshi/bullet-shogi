//! King Danger (KDGR) sparse feature 計算 (= 2026-06-15)。
//!
//! 推論側 `rshogi-core::nnue::king_danger` と完全一致する index 契約を保証する。
//! 既存 King Escape (KED) head の完全ミラー (= class 数だけ 4 → 8 に変更)。
//!
//! # 経路
//!
//! 1. bullet 側 `ShogiBoard` (= PSV 展開) を SFEN string に変換
//! 2. `rshogi-core::Position::set_sfen` で Position 構築
//! 3. `rshogi-core::nnue::king_danger::compute_king_danger_class` で feature 計算
//! 4. `local_idx(bucket, class)` で per-persp sparse index
//!
//! # weight layout (= per persp)
//!
//! - per_perspective dim = bucket × class = 8 × 8 = 64
//! - total = 2 perspective × 64 = 128
//!
//! NPS overhead: per-eval で SFEN parse + Position 構築が走る (= 訓練時のみ)。
//! 推論側 (= rshogi-usi) は直接 Position から計算するため bullet 経路は学習時のみ。

use crate::shogi::packed_sfen::ShogiBoard;

/// bucket 数 (= PSQT bucket と同期、 8 = progress8 系)
pub const KING_DANGER_NUM_BUCKETS: usize = 8;
/// king_danger class 数 (= 0..=7 の 8 class)
pub const KING_DANGER_NUM_CLASS: usize = 8;
/// perspective 数
pub const KING_DANGER_NUM_PERSP: usize = 2;

/// 1 perspective 分の disk dim (= bucket × class = 64)
pub const KING_DANGER_DIM_PER_PERSP: usize = KING_DANGER_NUM_BUCKETS * KING_DANGER_NUM_CLASS;

/// 全 weight 数 = 2 × 64 = 128
pub const KING_DANGER_TOTAL_COUNT: u32 = (2 * KING_DANGER_DIM_PER_PERSP) as u32;

/// per-perspective active 数 (= 1 fire only)
pub const KING_DANGER_MAX_ACTIVE_PER_PERSP: usize = 1;

/// disk section magic (= 推論側と完全一致、 P2KDGR + null padding)
pub const KING_DANGER_SUB_MAGIC: &[u8; 8] = b"P2KDGR\0\0";

/// section version (= 1、 2026-06-15)。
pub const KING_DANGER_SECTION_VERSION: u32 = 1;

/// 量子化スケール (= QA = 127、 KingEscape と同条件)
pub const KING_DANGER_QA: f64 = 127.0;

/// architecture marker (= ckpt log 用)
pub const KING_DANGER_ARCH_MARKER: &str = "KDGR=1";

/// sparse input 名 (= 内部識別、 per persp 1 active)
pub const KING_DANGER_SPARSE_NAME_BLACK: &str = "king_danger_black";
pub const KING_DANGER_SPARSE_NAME_WHITE: &str = "king_danger_white";

/// 自玉危険 class を rshogi-core 経由で計算する。
///
/// `persp_is_black` = true なら black perspective、 false なら white perspective。
pub fn compute_class_for_persp(board: &ShogiBoard, persp_is_black: bool) -> usize {
    use rshogi_core::nnue::king_danger::compute_king_danger_class;
    use rshogi_core::position::Position;
    use rshogi_core::types::Color;

    let sfen = board.to_sfen();
    let mut pos = Position::new();
    pos.set_sfen(&sfen).expect("valid sfen");

    let color = if persp_is_black { Color::Black } else { Color::White };
    compute_king_danger_class(&pos, color)
}

/// per-persp sparse input 内の local idx (= bucket × class flatten)
///
/// disk layout `[bucket][persp][class]` と整合させる: per-persp side では
/// `[bucket][class]` で row-major (= bucket-outer)。
#[inline]
pub const fn local_idx(bucket: usize, class: usize) -> usize {
    bucket * KING_DANGER_NUM_CLASS + class
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(KING_DANGER_NUM_BUCKETS, 8);
        assert_eq!(KING_DANGER_NUM_CLASS, 8);
        assert_eq!(KING_DANGER_DIM_PER_PERSP, 64);
        assert_eq!(KING_DANGER_TOTAL_COUNT, 128);
        assert_eq!(KING_DANGER_MAX_ACTIVE_PER_PERSP, 1);
        assert_eq!(KING_DANGER_SUB_MAGIC, b"P2KDGR\0\0");
        assert_eq!(KING_DANGER_SECTION_VERSION, 1);
    }

    #[test]
    fn test_local_idx() {
        assert_eq!(local_idx(0, 0), 0);
        assert_eq!(local_idx(7, 7), 63);
        assert_eq!(local_idx(7, 7) + 1, KING_DANGER_DIM_PER_PERSP);
    }
}
