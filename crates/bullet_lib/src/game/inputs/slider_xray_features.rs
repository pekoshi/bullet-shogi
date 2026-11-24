//! Slider X-ray (SXRY) sparse feature 計算 (= 2026-06-28)。
//!
//! 推論側 `rshogi-core::nnue::slider_xray` と完全一致する index 契約を保証する。
//! 既存 Attack Development (ADEV) / King Danger (KDGR) head の完全ミラー
//! (= feature 計算関数のみ差替え)。
//!
//! # 経路
//!
//! 1. bullet 側 `ShogiBoard` (= PSV 展開) を SFEN string に変換
//! 2. `rshogi-core::Position::set_sfen` で Position 構築
//! 3. `rshogi-core::nnue::slider_xray::compute_slider_xray_class` で feature 計算
//! 4. `local_idx(bucket, class)` で per-persp sparse index
//!
//! feature 計算を rshogi-core に委譲することで訓練 / 推論で bit-identical を保証する
//! (= ADEV / KDGR と同方式)。
//!
//! # weight layout (= per persp)
//!
//! - per_perspective dim = bucket × class = 8 × 8 = 64
//! - total = 2 perspective × 64 = 128

use crate::shogi::packed_sfen::ShogiBoard;

/// bucket 数 (= PSQT bucket と同期、 8 = progress8 系)
pub const SLIDER_XRAY_NUM_BUCKETS: usize = 8;
/// slider_xray class 数 (= 0..=7 の 8 class)
pub const SLIDER_XRAY_NUM_CLASS: usize = 8;
/// perspective 数
pub const SLIDER_XRAY_NUM_PERSP: usize = 2;

/// 1 perspective 分の disk dim (= bucket × class = 64)
pub const SLIDER_XRAY_DIM_PER_PERSP: usize = SLIDER_XRAY_NUM_BUCKETS * SLIDER_XRAY_NUM_CLASS;

/// 全 weight 数 = 2 × 64 = 128
pub const SLIDER_XRAY_TOTAL_COUNT: u32 = (2 * SLIDER_XRAY_DIM_PER_PERSP) as u32;

/// per-perspective active 数 (= 1 fire only)
pub const SLIDER_XRAY_MAX_ACTIVE_PER_PERSP: usize = 1;

/// disk section magic (= 推論側と完全一致、 P2SXRY + null padding)
pub const SLIDER_XRAY_SUB_MAGIC: &[u8; 8] = b"P2SXRY\0\0";

/// section version (= 1、 2026-06-28)。
pub const SLIDER_XRAY_SECTION_VERSION: u32 = 1;

/// 量子化スケール (= QA = 127、 ADEV / KDGR と同条件)
pub const SLIDER_XRAY_QA: f64 = 127.0;

/// architecture marker (= ckpt log 用)
pub const SLIDER_XRAY_ARCH_MARKER: &str = "SXRY=1";

/// sparse input 名 (= 内部識別、 per persp 1 active)
pub const SLIDER_XRAY_SPARSE_NAME_BLACK: &str = "slider_xray_black";
pub const SLIDER_XRAY_SPARSE_NAME_WHITE: &str = "slider_xray_white";

/// 自玉への敵 slider X-ray 照準 class を rshogi-core 経由で計算する。
///
/// `persp_is_black` = true なら black perspective、 false なら white perspective。
pub fn compute_class_for_persp(board: &ShogiBoard, persp_is_black: bool) -> usize {
    use rshogi_core::nnue::slider_xray::compute_slider_xray_class;
    use rshogi_core::position::Position;
    use rshogi_core::types::Color;

    let sfen = board.to_sfen();
    let mut pos = Position::new();
    pos.set_sfen(&sfen).expect("valid sfen");

    let color = if persp_is_black { Color::Black } else { Color::White };
    compute_slider_xray_class(&pos, color)
}

/// per-persp sparse input 内の local idx (= bucket × class flatten)
///
/// disk layout `[bucket][persp][class]` と整合させる: per-persp side では
/// `[bucket][class]` で row-major (= bucket-outer)。
#[inline]
pub const fn local_idx(bucket: usize, class: usize) -> usize {
    bucket * SLIDER_XRAY_NUM_CLASS + class
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(SLIDER_XRAY_NUM_BUCKETS, 8);
        assert_eq!(SLIDER_XRAY_NUM_CLASS, 8);
        assert_eq!(SLIDER_XRAY_DIM_PER_PERSP, 64);
        assert_eq!(SLIDER_XRAY_TOTAL_COUNT, 128);
        assert_eq!(SLIDER_XRAY_MAX_ACTIVE_PER_PERSP, 1);
        assert_eq!(SLIDER_XRAY_SUB_MAGIC, b"P2SXRY\0\0");
        assert_eq!(SLIDER_XRAY_SECTION_VERSION, 1);
    }

    #[test]
    fn test_local_idx() {
        assert_eq!(local_idx(0, 0), 0);
        assert_eq!(local_idx(7, 7), 63);
        assert_eq!(local_idx(7, 7) + 1, SLIDER_XRAY_DIM_PER_PERSP);
    }
}
