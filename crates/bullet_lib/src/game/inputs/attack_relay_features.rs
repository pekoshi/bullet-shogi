//! Attack Relay Head (ARH) sparse feature 計算 (= 2026-06-09)。
//!
//! 推論側 `rshogi-core::nnue::attack_relay` と完全一致する index 契約を保証する。
//!
//! 仕様: 敵玉中心 5x5 を inner (= 3x3) + outer ring に分け、 攻撃側 perspective の
//! (contact, reserve, relay) 3 軸を 4×4×4 = 64 cell に量子化、 8 bucket × 2 persp で
//! 1024 i16 weight (= 2 KiB)。 詳細は p2shogi 側
//! `docs/attack_relay_head_design.md`。
//!
//! # 経路 (= 既存 KED と同 pattern、 SFEN delegation)
//!
//! 1. bullet 側 `ShogiBoard` (= PSV 展開) を SFEN string に変換
//! 2. `rshogi-core::Position::set_sfen` で Position 構築
//! 3. `rshogi-core::nnue::attack_relay::compute_arh_classes` で 3-tuple 計算
//! 4. `local_idx(bucket, contact, reserve, relay)` で per-persp sparse index
//!
//! # weight layout (= per persp)
//!
//! - per_perspective dim = bucket × contact × reserve × relay = 8 × 4 × 4 × 4 = 512
//! - total = 2 perspective × 512 = 1024
//!
//! NPS overhead: per-eval で SFEN parse + Position 構築が走る (= 訓練時のみ)。

use crate::shogi::packed_sfen::ShogiBoard;

/// bucket 数 (= PSQT bucket と同期、 8 = progress8 系)
pub const ARH_NUM_BUCKETS: usize = 8;
/// perspective 数
pub const ARH_NUM_PERSP: usize = 2;
/// contact class 数 (= 0 / 1 / 2 / 3+)
pub const ARH_NUM_CONTACT: usize = 4;
/// reserve class 数 (= 0 / 1 / 2 / 3+)
pub const ARH_NUM_RESERVE: usize = 4;
/// relay class 数 (= 0 / 1 / 2 / 3+)
pub const ARH_NUM_RELAY: usize = 4;

/// 1 perspective 分の disk dim (= bucket × contact × reserve × relay = 512)
pub const ARH_DIM_PER_PERSP: usize = ARH_NUM_BUCKETS * ARH_NUM_CONTACT * ARH_NUM_RESERVE * ARH_NUM_RELAY;

/// 全 weight 数 = 2 × 512 = 1024
pub const ARH_TOTAL_COUNT: u32 = (2 * ARH_DIM_PER_PERSP) as u32;

/// per-perspective active 数 (= 1 fire only)
pub const ARH_MAX_ACTIVE_PER_PERSP: usize = 1;

/// disk section magic (= 推論側と完全一致)
pub const ARH_SUB_MAGIC: &[u8; 8] = b"P2AREL\0\0";

/// section version (= 1、 2026-06-09)
pub const ARH_SECTION_VERSION: u32 = 1;

/// 量子化スケール (= QA = 127、 KED / PairShape と同条件)
pub const ARH_QA: f64 = 127.0;

/// architecture marker (= ckpt log 用)
pub const ARH_ARCH_MARKER: &str = "ARH=1";

/// sparse input 名 (= 内部識別、 per persp 1 active)
pub const ARH_SPARSE_NAME_BLACK: &str = "attack_relay_black";
pub const ARH_SPARSE_NAME_WHITE: &str = "attack_relay_white";

/// 攻撃側 perspective の (contact, reserve, relay) 3-tuple を rshogi-core 経由で計算する。
///
/// `persp_is_black` = true なら black perspective、 false なら white perspective。
pub fn compute_classes_for_persp(board: &ShogiBoard, persp_is_black: bool) -> (usize, usize, usize) {
    use rshogi_core::nnue::attack_relay::compute_arh_classes;
    use rshogi_core::position::Position;
    use rshogi_core::types::Color;

    let sfen = board.to_sfen();
    let mut pos = Position::new();
    pos.set_sfen(&sfen).expect("valid sfen");

    let color = if persp_is_black { Color::Black } else { Color::White };
    compute_arh_classes(&pos, color)
}

/// per-persp sparse input 内の local idx。
///
/// `bucket × 64 + contact × 16 + reserve × 4 + relay`、 推論側
/// `rshogi-core::nnue::attack_relay::local_idx` と **完全同一**。
#[inline]
pub const fn local_idx(bucket: usize, contact: usize, reserve: usize, relay: usize) -> usize {
    ((bucket * ARH_NUM_CONTACT + contact) * ARH_NUM_RESERVE + reserve) * ARH_NUM_RELAY + relay
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(ARH_NUM_BUCKETS, 8);
        assert_eq!(ARH_NUM_PERSP, 2);
        assert_eq!(ARH_NUM_CONTACT, 4);
        assert_eq!(ARH_NUM_RESERVE, 4);
        assert_eq!(ARH_NUM_RELAY, 4);
        assert_eq!(ARH_DIM_PER_PERSP, 512);
        assert_eq!(ARH_TOTAL_COUNT, 1024);
        assert_eq!(ARH_MAX_ACTIVE_PER_PERSP, 1);
        assert_eq!(ARH_SUB_MAGIC, b"P2AREL\0\0");
        assert_eq!(ARH_SECTION_VERSION, 1);
    }

    #[test]
    fn test_local_idx_bounds() {
        assert_eq!(local_idx(0, 0, 0, 0), 0);
        assert_eq!(local_idx(7, 3, 3, 3), 511);
        assert_eq!(local_idx(7, 3, 3, 3) + 1, ARH_DIM_PER_PERSP);
    }

    #[test]
    fn test_local_idx_unique() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..ARH_NUM_BUCKETS {
            for c in 0..ARH_NUM_CONTACT {
                for r in 0..ARH_NUM_RESERVE {
                    for re in 0..ARH_NUM_RELAY {
                        let v = local_idx(b, c, r, re);
                        assert!(v < ARH_DIM_PER_PERSP);
                        assert!(seen.insert(v));
                    }
                }
            }
        }
        assert_eq!(seen.len(), ARH_DIM_PER_PERSP);
    }

    #[test]
    fn test_local_idx_matches_rshogi() {
        // rshogi-core 側 nnue-attack-relay feature が ON で build されている場合、
        // 同 index 計算と一致することを確認 (= cross-impl parity 第一段)。
        // feature OFF だと rshogi 側 symbol が無く本テストは無効化。
        #[cfg(feature = "nnue-attack-relay")]
        {
            use rshogi_core::nnue::attack_relay as rsh;
            for b in 0..ARH_NUM_BUCKETS {
                for c in 0..ARH_NUM_CONTACT {
                    for r in 0..ARH_NUM_RESERVE {
                        for re in 0..ARH_NUM_RELAY {
                            assert_eq!(local_idx(b, c, r, re), rsh::local_idx(b, c, r, re));
                        }
                    }
                }
            }
        }
    }
}
