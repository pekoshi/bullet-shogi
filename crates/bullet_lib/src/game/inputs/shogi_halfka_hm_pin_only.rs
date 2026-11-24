//! ShogiHalfKaHmPin — HalfKA_hm + HandCount + Pin-aware dense 補助入力
//!
//! 2026-05-25: LMAS 廃止に伴う後継 wrapper。 元の `ShogiHalfKaHmLMAS` は
//! LMAS + Pin の混合 wrapper だったが、 LMAS を物理削除した結果として
//! 残った "HandCount + Pin" 構成を担う。
//!
//! - sparse 部: `ShogiHalfKaHmHandCount` (= HalfKA_hm 73,305 dim + hand_count dense 14 dim) に delegate。
//! - 補助 dense 入力 (`pin_aware = true` のとき):
//!   - `pin_features_black` (2,268 dim)
//!   - `pin_features_white` (2,268 dim)
//!
//! pin_aware が false なら HandCount + hand_count dense と同じ振舞 (= 既存
//! 後方互換)。 LMAS との混在 path は廃止済のため shorthand に `+pin` 以外の
//! サフィックスは付かない。

use super::SparseInputType;
use super::shogi_halfka_hm_hand_count::ShogiHalfKaHmHandCount;
use super::shogi_halfka_hm_pin::{pin_aux_entries, try_fill_pin_aux};
use crate::shogi::PackedSfenValue;

/// HalfKA_hm + HandCount + 任意 Pin-aware の sparse input wrapper。
#[derive(Clone, Debug, Default)]
pub struct ShogiHalfKaHmPin {
    inner: ShogiHalfKaHmHandCount,
    /// Pin-aware 補助入力を有効化するか。
    ///
    /// `true` のとき `auxiliary_dense_dims` に `pin_features_black/white` (各 2,268 dim)
    /// を追加し、 fill 時に `pin_features::fill_pin_features_for_side` で dense vector
    /// を生成する。
    pin_aware: bool,
}

impl ShogiHalfKaHmPin {
    /// pin_aware off (= HandCount のみ) で構築。
    pub fn new() -> Self {
        Self::default()
    }

    /// pin_aware on で構築する builder。
    pub fn with_pin_aware(mut self, enabled: bool) -> Self {
        self.pin_aware = enabled;
        self
    }

    /// pin_aware の有効状態を返す。
    pub fn pin_aware(&self) -> bool {
        self.pin_aware
    }
}

impl SparseInputType for ShogiHalfKaHmPin {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        self.inner.num_inputs()
    }

    fn max_active(&self) -> usize {
        self.inner.max_active()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        self.inner.map_features(pos, f);
    }

    fn shorthand(&self) -> String {
        let pin_part = if self.pin_aware { "+pin" } else { "" };
        format!("shogi-73305x45hm+handcount14{}", pin_part)
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm + HandCount{}",
            if self.pin_aware { " + Pin-aware (2,268 × 2 perspective)" } else { "" }
        )
    }

    fn hand_count_dims(&self) -> Option<usize> {
        self.inner.hand_count_dims()
    }

    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        self.inner.fill_hand_count(pos, out);
    }

    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        // hand_count を先頭に置く (後方互換)。 続けて pin_aware なら pin 入力を追加。
        let mut dims: Vec<(String, usize)> = self.inner.auxiliary_dense_dims();
        if self.pin_aware {
            for (name, d) in pin_aux_entries().iter() {
                dims.push(((*name).to_string(), *d));
            }
        }
        dims
    }

    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        // pin_aware なら pin route を先に試す。 該当なしは inner (hand_count) に delegate。
        if self.pin_aware && try_fill_pin_aux(name, pos, out) {
            return;
        }
        self.inner.fill_auxiliary_dense(name, pos, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_no_pin() {
        let input = ShogiHalfKaHmPin::default();
        assert!(!input.pin_aware());
        let dims = input.auxiliary_dense_dims();
        assert_eq!(dims.len(), 1);
        assert_eq!(dims[0].0, "hand_count");
    }

    #[test]
    fn with_pin_adds_two_entries() {
        let input = ShogiHalfKaHmPin::new().with_pin_aware(true);
        assert!(input.pin_aware());
        let dims = input.auxiliary_dense_dims();
        assert_eq!(dims.len(), 3, "hand_count + pin_black + pin_white");
        assert_eq!(dims[0].0, "hand_count");
        assert_eq!(dims[1].0, "pin_features_black");
        assert_eq!(dims[1].1, 2268);
        assert_eq!(dims[2].0, "pin_features_white");
        assert_eq!(dims[2].1, 2268);
    }

    #[test]
    fn shorthand_reflects_pin() {
        assert_eq!(ShogiHalfKaHmPin::new().shorthand(), "shogi-73305x45hm+handcount14");
        assert_eq!(ShogiHalfKaHmPin::new().with_pin_aware(true).shorthand(), "shogi-73305x45hm+handcount14+pin");
    }
}
