//! ShogiHalfKaHmPolicySoft — HalfKA_hm sparse + visit distribution **soft** policy target。
//!
//! Phase 2 visit dist 教師訓練用 input type。
//! [`ShogiHalfKaHmPolicy`] (one-hot 版) との違い:
//! - RequiredDataType = [`PackedSfenWithVisits`] (172 bytes、 PSV + top-32 visits)
//! - fill_auxiliary_dense は **visit_count を正規化した確率分布** を 2187 dim に展開
//! - softmax_crossentropy_loss は soft target をそのまま受け取る (KL に等価)
//!
//! sparse 部分は ShogiHalfKA_hm と同じ encoding を **PackedSfenValue** (psv() 経由) に
//! 委譲する。

use super::SparseInputType;
use super::shogi_halfka::ShogiHalfKA_hm;
use super::shogi_halfka_hm_policy::{POLICY_TARGET_DIMS, make_move_label};
use crate::shogi::PackedSfenWithVisits;
use crate::shogi::types::Color;
use std::sync::OnceLock;

/// label smoothing α を env var `BULLET_LABEL_SMOOTH` から取得 (default 0.0)。
/// target = (1-α) * visit_dist + α / POLICY_TARGET_DIMS。
/// top-1 ranking 改善目的 (v22_big-20 top-1 36.26% から +pp 狙い、 2026-05-18)。
fn label_smooth_alpha() -> f32 {
    static ALPHA: OnceLock<f32> = OnceLock::new();
    *ALPHA.get_or_init(|| std::env::var("BULLET_LABEL_SMOOTH").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0))
}

/// auxiliary dense input 名 (one-hot 版とは別)。 graph builder で
/// `new_dense_input("policy_visit_dist", ...)` として参照する。
pub const POLICY_VISIT_DIST_NAME: &str = "policy_visit_dist";

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ShogiHalfKaHmPolicySoft;

impl SparseInputType for ShogiHalfKaHmPolicySoft {
    type RequiredDataType = PackedSfenWithVisits;

    fn num_inputs(&self) -> usize {
        ShogiHalfKA_hm.num_inputs()
    }

    fn max_active(&self) -> usize {
        // ShogiHalfKaHmPolicy と同じ理由で 60 に拡張。
        60
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        // PSV 部分に委譲。
        ShogiHalfKA_hm.map_features(pos.psv(), f)
    }

    fn shorthand(&self) -> String {
        "halfka_hm_policy_soft".to_string()
    }

    fn description(&self) -> String {
        "Shogi HalfKA_hm sparse + 2187 SOFT policy distribution (Phase 2 visit dist)".to_string()
    }

    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        vec![(POLICY_VISIT_DIST_NAME.to_string(), POLICY_TARGET_DIMS)]
    }

    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        if name != POLICY_VISIT_DIST_NAME {
            return;
        }
        debug_assert_eq!(out.len(), POLICY_TARGET_DIMS);
        // out は dataloader 側で 0 fill 済み。 ここでは visit_dist を確率に正規化し、
        // 該当 label の slot に書き込む。

        let total: u32 = pos.iter_visits().map(|m| m.visit as u32).sum();
        if total == 0 {
            // visit 0 (fallback): 既存 PSV の move16 を one-hot にする。
            let stm = side_to_move_from_psv(pos);
            let mv = pos.move16();
            if mv != 0 {
                if let Some(idx) = make_move_label(mv, stm) {
                    if idx < POLICY_TARGET_DIMS {
                        out[idx] = 1.0;
                    }
                }
            }
            return;
        }

        let stm = side_to_move_from_psv(pos);
        // 2-pass: (1) valid moves の visit 総和を計算、 (2) その総和で正規化。
        // raw total を使うと garbage move (encoder reject) を母数に含めてしまい、
        // target sum < 1.0 になり CE 信号が削減される (debug session 2026-05-18 で発覚、
        // gct hcpe3 aggregation で alignment ずれ由来の 27% bad moves 確認)。
        let mut valid_total: u32 = 0;
        for mv in pos.iter_visits() {
            if mv.visit == 0 {
                continue;
            }
            if let Some(idx) = make_move_label(mv.move16, stm) {
                if idx < POLICY_TARGET_DIMS {
                    valid_total += mv.visit as u32;
                }
            }
        }
        if valid_total == 0 {
            return;
        }
        let inv_valid_total = 1.0f32 / valid_total as f32;
        for mv in pos.iter_visits() {
            if mv.visit == 0 {
                continue;
            }
            if let Some(idx) = make_move_label(mv.move16, stm) {
                if idx < POLICY_TARGET_DIMS {
                    out[idx] += mv.visit as f32 * inv_valid_total;
                }
            }
        }

        // label smoothing (env var BULLET_LABEL_SMOOTH=α、 0..1)。
        // target = (1-α) * visit_dist + α / N。 α=0 で no-op (既存挙動互換)。
        let alpha = label_smooth_alpha();
        if alpha > 0.0 {
            let uniform = alpha / POLICY_TARGET_DIMS as f32;
            let scale = 1.0 - alpha;
            for v in out.iter_mut() {
                *v = scale * (*v) + uniform;
            }
        }
    }
}

#[inline]
fn side_to_move_from_psv(pos: &PackedSfenWithVisits) -> Color {
    let stm_bit = pos.sfen().data[0] & 1;
    if stm_bit == 0 { Color::Black } else { Color::White }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::{MoveVisit, PackedSfenValue, PackedSfenWithVisits};

    fn dummy_with_visits(visits: &[(u16, u16)]) -> PackedSfenWithVisits {
        // packed 構造のため field 直接書込み不可、 別 var に組んで move する。
        let mut visits_arr = [MoveVisit::default(); 32];
        for (i, &(mv, vs)) in visits.iter().enumerate() {
            if i >= 32 {
                break;
            }
            visits_arr[i] = MoveVisit { move16: mv, visit: vs };
        }
        PackedSfenWithVisits {
            psv: PackedSfenValue::default(),
            visit_count: visits.len() as u16,
            _pad: 0,
            visits: visits_arr,
        }
    }

    #[test]
    fn empty_visit_zero_fills_zero() {
        let p = dummy_with_visits(&[]);
        let mut out = vec![0.0f32; POLICY_TARGET_DIMS];
        ShogiHalfKaHmPolicySoft.fill_auxiliary_dense(POLICY_VISIT_DIST_NAME, &p, &mut out);
        let sum: f32 = out.iter().sum();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn visits_normalize_to_probability() {
        // 3 つの move、 visit 100/50/50 → probs 0.5 / 0.25 / 0.25
        // move16 はとりあえず非ゼロ。 to=10 from=20 (board move) = 10 | (20 << 7) = 2570
        let p = dummy_with_visits(&[(2570, 100), (2571, 50), (2572, 50)]);
        let mut out = vec![0.0f32; POLICY_TARGET_DIMS];
        ShogiHalfKaHmPolicySoft.fill_auxiliary_dense(POLICY_VISIT_DIST_NAME, &p, &mut out);
        let sum: f32 = out.iter().sum();
        // 確率分布なので合計 ~1.0 (浮動小数誤差込み)
        assert!((sum - 1.0).abs() < 1e-5, "sum should be 1.0 but got {}", sum);
    }
}
