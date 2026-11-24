//! 隣接 bucket smoothness regularization (= 2026-06-09)。
//!
//! 訓練時に bucket 別 weight tensor の隣接 bucket 間 L2 距離を penalty として
//! 加算する。 推論 NPS 不変 (= 訓練時のみ)、 `λ=0` で完全 no-op。
//!
//! # 設計
//!
//! `L_smooth = λ × mean_b ||w[b+1] - w[b]||²`
//!          = λ / (B-1) × Σ_b Σ_i (w[b+1][i] - w[b][i])²
//!
//! 隣接 bucket weight が完全 smooth (= 各 b で `w[b+1] = w[b]`) で 0、
//! 1 境界差で正値、 全 weight scale で linear。
//!
//! gradient:
//!   `dL/dw[b] = 2λ/(B-1) × ((w[b] - w[b-1]) + (w[b] - w[b+1]))` (= interior)
//!            = `2λ/(B-1) × (2w[b] - w[b-1] - w[b+1])`
//!   端 (= b=0 or b=B-1) は片方のみ:
//!   `dL/dw[0] = 2λ/(B-1) × (w[0] - w[1])`
//!   `dL/dw[B-1] = 2λ/(B-1) × (w[B-1] - w[B-2])`
//!
//! 使用前提: weights は **bucket-outer flat layout** (= `[B × dim]` row-major)。
//! bucket-inner (= `[dim × B]`) には未対応、 caller 側で transpose する。

/// 隣接 bucket smoothness loss 値を計算する。
///
/// # Arguments
/// * `weights` - bucket-outer flat layout `[B × dim]` の f32 slice (= row-major)
/// * `num_buckets` - bucket 数 B (= 通常 8)
/// * `dim_per_bucket` - 各 bucket の weight 数 (= weights.len() == B × dim_per_bucket 必須)
/// * `lambda` - 係数 λ (= 0 で完全 no-op、 loss 0 を返す)
///
/// # Returns
/// `L_smooth = λ × mean_b ||w[b+1] - w[b]||²` の値 (= f32)
///
/// # Panics
/// 訓練 hot path で呼ぶため `debug_assert!` で形状検査のみ。 release では caller の
/// 事前検証を信頼する。 `weights.len() != num_buckets * dim_per_bucket` の場合
/// debug build で panic、 release build で先頭部分のみで計算され誤った値となる。
pub fn smoothness_loss(weights: &[f32], num_buckets: usize, dim_per_bucket: usize, lambda: f32) -> f32 {
    debug_assert_eq!(weights.len(), num_buckets * dim_per_bucket);
    if lambda == 0.0 || num_buckets <= 1 {
        return 0.0;
    }
    let mut sum: f64 = 0.0;
    for b in 0..num_buckets - 1 {
        let off_b = b * dim_per_bucket;
        let off_b1 = (b + 1) * dim_per_bucket;
        for i in 0..dim_per_bucket {
            let d = (weights[off_b1 + i] - weights[off_b + i]) as f64;
            sum += d * d;
        }
    }
    let denom = (num_buckets - 1) as f64;
    (lambda as f64 * sum / denom) as f32
}

/// 隣接 bucket smoothness gradient を計算して `grad_out` に **加算** する。
///
/// `grad_out[b][i] += dL/dw[b][i]`
///   interior (`1 <= b <= B-2`):
///   `dL/dw[b] = 2λ/(B-1) × ((w[b] - w[b-1]) + (w[b] - w[b+1]))`
///   端:
///   `dL/dw[0] = 2λ/(B-1) × (w[0] - w[1])`
///   `dL/dw[B-1] = 2λ/(B-1) × (w[B-1] - w[B-2])`
///
/// # Arguments
/// * `weights` - 同上
/// * `grad_out` - 加算先 gradient slice (= 同 shape、 既存値に **+=** で加算)
/// * `num_buckets` - 同上
/// * `dim_per_bucket` - 同上
/// * `lambda` - 係数 λ (= 0 で何もしない、 grad_out 不変)
pub fn smoothness_grad_add(
    weights: &[f32],
    grad_out: &mut [f32],
    num_buckets: usize,
    dim_per_bucket: usize,
    lambda: f32,
) {
    debug_assert_eq!(weights.len(), num_buckets * dim_per_bucket);
    debug_assert_eq!(grad_out.len(), num_buckets * dim_per_bucket);
    if lambda == 0.0 || num_buckets <= 1 {
        return;
    }
    let factor = (2.0 * lambda as f64 / (num_buckets - 1) as f64) as f32;
    for b in 0..num_buckets {
        let off_b = b * dim_per_bucket;
        for i in 0..dim_per_bucket {
            let mut g = 0.0_f32;
            if b > 0 {
                let off_p = (b - 1) * dim_per_bucket;
                g += weights[off_b + i] - weights[off_p + i];
            }
            if b + 1 < num_buckets {
                let off_n = (b + 1) * dim_per_bucket;
                g += weights[off_b + i] - weights[off_n + i];
            }
            grad_out[off_b + i] += factor * g;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_lambda_noop() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut grad = vec![10.0; 8];
        let loss = smoothness_loss(&w, 4, 2, 0.0);
        assert_eq!(loss, 0.0);
        smoothness_grad_add(&w, &mut grad, 4, 2, 0.0);
        // grad 不変
        assert_eq!(grad, vec![10.0; 8]);
    }

    #[test]
    fn test_same_weights_zero_loss() {
        // 全 bucket 同一 weight → loss = 0、 grad = 0
        let w = vec![1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5];
        let mut grad = vec![0.0; 8];
        let loss = smoothness_loss(&w, 4, 2, 1.0);
        assert!(loss.abs() < 1e-6, "loss should be 0 for uniform buckets, got {loss}");
        smoothness_grad_add(&w, &mut grad, 4, 2, 1.0);
        for (i, &g) in grad.iter().enumerate() {
            assert!(g.abs() < 1e-6, "grad[{i}] should be 0 for uniform buckets, got {g}");
        }
    }

    #[test]
    fn test_one_boundary_diff_positive() {
        // 全 bucket 同一を base、 b=2 だけ weight 違う → loss > 0
        let mut w = vec![0.0; 8]; // 4 bucket × 2 dim
        w[4] = 1.0; // bucket 2, dim 0
        w[5] = 0.0; // bucket 2, dim 1
        let loss = smoothness_loss(&w, 4, 2, 1.0);
        // 隣接 b1→b2 で diff = (1, 0)、 b2→b3 で diff = (-1, 0)
        // sum = 1² + 0² + (-1)² + 0² = 2
        // mean = 2 / 3 = 0.6667
        let expected = 2.0 / 3.0;
        assert!((loss - expected).abs() < 1e-6, "expected {expected}, got {loss}");
        assert!(loss > 0.0);
    }

    #[test]
    fn test_bucket_order_reversal_invariant() {
        // bucket 順反転 (= [b0, b1, b2, b3] → [b3, b2, b1, b0]) で loss 一致
        let w = vec![1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 11.0];
        let mut w_rev = vec![0.0; 8];
        for b in 0..4 {
            for i in 0..2 {
                w_rev[(3 - b) * 2 + i] = w[b * 2 + i];
            }
        }
        let loss = smoothness_loss(&w, 4, 2, 1.5);
        let loss_rev = smoothness_loss(&w_rev, 4, 2, 1.5);
        assert!((loss - loss_rev).abs() < 1e-5, "order reversal should preserve loss, got {loss} vs {loss_rev}");
    }

    #[test]
    fn test_gradient_finite_diff() {
        // 数値微分 (= 中心差分) と analytical gradient の一致確認
        let w = vec![1.0, 0.5, 0.3, 0.7, -0.2, 1.5, 2.1, 0.8];
        let num_buckets = 4;
        let dim = 2;
        let lambda = 0.7;

        // analytical gradient
        let mut grad_analytical = vec![0.0; w.len()];
        smoothness_grad_add(&w, &mut grad_analytical, num_buckets, dim, lambda);

        // 数値微分 (= 中心差分)
        let eps = 1e-3_f32;
        for idx in 0..w.len() {
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();
            w_plus[idx] += eps;
            w_minus[idx] -= eps;
            let l_plus = smoothness_loss(&w_plus, num_buckets, dim, lambda);
            let l_minus = smoothness_loss(&w_minus, num_buckets, dim, lambda);
            let grad_num = (l_plus - l_minus) / (2.0 * eps);
            let diff = (grad_analytical[idx] - grad_num).abs();
            assert!(
                diff < 1e-2,
                "grad mismatch at idx {idx}: analytical={} numerical={} diff={diff}",
                grad_analytical[idx],
                grad_num
            );
        }
    }
}
