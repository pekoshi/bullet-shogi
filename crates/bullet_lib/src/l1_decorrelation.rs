//! L1 (fc0) の行間 decorrelation 正則化。
//!
//! # 背景 (2026-09-09)
//!
//! v7相当 (L1 = 8 幅) を Net2Net で 16 幅に拡張したが、 複製されたペアが分化せず
//! **実効次元が 8.11 / 15 のまま**だった (相関行列の固有値エントロピーで実測)。
//! 容量を 1.9 倍にしたのに使えているのは +1.4%。
//!
//! 複製ペアは入力も出力先の重みも対称なので同じ勾配を受け、 jitter を入れても
//! 差が増幅されない (Net2Net の symmetry breaking 問題)。 自然分化は -0.0038/SB で、
//! 相関 0.77 -> 0 まで単純計算 200 SB かかる。
//!
//! # 定義
//!
//! `L = λ × mean_b [ Σ_{i<j} (ŵ_i · ŵ_j)² ]`   (`ŵ` は L2 正規化した行ベクトル)
//!
//! 正規化してから内積を取るので、 重みのノルムに影響されない。
//! **特定のペアを仮定しない** (Net2Net のペア構造に依存しないため全行ペアに適用)。
//!
//! 勾配 (正規化を通した厳密な微分):
//! `c_ij = ŵ_i · ŵ_j` として
//! `∂L/∂w_i = (2λ/n_pairs) × (1/‖w_i‖) × Σ_{j≠i} c_ij × (ŵ_j − c_ij ŵ_i)`
//!
//! 括弧内の `− c_ij ŵ_i` は正規化の射影項で、 これが無いと重みのノルムを縮める方向に
//! 力が働いてしまう (decorrelation ではなく単なる縮小になる)。

/// 1 bucket ぶんの decorrelation loss。 `rows` 行 × `cols` 列の row-major を想定。
pub fn decorr_loss_one(w: &[f32], rows: usize, cols: usize) -> f32 {
    if rows < 2 {
        return 0.0;
    }
    let norms: Vec<f32> = (0..rows)
        .map(|i| {
            let s: f64 = w[i * cols..(i + 1) * cols].iter().map(|x| (*x as f64) * (*x as f64)).sum();
            (s.sqrt() as f32).max(1e-8)
        })
        .collect();
    let mut sum = 0.0_f64;
    for i in 0..rows {
        for j in (i + 1)..rows {
            let d: f64 = (0..cols).map(|k| (w[i * cols + k] as f64) * (w[j * cols + k] as f64)).sum();
            let c = d / (norms[i] as f64 * norms[j] as f64);
            sum += c * c;
        }
    }
    let n_pairs = (rows * (rows - 1) / 2) as f64;
    (sum / n_pairs) as f32
}

/// 全 bucket の decorrelation loss (平均)。
///
/// * `weights` - `num_buckets * rows * cols` の row-major
/// * `rows` - 1 bucket あたりの行数 (= L1 幅)。 skip 行を除くなら呼び出し側で `rows-1` を渡す
/// * `use_rows` - 先頭何行を対象にするか (skip 行を除外する用途)
pub fn decorr_loss(weights: &[f32], num_buckets: usize, rows: usize, cols: usize, use_rows: usize, lambda: f32) -> f32 {
    if lambda == 0.0 || use_rows < 2 {
        return 0.0;
    }
    let per = rows * cols;
    let mut sum = 0.0_f64;
    for b in 0..num_buckets {
        let off = b * per;
        sum += decorr_loss_one(&weights[off..off + use_rows * cols], use_rows, cols) as f64;
    }
    (lambda as f64 * sum / num_buckets as f64) as f32
}

/// decorrelation gradient を `grad_out` に **加算** する。
///
/// `grad_out[b][i][k] += ∂L/∂w[b][i][k]`
pub fn decorr_grad_add(
    weights: &[f32],
    grad_out: &mut [f32],
    num_buckets: usize,
    rows: usize,
    cols: usize,
    use_rows: usize,
    lambda: f32,
) {
    debug_assert_eq!(weights.len(), num_buckets * rows * cols);
    debug_assert_eq!(grad_out.len(), weights.len());
    if lambda == 0.0 || use_rows < 2 {
        return;
    }
    let per = rows * cols;
    let n_pairs = (use_rows * (use_rows - 1) / 2) as f32;
    let factor = 2.0 * lambda / (n_pairs * num_buckets as f32);
    let mut unit = vec![0.0_f32; use_rows * cols];
    let mut norms = vec![0.0_f32; use_rows];
    for b in 0..num_buckets {
        let off = b * per;
        // 正規化した行ベクトルを作る
        for i in 0..use_rows {
            let src = &weights[off + i * cols..off + (i + 1) * cols];
            let s: f64 = src.iter().map(|x| (*x as f64) * (*x as f64)).sum();
            let n = (s.sqrt() as f32).max(1e-8);
            norms[i] = n;
            for k in 0..cols {
                unit[i * cols + k] = src[k] / n;
            }
        }
        // ∂L/∂w_i = factor / ‖w_i‖ × Σ_{j≠i} c_ij (ŵ_j − c_ij ŵ_i)
        for i in 0..use_rows {
            let inv = factor / norms[i];
            for j in 0..use_rows {
                if i == j {
                    continue;
                }
                let c: f32 = (0..cols).map(|k| unit[i * cols + k] * unit[j * cols + k]).sum();
                if c == 0.0 {
                    continue;
                }
                for k in 0..cols {
                    grad_out[off + i * cols + k] += inv * c * (unit[j * cols + k] - c * unit[i * cols + k]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_lambda_is_noop() {
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let mut g = vec![5.0; 4];
        decorr_grad_add(&w, &mut g, 1, 2, 2, 2, 0.0);
        assert_eq!(g, vec![5.0; 4]);
    }

    #[test]
    fn orthogonal_rows_have_zero_loss() {
        // 直交する 2 行 -> 相関 0 -> loss 0
        let w = vec![1.0, 0.0, 0.0, 1.0];
        assert!(decorr_loss(&w, 1, 2, 2, 2, 1.0).abs() < 1e-6);
    }

    #[test]
    fn identical_rows_have_max_loss() {
        // 完全に同じ 2 行 -> 相関 1 -> loss = λ
        let w = vec![1.0, 2.0, 1.0, 2.0];
        assert!((decorr_loss(&w, 1, 2, 2, 2, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn grad_matches_numerical() {
        let w = vec![1.0, 2.0, 0.5, 3.0, 0.7, 1.1]; // 1 bucket, 3 行 x 2 列
        let (rows, cols, lam) = (3usize, 2usize, 0.7f32);
        let mut g = vec![0.0; w.len()];
        decorr_grad_add(&w, &mut g, 1, rows, cols, rows, lam);
        let eps = 1e-3_f32;
        for idx in 0..w.len() {
            let mut a = w.clone();
            let mut b = w.clone();
            a[idx] += eps;
            b[idx] -= eps;
            let num =
                (decorr_loss(&a, 1, rows, cols, rows, lam) - decorr_loss(&b, 1, rows, cols, rows, lam)) / (2.0 * eps);
            assert!((num - g[idx]).abs() < 2e-2, "idx={idx} numerical={num} analytical={}", g[idx]);
        }
    }
}
