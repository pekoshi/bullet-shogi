//! Phase 2 graph IR 統合 G4-G5 smoke: `builder.matmul_mixed()` で graph を構築し、
//! forward を実行する end-to-end smoke。
//!
//! 構造:
//!   weight w (m × k, not batched)
//!     × input x (k × batch)         ← MatmulF16Mixed (NobBat、 中間 f16 buffer 経由)
//!     → y (m × batch)
//!   ones (1 × m) · y で scalar 化 (graph root 要件)
//!
//! 検証:
//!   - 既存 `matmul` (f32 path) と `matmul_mixed` (IV案) の forward 結果を比較
//!   - f32 reference に対して rel_err 5% 以内
//!
//! NOTE: backward は当面 placeholder (G4 scope 縮小)。 forward 動作のみ確認。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::CudaDevice;

fn run_matmul_graph(mixed: bool, m: usize, k: usize, batch: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // master data
    use std::f32::consts::PI;
    let w_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.0317 * PI).sin() * 0.25).collect();
    let x_host: Vec<f32> = (0..k * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.35).collect();

    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    let w = builder.new_weights("w", Shape::new(m, k), InitSettings::Zeroed);
    let x = builder.new_dense_input("x", Shape::new(k, 1));

    let y = if mixed { w.matmul_mixed(x) } else { w.matmul(x) };

    // graph root を scalar 化
    let ones = builder.new_constant(Shape::new(1, m), &vec![1.0f32; m]);
    let _scalar = ones.matmul(y);
    let y_idx = y.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);

    // w / x 初期値
    {
        let w_ref = graph.get_weights("w");
        let mut w_dense = w_ref.dense_mut();
        w_dense.load_from_slice(None, &w_host).expect("w load");
    }
    {
        let x_ref = graph.get_input("x");
        let mut x_dense = x_ref.dense_mut();
        x_dense.load_from_slice(Some(batch), &x_host).expect("x load");
    }

    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync");

    let y_ref = graph.get(GraphNodeId::new(y_idx, GraphNodeIdTy::Values)).expect("y get");
    let mut y_host = vec![0.0f32; m * batch];
    y_ref.dense().write_to_slice(&mut y_host).expect("y write");

    (w_host, x_host, y_host)
}

fn main() {
    println!("=== matmul_mixed graph IR smoke (G4-G5) ===");

    let m = 16usize;
    let k = 32usize;
    let batch = 8usize;

    // f32 path
    let (w_host, x_host, y_f32_graph) = run_matmul_graph(false, m, k, batch);

    // mixed precision path
    let (_, _, y_mixed_graph) = run_matmul_graph(true, m, k, batch);

    // CPU reference: y[i, b] = sum_j W[i, j] * X[j, b]   (column-major)
    let mut max_abs_cpu = 0.0f32;
    let mut max_rel_cpu = 0.0f32;
    let mut max_abs_vs_f32 = 0.0f32;
    let mut max_rel_vs_f32 = 0.0f32;
    for i in 0..m {
        for b in 0..batch {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w_host[j * m + i] * x_host[b * k + j];
            }
            let mixed = y_mixed_graph[b * m + i];
            let f32_g = y_f32_graph[b * m + i];

            let abs_cpu = (mixed - acc).abs();
            let rel_cpu = if acc.abs() > 1e-2 { abs_cpu / acc.abs() } else { 0.0 };
            max_abs_cpu = max_abs_cpu.max(abs_cpu);
            max_rel_cpu = max_rel_cpu.max(rel_cpu);

            let abs_f = (mixed - f32_g).abs();
            let rel_f = if f32_g.abs() > 1e-2 { abs_f / f32_g.abs() } else { 0.0 };
            max_abs_vs_f32 = max_abs_vs_f32.max(abs_f);
            max_rel_vs_f32 = max_rel_vs_f32.max(rel_f);
        }
    }
    println!("matmul_mixed (m={m} k={k} batch={batch}):");
    println!("  vs CPU ref:  abs={:.6e}  rel={:.6e}", max_abs_cpu, max_rel_cpu);
    println!("  vs f32 graph: abs={:.6e}  rel={:.6e}", max_abs_vs_f32, max_rel_vs_f32);
    assert!(max_rel_cpu < 5e-2, "rel err vs CPU ref too large: {}", max_rel_cpu);
    assert!(max_rel_vs_f32 < 5e-2, "rel err vs f32 graph too large: {}", max_rel_vs_f32);

    println!("matmul_mixed graph IR smoke PASSED (G4-G5 完成)");
    println!("  → builder.matmul_mixed() が graph forward で動作");
}
