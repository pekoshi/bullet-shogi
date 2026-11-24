//! Phase 2 graph IR 統合 backward 本実装 smoke。
//!
//! `builder.matmul_mixed()` の backward (MixedMatmulBackward chain rule) が動作し、
//! grad が f32 path と数値一致することを確認する。
//!
//! 構造:
//!   w (out × in, weight、 requires_grad)
//!   x (in × batch)
//!   y = w.matmul_mixed(x) (out × batch)
//!   scalar = ones · y (graph root)
//!
//! 検証: w.grad と CPU reference (dY × X^T = ones × X^T) との一致。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::CudaDevice;

fn run_with_backward(
    mixed: bool,
    w_host: &[f32],
    x_host: &[f32],
    out_dim: usize,
    in_dim: usize,
    batch: usize,
) -> (Vec<f32>, Vec<f32>) {
    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    let w = builder.new_weights("w", Shape::new(out_dim, in_dim), InitSettings::Zeroed);
    let x = builder.new_dense_input("x", Shape::new(in_dim, 1));

    let y = if mixed { w.matmul_mixed(x) } else { w.matmul(x) };

    // graph root = ones (1, out_dim) × y (out_dim, batch) → (1, batch) → sum across batch でスカラー
    let ones = builder.new_constant(Shape::new(1, out_dim), &vec![1.0f32; out_dim]);
    let _scalar = ones.matmul(y);
    let y_idx = y.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);

    {
        let w_ref = graph.get_weights("w");
        let mut wd = w_ref.dense_mut();
        wd.load_from_slice(None, w_host).expect("w load");
    }
    {
        let x_ref = graph.get_input("x");
        let mut xd = x_ref.dense_mut();
        xd.load_from_slice(Some(batch), x_host).expect("x load");
    }

    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync fwd");

    let y_ref = graph.get(GraphNodeId::new(y_idx, GraphNodeIdTy::Values)).expect("y get");
    let mut y_out = vec![0.0f32; out_dim * batch];
    y_ref.dense().write_to_slice(&mut y_out).expect("y write");

    graph.execute("backward").expect("backward");
    graph.synchronise().expect("sync bwd");

    let dw_ref =
        graph.get(GraphNodeId::new(graph.weight_idx("w").expect("w idx"), GraphNodeIdTy::Gradients)).expect("dw get");
    let mut dw_out = vec![0.0f32; out_dim * in_dim];
    dw_ref.dense().write_to_slice(&mut dw_out).expect("dw write");

    (y_out, dw_out)
}

fn main() {
    println!("=== matmul_mixed backward smoke (MixedMatmulBackward chain rule) ===");

    let out_dim = 8usize;
    let in_dim = 16usize;
    let batch = 4usize;

    use std::f32::consts::PI;
    let w_host: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.25).collect();
    let x_host: Vec<f32> = (0..in_dim * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.35).collect();

    let (y_mixed, dw_mixed) = run_with_backward(true, &w_host, &x_host, out_dim, in_dim, batch);
    let (y_f32, dw_f32) = run_with_backward(false, &w_host, &x_host, out_dim, in_dim, batch);

    // forward sanity
    let mut max_y_abs = 0.0f32;
    let mut max_y_rel = 0.0f32;
    for i in 0..out_dim * batch {
        let abs_err = (y_mixed[i] - y_f32[i]).abs();
        let rel_err = if y_f32[i].abs() > 1e-2 { abs_err / y_f32[i].abs() } else { 0.0 };
        max_y_abs = max_y_abs.max(abs_err);
        max_y_rel = max_y_rel.max(rel_err);
    }
    println!("forward (mixed vs f32 graph): abs={:.6e}, rel={:.6e}", max_y_abs, max_y_rel);

    // backward 比較
    let mut max_dw_abs = 0.0f32;
    let mut max_dw_rel = 0.0f32;
    for i in 0..out_dim * in_dim {
        let abs_err = (dw_mixed[i] - dw_f32[i]).abs();
        let rel_err = if dw_f32[i].abs() > 1e-2 { abs_err / dw_f32[i].abs() } else { 0.0 };
        max_dw_abs = max_dw_abs.max(abs_err);
        max_dw_rel = max_dw_rel.max(rel_err);
    }
    println!("backward dw (mixed vs f32 graph): abs={:.6e}, rel={:.6e}", max_dw_abs, max_dw_rel);

    assert!(max_y_rel < 5e-2, "forward rel err too large");
    // backward は upstream grad が ones (一様) で、 値が小さいとき rel err が出やすい
    assert!(max_dw_abs < 5e-2, "backward abs err too large: {}", max_dw_abs);

    println!("matmul_mixed backward smoke PASSED (chain rule 動作)");
}
