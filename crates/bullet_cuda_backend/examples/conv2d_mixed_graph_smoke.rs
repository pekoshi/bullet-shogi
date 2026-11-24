//! Phase 2 CRCL-v2 Conv2d graph IR 統合 smoke。
//!
//! builder.conv2d_mixed() で graph を構築 + forward + backward を実行し、
//! 既存 builder.conv2d() (f32 NVRTC) と数値が一致確認する。
//!
//! 構造:
//!   x  (Cin × H × W、 batched)
//!   w  (Cout × Cin × K × K、 weight)
//!   y = x.conv2d_mixed(w, params)
//!   scalar = ones · y (graph root)

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::{
    CudaDevice,
    builder::{Conv2dParams, ConvBuilderExt},
};

fn run_conv_graph(mixed: bool, w_host: &[f32], x_host: &[f32], p: Conv2dParams, batch: usize) -> (Vec<f32>, Vec<f32>) {
    let in_size = p.input_size();
    let out_size = p.output_size();

    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    let w = builder.new_weights("w", Shape::new(p.weight_size(), 1), InitSettings::Zeroed);
    let x = builder.new_dense_input("x", Shape::new(in_size, 1));

    let y = if mixed { x.conv2d_mixed(w, p) } else { x.conv2d(w, p) };

    // graph root scalar 化
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
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
    let mut y_host = vec![0.0f32; out_size * batch];
    y_ref.dense().write_to_slice(&mut y_host).expect("y write");

    graph.execute("backward").expect("backward");
    graph.synchronise().expect("sync bwd");

    let dw_ref =
        graph.get(GraphNodeId::new(graph.weight_idx("w").expect("w idx"), GraphNodeIdTy::Gradients)).expect("dw get");
    let mut dw_host = vec![0.0f32; p.weight_size()];
    dw_ref.dense().write_to_slice(&mut dw_host).expect("dw write");

    (y_host, dw_host)
}

fn main() {
    println!("=== Conv2d mixed precision graph IR smoke ===");

    let p = Conv2dParams {
        in_channels: 4,
        out_channels: 8,
        spatial_h: 5,
        spatial_w: 5,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };
    let batch = 2usize;

    use std::f32::consts::PI;
    let w_host: Vec<f32> = (0..p.weight_size()).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.25).collect();
    let x_host: Vec<f32> = (0..p.input_size() * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();

    let (y_mixed, dw_mixed) = run_conv_graph(true, &w_host, &x_host, p, batch);
    let (y_f32, dw_f32) = run_conv_graph(false, &w_host, &x_host, p, batch);

    let mut max_y_abs = 0.0f32;
    let mut max_y_rel = 0.0f32;
    for i in 0..p.output_size() * batch {
        let abs_err = (y_mixed[i] - y_f32[i]).abs();
        let rel_err = if y_f32[i].abs() > 1e-2 { abs_err / y_f32[i].abs() } else { 0.0 };
        max_y_abs = max_y_abs.max(abs_err);
        max_y_rel = max_y_rel.max(rel_err);
    }
    let mut max_dw_abs = 0.0f32;
    let mut max_dw_rel = 0.0f32;
    for i in 0..p.weight_size() {
        let abs_err = (dw_mixed[i] - dw_f32[i]).abs();
        let rel_err = if dw_f32[i].abs() > 1e-2 { abs_err / dw_f32[i].abs() } else { 0.0 };
        max_dw_abs = max_dw_abs.max(abs_err);
        max_dw_rel = max_dw_rel.max(rel_err);
    }
    println!("forward (mixed vs f32 graph): abs={:.6e}, rel={:.6e}", max_y_abs, max_y_rel);
    println!("backward dw (mixed vs f32 graph): abs={:.6e}, rel={:.6e}", max_dw_abs, max_dw_rel);
    assert!(max_y_rel < 5e-2, "forward rel err too large");
    assert!(max_dw_rel < 5e-2, "backward rel err too large");

    println!("Conv2d mixed precision graph IR smoke PASSED");
    println!("  → builder.conv2d_mixed() が graph forward + backward で動作");
    println!("  → CRCL-v2 ResNet を IV案 で graph builder 経由で構築可能 (残: ResBlock pattern)");
}
