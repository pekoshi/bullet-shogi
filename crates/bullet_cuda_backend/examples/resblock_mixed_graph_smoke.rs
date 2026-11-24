//! Phase 2 CRCL-v2 ResNet block pattern smoke。
//!
//! 標準 ResBlock 構造 (CRCL-v2):
//!   y    = conv2d_mixed(x, w1)
//!   z    = silu_mixed(y)
//!   u    = conv2d_mixed(z, w2)
//!   res  = u + x       (residual / skip connection)
//!   out  = silu_mixed(res)
//!
//! 数 block を chain で stack、 graph 経由で forward + backward 動作確認。
//! CRCL-v2 ResNet (16 block × Cin=Nch) を builder で書ける ことの最終確認。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, GraphBuilderNode, InitSettings},
    },
};
use bullet_cuda_backend::{
    CudaDevice, CudaMarker,
    builder::{Conv2dParams, ConvBuilderExt},
};

/// ResBlock: y = silu(conv2(silu(conv1(x))) + x)
/// すべて mixed precision (Tensor Core 対応)
fn resblock_mixed<'a>(
    input: GraphBuilderNode<'a, CudaMarker>,
    w1: GraphBuilderNode<'a, CudaMarker>,
    w2: GraphBuilderNode<'a, CudaMarker>,
    p: Conv2dParams,
) -> GraphBuilderNode<'a, CudaMarker> {
    let y = input.conv2d_mixed(w1, p);
    let z = y.silu_mixed();
    let u = z.conv2d_mixed(w2, p);
    let res = u + input; // GraphBuilderNode の Add impl
    res.silu_mixed()
}

fn main() {
    println!("=== ResBlock mixed precision graph IR smoke (CRCL-v2 pattern) ===");

    let p = Conv2dParams {
        in_channels: 4,
        out_channels: 4, // ResBlock では Cin == Cout 必須 (residual)
        spatial_h: 5,
        spatial_w: 5,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };
    let n_blocks = 3usize;
    let batch = 2usize;

    let builder = GraphBuilder::<CudaMarker>::default();
    let mut weight_names = Vec::with_capacity(n_blocks * 2);
    let mut weight_nodes = Vec::with_capacity(n_blocks * 2);
    for b in 0..n_blocks {
        for k in 0..2 {
            let name = format!("w_b{}_{}", b, k);
            let w = builder.new_weights(&name, Shape::new(p.weight_size(), 1), InitSettings::Zeroed);
            weight_names.push(name);
            weight_nodes.push(w);
        }
    }

    let x = builder.new_dense_input("x", Shape::new(p.input_size(), 1));

    let mut h = x;
    for b in 0..n_blocks {
        let w1 = weight_nodes[b * 2];
        let w2 = weight_nodes[b * 2 + 1];
        h = resblock_mixed(h, w1, w2, p);
    }
    drop(weight_nodes);

    let out_size = p.output_size();
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _scalar = ones.matmul(h);
    let out_idx = h.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);
    println!("graph built: {} ResBlock chain (each: conv_mixed → silu_mixed → conv_mixed → +x → silu_mixed)", n_blocks);

    // x init
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..p.input_size() * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    {
        let x_ref = graph.get_input("x");
        let mut xd = x_ref.dense_mut();
        xd.load_from_slice(Some(batch), &x_host).expect("x load");
    }
    // weight init (random)
    for (i, name) in weight_names.iter().enumerate() {
        let w_host: Vec<f32> = (0..p.weight_size()).map(|j| ((i * 100 + j) as f32 * 0.0173 * PI).sin() * 0.2).collect();
        let w_ref = graph.get_weights(name);
        let mut wd = w_ref.dense_mut();
        wd.load_from_slice(None, &w_host).expect("w load");
    }

    // forward
    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync fwd");

    // output 確認 (size check + NaN/Inf 無いか)
    let out_ref = graph.get(GraphNodeId::new(out_idx, GraphNodeIdTy::Values)).expect("out get");
    let mut y_host = vec![0.0f32; out_size * batch];
    out_ref.dense().write_to_slice(&mut y_host).expect("y write");
    let n_finite = y_host.iter().filter(|v| v.is_finite()).count();
    let max_abs = y_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("forward done: {}/{} finite, max |y|={:.4}", n_finite, y_host.len(), max_abs);
    assert_eq!(n_finite, y_host.len(), "forward produced NaN/Inf");

    // backward
    graph.execute("backward").expect("backward");
    graph.synchronise().expect("sync bwd");

    // 各 weight の grad 確認 (NaN/Inf 無いか + 非ゼロかどうか)
    let mut total_nonzero_grads = 0;
    for name in weight_names.iter() {
        let dw_ref = graph
            .get(GraphNodeId::new(graph.weight_idx(name).expect("w idx"), GraphNodeIdTy::Gradients))
            .expect("dw get");
        let mut dw_host = vec![0.0f32; p.weight_size()];
        dw_ref.dense().write_to_slice(&mut dw_host).expect("dw write");
        let n_finite = dw_host.iter().filter(|v| v.is_finite()).count();
        let n_nonzero = dw_host.iter().filter(|v| v.abs() > 1e-10).count();
        let max_abs = dw_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert_eq!(n_finite, dw_host.len(), "{name} grad has NaN/Inf");
        println!("  {} grad: {}/{} non-zero, max |dw|={:.4e}", name, n_nonzero, dw_host.len(), max_abs);
        if n_nonzero > 0 {
            total_nonzero_grads += 1;
        }
    }
    assert_eq!(total_nonzero_grads, weight_names.len(), "some weight grad is all-zero (backward chain broken?)");

    println!("\nResBlock chain ({}) graph IR mixed precision smoke PASSED", n_blocks);
    println!("  → CRCL-v2 ResNet (16 block × Cin=Nch) を builder.conv2d_mixed + silu_mixed + add + silu_mixed");
    println!("    で構築可能、 forward + backward 完全動作");
}
