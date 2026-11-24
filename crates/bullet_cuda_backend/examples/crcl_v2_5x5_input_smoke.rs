//! CRCL-v2 改善 B: 入力層 5x5 large kernel smoke (dlshogi 知見)。
//!
//! 既存 builder.conv2d_mixed を kernel_size=5 で input 層に使い、
//! 後続 ResBlock は 3x3 のまま。 早期にグローバル特徴 (5x5 = 25 cells) を捉える効果。
//!
//! 構造:
//!   x (Cin × 9 × 9, batched)
//!     → conv2d_mixed (Cin → Cmid, K=5, Pad=2)    ← 5x5 input layer
//!     → silu_mixed
//!     → ResBlock × N (各 conv 3x3)
//!     → ...
//!
//! 動作確認: graph forward + backward 完走、 NaN/Inf なし。

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

/// ResBlock (CRCL-v2 標準、 3x3): y = silu(conv2(silu(conv1(x))) + x)
fn resblock_mixed<'a>(
    input: GraphBuilderNode<'a, CudaMarker>,
    w1: GraphBuilderNode<'a, CudaMarker>,
    w2: GraphBuilderNode<'a, CudaMarker>,
    p: Conv2dParams,
) -> GraphBuilderNode<'a, CudaMarker> {
    let y = input.conv2d_mixed(w1, p);
    let z = y.silu_mixed();
    let u = z.conv2d_mixed(w2, p);
    let res = u + input;
    res.silu_mixed()
}

fn main() {
    println!("=== CRCL-v2 改善 B: 5x5 input layer smoke ===");

    let cin = 4i32; // 入力チャンネル (NCHW)
    let cmid = 4i32; // ResBlock 内チャンネル (= Cin/Cout for residual)
    let h = 9i32;
    let w = 9i32;
    let batch = 2usize;
    let n_blocks = 2usize;

    // 入力層 5x5 params (pad=2 で同 spatial 維持)
    let p_input = Conv2dParams {
        in_channels: cin,
        out_channels: cmid,
        spatial_h: h,
        spatial_w: w,
        kernel_size: 5,
        padding: 2,
        stride: 1,
    };
    // ResBlock 内 3x3 params (pad=1 で同 spatial)
    let p_block = Conv2dParams {
        in_channels: cmid,
        out_channels: cmid,
        spatial_h: h,
        spatial_w: w,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };

    let builder = GraphBuilder::<CudaMarker>::default();
    // 入力層 weight (5x5)
    let w_in = builder.new_weights("w_in", Shape::new(p_input.weight_size(), 1), InitSettings::Zeroed);
    // ResBlock 内 weights
    let mut block_weight_names = Vec::with_capacity(n_blocks * 2);
    let mut block_weight_nodes = Vec::with_capacity(n_blocks * 2);
    for b in 0..n_blocks {
        for k in 0..2 {
            let name = format!("w_b{}_{}", b, k);
            let node = builder.new_weights(&name, Shape::new(p_block.weight_size(), 1), InitSettings::Zeroed);
            block_weight_names.push(name);
            block_weight_nodes.push(node);
        }
    }
    let x = builder.new_dense_input("x", Shape::new(p_input.input_size(), 1));

    // forward: 5x5 input layer → SiLU → ResBlock × n
    let stem = x.conv2d_mixed(w_in, p_input).silu_mixed();
    let mut h_node = stem;
    for b in 0..n_blocks {
        let w1 = block_weight_nodes[b * 2];
        let w2 = block_weight_nodes[b * 2 + 1];
        h_node = resblock_mixed(h_node, w1, w2, p_block);
    }
    drop(block_weight_nodes);

    // graph root scalar 化
    let out_size = p_block.output_size();
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _scalar = ones.matmul(h_node);
    let out_idx = h_node.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);
    println!("graph built: 5x5 input layer → SiLU → {} ResBlock (3x3) chain", n_blocks);

    // init
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..p_input.input_size() * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    let w_in_host: Vec<f32> = (0..p_input.weight_size()).map(|i| (i as f32 * 0.0119 * PI).sin() * 0.15).collect();
    {
        let xref = graph.get_input("x");
        let mut xd = xref.dense_mut();
        xd.load_from_slice(Some(batch), &x_host).expect("x load");
    }
    {
        let wref = graph.get_weights("w_in");
        let mut wd = wref.dense_mut();
        wd.load_from_slice(None, &w_in_host).expect("w_in load");
    }
    for (i, name) in block_weight_names.iter().enumerate() {
        let w_host: Vec<f32> =
            (0..p_block.weight_size()).map(|j| ((i * 100 + j) as f32 * 0.0173 * PI).sin() * 0.2).collect();
        let wref = graph.get_weights(name);
        let mut wd = wref.dense_mut();
        wd.load_from_slice(None, &w_host).expect("w load");
    }

    // forward
    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync fwd");
    let out_ref = graph.get(GraphNodeId::new(out_idx, GraphNodeIdTy::Values)).expect("out get");
    let mut y_host = vec![0.0f32; out_size * batch];
    out_ref.dense().write_to_slice(&mut y_host).expect("y write");
    let max_abs = y_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let n_finite = y_host.iter().filter(|v| v.is_finite()).count();
    println!("forward: {}/{} finite, max |y|={:.4}", n_finite, y_host.len(), max_abs);
    assert_eq!(n_finite, y_host.len());

    // backward
    graph.execute("backward").expect("backward");
    graph.synchronise().expect("sync bwd");

    // 入力層 weight grad 確認
    {
        let dw_ref = graph
            .get(GraphNodeId::new(graph.weight_idx("w_in").expect("w_in idx"), GraphNodeIdTy::Gradients))
            .expect("dw get");
        let mut dw_host = vec![0.0f32; p_input.weight_size()];
        dw_ref.dense().write_to_slice(&mut dw_host).expect("dw write");
        let nz = dw_host.iter().filter(|v| v.abs() > 1e-10).count();
        let max_abs = dw_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        println!("w_in grad: {}/{} non-zero, max |dw|={:.4e}", nz, dw_host.len(), max_abs);
        assert!(nz > 0, "w_in grad all-zero (backward chain broken)");
    }
    // ResBlock 内 weight grad 確認
    for name in block_weight_names.iter() {
        let dw_ref = graph
            .get(GraphNodeId::new(graph.weight_idx(name).expect("w idx"), GraphNodeIdTy::Gradients))
            .expect("dw get");
        let mut dw_host = vec![0.0f32; p_block.weight_size()];
        dw_ref.dense().write_to_slice(&mut dw_host).expect("dw write");
        let nz = dw_host.iter().filter(|v| v.abs() > 1e-10).count();
        assert!(nz > 0, "{name} grad all-zero");
    }

    println!("CRCL-v2 5x5 input layer smoke PASSED");
    println!("  → builder.conv2d_mixed(kernel=5) で 5x5 input + 3x3 ResBlock chain が動作");
    println!("  → dlshogi 知見「入力層ラージカーネル」 を CRCL-v2 に取り入れ完了");
}
