//! CRCL-v2 完全 ResNet end-to-end smoke。
//!
//! dlshogi 知見全部統合:
//!   - 5x5 input layer (入力 1 層のみ ラージカーネル)
//!   - 3x3 ResBlock chain (Cin == Cout 維持)
//!   - SEBlock を 5 block 間隔で挿入 (block 3 の後)
//!
//! 構造:
//!   x (Cin × H × W) → conv2d_mixed(5x5、 Cin → Cmid)
//!                  → silu_mixed
//!                  → ResBlock × 3 (3x3、 Cmid → Cmid)
//!                  → SEBlock (Cmid、 r=4)
//!                  → ResBlock × 3 (3x3、 Cmid → Cmid)
//!                  → ones · 全 cell = scalar root
//!
//! graph IR で forward + backward 完走 + 全 weight grad non-zero 確認。
//! CRCL-v2 ResNet build path の最終実証。

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
    ops::SEBlockMixed,
};

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
    println!("=== CRCL-v2 完全 ResNet end-to-end smoke ===");

    let cin = 4i32;
    let cmid = 8i32;
    let h = 9i32;
    let w_dim = 9i32;
    let batch = 2usize;
    let n_resblocks_part1 = 3usize; // 1st 半
    let n_resblocks_part2 = 3usize; // 2nd 半 (SE の後)
    let r_se = 4i32; // SE reduction ratio
    let c_r = cmid / r_se;

    // params
    let p_input = Conv2dParams {
        in_channels: cin,
        out_channels: cmid,
        spatial_h: h,
        spatial_w: w_dim,
        kernel_size: 5,
        padding: 2,
        stride: 1,
    };
    let p_block = Conv2dParams {
        in_channels: cmid,
        out_channels: cmid,
        spatial_h: h,
        spatial_w: w_dim,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };

    let builder = GraphBuilder::<CudaMarker>::default();

    // 入力層 weight (5x5)
    let w_in_node = builder.new_weights("w_in", Shape::new(p_input.weight_size(), 1), InitSettings::Zeroed);

    // ResBlock weights (part1 + part2)
    let mut rb_names = Vec::new();
    let mut rb_nodes = Vec::new();
    for half in &["p1", "p2"] {
        let n_blocks = if *half == "p1" { n_resblocks_part1 } else { n_resblocks_part2 };
        for b in 0..n_blocks {
            for k in 0..2 {
                let name = format!("w_{}_b{}_{}", half, b, k);
                let node = builder.new_weights(&name, Shape::new(p_block.weight_size(), 1), InitSettings::Zeroed);
                rb_names.push(name);
                rb_nodes.push(node);
            }
        }
    }

    // SE weights
    let w_se1_node = builder.new_weights("w_se1", Shape::new((c_r * cmid) as usize, 1), InitSettings::Zeroed);
    let w_se2_node = builder.new_weights("w_se2", Shape::new((cmid * c_r) as usize, 1), InitSettings::Zeroed);

    // input
    let x = builder.new_dense_input("x", Shape::new(p_input.input_size(), 1));

    // forward chain
    let stem = x.conv2d_mixed(w_in_node, p_input).silu_mixed();
    let mut h_node = stem;
    // part 1: ResBlock × n_resblocks_part1
    for b in 0..n_resblocks_part1 {
        let w1 = rb_nodes[b * 2];
        let w2 = rb_nodes[b * 2 + 1];
        h_node = resblock_mixed(h_node, w1, w2, p_block);
    }
    // 中間 SEBlock
    h_node = builder.apply(SEBlockMixed {
        input: h_node.annotated_node(),
        w1: w_se1_node.annotated_node(),
        w2: w_se2_node.annotated_node(),
        channels: cmid,
        spatial_h: h,
        spatial_w: w_dim,
        reduction: r_se,
    });
    // part 2: ResBlock × n_resblocks_part2
    let p2_off = n_resblocks_part1 * 2;
    for b in 0..n_resblocks_part2 {
        let w1 = rb_nodes[p2_off + b * 2];
        let w2 = rb_nodes[p2_off + b * 2 + 1];
        h_node = resblock_mixed(h_node, w1, w2, p_block);
    }
    drop(rb_nodes);
    drop(w_in_node);
    drop(w_se1_node);
    drop(w_se2_node);

    let out_size = p_block.output_size();
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _scalar = ones.matmul(h_node);
    let out_idx = h_node.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);
    println!(
        "graph built: 5x5 input → {} ResBlock → SEBlock(r={}) → {} ResBlock",
        n_resblocks_part1, r_se, n_resblocks_part2
    );

    // init
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..p_input.input_size() * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    // chain が長いので init を小さく (Kaiming init 風)
    let w_in_host: Vec<f32> = (0..p_input.weight_size()).map(|i| (i as f32 * 0.0119 * PI).sin() * 0.03).collect();
    let w_se1_host: Vec<f32> = (0..(c_r * cmid) as usize).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.05).collect();
    let w_se2_host: Vec<f32> = (0..(cmid * c_r) as usize).map(|i| (i as f32 * 0.0117 * PI).cos() * 0.05).collect();

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
    {
        let wref = graph.get_weights("w_se1");
        let mut wd = wref.dense_mut();
        wd.load_from_slice(None, &w_se1_host).expect("w_se1 load");
    }
    {
        let wref = graph.get_weights("w_se2");
        let mut wd = wref.dense_mut();
        wd.load_from_slice(None, &w_se2_host).expect("w_se2 load");
    }
    for (i, name) in rb_names.iter().enumerate() {
        let w_host: Vec<f32> =
            (0..p_block.weight_size()).map(|j| ((i * 100 + j) as f32 * 0.0173 * PI).sin() * 0.04).collect();
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

    // 各 weight grad 確認 (NaN/Inf なし + 非ゼロ)
    let all_weight_names: Vec<String> = std::iter::once("w_in".to_string())
        .chain(std::iter::once("w_se1".to_string()))
        .chain(std::iter::once("w_se2".to_string()))
        .chain(rb_names.iter().cloned())
        .collect();
    let mut total_nonzero_weights = 0;
    let mut max_grad = 0.0f32;
    for name in &all_weight_names {
        let dw_ref = graph
            .get(GraphNodeId::new(graph.weight_idx(name).expect("w idx"), GraphNodeIdTy::Gradients))
            .expect("dw get");
        let weight_size = match name.as_str() {
            "w_in" => p_input.weight_size(),
            "w_se1" => (c_r * cmid) as usize,
            "w_se2" => (cmid * c_r) as usize,
            _ => p_block.weight_size(),
        };
        let mut dw_host = vec![0.0f32; weight_size];
        dw_ref.dense().write_to_slice(&mut dw_host).expect("dw write");
        let n_finite = dw_host.iter().filter(|v| v.is_finite()).count();
        let n_nonzero = dw_host.iter().filter(|v| v.abs() > 1e-10).count();
        let max_abs = dw_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        max_grad = max_grad.max(max_abs);
        assert_eq!(n_finite, dw_host.len(), "{name} grad has NaN/Inf");
        if n_nonzero > 0 {
            total_nonzero_weights += 1;
        }
        if name.starts_with("w_in") || name.starts_with("w_se") {
            println!("  {} grad: {}/{} non-zero, max |dw|={:.4e}", name, n_nonzero, dw_host.len(), max_abs);
        }
    }
    assert_eq!(total_nonzero_weights, all_weight_names.len(), "some weight grad all-zero, backward chain broken");

    println!("\nCRCL-v2 完全 ResNet end-to-end smoke PASSED");
    println!("  → 5x5 input + 3 ResBlock + SEBlock + 3 ResBlock の forward + backward 動作");
    println!("  → 全 {} weight (w_in + w_se{{1,2}} + ResBlock × 6 × 2) で grad chain 完成", all_weight_names.len());
    println!("  → CRCL-v2 本格訓練の builder 構築 path 完全実証");
}
