//! CRCL-v2 ResNet 1-block smoke
//!
//! 構成: y = SiLU(Conv2(SiLU(Conv1(x)))) + x   (residual connection)
//!
//! 確認内容:
//!   - Conv → SiLU → Conv → +residual を builder で組める
//!   - graph compile が複数 conv をチェーンで通過する
//!   - 1-step training で loss が下がる (Conv1/Conv2 両方の weight に対する grad が流れる)
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_resnet_block_smoke --release

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        builder::{GraphBuilder, InitSettings},
        like::GraphLike,
    },
    trainer::optimiser::{
        Optimiser,
        ranger::{Ranger, RangerParams},
    },
};
use bullet_cuda_backend::{
    CudaDevice,
    builder::{Conv2dParams, ConvBuilderExt},
};

fn main() {
    println!("=== CRCL-v2 ResNet 1-block smoke ===");

    // 小さめの shape (compile 時間短縮)
    let p = Conv2dParams {
        in_channels: 16,
        out_channels: 16,
        spatial_h: 9,
        spatial_w: 9,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };
    let in_size = p.input_size();
    let out_size = p.output_size();
    assert_eq!(in_size, out_size, "residual には Cin==Cout かつ spatial 維持が必要");

    let builder = GraphBuilder::default();
    let input = builder.new_dense_input("input", Shape::new(in_size, 1));
    let w1 = builder.new_weights("w1", Shape::new(p.weight_size(), 1), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    let w2 = builder.new_weights("w2", Shape::new(p.weight_size(), 1), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    let target = builder.new_dense_input("target", Shape::new(out_size, 1));

    // ResNet 1-block: y = SiLU(Conv2(SiLU(Conv1(x)))) + x
    let c1 = input.conv2d(w1, p).silu();
    let c2 = c1.conv2d(w2, p).silu();
    let block_out = c2 + input;

    // loss = sum (block_out - target)^2
    let sq_err = block_out.squared_error(target);
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _root = ones.matmul(sq_err);

    let device = CudaDevice::new(0).expect("CudaDevice::new");
    let graph = builder.build(device);
    println!("graph built (Conv1 → SiLU → Conv2 → SiLU → +residual → squared_error)");

    let params = RangerParams { min_weight: -1000.0, max_weight: 1000.0, ..Default::default() };
    let mut optimiser = Optimiser::<_, _, Ranger<_>>::new(graph, params).expect("Optimiser::new");

    {
        let g = optimiser.graph.primary_mut();
        let in_ref = g.get_input("input");
        let mut in_dense = in_ref.dense_mut();
        in_dense.load_from_slice(Some(1), &vec![0.5f32; in_size]).expect("input load");
    }
    {
        let g = optimiser.graph.primary_mut();
        let tg_ref = g.get_input("target");
        let mut tg_dense = tg_ref.dense_mut();
        tg_dense.load_from_slice(Some(1), &vec![0.0f32; out_size]).expect("target load");
    }

    // 数 step 回して安定的に下がることを確認
    let mut losses = Vec::new();
    for step in 0..10 {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads");
        let loss = g.forward().expect("forward");
        g.backward().expect("backward");
        optimiser.update(1.0, 1e-5).expect("update");
        losses.push(loss);
        println!("step {step}: loss={loss:.6}");
    }

    let loss_first = losses[0];
    let loss_last = losses[losses.len() - 1];
    assert!(loss_last < loss_first, "loss should decrease over steps (first={loss_first} last={loss_last})");
    let drop_ratio = (loss_first - loss_last) / loss_first;
    println!("loss drop over 10 steps: {:.2}%", 100.0 * drop_ratio);
    println!("ResNet 1-block smoke PASSED");
}
