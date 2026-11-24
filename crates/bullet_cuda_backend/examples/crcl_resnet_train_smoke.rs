//! CRCL-v2 multi-block ResNet 訓練 smoke (FP32 graph IR、 GCP T4 で実訓練する基盤)
//!
//! 構成:
//!   - input → Conv1x1 (Cin=192) → ResBlock × N → value head (FC)
//!     ※ Conv1x1 は (1x1 Conv 相当の Affine) で代用
//!   - loss = (value - target)^2、 ones.matmul で scalar に reduce
//!   - Ranger optimizer + Normal init
//!
//! N block size を CRCL_BLOCKS env var で制御 (default 4)、 batch を CRCL_BATCH (default 32)、
//! 学習 step を CRCL_STEPS (default 100)。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_resnet_train_smoke --release

use std::time::Instant;

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        builder::{GraphBuilder, GraphBuilderNode, InitSettings},
        like::GraphLike,
    },
    trainer::optimiser::{
        Optimiser,
        ranger::{Ranger, RangerParams},
    },
};
use bullet_cuda_backend::{
    CudaDevice, CudaMarker,
    builder::{Conv2dParams, ConvBuilderExt},
};

fn build_resnet<'a>(
    builder: &'a GraphBuilder<CudaMarker>,
    input: GraphBuilderNode<'a, CudaMarker>,
    p: Conv2dParams,
    num_blocks: usize,
) -> GraphBuilderNode<'a, CudaMarker> {
    let mut x = input;
    for b in 0..num_blocks {
        let w1 = builder.new_weights(
            &format!("b{b}_w1"),
            Shape::new(p.weight_size(), 1),
            InitSettings::Normal { mean: 0.0, stdev: 0.05 },
        );
        let w2 = builder.new_weights(
            &format!("b{b}_w2"),
            Shape::new(p.weight_size(), 1),
            InitSettings::Normal { mean: 0.0, stdev: 0.05 },
        );
        let residual = x;
        let c1 = x.conv2d(w1, p).silu();
        let c2 = c1.conv2d(w2, p);
        x = (c2 + residual).silu();
    }
    x
}

fn main() {
    let num_blocks: usize = std::env::var("CRCL_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let batch: usize = std::env::var("CRCL_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let steps: usize = std::env::var("CRCL_STEPS").ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let cin: i32 = std::env::var("CRCL_CHANNELS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);

    println!("=== CRCL-v2 ResNet training smoke (blocks={num_blocks}, ch={cin}, batch={batch}, steps={steps}) ===");

    let p = Conv2dParams {
        in_channels: cin,
        out_channels: cin,
        spatial_h: 9,
        spatial_w: 9,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };
    let in_size = p.input_size();
    let out_size = p.output_size();

    let builder = GraphBuilder::default();
    let input = builder.new_dense_input("input", Shape::new(in_size, 1));
    let target = builder.new_dense_input("target", Shape::new(out_size, 1));

    let resnet_out = build_resnet(&builder, input, p, num_blocks);

    // loss = sum (resnet_out - target)^2
    let sq_err = resnet_out.squared_error(target);
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _root = ones.matmul(sq_err);

    let device = CudaDevice::new(0).expect("CudaDevice::new");
    let graph = builder.build(device);
    println!("graph built");

    let params = RangerParams { min_weight: -1000.0, max_weight: 1000.0, ..Default::default() };
    let mut optimiser = Optimiser::<_, _, Ranger<_>>::new(graph, params).expect("Optimiser::new");

    // dummy input/target (固定): 訓練が loss 減少することを確認
    {
        let g = optimiser.graph.primary_mut();
        let in_data: Vec<f32> = (0..(in_size * batch)).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
        g.get_input("input").dense_mut().load_from_slice(Some(batch), &in_data).expect("input load");
        g.get_input("target")
            .dense_mut()
            .load_from_slice(Some(batch), &vec![0.0f32; out_size * batch])
            .expect("target load");
    }

    // warmup forward
    {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads");
        let _ = g.forward().expect("forward warmup");
    }

    let mut losses = Vec::with_capacity(steps);
    let lr = 1e-5f32; // Ranger 初期 step は SGD-like、 lr 抑える

    let start = Instant::now();
    for step in 0..steps {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads");
        let loss = g.forward().expect("forward");
        g.backward().expect("backward");
        optimiser.update(1.0, lr).expect("update");
        losses.push(loss);
        if step < 5 || step % (steps / 10).max(1) == 0 {
            println!("step {step:3}: loss={loss:.4}");
        }
    }
    optimiser.graph.primary_mut().synchronise().expect("sync");
    let elapsed = start.elapsed();
    let per_step_ms = elapsed.as_secs_f64() * 1000.0 / steps as f64;

    let loss_first = losses[0];
    let loss_last = *losses.last().unwrap();
    let drop_ratio = (loss_first - loss_last) / loss_first.max(1e-6);
    println!(
        "training done: {steps} steps in {:.1} ms ({:.2} ms/step)、 \
         loss {loss_first:.4} → {loss_last:.4} (drop {:.1}%)",
        elapsed.as_secs_f64() * 1000.0,
        per_step_ms,
        drop_ratio * 100.0,
    );
    let samples_per_sec = (steps * batch) as f64 / elapsed.as_secs_f64();
    println!("training throughput: {:.1} samples/sec", samples_per_sec);

    assert!(loss_last < loss_first, "loss should decrease (first={loss_first} last={loss_last})");
    println!("CRCL-v2 ResNet training PASSED ({steps} steps、 loss drop {:.1}%)", drop_ratio * 100.0);
}
