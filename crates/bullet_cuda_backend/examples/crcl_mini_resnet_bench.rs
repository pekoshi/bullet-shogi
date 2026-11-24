//! CRCL-v2 mini ResNet (4 block × 64ch、 spatial 9x9) NPS 実測
//!
//! 1050 Ti で CRCL-v2 通常 conv (192ch × 9x9 × 3x3) の比例縮小版を組み、
//! forward / backward+update の per-batch 時間を計測して NPS を出す。
//!
//! 結果は memory に追記して CRCL-v2 設計の prediction (1450 leaf/sec) との照合に使う。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_mini_resnet_bench --release
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
    println!("=== CRCL-v2 mini ResNet (4 block × 64ch) bench ===");

    let p = Conv2dParams {
        in_channels: 64,
        out_channels: 64,
        spatial_h: 9,
        spatial_w: 9,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };
    let num_blocks = 4usize;
    let batch = 32usize;
    let in_size = p.input_size();
    let out_size = p.output_size();
    println!(
        "shape: Cin=Cout={} H=W={} K={} blocks={} batch={}",
        p.in_channels, p.spatial_h, p.kernel_size, num_blocks, batch
    );

    let builder = GraphBuilder::default();
    let input = builder.new_dense_input("input", Shape::new(in_size, 1));
    let target = builder.new_dense_input("target", Shape::new(out_size, 1));

    let resnet_out = build_resnet(&builder, input, p, num_blocks);

    let sq_err = resnet_out.squared_error(target);
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _root = ones.matmul(sq_err);

    let device = CudaDevice::new(0).expect("CudaDevice::new");
    let graph = builder.build(device);
    println!("graph built (waiting first compile to JIT all conv kernels)");

    let params = RangerParams { min_weight: -1000.0, max_weight: 1000.0, ..Default::default() };
    let mut optimiser = Optimiser::<_, _, Ranger<_>>::new(graph, params).expect("Optimiser::new");

    // Dummy input / target、 batch=32
    {
        let g = optimiser.graph.primary_mut();
        let in_data: Vec<f32> = (0..(in_size * batch)).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
        g.get_input("input").dense_mut().load_from_slice(Some(batch), &in_data).expect("input load");

        g.get_input("target")
            .dense_mut()
            .load_from_slice(Some(batch), &vec![0.0f32; out_size * batch])
            .expect("target load");
    }

    // warmup (first forward は kernel compile/cache warmup を含む)
    println!("warmup (5 iterations)...");
    for _ in 0..5 {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads warm");
        let _ = g.forward().expect("forward warm");
        g.backward().expect("backward warm");
    }
    optimiser.graph.primary_mut().synchronise().expect("sync warm");

    // forward only
    let n_iter = 50usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads");
        let _ = g.forward().expect("forward");
    }
    optimiser.graph.primary_mut().synchronise().expect("sync fw");
    let elapsed = start.elapsed();
    let fw_per_batch_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    let fw_nps = (n_iter * batch) as f64 / elapsed.as_secs_f64();
    println!(
        "forward only: {:.3} ms/batch ({} iter, batch={}) → {:.1} samples/sec",
        fw_per_batch_ms, n_iter, batch, fw_nps
    );

    // forward + backward + update
    let start = Instant::now();
    for _ in 0..n_iter {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads tr");
        let _ = g.forward().expect("forward tr");
        g.backward().expect("backward tr");
        optimiser.update(1.0, 1e-5).expect("update");
    }
    optimiser.graph.primary_mut().synchronise().expect("sync tr");
    let elapsed = start.elapsed();
    let tr_per_batch_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    let tr_nps = (n_iter * batch) as f64 / elapsed.as_secs_f64();
    println!(
        "fwd+bwd+upd : {:.3} ms/batch ({} iter, batch={}) → {:.1} samples/sec",
        tr_per_batch_ms, n_iter, batch, tr_nps
    );

    // 推定 CRCL-v2 通常版 (192ch × 16 block) との比例計算
    // 単純 cost ratio = (192/64)^2 * (16/4) ~ 9 * 4 = 36 倍
    let crcl_ratio = (192.0 / p.in_channels as f64).powi(2) * (16.0 / num_blocks as f64);
    let est_fw_full_ms = fw_per_batch_ms * crcl_ratio;
    let est_fw_full_nps = fw_nps / crcl_ratio;
    println!(
        "estimated CRCL-v2 full (192ch × 16 block, naive scaling x{:.0}): {:.1} ms/batch → {:.1} leaf/sec",
        crcl_ratio, est_fw_full_ms, est_fw_full_nps
    );

    println!("mini ResNet bench DONE");
}
