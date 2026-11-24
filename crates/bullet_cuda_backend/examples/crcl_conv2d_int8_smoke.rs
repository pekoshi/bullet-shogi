//! CRCL-v2 inference module (Conv2dInt8) の動作確認 + bench
//!
//! stand-alone Conv2dInt8 (graph IR 外) を 192ch × 9x9 × 3x3 で組み、
//! 既存 INT8 pipeline smoke と同じ batch 動作確認 + per-call 時間測定。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_conv2d_int8_smoke --release

use std::{sync::Arc, time::Instant};

use bullet_cuda_backend::{
    cudarc::{cublas::CudaBlas, driver::CudaContext},
    inference::{Conv2dInt8, Conv2dInt8Params, Int8Kernels},
};

fn main() {
    println!("=== CRCL-v2 inference::Conv2dInt8 smoke (192ch × 9x9 × 3x3) ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let blas = Arc::new(CudaBlas::new(stream.clone()).expect("CudaBlas::new"));

    let kernels = Int8Kernels::new(&ctx).expect("Int8Kernels::new");
    println!("Int8Kernels JIT compiled");

    let batch: i32 = std::env::var("CRCL_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    let cin = 192i32;
    let cout = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;

    // 入力 / weight 用 dummy data
    let x_host: Vec<f32> = (0..(batch * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host: Vec<f32> = (0..(cout * cin * k * k) as usize).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();
    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_x = 127.0 / x_max;
    let scale_w = 127.0 / w_max;
    println!("scale_x={scale_x:.2}, scale_w={scale_w:.2}");

    let params = Conv2dInt8Params {
        in_channels: cin,
        out_channels: cout,
        spatial_h: h,
        spatial_w: w,
        kernel_size: k,
        padding: 1,
        stride: 1,
        scale_x,
        scale_w,
    };

    let weight_f32 = stream.memcpy_stod(&w_host).unwrap();
    let mut conv = Conv2dInt8::from_f32_weight(params, &weight_f32, kernels, blas.clone(), stream.clone())
        .expect("Conv2dInt8 init");
    println!("Conv2dInt8 weight quantized to i8");

    let input = stream.memcpy_stod(&x_host).unwrap();
    let mut output = stream.alloc_zeros::<f32>(params.output_size(batch)).unwrap();

    // warmup + 動作確認
    conv.forward(batch, &input, &mut output).expect("forward");
    stream.synchronize().unwrap();
    let out_host = stream.memcpy_dtov(&output).unwrap();
    let hout = params.out_h();
    let wout = params.out_w();
    let center = out_host[((cout / 2 * hout + hout / 2) * wout + wout / 2) as usize];
    println!("forward done. output[center] = {:.4}", center);
    assert!(center.is_finite() && center.abs() < 1e3, "output abnormal: {center}");

    // === bench ===
    for _ in 0..5 {
        conv.forward(batch, &input, &mut output).expect("warm");
    }
    stream.synchronize().unwrap();

    let n_iter = 200usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        conv.forward(batch, &input, &mut output).expect("bench");
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let per_call_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    let full_ms = per_call_ms * 32.0;
    let full_lps = batch as f64 / (full_ms / 1000.0);
    println!(
        "Conv2dInt8 (batch={batch}): {:.3} ms/call → CRCL-v2 full (32 conv) {:.1} ms/batch → {:.1} leaf/sec",
        per_call_ms, full_ms, full_lps
    );
    let design = 1450.0f64;
    println!(
        "vs 設計目標 1450 leaf/sec: {:.0}% ({})",
        full_lps / design * 100.0,
        if full_lps >= design { "✓ ACHIEVED" } else { "未達" }
    );
}
