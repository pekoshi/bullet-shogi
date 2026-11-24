//! CRCL-v2 INT8 ResNet block end-to-end smoke + bench (1 block 単独)
//!
//! ResBlockInt8 1 個 (Conv1 → SiLU → Conv2 → +residual → SiLU、 192ch × 9x9) を測定し、
//! 16 block chain への外挿で CRCL-v2 full inference 時間を推定。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_resblock_int8_smoke --release

use std::{sync::Arc, time::Instant};

use bullet_cuda_backend::{
    CudaError,
    cudarc::{cublas::CudaBlas, driver::CudaContext},
    inference::{Conv2dInt8Params, Int8Kernels, ResBlockInt8},
};

fn main() -> Result<(), CudaError> {
    let nb: i32 = std::env::var("CRCL_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    println!("=== ResBlockInt8 1 block smoke (192ch × 9x9, batch={nb}) ===");

    let ctx = CudaContext::new(0).map_err(CudaError::Driver)?;
    // 仮説検証: event tracking を切ると複数 conv chain で event chain 累積がなくなり高速化?
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.default_stream();
    let blas = Arc::new(CudaBlas::new(stream.clone()).map_err(CudaError::Blas)?);
    let int8_kernels = Int8Kernels::new(&ctx)?;
    println!("kernels JIT compiled");

    let cin = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;
    let n_w = (cin * cin * k * k) as usize;
    let n_io = (nb * cin * h * w) as usize;

    let w_host: Vec<f32> = (0..n_w).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();
    let x_host: Vec<f32> = (0..n_io).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_w = 127.0 / w_max;
    let scale_x = 127.0 / x_max;

    let params = Conv2dInt8Params {
        in_channels: cin,
        out_channels: cin,
        spatial_h: h,
        spatial_w: w,
        kernel_size: k,
        padding: 1,
        stride: 1,
        scale_x,
        scale_w,
    };

    let w1 = stream.memcpy_stod(&w_host).map_err(CudaError::Driver)?;
    let w2 = stream.memcpy_stod(&w_host).map_err(CudaError::Driver)?;
    let mut block = ResBlockInt8::new(params, &w1, &w2, int8_kernels, blas, stream.clone())?;
    println!("ResBlockInt8 ready");

    let input = stream.memcpy_stod(&x_host).map_err(CudaError::Driver)?;
    let mut output = stream.alloc_zeros::<f32>(n_io).map_err(CudaError::Driver)?;

    // 動作確認
    block.forward(nb, &input, &mut output)?;
    stream.synchronize().map_err(CudaError::Driver)?;
    let host = stream.memcpy_dtov(&output).map_err(CudaError::Driver)?;
    println!("forward done. output[0..4]={:?}", &host[0..4]);
    assert!(host.iter().take(100).all(|v| v.is_finite()), "output has NaN/Inf");

    // bench
    let warmup = 50usize;
    for _ in 0..warmup {
        block.forward(nb, &input, &mut output)?;
    }
    stream.synchronize().map_err(CudaError::Driver)?;

    let n_iter = 200usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        block.forward(nb, &input, &mut output)?;
    }
    stream.synchronize().map_err(CudaError::Driver)?;
    let elapsed = start.elapsed();
    let per_block_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    println!("1 ResBlockInt8: {:.3} ms/call (batch={nb})", per_block_ms);

    // CRCL-v2 full (16 block + head) への外挿
    // 16 block + 入出力 conv 1x1 (~ 1 conv 相当 < block 1 個) + 5 head (軽量、 ~ 1 block 相当)
    // 概算: 16 block × per_block_ms + overhead 約 2 block 分
    let full_blocks_only = per_block_ms * 16.0;
    let full_with_io = per_block_ms * 18.0; // 16 block + IO conv + head 約 2 block 分相当
    let lps_blocks_only = nb as f64 / (full_blocks_only / 1000.0);
    let lps_with_io = nb as f64 / (full_with_io / 1000.0);
    println!(
        "CRCL-v2 full estimate:\n  16 blocks only: {:.1} ms/batch → {:.1} leaf/sec\n  16 blocks + IO + 5 head (~18 block): {:.1} ms/batch → {:.1} leaf/sec",
        full_blocks_only, lps_blocks_only, full_with_io, lps_with_io,
    );
    let design = 1450.0f64;
    println!(
        "vs 設計目標 1450 leaf/sec (blocks only): {:.0}% ({})",
        lps_blocks_only / design * 100.0,
        if lps_blocks_only >= design { "✓ ACHIEVED" } else { "未達" }
    );
    Ok(())
}
