//! Crclv2InferenceNet を `.crcli8` から構築 + forward smoke
//!
//! 使い方:
//!   cargo run -p bullet_cuda_backend --example crcl_net_load_smoke --release -- \
//!     /tmp/crcl_ckpt2/ptq_test-1.crcli8 <num_blocks> <channels>
//!
//! 例:
//!   ./crcl_net_load_smoke /tmp/crcl_ckpt2/ptq_test-1.crcli8 2 32

use std::{sync::Arc, time::Instant};

use bullet_cuda_backend::{
    CudaError,
    cudarc::{cublas::CudaBlas, driver::CudaContext},
    inference::{Crclv2InferenceNet, read_crcli8},
};

fn main() -> Result<(), CudaError> {
    let path = std::env::args().nth(1).expect("usage: crcl_net_load_smoke <.crcli8> <num_blocks> <channels>");
    let num_blocks: usize = std::env::args().nth(2).expect("num_blocks").parse().expect("usize");
    let channels: i32 = std::env::args().nth(3).expect("channels").parse().expect("i32");
    let batch: i32 = std::env::var("CRCL_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    println!("=== Crclv2InferenceNet smoke (blocks={num_blocks} ch={channels} batch={batch}) ===");

    // CUDA + cuBLAS
    let ctx = CudaContext::new(0).map_err(CudaError::Driver)?;
    let stream = ctx.default_stream();
    let blas = Arc::new(CudaBlas::new(stream.clone()).map_err(CudaError::Blas)?);

    // .crcli8 読込
    let layers = read_crcli8(&path).expect("read crcli8");
    println!("read {} layers from {path}", layers.len());
    assert!(layers.len() >= num_blocks * 2, "expected at least {} layers, got {}", num_blocks * 2, layers.len());

    // CRCL-v2 net 構築 (scale_x はとりあえず 254 (= 127/0.5) で placeholder)
    let scale_x_placeholder = 254.0f32;
    let mut net = Crclv2InferenceNet::from_crcli8_layers(
        layers,
        num_blocks,
        channels,
        9,
        9, // spatial 9x9
        3,
        1,
        1, // kernel=3, pad=1, stride=1
        scale_x_placeholder,
        &ctx,
        blas,
        stream.clone(),
    )?;
    println!("Crclv2InferenceNet built: {} ResBlocks", net.blocks.len());

    // dummy input
    let n_io = (batch as usize) * (channels as usize) * 81;
    let x_host: Vec<f32> = (0..n_io).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let input = stream.memcpy_stod(&x_host).map_err(CudaError::Driver)?;
    let mut output = stream.alloc_zeros::<f32>(n_io).map_err(CudaError::Driver)?;

    // 動作確認 1 回
    net.forward(batch, &input, &mut output)?;
    stream.synchronize().map_err(CudaError::Driver)?;
    let host = stream.memcpy_dtov(&output).map_err(CudaError::Driver)?;
    println!("forward done. output[0..4]={:?}", &host[0..4]);
    assert!(host.iter().take(100).all(|v| v.is_finite()), "output has NaN/Inf");

    // bench
    let warmup = 30usize;
    for _ in 0..warmup {
        net.forward(batch, &input, &mut output)?;
    }
    stream.synchronize().map_err(CudaError::Driver)?;

    let n_iter = 100usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        net.forward(batch, &input, &mut output)?;
    }
    stream.synchronize().map_err(CudaError::Driver)?;
    let elapsed = start.elapsed();
    let per_batch_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    let leaf_per_sec = batch as f64 / (per_batch_ms / 1000.0);

    println!("{} block chain: {:.3} ms/batch (batch={batch}) → {:.1} leaf/sec", num_blocks, per_batch_ms, leaf_per_sec);

    // 16 block 換算 (CRCL-v2 設計)
    let full_ms = per_batch_ms * 16.0 / num_blocks as f64;
    let full_lps = batch as f64 / (full_ms / 1000.0);
    println!("extrapolated to 16 blocks: {:.1} ms/batch → {:.1} leaf/sec", full_ms, full_lps);
    println!("Crclv2InferenceNet smoke PASSED");
    Ok(())
}
