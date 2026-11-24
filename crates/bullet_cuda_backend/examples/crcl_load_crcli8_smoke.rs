//! `.crcli8` を読んで Conv2dInt8::from_i8_weight で init し、 forward 動作確認する smoke。
//!
//! 使い方:
//!   cargo run -p bullet_cuda_backend --example crcl_load_crcli8_smoke --release -- \
//!     /tmp/crcl_ckpt2/ptq_test-1.crcli8 [channels]
//!
//! (channels 引数は省略時 32、 PTQ tool に渡したのと同じ値)

use std::sync::Arc;

use bullet_cuda_backend::{
    CudaError,
    cudarc::{cublas::CudaBlas, driver::CudaContext},
    inference::{Conv2dInt8, Conv2dInt8Params, Int8Kernels, read_crcli8},
};

fn main() -> Result<(), CudaError> {
    let path = std::env::args().nth(1).expect("usage: crcl_load_crcli8_smoke <.crcli8> [channels]");
    let channels: i32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(32);

    println!("=== load {path} (channels={channels}) ===");
    let layers = read_crcli8(&path).expect("read .crcli8");
    println!("read {} layers from .crcli8", layers.len());
    for layer in &layers {
        println!(
            "  {:>10}  n={:7}  scale={:8.2}  i8[0..4]={:?}",
            layer.name,
            layer.weight_i8.len(),
            layer.scale_w,
            &layer.weight_i8[..4.min(layer.weight_i8.len())]
        );
    }

    // CUDA + cuBLAS + kernels
    let ctx = CudaContext::new(0).map_err(CudaError::Driver)?;
    let stream = ctx.default_stream();
    let blas = Arc::new(CudaBlas::new(stream.clone()).map_err(CudaError::Blas)?);
    let kernels = Int8Kernels::new(&ctx)?;

    // Conv2dInt8 を 1 layer 分 init (動作確認用、 batch=1)
    let layer = &layers[0];
    let nb = 1i32;
    let params = Conv2dInt8Params {
        in_channels: channels,
        out_channels: channels,
        spatial_h: 9,
        spatial_w: 9,
        kernel_size: 3,
        padding: 1,
        stride: 1,
        scale_x: 254.0, // placeholder、 inference では calibration で決める
        scale_w: layer.scale_w,
    };
    let expected_n = (channels * channels * 3 * 3) as usize;
    assert_eq!(layer.weight_i8.len(), expected_n, "channels={channels} と .crcli8 layer size 不一致");

    let mut conv = Conv2dInt8::from_i8_weight(params, &layer.weight_i8, kernels, blas, stream.clone())?;
    println!("Conv2dInt8 init from i8 weight OK");

    // dummy input でも forward 実行
    let in_size = params.input_size();
    let x_host: Vec<f32> = (0..(nb as usize * in_size)).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let input = stream.memcpy_stod(&x_host).map_err(CudaError::Driver)?;
    let mut output = stream.alloc_zeros::<f32>(params.output_size(nb)).map_err(CudaError::Driver)?;

    conv.forward(nb, &input, &mut output)?;
    stream.synchronize().map_err(CudaError::Driver)?;
    let host = stream.memcpy_dtov(&output).map_err(CudaError::Driver)?;
    println!("forward done. output[0..4]={:?}", &host[0..4]);
    assert!(host.iter().take(50).all(|v| v.is_finite()), "output has NaN/Inf");
    println!("crcli8 → Conv2dInt8 inference smoke PASSED");
    Ok(())
}
