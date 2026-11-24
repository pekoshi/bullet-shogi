//! Phase 2 graph IR 統合 G1 smoke: `MixedUnary<CudaDevice>` の動作確認。
//!
//! Tensor (DenseMatrix f32) を input/output に、 内部 f16 cast → silu_f16 → cast の
//! mixed precision Unary が graph IR の DeviceOperation として動作することを確認する。
//!
//! 既存 Unary (純 f32 SiLU) と並べて、 f32 reference との一致を比較。

use std::{num::NonZeroUsize, sync::Arc};

use acyclib::device::{
    Device,
    function::{DeviceOperation, MixedUnary, UnaryOp},
    tensor::{Shape, Tensor, TensorRef},
};
use bullet_cuda_backend::{CudaDevice, CudaError};

fn main() -> Result<(), CudaError> {
    println!("=== MixedUnary smoke (graph IR G1) ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let single_size = 128usize;
    let batch_size = 16usize;
    let total = single_size * batch_size;

    // Tensor (DenseMatrix f32) を input/output として alloc
    let shape = Shape::new(single_size, 1);
    let mut t_in = Tensor::new(device.clone(), shape, None, true).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    let mut t_out = Tensor::new(device.clone(), shape, None, true).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    // input 初期化
    use std::f32::consts::PI;
    let input_host: Vec<f32> = (0..total).map(|i| (i as f32 * 0.0231 * PI).sin() * 1.5).collect();
    t_in.load_dense_from_slice(Some(batch_size), &input_host).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    // output も batch_size set (MixedUnary は input/output batch_size 同期 check)
    {
        let mut out_borrow = t_out.dense_mut().unwrap();
        out_borrow.set_batch_size(Some(batch_size)).map_err(|_| {
            CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
        })?;
    }

    let ref_in = TensorRef::new(t_in);
    let ref_out = TensorRef::new(t_out);

    // MixedUnary::Silu instance を作って execute
    let mixed = MixedUnary::<CudaDevice>::new(ref_in.clone(), ref_out.clone(), UnaryOp::Silu);
    mixed.execute().map_err(|e| {
        eprintln!("MixedUnary execute failed: {:?}", e);
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    device.synchronise()?;

    // output 取り出し
    let got: Vec<f32> = ref_out.get_dense_vals().map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    // CPU reference: SiLU(x) = x * sigmoid(x)
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..total {
        let x = input_host[i];
        let sig = 1.0 / (1.0 + (-x).exp());
        let ref_val = x * sig;
        let abs_err = (got[i] - ref_val).abs();
        let rel_err = if ref_val.abs() > 1e-2 { abs_err / ref_val.abs() } else { 0.0 };
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
    }
    println!("MixedUnary(Silu): single_size={single_size} batch={batch_size} (total={total})");
    println!("  max_abs_err = {:.6e}", max_abs);
    println!("  max_rel_err = {:.6e}", max_rel);
    assert!(max_rel < 5e-2, "rel err too large: {}", max_rel);
    assert!(max_abs < 5e-3, "abs err too large: {}", max_abs);

    // 同じ instance で 2 回目 execute (中間 buffer 再利用確認)
    mixed.execute().map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    device.synchronise()?;
    let got2: Vec<f32> = ref_out.get_dense_vals().map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    for i in 0..total {
        assert_eq!(got[i], got2[i], "2nd execute mismatch at {i}");
    }
    println!("2nd execute reused mid_f16 buffer: same result");

    let _ = NonZeroUsize::new(1);
    println!("MixedUnary smoke PASSED (graph IR G1 complete)");
    Ok(())
}
