//! Phase 2 Step 5 smoke: cast f32 ↔ f16 + HGEMM の round-trip 動作確認。
//!
//! 1) f32 行列 a_f32, b_f32 を cast f32→f16 で a_f16, b_f16 化
//! 2) HGEMM (CUDA_R_16F IO, COMPUTE_32F) で c_f16 = a_f16 × b_f16
//! 3) cast f16→f32 で c_recovered_f32 化
//! 4) CPU で f32 reference を計算、 mixed precision 結果と rel err 比較
//!
//! これで「f32 master → f16 cast → HGEMM → f32 cast 回収」 の一連 path が
//! 数値的に正しいことを確認 (Step 4 + Step 5 = IV案 の core pipeline)。

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer};
use bullet_cuda_backend::{
    CudaBuffer, CudaDevice, CudaError,
    cudarc::{
        cublas::{
            result::gemm_ex,
            sys::{
                cublasComputeType_t::CUBLAS_COMPUTE_32F, cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                cublasOperation_t::CUBLAS_OP_N, cudaDataType_t::CUDA_R_16F,
            },
        },
        driver::{DevicePtr, DevicePtrMut},
    },
};
use half::f16;

fn main() -> Result<(), CudaError> {
    println!("=== Mixed precision smoke (f32 → f16 → HGEMM → f32) ===");
    let device = Arc::new(CudaDevice::new(0)?);
    let stream = device.stream();

    // 8x8 の正方行列で確認
    let n = 8usize;
    let nn = n * n;

    // f32 master input (column-major)
    use std::f32::consts::PI;
    let a_f32_host: Vec<f32> = (0..nn).map(|i| (i as f32 * 0.0317 * PI).sin() * 0.5).collect();
    let b_f32_host: Vec<f32> = (0..nn).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.5).collect();

    let mut a_f32_buf = CudaBuffer::<f32>::new(device.clone(), nn)?;
    let mut b_f32_buf = CudaBuffer::<f32>::new(device.clone(), nn)?;
    a_f32_buf.load_from_slice(&a_f32_host)?;
    b_f32_buf.load_from_slice(&b_f32_host)?;

    let mut a_f16_buf = CudaBuffer::<f16>::new(device.clone(), nn)?;
    let mut b_f16_buf = CudaBuffer::<f16>::new(device.clone(), nn)?;
    let mut c_f16_buf = CudaBuffer::<f16>::new(device.clone(), nn)?;
    let mut c_f32_buf = CudaBuffer::<f32>::new(device.clone(), nn)?;

    // 1) f32 → f16 cast
    CudaDevice::cast_f32_to_f16(nn, &a_f32_buf, &mut a_f16_buf).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    CudaDevice::cast_f32_to_f16(nn, &b_f32_buf, &mut b_f16_buf).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    device.synchronise()?;

    // 2) HGEMM: c_f16 = a_f16 × b_f16
    {
        let alpha_f32: f32 = 1.0;
        let beta_f32: f32 = 0.0;
        let (a_ptr, _ra) = a_f16_buf.buf.device_ptr(&stream);
        let (b_ptr, _rb) = b_f16_buf.buf.device_ptr(&stream);
        let (c_ptr, _rc) = c_f16_buf.buf.device_ptr_mut(&stream);
        unsafe {
            gemm_ex(
                *device.blas().handle(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                n as i32,
                n as i32,
                &alpha_f32 as *const f32 as *const _,
                a_ptr as *const _,
                CUDA_R_16F,
                n as i32,
                b_ptr as *const _,
                CUDA_R_16F,
                n as i32,
                &beta_f32 as *const f32 as *const _,
                c_ptr as *mut _,
                CUDA_R_16F,
                n as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DFALT,
            )
            .map_err(CudaError::Blas)?;
        }
        device.synchronise()?;
    }
    println!("HGEMM done");

    // 3) f16 → f32 cast (回収)
    CudaDevice::cast_f16_to_f32(nn, &c_f16_buf, &mut c_f32_buf).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    device.synchronise()?;

    let mut c_recovered = vec![0.0f32; nn];
    c_f32_buf.write_into_slice(&mut c_recovered, nn)?;

    // 4) f32 reference (column-major)
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..n {
                acc += a_f32_host[kk * n + i] * b_f32_host[j * n + kk];
            }
            let got = c_recovered[j * n + i];
            let abs_err = (got - acc).abs();
            // rel err は acc が 0.01 以上のときだけ意味あり (0 付近では noise が膨らむ)
            let rel_err = if acc.abs() > 1e-2 { abs_err / acc.abs() } else { 0.0 };
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }
    }
    println!("round-trip: max_abs_err = {:.6e}, max_rel_err = {:.6e} (acc>0.01 のみ)", max_abs_err, max_rel_err);
    // f16 round 2 回 + HGEMM 内部 round で abs_err 1e-3 程度想定 (n=8 の matmul)
    assert!(max_abs_err < 5e-3, "round-trip abs err too large: {}", max_abs_err);
    assert!(max_rel_err < 5e-2, "round-trip rel err too large: {}", max_rel_err);

    println!("Mixed precision smoke PASSED (IV案 core pipeline 確認)");
    Ok(())
}
