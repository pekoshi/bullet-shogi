//! HGEMM smoke test (Phase 2 Step 1 後半)
//!
//! `CudaBuffer<half::f16>` を入出力で `cublasGemmEx` を呼び、
//! - IO  : CUDA_R_16F
//! - compute: CUBLAS_COMPUTE_32F (Tensor Core 経路、 sm_75+ で速い、 sm_61 でも fallback で動く)

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

fn hgemm_call(
    device: &CudaDevice,
    n: i32,
    a_buf: &CudaBuffer<f16>,
    b_buf: &CudaBuffer<f16>,
    c_buf: &mut CudaBuffer<f16>,
) -> Result<(), CudaError> {
    let stream = device.stream();
    let alpha_f32: f32 = 1.0;
    let beta_f32: f32 = 0.0;
    let (a_ptr, _ra) = a_buf.buf.device_ptr(&stream);
    let (b_ptr, _rb) = b_buf.buf.device_ptr(&stream);
    let (c_ptr, _rc) = c_buf.buf.device_ptr_mut(&stream);
    unsafe {
        gemm_ex(
            *device.blas().handle(),
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            n,
            n,
            n,
            &alpha_f32 as *const f32 as *const _,
            a_ptr as *const _,
            CUDA_R_16F,
            n,
            b_ptr as *const _,
            CUDA_R_16F,
            n,
            &beta_f32 as *const f32 as *const _,
            c_ptr as *mut _,
            CUDA_R_16F,
            n,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DFALT,
        )
        .map_err(CudaError::Blas)?;
    }
    Ok(())
}

fn main() -> Result<(), CudaError> {
    println!("=== HGEMM smoke (CUDA_R_16F IO, CUBLAS_COMPUTE_32F) ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let n = 8usize;
    let size = n * n;
    let mut a_buf: CudaBuffer<f16> = CudaBuffer::<f16>::new(device.clone(), size)?;
    let mut b_buf: CudaBuffer<f16> = CudaBuffer::<f16>::new(device.clone(), size)?;
    let mut c_buf: CudaBuffer<f16> = CudaBuffer::<f16>::new(device.clone(), size)?;

    // Step 1: identity 行列で check
    let mut a_host = vec![f16::ZERO; size];
    let mut b_host = vec![f16::ZERO; size];
    for i in 0..n {
        a_host[i * n + i] = f16::ONE;
        b_host[i * n + i] = f16::ONE;
    }
    a_buf.load_from_slice(&a_host)?;
    b_buf.load_from_slice(&b_host)?;
    println!("BufferF16 alloc + load OK ({} elem)", size);

    hgemm_call(&device, n as i32, &a_buf, &b_buf, &mut c_buf)?;
    device.synchronise()?;

    let mut c_host = vec![f16::ZERO; size];
    c_buf.write_into_slice(&mut c_host, size)?;
    let mut max_diff = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let got = c_host[j * n + i].to_f32();
            let diff = (got - expected).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }
    println!("identity check: max_diff = {:.6e}", max_diff);
    assert!(max_diff < 1e-3, "identity HGEMM failed: max_diff={}", max_diff);

    // Step 2: random 行列で f32 reference と比較
    use std::f32::consts::PI;
    for i in 0..size {
        a_host[i] = f16::from_f32((i as f32 * 0.0317 * PI).sin() * 0.5);
        b_host[i] = f16::from_f32((i as f32 * 0.0231 * PI).cos() * 0.5);
    }
    a_buf.load_from_slice(&a_host)?;
    b_buf.load_from_slice(&b_host)?;
    hgemm_call(&device, n as i32, &a_buf, &b_buf, &mut c_buf)?;
    device.synchronise()?;
    c_buf.write_into_slice(&mut c_host, size)?;

    let a_f32: Vec<f32> = a_host.iter().map(|v| v.to_f32()).collect();
    let b_f32: Vec<f32> = b_host.iter().map(|v| v.to_f32()).collect();
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..n {
                acc += a_f32[kk * n + i] * b_f32[j * n + kk];
            }
            let got = c_host[j * n + i].to_f32();
            let abs_err = (got - acc).abs();
            let rel_err = if acc.abs() > 1e-6 { abs_err / acc.abs() } else { abs_err };
            if abs_err > max_abs_err {
                max_abs_err = abs_err;
            }
            if rel_err > max_rel_err {
                max_rel_err = rel_err;
            }
        }
    }
    println!("random matmul: max_abs_err = {:.6e}, max_rel_err = {:.6e}", max_abs_err, max_rel_err);
    assert!(max_rel_err < 5e-2, "rel err too large: {}", max_rel_err);

    println!("HGEMM smoke PASSED");
    Ok(())
}
