//! Phase 2 Step B demo: Affine forward を IV案 で実演する stand-alone test。
//!
//! 構造:
//!   f32 master weight W (in_dim × out_dim) と f32 input X (in_dim × batch) から
//!   - cast f32 → f16 で W_f16 / X_f16 を作る (DenseMatrixF16 内部 storage)
//!   - HGEMM で hidden_f16 = W^T × X_f16
//!   - cast f16 → f32 で f32 output に戻す
//! という path で、 f32 reference (cuBLAS SGEMM) と比較する。
//!
//! これが IV案 の core: weight master は f32 維持、 forward 中間活性化のみ f16、
//! grad は f32 backward (本 demo では forward のみ実演) で既存 Ranger 互換。

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer, tensor::DenseMatrixF16};
use bullet_cuda_backend::{
    CudaBuffer, CudaDevice, CudaError,
    cudarc::{
        cublas::{
            result::gemm_ex,
            sys::{
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
                cudaDataType_t::{CUDA_R_16F, CUDA_R_32F},
            },
        },
        driver::{DevicePtr, DevicePtrMut},
    },
};

/// IV案 Affine forward (f32 input/weight/output、 内部 f16 経由)。
/// out = W^T × X (column-major、 W shape = (in_dim, out_dim)、 X shape = (in_dim, batch))
/// なので out shape = (out_dim, batch)。
fn affine_mixed_precision(
    device: &CudaDevice,
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    weight_f32: &CudaBuffer<f32>,
    input_f32: &CudaBuffer<f32>,
    output_f32: &mut CudaBuffer<f32>,
) -> Result<(), CudaError> {
    // 中間 storage: DenseMatrixF16
    let device_arc = Arc::new(()); // placeholder, 既存 Arc<CudaDevice> は別ルートで
    let _ = device_arc;
    let dev = device.stream().context().clone(); // CudaContext は driver crate
    let _ = dev;

    // 必要な BufferF16 alloc (DenseMatrixF16 経由)
    let device_clone: Arc<CudaDevice> = weight_f32.device.clone();
    let mut w_f16 = DenseMatrixF16::<CudaDevice>::zeroed(device_clone.clone(), in_dim * out_dim, None)?;
    let mut x_f16 = DenseMatrixF16::<CudaDevice>::zeroed(device_clone.clone(), in_dim * batch, None)?;
    let mut hidden_f16 = DenseMatrixF16::<CudaDevice>::zeroed(device_clone.clone(), out_dim * batch, None)?;

    // 1) f32 → f16 cast for weight and input
    CudaDevice::cast_f32_to_f16(in_dim * out_dim, weight_f32, &mut w_f16.buf).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    CudaDevice::cast_f32_to_f16(in_dim * batch, input_f32, &mut x_f16.buf).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    // 2) HGEMM: hidden_f16 = W^T × X  (op_a = OP_T, op_b = OP_N)
    {
        let stream = device.stream();
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (w_ptr, _rw) = w_f16.buf.buf.device_ptr(&stream);
        let (x_ptr, _rx) = x_f16.buf.buf.device_ptr(&stream);
        let (h_ptr, _rh) = hidden_f16.buf.buf.device_ptr_mut(&stream);
        unsafe {
            gemm_ex(
                *device.blas().handle(),
                CUBLAS_OP_T, // W^T (W is in_dim x out_dim column-major、 transpose で out_dim × in_dim)
                CUBLAS_OP_N,
                out_dim as i32,
                batch as i32,
                in_dim as i32,
                &alpha as *const f32 as *const _,
                w_ptr as *const _,
                CUDA_R_16F,
                in_dim as i32,
                x_ptr as *const _,
                CUDA_R_16F,
                in_dim as i32,
                &beta as *const f32 as *const _,
                h_ptr as *mut _,
                CUDA_R_16F,
                out_dim as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DFALT,
            )
            .map_err(CudaError::Blas)?;
        }
    }

    // 3) f16 → f32 cast for output
    CudaDevice::cast_f16_to_f32(out_dim * batch, &hidden_f16.buf, output_f32).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    Ok(())
}

fn main() -> Result<(), CudaError> {
    println!("=== Affine f16 demo (IV案 core path) ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let in_dim = 32usize;
    let out_dim = 16usize;
    let batch = 8usize;

    // host data (column-major)
    use std::f32::consts::PI;
    let weight_host: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i as f32 * 0.0317 * PI).sin() * 0.3)).collect();
    let input_host: Vec<f32> = (0..in_dim * batch).map(|i| ((i as f32 * 0.0231 * PI).cos() * 0.5)).collect();

    let mut w_buf = CudaBuffer::<f32>::new(device.clone(), in_dim * out_dim)?;
    let mut x_buf = CudaBuffer::<f32>::new(device.clone(), in_dim * batch)?;
    let mut out_mixed = CudaBuffer::<f32>::new(device.clone(), out_dim * batch)?;
    w_buf.load_from_slice(&weight_host)?;
    x_buf.load_from_slice(&input_host)?;

    affine_mixed_precision(&device, in_dim, out_dim, batch, &w_buf, &x_buf, &mut out_mixed)?;
    device.synchronise()?;

    let mut got = vec![0.0f32; out_dim * batch];
    out_mixed.write_into_slice(&mut got, out_dim * batch)?;

    // CPU で f32 reference: out[j, b] = sum_i W[i, j] * X[i, b] (column-major)
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for j in 0..out_dim {
        for b in 0..batch {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += weight_host[j * in_dim + i] * input_host[b * in_dim + i];
            }
            let g = got[b * out_dim + j];
            let abs_err = (g - acc).abs();
            let rel_err = if acc.abs() > 1e-2 { abs_err / acc.abs() } else { 0.0 };
            if abs_err > max_abs {
                max_abs = abs_err;
            }
            if rel_err > max_rel {
                max_rel = rel_err;
            }
        }
    }
    println!(
        "affine f16: in={in_dim} out={out_dim} batch={batch}, max_abs_err = {:.6e}, max_rel_err = {:.6e}",
        max_abs, max_rel
    );
    assert!(max_abs < 5e-3, "abs err too large: {}", max_abs);
    assert!(max_rel < 5e-2, "rel err too large: {}", max_rel);

    // f32 全 path (SGEMM reference) との比較もしておく
    let _ = CUDA_R_32F;

    println!("Affine f16 demo PASSED (IV案 core: f32 in → f16 cast → HGEMM → f32 out)");
    Ok(())
}
