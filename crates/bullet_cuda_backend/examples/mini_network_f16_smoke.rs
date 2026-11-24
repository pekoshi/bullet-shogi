//! Phase 2 Step B/C combined smoke: 多 op mini network の mixed precision forward。
//!
//! 構造 (2-layer NNUE-like、 sparse 入力なし、 dense input から):
//!   X (f32, in_dim × batch)
//!     → Affine1 (W1 f32 master, in_dim × hid_dim)  → h1_f16
//!     → SiLU                                       → a1_f16
//!     → Affine2 (W2 f32 master, hid_dim × out_dim) → out_f16
//!     → cast f16→f32                               → out_f32
//!
//! mixed precision path: 中間 activation (h1, a1, out_pre) は DenseMatrixF16、
//! weight master は CudaBuffer<f32> で保持、 forward 直前に f16 cast。
//!
//! 全 path 結果を f32 reference (純 SGEMM + SiLU) と数値比較する。
//! IV案 の training step forward が動作することを実演。
//!
//! NOTE: backward + optimizer 統合は本 demo 外 (Phase 2 Step C の本格 task、
//! acyclib autograd graph IR 改造が必要)。

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
                cudaDataType_t::CUDA_R_16F,
            },
        },
        driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
    },
};
use half::f16;

/// W (in×out, column-major) を transpose して f16 input X (in×batch) と HGEMM し、
/// f16 output (out×batch) を作る。 weight は f32 master、 forward 直前に f16 cast 化。
fn affine_f16_forward(
    device: &CudaDevice,
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    weight_f32: &CudaBuffer<f32>,
    input_f16: &DenseMatrixF16<CudaDevice>,
    output_f16: &mut DenseMatrixF16<CudaDevice>,
) -> Result<(), CudaError> {
    // weight を f16 に cast (中間 buffer)
    let mut w_f16 = DenseMatrixF16::<CudaDevice>::zeroed(weight_f32.device.clone(), in_dim * out_dim, None)?;
    CudaDevice::cast_f32_to_f16(in_dim * out_dim, weight_f32, &mut w_f16.buf).map_err(map_op_err)?;

    let stream = device.stream();
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let (w_ptr, _rw) = w_f16.buf.buf.device_ptr(&stream);
    let (x_ptr, _rx) = input_f16.buf.buf.device_ptr(&stream);
    let (o_ptr, _ro) = output_f16.buf.buf.device_ptr_mut(&stream);
    unsafe {
        gemm_ex(
            *device.blas().handle(),
            CUBLAS_OP_T,
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
            o_ptr as *mut _,
            CUDA_R_16F,
            out_dim as i32,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DFALT,
        )
        .map_err(CudaError::Blas)?;
    }
    Ok(())
}

/// SiLU on BufferF16 (kernels.cu の silu_f16 を呼ぶ)
fn silu_f16_inplace(device: &CudaDevice, size: usize, buf: &mut DenseMatrixF16<CudaDevice>) -> Result<(), CudaError> {
    let func = device.module().load_function("silu_f16").map_err(CudaError::Driver)?;
    let stream = device.stream();
    let threads: u32 = 256;
    let blocks = (size as u32).div_ceil(threads);
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    let (ptr, _r) = buf.buf.buf.device_ptr_mut(&stream);
    // in-place: input = output = ptr。 ただし silu_f16 は idx 毎独立なので OK
    unsafe {
        stream.launch_builder(&func).arg(&(size as i32)).arg(&ptr).arg(&ptr).launch(cfg).map_err(CudaError::Driver)?;
    }
    Ok(())
}

fn map_op_err<E: std::fmt::Debug>(_e: acyclib::device::OperationError<E>) -> CudaError {
    CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
}

fn main() -> Result<(), CudaError> {
    println!("=== Mini network mixed precision forward smoke ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let in_dim = 32usize;
    let hid_dim = 64usize;
    let out_dim = 8usize;
    let batch = 16usize;

    // host master data (column-major)
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..in_dim * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    let w1_host: Vec<f32> = (0..in_dim * hid_dim).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.25).collect();
    let w2_host: Vec<f32> = (0..hid_dim * out_dim).map(|i| (i as f32 * 0.0117 * PI).cos() * 0.35).collect();

    // device alloc (master weights = f32)
    let mut x_buf = CudaBuffer::<f32>::new(device.clone(), in_dim * batch)?;
    let mut w1_buf = CudaBuffer::<f32>::new(device.clone(), in_dim * hid_dim)?;
    let mut w2_buf = CudaBuffer::<f32>::new(device.clone(), hid_dim * out_dim)?;
    x_buf.load_from_slice(&x_host)?;
    w1_buf.load_from_slice(&w1_host)?;
    w2_buf.load_from_slice(&w2_host)?;

    // mixed precision forward
    // step 1: x_f32 → x_f16 (input cast)
    let mut x_f16 = DenseMatrixF16::<CudaDevice>::zeroed(device.clone(), in_dim * batch, None)?;
    CudaDevice::cast_f32_to_f16(in_dim * batch, &x_buf, &mut x_f16.buf).map_err(map_op_err)?;

    // step 2: h1 = W1^T × x (HGEMM, all f16)
    let mut h1_f16 = DenseMatrixF16::<CudaDevice>::zeroed(device.clone(), hid_dim * batch, None)?;
    affine_f16_forward(&device, in_dim, hid_dim, batch, &w1_buf, &x_f16, &mut h1_f16)?;

    // step 3: SiLU on h1
    silu_f16_inplace(&device, hid_dim * batch, &mut h1_f16)?;

    // step 4: out = W2^T × silu(h1) (HGEMM)
    let mut out_f16 = DenseMatrixF16::<CudaDevice>::zeroed(device.clone(), out_dim * batch, None)?;
    affine_f16_forward(&device, hid_dim, out_dim, batch, &w2_buf, &h1_f16, &mut out_f16)?;

    // step 5: cast f16 → f32 でホスト読み出し
    let mut out_f32_buf = CudaBuffer::<f32>::new(device.clone(), out_dim * batch)?;
    CudaDevice::cast_f16_to_f32(out_dim * batch, &out_f16.buf, &mut out_f32_buf).map_err(map_op_err)?;
    device.synchronise()?;

    let mut got = vec![0.0f32; out_dim * batch];
    out_f32_buf.write_into_slice(&mut got, out_dim * batch)?;

    // CPU reference: 純 f32 path で同じ network を forward
    let mut h1_ref = vec![0.0f32; hid_dim * batch];
    for j in 0..hid_dim {
        for b in 0..batch {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += w1_host[j * in_dim + i] * x_host[b * in_dim + i];
            }
            // SiLU
            let sig = 1.0 / (1.0 + (-acc).exp());
            h1_ref[b * hid_dim + j] = acc * sig;
        }
    }
    let mut out_ref = vec![0.0f32; out_dim * batch];
    for j in 0..out_dim {
        for b in 0..batch {
            let mut acc = 0.0f32;
            for i in 0..hid_dim {
                acc += w2_host[j * hid_dim + i] * h1_ref[b * hid_dim + i];
            }
            out_ref[b * out_dim + j] = acc;
        }
    }

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..out_dim * batch {
        let abs_err = (got[i] - out_ref[i]).abs();
        let rel_err = if out_ref[i].abs() > 1e-2 { abs_err / out_ref[i].abs() } else { 0.0 };
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
    }
    println!("mini network: in={in_dim} hid={hid_dim} out={out_dim} batch={batch}");
    println!("  max_abs_err = {:.6e}", max_abs);
    println!("  max_rel_err = {:.6e} (out_ref>0.01 のみ)", max_rel);
    // 2 HGEMM + SiLU + 数回 cast = f16 round 累積、 rel err 5e-2 程度想定
    assert!(max_rel < 5e-2, "rel err too large: {}", max_rel);
    assert!(max_abs < 5e-2, "abs err too large: {}", max_abs);

    println!("Mini network f16 forward smoke PASSED");
    println!("  → IV案 Step B/C forward path 動作実証 (backward + Ranger 統合は別 task)");
    Ok(())
}
