//! Phase 2 CRCL-v2 残課題 Conv2d backward mixed precision smoke。
//!
//! forward の im2col_f16 + HGEMM + permute path に対して、 backward chain rule を
//! HGEMM 2 回 + col2im で実装し、 CPU reference と比較する。
//!
//! 構造 (NCHW、 forward):
//!   X (N×Cin×H×W) → col (NHW × CinKK) f16 → HGEMM(weight^T × col) → Y (Cout × NHW) f16 → permute → Y f32 NCHW
//!
//! backward (upstream dY (N×Cout×Hout×Wout) f32 から):
//!   1. dY NCHW → gemm_in (Cout × NHW col-major) f16 (permute_nchw_f32_to_gemm_in_f16)
//!   2. dW = col × dY^T  (CinKK × NHW × NHW × Cout = CinKK × Cout)
//!      = HGEMM(op_a=N col, op_b=T gemm_in)
//!      → dW_f16 → cast → dW_f32 (Cout × CinKK row-major、 forward weight と同 layout)
//!   3. dX_col = dY^T × W^T = ... 実は dX_col = W × dY  (CinKK × NHW)
//!      column-major で考えると: dX_col (CinKK × NHW) = W (CinKK × Cout col-major) × gemm_in (Cout × NHW col-major)
//!      → HGEMM(op_a=N w, op_b=N gemm_in)
//!   4. col2im_f16_to_f32_nchw(dX_col) → dX f32 NCHW (atomic add で accumulate、 事前 0 fill)

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer};
use bullet_cuda_backend::{
    CudaBuffer, CudaDevice, CudaError,
    cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
};
use half::f16;

fn main() -> Result<(), CudaError> {
    println!("=== Conv2d mixed precision backward smoke ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let n_batch = 2i32;
    let cin = 4i32;
    let cout = 6i32;
    let h = 4i32;
    let w = 4i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;

    let in_size = (n_batch * cin * h * w) as usize;
    let weight_size = (cout * cin * k * k) as usize;
    let cin_kk = (cin * k * k) as usize;
    let nhw = (n_batch * hout * wout) as usize;
    let out_size = (n_batch * cout * hout * wout) as usize;
    let col_size = nhw * cin_kk;
    let gemm_io_size = nhw * cout as usize;
    let cout_us = cout as usize;

    // host data
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    let w_host: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.25).collect();
    // upstream dy (deterministic、 任意の f32 values)
    let dy_host: Vec<f32> = (0..out_size).map(|i| (i as f32 * 0.0117 * PI).sin() * 0.2).collect();

    // device alloc
    let mut x_buf = CudaBuffer::<f32>::new(device.clone(), in_size)?;
    let mut w_buf = CudaBuffer::<f32>::new(device.clone(), weight_size)?;
    let mut dy_buf = CudaBuffer::<f32>::new(device.clone(), out_size)?;
    let mut col_f16 = CudaBuffer::<f16>::new(device.clone(), col_size)?;
    let mut w_f16 = CudaBuffer::<f16>::new(device.clone(), weight_size)?;
    let mut gemm_in_f16 = CudaBuffer::<f16>::new(device.clone(), gemm_io_size)?;
    let mut dw_f16 = CudaBuffer::<f16>::new(device.clone(), weight_size)?;
    let mut dx_col_f16 = CudaBuffer::<f16>::new(device.clone(), col_size)?;
    let mut dw_f32 = CudaBuffer::<f32>::new(device.clone(), weight_size)?;
    let mut dx_f32 = CudaBuffer::<f32>::new(device.clone(), in_size)?;

    x_buf.load_from_slice(&x_host)?;
    w_buf.load_from_slice(&w_host)?;
    dy_buf.load_from_slice(&dy_host)?;

    // forward step: im2col_f16 (X → col)
    {
        let func = device.module().load_function("im2col_f16_nchw").map_err(CudaError::Driver)?;
        let stream = device.stream();
        let tpb: u32 = 64;
        let grid_y = ((cin_kk as u32) + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (nhw as u32, grid_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        let (xp, _rx) = x_buf.buf.device_ptr(&stream);
        let (cp, _rc) = col_f16.buf.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_batch)
                .arg(&cin)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&xp)
                .arg(&cp)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }

    // weight f32→f16 cast
    CudaDevice::cast_f32_to_f16(weight_size, &w_buf, &mut w_f16).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    // backward step 1: dY NCHW f32 → gemm_in (Cout × NHW col-major) f16
    {
        let func = device.module().load_function("permute_nchw_f32_to_gemm_in_f16").map_err(CudaError::Driver)?;
        let stream = device.stream();
        let cfg = LaunchConfig {
            grid_dim: (((cout as u32) + 15) / 16, ((nhw as u32) + 15) / 16, 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let (dp, _rd) = dy_buf.buf.device_ptr(&stream);
        let (gp, _rg) = gemm_in_f16.buf.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_batch)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&dp)
                .arg(&gp)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }

    // backward step 2: dW = col^T × dY (forward の weight 形状に accumulate)
    //   col layout: (NHW × CinKK) row-major = (CinKK × NHW) col-major
    //   gemm_in layout: (Cout × NHW) col-major
    //   dW_natural = (CinKK × Cout) col-major ← col (CinKK × NHW col-major) × gemm_in^T (NHW × Cout col-major)
    //   trans_a = false (col)、 trans_b = true (gemm_in^T)
    CudaDevice::hgemm_f16(
        cin_kk,
        cout_us,
        nhw,
        1.0,
        &col_f16,
        false,
        cin_kk,
        &gemm_in_f16,
        true,
        cout_us,
        0.0,
        &mut dw_f16,
        cin_kk,
    )
    .map_err(|_| CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)))?;
    CudaDevice::cast_f16_to_f32(weight_size, &dw_f16, &mut dw_f32).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    // backward step 3: dX_col = W × dY
    //   W layout (Cout × CinKK row-major = CinKK × Cout col-major)
    //   gemm_in (Cout × NHW col-major)
    //   dX_col (CinKK × NHW col-major) = W (CinKK × Cout col-major) × gemm_in (Cout × NHW col-major)
    //   trans_a = false (W)、 trans_b = false (gemm_in)
    CudaDevice::hgemm_f16(
        cin_kk,
        nhw,
        cout_us,
        1.0,
        &w_f16,
        false,
        cin_kk,
        &gemm_in_f16,
        false,
        cout_us,
        0.0,
        &mut dx_col_f16,
        cin_kk,
    )
    .map_err(|_| CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)))?;

    // backward step 4: col2im (dX_col → dX f32 NCHW、 0 init してから atomic add)
    dx_f32.set_zero()?;
    {
        let func = device.module().load_function("col2im_f16_to_f32_nchw").map_err(CudaError::Driver)?;
        let stream = device.stream();
        let tpb: u32 = 64;
        let grid_y = ((cin_kk as u32) + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (nhw as u32, grid_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        let (cp, _rc) = dx_col_f16.buf.device_ptr(&stream);
        let (xp, _rx) = dx_f32.buf.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_batch)
                .arg(&cin)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&cp)
                .arg(&xp)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }

    device.synchronise()?;

    // ===== CPU reference =====
    // dW_ref: dW[co, ci, kh, kw] = sum_n sum_ho sum_wo dY[n, co, ho, wo] * X[n, ci, hi, wi]
    let mut dw_ref = vec![0.0f32; weight_size];
    for n in 0..n_batch as usize {
        for co in 0..cout as usize {
            for ho in 0..hout as usize {
                for wo in 0..wout as usize {
                    let dy_val = dy_host[((n * cout as usize + co) * hout as usize + ho) * wout as usize + wo];
                    for ci in 0..cin as usize {
                        for kh in 0..k as usize {
                            for kw in 0..k as usize {
                                let hi = ho as i32 * stride + kh as i32 - pad;
                                let wi = wo as i32 * stride + kw as i32 - pad;
                                if hi < 0 || hi >= h || wi < 0 || wi >= w {
                                    continue;
                                }
                                let x_idx =
                                    ((n * cin as usize + ci) * h as usize + hi as usize) * w as usize + wi as usize;
                                let w_idx = ((co * cin as usize + ci) * k as usize + kh) * k as usize + kw;
                                dw_ref[w_idx] += dy_val * x_host[x_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // dX_ref: dX[n, ci, hi, wi] = sum_co sum_kh sum_kw dY[n, co, ho, wo] * W[co, ci, kh, kw]
    //   ho = hi - kh + pad、 wo = wi - kw + pad   (stride=1 で簡略)
    let mut dx_ref = vec![0.0f32; in_size];
    for n in 0..n_batch as usize {
        for co in 0..cout as usize {
            for ho in 0..hout as usize {
                for wo in 0..wout as usize {
                    let dy_val = dy_host[((n * cout as usize + co) * hout as usize + ho) * wout as usize + wo];
                    for ci in 0..cin as usize {
                        for kh in 0..k as usize {
                            for kw in 0..k as usize {
                                let hi = ho as i32 * stride + kh as i32 - pad;
                                let wi = wo as i32 * stride + kw as i32 - pad;
                                if hi < 0 || hi >= h || wi < 0 || wi >= w {
                                    continue;
                                }
                                let x_idx =
                                    ((n * cin as usize + ci) * h as usize + hi as usize) * w as usize + wi as usize;
                                let w_idx = ((co * cin as usize + ci) * k as usize + kh) * k as usize + kw;
                                dx_ref[x_idx] += dy_val * w_host[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    let mut dw_got = vec![0.0f32; weight_size];
    let mut dx_got = vec![0.0f32; in_size];
    dw_f32.write_into_slice(&mut dw_got, weight_size)?;
    dx_f32.write_into_slice(&mut dx_got, in_size)?;

    let mut max_dw_abs = 0.0f32;
    let mut max_dw_rel = 0.0f32;
    for i in 0..weight_size {
        let abs_err = (dw_got[i] - dw_ref[i]).abs();
        let rel_err = if dw_ref[i].abs() > 1e-2 { abs_err / dw_ref[i].abs() } else { 0.0 };
        max_dw_abs = max_dw_abs.max(abs_err);
        max_dw_rel = max_dw_rel.max(rel_err);
    }
    let mut max_dx_abs = 0.0f32;
    let mut max_dx_rel = 0.0f32;
    for i in 0..in_size {
        let abs_err = (dx_got[i] - dx_ref[i]).abs();
        let rel_err = if dx_ref[i].abs() > 1e-2 { abs_err / dx_ref[i].abs() } else { 0.0 };
        max_dx_abs = max_dx_abs.max(abs_err);
        max_dx_rel = max_dx_rel.max(rel_err);
    }
    println!("backward dW: max_abs={:.6e}, max_rel={:.6e}", max_dw_abs, max_dw_rel);
    println!("backward dX: max_abs={:.6e}, max_rel={:.6e}", max_dx_abs, max_dx_rel);
    assert!(max_dw_rel < 5e-2, "dW rel err too large");
    assert!(max_dx_rel < 5e-2, "dX rel err too large");

    println!("Conv2d mixed precision backward smoke PASSED");
    println!("  → CRCL-v2 ResNet 用 Conv2d forward + backward が mixed precision で完成");
    Ok(())
}
