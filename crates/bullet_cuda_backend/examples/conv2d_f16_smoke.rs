//! Phase 2 CRCL-v2 残課題: Conv2d mixed precision forward smoke。
//!
//! 構造 (CRCL-v2 inference 用):
//!   x f32 (N×Cin×H×W)
//!     → im2col_f16_nchw    → col f16 (NHW × CinKK row-major)
//!     → HGEMM              → out f16 (NHW × Cout)
//!       weight f16 (Cout × CinKK) を cast f32→f16 で作る
//!       cublasGemmEx: GEMM (M=Cout, N=NHW, K=CinKK) with op_a=transpose だが
//!       weight (Cout × CinKK row-major) = (CinKK × Cout col-major) と扱える
//!     → permute_gemm_out_f16_to_f32_nchw → y f32 (N×Cout×Hout×Wout)
//!
//! 検証: CPU reference (naive conv) と比較。
//!
//! これで CRCL-v2 ResNet で Conv2d を mixed precision (HGEMM Tensor Core) で
//! forward 実行できる base infrastructure 完成。 backward は次段 (im2col_grad + 2 HGEMM)。

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer};
use bullet_cuda_backend::{
    CudaBuffer, CudaDevice, CudaError,
    cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
};
use half::f16;

fn main() -> Result<(), CudaError> {
    println!("=== Conv2d mixed precision forward smoke ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let n_batch = 2i32;
    let cin = 4i32;
    let cout = 8i32;
    let h = 5i32;
    let w = 5i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;
    println!("shapes: N={n_batch} Cin={cin} H={h} W={w} → Cout={cout} Hout={hout} Wout={wout} K={k}");

    let in_size = (n_batch * cin * h * w) as usize;
    let weight_size = (cout * cin * k * k) as usize;
    let cin_kk = (cin * k * k) as usize;
    let nhw = (n_batch * hout * wout) as usize;
    let out_size = (n_batch * cout * hout * wout) as usize;
    let col_size = nhw * cin_kk;
    let gemm_out_size = nhw * (cout as usize);

    // host data
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..in_size).map(|i| ((i as f32 * 0.0231 * PI).cos() * 0.4)).collect();
    let w_host: Vec<f32> = (0..weight_size).map(|i| ((i as f32 * 0.0173 * PI).sin() * 0.25)).collect();

    // device alloc
    let mut x_buf = CudaBuffer::<f32>::new(device.clone(), in_size)?;
    let mut w_buf = CudaBuffer::<f32>::new(device.clone(), weight_size)?;
    let mut col_f16 = CudaBuffer::<f16>::new(device.clone(), col_size)?;
    let mut w_f16 = CudaBuffer::<f16>::new(device.clone(), weight_size)?;
    let mut gemm_out_f16 = CudaBuffer::<f16>::new(device.clone(), gemm_out_size)?;
    let mut y_buf = CudaBuffer::<f32>::new(device.clone(), out_size)?;
    x_buf.load_from_slice(&x_host)?;
    w_buf.load_from_slice(&w_host)?;

    // step 1: im2col_f16
    {
        let func = device.module().load_function("im2col_f16_nchw").map_err(CudaError::Driver)?;
        let stream = device.stream();
        // grid: (col_idx=N*HW, ceil_div(CinKK, tpb), 1)、 block: (tpb, 1, 1)
        let tpb: u32 = 64;
        let grid_x = nhw as u32;
        let grid_y = ((cin_kk as u32) + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
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

    // step 2: weight f32 → f16 cast
    CudaDevice::cast_f32_to_f16(weight_size, &w_buf, &mut w_f16).map_err(|_| {
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;

    // step 3: HGEMM
    //   col (NHW × CinKK) row-major = (CinKK × NHW) column-major
    //   weight (Cout × CinKK) row-major = (CinKK × Cout) column-major
    //   gemm: out (Cout × NHW) col-major = weight^T (Cout × CinKK col-major) × col (CinKK × NHW col-major)
    //         つまり trans_a=true (weight (CinKK × Cout) → (Cout × CinKK))、 trans_b=false
    CudaDevice::hgemm_f16(
        cout as usize,
        nhw,
        cin_kk,
        1.0,
        &w_f16,
        true,
        cin_kk, // weight^T、 leading dim = CinKK (col-major raw 表現)
        &col_f16,
        false,
        cin_kk, // col、 leading dim = CinKK
        0.0,
        &mut gemm_out_f16,
        cout as usize, // out (Cout × NHW)、 leading dim = Cout
    )
    .map_err(|_| CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN)))?;

    // step 4: permute f16 → f32 NCHW
    {
        let func = device.module().load_function("permute_gemm_out_f16_to_f32_nchw").map_err(CudaError::Driver)?;
        let stream = device.stream();
        let block_x: u32 = 16;
        let block_y: u32 = 16;
        let grid_x = ((cout as u32) + block_x - 1) / block_x;
        let grid_y = ((nhw as u32) + block_y - 1) / block_y;
        let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (block_x, block_y, 1), shared_mem_bytes: 0 };
        let (gp, _rg) = gemm_out_f16.buf.device_ptr(&stream);
        let (yp, _ry) = y_buf.buf.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_batch)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&gp)
                .arg(&yp)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }

    device.synchronise()?;

    // CPU reference
    let mut y_ref = vec![0.0f32; out_size];
    for n in 0..n_batch as usize {
        for co in 0..cout as usize {
            for ho in 0..hout as usize {
                for wo in 0..wout as usize {
                    let mut acc = 0.0f32;
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
                                acc += x_host[x_idx] * w_host[w_idx];
                            }
                        }
                    }
                    let y_idx = ((n * cout as usize + co) * hout as usize + ho) * wout as usize + wo;
                    y_ref[y_idx] = acc;
                }
            }
        }
    }

    let mut got = vec![0.0f32; out_size];
    y_buf.write_into_slice(&mut got, out_size)?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..out_size {
        let abs_err = (got[i] - y_ref[i]).abs();
        let rel_err = if y_ref[i].abs() > 1e-2 { abs_err / y_ref[i].abs() } else { 0.0 };
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
    }
    println!("vs CPU ref: max_abs={:.6e}, max_rel={:.6e}", max_abs, max_rel);
    assert!(max_rel < 5e-2, "rel err too large");
    assert!(max_abs < 1e-2, "abs err too large");
    println!("Conv2d mixed precision forward smoke PASSED");
    println!("  → im2col_f16 + HGEMM + permute path で CRCL-v2 Conv2d inference 可能");
    Ok(())
}
