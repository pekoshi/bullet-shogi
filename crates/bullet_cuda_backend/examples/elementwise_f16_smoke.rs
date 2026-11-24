//! Phase 2 Step 3 smoke: elementwise f16 kernel (silu_f16, add_silu_f16) の動作確認。
//!
//! 各 kernel を直接 launch し、 f32 reference と rel error を比較する。

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer};
use bullet_cuda_backend::{
    CudaBuffer, CudaDevice, CudaError,
    cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
};
use half::f16;

fn silu_ref(x: f32) -> f32 {
    let sig = 1.0 / (1.0 + (-x).exp());
    x * sig
}

fn main() -> Result<(), CudaError> {
    println!("=== elementwise f16 smoke (silu_f16, add_silu_f16) ===");
    let device = Arc::new(CudaDevice::new(0)?);
    let stream = device.stream();

    let n = 1024usize;

    // host input
    let a_host: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32 * 0.0271) - 12.0) * 0.5)).collect();
    let b_host: Vec<f16> = (0..n).map(|i| f16::from_f32(((i as f32 * 0.0181) - 8.0) * 0.3)).collect();

    let mut a_buf = CudaBuffer::<f16>::new(device.clone(), n)?;
    let mut b_buf = CudaBuffer::<f16>::new(device.clone(), n)?;
    let mut out_buf = CudaBuffer::<f16>::new(device.clone(), n)?;
    a_buf.load_from_slice(&a_host)?;
    b_buf.load_from_slice(&b_host)?;

    // === silu_f16 ===
    {
        let func = device.module().load_function("silu_f16").map_err(CudaError::Driver)?;
        let threads: u32 = 256;
        let blocks = (n as u32).div_ceil(threads);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let (a_ptr, _ra) = a_buf.buf.device_ptr(&stream);
        let (out_ptr, _ro) = out_buf.buf.device_ptr_mut(&stream);
        let n_i32 = n as i32;
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_i32)
                .arg(&a_ptr)
                .arg(&out_ptr)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        device.synchronise()?;
    }
    let mut out_host = vec![f16::ZERO; n];
    out_buf.write_into_slice(&mut out_host, n)?;

    let mut max_rel = 0.0f32;
    for i in 0..n {
        let x = a_host[i].to_f32();
        let ref_val = silu_ref(x);
        let got = out_host[i].to_f32();
        let abs_err = (got - ref_val).abs();
        let rel_err = if ref_val.abs() > 1e-3 { abs_err / ref_val.abs() } else { abs_err };
        if rel_err > max_rel {
            max_rel = rel_err;
        }
    }
    println!("silu_f16: max_rel_err = {:.6e} (n={n})", max_rel);
    assert!(max_rel < 5e-2, "silu_f16 rel err too large: {}", max_rel);

    // === add_silu_f16 ===
    {
        let func = device.module().load_function("add_silu_f16").map_err(CudaError::Driver)?;
        let threads: u32 = 256;
        let blocks = (n as u32).div_ceil(threads);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        let (a_ptr, _ra) = a_buf.buf.device_ptr(&stream);
        let (b_ptr, _rb) = b_buf.buf.device_ptr(&stream);
        let (out_ptr, _ro) = out_buf.buf.device_ptr_mut(&stream);
        let n_i32 = n as i32;
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&n_i32)
                .arg(&a_ptr)
                .arg(&b_ptr)
                .arg(&out_ptr)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        device.synchronise()?;
    }
    out_buf.write_into_slice(&mut out_host, n)?;

    let mut max_rel2 = 0.0f32;
    for i in 0..n {
        let x = a_host[i].to_f32() + b_host[i].to_f32();
        let ref_val = silu_ref(x);
        let got = out_host[i].to_f32();
        let abs_err = (got - ref_val).abs();
        let rel_err = if ref_val.abs() > 1e-3 { abs_err / ref_val.abs() } else { abs_err };
        if rel_err > max_rel2 {
            max_rel2 = rel_err;
        }
    }
    println!("add_silu_f16: max_rel_err = {:.6e} (n={n})", max_rel2);
    assert!(max_rel2 < 5e-2, "add_silu_f16 rel err too large: {}", max_rel2);

    println!("elementwise f16 smoke PASSED");
    Ok(())
}
