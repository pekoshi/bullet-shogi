//! sparse_to_dense_f16 smoke (Phase 2 Step 2)
//!
//! `CudaDevice::sparse_to_dense_f16` を呼び、 BufferI32 (sparse indices) →
//! BufferF16 (dense one-hot、 1.0 = f16::ONE) で active feature が正しく
//! 立っていることを確認する。

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer};
use bullet_cuda_backend::{CudaDevice, CudaError};
use half::f16;

fn main() -> Result<(), CudaError> {
    println!("=== sparse_to_dense_f16 smoke ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let batch_size = 4;
    let size = 16; // dense (input feature dim)
    let nnz = 3; // max_active

    // sparse indices: batch=4, 各 batch は max_active=3、 -1 で終端
    // batch 0: [2, 5, -1]
    // batch 1: [0, 7, 15]
    // batch 2: [-1, ...]   (active 0)
    // batch 3: [10, 11, -1]
    let mut sparse_host = vec![-1i32; batch_size * nnz];
    sparse_host[0] = 2;
    sparse_host[1] = 5;
    sparse_host[2] = -1;
    sparse_host[3] = 0;
    sparse_host[4] = 7;
    sparse_host[5] = 15;
    sparse_host[6] = -1;
    sparse_host[7] = -1;
    sparse_host[8] = -1;
    sparse_host[9] = 10;
    sparse_host[10] = 11;
    sparse_host[11] = -1;

    let mut sparse_buf = <CudaDevice as Device>::BufferI32::new(device.clone(), batch_size * nnz)?;
    sparse_buf.load_from_slice(&sparse_host)?;

    let mut dense_buf = <CudaDevice as Device>::BufferF16::new(device.clone(), batch_size * size)?;

    // kernel 呼び出し
    CudaDevice::sparse_to_dense_f16(batch_size, size, nnz, &sparse_buf, &mut dense_buf).map_err(|e| {
        eprintln!("sparse_to_dense_f16 op failed: {:?}", e);
        CudaError::Driver(cudarc::driver::DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_UNKNOWN))
    })?;
    device.synchronise()?;

    let mut dense_host = vec![f16::ZERO; batch_size * size];
    dense_buf.write_into_slice(&mut dense_host, batch_size * size)?;

    // 期待値: batch 0: [2, 5]=1, others=0
    //         batch 1: [0, 7, 15]=1, others=0
    //         batch 2: 全 0
    //         batch 3: [10, 11]=1, others=0
    let expected: [&[usize]; 4] = [&[2, 5], &[0, 7, 15], &[], &[10, 11]];

    for b in 0..batch_size {
        let slice = &dense_host[b * size..(b + 1) * size];
        for (i, v) in slice.iter().enumerate() {
            let want = if expected[b].contains(&i) { 1.0 } else { 0.0 };
            let got = v.to_f32();
            assert!((got - want).abs() < 1e-6, "batch {b} idx {i}: expected {want}, got {got}");
        }
        println!("batch {b}: OK (active = {:?})", expected[b]);
    }

    println!("sparse_to_dense_f16 smoke PASSED");
    Ok(())
}
