//! Phase 2 Step A1 smoke: `DenseMatrixF16<CudaDevice>` の動作確認。
//!
//! - zeroed alloc
//! - load_from_slice (host → device)
//! - write_to_slice (device → host) round-trip
//! - set_batch_size (resize, zero-fill)

use std::sync::Arc;

use acyclib::device::{Device, tensor::DenseMatrixF16};
use bullet_cuda_backend::{CudaDevice, CudaError};
use half::f16;

fn main() -> Result<(), CudaError> {
    println!("=== DenseMatrixF16 smoke ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let single_size = 8usize;
    let initial_batch = Some(4usize);
    let mut m = DenseMatrixF16::<CudaDevice>::zeroed(device.clone(), single_size, initial_batch)?;
    assert_eq!(m.single_size(), single_size);
    assert_eq!(m.batch_size(), Some(4));
    assert_eq!(m.size(), 32);

    // host data 流し込み
    let host_data: Vec<f16> = (0..32).map(|i| f16::from_f32((i as f32 * 0.0173) - 0.27)).collect();
    m.load_from_slice(Some(4), &host_data)?;

    // 取り出して一致確認
    let mut recovered = vec![f16::ZERO; 32];
    m.write_to_slice(&mut recovered)?;
    for i in 0..32 {
        assert_eq!(host_data[i].to_f32(), recovered[i].to_f32(), "round-trip mismatch at {i}");
    }
    println!("round-trip OK (32 elem)");

    // set_batch_size で resize (拡大)
    m.set_batch_size(Some(8))?;
    assert_eq!(m.size(), 64);
    // 拡大時は zero fill されている (拡大すると buf alloc し直し)
    let mut zeroed_check = vec![f16::ONE; 64]; // f16::ONE で埋めて diff 確認
    m.write_to_slice(&mut zeroed_check)?;
    for v in &zeroed_check {
        assert_eq!(v.to_f32(), 0.0, "resize zero-fill failed");
    }
    println!("set_batch_size resize + zero-fill OK (64 elem)");

    // set_batch_size で縮小 (内部 buf size 内、 zero-fill のみ)
    m.set_batch_size(Some(2))?;
    assert_eq!(m.size(), 16);
    let mut small = vec![f16::ONE; 16];
    m.write_to_slice(&mut small)?;
    for v in &small {
        assert_eq!(v.to_f32(), 0.0, "shrink zero-fill failed");
    }
    println!("set_batch_size shrink OK (16 elem)");

    println!("DenseMatrixF16 smoke PASSED");
    Ok(())
}
