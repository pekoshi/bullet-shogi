//! CRCL-v2 改善 A: SEBlock forward smoke (dlshogi 知見、 5 block 間隔挿入)。
//!
//! SEBlock 構造 (NCHW):
//!   x (N×C×H×W)
//!     → global_avg_pool (N×C)
//!     → FC1 (C → C/r、 r=4 reduction ratio)
//!     → ReLU
//!     → FC2 (C/r → C)
//!     → Sigmoid
//!     → channel_wise_mul (N×C×H×W * N×C → N×C×H×W)
//!   出力: x の channel-wise scaling
//!
//! hand-crafted で kernel + matmul の chain を実行、 CPU reference と一致確認。

use std::sync::Arc;

use acyclib::device::{Device, DeviceBuffer};
use bullet_cuda_backend::{
    CudaBuffer, CudaDevice, CudaError,
    cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
};

fn main() -> Result<(), CudaError> {
    println!("=== SEBlock forward smoke (CRCL-v2 改善 A) ===");

    let device = Arc::new(CudaDevice::new(0)?);

    let n_batch = 2usize;
    let c = 16usize;
    let h = 9usize;
    let w_dim = 9usize;
    let r = 4usize; // reduction ratio
    let c_r = c / r;

    let n_total = n_batch * c * h * w_dim;
    let nc = n_batch * c;
    let weight1_size = c * c_r;
    let weight2_size = c_r * c;

    // host init
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..n_total).map(|i| (i as f32 * 0.0231 * PI).sin() * 0.5).collect();
    let w1_host: Vec<f32> = (0..weight1_size).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.2).collect();
    let w2_host: Vec<f32> = (0..weight2_size).map(|i| (i as f32 * 0.0117 * PI).cos() * 0.25).collect();

    let mut x_buf = CudaBuffer::<f32>::new(device.clone(), n_total)?;
    let mut pool_buf = CudaBuffer::<f32>::new(device.clone(), nc)?;
    let mut fc1_buf = CudaBuffer::<f32>::new(device.clone(), n_batch * c_r)?;
    let mut fc2_buf = CudaBuffer::<f32>::new(device.clone(), nc)?;
    let mut out_buf = CudaBuffer::<f32>::new(device.clone(), n_total)?;
    let mut w1_buf = CudaBuffer::<f32>::new(device.clone(), weight1_size)?;
    let mut w2_buf = CudaBuffer::<f32>::new(device.clone(), weight2_size)?;
    x_buf.load_from_slice(&x_host)?;
    w1_buf.load_from_slice(&w1_host)?;
    w2_buf.load_from_slice(&w2_host)?;

    // step 1: global_avg_pool
    {
        let func = device.module().load_function("global_avg_pool_f32").map_err(CudaError::Driver)?;
        let stream = device.stream();
        let tpb: u32 = 64;
        let cfg = LaunchConfig { grid_dim: (nc as u32, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: tpb * 4 };
        let (xp, _rx) = x_buf.buf.device_ptr(&stream);
        let (pp, _rp) = pool_buf.buf.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&(n_batch as i32))
                .arg(&(c as i32))
                .arg(&(h as i32))
                .arg(&(w_dim as i32))
                .arg(&xp)
                .arg(&pp)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }

    // step 2: FC1 = W1 (C/r × C col-major) × pool (C × N col-major) → fc1 (C/r × N col-major)
    // ただし pool layout 上、 「N×C」 = column-major で (C, N)、 W1 = (C/r, C) col-major
    // GEMM (m=C/r, n=N, k=C): C = W1 × pool で fc1 = (C/r, N)
    // ただし cuBLAS は通常 col-major、 trans 不要
    use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
    {
        use cudarc::cublas::Gemm;
        let cfg = cudarc::cublas::GemmConfig {
            transa: CUBLAS_OP_N,
            transb: CUBLAS_OP_N,
            m: c_r as i32,
            n: n_batch as i32,
            k: c as i32,
            alpha: 1.0f32,
            lda: c_r as i32,
            ldb: c as i32,
            beta: 0.0f32,
            ldc: c_r as i32,
        };
        unsafe {
            device.blas().gemm(cfg, &w1_buf.buf, &pool_buf.buf, &mut fc1_buf.buf).map_err(CudaError::Blas)?;
        }
    }

    // step 3: ReLU (f32) on fc1
    // bullet には既存 relu kernel あり、 ただし hand smoke では自前 host computation 経由でも OK
    // ここでは既存 BaseOperations::diffable_from_output_fwd を呼ぶ簡易 (実は host 経由が簡単)、
    // ただし host 経由は遅い。 自前 relu kernel を launch:
    // 実は kernels.cu に既存 activate kernel "activate_ReLU_F32" or 同等あり、 grep で確認
    // 簡易: CPU で読み戻して ReLU → 書き戻し (smoke ok)
    device.synchronise()?;
    let mut fc1_host = vec![0.0f32; n_batch * c_r];
    fc1_buf.write_into_slice(&mut fc1_host, n_batch * c_r)?;
    for v in fc1_host.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
    fc1_buf.load_from_slice(&fc1_host)?;

    // step 4: FC2 = W2 (C × C/r col-major) × fc1 (C/r × N col-major) → fc2 (C × N col-major)
    {
        use cudarc::cublas::Gemm;
        let cfg = cudarc::cublas::GemmConfig {
            transa: CUBLAS_OP_N,
            transb: CUBLAS_OP_N,
            m: c as i32,
            n: n_batch as i32,
            k: c_r as i32,
            alpha: 1.0f32,
            lda: c as i32,
            ldb: c_r as i32,
            beta: 0.0f32,
            ldc: c as i32,
        };
        unsafe {
            device.blas().gemm(cfg, &w2_buf.buf, &fc1_buf.buf, &mut fc2_buf.buf).map_err(CudaError::Blas)?;
        }
    }

    // step 5: Sigmoid on fc2 (host 経由 simple smoke)
    device.synchronise()?;
    let mut fc2_host = vec![0.0f32; nc];
    fc2_buf.write_into_slice(&mut fc2_host, nc)?;
    for v in fc2_host.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
    fc2_buf.load_from_slice(&fc2_host)?;

    // step 6: channel_wise_mul (NCHW * NC11 → NCHW)
    {
        let func = device.module().load_function("channel_wise_mul_f32").map_err(CudaError::Driver)?;
        let stream = device.stream();
        let tpb: u32 = 256;
        let blocks = ((n_total as u32) + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        let (xp, _rx) = x_buf.buf.device_ptr(&stream);
        let (sp, _rs) = fc2_buf.buf.device_ptr(&stream);
        let (op, _ro) = out_buf.buf.device_ptr_mut(&stream);
        unsafe {
            stream
                .launch_builder(&func)
                .arg(&(n_batch as i32))
                .arg(&(c as i32))
                .arg(&(h as i32))
                .arg(&(w_dim as i32))
                .arg(&xp)
                .arg(&sp)
                .arg(&op)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }
    device.synchronise()?;

    // CPU reference
    // pool[n, c] = avg_{h, w} x[n, c, h, w]
    // fc1[n, c/r] = sum_c W1[c/r, c] * pool[n, c]
    // relu(fc1)
    // fc2[n, c] = sum_{c/r} W2[c, c/r] * relu_fc1[n, c/r]
    // sigmoid(fc2) = s
    // out[n, c, h, w] = x[n, c, h, w] * s[n, c]
    let hw = h * w_dim;
    let mut pool_ref = vec![0.0f32; nc];
    for n in 0..n_batch {
        for ci in 0..c {
            let mut acc = 0.0f32;
            for i in 0..hw {
                acc += x_host[(n * c + ci) * hw + i];
            }
            pool_ref[n * c + ci] = acc / (hw as f32);
        }
    }
    let mut fc1_ref = vec![0.0f32; n_batch * c_r];
    for n in 0..n_batch {
        for cr_i in 0..c_r {
            let mut acc = 0.0f32;
            for ci in 0..c {
                // W1 col-major: W1[c_r, c] at (cr_i, ci)、 idx = ci * c_r + cr_i
                acc += w1_host[ci * c_r + cr_i] * pool_ref[n * c + ci];
            }
            fc1_ref[n * c_r + cr_i] = acc.max(0.0); // ReLU
        }
    }
    let mut fc2_ref = vec![0.0f32; nc];
    for n in 0..n_batch {
        for ci in 0..c {
            let mut acc = 0.0f32;
            for cr_i in 0..c_r {
                // W2 col-major: W2[c, c_r] at (ci, cr_i)、 idx = cr_i * c + ci
                acc += w2_host[cr_i * c + ci] * fc1_ref[n * c_r + cr_i];
            }
            fc2_ref[n * c + ci] = 1.0 / (1.0 + (-acc).exp()); // Sigmoid
        }
    }
    let mut out_ref = vec![0.0f32; n_total];
    for n in 0..n_batch {
        for ci in 0..c {
            let s = fc2_ref[n * c + ci];
            for i in 0..hw {
                out_ref[(n * c + ci) * hw + i] = x_host[(n * c + ci) * hw + i] * s;
            }
        }
    }

    let mut got = vec![0.0f32; n_total];
    out_buf.write_into_slice(&mut got, n_total)?;

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..n_total {
        let abs_err = (got[i] - out_ref[i]).abs();
        let rel_err = if out_ref[i].abs() > 1e-2 { abs_err / out_ref[i].abs() } else { 0.0 };
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
    }
    println!("SEBlock forward (N={n_batch}, C={c}, H={h}, W={w_dim}, r={r}):");
    println!("  vs CPU ref: max_abs={:.6e}, max_rel={:.6e}", max_abs, max_rel);
    assert!(max_rel < 1e-3, "rel err too large: {}", max_rel);
    println!("SEBlock forward smoke PASSED");
    println!("  → global_avg_pool + 2 FC (FC1/ReLU/FC2/Sigmoid) + channel_wise_mul の chain 動作");
    println!("  → dlshogi 知見「SEBlock 5 block 間隔挿入」 を CRCL-v2 用 kernel として取り入れ");
    Ok(())
}
