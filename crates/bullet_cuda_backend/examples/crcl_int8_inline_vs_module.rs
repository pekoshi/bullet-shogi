//! CRCL-v2 INT8 inference: inline pipeline vs Conv2dInt8 module の 1:1 比較
//!
//! 同じ context/stream/buffer 上で両 path を直列に bench、 2x slowdown の有無を確認。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_int8_inline_vs_module --release
use std::{sync::Arc, time::Instant};

use bullet_cuda_backend::{
    CudaError,
    cudarc::{
        cublas::{
            CudaBlas, result,
            sys::{self, cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t},
        },
        driver::{CudaContext, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
    },
    inference::{Conv2dInt8, Conv2dInt8Params, Int8Kernels},
};

const N_ITER: usize = 200;
const WARMUP: usize = 100; // GPU clock boost に時間かかる、 cuBLAS algorithm cache も

fn main() -> Result<(), CudaError> {
    let nb: i32 = std::env::var("CRCL_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    println!("=== inline vs module bench (batch={nb}) ===");

    let ctx = CudaContext::new(0).map_err(CudaError::Driver)?;
    let stream = ctx.default_stream();
    let blas = Arc::new(CudaBlas::new(stream.clone()).map_err(CudaError::Blas)?);
    let kernels = Int8Kernels::new(&ctx)?;
    println!(
        "ctx multi_stream_mode={} event_tracking={} managing_sync={}",
        ctx.is_in_multi_stream_mode(),
        ctx.is_event_tracking(),
        ctx.is_managing_stream_synchronization(),
    );

    let cin = 192i32;
    let cout = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;
    let cin_kk = (cin * k * k) as usize;
    let nhw = (nb * hout * wout) as usize;

    let x_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host: Vec<f32> = (0..(cout * cin * k * k) as usize).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();
    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_x = 127.0f32 / x_max;
    let scale_w = 127.0f32 / w_max;
    let dq_scale = 1.0f32 / (scale_x * scale_w);

    let input = stream.memcpy_stod(&x_host).map_err(CudaError::Driver)?;
    let weight_f32 = stream.memcpy_stod(&w_host).map_err(CudaError::Driver)?;

    // === path A: inline pipeline ===
    // weight_i8 / col_i8 / gemm_out / output を pre-alloc
    let mut weight_i8_a = stream.alloc_zeros::<i8>((cout * cin * k * k) as usize).map_err(CudaError::Driver)?;
    let mut col_i8_a = stream.alloc_zeros::<i8>(cin_kk * nhw).map_err(CudaError::Driver)?;
    let mut gemm_out_a = stream.alloc_zeros::<i32>((cout as usize) * nhw).map_err(CudaError::Driver)?;
    let mut output_a = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).map_err(CudaError::Driver)?;

    // weight 量子化 1 度
    {
        let n_elem = (cout * cin * k * k) as i64;
        let tpb = 256u32;
        let blocks = ((n_elem as u64 + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&kernels.quant_weight)
                .arg(&n_elem)
                .arg(&scale_w)
                .arg(&weight_f32)
                .arg(&mut weight_i8_a)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }
    stream.synchronize().map_err(CudaError::Driver)?;

    // inline bench
    let inline_run = || -> Result<(), CudaError> {
        // step 1: im2col_i8
        let n_col = (nb * hout * wout) as u32;
        let cin_kk_u32 = (cin * k * k) as u32;
        let tpb = 256u32;
        let blocks_y = (cin_kk_u32 + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (n_col, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        // NOTE: closure 内で stream / col_i8_a の借用は ownership 移すので、 inner fn 化して回避する形に書き換える
        Ok(())
    };
    let _ = inline_run; // unused

    // 直接 bench loop (closure を使わず外でループ)
    println!("[Path A] inline pipeline warmup...");
    for _ in 0..WARMUP {
        // step 1
        let n_col = (nb * hout * wout) as u32;
        let cin_kk_u32 = (cin * k * k) as u32;
        let tpb = 256u32;
        let blocks_y = (cin_kk_u32 + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (n_col, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&kernels.im2col_i8)
                .arg(&nb)
                .arg(&cin)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&scale_x)
                .arg(&input)
                .arg(&mut col_i8_a)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        // step 2: GEMM
        let alpha: i32 = 1;
        let beta: i32 = 0;
        let m = cout;
        let n_dim = nhw as i32;
        let kk = cin_kk as i32;
        let handle = *blas.handle();
        let (a, _ra) = weight_i8_a.device_ptr(&stream);
        let (b, _rb) = col_i8_a.device_ptr(&stream);
        let (c, _rc) = gemm_out_a.device_ptr_mut(&stream);
        unsafe {
            result::gemm_ex(
                handle,
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m,
                n_dim,
                kk,
                &alpha as *const i32 as *const _,
                a as *const _,
                sys::cudaDataType_t::CUDA_R_8I,
                kk,
                b as *const _,
                sys::cudaDataType_t::CUDA_R_8I,
                kk,
                &beta as *const i32 as *const _,
                c as *mut _,
                sys::cudaDataType_t::CUDA_R_32I,
                m,
                cublasComputeType_t::CUBLAS_COMPUTE_32I,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(CudaError::Blas)?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        // step 3: dequant_permute
        let tile = 32u32;
        let n_col_o = (nb * hout * wout) as u32;
        let grid_x = (cout as u32 + tile - 1) / tile;
        let grid_y = (n_col_o + tile - 1) / tile;
        let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&kernels.dequant_permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(&gemm_out_a)
                .arg(&mut output_a)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }
    stream.synchronize().map_err(CudaError::Driver)?;

    let start = Instant::now();
    for _ in 0..N_ITER {
        let n_col = (nb * hout * wout) as u32;
        let cin_kk_u32 = (cin * k * k) as u32;
        let tpb = 256u32;
        let blocks_y = (cin_kk_u32 + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (n_col, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&kernels.im2col_i8)
                .arg(&nb)
                .arg(&cin)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&scale_x)
                .arg(&input)
                .arg(&mut col_i8_a)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        let alpha: i32 = 1;
        let beta: i32 = 0;
        let m = cout;
        let n_dim = nhw as i32;
        let kk = cin_kk as i32;
        let handle = *blas.handle();
        let (a, _ra) = weight_i8_a.device_ptr(&stream);
        let (b, _rb) = col_i8_a.device_ptr(&stream);
        let (c, _rc) = gemm_out_a.device_ptr_mut(&stream);
        unsafe {
            result::gemm_ex(
                handle,
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                m,
                n_dim,
                kk,
                &alpha as *const i32 as *const _,
                a as *const _,
                sys::cudaDataType_t::CUDA_R_8I,
                kk,
                b as *const _,
                sys::cudaDataType_t::CUDA_R_8I,
                kk,
                &beta as *const i32 as *const _,
                c as *mut _,
                sys::cudaDataType_t::CUDA_R_32I,
                m,
                cublasComputeType_t::CUBLAS_COMPUTE_32I,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .map_err(CudaError::Blas)?;
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
        let tile = 32u32;
        let n_col_o = (nb * hout * wout) as u32;
        let grid_x = (cout as u32 + tile - 1) / tile;
        let grid_y = (n_col_o + tile - 1) / tile;
        let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&kernels.dequant_permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(&gemm_out_a)
                .arg(&mut output_a)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
    }
    stream.synchronize().map_err(CudaError::Driver)?;
    let inline_elapsed = start.elapsed();
    let inline_ms = inline_elapsed.as_secs_f64() * 1000.0 / N_ITER as f64;
    println!("[Path A] inline: {:.3} ms/call", inline_ms);

    // === path B: Conv2dInt8 module、 同じ buffer / context ===
    println!("[Path B] Conv2dInt8 module warmup...");
    let params = Conv2dInt8Params {
        in_channels: cin,
        out_channels: cout,
        spatial_h: h,
        spatial_w: w,
        kernel_size: k,
        padding: pad,
        stride,
        scale_x,
        scale_w,
    };
    let mut conv = Conv2dInt8::from_f32_weight(params, &weight_f32, kernels.clone(), blas.clone(), stream.clone())?;
    let mut output_b = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).map_err(CudaError::Driver)?;
    for _ in 0..WARMUP {
        conv.forward(nb, &input, &mut output_b)?;
    }
    stream.synchronize().map_err(CudaError::Driver)?;

    let start = Instant::now();
    for _ in 0..N_ITER {
        conv.forward(nb, &input, &mut output_b)?;
    }
    stream.synchronize().map_err(CudaError::Driver)?;
    let module_elapsed = start.elapsed();
    let module_ms = module_elapsed.as_secs_f64() * 1000.0 / N_ITER as f64;
    println!("[Path B] module: {:.3} ms/call", module_ms);

    println!("ratio module/inline = {:.2}x   (1.0 = same)", module_ms / inline_ms);
    Ok(())
}
