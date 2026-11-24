//! CRCL-v2 INT8 cublasGemmEx (DP4A 経由) smoke + bench
//!
//! 1050 Ti (sm_61) は INT8 tensor core を持たないが、 DP4A 命令で 4x speedup 可能。
//! cublasGemmEx を CUDA_R_8I で呼び、 1050 Ti で動作 + FP32 比 何倍速度向上するか測定。
//!
//! 確認内容:
//!   1. host で fp32 input/weight を i8 量子化 (per-tensor scale max(abs) ベース)
//!   2. cublasGemmEx (a_type=CUDA_R_8I, b_type=CUDA_R_8I, c_type=CUDA_R_32I,
//!      compute_type=CUBLAS_COMPUTE_32I) で int8 SGEMM
//!   3. host で dequantize、 naive forward (FP32) と結果一致 (相対誤差 5% 以内)
//!   4. batch=32 で per-call 時間を FP32 (smoke 3.35ms) と比較
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_int8_gemm_smoke --release

use std::time::Instant;

use bullet_cuda_backend::cudarc::{
    cublas::{
        CudaBlas, Gemm, GemmConfig, result,
        sys::{self, cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t},
    },
    driver::{CudaContext, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

const IM2COL_SRC: &str = r#"
extern "C" __global__ void Im2ColNCHWFp32(
    int N, int Cin, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride,
    const float* __restrict__ x,
    float* __restrict__ col)
{
    long total = (long)N * Cin * K * K * Hout * Wout;
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    long whw = (long)Hout * Wout;
    int CinKK = Cin * K * K;
    int wo = tid % Wout;
    int ho = (tid / Wout) % Hout;
    long rest_pre = tid / whw;
    int n = rest_pre / CinKK;
    int rest = rest_pre % CinKK;
    int ci = rest / (K * K);
    int kh = (rest / K) % K;
    int kw = rest % K;
    int hi = ho * Stride + kh - Pad;
    int wi = wo * Stride + kw - Pad;
    long col_idx = (long)n * whw + (long)ho * Wout + wo;
    long col_cm_idx = col_idx * CinKK + rest;
    float v = 0.0f;
    if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
        v = x[((n * Cin + ci) * H + hi) * W + wi];
    }
    col[col_cm_idx] = v;
}

// 量子化 kernel: fp32 → i8、 per-tensor scale
extern "C" __global__ void QuantizeF32ToI8(
    long n_elem, float scale, const float* __restrict__ x, signed char* __restrict__ y)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elem) return;
    float v = x[tid] * scale;
    if (v >  127.0f) v =  127.0f;
    if (v < -127.0f) v = -127.0f;
    y[tid] = (signed char)__float2int_rn(v);
}

// dequantize i32 → fp32
extern "C" __global__ void DequantizeI32ToF32(
    long n_elem, float scale, const int* __restrict__ x, float* __restrict__ y)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elem) return;
    y[tid] = (float)x[tid] * scale;
}
"#;

fn main() {
    println!("=== CRCL-v2 INT8 cublasGemmEx smoke (1050 Ti / sm_61 / DP4A) ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("CudaBlas::new");

    let ptx = compile_ptx(IM2COL_SRC).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let im2col = module.load_function("Im2ColNCHWFp32").expect("im2col");
    let quant = module.load_function("QuantizeF32ToI8").expect("quant");
    let dequant = module.load_function("DequantizeI32ToF32").expect("dequant");

    // CRCL-v2 通常 Conv: 192ch × 9x9 × 3x3
    let nb = 32i32;
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

    // input/weight 用意 (small random)
    let x_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host: Vec<f32> = (0..(cout * cin * k * k) as usize).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();
    let x = stream.memcpy_stod(&x_host).unwrap();
    let weight = stream.memcpy_stod(&w_host).unwrap();
    let mut col = stream.alloc_zeros::<f32>(cin_kk * nhw).unwrap();
    let mut col_i8 = stream.alloc_zeros::<i8>(cin_kk * nhw).unwrap();
    let mut weight_i8 = stream.alloc_zeros::<i8>((cout * cin * k * k) as usize).unwrap();
    let mut gemm_out_i32 = stream.alloc_zeros::<i32>((cout as usize) * nhw).unwrap();
    let mut gemm_out_f32 = stream.alloc_zeros::<f32>((cout as usize) * nhw).unwrap();

    // === Phase 1: im2col fp32 ===
    {
        let total = (nb as u64) * (cin as u64) * (k as u64) * (k as u64) * (hout as u64) * (wout as u64);
        let tpb = 256u32;
        let blocks = ((total + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&im2col)
                .arg(&nb)
                .arg(&cin)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&x)
                .arg(&mut col)
                .launch(cfg)
                .unwrap();
        }
    }

    // === Phase 2: 量子化 (per-tensor scale = max(abs) ベース) ===
    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_x = 127.0 / x_max;
    let scale_w = 127.0 / w_max;
    println!("quantize scales: x_max={x_max:.4} → scale_x={scale_x:.2}, w_max={w_max:.4} → scale_w={scale_w:.2}");

    // col_i8 = quant(col, scale_x)
    {
        let n_elem = (cin_kk * nhw) as i64;
        let tpb = 256u32;
        let blocks = ((n_elem as u64 + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream.launch_builder(&quant).arg(&n_elem).arg(&scale_x).arg(&col).arg(&mut col_i8).launch(cfg).unwrap();
        }
    }
    // weight_i8 = quant(weight, scale_w)
    {
        let n_elem = (cout * cin * k * k) as i64;
        let tpb = 256u32;
        let blocks = ((n_elem as u64 + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&quant)
                .arg(&n_elem)
                .arg(&scale_w)
                .arg(&weight)
                .arg(&mut weight_i8)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    // === Phase 3: cublasGemmEx INT8 ===
    // C[Cout, NHW] = W^T[Cout, Cin*K*K] × col[Cin*K*K, NHW]
    // CUDA_R_8I for A, B、 CUDA_R_32I for C、 CUBLAS_COMPUTE_32I
    let alpha: i32 = 1;
    let beta: i32 = 0;
    let m = cout as i32;
    let n_dim = nhw as i32;
    let kk = cin_kk as i32;
    let handle = *blas.handle();
    let (a_ptr, _ra) = weight_i8.device_ptr(&stream);
    let (b_ptr, _rb) = col_i8.device_ptr(&stream);
    let (c_ptr, _rc) = gemm_out_i32.device_ptr_mut(&stream);

    let int8_result = unsafe {
        result::gemm_ex(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n_dim,
            kk,
            &alpha as *const i32 as *const _,
            a_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_8I,
            kk,
            b_ptr as *const _,
            sys::cudaDataType_t::CUDA_R_8I,
            kk,
            &beta as *const i32 as *const _,
            c_ptr as *mut _,
            sys::cudaDataType_t::CUDA_R_32I,
            m,
            cublasComputeType_t::CUBLAS_COMPUTE_32I,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )
    };
    drop(_ra);
    drop(_rb);
    drop(_rc);
    match int8_result {
        Ok(_) => println!("cublasGemmEx INT8 → SUCCESS"),
        Err(e) => {
            println!("cublasGemmEx INT8 → FAILED: {e:?}");
            println!("1050 Ti では INT8 cublas が動かない可能性、 続行不可");
            std::process::exit(2);
        }
    }
    stream.synchronize().unwrap();

    // === Phase 4: dequantize ===
    // fp32_out = i32_out / (scale_x * scale_w)
    let dq_scale: f32 = 1.0 / (scale_x * scale_w);
    {
        let n_elem = ((cout as usize) * nhw) as i64;
        let tpb = 256u32;
        let blocks = ((n_elem as u64 + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&dequant)
                .arg(&n_elem)
                .arg(&dq_scale)
                .arg(&gemm_out_i32)
                .arg(&mut gemm_out_f32)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    // === Phase 5: 比較用 FP32 reference (cuBLAS SGEMM) ===
    let mut gemm_ref = stream.alloc_zeros::<f32>((cout as usize) * nhw).unwrap();
    let cfg_g = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m,
        n: n_dim,
        k: kk,
        alpha: 1.0,
        lda: kk,
        ldb: kk,
        beta: 0.0,
        ldc: m,
    };
    unsafe {
        blas.gemm(cfg_g, &weight, &col, &mut gemm_ref).unwrap();
    }
    stream.synchronize().unwrap();

    let int8_host = stream.memcpy_dtov(&gemm_out_f32).unwrap();
    let fp32_host = stream.memcpy_dtov(&gemm_ref).unwrap();

    let mut max_abs_err = 0.0f32;
    let mut fp32_max = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_fp32 = 0.0f64;
    for i in 0..int8_host.len() {
        let err = (int8_host[i] - fp32_host[i]).abs();
        max_abs_err = max_abs_err.max(err);
        fp32_max = fp32_max.max(fp32_host[i].abs());
        sum_sq_err += (err as f64) * (err as f64);
        sum_sq_fp32 += (fp32_host[i] as f64) * (fp32_host[i] as f64);
    }
    let rmse = (sum_sq_err / int8_host.len() as f64).sqrt() as f32;
    let rms_fp32 = (sum_sq_fp32 / int8_host.len() as f64).sqrt() as f32;
    let rel_rmse = rmse / rms_fp32;
    println!(
        "INT8 vs FP32: max_abs_err={max_abs_err:.5} (FP32 max={fp32_max:.3}), \
         rmse={rmse:.5} / rms_fp32={rms_fp32:.5} → rel_rmse={:.2}%",
        rel_rmse * 100.0
    );
    // RMSE 基準: 平均的な誤差が FP32 出力の 5% 以内なら OK (per-tensor 量子化の典型値)
    assert!(rel_rmse < 0.05, "INT8 dequantize RMSE too high: {:.2}%", rel_rmse * 100.0);
    assert!(max_abs_err < fp32_max * 0.25, "INT8 max abs err > 25% of FP32 max");

    // === Phase 6: bench (INT8 GEMM only、 batch=32) ===
    // warmup
    for _ in 0..5 {
        let (a, _ra) = weight_i8.device_ptr(&stream);
        let (b, _rb) = col_i8.device_ptr(&stream);
        let (c, _rc) = gemm_out_i32.device_ptr_mut(&stream);
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
            .unwrap();
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
    }
    stream.synchronize().unwrap();
    let n_iter = 200usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        let (a, _ra) = weight_i8.device_ptr(&stream);
        let (b, _rb) = col_i8.device_ptr(&stream);
        let (c, _rc) = gemm_out_i32.device_ptr_mut(&stream);
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
            .unwrap();
        }
        drop(_ra);
        drop(_rb);
        drop(_rc);
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let int8_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    println!("INT8 GEMM only: {:.3} ms/call (batch={nb}, m={m}, n={n_dim}, k={kk})", int8_ms);

    // FP32 reference 同条件
    let start = Instant::now();
    for _ in 0..n_iter {
        unsafe {
            blas.gemm(cfg_g, &weight, &col, &mut gemm_ref).unwrap();
        }
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let fp32_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    println!("FP32 GEMM only: {:.3} ms/call (batch={nb})", fp32_ms);

    let speedup = fp32_ms / int8_ms;
    println!("INT8 speedup over FP32: {:.2}x", speedup);

    // CRCL-v2 full 32 conv 予測 (INT8、 GEMM のみ、 im2col+permute+quant overhead 別)
    let full_int8_ms = int8_ms * 32.0;
    let full_leaf_per_sec = nb as f64 / (full_int8_ms / 1000.0);
    println!(
        "CRCL-v2 full (32 conv) INT8 GEMM-only estimate: {:.1} ms/batch → {:.1} leaf/sec",
        full_int8_ms, full_leaf_per_sec
    );
    println!("(GEMM 以外の im2col + quant + permute overhead 加味で実効 70% 程度想定)");
}
