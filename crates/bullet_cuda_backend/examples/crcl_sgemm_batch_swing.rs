//! CRCL-v2 FP32 cuBLAS SGEMM の batch swing bench
//!
//! 192ch × 9x9 × 3x3 Conv の im2col + SGEMM パイプラインで batch サイズを
//! 16 / 32 / 64 / 128 / 256 と変えて per-sample 時間を測定。
//!
//! 目的:
//!   - 1050 Ti FP32 SGEMM の物理上限を実証
//!   - CRCL-v2 設計 1450 leaf/sec が FP32 では batch=? でも到達不可能であることの確認
//!   - INT8 path 必須の根拠データを取る
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_sgemm_batch_swing --release

use std::time::Instant;

use bullet_cuda_backend::cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t},
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

const IM2COL_SRC: &str = r#"
// 1D grid (grid_dim.z <= 65535 制約回避)、 thread per (n, ci, kh, kw, ho, wo)
extern "C" __global__ void Im2ColNCHW(
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
"#;

#[allow(clippy::too_many_arguments)]
fn bench_one_batch(
    blas: &CudaBlas,
    stream: &std::sync::Arc<bullet_cuda_backend::cudarc::driver::CudaStream>,
    im2col: &bullet_cuda_backend::cudarc::driver::CudaFunction,
    nb: i32,
    cin: i32,
    cout: i32,
    h: i32,
    w: i32,
    k: i32,
    pad: i32,
    stride: i32,
) {
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;
    let cin_kk = (cin * k * k) as usize;
    let nhw = (nb * hout * wout) as usize;

    let x_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host = vec![0.01f32; (cout * cin * k * k) as usize];
    let x = stream.memcpy_stod(&x_host).unwrap();
    let weight = stream.memcpy_stod(&w_host).unwrap();
    let mut col = stream.alloc_zeros::<f32>(cin_kk * nhw).unwrap();
    let mut gemm_out = stream.alloc_zeros::<f32>((cout as usize) * nhw).unwrap();

    // warmup
    for _ in 0..5 {
        let total = (nb as u64) * (cin as u64) * (k as u64) * (k as u64) * (hout as u64) * (wout as u64);
        let threads_per_block = 256u32;
        let blocks = ((total + threads_per_block as u64 - 1) / threads_per_block as u64) as u32;
        let cfg_i =
            LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads_per_block, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(im2col)
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
                .launch(cfg_i)
                .unwrap();
        }
        let cfg_g = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: cout,
            n: nhw as i32,
            k: cin_kk as i32,
            alpha: 1.0,
            lda: cin_kk as i32,
            ldb: cin_kk as i32,
            beta: 0.0,
            ldc: cout,
        };
        unsafe {
            blas.gemm(cfg_g, &weight, &col, &mut gemm_out).unwrap();
        }
    }
    stream.synchronize().unwrap();

    let n_iter = 100usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        let total = (nb as u64) * (cin as u64) * (k as u64) * (k as u64) * (hout as u64) * (wout as u64);
        let threads_per_block = 256u32;
        let blocks = ((total + threads_per_block as u64 - 1) / threads_per_block as u64) as u32;
        let cfg_i =
            LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads_per_block, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(im2col)
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
                .launch(cfg_i)
                .unwrap();
        }
        let cfg_g = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: cout,
            n: nhw as i32,
            k: cin_kk as i32,
            alpha: 1.0,
            lda: cin_kk as i32,
            ldb: cin_kk as i32,
            beta: 0.0,
            ldc: cout,
        };
        unsafe {
            blas.gemm(cfg_g, &weight, &col, &mut gemm_out).unwrap();
        }
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let per_call_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    let per_sample_us = per_call_ms * 1000.0 / nb as f64;
    let samples_per_sec = nb as f64 / (per_call_ms / 1000.0);

    // 32 conv の CRCL-v2 full への単純比例予測 (batch=nb 維持)
    let full_ms = per_call_ms * 32.0;
    let full_leaf_per_sec = nb as f64 / (full_ms / 1000.0);

    // 1 Conv の理論最小時間 (1050 Ti SGEMM 1.5 TFLOPS 100% efficient)
    let flops = (cout as f64) * (nhw as f64) * (cin_kk as f64) * 2.0; // 2*FMA
    let theoretical_ms = flops / 1.5e12 * 1000.0;
    let efficiency = theoretical_ms / per_call_ms * 100.0;

    println!(
        "batch={:>4}  1 Conv: {:>7.3} ms/call ({:>6.1} samples/sec, {:>4.1} us/sample), \
         full 32-conv estimate: {:>5.1} ms/batch → {:>6.1} leaf/sec, \
         theoretical_min={:.2} ms (cuBLAS eff {:>4.1}%)",
        nb, per_call_ms, samples_per_sec, per_sample_us, full_ms, full_leaf_per_sec, theoretical_ms, efficiency,
    );
}

fn main() {
    println!("=== CRCL-v2 FP32 cuBLAS SGEMM batch swing bench (192ch × 9x9 × 3x3) ===");
    println!("1050 Ti FP32 peak 2.1 TFLOPS、 cuBLAS 効率 70% 想定で 1.5 TFLOPS を基準\n");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("CudaBlas::new");

    let ptx = compile_ptx(IM2COL_SRC).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let im2col = module.load_function("Im2ColNCHW").expect("im2col");

    let cin = 192i32;
    let cout = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;

    for nb in [8, 16, 32, 64, 128, 256] {
        bench_one_batch(&blas, &stream, &im2col, nb, cin, cout, h, w, k, pad, stride);
    }

    println!("\n=== CRCL-v2 設計 target ===");
    println!("FP32 1450 leaf/sec @ batch=32 → 22 ms/batch、 1 conv 0.69 ms/call が必要");
    println!("INT8 (DP4A、 4x speedup 期待) → FP32 ~890 leaf/sec ライン × 4 = 3555 leaf/sec で設計超え");
}
