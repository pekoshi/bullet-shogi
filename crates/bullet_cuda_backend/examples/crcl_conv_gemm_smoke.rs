//! CRCL-v2 im2col + cuBLAS SGEMM 経路の forward smoke
//!
//! naive Conv2dForwardNCHW kernel が遅い (1050 Ti で CRCL-v2 full 予測 46 leaf/sec) ため、
//! im2col → SGEMM → reshape の 3 段路線で高速化を目指す。
//!
//! 確認内容:
//!   1. im2col_nchw_to_col_cm: input [N,Cin,H,W] (NCHW row-major) を col [Cin*K*K, N*Hout*Wout]
//!      (column-major) に展開
//!   2. cuBLAS SGEMM: weight^T [Cout, Cin*K*K] × col [Cin*K*K, N*Hout*Wout]
//!      = gemm_out [Cout, N*Hout*Wout] (column-major)
//!   3. permute_cm_to_nchw: gemm_out (column-major、 C×NHW) を output [N,Cout,Hout,Wout]
//!      (NCHW row-major) に並べ替え
//!   4. 結果が naive forward と一致 (center=1728 等) することを確認
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_conv_gemm_smoke --release

use std::time::Instant;

use bullet_cuda_backend::cudarc::{
    cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t},
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

const HELPER_KERNELS: &str = r#"
// input: x [N,Cin,H,W] row-major
// output: col [Cin*K*K, N*Hout*Wout] column-major
//         col_cm_layout: col[col_idx * (Cin*K*K) + row_idx]
//         row_idx = ci*K*K + kh*K + kw
//         col_idx = n*Hout*Wout + ho*Wout + wo
extern "C" __global__ void Im2ColNCHW(
    int N, int Cin, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride,
    const float* __restrict__ x,
    float* __restrict__ col)
{
    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;  // 0 .. N*Cin*K*K-1
    int nckk = N * Cin * K * K;
    if (wo >= Wout || ho >= Hout || nc >= nckk) return;

    int CinKK = Cin * K * K;
    int n = nc / CinKK;
    int rest = nc % CinKK;
    int ci = rest / (K * K);
    int kh = (rest / K) % K;
    int kw = rest % K;

    int hi = ho * Stride + kh - Pad;
    int wi = wo * Stride + kw - Pad;

    int row_idx = rest;
    int col_idx = n * Hout * Wout + ho * Wout + wo;
    int col_cm_idx = col_idx * CinKK + row_idx;

    float v = 0.0f;
    if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
        v = x[((n * Cin + ci) * H + hi) * W + wi];
    }
    col[col_cm_idx] = v;
}

// gemm_out: [Cout, N*Hout*Wout] column-major
//   gemm_out[col_idx * Cout + co]
//   col_idx = n*Hout*Wout + ho*Wout + wo
// output: [N, Cout, Hout, Wout] row-major
//   output[((n*Cout + co)*Hout + ho)*Wout + wo]
extern "C" __global__ void PermuteCmToNCHW(
    int N, int Cout, int Hout, int Wout,
    const float* __restrict__ gemm_out,
    float* __restrict__ output)
{
    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;  // 0..N*Cout-1
    if (wo >= Wout || ho >= Hout || nc >= N * Cout) return;

    int n  = nc / Cout;
    int co = nc % Cout;

    int col_idx = n * Hout * Wout + ho * Wout + wo;
    int gemm_idx = col_idx * Cout + co;
    int out_idx = ((n * Cout + co) * Hout + ho) * Wout + wo;
    output[out_idx] = gemm_out[gemm_idx];
}
"#;

fn main() {
    println!("=== CRCL-v2 Conv im2col + cuBLAS SGEMM smoke ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("CudaBlas::new");
    println!("CUDA context + cuBLAS created");

    let ptx = compile_ptx(HELPER_KERNELS).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let im2col = module.load_function("Im2ColNCHW").expect("im2col");
    let permute = module.load_function("PermuteCmToNCHW").expect("permute");

    // CRCL-v2 通常 conv: 192ch × 9x9 × 3x3 padding 1 stride 1
    let n = 1i32;
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
    let nhw = (n * hout * wout) as usize;

    // input/weight all-ones
    let x_host = vec![1.0f32; (n * cin * h * w) as usize];
    let w_host = vec![1.0f32; (cout * cin * k * k) as usize];
    let x = stream.memcpy_stod(&x_host).unwrap();
    let weight = stream.memcpy_stod(&w_host).unwrap();
    let mut col = stream.alloc_zeros::<f32>(cin_kk * nhw).unwrap();
    let mut gemm_out = stream.alloc_zeros::<f32>((cout as usize) * nhw).unwrap();
    let mut output = stream.alloc_zeros::<f32>((n * cout * hout * wout) as usize).unwrap();

    // === phase 1: im2col ===
    let block = (16u32, 16u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (n * cin * k * k) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&im2col)
            .arg(&n)
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
    stream.synchronize().unwrap();
    println!("im2col done (col shape Cin*K*K={cin_kk} × N*Hout*Wout={nhw})");

    // === phase 2: cuBLAS SGEMM ===
    // C[Cout, NHW] = W^T[Cout, Cin*K*K] × col[Cin*K*K, NHW]
    // cuBLAS column-major:
    //   A = weight (memory layout row-major Cout × Cin*K*K = column-major (Cin*K*K, Cout)):
    //       渡し時に shape を (Cin*K*K, Cout) と見て transa=T で論理 (Cout, Cin*K*K)
    //   B = col (column-major (Cin*K*K, NHW)): そのまま、 transb=N
    //   C = gemm_out (column-major (Cout, NHW))
    let m = cout as i32; // output rows (Cout)
    let n_dim = nhw as i32; // output cols (NHW)
    let kk = cin_kk as i32; // inner (Cin*K*K)
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m,
        n: n_dim,
        k: kk,
        alpha: 1.0,
        lda: kk, // weight column-major (CinKK rows)、 lda = rows
        ldb: kk, // col column-major (CinKK rows)、 ldb = CinKK
        beta: 0.0,
        ldc: m, // gemm_out column-major (Cout rows)
    };
    unsafe {
        blas.gemm(cfg, &weight, &col, &mut gemm_out).expect("SGEMM");
    }
    stream.synchronize().unwrap();
    println!("SGEMM done");

    // === phase 3: permute column-major C×NHW → NCHW row-major ===
    let cfg = LaunchConfig {
        grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (n * cout) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&permute)
            .arg(&n)
            .arg(&cout)
            .arg(&hout)
            .arg(&wout)
            .arg(&gemm_out)
            .arg(&mut output)
            .launch(cfg)
            .unwrap();
    }
    stream.synchronize().unwrap();
    println!("permute done");

    let host = stream.memcpy_dtov(&output).unwrap();
    let center = host[((cout / 2 * hout + hout / 2) * wout + wout / 2) as usize];
    let corner = host[0];
    let edge = host[(wout / 2) as usize];
    println!(
        "im2col+gemm forward: center={center} (expected 1728) corner={corner} (expected 768) edge={edge} (expected 1152)"
    );
    let expected_center = (cin * k * k) as f32;
    let expected_corner = (cin * 4) as f32;
    let expected_edge = (cin * 6) as f32;
    assert!((center - expected_center).abs() < 1e-3);
    assert!((corner - expected_corner).abs() < 1e-3);
    assert!((edge - expected_edge).abs() < 1e-3);
    println!("im2col + SGEMM forward smoke PASSED");

    // ===== bench: batch=32 で im2col + gemm + permute の時間計測 =====
    println!("--- bench (batch=32) ---");
    let nb = 32i32;
    let cin_kk_u = cin_kk;
    let nhw_b = (nb * hout * wout) as usize;
    let xb_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let xb = stream.memcpy_stod(&xb_host).unwrap();
    let mut colb = stream.alloc_zeros::<f32>(cin_kk_u * nhw_b).unwrap();
    let mut gemm_out_b = stream.alloc_zeros::<f32>((cout as usize) * nhw_b).unwrap();
    let mut output_b = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).unwrap();

    // warmup
    for _ in 0..3 {
        let cfg_i = LaunchConfig {
            grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (nb * cin * k * k) as u32),
            block_dim: block,
            shared_mem_bytes: 0,
        };
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
                .arg(&xb)
                .arg(&mut colb)
                .launch(cfg_i)
                .unwrap();
        }
        let cfg_g = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: cout,
            n: nhw_b as i32,
            k: cin_kk_u as i32,
            alpha: 1.0,
            lda: cin_kk_u as i32,
            ldb: cin_kk_u as i32,
            beta: 0.0,
            ldc: cout,
        };
        unsafe {
            blas.gemm(cfg_g, &weight, &colb, &mut gemm_out_b).unwrap();
        }
        let cfg_p = LaunchConfig {
            grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (nb * cout) as u32),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&gemm_out_b)
                .arg(&mut output_b)
                .launch(cfg_p)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    let n_iter = 100usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        let cfg_i = LaunchConfig {
            grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (nb * cin * k * k) as u32),
            block_dim: block,
            shared_mem_bytes: 0,
        };
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
                .arg(&xb)
                .arg(&mut colb)
                .launch(cfg_i)
                .unwrap();
        }
        let cfg_g = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: cout,
            n: nhw_b as i32,
            k: cin_kk_u as i32,
            alpha: 1.0,
            lda: cin_kk_u as i32,
            ldb: cin_kk_u as i32,
            beta: 0.0,
            ldc: cout,
        };
        unsafe {
            blas.gemm(cfg_g, &weight, &colb, &mut gemm_out_b).unwrap();
        }
        let cfg_p = LaunchConfig {
            grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (nb * cout) as u32),
            block_dim: block,
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&gemm_out_b)
                .arg(&mut output_b)
                .launch(cfg_p)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let per_call_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    let nps = (n_iter as i32 * nb) as f64 / elapsed.as_secs_f64();
    println!("1 Conv (Cin=Cout={cout}, 9x9, K=3, batch={nb}): {:.3} ms/call → {:.1} samples/sec", per_call_ms, nps);
}
