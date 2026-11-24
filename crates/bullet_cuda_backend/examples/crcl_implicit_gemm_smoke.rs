//! CRCL-v2 Implicit GEMM INT8 conv kernel 自前実装 smoke
//!
//! 1 kernel で im2col + GEMM + dequant + post-op をすべて統合。
//! 中間 col buffer 不要、 memory traffic 大幅削減で 1450 leaf/sec 達成を狙う。
//!
//! Tile 設計:
//!   - BLOCK_M = 32 (Cout 方向)
//!   - BLOCK_N = 32 (NHW 方向 = batch * Hout * Wout)
//!   - BLOCK_K = 16 (Cin * K * K 方向、 inner reduction)
//!   - thread block = 32 × 32 = 1024 thread (sm_61 max)
//!   - shared memory: 32×16 + 16×32 = 1024 i8 = 1 KB (per block)
//!
//! 各 thread (tx, ty) は出力 1 要素 output[m=Cout_block+tx, n=NHW_block+ty] を担当、
//! K loop で内積を累積。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_implicit_gemm_smoke --release

use std::time::Instant;

use bullet_cuda_backend::cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

const IMPLICIT_GEMM_SRC: &str = r#"
// post_op: 0 = none、 1 = silu(v)、 2 = silu(v + residual)
extern "C" __global__ void Conv2dInt8ImplicitGemm(
    int N, int Cin, int Cout, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride, float scale_x, float dq_scale,
    int post_op,
    const signed char* __restrict__ weight_i8,  // [Cout, CinKK] row-major
    const float* __restrict__ input,            // [N, Cin, H, W] NCHW row-major
    const float* __restrict__ residual,         // [N, Cout, Hout, Wout] (post_op==2 のみ使用)
    float* __restrict__ output)                  // [N, Cout, Hout, Wout] NCHW row-major
{
    const int BLOCK_M = 32;
    const int BLOCK_N = 32;
    const int BLOCK_K = 16;

    int tx = threadIdx.x;  // 0..BLOCK_M-1 = Cout local index
    int ty = threadIdx.y;  // 0..BLOCK_N-1 = NHW local index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int m = bx * BLOCK_M + tx;
    int n_idx = by * BLOCK_N + ty;

    int CinKK = Cin * K * K;
    int whw = Hout * Wout;
    int NHW = N * whw;

    __shared__ signed char A_sh[BLOCK_M][BLOCK_K];
    __shared__ signed char B_sh[BLOCK_K][BLOCK_N];

    int acc = 0;

    int k_blocks = (CinKK + BLOCK_K - 1) / BLOCK_K;
    for (int kb = 0; kb < k_blocks; ++kb) {
        int k_base = kb * BLOCK_K;

        // A_sh[tx][ty]: weight_i8[m, k_base+ty] (ty < BLOCK_K のみ)
        if (ty < BLOCK_K) {
            int kk = k_base + ty;
            signed char v = 0;
            if (m < Cout && kk < CinKK) {
                v = weight_i8[m * CinKK + kk];
            }
            A_sh[tx][ty] = v;
        }

        // B_sh[tx][ty]: im2col virtual col[k_base+tx, n_idx] (tx < BLOCK_K のみ)
        if (tx < BLOCK_K) {
            int kk = k_base + tx;
            signed char v = 0;
            if (n_idx < NHW && kk < CinKK) {
                int wo = n_idx % Wout;
                int ho = (n_idx / Wout) % Hout;
                int n = n_idx / whw;
                int ci = kk / (K * K);
                int kh = (kk / K) % K;
                int kw = kk % K;
                int hi = ho * Stride + kh - Pad;
                int wi = wo * Stride + kw - Pad;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    float xv = input[((n * Cin + ci) * H + hi) * W + wi];
                    float q = xv * scale_x;
                    if (q >  127.0f) q =  127.0f;
                    if (q < -127.0f) q = -127.0f;
                    v = (signed char)__float2int_rn(q);
                }
            }
            B_sh[tx][ty] = v;
        }
        __syncthreads();

        if (m < Cout && n_idx < NHW) {
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; ++kk) {
                acc += (int)A_sh[tx][kk] * (int)B_sh[kk][ty];
            }
        }
        __syncthreads();
    }

    // Epilogue: dequant + post-op + NCHW permute
    if (m < Cout && n_idx < NHW) {
        int wo = n_idx % Wout;
        int ho = (n_idx / Wout) % Hout;
        int n = n_idx / whw;
        long out_idx = ((long)(n * Cout + m) * Hout + ho) * Wout + wo;

        float v = (float)acc * dq_scale;
        if (post_op == 2) {
            v += residual[out_idx];
        }
        if (post_op >= 1) {
            v = v / (1.0f + expf(-v));
        }
        output[out_idx] = v;
    }
}

// f32 weight → i8 weight (PTQ 用)
extern "C" __global__ void QuantizeWeightI8(
    long n_elem, float scale, const float* __restrict__ x, signed char* __restrict__ y)
{
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_elem) return;
    float v = x[tid] * scale;
    if (v >  127.0f) v =  127.0f;
    if (v < -127.0f) v = -127.0f;
    y[tid] = (signed char)__float2int_rn(v);
}
"#;

fn main() {
    println!("=== CRCL-v2 Implicit GEMM INT8 smoke ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();

    let ptx = compile_ptx(IMPLICIT_GEMM_SRC).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let conv_kernel = module.load_function("Conv2dInt8ImplicitGemm").unwrap();
    let quant_w = module.load_function("QuantizeWeightI8").unwrap();
    println!("kernels JIT compiled");

    // CRCL-v2 通常 Conv: 192ch × 9x9 × 3x3
    let nb: i32 = std::env::var("CRCL_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(32);
    let cin = 192i32;
    let cout = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;
    let nhw = (nb * hout * wout) as usize;

    // input / weight
    let x_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host: Vec<f32> = (0..(cout * cin * k * k) as usize).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();

    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_x = 127.0 / x_max;
    let scale_w = 127.0 / w_max;
    let dq_scale = 1.0 / (scale_x * scale_w);
    println!("scale_x={scale_x:.2}, scale_w={scale_w:.2}");

    let input = stream.memcpy_stod(&x_host).unwrap();
    let weight = stream.memcpy_stod(&w_host).unwrap();
    let mut weight_i8 = stream.alloc_zeros::<i8>((cout * cin * k * k) as usize).unwrap();
    let mut output = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).unwrap();
    let residual_dummy = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).unwrap();

    // weight 量子化
    {
        let n_elem = (cout * cin * k * k) as i64;
        let tpb = 256u32;
        let blocks = ((n_elem as u64 + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&quant_w)
                .arg(&n_elem)
                .arg(&scale_w)
                .arg(&weight)
                .arg(&mut weight_i8)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    // === implicit GEMM forward (post_op=0) ===
    let block_m = 32u32;
    let block_n = 32u32;
    let grid_x = (cout as u32 + block_m - 1) / block_m;
    let grid_y = (nhw as u32 + block_n - 1) / block_n;
    let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (block_m, block_n, 1), shared_mem_bytes: 0 };
    let post_op: i32 = 0;
    println!("launch grid={:?} block={:?}", cfg.grid_dim, cfg.block_dim);
    unsafe {
        stream
            .launch_builder(&conv_kernel)
            .arg(&nb)
            .arg(&cin)
            .arg(&cout)
            .arg(&h)
            .arg(&w)
            .arg(&hout)
            .arg(&wout)
            .arg(&k)
            .arg(&pad)
            .arg(&stride)
            .arg(&scale_x)
            .arg(&dq_scale)
            .arg(&post_op)
            .arg(&weight_i8)
            .arg(&input)
            .arg(&residual_dummy)
            .arg(&mut output)
            .launch(cfg)
            .unwrap();
    }
    stream.synchronize().unwrap();
    println!("implicit GEMM kernel done");

    let host = stream.memcpy_dtov(&output).unwrap();
    let center = host[((cout / 2 * hout + hout / 2) * wout + wout / 2) as usize];
    println!("output[center]={:.4}, [0..4]={:?}", center, &host[0..4]);
    assert!(center.is_finite() && center.abs() < 1e3, "abnormal");

    // bench
    let warmup = 50;
    for _ in 0..warmup {
        unsafe {
            stream
                .launch_builder(&conv_kernel)
                .arg(&nb)
                .arg(&cin)
                .arg(&cout)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&scale_x)
                .arg(&dq_scale)
                .arg(&post_op)
                .arg(&weight_i8)
                .arg(&input)
                .arg(&residual_dummy)
                .arg(&mut output)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    let n_iter = 200usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        unsafe {
            stream
                .launch_builder(&conv_kernel)
                .arg(&nb)
                .arg(&cin)
                .arg(&cout)
                .arg(&h)
                .arg(&w)
                .arg(&hout)
                .arg(&wout)
                .arg(&k)
                .arg(&pad)
                .arg(&stride)
                .arg(&scale_x)
                .arg(&dq_scale)
                .arg(&post_op)
                .arg(&weight_i8)
                .arg(&input)
                .arg(&residual_dummy)
                .arg(&mut output)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let per_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    println!("implicit GEMM 1 Conv: {:.3} ms/call (batch={nb})", per_ms);

    let full_ms = per_ms * 32.0;
    let lps = nb as f64 / (full_ms / 1000.0);
    println!("CRCL-v2 full (32 conv) estimate: {:.1} ms/batch → {:.1} leaf/sec", full_ms, lps);
    let design = 1450.0;
    println!(
        "vs 設計目標 1450 leaf/sec: {:.0}% ({})",
        lps / design * 100.0,
        if lps >= design { "✓ ACHIEVED" } else { "未達" }
    );
}
