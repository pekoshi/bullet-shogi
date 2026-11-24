//! CRCL-v2 Implicit GEMM Phase 3B: double-buffered shared memory
//!
//! Phase 2 base (BLOCK_M=BLOCK_N=64, TM=TN=4) + double buffer (A_sh / B_sh を 2 set):
//!   - inner loop で next tile load と current tile compute を overlap
//!   - prologue で最初の tile を pre-load、 epilogue で最後の compute
//!   - 期待: 30-50% 改善 (Phase 2 2.84 → 1.5-2.0 ms/conv)
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_implicit_gemm_phase3b_smoke --release

use std::time::Instant;

use bullet_cuda_backend::cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

const SRC: &str = r#"
extern "C" __global__ void PackWeightInt8ToInt32(
    int Cout, int CinKK, int CinKKPack,
    const signed char* __restrict__ w_i8,
    int* __restrict__ w_packed)
{
    int m = blockIdx.x;
    int p = blockIdx.y * blockDim.x + threadIdx.x;
    if (m >= Cout || p >= CinKKPack) return;
    int k_base = p * 4;
    int v = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        int kk = k_base + j;
        signed char wv = 0;
        if (kk < CinKK) wv = w_i8[m * CinKK + kk];
        v |= ((int)((unsigned char)wv)) << (8 * j);
    }
    w_packed[m * CinKKPack + p] = v;
}

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

// double-buffered shared memory、 prologue + main loop + epilogue
extern "C" __global__ void Conv2dInt8ImplicitGemmPhase3B(
    int N, int Cin, int Cout, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride, float scale_x, float dq_scale,
    int post_op,
    const int* __restrict__ w_packed,
    int CinKKPack,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output)
{
    const int BLOCK_M = 64;
    const int BLOCK_N = 64;
    const int BLOCK_K_PACK = 4;
    const int TM = 4;
    const int TN = 4;
    const int THREADS_N = BLOCK_N / TN;  // 16

    int mt = threadIdx.x;
    int nt = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = mt * THREADS_N + nt;  // 0..255

    int whw = Hout * Wout;
    int NHW = N * whw;
    int CinKK = Cin * K * K;

    // double buffer (2 set)
    __shared__ int A_sh[2][BLOCK_M][BLOCK_K_PACK + 1];   // 2 × 64 × 5 = 640 int32 = 2.5 KB
    __shared__ int B_sh[2][BLOCK_K_PACK][BLOCK_N];       // 2 × 4 × 64 = 512 int32 = 2 KB

    int acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0;
    }

    int k_blocks = (CinKKPack + BLOCK_K_PACK - 1) / BLOCK_K_PACK;

    // ----- helper macro -----
    // load tile (A_sh[buf] と B_sh[buf]) from p_base
    auto load_tile = [&](int buf, int p_base) {
        // A_sh load: 64*4 = 256 cell、 256 thread で 1 cell/thread
        {
            int row = tid / BLOCK_K_PACK;  // 0..63
            int col = tid % BLOCK_K_PACK;  // 0..3
            int m_g = bx * BLOCK_M + row;
            int p_g = p_base + col;
            int v = 0;
            if (m_g < Cout && p_g < CinKKPack) v = w_packed[m_g * CinKKPack + p_g];
            A_sh[buf][row][col] = v;
        }
        // B_sh load: 4*64 = 256 cell、 256 thread で 1 cell/thread
        {
            int row = tid / BLOCK_N;
            int col = tid % BLOCK_N;
            int kk_base = (p_base + row) * 4;
            int n_g = by * BLOCK_N + col;
            int v = 0;
            if (n_g < NHW) {
                int wo = n_g % Wout;
                int ho = (n_g / Wout) % Hout;
                int n = n_g / whw;
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int kk = kk_base + j;
                    signed char bv = 0;
                    if (kk < CinKK) {
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
                            bv = (signed char)__float2int_rn(q);
                        }
                    }
                    v |= ((int)((unsigned char)bv)) << (8 * j);
                }
            }
            B_sh[buf][row][col] = v;
        }
    };

    auto compute_tile = [&](int buf) {
        #pragma unroll
        for (int k4 = 0; k4 < BLOCK_K_PACK; ++k4) {
            int a_reg[TM];
            int b_reg[TN];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) a_reg[tm] = A_sh[buf][mt * TM + tm][k4];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) b_reg[tn] = B_sh[buf][k4][nt * TN + tn];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] = __dp4a(a_reg[tm], b_reg[tn], acc[tm][tn]);
                }
            }
        }
    };

    // === prologue: load first tile ===
    load_tile(0, 0);
    __syncthreads();

    // === main loop: load next while compute current ===
    int cur = 0;
    for (int kb = 0; kb < k_blocks - 1; ++kb) {
        int next = 1 - cur;
        load_tile(next, (kb + 1) * BLOCK_K_PACK);
        compute_tile(cur);
        __syncthreads();
        cur = next;
    }

    // === epilogue: compute last tile ===
    compute_tile(cur);

    // === store output ===
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        int m = bx * BLOCK_M + mt * TM + tm;
        if (m >= Cout) continue;
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            int n_idx = by * BLOCK_N + nt * TN + tn;
            if (n_idx >= NHW) continue;
            int wo = n_idx % Wout;
            int ho = (n_idx / Wout) % Hout;
            int n = n_idx / whw;
            long out_idx = ((long)(n * Cout + m) * Hout + ho) * Wout + wo;
            float v = (float)acc[tm][tn] * dq_scale;
            if (post_op == 2) v += residual[out_idx];
            if (post_op >= 1) v = v / (1.0f + expf(-v));
            output[out_idx] = v;
        }
    }
}
"#;

fn main() {
    println!("=== Phase 3B: double-buffered shared memory ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let opts = CompileOptions { arch: Some("compute_61"), ..Default::default() };
    let ptx = compile_ptx_with_opts(SRC, opts).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let pack_w = module.load_function("PackWeightInt8ToInt32").unwrap();
    let conv_kernel = module.load_function("Conv2dInt8ImplicitGemmPhase3B").unwrap();
    let quant_w = module.load_function("QuantizeWeightI8").unwrap();

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
    let cin_kk = cin * k * k;
    let cin_kk_pack = (cin_kk + 3) / 4;
    let nhw = (nb * hout * wout) as usize;

    let x_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host: Vec<f32> = (0..(cout * cin * k * k) as usize).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();
    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_x = 127.0 / x_max;
    let scale_w = 127.0 / w_max;
    let dq_scale = 1.0 / (scale_x * scale_w);

    let input = stream.memcpy_stod(&x_host).unwrap();
    let weight = stream.memcpy_stod(&w_host).unwrap();
    let mut weight_i8 = stream.alloc_zeros::<i8>((cout * cin_kk) as usize).unwrap();
    let mut weight_packed = stream.alloc_zeros::<i32>((cout * cin_kk_pack) as usize).unwrap();
    let mut output = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).unwrap();
    let residual_dummy = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).unwrap();

    {
        let n_elem = (cout * cin_kk) as i64;
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
    {
        let tpb = 64u32;
        let blocks_y = (cin_kk_pack as u32 + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (cout as u32, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&pack_w)
                .arg(&cout)
                .arg(&cin_kk)
                .arg(&cin_kk_pack)
                .arg(&weight_i8)
                .arg(&mut weight_packed)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    let block_m = 64u32;
    let block_n = 64u32;
    let grid_x = (cout as u32 + block_m - 1) / block_m;
    let grid_y = (nhw as u32 + block_n - 1) / block_n;
    let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (16, 16, 1), shared_mem_bytes: 0 };
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
            .arg(&weight_packed)
            .arg(&cin_kk_pack)
            .arg(&input)
            .arg(&residual_dummy)
            .arg(&mut output)
            .launch(cfg)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let host = stream.memcpy_dtov(&output).unwrap();
    let center = host[((cout / 2 * hout + hout / 2) * wout + wout / 2) as usize];
    println!("output[center]={:.4}, [0..4]={:?}", center, &host[0..4]);
    assert!(center.is_finite() && center.abs() < 1e3);

    for _ in 0..50 {
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
                .arg(&weight_packed)
                .arg(&cin_kk_pack)
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
                .arg(&weight_packed)
                .arg(&cin_kk_pack)
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
    let full_ms = per_ms * 32.0;
    let lps = nb as f64 / (full_ms / 1000.0);
    println!("Phase 3B 1 Conv: {:.3} ms/call (batch={nb}) → 32 conv {:.1} ms → {:.1} leaf/sec", per_ms, full_ms, lps);
    let design = 1450.0;
    println!(
        "vs 設計目標 1450 leaf/sec: {:.0}% ({})",
        lps / design * 100.0,
        if lps >= design { "✓ ACHIEVED" } else { "未達" }
    );
}
