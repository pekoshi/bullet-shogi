//! CRCL-v2 INT8 inference pipeline end-to-end smoke + bench
//!
//! crcl_int8_gemm_smoke では GEMM だけ計測。 ここでは fused kernel で:
//!   - Im2ColI8NCHW: input(fp32) + scale → col(i8) in 1 kernel (im2col + quantize)
//!   - cublasGemmEx INT8: weight_i8 × col_i8 → gemm_out_i32
//!   - DequantPermuteI32: gemm_out_i32 + scale_inv → output(fp32, NCHW row-major) in 1 kernel
//! を 1 Conv pipeline として組み、 batch=32 で per-call 時間を測定。
//!
//! weight_i8 は事前 quantize 済 (inference 用 PTQ 後)、 毎 inference では quantize しない想定。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_int8_pipeline_smoke --release

use std::time::Instant;

use bullet_cuda_backend::cudarc::{
    cublas::{
        CudaBlas, result,
        sys::{self, cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t},
    },
    driver::{CudaContext, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};

const FUSED_KERNELS: &str = r#"
// fused im2col + quantize, coalesced col write.
// thread mapping: blockIdx.x = col_idx (0..N*Hout*Wout-1)
//                 (blockIdx.y * blockDim.x + threadIdx.x) = rest (0..CinKK-1)
// thread が x 内で連続 = rest 連続 = col_cm_idx 連続 → coalesced i8 write
extern "C" __global__ void Im2ColI8NCHW(
    int N, int Cin, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride, float scale_x,
    const float* __restrict__ x,
    signed char* __restrict__ col)
{
    int col_idx = blockIdx.x;
    int rest = blockIdx.y * blockDim.x + threadIdx.x;
    int CinKK = Cin * K * K;
    int whw = Hout * Wout;
    if (col_idx >= N * whw || rest >= CinKK) return;

    int wo = col_idx % Wout;
    int ho = (col_idx / Wout) % Hout;
    int n = col_idx / whw;
    int ci = rest / (K * K);
    int kh = (rest / K) % K;
    int kw = rest % K;
    int hi = ho * Stride + kh - Pad;
    int wi = wo * Stride + kw - Pad;

    float v = 0.0f;
    if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
        v = x[((n * Cin + ci) * H + hi) * W + wi];
    }
    float q = v * scale_x;
    if (q >  127.0f) q =  127.0f;
    if (q < -127.0f) q = -127.0f;
    col[(long)col_idx * CinKK + rest] = (signed char)__float2int_rn(q);
}

// fused dequant + permute, shared memory tile transpose で gemm_out read と output write 両方 coalesce
// gemm_out: column-major (Cout × NHW)、 gemm_out[col_idx * Cout + co]
// output: NCHW row-major、 output[((n*Cout+co)*Hout+ho)*Wout+wo]
//
// 32x32 tile: blockIdx.x = co_tile (0..Cout/32-1)、 blockIdx.y = col_tile (0..NHW/32-1)
// 各 block で gemm_out[(col_tile*32+ty), (co_tile*32+tx)] を tile に coalesced read
// 転置して output に coalesced write
#define DEQUANT_TILE 32
extern "C" __global__ void DequantPermuteI32ToNCHW(
    int N, int Cout, int Hout, int Wout, float scale_inv,
    const int* __restrict__ gemm_out,
    float* __restrict__ output)
{
    __shared__ float tile[DEQUANT_TILE][DEQUANT_TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int co_tile = blockIdx.x;
    int col_tile = blockIdx.y;
    int whw = Hout * Wout;
    int nhw_total = N * whw;

    // === phase 1: gemm_out coalesced read into shared memory ===
    // thread (tx, ty) は (col=col_tile*32+ty, row=co_tile*32+tx) を read
    // gemm_out[col * Cout + row]、 tx 連続で row 連続 → 4 byte stride i32 → coalesce
    int row_g = co_tile * DEQUANT_TILE + tx;
    int col_g = col_tile * DEQUANT_TILE + ty;
    if (row_g < Cout && col_g < nhw_total) {
        long gidx = (long)col_g * Cout + row_g;
        tile[ty][tx] = (float)gemm_out[gidx] * scale_inv;
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    // === phase 2: transposed write into output (NCHW) ===
    // thread (tx, ty) は (col=col_tile*32+tx, co=co_tile*32+ty) の output を write
    // output[((n*Cout+co)*Hout+ho)*Wout+wo] where col = n*Hout*Wout+ho*Wout+wo
    // tx 連続で col 連続、 wo 連続 (Wout 単位) → output 連続 → coalesce (ただし wo が tile 内なら)
    int co_out = co_tile * DEQUANT_TILE + ty;
    int col_out = col_tile * DEQUANT_TILE + tx;
    if (co_out < Cout && col_out < nhw_total) {
        int wo = col_out % Wout;
        int ho = (col_out / Wout) % Hout;
        int n = col_out / whw;
        long oidx = ((long)(n * Cout + co_out) * Hout + ho) * Wout + wo;
        output[oidx] = tile[tx][ty];
    }
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
"#;

fn main() {
    println!("=== CRCL-v2 INT8 inference pipeline smoke (192ch × 9x9 × 3x3, batch=32) ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone()).expect("CudaBlas::new");

    let ptx = compile_ptx(FUSED_KERNELS).expect("nvrtc compile");
    let module = ctx.load_module(ptx).expect("load_module");
    let im2col_i8 = module.load_function("Im2ColI8NCHW").unwrap();
    let dequant_permute = module.load_function("DequantPermuteI32ToNCHW").unwrap();
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
    let cin_kk = (cin * k * k) as usize;
    let nhw = (nb * hout * wout) as usize;

    let x_host: Vec<f32> = (0..(nb * cin * h * w) as usize).map(|i| (i as f32 * 0.0317).sin() * 0.5).collect();
    let w_host: Vec<f32> = (0..(cout * cin * k * k) as usize).map(|i| (i as f32 * 0.0123).cos() * 0.05).collect();
    let x = stream.memcpy_stod(&x_host).unwrap();
    let weight = stream.memcpy_stod(&w_host).unwrap();
    let mut col_i8 = stream.alloc_zeros::<i8>(cin_kk * nhw).unwrap();
    let mut weight_i8 = stream.alloc_zeros::<i8>((cout * cin * k * k) as usize).unwrap();
    let mut gemm_out_i32 = stream.alloc_zeros::<i32>((cout as usize) * nhw).unwrap();
    let mut output = stream.alloc_zeros::<f32>((nb * cout * hout * wout) as usize).unwrap();

    // 量子化 scale (per-tensor)
    let x_max = x_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let w_max = w_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_x = 127.0 / x_max;
    let scale_w = 127.0 / w_max;
    let dq_scale: f32 = 1.0 / (scale_x * scale_w);
    println!("scale_x={scale_x:.2}, scale_w={scale_w:.2}, dq_scale={dq_scale:.6e}");

    // weight_i8 を 1 度だけ量子化 (inference 用 PTQ)
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

    // === inference pipeline 1 回 (動作確認) ===
    // step 1: im2col + quantize (coalesced 2D grid)
    {
        let n_col = (nb * hout * wout) as u32;
        let cin_kk_u32 = (cin * k * k) as u32;
        let tpb = 256u32;
        let blocks_y = (cin_kk_u32 + tpb - 1) / tpb;
        let cfg = LaunchConfig { grid_dim: (n_col, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&im2col_i8)
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
                .arg(&x)
                .arg(&mut col_i8)
                .launch(cfg)
                .unwrap();
        }
    }

    // step 2: cublasGemmEx INT8
    let alpha: i32 = 1;
    let beta: i32 = 0;
    let m = cout as i32;
    let n_dim = nhw as i32;
    let kk = cin_kk as i32;
    let handle = *blas.handle();
    {
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

    // step 3: dequant + permute (shared memory tile transpose)
    {
        let tile = 32u32;
        let n_col_o = (nb * hout * wout) as u32;
        let grid_x = (cout as u32 + tile - 1) / tile;
        let grid_y = (n_col_o + tile - 1) / tile;
        let cfg = LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&dequant_permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(&gemm_out_i32)
                .arg(&mut output)
                .launch(cfg)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    println!("pipeline 1 iteration done");

    // === bench: pipeline (3 step) のみ、 weight quant 除く ===
    let warmup = 10;
    for _ in 0..warmup {
        // step 1: im2col_i8 (coalesced 2D grid: x=col_idx, y=rest/tpb)
        let n_col = (nb * hout * wout) as u32;
        let cin_kk_u32 = (cin * k * k) as u32;
        let tpb = 256u32;
        let blocks_y = (cin_kk_u32 + tpb - 1) / tpb;
        let cfg_i = LaunchConfig { grid_dim: (n_col, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&im2col_i8)
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
                .arg(&x)
                .arg(&mut col_i8)
                .launch(cfg_i)
                .unwrap();
        }
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
        // dequant_permute: 32x32 tile transpose
        let tile = 32u32;
        let n_col_o = (nb * hout * wout) as u32;
        let grid_x_o = (cout as u32 + tile - 1) / tile;
        let grid_y_o = (n_col_o + tile - 1) / tile;
        let cfg_o = LaunchConfig { grid_dim: (grid_x_o, grid_y_o, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&dequant_permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(&gemm_out_i32)
                .arg(&mut output)
                .launch(cfg_o)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();

    let n_iter = 200usize;
    let start = Instant::now();
    for _ in 0..n_iter {
        let total = (nb as u64) * (cin as u64) * (k as u64) * (k as u64) * (hout as u64) * (wout as u64);
        let tpb = 256u32;
        let blocks = ((total + tpb as u64 - 1) / tpb as u64) as u32;
        let cfg_i = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&im2col_i8)
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
                .arg(&x)
                .arg(&mut col_i8)
                .launch(cfg_i)
                .unwrap();
        }
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
        // dequant_permute: 32x32 tile transpose
        let tile = 32u32;
        let n_col_o = (nb * hout * wout) as u32;
        let grid_x_o = (cout as u32 + tile - 1) / tile;
        let grid_y_o = (n_col_o + tile - 1) / tile;
        let cfg_o = LaunchConfig { grid_dim: (grid_x_o, grid_y_o, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&dequant_permute)
                .arg(&nb)
                .arg(&cout)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(&gemm_out_i32)
                .arg(&mut output)
                .launch(cfg_o)
                .unwrap();
        }
    }
    stream.synchronize().unwrap();
    let elapsed = start.elapsed();
    let pipeline_ms = elapsed.as_secs_f64() * 1000.0 / n_iter as f64;
    println!("INT8 pipeline (im2col_i8 + GEMM + dequant_permute): {:.3} ms/call (batch={nb})", pipeline_ms);

    // CRCL-v2 full 32-conv 予測
    let full_ms = pipeline_ms * 32.0;
    let full_lps = nb as f64 / (full_ms / 1000.0);
    println!("CRCL-v2 full (32 conv) INT8 pipeline estimate: {:.1} ms/batch → {:.1} leaf/sec", full_ms, full_lps);

    // 設計目標との比較
    let design_target = 1450.0f64;
    let ratio = full_lps / design_target;
    println!("vs 設計目標 1450 leaf/sec: {:.0}% ({})", ratio * 100.0, if ratio >= 1.0 { "✓ ACHIEVED" } else { "未達" });
}
