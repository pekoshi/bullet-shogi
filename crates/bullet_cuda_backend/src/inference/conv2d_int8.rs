//! CRCL-v2 INT8 inference Conv2d (graph IR 非依存、 stand-alone モジュール)
//!
//! 訓練後 PTQ で量子化された i8 weight を読み込み、 f32 input → f32 output の forward を
//! 1050 Ti DP4A 経由 cublasGemmEx INT8 で実行。
//!
//! 設計:
//!   - 訓練は既存 FP32 Conv2d (graph IR) で行う
//!   - 保存時に weight を PTQ で i8 化、 scale_w を別途記録
//!   - inference 時は本モジュールの `Conv2dInt8` を chain して使う
//!   - 入力 scale_x は per-tensor (max(abs)) ベース、 batch 毎 / layer 毎で fit
//!
//! 1 Conv の pipeline:
//!   input(f32 NCHW) → im2col_i8(quantize 込) → cublasGemmEx INT8 (weight_i8 × col_i8 → i32)
//!     → dequant_permute(scale_inv) → output(f32 NCHW)

use std::sync::Arc;

use cudarc::{
    cublas::{
        CudaBlas, result,
        sys::{self, cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t},
    },
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
        PushKernelArg,
    },
    nvrtc::compile_ptx,
};

use crate::CudaError;

const KERNELS_SRC: &str = r#"
// fused im2col + quantize、 col write coalesced (rest 連続を 1 block で担当)
// grid: (col_idx, rest_block, 1)、 block: (tpb, 1, 1)
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

// fused dequant + permute, shared memory 32×32 tile transpose
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

    int row_g = co_tile * DEQUANT_TILE + tx;
    int col_g = col_tile * DEQUANT_TILE + ty;
    if (row_g < Cout && col_g < nhw_total) {
        long gidx = (long)col_g * Cout + row_g;
        tile[ty][tx] = (float)gemm_out[gidx] * scale_inv;
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

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

// f32 weight → i8 weight (PTQ 用、 inference 開始時 1 度だけ呼ぶ)
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

// fused dequant + permute + SiLU (post-op = silu のみ)
// output[i] = silu(gemm_out * scale_inv) where silu(z) = z / (1 + exp(-z))
extern "C" __global__ void DequantPermuteSiLUI32ToNCHW(
    int N, int Cout, int Hout, int Wout, float scale_inv,
    const int* __restrict__ gemm_out,
    float* __restrict__ output)
{
    __shared__ float tile[32][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int co_tile = blockIdx.x;
    int col_tile = blockIdx.y;
    int whw = Hout * Wout;
    int nhw_total = N * whw;

    int row_g = co_tile * 32 + tx;
    int col_g = col_tile * 32 + ty;
    if (row_g < Cout && col_g < nhw_total) {
        long gidx = (long)col_g * Cout + row_g;
        tile[ty][tx] = (float)gemm_out[gidx] * scale_inv;
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    int co_out = co_tile * 32 + ty;
    int col_out = col_tile * 32 + tx;
    if (co_out < Cout && col_out < nhw_total) {
        int wo = col_out % Wout;
        int ho = (col_out / Wout) % Hout;
        int n = col_out / whw;
        long oidx = ((long)(n * Cout + co_out) * Hout + ho) * Wout + wo;
        float v = tile[tx][ty];
        output[oidx] = v / (1.0f + expf(-v));  // silu
    }
}

// fused dequant + permute + add residual + SiLU (post-op = add+silu)
// output[i] = silu((gemm_out * scale_inv) + residual[i])
extern "C" __global__ void DequantPermuteResidualSiLUI32ToNCHW(
    int N, int Cout, int Hout, int Wout, float scale_inv,
    const int* __restrict__ gemm_out,
    const float* __restrict__ residual,
    float* __restrict__ output)
{
    __shared__ float tile[32][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int co_tile = blockIdx.x;
    int col_tile = blockIdx.y;
    int whw = Hout * Wout;
    int nhw_total = N * whw;

    int row_g = co_tile * 32 + tx;
    int col_g = col_tile * 32 + ty;
    if (row_g < Cout && col_g < nhw_total) {
        long gidx = (long)col_g * Cout + row_g;
        tile[ty][tx] = (float)gemm_out[gidx] * scale_inv;
    } else {
        tile[ty][tx] = 0.0f;
    }
    __syncthreads();

    int co_out = co_tile * 32 + ty;
    int col_out = col_tile * 32 + tx;
    if (co_out < Cout && col_out < nhw_total) {
        int wo = col_out % Wout;
        int ho = (col_out / Wout) % Hout;
        int n = col_out / whw;
        long oidx = ((long)(n * Cout + co_out) * Hout + ho) * Wout + wo;
        float v = tile[tx][ty] + residual[oidx];
        output[oidx] = v / (1.0f + expf(-v));  // silu
    }
}
"#;

/// CRCL-v2 inference Conv2d の shape + 量子化 params
#[derive(Clone, Copy, Debug)]
pub struct Conv2dInt8Params {
    pub in_channels: i32,
    pub out_channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub kernel_size: i32,
    pub padding: i32,
    pub stride: i32,
    /// input quantize scale (= 127 / max(abs(input)))、 calibration で確定
    pub scale_x: f32,
    /// weight quantize scale (= 127 / max(abs(weight)))
    pub scale_w: f32,
}

impl Conv2dInt8Params {
    pub fn out_h(&self) -> i32 {
        (self.spatial_h + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
    pub fn out_w(&self) -> i32 {
        (self.spatial_w + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
    pub fn output_size(&self, batch: i32) -> usize {
        (batch * self.out_channels * self.out_h() * self.out_w()) as usize
    }
    pub fn input_size(&self) -> usize {
        (self.in_channels * self.spatial_h * self.spatial_w) as usize
    }
    pub fn dq_scale(&self) -> f32 {
        1.0 / (self.scale_x * self.scale_w)
    }
}

/// 共有 kernels (Module 1 個に 5 kernel)
#[derive(Clone)]
pub struct Int8Kernels {
    pub _module: Arc<CudaModule>,
    pub im2col_i8: CudaFunction,
    pub dequant_permute: CudaFunction,
    pub dequant_permute_silu: CudaFunction,
    pub dequant_permute_residual_silu: CudaFunction,
    pub quant_weight: CudaFunction,
}

impl Int8Kernels {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, CudaError> {
        let ptx = compile_ptx(KERNELS_SRC).map_err(CudaError::RuntimeCompile)?;
        let module = ctx.load_module(ptx).map_err(CudaError::Driver)?;
        let im2col_i8 = module.load_function("Im2ColI8NCHW").map_err(CudaError::Driver)?;
        let dequant_permute = module.load_function("DequantPermuteI32ToNCHW").map_err(CudaError::Driver)?;
        let dequant_permute_silu = module.load_function("DequantPermuteSiLUI32ToNCHW").map_err(CudaError::Driver)?;
        let dequant_permute_residual_silu =
            module.load_function("DequantPermuteResidualSiLUI32ToNCHW").map_err(CudaError::Driver)?;
        let quant_weight = module.load_function("QuantizeWeightI8").map_err(CudaError::Driver)?;
        Ok(Self {
            _module: module,
            im2col_i8,
            dequant_permute,
            dequant_permute_silu,
            dequant_permute_residual_silu,
            quant_weight,
        })
    }
}

/// stand-alone Conv2d INT8 inference op
///
/// - `weight_i8` は PTQ 後の i8 weight (`Conv2dInt8::quantize_weight` で f32 → i8 変換)
/// - 内部の scratch (col_i8, gemm_out_i32) は batch_size に応じて re-alloc される
pub struct Conv2dInt8 {
    pub params: Conv2dInt8Params,
    pub weight_i8: Arc<CudaSlice<i8>>,
    pub kernels: Int8Kernels,
    pub blas: Arc<CudaBlas>,
    pub stream: Arc<CudaStream>,
    /// (batch_size, col_i8 buffer) — batch 変わったら re-alloc
    scratch_col: Option<(i32, CudaSlice<i8>)>,
    scratch_gemm: Option<(i32, CudaSlice<i32>)>,
}

impl Conv2dInt8 {
    /// f32 weight (`weight_f32`、 not batched、 size = Cout*Cin*K*K) を i8 に量子化して保持する。
    /// scale_w は `params.scale_w` をそのまま使う (calibration 後の値前提)。
    pub fn from_f32_weight(
        params: Conv2dInt8Params,
        weight_f32: &CudaSlice<f32>,
        kernels: Int8Kernels,
        blas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        let n_w = (params.out_channels * params.in_channels * params.kernel_size * params.kernel_size) as usize;
        assert_eq!(weight_f32.len(), n_w, "weight_f32 size mismatch");
        let mut weight_i8 = stream.alloc_zeros::<i8>(n_w).map_err(CudaError::Driver)?;
        let n_elem = n_w as i64;
        let scale_w = params.scale_w;
        let tpb = 256u32;
        let blocks = (n_w as u64).div_ceil(tpb as u64) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream
                .launch_builder(&kernels.quant_weight)
                .arg(&n_elem)
                .arg(&scale_w)
                .arg(weight_f32)
                .arg(&mut weight_i8)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(Self {
            params,
            weight_i8: Arc::new(weight_i8),
            kernels,
            blas,
            stream,
            scratch_col: None,
            scratch_gemm: None,
        })
    }

    /// 既に i8 量子化済 weight を直接受け取る (.crcli8 reader の出力をそのまま load する用)。
    /// `params.scale_w` は PTQ 時に算出されたものをそのまま使う。
    pub fn from_i8_weight(
        params: Conv2dInt8Params,
        weight_i8_host: &[i8],
        kernels: Int8Kernels,
        blas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        let n_w = (params.out_channels * params.in_channels * params.kernel_size * params.kernel_size) as usize;
        assert_eq!(weight_i8_host.len(), n_w, "weight_i8 size mismatch");
        let weight_i8 = stream.memcpy_stod(weight_i8_host).map_err(CudaError::Driver)?;
        Ok(Self {
            params,
            weight_i8: Arc::new(weight_i8),
            kernels,
            blas,
            stream,
            scratch_col: None,
            scratch_gemm: None,
        })
    }

    fn ensure_scratch(&mut self, batch: i32) -> Result<(), CudaError> {
        let p = &self.params;
        let cin_kk = (p.in_channels * p.kernel_size * p.kernel_size) as usize;
        let nhw = (batch * p.out_h() * p.out_w()) as usize;
        let col_size = cin_kk * nhw;
        let gemm_size = (p.out_channels as usize) * nhw;

        if self.scratch_col.as_ref().map(|(b, _)| *b) != Some(batch) {
            let buf = self.stream.alloc_zeros::<i8>(col_size).map_err(CudaError::Driver)?;
            self.scratch_col = Some((batch, buf));
        }
        if self.scratch_gemm.as_ref().map(|(b, _)| *b) != Some(batch) {
            let buf = self.stream.alloc_zeros::<i32>(gemm_size).map_err(CudaError::Driver)?;
            self.scratch_gemm = Some((batch, buf));
        }
        Ok(())
    }

    /// 内部用: step 1 (im2col_i8) + step 2 (cublasGemmEx) を実行。 dequant_permute は呼び出し側で行う。
    fn forward_to_i32(&mut self, batch: i32, input: &CudaSlice<f32>) -> Result<(), CudaError> {
        self.ensure_scratch(batch)?;
        let p = self.params;
        let cin_kk = p.in_channels * p.kernel_size * p.kernel_size;
        let nhw = batch * p.out_h() * p.out_w();

        // step 1: im2col_i8
        {
            let col_i8 = &mut self.scratch_col.as_mut().unwrap().1;
            let tpb = 256u32;
            let blocks_y = (cin_kk as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (nhw as u32, blocks_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let hout = p.out_h();
            let wout = p.out_w();
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.im2col_i8)
                    .arg(&batch)
                    .arg(&p.in_channels)
                    .arg(&p.spatial_h)
                    .arg(&p.spatial_w)
                    .arg(&hout)
                    .arg(&wout)
                    .arg(&p.kernel_size)
                    .arg(&p.padding)
                    .arg(&p.stride)
                    .arg(&p.scale_x)
                    .arg(input)
                    .arg(col_i8)
                    .launch(cfg)
                    .map_err(CudaError::Driver)?;
            }
        }

        // step 2: cublasGemmEx INT8
        {
            let col_i8 = &self.scratch_col.as_ref().unwrap().1;
            let gemm_i32 = &mut self.scratch_gemm.as_mut().unwrap().1;
            let alpha: i32 = 1;
            let beta: i32 = 0;
            let (a, _ra) = self.weight_i8.device_ptr(&self.stream);
            let (b, _rb) = col_i8.device_ptr(&self.stream);
            let (c, _rc) = gemm_i32.device_ptr_mut(&self.stream);
            unsafe {
                result::gemm_ex(
                    *self.blas.handle(),
                    cublasOperation_t::CUBLAS_OP_T,
                    cublasOperation_t::CUBLAS_OP_N,
                    p.out_channels,
                    nhw,
                    cin_kk,
                    &alpha as *const i32 as *const _,
                    a as *const _,
                    sys::cudaDataType_t::CUDA_R_8I,
                    cin_kk,
                    b as *const _,
                    sys::cudaDataType_t::CUDA_R_8I,
                    cin_kk,
                    &beta as *const i32 as *const _,
                    c as *mut _,
                    sys::cudaDataType_t::CUDA_R_32I,
                    p.out_channels,
                    cublasComputeType_t::CUBLAS_COMPUTE_32I,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(CudaError::Blas)?;
            }
            drop(_ra);
            drop(_rb);
            drop(_rc);
        }
        Ok(())
    }

    fn dequant_tile_grid(&self, batch: i32) -> LaunchConfig {
        let p = &self.params;
        let nhw = batch * p.out_h() * p.out_w();
        let tile = 32u32;
        let grid_x = (p.out_channels as u32).div_ceil(tile);
        let grid_y = (nhw as u32).div_ceil(tile);
        LaunchConfig { grid_dim: (grid_x, grid_y, 1), block_dim: (tile, tile, 1), shared_mem_bytes: 0 }
    }

    /// forward inference を 1 step 実行する (post-op なし)。
    /// - `input`: shape [batch, Cin, H, W] flatten NCHW row-major
    /// - `output`: shape [batch, Cout, Hout, Wout] flatten NCHW row-major、 caller 確保済
    pub fn forward(
        &mut self,
        batch: i32,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        self.forward_to_i32(batch, input)?;
        let p = self.params;
        let hout = p.out_h();
        let wout = p.out_w();
        let gemm_i32 = &self.scratch_gemm.as_ref().unwrap().1;
        let cfg = self.dequant_tile_grid(batch);
        let dq_scale = p.dq_scale();
        unsafe {
            self.stream
                .launch_builder(&self.kernels.dequant_permute)
                .arg(&batch)
                .arg(&p.out_channels)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(gemm_i32)
                .arg(output)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    /// forward + SiLU activation を fused dequant_permute_silu kernel で実行
    pub fn forward_with_silu(
        &mut self,
        batch: i32,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        self.forward_to_i32(batch, input)?;
        let p = self.params;
        let hout = p.out_h();
        let wout = p.out_w();
        let gemm_i32 = &self.scratch_gemm.as_ref().unwrap().1;
        let cfg = self.dequant_tile_grid(batch);
        let dq_scale = p.dq_scale();
        unsafe {
            self.stream
                .launch_builder(&self.kernels.dequant_permute_silu)
                .arg(&batch)
                .arg(&p.out_channels)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(gemm_i32)
                .arg(output)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    /// forward + residual add + SiLU を 1 fused kernel で実行 (ResNet block 終端用)
    /// output[i] = silu((conv(input)[i]) + residual[i])
    pub fn forward_with_residual_silu(
        &mut self,
        batch: i32,
        input: &CudaSlice<f32>,
        residual: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        self.forward_to_i32(batch, input)?;
        let p = self.params;
        let hout = p.out_h();
        let wout = p.out_w();
        let gemm_i32 = &self.scratch_gemm.as_ref().unwrap().1;
        let cfg = self.dequant_tile_grid(batch);
        let dq_scale = p.dq_scale();
        unsafe {
            self.stream
                .launch_builder(&self.kernels.dequant_permute_residual_silu)
                .arg(&batch)
                .arg(&p.out_channels)
                .arg(&hout)
                .arg(&wout)
                .arg(&dq_scale)
                .arg(gemm_i32)
                .arg(residual)
                .arg(output)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }
}
