mod base;
mod blas;
mod buffer;

pub use blas::convert_gemm_config;
pub use buffer::CudaBuffer;

use std::sync::{Arc, Mutex};

use acyclib::{
    device::{Device, DeviceBuffer, OperationError, OperationResult, operation::CoreDeviceOps},
    graph::ir::BackendMarker,
};
use cudarc::{
    cublas::{CudaBlas, result::CublasError},
    driver::{CudaContext, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::{self, CompileError},
};

#[derive(Debug, Default)]
pub enum CudaError {
    #[default]
    Generic,
    RuntimeCompile(CompileError),
    Driver(DriverError),
    Blas(CublasError),
    ExpectedIllegalAddressAccess,
    #[cfg(feature = "nccl")]
    Nccl(cudarc::nccl::result::NcclError),
}

#[derive(Debug)]
pub struct CudaDevice {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    module: Arc<CudaModule>,
    copystream: Arc<CudaStream>,
    ones: Mutex<CudaSlice<f32>>,
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new(0).unwrap()
    }
}

impl CudaDevice {
    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    pub fn module(&self) -> Arc<CudaModule> {
        self.module.clone()
    }

    pub fn copystream(&self) -> Arc<CudaStream> {
        self.copystream.clone()
    }

    pub fn with_ones<T, F: FnMut(&CudaSlice<f32>) -> Result<T, CudaError>>(
        self: Arc<Self>,
        count: usize,
        mut f: F,
    ) -> Result<T, CudaError> {
        let mut ones = self.ones.try_lock().unwrap();

        if count > ones.len() {
            *ones = self.stream.alloc_zeros(count).map_err(CudaError::Driver)?;

            base::set_to(self.clone(), &mut ones, count, 1.0).map_err(CudaError::Driver)?;
        }

        f(&ones)
    }

    pub fn elementwise_launch_params(size: usize, threads: u32) -> LaunchConfig {
        let float4_size = (size as u32).div_ceil(4);
        let blocks = float4_size.div_ceil(threads);
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
    }

    pub fn elementwise_launch_params_single(size: usize, threads: u32) -> LaunchConfig {
        let blocks = (size as u32).div_ceil(threads);
        LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaMarker;
impl BackendMarker for CudaMarker {
    type Backend = CudaDevice;
}

impl Device for CudaDevice {
    type Marker = CudaMarker;
    type IdType = usize;
    type DeviceError = CudaError;
    type BufferF32 = CudaBuffer<f32>;
    type BufferI32 = CudaBuffer<i32>;
    type BufferF16 = CudaBuffer<half::f16>;

    fn new(id: Self::IdType) -> Result<Self, Self::DeviceError> {
        let ctx = CudaContext::new(id).map_err(CudaError::Driver)?;
        ctx.set_blocking_synchronize().map_err(CudaError::Driver)?;
        let stream = ctx.default_stream();
        let copystream = ctx.new_stream().map_err(CudaError::Driver)?;
        let blas = CudaBlas::new(stream.clone()).map_err(CudaError::Blas)?;

        static KERNELS: &str = include_str!("kernels.cu");
        // cuda_fp16.h を nvrtc に解決させるため CUDA include dir を渡す。
        // build.rs 経由で CUDA_PATH 等から拾えなければ、 default の /usr/local/cuda(-XX)/include や
        // %CUDA_PATH%/include を試す。
        let cuda_include = std::env::var("CUDA_PATH").ok().map(|p| format!("{}/include", p)).or_else(|| {
            // fallback: ビルド時に compile_time に組み込まれた CUDA_PATH
            option_env!("CUDA_PATH").map(|p| format!("{}/include", p))
        });
        let compile_opts =
            nvrtc::CompileOptions { include_paths: cuda_include.into_iter().collect(), ..Default::default() };
        let ptx = nvrtc::compile_ptx_with_opts(KERNELS, compile_opts).map_err(CudaError::RuntimeCompile)?;

        let module = ctx.load_module(ptx).map_err(CudaError::Driver)?;

        let ones = Mutex::new(stream.alloc_zeros::<f32>(0).map_err(CudaError::Driver)?);

        Ok(Self { stream, blas, module, copystream, ones })
    }

    fn synchronise(&self) -> Result<(), Self::DeviceError> {
        self.stream.synchronize().map_err(CudaError::Driver)?;
        self.copystream.synchronize().map_err(CudaError::Driver)
    }

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError> {
        self.synchronise()
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if batch_size * nnz > sparse.size() || batch_size * size > dense.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        dense.set_zero()?;

        let func = sparse.device.module.load_function("sparse_to_dense").unwrap();
        let threads = batch_size.min(1024);
        let blocks = batch_size.div_ceil(threads) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            sparse
                .device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&(batch_size as i32))
                .arg(&(nnz as i32))
                .arg(&sparse.buf)
                .arg(&mut dense.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    /// Phase 2 Step 4: BufferF16 → BufferF32 cast (`cast_f16_to_f32` kernel)。
    fn cast_f16_to_f32(
        size: usize,
        src: &Self::BufferF16,
        dst: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if size > src.size() || size > dst.size() {
            return Err(OperationError::IndexOutOfBounds);
        }
        let func = src.device.module.load_function("cast_f16_to_f32").unwrap();
        let threads: u32 = 256;
        let blocks = (size as u32).div_ceil(threads);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            src.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&src.buf)
                .arg(&mut dst.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    /// Phase 2 Step G1: SiLU on BufferF16 (in-place 可、 silu_f16 kernel)。
    fn silu_f16(size: usize, src: &Self::BufferF16, dst: &mut Self::BufferF16) -> OperationResult<Self::DeviceError> {
        if size > src.size() || size > dst.size() {
            return Err(OperationError::IndexOutOfBounds);
        }
        let func = src.device.module.load_function("silu_f16").unwrap();
        let threads: u32 = 256;
        let blocks = (size as u32).div_ceil(threads);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            src.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&src.buf)
                .arg(&mut dst.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    /// Phase 2 Step G4: HGEMM on BufferF16 (cublasGemmEx + CUDA_R_16F IO + CUBLAS_COMPUTE_32F)。
    #[allow(clippy::too_many_arguments)]
    fn hgemm_f16(
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &Self::BufferF16,
        trans_a: bool,
        lda: usize,
        b: &Self::BufferF16,
        trans_b: bool,
        ldb: usize,
        beta: f32,
        c: &mut Self::BufferF16,
        ldc: usize,
    ) -> OperationResult<Self::DeviceError> {
        use cudarc::{
            cublas::{
                result::gemm_ex,
                sys::{
                    self,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                    cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
                    cudaDataType_t::CUDA_R_16F,
                },
            },
            driver::{DevicePtr, DevicePtrMut},
        };
        let stream = a.device.stream();
        let op_a = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let op_b = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let (ap, _ra) = a.buf.device_ptr(&stream);
        let (bp, _rb) = b.buf.device_ptr(&stream);
        let (cp, _rc) = c.buf.device_ptr_mut(&stream);
        let _ = sys::cublasComputeType_t::CUBLAS_COMPUTE_32F;
        unsafe {
            gemm_ex(
                *a.device.blas().handle(),
                op_a,
                op_b,
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                ap as *const _,
                CUDA_R_16F,
                lda as i32,
                bp as *const _,
                CUDA_R_16F,
                ldb as i32,
                &beta as *const f32 as *const _,
                cp as *mut _,
                CUDA_R_16F,
                ldc as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DFALT,
            )
            .map_err(CudaError::Blas)?;
        }
        Ok(())
    }

    /// Phase 2 Step G1: SiLU backward on BufferF16 (silu_grad_f16 kernel)。
    fn silu_grad_f16(
        size: usize,
        x: &Self::BufferF16,
        upstream: &Self::BufferF16,
        dst: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        if size > x.size() || size > upstream.size() || size > dst.size() {
            return Err(OperationError::IndexOutOfBounds);
        }
        let func = x.device.module.load_function("silu_grad_f16").unwrap();
        let threads: u32 = 256;
        let blocks = (size as u32).div_ceil(threads);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            x.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&x.buf)
                .arg(&upstream.buf)
                .arg(&mut dst.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    /// Phase 2 Step 4: BufferF32 → BufferF16 cast (`cast_f32_to_f16` kernel)。
    fn cast_f32_to_f16(
        size: usize,
        src: &Self::BufferF32,
        dst: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        if size > src.size() || size > dst.size() {
            return Err(OperationError::IndexOutOfBounds);
        }
        let func = src.device.module.load_function("cast_f32_to_f16").unwrap();
        let threads: u32 = 256;
        let blocks = (size as u32).div_ceil(threads);
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            src.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&src.buf)
                .arg(&mut dst.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }
        Ok(())
    }

    /// Phase 2 (IV案): sparse → BufferF16 直接出力 (kernels.cu の `sparse_to_dense_f16` を呼ぶ)。
    /// 既存 `sparse_to_dense` の f16 版で、 1.0 を f16 として書き込む。
    fn sparse_to_dense_f16(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        if batch_size * nnz > sparse.size() || batch_size * size > dense.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        dense.set_zero()?;

        let func = sparse.device.module.load_function("sparse_to_dense_f16").unwrap();
        let threads = batch_size.min(1024);
        let blocks = batch_size.div_ceil(threads) as u32;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads as u32, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            sparse
                .device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&(batch_size as i32))
                .arg(&(nnz as i32))
                .arg(&sparse.buf)
                .arg(&mut dense.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}

impl CoreDeviceOps for CudaDevice {
    fn select(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if if input_batched { batch_size } else { 1 } * input_size > input.size()
            || batch_size > indices.size()
            || batch_size * output_size > output.size()
        {
            return OperationResult::Err(OperationError::IndexOutOfBounds);
        }

        let func = input.device.module.load_function("select").map_err(CudaError::Driver)?;

        let threads = 1024;
        let grid_dim = (((batch_size * output_size) as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            input
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_batched as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&indices.buf)
                .arg(&input.buf)
                .arg(&mut output.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn select_backprop(
        batch_size: usize,
        input_grad_batched: bool,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if if input_grad_batched { batch_size } else { 1 } * input_size > input_grad.size()
            || batch_size > indices.size()
            || batch_size * output_size > output_grad.size()
        {
            return OperationResult::Err(OperationError::IndexOutOfBounds);
        }

        let func = input_grad.device.module.load_function("select_backprop").map_err(CudaError::Driver)?;

        let threads = 1024;
        let grid_dim = (((batch_size * output_size) as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            input_grad
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_grad_batched as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&indices.buf)
                .arg(&output_grad.buf)
                .arg(&mut input_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let size = batch_size * single_size;
        if size > input.size() || size > output.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func = input.device.module.load_function("softmax").unwrap();

        let threads = 512u32;
        let grid_dim = ((batch_size as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            input
                .device
                .stream
                .launch_builder(&func)
                .arg(&(single_size as i32))
                .arg(&(batch_size as i32))
                .arg(&input.buf.slice(0..size))
                .arg(&mut output.buf.slice_mut(0..size))
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if size > pred.size() || size > target.size() || size > output.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func = pred.device.module.load_function("cross_entropy").unwrap();
        let threads = 512u32;
        let grid_dim = ((size as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            pred.device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&pred.buf.slice(0..size))
                .arg(&target.buf.slice(0..size))
                .arg(&mut output.buf.slice_mut(0..size))
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if size > softmaxed.size() || size > target.size() || size > output_grad.size() || size > input_grad.size() {
            return Err(OperationError::IndexOutOfBounds);
        }

        let func = softmaxed.device.module.load_function("backprop_softmax_cross_entropy").unwrap();
        let threads = 512u32;
        let grid_dim = ((size as u32).div_ceil(threads), 1, 1);
        let cfg = LaunchConfig { grid_dim, block_dim: (threads, 1, 1), shared_mem_bytes: 0 };

        unsafe {
            softmaxed
                .device
                .stream
                .launch_builder(&func)
                .arg(&(size as i32))
                .arg(&softmaxed.buf.slice(0..size))
                .arg(&target.buf.slice(0..size))
                .arg(&output_grad.buf.slice(0..size))
                .arg(&mut input_grad.buf.slice_mut(0..size))
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    // ===== gather affine (2026-11-14 OPNARCH T2)。 kernel は kernels.cu の bucketed_affine_* =====
    // どの kernel も block = 256 threads (8 warps) 固定 (kernel 側が warp 数 8 を前提にしている)。
    fn bucketed_affine_fwd(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        num_buckets: usize,
        weights: &Self::BufferF32,
        bias: &Self::BufferF32,
        input: &Self::BufferF32,
        buckets: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let rows = num_buckets * output_size;
        let shared = 8 * output_size * std::mem::size_of::<f32>();
        if rows * input_size > weights.size()
            || rows > bias.size()
            || batch_size * input_size > input.size()
            || batch_size > buckets.size()
            || batch_size * output_size > output.size()
            || shared > BUCKETED_MAX_SHARED_BYTES
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        if batch_size == 0 {
            return Ok(());
        }

        let func = input.device.module.load_function("bucketed_affine_fwd").map_err(CudaError::Driver)?;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (BUCKETED_THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };

        // SAFETY: 上で全 buffer の長さを kernel が触る範囲 (weights rows*I / bias rows / input B*I /
        //   buckets B / output B*O) 以上と確かめた。 範囲外の bucket 値は kernel 側で読み飛ばす。
        //   共有メモリは 8*O floats を要求し、 kernel はその範囲だけを使う (block = 256 = 8 warps が不変条件)。
        unsafe {
            input
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&(num_buckets as i32))
                .arg(&weights.buf)
                .arg(&bias.buf)
                .arg(&input.buf)
                .arg(&buckets.buf)
                .arg(&mut output.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn bucketed_affine_bwd_input(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        num_buckets: usize,
        weights: &Self::BufferF32,
        buckets: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let rows = num_buckets * output_size;
        let shared = output_size * std::mem::size_of::<f32>();
        if rows * input_size > weights.size()
            || batch_size > buckets.size()
            || batch_size * output_size > output_grad.size()
            || batch_size * input_size > input_grad.size()
            || shared > BUCKETED_MAX_SHARED_BYTES
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        if batch_size == 0 {
            return Ok(());
        }

        let func = weights.device.module.load_function("bucketed_affine_bwd_input").map_err(CudaError::Driver)?;
        let cfg = LaunchConfig {
            grid_dim: (batch_size as u32, 1, 1),
            block_dim: (BUCKETED_THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };

        // SAFETY: buffer の長さは上で確認済み (weights rows*I / buckets B / output_grad B*O / input_grad B*I)。
        //   共有メモリは O floats (dy の 1 行)。 各 block は自分の局面 b の dx 行だけに書く (競合なし)。
        unsafe {
            weights
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&(num_buckets as i32))
                .arg(&weights.buf)
                .arg(&buckets.buf)
                .arg(&output_grad.buf)
                .arg(&mut input_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn bucketed_affine_bwd_weights(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        num_buckets: usize,
        input: &Self::BufferF32,
        buckets: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        weights_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let rows = num_buckets * output_size;
        let shared = batch_size * std::mem::size_of::<i32>();
        if rows * input_size > weights_grad.size()
            || batch_size * input_size > input.size()
            || batch_size > buckets.size()
            || batch_size * output_size > output_grad.size()
            || shared > BUCKETED_MAX_SHARED_BYTES
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        if batch_size == 0 || num_buckets == 0 || output_size == 0 {
            return Ok(());
        }
        // 1 block が受け持つ列の数。 O * kc がおよそ 1024 (= 1 thread 4 要素) になるようにする。
        let kc = (1024 / output_size).clamp(1, input_size.max(1));

        let func = input.device.module.load_function("bucketed_affine_bwd_weights").map_err(CudaError::Driver)?;
        let cfg = LaunchConfig {
            grid_dim: (num_buckets as u32, input_size.div_ceil(kc) as u32, 1),
            block_dim: (BUCKETED_THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };

        // SAFETY: buffer の長さは上で確認済み。 共有メモリは B 個の i32 (bucket ごとの局面の一覧) で、
        //   kernel は一覧を B 個以内に詰める。 block (bk, kt) は dW の行 bk*O.. / 列 kt*kc.. の要素だけに書くので
        //   block 間で書き込みが重ならない (atomic 不要)。
        unsafe {
            input
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(input_size as i32))
                .arg(&(output_size as i32))
                .arg(&(num_buckets as i32))
                .arg(&(kc as i32))
                .arg(&input.buf)
                .arg(&buckets.buf)
                .arg(&output_grad.buf)
                .arg(&mut weights_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }

    fn bucketed_affine_bwd_bias(
        batch_size: usize,
        output_size: usize,
        num_buckets: usize,
        buckets: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        bias_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let shared = batch_size * std::mem::size_of::<i32>();
        if num_buckets * output_size > bias_grad.size()
            || batch_size > buckets.size()
            || batch_size * output_size > output_grad.size()
            || shared > BUCKETED_MAX_SHARED_BYTES
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        if batch_size == 0 || num_buckets == 0 {
            return Ok(());
        }

        let func = buckets.device.module.load_function("bucketed_affine_bwd_bias").map_err(CudaError::Driver)?;
        let cfg = LaunchConfig {
            grid_dim: (num_buckets as u32, 1, 1),
            block_dim: (BUCKETED_THREADS, 1, 1),
            shared_mem_bytes: shared as u32,
        };

        // SAFETY: buffer の長さは上で確認済み。 block bk は dbias の bk*O.. の O 要素だけに書く (競合なし)。
        unsafe {
            buckets
                .device
                .stream
                .launch_builder(&func)
                .arg(&(batch_size as i32))
                .arg(&(output_size as i32))
                .arg(&(num_buckets as i32))
                .arg(&buckets.buf)
                .arg(&output_grad.buf)
                .arg(&mut bias_grad.buf)
                .launch(cfg)
                .map_err(CudaError::Driver)?;
        }

        Ok(())
    }
}

/// gather affine の kernel の block の大きさ (kernels.cu が 8 warps を前提にしている)。
const BUCKETED_THREADS: u32 = 256;
/// gather affine の kernel が要求してよい動的共有メモリの上限 (既定の 48 KB から静的な cnt[9] の分を引いた値)。
const BUCKETED_MAX_SHARED_BYTES: usize = 48 * 1024 - 64;
