pub mod cpu;
pub mod function;
pub mod multi;
pub mod operation;
pub mod tensor;

use std::{fmt::Debug, sync::Arc};

use cpu::CpuThread;

use crate::device::operation::{BaseOperations, BlasOperations};

#[derive(Debug)]
pub enum OperationError<T: Debug> {
    TensorOptimisedOut,
    InvalidTensorFormat,
    IndexOutOfBounds,
    UnsupportedOperation,
    MismatchedBatchSizes,
    NoWeightWithID(String),
    WeightLoadingError(String, Option<(usize, usize)>),
    DeviceError(Box<T>),
}

impl<T: Debug> From<T> for OperationError<T> {
    fn from(value: T) -> Self {
        Self::DeviceError(Box::new(value))
    }
}

pub type OperationResult<T> = Result<(), OperationError<T>>;

pub trait DeviceBuffer<D, T>: Sized + Send + Sync {
    type BufferError;

    fn new(device: Arc<D>, size: usize) -> Result<Self, Self::BufferError>;

    fn size(&self) -> usize;

    fn device(&self) -> Arc<D>;

    fn set_zero(&mut self) -> Result<(), Self::BufferError>;

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), Self::BufferError>;

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), Self::BufferError>;

    /// # Safety
    /// Needs to be followed by a synchronise before `buf` is dropped!
    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError>;

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), Self::BufferError>;
}

pub trait Device: Sized + 'static {
    type IdType;
    type Marker;
    type DeviceError: std::fmt::Debug + Default + Send + Sync;
    type BufferI32: DeviceBuffer<Self, i32, BufferError = Self::DeviceError>;
    type BufferF32: DeviceBuffer<Self, f32, BufferError = Self::DeviceError>
        + BaseOperations<BaseError = Self::DeviceError>
        + BlasOperations<BlasError = Self::DeviceError>;

    /// Phase 2 mixed precision 用 f16 buffer (op の中間 storage 用、 weight master は f32 のまま)。
    /// Buffer 機能のみ要求し、 BaseOperations/BlasOperations は要求しない (BufferF32 master + 内部一時 f16 という設計)。
    type BufferF16: DeviceBuffer<Self, half::f16, BufferError = Self::DeviceError>;

    fn new(id: Self::IdType) -> Result<Self, Self::DeviceError>;

    fn synchronise(&self) -> Result<(), Self::DeviceError>;

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError>;

    fn sanity_check(self: Arc<Self>) {
        println!("\x1b[34;1mRunning Sanity Checks\x1b[0m");
        CpuThread::compare_linear_comb(self.clone());
        CpuThread::compare_gemm(self.clone());
        CpuThread::compare_gebmm(self.clone());
        CpuThread::compare_activate(self.clone());
        CpuThread::compare_power_error(self.clone());
        CpuThread::compare_pairwise(self.clone());
        CpuThread::compare_clip(self.clone());
        CpuThread::compare_adam(self.clone());
        CpuThread::compare_copy_or_add_strided(self.clone());
        CpuThread::compare_add(self.clone());
        CpuThread::compare_abs_pow(self.clone());
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    /// Phase 2 (IV案) mixed precision 用: sparse → BufferF16 直接出力。
    /// 既存 `sparse_to_dense` の f16 出力版。 BufferF16 を呼出側で 0 fill した後に
    /// active feature 部分にのみ 1.0 (= `half::f16::ONE`) を書き込む。
    /// 未対応 backend は `OperationError::UnsupportedOperation` を返す。
    fn sparse_to_dense_f16(
        _batch_size: usize,
        _size: usize,
        _nnz: usize,
        _sparse: &Self::BufferI32,
        _dense: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    /// Phase 2 Step 4: BufferF16 → BufferF32 cast (forward 活性化を grad 計算用 f32 path に渡す境界)。
    /// in-place 不可、 `src.size()` ≤ `dst.size()` の前提。
    fn cast_f16_to_f32(
        _size: usize,
        _src: &Self::BufferF16,
        _dst: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    /// Phase 2 Step 4: BufferF32 → BufferF16 cast (weight master / activation を f16 buffer に書き込む境界)。
    fn cast_f32_to_f16(
        _size: usize,
        _src: &Self::BufferF32,
        _dst: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    /// Phase 2 Step G1: SiLU on BufferF16 (elementwise、 in-place 可)。
    fn silu_f16(
        _size: usize,
        _src: &Self::BufferF16,
        _dst: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    /// Phase 2 Step G1: SiLU backward on BufferF16 (silu'(x) * upstream)。
    fn silu_grad_f16(
        _size: usize,
        _x: &Self::BufferF16,
        _upstream: &Self::BufferF16,
        _dst: &mut Self::BufferF16,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    /// Phase 2 Step G4: HGEMM on BufferF16 (CUDA_R_16F IO, CUBLAS_COMPUTE_32F)。
    /// C = alpha * op(A) * op(B) + beta * C (column-major)
    /// op(A) = A or A^T (trans_a で指定)、 lda/ldb/ldc は column-major leading dim。
    #[allow(clippy::too_many_arguments)]
    fn hgemm_f16(
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &Self::BufferF16,
        _trans_a: bool,
        _lda: usize,
        _b: &Self::BufferF16,
        _trans_b: bool,
        _ldb: usize,
        _beta: f32,
        _c: &mut Self::BufferF16,
        _ldc: usize,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}
