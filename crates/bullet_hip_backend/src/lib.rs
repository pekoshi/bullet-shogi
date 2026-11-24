mod backend;
pub mod dense;
mod matmul;
pub mod sparse;

pub use backend::ExecutionContext;
use backend::{Buffer, bindings, ops, util};

use acyclib::{
    device::{
        Device, DeviceBuffer, OperationError,
        operation::{
            AdamConfig, BaseOperations, BlasOperations, CoreDeviceOps, DiffableFromOutput, GemmConfig, Ranger21Config,
            SparseAffineOps,
        },
        tensor::{self, Shape},
    },
    graph::ir::BackendMarker,
};

pub type DenseMatrix = tensor::DenseMatrix<ExecutionContext>;
pub type SparseMatrix = tensor::SparseMatrix<ExecutionContext>;
pub type Matrix = tensor::Matrix<ExecutionContext>;
pub type Tensor = tensor::Tensor<ExecutionContext>;

#[derive(Debug, Default)]
pub enum DeviceError {
    #[default]
    Generic,
    Cuda(bindings::cudaError_t),
    Cublas(bindings::cublasStatus_t),
    ExpectedIllegalAddressAccess,
}

impl From<bindings::cublasStatus_t> for Result<(), DeviceError> {
    fn from(value: bindings::cublasStatus_t) -> Self {
        if value == bindings::CUBLAS_SUCCESS { Ok(()) } else { Err(DeviceError::Cublas(value)) }
    }
}

impl From<bindings::cudaError_t> for Result<(), DeviceError> {
    fn from(value: bindings::cudaError_t) -> Self {
        if value == bindings::SUCCESS { Ok(()) } else { Err(DeviceError::Cuda(value)) }
    }
}

pub(crate) type OperationResult = Result<(), OperationError<DeviceError>>;

impl BlasOperations for Buffer<f32> {
    type BlasError = DeviceError;

    fn gemm(&mut self, config: &GemmConfig, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        matmul::sgemm(config, a, b, self)
    }

    fn gebmm(&mut self, config: &GemmConfig, batch_size: usize, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        matmul::sgemm_batched(config, batch_size, a, b, self)
    }
}

impl BaseOperations for Buffer<f32> {
    type BaseError = DeviceError;

    fn set_to(&mut self, size: usize, val: f32) -> Result<(), Self::BaseError> {
        if size > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }

        unsafe {
            ops::set(self.mut_ptr(), size, val);
        }

        Ok(())
    }

    fn set_range_to(&mut self, start: usize, end: usize, val: f32) -> Result<(), Self::BaseError> {
        if start > end || end > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }
        if start == end {
            return Ok(());
        }

        // SAFETY: `start < end <= self.size()` を上で検証済み。 `mut_ptr().add(start)` は
        // buffer 内の有効な device pointer であり、 書き込み範囲 `end - start` は buffer 末尾を
        // 超えない。
        unsafe {
            ops::set(self.mut_ptr().add(start), end - start, val);
        }

        Ok(())
    }

    fn diffable_from_output_fwd(
        &mut self,
        size: usize,
        a: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError> {
        match act {
            DiffableFromOutput::Identity => panic!("No-op!"),
            DiffableFromOutput::ReLU => dense::relu(size, a, self),
            DiffableFromOutput::CReLU => dense::crelu(size, a, self),
            DiffableFromOutput::SCReLU => dense::screlu(size, a, self),
            DiffableFromOutput::SqrReLU => dense::sqrrelu(size, a, self),
            DiffableFromOutput::Sigmoid => dense::sigmoid(size, a, self),
            DiffableFromOutput::LeakyReLU => dense::leaky_relu(size, a, self),
        }
    }

    fn diffable_from_output_bwd(
        &mut self,
        size: usize,
        a: &Self,
        grd: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError> {
        match act {
            DiffableFromOutput::Identity => panic!("No-op!"),
            DiffableFromOutput::ReLU => dense::relu_backward(size, a, self, grd),
            DiffableFromOutput::CReLU => dense::crelu_backward(size, a, self, grd),
            DiffableFromOutput::SCReLU => dense::screlu_backward(size, a, self, grd),
            DiffableFromOutput::SqrReLU => dense::sqrrelu_backward(size, a, self, grd),
            DiffableFromOutput::Sigmoid => dense::sigmoid_backward(size, a, self, grd),
            DiffableFromOutput::LeakyReLU => dense::leaky_relu_backward(size, a, self, grd),
        }
    }

    fn mul_scalar(&mut self, size: usize, alpha: f32) -> Result<(), Self::BaseError> {
        if size > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }

        unsafe {
            ops::scale_assign(size, self.mut_ptr(), alpha);
        }

        Ok(())
    }

    fn add_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        if size > input.size() || size > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }

        unsafe {
            ops::add_scalar(size, alpha, input.ptr(), self.mut_ptr());
        }

        Ok(())
    }

    fn linear_comb(&mut self, size: usize, alpha: f32, beta: f32, input: &Self) -> Result<(), Self::BaseError> {
        if size > input.size() || size > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }

        unsafe {
            ops::scale_add_assign(size, alpha, self.mut_ptr(), beta, input.ptr());
        }

        Ok(())
    }

    fn linear_comb_splat(
        &mut self,
        size: usize,
        reps: usize,
        alpha: f32,
        beta: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError> {
        let device = self.device();

        device.with_ones(reps, |ones| {
            let cfg = GemmConfig::new(beta, alpha, Shape::new(size, 1), false, Shape::new(1, reps), false);
            self.gemm(&cfg, input, ones)
        })
    }

    fn reduce_across_batch(
        &mut self,
        size: usize,
        reps: usize,
        output_mul: f32,
        input_mul: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError> {
        let device = self.device();

        device.with_ones(reps, |ones| {
            let cfg = GemmConfig::new(input_mul, output_mul, Shape::new(size, reps), false, Shape::new(reps, 1), false);
            self.gemm(&cfg, input, ones)
        })
    }

    fn abs_pow_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError> {
        if size > input.size() || size > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }

        unsafe {
            ops::abs_pow_scalar(size, alpha, input.ptr(), self.mut_ptr());
        }

        Ok(())
    }

    fn silu_fwd(&mut self, size: usize, input: &Self) -> Result<(), Self::BaseError> {
        dense::silu(size, input, self)
    }

    fn silu_bwd(&mut self, size: usize, input: &Self, output_grad: &Self) -> Result<(), Self::BaseError> {
        dense::silu_backward(size, input, self, output_grad)
    }

    fn rms_norm_fwd(
        &mut self,
        _feature_size: usize,
        _batch_size: usize,
        _eps: f32,
        _input: &Self,
    ) -> Result<(), Self::BaseError> {
        unimplemented!("RmsNorm not yet implemented for HIP backend; use CPU or CUDA backend")
    }

    fn rms_norm_bwd(
        &mut self,
        _feature_size: usize,
        _batch_size: usize,
        _eps: f32,
        _input: &Self,
        _output_grad: &Self,
    ) -> Result<(), Self::BaseError> {
        unimplemented!("RmsNorm not yet implemented for HIP backend; use CPU or CUDA backend")
    }

    fn softmax_fwd(&mut self, _feature_size: usize, _batch_size: usize, _input: &Self) -> Result<(), Self::BaseError> {
        unimplemented!("standalone Softmax not yet implemented for HIP backend; use CPU or CUDA backend")
    }

    fn softmax_bwd(
        &mut self,
        _feature_size: usize,
        _batch_size: usize,
        _softmax_out: &Self,
        _output_grad: &Self,
    ) -> Result<(), Self::BaseError> {
        unimplemented!("standalone Softmax not yet implemented for HIP backend; use CPU or CUDA backend")
    }

    fn abs_pow_scalar_backward(
        &mut self,
        size: usize,
        alpha: f32,
        input: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        if size > input.size() || size > grd.size() || size > self.size() {
            return Err(DeviceError::ExpectedIllegalAddressAccess);
        }

        unsafe {
            ops::abs_pow_scalar_backward(size, alpha, input.ptr(), grd.ptr(), self.mut_ptr());
        }

        Ok(())
    }

    fn copy_or_add_strided(
        &mut self,
        add: bool,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        a: &Self,
        offset_a: usize,
        stride_a: usize,
    ) -> Result<(), Self::BaseError> {
        dense::copy_or_add_strided(rows, cols, a, offset_a, stride_a, self, offset, stride, add)
    }

    fn pairwise_fwd(
        &mut self,
        offset: usize,
        stride: usize,
        size: usize,
        batch_size: usize,
        a: &Self,
    ) -> Result<(), Self::BaseError> {
        dense::pairwise(offset, stride, size, batch_size, a, self)
    }

    fn pairwise_bwd(
        &mut self,
        offset: usize,
        stride: usize,
        size: usize,
        batch_size: usize,
        a: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        dense::backprop_pairwise(offset, stride, size, batch_size, a, grd, self)
    }

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError> {
        dense::abs_power_error(power, size, a, b, self)
    }

    fn power_error_bwd(
        &mut self,
        power: f32,
        size: usize,
        a: &Self,
        b: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError> {
        dense::backprop_abs_power_error_single(power, size, a, b, grd, self)
    }

    fn adam(
        &mut self,
        config: &AdamConfig,
        size: usize,
        grd: &Self,
        mom: &mut Self,
        vel: &mut Self,
    ) -> Result<(), Self::BaseError> {
        dense::adam(size, self, grd, mom, vel, config)
    }

    fn ranger21(
        &mut self,
        _config: &Ranger21Config,
        _offset: usize,
        _size: usize,
        _grd: &Self,
        _momentum: &mut Self,
        _velocity: &mut Self,
        _max_velocity: &mut Self,
    ) -> Result<(), Self::BaseError> {
        panic!("Ranger21 core is currently supported by the CUDA and CPU backends only")
    }

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError> {
        dense::clip(size, self, min, max)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HipMarker;
impl BackendMarker for HipMarker {
    type Backend = ExecutionContext;
}

impl Device for ExecutionContext {
    type Marker = HipMarker;
    type BufferF32 = Buffer<f32>;
    type BufferI32 = Buffer<i32>;
    type BufferF16 = Buffer<half::f16>;
    type DeviceError = DeviceError;
    type IdType = ();

    fn new(_: Self::IdType) -> Result<Self, DeviceError> {
        Ok(Self::default())
    }

    fn synchronise(&self) -> Result<(), DeviceError> {
        util::device_synchronise()
    }

    fn get_last_device_error(&self) -> Result<(), DeviceError> {
        util::get_last_error()
    }

    fn sparse_to_dense(
        batch_size: usize,
        size: usize,
        nnz: usize,
        sparse: &Self::BufferI32,
        dense: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::sparse_to_dense(batch_size, size, nnz, sparse, dense)
    }
}

impl SparseAffineOps for ExecutionContext {
    fn backprop_sparse_affine_activate(
        batch_size: usize,
        activation: DiffableFromOutput,
        input_a_grad: &mut Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c_grad: Option<&mut Self::BufferF32>,
        input_c_batched: bool,
        outputs: &Self::BufferF32,
        output_grad: &Self::BufferF32,
    ) -> OperationResult {
        sparse::backprop_sparse_affine(
            batch_size,
            activation,
            input_a_grad,
            shape_a,
            input_b,
            input_b_vals,
            shape_b,
            nnz,
            input_c_grad,
            input_c_batched,
            outputs,
            output_grad,
        )
    }

    fn sparse_affine_activate(
        batch_size: usize,
        activation: DiffableFromOutput,
        input_a: &Self::BufferF32,
        shape_a: Shape,
        input_b: &Self::BufferI32,
        input_b_vals: Option<&Self::BufferF32>,
        shape_b: Shape,
        nnz: usize,
        input_c: Option<&Self::BufferF32>,
        input_c_batched: bool,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::sparse_affine(
            batch_size,
            activation,
            input_a,
            shape_a,
            input_b,
            input_b_vals,
            shape_b,
            nnz,
            input_c,
            input_c_batched,
            output,
        )
    }
}

impl CoreDeviceOps for ExecutionContext {
    fn select(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::select(batch_size, input_batched, input_size, output_size, input, indices, output)
    }

    fn select_backprop(
        batch_size: usize,
        input_grad_batched: bool,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult {
        sparse::select_backprop(
            batch_size,
            input_grad_batched,
            input_size,
            output_size,
            indices,
            output_grad,
            input_grad,
        )
    }

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        dense::softmax_across_batch(batch_size, single_size, input, output)
    }

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult {
        dense::crossentropy(size, pred, target, output)
    }

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult {
        dense::backprop_softmax_crossentropy(size, softmaxed, target, output_grad, input_grad)
    }
}
