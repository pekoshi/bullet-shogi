use crate::device::{
    Device, OperationError,
    function::{DeviceOperation, UnaryOp},
    operation::{BaseOperations, CoreDeviceOps},
    tensor::TensorRef,
};

#[derive(Clone)]
pub struct AbsPowerError<D: Device> {
    pub a: TensorRef<D>,
    pub b: TensorRef<D>,
    pub power: f32,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for AbsPowerError<D> {
    fn opname(&self) -> String {
        format!("AbsPowerError({})", self.power)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let a = self.a.dense();
        let b = self.b.dense();
        let mut output = self.output.dense_mut();

        if a.batch_size() != b.batch_size() || a.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if a.single_size() != b.single_size() || a.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        output.buf.power_error_fwd(self.power, a.size(), &a.buf, &b.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct UnaryBackward<D: Device> {
    pub input: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
    pub op: UnaryOp,
}

impl<D: Device> DeviceOperation<D> for UnaryBackward<D> {
    fn opname(&self) -> String {
        format!("UnaryBackward({:?})", self.op)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let mut input_grad = self.input_grad.dense_mut();
        let output_grad = self.output_grad.dense();
        let input = self.input.dense();

        if input.batch_size() != input_grad.batch_size() || input.batch_size() != output_grad.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != input_grad.single_size() || input.single_size() != output_grad.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = output_grad.size();
        let igrd = &mut input_grad.buf;
        let ogrd = &output_grad.buf;

        match self.op {
            UnaryOp::AbsPow(p) => igrd.abs_pow_scalar_backward(size, p, &input.buf, ogrd)?,
            UnaryOp::Add(_) => igrd.linear_comb(size, 1.0, 1.0, ogrd)?,
            UnaryOp::Mul(x) => igrd.linear_comb(size, 1.0, x, ogrd)?,
            UnaryOp::DiffableFromOutput(act) => igrd.diffable_from_output_bwd(size, &input.buf, ogrd, act)?,
            UnaryOp::Silu => igrd.silu_bwd(size, &input.buf, ogrd)?,
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct SoftmaxBackward<D: Device> {
    pub softmax_out: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
    pub feature_size: usize,
}

impl<D: Device> DeviceOperation<D> for SoftmaxBackward<D> {
    fn opname(&self) -> String {
        "SoftmaxBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let mut input_grad = self.input_grad.dense_mut();
        let output_grad = self.output_grad.dense();
        let softmax_out = self.softmax_out.dense();

        if softmax_out.batch_size() != input_grad.batch_size() || softmax_out.batch_size() != output_grad.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if softmax_out.single_size() != input_grad.single_size()
            || softmax_out.single_size() != output_grad.single_size()
            || softmax_out.single_size() != self.feature_size
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = softmax_out.batch_size().unwrap_or(1);
        input_grad.buf.softmax_bwd(self.feature_size, bs, &softmax_out.buf, &output_grad.buf)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct RmsNormBackward<D: Device> {
    pub input: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
    pub feature_size: usize,
    pub eps: f32,
}

impl<D: Device> DeviceOperation<D> for RmsNormBackward<D> {
    fn opname(&self) -> String {
        "RmsNormBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let mut input_grad = self.input_grad.dense_mut();
        let output_grad = self.output_grad.dense();
        let input = self.input.dense();

        if input.batch_size() != input_grad.batch_size() || input.batch_size() != output_grad.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if input.single_size() != input_grad.single_size()
            || input.single_size() != output_grad.single_size()
            || input.single_size() != self.feature_size
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = input.batch_size().unwrap_or(1);
        input_grad.buf.rms_norm_bwd(self.feature_size, bs, self.eps, &input.buf, &output_grad.buf)?;
        Ok(())
    }
}

pub struct PairwiseMulBackward<D: Device> {
    pub offset: usize,
    pub values: TensorRef<D>,
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for PairwiseMulBackward<D> {
    fn opname(&self) -> String {
        "PairwiseMulBackward".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let values = self.values.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() || input.batch_size() != values.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if output.single_size() > 2 * input.single_size() || output.single_size() != values.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let single_size = output.single_size();
        let stride = input.single_size();
        let batch_size = input.batch_size().unwrap_or(1);

        output.buf.pairwise_bwd(self.offset, stride, single_size, batch_size, &values.buf, &input.buf)?;

        Ok(())
    }
}

pub struct Select<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub buckets: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for Select<D> {
    fn opname(&self) -> String {
        "Select".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();
        let buckets = self.buckets.sparse();

        if output.batch_size() != buckets.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let input_size = input.single_size();
        let output_size = output.single_size();

        if input_size != buckets.single_size() * output_size || buckets.nnz() != 1 {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::select(
            output.batch_size().unwrap_or(1),
            input.batch_size().is_some(),
            input_size,
            output_size,
            &input.buf,
            &buckets.buf,
            &mut output.buf,
        )
    }
}

pub struct SelectBackprop<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub buckets: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for SelectBackprop<D> {
    fn opname(&self) -> String {
        "SelectBackprop".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();
        let buckets = self.buckets.sparse();

        if input.batch_size() != buckets.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let input_size = input.single_size();
        let output_size = output.single_size();

        if output_size != buckets.single_size() * input_size || buckets.nnz() != 1 {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::select_backprop(
            input.batch_size().unwrap_or(1),
            output.batch_size().is_some(),
            output_size,
            input_size,
            &buckets.buf,
            &input.buf,
            &mut output.buf,
        )
    }
}

pub struct CrossEntropy<D: Device> {
    pub a: TensorRef<D>,
    pub b: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for CrossEntropy<D> {
    fn opname(&self) -> String {
        "CrossEntropy".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let a = self.a.dense();
        let b = self.b.dense();
        let mut output = self.output.dense_mut();

        if a.batch_size() != b.batch_size() || a.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if a.single_size() != b.single_size() || a.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::crossentropy(a.size(), &a.buf, &b.buf, &mut output.buf)
    }
}

// ===== gather affine (選んだ bucket の行だけを計算する affine)。 2026-11-14 OPNARCH T2 =====
// 形は `CoreDeviceOps::bucketed_affine_*` の説明を参照。 `output_size` は 1 bucket あたりの出力数。
// num_buckets は buckets (sparse, nnz = 1) の single_size から取る。

/// gather affine の順伝播 (出力は上書き)。
pub struct BucketedAffineFwd<D: Device> {
    pub weights: TensorRef<D>,
    pub bias: TensorRef<D>,
    pub input: TensorRef<D>,
    pub buckets: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for BucketedAffineFwd<D> {
    fn opname(&self) -> String {
        "BucketedAffineFwd".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let weights = self.weights.dense();
        let bias = self.bias.dense();
        let input = self.input.dense();
        let buckets = self.buckets.sparse();
        let mut output = self.output.dense_mut();

        if weights.batch_size().is_some() || bias.batch_size().is_some() {
            return Err(OperationError::InvalidTensorFormat);
        }
        let batch_size = buckets.batch_size().unwrap_or(1);
        if input.batch_size().unwrap_or(1) != batch_size || output.batch_size().unwrap_or(1) != batch_size {
            return Err(OperationError::MismatchedBatchSizes);
        }
        let num_buckets = buckets.single_size();
        let output_size = output.single_size();
        let input_size = input.single_size();
        if buckets.nnz() != 1
            || weights.single_size() != num_buckets * output_size * input_size
            || bias.single_size() != num_buckets * output_size
        {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::bucketed_affine_fwd(
            batch_size,
            input_size,
            output_size,
            num_buckets,
            &weights.buf,
            &bias.buf,
            &input.buf,
            &buckets.buf,
            &mut output.buf,
        )
    }
}

/// gather affine の入力の勾配 (加算)。
pub struct BucketedAffineBwdInput<D: Device> {
    pub weights: TensorRef<D>,
    pub buckets: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub input_grad: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for BucketedAffineBwdInput<D> {
    fn opname(&self) -> String {
        "BucketedAffineBwdInput".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let weights = self.weights.dense();
        let buckets = self.buckets.sparse();
        let output_grad = self.output_grad.dense();
        let mut input_grad = self.input_grad.dense_mut();

        let batch_size = buckets.batch_size().unwrap_or(1);
        if output_grad.batch_size().unwrap_or(1) != batch_size || input_grad.batch_size().unwrap_or(1) != batch_size {
            return Err(OperationError::MismatchedBatchSizes);
        }
        let num_buckets = buckets.single_size();
        let output_size = output_grad.single_size();
        let input_size = input_grad.single_size();
        if buckets.nnz() != 1 || weights.single_size() != num_buckets * output_size * input_size {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::bucketed_affine_bwd_input(
            batch_size,
            input_size,
            output_size,
            num_buckets,
            &weights.buf,
            &buckets.buf,
            &output_grad.buf,
            &mut input_grad.buf,
        )
    }
}

/// gather affine の重みの勾配 (加算)。
pub struct BucketedAffineBwdWeights<D: Device> {
    pub input: TensorRef<D>,
    pub buckets: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub weights_grad: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for BucketedAffineBwdWeights<D> {
    fn opname(&self) -> String {
        "BucketedAffineBwdWeights".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let buckets = self.buckets.sparse();
        let output_grad = self.output_grad.dense();
        let mut weights_grad = self.weights_grad.dense_mut();

        let batch_size = buckets.batch_size().unwrap_or(1);
        if input.batch_size().unwrap_or(1) != batch_size || output_grad.batch_size().unwrap_or(1) != batch_size {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if weights_grad.batch_size().is_some() {
            return Err(OperationError::InvalidTensorFormat);
        }
        let num_buckets = buckets.single_size();
        let output_size = output_grad.single_size();
        let input_size = input.single_size();
        if buckets.nnz() != 1 || weights_grad.single_size() != num_buckets * output_size * input_size {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::bucketed_affine_bwd_weights(
            batch_size,
            input_size,
            output_size,
            num_buckets,
            &input.buf,
            &buckets.buf,
            &output_grad.buf,
            &mut weights_grad.buf,
        )
    }
}

/// gather affine の bias の勾配 (加算)。
pub struct BucketedAffineBwdBias<D: Device> {
    pub buckets: TensorRef<D>,
    pub output_grad: TensorRef<D>,
    pub bias_grad: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for BucketedAffineBwdBias<D> {
    fn opname(&self) -> String {
        "BucketedAffineBwdBias".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let buckets = self.buckets.sparse();
        let output_grad = self.output_grad.dense();
        let mut bias_grad = self.bias_grad.dense_mut();

        let batch_size = buckets.batch_size().unwrap_or(1);
        if output_grad.batch_size().unwrap_or(1) != batch_size {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if bias_grad.batch_size().is_some() {
            return Err(OperationError::InvalidTensorFormat);
        }
        let num_buckets = buckets.single_size();
        let output_size = output_grad.single_size();
        if buckets.nnz() != 1 || bias_grad.single_size() != num_buckets * output_size {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::bucketed_affine_bwd_bias(
            batch_size,
            output_size,
            num_buckets,
            &buckets.buf,
            &output_grad.buf,
            &mut bias_grad.buf,
        )
    }
}
