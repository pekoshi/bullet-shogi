use crate::device::{
    Device, DeviceBuffer, OperationError,
    function::DeviceOperation,
    operation::{BaseOperations, CoreDeviceOps, DiffableFromOutput},
    tensor::{DenseMatrixF16, Matrix, TensorRef},
};
use parking_lot::Mutex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnaryOp {
    DiffableFromOutput(DiffableFromOutput),
    Add(f32),
    Mul(f32),
    AbsPow(f32),
    /// SiLU (Swish) 活性化: f(x) = x * sigmoid(x)
    Silu,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reduce {
    Sum,
    Avg,
}

#[derive(Clone)]
pub struct MaybeUpdateBatchSize<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for MaybeUpdateBatchSize<D> {
    fn opname(&self) -> String {
        "MaybeUpdateBatchSize".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let batch_size = self.input.batch_size();

        if self.output.batch_size() != batch_size {
            match &mut self.output.borrow_mut().values {
                Matrix::Dense(x) => x.set_batch_size(batch_size)?,
                Matrix::Sparse(x) => x.set_batch_size(batch_size)?,
            }
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct ReduceAcrossBatch<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub input_mul: f32,
    pub output_mul: f32,
    pub reduction: Reduce,
}

impl<D: Device> DeviceOperation<D> for ReduceAcrossBatch<D> {
    fn opname(&self) -> String {
        format!("ReduceAcrossBatch({:?})", self.reduction)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size().is_none() || output.batch_size().is_some() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = input.batch_size().unwrap_or(1);

        let scale = match self.reduction {
            Reduce::Avg => 1.0 / bs as f32,
            Reduce::Sum => 1.0,
        };

        output.buf.reduce_across_batch(input.single_size(), bs, self.output_mul, self.input_mul * scale, &input.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct SplatAcrossBatch<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub input_mul: f32,
    pub output_mul: f32,
    pub reduction: Reduce,
}

impl<D: Device> DeviceOperation<D> for SplatAcrossBatch<D> {
    fn opname(&self) -> String {
        format!("SplatAcrossBatch({:?})", self.reduction)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size().is_some() || output.batch_size().is_none() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = output.batch_size().unwrap_or(1);

        let scale = match self.reduction {
            Reduce::Avg => 1.0 / bs as f32,
            Reduce::Sum => 1.0,
        };

        output.buf.linear_comb_splat(input.single_size(), bs, self.output_mul, self.input_mul * scale, &input.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct LinearCombination<D: Device> {
    pub input_mul: f32,
    pub output_mul: f32,
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for LinearCombination<D> {
    fn opname(&self) -> String {
        "LinearCombination".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        output.buf.linear_comb(input.size(), self.output_mul, self.input_mul, &input.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct LinearCombinationSplat<D: Device> {
    pub input_mul: f32,
    pub output_mul: f32,
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for LinearCombinationSplat<D> {
    fn opname(&self) -> String {
        "LinearCombinationSplat".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size().is_some() || output.batch_size().is_none() {
            println!("{:?} {:?}", input.batch_size(), output.batch_size());
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = output.batch_size().unwrap_or(1);
        output.buf.linear_comb_splat(input.size(), bs, self.output_mul, self.input_mul, &input.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct SparseToDense<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for SparseToDense<D> {
    fn opname(&self) -> String {
        "SparseToDense".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.sparse();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::sparse_to_dense(
            input.batch_size().unwrap_or(1),
            input.single_size(),
            input.nnz(),
            &input.buf,
            &mut output.buf,
        )
    }
}

#[derive(Clone)]
pub struct PairwiseMul<D: Device> {
    pub offset: usize,
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: Device> DeviceOperation<D> for PairwiseMul<D> {
    fn opname(&self) -> String {
        "PairwiseMul".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() > 2 * output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let single_size = input.single_size();
        let stride = output.single_size();
        let batch_size = input.batch_size().unwrap_or(1);

        output.buf.pairwise_fwd(self.offset, stride, single_size, batch_size, &input.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct Unary<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub op: UnaryOp,
}

impl<D: Device> DeviceOperation<D> for Unary<D> {
    fn opname(&self) -> String {
        format!("Unary({:?})", self.op)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = input.size();

        match self.op {
            UnaryOp::AbsPow(p) => output.buf.abs_pow_scalar(size, p, &input.buf)?,
            UnaryOp::Add(x) => output.buf.add_scalar(size, x, &input.buf)?,
            UnaryOp::Mul(x) => output.buf.linear_comb(size, 0.0, x, &input.buf)?,
            UnaryOp::DiffableFromOutput(act) => output.buf.diffable_from_output_fwd(size, &input.buf, act)?,
            UnaryOp::Silu => output.buf.silu_fwd(size, &input.buf)?,
        }

        Ok(())
    }
}

// ============================================================================
// Phase 2 IV案: MixedUnary (中間 f16 buffer 経由の unary 演算)
// ============================================================================
//
// 既存 `Unary<D>` (f32) を無改造で残し、 新規 `MixedUnary<D>` を提供する。
// 入出力 TensorRef は f32 (DenseMatrix<D>) のままで、 execute() の内部で
// 中間 f16 buffer (DenseMatrixF16<D>) を lazy alloc し、 cast→f16 kernel→cast
// の chain を実行する。
//
// 対応 op: Silu (他 op は trait に method 追加すれば拡張可能)。
// 未対応 backend (CpuThread) は `UnsupportedOperation` を返す。

pub struct MixedUnary<D: Device> {
    pub input: TensorRef<D>,  // f32
    pub output: TensorRef<D>, // f32
    pub op: UnaryOp,
    /// 中間 f16 buffer (lazy alloc、 batch 変動対応)。
    pub mid_f16: Mutex<Option<DenseMatrixF16<D>>>,
}

impl<D: Device> MixedUnary<D> {
    pub fn new(input: TensorRef<D>, output: TensorRef<D>, op: UnaryOp) -> Self {
        Self { input, output, op, mid_f16: Mutex::new(None) }
    }
}

impl<D: Device> DeviceOperation<D> for MixedUnary<D> {
    fn opname(&self) -> String {
        format!("MixedUnary({:?})", self.op)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if input.single_size() != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = input.size();
        let single_size = input.single_size();
        let batch_size = input.batch_size();

        // 中間 f16 buffer を lazy alloc / resize
        let mut mid_guard = self.mid_f16.lock();
        let need_alloc = match mid_guard.as_ref() {
            None => true,
            Some(m) => m.single_size() != single_size || m.batch_size() != batch_size,
        };
        if need_alloc {
            let device = input.buf.device();
            *mid_guard = Some(DenseMatrixF16::zeroed(device, single_size, batch_size)?);
        }
        let mid = mid_guard.as_mut().unwrap();

        // cast f32 → f16
        D::cast_f32_to_f16(size, &input.buf, &mut mid.buf)?;

        // op を f16 path で実行
        match self.op {
            UnaryOp::Silu => {
                // in-place で OK (silu_f16 kernel は idx 毎独立)
                // ただし borrow checker 上、 buf を src/dst 両方 borrow できないので
                // 別 f16 buffer を作る必要がある。 一時 buffer は mid 内で完結すべく
                // 「mid → tmp → mid copy」 の 3-step は避ける。
                // 代わりに silu_f16(src=mid, dst=mid) を unsafe 経由 (kernel 自体は安全) で呼ぶ。
                // ここでは Mutex で borrow 衝突回避済み、 mid を 2 度 borrow しない設計に。
                // 簡易解: 一時 mid2 を alloc、 silu_f16(mid, mid2)、 mid に swap back。
                let device = input.buf.device();
                let mut mid2 = DenseMatrixF16::<D>::zeroed(device, single_size, batch_size)?;
                D::silu_f16(size, &mid.buf, &mut mid2.buf)?;
                mid.swap_with(&mut mid2)?;
            }
            _ => return Err(OperationError::UnsupportedOperation),
        }

        // cast f16 → f32 output
        D::cast_f16_to_f32(size, &mid.buf, &mut output.buf)?;

        Ok(())
    }
}

// MixedUnaryBackward: graph IR の上流 grad (output_grad) を mixed precision で
// input_grad に accumulate する instruction。 forward instance とは独立に持つ
// 中間 buffer (f16 × 3 + f32 × 1) を lazy alloc。
pub struct MixedUnaryBackward<D: Device> {
    pub input: TensorRef<D>,       // forward input (f32)、 derivative 計算に使う
    pub input_grad: TensorRef<D>,  // accumulate target (f32)
    pub output_grad: TensorRef<D>, // 上流 grad (f32)
    pub op: UnaryOp,
    /// mid buffers (input_f16, upstream_f16, grad_f16) と temp_f32 grad (cast 戻し用)
    pub mid_x: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_up: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_g: Mutex<Option<DenseMatrixF16<D>>>,
    pub tmp_g32: Mutex<Option<crate::device::tensor::DenseMatrix<D>>>,
}

impl<D: Device> MixedUnaryBackward<D> {
    pub fn new(input: TensorRef<D>, input_grad: TensorRef<D>, output_grad: TensorRef<D>, op: UnaryOp) -> Self {
        Self {
            input,
            input_grad,
            output_grad,
            op,
            mid_x: Mutex::new(None),
            mid_up: Mutex::new(None),
            mid_g: Mutex::new(None),
            tmp_g32: Mutex::new(None),
        }
    }
}

fn ensure_mid_f16<D: Device>(
    slot: &mut Option<DenseMatrixF16<D>>,
    single_size: usize,
    batch_size: Option<usize>,
    device: std::sync::Arc<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let need = match slot.as_ref() {
        None => true,
        Some(m) => m.single_size() != single_size || m.batch_size() != batch_size,
    };
    if need {
        *slot = Some(DenseMatrixF16::zeroed(device, single_size, batch_size)?);
    }
    Ok(())
}

fn ensure_mid_f32<D: Device>(
    slot: &mut Option<crate::device::tensor::DenseMatrix<D>>,
    single_size: usize,
    batch_size: Option<usize>,
    device: std::sync::Arc<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let need = match slot.as_ref() {
        None => true,
        Some(m) => m.single_size() != single_size || m.batch_size() != batch_size,
    };
    if need {
        *slot = Some(crate::device::tensor::DenseMatrix::zeroed(device, single_size, batch_size)?);
    }
    Ok(())
}

impl<D: Device> DeviceOperation<D> for MixedUnaryBackward<D> {
    fn opname(&self) -> String {
        format!("MixedUnaryBackward({:?})", self.op)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let output_grad = self.output_grad.dense();
        let mut input_grad = self.input_grad.dense_mut();

        if input.batch_size() != output_grad.batch_size() || input.batch_size() != input_grad.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if input.single_size() != input_grad.single_size() || input.single_size() != output_grad.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        let size = input.size();
        let single_size = input.single_size();
        let batch_size = input.batch_size();
        let device = input.buf.device();

        let mut sx = self.mid_x.lock();
        let mut sup = self.mid_up.lock();
        let mut sg = self.mid_g.lock();
        let mut sg32 = self.tmp_g32.lock();
        ensure_mid_f16::<D>(&mut sx, single_size, batch_size, device.clone())?;
        ensure_mid_f16::<D>(&mut sup, single_size, batch_size, device.clone())?;
        ensure_mid_f16::<D>(&mut sg, single_size, batch_size, device.clone())?;
        ensure_mid_f32::<D>(&mut sg32, single_size, batch_size, device.clone())?;
        let mid_x = sx.as_mut().unwrap();
        let mid_up = sup.as_mut().unwrap();
        let mid_g = sg.as_mut().unwrap();
        let tmp_g32 = sg32.as_mut().unwrap();

        // cast f32 → f16 for forward input + upstream grad
        D::cast_f32_to_f16(size, &input.buf, &mut mid_x.buf)?;
        D::cast_f32_to_f16(size, &output_grad.buf, &mut mid_up.buf)?;

        // op derivative on f16 path
        match self.op {
            UnaryOp::Silu => {
                D::silu_grad_f16(size, &mid_x.buf, &mid_up.buf, &mut mid_g.buf)?;
            }
            _ => return Err(OperationError::UnsupportedOperation),
        }

        // cast f16 → f32 (tmp_g32)
        D::cast_f16_to_f32(size, &mid_g.buf, &mut tmp_g32.buf)?;

        // input_grad += tmp_g32 (accumulate)
        input_grad.buf.linear_comb(size, 1.0, 1.0, &tmp_g32.buf)?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct SoftmaxOp<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub feature_size: usize,
}

impl<D: Device> DeviceOperation<D> for SoftmaxOp<D> {
    fn opname(&self) -> String {
        "Softmax".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if input.single_size() != output.single_size() || input.single_size() != self.feature_size {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = input.batch_size().unwrap_or(1);
        output.buf.softmax_fwd(self.feature_size, bs, &input.buf)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct RmsNorm<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub feature_size: usize,
    pub eps: f32,
}

impl<D: Device> DeviceOperation<D> for RmsNorm<D> {
    fn opname(&self) -> String {
        "RmsNorm".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if input.single_size() != output.single_size() || input.single_size() != self.feature_size {
            return Err(OperationError::InvalidTensorFormat);
        }

        let bs = input.batch_size().unwrap_or(1);
        output.buf.rms_norm_fwd(self.feature_size, bs, self.eps, &input.buf)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct CopyOrAddStrided<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
    pub input_offset: usize,
    pub output_offset: usize,
    pub add: bool,
    pub len_is_out: bool,
}

impl<D: Device> DeviceOperation<D> for CopyOrAddStrided<D> {
    fn opname(&self) -> String {
        "CopyOrAddStrided".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        if input.batch_size() != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let output_size = output.single_size();
        let rows = if self.len_is_out { output_size } else { input.single_size() };

        output.buf.copy_or_add_strided(
            self.add,
            rows,
            input.batch_size().unwrap_or(1),
            self.output_offset,
            output_size,
            &input.buf,
            self.input_offset,
            input.single_size(),
        )?;

        Ok(())
    }
}

#[derive(Clone)]
pub struct Softmax<D: Device> {
    pub input: TensorRef<D>,
    pub output: TensorRef<D>,
}

impl<D: CoreDeviceOps> DeviceOperation<D> for Softmax<D> {
    fn opname(&self) -> String {
        "Softmax".to_string()
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let input = self.input.dense();
        let mut output = self.output.dense_mut();

        let batch_size = input.batch_size();
        let single_size = input.single_size();

        if batch_size != output.batch_size() {
            return Err(OperationError::MismatchedBatchSizes);
        }

        if single_size != output.single_size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        D::softmax_across_batch(batch_size.unwrap_or(1), single_size, &input.buf, &mut output.buf)
    }
}
