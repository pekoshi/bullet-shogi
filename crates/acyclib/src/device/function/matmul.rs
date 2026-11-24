use crate::device::{
    Device, DeviceBuffer, OperationError,
    function::DeviceOperation,
    operation::{BaseOperations, BlasOperations, GemmConfig},
    tensor::{DenseMatrix, DenseMatrixF16, Shape, TensorRef},
};
use parking_lot::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulType {
    BatBat,
    NobBat,
    NobNob,
    BatBatRed,
}

#[derive(Clone)]
pub struct Matmul<D: Device> {
    pub cfg: GemmConfig,
    pub input_a: TensorRef<D>,
    pub input_b: TensorRef<D>,
    pub output: TensorRef<D>,
    pub ty: MatmulType,
}

impl<D: Device> DeviceOperation<D> for Matmul<D> {
    fn opname(&self) -> String {
        format!("Matmul({:?})", self.ty)
    }

    fn execute(&self) -> Result<(), OperationError<<D as Device>::DeviceError>> {
        let Matmul { cfg, input_a, input_b, output, ty } = self;

        let input_a = input_a.dense();
        let input_b = input_b.dense();
        let mut output = output.dense_mut();

        if input_a.single_size() != cfg.shape_a.size() || input_b.single_size() != cfg.shape_b.size() {
            return Err(OperationError::InvalidTensorFormat);
        }

        match ty {
            MatmulType::BatBat => {
                let bs = input_a.batch_size();

                if bs != input_b.batch_size() || bs != output.batch_size() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                output.buf.gebmm(cfg, bs.unwrap_or(1), &input_a.buf, &input_b.buf)?;
            }
            MatmulType::NobNob => {
                if input_a.batch_size().is_some() || input_b.batch_size().is_some() || output.batch_size().is_some() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                output.buf.gemm(cfg, &input_a.buf, &input_b.buf)?;
            }
            MatmulType::NobBat => {
                if input_a.batch_size().is_some()
                    || input_b.batch_size().is_none()
                    || input_b.batch_size() != output.batch_size()
                {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                if cfg.trans_b {
                    return Err(OperationError::UnsupportedOperation);
                }

                let bs = input_b.batch_size().unwrap();

                let shape_b = Shape::new(cfg.shape_b.rows(), bs * cfg.shape_b.cols());
                let cfg = GemmConfig { shape_b, ..*cfg };
                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
            MatmulType::BatBatRed => {
                if input_a.batch_size() != input_b.batch_size() || output.batch_size().is_some() {
                    return Err(OperationError::MismatchedBatchSizes);
                }

                if !cfg.trans_b {
                    return Err(OperationError::UnsupportedOperation);
                }

                let bs = input_a.batch_size().unwrap_or(1);

                let cfg = GemmConfig {
                    shape_a: Shape::new(self.cfg.shape_a.rows(), bs * self.cfg.shape_a.cols()),
                    shape_b: Shape::new(self.cfg.shape_b.rows(), bs * self.cfg.shape_b.cols()),
                    ..self.cfg
                };

                output.buf.gemm(&cfg, &input_a.buf, &input_b.buf)?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Phase 2 graph IR 統合 G4: MixedMatmul (mixed precision matmul、 NobBat 専用)
// ============================================================================
//
// 既存 Matmul は f32 path (DenseMatrix → BlasOperations::gemm/gebmm)。
// MixedMatmul は IV案 = input/output f32、 内部 f16 cast + HGEMM (CUDA_R_16F + COMPUTE_32F)。
// scope 縮小: 当面 `MatmulType::NobBat` (input_a not-batched = weight、 input_b batched = activation) のみ対応。
// 他 type は UnsupportedOperation。

pub struct MixedMatmul<D: Device> {
    pub cfg: GemmConfig,
    pub input_a: TensorRef<D>, // f32 weight (not batched)
    pub input_b: TensorRef<D>, // f32 activation (batched)
    pub output: TensorRef<D>,  // f32 output (batched)
    pub ty: MatmulType,        // 当面 NobBat のみ accept
    pub mid_a: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_b: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_c: Mutex<Option<DenseMatrixF16<D>>>,
}

impl<D: Device> MixedMatmul<D> {
    pub fn new(
        cfg: GemmConfig,
        input_a: TensorRef<D>,
        input_b: TensorRef<D>,
        output: TensorRef<D>,
        ty: MatmulType,
    ) -> Self {
        Self {
            cfg,
            input_a,
            input_b,
            output,
            ty,
            mid_a: Mutex::new(None),
            mid_b: Mutex::new(None),
            mid_c: Mutex::new(None),
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

impl<D: Device> DeviceOperation<D> for MixedMatmul<D> {
    fn opname(&self) -> String {
        format!("MixedMatmul({:?})", self.ty)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        let MixedMatmul { cfg, input_a, input_b, output, ty, .. } = self;
        if *ty != MatmulType::NobBat {
            return Err(OperationError::UnsupportedOperation);
        }

        let input_a = input_a.dense();
        let input_b = input_b.dense();
        let mut output = output.dense_mut();

        if input_a.single_size() != cfg.shape_a.size() || input_b.single_size() != cfg.shape_b.size() {
            return Err(OperationError::InvalidTensorFormat);
        }
        if input_a.batch_size().is_some()
            || input_b.batch_size().is_none()
            || input_b.batch_size() != output.batch_size()
        {
            return Err(OperationError::MismatchedBatchSizes);
        }
        if cfg.trans_b {
            return Err(OperationError::UnsupportedOperation);
        }

        let bs = input_b.batch_size().unwrap();
        let m_dim = if cfg.trans_a { cfg.shape_a.cols() } else { cfg.shape_a.rows() };
        let k_dim = if cfg.trans_a { cfg.shape_a.rows() } else { cfg.shape_a.cols() };
        let n_dim = bs * cfg.shape_b.cols();
        let lda = cfg.shape_a.rows();
        let ldb = cfg.shape_b.rows();
        let ldc = m_dim;

        let device = input_a.buf.device();

        // 中間 f16 buffer alloc
        let mut sa = self.mid_a.lock();
        let mut sb = self.mid_b.lock();
        let mut sc = self.mid_c.lock();
        ensure_mid_f16::<D>(&mut sa, input_a.size(), None, device.clone())?;
        ensure_mid_f16::<D>(&mut sb, input_b.single_size(), Some(bs), device.clone())?;
        ensure_mid_f16::<D>(&mut sc, output.single_size(), Some(bs), device.clone())?;
        let mid_a = sa.as_mut().unwrap();
        let mid_b = sb.as_mut().unwrap();
        let mid_c = sc.as_mut().unwrap();

        // f32 → f16 cast (input_a, input_b)
        D::cast_f32_to_f16(input_a.size(), &input_a.buf, &mut mid_a.buf)?;
        D::cast_f32_to_f16(input_b.size(), &input_b.buf, &mut mid_b.buf)?;

        // HGEMM (output_f16 = alpha * op(A) * op(B) + beta * 0)
        D::hgemm_f16(
            m_dim,
            n_dim,
            k_dim,
            cfg.alpha,
            &mid_a.buf,
            cfg.trans_a,
            lda,
            &mid_b.buf,
            false,
            ldb,
            0.0,
            &mut mid_c.buf,
            ldc,
        )?;

        // f16 → f32 cast (output)
        D::cast_f16_to_f32(output.size(), &mid_c.buf, &mut output.buf)?;

        Ok(())
    }
}

// Backward は 2 つの matmul + accumulate。 当面 forward のみで smoke 通す。
// `MixedMatmulBackward` は後続 turn で実装 (chain rule:
//   dA = dC × B^T  (NobNob か NobBatRed)
//   dB = A^T × dC  (NobBat か他 type、 cfg は forward と異なる)
// + tmp_f32 経由 accumulate)。

// MixedMatmulBackward (NobBat 専用、 chain rule 本実装):
//
// forward:   Y = A × B   (A: m×k not batched, B: k×batch batched, Y: m×batch batched)
// backward:
//   dA = dY × B^T  (NobBat→Not-batched: sum across batch)  - shape (m, k)
//   dB = A^T × dY  (NobBat)                                  - shape (k, batch)
//
// 入出力 grad は f32、 内部で f16 cast 経由 HGEMM、 tmp_f32 経由 accumulate。
pub struct MixedMatmulBackward<D: Device> {
    pub input_a: TensorRef<D>,              // forward A (f32, m×k not batched)
    pub input_b: TensorRef<D>,              // forward B (f32, k×batch batched)
    pub output_grad: TensorRef<D>,          // dY (f32, m×batch batched)
    pub input_a_grad: Option<TensorRef<D>>, // optional dA (f32, m×k not batched)
    pub input_b_grad: Option<TensorRef<D>>, // optional dB (f32, k×batch batched)
    pub cfg: GemmConfig,
    pub ty: MatmulType,
    // 中間 f16 buffer (forward 入力 / 上流 grad / backward 結果)
    pub mid_a: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_b: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_dy: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_da: Mutex<Option<DenseMatrixF16<D>>>,
    pub mid_db: Mutex<Option<DenseMatrixF16<D>>>,
    pub tmp_da32: Mutex<Option<DenseMatrix<D>>>,
    pub tmp_db32: Mutex<Option<DenseMatrix<D>>>,
}

impl<D: Device> MixedMatmulBackward<D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_a: TensorRef<D>,
        input_b: TensorRef<D>,
        output_grad: TensorRef<D>,
        input_a_grad: Option<TensorRef<D>>,
        input_b_grad: Option<TensorRef<D>>,
        cfg: GemmConfig,
        ty: MatmulType,
    ) -> Self {
        Self {
            input_a,
            input_b,
            output_grad,
            input_a_grad,
            input_b_grad,
            cfg,
            ty,
            mid_a: Mutex::new(None),
            mid_b: Mutex::new(None),
            mid_dy: Mutex::new(None),
            mid_da: Mutex::new(None),
            mid_db: Mutex::new(None),
            tmp_da32: Mutex::new(None),
            tmp_db32: Mutex::new(None),
        }
    }
}

impl<D: Device> DeviceOperation<D> for MixedMatmulBackward<D> {
    fn opname(&self) -> String {
        format!("MixedMatmulBackward({:?})", self.ty)
    }

    fn execute(&self) -> Result<(), OperationError<D::DeviceError>> {
        if self.ty != MatmulType::NobBat {
            return Err(OperationError::UnsupportedOperation);
        }
        if self.cfg.trans_a || self.cfg.trans_b {
            // 当面 simple form (no transpose) のみ
            return Err(OperationError::UnsupportedOperation);
        }

        let input_a = self.input_a.dense(); // (m, k) not batched
        let input_b = self.input_b.dense(); // (k, batch) batched
        let output_grad = self.output_grad.dense(); // (m, batch) batched

        let m_dim = input_a.single_size() / self.cfg.shape_a.cols();
        let k_dim = self.cfg.shape_a.cols();
        if m_dim * k_dim != input_a.size() {
            return Err(OperationError::InvalidTensorFormat);
        }
        let batch = input_b.batch_size().ok_or(OperationError::MismatchedBatchSizes)?;
        if output_grad.batch_size() != Some(batch) {
            return Err(OperationError::MismatchedBatchSizes);
        }

        let device = input_a.buf.device();

        // 中間 f16 buffer alloc
        let mut sa = self.mid_a.lock();
        let mut sb = self.mid_b.lock();
        let mut sdy = self.mid_dy.lock();
        let mut sda = self.mid_da.lock();
        let mut sdb = self.mid_db.lock();
        let mut sda32 = self.tmp_da32.lock();
        let mut sdb32 = self.tmp_db32.lock();
        ensure_mid_f16::<D>(&mut sa, input_a.size(), None, device.clone())?;
        ensure_mid_f16::<D>(&mut sb, input_b.single_size(), Some(batch), device.clone())?;
        ensure_mid_f16::<D>(&mut sdy, output_grad.single_size(), Some(batch), device.clone())?;
        ensure_mid_f16::<D>(&mut sda, input_a.size(), None, device.clone())?;
        ensure_mid_f16::<D>(&mut sdb, input_b.single_size(), Some(batch), device.clone())?;
        ensure_tmp_f32::<D>(&mut sda32, m_dim * k_dim, None, device.clone())?;
        ensure_tmp_f32::<D>(&mut sdb32, input_b.single_size(), Some(batch), device.clone())?;
        let mid_a = sa.as_mut().unwrap();
        let mid_b = sb.as_mut().unwrap();
        let mid_dy = sdy.as_mut().unwrap();
        let mid_da = sda.as_mut().unwrap();
        let mid_db = sdb.as_mut().unwrap();
        let tmp_da32 = sda32.as_mut().unwrap();
        let tmp_db32 = sdb32.as_mut().unwrap();

        // cast f32 → f16 (forward input + upstream grad)
        D::cast_f32_to_f16(input_a.size(), &input_a.buf, &mut mid_a.buf)?;
        D::cast_f32_to_f16(input_b.size(), &input_b.buf, &mut mid_b.buf)?;
        D::cast_f32_to_f16(output_grad.size(), &output_grad.buf, &mut mid_dy.buf)?;

        // dA = dY × B^T   (m × batch) × (batch × k) = (m × k)
        //   column-major: dY (m, batch)、 B (k, batch)、 dA (m, k)
        //   trans_a=false (m, batch)、 trans_b=true (batch, k)
        if self.input_a_grad.is_some() {
            D::hgemm_f16(
                m_dim,
                k_dim,
                batch,
                1.0,
                &mid_dy.buf,
                false,
                m_dim, // dY (m × batch)
                &mid_b.buf,
                true,
                k_dim, // B^T (column-major B (k, batch) を trans で (batch, k))
                0.0,
                &mut mid_da.buf,
                m_dim,
            )?;
            // f16 → f32 cast
            D::cast_f16_to_f32(m_dim * k_dim, &mid_da.buf, &mut tmp_da32.buf)?;
            // input_a_grad += tmp_da32
            let mut da32 = self.input_a_grad.as_ref().unwrap().dense_mut();
            da32.buf.linear_comb(m_dim * k_dim, 1.0, 1.0, &tmp_da32.buf)?;
        }

        // dB = A^T × dY   (k × m) × (m × batch) = (k × batch)
        //   trans_a=true (m, k) → (k, m)、 trans_b=false (m, batch)
        if self.input_b_grad.is_some() {
            D::hgemm_f16(
                k_dim,
                batch,
                m_dim,
                1.0,
                &mid_a.buf,
                true,
                m_dim, // A^T
                &mid_dy.buf,
                false,
                m_dim, // dY
                0.0,
                &mut mid_db.buf,
                k_dim,
            )?;
            D::cast_f16_to_f32(k_dim * batch, &mid_db.buf, &mut tmp_db32.buf)?;
            let mut db32 = self.input_b_grad.as_ref().unwrap().dense_mut();
            db32.buf.linear_comb(k_dim * batch, 1.0, 1.0, &tmp_db32.buf)?;
        }

        Ok(())
    }
}

fn ensure_tmp_f32<D: Device>(
    slot: &mut Option<DenseMatrix<D>>,
    single_size: usize,
    batch_size: Option<usize>,
    device: std::sync::Arc<D>,
) -> Result<(), OperationError<D::DeviceError>> {
    let need = match slot.as_ref() {
        None => true,
        Some(m) => m.single_size() != single_size || m.batch_size() != batch_size,
    };
    if need {
        *slot = Some(DenseMatrix::zeroed(device, single_size, batch_size)?);
    }
    Ok(())
}
