//! SEBlock (Squeeze-Excitation) mixed precision IR op (CRCL-v2 改善 A、 dlshogi 知見)。
//!
//! 構造:
//!   input (N×C×H×W、 f32 batched)
//!     → global_avg_pool      (N×C)
//!     → FC1 (C → C/r)         W1 (C/r × C)
//!     → ReLU
//!     → FC2 (C/r → C)         W2 (C × C/r)
//!     → Sigmoid               → s (N×C)
//!     → channel_wise_mul      output = input * s broadcast
//!
//! 内部の FC は f32 SGEMM (mixed precision の HGEMM ではない、 simple path)。
//! 中間 buffer は lazy alloc + Mutex で保持 (既存 conv_mixed と同 pattern)。
//!
//! backward は当面 placeholder (chain rule 6 step、 次段で本実装)。

use std::sync::Arc;

use acyclib::{
    dag::NodeId,
    device::{
        Device, DeviceBuffer, OperationError, OperationResult,
        function::{DeviceFunction, DeviceOperation},
        tensor::{DenseMatrix, Shape, TensorRef},
    },
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        },
    },
};
use cudarc::driver::{DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;

use crate::{CudaDevice, CudaMarker};

#[derive(Clone, Copy, Debug)]
pub struct SEBlockMixed {
    pub input: AnnotatedNode,
    pub w1: AnnotatedNode,
    pub w2: AnnotatedNode,
    pub channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub reduction: i32,
}

impl GraphIROperationBase<CudaMarker> for SEBlockMixed {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.w1, self.w2]
    }

    fn output_shape(&self, ir: &GraphIR<CudaMarker>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.w1, true)?;
        util::check_dense_eq(ir, &self.w2, true)?;
        util::check_not_batched(ir, &self.w1)?;
        util::check_not_batched(ir, &self.w2)?;
        util::check_has_grad(ir, &[&self.w1, &self.w2])?;

        let chw = (self.channels * self.spatial_h * self.spatial_w) as usize;
        if self.input.shape.size() != chw {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.input.shape)));
        }
        let c_r = self.channels / self.reduction;
        if self.w1.shape.size() != (c_r * self.channels) as usize {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.w1.shape)));
        }
        if self.w2.shape.size() != (self.channels * c_r) as usize {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.w2.shape)));
        }
        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        format!("SEBlockMixed(C={},H={},W={},r={})", self.channels, self.spatial_h, self.spatial_w, self.reduction)
    }
}

impl GraphIROperationCompilable<CudaMarker> for SEBlockMixed {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let w1 = graph.get_ref(self.w1.idx, GraphNodeIdTy::Values);
        let w2 = graph.get_ref(self.w2.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        func.push(acyclib::device::function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(SEBlockMixedForward {
            input,
            w1,
            w2,
            output,
            channels: self.channels,
            spatial_h: self.spatial_h,
            spatial_w: self.spatial_w,
            reduction: self.reduction,
            pool_buf: Mutex::new(None),
            fc1_buf: Mutex::new(None),
            fc2_buf: Mutex::new(None),
        });
        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let w1 = graph.get_ref(self.w1.idx, GraphNodeIdTy::Values);
        let w2 = graph.get_ref(self.w2.idx, GraphNodeIdTy::Values);
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
        let input_grad = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients);
        let w1_grad = graph.get_ref(self.w1.idx, GraphNodeIdTy::Gradients);
        let w2_grad = graph.get_ref(self.w2.idx, GraphNodeIdTy::Gradients);

        if let Some(ref ig) = input_grad {
            func.push(acyclib::device::function::MaybeUpdateBatchSize { input: input.clone(), output: ig.clone() });
        }

        func.push(SEBlockMixedBackward {
            input,
            w1,
            w2,
            output_grad,
            input_grad,
            w1_grad,
            w2_grad,
            channels: self.channels,
            spatial_h: self.spatial_h,
            spatial_w: self.spatial_w,
            reduction: self.reduction,
            pool_buf: Mutex::new(None),
            fc1_buf: Mutex::new(None),
            fc1_relu_buf: Mutex::new(None),
            fc2_buf: Mutex::new(None),
            s_buf: Mutex::new(None),
            ds_buf: Mutex::new(None),
            ds_sigmoid_buf: Mutex::new(None),
            dfc1_buf: Mutex::new(None),
            dfc1_relu_buf: Mutex::new(None),
            dpool_buf: Mutex::new(None),
            tmp_dx_via_mul: Mutex::new(None),
            tmp_dx_via_pool: Mutex::new(None),
        });
        func
    }
}

pub struct SEBlockMixedBackward {
    pub input: TensorRef<CudaDevice>,
    pub w1: TensorRef<CudaDevice>,
    pub w2: TensorRef<CudaDevice>,
    pub output_grad: TensorRef<CudaDevice>,
    pub input_grad: Option<TensorRef<CudaDevice>>,
    pub w1_grad: TensorRef<CudaDevice>,
    pub w2_grad: TensorRef<CudaDevice>,
    pub channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub reduction: i32,
    // forward redo 用 buffer (pool, fc1 pre-ReLU, fc1_relu, fc2 pre-sigmoid, s = sigmoid(fc2))
    pub pool_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub fc1_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub fc1_relu_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub fc2_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub s_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    // backward 用 buffer
    pub ds_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub ds_sigmoid_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub dfc1_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub dfc1_relu_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub dpool_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub tmp_dx_via_mul: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub tmp_dx_via_pool: Mutex<Option<DenseMatrix<CudaDevice>>>,
}

impl DeviceOperation<CudaDevice> for SEBlockMixedBackward {
    fn opname(&self) -> String {
        "SEBlockMixedBackward".to_string()
    }

    fn execute(&self) -> OperationResult<<CudaDevice as Device>::DeviceError> {
        use acyclib::device::operation::BaseOperations;
        use cudarc::cublas::{
            Gemm,
            sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
        };

        let input = self.input.dense();
        let w1 = self.w1.dense();
        let w2 = self.w2.dense();
        let output_grad = self.output_grad.dense();
        let batch = input.batch_size().ok_or(OperationError::MismatchedBatchSizes)?;
        let c = self.channels as usize;
        let h = self.spatial_h as usize;
        let w_dim = self.spatial_w as usize;
        let c_r = (self.channels / self.reduction) as usize;
        let nc = batch * c;
        let hw = h * w_dim;
        let n_input = nc * hw;
        let device = input.buf.device();

        // alloc 全 buffer (forward redo + backward intermediate)
        let mut pool_lk = self.pool_buf.lock();
        let mut fc1_lk = self.fc1_buf.lock();
        let mut fc1r_lk = self.fc1_relu_buf.lock();
        let mut fc2_lk = self.fc2_buf.lock();
        let mut s_lk = self.s_buf.lock();
        let mut ds_lk = self.ds_buf.lock();
        let mut dss_lk = self.ds_sigmoid_buf.lock();
        let mut dfc1_lk = self.dfc1_buf.lock();
        let mut dfc1r_lk = self.dfc1_relu_buf.lock();
        let mut dpool_lk = self.dpool_buf.lock();
        let mut dxm_lk = self.tmp_dx_via_mul.lock();
        let mut dxp_lk = self.tmp_dx_via_pool.lock();
        ensure_f32(&mut pool_lk, c, Some(batch), device.clone())?;
        ensure_f32(&mut fc1_lk, c_r, Some(batch), device.clone())?;
        ensure_f32(&mut fc1r_lk, c_r, Some(batch), device.clone())?;
        ensure_f32(&mut fc2_lk, c, Some(batch), device.clone())?;
        ensure_f32(&mut s_lk, c, Some(batch), device.clone())?;
        ensure_f32(&mut ds_lk, c, Some(batch), device.clone())?;
        ensure_f32(&mut dss_lk, c, Some(batch), device.clone())?;
        ensure_f32(&mut dfc1_lk, c_r, Some(batch), device.clone())?;
        ensure_f32(&mut dfc1r_lk, c_r, Some(batch), device.clone())?;
        ensure_f32(&mut dpool_lk, c, Some(batch), device.clone())?;
        ensure_f32(&mut dxm_lk, c * hw, Some(batch), device.clone())?;
        ensure_f32(&mut dxp_lk, c * hw, Some(batch), device.clone())?;
        let pool = pool_lk.as_mut().unwrap();
        let fc1 = fc1_lk.as_mut().unwrap();
        let fc1r = fc1r_lk.as_mut().unwrap();
        let fc2 = fc2_lk.as_mut().unwrap();
        let s = s_lk.as_mut().unwrap();
        let ds = ds_lk.as_mut().unwrap();
        let dss = dss_lk.as_mut().unwrap();
        let dfc1 = dfc1_lk.as_mut().unwrap();
        let dfc1r = dfc1r_lk.as_mut().unwrap();
        let dpool = dpool_lk.as_mut().unwrap();
        let dxm = dxm_lk.as_mut().unwrap();
        let dxp = dxp_lk.as_mut().unwrap();

        // ===== forward redo =====
        // step 1: pool
        {
            let func = device.module().load_function("global_avg_pool_f32").unwrap();
            let stream = device.stream();
            let tpb: u32 = 64;
            let cfg = LaunchConfig { grid_dim: (nc as u32, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: tpb * 4 };
            let (xp, _r1) = input.buf.buf.device_ptr(&stream);
            let (pp, _r2) = pool.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&(batch as i32))
                    .arg(&self.channels)
                    .arg(&(h as i32))
                    .arg(&(w_dim as i32))
                    .arg(&xp)
                    .arg(&pp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        // step 2: fc1 = W1 × pool
        {
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_N,
                m: c_r as i32,
                n: batch as i32,
                k: c as i32,
                alpha: 1.0,
                lda: c_r as i32,
                ldb: c as i32,
                beta: 0.0,
                ldc: c_r as i32,
            };
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &w1.buf.buf, &pool.buf.buf, &mut fc1.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }
        // step 3: fc1_relu = ReLU(fc1)
        {
            let func = device.module().load_function("ForwardReluKernel").unwrap();
            let stream = device.stream();
            let n = (batch * c_r) as i32;
            let cfg = CudaDevice::elementwise_launch_params_single(n as usize, 256);
            let (ip, _r1) = fc1.buf.buf.device_ptr(&stream);
            let (op, _r2) = fc1r.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n)
                    .arg(&ip)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        // step 4: fc2 = W2 × fc1_relu
        {
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_N,
                m: c as i32,
                n: batch as i32,
                k: c_r as i32,
                alpha: 1.0,
                lda: c as i32,
                ldb: c_r as i32,
                beta: 0.0,
                ldc: c as i32,
            };
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &w2.buf.buf, &fc1r.buf.buf, &mut fc2.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }
        // step 5: s = sigmoid(fc2)
        {
            let func = device.module().load_function("ForwardSigmoidKernel").unwrap();
            let stream = device.stream();
            let n = (batch * c) as i32;
            let cfg = CudaDevice::elementwise_launch_params_single(n as usize, 256);
            let (ip, _r1) = fc2.buf.buf.device_ptr(&stream);
            let (op, _r2) = s.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n)
                    .arg(&ip)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }

        // ===== backward chain rule =====
        // step 6: dx_via_mul = dy * s broadcast (channel_wise_mul_f32_backward_x)
        {
            let func = device.module().load_function("channel_wise_mul_f32_backward_x").unwrap();
            let stream = device.stream();
            let tpb: u32 = 256;
            let blocks = (n_input as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let (dp, _r1) = output_grad.buf.buf.device_ptr(&stream);
            let (sp, _r2) = s.buf.buf.device_ptr(&stream);
            let (op, _r3) = dxm.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&(batch as i32))
                    .arg(&self.channels)
                    .arg(&(h as i32))
                    .arg(&(w_dim as i32))
                    .arg(&dp)
                    .arg(&sp)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        // step 7: ds = sum_hw(dy * x) (channel_wise_mul_f32_backward_s、 atomic add)
        ds.buf.set_zero().map_err(|e| OperationError::DeviceError(Box::new(e)))?;
        {
            let func = device.module().load_function("channel_wise_mul_f32_backward_s").unwrap();
            let stream = device.stream();
            let tpb: u32 = 256;
            let blocks = (n_input as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let (dp, _r1) = output_grad.buf.buf.device_ptr(&stream);
            let (xp, _r2) = input.buf.buf.device_ptr(&stream);
            let (op, _r3) = ds.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&(batch as i32))
                    .arg(&self.channels)
                    .arg(&(h as i32))
                    .arg(&(w_dim as i32))
                    .arg(&dp)
                    .arg(&xp)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        // step 8: dss = primeSigmoid(fc2) * ds (BackwardSigmoidKernel)
        {
            let func = device.module().load_function("BackwardSigmoidKernel").unwrap();
            let stream = device.stream();
            let n = (batch * c) as i32;
            let cfg = CudaDevice::elementwise_launch_params_single(n as usize, 256);
            let (ip, _r1) = fc2.buf.buf.device_ptr(&stream);
            let (gp, _r2) = ds.buf.buf.device_ptr(&stream);
            let (op, _r3) = dss.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n)
                    .arg(&ip)
                    .arg(&gp)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        // step 9: dW2 = dss × fc1_relu^T  (C × N) × (N × C/r) = (C × C/r)、 weight_grad に accumulate
        {
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_T,
                m: c as i32,
                n: c_r as i32,
                k: batch as i32,
                alpha: 1.0,
                lda: c as i32,
                ldb: c_r as i32,
                beta: 1.0,
                ldc: c as i32,
            };
            let mut w2g = self.w2_grad.dense_mut();
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &dss.buf.buf, &fc1r.buf.buf, &mut w2g.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }
        // step 10: dfc1_relu = W2^T × dss  (C/r × C) × (C × N) = (C/r × N)
        {
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_T,
                transb: CUBLAS_OP_N,
                m: c_r as i32,
                n: batch as i32,
                k: c as i32,
                alpha: 1.0,
                lda: c as i32,
                ldb: c as i32,
                beta: 0.0,
                ldc: c_r as i32,
            };
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &w2.buf.buf, &dss.buf.buf, &mut dfc1r.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }
        // step 11: dfc1 = primeReLU(fc1) * dfc1_relu (BackwardReluKernel)
        {
            let func = device.module().load_function("BackwardReluKernel").unwrap();
            let stream = device.stream();
            let n = (batch * c_r) as i32;
            let cfg = CudaDevice::elementwise_launch_params_single(n as usize, 256);
            let (ip, _r1) = fc1.buf.buf.device_ptr(&stream);
            let (gp, _r2) = dfc1r.buf.buf.device_ptr(&stream);
            let (op, _r3) = dfc1.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n)
                    .arg(&ip)
                    .arg(&gp)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        // step 12: dW1 = dfc1 × pool^T  (C/r × N) × (N × C) = (C/r × C)、 weight_grad に accumulate
        {
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_T,
                m: c_r as i32,
                n: c as i32,
                k: batch as i32,
                alpha: 1.0,
                lda: c_r as i32,
                ldb: c as i32,
                beta: 1.0,
                ldc: c_r as i32,
            };
            let mut w1g = self.w1_grad.dense_mut();
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &dfc1.buf.buf, &pool.buf.buf, &mut w1g.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }
        // step 13: dpool = W1^T × dfc1  (C × C/r) × (C/r × N) = (C × N)
        {
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_T,
                transb: CUBLAS_OP_N,
                m: c as i32,
                n: batch as i32,
                k: c_r as i32,
                alpha: 1.0,
                lda: c_r as i32,
                ldb: c_r as i32,
                beta: 0.0,
                ldc: c as i32,
            };
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &w1.buf.buf, &dfc1.buf.buf, &mut dpool.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }
        // step 14: dx_via_pool = dpool / hw broadcast (global_avg_pool_f32_backward)
        if self.input_grad.is_some() {
            {
                let func = device.module().load_function("global_avg_pool_f32_backward").unwrap();
                let stream = device.stream();
                let tpb: u32 = 256;
                let blocks = (n_input as u32).div_ceil(tpb);
                let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
                let (gp, _r1) = dpool.buf.buf.device_ptr(&stream);
                let (op, _r2) = dxp.buf.buf.device_ptr_mut(&stream);
                unsafe {
                    stream
                        .launch_builder(&func)
                        .arg(&(batch as i32))
                        .arg(&self.channels)
                        .arg(&(h as i32))
                        .arg(&(w_dim as i32))
                        .arg(&gp)
                        .arg(&op)
                        .launch(cfg)
                        .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
                }
            }
            // step 15: input_grad += dxm + dxp
            let mut ig = self.input_grad.as_ref().unwrap().dense_mut();
            ig.buf.linear_comb(n_input, 1.0, 1.0, &dxm.buf)?;
            ig.buf.linear_comb(n_input, 1.0, 1.0, &dxp.buf)?;
        }

        Ok(())
    }
}

pub struct SEBlockMixedForward {
    pub input: TensorRef<CudaDevice>,  // f32 (N×C×H×W batched)
    pub w1: TensorRef<CudaDevice>,     // f32 (C/r × C not batched)
    pub w2: TensorRef<CudaDevice>,     // f32 (C × C/r not batched)
    pub output: TensorRef<CudaDevice>, // f32 (N×C×H×W batched)
    pub channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub reduction: i32,
    pub pool_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub fc1_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
    pub fc2_buf: Mutex<Option<DenseMatrix<CudaDevice>>>,
}

fn ensure_f32(
    slot: &mut Option<DenseMatrix<CudaDevice>>,
    single_size: usize,
    batch_size: Option<usize>,
    device: Arc<CudaDevice>,
) -> OperationResult<<CudaDevice as Device>::DeviceError> {
    let need = match slot.as_ref() {
        None => true,
        Some(m) => m.single_size() != single_size || m.batch_size() != batch_size,
    };
    if need {
        *slot = Some(DenseMatrix::zeroed(device, single_size, batch_size)?);
    }
    Ok(())
}

impl DeviceOperation<CudaDevice> for SEBlockMixedForward {
    fn opname(&self) -> String {
        "SEBlockMixedForward".to_string()
    }

    fn execute(&self) -> OperationResult<<CudaDevice as Device>::DeviceError> {
        let input = self.input.dense();
        let w1 = self.w1.dense();
        let w2 = self.w2.dense();
        let mut output = self.output.dense_mut();
        let batch = input.batch_size().ok_or(OperationError::MismatchedBatchSizes)?;
        let c = self.channels as usize;
        let h = self.spatial_h as usize;
        let w_dim = self.spatial_w as usize;
        let c_r = (self.channels / self.reduction) as usize;
        let nc = batch * c;
        let hw = h * w_dim;
        let device = input.buf.device();

        let mut pool_lock = self.pool_buf.lock();
        let mut fc1_lock = self.fc1_buf.lock();
        let mut fc2_lock = self.fc2_buf.lock();
        // pool: (C, N) col-major で gemm 入力にするため single_size=C, batch=N
        ensure_f32(&mut pool_lock, c, Some(batch), device.clone())?;
        ensure_f32(&mut fc1_lock, c_r, Some(batch), device.clone())?;
        ensure_f32(&mut fc2_lock, c, Some(batch), device.clone())?;
        let pool = pool_lock.as_mut().unwrap();
        let fc1 = fc1_lock.as_mut().unwrap();
        let fc2 = fc2_lock.as_mut().unwrap();

        // step 1: global_avg_pool
        {
            let func = device.module().load_function("global_avg_pool_f32").unwrap();
            let stream = device.stream();
            let tpb: u32 = 64;
            let cfg = LaunchConfig { grid_dim: (nc as u32, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: tpb * 4 };
            let (xp, _rx) = input.buf.buf.device_ptr(&stream);
            let (pp, _rp) = pool.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&(batch as i32))
                    .arg(&self.channels)
                    .arg(&(h as i32))
                    .arg(&(w_dim as i32))
                    .arg(&xp)
                    .arg(&pp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }

        // step 2: FC1 = W1 (C/r × C col-major) × pool (C × N col-major) → fc1 (C/r × N)
        {
            use cudarc::cublas::{Gemm, sys::cublasOperation_t::CUBLAS_OP_N};
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_N,
                m: c_r as i32,
                n: batch as i32,
                k: c as i32,
                alpha: 1.0f32,
                lda: c_r as i32,
                ldb: c as i32,
                beta: 0.0f32,
                ldc: c_r as i32,
            };
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &w1.buf.buf, &pool.buf.buf, &mut fc1.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }

        // step 3: ReLU on fc1 (既存 ForwardReluKernel、 tmp 経由で in-place 回避)
        let mut tmp_relu = DenseMatrix::<CudaDevice>::zeroed(device.clone(), c_r, Some(batch))?;
        {
            let func = device.module().load_function("ForwardReluKernel").unwrap();
            let stream = device.stream();
            let n_total = (batch * c_r) as i32;
            let cfg = CudaDevice::elementwise_launch_params_single(n_total as usize, 256);
            let (i_ptr, _ri) = fc1.buf.buf.device_ptr(&stream);
            let (o_ptr, _ro) = tmp_relu.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_total)
                    .arg(&i_ptr)
                    .arg(&o_ptr)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        fc1.buf.load_from_device(&tmp_relu.buf, batch * c_r).map_err(|e| OperationError::DeviceError(Box::new(e)))?;

        // step 4: FC2 = W2 (C × C/r) × fc1 (C/r × N) → fc2 (C × N)
        {
            use cudarc::cublas::{Gemm, sys::cublasOperation_t::CUBLAS_OP_N};
            let cfg = cudarc::cublas::GemmConfig {
                transa: CUBLAS_OP_N,
                transb: CUBLAS_OP_N,
                m: c as i32,
                n: batch as i32,
                k: c_r as i32,
                alpha: 1.0f32,
                lda: c as i32,
                ldb: c_r as i32,
                beta: 0.0f32,
                ldc: c as i32,
            };
            unsafe {
                device
                    .blas()
                    .gemm(cfg, &w2.buf.buf, &fc1.buf.buf, &mut fc2.buf.buf)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Blas(e))))?;
            }
        }

        // step 5: Sigmoid on fc2 (既存 ForwardSigmoidKernel、 tmp 経由)
        let mut tmp_sig = DenseMatrix::<CudaDevice>::zeroed(device.clone(), c, Some(batch))?;
        {
            let func = device.module().load_function("ForwardSigmoidKernel").unwrap();
            let stream = device.stream();
            let n_total = (batch * c) as i32;
            let cfg = CudaDevice::elementwise_launch_params_single(n_total as usize, 256);
            let (i_ptr, _ri) = fc2.buf.buf.device_ptr(&stream);
            let (o_ptr, _ro) = tmp_sig.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_total)
                    .arg(&i_ptr)
                    .arg(&o_ptr)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        fc2.buf.load_from_device(&tmp_sig.buf, batch * c).map_err(|e| OperationError::DeviceError(Box::new(e)))?;

        // step 6: channel_wise_mul (input * fc2 broadcast → output)
        {
            let func = device.module().load_function("channel_wise_mul_f32").unwrap();
            let stream = device.stream();
            let n_total = (batch * c * hw) as i32;
            let tpb: u32 = 256;
            let blocks = (n_total as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let (xp, _rx) = input.buf.buf.device_ptr(&stream);
            let (sp, _rs) = fc2.buf.buf.device_ptr(&stream);
            let (op, _ro) = output.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&(batch as i32))
                    .arg(&self.channels)
                    .arg(&(h as i32))
                    .arg(&(w_dim as i32))
                    .arg(&xp)
                    .arg(&sp)
                    .arg(&op)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }

        Ok(())
    }
}
