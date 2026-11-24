//! Conv2d mixed precision IR op (Phase 2 IV案、 CRCL-v2 ResNet 用)。
//!
//! 既存 `Conv2d` (NVRTC direct conv kernel) と並列で、 mixed precision (im2col_f16 + HGEMM
//! Tensor Core + permute/col2im) 経由の forward + backward を持つ。
//!
//! - 入出力 Tensor: f32 (DenseMatrix、 既存 graph IR の中間 storage と互換)
//! - 内部: DenseMatrixF16 ×複数 を lazy alloc、 cast 経由
//! - forward: im2col_f16 + cast(W) + HGEMM + permute → output f32
//! - backward: permute(dY) + cast → 2 HGEMM (dW, dX_col) + cast → col2im → dX f32

use std::sync::Arc;

use acyclib::{
    dag::NodeId,
    device::{
        Device, DeviceBuffer, OperationError, OperationResult,
        function::{DeviceFunction, DeviceOperation},
        operation::BaseOperations,
        tensor::{DenseMatrix, DenseMatrixF16, Shape, TensorRef},
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
pub struct Conv2dF16Mixed {
    pub input: AnnotatedNode,
    pub weight: AnnotatedNode,
    pub in_channels: i32,
    pub out_channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub kernel_size: i32,
    pub padding: i32,
    pub stride: i32,
}

impl Conv2dF16Mixed {
    pub fn out_h(&self) -> i32 {
        (self.spatial_h + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
    pub fn out_w(&self) -> i32 {
        (self.spatial_w + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
}

impl GraphIROperationBase<CudaMarker> for Conv2dF16Mixed {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input, self.weight]
    }
    fn output_shape(&self, ir: &GraphIR<CudaMarker>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        util::check_dense_eq(ir, &self.weight, true)?;
        util::check_not_batched(ir, &self.weight)?;
        util::check_has_grad(ir, &[&self.weight])?;
        let expected_input = (self.in_channels * self.spatial_h * self.spatial_w) as usize;
        if self.input.shape.size() != expected_input {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.input.shape)));
        }
        let expected_weight = (self.out_channels * self.in_channels * self.kernel_size * self.kernel_size) as usize;
        if self.weight.shape.size() != expected_weight {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.weight.shape)));
        }
        let out_size = (self.out_channels * self.out_h() * self.out_w()) as usize;
        Ok(Shape::new(out_size, 1))
    }
    fn shorthand(&self) -> String {
        format!(
            "Conv2dF16Mixed(Cin={},Cout={},H={},W={},K={},Pad={},Stride={})",
            self.in_channels,
            self.out_channels,
            self.spatial_h,
            self.spatial_w,
            self.kernel_size,
            self.padding,
            self.stride,
        )
    }
}

impl GraphIROperationCompilable<CudaMarker> for Conv2dF16Mixed {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let weight = graph.get_ref(self.weight.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        func.push(acyclib::device::function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(MixedConv2dForward {
            input,
            weight,
            output,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            spatial_h: self.spatial_h,
            spatial_w: self.spatial_w,
            kernel_size: self.kernel_size,
            padding: self.padding,
            stride: self.stride,
            col_f16: Mutex::new(None),
            w_f16: Mutex::new(None),
            gemm_out_f16: Mutex::new(None),
        });
        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let weight = graph.get_ref(self.weight.idx, GraphNodeIdTy::Values);
        let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
        let input_grad = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients);
        let weight_grad = graph.get_ref(self.weight.idx, GraphNodeIdTy::Gradients);
        if let Some(ref ig) = input_grad {
            func.push(acyclib::device::function::MaybeUpdateBatchSize { input: input.clone(), output: ig.clone() });
        }
        func.push(MixedConv2dBackward {
            input,
            weight,
            output_grad,
            input_grad,
            weight_grad,
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            spatial_h: self.spatial_h,
            spatial_w: self.spatial_w,
            kernel_size: self.kernel_size,
            padding: self.padding,
            stride: self.stride,
            col_f16: Mutex::new(None),
            w_f16: Mutex::new(None),
            gemm_in_f16: Mutex::new(None),
            dw_f16: Mutex::new(None),
            dx_col_f16: Mutex::new(None),
            tmp_dw_f32: Mutex::new(None),
        });
        func
    }
}

// ============================================================================
// MixedConv2dForward DeviceOperation
// ============================================================================

pub struct MixedConv2dForward {
    pub input: TensorRef<CudaDevice>,
    pub weight: TensorRef<CudaDevice>,
    pub output: TensorRef<CudaDevice>,
    pub in_channels: i32,
    pub out_channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub kernel_size: i32,
    pub padding: i32,
    pub stride: i32,
    pub col_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub w_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub gemm_out_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
}

fn out_hw(spatial: i32, k: i32, pad: i32, stride: i32) -> i32 {
    (spatial + 2 * pad - k) / stride + 1
}

fn ensure_f16(
    slot: &mut Option<DenseMatrixF16<CudaDevice>>,
    single_size: usize,
    batch_size: Option<usize>,
    device: Arc<CudaDevice>,
) -> OperationResult<<CudaDevice as Device>::DeviceError> {
    let need = match slot.as_ref() {
        None => true,
        Some(m) => m.single_size() != single_size || m.batch_size() != batch_size,
    };
    if need {
        *slot = Some(DenseMatrixF16::zeroed(device, single_size, batch_size)?);
    }
    Ok(())
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

impl DeviceOperation<CudaDevice> for MixedConv2dForward {
    fn opname(&self) -> String {
        "MixedConv2dForward".to_string()
    }

    fn execute(&self) -> OperationResult<<CudaDevice as Device>::DeviceError> {
        let input = self.input.dense();
        let weight = self.weight.dense();
        let mut output = self.output.dense_mut();
        let batch = input.batch_size().ok_or(OperationError::MismatchedBatchSizes)?;
        let hout = out_hw(self.spatial_h, self.kernel_size, self.padding, self.stride);
        let wout = out_hw(self.spatial_w, self.kernel_size, self.padding, self.stride);
        let cin_kk = (self.in_channels * self.kernel_size * self.kernel_size) as usize;
        let nhw = (batch as i32 * hout * wout) as usize;
        let cout = self.out_channels as usize;
        let weight_size = weight.size();
        let device = input.buf.device();

        let mut cs = self.col_f16.lock();
        let mut ws = self.w_f16.lock();
        let mut gs = self.gemm_out_f16.lock();
        ensure_f16(&mut cs, nhw * cin_kk, None, device.clone())?;
        ensure_f16(&mut ws, weight_size, None, device.clone())?;
        ensure_f16(&mut gs, nhw * cout, None, device.clone())?;
        let col = cs.as_mut().unwrap();
        let wf16 = ws.as_mut().unwrap();
        let gout = gs.as_mut().unwrap();

        // step 1: im2col_f16
        {
            let func = device.module().load_function("im2col_f16_nchw").unwrap();
            let stream = device.stream();
            let tpb: u32 = 64;
            let grid_y = (cin_kk as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (nhw as u32, grid_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let n_batch = batch as i32;
            let (xp, _rx) = input.buf.buf.device_ptr(&stream);
            let (cp, _rc) = col.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_batch)
                    .arg(&self.in_channels)
                    .arg(&self.spatial_h)
                    .arg(&self.spatial_w)
                    .arg(&hout)
                    .arg(&wout)
                    .arg(&self.kernel_size)
                    .arg(&self.padding)
                    .arg(&self.stride)
                    .arg(&xp)
                    .arg(&cp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }

        // step 2: weight cast f32 → f16
        CudaDevice::cast_f32_to_f16(weight_size, &weight.buf, &mut wf16.buf)?;

        // step 3: HGEMM out (Cout × NHW col-major) = W^T (Cout × CinKK col-major) × col (CinKK × NHW col-major)
        CudaDevice::hgemm_f16(
            cout,
            nhw,
            cin_kk,
            1.0,
            &wf16.buf,
            true,
            cin_kk,
            &col.buf,
            false,
            cin_kk,
            0.0,
            &mut gout.buf,
            cout,
        )?;

        // step 4: permute f16 → f32 NCHW output
        {
            let func = device.module().load_function("permute_gemm_out_f16_to_f32_nchw").unwrap();
            let stream = device.stream();
            let block_x: u32 = 16;
            let block_y: u32 = 16;
            let cfg = LaunchConfig {
                grid_dim: ((cout as u32).div_ceil(block_x), (nhw as u32).div_ceil(block_y), 1),
                block_dim: (block_x, block_y, 1),
                shared_mem_bytes: 0,
            };
            let n_batch = batch as i32;
            let (gp, _rg) = gout.buf.buf.device_ptr(&stream);
            let (yp, _ry) = output.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_batch)
                    .arg(&self.out_channels)
                    .arg(&hout)
                    .arg(&wout)
                    .arg(&gp)
                    .arg(&yp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        Ok(())
    }
}

// ============================================================================
// MixedConv2dBackward DeviceOperation
// ============================================================================

pub struct MixedConv2dBackward {
    pub input: TensorRef<CudaDevice>,
    pub weight: TensorRef<CudaDevice>,
    pub output_grad: TensorRef<CudaDevice>,
    pub input_grad: Option<TensorRef<CudaDevice>>,
    pub weight_grad: TensorRef<CudaDevice>,
    pub in_channels: i32,
    pub out_channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub kernel_size: i32,
    pub padding: i32,
    pub stride: i32,
    pub col_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub w_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub gemm_in_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub dw_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub dx_col_f16: Mutex<Option<DenseMatrixF16<CudaDevice>>>,
    pub tmp_dw_f32: Mutex<Option<DenseMatrix<CudaDevice>>>,
}

impl DeviceOperation<CudaDevice> for MixedConv2dBackward {
    fn opname(&self) -> String {
        "MixedConv2dBackward".to_string()
    }

    fn execute(&self) -> OperationResult<<CudaDevice as Device>::DeviceError> {
        let input = self.input.dense();
        let weight = self.weight.dense();
        let output_grad = self.output_grad.dense();
        let batch = input.batch_size().ok_or(OperationError::MismatchedBatchSizes)?;
        let hout = out_hw(self.spatial_h, self.kernel_size, self.padding, self.stride);
        let wout = out_hw(self.spatial_w, self.kernel_size, self.padding, self.stride);
        let cin_kk = (self.in_channels * self.kernel_size * self.kernel_size) as usize;
        let nhw = (batch as i32 * hout * wout) as usize;
        let cout = self.out_channels as usize;
        let weight_size = weight.size();
        let device = input.buf.device();

        let mut cs = self.col_f16.lock();
        let mut ws = self.w_f16.lock();
        let mut gs = self.gemm_in_f16.lock();
        let mut dws = self.dw_f16.lock();
        let mut dxs = self.dx_col_f16.lock();
        let mut dws32 = self.tmp_dw_f32.lock();
        ensure_f16(&mut cs, nhw * cin_kk, None, device.clone())?;
        ensure_f16(&mut ws, weight_size, None, device.clone())?;
        ensure_f16(&mut gs, nhw * cout, None, device.clone())?;
        ensure_f16(&mut dws, weight_size, None, device.clone())?;
        ensure_f16(&mut dxs, nhw * cin_kk, None, device.clone())?;
        ensure_f32(&mut dws32, weight_size, None, device.clone())?;
        let col = cs.as_mut().unwrap();
        let wf16 = ws.as_mut().unwrap();
        let gin = gs.as_mut().unwrap();
        let dwf16 = dws.as_mut().unwrap();
        let dxcol = dxs.as_mut().unwrap();
        let tmp_dw = dws32.as_mut().unwrap();

        // step 1: forward 同様 im2col_f16 (X → col) + cast W
        {
            let func = device.module().load_function("im2col_f16_nchw").unwrap();
            let stream = device.stream();
            let tpb: u32 = 64;
            let grid_y = (cin_kk as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (nhw as u32, grid_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let n_batch = batch as i32;
            let (xp, _rx) = input.buf.buf.device_ptr(&stream);
            let (cp, _rc) = col.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_batch)
                    .arg(&self.in_channels)
                    .arg(&self.spatial_h)
                    .arg(&self.spatial_w)
                    .arg(&hout)
                    .arg(&wout)
                    .arg(&self.kernel_size)
                    .arg(&self.padding)
                    .arg(&self.stride)
                    .arg(&xp)
                    .arg(&cp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        CudaDevice::cast_f32_to_f16(weight_size, &weight.buf, &mut wf16.buf)?;

        // step 2: dY (NCHW f32) → gemm_in (Cout × NHW col-major) f16
        {
            let func = device.module().load_function("permute_nchw_f32_to_gemm_in_f16").unwrap();
            let stream = device.stream();
            let cfg = LaunchConfig {
                grid_dim: ((cout as u32).div_ceil(16), (nhw as u32).div_ceil(16), 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let n_batch = batch as i32;
            let (dp, _rd) = output_grad.buf.buf.device_ptr(&stream);
            let (gp, _rg) = gin.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_batch)
                    .arg(&self.out_channels)
                    .arg(&hout)
                    .arg(&wout)
                    .arg(&dp)
                    .arg(&gp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }

        // step 3: dW = col × gemm_in^T  (CinKK × NHW × NHW × Cout = CinKK × Cout col-major)
        CudaDevice::hgemm_f16(
            cin_kk,
            cout,
            nhw,
            1.0,
            &col.buf,
            false,
            cin_kk,
            &gin.buf,
            true,
            cout,
            0.0,
            &mut dwf16.buf,
            cin_kk,
        )?;
        // cast f16 → f32 → weight_grad に accumulate (linear_comb 1, 1)
        CudaDevice::cast_f16_to_f32(weight_size, &dwf16.buf, &mut tmp_dw.buf)?;
        {
            let mut wg = self.weight_grad.dense_mut();
            wg.buf.linear_comb(weight_size, 1.0, 1.0, &tmp_dw.buf)?;
        }

        // step 4 (input_grad あれば): dX_col = W × gemm_in  (CinKK × Cout × Cout × NHW)
        if let Some(ig_ref) = self.input_grad.as_ref() {
            CudaDevice::hgemm_f16(
                cin_kk,
                nhw,
                cout,
                1.0,
                &wf16.buf,
                false,
                cin_kk,
                &gin.buf,
                false,
                cout,
                0.0,
                &mut dxcol.buf,
                cin_kk,
            )?;
            // step 5: col2im (dX_col → input_grad f32 NCHW)
            // input_grad は zero_grads 後 (= 0) 想定で atomic add で accumulate。
            let func = device.module().load_function("col2im_f16_to_f32_nchw").unwrap();
            let stream = device.stream();
            let tpb: u32 = 64;
            let grid_y = (cin_kk as u32).div_ceil(tpb);
            let cfg = LaunchConfig { grid_dim: (nhw as u32, grid_y, 1), block_dim: (tpb, 1, 1), shared_mem_bytes: 0 };
            let n_batch = batch as i32;
            let mut ig_mut = ig_ref.dense_mut();
            let (cp, _rc) = dxcol.buf.buf.device_ptr(&stream);
            let (xp, _rx) = ig_mut.buf.buf.device_ptr_mut(&stream);
            unsafe {
                stream
                    .launch_builder(&func)
                    .arg(&n_batch)
                    .arg(&self.in_channels)
                    .arg(&self.spatial_h)
                    .arg(&self.spatial_w)
                    .arg(&hout)
                    .arg(&wout)
                    .arg(&self.kernel_size)
                    .arg(&self.padding)
                    .arg(&self.stride)
                    .arg(&cp)
                    .arg(&xp)
                    .launch(cfg)
                    .map_err(|e| OperationError::DeviceError(Box::new(crate::CudaError::Driver(e))))?;
            }
        }
        Ok(())
    }
}
