//! Conv 2D 3x3 (NCHW) Operation
//!
//! CRCL-v2 用。 1050 Ti (sm_61) で cuDNN 9 が EXECUTION_FAILED するため、 自前 NVRTC kernel
//! で実装する。 Conv2dForwardNCHW、 backward_data、 backward_filter の 3 kernel を持つ。
//!
//! - 入力 input: shape (Cin * H * W, 1)、 batched (= batch_size = N)
//! - 入力 weight: shape (Cout * Cin * K * K, 1)、 not batched、 requires_grad
//! - 出力: shape (Cout * Hout * Wout, 1)、 batched
//!
//! Hout = (H + 2*Pad - K) / Stride + 1、 Wout 同様
use acyclib::{
    dag::NodeId,
    device::{function::DeviceFunction, tensor::Shape},
    graph::{
        Graph, GraphNodeIdTy,
        ir::{
            GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        },
    },
};

use crate::{
    CudaDevice, CudaMarker,
    kernel::{Expr, Kernel, KernelArgs, KernelInput},
};

#[derive(Clone, Copy, Debug)]
pub struct Conv2d {
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

impl Conv2d {
    pub fn out_h(&self) -> i32 {
        (self.spatial_h + 2 * self.padding - self.kernel_size) / self.stride + 1
    }

    pub fn out_w(&self) -> i32 {
        (self.spatial_w + 2 * self.padding - self.kernel_size) / self.stride + 1
    }

    fn substitute(&self, code: &str) -> String {
        code.lines()
            .skip(11)
            .map(|x| format!("{x}\n"))
            .collect::<String>()
            .replace("DECL_CIN", &self.in_channels.to_string())
            .replace("DECL_COUT", &self.out_channels.to_string())
            .replace("DECL_HOUT", &self.out_h().to_string())
            .replace("DECL_WOUT", &self.out_w().to_string())
            .replace("DECL_H", &self.spatial_h.to_string())
            .replace("DECL_W", &self.spatial_w.to_string())
            .replace("DECL_K", &self.kernel_size.to_string())
            .replace("DECL_PAD", &self.padding.to_string())
            .replace("DECL_STRIDE", &self.stride.to_string())
    }
}

impl GraphIROperationBase<CudaMarker> for Conv2d {
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
        if self.kernel_size <= 0 || self.padding < 0 || self.stride <= 0 {
            return Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(self.weight.shape)));
        }

        let out_size = (self.out_channels * self.out_h() * self.out_w()) as usize;
        Ok(Shape::new(out_size, 1))
    }

    fn shorthand(&self) -> String {
        format!(
            "Conv2d(Cin={},Cout={},H={},W={},K={},Pad={},Stride={})",
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

const BLOCK_X: i32 = 16;
const BLOCK_Y: i32 = 16;

fn ceil_div(a: i32, b: i32) -> i32 {
    (a + b - 1) / b
}

impl GraphIROperationCompilable<CudaMarker> for Conv2d {
    fn forward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();

        let input_ref = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let weight_ref = graph.get_ref(self.weight.idx, GraphNodeIdTy::Values);
        let output_ref = graph.get_ref(output_node, GraphNodeIdTy::Values);

        // output buffer の batch_size を input に合わせる
        func.push(acyclib::device::function::MaybeUpdateBatchSize {
            input: input_ref.clone(),
            output: output_ref.clone(),
        });

        let out_size = (self.out_channels * self.out_h() * self.out_w()) as usize;
        let inputs = vec![
            KernelInput::Size(Expr::Var), // N (batch_size)
            KernelInput::Slice {
                slice: input_ref,
                layout: None,
                mutable: false,
                batched: true,
                shape: self.input.shape,
            },
            KernelInput::Slice {
                slice: weight_ref,
                layout: None,
                mutable: false,
                batched: false,
                shape: self.weight.shape,
            },
            KernelInput::Slice {
                slice: output_ref,
                layout: None,
                mutable: true,
                batched: true,
                shape: Shape::new(out_size, 1),
            },
        ];

        let grid_dim = [
            Expr::Const(ceil_div(self.out_w(), BLOCK_X)),
            Expr::Const(ceil_div(self.out_h(), BLOCK_Y)),
            Expr::Var * self.out_channels,
        ];
        let block_dim = [Expr::Const(BLOCK_X), Expr::Const(BLOCK_Y), Expr::Const(1)];

        let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes: Expr::Const(0) };
        let code = self.substitute(include_str!("conv/fwd.cu"));
        let kernel = unsafe { Kernel::new("Conv2dForward".to_string(), code, args) }.unwrap();
        func.push(kernel);

        func
    }

    fn backward_pass(&self, graph: &Graph<CudaDevice>, output_node: NodeId) -> DeviceFunction<CudaDevice> {
        let mut func = DeviceFunction::default();

        let input_ref = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let weight_ref = graph.get_ref(self.weight.idx, GraphNodeIdTy::Values);
        let output_grad_ref = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

        let weight_grad_ref = graph.get_ref(self.weight.idx, GraphNodeIdTy::Gradients);

        // backward_data: 入力に grad があるときのみ実行
        if let Some(input_grad_ref) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            func.push(acyclib::device::function::MaybeUpdateBatchSize {
                input: input_ref.clone(),
                output: input_grad_ref.clone(),
            });

            let inputs = vec![
                KernelInput::Size(Expr::Var),
                KernelInput::Slice {
                    slice: output_grad_ref.clone(),
                    layout: None,
                    mutable: false,
                    batched: true,
                    shape: Shape::new((self.out_channels * self.out_h() * self.out_w()) as usize, 1),
                },
                KernelInput::Slice {
                    slice: weight_ref.clone(),
                    layout: None,
                    mutable: false,
                    batched: false,
                    shape: self.weight.shape,
                },
                KernelInput::Slice {
                    slice: input_grad_ref,
                    layout: None,
                    mutable: true,
                    batched: true,
                    shape: self.input.shape,
                },
            ];
            let grid_dim = [
                Expr::Const(ceil_div(self.spatial_w, BLOCK_X)),
                Expr::Const(ceil_div(self.spatial_h, BLOCK_Y)),
                Expr::Var * self.in_channels,
            ];
            let block_dim = [Expr::Const(BLOCK_X), Expr::Const(BLOCK_Y), Expr::Const(1)];
            let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes: Expr::Const(0) };
            let code = self.substitute(include_str!("conv/bwd_data.cu"));
            let kernel = unsafe { Kernel::new("Conv2dBackwardData".to_string(), code, args) }.unwrap();
            func.push(kernel);
        }

        // backward_filter
        let inputs = vec![
            KernelInput::Size(Expr::Var),
            KernelInput::Slice {
                slice: input_ref,
                layout: None,
                mutable: false,
                batched: true,
                shape: self.input.shape,
            },
            KernelInput::Slice {
                slice: output_grad_ref,
                layout: None,
                mutable: false,
                batched: true,
                shape: Shape::new((self.out_channels * self.out_h() * self.out_w()) as usize, 1),
            },
            KernelInput::Slice {
                slice: weight_grad_ref,
                layout: None,
                mutable: true,
                batched: false,
                shape: self.weight.shape,
            },
        ];
        let grid_dim = [
            Expr::Const(ceil_div(self.kernel_size, BLOCK_X)),
            Expr::Const(ceil_div(self.kernel_size, BLOCK_Y)),
            Expr::Const(self.out_channels * self.in_channels),
        ];
        let block_dim = [Expr::Const(BLOCK_X), Expr::Const(BLOCK_Y), Expr::Const(1)];
        let args = KernelArgs { inputs, grid_dim, block_dim, shared_mem_bytes: Expr::Const(0) };
        let code = self.substitute(include_str!("conv/bwd_filter.cu"));
        let kernel = unsafe { Kernel::new("Conv2dBackwardFilter".to_string(), code, args) }.unwrap();
        func.push(kernel);

        func
    }
}
