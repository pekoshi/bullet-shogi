use crate::{
    dag::NodeId,
    device::{
        function::{self, Reduce, UnaryOp},
        tensor::Shape,
    },
    graph::{
        DeviceFunction, Graph, GraphNodeIdTy,
        ir::{
            BackendMarker, GraphIR, GraphIRError,
            node::AnnotatedNode,
            operation::{GraphIROperationBase, GraphIROperationCompilable, GraphIROperationError, util},
        },
    },
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unary {
    pub input: AnnotatedNode,
    pub op: UnaryOp,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Unary {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        match self.op {
            UnaryOp::AbsPow(p) => format!("|x|^{p}"),
            UnaryOp::Add(x) => format!("+ {x}"),
            UnaryOp::Mul(x) => format!("* {x}"),
            UnaryOp::DiffableFromOutput(act) => format!("{act:?}"),
            UnaryOp::Silu => "SiLU".to_string(),
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Unary {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::Unary { input, output, op: self.op });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(grd) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: grd.clone() });

            func.push(function::UnaryBackward {
                input,
                input_grad: grd,
                output_grad: graph.get_ref(output_node, GraphNodeIdTy::Gradients),
                op: self.op,
            });
        }

        func
    }
}

// ============================================================================
// Phase 2 graph IR 統合 G2: UnaryF16Mixed (mixed precision unary op)
// ============================================================================
//
// 既存 `Unary` (f32 全 path) と並ぶ新規 IR op。 input/output は f32 Tensor だが、
// 内部で MixedUnary / MixedUnaryBackward (中間 f16 buffer + cast 経由) を push する。
// 既存 `Unary` 無改造。 当面は `UnaryOp::Silu` のみ対応。

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnaryF16Mixed {
    pub input: AnnotatedNode,
    pub op: UnaryOp,
}

impl<B: BackendMarker> GraphIROperationBase<B> for UnaryF16Mixed {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        format!(
            "{}(mixed-f16)",
            match self.op {
                UnaryOp::Silu => "SiLU".to_string(),
                UnaryOp::DiffableFromOutput(act) => format!("{act:?}"),
                UnaryOp::AbsPow(p) => format!("|x|^{p}"),
                UnaryOp::Add(x) => format!("+ {x}"),
                UnaryOp::Mul(x) => format!("* {x}"),
            }
        )
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for UnaryF16Mixed {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::MixedUnary::new(input, output, self.op));
        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();
        if let Some(grd) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: grd.clone() });
            func.push(function::MixedUnaryBackward::new(
                input,
                grd,
                graph.get_ref(output_node, GraphNodeIdTy::Gradients),
                self.op,
            ));
        }
        func
    }
}

#[derive(Debug)]
pub struct ReduceAcrossBatch {
    pub input: AnnotatedNode,
    pub reduction: Reduce,
}

impl<B: BackendMarker> GraphIROperationBase<B> for ReduceAcrossBatch {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_batched(&self, _: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(false)
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        if util::check_not_batched(ir, &self.input).is_ok() {
            return Err(GraphIRError::Op(GraphIROperationError::MismatchedBatching));
        }

        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        format!("Reduce{:?}AcrossBatch", self.reduction)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ReduceAcrossBatch {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::ReduceAcrossBatch {
            input,
            output,
            input_mul: 1.0,
            output_mul: 0.0,
            reduction: self.reduction,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(input_grad) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
            let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input, output: input_grad.clone() });

            func.push(function::SplatAcrossBatch {
                input: output_grad,
                output: input_grad,
                reduction: self.reduction,
                input_mul: 1.0,
                output_mul: 1.0,
            });
        }

        func
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Softmax {
    pub input: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Softmax {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        "Softmax".to_string()
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Softmax {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let feature_size = self.input.shape.size();

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::SoftmaxOp { input, output, feature_size });
        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(grd) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let softmax_out = graph.get_ref(output_node, GraphNodeIdTy::Values);
            let output_grad = graph.get_ref(output_node, GraphNodeIdTy::Gradients);
            let feature_size = self.input.shape.size();

            func.push(function::MaybeUpdateBatchSize { input: softmax_out.clone(), output: grd.clone() });
            func.push(function::SoftmaxBackward { softmax_out, output_grad, input_grad: grd, feature_size });
        }

        func
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNorm {
    pub input: AnnotatedNode,
    pub eps: f32,
}

impl<B: BackendMarker> GraphIROperationBase<B> for RmsNorm {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        Ok(self.input.shape)
    }

    fn shorthand(&self) -> String {
        format!("RmsNorm(eps={})", self.eps)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for RmsNorm {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);
        let feature_size = self.input.shape.size();

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::RmsNorm { input, output, feature_size, eps: self.eps });
        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(grd) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
            let feature_size = self.input.shape.size();

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: grd.clone() });
            func.push(function::RmsNormBackward {
                input,
                input_grad: grd,
                output_grad: graph.get_ref(output_node, GraphNodeIdTy::Gradients),
                feature_size,
                eps: self.eps,
            });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct PairwiseMul {
    pub input: AnnotatedNode,
}

impl<B: BackendMarker> GraphIROperationBase<B> for PairwiseMul {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        let is = self.input.shape;

        if is.rows() % 2 == 0 {
            Ok(Shape::new(is.rows() / 2, is.cols()))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::InvalidInputShape(is)))
        }
    }

    fn shorthand(&self) -> String {
        "PairwiseMul".to_string()
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for PairwiseMul {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::PairwiseMul { offset: 0, input, output });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::PairwiseMulBackward {
                offset: 0,
                values: graph.get_ref(self.input.idx, GraphNodeIdTy::Values),
                input,
                output,
            });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct Slice {
    pub input: AnnotatedNode,
    pub start: usize,
    pub end: usize,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Slice {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;
        let is = self.input.shape;
        if self.end > self.start && self.end <= is.rows() && is.cols() == 1 {
            Ok(Shape::new(self.end - self.start, 1))
        } else {
            Err(GraphIRError::Op(GraphIROperationError::OutOfBounds(is, [self.start, self.end])))
        }
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Slice {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::CopyOrAddStrided {
            input,
            output,
            input_offset: self.start,
            output_offset: 0,
            add: false,
            len_is_out: true,
        });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::CopyOrAddStrided {
                input,
                output,
                input_offset: 0,
                output_offset: self.start,
                add: true,
                len_is_out: false,
            });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct ToDense(pub AnnotatedNode);

impl<B: BackendMarker> GraphIROperationBase<B> for ToDense {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.0]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.0, false)?;
        Ok(self.0.shape)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ToDense {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.0.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::SparseToDense { input, output });

        func
    }

    fn backward_pass(&self, _graph: &Graph<B::Backend>, _output_node: NodeId) -> DeviceFunction<B::Backend> {
        DeviceFunction::default()
    }
}

#[derive(Clone, Debug)]
pub struct Copy {
    pub input: AnnotatedNode,
    pub stop_grad: bool,
}

impl<B: BackendMarker> GraphIROperationBase<B> for Copy {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }

    fn output_requires_grad(&self, ir: &GraphIR<B>) -> Result<bool, GraphIRError> {
        Ok(!self.stop_grad && ir.get(self.input.idx).unwrap().ty().requires_grad)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for Copy {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();
        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::LinearCombination { input_mul: 1.0, output_mul: 0.0, input, output });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if !self.stop_grad
            && let Some(output) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients)
        {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct ClipPassThroughGrad {
    pub input: AnnotatedNode,
    pub min: f32,
    pub max: f32,
}

impl<B: BackendMarker> GraphIROperationBase<B> for ClipPassThroughGrad {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }
}

impl<B: BackendMarker> GraphIROperationCompilable<B> for ClipPassThroughGrad {
    fn forward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
        let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

        let mut func = DeviceFunction::default();

        func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
        func.push(function::LinearCombination { input_mul: 1.0, output_mul: 0.0, input, output: output.clone() });
        func.push(function::ClipInPlace { value: output, min: self.min, max: self.max });

        func
    }

    fn backward_pass(&self, graph: &Graph<B::Backend>, output_node: NodeId) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }
}

#[derive(Clone, Debug)]
pub struct FauxQuantise {
    pub input: AnnotatedNode,
    pub value: f32,
    pub round: bool,
}

impl<B: BackendMarker> GraphIROperationBase<B> for FauxQuantise {
    fn nodes(&self) -> Vec<AnnotatedNode> {
        vec![self.input]
    }

    fn output_shape(&self, ir: &GraphIR<B>) -> Result<Shape, GraphIRError> {
        util::check_dense_eq(ir, &self.input, true)?;

        Ok(self.input.shape)
    }
}

impl FauxQuantise {
    pub fn backward<B: BackendMarker>(
        &self,
        graph: &Graph<B::Backend>,
        output_node: NodeId,
    ) -> DeviceFunction<B::Backend> {
        let mut func = DeviceFunction::default();

        if let Some(output) = graph.maybe_get_ref(self.input.idx, GraphNodeIdTy::Gradients) {
            let input = graph.get_ref(output_node, GraphNodeIdTy::Gradients);

            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(function::LinearCombination { input_mul: 1.0, output_mul: 1.0, input, output });
        }

        func
    }
}
