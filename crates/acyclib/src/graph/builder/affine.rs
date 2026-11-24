use crate::{
    device::operation::DiffableFromOutput,
    graph::ir::{
        BackendMarker,
        operation::{GraphIROperationCompilable, affine::BucketedAffine, sparse::SparseAffineActivate},
    },
};

use super::{GraphBuilderNode, InitSettings};

#[derive(Clone, Copy)]
pub struct Affine<'a, B: BackendMarker> {
    pub weights: GraphBuilderNode<'a, B>,
    pub bias: GraphBuilderNode<'a, B>,
}

impl<'a, B: BackendMarker> Affine<'a, B>
where
    SparseAffineActivate: GraphIROperationCompilable<B>,
{
    pub fn forward(self, input: GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B> {
        self.weights.matmul(input) + self.bias
    }

    /// Sparse index-only affine transform for binary feature inputs.
    /// The index node is batched by the value loader; no explicit feature values are needed.
    pub fn forward_sparse(self, input: GraphBuilderNode<'a, B>) -> GraphBuilderNode<'a, B> {
        input.builder.apply(SparseAffineActivate {
            weights: self.weights.node,
            indices: input.node,
            values: None,
            biases: Some(self.bias.node),
            activation: DiffableFromOutput::Identity,
        })
    }

    pub fn init_with_effective_input_size(&self, size: usize) {
        let builder = self.weights.builder.ir();
        let id = builder.get_id(self.weights.node.idx).unwrap();
        *self.weights.builder.init().get_mut(&id).unwrap() =
            InitSettings::Normal { mean: 0.0, stdev: (2.0 / size as f32).sqrt() };
    }

    pub fn forward_sparse_with_values(
        self,
        stm: GraphBuilderNode<'a, B>,
        vals: GraphBuilderNode<'a, B>,
    ) -> GraphBuilderNode<'a, B> {
        stm.builder.apply(SparseAffineActivate {
            weights: self.weights.node,
            indices: stm.node,
            values: Some(vals.node),
            biases: Some(self.bias.node),
            activation: DiffableFromOutput::Identity,
        })
    }
}

impl<'a, B: BackendMarker> Affine<'a, B>
where
    BucketedAffine: GraphIROperationCompilable<B>,
{
    /// gather affine (2026-11-14 OPNARCH T2): `self.forward(input).select(buckets)` と同じ値・同じ勾配を、
    /// 選んだ bucket の行だけ計算して出す。 重みは [NB * O, I]、 bias は [NB * O, 1] (Affine と同じ形)。
    pub fn forward_bucketed(
        self,
        input: GraphBuilderNode<'a, B>,
        buckets: GraphBuilderNode<'a, B>,
    ) -> GraphBuilderNode<'a, B> {
        input.builder.apply(BucketedAffine {
            weights: self.weights.node,
            bias: self.bias.node,
            input: input.node,
            buckets: buckets.node,
        })
    }
}
