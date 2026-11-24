pub mod base;
pub mod blas;
pub mod cmp;
pub mod multi;
pub mod sparse;

#[cfg(test)]
mod bucketed_tests;

use std::sync::Arc;

use crate::{
    device::{Device, DeviceBuffer, OperationError, OperationResult, operation::CoreDeviceOps},
    graph::ir::BackendMarker,
};

#[derive(Debug, Default)]
pub struct CpuError;

#[derive(Debug, Default)]
pub struct CpuThread;

pub struct CpuBuffer<T> {
    buf: Vec<T>,
    device: Arc<CpuThread>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuMarker;

impl BackendMarker for CpuMarker {
    type Backend = CpuThread;
}

impl<T: Copy + Default + Send + Sync> DeviceBuffer<CpuThread, T> for CpuBuffer<T> {
    type BufferError = CpuError;

    fn device(&self) -> Arc<CpuThread> {
        self.device.clone()
    }

    fn new(device: Arc<CpuThread>, size: usize) -> Result<Self, CpuError> {
        Ok(Self { buf: vec![T::default(); size], device })
    }

    fn size(&self) -> usize {
        self.buf.len()
    }

    fn set_zero(&mut self) -> Result<(), CpuError> {
        for elem in &mut self.buf {
            *elem = T::default();
        }

        Ok(())
    }

    fn load_from_device(&mut self, buf: &Self, num: usize) -> Result<(), CpuError> {
        self.buf[..num].copy_from_slice(&buf.buf[..num]);
        Ok(())
    }

    fn load_from_slice(&mut self, buf: &[T]) -> Result<(), CpuError> {
        self.buf[..buf.len()].copy_from_slice(buf);
        Ok(())
    }

    unsafe fn load_non_blocking_from_host(&mut self, buf: &[T]) -> Result<(), Self::BufferError> {
        self.load_from_slice(buf)
    }

    fn write_into_slice(&self, buf: &mut [T], num: usize) -> Result<(), CpuError> {
        buf[..num].copy_from_slice(&self.buf[..num]);
        Ok(())
    }
}

impl Device for CpuThread {
    type Marker = CpuMarker;

    type BufferF32 = CpuBuffer<f32>;
    type BufferI32 = CpuBuffer<i32>;
    type BufferF16 = CpuBuffer<half::f16>;

    type DeviceError = CpuError;

    type IdType = ();

    fn new(_id: Self::IdType) -> Result<Self, Self::DeviceError> {
        Ok(Self)
    }

    fn synchronise(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn get_last_device_error(&self) -> Result<(), Self::DeviceError> {
        Ok(())
    }

    fn sparse_to_dense(
        _batch_size: usize,
        _size: usize,
        _nnz: usize,
        _sparse: &Self::BufferI32,
        _dense: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }
}

impl CoreDeviceOps for CpuThread {
    fn select(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        for i in 0..batch_size {
            let bucket = indices.buf[i] as usize;
            let offset = if input_batched { i * input_size } else { 0 };

            for j in 0..output_size {
                output.buf[output_size * i + j] = input.buf[offset + output_size * bucket + j];
            }
        }

        Ok(())
    }

    fn select_backprop(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        for i in 0..batch_size {
            let bucket = indices.buf[i] as usize;
            let offset = if input_batched { i * input_size } else { 0 };

            for j in 0..output_size {
                input_grad.buf[offset + output_size * bucket + j] += output_grad.buf[output_size * i + j];
            }
        }

        Ok(())
    }

    #[allow(unused)]
    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    #[allow(unused)]
    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    #[allow(unused)]
    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(OperationError::UnsupportedOperation)
    }

    // ===== gather affine (2026-11-14 OPNARCH T2)。 素朴な参照実装 (CUDA 版の照合用)。 =====
    fn bucketed_affine_fwd(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        num_buckets: usize,
        weights: &Self::BufferF32,
        bias: &Self::BufferF32,
        input: &Self::BufferF32,
        buckets: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let rows = num_buckets * output_size;
        if rows * input_size > weights.buf.len()
            || rows > bias.buf.len()
            || batch_size * input_size > input.buf.len()
            || batch_size > buckets.buf.len()
            || batch_size * output_size > output.buf.len()
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        for b in 0..batch_size {
            let bk = buckets.buf[b];
            let out = &mut output.buf[b * output_size..(b + 1) * output_size];
            if bk < 0 || bk as usize >= num_buckets {
                out.fill(0.0);
                continue;
            }
            let base = bk as usize * output_size;
            let x = &input.buf[b * input_size..(b + 1) * input_size];
            for (j, o) in out.iter_mut().enumerate() {
                let mut acc = bias.buf[base + j];
                for (k, &xk) in x.iter().enumerate() {
                    acc += weights.buf[k * rows + base + j] * xk;
                }
                *o = acc;
            }
        }
        Ok(())
    }

    fn bucketed_affine_bwd_input(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        num_buckets: usize,
        weights: &Self::BufferF32,
        buckets: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let rows = num_buckets * output_size;
        if rows * input_size > weights.buf.len()
            || batch_size > buckets.buf.len()
            || batch_size * output_size > output_grad.buf.len()
            || batch_size * input_size > input_grad.buf.len()
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        for b in 0..batch_size {
            let bk = buckets.buf[b];
            if bk < 0 || bk as usize >= num_buckets {
                continue;
            }
            let base = bk as usize * output_size;
            let g = &output_grad.buf[b * output_size..(b + 1) * output_size];
            for k in 0..input_size {
                let mut acc = 0.0f32;
                for (j, &gj) in g.iter().enumerate() {
                    acc += weights.buf[k * rows + base + j] * gj;
                }
                input_grad.buf[b * input_size + k] += acc;
            }
        }
        Ok(())
    }

    fn bucketed_affine_bwd_weights(
        batch_size: usize,
        input_size: usize,
        output_size: usize,
        num_buckets: usize,
        input: &Self::BufferF32,
        buckets: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        weights_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        let rows = num_buckets * output_size;
        if rows * input_size > weights_grad.buf.len()
            || batch_size * input_size > input.buf.len()
            || batch_size > buckets.buf.len()
            || batch_size * output_size > output_grad.buf.len()
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        for b in 0..batch_size {
            let bk = buckets.buf[b];
            if bk < 0 || bk as usize >= num_buckets {
                continue;
            }
            let base = bk as usize * output_size;
            let g = &output_grad.buf[b * output_size..(b + 1) * output_size];
            let x = &input.buf[b * input_size..(b + 1) * input_size];
            for (k, &xk) in x.iter().enumerate() {
                for (j, &gj) in g.iter().enumerate() {
                    weights_grad.buf[k * rows + base + j] += gj * xk;
                }
            }
        }
        Ok(())
    }

    fn bucketed_affine_bwd_bias(
        batch_size: usize,
        output_size: usize,
        num_buckets: usize,
        buckets: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        bias_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        if num_buckets * output_size > bias_grad.buf.len()
            || batch_size > buckets.buf.len()
            || batch_size * output_size > output_grad.buf.len()
        {
            return Err(OperationError::IndexOutOfBounds);
        }
        for b in 0..batch_size {
            let bk = buckets.buf[b];
            if bk < 0 || bk as usize >= num_buckets {
                continue;
            }
            let base = bk as usize * output_size;
            for j in 0..output_size {
                bias_grad.buf[base + j] += output_grad.buf[b * output_size + j];
            }
        }
        Ok(())
    }
}

// ============================================================================
// 2026-11-15 QAT: `FauxQuantise` の CPU 実装
// ============================================================================
//
// 順伝播は CUDA 版 (bullet_cuda_backend/src/ops/qat.rs) と同じ式 `roundf(Q * x) / Q`
// (round = false なら `truncf`) を f32 で計算する。 Rust の `f32::round` は roundf と同じく
// 0 から遠い側への丸め。 逆伝播は IR 側の共通実装 (`FauxQuantise::backward` = 勾配をそのまま足す STE)。
// CPU 版は学習器の検証 (量子化後の整数推論との突き合わせ) に使う。
mod faux_quantise_cpu {
    use crate::{
        dag::NodeId,
        device::{
            OperationError,
            cpu::{CpuError, CpuMarker, CpuThread},
            function::{self, DeviceFunction, DeviceOperation},
            tensor::TensorRef,
        },
        graph::{
            Graph, GraphNodeIdTy,
            ir::operation::{GraphIROperationCompilable, unary::FauxQuantise},
        },
    };

    struct CpuFauxQuantise {
        input: TensorRef<CpuThread>,
        output: TensorRef<CpuThread>,
        value: f32,
        round: bool,
    }

    impl DeviceOperation<CpuThread> for CpuFauxQuantise {
        fn opname(&self) -> String {
            "FauxQuantise".to_string()
        }

        fn execute(&self) -> Result<(), OperationError<CpuError>> {
            let input = self.input.dense();
            let mut output = self.output.dense_mut();
            let size = input.size();
            if output.size() != size || input.buf.buf.len() < size || output.buf.buf.len() < size {
                return Err(OperationError::IndexOutOfBounds);
            }
            let q = self.value;
            for (o, &x) in output.buf.buf[..size].iter_mut().zip(&input.buf.buf[..size]) {
                let y = q * x;
                *o = if self.round { y.round() } else { y.trunc() } / q;
            }
            Ok(())
        }
    }

    impl GraphIROperationCompilable<CpuMarker> for FauxQuantise {
        fn forward_pass(&self, graph: &Graph<CpuThread>, output_node: NodeId) -> DeviceFunction<CpuThread> {
            let input = graph.get_ref(self.input.idx, GraphNodeIdTy::Values);
            let output = graph.get_ref(output_node, GraphNodeIdTy::Values);

            let mut func = DeviceFunction::default();
            func.push(function::MaybeUpdateBatchSize { input: input.clone(), output: output.clone() });
            func.push(CpuFauxQuantise { input, output, value: self.value, round: self.round });
            func
        }

        fn backward_pass(&self, graph: &Graph<CpuThread>, output_node: NodeId) -> DeviceFunction<CpuThread> {
            self.backward::<CpuMarker>(graph, output_node)
        }
    }
}
