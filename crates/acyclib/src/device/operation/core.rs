use crate::device::{Device, OperationResult};

pub trait CoreDeviceOps: Device {
    fn select(
        batch_size: usize,
        input_batched: bool,
        input_size: usize,
        output_size: usize,
        input: &Self::BufferF32,
        indices: &Self::BufferI32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn select_backprop(
        batch_size: usize,
        input_grad_batched: bool,
        input_size: usize,
        output_size: usize,
        indices: &Self::BufferI32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn softmax_across_batch(
        batch_size: usize,
        single_size: usize,
        input: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn crossentropy(
        size: usize,
        pred: &Self::BufferF32,
        target: &Self::BufferF32,
        output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    fn backprop_softmax_crossentropy(
        size: usize,
        softmaxed: &Self::BufferF32,
        target: &Self::BufferF32,
        output_grad: &Self::BufferF32,
        input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError>;

    // ===== gather affine (選んだ bucket の行だけを計算する affine)。 2026-11-14 OPNARCH T2 =====
    // 記号: B = batch_size、 I = input_size、 O = output_size (1 bucket あたりの出力)、
    //   NB = num_buckets、 rows = NB * O。 重み W は列優先 [rows, I] (W[r, k] = w[k * rows + r])、
    //   bias は [rows]、 入力 x は [B, I] (x[b, k] = x[b * I + k])、 出力 y は [B, O]、
    //   buckets は batch ごとに 1 個の i32 (0 <= bucket < NB。 範囲外の局面は何も計算しない = 出力 0 / 勾配 0)。
    // `Select(Affine(W, x, bias), buckets)` と同じ値・同じ勾配 (浮動小数の加算順は違いうる)。
    // 既定は未対応 (HIP などの backend はそのまま)。 CPU と CUDA が実装する。

    /// 順伝播: `y[b, j] = Σ_k W[bk * O + j, k] * x[b, k] + bias[bk * O + j]` (bk = buckets[b])。 y は上書き。
    #[allow(clippy::too_many_arguments)]
    fn bucketed_affine_fwd(
        _batch_size: usize,
        _input_size: usize,
        _output_size: usize,
        _num_buckets: usize,
        _weights: &Self::BufferF32,
        _bias: &Self::BufferF32,
        _input: &Self::BufferF32,
        _buckets: &Self::BufferI32,
        _output: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(crate::device::OperationError::UnsupportedOperation)
    }

    /// 入力の勾配 (加算): `dx[b, k] += Σ_j W[bk * O + j, k] * dy[b, j]`。
    #[allow(clippy::too_many_arguments)]
    fn bucketed_affine_bwd_input(
        _batch_size: usize,
        _input_size: usize,
        _output_size: usize,
        _num_buckets: usize,
        _weights: &Self::BufferF32,
        _buckets: &Self::BufferI32,
        _output_grad: &Self::BufferF32,
        _input_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(crate::device::OperationError::UnsupportedOperation)
    }

    /// 重みの勾配 (加算): `dW[bk * O + j, k] += Σ_{b: buckets[b] = bk} dy[b, j] * x[b, k]`。
    #[allow(clippy::too_many_arguments)]
    fn bucketed_affine_bwd_weights(
        _batch_size: usize,
        _input_size: usize,
        _output_size: usize,
        _num_buckets: usize,
        _input: &Self::BufferF32,
        _buckets: &Self::BufferI32,
        _output_grad: &Self::BufferF32,
        _weights_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(crate::device::OperationError::UnsupportedOperation)
    }

    /// bias の勾配 (加算): `dbias[bk * O + j] += Σ_{b: buckets[b] = bk} dy[b, j]`。
    fn bucketed_affine_bwd_bias(
        _batch_size: usize,
        _output_size: usize,
        _num_buckets: usize,
        _buckets: &Self::BufferI32,
        _output_grad: &Self::BufferF32,
        _bias_grad: &mut Self::BufferF32,
    ) -> OperationResult<Self::DeviceError> {
        Err(crate::device::OperationError::UnsupportedOperation)
    }
}
