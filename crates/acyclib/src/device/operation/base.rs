#[derive(Clone, Copy, Debug)]
pub struct AdamConfig {
    pub beta1: f32,
    pub beta2: f32,
    pub gradient_factor: f32,
    pub learning_rate: f32,
    /// Adam分母に加えるepsilon。既存optimizerは1e-8を指定する。
    pub eps: f32,
    pub denom: bool,
    pub decay: f32,
    pub clip: Option<(f32, f32)>,
}

/// Ranger21 coreの一更新に必要な係数。
///
/// `momentum` はPNMに必要な直前EMAであり、更新前の値をkernel内で参照する。
/// `max_velocity` はAMSGradのmax second momentである。`step_size`にはLR schedule、bias
/// correction、PNM noise normalizationを事前に畳み込む。
#[derive(Clone, Copy, Debug)]
pub struct Ranger21Config {
    pub beta0: f32,
    pub beta1_sq: f32,
    pub beta2: f32,
    pub gradient_factor: f32,
    pub step_size: f32,
    pub inv_sqrt_bias_correction2: f32,
    pub eps: f32,
    pub decay: f32,
    pub min_weight: f32,
    pub max_weight: f32,
    pub clamp_weights: bool,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffableFromOutput {
    Identity = 0,
    ReLU = 1,
    CReLU = 2,
    SCReLU = 3,
    SqrReLU = 4,
    Sigmoid = 5,
    /// LeakyReLU (slope 0.01): f(x) = max(0.01*x, x)、 dying ReLU 回避用
    LeakyReLU = 6,
}

pub trait BaseOperations {
    type BaseError;

    fn set_to(&mut self, size: usize, val: f32) -> Result<(), Self::BaseError>;

    /// buffer の連続部分範囲 `[start, end)` を `val` で埋める。
    ///
    /// `run_with_zeroed_rows` の post-step zero projection 用: 全 buffer の
    /// GPU→CPU→GPU 往復 (l0w 数百 MB/step) を避け、 device 上で部分範囲のみ書く。
    /// `start > end` または `end > size` はエラー。
    fn set_range_to(&mut self, start: usize, end: usize, val: f32) -> Result<(), Self::BaseError>;

    #[allow(clippy::too_many_arguments)]
    fn copy_or_add_strided(
        &mut self,
        add: bool,
        rows: usize,
        cols: usize,
        offset: usize,
        stride: usize,
        a: &Self,
        offset_a: usize,
        stride_a: usize,
    ) -> Result<(), Self::BaseError>;

    fn diffable_from_output_fwd(
        &mut self,
        size: usize,
        a: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError>;

    fn diffable_from_output_bwd(
        &mut self,
        size: usize,
        a: &Self,
        grd: &Self,
        act: DiffableFromOutput,
    ) -> Result<(), Self::BaseError>;

    fn linear_comb(&mut self, size: usize, alpha: f32, beta: f32, input: &Self) -> Result<(), Self::BaseError>;

    fn linear_comb_splat(
        &mut self,
        size: usize,
        reps: usize,
        alpha: f32,
        beta: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError>;

    fn reduce_across_batch(
        &mut self,
        size: usize,
        batch_size: usize,
        output_mul: f32,
        input_mul: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError>;

    fn mul_scalar(&mut self, size: usize, alpha: f32) -> Result<(), Self::BaseError>;

    fn add_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError>;

    fn abs_pow_scalar(&mut self, size: usize, alpha: f32, input: &Self) -> Result<(), Self::BaseError>;

    fn abs_pow_scalar_backward(
        &mut self,
        size: usize,
        alpha: f32,
        input: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError>;

    /// SiLU forward: out[i] = in[i] * sigmoid(in[i])
    fn silu_fwd(&mut self, size: usize, input: &Self) -> Result<(), Self::BaseError>;

    /// SiLU backward (accumulate): input_grad[i] += f'(x) * output_grad[i]
    /// f'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    fn silu_bwd(&mut self, size: usize, input: &Self, output_grad: &Self) -> Result<(), Self::BaseError>;

    /// RMSNorm forward: out[i] = in[i] / sqrt(mean(in^2) + eps), per-sample reduction over `feature_size`.
    /// `batch_size` は sample 数 (= total_size / feature_size)。
    fn rms_norm_fwd(
        &mut self,
        feature_size: usize,
        batch_size: usize,
        eps: f32,
        input: &Self,
    ) -> Result<(), Self::BaseError>;

    /// RMSNorm backward (accumulate):
    /// input_grad[i] += output_grad[i] / rms - x[i] * sum(g·x) / (N * rms^3)
    fn rms_norm_bwd(
        &mut self,
        feature_size: usize,
        batch_size: usize,
        eps: f32,
        input: &Self,
        output_grad: &Self,
    ) -> Result<(), Self::BaseError>;

    /// Standalone Softmax forward (per-sample feature 軸 reduction):
    /// out[i] = exp(in[i] - max) / sum_j exp(in[j] - max)
    fn softmax_fwd(&mut self, feature_size: usize, batch_size: usize, input: &Self) -> Result<(), Self::BaseError>;

    /// Softmax backward (accumulate):
    /// input_grad[i] += y[i] * (output_grad[i] - Σ_j y[j] * output_grad[j])
    /// where y = softmax(input). 引数 `softmax_out` は forward の結果 (y) を渡す。
    fn softmax_bwd(
        &mut self,
        feature_size: usize,
        batch_size: usize,
        softmax_out: &Self,
        output_grad: &Self,
    ) -> Result<(), Self::BaseError>;

    fn pairwise_fwd(
        &mut self,
        offset: usize,
        stride: usize,
        size: usize,
        batch_size: usize,
        a: &Self,
    ) -> Result<(), Self::BaseError>;

    fn pairwise_bwd(
        &mut self,
        offset: usize,
        stride: usize,
        size: usize,
        batch_size: usize,
        a: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError>;

    fn power_error_fwd(&mut self, power: f32, size: usize, a: &Self, b: &Self) -> Result<(), Self::BaseError>;

    fn power_error_bwd(
        &mut self,
        power: f32,
        size: usize,
        a: &Self,
        b: &Self,
        grd: &Self,
    ) -> Result<(), Self::BaseError>;

    fn clip(&mut self, size: usize, min: f32, max: f32) -> Result<(), Self::BaseError>;

    fn adam(
        &mut self,
        config: &AdamConfig,
        size: usize,
        grd: &Self,
        mom: &mut Self,
        vel: &mut Self,
    ) -> Result<(), Self::BaseError>;

    /// 区間 `[offset, offset + size)` の要素だけを Ranger21 で更新する。
    /// 5 本の buffer (重み / 勾配 / momentum / velocity / max_velocity) は同じ区間を使う。
    fn ranger21(
        &mut self,
        config: &Ranger21Config,
        offset: usize,
        size: usize,
        grd: &Self,
        momentum: &mut Self,
        velocity: &mut Self,
        max_velocity: &mut Self,
    ) -> Result<(), Self::BaseError>;
}
