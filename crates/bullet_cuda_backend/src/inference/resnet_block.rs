//! CRCL-v2 ResNet block (INT8 inference、 fused 版)
//!
//! 構造: y = SiLU(Conv2(SiLU(Conv1(x))) + x)   (residual + post-add SiLU)
//!
//! 実装 (fused):
//!   - conv1.forward_with_silu(input, a)        ← Conv1 + SiLU を 1 path
//!   - conv2.forward_with_residual_silu(a, input, output)  ← Conv2 + +residual + SiLU を 1 path
//!   - scratch buffer は a (Conv1 出力 = Conv2 入力) のみ
//!   - elementwise kernel・ memcpy_dtod 不要

use std::sync::Arc;

use cudarc::{
    cublas::CudaBlas,
    driver::{CudaSlice, CudaStream},
};

use crate::{
    CudaError,
    inference::conv2d_int8::{Conv2dInt8, Conv2dInt8Params, Int8Kernels},
};

/// CRCL-v2 1 ResNet block: y = SiLU(Conv2(SiLU(Conv1(x))) + x)
///
/// in_channels == out_channels、 spatial サイズ維持必須 (padding=1 stride=1 推奨)。
pub struct ResBlockInt8 {
    conv1: Conv2dInt8,
    conv2: Conv2dInt8,
    stream: Arc<CudaStream>,
    /// (batch, intermediate buffer)、 Conv1 出力 = Conv2 入力 (SiLU 済)
    scratch_a: Option<(i32, CudaSlice<f32>)>,
}

impl ResBlockInt8 {
    pub fn new(
        params: Conv2dInt8Params,
        weight1_f32: &CudaSlice<f32>,
        weight2_f32: &CudaSlice<f32>,
        int8_kernels: Int8Kernels,
        blas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        assert_eq!(params.in_channels, params.out_channels, "residual には Cin==Cout 必須");
        let conv1 =
            Conv2dInt8::from_f32_weight(params, weight1_f32, int8_kernels.clone(), blas.clone(), stream.clone())?;
        let conv2 = Conv2dInt8::from_f32_weight(params, weight2_f32, int8_kernels, blas, stream.clone())?;
        Ok(Self { conv1, conv2, stream, scratch_a: None })
    }

    /// 既に i8 量子化済 weight (`.crcli8` reader 出力) から block を構築
    pub fn from_i8_weights(
        params1: Conv2dInt8Params,
        weight1_i8: &[i8],
        params2: Conv2dInt8Params,
        weight2_i8: &[i8],
        int8_kernels: Int8Kernels,
        blas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        assert_eq!(params1.in_channels, params1.out_channels, "residual には Cin==Cout 必須");
        assert_eq!(params2.in_channels, params2.out_channels, "residual には Cin==Cout 必須");
        let conv1 =
            Conv2dInt8::from_i8_weight(params1, weight1_i8, int8_kernels.clone(), blas.clone(), stream.clone())?;
        let conv2 = Conv2dInt8::from_i8_weight(params2, weight2_i8, int8_kernels, blas, stream.clone())?;
        Ok(Self { conv1, conv2, stream, scratch_a: None })
    }

    fn ensure_scratch(&mut self, batch: i32) -> Result<(), CudaError> {
        let n = self.conv1.params.output_size(batch);
        if self.scratch_a.as_ref().map(|(b, _)| *b) != Some(batch) {
            let buf = self.stream.alloc_zeros::<f32>(n).map_err(CudaError::Driver)?;
            self.scratch_a = Some((batch, buf));
        }
        Ok(())
    }

    /// forward: y = SiLU(Conv2(SiLU(Conv1(x))) + x)、 全 fused
    pub fn forward(
        &mut self,
        batch: i32,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        self.ensure_scratch(batch)?;
        // step 1: a = SiLU(Conv1(input))   (fused)
        {
            let a = &mut self.scratch_a.as_mut().unwrap().1;
            self.conv1.forward_with_silu(batch, input, a)?;
        }
        // step 2: output = SiLU(Conv2(a) + input)   (fused with residual+silu)
        {
            let a = &self.scratch_a.as_ref().unwrap().1;
            self.conv2.forward_with_residual_silu(batch, a, input, output)?;
        }
        Ok(())
    }
}

// `ElementwiseKernels` は fused dequant kernel 化で不要になったが、 既存 API 互換のため空 stub を残す。
// 旧 example が compile 通るように。
#[derive(Clone)]
pub struct ElementwiseKernels;

impl ElementwiseKernels {
    pub fn new(_ctx: &Arc<cudarc::driver::CudaContext>) -> Result<Self, CudaError> {
        Ok(Self)
    }
}
