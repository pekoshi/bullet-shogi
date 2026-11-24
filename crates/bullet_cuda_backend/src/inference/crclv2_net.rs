//! CRCL-v2 N-block ResNet inference net (INT8、 .crcli8 load 対応)
//!
//! 構成:
//!   input (FP32 NCHW、 Cin*H*W) → ResBlockInt8 × N → output (FP32 NCHW)
//!   ※ value/policy head は別途 (selfplay engine 側で実装)
//!
//! ping-pong buffer で chain 内 N block を切替: input → buf_a → buf_b → buf_a → ...

use std::sync::Arc;

use cudarc::{
    cublas::CudaBlas,
    driver::{CudaContext, CudaSlice, CudaStream},
};

use crate::{
    CudaError,
    inference::{
        Conv2dInt8Params, Int8Kernels, ResBlockInt8,
        crcli8::{Crcli8Layer, layers_to_map},
    },
};

/// CRCL-v2 inference net: ResBlock chain のみ (head 別途)
pub struct Crclv2InferenceNet {
    pub blocks: Vec<ResBlockInt8>,
    pub channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    stream: Arc<CudaStream>,
    /// ping-pong (batch, buffer)、 chain で交互に使用
    buf_a: Option<(i32, CudaSlice<f32>)>,
    buf_b: Option<(i32, CudaSlice<f32>)>,
}

impl Crclv2InferenceNet {
    /// `.crcli8` から ResBlock chain を構築
    /// - `num_blocks`: ResNet block 数 (= conv layer 数 / 2)
    /// - `channels`, `spatial_h/w`, `kernel_size`, `padding`, `stride`: Conv2dInt8Params の共通 params
    /// - `scale_x`: input quantize scale (全 layer 共通の placeholder、 calibration で改善)
    pub fn from_crcli8_layers(
        layers: Vec<Crcli8Layer>,
        num_blocks: usize,
        channels: i32,
        spatial_h: i32,
        spatial_w: i32,
        kernel_size: i32,
        padding: i32,
        stride: i32,
        scale_x: f32,
        ctx: &Arc<CudaContext>,
        blas: Arc<CudaBlas>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, CudaError> {
        let map = layers_to_map(layers);
        let int8_kernels = Int8Kernels::new(ctx)?;

        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let n1 = format!("b{b}_w1");
            let n2 = format!("b{b}_w2");
            let layer1 = map.get(&n1).unwrap_or_else(|| panic!("missing layer {n1}"));
            let layer2 = map.get(&n2).unwrap_or_else(|| panic!("missing layer {n2}"));

            let p1 = Conv2dInt8Params {
                in_channels: channels,
                out_channels: channels,
                spatial_h,
                spatial_w,
                kernel_size,
                padding,
                stride,
                scale_x,
                scale_w: layer1.scale_w,
            };
            let p2 = Conv2dInt8Params {
                in_channels: channels,
                out_channels: channels,
                spatial_h,
                spatial_w,
                kernel_size,
                padding,
                stride,
                scale_x,
                scale_w: layer2.scale_w,
            };
            let block = ResBlockInt8::from_i8_weights(
                p1,
                &layer1.weight_i8,
                p2,
                &layer2.weight_i8,
                int8_kernels.clone(),
                blas.clone(),
                stream.clone(),
            )?;
            blocks.push(block);
        }

        Ok(Self { blocks, channels, spatial_h, spatial_w, stream, buf_a: None, buf_b: None })
    }

    fn ensure_scratch(&mut self, batch: i32) -> Result<(), CudaError> {
        let n = (batch * self.channels * self.spatial_h * self.spatial_w) as usize;
        if self.buf_a.as_ref().map(|(b, _)| *b) != Some(batch) {
            let buf = self.stream.alloc_zeros::<f32>(n).map_err(CudaError::Driver)?;
            self.buf_a = Some((batch, buf));
        }
        if self.buf_b.as_ref().map(|(b, _)| *b) != Some(batch) {
            let buf = self.stream.alloc_zeros::<f32>(n).map_err(CudaError::Driver)?;
            self.buf_b = Some((batch, buf));
        }
        Ok(())
    }

    /// chain forward: input → block[0] → buf_a → block[1] → buf_b → ... → output
    pub fn forward(
        &mut self,
        batch: i32,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<(), CudaError> {
        self.ensure_scratch(batch)?;
        let n_blocks = self.blocks.len();

        // 簡略化: input → buf_a → buf_b → buf_a → ... → 最後の block で output に書く
        // i=0:        block[0].forward(input,  &mut buf_a)
        // i=1..n-1:   block[i].forward(prev, &mut next)
        // i=n-1:      block[n-1].forward(prev, &mut output)
        if n_blocks == 0 {
            // chain なし、 input → output コピーで終わる
            self.stream.memcpy_dtod(input, output).map_err(CudaError::Driver)?;
            return Ok(());
        }
        if n_blocks == 1 {
            self.blocks[0].forward(batch, input, output)?;
            return Ok(());
        }

        // 1 block: input → buf_a
        {
            let buf_a = &mut self.buf_a.as_mut().unwrap().1;
            self.blocks[0].forward(batch, input, buf_a)?;
        }

        // 中間 block: ping-pong
        for i in 1..(n_blocks - 1) {
            if i % 2 == 1 {
                // buf_a → buf_b
                let (src, dst) = split_pair(&mut self.buf_a, &mut self.buf_b);
                self.blocks[i].forward(batch, src, dst)?;
            } else {
                // buf_b → buf_a
                let (src, dst) = split_pair(&mut self.buf_b, &mut self.buf_a);
                self.blocks[i].forward(batch, src, dst)?;
            }
        }

        // 最後の block: prev_buf → output
        let last = n_blocks - 1;
        let prev = if last == 0 {
            // unreachable here (n_blocks >= 2 のため)
            input
        } else if last % 2 == 1 {
            &self.buf_a.as_ref().unwrap().1
        } else {
            &self.buf_b.as_ref().unwrap().1
        };
        self.blocks[last].forward(batch, prev, output)?;
        Ok(())
    }
}

// Helper: 2 つの Option<(i32, CudaSlice<f32>)> から (read, write) reference を取り出す
fn split_pair<'a>(
    src: &'a mut Option<(i32, CudaSlice<f32>)>,
    dst: &'a mut Option<(i32, CudaSlice<f32>)>,
) -> (&'a CudaSlice<f32>, &'a mut CudaSlice<f32>) {
    let src_ref: &CudaSlice<f32> = &src.as_ref().unwrap().1;
    let dst_ref: &mut CudaSlice<f32> = &mut dst.as_mut().unwrap().1;
    // SAFETY: src と dst は別の Option (異なる buffer)、 borrow checker を helper 経由で回避
    let src_ref: &'a CudaSlice<f32> = unsafe { std::mem::transmute(src_ref) };
    (src_ref, dst_ref)
}
