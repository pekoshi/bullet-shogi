//! User-facing ext traits to add CRCL-v2 ops (Conv2d 等) onto `GraphBuilderNode<CudaMarker>`
//!
//! 使い方:
//! ```ignore
//! use bullet_cuda_backend::builder::{Conv2dParams, ConvBuilderExt};
//!
//! let conv = input.conv2d(weight, Conv2dParams {
//!     in_channels: 192, out_channels: 192,
//!     spatial_h: 9, spatial_w: 9,
//!     kernel_size: 3, padding: 1, stride: 1,
//! });
//! ```

use acyclib::graph::builder::GraphBuilderNode;

use crate::{
    CudaMarker,
    ops::{Conv2d, Conv2dF16Mixed},
};

#[derive(Clone, Copy, Debug)]
pub struct Conv2dParams {
    pub in_channels: i32,
    pub out_channels: i32,
    pub spatial_h: i32,
    pub spatial_w: i32,
    pub kernel_size: i32,
    pub padding: i32,
    pub stride: i32,
}

impl Conv2dParams {
    pub fn out_h(&self) -> i32 {
        (self.spatial_h + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
    pub fn out_w(&self) -> i32 {
        (self.spatial_w + 2 * self.padding - self.kernel_size) / self.stride + 1
    }
    pub fn input_size(&self) -> usize {
        (self.in_channels * self.spatial_h * self.spatial_w) as usize
    }
    pub fn output_size(&self) -> usize {
        (self.out_channels * self.out_h() * self.out_w()) as usize
    }
    pub fn weight_size(&self) -> usize {
        (self.out_channels * self.in_channels * self.kernel_size * self.kernel_size) as usize
    }
}

pub trait ConvBuilderExt: Sized {
    /// Apply a Conv2d on `self` with the given `weight` and `params`. Returns the conv output node.
    fn conv2d(self, weight: Self, params: Conv2dParams) -> Self;

    /// Phase 2 IV案: mixed precision Conv2d (im2col_f16 + HGEMM Tensor Core + permute)。
    /// T4 等 sm_70+ で速度向上、 1050 Ti でも crash なく動作 (fallback)。
    fn conv2d_mixed(self, weight: Self, params: Conv2dParams) -> Self;
}

impl<'a> ConvBuilderExt for GraphBuilderNode<'a, CudaMarker> {
    fn conv2d(self, weight: Self, p: Conv2dParams) -> Self {
        let input_node = self.annotated_node();
        let weight_node = weight.annotated_node();
        self.graph_builder().apply(Conv2d {
            input: input_node,
            weight: weight_node,
            in_channels: p.in_channels,
            out_channels: p.out_channels,
            spatial_h: p.spatial_h,
            spatial_w: p.spatial_w,
            kernel_size: p.kernel_size,
            padding: p.padding,
            stride: p.stride,
        })
    }

    fn conv2d_mixed(self, weight: Self, p: Conv2dParams) -> Self {
        let input_node = self.annotated_node();
        let weight_node = weight.annotated_node();
        self.graph_builder().apply(Conv2dF16Mixed {
            input: input_node,
            weight: weight_node,
            in_channels: p.in_channels,
            out_channels: p.out_channels,
            spatial_h: p.spatial_h,
            spatial_w: p.spatial_w,
            kernel_size: p.kernel_size,
            padding: p.padding,
            stride: p.stride,
        })
    }
}
