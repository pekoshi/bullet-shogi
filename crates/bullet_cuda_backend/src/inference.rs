//! CRCL-v2 inference 専用 stand-alone module (graph IR 非依存)
//!
//! 訓練後 PTQ 済 i8 weight + scale を読み込んで、 1050 Ti DP4A 経路で INT8 forward を行う。
//! selfplay や USI engine から直接呼ぶ前提。 graph IR の autograd は不要。

pub mod conv2d_int8;
pub mod crcli8;
pub mod crclv2_net;
pub mod resnet_block;

pub use conv2d_int8::{Conv2dInt8, Conv2dInt8Params, Int8Kernels};
pub use crcli8::{Crcli8Error, Crcli8Layer, layers_to_map, read_crcli8};
pub use crclv2_net::Crclv2InferenceNet;
pub use resnet_block::{ElementwiseKernels, ResBlockInt8};
