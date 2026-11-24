use std::sync::OnceLock;

use acyclib::device::{operation, tensor::Shape};
use cudarc::{
    cublas::{
        Gemm, GemmConfig, StridedBatchedConfig, result,
        sys::{
            self,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
        },
    },
    driver::{DevicePtr, DevicePtrMut},
};

use crate::{CudaBuffer, CudaError};

/// FP16-accelerated FP32 gemm を有効化するかどうかを env var で判定する。
///
/// `BULLET_USE_FAST_FP16=1` で `cublasGemmEx` + `CUBLAS_COMPUTE_32F_FAST_16F` 経路に切替える。
/// 入出力は f32 のまま、 内部 accumulate のみ f16 を経由するため、 weights/Adam state は f32 維持。
/// Tensor Core (sm_70+) 搭載 GPU で 1.5-2x の速度向上が期待できる。
/// 値は process 起動時に固定キャッシュされる。
fn use_fast_fp16() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("BULLET_USE_FAST_FP16").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
    })
}

#[allow(unused)]
impl operation::BlasOperations for CudaBuffer<f32> {
    type BlasError = CudaError;

    fn gemm(&mut self, config: &operation::GemmConfig, a: &Self, b: &Self) -> Result<(), Self::BlasError> {
        let (cfg, _) = convert_gemm_config(config);

        if use_fast_fp16() {
            // f32 IO + internal f16 compute (Tensor Core path)
            let stream = self.device.stream();
            let (a_ptr, _ra) = a.buf.device_ptr(&stream);
            let (b_ptr, _rb) = b.buf.device_ptr(&stream);
            let (c_ptr, _rc) = self.buf.device_ptr_mut(&stream);
            unsafe {
                result::gemm_ex(
                    *self.device.blas().handle(),
                    cfg.transa,
                    cfg.transb,
                    cfg.m,
                    cfg.n,
                    cfg.k,
                    &cfg.alpha as *const f32 as *const _,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    cfg.lda,
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    cfg.ldb,
                    &cfg.beta as *const f32 as *const _,
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    cfg.ldc,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(CudaError::Blas)
            }
        } else {
            unsafe { self.device.blas().gemm(cfg, &a.buf, &b.buf, &mut self.buf).map_err(CudaError::Blas) }
        }
    }

    fn gebmm(
        &mut self,
        config: &operation::GemmConfig,
        batch_size: usize,
        a: &Self,
        b: &Self,
    ) -> Result<(), Self::BlasError> {
        let (gemm, shape_o) = convert_gemm_config(config);

        let stride_a = config.shape_a.size() as i64;
        let stride_b = config.shape_b.size() as i64;
        let stride_c = shape_o.size() as i64;
        let batch_size = batch_size as i32;

        if use_fast_fp16() {
            let stream = self.device.stream();
            let (a_ptr, _ra) = a.buf.device_ptr(&stream);
            let (b_ptr, _rb) = b.buf.device_ptr(&stream);
            let (c_ptr, _rc) = self.buf.device_ptr_mut(&stream);
            unsafe {
                result::gemm_strided_batched_ex(
                    *self.device.blas().handle(),
                    gemm.transa,
                    gemm.transb,
                    gemm.m,
                    gemm.n,
                    gemm.k,
                    &gemm.alpha as *const f32 as *const _,
                    a_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    gemm.lda,
                    stride_a,
                    b_ptr as *const _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    gemm.ldb,
                    stride_b,
                    &gemm.beta as *const f32 as *const _,
                    c_ptr as *mut _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    gemm.ldc,
                    stride_c,
                    batch_size,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16F,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .map_err(CudaError::Blas)
            }
        } else {
            let cfg = StridedBatchedConfig { gemm, batch_size, stride_a, stride_b, stride_c };
            unsafe {
                self.device.blas().gemm_strided_batched(cfg, &a.buf, &b.buf, &mut self.buf).map_err(CudaError::Blas)
            }
        }
    }
}

pub fn convert_gemm_config(config: &operation::GemmConfig) -> (GemmConfig<f32>, Shape) {
    let operation::GemmConfig { alpha, beta, shape_a, trans_a, shape_b, trans_b } = *config;
    let shape_o = shape_a.maybe_transpose(trans_a) * shape_b.maybe_transpose(trans_b);

    let m = if trans_a { shape_a.cols() } else { shape_a.rows() };
    let n = if trans_b { shape_b.rows() } else { shape_b.cols() };
    let k = if trans_a { shape_a.rows() } else { shape_a.cols() };

    if trans_b {
        assert_eq!(shape_b.cols(), k);
    } else {
        assert_eq!(shape_b.rows(), k);
    }

    assert_eq!(shape_o.rows(), m);
    assert_eq!(shape_o.cols(), n);

    let transa = if trans_a { CUBLAS_OP_T } else { CUBLAS_OP_N };
    let transb = if trans_b { CUBLAS_OP_T } else { CUBLAS_OP_N };

    let m = m as i32;
    let n = n as i32;
    let k = k as i32;

    let lda = shape_a.rows() as i32;
    let ldb = shape_b.rows() as i32;
    let ldc = shape_o.rows() as i32;

    let cfg = GemmConfig { transa, transb, m, n, k, alpha, lda, ldb, beta, ldc };

    (cfg, shape_o)
}
