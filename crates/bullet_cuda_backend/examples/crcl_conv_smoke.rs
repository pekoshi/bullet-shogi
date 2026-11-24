//! CRCL-v2 Conv 3x3 自前 kernel smoke test
//!
//! 1050 Ti (sm_61) で cuDNN 9.x が EXECUTION_FAILED するため、 自前 CUDA kernel を
//! NVRTC で JIT compile する路線の動作確認。
//!
//! 確認内容:
//!   1. Conv2dForwardNCHW: all-ones input/weight で center=Cin*K*K, corner=Cin*4
//!   2. Conv2dBackwardDataNCHW: 全 dy=1 から dx を計算、 中央は Cout*K*K、 corner は Cout*4
//!   3. Conv2dBackwardFilterNCHW: 全 dy=1, x=1 から dw を計算、 各要素は N*Hout_used*Wout_used
//!   4. numerical grad check: forward を eps shift して中心差分 ≈ analytic grad
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_conv_smoke --release

use bullet_cuda_backend::cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::compile_ptx,
};
use std::sync::Arc;

const CONV_KERNEL_SRC: &str = r#"
extern "C" __global__ void Conv2dForwardNCHW(
    int N, int Cin, int Cout, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride,
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y)
{
    int wo = blockIdx.x * blockDim.x + threadIdx.x;
    int ho = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    if (wo >= Wout || ho >= Hout || nc >= N * Cout) return;

    int n  = nc / Cout;
    int co = nc % Cout;

    float acc = 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            int hi = ho * Stride + kh - Pad;
            if (hi < 0 || hi >= H) continue;
            for (int kw = 0; kw < K; ++kw) {
                int wi = wo * Stride + kw - Pad;
                if (wi < 0 || wi >= W) continue;
                float xv = x[((n * Cin + ci) * H + hi) * W + wi];
                float wv = w[((co * Cin + ci) * K + kh) * K + kw];
                acc += xv * wv;
            }
        }
    }
    y[((n * Cout + co) * Hout + ho) * Wout + wo] = acc;
}

// dx[n,ci,hi,wi] = sum_{co,kh,kw} dy[n,co,ho,wo] * w[co,ci,kh,kw]
// ho = (hi + Pad - kh) / Stride、 wo 同様。 ho が Stride で割り切れ、 ho ∈ [0, Hout) のとき有効
extern "C" __global__ void Conv2dBackwardDataNCHW(
    int N, int Cin, int Cout, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride,
    const float* __restrict__ dy,
    const float* __restrict__ w,
    float* __restrict__ dx)
{
    int wi = blockIdx.x * blockDim.x + threadIdx.x;
    int hi = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z;
    if (wi >= W || hi >= H || nc >= N * Cin) return;

    int n  = nc / Cin;
    int ci = nc % Cin;

    float acc = 0.0f;
    for (int co = 0; co < Cout; ++co) {
        for (int kh = 0; kh < K; ++kh) {
            int ho_num = hi + Pad - kh;
            if (ho_num < 0 || (ho_num % Stride) != 0) continue;
            int ho = ho_num / Stride;
            if (ho >= Hout) continue;
            for (int kw = 0; kw < K; ++kw) {
                int wo_num = wi + Pad - kw;
                if (wo_num < 0 || (wo_num % Stride) != 0) continue;
                int wo = wo_num / Stride;
                if (wo >= Wout) continue;
                float dyv = dy[((n * Cout + co) * Hout + ho) * Wout + wo];
                float wv  = w[((co * Cin + ci) * K + kh) * K + kw];
                acc += dyv * wv;
            }
        }
    }
    dx[((n * Cin + ci) * H + hi) * W + wi] = acc;
}

// dw[co,ci,kh,kw] = sum_{n,ho,wo} x[n,ci,hi,wi] * dy[n,co,ho,wo]
// hi = ho*Stride + kh - Pad
extern "C" __global__ void Conv2dBackwardFilterNCHW(
    int N, int Cin, int Cout, int H, int W, int Hout, int Wout,
    int K, int Pad, int Stride,
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float* __restrict__ dw)
{
    int kw = blockIdx.x * blockDim.x + threadIdx.x;
    int kh = blockIdx.y * blockDim.y + threadIdx.y;
    int co_ci = blockIdx.z;
    if (kw >= K || kh >= K || co_ci >= Cout * Cin) return;

    int co = co_ci / Cin;
    int ci = co_ci % Cin;

    float acc = 0.0f;
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < Hout; ++ho) {
            int hi = ho * Stride + kh - Pad;
            if (hi < 0 || hi >= H) continue;
            for (int wo = 0; wo < Wout; ++wo) {
                int wi = wo * Stride + kw - Pad;
                if (wi < 0 || wi >= W) continue;
                float xv  = x[((n * Cin + ci) * H + hi) * W + wi];
                float dyv = dy[((n * Cout + co) * Hout + ho) * Wout + wo];
                acc += xv * dyv;
            }
        }
    }
    dw[((co * Cin + ci) * K + kh) * K + kw] = acc;
}
"#;

#[allow(clippy::too_many_arguments)]
fn launch_conv_forward(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    n: i32,
    cin: i32,
    cout: i32,
    h: i32,
    w: i32,
    hout: i32,
    wout: i32,
    k: i32,
    pad: i32,
    stride: i32,
    x: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) {
    let block = (16u32, 16u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: ((wout as u32).div_ceil(block.0), (hout as u32).div_ceil(block.1), (n * cout) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(func)
            .arg(&n)
            .arg(&cin)
            .arg(&cout)
            .arg(&h)
            .arg(&w)
            .arg(&hout)
            .arg(&wout)
            .arg(&k)
            .arg(&pad)
            .arg(&stride)
            .arg(x)
            .arg(weights)
            .arg(y)
            .launch(cfg)
            .unwrap();
    }
}

fn main() {
    println!("=== CRCL-v2 Conv 3x3 自前 kernel smoke ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new failed");
    let stream = ctx.default_stream();
    println!("CUDA context created (device 0)");

    let ptx = compile_ptx(CONV_KERNEL_SRC).expect("nvrtc compile failed");
    let module = ctx.load_module(ptx).expect("load_module failed");
    let fw = module.load_function("Conv2dForwardNCHW").unwrap();
    let bw_data = module.load_function("Conv2dBackwardDataNCHW").unwrap();
    let bw_filter = module.load_function("Conv2dBackwardFilterNCHW").unwrap();
    println!("NVRTC compiled fw + backward_data + backward_filter");

    let n = 1i32;
    let cin = 192i32;
    let cout = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;

    // ----- Phase 1: forward all-ones -----
    let x_host = vec![1.0f32; (n * cin * h * w) as usize];
    let w_host = vec![1.0f32; (cout * cin * k * k) as usize];
    let x = stream.memcpy_stod(&x_host).unwrap();
    let weights = stream.memcpy_stod(&w_host).unwrap();
    let mut y = stream.alloc_zeros::<f32>((n * cout * hout * wout) as usize).unwrap();

    launch_conv_forward(&stream, &fw, n, cin, cout, h, w, hout, wout, k, pad, stride, &x, &weights, &mut y);
    stream.synchronize().unwrap();

    let y_host = stream.memcpy_dtov(&y).unwrap();
    let center = y_host[((cout / 2 * hout + hout / 2) * wout + wout / 2) as usize];
    let corner = y_host[0];
    let edge = y_host[(wout / 2) as usize];
    println!("forward: center={center} corner={corner} edge={edge}");
    assert!((center - (cin * k * k) as f32).abs() < 1e-3);
    assert!((corner - (cin * 4) as f32).abs() < 1e-3);
    assert!((edge - (cin * 6) as f32).abs() < 1e-3);
    println!("forward all-ones PASSED");

    // ----- Phase 2: backward_data all-ones -----
    // dy=全1, w=全1 → dx[hi,wi] = Cout * (有効な (kh,kw,ho,wo) ペア数)
    // 中央 (hi=4,wi=4) padding=1 stride=1: 全 K*K=9 が valid → Cout*9
    // corner (hi=0,wi=0): kh∈{0,1}, kw∈{0,1} のみ valid → Cout*4
    let dy = stream.memcpy_stod(&vec![1.0f32; (n * cout * hout * wout) as usize]).unwrap();
    let mut dx = stream.alloc_zeros::<f32>((n * cin * h * w) as usize).unwrap();

    let block = (16u32, 16u32, 1u32);
    let cfg = LaunchConfig {
        grid_dim: ((w as u32).div_ceil(block.0), (h as u32).div_ceil(block.1), (n * cin) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&bw_data)
            .arg(&n)
            .arg(&cin)
            .arg(&cout)
            .arg(&h)
            .arg(&w)
            .arg(&hout)
            .arg(&wout)
            .arg(&k)
            .arg(&pad)
            .arg(&stride)
            .arg(&dy)
            .arg(&weights)
            .arg(&mut dx)
            .launch(cfg)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let dx_host = stream.memcpy_dtov(&dx).unwrap();
    let dx_center = dx_host[((cin / 2 * h + h / 2) * w + w / 2) as usize];
    let dx_corner = dx_host[0];
    println!("bw_data: center={dx_center} corner={dx_corner}");
    assert!((dx_center - (cout * 9) as f32).abs() < 1e-3);
    assert!((dx_corner - (cout * 4) as f32).abs() < 1e-3);
    println!("backward_data all-ones PASSED");

    // ----- Phase 3: backward_filter all-ones -----
    // x=全1, dy=全1 → dw[co,ci,kh,kw] = N * (有効な (ho,wo) ペア数)
    // 中央 kernel pos (kh=1,kw=1): 全 Hout*Wout で valid → N*Hout*Wout = 81
    // corner kernel pos (kh=0,kw=0): ho=0 で hi=-1 invalid、 ho>=1 から有効 → (Hout-1)*(Wout-1) = 64
    let mut dw = stream.alloc_zeros::<f32>((cout * cin * k * k) as usize).unwrap();
    let cfg2 = LaunchConfig {
        grid_dim: ((k as u32).div_ceil(block.0), (k as u32).div_ceil(block.1), (cout * cin) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&bw_filter)
            .arg(&n)
            .arg(&cin)
            .arg(&cout)
            .arg(&h)
            .arg(&w)
            .arg(&hout)
            .arg(&wout)
            .arg(&k)
            .arg(&pad)
            .arg(&stride)
            .arg(&x)
            .arg(&dy)
            .arg(&mut dw)
            .launch(cfg2)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let dw_host = stream.memcpy_dtov(&dw).unwrap();
    let dw_center = dw_host[(((0 * cin + 0) * k + 1) * k + 1) as usize];
    let dw_corner = dw_host[(((0 * cin + 0) * k + 0) * k + 0) as usize];
    println!("bw_filter: center={dw_center} corner={dw_corner}");
    assert!((dw_center - (n * hout * wout) as f32).abs() < 1e-3);
    assert!((dw_corner - (n * (hout - 1) * (wout - 1)) as f32).abs() < 1e-3);
    println!("backward_filter all-ones PASSED");

    // ----- Phase 4: numerical grad check (small shape, random x/w) -----
    let n2 = 2i32;
    let cin2 = 3i32;
    let cout2 = 4i32;
    let h2 = 5i32;
    let w2 = 5i32;
    let hout2 = (h2 + 2 * pad - k) / stride + 1;
    let wout2 = (w2 + 2 * pad - k) / stride + 1;
    let nx2 = (n2 * cin2 * h2 * w2) as usize;
    let nw2 = (cout2 * cin2 * k * k) as usize;
    let ny2 = (n2 * cout2 * hout2 * wout2) as usize;

    let x2_host: Vec<f32> = (0..nx2).map(|i| (i as f32 * 0.137).sin() * 0.5).collect();
    let w2_host: Vec<f32> = (0..nw2).map(|i| (i as f32 * 0.089).cos() * 0.3).collect();
    let x2 = stream.memcpy_stod(&x2_host).unwrap();
    let w2_dev = stream.memcpy_stod(&w2_host).unwrap();

    let mut y2 = stream.alloc_zeros::<f32>(ny2).unwrap();
    launch_conv_forward(&stream, &fw, n2, cin2, cout2, h2, w2, hout2, wout2, k, pad, stride, &x2, &w2_dev, &mut y2);

    // analytic dx, dw with dy = all-ones
    let dy2 = stream.memcpy_stod(&vec![1.0f32; ny2]).unwrap();
    let mut dx2 = stream.alloc_zeros::<f32>(nx2).unwrap();
    let cfg3 = LaunchConfig {
        grid_dim: ((w2 as u32).div_ceil(block.0), (h2 as u32).div_ceil(block.1), (n2 * cin2) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&bw_data)
            .arg(&n2)
            .arg(&cin2)
            .arg(&cout2)
            .arg(&h2)
            .arg(&w2)
            .arg(&hout2)
            .arg(&wout2)
            .arg(&k)
            .arg(&pad)
            .arg(&stride)
            .arg(&dy2)
            .arg(&w2_dev)
            .arg(&mut dx2)
            .launch(cfg3)
            .unwrap();
    }
    let mut dw2 = stream.alloc_zeros::<f32>(nw2).unwrap();
    let cfg4 = LaunchConfig {
        grid_dim: ((k as u32).div_ceil(block.0), (k as u32).div_ceil(block.1), (cout2 * cin2) as u32),
        block_dim: block,
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&bw_filter)
            .arg(&n2)
            .arg(&cin2)
            .arg(&cout2)
            .arg(&h2)
            .arg(&w2)
            .arg(&hout2)
            .arg(&wout2)
            .arg(&k)
            .arg(&pad)
            .arg(&stride)
            .arg(&x2)
            .arg(&dy2)
            .arg(&mut dw2)
            .launch(cfg4)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let dx2_host = stream.memcpy_dtov(&dx2).unwrap();
    let dw2_host = stream.memcpy_dtov(&dw2).unwrap();

    // numerical: sum(y) を eps shift で評価
    let eps = 1e-3f32;
    let mut max_err_x = 0.0f32;
    for probe_idx in &[0usize, 7, 31, 64, 100, 149] {
        let mut x_plus = x2_host.clone();
        let mut x_minus = x2_host.clone();
        x_plus[*probe_idx] += eps;
        x_minus[*probe_idx] -= eps;
        let x_p = stream.memcpy_stod(&x_plus).unwrap();
        let x_m = stream.memcpy_stod(&x_minus).unwrap();
        let mut y_p = stream.alloc_zeros::<f32>(ny2).unwrap();
        let mut y_m = stream.alloc_zeros::<f32>(ny2).unwrap();
        launch_conv_forward(
            &stream, &fw, n2, cin2, cout2, h2, w2, hout2, wout2, k, pad, stride, &x_p, &w2_dev, &mut y_p,
        );
        launch_conv_forward(
            &stream, &fw, n2, cin2, cout2, h2, w2, hout2, wout2, k, pad, stride, &x_m, &w2_dev, &mut y_m,
        );
        stream.synchronize().unwrap();
        let y_p_host = stream.memcpy_dtov(&y_p).unwrap();
        let y_m_host = stream.memcpy_dtov(&y_m).unwrap();
        let numerical = (y_p_host.iter().sum::<f32>() - y_m_host.iter().sum::<f32>()) / (2.0 * eps);
        let analytic = dx2_host[*probe_idx];
        let err = (numerical - analytic).abs();
        max_err_x = max_err_x.max(err);
        println!("  dx idx={probe_idx}: num={numerical:.6} ana={analytic:.6} err={err:.2e}");
    }
    assert!(max_err_x < 1e-2, "dx max err {max_err_x} too high");

    let mut max_err_w = 0.0f32;
    for probe_idx in &[0usize, 5, 13, 40, 80] {
        let mut w_plus = w2_host.clone();
        let mut w_minus = w2_host.clone();
        w_plus[*probe_idx] += eps;
        w_minus[*probe_idx] -= eps;
        let w_p = stream.memcpy_stod(&w_plus).unwrap();
        let w_m = stream.memcpy_stod(&w_minus).unwrap();
        let mut y_p = stream.alloc_zeros::<f32>(ny2).unwrap();
        let mut y_m = stream.alloc_zeros::<f32>(ny2).unwrap();
        launch_conv_forward(&stream, &fw, n2, cin2, cout2, h2, w2, hout2, wout2, k, pad, stride, &x2, &w_p, &mut y_p);
        launch_conv_forward(&stream, &fw, n2, cin2, cout2, h2, w2, hout2, wout2, k, pad, stride, &x2, &w_m, &mut y_m);
        stream.synchronize().unwrap();
        let y_p_host = stream.memcpy_dtov(&y_p).unwrap();
        let y_m_host = stream.memcpy_dtov(&y_m).unwrap();
        let numerical = (y_p_host.iter().sum::<f32>() - y_m_host.iter().sum::<f32>()) / (2.0 * eps);
        let analytic = dw2_host[*probe_idx];
        let err = (numerical - analytic).abs();
        max_err_w = max_err_w.max(err);
        println!("  dw idx={probe_idx}: num={numerical:.6} ana={analytic:.6} err={err:.2e}");
    }
    assert!(max_err_w < 1e-2, "dw max err {max_err_w} too high");

    println!("numerical grad check PASSED (max_err_x={:.2e}, max_err_w={:.2e})", max_err_x, max_err_w);
    println!("ALL Conv 3x3 kernels OK on 1050 Ti");
}
