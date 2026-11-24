//! SEBlock backward graph IR smoke (CRCL-v2 改善 A、 chain rule 完成)。
//!
//! `builder.apply(SEBlockMixed)` + `graph.execute("backward")` で
//! dW1, dW2 の grad が CPU reference と一致確認。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::{CudaDevice, ops::SEBlockMixed};

fn main() {
    println!("=== SEBlock backward graph IR smoke ===");

    let c = 8i32;
    let h = 5i32;
    let w_dim = 5i32;
    let r = 2i32;
    let c_r = c / r;
    let batch = 2usize;
    let chw = (c * h * w_dim) as usize;

    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    let w1 = builder.new_weights("w1", Shape::new((c_r * c) as usize, 1), InitSettings::Zeroed);
    let w2 = builder.new_weights("w2", Shape::new((c * c_r) as usize, 1), InitSettings::Zeroed);
    let x = builder.new_dense_input("x", Shape::new(chw, 1));

    let y = builder.apply(SEBlockMixed {
        input: x.annotated_node(),
        w1: w1.annotated_node(),
        w2: w2.annotated_node(),
        channels: c,
        spatial_h: h,
        spatial_w: w_dim,
        reduction: r,
    });
    let ones = builder.new_constant(Shape::new(1, chw), &vec![1.0f32; chw]);
    let _scalar = ones.matmul(y);

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);

    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..chw * batch).map(|i| (i as f32 * 0.0231 * PI).sin() * 0.5).collect();
    let w1_host: Vec<f32> = (0..(c_r * c) as usize).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.2).collect();
    let w2_host: Vec<f32> = (0..(c * c_r) as usize).map(|i| (i as f32 * 0.0117 * PI).cos() * 0.25).collect();

    {
        let xref = graph.get_input("x");
        let mut xd = xref.dense_mut();
        xd.load_from_slice(Some(batch), &x_host).expect("x load");
    }
    {
        let wref = graph.get_weights("w1");
        let mut wd = wref.dense_mut();
        wd.load_from_slice(None, &w1_host).expect("w1 load");
    }
    {
        let wref = graph.get_weights("w2");
        let mut wd = wref.dense_mut();
        wd.load_from_slice(None, &w2_host).expect("w2 load");
    }

    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync fwd");
    graph.execute("backward").expect("backward");
    graph.synchronise().expect("sync bwd");

    let dw1_ref =
        graph.get(GraphNodeId::new(graph.weight_idx("w1").expect("w1"), GraphNodeIdTy::Gradients)).expect("dw1");
    let mut dw1_got = vec![0.0f32; (c_r * c) as usize];
    dw1_ref.dense().write_to_slice(&mut dw1_got).expect("dw1 write");
    let dw2_ref =
        graph.get(GraphNodeId::new(graph.weight_idx("w2").expect("w2"), GraphNodeIdTy::Gradients)).expect("dw2");
    let mut dw2_got = vec![0.0f32; (c * c_r) as usize];
    dw2_ref.dense().write_to_slice(&mut dw2_got).expect("dw2 write");

    // CPU reference (numerical gradient via finite diff、 簡便)
    // forward 関数: f(W1, W2) = sum_all(y) = sum_all(x * sigmoid(W2 × relu(W1 × pool(x))))
    // upstream grad = ones (graph root の linear matmul 結果)
    let c_us = c as usize;
    let c_r_us = c_r as usize;
    let hw = (h * w_dim) as usize;
    let n_chw = chw;
    let n_input = n_chw * batch;

    fn forward(x: &[f32], w1: &[f32], w2: &[f32], batch: usize, c: usize, c_r: usize, hw: usize) -> Vec<f32> {
        let mut pool = vec![0.0f32; batch * c];
        for n in 0..batch {
            for ci in 0..c {
                let mut acc = 0.0;
                for i in 0..hw {
                    acc += x[(n * c + ci) * hw + i];
                }
                pool[n * c + ci] = acc / hw as f32;
            }
        }
        let mut fc1 = vec![0.0f32; batch * c_r];
        for n in 0..batch {
            for cri in 0..c_r {
                let mut acc = 0.0;
                for ci in 0..c {
                    acc += w1[ci * c_r + cri] * pool[n * c + ci];
                }
                fc1[n * c_r + cri] = acc.max(0.0);
            }
        }
        let mut s = vec![0.0f32; batch * c];
        for n in 0..batch {
            for ci in 0..c {
                let mut acc = 0.0;
                for cri in 0..c_r {
                    acc += w2[cri * c + ci] * fc1[n * c_r + cri];
                }
                s[n * c + ci] = 1.0 / (1.0 + (-acc).exp());
            }
        }
        let mut y = vec![0.0f32; n_input_val(batch, c, hw)];
        for n in 0..batch {
            for ci in 0..c {
                let sc = s[n * c + ci];
                for i in 0..hw {
                    y[(n * c + ci) * hw + i] = x[(n * c + ci) * hw + i] * sc;
                }
            }
        }
        y
    }
    fn n_input_val(batch: usize, c: usize, hw: usize) -> usize {
        batch * c * hw
    }

    fn forward_sum(x: &[f32], w1: &[f32], w2: &[f32], batch: usize, c: usize, c_r: usize, hw: usize) -> f32 {
        forward(x, w1, w2, batch, c, c_r, hw).iter().sum()
    }

    // numerical grad: dL/dW1[i, j] ≈ (f(W1+e) - f(W1-e)) / 2e
    let eps = 1e-3f32;
    let mut dw1_num = vec![0.0f32; (c_r * c) as usize];
    for i in 0..dw1_num.len() {
        let mut w1p = w1_host.clone();
        w1p[i] += eps;
        let mut w1m = w1_host.clone();
        w1m[i] -= eps;
        let fp = forward_sum(&x_host, &w1p, &w2_host, batch, c_us, c_r_us, hw);
        let fm = forward_sum(&x_host, &w1m, &w2_host, batch, c_us, c_r_us, hw);
        dw1_num[i] = (fp - fm) / (2.0 * eps);
    }
    let mut dw2_num = vec![0.0f32; (c * c_r) as usize];
    for i in 0..dw2_num.len() {
        let mut w2p = w2_host.clone();
        w2p[i] += eps;
        let mut w2m = w2_host.clone();
        w2m[i] -= eps;
        let fp = forward_sum(&x_host, &w1_host, &w2p, batch, c_us, c_r_us, hw);
        let fm = forward_sum(&x_host, &w1_host, &w2m, batch, c_us, c_r_us, hw);
        dw2_num[i] = (fp - fm) / (2.0 * eps);
    }

    let mut max_dw1_abs = 0.0f32;
    let mut max_dw1_rel = 0.0f32;
    for i in 0..dw1_got.len() {
        let abs_err = (dw1_got[i] - dw1_num[i]).abs();
        let rel_err = if dw1_num[i].abs() > 0.05 { abs_err / dw1_num[i].abs() } else { 0.0 };
        max_dw1_abs = max_dw1_abs.max(abs_err);
        max_dw1_rel = max_dw1_rel.max(rel_err);
    }
    let mut max_dw2_abs = 0.0f32;
    let mut max_dw2_rel = 0.0f32;
    for i in 0..dw2_got.len() {
        let abs_err = (dw2_got[i] - dw2_num[i]).abs();
        let rel_err = if dw2_num[i].abs() > 0.05 { abs_err / dw2_num[i].abs() } else { 0.0 };
        max_dw2_abs = max_dw2_abs.max(abs_err);
        max_dw2_rel = max_dw2_rel.max(rel_err);
    }
    println!("dW1 vs numerical: abs={:.4e}, rel={:.4e}", max_dw1_abs, max_dw1_rel);
    println!("dW2 vs numerical: abs={:.4e}, rel={:.4e}", max_dw2_abs, max_dw2_rel);
    assert!(max_dw1_rel < 0.05, "dW1 rel too large");
    assert!(max_dw2_rel < 0.05, "dW2 rel too large");

    println!("SEBlock backward smoke PASSED (chain rule 動作)");
    let _ = n_input;
    let _ = n_chw;
}
