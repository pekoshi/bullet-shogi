//! SEBlock graph IR 統合 smoke (CRCL-v2 改善 A、 monolithic IR op)。
//!
//! `builder.apply(SEBlockMixed { ... })` で graph 構築 + forward 実行、
//! hand-crafted CPU reference と一致確認。 backward は当面 placeholder。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::{CudaDevice, ops::SEBlockMixed};

fn main() {
    println!("=== SEBlock graph IR smoke ===");

    let c = 16i32;
    let h = 9i32;
    let w_dim = 9i32;
    let r = 4i32;
    let c_r = c / r;
    let batch = 2usize;
    let chw = (c * h * w_dim) as usize;

    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    // weights
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
    let y_idx = y.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);
    println!("graph built: SEBlock (C={c}, H={h}, W={w_dim}, r={r})");

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
    graph.synchronise().expect("sync");

    let y_ref = graph.get(GraphNodeId::new(y_idx, GraphNodeIdTy::Values)).expect("y get");
    let mut got = vec![0.0f32; chw * batch];
    y_ref.dense().write_to_slice(&mut got).expect("y write");

    // CPU reference
    let c_us = c as usize;
    let c_r_us = c_r as usize;
    let hw = (h * w_dim) as usize;
    let mut pool = vec![0.0f32; batch * c_us];
    for n in 0..batch {
        for ci in 0..c_us {
            let mut acc = 0.0f32;
            for i in 0..hw {
                acc += x_host[(n * c_us + ci) * hw + i];
            }
            pool[n * c_us + ci] = acc / hw as f32;
        }
    }
    let mut fc1 = vec![0.0f32; batch * c_r_us];
    for n in 0..batch {
        for cri in 0..c_r_us {
            let mut acc = 0.0f32;
            for ci in 0..c_us {
                acc += w1_host[ci * c_r_us + cri] * pool[n * c_us + ci];
            }
            fc1[n * c_r_us + cri] = acc.max(0.0);
        }
    }
    let mut s = vec![0.0f32; batch * c_us];
    for n in 0..batch {
        for ci in 0..c_us {
            let mut acc = 0.0f32;
            for cri in 0..c_r_us {
                acc += w2_host[cri * c_us + ci] * fc1[n * c_r_us + cri];
            }
            s[n * c_us + ci] = 1.0 / (1.0 + (-acc).exp());
        }
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for n in 0..batch {
        for ci in 0..c_us {
            let scale = s[n * c_us + ci];
            for i in 0..hw {
                let idx = (n * c_us + ci) * hw + i;
                let want = x_host[idx] * scale;
                let abs_err = (got[idx] - want).abs();
                let rel_err = if want.abs() > 1e-2 { abs_err / want.abs() } else { 0.0 };
                max_abs = max_abs.max(abs_err);
                max_rel = max_rel.max(rel_err);
            }
        }
    }
    println!("vs CPU ref: max_abs={:.6e}, max_rel={:.6e}", max_abs, max_rel);
    assert!(max_rel < 5e-3, "rel err too large");
    println!("SEBlock graph IR smoke PASSED");
    println!("  → builder.apply(SEBlockMixed) で graph forward が動作");
}
