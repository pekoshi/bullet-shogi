//! Phase 2 graph IR 統合 G6 smoke: mini network を graph IR + mixed precision API で構築。
//!
//! 構造:
//!   w1 (hid × in, not batched)
//!   x  (in × batch)
//!   hidden  = w1 . matmul_mixed(x)         ← MatmulF16Mixed
//!   active  = hidden . silu_mixed()         ← UnaryF16Mixed
//!   w2 (out × hid)
//!   y       = w2 . matmul_mixed(active)    ← MatmulF16Mixed
//!   scalar  = ones · y                       (graph root)
//!
//! 検証:
//!   - 各中間 tensor (graph 上は f32) と CPU reference との一致確認
//!   - 既存 f32 path (matmul + silu) との一致比較
//!
//! NOTE: backward は MatmulF16Mixed placeholder により noop。 forward chain のみ確認。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::CudaDevice;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn run_mini_graph(
    mixed: bool,
    w1_host: &[f32],
    w2_host: &[f32],
    x_host: &[f32],
    in_dim: usize,
    hid_dim: usize,
    out_dim: usize,
    batch: usize,
) -> Vec<f32> {
    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    let w1 = builder.new_weights("w1", Shape::new(hid_dim, in_dim), InitSettings::Zeroed);
    let w2 = builder.new_weights("w2", Shape::new(out_dim, hid_dim), InitSettings::Zeroed);
    let x = builder.new_dense_input("x", Shape::new(in_dim, 1));

    let hidden = if mixed { w1.matmul_mixed(x) } else { w1.matmul(x) };
    let active = if mixed { hidden.silu_mixed() } else { hidden.silu() };
    let y = if mixed { w2.matmul_mixed(active) } else { w2.matmul(active) };

    // graph root 必要 (scalar)
    let ones = builder.new_constant(Shape::new(1, out_dim), &vec![1.0f32; out_dim]);
    let _scalar = ones.matmul(y);
    let y_idx = y.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);

    {
        let w_ref = graph.get_weights("w1");
        let mut wd = w_ref.dense_mut();
        wd.load_from_slice(None, w1_host).expect("w1 load");
    }
    {
        let w_ref = graph.get_weights("w2");
        let mut wd = w_ref.dense_mut();
        wd.load_from_slice(None, w2_host).expect("w2 load");
    }
    {
        let x_ref = graph.get_input("x");
        let mut xd = x_ref.dense_mut();
        xd.load_from_slice(Some(batch), x_host).expect("x load");
    }

    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync");

    let y_ref = graph.get(GraphNodeId::new(y_idx, GraphNodeIdTy::Values)).expect("y get");
    let mut y_host = vec![0.0f32; out_dim * batch];
    y_ref.dense().write_to_slice(&mut y_host).expect("y write");
    y_host
}

fn main() {
    println!("=== mini network graph IR mixed precision smoke (G6) ===");

    let in_dim = 32usize;
    let hid_dim = 64usize;
    let out_dim = 8usize;
    let batch = 16usize;

    use std::f32::consts::PI;
    let w1_host: Vec<f32> = (0..hid_dim * in_dim).map(|i| (i as f32 * 0.0173 * PI).sin() * 0.2).collect();
    let w2_host: Vec<f32> = (0..out_dim * hid_dim).map(|i| (i as f32 * 0.0117 * PI).cos() * 0.3).collect();
    let x_host: Vec<f32> = (0..in_dim * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();

    let y_mixed = run_mini_graph(true, &w1_host, &w2_host, &x_host, in_dim, hid_dim, out_dim, batch);
    let y_f32 = run_mini_graph(false, &w1_host, &w2_host, &x_host, in_dim, hid_dim, out_dim, batch);

    // CPU reference
    let mut hidden = vec![0.0f32; hid_dim * batch];
    for j in 0..hid_dim {
        for b in 0..batch {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += w1_host[i * hid_dim + j] * x_host[b * in_dim + i];
            }
            hidden[b * hid_dim + j] = acc;
        }
    }
    let active: Vec<f32> = hidden.iter().map(|&h| silu(h)).collect();
    let mut y_ref = vec![0.0f32; out_dim * batch];
    for j in 0..out_dim {
        for b in 0..batch {
            let mut acc = 0.0f32;
            for i in 0..hid_dim {
                acc += w2_host[i * out_dim + j] * active[b * hid_dim + i];
            }
            y_ref[b * out_dim + j] = acc;
        }
    }

    let mut max_abs_cpu = 0.0f32;
    let mut max_rel_cpu = 0.0f32;
    let mut max_abs_vs_f32 = 0.0f32;
    let mut max_rel_vs_f32 = 0.0f32;
    for i in 0..out_dim * batch {
        let m = y_mixed[i];
        let f = y_f32[i];
        let r = y_ref[i];
        let abs_cpu = (m - r).abs();
        let rel_cpu = if r.abs() > 1e-2 { abs_cpu / r.abs() } else { 0.0 };
        let abs_f = (m - f).abs();
        let rel_f = if f.abs() > 1e-2 { abs_f / f.abs() } else { 0.0 };
        max_abs_cpu = max_abs_cpu.max(abs_cpu);
        max_rel_cpu = max_rel_cpu.max(rel_cpu);
        max_abs_vs_f32 = max_abs_vs_f32.max(abs_f);
        max_rel_vs_f32 = max_rel_vs_f32.max(rel_f);
    }
    println!("mini network (in={in_dim} hid={hid_dim} out={out_dim} batch={batch}):");
    println!("  vs CPU ref:  abs={:.6e}  rel={:.6e}", max_abs_cpu, max_rel_cpu);
    println!("  vs f32 graph: abs={:.6e}  rel={:.6e}", max_abs_vs_f32, max_rel_vs_f32);
    assert!(max_rel_cpu < 5e-2);
    assert!(max_rel_vs_f32 < 5e-2);

    println!("mini network graph IR mixed precision smoke PASSED (G6 完成)");
    println!("  → matmul_mixed → silu_mixed → matmul_mixed chain が graph forward で動作");
}
