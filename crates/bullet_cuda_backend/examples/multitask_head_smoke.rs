//! CRCL-v2 Multi-task head (policy + value + 詰将棋 dual) graph IR smoke。
//!
//! 構造:
//!   input (Cin = backbone 出力 channels × spatial flatten = 例: 192 * 9 * 9)
//!     → matmul_mixed (Cin → hidden) → silu_mixed   ← 共有 backbone neck
//!     → 3 head 分岐:
//!         (a) policy: matmul_mixed (hidden → policy_dim) → softmax_crossentropy_loss(target_policy)
//!         (b) value:  matmul_mixed (hidden → 1) → power_error(target_value, 2.0)  ← MSE
//!         (c) 詰将棋:  matmul_mixed (hidden → 1) → sigmoid → power_error(target_mate, 2.0)
//!     → total_loss = wp * policy_loss + wv * value_loss + wm * mate_loss
//!     → graph root
//!
//! forward + backward 動作 + 全 head weight grad non-zero 確認。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::{CudaDevice, CudaMarker};

fn main() {
    println!("=== Multi-task head graph IR smoke ===");

    let cin = 64usize; // mini backbone 出力 dim
    let hidden = 32usize;
    let policy_dim = 16usize; // mini policy 出力 (実際は ~7000)
    let batch = 4usize;
    let wp = 1.0f32;
    let wv = 1.0f32;
    let wm = 0.5f32;

    let builder = GraphBuilder::<CudaMarker>::default();
    // 共有 backbone weights
    let w_neck = builder.new_weights("w_neck", Shape::new(hidden, cin), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    // policy head
    let w_policy =
        builder.new_weights("w_policy", Shape::new(policy_dim, hidden), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    // value head
    let w_value = builder.new_weights("w_value", Shape::new(1, hidden), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    // mate head
    let w_mate = builder.new_weights("w_mate", Shape::new(1, hidden), InitSettings::Normal { mean: 0.0, stdev: 0.1 });

    let x = builder.new_dense_input("x", Shape::new(cin, 1));
    let target_policy = builder.new_dense_input("target_policy", Shape::new(policy_dim, 1));
    let target_value = builder.new_dense_input("target_value", Shape::new(1, 1));
    let target_mate = builder.new_dense_input("target_mate", Shape::new(1, 1));

    // 共有 neck (backbone)
    let neck = w_neck.matmul_mixed(x).silu_mixed();

    // policy head: neck → policy_logits → softmax_crossentropy_loss
    let policy_logits = w_policy.matmul_mixed(neck);
    let policy_loss = policy_logits.softmax_crossentropy_loss(target_policy);

    // value head: neck → value_scalar → MSE
    let value_pred = w_value.matmul_mixed(neck);
    let value_loss = value_pred.power_error(target_value, 2.0);

    // mate head: neck → mate_scalar → sigmoid → MSE
    let mate_pre = w_mate.matmul_mixed(neck);
    let mate_pred = mate_pre.sigmoid();
    let mate_loss = mate_pred.power_error(target_mate, 2.0);

    // 各 loss は shape が異なる (policy: (policy_dim, 1), value/mate: (1, 1) batched)
    // ones で reduce して同 shape (1, 1) batched に揃えてから add
    let ones_p = builder.new_constant(Shape::new(1, policy_dim), &vec![1.0f32; policy_dim]);
    let pl_reduced = ones_p.matmul(policy_loss); // (1, 1) batched
    // value/mate は元々 (1, 1) batched
    let pl_weighted = pl_reduced * wp;
    let vl_weighted = value_loss * wv;
    let ml_weighted = mate_loss * wm;
    let total_loss = pl_weighted + vl_weighted + ml_weighted;
    let _scalar = total_loss; // すでに (1, 1) batched、 graph builder が ReduceAcrossBatch 自動追加

    let policy_loss_idx = policy_loss.annotated_node().idx;
    let value_loss_idx = value_loss.annotated_node().idx;
    let mate_loss_idx = mate_loss.annotated_node().idx;

    let device = CudaDevice::new(0).expect("device");
    let mut graph = builder.build(device);
    println!("graph built: backbone (matmul_mixed→silu) + 3 head (policy ce + value mse + mate sigmoid+mse)");

    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..cin * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    let t_policy_host: Vec<f32> = {
        // 各 batch sample で 1-hot に近い分布 (sum = 1)
        let mut v = vec![0.0f32; policy_dim * batch];
        for b in 0..batch {
            v[b * policy_dim + (b % policy_dim)] = 1.0;
        }
        v
    };
    let t_value_host: Vec<f32> = (0..batch).map(|i| (i as f32 * 0.3 - 0.5)).collect();
    let t_mate_host: Vec<f32> = (0..batch).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();

    {
        let xref = graph.get_input("x");
        let mut xd = xref.dense_mut();
        xd.load_from_slice(Some(batch), &x_host).expect("x");
    }
    {
        let tref = graph.get_input("target_policy");
        let mut td = tref.dense_mut();
        td.load_from_slice(Some(batch), &t_policy_host).expect("t_policy");
    }
    {
        let tref = graph.get_input("target_value");
        let mut td = tref.dense_mut();
        td.load_from_slice(Some(batch), &t_value_host).expect("t_value");
    }
    {
        let tref = graph.get_input("target_mate");
        let mut td = tref.dense_mut();
        td.load_from_slice(Some(batch), &t_mate_host).expect("t_mate");
    }

    // forward
    graph.execute("forward").expect("forward");
    graph.synchronise().expect("sync fwd");

    let pl_ref = graph.get(GraphNodeId::new(policy_loss_idx, GraphNodeIdTy::Values)).expect("pl");
    let vl_ref = graph.get(GraphNodeId::new(value_loss_idx, GraphNodeIdTy::Values)).expect("vl");
    let ml_ref = graph.get(GraphNodeId::new(mate_loss_idx, GraphNodeIdTy::Values)).expect("ml");
    let mut pl_host = vec![0.0f32; policy_dim * batch];
    let mut vl_host = vec![0.0f32; batch];
    let mut ml_host = vec![0.0f32; batch];
    pl_ref.dense().write_to_slice(&mut pl_host).expect("pl write");
    vl_ref.dense().write_to_slice(&mut vl_host).expect("vl write");
    ml_ref.dense().write_to_slice(&mut ml_host).expect("ml write");
    let pl_total: f32 = pl_host.iter().sum();
    let vl_total: f32 = vl_host.iter().sum();
    let ml_total: f32 = ml_host.iter().sum();
    println!(
        "forward loss components: policy={:.4} value={:.4} mate={:.4} (weighted total={:.4})",
        pl_total,
        vl_total,
        ml_total,
        wp * pl_total + wv * vl_total + wm * ml_total
    );

    // backward
    graph.execute("backward").expect("backward");
    graph.synchronise().expect("sync bwd");

    // 全 head weight grad 確認
    for (name, size) in
        [("w_neck", hidden * cin), ("w_policy", policy_dim * hidden), ("w_value", hidden), ("w_mate", hidden)]
    {
        let dw_ref =
            graph.get(GraphNodeId::new(graph.weight_idx(name).expect(name), GraphNodeIdTy::Gradients)).expect("dw");
        let mut dw_host = vec![0.0f32; size];
        dw_ref.dense().write_to_slice(&mut dw_host).expect("dw write");
        let nz = dw_host.iter().filter(|v| v.abs() > 1e-10).count();
        let max_abs = dw_host.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let n_finite = dw_host.iter().filter(|v| v.is_finite()).count();
        assert_eq!(n_finite, dw_host.len(), "{name} grad NaN/Inf");
        assert!(nz > 0, "{name} grad all-zero");
        println!("  {name} grad: {}/{} non-zero, max |dw|={:.4e}", nz, size, max_abs);
    }

    println!("\nMulti-task head smoke PASSED");
    println!("  → policy (CE) + value (MSE) + mate (sigmoid+MSE) の 3 head 同時訓練 path 動作");
    println!("  → CRCL-v2 dual head 構造 (dlshogi/tanuki 系) を builder で構築可能");
}
