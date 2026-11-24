//! Phase 2 graph IR 統合 最終確認: multi-step train で loss decreasing を確認。
//!
//! train_step_mixed_smoke を 100 step に拡張し、 IV案 + graph IR + Ranger の
//! training cycle が実用的に loss を下げられることを実証する。
//!
//! f32 path と並べて loss curve を比較 (同 init、 同 data、 同 lr で multi-step)。

use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        builder::{GraphBuilder, InitSettings},
        like::GraphLike,
    },
    trainer::optimiser::{
        Optimiser,
        ranger::{Ranger, RangerParams},
    },
};
use bullet_cuda_backend::CudaDevice;

fn train_run(mixed: bool, n_steps: usize, lr: f32) -> Vec<f32> {
    let in_dim = 16usize;
    let hid_dim = 32usize;
    let out_dim = 4usize;
    let batch = 8usize;

    let builder = GraphBuilder::<bullet_cuda_backend::CudaMarker>::default();
    let w1 = builder.new_weights("w1", Shape::new(hid_dim, in_dim), InitSettings::Normal { mean: 0.0, stdev: 0.2 });
    let w2 = builder.new_weights("w2", Shape::new(out_dim, hid_dim), InitSettings::Normal { mean: 0.0, stdev: 0.2 });
    let x = builder.new_dense_input("x", Shape::new(in_dim, 1));
    let target = builder.new_dense_input("target", Shape::new(out_dim, 1));

    let hidden = if mixed { w1.matmul_mixed(x) } else { w1.matmul(x) };
    let active = if mixed { hidden.silu_mixed() } else { hidden.silu() };
    let y_pred = if mixed { w2.matmul_mixed(active) } else { w2.matmul(active) };

    let sq_err = y_pred.squared_error(target);
    let ones = builder.new_constant(Shape::new(1, out_dim), &vec![1.0f32; out_dim]);
    let _root = ones.matmul(sq_err);

    let device = CudaDevice::new(0).expect("device");
    let graph = builder.build(device);

    let params = RangerParams { min_weight: -1000.0, max_weight: 1000.0, ..Default::default() };
    let mut optimiser = Optimiser::<_, _, Ranger<_>>::new(graph, params).expect("Optimiser::new");

    // 固定 data set
    use std::f32::consts::PI;
    let x_host: Vec<f32> = (0..in_dim * batch).map(|i| (i as f32 * 0.0231 * PI).cos() * 0.4).collect();
    let t_host: Vec<f32> = (0..out_dim * batch).map(|i| (i as f32 * 0.0149 * PI).sin() * 0.6).collect();
    {
        let g = optimiser.graph.primary_mut();
        g.get_input("x").dense_mut().load_from_slice(Some(batch), &x_host).expect("x");
        g.get_input("target").dense_mut().load_from_slice(Some(batch), &t_host).expect("t");
    }

    let mut losses = Vec::with_capacity(n_steps + 1);

    for step in 0..n_steps {
        let loss = {
            let g = optimiser.graph.primary_mut();
            g.zero_grads().expect("zero_grads");
            g.forward().expect("forward")
        };
        if step == 0 || (step + 1) % 10 == 0 {
            losses.push(loss);
        }
        {
            let g = optimiser.graph.primary_mut();
            g.backward().expect("backward");
        }
        optimiser.update(1.0, lr).expect("update");
    }
    // final loss
    let final_loss = {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads final");
        g.forward().expect("forward final")
    };
    losses.push(final_loss);
    losses
}

fn main() {
    println!("=== train multi-step mixed precision smoke ===");
    let n_steps = 100usize;
    let lr = 1e-3f32;

    println!("running f32 path ({} steps, lr={}) ...", n_steps, lr);
    let f32_losses = train_run(false, n_steps, lr);
    println!("running mixed precision path ({} steps, lr={}) ...", n_steps, lr);
    let mixed_losses = train_run(true, n_steps, lr);

    println!("\n  step | f32 loss   | mixed loss");
    println!("  -----+------------+------------");
    let n_print = f32_losses.len();
    for (i, (a, b)) in f32_losses.iter().zip(mixed_losses.iter()).enumerate() {
        let step_label = if i == n_print - 1 {
            format!("{:>4}", n_steps)
        } else if i == 0 {
            "   0".to_string()
        } else {
            format!("{:>4}", i * 10)
        };
        println!("  {}  | {:>10.6} | {:>10.6}", step_label, a, b);
    }

    let f32_initial = f32_losses[0];
    let f32_final = *f32_losses.last().unwrap();
    let mixed_initial = mixed_losses[0];
    let mixed_final = *mixed_losses.last().unwrap();

    let f32_drop = (f32_initial - f32_final) / f32_initial * 100.0;
    let mixed_drop = (mixed_initial - mixed_final) / mixed_initial * 100.0;
    println!("\nf32   loss drop: {:.2}%  ({:.4} → {:.4})", f32_drop, f32_initial, f32_final);
    println!("mixed loss drop: {:.2}%  ({:.4} → {:.4})", mixed_drop, mixed_initial, mixed_final);

    if mixed_final < mixed_initial {
        println!("\nmixed precision multi-step train PASSED: loss decreased over {} steps", n_steps);
    } else {
        println!("\nWARN: mixed loss did not decrease, lr or init を見直す必要");
    }
}
