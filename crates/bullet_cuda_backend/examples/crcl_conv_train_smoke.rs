//! CRCL-v2 Conv2d で 1-step training smoke
//!
//! 確認内容:
//!   - GraphBuilder で Conv2d → squared_error(zeros) → ones.matmul で scalar loss を作る
//!   - AdamW Optimiser を建てて forward → backward → update を 1 step 実行
//!   - loss が下がる (weight が target=zeros に向かって動く) ことを確認
//!
//! これで forward + backward (data + filter) + 重み更新がすべて graph IR 経由で動くことを実証。
//!
//! 実行:
//!   cargo run -p bullet_cuda_backend --example crcl_conv_train_smoke --release

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
use bullet_cuda_backend::{
    CudaDevice,
    builder::{Conv2dParams, ConvBuilderExt},
};

fn main() {
    println!("=== CRCL-v2 Conv2d 1-step training smoke ===");

    let p = Conv2dParams {
        in_channels: 32,
        out_channels: 32,
        spatial_h: 9,
        spatial_w: 9,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };
    let in_size = p.input_size();
    let out_size = p.output_size();
    let weight_size = p.weight_size();

    let builder = GraphBuilder::default();
    let input = builder.new_dense_input("input", Shape::new(in_size, 1));
    let weight =
        builder.new_weights("weight", Shape::new(weight_size, 1), InitSettings::Normal { mean: 0.0, stdev: 0.1 });
    let target = builder.new_dense_input("target", Shape::new(out_size, 1));

    let conv = input.conv2d(weight, p);

    let sq_err = conv.squared_error(target);
    // ones で scalar に reduce (build 時 ReduceAcrossBatch も自動追加される)
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _root = ones.matmul(sq_err);

    let device = CudaDevice::new(0).expect("CudaDevice::new");
    let graph = builder.build(device);
    println!("graph built");

    let params = RangerParams { min_weight: -1000.0, max_weight: 1000.0, ..Default::default() };
    let mut optimiser = Optimiser::<_, _, Ranger<_>>::new(graph, params).expect("Optimiser::new");

    // input / target を all-ones / zeros で固定
    {
        let g = optimiser.graph.primary_mut();
        let in_ref = g.get_input("input");
        let mut in_dense = in_ref.dense_mut();
        in_dense.load_from_slice(Some(1), &vec![1.0f32; in_size]).expect("input load");
    }
    {
        let g = optimiser.graph.primary_mut();
        let tg_ref = g.get_input("target");
        let mut tg_dense = tg_ref.dense_mut();
        tg_dense.load_from_slice(Some(1), &vec![0.0f32; out_size]).expect("target load");
    }

    let loss_before = {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads");
        let loss = g.forward().expect("forward");
        loss
    };
    println!("loss BEFORE step: {loss_before:.6}");

    // backward + update
    {
        let g = optimiser.graph.primary_mut();
        g.backward().expect("backward");
    }
    // Ranger (RAdam+Lookahead) は 1 step では RAdam warmup 中 (rho_t<=4) で SGD-like に動くため、
    // raw gradient のスケール (cin*hout*wout ≈ 288 * 81 ~ 几百 オーダー) で update してしまう。
    // lr を 1e-5 まで落として overshoot を防ぐ。
    optimiser.update(1.0, 1e-5).expect("optimiser update");

    let loss_after = {
        let g = optimiser.graph.primary_mut();
        g.zero_grads().expect("zero_grads 2");
        g.forward().expect("forward 2")
    };
    println!("loss AFTER  step: {loss_after:.6}");

    assert!(
        loss_after < loss_before,
        "loss should decrease after AdamW step (before={loss_before} after={loss_after})"
    );
    let drop_ratio = (loss_before - loss_after) / loss_before;
    println!("loss drop ratio: {:.2}%", 100.0 * drop_ratio);
    println!("Conv2d 1-step training smoke PASSED");
}
