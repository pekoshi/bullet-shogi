//! shogi_layerstack_policy_v17_linear_wdl — value 学習修正版
//!
//! v16 (ConstantWDL=0.5) では target 固定で value 学習不能。
//! v17 では LinearWDL(start=0.0, end=1.0) で sb 進行で score → outcome 推移、
//! value head が局面別 score を学習する。
//!
//! arch は v16 と完全同じ (PH=512 + LeakyReLU value)。

use acyclib::device::tensor::Shape;
use bullet_lib::{
    game::inputs::{POLICY_VISIT_DIST_NAME, ShogiHalfKaHmPolicySoft, SparseInputType},
    nn::optimiser::{self, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

const L1_SIZE: usize = 2048;
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 96;
const POLICY_HIDDEN: usize = 512;
const POLICY_DIMS: usize = 2187;
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "v17_linear_wdl".to_string());
    let inputs = ShogiHalfKaHmPolicySoft;
    let policy_weight: f32 = std::env::var("BULLET_POLICY_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let wdl_start: f32 = std::env::var("BULLET_WDL_START").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let wdl_end: f32 = std::env::var("BULLET_WDL_END").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    eprintln!("[v17_linear_wdl] policy_weight={policy_weight} lr_start={lr_start} wdl=[{wdl_start}, {wdl_end}]");

    let save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("l2w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l2b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("v_outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("v_outb").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_w1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_w1b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_w3w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_w3b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_ow").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_ob").quantise::<i16>(255 * 64).round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(optimiser::AdamW)
        .save_format(&save_format)
        .use_threads(4)
        .build_custom(|builder, (stm_inputs, ntm_inputs), targets| {
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1_SIZE);
            let stm_h = l0.forward(stm_inputs).screlu();
            let ntm_h = l0.forward(ntm_inputs).screlu();
            let concat = stm_h.concat(ntm_h);

            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let v_out = builder.new_affine("v_out", L3_SIZE, 1);
            let h1 = l1.forward(concat).leaky_relu();
            let h2 = l2.forward(h1).leaky_relu();
            let value_out = v_out.forward(h2);
            let value_loss = value_out.sigmoid().squared_error(targets);

            let pol_w1 = builder.new_affine("pol_w1", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_w3 = builder.new_affine("pol_w3", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_o = builder.new_affine("pol_o", POLICY_HIDDEN, POLICY_DIMS);
            let gate_pre = pol_w1.forward(concat);
            let value_pre = pol_w3.forward(concat);
            let silu_gate = gate_pre.silu();
            let swiglu_out = silu_gate * value_pre;
            let policy_logits = pol_o.forward(swiglu_out);

            let policy_target = builder.new_dense_input(POLICY_VISIT_DIST_NAME, Shape::new(POLICY_DIMS, 1));
            let per_class_loss = policy_logits.softmax_crossentropy_loss(policy_target);
            let ones_row = builder.new_constant(Shape::new(1, POLICY_DIMS), &vec![1.0f32; POLICY_DIMS]);
            let policy_loss = ones_row.matmul(per_class_loss);

            let total_loss = value_loss + policy_loss * policy_weight;
            (value_out, total_loss)
        });

    trainer.optimiser.set_params(AdamWParams::default().into());

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let batches_per_sb: usize = std::env::var("BULLET_BATCHES_PER_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(200);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: batches_per_sb,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::LinearWDL { start: wdl_start, end: wdl_end }, // ★ outcome 重視へ推移
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.992, step: 1 },
        save_rate: 5,
        log_rate: 100,
    };
    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
        on_checkpoint_saved: None,
    };

    let data_files: Vec<String> = match std::env::var("BULLET_DATA") {
        Ok(s) if !s.is_empty() => s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => vec!["E:/shogi/p2shogi/data/_psv/gct_strict_visits_50M_shuf.bin".to_string()],
    };
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);
    trainer.run(&schedule, &settings, &data_loader);
}
