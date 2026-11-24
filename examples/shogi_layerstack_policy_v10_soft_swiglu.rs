//! shogi_layerstack_policy_v10_soft_swiglu — SwiGLU policy head 版
//!
//! v10_soft からの差分:
//! - policy head の hidden を `screlu(pol_h)` → `SwiGLU(pol_w1, pol_w3)` に置換
//! - POLICY_HIDDEN は 2/3 縮小 (2048 → 1365、 16 alignment) で param 数を同等に
//!
//! SwiGLU(x) = (SiLU(W1 x)) ⊙ (W3 x)
//!   SiLU(z) = z * sigmoid(z)  (= bullet 既存 op の copy + sigmoid + Mul)
//!   ⊙       = element-wise multiply  (= `Node * Node` で `concat.pairwise_mul()`)
//!
//! 量子化: SiLU は i16 量子化困難 → 推論側は f32 / GPU 前提
//! save_format: pol_w1 / pol_w3 / pol_o の 3 weight slot (i16 quantise は実行時 OK)

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
/// SwiGLU は W1 と W3 の 2 行列を持つため、 param 同等にするには 2/3 縮小。
/// 2048 * 2/3 = 1365.33 → 1376 (16 alignment、 上方丸め、 SIMD efficient)。
const POLICY_HIDDEN: usize = 1376;
const POLICY_DIMS: usize = 2187;
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "v10_swiglu".to_string());

    let inputs = ShogiHalfKaHmPolicySoft;

    let policy_weight: f32 = std::env::var("BULLET_POLICY_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    eprintln!("[v10_swiglu] policy_weight={policy_weight} lr_start={lr_start}");

    let save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("l2w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l2b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("v_outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("v_outb").quantise::<i16>(255 * 64).round(),
        // SwiGLU policy head の 3 weight (W1 / W3 / output)
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

            // Value head (v10_soft と同じ)
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let v_out = builder.new_affine("v_out", L3_SIZE, 1);
            let h1 = l1.forward(concat).screlu();
            let h2 = l2.forward(h1).screlu();
            let value_out = v_out.forward(h2);
            let value_loss = value_out.sigmoid().squared_error(targets);

            // SwiGLU policy head
            //   gate_pre = W1(concat)
            //   value_pre = W3(concat)
            //   swiglu_out = SiLU(gate_pre) * value_pre
            //              = (gate_pre * sigmoid(gate_pre)) * value_pre
            //   logits = W_o(swiglu_out)
            let pol_w1 = builder.new_affine("pol_w1", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_w3 = builder.new_affine("pol_w3", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_o = builder.new_affine("pol_o", POLICY_HIDDEN, POLICY_DIMS);

            let gate_pre = pol_w1.forward(concat);
            let value_pre = pol_w3.forward(concat);
            // SiLU(gate_pre) = gate_pre * sigmoid(gate_pre)
            let silu_gate = gate_pre.copy() * gate_pre.sigmoid();
            // SwiGLU = SiLU(gate) * value
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

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(200);
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
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
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
        _ => vec!["E:/shogi/p2shogi/data/_psv/gct_strict_visits_177M_shuf.bin".to_string()],
    };
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    eprintln!("[v10_swiglu] data files ({}):", data_files_refs.len());
    for p in &data_files_refs {
        eprintln!("  {p}");
    }

    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);
    trainer.run(&schedule, &settings, &data_loader);
}
