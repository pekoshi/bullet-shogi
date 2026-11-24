//! shogi_layerstack_policy_v2 — multi-source training via ShogiPackLoader。
//!
//! Phase 1 baseline (hao_depth9 .bin only) は loss 4.27 / top1 1.4% で plateau。
//! 教師の質と多様性が bottleneck と判断、 .spk (game-pack) 形式の複数 source を
//! mix して学習する v2 を構築。
//!
//! ## v1 との差分
//!
//! - DirectSequentialDataLoader (hao .bin x 10) → ShogiPackLoader::new_concat_multiple
//! - data: combined_v17 (37GB) + gct_hcpe (1.4GB) + suisho5_ek (2.3GB) + mekitto (10.8GB)
//! - arch / hyperparam は v1 同一 (比較可能性のため)
//!
//! BULLET_DATA env var で path 上書き可。

use acyclib::device::tensor::Shape;
use bullet_lib::{
    game::inputs::{ShogiHalfKaHmPolicy, SparseInputType},
    nn::optimiser::{self, RAdamParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

// v7: L1=1024 (was 768)、 容量増加で top1 上限を押し上げ。
const L1_SIZE: usize = 1024;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;
const POLICY_HIDDEN: usize = 768;
const POLICY_DIMS: usize = 2187;
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "halfka_policy_v2".to_string());

    let inputs = ShogiHalfKaHmPolicy;

    let policy_weight: f32 = std::env::var("BULLET_POLICY_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    eprintln!("[v2] policy_weight={policy_weight} lr_start={lr_start}");

    let save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("l2w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l2b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("v_outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("v_outb").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_hw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_hb").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_ow").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_ob").quantise::<i16>(255 * 64).round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(optimiser::RAdam)
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

            let h1 = l1.forward(concat).screlu();
            let h2 = l2.forward(h1).screlu();
            let value_out = v_out.forward(h2);

            let value_loss = value_out.sigmoid().squared_error(targets);

            let pol_h = builder.new_affine("pol_h", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_o = builder.new_affine("pol_o", POLICY_HIDDEN, POLICY_DIMS);
            let policy_hidden = pol_h.forward(concat).screlu();
            let policy_logits = pol_o.forward(policy_hidden);

            let policy_target =
                builder.new_dense_input(bullet_lib::game::inputs::POLICY_TARGET_NAME, Shape::new(POLICY_DIMS, 1));
            let per_class_loss = policy_logits.softmax_crossentropy_loss(policy_target);
            let ones_row = builder.new_constant(Shape::new(1, POLICY_DIMS), &vec![1.0f32; POLICY_DIMS]);
            let policy_loss = ones_row.matmul(per_class_loss);

            let total_loss = value_loss + policy_loss * policy_weight;

            (value_out, total_loss)
        });

    trainer.optimiser.set_params(RAdamParams::default().into());

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(50);
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

    // 多源混合 PSV-40 fixed format。 dlshogi 系 (AobaZero teacher + gct selfplay) を
    // 追加して教師多様化、 interleave で size 偏りを補正。
    let data_files_default: Vec<String> = {
        let mut v = vec![
            // 既存 main
            "E:/shogi/p2shogi/data/bullet/combined_v17.spk".to_string(),
            "E:/shogi/p2shogi/data/bullet/hao_subset.spk".to_string(),
        ];
        // hao_depth9 (一部)
        for i in 0..3 {
            v.push(format!(
                "E:/shogi/shogi_hao_depth9/kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index={i:03}.bin"
            ));
        }
        // AobaZero teacher (PUCT-based、 dlshogi 系)
        for arch in 115..=119 {
            v.push(format!("E:/shogi/p2shogi/data/_psv/aoba_teacher/aobazero_teacher-arch0000{arch}.psv"));
        }
        // dlshogi gct (selfplay-unique 90000..90009)
        for i in 0..10 {
            v.push(format!("E:/shogi/p2shogi/data/_psv/gct/selfplay-unique-9000{i}.psv"));
        }
        v
    };

    let data_files: Vec<String> = match std::env::var("BULLET_DATA") {
        Ok(s) if !s.is_empty() => s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => data_files_default,
    };
    let interleave_batches: usize =
        std::env::var("BULLET_INTERLEAVE_BATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();

    eprintln!("[v2] data files ({}) interleave_batches={}:", data_files_refs.len(), interleave_batches);
    for p in &data_files_refs {
        eprintln!("  {}", p);
    }

    let data_loader = DirectSequentialDataLoader::new(&data_files_refs).with_interleave_batches(interleave_batches);
    trainer.run(&schedule, &settings, &data_loader);
}
