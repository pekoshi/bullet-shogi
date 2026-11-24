//! debug_halfka_policy_input — debug_halfka_direct から **input 型のみ**
//! `ShogiHalfKA_hm` → `ShogiHalfKaHmPolicy` に置換。 他は全て同一。
//!
//! - 動けば: HalfKaHmPolicy delegate は OK、 我々の元 example の他の要素が原因
//! - 動かなければ: HalfKaHmPolicy 自体に問題、 ShogiHalfKA_hm delegate 経路が grad 不通

use bullet_lib::{
    game::inputs::{ShogiHalfKaHmPolicy, SparseInputType},
    nn::optimiser,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

const L1_SIZE: usize = 768;
const L2_SIZE: usize = 16;
const L3_SIZE: usize = 32;
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "halfka_policy_input".to_string());
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);

    // ★ direct と違うのはここだけ
    let inputs = ShogiHalfKaHmPolicy;

    let save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("l2w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l2b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("outb").quantise::<i16>(255 * 64).round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(optimiser::AdamW)
        .save_format(&save_format)
        .use_threads(4)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1_SIZE);
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let out = builder.new_affine("out", L3_SIZE, 1);

            let stm_h = l0.forward(stm_inputs).screlu();
            let ntm_h = l0.forward(ntm_inputs).screlu();
            let combined = stm_h.concat(ntm_h);
            let h1 = l1.forward(combined).screlu();
            let h2 = l2.forward(h1).screlu();
            out.forward(h2)
        });

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let batches_per_sb: usize = std::env::var("BULLET_BATCHES_PER_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(100);

    eprintln!("[debug_halfka_policy_input] lr={lr_start} superbatches={superbatches}");

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
        save_rate: 100,
        log_rate: 100,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
        on_checkpoint_saved: None,
    };

    let data_files: Vec<String> = (0..3)
        .map(|i| {
            format!(
                "E:\\shogi\\shogi_hao_depth9\\kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index={i:03}.bin"
            )
        })
        .collect();
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);

    trainer.run(&schedule, &settings, &data_loader);
}
