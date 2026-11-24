//! shogi_halfka_policy_value_only — debug 用、 ShogiHalfKaHmPolicy 入力 +
//! 標準 .build() / .loss_fn() による value-only 訓練。 policy aux は無視 (dense
//! input が宣言されないだけで、 dataloader 側は 0 で埋めるが loss に未接続)。
//!
//! shogi_layerstack_policy.rs の build_custom 経路で loss 不動の原因切り分けに使う:
//! - ここで loss が下がる → build_custom 経路の問題
//! - ここでも下がらない → ShogiHalfKaHmPolicy or データ側の問題

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
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "value_only_debug".to_string());
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);

    let inputs = ShogiHalfKaHmPolicy;

    let save_format = vec![
        SavedFormat::id("ftw").quantise::<i16>(255).round(),
        SavedFormat::id("ftb").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(255 * 64).round(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(optimiser::RAdam)
        .save_format(&save_format)
        .use_threads(4)
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let ft = builder.new_affine("ft", inputs.num_inputs(), L1_SIZE);
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, 1);

            let stm_h = ft.forward(stm_inputs).crelu();
            let ntm_h = ft.forward(ntm_inputs).crelu();
            let concat = stm_h.concat(ntm_h);
            l1.forward(concat)
        });

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let batches_per_sb: usize = std::env::var("BULLET_BATCHES_PER_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(100);

    eprintln!(
        "[value_only_debug] lr={lr_start} superbatches={superbatches} batch={batch_size} batches_per_sb={batches_per_sb}"
    );

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
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.3, step: 5 },
        save_rate: 1,
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
