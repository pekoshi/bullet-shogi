//! v20_clean — Value-only fine-tune on clean_train_v1.spk
//!
//! 目的:
//!   v16 の value head は visit-only data で score 教師なし → flat (mean 0、 stdev <0.001)
//!   AobaNNUE 評価値付け 12h vs. 既存 clean_train_v1.spk (24 GB、 454M positions、 cp 付き) 直接利用
//!
//! 構成:
//!   - inputs: ShogiHalfKA_hm (PackedSfenValue、 40 byte/record)
//!   - data: clean_train_v1.spk (ShogiPackLoader)
//!   - arch: v16 と同じ (L1=2048, L2=32, L3=96)、 ただし value head のみ
//!   - WDL: ConstantWDL(0.0) = pure score 教師 (game outcome 不要)
//!   - resume from v16 ckpt: 同じ weight 名 (l0/l1/l2/v_out) のみ load、 policy weights は v20 にないので無視
//!
//! 訓練後の merge:
//!   v20 ckpt (l0/l1/l2/v_out) + v16 ckpt の policy 部分 (pol_*) を別 script で合体 → final ckpt
//!
//! 使い方:
//!   BULLET_DATA=E:/shogi/p2shogi/data/bullet/clean_train_v1.spk \
//!   BULLET_SUPERBATCHES=20 BULLET_BATCH=16384 \
//!   cargo run --release --example shogi_layerstack_value_v20_clean -- v20_clean

use bullet_lib::{
    game::inputs::{ShogiHalfKA_hm, SparseInputType},
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
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "v20_clean".to_string());
    let inputs = ShogiHalfKA_hm;

    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let wdl_val: f32 = std::env::var("BULLET_WDL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    eprintln!("[v20_clean] lr_start={lr_start} WDL={wdl_val} L1={L1_SIZE} (value-only)");

    let save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("l1w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l1b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("l2w").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("l2b").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("v_outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("v_outb").quantise::<i16>(255 * 64).round(),
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

            (value_out, value_loss)
        });

    trainer.optimiser.set_params(AdamWParams::default().into());

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(16384);
    let batches_per_sb: usize =
        std::env::var("BULLET_BATCHES_PER_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(6104);

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: batches_per_sb,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl_val },
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.992, step: 1 },
        save_rate: 5,
        log_rate: 100,
    };
    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "E:/shogi/p2shogi/checkpoints",
        batch_queue_size: 32,
        on_checkpoint_saved: None,
    };

    let data_files: Vec<String> = match std::env::var("BULLET_DATA") {
        Ok(s) if !s.is_empty() => s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => vec!["E:/shogi/p2shogi/data/bullet/clean_train_v1.spk".to_string()],
    };
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    eprintln!("[v20_clean] data files ({}):", data_files_refs.len());
    for p in &data_files_refs {
        eprintln!("  {p}");
    }

    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);
    trainer.run(&schedule, &settings, &data_loader);
}
