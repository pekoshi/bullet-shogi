//! CRCL-v2 ResNet 訓練 (実 PSV dataset 投入版)
//!
//! 既存 NNUE 訓練 path (ShogiHalfKA_hm sparse + DirectSequentialDataLoader) を CRCL-v2 ResNet に接続。
//!
//! 構成:
//!   - inputs: ShogiHalfKA_hm sparse、 dual_perspective (stm + ntm)
//!   - l0 (Affine): sparse 73,305 → dense (Cin*81)、 SiLU
//!   - combined = stm_h + ntm_h  (perspective-symmetric)
//!   - ResBlock × N (Conv 3x3 + SiLU + +residual + SiLU)
//!   - value head (Affine Cin*81 → 1) + sigmoid
//!   - loss = squared_error(target)
//!
//! env var:
//!   CRCL_BLOCKS    (default 4)    ResNet block 数
//!   CRCL_CHANNELS  (default 64)   filter ch
//!   BULLET_DATA                   PSV file path (;-separated for multi)
//!   BULLET_SUPERBATCHES  (default 5)
//!   BULLET_BATCH         (default 4096)  T4 16GB なら ch=192 で batch=4096 余裕、 ch=64 なら 16384 可
//!
//! 実行 (1050 Ti smoke):
//!   BULLET_DATA=E:/shogi/p2shogi/data/bullet/clean_train_v1.spk \
//!   CRCL_BLOCKS=4 CRCL_CHANNELS=64 BULLET_BATCH=4096 BULLET_SUPERBATCHES=2 \
//!   cargo run --release --example crcl_resnet_psv_train -- v_smoke
//!
//! 実行 (GCP T4 本格、 CRCL-v2 設計):
//!   BULLET_DATA=/data/dataset/aoba_psv/*.spk \
//!   CRCL_BLOCKS=16 CRCL_CHANNELS=192 BULLET_BATCH=4096 BULLET_SUPERBATCHES=100 \
//!   cargo run --release --example crcl_resnet_psv_train -- crcl_v2_stage1

use std::sync::Arc;

use bullet_cuda_backend::builder::{Conv2dParams, ConvBuilderExt};
use bullet_lib::{
    acyclib::{device::tensor::Shape, graph::builder::InitSettings},
    game::inputs::{ShogiHalfKA_hm, SparseInputType},
    nn::optimiser::{self, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "crcl_v_smoke".to_string());
    let inputs = ShogiHalfKA_hm;

    let num_blocks: usize = std::env::var("CRCL_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let cin: usize = std::env::var("CRCL_CHANNELS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let spatial_size: usize = 9 * 9;
    let dense_size = cin * spatial_size;

    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    let wdl_val: f32 = std::env::var("BULLET_WDL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    eprintln!("[crcl_resnet_psv_train] blocks={num_blocks} ch={cin} dense={dense_size} lr={lr_start} WDL={wdl_val}");

    let mut save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("v_outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("v_outb").quantise::<i16>(255 * 64).round(),
    ];
    // ResBlock conv weights を raw f32 で raw.bin にも保存 (PTQ tool で読む用)
    // raw.bin format = saved_format 順の f32 bytes 直接 concat (ID なし)
    let conv_weight_names: Vec<String> =
        (0..num_blocks).flat_map(|b| [format!("b{b}_w1"), format!("b{b}_w2")]).collect();
    for name in &conv_weight_names {
        save_format.push(SavedFormat::id(name));
    }
    let save_format = save_format;

    // dual_perspective sparse input + custom arch
    let conv_p = Conv2dParams {
        in_channels: cin as i32,
        out_channels: cin as i32,
        spatial_h: 9,
        spatial_w: 9,
        kernel_size: 3,
        padding: 1,
        stride: 1,
    };

    let trainer = Arc::new(std::sync::Mutex::new(None));
    let trainer_clone = trainer.clone();

    let built = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(optimiser::AdamW)
        .save_format(&save_format)
        .use_threads(4)
        .build_custom(move |builder, (stm_inputs, ntm_inputs), targets| {
            // l0: sparse → dense (Cin*81)
            let l0 = builder.new_affine("l0", inputs.num_inputs(), dense_size);
            let stm_h = l0.forward(stm_inputs).silu();
            let ntm_h = l0.forward(ntm_inputs).silu();
            // perspective-symmetric: 加算
            let mut x = stm_h + ntm_h;

            // ResBlock chain
            for b in 0..num_blocks {
                let w1 = builder.new_weights(
                    &format!("b{b}_w1"),
                    Shape::new((cin * cin * 3 * 3) as usize, 1),
                    InitSettings::Normal { mean: 0.0, stdev: 0.05 },
                );
                let w2 = builder.new_weights(
                    &format!("b{b}_w2"),
                    Shape::new((cin * cin * 3 * 3) as usize, 1),
                    InitSettings::Normal { mean: 0.0, stdev: 0.05 },
                );
                let residual = x;
                let c1 = x.conv2d(w1, conv_p).silu();
                let c2 = c1.conv2d(w2, conv_p);
                x = (c2 + residual).silu();
            }

            // value head: dense → 1
            let v_out = builder.new_affine("v_out", dense_size, 1);
            let value_out = v_out.forward(x);
            let value_loss = value_out.sigmoid().squared_error(targets);

            (value_out, value_loss)
        });
    *trainer_clone.lock().unwrap() = Some(());

    let mut trainer = built;
    trainer.optimiser.set_params(AdamWParams::default().into());

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(4096);
    let batches_per_sb: usize =
        std::env::var("BULLET_BATCHES_PER_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);

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
        log_rate: 50,
    };
    let out_dir = std::env::var("CRCL_CKPT_DIR").unwrap_or_else(|_| "/data/ckpts".to_string());
    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: out_dir.leak(),
        batch_queue_size: 16,
        on_checkpoint_saved: None,
    };

    let data_files: Vec<String> = match std::env::var("BULLET_DATA") {
        Ok(s) if !s.is_empty() => s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => vec!["E:/shogi/p2shogi/data/bullet/clean_train_v1.spk".to_string()],
    };
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    eprintln!("[crcl_resnet_psv_train] data files ({}):", data_files_refs.len());
    for p in &data_files_refs {
        eprintln!("  {p}");
    }

    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);
    trainer.run(&schedule, &settings, &data_loader);
}
