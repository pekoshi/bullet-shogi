//! CRCL-v2 ResNet 訓練 + one-hot policy + value (joint training)
//!
//! v10_soft (visit dist) は dataset (172 byte PackedSfenWithVisits) が必要だが、
//! その raw hcpe3 format に score 欠落で blocked。
//! 代替: 既存 PSV (40 byte、 best_move 含む) で one-hot policy + value joint training。
//!
//! base: crcl_resnet_psv_train.rs (value-only) を base に policy head 追加。
//! input: ShogiHalfKaHmPolicy (PSV の bestMove から one-hot target 生成)
//! v2 (shogi_layerstack_policy_v2.rs) と同じ policy 訓練 pattern を CRCL-v2 ResNet に統合。
//!
//! 構成:
//!   - inputs: ShogiHalfKaHmPolicy (PSV 40 byte、 既存 aoba_psv/dlsuisho/nnue16b 等で OK)
//!   - l0 (Affine): sparse → Cin*81、 SiLU
//!   - combined = stm_h + ntm_h (perspective-symmetric)
//!   - ResBlock × N (Conv 3x3 + SiLU + residual)
//!   - value head: Affine → 1 + sigmoid + squared_error
//!   - policy head: Affine → POLICY_HIDDEN → 2187 + softmax_CE(one-hot best_move)
//!
//! env var:
//!   CRCL_BLOCKS / CRCL_CHANNELS / CRCL_POLICY_HIDDEN
//!   BULLET_DATA (PSV 40 byte file)
//!   BULLET_BATCH / BULLET_SUPERBATCHES / BULLET_LR / BULLET_POLICY_WEIGHT
//!
//! 実行 (GCP T4 並行訓練、 PUCT 用):
//!   BULLET_DATA=/data/dataset/aoba_psv/aoba_psv_000.bin \
//!   CRCL_BLOCKS=16 CRCL_CHANNELS=96 BULLET_BATCH=48 BULLET_LR=1e-4 \
//!   BULLET_SUPERBATCHES=30 \
//!   cargo run --release --example crcl_resnet_psv_train_policy_onehot -- crcl_v2_joint

use std::sync::Arc;

use bullet_cuda_backend::builder::{Conv2dParams, ConvBuilderExt};
use bullet_lib::{
    acyclib::{device::tensor::Shape, graph::builder::InitSettings},
    game::inputs::{POLICY_TARGET_NAME, ShogiHalfKaHmPolicy, SparseInputType},
    nn::optimiser::{self, AdamWParams},
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

const SCALE: f32 = 600.0;
const POLICY_DIMS: usize = 2187;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "crcl_v2_joint_onehot".to_string());
    let inputs = ShogiHalfKaHmPolicy;

    let num_blocks: usize = std::env::var("CRCL_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let cin: usize = std::env::var("CRCL_CHANNELS").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let policy_hidden: usize = std::env::var("CRCL_POLICY_HIDDEN").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let spatial_size: usize = 9 * 9;
    let dense_size = cin * spatial_size;

    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-4);
    let wdl_val: f32 = std::env::var("BULLET_WDL").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let policy_weight: f32 = std::env::var("BULLET_POLICY_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    eprintln!(
        "[crcl_resnet_psv_train_policy_onehot] blocks={num_blocks} ch={cin} dense={dense_size} \
         policy_hidden={policy_hidden} lr={lr_start} WDL={wdl_val} policy_weight={policy_weight}"
    );

    let mut save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
        SavedFormat::id("v_outw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("v_outb").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_hw").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_hb").quantise::<i16>(255 * 64).round(),
        SavedFormat::id("pol_ow").quantise::<i16>(64).transpose().round(),
        SavedFormat::id("pol_ob").quantise::<i16>(255 * 64).round(),
    ];
    // ResBlock conv weights (raw f32 for PTQ)
    let conv_weight_names: Vec<String> =
        (0..num_blocks).flat_map(|b| [format!("b{b}_w1"), format!("b{b}_w2")]).collect();
    for name in &conv_weight_names {
        save_format.push(SavedFormat::id(name));
    }
    let save_format = save_format;

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

            // value head: dense → 1 + sigmoid + squared_error
            let v_out = builder.new_affine("v_out", dense_size, 1);
            let value_out = v_out.forward(x);
            let value_loss = value_out.sigmoid().squared_error(targets);

            // policy head: dense → POLICY_HIDDEN → 2187 + softmax_CE(one-hot)
            let pol_h = builder.new_affine("pol_h", dense_size, policy_hidden);
            let pol_o = builder.new_affine("pol_o", policy_hidden, POLICY_DIMS);
            let policy_hidden_out = pol_h.forward(x).silu();
            let policy_logits = pol_o.forward(policy_hidden_out);

            // one-hot target (ShogiHalfKaHmPolicy が PSV の best_move から自動生成)
            let policy_target = builder.new_dense_input(POLICY_TARGET_NAME, Shape::new(POLICY_DIMS, 1));
            let per_class_loss = policy_logits.softmax_crossentropy_loss(policy_target);
            let ones_row = builder.new_constant(Shape::new(1, POLICY_DIMS), &vec![1.0f32; POLICY_DIMS]);
            let policy_loss = ones_row.matmul(per_class_loss);

            let total_loss = value_loss + policy_loss * policy_weight;

            (value_out, total_loss)
        });
    *trainer_clone.lock().unwrap() = Some(());

    let mut trainer = built;
    trainer.optimiser.set_params(AdamWParams::default().into());

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
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
        _ => vec!["E:/shogi/p2shogi/data/_psv/aoba_full/aoba_psv_000.bin".to_string()],
    };
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    eprintln!("[crcl_resnet_psv_train_policy_onehot] data files ({}):", data_files_refs.len());
    for p in &data_files_refs {
        eprintln!("  {p}");
    }

    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);
    trainer.run(&schedule, &settings, &data_loader);
}
