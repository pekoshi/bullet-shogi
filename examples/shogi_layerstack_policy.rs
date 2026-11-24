//! shogi_layerstack_policy — HalfKA_hm + value LayerStack + 2187-policy head の
//! multi-head 訓練 example (Phase 1、 one-hot policy 教師)。
//!
//! 元 `simple.rs` を base に、 ShogiHalfKaHmPolicy input + build_custom で
//! multi-head graph を構築する。
//!
//! 訓練 graph:
//!
//! ```text
//! HalfKA_hm sparse (73,305) ─┐
//!                            ├─ FT ─┐
//! HalfKA_hm sparse (NSTM) ───┘     │
//!                                  ├─ stm/ntm concat (2L1)
//!                                  ↓
//!                       ┌──────────┴──────────┐
//!                       │                     │
//!                  value branch          policy branch
//!                  (LayerStack)          (2L1 → ph → 2187)
//!                       │                     │
//!                       ↓                     ↓
//!                 value_out (1)          policy_logits (2187)
//!                       │                     │
//!                       ↓                     ↓
//!                value MSE loss      softmax CE loss
//!                       └──────────┬──────────┘
//!                                  ↓
//!                       total = v + λ × p
//! ```
//!
//! ## 教師
//!
//! - value: `output.sigmoid().squared_error(targets)`、 targets は wdl + sigmoid(cp/SCALE)
//! - policy: `policy_logits.softmax_crossentropy_loss(policy_target)`、
//!   policy_target は ShogiHalfKaHmPolicy が move16 から作る 2187-dim one-hot
//!
//! ## 注意
//!
//! - Phase 1 は one-hot 教師 (PSV の move16 = bestmove)。 visit dist 教師は
//!   Phase 2 で扱う (hcpe3 拡張)。
//! - 既存 hao_depth9 .pack/.spk をそのまま流用可能。
//! - L1 / バケット数 / arch は P0 (1536-32-96) より小さい baseline。 動作確認後
//!   shogi_layerstack.rs の本体パターンへ統合する想定。

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

// =============================================================================
// arch (Phase 1 baseline、 P0 比 1/2)
// =============================================================================

/// FT 出力次元 per side (concat 後は 2 倍)。
const L1_SIZE: usize = 768;
/// value 中間 (前段)。
const L2_SIZE: usize = 16;
/// value 中間 (後段)。
const L3_SIZE: usize = 32;
/// policy 中間。
const POLICY_HIDDEN: usize = 512;
/// policy 出力 (dlshogi `make_move_label`)。
const POLICY_DIMS: usize = 2187;

/// 評価 cp → sigmoid スケール。
const SCALE: f32 = 600.0;

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "halfka_policy_v1".to_string());

    let inputs = ShogiHalfKaHmPolicy;

    // debug 用: env vars で hyperparam 上書き。
    let policy_weight: f32 = std::env::var("BULLET_POLICY_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    eprintln!("[shogi_layerstack_policy] policy_weight={policy_weight} lr_start={lr_start}");

    // `new_affine("name", ...)` は weight = `nameW`、 bias = `nameB` を作る (Bullet 命名規則)。
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
        // RAdam を使う場合は **build 後に明示的に set_params** が必要 (debug 結果
        // [[session_state_20260518_halfka_pivot]] 参照)。 そうしないと ValueTrainerBuilder の
        // Default::default() が壊れた params を渡してしまい全く学習しない。
        .optimiser(optimiser::RAdam)
        .save_format(&save_format)
        .use_threads(4)
        // multi-head loss は build_custom 内で組み立てる。
        // output_buckets なし → closure 引数は (builder, (stm, ntm), targets)。
        .build_custom(|builder, (stm_inputs, ntm_inputs), targets| {
            // ---- shared FT ----
            // 活性は SCReLU (shogi_simple / shogi_layerstack 互換)、 CReLU だと
            // gradient が早く飽和して loss が動かない問題を踏んだ。
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1_SIZE);
            let stm_h = l0.forward(stm_inputs).screlu();
            let ntm_h = l0.forward(ntm_inputs).screlu();
            let concat = stm_h.concat(ntm_h); // [2*L1, 1]

            // ---- value branch (4 層、 shogi_simple 互換) ----
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let v_out = builder.new_affine("v_out", L3_SIZE, 1);

            let h1 = l1.forward(concat).screlu();
            let h2 = l2.forward(h1).screlu();
            let value_out = v_out.forward(h2); // [1, 1]

            let value_loss = value_out.sigmoid().squared_error(targets);

            // ---- policy branch ----
            // 共有 FT 出力 (concat) → 1 段 MLP → 2187 logits
            let pol_h = builder.new_affine("pol_h", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_o = builder.new_affine("pol_o", POLICY_HIDDEN, POLICY_DIMS);
            let policy_hidden = pol_h.forward(concat).screlu();
            let policy_logits = pol_o.forward(policy_hidden); // [POLICY_DIMS, 1]

            // policy_target は ShogiHalfKaHmPolicy::fill_auxiliary_dense が 0/1 で
            // 埋める one-hot 2187-dim dense input。
            let policy_target =
                builder.new_dense_input(bullet_lib::game::inputs::POLICY_TARGET_NAME, Shape::new(POLICY_DIMS, 1));
            // softmax_crossentropy_loss は per-class loss を返す ([POLICY_DIMS, 1])。
            // mnist example と同様、 行ベクトル ones × losses の matmul で scalar 化。
            let per_class_loss = policy_logits.softmax_crossentropy_loss(policy_target);
            let ones_row = builder.new_constant(Shape::new(1, POLICY_DIMS), &vec![1.0f32; POLICY_DIMS]);
            let policy_loss = ones_row.matmul(per_class_loss); // [1, 1]

            // ---- 合成 loss ----
            let total_loss = value_loss + policy_loss * policy_weight;

            (value_out, total_loss)
        });

    // ★ RAdam の params を明示的に設定 (shogi_layerstack 同様)。
    trainer.optimiser.set_params(RAdamParams::default().into());

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let batch_size: usize = std::env::var("BULLET_BATCH").ok().and_then(|s| s.parse().ok()).unwrap_or(8_192);
    let batches_per_sb: usize =
        std::env::var("BULLET_BATCHES_PER_SB").ok().and_then(|s| s.parse().ok()).unwrap_or(6_104);

    let schedule = TrainingSchedule {
        net_id: net_id.clone(),
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size,
            batches_per_superbatch: batches_per_sb,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.5 },
        // shogi_simple と同じ slow decay (debug 結果: gamma 0.3/step 5 でも loss 推移は
        // 似ているが、 baseline と揃えて再現性確保)
        lr_scheduler: lr::StepLR { start: lr_start, gamma: 0.992, step: 1 },
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

    // hao_depth9 thread_000..009 を 既存 .pack 形式で読む想定。
    // パスは BULLET_DATA 環境変数で上書き可能。
    let data_files_default: Vec<String> = (0..10)
        .map(|i| {
            format!(
                "E:\\shogi\\shogi_hao_depth9\\kifu.tag=train.depth=9.num_positions=1000000000.start_time=1695340981.thread_index={i:03}.bin"
            )
        })
        .collect();

    let data_files: Vec<String> = match std::env::var("BULLET_DATA") {
        Ok(s) if !s.is_empty() => s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => data_files_default,
    };

    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    // PackedSfenValue は CanBeDirectlySequentiallyLoaded を impl 済 (packed_sfen.rs:446)
    // なので、 .bin file (PSV 40byte 連結) をそのまま直接読める。
    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);

    trainer.run(&schedule, &settings, &data_loader);
}
