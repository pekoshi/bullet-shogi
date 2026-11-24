//! v21 — Attention-Augmented NNUE (1050 Ti 水匠5 勝ち越し狙い arch)
//!
//! 目的:
//!   v16/v20 系列の FC stack で plateau (top-1 60.95%) を突破。
//!   Conv なし (= ResNet なし) で表現力 up する path = Self-Attention block を NNUE に組込み。
//!
//! 構成:
//!   Input: HalfKAv2_hm sparse → FT (L1 = D_MODEL * NUM_TOKENS = 128 * 16 = 2048)
//!     ↓ reshape to (D_MODEL=128, NUM_TOKENS=16)
//!     ↓ Self-Attention block × ATTN_LAYERS (Q/K/V = Affine、 softmax(QK^T) @ V、 output proj)
//!     ↓ reshape back to (L1)
//!     ↓ 既存 value head (LayerStack L2=32, L3=96, LeakyReLU) + policy head (SwiGLU PH=512)
//!
//! 期待:
//!   - policy top-1 65-70% (plateau 60.95% 突破)
//!   - value head の局面間 std up (Attention で駒間関係を直接学習)
//!   - 1050 Ti NPS: forward 比 v16 ×1.3、 batch=32 で 3500 leaf/sec
//!
//! 使い方:
//!   BULLET_DATA=...visit dist data.bin BULLET_SUPERBATCHES=20 \
//!     cargo run --release --example shogi_layerstack_value_v21_attention -- v21_attn

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
use acyclib::device::tensor::Shape;

const D_MODEL: usize = 128;
const NUM_TOKENS: usize = 16;
const L1_SIZE: usize = D_MODEL * NUM_TOKENS; // = 2048
const L2_SIZE: usize = 32;
const L3_SIZE: usize = 96;
const POLICY_HIDDEN: usize = 512;
const POLICY_DIMS: usize = 2187;
const SCALE: f32 = 600.0;
const ATTN_LAYERS: usize = 2;
const ATTN_SCALE: f32 = 0.08838834764; // = 1 / sqrt(128)

fn main() {
    let net_id = std::env::args().nth(1).unwrap_or_else(|| "v21_attn".to_string());
    let inputs = ShogiHalfKaHmPolicySoft;

    let policy_weight: f32 = std::env::var("BULLET_POLICY_WEIGHT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let lr_start: f32 = std::env::var("BULLET_LR").ok().and_then(|s| s.parse().ok()).unwrap_or(1e-3);
    eprintln!("[v21_attn] policy_weight={policy_weight} lr_start={lr_start} L1={L1_SIZE} d_model={D_MODEL} tokens={NUM_TOKENS} attn_layers={ATTN_LAYERS}");

    let mut save_format = vec![
        SavedFormat::id("l0w").quantise::<i16>(255).round(),
        SavedFormat::id("l0b").quantise::<i16>(255).round(),
    ];
    for i in 0..ATTN_LAYERS {
        save_format.push(SavedFormat::id(&format!("attn{i}_qw")).quantise::<i16>(64).transpose().round());
        save_format.push(SavedFormat::id(&format!("attn{i}_qb")).quantise::<i16>(255 * 64).round());
        save_format.push(SavedFormat::id(&format!("attn{i}_kw")).quantise::<i16>(64).transpose().round());
        save_format.push(SavedFormat::id(&format!("attn{i}_kb")).quantise::<i16>(255 * 64).round());
        save_format.push(SavedFormat::id(&format!("attn{i}_vw")).quantise::<i16>(64).transpose().round());
        save_format.push(SavedFormat::id(&format!("attn{i}_vb")).quantise::<i16>(255 * 64).round());
        save_format.push(SavedFormat::id(&format!("attn{i}_ow")).quantise::<i16>(64).transpose().round());
        save_format.push(SavedFormat::id(&format!("attn{i}_ob")).quantise::<i16>(255 * 64).round());
    }
    save_format.extend(vec![
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
    ]);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .inputs(inputs)
        .optimiser(optimiser::AdamW)
        .save_format(&save_format)
        .use_threads(4)
        .build_custom(|builder, (stm_inputs, ntm_inputs), targets| {
            // FT (= HalfKAv2_hm sparse → L1=2048)
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1_SIZE);
            let stm_h = l0.forward(stm_inputs).screlu();
            let ntm_h = l0.forward(ntm_inputs).screlu();
            let concat = stm_h.concat(ntm_h); // (2 * L1_SIZE, 1)

            // 既存 path とほぼ同じ構造で「stm の L1 部分のみ」 を attention に通し、 concat 後の value head は維持。
            // ここでは concat (2*L1) を base にして reshape (D_MODEL, 2*NUM_TOKENS) として attention をかける。
            let attn_tokens = 2 * NUM_TOKENS; // = 32
            let mut x = concat.reshape(Shape::new(D_MODEL, attn_tokens));

            for i in 0..ATTN_LAYERS {
                // QKV projections (Affine: out_dim = D_MODEL, in_dim = D_MODEL)
                let q_w = builder.new_affine(&format!("attn{i}_q"), D_MODEL, D_MODEL);
                let k_w = builder.new_affine(&format!("attn{i}_k"), D_MODEL, D_MODEL);
                let v_w = builder.new_affine(&format!("attn{i}_v"), D_MODEL, D_MODEL);
                let o_w = builder.new_affine(&format!("attn{i}_o"), D_MODEL, D_MODEL);

                let q = q_w.forward(x.clone()); // (D_MODEL, attn_tokens)
                let k = k_w.forward(x.clone());
                let v = v_w.forward(x.clone());

                // scores = Q^T @ K → (attn_tokens, attn_tokens) (transa = true: Q transposed)
                let scores = q.gemm(true, k, false);

                // softmax over tokens (feature 軸 = 行方向)
                let scale = builder.new_constant(Shape::new(1, 1), &[ATTN_SCALE]);
                let scaled = scores * scale;
                let attn = scaled.softmax();

                // out = V @ attn → (D_MODEL, attn_tokens)
                let out = v.matmul(attn);

                // output projection
                let out_proj = o_w.forward(out);

                // residual ではなく out_proj を次に渡す (user 指定: 非 ResNet = residual なし)
                x = out_proj;
            }

            // reshape back to flat (2 * L1_SIZE, 1)
            let attn_out = x.reshape(Shape::new(2 * L1_SIZE, 1));

            // value head (既存と同型、 LeakyReLU)
            let l1 = builder.new_affine("l1", 2 * L1_SIZE, L2_SIZE);
            let l2 = builder.new_affine("l2", L2_SIZE, L3_SIZE);
            let v_out = builder.new_affine("v_out", L3_SIZE, 1);
            let h1 = l1.forward(attn_out.clone()).leaky_relu();
            let h2 = l2.forward(h1).leaky_relu();
            let value_out = v_out.forward(h2);
            let value_loss = value_out.sigmoid().squared_error(targets);

            // policy head (SwiGLU PH=512、 既存と同型)
            let pol_w1 = builder.new_affine("pol_w1", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_w3 = builder.new_affine("pol_w3", 2 * L1_SIZE, POLICY_HIDDEN);
            let pol_o = builder.new_affine("pol_o", POLICY_HIDDEN, POLICY_DIMS);
            let gate_pre = pol_w1.forward(attn_out.clone());
            let value_pre = pol_w3.forward(attn_out);
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

    let superbatches: usize = std::env::var("BULLET_SUPERBATCHES").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
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
        output_directory: "E:/shogi/p2shogi/checkpoints",
        batch_queue_size: 32,
        on_checkpoint_saved: None,
    };

    let data_files: Vec<String> = match std::env::var("BULLET_DATA") {
        Ok(s) if !s.is_empty() => s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect(),
        _ => vec!["E:/shogi/p2shogi/data/_psv/gct_strict_visits_50M_shuf.bin".to_string()],
    };
    let data_files_refs: Vec<&str> = data_files.iter().map(|s| s.as_str()).collect();
    eprintln!("[v21_attn] data files ({}):", data_files_refs.len());
    for p in &data_files_refs {
        eprintln!("  {p}");
    }

    let data_loader = DirectSequentialDataLoader::new(&data_files_refs);
    trainer.run(&schedule, &settings, &data_loader);
}
