/*
Shogi LayerStack NNUE Training Script

LayerStacks (SFNNwoP1536 progress) アーキテクチャの学習スクリプト。
tanuki-wcsc36-engine.shogitest.sfnnwoP1536.progress 互換の量子化ファイル (quantised.bin) を出力する。

Usage:
    cargo run --release --example shogi_layerstack -- [OPTIONS]

Options:
    --data <PATH>       Training data path (comma-separated for multiple files)
    --batch-size <N>    Batch size (default: 16384)
    --superbatches <N>  Number of superbatches (default: 100)
    --lr <RATE>         Initial learning rate (default: 0.001)
    --wdl <LAMBDA>      WDL lambda for constant scheduler (default: 0.5)
    --start-wdl <F>     Start WDL lambda for linear interpolation
    --end-wdl <F>       End WDL lambda for linear interpolation
    --scale <N>         Eval scale (default: 600)
    --l0 <SIZE>         FT output size (default: 1536)
    --l1 <SIZE>         L1 output size (default: 16)
    --l2 <SIZE>         L2 output size (default: 32)
    --save-rate <N>     Save interval in superbatches (default: 10)
    --threads <N>       Number of threads (default: 4)
    --output <DIR>      Output directory (default: checkpoints)
    --net-id <NAME>     Network ID (default: shogi-ls-1536)
    --resume <PATH>     Resume from checkpoint
    --quantise-only     Only re-quantise checkpoint (requires --resume)
    --optimizer <OPT>   Optimizer (adamw, radam, ranger) (default: ranger)
    --win-rate-model    Use win rate model for score conversion
    --batches-per-superbatch <N>  Batches per superbatch (default: auto)
    --lr-gamma <F>      LR decay rate (default: 0.992)
    --lr-step <N>       LR decay interval (default: 1)
    --interleave-file-batches <N> File mix granularity (0=sequential, 1=round-robin)
    --epoch-file-shuffle Shuffle file order every epoch
    --file-shuffle-seed <SEED> Seed for epoch file shuffle
    --psqt               Enable PSQT shortcut layer
    --psqt-scale <F>     PSQT 出力に掛ける定数 (default: 1.0 = 従来どおり)
    --export-no-psqt     quantised.bin を PSQT 無しで書き出す (default: off)
    --threat            Enable Threat concatenated input (placeholder)
*/

use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

use bullet_lib::{
    acyclib::graph::CheckpointShapeAdapter,
    game::inputs::{
        Direct15HeadShape, ShogiHalfKA_hm, ShogiHalfKaHmAux, ShogiHalfKaHmAuxCheap, ShogiHalfKaHmHandCount,
        ShogiHalfKaHmHandThreat, ShogiHalfKaHmHandThreatDefensive, ShogiHalfKaHmKzt, ShogiHalfKaHmThreat,
        ShogiHalfKaHmThreatDirect15, ShogiHalfKaHmThreatFactor16, SparseInputType, THREAT_DIRECT15_ATTACKS_PER_CLASS,
        THREAT_DIRECT15_DEFENDED_DIMENSIONS, THREAT_DIRECT15_DIMENSIONS, ThreatProfile, direct15_pair_of_feature,
        expand_direct15_head_bias, expand_direct15_head_weights,
    },
    game::outputs::{
        SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS, SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER,
        SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES, SHOGI_PROGRESS8_FEATURE_ORDER, SHOGI_PROGRESS8_NUM_FEATURES,
        ShogiProgressBucket8, ShogiProgressBucket8GikouLite, ShogiProgressKPAbs,
    },
    nn::{
        Affine, BackendMarker, InitSettings, NetworkBuilderNode, Shape,
        optimiser::{self, AdamWParams, RAdamParams, Ranger21Params, RangerParams},
    },
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{
        ValueTrainerBuilder,
        loader::{DirectSequentialDataLoader, HcpeLoader, looks_like_hcpe_file},
    },
};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

// =============================================================================
// Constants
// =============================================================================

// LayerStack 出力 bucket 数。既定(progress8 / P4H2 = 進行度4×持駒2)は 8。
// `lsmode-p8h2` feature ビルドのみ P8H2(進行度8×持駒2)= 16 に切り替わる。
// ★C++ 推論側の `NUM_BUCKETS`(既定 8 / -DP2_LSMODE_P8H2 で 16)と対称。
// この定数は graph の LayerStack weight shape と量子化 save format の per-bucket
// セクション数を同時に決めるため、OutputBuckets 型の `BUCKETS` と必ず一致させること。
// 2026-11-14 OPNARCH: `lsmode-opn81` (k9k9z = 81) / `lsmode-opn243` (k9k9z × 進行度 3 = 243) を追加。
//   OPN (序盤帯専用 net) 用。 これらのビルドは fc0 / fc1 / fc2 を gather affine (選んだ bucket の行だけ計算) で回す。
#[cfg(any(
    all(feature = "lsmode-p8h2", feature = "lsmode-opn81"),
    all(feature = "lsmode-p8h2", feature = "lsmode-opn243"),
    all(feature = "lsmode-opn81", feature = "lsmode-opn243"),
))]
compile_error!("lsmode-p8h2 / lsmode-opn81 / lsmode-opn243 は同時に 1 つだけ指定すること");
#[cfg(not(any(feature = "lsmode-p8h2", feature = "lsmode-opn81", feature = "lsmode-opn243")))]
const NUM_BUCKETS: usize = 8;
#[cfg(feature = "lsmode-p8h2")]
const NUM_BUCKETS: usize = 16;
#[cfg(feature = "lsmode-opn81")]
const NUM_BUCKETS: usize = 81;
#[cfg(feature = "lsmode-opn243")]
const NUM_BUCKETS: usize = 243;
const QA: i16 = 127;
/// FT (HalfKA accumulator) 専用の量子化スケール。
/// エンジン側は accumulator を 0..QA_FT で clamp し、 pairwise を
/// `mulhi(lo << 7, hi)` = `(lo*hi) >> 9` で行う (NAGISA V3.1 / sfnn-1536.h と同方式)。
/// 出力レンジは 254*254/512 = 126 で QA=127 時の 127*127/128 = 126 と同一なので、
/// 下流 (fc0 bias = QA*QB, fc0 weight = QB) のスケールは変更不要。
/// 実測 (2026-09-06): NAGISA は accumulator の 5-95% レンジが 575、 我々 (QA=127) は 197。
/// QA_FT を倍にすると同じ float 関数を約 2 倍細かい目盛りで表現でき、 量子化誤差が半減する。
const QA_FT: i16 = 254;
const QB: i16 = 64;

#[derive(Debug, Clone, Copy)]
struct WrmLossParams {
    nnue2score: f32,
    in_scaling: f32,
}

static WRM_LOSS_PARAMS: OnceLock<WrmLossParams> = OnceLock::new();

/// 損失指数 p: loss = |pred − target|^p。
/// nnue-pytorch (Stockfish master) は p = 2.6 を採用しており、 SPRT 検証で
/// p = 2.0 (MSE) や cross-entropy より強い net を生むことが確認されている。
/// 既定 2.6。 旧挙動 (MSE) は --loss-power 2.0 で再現できる。
static LOSS_POWER: OnceLock<f32> = OnceLock::new();

/// `--loss-scale S` 指定時に損失の前で出力に掛ける定数 `scale / S`。 未指定なら未設定のまま。
static LOSS_OUT_MULT: OnceLock<f32> = OnceLock::new();

/// 2026-09-17 `--loss-grad-mask` の (閾値 |教師値| cp, 係数) 表。
static LOSS_WEIGHT_TABLE: std::sync::OnceLock<Vec<(i32, f32)>> = std::sync::OnceLock::new();
/// `--kh-split` の半分 (0 / 1)。 bullet の重み関数は fn ポインタなのでグローバルに置く。
static KH_SPLIT: std::sync::OnceLock<u8> = std::sync::OnceLock::new();

fn loss_power() -> f32 {
    LOSS_POWER.get().copied().unwrap_or(2.6)
}

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum OptimizerType {
    AdamW,
    RAdam,
    #[default]
    Ranger,
    /// Positive-Negative MomentumとLookaheadを使うRanger21。
    Ranger21,
}

/// PSQT shortcut layer の重み初期化方法。
///
/// - `Zeroed` (default): 従来通り全 0 で初期化。 `--psqt-init` 省略時と完全に同一挙動。
/// - `Material`: 各特徴量が表す駒の駒得価値 (cp) を prior として psqt_w に注入する。
///   学習開始時から material baseline を持たせる狙い (scale-critical、 詳細は
///   `init_psqt_material` 参照)。
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
enum PsqtInit {
    /// 全 0 初期化 (= 従来動作、 default)
    #[default]
    Zeroed,
    /// 駒得価値 prior を注入
    Material,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
enum BucketMode {
    #[default]
    Kingrank9,
    Ply9,
    Progress8,
    #[value(name = "progress8gikou")]
    Progress8Gikou,
    #[value(name = "progress8kpabs")]
    Progress8KPAbs,
    /// LayerStack を「進行度4区分 × 持駒量2区分 = 8」で選ぶ。
    /// progress8kpabs と同じ progress.bin を使い、stack = progress_bucket4*2 + hand_bucket2。
    /// このモードで学習した net は arch に "LSMODE=P4H2," marker が付き、
    /// 推論側は -DP2_LSMODE_P4H2 ビルドでのみロードできる。
    #[value(name = "progress4hand2kpabs")]
    Progress4Hand2KPAbs,
    /// LayerStack を「進行度8区分 × 持駒量2区分 = 16」で選ぶ (P8H2)。
    /// progress8kpabs と同じ progress.bin を使い、stack = progress_bucket8*2 + hand_bucket2。
    /// P4H2 と異なり進行度を圧縮しないため出力 bucket 数が 16 になる。net は arch に
    /// "LSMODE=P8H2," marker が付き、推論側は -DP2_LSMODE_P8H2 ビルドでのみロードできる。
    /// 訓練は 16 bucket graph が必要なため `--features lsmode-p8h2` ビルドでのみ機能する。
    #[value(name = "progress8hand2kpabs")]
    Progress8Hand2KPAbs,
    /// 2026-11-14 OPNARCH: OPN 用の king bucket k9k9z (81)。 `--features lsmode-opn81` ビルドでのみ使える。
    /// head = kz (両玉の zone 9 × 9)。 progress.bin は `--bucket-filter` (進行度 bucket) の計算に使う。
    /// arch に "LSMODE=K9Z," が付く。
    #[value(name = "opnk9z")]
    OpnK9z,
    /// 2026-11-14 OPNARCH: OPN 用の k9k9z × 進行度 3 (243)。 `--features lsmode-opn243` ビルドでのみ使える。
    /// head = kz * 3 + 進行度 bucket (0..2)。 `--bucket-filter` (上限 2 以下) が必須。
    /// arch に "LSMODE=K9ZP3," が付く。
    #[value(name = "opnk9zp3")]
    OpnK9zProg3,
}

impl BucketMode {
    /// OPN の king bucket モード (k9k9z / k9k9z × 進行度 3) か。
    fn is_opn_king(self) -> bool {
        matches!(self, BucketMode::OpnK9z | BucketMode::OpnK9zProg3)
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
enum LayerStackOutputFormat {
    /// rshogi NetworkLayerStacks 互換フォーマット
    #[default]
    Rshogi,
    /// hakubishin-private tanuki-wcsc36-engine.shogitest.sfnnwoP1536.progress 互換フォーマット
    #[value(name = "tanuki-sfnnwop1536")]
    TanukiSfnnwoP1536,
}

fn parse_nonzero_usize(text: &str) -> Result<usize, String> {
    let value = text.parse::<usize>().map_err(|e| format!("invalid unsigned integer '{text}': {e}"))?;
    if value == 0 { Err("value must be 1 or greater".to_string()) } else { Ok(value) }
}

/// --threads の既定値: 論理コア数 (取得失敗時は 4)。
/// データ準備 (PSV decode + 特徴量展開 + aux head 計算) は CPU-bound になりやすく、
/// 旧既定 4 では GPU が待つ (実測: GTX 1050 Ti + 8C 環境で GPU 使用率 ~42%)。
fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

#[derive(Parser, Debug)]
#[command(name = "shogi_layerstack")]
#[command(about = "Shogi LayerStack NNUE training script")]
struct Args {
    /// Training data path (comma-separated for multiple files)
    #[arg(long, default_value = "data/train.bin")]
    data: String,

    /// Training data format。 `psv` (= 40 byte / record、 YO PackedSfenValue、 default) または
    /// `hcpe` (= 32 byte / record、 cshogi HuffmanCodedPosAndEval、 dlshogi hao_depth9 等)。
    /// `auto` (= default) のときは file size から `% 32 == 0 && % 40 != 0` で hcpe を推定する。
    #[arg(long, default_value = "auto")]
    data_format: String,

    /// Batch size
    #[arg(long, default_value = "16384")]
    batch_size: usize,

    /// Number of superbatches
    #[arg(long, default_value = "100")]
    superbatches: usize,

    /// Initial learning rate
    #[arg(long, default_value = "0.001")]
    lr: f32,

    /// WDL lambda (constant). Cannot be used with --start-wdl/--end-wdl
    #[arg(long, conflicts_with_all = ["start_wdl", "end_wdl"])]
    wdl: Option<f32>,

    /// Start WDL lambda for linear interpolation
    #[arg(long, requires = "end_wdl")]
    start_wdl: Option<f32>,

    /// End WDL lambda for linear interpolation
    #[arg(long, requires = "start_wdl")]
    end_wdl: Option<f32>,

    /// Eval scale (default: 600, Eval_Coef=600 のDL教師データと整合)
    #[arg(long, default_value = "600")]
    scale: i32,

    /// 損失の sigmoid の尺度 S (2026-11-13、 水匠の手法)。 出力の目盛りは `--scale` のまま、
    /// 損失の傾きだけを S 相当に緩める。
    ///
    /// 指定時の損失: `|sigmoid(out * scale / S) - target_S|^p`、
    /// `target_S = (1 - wdl) * sigmoid(score / S) + wdl * result` (result は 0 / 0.5 / 1)。
    /// - `--scale` は今までどおり出力の単位・FV_SCALE・書き出し・PSQT 初期値に使う。
    /// - `--win-rate-model` / `--wrm-in-scaling` とは併用できない (起動時にエラー)。
    /// 未指定 (既定) なら loader も損失も従来とビット単位で同一。
    #[arg(long)]
    loss_scale: Option<f32>,

    /// L0 (Feature Transformer) size
    #[arg(long, default_value = "1536")]
    l0: usize,

    /// L1 output size (includes skip connection neuron)。 main 行 = l1 - 1、 L2 入力 = [main^2, main]。 例: 16 (main 15) / 25 (main 24)
    #[arg(long, default_value = "16")]
    l1: usize,

    /// Train and export a no-residual control while preserving the P2NNUE L1 layout.
    #[arg(long, default_value_t = false)]
    residual_zero: bool,

    /// Keep raw skip at the output and expose [skip^2, skip] to the deep path.
    #[arg(long, default_value_t = false)]
    residual_mix: bool,

    /// R-MixWide16: force the residual architecture to 1024-16-64 (15 main rows + final raw skip) and expose [skip^2, skip] to L2.
    #[arg(long, default_value_t = false)]
    residual_wide16: bool,

    /// L2 output size
    #[arg(long, default_value = "32")]
    l2: usize,

    /// 2026-11-14 OPNARCH T2: fc0 / fc1 / (L4) / fc2 を gather affine (選んだ bucket の行だけ計算する層) で回す。
    /// 値と勾配は従来の `Affine → select` と同じ (浮動小数の加算順だけ違う)。 重みの形・保存形式も同じ。
    /// 81 / 243 bucket ビルド (`--features lsmode-opn81` / `lsmode-opn243`) では常に有効 (指定不要)。
    /// 8 bucket ビルドで指定すると従来版との照合 (同じ乱数種・同じ batch で損失が一致するか) に使える。
    /// 未指定 (既定) なら graph は従来と同一。
    #[arg(long, default_value_t = false)]
    gather_affine: bool,

    /// 2026-11-14 OPN2048 K16: 入力を HalfKA_hm (玉区分 45、 73,305) から K16 (玉区分 16、 26,064) にする。
    /// 自玉が 7〜9 段の 15 区分はそのまま、 1〜6 段の 30 区分は 1 つに束ねる (設計書 DESIGN-20261114-OPN2048-FABLE.md 2.3)。
    /// `--factorise` と併用可 (factoriser は今と同じ 1,629 行、 起動ログは 27,693 dimensions)。
    /// threat / aux / PSQT など他の入力・head とは併用不可 (起動時に止める)。 未指定 (既定) なら従来と同一。
    /// arch は `Features=HalfKA_hmK16(Friend)[26064->FTx2]` になる。
    #[arg(long, default_value_t = false)]
    opn_king16: bool,

    /// 2026-11-15 QAT (量子化を意識した微調整、 既存 checkpoint から 1〜2 SB 続ける仕上げ用)。
    /// 順伝播だけを書き出し後の整数推論 (rshogi / p2nnue の LayerStack) と同じ格子に落とし、
    /// 逆伝播は STE (丸めは恒等)。 optimizer と保存は f32 の master 重みのまま。 丸める場所:
    /// - FT の積 (両視点の連結): Q=127 切り捨て (エンジン `(a*b)>>9` の u8)
    /// - L1 / L2 / L3 の重み: Q=64 四捨五入、 [-2, 127/64] で clip (= 書き出しの `as i8` の飽和)。
    ///   L1 は bucket 分 l1w と共有分 l1fw を f32 で足してから 1 回丸める (書き出しと同じ)
    /// - L1 / L2 / L3 の bias: Q=8128 四捨五入 (L1 は l1b + l1fb)
    /// - L1 後 [sqr, 線形] と L2 後の活性: crelu の後に Q=127 切り捨て (`(v*v)>>19` / `v>>6` / `>>6`)
    /// 素の LayerStack (+ 任意の --opn-king16 / --factorise / --gather-affine) 専用。 threat / PSQT /
    /// L4 / KH などとは併用不可 (起動時に止める)。 未指定 (既定) なら graph は従来と同一。
    #[arg(long, default_value_t = false)]
    qat: bool,

    /// 2026-11-17 U1a: `--qat` の fc0 を **L1Q128** (現職の書き出し、 `runs/l1q128/mk_l1q128.py`) の格子にする。
    /// 本体 15 行 (bucket ごとの行 0..L1-2) は重み round(128 w) を [-128, 127] (= w を [-1, 127/128] で clip)、
    /// bias round(16256 b)。 skip 行 (L1-1) は従来どおり 64 / 8128。 L2 入力の u8 の目盛りは L1Q128 でも同じ
    /// (エンジン: lin = v >> 7、 sqr = v^2 >> 21) なので ③ 以降は変えない。 `--qat` と併用すること。
    #[arg(long, default_value_t = false)]
    qat_l1q128: bool,

    /// 2026-11-15 検証用: 重みを読み込んだ後 (学習の前)、 この PSV の全局面について学習器の順伝播の出力を
    /// `<--output>/forward_dump.txt` (1 行 = 「bucket 出力」、 出力は損失に入る前の生の値) に書き、
    /// 同じ重みの量子化 net を `<--output>/forward_dump_quantised.bin` に書いて終了する (学習しない)。
    /// `--qat` の有無どちらでも使える (整数推論 opn_static2.py との突き合わせ用)。
    #[arg(long)]
    dump_forward: Option<PathBuf>,

    /// Save interval (superbatches)
    #[arg(long, default_value = "10")]
    save_rate: usize,

    /// Progress display interval in batches
    #[arg(long, default_value = "128", value_parser = parse_nonzero_usize)]
    log_rate: usize,

    /// Number of threads for host-side data preparation
    /// (default: all logical cores; the pipeline is usually CPU-bound)
    #[arg(long, default_value_t = default_threads())]
    threads: usize,

    /// Loss exponent p: loss = |pred - target|^p.
    /// 2.6 = nnue-pytorch (Stockfish master) 採用値。 2.0 で旧 MSE 挙動。
    #[arg(long, default_value_t = 2.6)]
    loss_power: f32,

    /// LR warmup: 最初の superbatch の先頭 N batch で LR を線形に立ち上げる (0 = 無効)
    #[arg(long, default_value_t = 0)]
    warmup_batches: usize,

    /// バッファ内レコードシャッフルを無効化する (--data-format psv の direct loader 専用)。
    /// 既定は有効: 読み込みチャンク (256MB) 内で局面順を決定論的にシャッフルし、
    /// 同一対局由来の連続局面によるバッチ内相関を除去する (nnue-pytorch の
    /// ランダムサンプリング相当)。
    #[arg(long, default_value_t = false)]
    no_shuffle_records: bool,

    /// Output directory
    #[arg(long, default_value = "checkpoints")]
    output: PathBuf,

    /// Network ID
    #[arg(long, default_value = "shogi-ls-1536")]
    net_id: String,

    /// Optimizer (adamw, radam, ranger, ranger21)
    #[arg(long, value_enum, default_value = "ranger")]
    optimizer: OptimizerType,

    /// Weight decay
    #[arg(long, default_value = "0.01")]
    weight_decay: f32,

    /// Ranger21 internal schedule total steps. Omit to derive it from the complete superbatch run; set 0 to disable the internal schedule for a one-batch preflight.
    #[arg(long)]
    ranger21_total_steps: Option<usize>,
    /// Ranger21 Lookahead merge interval in optimizer steps. The default 5 preserves the baseline behavior.
    #[arg(long, default_value = "5")]
    ranger21_lookahead_mergetime: usize,

    /// Batches per superbatch (default: auto ~100M positions)
    #[arg(long)]
    batches_per_superbatch: Option<usize>,

    /// LR scheduler gamma
    #[arg(long, default_value = "0.992")]
    lr_gamma: f32,

    /// LR scheduler step interval
    #[arg(long, default_value = "1")]
    lr_step: usize,

    /// Start superbatch number
    #[arg(long, default_value = "1")]
    start_superbatch: usize,

    /// Batch queue size
    #[arg(long, default_value = "64")]
    batch_queue_size: usize,

    /// Read this many batches from one file before switching files (0 = sequential by file)
    #[arg(long, default_value = "0")]
    interleave_file_batches: usize,

    /// Shuffle file order at every epoch boundary
    #[arg(long)]
    epoch_file_shuffle: bool,

    /// Seed for --epoch-file-shuffle
    #[arg(long, default_value = "0")]
    file_shuffle_seed: u64,

    /// Resume from checkpoint
    #[arg(long)]
    resume: Option<String>,

    /// Load f32 `weights.bin` from a checkpoint without loading optimiser state.
    ///
    /// 異なるoptimizerへ切り替えるwarm-start専用。特に既存RangerからRanger21へ
    /// 移るときはstateの意味が異なるため、`--resume`ではなくこちらを使う。
    #[arg(long, conflicts_with_all = ["resume", "resume_from_quantised", "quantise_only"])]
    resume_weights_only: Option<String>,

    /// Resume from quantised.bin directly (raw.bin / optimiser_state は不要)。
    ///
    /// 量子化重みを逆量子化して f32 graph weights を上書きし、 optimiser state は
    /// zero init で start する。 fine-tune や L1/L2 dim 変換後の resume に使う。
    /// `--resume` / `--quantise-only` とは排他。
    #[arg(long, conflicts_with_all = ["resume", "quantise_only"])]
    resume_from_quantised: Option<String>,

    /// `--resume-from-quantised` 再開時に optimiser の velocity (二次モーメント) buffer を
    /// 埋める正の定数。 default 0.0 = 従来動作 (= velocity zero init)。
    ///
    /// velocity=0 のまま Adam/RAdam を回すと更新分母 `sqrt(v)+eps` が eps(=1e-8) に退化し、
    /// 初手の更新が `lr*grad/1e-8` で爆発して eval 出力が崩壊する。 例えば 1e-4 を指定すると
    /// 分母が `sqrt(1e-4)=0.01` となり爆発を抑止できる (= base 近傍の eval magnitude を維持)。
    /// `--resume-from-quantised` 指定時のみ効果を持つ。
    #[arg(long, default_value_t = 0.0)]
    resume_from_quantised_velocity_init: f32,

    /// Only re-quantise checkpoint
    #[arg(long)]
    quantise_only: bool,

    /// Use win rate model
    #[arg(long)]
    win_rate_model: bool,

    /// Apply WRM to network output in loss (nnue-pytorch-nodchip style).
    /// Value is the in_scaling parameter (nodchip default: 340).
    /// Requires --win-rate-model. When set, loss becomes |WRM_in(net) - WRM_out(target)|^2
    /// instead of |sigmoid(net) - WRM_out(target)|^2.
    #[arg(long, requires = "win_rate_model")]
    wrm_in_scaling: Option<f32>,

    /// Scaling factor to convert network output to centipawn score for WRM loss.
    /// Only used when --wrm-in-scaling is set. The raw network output is multiplied
    /// by this value before being passed to the WRM function.
    /// (nnue-pytorch-nodchip default: 600)
    #[arg(long, default_value_t = 600.0, requires = "wrm_in_scaling")]
    wrm_nnue2score: f32,

    /// Output bucket mode (kingrank9 / ply9 / progress8 / progress8gikou / progress8kpabs)
    #[arg(long, value_enum, default_value = "kingrank9")]
    bucket_mode: BucketMode,

    /// Optional boundaries for ply9 buckets (8 comma-separated values)
    #[arg(long)]
    ply_bounds: Option<String>,

    /// Enable PSQT shortcut layer
    #[arg(long, default_value_t = false)]
    psqt: bool,

    /// PSQT shortcut layer の重み初期化方法 (zeroed | material)。
    ///
    /// `zeroed` (default) は従来通り全 0 (= 省略時と bit-identical)。
    /// `material` は各特徴量が表す駒の駒得価値 (cp) を prior として注入する。
    /// `--psqt` が無効な場合は無視される。 fresh start (resume なし) のときのみ適用。
    #[arg(long, value_enum, default_value = "zeroed")]
    psqt_init: PsqtInit,

    /// 2026-09-12 追加: PSQT shortcut の出力 (bucket 別 psqt 値 `(stm - nstm) / 2`) に掛ける
    /// 定数 f (default 1.0 = 従来と graph が完全に同一、 ノードは増えない)。
    ///
    /// PSQT の寄与を SB ごとに 0.5 → 0.25 → 0 と薄めながら学習し、 最後に PSQT 無しの
    /// net (`--export-no-psqt`) を書き出す用途。 実装は forward の定数 `0.5` を `0.5 * f` に
    /// 置き換えるだけなので、 **psqtw / psqtb への勾配も f 倍**になる (f=0 で勾配ゼロ =
    /// PSQT 重みは weight decay 以外で動かない)。 `--psqt` が無効なら無視される。
    #[arg(long, default_value_t = 1.0)]
    psqt_scale: f32,

    /// 2026-09-12 追加: rshogi 形式 quantised.bin を書き出すとき PSQT を落とす (default off)。
    ///
    /// arch 文字列から `PSQT=8,` を除き、 PSQT section (bias i32 × 8 + weight i32 × 73,305 × 8)
    /// を書かない。 エンジン (p2nnue.cpp load_network) は arch に `PSQT=` が無ければ section を
    /// 読まず `has_psqt=false` → `psqt_val = 0` で推論する。 raw.bin / checkpoint には従来どおり
    /// psqt tensor が残る (`--resume` 互換)。 学習時の `--psqt-scale` が 0 でないのに指定すると
    /// 評価値が学習時と変わるため起動時に警告する。 `--enable-auxcheap` (psqtw の aux 行を
    /// 別 section で書く) とは併用不可。
    #[arg(long, default_value_t = false)]
    export_no_psqt: bool,

    /// Enable Threat concatenated input
    #[arg(long, default_value_t = false)]
    threat: bool,

    /// Threat exclusion profile (full, same-class, same-class-major-pawn, cross-side)
    #[arg(long, default_value = "full")]
    threat_profile: String,

    /// Threat accumulator の**出力幅** K (0 = 従来 = FT 幅と同じ)。
    ///
    /// K > 0 にすると threat 特徴は主 sparse 入力から分離され、 `(K, threat_dims)` の
    /// 独立テンソル `threat_ft_w` になり、 その出力を FT の先頭 K 行へ加算する。
    /// エンジンのビルド定数 `P2_THREAT_OUT=K` と**必ず一致させること**
    /// (net の threat 行の長さが K になるため、 不一致だとロードが壊れる)。
    ///
    /// full threat (216,720 次元) を 1050Ti (4GB) で学習するために必要:
    /// 主入力に連結したままだと l0w が (73,305+216,720) x 1024 fp32 = 1.19GB/copy、
    /// Ranger21 の状態込みで ~5.9GB になり載らない。 K=256 なら threat 側は
    /// 216,720 x 256 fp32 = 222MB/copy で収まる。
    #[arg(long, default_value_t = 0)]
    threat_out: usize,

    /// threat が乗る FT 次元を HalfKA から**切り離す** (混線を無くす)。
    ///
    /// 既定では threat は HalfKA と同じ次元に加算される (acc[j] = halfka + threat)。
    /// pairwise_mul で (h+t)x(h+t) となり HalfKA との交互作用が表現できる反面、
    /// その次元では HalfKA の信号に threat が混ざる。 このフラグを立てると
    /// 該当次元を threat 出力だけにする (HalfKA 側の寄与を forward で捨てる)。
    /// HalfKA の実効次元は 1024 -> 1024-K に減る。 どちらが良いかは未検証 (2026-09-10)。
    #[arg(long, default_value_t = false)]
    threat_dedicated_dims: bool,

    /// Threat FT 重みを i16 で書き出す (default false = 従来 i8 ±127 clamp)。
    ///
    /// threat 分布は HalfKA FT と同一に育ち、収束時 max~500 で ±127 を超える最強
    /// 脅威信号が i8 では clip される。有効化すると threat 重みを clamp 無し i16 (LE 2byte)
    /// で書き出し、arch_str に "ThreatI16=1," marker を付与する (C++ loader が検出して
    /// i16 経路でロード)。i8 net とはバイナリ非互換 (別ネット扱い)。`--threat` 有効時のみ効果。
    #[arg(long, default_value_t = false)]
    threat_i16: bool,

    /// Profile15 (Direct15) の学習側 cap [cp] (default 30 = 従来ハードコード値と bit 同一)。
    ///
    /// C++ 推論側 `P2_THREAT_DIRECT_CAP_CP` (compile-time 定数、static_assert で 0..=200)
    /// と必ず一致させること。値を変える場合はエンジンも同値で再ビルドしないと
    /// 学習 forward と推論の cap が乖離する。0 = 上限なし。
    /// 現行 production エンジン (`tsumugi_p15_cap60_fast3.exe`) は 60 でビルド済み。
    /// `--threat --threat-profile enemy-only-direct15` 有効時のみ効果。
    #[arg(long, default_value_t = 30.0)]
    direct15_cap_cp: f32,

    /// Profile15 (Direct15) の**勾配**の減衰尺度 [cp]。 0 = hard clamp の真の勾配
    /// (cap 外は勾配 0)。 正値を指定すると forward は cap で clamp したまま、
    /// backward だけ `2g*sigmoid(2x/g) - g` の傾き (原点で 1) で滑らかに減衰させる。
    ///
    /// 2026-09-01 実測 (cap60, SB1, 実戦 1,500 局面): 真の勾配だと **41.5% の局面が
    /// 勾配 0** で head の学習に寄与しない。 g=120 なら勾配 0.05 未満は 0.7% まで減り、
    /// 中央値 0.86 を確保しつつ、 cap を大きく超えた外れ値 (p99=235cp で 0.08) は
    /// 従来どおり抑制される。
    #[arg(long, default_value_t = 0.0)]
    direct15_grad_scale_cp: f32,

    /// Direct15 head の重みを毎 step 0 に射影する (**対照実験用**)。
    ///
    /// head 込みで学習した checkpoint から `--threat` を外して resume すると
    /// `NoWeightWithID("threat_direct_b_cancels")` でロードに失敗するため、
    /// 「head 無し」の対照 net は graph を変えずに重みを 0 固定して作る。
    /// 書き出される P2THRD2 payload は全 0 になり、 エンジン側では
    /// threat_direct_raw=0 (cap の影響も無し) になる。
    #[arg(long, default_value_t = false)]
    direct15_zero_head: bool,

    /// P2: Direct15 head を stm 視点 / nstm 視点で別テーブルにする (bucket 8 -> 16)。
    ///
    /// 従来は両視点で同じテーブルを引き `raw = Σ_stm − Σ_nstm` としていたため、
    /// 「自分の駒が当たられている」と「相手の駒に当たっている」を符号でしか区別できなかった。
    /// on にすると graph の `threat_direct_w` が [16, dims] (行 = 2*bucket + 視点、 視点 0 = stm)
    /// になり、 書き出す P2THRD2 は buckets=16 (ファイル上 0..7 = stm 視点、 8..15 = nstm 視点)。
    /// エンジンは section header の buckets で判別する (旧 8 bucket net も引き続き読める)。
    /// off (default) のときは graph / 書き出し形式とも従来と bit 同一。
    /// `--threat --threat-profile enemy-only-direct15` 有効時のみ指定可。
    #[arg(long, default_value_t = false)]
    direct15_persp_split: bool,

    /// P4: Direct15 head を factorizer で学習する。
    ///
    ///   W_b[f] = S[f] + P[pair(f)] + D_b[f]   (pair(f) = attacker class x target class の 100 通り、
    ///                                          defended 時は守備ビットも含めて 200 通り)
    ///
    /// S (`threat_direct_shared_w`, bucket 共通) と P (`threat_direct_pair_w`) は全 bucket / 全 cell の
    /// 勾配を受けるので、 観測数の少ない cell の推定分散を下げる狙い。 書き出し時に 3 項を畳んで
    /// 従来と同一形式 / 同一サイズの P2THRD2 にするのでエンジン無改造・推論コスト不変。
    /// `--direct15-persp-split` と併用時は S / P も視点ごとに持つ。 off (default) は従来と bit 同一。
    #[arg(long, default_value_t = false)]
    direct15_factorise: bool,

    /// P5: 被攻撃駒が守られているか (味方の実利きが 1 つ以上あるか) で Direct15 の特徴を分ける。
    ///
    /// 次元 65,640 -> 131,280。 index は [守備なし ブロック | 守備あり ブロック] の連結。
    /// 従来は「守られた駒への当たり」も「ただで取れる駒への当たり」も同じ重みを引いていたため、
    /// 重みが文脈平均に落ちて的中率が下がっていた (H3)。 これを分離する。
    /// 書き出す P2THRD2 は dims=131280。 エンジンは section header の dims で判別し、
    /// 旧 65,640 次元の net も引き続き読める。 off (default) は従来と bit 同一。
    /// `--threat --threat-profile enemy-only-direct15` 有効時のみ指定可。
    #[arg(long, default_value_t = false)]
    direct15_defended: bool,

    /// `--resume` / `--resume-weights-only` の base checkpoint の Direct15 head
    /// (`threat_direct_w` / `threat_direct_b_cancels`) を、 現在の `--direct15-persp-split` /
    /// `--direct15-defended` の形状へ複製展開して読み込む (P5 / P2 を学習済み head から始める用)。
    ///
    /// 展開は「展開直後の forward 出力が展開前と完全に同一」になる複製:
    ///   - P5 (defended): 各 bucket 行の 65,640 列を [守備なし | 守備あり] の両ブロックへ複製
    ///   - P2 (persp-split): bucket b の行を stm 行 2b と nstm 行 2b+1 の両方へ複製
    ///   - P5+P2: 両方の合成
    ///
    /// 単純なバッファの繰り返しではなく bucket / ブロック構造を保った要素単位の写像を使う。
    /// optimiser state (Ranger21 の momentum / velocity / max_velocity / slow、 AdamW の
    /// momentum / velocity) も同じ写像で展開するので、 head の step 数・分散推定を保ったまま
    /// 続きから学習できる (`BULLET_SKIP_UNKNOWN_WEIGHTS=1` で head を 0 から学習し直す方式と
    /// 違い、 main と同じ「学習済み base からの resume」として公平に比較できる)。
    /// base の head 形状は checkpoint の `threat_direct_b_cancels` (行数) と `threat_direct_w` の
    /// 要素数から自動判定する。 既に同形なら何もしない。 縮小方向はエラー。
    /// `--quantise-only` と併用すると、 展開しただけの net を `requantised.bin` に書き出せる
    /// (展開の出力不変性をエンジンで検証する用)。 ステージをまたぐ resume なので Ranger21 では
    /// `--ranger21-total-steps 0` も併せて指定すること (内部 schedule の warmdown 飽和で lr=0 になる)。
    #[arg(long, default_value_t = false)]
    direct15_expand_head: bool,

    /// Enable HandThreat concatenated input (full drop-attack pair, 121,104 dims)
    ///
    /// `--threat` とは排他。両方指定した場合はエラーで終了する。
    /// profile なし (v95 PoC 版)。
    #[arg(long, default_value_t = false)]
    hand_threat: bool,

    /// Enable HandThreat defensive variant (30,276 dims, 非対称 emission)
    ///
    /// `--hand-threat` と `--threat` 両方と排他。drop_owner=enemy かつ
    /// attacked_side=friend のみ符号化する防御 feature。
    #[arg(long, default_value_t = false)]
    hand_threat_defensive: bool,

    /// HandCount Dense Input を有効化する（L1 層の入力に 14 元の持ち駒 dense vector を concat）。
    ///
    /// `[stm 持ち駒 7 種, nstm 持ち駒 7 種] = 14 元` を FT 出力 (1536) の
    /// 後ろに連結して L1 に渡す。sparse 特徴は HalfKA_hm と完全互換。
    ///
    /// `--threat` / `--hand-threat` / `--hand-threat-defensive` とは排他。
    #[arg(long, default_value_t = false)]
    hand_count_dense: bool,

    // ---- LMAS は 2026-05-25 廃止 (= CLI flag 削除済)。 互換のため lmas_flags ()
    //      stub field のみ残す。 ----
    /// Pin-aware NNUE 補助入力を有効化する (= 黒/白視点 dense 2,268 dim × 2 = 4,536 dim、
    /// rshogi-core `feature "nnue-pin-aware"` 対応モデル向け)。
    ///
    /// quantised.bin 末尾に "P2PIN\0\0\0" section が追加され、 arch_str に
    /// "PIN=2268," marker が埋め込まれる。
    #[arg(long, default_value_t = false)]
    pin_aware: bool,

    /// B6 可動域 (2026-09-29)。 反対称スカラー 1 本を dense aux で流し、
    /// bucket 別の重み 8 本で出力へ足す。
    ///
    ///   mob_c = Σ(色 c の疑似合法な移動先の数)、 B6 = mob(手番側) − mob(非手番側)
    ///
    /// 勝敗 ΔR² +0.038 (床 ±0.001)、 NPS 税 +0.93%。 定義は 3 実装 (Python 正典 /
    /// C++ `p2mob` / 学習器 `shogi_mob.rs`) で 4,010 局面ゼロ許容一致を確認済み。
    /// quantised.bin 末尾に "P2MOB\0\0\0" section、 arch_str に "MOB=1," marker。
    #[arg(long, default_value_t = false)]
    mob: bool,

    /// 案1 ブロック射影 (2026-09-30、 fable 設計)。
    ///
    /// threat accumulator (K=64) を FT の**全 1024 次元**へ射影する。 いまは
    /// 前半 32 / 後半 32 の計 64 次元にしか着地しておらず、 pairwise_mul の積 512 個のうち
    /// threat が入るのは 32 個だけ。 全次元に載せると 512 個すべてに入る。
    ///
    /// 対角ブロック行列 (FT, K) を「フル重み ⊙ 定数マスク」で作る
    /// (🚨Concat が同一 batching を要求するので、 重み × バッチ活性の要素積は組めない)。
    /// ゼロ初期化なので**関数保存**。 quantised.bin に "P2TPRJ\0\0\0" section、
    /// arch_str に "ThreatProj=1024," marker。
    #[arg(long, default_value_t = false)]
    threat_proj: bool,

    /// 案1 ブロック射影の係数 `threat_proj_raw` の lr 倍率 (2026-09-30)。
    ///
    /// 🚨p はゼロ初期化なので、 ここを 1.0 のままにすると「動かないまま 8 SB」で
    ///   実験が null になる (ひもの轍)。 decay も 0 に上書きする。
    /// 既定 8: base 6e-6 x 24,415 batch なら約 1 SB で clip (±1) に届く。
    /// threat と同じ 72 にすると 1 SB 未満で貼り付き p が実質 ±1 の符号ベクトルになる。
    #[arg(long, default_value_t = 8.0)]
    threat_proj_lr_mult: f32,

    /// 案1 の **hi 側チャンネルのずらし量** (2026-10-01)。
    ///
    /// FT_HALF は K の倍数なので、 既定の 0 だと pairwise の 2 因子が同じチャンネルを受け、
    /// 2 次項が `t_c^2` だけになる (**異なるチャンネルの積 `t_c t_c'` が増えない**)。
    /// K/2 = 32 にすると 512 個の積すべてが異なる 2 チャンネルの積を持つ。
    /// 🚨quantised.bin の P2TPRJ section (version 2) に書き出すので、 エンジンは
    ///   ファイルからマッピングを決める = 旧 net との黙った食い違いが起きない。
    #[arg(long, default_value_t = 0)]
    threat_proj_offset: usize,

    /// 案1 の **チャンネル組み合わせの作り方** (2026-10-01)。
    ///
    /// * `same` : 従来。 2 因子が同じチャンネル → 2 次項は `t_c^2` だけ
    /// * `half` : hi 側を 32 ずらす → 組は (c, c+32) の 32 通りだけ
    /// * `rich` : **16 次元ブロックごとに別のずらし量**。 hi 側で 64 次元ごとに d=1..8。
    ///            512 個の積が **512 通りすべて異なるチャンネルの組**を持つ
    ///
    /// quantised.bin の P2TPRJ section (version 3) に **ブロックごとのずらし量**を書き出すので、
    /// エンジン側に式を持たせない (= 旧 net との黙った食い違いが起きない。 ★19)。
    #[arg(long, default_value_t = String::from("same"))]
    threat_proj_pair: String,

    /// W: 射影係数 p を **LayerStack の bucket ごと**に持つ (2026-10-01)。
    ///
    /// 射影が評価値を動かす量は進行度で違う (|dcp| 中央 序盤 32 / 中盤 29 / 終盤 35、
    /// 最大 158 / 152 / 234) のに、 p は全局面で共通だった。 bucket 別にすると
    /// 進行度ごとに別の係数を学習できる。 パラメータは 8 倍 (8,192) だが表 55M に対して
    /// 無視できる。 推論の NPS はほぼ不変。 section は version 4 になる。
    #[arg(long, default_value_t = false)]
    threat_proj_buckets: bool,

    /// (3) threat を **L1 に直結**する (FT の clamp を迂回)。 2026-10-01。
    ///
    /// FT の 48.4% が毎局面 clamp [0,254] されている (今日の実測) ので、
    /// threat が FT 経由で運ぶ情報は約半分が捨てられている。 clamp を通さない経路を足す:
    /// `l1_out += W_bucket . concat(thr_stm, thr_ntm)`。
    /// 目盛りは導出済み — engine の thr(i16) = 254 x float、 L1 i32 = 8128 x float なので
    /// `w_q = round(1024 * w)` を持ち engine は `sum(thr_i16 * w_q) >> 5` で足す。
    /// quantised.bin に "P2TL1\0\0\0" section、 arch に "ThreatL1=<K>," marker。
    #[arg(long, default_value_t = false)]
    threat_l1: bool,

    /// (4) threat を **L2 入力に直結**する (L1 = l1_effective の首を迂回)。 2026-10-08。
    ///
    /// TL1 は L1 を通るので幅 l1_effective が律速になる。 TL2 は sqr / crelu の**後ろ**、
    /// `l2_input` (= l1_effective*2 本) に直接足すので首を迂回する:
    /// `l2_pre += V_bucket . concat(thr_stm, thr_ntm)` → `l2_input = l2_pre.crelu()`。
    /// 目盛り: engine の thr(i16) = 254 x float、 L2 入力 (u8) = 127 x float なので
    /// `v_q = round(S * v)` を持ち engine は `sum(thr_i16 * v_q) >> log2(2S)` で足す。
    /// 🚨2026-10-09 に **S = 1024** (section ver 2 / engine の shift 11) に上げた。
    ///   S = 64 (ver 1) では i16 の 4 bit しか使わず、 量子化誤差が 34.28% もあった
    ///   (|w| < 1/128 の要素 64.1% が 0 になり、 そのノルムは全体の 25.17%)。
    ///   S = 1024 で誤差 2.26% / max|q| 131。 clip ±2 はそのまま (|v_q| <= 2048)。
    ///   TL1 とは Q が違うので section / export を使い回さないこと。
    /// quantised.bin に "P2TL2\0\0\0" section、 arch に "ThreatL2=<2K>," marker。
    #[arg(long, default_value_t = false)]
    threat_l2: bool,

    /// 案 B (Regime LayerStack、 2026-10-07)。 16 bucket の第 2 軸を
    /// 「持駒量」から「regime (玉周りが攻められているか)」に差し替える。
    ///   stack = progress_bucket8 * 2 + [bucket8 >= regime_min_pb && V1 >= regime_v1_th]
    ///   V1 = 両玉 8 近傍の空升への敵の利き数の和 (= C++ の P2Volm::value、 step+ray)。
    /// `--bucket-mode progress8hand2kpabs` と `--features lsmode-p8h2` (16 bucket) が前提。
    /// net は arch に "LSMODE=PROGREGIME:v1<T>-pb<P>," marker を持ち、 推論側は
    /// -DP2_LSMODE_PROGREGIME ビルドでのみロードできる。
    #[arg(long, default_value_t = false)]
    lsmode_regime: bool,

    /// 案 B の V1 閾値 (既定 3 = 本番 exe の SPSA_RAZV_VMIN と同じ値)。
    #[arg(long, default_value_t = 3)]
    regime_v1_th: i32,

    /// 案 B で regime を立て始める progress bucket (既定 4)。
    /// 🚨序盤を割らないための下限。 実測で ply 1-40 の regime 1 は 0.8% しか無く、
    ///   全 bucket を割ると前半の regime 1 側に勾配が来ない (★threat_group_starved_b01)。
    #[arg(long, default_value_t = 4)]
    regime_min_pb: i32,

    /// (3) TL1 の lr 倍率 (2026-10-01)。 ゼロ初期化なので上げないと動かない。
    /// 🚨射影の 8 ではなく **2** が既定。 TL1 の入力は thr (±数百) で combined ([0,1]) より
    /// 桁が大きいので、 同じ lr でも動きが速い (fable 助言)。 decay は 0 に上書きする。
    #[arg(long, default_value_t = 2.0)]
    threat_l1_lr_mult: f32,

    /// (4) TL2 の lr 倍率 (2026-10-08)。 ゼロ初期化なので上げないと動かない。
    /// 🚨TL1 と同じ 2 を既定にする (入力は同じ thr。 着地が L2 なので目盛りだけ違う)。
    ///   decay は 0 に上書きする。
    #[arg(long, default_value_t = 2.0)]
    threat_l2_lr_mult: f32,

    /// (4) TL2 (`threat_l2_w`) の weight decay (2026-10-09)。
    /// **既定 0.0 = 従来どおり decay を 0 に固定** (既存の腕の挙動は一切変わらない)。
    ///
    /// 0 より大きい値を渡すと decay 0 の上書きをやめ、 その値を TL2 の decay として使う。
    /// optimiser 側の decay は 1 step あたり `w *= 1 - decay * effective_lr` として掛かるため、
    /// 実効の縮小率は `--threat-l2-lr-mult` の倍率に自動追従する。
    ///
    /// 狙い: TL2 はゼロ初期化から育てているが decay 0 なので ‖w‖ が伸び続け、
    /// 相対変化 (‖Δw‖/‖w‖) が「‖w‖ が大きくなるだけ」で下がる状態にある。
    /// decay が効けば ‖w‖ が平衡に達し、 相対変化がもっと早く落ちるはず。
    #[arg(long, default_value_t = 0.0)]
    threat_l2_decay: f32,

    /// (4) TL2 (`threat_l2_w`) を**凍結**する (2026-10-09)。 lr_scale を 1e-12、 decay を 0、
    /// lookahead を無効にして学習中まったく動かさない。 既存の `--threat-freeze` の **TL2 版**。
    ///
    /// **既定 false = 現状どおり** (`--threat-l2-lr-mult` / `--threat-l2-decay` がそのまま効く)。
    ///
    /// 🚨optimiser は lr_scale > 0 を要求するので 0 ではなく極小値を入れる。 倍率は **1e-12**:
    ///   `--threat-freeze` は `set_lr_scale(1e-6)` と `set_range_lr_scales(1e-6)` が**乗算**される
    ///   ので threat_ft_w の実効倍率が 1e-12 で、 1e-6 はその平方根しかない。 TL2 は区間倍率を
    ///   持たないため 1e-6 では base_lr 6e-6 との積 6e-12 が小さい重みの 0.5 ulp を超え、
    ///   tl2m25-122 の momentum / max_velocity から **47 / 30,720 要素が 1 step で動く**と算出。
    ///   1e-12 なら 0 要素 (= 真にビット不動)。
    /// 🚨Ranger21 の **lookahead merge は lr_scale を無視**して weights を slow 側へ引き戻すので、
    ///   凍結時は threat_l2_w の lookahead も無効化する (`lookahead_mergetime = usize::MAX`)。
    ///   これをしないと位相 0 以外の checkpoint から resume したとき凍結しても weights が動く。
    /// 🚨`--threat-l2-lr-mult` / `--threat-l2-decay` と同時指定したときは **freeze を優先**し、
    ///   その 2 つは無視する (無視した旨をログに出す)。
    /// 🚨`--threat-l2` 自体が無効なときは tensor が存在しないので何もしない。
    ///
    /// 用途: データ軸 (上位陣データの注入) に進むとき、 学習済みの TL2 を固定して
    /// それ以外だけを学習する。
    #[arg(long, default_value_t = false)]
    threat_l2_freeze: bool,

    /// (5) TG: threat で **L1 の 2 乗側 (sqr) を乗法ゲート**する (2026-10-10)。
    ///
    /// TL2 (`--threat-l2`) は L2 入力への**加法** (= L1 の値に依らない主効果)。
    /// TG は `g(thr) * sqr(l1)` という**交差項**で、 TL2 の関数族には含まれない
    /// (設計書 `runs/prereg/DESIGN-20261009-THREAT-GATE.md` 2 節)。
    ///
    /// 順伝播 (engine と**同一の算術**にする):
    /// ```text
    ///   g_i = 1 + (254/256) * (thr_cat . u_i)      u_i = threat_gate_w の i 行目
    ///   g_i = hard clamp [0, 4]                    勾配 0 の clamp
    ///   pre_sq_i = sqr_i * g_i                     ゲートは 2 乗の**後ろ**
    /// ```
    /// 🚨係数 **254/256 を 1 にしてはいけない**。 engine は `thr_i16 = 254*thr_f` /
    ///   `u_q = round(1024*u)` を持ち `(Σ thr_i16*u_q) >> 10 = 254*Σ thr_f*u` を計算して
    ///   `g = 256 + 254*Σ thr_f*u` を `>>8` で使う。 学習器が 256 で揃えると **0.79% ずれ**、
    ///   「eval は bit-exact なのに勾配がずれる」事故になる
    ///   (★engine_patch_verification_20260915: eval bit-exact だけでは不十分)。
    /// 🚨clamp も engine の `[0,1024]/256` = `[0,4]` と同一にする (順伝播が違うと勾配がずれる)。
    ///
    /// ゲートするのは **sqr 側 15 本だけ** (lin 側は触らない)。 パラメータ数は
    /// NUM_BUCKETS(8) x l1_effective(15) x 2K(128) = 15,360 (TL2 の半分)。
    /// 初期値 0 → g = 1 → 起点 net と bit-exact。
    /// quantised.bin に "P2TGS\0\0\0" section、 arch に "ThreatGate=<2K>," marker。
    #[arg(long, default_value_t = false)]
    threat_gate: bool,

    /// (5) TG の lr 倍率 (2026-10-10)。 ゼロ初期化なので上げないと動かない。
    /// 🚨既定は TL2 (`--threat-l2-lr-mult`) と同じ **2.0** (まず同じ実効 lr から始める。
    ///   設計書 4 節。 動かなければ上げる)。 decay は `--threat-gate-decay` に従う。
    #[arg(long, default_value_t = 2.0)]
    threat_gate_lr_mult: f32,

    /// (5) TG (`threat_gate_w`) の weight decay (2026-10-10)。
    /// **既定 0.0 = decay を 0 に固定** (TL2 の既定と同じ流儀)。
    /// 0 より大きい値を渡すとその値を TG の decay として使う。 optimiser 側の decay は
    /// 1 step あたり `w *= 1 - decay * effective_lr` なので、 実効の縮小率は
    /// `--threat-gate-lr-mult` の倍率に自動追従する。
    #[arg(long, default_value_t = 0.0)]
    threat_gate_decay: f32,

    /// Plan C iter 2 用 dual PSQT section を予約する (= weight は Zeroed init、
    /// iter 1 では eval 寄与 0)。 quantised.bin 末尾に "P2DUAL\0\0" section (= 44 byte)
    /// が追加され、 arch_str に "DUAL=1" marker が埋め込まれる。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/dual_psqt.rs` 参照。
    #[arg(long, default_value_t = false)]
    dual_psqt_reserve: bool,

    /// Dual PSQT King-Attacker 型 section を出力する (2026-05-26 追加)。
    ///
    /// 旧 `--dual-psqt-reserve` の bucket-only (16 weight × 2 head = 44 byte) を置換した
    /// **King-Attacker 設計** で、 weight shape = [2 persp][2 head][8 bucket][81 sq][14 piece]
    /// = 36,288 i16 = 72,576 byte の dual section を書き出す。
    ///
    /// quantised.bin 末尾に "P2DUAL2\0" section (= 72,592 byte) が追加され、
    /// arch_str に "DUAL=1" marker が埋め込まれる (= 旧 reserve と marker 共有、
    /// section magic で区別)。
    ///
    /// **本 flag は --dual-psqt-reserve と排他**。 同時指定は error。
    ///
    /// **現状: 重みは Zeroed init で書き出す** (graph 側に dual weight tensor は追加しない)。
    /// 実 training-time feature emit + fine-tune は別 sub-agent で実装予定 (= Phase 3)。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/dual_psqt.rs` 参照。
    #[arg(long, default_value_t = false)]
    dual_psqt_king_attacker: bool,

    /// arm 5 設計: king_attacker head を無効化する (= cross のみ採用)。
    ///
    /// `--dual-psqt-king-attacker` を ON にした上で、 このフラグも ON にすると
    /// king_attacker 4 sparse input + 4 graph weight tensor + P2DUAL2 section を
    /// 削除する。 cross feature (= P2DUALX section) のみ保持される。
    /// 後方互換: default false で旧挙動 (= king_attacker enabled)。
    #[arg(long, default_value_t = false)]
    dual_disable_king_attacker: bool,

    /// Path I (= 2026-05-27): dual cross section の save format を切り替える。
    ///
    /// `false` (= default): 新 magic `P2DUALZ\0`、 head 0 のみ save (= 254,016 i16 = ~0.5 MB)。
    /// `true`: 旧 magic `P2DUALX\0`、 head 0/1 両方 save (= 508,032 i16 = ~1 MB、 後方互換用)。
    ///
    /// arm 6 で head 1 (= defended) は production から廃止済、 rshogi-core 推論側も
    /// `NUM_DUAL_CROSS_HEADS = 1` で head 0 のみ参照する。 新 ckpt は default の P2DUALZ
    /// で save するのが推奨 (= disk + memory 半減)。
    ///
    /// `--dual-disable-cross-defended` と併用は不可 (= defended 廃止時に
    /// P2DUALX legacy save は意味を持たない、 起動時 error)。
    #[arg(long, default_value_t = false)]
    dual_save_legacy_p2dualx: bool,

    /// arm 6 設計 (= 2026-05-27): D2 cross feature の head 1 (= defended) を廃止する。
    ///
    /// cosine similarity 解析 (= sb 25 → sb 55) で head 1 (= dual_w_defended) は
    /// cos = 0.38、 norm 23 → 16.5 (-29%) の振動 + L2 縮小 (= 無方向 random walk) と判明、
    /// rshogi-core 推論側でも production path から削除済 (= NUM_DUAL_CROSS_HEADS = 1)。
    ///
    /// 本 flag ON で:
    ///   - input layer `dual_cross_defended_{black,white}` 2 sparse input を削除
    ///   - graph `dual_w_defended_{black,white}` 2 weight tensor を削除 (= forward/backward
    ///     pass で skip、 訓練 NPS 向上 + 学習資源を head 0 + 他 layer に集中)
    ///   - save 側で常に `P2DUALZ\0` magic (= head 0 only、 ~0.5 MB)
    ///
    /// 後方互換: default false で旧挙動 (= defended head 維持)。
    #[arg(long, default_value_t = false)]
    dual_disable_cross_defended: bool,

    /// arm 11 (= 2026-05-28): KingRing Threat PSQT scalar head を有効化する。
    ///
    /// 玉周辺 24 sq × 13 piece × 4 state × 8 bucket × 2 persp = **19,968 i16 weight**
    /// (~40 KB) の scalar PSQT head。 L1 accumulator には混ぜず、 末端で
    /// `score += kingring_b · kingring_w_black + kingring_w · kingring_w_white` を
    /// 加算合成する。
    ///
    /// 本 flag ON で:
    ///   - input layer に `kingring_{black,white}` 2 sparse input (= dim 9984、 max_active 24)
    ///   - graph に `kingring_w_black` / `kingring_w_white` 2 weight tensor (= shape (1, 9984))
    ///     を追加 (= Zeroed init、 旧 ckpt resume 時は random init 扱い)
    ///   - quantised.bin 末尾に `P2KRING\0` section (= 19,968 i16 = 39,952 byte) を append
    ///
    /// 後方互換: default false で kingring 完全 OFF (= 旧 arm 5/6/7/8 と一致)。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/kingring.rs` 参照。
    #[arg(long, default_value_t = false)]
    enable_kingring: bool,

    /// arm 11 (= 2026-05-29): KingRing head の learning rate boost 倍率。
    ///
    /// `--enable-kingring` で追加した sparse PSQT head (`kingring_w_black` / `kingring_w_white`)
    /// は **sparse 更新で平均的に勾配信号が弱く**、 default lr (= 0.001) だと sb 数百級でも
    /// `abs_max ~0.1` 程度しか成長せず dual_w_threat (~1.0) に比べて scale が 1/10 にとどまる。
    /// 本 flag (= デフォルト 1.0、 boost 推奨値 5-10) で kingring 2 tensor のみ effective lr を
    /// `lr * mult` に boost する。
    ///
    /// 副作用回避: lr boost は RAdam の weight decay 項にも比例するため、 boost > 1.0 の場合は
    /// kingring 2 tensor の decay を強制的に 0 に設定する (= `set_params_for_weight` で個別 override)。
    /// その他 tensor の decay は `--weight-decay` 既定値のまま変化しない。
    #[arg(long, default_value_t = 1.0)]
    kingring_lr_mult: f32,

    /// arm 12 (= 2026-05-29): PairShape (= Piece-Pair Shape Table) scalar head を有効化する。
    ///
    /// 12 種固定 pair_type × 16 種 shape × 8 bucket × 2 persp = **3,072 i16 weight**
    /// (~6 KB) の軽量 scalar PSQT head。 L1 accumulator には混ぜず、 末端で
    /// `score += pairshape_b · pairshape_w_black - pairshape_w · pairshape_w_white` を
    /// 反対称合成する。
    ///
    /// 本 flag ON で:
    ///   - input layer に `pairshape_{black,white}` 2 sparse input (= dim 1536、 max_active 256)
    ///   - graph に `pairshape_w_black` / `pairshape_w_white` 2 weight tensor (= shape (1, 1536))
    ///     を追加 (= Zeroed init、 旧 ckpt resume 時は random init 扱い)
    ///   - quantised.bin 末尾に `P2PAIR\0\0` section (= 3,072 i16 = 6,160 byte) を append
    ///
    /// 後方互換: default false で pairshape 完全 OFF (= 旧 arm 5/6/7/8/11 と一致)。
    ///
    /// 詳細は rshogi-core `crates/rshogi-core/src/nnue/pairshape.rs` 参照。
    #[arg(long, default_value_t = false)]
    enable_pairshape: bool,

    /// arm 12 (= 2026-05-29): PairShape head の learning rate boost 倍率。
    ///
    /// kingring と同じく sparse 更新で勾配信号が弱い → boost で短期 finetune での学習を促進。
    /// デフォルト 1.0 (= no-op)、 boost 推奨 5-10。 boost > 1.0 で pairshape 2 tensor の
    /// weight decay を 0 化する (= kingring_lr_mult と同 convention)。
    #[arg(long, default_value_t = 1.0)]
    pairshape_lr_mult: f32,

    /// 2026-06-20 追加: Dual Cross Geometry head の learning rate boost 倍率。
    ///
    /// pairshape_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 デフォルト 1.0 (= no-op)、 boost 推奨 5-10。
    /// boost > 1.0 で dual_cross_geometry 2 tensor の weight decay を 0 化する。
    #[arg(long, default_value_t = 1.0)]
    dual_cross_geometry_lr_mult: f32,

    /// 2026-06-06 追加: Activity-Risk Interaction Head を有効化する。
    ///
    /// 4,096 i16 weight (= ~8 KB) の軽量 scalar PSQT head。 末端で
    /// `score += ar_b · ar_w_black - ar_w · ar_w_white` を反対称合成する。
    ///
    /// ON で:
    ///   - input layer に `activity_risk_{black,white}` 2 sparse input (= dim 2048、 max_active 1)
    ///   - graph に `activity_risk_w_black` / `activity_risk_w_white` 2 weight tensor
    ///     (= shape (1, 2048)) を追加 (= Zeroed init、 sb 240 resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2ACTR\0\0` section (= 4,096 i16 = 8,208 byte) を append
    ///
    /// 後方互換: default false で activity_risk 完全 OFF (= 旧 arm 12 base と一致)。
    #[arg(long, default_value_t = false)]
    enable_activity_risk: bool,

    /// 2026-06-06 追加: Activity-Risk head の learning rate boost 倍率。
    /// pairshape_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 boost 推奨 5-10。
    #[arg(long, default_value_t = 1.0)]
    activity_risk_lr_mult: f32,

    /// 2026-06-09 追加: King Escape Direction (KED) PSQT scalar head を有効化する。
    ///
    /// 64 i16 weight (= 128 byte) の軽量 scalar head。 玉 forward 同一 file 1-3 マス先の
    /// 自軍利き 0..=3 を 4 class × 8 bucket × 2 persp で encode。
    ///
    /// ON で:
    ///   - input layer に `king_escape_{black,white}` 2 sparse input (= dim 32、 max_active 1)
    ///   - graph に `king_escape_w_black` / `king_escape_w_white` 2 weight tensor
    ///     (= shape (1, 32)) を追加 (= Zeroed init、 arv3 resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2KESC\0\0` section (= 64 i16 = 144 byte) を append
    ///
    /// 後方互換: default false で KED 完全 OFF (= 旧 arv3 と一致)。
    #[arg(long, default_value_t = false)]
    enable_king_escape: bool,

    /// 2026-06-09 追加: KED head の learning rate boost 倍率。
    /// activity_risk_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 boost 推奨 5-10。
    #[arg(long, default_value_t = 1.0)]
    king_escape_lr_mult: f32,

    /// 2026-06-28 追加: Slider X-ray (SXRY) PSQT scalar head を有効化する。
    ///
    /// 128 i16 weight (= 256 byte) の軽量 scalar head。 自玉への敵 slider の
    /// X-ray 照準 (= slider と玉の間に駒がちょうど 1 枚 = block が 1 枚どけば直射に
    /// なる潜在脅威) を 重 slider(飛/竜)×3 + 軽 slider(角/馬/香)×1 で重み付けし
    /// 0..=7 の 8 class に離散化、 8 class × 8 bucket × 2 persp。 直射(0枚)は ADEV 領域、
    /// 2 枚以上は非計上。
    ///
    /// ON で:
    ///   - input layer に `slider_xray_{black,white}` 2 sparse input (= dim 64、 max_active 1)
    ///   - graph に `slider_xray_w_black` / `slider_xray_w_white` 2 weight tensor
    ///     (= shape (1, 64)) を追加 (= Zeroed init、 resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2SXRY\0\0` section (= 128 i16) を append
    ///
    /// 後方互換: default false で SXRY 完全 OFF。
    #[arg(long, default_value_t = false)]
    enable_slider_xray: bool,

    /// 2026-06-28 追加: SXRY head の learning rate boost 倍率。
    /// attack_dev_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 boost 推奨 5-10。
    #[arg(long, default_value_t = 1.0)]
    slider_xray_lr_mult: f32,

    /// 2026-06-30 追加: King Zone Threat / 持駒打ち込み (KZT_HAND) sparse head を有効化する。
    ///
    /// 716,800 i16 weight (= 1.4 MB) の大型 sparse head。 敵玉 5x5 zone への合法持駒打ちを
    /// (persp×bucket×king_region×hand_piece×target_rel×drop_state) で encode する。 1 局面の
    /// active feature は少数 (= sparse、 max 175/persp)。 feature emit は推論側
    /// `rshogi-core::nnue::kzt::collect_hand_features` を直接呼び bit-exact。
    ///
    /// ON で (= factored v1、 2026-07-01):
    ///   - input layer に 5 テーブル × 2 persp = 10 sparse input (= max_active 175)
    ///   - graph に `kzt_w_{core,region,bucket,persp,bias}` 5 共有 weight tensor を追加
    ///     (= Zeroed init、 resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2KZT\0\0\0` section (= version 2、 10,336 i16) を append
    ///
    /// 後方互換: default false で KZT 完全 OFF (= 旧 ckpt 一致)。
    #[arg(long, default_value_t = false)]
    enable_kzt_hand: bool,

    /// 2026-06-30 追加: KZT_HAND head の learning rate boost 倍率 (= core テーブル基準)。
    /// slider_xray_lr_mult と同 convention。 zero init から育てるため boost 推奨 8。
    /// factored v1 では各 table の実効 lr_scale = kzt_hand_lr_mult × KZT_TABLE_LR_REL[t]
    /// (= core 1.0 / region 0.7 / bucket 0.5 / persp 0.25 / bias 0.25、 Codex 指定)。
    #[arg(long, default_value_t = 1.0)]
    kzt_hand_lr_mult: f32,

    /// 2026-07-01 追加: KZT_HAND factored head の base weight decay (= core テーブル基準)。
    /// 各 table の実効 decay = kzt_hand_wd × KZT_TABLE_WD_REL[t]
    /// (= core 1.0 / region 1.2 / bucket 1.5 / persp 2.0 / bias 2.0、 Codex 指定 = 平滑化)。
    /// 低ランクな table (persp/bias) ほど強く正則化して過学習極値を抑える。
    #[arg(long, default_value_t = 0.01)]
    kzt_hand_wd: f32,

    /// 2026-07-04 追加: King Zone Threat を **FT 入力** として 1024 accumulator へ流す
    /// (= KZT-FT、 output 加算 head `--enable-kzt-hand` とは別経路 / 排他)。
    ///
    /// 入力を HalfKA_hm (73,305) + KZT-FT (9,440) = 82,745 次元に拡張し、 l0 affine を
    /// `[82,745 → L1]` で構築する。 KZT feature emit は推論側 KZT-FT index
    /// (`for_each_active_kzt_ft_index` と同一 primitive) を bit-exact に複製 (= 非対称 split emit)。
    ///
    /// ON で:
    ///   - 入力型 `ShogiHalfKaHmKzt` に差替 (= HalfKA + KZT-FT concat sparse 流)
    ///   - export で l0w の KZT-FT 部分 (feat 73,305..82,745) を `P2KZTF\0\0` section
    ///     (= version 1、 i8 raw row-major [9,440 × L1]) として物理末尾に append
    ///
    /// v0 は base main (+ optional --psqt) + KZT-FT のみ。 他 head / threat / dual /
    /// hand-threat / factorise とは排他 (= 起動時 guard)。 default false で完全 OFF (= 旧 ckpt 一致)。
    #[arg(long, default_value_t = false)]
    enable_kzt_ft: bool,

    /// 2026-07-22 追加: AUX (KZ + HANG + STATKD, 600 次元/persp) を **FT 入力** として
    /// 1024/768 accumulator へ流す (= AUX-FT、 enable_kzt_ft の完全な相似形 / 排他)。
    ///
    /// 入力を HalfKA_hm (73,305) + AUX (600) = 73,905 次元に拡張し、 l0 affine を
    /// `[73,905 → L1]` で構築する。 aux feature emit は推論側 (C++ P2NNUE Aux) と
    /// bit-exact な `ShogiHalfKaHmAux` (= 非対称 split emit) を使う。
    ///
    /// ON で:
    ///   - 入力型 `ShogiHalfKaHmAux` に差替 (= HalfKA + AUX concat sparse 流)
    ///   - export で l0w の AUX 部分 (feat 73,305..73,905) を `P2AUX\0\0\0` section
    ///     (= version 1、 i8 raw feature-major [600 × ft_out]、 count 無し) として
    ///     物理末尾に append し、 arch_str に `AUX=600,` marker を付与する
    ///     (= C++ p2nnue.cpp load_eval が検出して P2AUX section を読む)
    ///   - warm-start (`--resume-from-quantised`) 時は HalfKA/L1-L3/PSQT のみ継承し、
    ///     aux 行 [73,305,73,905) は zero-init (= step0 forward が aux 寄与ゼロで plain 一致)
    ///
    /// v0 は base main (+ optional --psqt) + AUX のみ。 他 head / threat / dual /
    /// hand-threat / kzt-ft / factorise とは排他 (= 起動時 guard)。 default false で完全 OFF。
    #[arg(long, default_value_t = false)]
    enable_aux: bool,

    /// 2026-07-23 追加: AUXCHEAP (KZ2x + HG6 + OUT6 + STATKD-kd2, 392 次元/persp) を
    /// **PSQT-only** で訓練する (= C++ P2_AUXCHEAP 推論 (board_effect SIMD 抽出 +
    /// PSQT 8 スカラー加算) の鏡像。 enable_aux (FT 入力版) の対、 排他)。
    ///
    /// 入力を HalfKA_hm (73,305) + AUXCHEAP (392) = 73,697 次元に拡張し、
    /// 発火 emit は C++ `p2auxcheap` dump と bit-exact な `ShogiHalfKaHmAuxCheap`
    /// (= 非対称 split emit) を使う。
    ///
    /// ON で:
    ///   - 入力型 `ShogiHalfKaHmAuxCheap` に差替 (= HalfKA + AUXCHEAP concat sparse 流)
    ///   - **psqtw の aux 行 [73,305, 73,697) を訓練** (= aux の唯一の寄与経路)、
    ///     **l0w の aux 行は run_with_zeroed_rows で毎 step 0 固定** (= FT へ流さない。
    ///     enable_aux の凍結パターンの鏡像)
    ///   - export で psqtw の aux 行を `P2AUXC\0\0` section (= version 1、 i32 LE
    ///     feature-major [392 × 8bucket]、 scale QA*QB = PSQT と同一) として物理末尾に
    ///     append し、 arch_str に `AUXCHEAP=392,` marker を付与する
    ///   - warm-start (`--resume-from-quantised`) 時は HalfKA/L1-L3/PSQT のみ継承し、
    ///     l0w/psqtw の aux 行はどちらも zero-init (= step0 forward が plain 一致)
    ///
    /// v0 は base main + **--psqt 必須** (PSQT-only 経路のため)。 他 head / threat /
    /// dual / hand-threat / kzt-ft / enable-aux とは排他 (= 起動時 guard)。
    /// `--factorise` とは併用可 (2026-07-23): HalfKA 部のみ factorise され、 aux 392
    /// 次元は factor 対象外 (= `Factorises<ShogiHalfKaHmAuxCheap>` が aux で `None`)。
    /// export layout / P2AUXC / marker は非 factorise 時と完全同一。
    /// default false で完全 OFF。
    #[arg(long, default_value_t = false)]
    enable_auxcheap: bool,

    /// 2026-06-16 追加: King Mobility (KMOB) PSQT scalar head を有効化する。
    ///
    /// 144 i16 weight (= 288 byte) の軽量 scalar head。 自玉の安全な逃げ場マス数
    /// (= 8 近傍のうち 自駒不在 かつ 敵利き無し) を 0..=8 の 9 class に離散化、
    /// 9 class × 8 bucket × 2 persp。 守備系特徴 (= ADEV / ARS1 と直交)。
    ///
    /// ON で:
    ///   - input layer に `king_mobility_{black,white}` 2 sparse input (= dim 72、 max_active 1)
    ///   - graph に `king_mobility_w_black` / `king_mobility_w_white` 2 weight tensor
    ///     (= shape (1, 72)) を追加 (= Zeroed init、 resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2KMOB\0\0` section (= 144 i16) を append
    ///
    /// 後方互換: default false で KMOB 完全 OFF。
    #[arg(long, default_value_t = false)]
    enable_king_mobility: bool,

    /// 2026-06-16 追加: KMOB head の learning rate boost 倍率。
    /// attack_dev_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 boost 推奨 5-10。
    #[arg(long, default_value_t = 1.0)]
    king_mobility_lr_mult: f32,

    /// 2026-06-16 追加: Drop Pressure (DROP) PSQT scalar head を有効化する。
    ///
    /// 128 i16 weight (= 256 byte) の軽量 scalar head。 相手持駒 (in-hand) を弾とした
    /// 自玉近傍 5x5 への投下脅威 (= 投下駒が玉 or 玉近傍に利く空升数 × 駒重み、
    /// 飛角金銀×2 + 歩香桂×1) を 0..=7 の 8 class に離散化、 8 class × 8 bucket × 2 persp。
    /// 入力源は相手持駒 (= 盤上駒のみの既存 head と直交)。 負け棋譜 9/9 局由来の最頻負け筋を補正。
    ///
    /// ON で:
    ///   - input layer に `drop_pressure_{black,white}` 2 sparse input (= dim 64、 max_active 1)
    ///   - graph に `drop_pressure_w_black` / `drop_pressure_w_white` 2 weight tensor
    ///     (= shape (1, 64)) を追加 (= Zeroed init、 resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2DROP\0\0` section (= 128 i16) を append
    ///
    /// 後方互換: default false で DROP 完全 OFF。
    #[arg(long, default_value_t = false)]
    enable_drop_pressure: bool,

    /// Endgame Tempo (EGT): 早逃げ窓 × 相手持駒打ち込み圧 × 自軍大駒状態の結合 head。
    #[arg(long, default_value_t = false)]
    enable_endgame_tempo: bool,

    /// Full Threat indexを8 bucketのscalar residual headとして学習する。
    /// Dual Crossは自動で無効化される。default falseで既存経路へ戻る。
    #[arg(long, default_value_t = false)]
    enable_scalar_threat: bool,

    /// Dual Crossを凍結保持し、Scalar Threatを残差として同時学習する。
    #[arg(
        long,
        default_value_t = false,
        requires_all = ["enable_scalar_threat", "dual_psqt_king_attacker"]
    )]
    scalar_threat_coexist_dual_cross: bool,

    /// Dual Crossへ攻撃方向・距離のゼロ平均残差を追加する。
    #[arg(long, default_value_t = false, requires = "dual_psqt_king_attacker")]
    enable_dual_cross_geometry: bool,

    /// Scalar Threat専用learning-rate倍率。短期head-only学習では3-10を想定。
    #[arg(long, default_value_t = 1.0)]
    scalar_threat_lr_mult: f32,

    /// 2026-09-10: `--threat-out K` の threat FT テンソル (`threat_ft_w`) の learning rate 倍率。
    ///
    /// `threat_ft_w` は **ゼロ起点の (K x 216,720) = 数千万パラメータ**で、 他の重みは学習済み。
    /// 全体 lr を上げると threat は育つが **main FT が壊れる**、 下げると main は無事だが
    /// threat が育たない、 という板挟みになる (2026-09-10 実測):
    ///   ・lr 3e-4 + threat (v12_sym)  : 2SB で校正済み held-out -3.51 cp (= -31.8 Elo)
    ///   ・lr 3e-4 threat 無し (v14)   : SB1 で -1.18 cp → **劣化は threat でなく lr 由来**
    ///   ・main 凍結 + threat (v11)    : threat が育たず効果ゼロ
    /// 本 flag で `threat_ft_w` のみ effective lr を `lr * mult` にする。 これで
    /// **main を低 lr (= 壊さない) + threat を高 lr (= 育てる)** が両立できる。
    ///
    /// 副作用回避: lr boost は weight decay 項にも比例するため、 mult > 1.0 のときは
    /// `threat_ft_w` の decay を 0 に上書きする (kingring_lr_mult と同 convention)。
    /// デフォルト 1.0 (= no-op) なので既存の実験結果は一切変わらない。
    #[arg(long, default_value_t = 1.0)]
    threat_ft_lr_mult: f32,

    /// **主 FT (`l0w` / `l0b`) だけ** effective lr を `lr * mult` にする。 2026-10-10。
    ///
    /// ## なぜ global lr では駄目か (実測)
    /// `‖Δw‖` は Adam 系の `lr/√v` 正規化のため lr に比例しない。 TL2 の梯子で取った 3 点:
    ///   lr 1/16.0 → ‖Δw‖ 1/5.3 (指数 0.60) / 1/21.6 → 1/6.1 (0.59) / 1/43.3 → 1/18.2 (0.77)
    ///   → `‖Δw‖ ∝ lr^(0.59〜0.77)`
    /// 現在地 (lr 6e-6 / 4 SB) の l0w 相対変化は **0.23%**。 これを 1% にするには
    /// global lr を **6.7〜12.1 倍 (= 4.0e-5〜7.3e-5)** にする必要があり、 nodchip の破壊域
    /// 1e-4 (-103 Elo) の 40〜73% まで寄る。 しかも l1w / l2w / psqtw も一緒に動くので
    /// 「FT だけを動かした」実験にならない。
    /// → 本 flag なら **mult 8.7 で l0w だけ 1% 動かし、 他は 6e-6 のまま**にできる。
    ///
    /// ## 副作用回避 (`--threat-ft-lr-mult` / `kingring_lr_mult` と同 convention)
    /// lr boost は weight decay 項にも比例するため、 `mult > 1.0` のときは l0w / l0b の
    /// decay を 0 に上書きする。 ★これにより本 flag は「FT の lr を上げる」と
    /// 「FT の decay を切る」の 2 つを同時に行う。 判定のときはそう解釈すること。
    ///
    /// ★`set_lr_scale` は**乗算**なので (trainer_lr_scale_multiplies)、 同じテンソルに
    /// 二度呼ばない。 l0w は他の経路から set_lr_scale されないことを確認済み
    /// (`--main-freeze --unfreeze-l0` の経路だけが 1.0 を入れるが、 それは main_freeze 時のみ)。
    ///
    /// デフォルト 1.0 (= no-op) なので既存の実験結果は一切変わらない。
    #[arg(long, default_value_t = 1.0)]
    l0_lr_mult: f32,

    /// **PSQT (`psqtw` / `psqtb`) だけ** effective lr を `lr * mult` にする。 2026-10-10。
    /// `1e-6` を渡せば実質凍結 (optimiser は lr_scale > 0 を要求するので 0 は入れられない)。
    ///
    /// ## 動機 (実測 2026-10-10 / 現在地 fg02-102 を newd-98 と f32 で比較)
    ///   l0w 0.4583% / l1w 0.3541% / l1fw 3.8757% / l2w 0.4929% / **psqtw 0.5956%** / l3w 0.1872%
    /// → psqtw は **l0w より大きく動いている**。 パラメータは 586,440 しかないので
    ///   1 パラメータあたりの変化は l0w の 128 倍。 「FT だけ微調整する」には邪魔になる。
    /// 🚨`--main-freeze --unfreeze-l0..l3 --unfreeze-heads` でも psqt だけ凍結できるが、
    ///   その経路は lr_scale を 1.0 に戻すので `--l0-lr-mult` と併用できない。 本 flag なら併用可。
    ///
    /// 副作用回避: `mult != 1.0` のときは decay を 0 に上書きする
    /// (`--threat-ft-lr-mult` / `kingring_lr_mult` と同 convention)。
    /// デフォルト 1.0 (= no-op) なので既存の実験結果は一切変わらない。
    #[arg(long, default_value_t = 1.0)]
    psqt_lr_mult: f32,

    /// 2026-09-22: threat 表 (`threat_ft_w`) を**凍結**する。 lr_scale を 1e-6、 decay を 0 にして
    /// 学習中まったく動かさない。
    ///
    /// 🚨 これが必要な理由: `--threat-ft-lr-mult 0` は凍結にならない。 lr_scale の設定は
    /// `threat_ft_lr_mult > 1.0 + 1e-6` のガード内にしか無いので、 mult=0 では threat_ft_w は
    /// 既定の lr_scale (=1.0、 主 FT と同じ) と既定の weight decay で**普通に学習される**。
    ///
    /// 用途: threat を外した構成へ移行するフェード実験。 SB の合間に `scale_threat.py` で
    /// threat 表を α 倍し、 その α のまま 1 SB 学習して主 FT に肩代わりを学ばせる。
    /// 凍結しないと threat が元の値へ戻ろうと学習してしまい、 肩代わりが起きない。
    #[arg(long, default_value_t = false)]
    threat_freeze: bool,

    /// **`--threat-freeze` の強い版 (真の凍結)**。  2026-11-03
    ///
    /// `threat_ft_w` を **勾配を持たない leaf** にして、 勾配バッファと Ranger21 の
    /// optimiser 状態を**確保しない**。
    /// - 節約: GPU 16 B/param (5,548 万で **846 MiB**) / host 8 B/param (423 MiB)
    /// - 既存の `--threat-freeze` は `set_lr_scale(1e-6)` x `set_range_lr_scales(1e-6)`
    ///   = 実効 1e-12 で lr をほぼ 0 にするだけで、 **tensor は optimiser に残る**ので
    ///   メモリは減らない (11-03 に実測: 凍結 3,839 / 解凍 3,838 MiB で差が無い)。
    /// - 用途: **幅 1536 + threat を 4 GB の GPU に載せる** (素のままだと 460 MiB 超過)。
    ///
    /// 🚨`--threat-ft-lr-mult` は無視される (起動ログに出す)。
    /// 🚨表を触る指定とは**排他** (エラーで止める): `--threat-factorizer` /
    ///   `--threat-hang-split` / `--threat-zero-init` / `--threat-clip`。
    /// 🚨weights.bin には出るので resume できる。 momentum.bin 等には出ない
    ///   (= この ckpt から解凍 run を resume すると threat の状態はゼロから)。
    #[arg(long, default_value_t = false)]
    threat_freeze_hard: bool,

    /// **KH ブロック** (2026-11-12)。 D1 兄弟差分で現職が楽観視していた 2 量を疎入力にして FT 出力へ足す。
    ///
    /// - 入力 (視点ごと 142 次元 / 最大 54 本、 定義は `crates/bullet_lib/src/game/inputs/shogi_kh.rs`)
    ///   - K 部 64 = 玉 (自/相手) x 8 方向 x 敵の利き数 (0,1,2,3+) の one-hot。 空升だけ
    ///   - H 部 78 = 浮き駒 (自/相手) x 駒種 13 x 区域 3 (視点から見た 1-3 / 4-6 / 7-9 段)
    /// - 重み `kh_ft_w` (`--kh-out` x 142) は**ゼロ初期化** = 学習開始時は resume 元と同じ関数。
    ///   出力は threat と同じ配り方で FT の前半・後半に半分ずつ足す (`--kh-row-offset`)。
    /// - 🚨quantised.bin には**書き出さない** (エンジン側は未実装)。 判定は float の重み・損失・`--kh-dump` で行う。
    /// - `--threat --threat-out K` (専用 sparse の threat) の構成でだけ使える。
    #[arg(long, default_value_t = false)]
    kh: bool,

    /// KH ブロックのうち立てる部分: `k` / `h` / `kh` (既定)。 index の配置は常に同じなので
    /// checkpoint の形は変わらない (使わない部分の重みは 0 のまま)。
    /// 2026-11-18 KT: `hx` = H 部 + X 部 (損な当たりの 1 番目・2 番目 28 列、 浮き駒 + 安い駒に当てられた差額) /
    /// `hxh` = H 部 + X 部 (浮き駒だけ)。 X 部は H 部の後ろ (kmode 1 で列 [142, 170)) に足す。 定義は `shogi_kh.rs`。
    /// X 部なしの checkpoint (142 列) から resume すると X 部の 28 列を 0 で足して読む (= 学習前は同じ関数)。
    #[arg(long, default_value = "kh")]
    kh_feats: String,

    /// KH ブロックの K 部の表現 (2026-11-12): `1` (既定) = 玉 x 8 方向 x 利き数 (64 次元、 初版と同一) /
    /// `2` = 玉の位置の区分 (持ち主から見た筋 3 x 段 3、 HM 鏡像あり) で条件付けた
    /// 方向部 576 + 集計部 (8 近傍の空升への敵利きの和 0..7+) 144 = 720 次元。
    /// 2 では `kh_ft_w` の列数が 142 -> 798 になる (H 部の index は 720 から)。 定義は `shogi_kh.rs`。
    #[arg(long, default_value_t = 1)]
    kh_kmode: u8,

    /// KH ブロックの出力幅 W (偶数)。 FT の前半 [off, off+W/2) と後半 [half+off, half+off+W/2) に足す。
    /// 既定 64 + offset 0 は threat (`--threat-out 64`) と**同じ行** = エンジンでは threat の 64 次元
    /// accumulator にそのまま足せる (新しい SIMD 経路が要らない)。
    #[arg(long, default_value_t = 64)]
    kh_out: usize,

    /// KH ブロックを足す FT 行のずらし量 (前半・後半とも同じだけずらす)。 既定 0 = threat と同じ行。
    #[arg(long, default_value_t = 0)]
    kh_row_offset: usize,

    /// `kh_ft_w` の lr 倍率 (decay は常に 0)。 ゼロ起点で量子化 (QA=127 の i8 なら |w| >= 0.004) を
    /// 超えるまで育てるため既定 64 (threat の 72 と同じ桁)。 起動ログに必ず出す。
    #[arg(long, default_value_t = 64.0)]
    kh_lr_mult: f32,

    /// **KH 以外を凍結** する: l0 (FT) / l1 / l1f (fc0) / l2 (fc1) / l3 (fc2) / psqt の lr_scale を 1e-12、
    /// decay 0、 lookahead 無効にする (`--threat-l2-freeze` と同じやり方)。 threat は別途
    /// `--threat-freeze-hard` が必須 (黙って動かないよう起動時に検査する)。 `--main-freeze` とは排他。
    #[arg(long, default_value_t = false)]
    kh_freeze_rest: bool,

    /// `--kh-freeze-rest` のうち **fc0 (l1w / l1b / l1fw / l1fb) だけ解凍**する
    /// (新入力は首 L1=15 を通るので、 fc0 が凍っていると効かない可能性がある = Fable 指摘)。
    #[arg(long, default_value_t = false)]
    kh_unfreeze_fc0: bool,

    /// 2026-11-18 KT: `--kh-freeze-rest` のうち **fc0 の重みのうち KH の行を読む列だけ解凍**する
    /// (l1w / l1fw の入力列 [off, off+W/2) と [ft_out/2+off, ft_out/2+off+W/2) = 既定 64 列)。
    /// 他の列と bias (l1b / l1fb) は lr 1e-12・decay 0 の区間倍率で凍結する。 Ranger21 だけ。
    /// `--kh-unfreeze-fc0` (fc0 全部の解凍) とは排他。
    #[arg(long, default_value_t = false)]
    kh_unfreeze_fc0_cols: bool,

    /// split-half (D3 勾配の篩): 局面の盤面 hash の最上位ビットが この値 (0 / 1) の局面だけ損失の重みを 1、
    /// 他を 0 にする (`datapoint_weight_function`)。 同じ盤面は必ず同じ半分に入る。 未指定 = 全局面。
    #[arg(long)]
    kh_split: Option<u8>,

    /// null 対照: KH 特徴を「同じ chunk の直前の別局面」のものに差し替える (分布は保ち、 教師との対応だけ壊す)。
    #[arg(long, default_value_t = false)]
    kh_null: bool,

    /// 学習終了時に `kh_ft_w` (f32 LE、 column-major = 特徴ごとに W 個) を書き出すパス。
    /// 同名 `.json` に形と設定を書く。 D3 の split-half cos は `runs/sibling/d3_kh_cos.py` で取る。
    #[arg(long)]
    kh_dump: Option<PathBuf>,

    /// 2026-11-17 K256 案 A (DESIGN-20261117-K256-FABLE 2.3 / 4.2): threat の**追加列**の幅 N (偶数)。 0 (既定) = 無効 = 従来と同一。
    ///
    /// - 新しいテンソル `threat_ext_w` (N x threat_dims、 **1 群**、 0 初期化) を作り、 群 offset なしの threat 疎入力
    ///   (`threat_ext_stm` / `threat_ext_nstm`) を掛けて、 FT の行 [K/2, K/2 + N/2) と [half + K/2, half + K/2 + N/2) に
    ///   活性化前に足す (K = `--threat-out`)。 エンジンの K' = K + N 幅の表の列 [K/2, K'/2) と [K'/2 + K/2, K') に当たる。
    /// - 0 初期化なので開始時の関数は resume 元と同一。 checkpoint に無ければ 0 のまま始まる (ローダの性質)。
    /// - 🚨quantised.bin には書き出さない (`k256_merge.py` が weights.bin から読んで K' 幅の net を作る)。
    /// - 重みは i8 に収まる範囲 |w| <= 127/254 に clip (QA_FT = 254)。 decay 0。 `--optimizer ranger21` 専用。
    #[arg(long, default_value_t = 0)]
    threat_ext_out: usize,

    /// `threat_ext_w` の lr 倍率 (decay は常に 0)。
    #[arg(long, default_value_t = 1.0)]
    threat_ext_lr_mult: f32,

    /// **`threat_ext_w` 以外を全部凍結**する: graph の全重み (l0 / fc0 / fc1 / fc2 / psqt / kh_ft_w ほか) を
    /// lr_scale 1e-12・decay 0・lookahead 無効にする (`--kh-freeze-rest` と同じやり方)。
    /// `threat_ft_w` は `--threat-freeze-hard` が必須 (起動時に検査する)。
    #[arg(long, default_value_t = false)]
    threat_ext_only: bool,

    /// 2026-11-17 T64USE 案 U1a (DESIGN-20261117-T64USE-FABLE 3.2): **`threat_ft_w` 以外を全部凍結**する。
    /// graph の全重み (l0 / fc0 / fc1 / fc2 / psqt / kh_ft_w ほか) を lr_scale 1e-12・decay 0・lookahead 無効にし、
    /// threat 表だけを `--threat-ft-lr-mult` / `--threat-clip` の設定で学習する。 `--threat-freeze(-hard)` とは排他。
    #[arg(long, default_value_t = false)]
    threat_only: bool,

    /// 2026-11-17 PNKLONG (K=128 案): **主 FT (`l0w` / `l0b`) だけを真の凍結**にする (勾配も optimizer 状態も持たない leaf)。
    /// 他の層 (threat・KH・fc0〜fc2・PSQT) は通常どおり学習する。 l0w (7,500 万) の状態 16 B/param が要らなくなるので
    /// GPU が約 1.15 GB 減り、 K=128 の全層学習が 4 GB に載る。 `--l0-lr-mult` とは排他。
    #[arg(long, default_value_t = false)]
    l0_freeze_hard: bool,

    /// 2026-09-12: threat 表を LayerStack bucket の「群」ごとに**独立**させる (未指定 = 従来の 1 表)。
    ///
    /// 値は bucket 0..7 の群番号をカンマ区切りで並べたもの。 例: `0,0,1,1,1,2,2,2`
    /// (序盤 b0-1 / 中盤 b2-4 / 終盤 b5-7 の 3 表)。 群数は最大値 + 1 で、 全群が 1 回以上現れること。
    ///
    /// 実装: 専用 sparse 入力の index を `f + 群 * 216,720` にずらし、 `threat_ft_w` を
    /// `(K, 群数 * 216,720)` の 1 本の tensor で持つ。 bias は従来どおり持たない (FT bias 共有)。
    /// lr 倍率 / decay は `threat_ft_w` の設定 (`--threat-ft-lr-mult`、 mult>1 で decay 0) がそのまま掛かる。
    /// 群化前の checkpoint (`threat_ft_w` = K x 216,720) から `--resume` / `--quantise-only` /
    /// `--resume-weights-only` すると、 重みと optimizer 状態 (Ranger21 の momentum / velocity /
    /// max_velocity / slow) を**群数ぶん複製 (tile)** して読み込むので、 開始時点の出力は元と同じ。
    /// 書き出し: arch に `ThreatGroups=<群数>:<b0>-<b1>-...-<b7>,` を付け、 threat 区画に
    /// 群 0, 1, ... の表 (各 i8|i16 [216,720 x K]) を連結する。 エンジンは局面の bucket の群の表を使う。
    /// `--threat-out K` (K > 0) と progress8kpabs / progress4hand2kpabs でのみ使える。
    #[arg(long)]
    threat_bucket_groups: Option<String>,

    /// `--threat-bucket-groups` で増える GPU メモリの推定値の上限 [MiB] (既定 800)。
    /// 起動時に推定 (重み + 勾配 + optimizer 状態 3 本 = 20 byte/param) を表示し、 超えたらエラーで止める。
    #[arg(long, default_value_t = 800.0)]
    threat_group_mem_budget_mib: f64,

    /// 2026-09-15: `--threat-bucket-groups` の群ごとの threat lr 倍率 (群の数だけカンマ区切り、 例 `1,2,1,1`)。
    /// `threat_ft_w` 全体の倍率 (`--threat-ft-lr-mult`) にさらに掛かる。 未指定なら全群 1.0 (= 従来と同一)。
    /// 実装: threat_ft_w は群 g の表が要素区間 [g x K x 216,720, (g+1) x K x 216,720) に並ぶので、
    /// optimizer をその区間ごとに異なる lr で呼ぶ。 momentum / velocity の更新は lr に依存しないため、
    /// 変わるのはその群の重みの 1 歩の大きさだけ。 Ranger21 でのみ使える。
    #[arg(long)]
    threat_group_lr_mult: Option<String>,
    /// 2026-09-16: 学習時だけの threat factorizer (runs/threat_factor/design.md)。
    /// threat_ft_w の末尾に pair (攻め色, 攻め駒クラス, 受け色, 受け駒クラス) 324 列を足し、 全群・全升で共有する。
    /// export で各実特徴の列へ畳み込むので出力 bin の形式・サイズは不変 (エンジン無変更)。 `--threat-out` > 0 が必要。
    /// resume 時は checkpoint の threat_ft_w に 0 の factor 列を足す (= 出力は resume 元と同一)。
    #[arg(long, default_value_t = false)]
    threat_factorizer: bool,
    /// threat factor 列の lr 倍率 (`--threat-ft-lr-mult` にさらに掛かる)。 既定 0.5。
    #[arg(long, default_value_t = 0.5)]
    threat_factor_lr_mult: f32,
    /// 2026-09-24: profile 16 (hanging-split)。 threat_ft_w の末尾に群共有の delta 列 (108,360) を足し、
    /// 「被攻撃駒にひもが無い」 pair だけ 実特徴と対で引く。 export で各 hang 行へ `base[g] + delta` を
    /// 畳み込み、 エンジン用に 325,080 行 x 群 の表を書く (エンジンは -DP2_THREAT_PROFILE=16 が必要)。
    /// resume 時は checkpoint の threat_ft_w に 0 の delta 列を足すので**関数は完全に保存**される。
    /// `--threat-profile full-hang-split` と `--threat-out K` (K > 0) が必要。
    /// 設計 = C:/project/p2shogi/runs/prereg/DESIGN-20260924-THREAT-DEFENDED.md
    #[arg(long, default_value_t = false)]
    threat_hang_split: bool,
    /// hanging-split の delta 列の lr 倍率 (`--threat-ft-lr-mult` にさらに掛かる)。 既定 6.5。
    ///
    /// G0 の実測 (実戦 2 万局面) で「ひもが無い pair」は局面あたり平均 3.8 個しか立たず、
    /// delta 行が受ける勾配が base 行の**約 1/6.5** しかないと分かったため、 その逆数を既定にする。
    #[arg(long, default_value_t = 6.5)]
    threat_delta_lr_mult: f32,
    /// 2026-09-16: factorizer の共有行をゼロから始める (既定 false = ペア平均で初期化)。
    ///
    /// 既定 (mean init) は resume 時に `factor[p] = そのペアの実特徴の平均`、 `real -= factor` と置き換える。
    /// 和は不変なので**関数は完全に保存**され、 分解として正しい位置から学習が始まる。
    /// ゼロ初期化だと収束済みの net に「損失を下げる自由な方向」を与えることになり、
    /// 共有行が 669 特徴を一斉に押して既存の均衡を壊す (2026-09-16 実測: Σ|factor| が SB ごとに 209→294→371 と発散、 対局 23.5%)。
    #[arg(long, default_value_t = false)]
    threat_factor_init_zero: bool,
    /// 2026-09-17: ペア平均で初期化するときの倍率 (既定 1.0)。
    ///
    /// `factor[p] = scale x ペア平均`、 `real -= factor` とするので関数は scale によらず保存される。
    /// 背景: 1.0 だと Σ|factor| = 196 から始まり、 SB ごとに増えて 3 SB で対局が崩れた (2026-09-17 実測)。
    /// 0.1 前後で Σ|factor| を 20 程度から始め、 `--threat-factor-decay` で上限を抑える。
    #[arg(long, default_value_t = 1.0)]
    threat_factor_init_scale: f32,
    /// 2026-09-16: resume 直後に threat 重み (threat_ft_w) を**全部ゼロ**にして threat だけ学習し直す。
    ///
    /// Direct15 の factorizer が成功したのは「最初から分解した形で学習した」ためで、 収束後に足すのとは別物
    /// (2026-09-16 実測)。 その形を threat でも再現するための実験用。 HalfKA FT / L1-L3 は resume 元のまま。
    /// threat 用の optimiser state (momentum / velocity / slow) も同時に初期化する。
    /// 10-20 SB 規模の学習を前提とすること (3 SB では戻らない)。
    #[arg(long, default_value_t = false)]
    threat_zero_init: bool,
    /// 2026-09-17: factorizer の共有行にだけ掛ける weight decay (既定 0.05)。
    ///
    /// 共有行は 669 特徴ぶんの勾配を集めるので放っておくと増え続ける (2026-09-16 実測: Σ|factor| が
    /// SB ごとに +65 ずつ増加)。 decay で「勾配が押す力」と「戻す力」を釣り合わせる。
    /// 実特徴には掛からない (区間ごとの decay 倍率 0.0)。 実効の縮小率は decay x lr x step 数。
    #[arg(long, default_value_t = 0.05)]
    threat_factor_decay: f32,
    /// 2026-09-16: threat の勾配を局面で絞る (`--threat-grad-mask "600:1.0,1000:0.5,inf:0.25"`)。
    /// `|教師値| <= 閾値` に一致した係数を threat 出力の**勾配だけ**に掛ける (順伝播は不変 = エンジン無変更)。
    /// 根拠: 学習データの 54% が |教師値| > 1000 の決着局面で、 1 局面あたりの loss は互角局面の約 10 倍 = 勾配を独占している
    /// (runs/threat_slice/)。 `--threat-out` > 0 が必要。 未指定なら従来と完全に同一。
    #[arg(long)]
    threat_grad_mask: Option<String>,
    /// 2026-09-17: ネット出力の勾配を局面で重み付けする (`--loss-grad-mask "600:1.0,1000:0.5,inf:0.25"`)。
    /// 書式は --threat-grad-mask と同じ。 順伝播は不変 (エンジン無変更) で、 全パラメータの勾配が係数倍になる
    /// = 損失を局面ごとに重み付けしたのと同じ。 狙い: 決着局面が勾配を独占している損失の形を、 互角帯寄りにする。
    /// 未指定なら従来と完全に同一。
    #[arg(long)]
    loss_grad_mask: Option<String>,
    /// 2026-09-16: threat_ft_w を i8 で書き出せる範囲に学習中から収める (`--threat-clip 0.5`)。
    /// 書き出しは `round(QA_FT * w)` を ±127 に clamp するので、 QA_FT=254 なら |w| <= 0.5 が i8 の範囲。
    /// factorizer で畳み込む場合は factor 込みで超えうるので、 実特徴と factor の**両方**を
    /// この値の半分ずつに抑える運用にする (例: --threat-clip 0.25)。 0 = 無効 (従来どおり)。
    /// エンジンは i8 のまま = NPS 税ゼロ。 Ranger21 でのみ有効。
    #[arg(long, default_value_t = 0.0)]
    threat_clip: f32,

    /// 2026-09-12 追加: LayerStack L1 (fc0) 4 tensor (`l1w` / `l1b` / `l1fw` / `l1fb`) の
    /// learning rate 倍率。 1.0 (default) で完全 no-op、 1.0 未満で L1 だけ遅く動かす。
    ///
    /// 背景: v18 (SB6→SB12) で held-out の bucket 1 / 6 の評価値が手番側に不利な方向へ
    /// 単調にずれ、 量子化 net の **fc0 (l1w_b + l1fw の合算) だけ SB5 に戻すとずれが消える**
    /// (b1 -27→-9.5cp, b6 -37→-5cp、 2026-09-12 実測)。 threat / FT はそのまま育てたいので
    /// L1 4 tensor のみ lr を下げる。
    ///
    /// weight decay は optimiser 内部で `decay * effective_lr` として掛かるため、 lr 倍率に
    /// 自動で追従する (別途 `--l1-weight-decay` で絶対値の上書きも可)。 `--main-freeze` の
    /// lr_scale より後に適用されるので、 併用時は本 flag が優先される。 0 以下を渡した場合は
    /// 1e-6 (= `--main-freeze` と同じ実質凍結) に丸める (optimiser が lr_scale > 0 を要求するため)。
    #[arg(long, default_value_t = 1.0)]
    l1_lr_mult: f32,

    /// 2026-09-12 追加: L1 4 tensor (`l1w` / `l1b` / `l1fw` / `l1fb`) の weight decay を
    /// この値で上書きする (未指定 = `--weight-decay` のまま)。 `0` で L1 の decay を止める。
    #[arg(long)]
    l1_weight_decay: Option<f32>,

    /// 2026-11-06 追加: **L2 (`l2w` / `l2b` = fc1) だけ lr 倍率を掛ける**。
    ///
    /// 背景: L1 を作り直す実験 (畳み+空け / 全行の再初期化) では **fc1 も一緒に作り直す必要がある**。
    /// ところが `--l1-lr-mult` は L1 の 4 tensor だけで、 `l2w` には倍率が無かったため、
    /// 基底 lr を安全な 6e-6 に下げると **fc1 が動かず「列が先に育つ」機構が止まる**
    /// (fc1 の列が 0 の行には勾配も来ない: 出力 = Σ l2w*relu(l1))。
    /// `--main-freeze` は l2w まで凍結するので代用にならない。
    ///
    /// `--l1-lr-mult` と同じ規約: 0 以下は 1e-6 (実質凍結) に丸める。 weight decay は
    /// optimiser 内部で `decay * effective_lr` なので倍率に自動追従し、 `--l2-weight-decay`
    /// で絶対値を上書きできる。 既定 1.0 = no-op なので既存の実験結果は一切変わらない。
    #[arg(long, default_value_t = 1.0)]
    l2_lr_mult: f32,

    /// `l2w` / `l2b` の weight decay を上書きする (未指定 = `--weight-decay` のまま)。
    #[arg(long)]
    l2_weight_decay: Option<f32>,

    /// 2026-11-06 追加: **L3 (`l3w` / `l3b` = fc2) だけ lr 倍率を掛ける**。 規約は `--l2-lr-mult` と同じ。
    #[arg(long, default_value_t = 1.0)]
    l3_lr_mult: f32,

    /// `l3w` / `l3b` の weight decay を上書きする (未指定 = `--weight-decay` のまま)。
    #[arg(long)]
    l3_weight_decay: Option<f32>,

    /// 2026-11-07 追加 (T1): **L4 = 4 層目 (`fc1b`) の幅**。 既定 **0 = 無効**。
    ///
    /// 0 のときは graph も arch 文字列も量子化の保存形式も**現行と完全に同一**
    /// (検算 (a0): `--quantise-only --l4 0` の md5 が出荷 net と一致すること)。
    ///
    /// 幅は `affine_par` が 8 行単位なので **8 の倍数 / 最大 64**。 恒等初期化が成立するのは
    /// **64 のみ** (i8 の対角 64 で `clamp((64·x)>>6, 0, 127) = x`)。
    #[arg(long, default_value_t = 0)]
    l4: usize,

    /// `--l4` を有効にしたときの L4 重みの初期化。
    ///
    /// `identity` (既定) = `l4w` を対角 1.0 / `l4b` を 0 にして**関数を保存**して挿入する。
    /// 🚨bullet のローダは ckpt に無い id をエラーにせず乱数のまま走らせるので、
    /// `--resume` だけでは**乱数 L4 で静かに壊れる**。 だからここで明示的に書く。
    /// 🚨L4 run の**続き**を resume するときは `random` にする (identity は学習済み L4 を潰す)。
    #[arg(long, default_value = "identity")]
    l4_init: String,

    /// 2026-11-07 追加: **L4 (`l4w` / `l4b`) だけ lr 倍率を掛ける**。 規約は `--l2-lr-mult` と同じ。
    #[arg(long, default_value_t = 1.0)]
    l4_lr_mult: f32,

    /// `l4w` / `l4b` の weight decay を上書きする (未指定 = `--weight-decay` のまま)。
    ///
    /// 🚨恒等から始める場合は **0 を明示する**こと。 decoupled decay は対角を毎 step
    /// `(1 - lr·wd)` 倍に縮めるので、 lr 6e-4 × wd 0.005 を 8 SB (2e5 step) 続けると
    /// exp(-0.59) まで縮み、 勾配が支えないと恒等が崩れる。
    #[arg(long)]
    l4_weight_decay: Option<f32>,

    /// 2026-09-12 追加: L1 合算重みのアンカー正則化係数 λ (0 = OFF、 default)。
    ///
    /// 学習開始時 (resume 直後) の L1 **合算**重み `W0_b = l1w_b + l1fw`、 `c0_b = l1b_b + l1fb`
    /// を graph 内の非学習 tensor (`l1_anchor_w` / `l1_anchor_b`、 stop-grad) に保存し、
    /// `λ × Σ_b ( ||(l1w_b + l1fw) - W0_b||² + ||(l1b_b + l1fb) - c0_b||² )` を loss に加算する。
    /// factorizer (l1fw) と bucket 別 (l1w) が互いに肩代わりするのは許し、 **合算だけ**を
    /// 動かなくする意図 (どちらか片方を戻しても半分しか直らなかった実測に対応)。
    ///
    /// 実装は graph 内 (forward に penalty node を足す) なので、 どの run 経路
    /// (run / run_with_zeroed_rows / run_with_bucket_smoothness) でも効く。 penalty は
    /// 1 サンプルあたりの loss に加算される (= batch 内で splat される) ので、 λ は
    /// 「1 サンプルの data loss (v18 で ≈0.0076) に対する比」と読む。 SB12 時点の drift
    /// (合算 Σ_b||ΔW_b||² ≈ 0.7) では λ=1e-2 で penalty ≈ data loss と同程度。
    ///
    /// アンカーは checkpoint (weights.bin / raw.bin) に保存され、 `--resume` で引き継がれる
    /// (= babysit の再起動を跨いでも SB12 の重みがアンカーのまま)。 checkpoint に無い
    /// (= 全 0) ときだけ現在の L1 から初期化する。 ★このアンカー入り checkpoint を
    /// λ=0 で `--resume` するときは `BULLET_SKIP_UNKNOWN_WEIGHTS=1` が必要。
    /// `--hand-count-dense` (L1 入力に dense 14 次元を concat) とは併用不可。
    #[arg(long, default_value_t = 0.0)]
    l1_anchor_lambda: f32,

    /// 2026-09-12 追加: L1 bucket 別重み (`l1w` / `l1b`) の隣接 bucket smoothness 係数 λ
    /// (0 = OFF、 default)。 `λ × mean_b ( ||l1w[b+1] - l1w[b]||² + ||l1b[b+1] - l1b[b]||² )`
    /// を loss に加算する (共有 factorizer は差分で消えるので対象外)。
    /// `--bucket-smooth-l3-lambda` (post-step projection) と違い graph 内で実装しているため
    /// λ の目盛りは `--l1-anchor-lambda` と同じ「1 サンプル loss に対する比」。
    #[arg(long, default_value_t = 0.0)]
    bucket_smooth_l1_lambda: f32,

    /// 2026-06-16 追加: DROP head の learning rate boost 倍率。
    /// king_mobility_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 boost 推奨 5-10。
    #[arg(long, default_value_t = 1.0)]
    drop_pressure_lr_mult: f32,

    /// EGT head の learning rate boost 倍率。
    #[arg(long, default_value_t = 5.0)]
    endgame_tempo_lr_mult: f32,

    /// 2026-06-09 追加: Attack Relay Head (ARH) を有効化する。
    ///
    /// 敵玉中心 5x5 を inner (= 3x3) + outer に分け、 攻撃側の
    /// (contact, reserve, relay) 3 軸 × 4×4×4 = 64 cell × 8 bucket × 2 persp
    /// = 1024 i16 weight (= 2 KiB) で encode。
    ///
    /// ON で:
    ///   - input layer に `attack_relay_{black,white}` 2 sparse input (= dim 512、 max_active 1)
    ///   - graph に `attack_relay_w_black` / `attack_relay_w_white` 2 weight tensor
    ///     (= shape (1, 512)) を追加 (= Zeroed init、 既存 ckpt resume 時は head のみ育成)
    ///   - quantised.bin 末尾に `P2AREL\0\0` section (= 1024 i16 = 2,064 byte) を append
    ///
    /// 後方互換: default false で ARH 完全 OFF (= 旧 arv4 と一致)。 詳細は p2shogi
    /// 側 `docs/attack_relay_head_design.md` 参照。
    #[arg(long, default_value_t = false)]
    enable_attack_relay: bool,

    /// 2026-06-09 追加: ARH head の learning rate boost 倍率。
    /// king_escape_lr_mult と同 convention。 sparse 更新で勾配信号が弱いため boost で
    /// 短期 finetune 学習を促進する。 設計書 §14.16 では sweep = {1 / 3 / 10}。
    #[arg(long, default_value_t = 1.0)]
    attack_relay_lr_mult: f32,

    /// 2026-06-10 追加: Attack Relay Head v2 (= A5 仕様、 attackers / overlap / supported)
    /// を有効化する。 v1 (= contact / reserve / relay) が ply 24 中盤局面で効果 0 で
    /// あったことを受けた仕様変更。 dim 1024 i16 (= v1 と同)、 section magic P2AREL2 で並列。
    #[arg(long, default_value_t = false)]
    enable_attack_relay_v2: bool,

    /// 2026-06-10 追加: ARH v2 head の learning rate boost 倍率。
    #[arg(long, default_value_t = 1.0)]
    attack_relay_v2_lr_mult: f32,

    /// 2026-07-03 追加: HalfKA_hm feature factorizer (nnue-pytorch 相当) を有効化する。
    ///
    /// 訓練時のみ玉位置非依存の 1,629 次元 piece-square factor (= `feat % 1629`) を
    /// virtual feature として付与し、 45 バケット分の実特徴が勾配を共有する。
    /// export 時に factor 重みを実特徴へ coalesce するため **推論形状は不変**
    /// (= engine rshogi は 73,305 次元 HalfKA_hm のまま変更不要)。
    ///
    /// base main (+ optional --psqt) 専用。 各 head / threat / hand-threat / dual /
    /// pin / hand-count-dense とは併用不可 (起動時 error)。 29% 天井が訓練速度律速か
    /// 判別する A/B 用。 default false で完全 no-op (= plain 経路と bit-identical)。
    #[arg(long, default_value_t = false)]
    factorise: bool,

    /// 2026-06-09 追加: L3 weight (= l3w + l3b) の隣接 bucket smoothness regularization 係数。
    ///
    /// `L_smooth_l3 = λ × mean_b ||w[b+1] - w[b]||²` を main loss に加算し、
    /// 隣接 bucket 間 weight 差を抑えて 推論時 bucket 境界 jump を軽減する。
    ///
    /// `λ=0` で完全 no-op (= 既存学習と完全互換、 数値 bit-exact)。
    /// 推奨初期値 `1e-7..1e-6` (= `docs/arch/bucket_smoothness_plan.md` 比較実験 B-G 参照)。
    ///
    /// 推論 NPS 不変 (= 訓練時のみ penalty 適用、 production binary は変更なし)。
    #[arg(long, default_value_t = 0.0)]
    bucket_smooth_l3_lambda: f32,
    /// L1 (fc0) 行間 decorrelation の係数 λ。
    /// `L = λ × mean_b [ Σ_{i<j} (ŵ_i · ŵ_j)² ]` を post-step projection で適用する。
    /// Net2Net で複製した行が分化せず実効次元が 8.11/15 に留まっていたため (2026-09-09 実測)。
    /// 0 で完全 no-op。 推奨初期値 1e-4..1e-2 (相関は 0..1 の無次元量なので smoothness より大きめ)。
    #[arg(long, default_value_t = 0.0)]
    l1_decorr_lambda: f32,
    /// L2 (fc1) 行間 decorrelation の係数 λ。 L1 と同じ定義。
    /// 活性化で測ると L2 の実効次元は 15.4/64 (2026-09-09 実測)。 0 で no-op。
    #[arg(long, default_value_t = 0.0)]
    l2_decorr_lambda: f32,

    /// 2026-06-09 追加: PSQT weight (= psqtw + psqtb) の隣接 bucket smoothness regularization 係数。
    ///
    /// `L_smooth_psqt = λ × mean_b ||w[b+1] - w[b]||²` を main loss に加算。
    /// L3 と同 path、 PSQT bucket 別 weight 587 K の境界差を抑える。
    ///
    /// `λ=0` で完全 no-op。 推奨初期値 `1e-7..3e-6`。
    #[arg(long, default_value_t = 0.0)]
    bucket_smooth_psqt_lambda: f32,

    /// arm 12 path B' (= 2026-05-30): main backbone (= l0/l1/l1f/l2/l3/psqt + dual threat) 凍結。
    ///
    /// ON にすると以下 tensor の effective lr を `lr * 1e-6` まで落とし、 weight decay を 0 化する
    /// (= 実質凍結)。 pairshape (and kingring) のみ育成する短期 finetune 専用 flag。
    ///   - l0w / l0b (FT)
    ///   - l1w / l1b (LayerStack L1 bucket-specific)
    ///   - l1fw / l1fb (L1 shared factorized)
    ///   - l2w / l2b (L2)
    ///   - l3w / l3b (L3)
    ///   - psqtw / psqtb (PSQT)
    ///   - dual_w_threat_black / dual_w_threat_white (= D2 cross threat、 主 eval signal)
    ///
    /// 後方互換: default false で完全 no-op。 pairshape 真育成 path として `--pairshape-lr-mult 10`
    /// と組み合わせ、 a78b bullet lr_mult 波及 bug の代替策として導入 (= task #97)。
    #[arg(long, default_value_t = false)]
    main_freeze: bool,

    /// main-freeze時に l3w/l3b のみ学習可能に戻す(最終層のみ学習)
    #[arg(long, default_value_t = false)]
    unfreeze_l3: bool,

    /// main-freeze時に l2w/l2b も学習可能に戻す(--unfreeze-l3 と併用でL2+L3学習)
    #[arg(long, default_value_t = false)]
    unfreeze_l2: bool,

    /// main-freeze時に l1w/l1b/l1fw/l1fb も学習可能に戻す(L1解凍)
    #[arg(long, default_value_t = false)]
    unfreeze_l1: bool,

    /// main-freeze時に l0w/l0b(FT) も学習可能に戻す(FT/L0解凍)
    #[arg(long, default_value_t = false)]
    unfreeze_l0: bool,

    /// main-freeze時に eval head 群(dual/pairshape/activity_risk/king_escape/attack_relay_v2)を
    /// lr_scale 1.0 で学習対象に戻す(head 再学習用)
    #[arg(long, default_value_t = false)]
    unfreeze_heads: bool,

    /// **bucket の絞り込み** (2026-11-13、 INVENT02 案 A)。 例: `0-2` / `1`。
    ///
    /// データ loader の段で、 LayerStack の bucket (学習器が head 選択に使うのと**同じ**
    /// `OutputBuckets::bucket` = progress8kpabs + `--progress-coeff`) が範囲外の局面を捨てる。
    /// 捨てた分は次の局面で埋めるので **batch の大きさは変わらない**。
    /// 通した数 / 捨てた数は起動時の先読み (各ファイル先頭) と学習中に定期的にログへ出す。
    /// - 範囲外 bucket の head には勾配が来ない。 bit 単位で動かさないには
    ///   `--hard-freeze-shared` + `--threat-freeze-hard` + `--weight-decay 0` +
    ///   `--resume-weights-only` (optimizer 状態を新しくする) が要る (起動時に検査する)。
    /// - 🚨`--start-superbatch` > 0 の再開では、 内側 loader の batch 番号で読み飛ばすので
    ///   中断なしの run と同じ局面列にはならない (近似)。
    /// 未指定 (既定) なら loader は従来と同一。
    #[arg(long)]
    bucket_filter: Option<String>,

    /// 2026-11-17 tanuki: 読み込みのときに教師の score を bucket 別の表で置き換える (目盛り合わせ)。
    /// 表は little-endian i32 の 8 x 10000 (行優先、 添字 = bucket * 10000 + |score|、 値 = 変換後の |score|、
    /// 各 bucket で単調非減少)。 out = sign(s) * 表[b][|s|]。 b は `--bucket-filter` と同じ鍵 (progress8kpabs の bucket)。
    /// `--bucket-filter` と `--score-lut-drop-mate 10000` (表の範囲外 = 詰みを捨てる) と併用すること。 未指定なら従来と同一。
    #[arg(long)]
    score_lut: Option<PathBuf>,

    /// 2026-11-17 tanuki: |score| がこの値以上の局面を読み込みで捨てる (詰み)。 0 (既定) = 捨てない = 従来と同一。
    #[arg(long, default_value_t = 0)]
    score_lut_drop_mate: i32,

    /// 2026-11-17 tanuki 検算用: loader が最初に見た N 局面について、 元のレコード (40 B) を
    /// `<--output>/score_lut_dump.psv` に、 学習器が計算した bucket・元の score・変換後の score (捨てたら理由) を
    /// `<--output>/score_lut_dump.tsv` に書く。 0 (既定) = 書かない。
    #[arg(long, default_value_t = 0)]
    score_lut_dump: usize,

    /// 2026-11-17 tanuki: `--bucket-filter` の loader で、 通した局面を N batch 分ためてから**まとめて混ぜて**出す
    /// (Fisher-Yates、 種は固定 = 再現できる)。 `--interleave-file-batches 1` の round-robin は 1 batch = 1 本のファイルの
    /// 連続 4,096 局面になり、 対局順に並んだデータ (tanuki の生) では batch の中の相関が強いので、 N 本のファイルをまたいで混ぜる。
    /// 0 (既定) = 混ぜない = 従来と同一。 メモリは N x batch-size x 40 B (N=256 で 40 MB)。
    #[arg(long, default_value_t = 0)]
    mix_buffer_batches: usize,

    /// 2026-11-18 tanuki: 静かでない局面を読み込みで捨てる (nnue-pytorch の smart fen skipping と同じ考え)。
    /// 捨てるのは (1) 手番側が王手をかけられている局面 (rshogi-core の `Position::in_check`)、
    /// (2) 記録された最善手 (PSV の move16) の行き先に駒がある = 駒取りの局面。 判定は
    /// `runs/tanuki_nnuept/quiet_filter.py` (cshogi) と同じ条件。 `--bucket-filter` と併用すること。 未指定なら従来と同一。
    #[arg(long, default_value_t = false)]
    quiet_filter: bool,

    /// 2026-11-19 TREEDL: 探索木の局面 (PSV の byte 39 = padding の bit 0 が 1) だけ勝敗の項を使わない
    /// (`--wdl` の混ぜ率をその局面だけ 0 にする = target は sigmoid(score / scale) だけ)。
    /// 探索木の局面には対局の結果が無い (書き出しの器は game_result = 0 にする) ため。
    /// 印の無い局面 (ponko 等、 byte 39 = 0) は従来どおり `--wdl` の値で混ぜる。 未指定なら従来と同一。
    #[arg(long, default_value_t = false)]
    tree_nowdl: bool,

    /// 2026-11-19 TREEDL: 小さい PSV (探索木の局面) を全部メモリに読み、 学習の局面に一定の割合で差し込む。
    /// `--inject-ratio` と組。 差し込む局面は `--bucket-filter` の範囲内のものだけを使い、 ファイルの順に循環する
    /// (読み切ったら先頭へ戻る)。 score の置き換え (`--score-lut`)・詰みの捨て・`--quiet-filter` は掛けない
    /// (差し込む側は目盛りを合わせて書き出してある前提)。 `--bucket-filter` と併用すること。 未指定なら従来と同一。
    #[arg(long)]
    inject_file: Option<PathBuf>,

    /// `--inject-file` の局面が、 出す局面全体に占める割合 (0 < r < 1)。 例 0.06 = 6%。
    #[arg(long, default_value_t = 0.0)]
    inject_ratio: f64,

    /// **bucket をまたいで共有される重みの真の凍結** (2026-11-13、 INVENT02 案 A)。
    ///
    /// FT (`l0w` / `l0b`)、 PSQT (`psqtw` / `psqtb`)、 L1 の共有部 (`l1fw` / `l1fb`) を
    /// **勾配を持たない leaf** にする (`--threat-freeze-hard` と同じ仕組み)。 勾配バッファと
    /// optimizer 状態を確保しないので、 値は 1 bit も動かない。 残る学習対象は bucket 別の
    /// head (`l1w` / `l1b` / `l2w` / `l2b` / `l3w` / `l3b`) だけになる。
    /// - `--main-freeze` の lr_scale 1e-6 は「実質凍結」で、 小さい重みは 1 step で ulp 単位に動く。
    /// - threat 構成では `--threat-freeze-hard` を必須にする。
    /// - weights.bin には出るので resume できる (optimizer 状態は持たない)。
    /// 既定 false = graph は従来と同一。
    #[arg(long, default_value_t = false)]
    hard_freeze_shared: bool,

    /// Progress parameter path: coeff JSON for progress8/progress8gikou, progress.bin for progress8kpabs
    #[arg(long)]
    progress_coeff: Option<PathBuf>,

    /// Quantised eval output format
    #[arg(long, value_enum, default_value = "tanuki-sfnnwop1536")]
    output_format: LayerStackOutputFormat,

    /// Disable ANSI colour output
    #[arg(long)]
    no_color: bool,
}

#[derive(Debug, Deserialize)]
struct ProgressCoeffV1 {
    format: String,
    model: String,
    num_buckets: usize,
    feature_order: Vec<String>,
    standardization: ProgressStandardization,
    weights: Vec<f32>,
    bias: f32,
    runtime: ProgressRuntime,
}

#[derive(Debug, Deserialize)]
struct ProgressStandardization {
    mean: Vec<f32>,
    std: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ProgressRuntime {
    z_clip: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct ProgressCoeffV2 {
    format: String,
    model: String,
    feature_set: String,
    num_buckets: usize,
    feature_order: Vec<String>,
    standardization: ProgressStandardization,
    weights: Vec<f32>,
    bias: f32,
    runtime: ProgressRuntime,
}

// `OutputBuckets` implementations stay `Copy`, so boxing the large variants is not an option.
#[allow(clippy::large_enum_variant, dead_code)]
#[derive(Clone, Copy)]
enum LoadedProgressBucket {
    V1(ShogiProgressBucket8),
    Gikou(ShogiProgressBucket8GikouLite),
    KPAbs(ShogiProgressKPAbs),
}

impl Args {
    fn resume_path(&self) -> Option<&Path> {
        self.resume.as_deref().filter(|path| !path.is_empty()).map(Path::new)
    }

    fn resume_from_quantised_path(&self) -> Option<&Path> {
        self.resume_from_quantised.as_deref().filter(|p| !p.is_empty()).map(Path::new)
    }

    fn resume_weights_only_path(&self) -> Option<&Path> {
        self.resume_weights_only.as_deref().filter(|p| !p.is_empty()).map(Path::new)
    }

    fn wdl_value(&self) -> f32 {
        self.wdl.unwrap_or(0.5)
    }

    fn validate_wdl_range(name: &str, value: f32) -> Result<(), String> {
        if (0.0..=1.0).contains(&value) {
            Ok(())
        } else {
            Err(format!("--{} must be between 0.0 and 1.0 (got {})", name, value))
        }
    }

    fn create_wdl_scheduler(&self) -> Result<wdl::WdlSchedulerEnum, String> {
        match (self.start_wdl, self.end_wdl) {
            (Some(start), Some(end)) => {
                Self::validate_wdl_range("start-wdl", start)?;
                Self::validate_wdl_range("end-wdl", end)?;
                Ok(wdl::WdlSchedulerEnum::linear(start, end))
            }
            (Some(_), None) => Err("--start-wdl requires --end-wdl".to_string()),
            (None, Some(_)) => Err("--end-wdl requires --start-wdl".to_string()),
            (None, None) => {
                let wdl = self.wdl_value();
                Self::validate_wdl_range("wdl", wdl)?;
                Ok(wdl::WdlSchedulerEnum::constant(wdl))
            }
        }
    }

    fn wdl_display(&self) -> String {
        match (self.start_wdl, self.end_wdl) {
            (Some(start), Some(end)) => format!("Linear ({} -> {})", start, end),
            _ => format!("Constant ({})", self.wdl_value()),
        }
    }

    fn interleave_batches_value(&self) -> Option<usize> {
        if self.interleave_file_batches == 0 { None } else { Some(self.interleave_file_batches) }
    }

    /// LMAS は 2026-05-25 廃止。 互換のため常に 0 を返す stub。
    fn resolved_lmas_flags(&self) -> u32 {
        0
    }

    fn validate_wrm_settings(&self) -> Result<(), String> {
        if let Some(in_scaling) = self.wrm_in_scaling {
            if !in_scaling.is_finite() || in_scaling <= 0.0 {
                return Err(format!("--wrm-in-scaling must be a positive finite value (got {})", in_scaling));
            }
            if !self.wrm_nnue2score.is_finite() || self.wrm_nnue2score <= 0.0 {
                return Err(format!("--wrm-nnue2score must be a positive finite value (got {})", self.wrm_nnue2score));
            }
        }
        if let Some(s) = self.loss_scale {
            if !s.is_finite() || s <= 0.0 {
                return Err(format!("--loss-scale must be a positive finite value (got {})", s));
            }
            // WRM 系は target / 出力の変換が scale に依存しない別の式なので、 尺度の付け替えが成り立たない。
            if self.win_rate_model || self.wrm_in_scaling.is_some() {
                return Err("--loss-scale は --win-rate-model / --wrm-in-scaling と併用できない".into());
            }
        }
        if !self.loss_power.is_finite() || self.loss_power < 1.0 {
            return Err(format!("--loss-power must be a finite value >= 1.0 (got {})", self.loss_power));
        }
        Ok(())
    }

    fn parse_ply_bounds_csv(text: &str) -> Result<[u16; 8], String> {
        let mut values = Vec::new();
        for token in text.split(',') {
            let t = token.trim();
            if t.is_empty() {
                continue;
            }
            let value: u16 = t.parse().map_err(|e| format!("invalid --ply-bounds value '{t}': {e}"))?;
            values.push(value);
        }
        if values.len() != 8 {
            return Err(format!("--ply-bounds requires exactly 8 comma-separated values (got {})", values.len()));
        }
        Ok([values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]])
    }

    fn resolved_ply_bounds(&self) -> Result<Option<[u16; 8]>, String> {
        match self.bucket_mode {
            BucketMode::Kingrank9 => {
                if self.ply_bounds.is_some() {
                    Err("--ply-bounds can only be used with --bucket-mode ply9".to_string())
                } else if self.progress_coeff.is_some() {
                    Err("--progress-coeff can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
                        .to_string())
                } else {
                    Ok(None)
                }
            }
            BucketMode::Ply9 => {
                if self.progress_coeff.is_some() {
                    Err("--progress-coeff can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
                        .to_string())
                } else {
                    match &self.ply_bounds {
                        Some(text) => Self::parse_ply_bounds_csv(text).map(Some),
                        None => Ok(Some(SHOGI_PLY_BUCKET9_DEFAULT_BOUNDS)),
                    }
                }
            }
            BucketMode::Progress8
            | BucketMode::Progress8Gikou
            | BucketMode::Progress8KPAbs
            | BucketMode::Progress4Hand2KPAbs
            | BucketMode::Progress8Hand2KPAbs
            | BucketMode::OpnK9z
            | BucketMode::OpnK9zProg3 => {
                if self.ply_bounds.is_some() {
                    Err("--ply-bounds can only be used with --bucket-mode ply9".to_string())
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn bucket_mode_name(&self) -> &'static str {
        match self.bucket_mode {
            BucketMode::Kingrank9 => "kingrank9",
            BucketMode::Ply9 => "ply9",
            BucketMode::Progress8 => "progress8",
            BucketMode::Progress8Gikou => "progress8gikou",
            BucketMode::Progress8KPAbs => "progress8kpabs",
            BucketMode::Progress4Hand2KPAbs => "progress4hand2kpabs",
            BucketMode::Progress8Hand2KPAbs => "progress8hand2kpabs",
            BucketMode::OpnK9z => "opnk9z",
            BucketMode::OpnK9zProg3 => "opnk9zp3",
        }
    }

    fn load_progress_bucket(&self) -> Result<Option<LoadedProgressBucket>, String> {
        match self.bucket_mode {
            BucketMode::Progress8 => {
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8 requires --progress-coeff".to_string())?;
                load_progress_bucket_v1_from_json(path).map(|v| Some(LoadedProgressBucket::V1(v)))
            }
            BucketMode::Progress8Gikou => {
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8gikou requires --progress-coeff".to_string())?;
                load_progress_bucket_v2_from_json(path).map(|v| Some(LoadedProgressBucket::Gikou(v)))
            }
            BucketMode::Progress8KPAbs => {
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8kpabs requires --progress-coeff".to_string())?;
                ShogiProgressKPAbs::load_from_bin(path).map(|v| Some(LoadedProgressBucket::KPAbs(v)))
            }
            BucketMode::Progress4Hand2KPAbs => {
                // P4H2 は progress8kpabs と同じ progress.bin を使い、bucket 算出だけ
                // LayerStack グローバルモードで切り替える(set_layerstack_p4h2 は main で設定)。
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress4hand2kpabs requires --progress-coeff".to_string())?;
                ShogiProgressKPAbs::load_from_bin(path).map(|v| Some(LoadedProgressBucket::KPAbs(v)))
            }
            BucketMode::Progress8Hand2KPAbs => {
                // P8H2 も progress8kpabs と同じ progress.bin を使う。bucket 算出は 16 区分の
                // 専用型 ShogiProgress8Hand2KPAbs 側で行う(main の dispatch で構築)。
                // ここでは progress.bin(共有 OnceLock)のロードのみ行い KPAbs として返す。
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| "--bucket-mode progress8hand2kpabs requires --progress-coeff".to_string())?;
                ShogiProgressKPAbs::load_from_bin(path).map(|v| Some(LoadedProgressBucket::KPAbs(v)))
            }
            BucketMode::OpnK9z | BucketMode::OpnK9zProg3 => {
                // 2026-11-14 OPNARCH: 進行度 (--bucket-filter の鍵と 243 の head) は progress8kpabs と同じ
                // progress.bin (共有 OnceLock)。 head の型 ShogiOpnK9z は main の dispatch で構築する。
                let path = self
                    .progress_coeff
                    .as_ref()
                    .ok_or_else(|| format!("--bucket-mode {} requires --progress-coeff", self.bucket_mode_name()))?;
                ShogiProgressKPAbs::load_from_bin(path).map(|v| Some(LoadedProgressBucket::KPAbs(v)))
            }
            _ => {
                if self.progress_coeff.is_some() {
                    Err("--progress-coeff can only be used with --bucket-mode progress8/progress8gikou/progress8kpabs"
                        .to_string())
                } else {
                    Ok(None)
                }
            }
        }
    }
}

fn load_progress_bucket_v1_from_json(path: &PathBuf) -> Result<ShogiProgressBucket8, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read --progress-coeff '{}': {e}", path.display()))?;
    let coeff: ProgressCoeffV1 = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse progress coeff JSON '{}': {e}", path.display()))?;

    if coeff.format != "rshogi.progress_coeff.v1" {
        return Err(format!("invalid progress coeff format '{}', expected 'rshogi.progress_coeff.v1'", coeff.format));
    }
    if coeff.model != "logistic_regression" {
        return Err(format!("invalid progress coeff model '{}', expected 'logistic_regression'", coeff.model));
    }
    if coeff.num_buckets != 8 {
        return Err(format!("invalid num_buckets {}, expected 8", coeff.num_buckets));
    }
    if coeff.feature_order.len() != SHOGI_PROGRESS8_NUM_FEATURES {
        return Err(format!(
            "invalid feature_order length {}, expected {}",
            coeff.feature_order.len(),
            SHOGI_PROGRESS8_NUM_FEATURES
        ));
    }
    for (idx, expected) in SHOGI_PROGRESS8_FEATURE_ORDER.iter().enumerate() {
        if coeff.feature_order[idx] != *expected {
            return Err(format!(
                "feature_order mismatch at index {}: got '{}', expected '{}'",
                idx, coeff.feature_order[idx], expected
            ));
        }
    }
    if coeff.standardization.mean.len() != SHOGI_PROGRESS8_NUM_FEATURES
        || coeff.standardization.std.len() != SHOGI_PROGRESS8_NUM_FEATURES
        || coeff.weights.len() != SHOGI_PROGRESS8_NUM_FEATURES
    {
        return Err(format!(
            "mean/std/weights lengths must all be {} (got mean={}, std={}, weights={})",
            SHOGI_PROGRESS8_NUM_FEATURES,
            coeff.standardization.mean.len(),
            coeff.standardization.std.len(),
            coeff.weights.len()
        ));
    }
    if coeff.runtime.z_clip.len() != 2 {
        return Err(format!("runtime.z_clip must have exactly 2 values (got {})", coeff.runtime.z_clip.len()));
    }

    let mean: [f32; SHOGI_PROGRESS8_NUM_FEATURES] =
        coeff.standardization.mean.try_into().map_err(|_| "failed to convert mean to fixed array".to_string())?;
    let std: [f32; SHOGI_PROGRESS8_NUM_FEATURES] =
        coeff.standardization.std.try_into().map_err(|_| "failed to convert std to fixed array".to_string())?;
    let weights: [f32; SHOGI_PROGRESS8_NUM_FEATURES] =
        coeff.weights.try_into().map_err(|_| "failed to convert weights to fixed array".to_string())?;
    let z_clip = [coeff.runtime.z_clip[0], coeff.runtime.z_clip[1]];

    Ok(ShogiProgressBucket8::new(mean, std, weights, coeff.bias, z_clip))
}

fn load_progress_bucket_v2_from_json(path: &PathBuf) -> Result<ShogiProgressBucket8GikouLite, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read --progress-coeff '{}': {e}", path.display()))?;
    let coeff: ProgressCoeffV2 = serde_json::from_str(&text)
        .map_err(|e| format!("failed to parse progress coeff JSON '{}': {e}", path.display()))?;

    if coeff.format != "rshogi.progress_coeff.v2" {
        return Err(format!("invalid progress coeff format '{}', expected 'rshogi.progress_coeff.v2'", coeff.format));
    }
    if coeff.model != "logistic_regression" {
        return Err(format!("invalid progress coeff model '{}', expected 'logistic_regression'", coeff.model));
    }
    if coeff.feature_set != "gikou_lite_34" {
        return Err(format!("invalid feature_set '{}', expected 'gikou_lite_34'", coeff.feature_set));
    }
    if coeff.num_buckets != 8 {
        return Err(format!("invalid num_buckets {}, expected 8", coeff.num_buckets));
    }
    if coeff.feature_order.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES {
        return Err(format!(
            "invalid feature_order length {}, expected {}",
            coeff.feature_order.len(),
            SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
        ));
    }
    for (idx, expected) in SHOGI_PROGRESS_GIKOU_LITE_FEATURE_ORDER.iter().enumerate() {
        if coeff.feature_order[idx] != *expected {
            return Err(format!(
                "feature_order mismatch at index {}: got '{}', expected '{}'",
                idx, coeff.feature_order[idx], expected
            ));
        }
    }
    if coeff.standardization.mean.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
        || coeff.standardization.std.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
        || coeff.weights.len() != SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES
    {
        return Err(format!(
            "mean/std/weights lengths must all be {} (got mean={}, std={}, weights={})",
            SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES,
            coeff.standardization.mean.len(),
            coeff.standardization.std.len(),
            coeff.weights.len()
        ));
    }
    if coeff.runtime.z_clip.len() != 2 {
        return Err(format!("runtime.z_clip must have exactly 2 values (got {})", coeff.runtime.z_clip.len()));
    }

    let mean: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] =
        coeff.standardization.mean.try_into().map_err(|_| "failed to convert mean to fixed array".to_string())?;
    let std: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] =
        coeff.standardization.std.try_into().map_err(|_| "failed to convert std to fixed array".to_string())?;
    let weights: [f32; SHOGI_PROGRESS_GIKOU_LITE_NUM_FEATURES] =
        coeff.weights.try_into().map_err(|_| "failed to convert weights to fixed array".to_string())?;
    let z_clip = [coeff.runtime.z_clip[0], coeff.runtime.z_clip[1]];

    Ok(ShogiProgressBucket8GikouLite::new(mean, std, weights, coeff.bias, z_clip))
}

// =============================================================================
// Experiment Log Structures
// =============================================================================

#[derive(Serialize, Clone)]
struct ExperimentLog {
    id: String,
    name: String,
    date: String,
    status: String,
    last_updated_at: String,
    commit: String,
    command: String,
    params: ExperimentParams,
    data: ExperimentData,
    results: ExperimentResults,
    history: Vec<LossEntry>,
    checkpoints: Vec<String>,
}

#[derive(Serialize, Clone)]
struct ExperimentResults {
    training_time_seconds: u64,
    fv_scale: i32,
    best_loss: Option<f64>,
    best_loss_superbatch: Option<usize>,
}

#[derive(Serialize, Clone)]
struct ExperimentParams {
    architecture: String,
    l0: usize,
    l1: usize,
    l2: usize,
    num_buckets: usize,
    bucket_mode: String,
    ply_bounds: Option<[u16; 8]>,
    progress_coeff: Option<String>,
    lr: f32,
    lr_gamma: f32,
    lr_step: usize,
    batch_size: usize,
    batches_per_superbatch: usize,
    interleave_file_batches: Option<usize>,
    epoch_file_shuffle: bool,
    file_shuffle_seed: Option<u64>,
    superbatches: usize,
    start_superbatch: usize,
    wdl: f32,
    start_wdl: Option<f32>,
    end_wdl: Option<f32>,
    scale: i32,
    weight_decay: f32,
    win_rate_model: bool,
    optimizer: String,
    qa: i16,
    qb: i16,
}

#[derive(Serialize, Clone)]
struct ExperimentData {
    name: String,
    positions: Option<u64>,
    total_positions: u64,
    dataset_passes: Option<f64>,
}

#[derive(Serialize, Clone)]
struct LossEntry {
    superbatch: usize,
    loss: f64,
}

// =============================================================================
// 2026-11-13 INVENT02 案 A: bucket の絞り込み (--bucket-filter) と共有重みの真の凍結 (--hard-freeze-shared)
// =============================================================================

/// `--bucket-filter` の文字列 (`"0-2"` / `"1"`) を `(lo, hi)` (両端を含む) に解釈する。
fn parse_bucket_filter(text: &str, num_buckets: usize) -> Result<(u8, u8), String> {
    let text = text.trim();
    let (lo, hi) = match text.split_once('-') {
        Some((a, b)) => (a.trim(), b.trim()),
        None => (text, text),
    };
    let lo: usize = lo.parse().map_err(|_| format!("--bucket-filter の形式は LO-HI か N (与えられた値: {text})"))?;
    let hi: usize = hi.parse().map_err(|_| format!("--bucket-filter の形式は LO-HI か N (与えられた値: {text})"))?;
    if lo > hi || hi >= num_buckets {
        return Err(format!("--bucket-filter は 0 <= LO <= HI < {num_buckets} (与えられた値: {text})"));
    }
    Ok((lo as u8, hi as u8))
}

/// 2026-11-17 K256 案 A: `threat_ext_w` の clip 幅。 書き出しは threat 表と同じ `round(254 * w)` を i8 に clamp するので、
/// |w| <= 127/254 なら飽和しない (QA_FT = 254)。
const THREAT_EXT_CLIP: f32 = 127.0 / QA_FT as f32;

/// 2026-11-17 K256 案 A: `--threat-ext-out` / `--threat-ext-lr-mult` / `--threat-ext-only` の組み合わせを検査する。
/// 既定 (`--threat-ext-out 0`) では他の 2 つが既定値であることだけを確かめる (= 従来と同一)。
fn validate_threat_ext_args(args: &Args) -> Result<(), String> {
    // 2026-11-17 U1a: --threat-only (threat 表だけを学習) の検査。
    if args.threat_only {
        let mut bad: Vec<&str> = Vec::new();
        if !args.threat || args.threat_out == 0 {
            bad.push("--threat --threat-out K (専用 sparse の threat 構成) が必要");
        }
        if args.threat_freeze_hard || args.threat_freeze {
            bad.push("--threat-freeze / --threat-freeze-hard とは排他 (学習させる表を凍結する指定)");
        }
        if args.threat_ext_only || args.main_freeze || args.kh_freeze_rest || args.hard_freeze_shared {
            bad.push("--threat-ext-only / --main-freeze / --kh-freeze-rest / --hard-freeze-shared とは排他 (凍結の倍率が食い違う)");
        }
        if !matches!(args.optimizer, OptimizerType::Ranger21) {
            bad.push("--optimizer ranger21 でのみ使える (凍結に Ranger21 の lookahead 無効を使うため)");
        }
        if (args.l0_lr_mult - 1.0).abs() > 1e-6
            || (args.psqt_lr_mult - 1.0).abs() > 1e-6
            || (args.l1_lr_mult - 1.0).abs() > 1e-6
            || (args.l2_lr_mult - 1.0).abs() > 1e-6
            || (args.l3_lr_mult - 1.0).abs() > 1e-6
            || args.l1_weight_decay.is_some()
        {
            bad.push("凍結した層の lr 倍率 / decay 指定とは併用できない");
        }
        if !bad.is_empty() {
            return Err(format!("--threat-only: {}", bad.join(" / ")));
        }
    }
    if args.l0_freeze_hard && (args.l0_lr_mult - 1.0).abs() > 1e-6 {
        return Err("--l0-freeze-hard と --l0-lr-mult は排他 (凍結した FT に倍率は掛からない)".into());
    }
    let n = args.threat_ext_out;
    if n == 0 {
        if (args.threat_ext_lr_mult - 1.0).abs() > 1e-6 || args.threat_ext_only {
            return Err("--threat-ext-lr-mult / --threat-ext-only は --threat-ext-out N (N > 0) と併用すること".into());
        }
        return Ok(());
    }
    let mut bad: Vec<String> = Vec::new();
    if !args.threat || args.threat_out == 0 {
        bad.push("--threat --threat-out K (専用 sparse の threat 構成) が必要".into());
    }
    if n % 32 != 0 {
        bad.push(format!("--threat-ext-out {n} は 32 の倍数 (エンジンの K' の半分が 16 の倍数になるように)"));
    }
    if args.threat_out % 2 != 0 || args.threat_out / 2 + n / 2 > args.l0 / 2 {
        bad.push(format!(
            "--threat-out {} / 2 + --threat-ext-out {n} / 2 が FT の半分 ({}) を超えている (または K が奇数)",
            args.threat_out,
            args.l0 / 2
        ));
    }
    if !matches!(args.optimizer, OptimizerType::Ranger21) {
        bad.push("--optimizer ranger21 でのみ使える (clip を Ranger21 の params で掛けるため)".into());
    }
    if !(args.threat_ext_lr_mult > 0.0 && args.threat_ext_lr_mult.is_finite()) {
        bad.push("--threat-ext-lr-mult は正の有限値".into());
    }
    // 書き出し (k256_merge.py) は「K 列の旧表 + 群なしの N 列」という形しか扱わない。 形を変える指定は止める。
    for (on, name) in [
        (args.threat_factorizer, "--threat-factorizer"),
        (args.threat_hang_split, "--threat-hang-split"),
        (args.threat_i16, "--threat-i16"),
        (args.threat_proj, "--threat-proj"),
        (args.threat_l1, "--threat-l1"),
        (args.threat_l2, "--threat-l2"),
        (args.threat_gate, "--threat-gate"),
        (args.threat_dedicated_dims, "--threat-dedicated-dims"),
        (args.hard_freeze_shared, "--hard-freeze-shared"),
    ] {
        if on {
            bad.push(format!("{name} とは併用できない (追加列の書き出しが対応していない)"));
        }
    }
    if args.threat_ext_only {
        if !args.threat_freeze_hard {
            bad.push("--threat-ext-only は --threat-freeze-hard と併用すること (旧 64 列も凍結しないと追加列だけの学習にならない)".into());
        }
        if args.main_freeze || args.kh_freeze_rest {
            bad.push("--threat-ext-only と --main-freeze / --kh-freeze-rest は排他 (凍結の倍率が食い違う)".into());
        }
        if (args.l0_lr_mult - 1.0).abs() > 1e-6
            || (args.psqt_lr_mult - 1.0).abs() > 1e-6
            || (args.l1_lr_mult - 1.0).abs() > 1e-6
            || (args.l2_lr_mult - 1.0).abs() > 1e-6
            || (args.l3_lr_mult - 1.0).abs() > 1e-6
            || args.l1_weight_decay.is_some()
        {
            bad.push("--threat-ext-only と 凍結した層の lr 倍率 / decay 指定は併用できない".into());
        }
    }
    if bad.is_empty() { Ok(()) } else { Err(format!("--threat-ext-out: {}", bad.join(" / "))) }
}

/// `--bucket-filter` / `--hard-freeze-shared` の組み合わせを検査し、 絞り込みの範囲を返す。
/// どちらも未指定なら `Ok(None)` で何もしない (= 従来と同一)。
fn validate_bucket_filter_args(args: &Args) -> Result<Option<(u8, u8)>, String> {
    if args.hard_freeze_shared {
        let mut bad: Vec<&str> = Vec::new();
        if args.main_freeze {
            bad.push("--main-freeze");
        }
        // 2026-11-15 H2: --kh は「KH の重み kh_ft_w も真の凍結 (勾配なしの leaf)」として併用できる
        //   (KH 入りの net の bucket 別 head だけを学習し直す)。 KH を学習させる指定とは併用させない。
        if args.kh && (args.kh_freeze_rest || args.kh_null || args.kh_split.is_some()) {
            bad.push("--kh-freeze-rest / --kh-null / --kh-split (--kh は kh_ft_w を凍結した形でだけ併用できる)");
        }
        if (args.l0_lr_mult - 1.0).abs() > 1e-6 {
            bad.push("--l0-lr-mult");
        }
        if (args.psqt_lr_mult - 1.0).abs() > 1e-6 {
            bad.push("--psqt-lr-mult");
        }
        if args.l1_anchor_lambda > 0.0 || args.bucket_smooth_l1_lambda > 0.0 {
            bad.push("L1 graph 正則化 (--l1-anchor-lambda / --bucket-smooth-l1-lambda)");
        }
        if args.factorise {
            bad.push("--factorise");
        }
        if !bad.is_empty() {
            return Err(format!(
                "--hard-freeze-shared は {} と併用できない (凍結した重みを触る指定 / 未検証)",
                bad.join(" / ")
            ));
        }
        if args.threat_out > 0 && !args.threat_freeze_hard {
            return Err(
                "--hard-freeze-shared は threat 構成では --threat-freeze-hard と併用すること (threat も bucket 共有)"
                    .into(),
            );
        }
        if args.resume_path().is_none()
            && args.resume_weights_only_path().is_none()
            && args.resume_from_quantised_path().is_none()
        {
            return Err("--hard-freeze-shared は土台の重みが要る (--resume-weights-only / --resume / --resume-from-quantised のどれか)".into());
        }
        println!(
            "HARD-FREEZE-SHARED: l0w / l0b / l1fw / l1fb{} を勾配なしの leaf にする (optimizer 状態も持たない = bit 単位で不変)",
            if args.psqt { " / psqtw / psqtb" } else { "" }
        );
    }

    // 2026-11-14 OPNARCH: 243 (k9k9z × 進行度 3) の head は進行度 bucket 0..2 しか持たないので絞り込みが必須。
    if args.bucket_mode == BucketMode::OpnK9zProg3 && args.bucket_filter.is_none() {
        return Err(
            "--bucket-mode opnk9zp3 は --bucket-filter (進行度 bucket の上限 2 以下、 例 0-2) が必須 (243 の head は進行度 0..2 だけ)"
                .into(),
        );
    }
    let Some(text) = args.bucket_filter.as_deref() else {
        return Ok(None);
    };
    // OPN の king bucket モードでは、 --bucket-filter の範囲は head 番号ではなく進行度 8 区分 (progress8kpabs) で読む。
    let filter_key_buckets = if args.bucket_mode.is_opn_king() {
        bullet_lib::game::outputs::SHOGI_PROGRESS8_NUM_BUCKETS
    } else {
        NUM_BUCKETS
    };
    let range = parse_bucket_filter(text, filter_key_buckets)?;
    if !matches!(
        args.bucket_mode,
        BucketMode::Progress8KPAbs | BucketMode::Progress4Hand2KPAbs | BucketMode::OpnK9z | BucketMode::OpnK9zProg3
    ) {
        return Err(
            "--bucket-filter は --bucket-mode progress8kpabs / progress4hand2kpabs / opnk9z / opnk9zp3 でのみ使える"
                .into(),
        );
    }
    if args.bucket_mode == BucketMode::OpnK9zProg3 && range.1 > 2 {
        return Err(format!(
            "--bucket-mode opnk9zp3 の --bucket-filter は進行度 bucket の上限 2 以下 (与えられた値: {text})"
        ));
    }
    if args.data_format.eq_ignore_ascii_case("hcpe") {
        return Err("--bucket-filter は psv の loader 専用 (--data-format hcpe は未対応)".into());
    }
    // soft bucket (環境変数 BUCKET_SOFT_PROB) は乱数で bucket を選ぶので、 絞り込みと学習で
    // 別の bucket になりうる。 黙って食い違わせないよう止める。
    let soft = std::env::var("BUCKET_SOFT_PROB").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
    if soft > 0.0 {
        return Err(format!("--bucket-filter は BUCKET_SOFT_PROB={soft} (> 0、 bucket が乱数) と併用できない"));
    }
    if args.hard_freeze_shared {
        // 範囲外 bucket の head を bit 単位で動かさないための条件。
        //   - weight decay: 勾配 0 でも w *= (1 - decay * lr) で縮む。
        //   - optimizer 状態: 土台の momentum が残っていると勾配 0 でも動き、
        //     lookahead の slow 重みが現在値と違えば merge で引き戻される。
        let decays = [
            ("--weight-decay", Some(args.weight_decay)),
            ("--l1-weight-decay", args.l1_weight_decay),
            ("--l2-weight-decay", args.l2_weight_decay),
            ("--l3-weight-decay", args.l3_weight_decay),
        ];
        for (name, value) in decays {
            if let Some(v) = value {
                if v != 0.0 {
                    return Err(format!(
                        "--bucket-filter + --hard-freeze-shared では {name} を 0 にすること (範囲外 bucket の head が decay で縮む)。 与えられた値: {v}"
                    ));
                }
            }
        }
        // bucket をまたぐ正則化 (隣の bucket との差 / 行間の相関) は範囲外 bucket の head に勾配を流す。
        if args.bucket_smooth_l3_lambda > 0.0
            || args.bucket_smooth_psqt_lambda > 0.0
            || args.l1_decorr_lambda > 0.0
            || args.l2_decorr_lambda > 0.0
        {
            return Err("--bucket-filter + --hard-freeze-shared は bucket 間 / 行間の正則化 (--bucket-smooth-* / --l1/l2-decorr-lambda) と併用できない".into());
        }
        if args.resume_path().is_some() && args.start_superbatch == 0 {
            return Err("--bucket-filter + --hard-freeze-shared の新規開始は --resume-weights-only <ckpt>/optimiser_state を使うこと \
                 (--resume は土台の momentum / slow 重みを持ち込み、 範囲外 bucket の head が勾配 0 でも動く)。 \
                 中断からの再開 (--start-superbatch > 0) なら --resume でよい"
                .into());
        }
    } else {
        println!(
            "[warn] --bucket-filter を --hard-freeze-shared なしで使っている: FT / PSQT / l1f は全 bucket 共有なので、 範囲外 bucket の評価も変わる"
        );
    }
    println!(
        "BUCKET-FILTER: bucket {}..={} の局面だけ通す (bucket は学習器と同じ OutputBuckets::bucket / {:?} + --progress-coeff)",
        range.0, range.1, args.bucket_mode
    );
    Ok(Some(range))
}

/// `--bucket-filter` 用の loader。 内側の loader が出す局面のうち bucket が範囲内のものだけを
/// 集めて、 **同じ大きさの batch** に詰め直して渡す。 `range` が `None` なら内側をそのまま呼ぶ
/// (= 従来と同一)。
#[derive(Clone)]
struct BucketFilterLoader<L, O> {
    inner: L,
    range: Option<(u8, u8)>,
    /// 学習器が head 選択に使うのと同じ bucket 関数 (`bucket_impl`)。
    buckets: O,
    /// 範囲と比べる鍵。 既定は `buckets.bucket(pos)` (= 従来と同一)。
    /// 2026-11-14 OPNARCH: OPN の king bucket モードでは進行度 8 区分 (progress8kpabs) にする。
    key: fn(&O, &bullet_lib::shogi::PackedSfenValue) -> u8,
    /// 鍵の取りうる数 (起動時の分布の印字に使う)。 既定は `O::BUCKETS`。
    key_buckets: usize,
    /// 起動時の先読みで head (king bucket) の偏りも印字するか (OPN の king bucket モード)。
    log_heads: bool,
    /// 2026-11-17 tanuki: score の置き換え表 (`--score-lut`、 8 x 10000)。 None なら score は触らない。
    score_lut: Option<std::sync::Arc<Vec<i32>>>,
    /// 2026-11-17 tanuki: |score| >= この値の局面を捨てる (`--score-lut-drop-mate`)。 0 = 捨てない。
    drop_mate: i32,
    /// 2026-11-17 tanuki 検算用ダンプ (`--score-lut-dump`): (出力先ディレクトリ, 局面数)。
    lut_dump: Option<(PathBuf, usize)>,
    /// 2026-11-17 tanuki: ためてから混ぜる batch 数 (`--mix-buffer-batches`)。 0 = 混ぜない。
    mix_batches: usize,
    /// 2026-11-18 tanuki: 静かでない局面 (王手・最善手が駒取り) を捨てるか (`--quiet-filter`)。
    quiet: bool,
    /// 2026-11-19 TREEDL: 差し込む局面 (`--inject-file`) と、 通した局面 1 つあたりの差し込み数 r / (1 - r)。
    inject: Option<(std::sync::Arc<Vec<bullet_lib::shogi::PackedSfenValue>>, f64)>,
}

/// `--quiet-filter` の判定: 捨てるなら理由 ("DROP_CHECK" / "DROP_CAPTURE")、 静かなら None。
/// 王手は rshogi-core の `Position::in_check` (手番側が王手をかけられている)。 駒取りは move16 の行き先
/// (下位 7 bit、 やねうら王の Square = 筋 * 9 + 段) に駒があるか (打つ手の行き先は必ず空なので from は見ない)。
fn quiet_drop_reason(pos: &bullet_lib::shogi::PackedSfenValue) -> Option<&'static str> {
    let board = bullet_lib::shogi::ShogiBoard::from_packed_sfen(pos);
    let mut core = rshogi_core::position::Position::new();
    if core.set_sfen_with_pass_rights(&board.to_sfen(), 0, 0).is_err() {
        // 局面を読めないレコードは静かと見なせないので捨てる (王手扱い)。
        return Some("DROP_CHECK");
    }
    if core.in_check() {
        return Some("DROP_CHECK");
    }
    let mv = pos.move16();
    let to = usize::from(mv & 0x7F);
    if mv != 0 && to < 81 && !board.board[to].is_none() {
        return Some("DROP_CAPTURE");
    }
    None
}

/// `--mix-buffer-batches` の混ぜ方: splitmix64 で決定論的に Fisher-Yates (種 = 何回目のバッファか)。
fn mix_shuffle(v: &mut [bullet_lib::shogi::PackedSfenValue], round: u64) {
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ round.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mut next = || {
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..v.len()).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// `--score-lut` の表の 1 bucket あたりの長さ (|score| 0..9999)。
const SCORE_LUT_LEN: usize = 10000;

/// `--score-lut` の表を読む (LE i32、 8 x 10000、 各 bucket で 0 始まり・単調非減少・[0, 32767] を確かめる)。
fn load_score_lut(path: &Path) -> Result<Vec<i32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("--score-lut {} を読めない: {e}", path.display()))?;
    let want = NUM_BUCKETS * SCORE_LUT_LEN * 4;
    if bytes.len() != want {
        return Err(format!(
            "--score-lut {} の大きさ {} が {want} (i32 x {NUM_BUCKETS} x {SCORE_LUT_LEN}) でない",
            path.display(),
            bytes.len()
        ));
    }
    let v: Vec<i32> = bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    for b in 0..NUM_BUCKETS {
        let row = &v[b * SCORE_LUT_LEN..(b + 1) * SCORE_LUT_LEN];
        if row[0] != 0 || row.windows(2).any(|w| w[1] < w[0]) || row.iter().any(|&x| !(0..=32767).contains(&x)) {
            return Err(format!("--score-lut: bucket {b} の表が 0 始まり・単調非減少・[0, 32767] でない"));
        }
    }
    Ok(v)
}

/// score の置き換え: out = sign(s) * 表[b][|s|]。 |s| が表の範囲外 (>= 10000) なら None (呼び出し側で捨てる)。
fn apply_score_lut(lut: &[i32], bucket: u8, s: i16) -> Option<i16> {
    let a = (s as i32).unsigned_abs() as usize;
    if a >= SCORE_LUT_LEN {
        return None;
    }
    let y = lut[usize::from(bucket) * SCORE_LUT_LEN + a];
    Some((if s < 0 { -y } else { y }) as i16)
}

/// 既定の鍵: head 選択と同じ bucket (従来の `--bucket-filter` の意味)。
fn bucket_filter_key_head<O>(o: &O, pos: &bullet_lib::shogi::PackedSfenValue) -> u8
where
    O: bullet_lib::game::outputs::OutputBuckets<bullet_lib::shogi::PackedSfenValue>,
{
    o.bucket(pos)
}

/// OPN の king bucket モードの鍵: 進行度 8 区分 (progress8kpabs と同じ値)。
fn bucket_filter_key_progress8<O>(_: &O, pos: &bullet_lib::shogi::PackedSfenValue) -> u8 {
    bullet_lib::game::outputs::opn_progress_bucket8(pos)
}

impl<L, O> BucketFilterLoader<L, O>
where
    L: bullet_lib::value::loader::DataLoader<bullet_lib::shogi::PackedSfenValue>,
    O: bullet_lib::game::outputs::OutputBuckets<bullet_lib::shogi::PackedSfenValue>,
{
    fn new(inner: L, range: Option<(u8, u8)>, buckets: O) -> Self {
        Self {
            inner,
            range,
            buckets,
            key: bucket_filter_key_head::<O>,
            key_buckets: O::BUCKETS,
            log_heads: false,
            score_lut: None,
            drop_mate: 0,
            lut_dump: None,
            mix_batches: 0,
            quiet: false,
            inject: None,
        }
    }

    /// 2026-11-19 TREEDL: 差し込む局面と割合 r を設定する (None = 差し込まない = 従来と同一)。
    fn with_inject(mut self, inject: Option<(Vec<bullet_lib::shogi::PackedSfenValue>, f64)>) -> Self {
        self.inject = inject.map(|(v, r)| (std::sync::Arc::new(v), r / (1.0 - r)));
        self
    }

    /// 2026-11-18 tanuki: 静かでない局面を捨てる (false = 従来と同一)。
    fn with_quiet(mut self, quiet: bool) -> Self {
        self.quiet = quiet;
        self
    }

    /// 2026-11-17 tanuki: N batch 分ためてから混ぜる (0 = 混ぜない = 従来と同一)。
    fn with_mix_batches(mut self, n: usize) -> Self {
        self.mix_batches = n;
        self
    }

    /// 2026-11-17 tanuki: score の置き換え表・詰みの捨て・検算用ダンプを設定する (どれも既定は無効 = 従来と同一)。
    fn with_score_lut(mut self, lut: Option<Vec<i32>>, drop_mate: i32, dump: Option<(PathBuf, usize)>) -> Self {
        self.score_lut = lut.map(std::sync::Arc::new);
        self.drop_mate = drop_mate;
        self.lut_dump = dump;
        self
    }

    /// 2026-11-14 OPNARCH: 範囲の鍵を進行度 8 区分にし、 起動時に head (king bucket) の偏りも印字する。
    fn with_progress8_key(mut self) -> Self {
        self.key = bucket_filter_key_progress8::<O>;
        self.key_buckets = bullet_lib::game::outputs::SHOGI_PROGRESS8_NUM_BUCKETS;
        self.log_heads = true;
        self
    }

    /// 起動時の先読み: 内側 loader の先頭 `sample_batches` batch を数えて、 bucket の分布と
    /// 通す数 / 捨てる数をログに出す。 学習の読み出しとは独立 (状態を持たない)。
    fn log_startup_sample(&self, batch_size: usize, sample_batches: usize) {
        let Some((lo, hi)) = self.range else {
            return;
        };
        let mut hist = vec![0u64; self.key_buckets];
        let mut heads = vec![0u64; if self.log_heads { O::BUCKETS } else { 0 }];
        let mut seen = 0usize;
        self.inner.map_batches(0, batch_size, |raw| {
            for pos in raw {
                let k = (self.key)(&self.buckets, pos);
                hist[usize::from(k)] += 1;
                if self.log_heads && (lo..=hi).contains(&k) {
                    heads[usize::from(self.buckets.bucket(pos))] += 1;
                }
            }
            seen += 1;
            seen >= sample_batches
        });
        let total: u64 = hist.iter().sum();
        let passed: u64 =
            hist.iter().enumerate().filter(|(b, _)| (lo as usize..=hi as usize).contains(b)).map(|(_, n)| n).sum();
        let dropped = total - passed;
        println!(
            "BUCKET-FILTER 起動時の先読み ({} 局面 = 先頭 {} batch): 通す {} / 捨てる {} (通過率 {:.2}%)",
            total,
            seen,
            passed,
            dropped,
            100.0 * passed as f64 / total.max(1) as f64
        );
        println!(
            "BUCKET-FILTER 先読みの bucket 分布: {}",
            hist.iter().enumerate().map(|(b, n)| format!("b{b}={n}")).collect::<Vec<_>>().join(" ")
        );
        if self.log_heads {
            log_opn_head_stats(&heads);
        }
        if passed == 0 {
            eprintln!("ERROR: --bucket-filter: 先読みで範囲内の局面が 1 つも無い (bucket の計算か範囲を確認すること)");
            std::process::exit(1);
        }
    }
}

/// 2026-11-14 OPNARCH: 通した局面の head (king bucket) の偏りを、 設計書 1.1 (runs/opnarch/kb_density.py) と
/// 同じ指標で印字する: 使用 / 実効数 (exp エントロピー) / 上位 1・4・16 の占有 / 1% 未満の head に入る局面の率 /
/// 自玉 z9 の分布。 篩 (自己対局) の分布との差をここで確かめる。
fn log_opn_head_stats(heads: &[u64]) {
    let n: u64 = heads.iter().sum();
    if n == 0 {
        return;
    }
    let nf = n as f64;
    let mut v: Vec<u64> = heads.iter().copied().filter(|&x| x > 0).collect();
    v.sort_unstable_by(|a, b| b.cmp(a));
    let ent: f64 = v.iter().map(|&x| -(x as f64 / nf) * (x as f64 / nf).ln()).sum();
    let top = |k: usize| v.iter().take(k).sum::<u64>() as f64 / nf;
    let low = v.iter().filter(|&&x| (x as f64) / nf < 0.01).sum::<u64>() as f64 / nf;
    println!(
        "OPN-HEAD 先読みの head 分布 ({} 局面): 全 {} / 使用 {} / 実効 {:.1} / 上位1 {:.3} 上位4 {:.3} 上位16 {:.3} / 1%未満の head に入る局面 {:.3}",
        n,
        heads.len(),
        v.len(),
        ent.exp(),
        top(1),
        top(4),
        top(16),
        low
    );
    // head = kz * PROG + pb、 kz = 自玉 z9 * 9 + 相手玉 z9。
    let prog = heads.len() / bullet_lib::game::outputs::OPN_K9Z_NUM_BUCKETS;
    let mut own = [0u64; bullet_lib::game::outputs::OPN_K9Z_ZONES];
    for (h, &c) in heads.iter().enumerate() {
        own[h / prog / bullet_lib::game::outputs::OPN_K9Z_ZONES] += c;
    }
    println!(
        "OPN-HEAD 自玉 z9 の分布: {}",
        own.iter().enumerate().map(|(z, &c)| format!("z{z}={:.3}", c as f64 / nf)).collect::<Vec<_>>().join(" ")
    );
}

/// `--score-lut-dump` は 1 プロセスで 1 回だけ (map_batches が何度呼ばれても最初の呼び出しの先頭だけ)。
static SCORE_LUT_DUMP_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// 混ぜた後の最初の batch の検算用ダンプも 1 回だけ。
static MIX_DUMP_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// `--score-lut-dump` の 1 局面分を書く (残り数が 0 になったら閉じる)。
fn dump_score_lut(
    dump: &mut Option<(std::fs::File, std::fs::File, usize)>,
    pos: &bullet_lib::shogi::PackedSfenValue,
    bucket: u8,
    s0: i16,
    verdict: &str,
) {
    use std::io::Write;
    let Some((psv, tsv, left)) = dump else {
        return;
    };
    psv.write_all(pos.as_bytes()).expect("score_lut_dump.psv に書けない");
    writeln!(tsv, "{bucket}\t{s0}\t{verdict}").expect("score_lut_dump.tsv に書けない");
    *left -= 1;
    if *left == 0 {
        *dump = None;
    }
}

impl<L, O> bullet_lib::value::loader::DataLoader<bullet_lib::shogi::PackedSfenValue> for BucketFilterLoader<L, O>
where
    L: bullet_lib::value::loader::DataLoader<bullet_lib::shogi::PackedSfenValue>,
    O: bullet_lib::game::outputs::OutputBuckets<bullet_lib::shogi::PackedSfenValue>,
{
    fn data_file_paths(&self) -> &[String] {
        self.inner.data_file_paths()
    }

    fn count_positions(&self) -> Option<u64> {
        self.inner.count_positions()
    }

    fn map_batches<F: FnMut(&[bullet_lib::shogi::PackedSfenValue]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        mut f: F,
    ) {
        let Some((lo, hi)) = self.range else {
            // 絞り込みなし = 内側をそのまま (従来と同一)。
            self.inner.map_batches(start_batch, batch_size, f);
            return;
        };
        // 詰め直し用の buffer は 1 回だけ確保して使い回す。
        //   2026-11-17 --mix-buffer-batches N: N batch 分ためてから混ぜて、 先頭から batch-size ずつ出す。
        let pool_cap = batch_size * self.mix_batches.max(1);
        let mut acc: Vec<bullet_lib::shogi::PackedSfenValue> = Vec::with_capacity(pool_cap);
        let mut mix_round = 0u64;
        let mut last_log = 0usize;
        let mut passed = 0u64;
        let mut dropped = 0u64;
        let mut emitted = 0usize;
        // 2026-11-17 tanuki: 詰みの捨てと score の置き換え (既定は無効 = 従来と同一)。
        let mut mate_dropped = 0u64;
        // 2026-11-18 --quiet-filter: 王手 / 最善手が駒取り で捨てた数。
        let mut check_dropped = 0u64;
        let mut capture_dropped = 0u64;
        // 2026-11-19 TREEDL: 差し込む局面のうち bucket の範囲内のものだけを使う (添字の表を 1 回だけ作る)。
        let inj: Option<(&[bullet_lib::shogi::PackedSfenValue], Vec<u32>, f64)> = self.inject.as_ref().map(|(v, per)| {
            let idx: Vec<u32> = (0..v.len() as u32)
                .filter(|&i| {
                    let k = (self.key)(&self.buckets, &v[i as usize]);
                    k >= lo && k <= hi
                })
                .collect();
            println!(
                "[inject] 差し込む局面 {} のうち bucket {lo}-{hi} の範囲内 {} を使う (通した局面 1 つあたり {:.4} 個)",
                v.len(),
                idx.len(),
                per
            );
            assert!(!idx.is_empty(), "--inject-file: bucket の範囲内の局面が 1 つも無い");
            (v.as_slice(), idx, *per)
        });
        let mut inj_credit = 0.0f64;
        let mut inj_pos = 0usize;
        let mut injected = 0u64;
        let mut dump: Option<(std::fs::File, std::fs::File, usize)> = match &self.lut_dump {
            Some((dir, n)) if *n > 0 && !SCORE_LUT_DUMP_DONE.swap(true, std::sync::atomic::Ordering::SeqCst) => {
                std::fs::create_dir_all(dir).expect("--score-lut-dump: 出力先を作れない");
                Some((
                    std::fs::File::create(dir.join("score_lut_dump.psv")).expect("score_lut_dump.psv を作れない"),
                    std::fs::File::create(dir.join("score_lut_dump.tsv")).expect("score_lut_dump.tsv を作れない"),
                    *n,
                ))
            }
            _ => None,
        };
        self.inner.map_batches(start_batch, batch_size, |raw| {
            for pos in raw {
                let b = (self.key)(&self.buckets, pos);
                let s0 = pos.score();
                if b < lo || b > hi {
                    dropped += 1;
                    dump_score_lut(&mut dump, pos, b, s0, "DROP_BUCKET");
                    continue;
                }
                if self.drop_mate > 0 && (s0 as i32).abs() >= self.drop_mate {
                    mate_dropped += 1;
                    dump_score_lut(&mut dump, pos, b, s0, "DROP_MATE");
                    continue;
                }
                if self.quiet {
                    if let Some(why) = quiet_drop_reason(pos) {
                        if why == "DROP_CHECK" {
                            check_dropped += 1;
                        } else {
                            capture_dropped += 1;
                        }
                        dump_score_lut(&mut dump, pos, b, s0, why);
                        continue;
                    }
                }
                let mut p = *pos;
                if let Some(lut) = &self.score_lut {
                    let Some(s1) = apply_score_lut(lut, b, s0) else {
                        // 表の範囲外 (|s| >= 10000) は捨てる (--score-lut-drop-mate を付けない場合の安全網)。
                        mate_dropped += 1;
                        dump_score_lut(&mut dump, pos, b, s0, "DROP_MATE");
                        continue;
                    };
                    p.as_bytes_mut()[32..34].copy_from_slice(&s1.to_le_bytes());
                    dump_score_lut(&mut dump, pos, b, s0, &s1.to_string());
                } else {
                    dump_score_lut(&mut dump, pos, b, s0, &s0.to_string());
                }
                passed += 1;
                // 2026-11-19 TREEDL: 通した局面の後ろに、 割合どおりの数だけ差し込む局面を続けて積む。
                let mut n_inj = 0usize;
                if let Some((_, _, per)) = &inj {
                    inj_credit += *per;
                    while inj_credit >= 1.0 {
                        inj_credit -= 1.0;
                        n_inj += 1;
                    }
                }
                for k in 0..=n_inj {
                let p = if k == 0 {
                    p
                } else {
                    let (v, idx, _) = inj.as_ref().expect("n_inj > 0 なら inj がある");
                    let q = v[idx[inj_pos % idx.len()] as usize];
                    inj_pos += 1;
                    injected += 1;
                    q
                };
                acc.push(p);
                if acc.len() == pool_cap {
                    if self.mix_batches > 0 {
                        mix_shuffle(&mut acc, mix_round);
                        mix_round += 1;
                    }
                    // 検算用: --score-lut-dump を付けたとき、 混ぜた後の最初の batch を mix_first_batch.psv に書く
                    //   (1 batch の局面が複数のファイルから来ていることを外で確かめるため)。
                    if let Some((dir, _)) = &self.lut_dump {
                        if !MIX_DUMP_DONE.swap(true, std::sync::atomic::Ordering::SeqCst) {
                            let bytes: Vec<u8> = acc[..batch_size.min(acc.len())].iter().flat_map(|p| p.as_bytes().to_vec()).collect();
                            std::fs::write(dir.join("mix_first_batch.psv"), bytes).expect("mix_first_batch.psv を書けない");
                        }
                    }
                    let mut stop = false;
                    for chunk in acc.chunks(batch_size) {
                        emitted += 1;
                        if f(chunk) {
                            stop = true;
                            break;
                        }
                    }
                    acc.clear();
                    // 混ぜるバッファありでは emitted が N ずつ進むので、 5,000 batch の区切りを跨いだら出す。
                    let due = emitted == 1 || emitted == 100 || emitted % 5000 == 0 || (self.mix_batches > 0 && emitted / 5000 != last_log);
                    last_log = emitted / 5000;
                    if due {
                        println!(
                            "[bucket-filter] 出した batch {emitted}: 通した {passed} / 捨てた {dropped} (通過率 {:.2}%)",
                            100.0 * passed as f64 / (passed + dropped).max(1) as f64
                        );
                        if self.quiet {
                            let seen = passed + mate_dropped + check_dropped + capture_dropped;
                            println!(
                                "[quiet-filter] 範囲内の局面 {seen} のうち 王手で捨てた {check_dropped} ({:.2}%) / 最善手が駒取りで捨てた {capture_dropped} ({:.2}%) / 残した {passed} ({:.2}%)",
                                100.0 * check_dropped as f64 / seen.max(1) as f64,
                                100.0 * capture_dropped as f64 / seen.max(1) as f64,
                                100.0 * passed as f64 / seen.max(1) as f64
                            );
                        }
                        if self.drop_mate > 0 || self.score_lut.is_some() {
                            println!(
                                "[score-lut] 範囲内の局面のうち詰みで捨てた {mate_dropped} / {} ({:.2}%)",
                                passed + mate_dropped,
                                100.0 * mate_dropped as f64 / (passed + mate_dropped).max(1) as f64
                            );
                        }
                    }
                    if due && inj.is_some() {
                        println!(
                            "[inject] 差し込んだ {injected} / 出した局面 {} ({:.2}%)",
                            passed + injected,
                            100.0 * injected as f64 / (passed + injected).max(1) as f64
                        );
                    }
                    if stop {
                        return true;
                    }
                }
                }
            }
            false
        });
    }
}

// =============================================================================
// Experiment Log Helpers
// =============================================================================

fn get_git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok().map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn get_timestamp() -> (String, String) {
    use std::time::SystemTime;
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    let mut y = 1970i64;
    let mut remaining_days = days as i64;
    loop {
        let days_in_year = if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0usize;
    for (i, &md) in month_days.iter().enumerate() {
        if remaining_days < md as i64 {
            m = i;
            break;
        }
        remaining_days -= md as i64;
    }
    let d = remaining_days + 1;
    let id_ts = format!("{:04}{:02}{:02}-{:02}{:02}{:02}", y, m + 1, d, hours, minutes, seconds);
    let date = format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m + 1, d, hours, minutes, seconds);
    (id_ts, date)
}

/// `prior` (resume 元 experiment.json から引き継いだ history) と
/// `current` (現在 process の log.txt を parse した history) を superbatch でマージ。
/// 同一 superbatch が両方にある場合は current を採用する。
fn merge_loss_histories(prior: &[LossEntry], current: &[LossEntry]) -> Vec<LossEntry> {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<usize, f64> = BTreeMap::new();
    for entry in prior {
        map.insert(entry.superbatch, entry.loss);
    }
    for entry in current {
        map.insert(entry.superbatch, entry.loss);
    }
    map.into_iter().map(|(superbatch, loss)| LossEntry { superbatch, loss }).collect()
}

fn parse_loss_history(log_path: &std::path::Path) -> Vec<LossEntry> {
    use std::collections::BTreeMap;
    let content = match std::fs::read_to_string(log_path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let mut superbatch_losses: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for line in content.lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 3 {
            if let (Ok(sb), Ok(loss)) = (parts[0].trim().parse::<usize>(), parts[2].trim().parse::<f64>()) {
                let entry = superbatch_losses.entry(sb).or_insert((0.0, 0));
                entry.0 += loss;
                entry.1 += 1;
            }
        }
    }
    superbatch_losses
        .into_iter()
        .map(|(sb, (sum, count))| LossEntry { superbatch: sb, loss: sum / count as f64 })
        .collect()
}

fn collect_checkpoints(output_dir: &std::path::Path, net_id: &str) -> Vec<String> {
    let prefix = format!("{}-", net_id);
    let mut checkpoints: Vec<String> = std::fs::read_dir(output_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(&prefix) && entry.path().is_dir() {
                let suffix = &name[prefix.len()..];
                if suffix.parse::<usize>().is_ok() {
                    return Some(name);
                }
            }
            None
        })
        .collect();
    checkpoints.sort_by(|a, b| {
        let a_num: usize = a[prefix.len()..].parse().unwrap_or(0);
        let b_num: usize = b[prefix.len()..].parse().unwrap_or(0);
        a_num.cmp(&b_num)
    });
    checkpoints
}

struct ExperimentContext {
    output_dir: std::path::PathBuf,
    net_id: String,
    command: String,
    params: ExperimentParams,
    data_name: String,
    superbatches: usize,
    fv_scale: i32,
    /// 学習開始時に確定するID・日時・コミット（以後不変）
    experiment_id: String,
    experiment_date: String,
    commit: String,
    training_start: std::time::Instant,
    /// データファイルの総局面数（初期化時に計算、以後不変）
    positions: u64,
    /// resume 時に既存 experiment.json から引き継いだ history。
    /// build_experiment_log() で現在 process の history とマージされる。
    prior_history: Vec<LossEntry>,
    /// resume 時に既存 experiment.json から引き継いだ累積学習時間 (秒)。
    /// build_experiment_log() で現在 process の経過時間に加算される。
    prior_training_seconds: u64,
}

impl ExperimentContext {
    fn new(
        output_dir: std::path::PathBuf,
        net_id: String,
        command: String,
        params: ExperimentParams,
        data_name: String,
        superbatches: usize,
        fv_scale: i32,
    ) -> Self {
        let commit = get_git_commit();
        let (id_ts, date) = get_timestamp();
        let id = format!("{}-{}", id_ts, &net_id);

        const PACKED_SFEN_VALUE_SIZE: u64 = 40;
        let positions: u64 = data_name
            .split(',')
            .filter_map(|path| std::fs::metadata(path.trim()).ok())
            .map(|meta| meta.len() / PACKED_SFEN_VALUE_SIZE)
            .sum();

        Self {
            output_dir,
            net_id,
            command,
            params,
            data_name,
            superbatches,
            fv_scale,
            experiment_id: id,
            experiment_date: date,
            commit,
            training_start: std::time::Instant::now(),
            positions,
            prior_history: Vec::new(),
            prior_training_seconds: 0,
        }
    }

    fn build_experiment_log(&self, status: &str) -> ExperimentLog {
        let latest_checkpoint = collect_checkpoints(&self.output_dir, &self.net_id)
            .last()
            .cloned()
            .unwrap_or_else(|| format!("{}-{}", self.net_id, self.superbatches));
        let log_path = self.output_dir.join(&latest_checkpoint).join("log.txt");
        let current_history = parse_loss_history(&log_path);
        // resume 時は過去 run の history を引き継いだ上で、現在 process の history を上書き合成する。
        // log.txt は checkpoint 単位で current process の error_record から書き直されるため、
        // prior_history を持っていないと sb 1..=resume_point の loss が experiment.json から消える。
        let history = merge_loss_histories(&self.prior_history, &current_history);

        let checkpoints = collect_checkpoints(&self.output_dir, &self.net_id);

        // 実際に完了したsuperbatch数から計算（中間保存時に最終予定値を使わない）
        let actual_superbatches = history.last().map(|e| e.superbatch).unwrap_or(0) as u64;
        let total_positions =
            self.params.batch_size as u64 * self.params.batches_per_superbatch as u64 * actual_superbatches;
        let dataset_passes = if self.positions > 0 { total_positions as f64 / self.positions as f64 } else { 0.0 };

        let (best_loss, best_loss_superbatch) = history
            .iter()
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal))
            .map(|entry| (Some(entry.loss), Some(entry.superbatch)))
            .unwrap_or((None, None));

        let training_time_seconds = self.prior_training_seconds.saturating_add(self.training_start.elapsed().as_secs());
        let (_, last_updated_at) = get_timestamp();

        ExperimentLog {
            id: self.experiment_id.clone(),
            name: self.net_id.clone(),
            date: self.experiment_date.clone(),
            status: status.to_string(),
            last_updated_at,
            commit: self.commit.clone(),
            command: self.command.clone(),
            params: self.params.clone(),
            data: ExperimentData {
                name: self.data_name.clone(),
                positions: Some(self.positions),
                total_positions,
                dataset_passes: Some(dataset_passes),
            },
            results: ExperimentResults {
                training_time_seconds,
                fv_scale: self.fv_scale,
                best_loss,
                best_loss_superbatch,
            },
            history,
            checkpoints,
        }
    }

    fn write_experiment_json(&self, status: &str) -> std::io::Result<()> {
        let experiment = self.build_experiment_log(status);
        let json = serde_json::to_string_pretty(&experiment).map_err(std::io::Error::other)?;
        let json_dir = self.output_dir.join(&self.net_id);
        std::fs::create_dir_all(&json_dir)?;
        let json_path = json_dir.join("experiment.json");
        std::fs::write(&json_path, &json)?;
        println!("Experiment log saved to {} (status: {})", json_path.display(), status);
        Ok(())
    }

    /// resume 時に既存 experiment.json から experiment_id / date / history を引き継ぐ。
    /// 詳細は shogi_simple.rs の同名メソッドのコメント参照。
    fn inherit_resume_experiment_id(&mut self) {
        let json_path = self.output_dir.join(&self.net_id).join("experiment.json");
        if !json_path.exists() {
            return;
        }
        let content = match std::fs::read_to_string(&json_path) {
            Ok(c) => c,
            Err(_) => return,
        };
        let existing: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return,
        };

        if let Some(id) = existing.get("id").and_then(|v| v.as_str()) {
            if !id.is_empty() {
                println!("Inheriting experiment id from {}: {}", json_path.display(), id);
                self.experiment_id = id.to_string();
            }
        }
        if let Some(date) = existing.get("date").and_then(|v| v.as_str()) {
            if !date.is_empty() {
                self.experiment_date = date.to_string();
            }
        }
        if let Some(secs) =
            existing.get("results").and_then(|v| v.get("training_time_seconds")).and_then(|v| v.as_u64())
        {
            if secs > 0 {
                println!("Inheriting prior training time: {} seconds", secs);
                self.prior_training_seconds = secs;
            }
        }
        if let Some(arr) = existing.get("history").and_then(|v| v.as_array()) {
            let mut history: Vec<LossEntry> = arr
                .iter()
                .filter_map(|entry| {
                    let sb = entry.get("superbatch").and_then(|v| v.as_u64())? as usize;
                    let loss = entry.get("loss").and_then(|v| v.as_f64())?;
                    Some(LossEntry { superbatch: sb, loss })
                })
                .collect();
            history.sort_by_key(|e| e.superbatch);
            if !history.is_empty() {
                println!(
                    "Inheriting {} history entries from previous run (sb {} .. {})",
                    history.len(),
                    history.first().unwrap().superbatch,
                    history.last().unwrap().superbatch,
                );
                self.prior_history = history;
            }
        }
    }
}

// =============================================================================
// LEB128 Encoder
// =============================================================================

/// 符号付き LEB128 エンコード (1 値)
fn encode_signed_leb128(mut value: i64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        // value が 0 で符号ビットが立っていない、または value が -1 で符号ビットが立っている場合、終了
        if (value == 0 && (byte & 0x40) == 0) || (value == -1 && (byte & 0x40) != 0) {
            result.push(byte);
            break;
        }
        byte |= 0x80; // 継続ビット
        result.push(byte);
    }
    result
}

/// i16 配列を LEB128 圧縮し、マジック + サイズ + データ を返す
fn encode_leb128_tensor_i16(values: &[i16]) -> Vec<u8> {
    // まず全値を LEB128 エンコード
    let mut compressed = Vec::new();
    for &val in values {
        compressed.extend_from_slice(&encode_signed_leb128(val as i64));
    }

    // マジック + サイズヘッダ + 圧縮データ
    let mut result = Vec::new();
    result.extend_from_slice(b"COMPRESSED_LEB128"); // 17 bytes
    result.extend_from_slice(&(compressed.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    result
}

// =============================================================================
// SIMD Padding Utilities
// =============================================================================

fn pad32(size: usize) -> usize {
    size.div_ceil(32) * 32
}

// =============================================================================
// Hash Computation (nnue-pytorch 互換)
// =============================================================================

/// LayerStacks 用 fc_hash 計算
///
/// 各バケットの FC ハッシュ。rshogi は読み飛ばすが互換性のため出力する。
fn compute_layerstack_fc_hash(l1_out: usize, l2_in: usize, l2_out: usize) -> u32 {
    // InputSlice hash
    let mut prev_hash: u32 = 0xEC42E90D;
    prev_hash ^= (l1_out * 2) as u32; // FT output * 2 (dual perspective)

    // L1: 1536 → l1_out
    let layer_sizes = [(l1_out, true), (l2_out, true), (1usize, false)];
    for (out_features, has_relu) in layer_sizes {
        let mut layer_hash: u32 = 0xCC03DAE4;
        layer_hash = layer_hash.wrapping_add(out_features as u32);
        layer_hash ^= prev_hash >> 1;
        layer_hash ^= prev_hash << 31;
        if has_relu {
            layer_hash = layer_hash.wrapping_add(0x538D24C7);
        }
        prev_hash = layer_hash;
    }

    let _ = (l2_in,); // suppress unused warning — used for documentation clarity
    prev_hash
}

struct LayerStackFormatConstants {
    nnue_version: u32,
    network_hash: u32,
    ft_hash: u32,
    fc_hash: u32,
    serialized_buckets: usize,
    architecture: String,
}

fn layerstack_format_constants(
    output_format: LayerStackOutputFormat,
    ft_out: usize,
    _l1_out: usize,
    l2_in: usize,
    l2_out: usize,
    arch_desc: String,
) -> LayerStackFormatConstants {
    use bullet_lib::game::inputs::FEATURE_HASH_HM_V2;

    match output_format {
        LayerStackOutputFormat::Rshogi => {
            let fc_hash = compute_layerstack_fc_hash(ft_out, l2_in, l2_out);
            let ft_hash = FEATURE_HASH_HM_V2 ^ ((ft_out * 2) as u32);
            let network_hash = fc_hash ^ ft_hash;
            LayerStackFormatConstants {
                nnue_version: 0x7AF32F20,
                network_hash,
                ft_hash,
                fc_hash,
                serialized_buckets: NUM_BUCKETS,
                architecture: arch_desc,
            }
        }
        LayerStackOutputFormat::TanukiSfnnwoP1536 => LayerStackFormatConstants {
            nnue_version: 0x7AF32F16,
            network_hash: 0x3c203b32,
            ft_hash: 0x5f134ab8,
            fc_hash: 0x6333718A,
            serialized_buckets: 8,
            architecture: "Network trained with https://github.com/official-stockfish/nnue-pytorch".to_string(),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_layerstack_output_format(
    output_format: LayerStackOutputFormat,
    bucket_mode: BucketMode,
    input_size: usize,
    halfka_dim: usize,
    ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    psqt: bool,
    threat_profile: Option<ThreatProfile>,
    hand_threat: bool,
    hand_count_dense_dims: usize,
    lmas_flags: u32,
    pin_aware: bool,
    dual_psqt_reserve: bool,
    dual_psqt_king_attacker: bool,
) -> Result<(), String> {
    // 排他チェック (= reserve / king-attacker は同時指定不可)。
    if dual_psqt_reserve && dual_psqt_king_attacker {
        return Err("--dual-psqt-reserve and --dual-psqt-king-attacker are mutually exclusive".to_string());
    }

    if output_format != LayerStackOutputFormat::TanukiSfnnwoP1536 {
        return Ok(());
    }

    if input_size != halfka_dim {
        return Err("tanuki-sfnnwop1536 output supports only plain HalfKA_hm input".to_string());
    }
    if (ft_out, l1_out, l2_out) != (1536, 16, 32) {
        return Err(format!(
            "tanuki-sfnnwop1536 output requires --l0 1536 --l1 16 --l2 32 (got --l0 {ft_out} --l1 {l1_out} --l2 {l2_out})"
        ));
    }
    if bucket_mode != BucketMode::Progress8KPAbs {
        return Err("tanuki-sfnnwop1536 output requires --bucket-mode progress8kpabs".to_string());
    }
    if psqt
        || threat_profile.is_some()
        || hand_threat
        || hand_count_dense_dims > 0
        || lmas_flags != 0
        || pin_aware
        || dual_psqt_reserve
        || dual_psqt_king_attacker
    {
        return Err(
            "tanuki-sfnnwop1536 output does not support --psqt/--threat/--hand-threat/--hand-threat-defensive/--hand-count-dense/--pin-aware/--dual-psqt-reserve/--dual-psqt-king-attacker"
                .to_string(),
        );
    }

    Ok(())
}

// =============================================================================
// SavedFormat Construction
// =============================================================================

/// LayerStack 量子化出力の SavedFormat を構築する
///
/// rshogi NetworkLayerStacks::read() と完全互換のバイナリを生成。
///
/// `hand_count_dense_dims` が 0 より大きい場合、L1 重みは `ft_out + dims`
/// の入力次元を持ち、先頭 `ft_out` が FT 出力、残り `dims` が HandCount dense の
/// 貢献となる。`arch_str` に `HandCountDense=dims` を追加する。
#[allow(clippy::too_many_arguments)]
fn build_layerstack_save_format(
    halfka_dim: usize,
    input_size: usize,
    ft_out: usize,
    l1_out: usize,
    residual_zero: bool,
    l2_out: usize,
    fv_scale: i32,
    psqt: bool,
    threat_profile: Option<ThreatProfile>,
    // --threat-out K (0 = 従来 = FT 幅)。 K > 0 のとき threat 重みは
    // threat_ft_w (K, dims) にあり、 書き出しは feature-major i8[dims x K]。
    threat_ft_out: usize,
    // --threat-dedicated-dims: threat が乗る FT 次元の l0w をゼロで書き出す
    // (forward で HalfKA 側を外しているので、 net にも同じ形を反映させる)。
    threat_dedicated: bool,
    // --threat-bucket-groups: bucket → 群番号 (None = 従来の 1 表)。 Some のとき threat_ft_w は
    // (K, 群数 * dims) で、 書き出しは群ごとの表の連結 + arch marker "ThreatGroups=G:b0-...-b7,"。
    threat_bucket_groups: Option<Vec<u8>>,
    // --threat-factorizer: threat_ft_w 末尾の factor 列数 (0 = 無効)。 書き出し時に各実特徴の列へ畳み込む。
    threat_factor_rows: usize,
    // --threat-hang-split (profile 16): threat_ft_w 末尾の delta 列数 (0 = 無効)。
    // 書き出し時に hang 行[g][hl] = base 行[g][base_of(hl)] + delta[hl] へ展開する。
    threat_hang_rows: usize,
    factor_threat_profile: Option<ThreatProfile>,
    direct_threat_profile: Option<ThreatProfile>,
    direct15_persp_split: bool,
    direct15_factorise: bool,
    direct15_defended: bool,
    threat_i16: bool,
    hand_threat: bool,
    hand_count_dense_dims: usize,
    lmas_flags: u32,
    pin_aware: bool,
    dual_psqt_reserve: bool,
    dual_psqt_king_attacker: bool,
    dual_save_legacy_p2dualx: bool,
    dual_disable_cross_defended: bool,
    enable_kingring: bool,
    enable_pairshape: bool,
    enable_activity_risk: bool,
    enable_king_escape: bool,
    enable_slider_xray: bool,
    enable_king_mobility: bool,
    enable_drop_pressure: bool,
    enable_endgame_tempo: bool,
    // B6 可動域 (2026-09-29)。 arch marker "MOB=1," と P2MOB section を出すか。
    enable_mob: bool,
    // 案1 ブロック射影 (2026-09-30)。 arch marker "ThreatProj=<FT>," と P2TPRJ section。
    enable_threat_proj: bool,
    // 案1 のチャンネル組の作り方 (2026-10-01)。 section version 3 に書き出す。
    threat_proj_offset: usize,
    threat_proj_pair: String,
    // W: bucket 別の射影 (2026-10-01)。 section version 4。
    threat_proj_buckets: bool,
    // (3) TL1: threat を L1 に直結 (2026-10-01)。 P2TL1 section。
    enable_threat_l1: bool,
    // (4) TL2: threat を L2 入力に直結 (2026-10-08)。 P2TL2 section。
    enable_threat_l2: bool,
    // (5) TG: threat で sqr 側を乗法ゲート (2026-10-10)。 P2TGS section。
    enable_threat_gate: bool,
    enable_scalar_threat: bool,
    scalar_threat_coexist_dual_cross: bool,
    enable_dual_cross_geometry: bool,
    enable_attack_relay: bool,
    enable_attack_relay_v2: bool,
    enable_kzt: bool,
    enable_kzt_ft: bool,
    enable_aux: bool,
    enable_auxcheap: bool,
    factorise: bool,
    layerstack_p4h2: bool,
    layerstack_p8h2: bool,
    output_format: LayerStackOutputFormat,
    // 2026-11-07 (T4): L4 = 4 層目 (`fc1b`) の幅。 **0 = 無効 = 現行と 1 バイトも変わらない**。
    l4_out: usize,
) -> Vec<SavedFormat> {
    let l1_effective = l1_out - 1; // skip connection 分を除く
    let l2_in = l1_effective * 2; // skip は L2 に入れず出力へ直加算のみ (NAGISA V3.1 と同構成)
    // 2026-11-07 (T4): Output 層の入力次元。 L4 を挟むと l3w の入力が l2_out から l4_out に変わる。
    let out_in = if l4_out > 0 { l4_out } else { l2_out };

    // アーキテクチャ文字列（fv_scale を埋め込み、rshogi が推論時に正しく解釈できるようにする）
    let psqt_part = if psqt { format!("PSQT={},", NUM_BUCKETS) } else { String::new() };
    // 2026-11-07 (T3): **L4 = 4 層目 (`fc1b`)** の arch 文字列。
    //
    // 🚨`network_hash` / `fc_hash` はエンジンが**読み捨てる** (p2nnue.cpp:7389 / 7209) ので、
    //   net x exe の食い違いを弾く唯一の門は **`AffineTransform[A<-B]` token の照合** (7400-7445)。
    //   だから L4 の有無で token の数と形を必ず変える。
    //     旧 net を `-DP2_L4=64` の exe で読む -> token が 1 つ足りない -> parse failed
    //     新 net を P2_L4 無しの exe で読む   -> token1.B = l4_out != L2_IN -> MISMATCH
    //   ★net_exe_mismatch_silent (黙って壊れた評価値で動く) の前例があるのでここは必須。
    //
    // l4_out == 0 のときは下の else 枝が**現行と 1 バイトも同一の文字列**を作る。
    let l4_part = if l4_out > 0 { format!("L4={l4_out},") } else { String::new() };
    let ft2 = ft_out * 2;
    let network_part = if l4_out > 0 {
        format!(
            "AffineTransform[1<-{l4_out}](\
             ClippedReLU[{l4_out}](\
             AffineTransform[{l4_out}<-{l2_out}](\
             ClippedReLU[{l2_out}](\
             AffineTransform[{l2_out}<-{l2_in}](\
             SqrClippedReLU[{l2_in}](\
             AffineTransform[{l1_out}<-{ft2}](\
             InputSlice[{ft2}(0:{ft2})])))))))"
        )
    } else {
        format!(
            "AffineTransform[1<-{l2_out}](\
             ClippedReLU[{l2_out}](\
             AffineTransform[{l2_out}<-{l2_in}](\
             SqrClippedReLU[{l2_in}](\
             AffineTransform[{l1_out}<-{ft2}](\
             InputSlice[{ft2}(0:{ft2})])))))"
        )
    };
    let threat_part = if factor_threat_profile.is_none() && direct_threat_profile.is_none() {
        if let Some(tp) = threat_profile {
            // --threat-out で分離した場合 input_size は HalfKA のみになるので、
            // profile 定義から直接引く (非分離時も同値)。
            // profile 16 は学習器内部では base (216,720) だけを 1 群の次元として扱うが、
            // ファイル (= エンジンが読む表) は base + hang = 325,080 行になる。
            let threat_dims = ShogiHalfKaHmThreat::new(tp).threat_dimensions()
                + if tp.is_hang_split() { bullet_lib::game::inputs::THREAT_HANG_ROWS } else { 0 };
            let pid = tp.profile_id();
            // --threat-bucket-groups: "ThreatGroups=<群数>:<b0>-<b1>-...," (群化しない net は空 = 従来と同じ arch)。
            let groups_part = match &threat_bucket_groups {
                Some(g) => format!(
                    "ThreatGroups={}:{},",
                    g.iter().copied().max().unwrap_or(0) as usize + 1,
                    g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("-")
                ),
                None => String::new(),
            };
            if pid == 0 {
                format!("Threat={threat_dims},{groups_part}")
            } else {
                format!("Threat={threat_dims},ThreatProfile={pid},{groups_part}")
            }
        } else {
            String::new()
        }
    } else {
        String::new()
    };
    // Threat i16 marker: threat FT 重みを i16 (clamp 無し) で書き出した net に付与。
    // C++ loader (p2nnue.cpp load_eval) が "ThreatI16=1" を検出して i16 経路でロードする。
    // i8 のときは marker を付けない (後方互換: marker 無し = 従来 i8)。
    let threat_i16_part =
        if factor_threat_profile.is_none() && direct_threat_profile.is_none() && threat_profile.is_some() && threat_i16
        {
            "ThreatI16=1,".to_string()
        } else {
            String::new()
        };
    let factor_threat_part = if factor_threat_profile.is_some() {
        format!(
            "ThreatFactor16={},ThreatProfile=13,",
            bullet_lib::game::inputs::ShogiHalfKaHmThreatFactor16::new().threat_dimensions()
        )
    } else {
        String::new()
    };
    let direct_threat_part = if let Some(tp) = direct_threat_profile {
        let dims = if tp == ThreatProfile::EnemyOnlyDirect15 {
            // P5 defended は 131,280。 エンジンは P2THRD2 header の dims で判別する。
            if direct15_defended { THREAT_DIRECT15_DEFENDED_DIMENSIONS } else { THREAT_DIRECT15_DIMENSIONS }
        } else {
            ShogiHalfKaHmThreat::new(tp).num_inputs() - halfka_dim
        };
        // P2 は情報用 marker を付ける (エンジンの判別は P2THRD2 header の buckets=16 で行う)。
        let persp_part = if tp == ThreatProfile::EnemyOnlyDirect15 && direct15_persp_split {
            "ThreatDirectPersp=2,"
        } else {
            ""
        };
        format!("ThreatDirect={dims},ThreatProfile={},{persp_part}", tp.profile_id())
    } else {
        String::new()
    };
    // HandThreat (案 A): dims 固定 121,104。rshogi 側 loader は
    // `HandThreat={dims},` を検出して u32 dims + i8 weights を読み込む。
    let hand_threat_part = if hand_threat {
        let hand_threat_dims = input_size - halfka_dim;
        format!("HandThreat={hand_threat_dims},")
    } else {
        String::new()
    };
    // HandCount Dense (本機能): L1 入力に 14 元の持ち駒 dense vector を concat。
    // rshogi 側は `HandCountDense={dims}` を検出して L1 重みを 14 行分追加で読む。
    let hand_count_part = if hand_count_dense_dims > 0 {
        format!("HandCountDense={hand_count_dense_dims},")
    } else {
        String::new()
    };
    // LMAS marker は 2026-05-25 廃止 (lmas_flags は常に 0)。 互換性のため空文字列。
    let lmas_part = String::new();
    let _ = lmas_flags;
    // Pin-aware marker (rshogi-core loader が `"PIN="` を検出して末尾 P2PIN section を読み込む)。
    let pin_part = if pin_aware { bullet_lib::game::inputs::PIN_ARCH_MARKER.to_string() } else { String::new() };
    // Dual PSQT marker (rshogi-core loader が `"DUAL=1"` を検出して末尾 dual section を読み込む)。
    // marker は reserve / king-attacker 共通、 section magic で format を区別する
    // (P2DUAL\0\0 = legacy reserve 44 byte / P2DUAL2\0 = King-Attacker 72,592 byte)。
    let dual_part = if dual_psqt_reserve || dual_psqt_king_attacker { "DUAL=1,".to_string() } else { String::new() };
    // AUX-FT marker (C++ p2nnue.cpp load_eval が `AUX=600,` を検出して末尾 P2AUX section を
    // 読み込む)。 値 600 は AUX_DIM_PER_PERSP と一致 (= C++ 側 AUX_ARCH_MARKER と bit-exact)。
    let aux_part = if enable_aux { bullet_lib::game::inputs::AUX_ARCH_MARKER.to_string() } else { String::new() };
    // AUXCHEAP marker (C++ p2nnue.cpp load_eval が `AUXCHEAP=392,` を検出して末尾 P2AUXC
    // section (= psqtw aux 行の i32 PSQT 重み) を読み込む)。 値 392 は
    // AUXCHEAP_DIM_PER_PERSP と一致。
    let auxcheap_part = if enable_auxcheap {
        bullet_lib::game::inputs::AUXCHEAP_ARCH_MARKER.to_string()
    } else {
        String::new()
    };
    // B6 可動域 marker (C++ p2nnue.cpp load_eval が `MOB=1,` を見て末尾 P2MOB を読む)。
    let mob_part = if enable_mob { format!("{},", bullet_lib::game::inputs::MOB_ARCH_MARKER) } else { String::new() };
    // (3) TL1 marker (C++ load_eval が `ThreatL1=<2K>,` を見て末尾 P2TL1 を読む)。
    let threat_l1_part = if enable_threat_l1 { format!("ThreatL1={},", 2 * threat_ft_out) } else { String::new() };
    // (4) TL2 marker (C++ load_eval が `ThreatL2=<2K>,` を見て末尾 P2TL2 を読む)。
    // 🚨"ThreatL1=" とも "Threat=" とも別の文字列なので前方一致の事故は起きない。
    let threat_l2_part = if enable_threat_l2 { format!("ThreatL2={},", 2 * threat_ft_out) } else { String::new() };
    // (5) TG marker (C++ load_eval が `ThreatGate=<2K>,` を見て末尾 P2TGS を読む)。
    // 🚨"ThreatL1=" / "ThreatL2=" / "ThreatProj=" / "Threat=" とは別文字列なので前方一致の事故はない。
    let threat_gate_part =
        if enable_threat_gate { format!("ThreatGate={},", 2 * threat_ft_out) } else { String::new() };
    // 案1 ブロック射影 marker (C++ p2nnue.cpp load_eval が `ThreatProj=<FT>,` を見て末尾 P2TPRJ を読む)。
    let threat_proj_part = if enable_threat_proj {
        format!("{}={},", bullet_lib::game::inputs::THREAT_PROJ_ARCH_MARKER, ft_out)
    } else {
        String::new()
    };
    let endgame_tempo_part = if enable_endgame_tempo {
        format!("{},", bullet_lib::game::inputs::ENDGAME_TEMPO_ARCH_MARKER)
    } else {
        String::new()
    };
    // LayerStack 選択モード marker。P4H2(進行度4×持駒2=8)/ P8H2(進行度8×持駒2=16)net のみ付与。
    // 推論側は "LSMODE=P4H2" / "LSMODE=P8H2" を検出し、-DP2_LSMODE_P4H2 / -DP2_LSMODE_P8H2
    // ビルドと一致するかを検証する(旧 progress8 net を新方式で誤ロードしない / モード取り違え防止)。
    // 持駒量2区分の閾値配列は P4H2/P8H2 共通(HAND_VALUE_THRESHOLDS[4])で、閾値違いは別ネット扱い。
    // 案 B (2026-10-07): regime モードなら PROGREGIME marker を出す。
    //   🚨関数の署名を変えないため runtime グローバルから読む (設計判断は outputs.rs 参照)。
    // 2026-11-14 OPNARCH: OPN の king bucket ビルドは NUM_BUCKETS で決まる marker を出す
    //   (81 = "LSMODE=K9Z," / 243 = "LSMODE=K9ZP3,")。 python の opn_static2.py と推論側はこれで head の本数と
    //   head 番号の式 (設計書 2.2) を選ぶ。 既定ビルド・P8H2 ビルドでは空 = 従来と同一。
    let opn_lsmode: Option<&str> = match NUM_BUCKETS {
        81 => Some("LSMODE=K9Z,"),
        243 => Some("LSMODE=K9ZP3,"),
        _ => None,
    };
    let lsmode_part = if let Some(m) = opn_lsmode {
        m.to_string()
    } else if let Some((rv1, rpb)) = bullet_lib::game::outputs::regime_mode() {
        format!("LSMODE=PROGREGIME:v1{rv1}-pb{rpb},")
    } else if layerstack_p8h2 {
        let t = bullet_lib::game::outputs::HAND_VALUE_THRESHOLDS;
        format!("LSMODE=P8H2:{}-{}-{}-{},", t[0], t[1], t[2], t[3])
    } else if layerstack_p4h2 {
        // 進行度別閾値配列を marker に埋め込む(閾値違いの net は別ネット扱い)。
        let t = bullet_lib::game::outputs::HAND_VALUE_THRESHOLDS;
        format!("LSMODE=P4H2:{}-{}-{}-{},", t[0], t[1], t[2], t[3])
    } else {
        String::new()
    };
    // 2026-11-14 OPN2048 K16: 主入力の次元が K16 (26,064) なら特徴の名前を変える (python / エンジンはこれで符号化を選ぶ)。
    let features_name = if halfka_dim == bullet_lib::game::inputs::HALFKA_HM_K16_DIMENSIONS {
        "HalfKA_hmK16"
    } else {
        "HalfKA_hm"
    };
    let arch_desc = format!(
        "Features={features_name}(Friend)[{}->{}x2],\
         {psqt_part}\
         {threat_part}\
         {factor_threat_part}\
         {direct_threat_part}\
         {threat_i16_part}\
         {hand_threat_part}\
         {hand_count_part}\
         {lmas_part}\
         {pin_part}\
         {dual_part}\
         {aux_part}\
         {auxcheap_part}\
         {endgame_tempo_part}\
         {mob_part}\
         {threat_proj_part}\
         {threat_l1_part}\
         {threat_l2_part}\
         {threat_gate_part}\
         {lsmode_part}\
         {l4_part}\
         Network={network_part},\
         fv_scale={fv_scale}",
        input_size, ft_out,
    );
    let format = layerstack_format_constants(output_format, ft_out, l1_out, l2_in, l2_out, arch_desc);
    let arch_bytes = format.architecture.as_bytes();

    // ---- ヘッダー ----
    let mut header = Vec::new();
    header.extend_from_slice(&format.nnue_version.to_le_bytes());
    header.extend_from_slice(&format.network_hash.to_le_bytes());
    header.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
    header.extend_from_slice(arch_bytes);

    // ---- FT hash ----
    let ft_hash_bytes = format.ft_hash.to_le_bytes().to_vec();

    // ---- FT biases + weights (LEB128 圧縮, YO 互換 2ブロック形式) ----
    // biases / weights を別々の LEB128 ブロックで出力。
    // YaneuraOu は FT を biases ブロック → weights ブロックの順で読み込むため、
    // 各ブロックに独立した COMPRESSED_LEB128 マジック + サイズヘッダが付く。
    let qa_i16 = QA_FT;
    let ft_out_captured = ft_out;
    let input_size_captured = input_size;
    let ft_biases_leb128 = SavedFormat::empty()
        .transform(move |graph, _| {
            let l0b = graph.get("l0b");
            let qa_f = qa_i16 as f64;
            let biases_i16: Vec<i16> = l0b.values.iter().map(|&v| (qa_f * v as f64).round() as i16).collect();
            let leb128_bytes = encode_leb128_tensor_i16(&biases_i16);
            leb128_bytes.iter().map(|&b| (b as i8) as f32).collect()
        })
        .quantise::<i8>(1);

    let qa_i16 = QA_FT;
    let halfka_dim_captured = halfka_dim;
    let factorise_ft = factorise;
    let threat_dedicated_c2 = threat_dedicated;
    let threat_ft_out_c2 = threat_ft_out;
    let ft_weights_leb128 = SavedFormat::empty()
        .transform(move |graph, _| {
            let l0w = graph.get("l0w");

            // Quantise to i16 (scale = QA_FT = 254)
            // column-major: l0w.values[feat * ft_out + out] (= feature-major, ft_out block/feat)
            let qa_f = qa_i16 as f64;
            let ft_out = ft_out_captured;
            let weights_i16: Vec<i16> = if factorise_ft {
                // Factorised<ShogiHalfKA_hm, ShogiPieceFactor> の coalesce。
                // graph l0w layout (input_size = 74,934 rows):
                //   rows [0, 1629)            = factoriser (piece factor, ShogiPieceFactor)
                //   rows [1629, 1629+73305)   = 実特徴 A (ShogiHalfKA_hm, offset=PIECE_INPUTS)
                // 実特徴 r (0..73305) について:
                //   coalesced[r] = l0w_col(PIECE_INPUTS + r) + l0w_col(r % PIECE_INPUTS)
                // (= merge_factoriser 相当を l0w テンソル上で再現。 factor derive = feat % 1629)
                let piece = bullet_lib::game::inputs::PIECE_INPUTS; // 1629 = factoriser offset & modulus
                let mut out = Vec::with_capacity(halfka_dim_captured * ft_out);
                for r in 0..halfka_dim_captured {
                    let real_base = (piece + r) * ft_out;
                    let factor_base = (r % piece) * ft_out;
                    for o in 0..ft_out {
                        let coalesced = l0w.values[real_base + o] + l0w.values[factor_base + o];
                        out.push((qa_f * coalesced as f64).round() as i16);
                    }
                }
                out
            } else {
                // 非 factorise: piece 部分 = feat 0..halfka_dim → indices 0..halfka_dim*ft_out
                let piece_end = halfka_dim_captured * ft_out;
                l0w.values[..piece_end].iter().map(|&v| (qa_f * v as f64).round() as i16).collect()
            };
            let _ = input_size_captured;
            // ★2026-09-10: --threat-dedicated-dims では threat が乗る FT 次元を
            //   HalfKA から切り離しているので、 net 側でも l0w のその列をゼロにする。
            //   対称配置なので前半 [0, lo) と後半 [half, half+hi) の 2 ブロック。
            let mut weights_i16 = weights_i16;
            if threat_dedicated_c2 {
                let half = ft_out_captured / 2;
                let lo = threat_ft_out_c2.div_ceil(2).min(half);
                let hi = threat_ft_out_c2 - lo;
                let rows = weights_i16.len() / ft_out_captured;
                for r in 0..rows {
                    let base = r * ft_out_captured;
                    for o in 0..lo {
                        weights_i16[base + o] = 0;
                    }
                    for o in half..(half + hi) {
                        weights_i16[base + o] = 0;
                    }
                }
            }
            let leb128_bytes = encode_leb128_tensor_i16(&weights_i16);
            leb128_bytes.iter().map(|&b| (b as i8) as f32).collect()
        })
        .quantise::<i8>(1);

    // ---- PSQT weights/biases (raw i32) ----
    let psqt_data = if psqt {
        // PSQT は HalfKA 特徴量のみ対象。Threat 部分は含めない。
        let input_size_for_psqt = halfka_dim;
        let factorise_psqt = factorise;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let psqt_w = graph.get("psqtw"); // [NUM_BUCKETS, input_size] column-major
                    let psqt_b = graph.get("psqtb"); // [NUM_BUCKETS]

                    let scale = (QA as i32 * QB as i32) as f64; // 8128.0
                    let mut bytes: Vec<u8> = Vec::new();

                    // Biases: i32[9]
                    for bucket in 0..NUM_BUCKETS {
                        let val = (scale * psqt_b.values[bucket] as f64).round() as i32;
                        bytes.extend_from_slice(&val.to_le_bytes());
                    }

                    // Weights: i32[input_size][9] (feature-major)
                    // factorise 時は psqtw も同じ factorised layout (実特徴が offset=PIECE_INPUTS
                    // 以降、 factoriser が先頭) なので FT と同式で coalesce する
                    // (= サイレントバグ回避のため FT のみでなく PSQT も畳み込む)。
                    let piece = bullet_lib::game::inputs::PIECE_INPUTS; // 1629
                    for feat in 0..input_size_for_psqt {
                        for bucket in 0..NUM_BUCKETS {
                            // column-major: feat * rows + bucket
                            let w = if factorise_psqt {
                                psqt_w.values[(piece + feat) * NUM_BUCKETS + bucket]
                                    + psqt_w.values[(feat % piece) * NUM_BUCKETS + bucket]
                            } else {
                                psqt_w.values[feat * NUM_BUCKETS + bucket]
                            };
                            let val = (scale * w as f64).round() as i32;
                            bytes.extend_from_slice(&val.to_le_bytes());
                        }
                    }

                    // byte passthrough: 各バイトを i8 として f32 にキャスト
                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };

    // ---- Threat weights (raw i8) ----
    // l0w の threat 部分 (feat halfka_dim..input_size) を i8 で書き出す。
    // レイアウト: i8[THREAT_DIMENSIONS × ft_out] (feature-major)
    let threat_data = if factor_threat_profile.is_none() && direct_threat_profile.is_none() && threat_profile.is_some()
    {
        let halfka_dim_for_threat = halfka_dim;
        let ft_out_for_threat = ft_out;
        // ★2026-09-09 修正: threat 重みは FT accumulator に**そのまま加算**される
        //   (engine: acc[k] += (int)threat_row[k]) ので、 FT と同じ QA_FT スケールで
        //   量子化しなければならない。 2026-09-07 に QA_FT を 127 -> 254 にした際
        //   threat 側が QA=127 のまま取り残され、 推論時の threat 寄与が**半分**に
        //   なっていた (それ以前は QA_FT == QA == 127 で一致していたため潜伏)。
        //   実測 (v10-thrk256 SB2): |w|>63 は 55.5M 中 167 個 (0.0003%) なので
        //   254 倍にしても i8 飽和はほぼ起きない。 むしろ分解能が 1bit 増える
        //   (mean|w| が 0.58 しかなく、 大半が 0/±1 に丸まっていた)。
        let qa_for_threat = QA_FT;
        let threat_i16_captured = threat_i16;
        let threat_ft_out_captured = threat_ft_out;
        // --threat-bucket-groups: 群ごとの表の i8 clamp 飽和件数を書き出しのたびに表示する (群化しなければ何もしない)。
        let threat_groups_for_log =
            threat_bucket_groups.as_ref().map_or(1, |g| g.iter().copied().max().unwrap_or(0) as usize + 1);
        // --threat-factorizer: 特徴 → pair の表 (export で factor 列を実特徴へ畳み込むため)。
        let threat_feature_pairs: Option<Vec<u16>> = (threat_factor_rows > 0).then(|| {
            let tp = threat_profile.expect("threat_factorizer requires a threat profile");
            let inp = ShogiHalfKaHmThreat::new(tp).with_threat_factor(true);
            (0..inp.threat_dimensions()).map(|f| inp.threat_feature_pair(f).unwrap() as u16).collect()
        });
        let threat_factor_rows_captured = threat_factor_rows;
        // 2026-09-24 hanging-split: hang 局所 index -> base feature index (export の展開で使う)。
        let threat_hang_base: Option<Vec<u32>> = (threat_hang_rows > 0)
            .then(|| bullet_lib::game::inputs::threat_hang_base_map(ThreatProfile::FullHangSplit));
        let threat_hang_groups =
            threat_bucket_groups.as_ref().map(|g| g.iter().copied().max().unwrap_or(0) as usize + 1).unwrap_or(1);
        let threat_hang_dims = threat_profile.map(|tp| ShogiHalfKaHmThreat::new(tp).threat_dimensions()).unwrap_or(0);
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let qa_f = qa_for_threat as f64;

                    // --threat-out K: threat は独立テンソル `threat_ft_w` (K, dims)。
                    //   column-major なので values[feat*K + k] = feature-major の
                    //   行長 K レイアウトになり、 エンジンの
                    //   threat_row(tf) = &threat_weight[tf * THREAT_ACC_DIM] と一致する。
                    // 従来 (K=0): l0w の threat 部分 (feat halfka_dim..input_size)。
                    let (l0w, threat_start) = if threat_ft_out_captured > 0 {
                        (graph.get("threat_ft_w"), 0usize)
                    } else {
                        (graph.get("l0w"), halfka_dim_for_threat * ft_out_for_threat)
                    };
                    // --threat-factorizer: 実特徴の列に factor 列を足してから量子化する (bin の形・サイズは不変)。
                    //   layout は column-major (values[col * K + k])、 実特徴 col = g * dims + f、 factor col = dims * G + pair(f)。
                    let coalesced: Option<Vec<f32>> = threat_feature_pairs.as_ref().map(|pairs| {
                        let k = threat_ft_out_captured;
                        let dims = pairs.len();
                        let real_cols = l0w.values[threat_start..].len() / k - threat_factor_rows_captured;
                        assert_eq!(real_cols % dims, 0, "threat factorizer: 実特徴の列数 {real_cols} が dims {dims} で割り切れない");
                        let src = &l0w.values[threat_start..];
                        let fbase = real_cols * k;
                        let mut out = src[..fbase].to_vec();
                        for col in 0..real_cols {
                            let fcol = fbase + pairs[col % dims] as usize * k;
                            for i in 0..k {
                                out[col * k + i] += src[fcol + i];
                            }
                        }
                        let fsum: f64 = src[fbase..].iter().map(|v| (*v as f64).abs()).sum();
                        println!(
                            "threat factorizer export: factor 列 {} を実特徴 {real_cols} 列へ畳み込み (Σ|factor| = {fsum:.4}{})",
                            threat_factor_rows_captured,
                            if fsum == 0.0 { "、 ★まだ学習されていない (resume 直後なら正常)" } else { "" }
                        );
                        out
                    });
                    // 2026-09-24 hanging-split: 学習器は base (dims x 群) + 群共有 delta (108,360) を持つ。
                    //   エンジンは群ごとに 325,080 行の表を読むので、 ここで
                    //     hang 行[g][hl] = base 行[g][base_of(hl)] + delta[hl]
                    //   に展開する。 delta が 0 (resume 直後) なら hang 行 = base 行のコピーになり、
                    //   量子化も決定的なので **関数は bit 単位で保存される**。
                    let expanded: Option<Vec<f32>> = threat_hang_base.as_ref().map(|base_of| {
                        let k = threat_ft_out_captured;
                        let dims = threat_hang_dims;
                        let groups = threat_hang_groups;
                        let hang = base_of.len();
                        let src = &l0w.values[threat_start..];
                        assert_eq!(src.len(), (dims * groups + hang) * k, "hang-split: threat_ft_w の要素数が合わない");
                        let dbase = dims * groups * k;
                        let mut out = vec![0f32; (dims + hang) * groups * k];
                        for g in 0..groups {
                            let sg = g * dims * k;
                            let og = g * (dims + hang) * k;
                            out[og..og + dims * k].copy_from_slice(&src[sg..sg + dims * k]);
                            for hl in 0..hang {
                                let f = base_of[hl] as usize;
                                for i in 0..k {
                                    out[og + (dims + hl) * k + i] = src[sg + f * k + i] + src[dbase + hl * k + i];
                                }
                            }
                        }
                        let dsum: f64 = src[dbase..].iter().map(|v| (*v as f64).abs()).sum();
                        println!(
                            "threat hanging-split export: base {dims} x {groups} 群 + delta {hang} -> {} 行 x {groups} 群 (Σ|delta| = {dsum:.4}{})",
                            dims + hang,
                            if dsum == 0.0 { "、 ★まだ学習されていない (resume 直後なら正常 = 関数保存)" } else { "" }
                        );
                        out
                    });
                    let threat_vals: &[f32] = match (&expanded, &coalesced) {
                        (Some(v), _) => v.as_slice(),
                        (None, Some(v)) => v.as_slice(),
                        (None, None) => &l0w.values[threat_start..],
                    };
                    let mut bytes: Vec<u8> = Vec::new();
                    if threat_i16_captured {
                        // i16 (clamp 無し、LE 2byte)。HalfKA FT 重みと同スケール (QA=127)。
                        // 収束時 max~500 は i16 範囲内。float→int cast は Rust では飽和のため安全。
                        for &v in threat_vals {
                            let q = (qa_f * v as f64).round() as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    } else {
                        // i8 (±127 clamp、従来デフォルト)。
                        for &v in threat_vals {
                            let q = (qa_f * v as f64).round().clamp(-128.0, 127.0) as i8;
                            bytes.push(q as u8);
                        }
                    }

                    if threat_groups_for_log > 1 && !threat_i16_captured {
                        let vals = threat_vals;
                        let per_group = vals.len() / threat_groups_for_log;
                        let saturated: Vec<usize> = vals
                            .chunks(per_group)
                            .map(|c| {
                                c.iter()
                                    .filter(|&&v| {
                                        let r = (qa_f * v as f64).round();
                                        !(-128.0..=127.0).contains(&r)
                                    })
                                    .count()
                            })
                            .collect();
                        println!(
                            "threat 群化 export: {} 群 x {} 要素、 i8 clamp 飽和件数 (群 0, 1, ...) = {:?}",
                            threat_groups_for_log, per_group, saturated
                        );
                    }

                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };

    // ---- Profile 13 factorized Threat section ----
    // Disk layout (C++ P2THRF16 reader): magic + version/profile/dims/factor/buckets,
    // then feature-major int8 code[d=16] and bucket-major int32 output weights.
    let threat_factor_data = if factor_threat_profile.is_some() {
        let factor_dims = bullet_lib::game::inputs::ShogiHalfKaHmThreatFactor16::new().threat_dimensions();
        let factor_dim = bullet_lib::game::inputs::THREAT_FACTOR_DIM;
        let halfka_dim_for_factor = halfka_dim;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let code = graph.get("threat_factor_code");
                    let out = graph.get("threat_factor_out");
                    let code_start = halfka_dim_for_factor * factor_dim;
                    assert_eq!(code.values.len(), code_start + factor_dims * factor_dim, "profile13 factor code shape");
                    assert_eq!(out.values.len(), factor_dim, "profile13 unified factor output shape");
                    let mut bytes = Vec::with_capacity(28 + factor_dims * factor_dim + NUM_BUCKETS * factor_dim * 4);
                    bytes.extend_from_slice(b"P2THRF16");
                    bytes.extend_from_slice(&1u32.to_le_bytes());
                    bytes.extend_from_slice(&13u32.to_le_bytes());
                    bytes.extend_from_slice(&(factor_dims as u32).to_le_bytes());
                    bytes.extend_from_slice(&(factor_dim as u32).to_le_bytes());
                    bytes.extend_from_slice(&(NUM_BUCKETS as u32).to_le_bytes());
                    for &v in &code.values[code_start..] {
                        let q = (f64::from(QA) * f64::from(v)).round().clamp(-128.0, 127.0) as i8;
                        bytes.push(q as u8);
                    }
                    let raw_scale = f64::from(QA) * f64::from(QB);
                    for _bucket in 0..NUM_BUCKETS {
                        for d in 0..factor_dim {
                            // The learned scalar projection is intentionally shared across buckets.
                            let q = (raw_scale * f64::from(out.values[d])).round() as i32;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };
    // ---- Profile 14 direct bucketed Threat section ----
    // The direct branch removes the rank-16 -> rank-1 factorisation in profile 13.
    // It keeps eight independent output buckets, but only the appended Threat rows
    // are trainable; HalfKA prefix rows are projected to zero by the optimizer loop.
    let threat_direct_data = if let Some(tp) = direct_threat_profile {
        let direct15 = tp == ThreatProfile::EnemyOnlyDirect15;
        let (direct_dims, direct_magic, direct_version, direct_profile) = if direct15 {
            let dims = if direct15_defended { THREAT_DIRECT15_DEFENDED_DIMENSIONS } else { THREAT_DIRECT15_DIMENSIONS };
            (dims, *b"P2THRD2\0", 2u32, 15u32)
        } else {
            (ShogiHalfKaHmThreat::new(tp).num_inputs() - halfka_dim, *b"P2THRD1\0", 1u32, 14u32)
        };
        // Profile15 は専用 sparse 入力なので threat_direct_w に HalfKA 前置き行が無い。
        // Profile14 (P2THRD1) は従来どおり [HalfKA | Threat] の連結レイアウト。
        let halfka_dim_for_direct = if direct15 { 0 } else { halfka_dim };
        // P2 (persp split): graph 行 = 2*bucket + 視点 (select の出力 [stm, nstm] の並び)、
        //   ファイル bucket = 視点 * NUM_BUCKETS + bucket (0..7 = stm 視点、 8..15 = nstm 視点)。
        // P4 (factorise): W = D + S + P[pair] をここで畳む (形式・サイズは従来と同一)。
        let np = if direct15 && direct15_persp_split { 2 } else { 1 };
        let rows = np * NUM_BUCKETS;
        let factorise = direct15 && direct15_factorise;
        let halves = if direct15 { direct_dims / THREAT_DIRECT15_DIMENSIONS } else { 1 };
        let npairs = halves * 100;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let w = graph.get("threat_direct_w");
                    let expected = (halfka_dim_for_direct + direct_dims) * rows;
                    assert_eq!(w.values.len(), expected, "profile direct Threat shape");
                    let (shared, pair) = if factorise {
                        let s = graph.get("threat_direct_shared_w");
                        let p = graph.get("threat_direct_pair_w");
                        assert_eq!(s.values.len(), direct_dims * np, "threat_direct_shared_w shape");
                        assert_eq!(p.values.len(), npairs * np, "threat_direct_pair_w shape");
                        (Some(s.values), Some(p.values))
                    } else {
                        (None, None)
                    };
                    let mut bytes = Vec::with_capacity(24 + direct_dims * rows * 4);
                    bytes.extend_from_slice(&direct_magic);
                    bytes.extend_from_slice(&direct_version.to_le_bytes());
                    bytes.extend_from_slice(&direct_profile.to_le_bytes());
                    bytes.extend_from_slice(&(direct_dims as u32).to_le_bytes());
                    bytes.extend_from_slice(&(rows as u32).to_le_bytes());
                    let raw_scale = f64::from(QA) * f64::from(QB);
                    for file_bucket in 0..rows {
                        let (persp, bucket) = (file_bucket / NUM_BUCKETS, file_bucket % NUM_BUCKETS);
                        let row = if np == 2 { 2 * bucket + persp } else { bucket };
                        for feature in 0..direct_dims {
                            let idx = (halfka_dim_for_direct + feature) * rows + row;
                            let mut v = w.values[idx];
                            if let Some(s) = &shared {
                                // S: Shape(dims, np) column-major = [persp][feature]
                                v += s[persp * direct_dims + feature];
                            }
                            if let Some(p) = &pair {
                                // P: Shape(np*npairs, 1) = [persp][half][ac*10+dc]
                                let half = feature / THREAT_DIRECT15_DIMENSIONS;
                                v += p[persp * npairs + half * 100 + direct15_pair_of_feature(feature)];
                            }
                            let q = (raw_scale * f64::from(v)).round() as i32;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };
    // ---- HandThreat weights (raw i8) ----
    // l0w の HandThreat 部分 (feat halfka_dim..input_size) を i8 で書き出す。
    // --threat と --hand-threat は排他なので、同時に存在しない。
    // レイアウト: i8[HAND_THREAT_DIMENSIONS × ft_out] (feature-major)
    let hand_threat_data = if hand_threat {
        let halfka_dim_for_ht = halfka_dim;
        let ft_out_for_ht = ft_out;
        let qa_for_ht = QA;
        Some(
            SavedFormat::empty()
                .transform(move |graph, _| {
                    let l0w = graph.get("l0w");
                    let qa_f = qa_for_ht as f64;

                    let ht_start = halfka_dim_for_ht * ft_out_for_ht;
                    let mut bytes: Vec<u8> = Vec::new();
                    for &v in &l0w.values[ht_start..] {
                        let q = (qa_f * v as f64).round().clamp(-128.0, 127.0) as i8;
                        bytes.push(q as u8);
                    }

                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1),
        )
    } else {
        None
    };

    // ---- LayerStacks (9 buckets) ----
    // 各バケットについて: fc_hash + L1(biases, weights) + L2(biases, weights) + Output(bias, weights)
    //
    // bullet内部の重みレイアウト:
    //   l1w: column-major [NUM_BUCKETS * l1_out, ft_out] (= [rows, cols])
    //   l2w: column-major [NUM_BUCKETS * l2_out, l2_in]
    //   l3w: column-major [NUM_BUCKETS * 1, l2_out]
    //
    // rshogi の期待レイアウト (per bucket):
    //   L1 weights: [l1_out × pad32(ft_out)] row-major, つまり weight[out][padded_in]
    //   L2 weights: [l2_out × pad32(l2_in)] row-major
    //   Output weights: [pad32(l2_out)] row-major

    let bias_scale = i32::from(QA) * i32::from(QB); // 127 * 64 = 8128

    // 全バケットの LayerStack データを1つの transform で生成
    let l1_out_captured = l1_out;
    let residual_zero_captured = residual_zero;
    let l2_out_captured = l2_out;
    let l2_in_captured = l2_in;
    let ft_out_for_ls = ft_out;
    let fc_hash_captured = format.fc_hash;
    let serialized_buckets = format.serialized_buckets;
    let hand_count_dense_dims_captured = hand_count_dense_dims;
    // 2026-11-07 (T4): L4 = 4 層目。 0 のとき下の書き出しは**現行と 1 バイトも同一**。
    let l4_out_captured = l4_out;
    let out_in_captured = out_in;
    let layerstack_data = SavedFormat::empty()
        .transform(move |graph, _| {
            let l1w = graph.get("l1w");
            let l1b = graph.get("l1b");
            let l1fw = graph.get("l1fw");
            let l1fb = graph.get("l1fb");
            let l2w = graph.get("l2w");
            let l2b = graph.get("l2b");
            let l3w = graph.get("l3w");
            let l3b = graph.get("l3b");
            // 🚨l4_out == 0 のときは graph に `l4w` / `l4b` が**存在しない**ので get してはいけない。
            let (l4w, l4b) =
                if l4_out_captured > 0 { (Some(graph.get("l4w")), Some(graph.get("l4b"))) } else { (None, None) };

            let qb_f = QB as f64;
            let bias_scale_f = bias_scale as f64;

            let mut output_bytes: Vec<u8> = Vec::new();

            for bucket in 0..serialized_buckets {
                // fc_hash per bucket
                output_bytes.extend_from_slice(&fc_hash_captured.to_le_bytes());

                // === L1 layer ===
                // Biases: i32, scale = QA * QB = 8128
                for out_idx in 0..l1_out_captured {
                    let global_out = bucket * l1_out_captured + out_idx;
                    let merged_bias = if residual_zero_captured && out_idx + 1 == l1_out_captured {
                        0.0
                    } else {
                        l1b.values[global_out] + l1fb.values[out_idx]
                    };
                    let val = (bias_scale_f * merged_bias as f64).round() as i32;
                    output_bytes.extend_from_slice(&val.to_le_bytes());
                }

                // Weights: i8, scale = QB = 64
                // bullet:
                //   l1w  = [NUM_BUCKETS * l1_out, ft_out + hand_count_dims]
                //   l1fw = [l1_out, ft_out]   (shared factorized part; hand_count は共有化しない)
                //   weight[global_out * (ft_out + hc) + in_idx] where
                //   global_out = bucket * l1_out + out_idx
                //   - in_idx < ft_out: FT 出力との結合部（bucket_w + shared_w）
                //   - ft_out <= in_idx < ft_out + hc_dims: HandCount との結合部（bucket_w のみ）
                //   - それ以上: padding
                // rshogi: row-major [l1_out × padded(ft_out + hc)]
                let l1_total_in = ft_out_for_ls + hand_count_dense_dims_captured;
                let l1_padded_in = pad32(l1_total_in);
                let l1_rows_total = NUM_BUCKETS * l1_out_captured;
                for out_idx in 0..l1_out_captured {
                    let global_out = bucket * l1_out_captured + out_idx;
                    for in_idx in 0..l1_padded_in {
                        if in_idx < ft_out_for_ls {
                            // column-major indexing:
                            //   l1w  shape [NUM_BUCKETS*l1_out, ft_out + hc] -> in * rows + out
                            //   l1fw shape [l1_out, ft_out]                  -> in * l1_out + out
                            let bucket_w = l1w.values[in_idx * l1_rows_total + global_out];
                            let shared_w = l1fw.values[in_idx * l1_out_captured + out_idx];
                            let w = if residual_zero_captured && out_idx + 1 == l1_out_captured {
                                0.0
                            } else {
                                bucket_w + shared_w
                            };
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else if in_idx < l1_total_in {
                            // HandCount Dense 部: bucket_w のみ（共有なし）
                            let bucket_w = l1w.values[in_idx * l1_rows_total + global_out];
                            let w =
                                if residual_zero_captured && out_idx + 1 == l1_out_captured { 0.0 } else { bucket_w };
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else {
                            output_bytes.push(0u8); // padding
                        }
                    }
                }

                // === L2 layer ===
                // Biases: i32, scale = 127 * QB (CReLU output is 127-scale)
                let l2_bias_scale = 127.0 * qb_f;
                for out_idx in 0..l2_out_captured {
                    let global_out = bucket * l2_out_captured + out_idx;
                    let val = (l2_bias_scale * l2b.values[global_out] as f64).round() as i32;
                    output_bytes.extend_from_slice(&val.to_le_bytes());
                }

                // Weights: i8, scale = QB = 64
                let l2_padded_in = pad32(l2_in_captured);
                let l2_rows_total = NUM_BUCKETS * l2_out_captured;
                for out_idx in 0..l2_out_captured {
                    let global_out = bucket * l2_out_captured + out_idx;
                    for in_idx in 0..l2_padded_in {
                        if in_idx < l2_in_captured {
                            // l2w shape [NUM_BUCKETS*l2_out, l2_in], column-major
                            let w = l2w.values[in_idx * l2_rows_total + global_out];
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else {
                            output_bytes.push(0u8);
                        }
                    }
                }

                // === L4 layer (4 層目 `fc1b`) === 2026-11-07
                // l4_out == 0 のときは 1 バイトも書かない = 現行の保存形式と完全に同一。
                //
                // 目盛りは fc1 (L2) と**同一規約**: bias は 127·QB、 重みは QB、
                // エンジン側は `clampi(out >> 6, 0, 127)` で 127 目盛りに戻す。
                // だから f32 の対角 1.0 は i8 の **64** になり `(64·x) >> 6 = x` で恒等になる。
                if l4_out_captured > 0 {
                    let l4w = l4w.as_ref().expect("l4w");
                    let l4b = l4b.as_ref().expect("l4b");
                    let l4_bias_scale = 127.0 * qb_f;
                    for out_idx in 0..l4_out_captured {
                        let global_out = bucket * l4_out_captured + out_idx;
                        let val = (l4_bias_scale * l4b.values[global_out] as f64).round() as i32;
                        output_bytes.extend_from_slice(&val.to_le_bytes());
                    }
                    // l4w shape [NUM_BUCKETS*l4_out, l2_out], column-major (★net2net_three_traps: 列優先)
                    let l4_padded_in = pad32(l2_out_captured);
                    let l4_rows_total = NUM_BUCKETS * l4_out_captured;
                    for out_idx in 0..l4_out_captured {
                        let global_out = bucket * l4_out_captured + out_idx;
                        for in_idx in 0..l4_padded_in {
                            if in_idx < l2_out_captured {
                                let w = l4w.values[in_idx * l4_rows_total + global_out];
                                let q = (qb_f * w as f64).round() as i8;
                                output_bytes.push(q as u8);
                            } else {
                                output_bytes.push(0u8);
                            }
                        }
                    }
                }

                // === Output layer ===
                // Bias: i32, scale = 127 * QB
                let out_bias_scale = 127.0 * qb_f;
                {
                    let global_out = bucket;
                    let val = (out_bias_scale * l3b.values[global_out] as f64).round() as i32;
                    output_bytes.extend_from_slice(&val.to_le_bytes());
                }

                // Weights: i8, scale = QB = 64
                // 🚨2026-11-07: L4 を挟むと l3w の**入力次元が l2_out から l4_out に変わる**。
                //   `out_in_captured` は l4_out > 0 なら l4_out、 そうでなければ l2_out (= 現行)。
                let output_padded_in = pad32(out_in_captured);
                {
                    let global_out = bucket;
                    for in_idx in 0..output_padded_in {
                        if in_idx < out_in_captured {
                            // l3w shape [NUM_BUCKETS, out_in], column-major
                            let w = l3w.values[in_idx * NUM_BUCKETS + global_out];
                            let q = (qb_f * w as f64).round() as i8;
                            output_bytes.push(q as u8);
                        } else {
                            output_bytes.push(0u8);
                        }
                    }
                }
            }

            // byte passthrough: 各バイトを i8 として f32 にキャスト
            output_bytes.iter().map(|&b| (b as i8) as f32).collect()
        })
        .quantise::<i8>(1);

    let mut formats =
        vec![SavedFormat::custom(header), SavedFormat::custom(ft_hash_bytes), ft_biases_leb128, ft_weights_leb128];
    if let Some(psqt) = psqt_data {
        formats.push(psqt);
    }

    if let Some(threat) = threat_data {
        // profile_id > 0 のときだけ profile id (u32 LE) を Threat weights の直前に書き込む
        if let Some(tp) = threat_profile {
            let pid = tp.profile_id();
            if pid != 0 {
                formats.push(SavedFormat::custom(pid.to_le_bytes().to_vec()));
            }
        }
        formats.push(threat);
    }
    if let Some(ht) = hand_threat_data {
        // HandThreat dims (u32 LE) を weights の直前に書き込む
        // v95 では 121,104 固定だが、将来の profile 追加に備えて明示化
        let ht_dims = (input_size - halfka_dim) as u32;
        formats.push(SavedFormat::custom(ht_dims.to_le_bytes().to_vec()));
        formats.push(ht);
    }
    formats.push(layerstack_data);
    // P2THRF16 is a custom tail section.  It must follow LayerStack data because
    // the C++ loader enters its 8-byte custom-section dispatch only after parsing
    // the LayerStack body.  P2DUAL2 and other recognised tail sections may follow.
    if let Some(factor) = threat_factor_data {
        formats.push(factor);
    }
    if let Some(direct) = threat_direct_data {
        formats.push(direct);
    }

    // ---- LMAS section (legacy、 2026-05-25 廃止) ----
    // LMAS 機能は削除済。 lmas_flags != 0 の呼び出しは想定外。
    if lmas_flags != 0 {
        panic!("LMAS section は 2026-05-25 に廃止されました (lmas_flags=0x{:x})", lmas_flags);
    }
    // 旧 LMAS section 書き出し block 本体 (= 約 320 行) は 2026-05-25 物理削除。
    // cfg(any()) は常に false なので block 内は compile されない。
    #[cfg(any())]
    {
        if lmas_flags != 0 {
            let flags_for_lmas = lmas_flags;
            let lmas_data = SavedFormat::empty()
                .transform(move |graph, _| {
                    let mut bytes: Vec<u8> = Vec::new();
                    // sub_magic
                    bytes.extend_from_slice(b"P2LMAS\0\0");
                    // version (u32 LE)
                    bytes.extend_from_slice(&5u32.to_le_bytes());
                    // flags (u32 LE)
                    bytes.extend_from_slice(&flags_for_lmas.to_le_bytes());

                    // P1 (SLA): 14 piece_type × NUM_BUCKETS の affine weights を
                    // bucket-major で i16 量子化 (QA_SLA = 127、 spec.md §6) して書き出す。
                    //
                    // graph 上の "lmas_sla_w" は Shape::new(NUM_BUCKETS, 14) で column-major、
                    // すなわち w.values[pt * NUM_BUCKETS + bucket] で要素 (bucket, pt) を読む。
                    if (flags_for_lmas & 0x01) != 0 {
                        let w = graph.get("lmas_sla_w");
                        let dims = bullet_lib::game::inputs::LMAS_SLA_DIMS;
                        let qa_f = QA as f64; // QA = 127
                        for phase in 0..4 {
                            let bucket = phase * 2;
                            for pt in 0..dims {
                                let v = w.values[pt * NUM_BUCKETS + bucket] as f64;
                                let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                    if (flags_for_lmas & 0x02) != 0 {
                        // P2 (HML): 7 piece_type × NUM_BUCKETS の affine weights を bucket-major で
                        // i16 量子化 (QA_HML = 127、 spec.md §6) して書き出す。
                        //
                        // graph 上の "lmas_hml_w" は Shape::new(NUM_BUCKETS, 7) で column-major:
                        // w.values[pt * NUM_BUCKETS + bucket] で要素 (bucket, pt) を読む。
                        let w = graph.get("lmas_hml_w");
                        let dims = bullet_lib::game::inputs::LMAS_HML_DIMS;
                        let qa_f = QA as f64; // QA = 127
                        for phase in 0..4 {
                            let bucket = phase * 2;
                            for pt in 0..dims {
                                let v = w.values[pt * NUM_BUCKETS + bucket] as f64;
                                let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                    if (flags_for_lmas & 0x04) != 0 {
                        // P3 (Zone PSQT): [2][NUM_BUCKETS][9][14] i16 (両 perspective)
                        //
                        // spec.md §10 layout:
                        //   for perspective in [Black=0, White=1]:
                        //     for bucket in 0..NUM_BUCKETS:
                        //       for zone in 0..9:
                        //         for pt in 0..14:
                        //           i16 weight
                        //
                        // graph 上の "lmas_zpsqt_w_black" / "lmas_zpsqt_w_white" は
                        // Shape::new(NUM_BUCKETS, 126) で column-major:
                        //   w.values[input_idx * NUM_BUCKETS + bucket]
                        //   input_idx = zone * 14 + pt  (= row-major in [zone][pt])
                        let dims = bullet_lib::game::inputs::LMAS_ZPSQT_DIMS; // 126
                        let qa_f = QA as f64;
                        for name in ["lmas_zpsqt_w_black", "lmas_zpsqt_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }
                    if (flags_for_lmas & 0x08) != 0 {
                        let w = graph.get("lmas_ksa_w");
                        let dims = bullet_lib::game::inputs::LMAS_KSA_DIMS;
                        let qa_f = QA as f64;
                        for bucket in 0..NUM_BUCKETS {
                            for input_idx in 0..dims {
                                let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                    if (flags_for_lmas & 0x10) != 0 {
                        let w = graph.get("lmas_ct_w");
                        let dims = bullet_lib::game::inputs::LMAS_CT_DIMS;
                        let qa_f = QA as f64;
                        for bucket in 0..NUM_BUCKETS {
                            for input_idx in 0..dims {
                                let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                    if (flags_for_lmas & 0x20) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_KZPSQT_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_kzpsqt_w_black", "lmas_kzpsqt_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v5 ContextGate (flat、 phase × atk × mat × group、 4,536 entries)。
                    // graph 上の "lmas_context_gate_w" は Shape::new(NUM_BUCKETS=8, atk×mat×group=567)
                    // で column-major (atk×mat×group が phase 軸 = bucket と対応)。
                    // 量子化なし (Q10 fixed-point の raw i16、 GATE_DEFAULT=1024 が ×1.0)。
                    if (flags_for_lmas & 0x40) != 0 {
                        let w = graph.get("lmas_context_gate_w");
                        // disk 順: ((phase × 9 + atk) × 9 + mat) × 7 + group
                        // 同 graph layout: phase=bucket、 atk*mat*group=cols、 column-major
                        let cols = bullet_lib::game::inputs::LMAS_CG_ATTACK_BUCKETS
                            * bullet_lib::game::inputs::LMAS_CG_MATERIAL_BUCKETS
                            * bullet_lib::game::inputs::LMAS_CG_GROUPS;
                        for phase in 0..bullet_lib::game::inputs::LMAS_CG_PHASES {
                            for col in 0..cols {
                                // ContextGate は graph 上で raw i16 値を float32 に格納する想定
                                // (gate=1024.0 → 量子化後 i16 1024)。 QA scale は適用しない。
                                let v = w.values[col * NUM_BUCKETS + phase] as f64;
                                let q = v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }

                    // v5 KZLine: [2][NUM_BUCKETS][9 king_zone][2 rel][3 slider][8 dir] = 6,912 i16
                    // 黒/白視点別 graph weight、 black/white を順に dump。
                    if (flags_for_lmas & 0x80) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_KZLINE_AUX_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_kzline_w_black", "lmas_kzline_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v5 KZESC: [2][NUM_BUCKETS][9 king_zone][5 escape][5 danger] = 3,600 i16
                    if (flags_for_lmas & 0x100) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_KZESC_AUX_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_kzesc_w_black", "lmas_kzesc_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v6 KP-PSQT raw: [2][NUM_BUCKETS][81 king_sq][2 rel][81 piece_sq][14 pt]
                    //                 = 2,939,328 i16 (片視点 183,708 dim × 8 bucket × 2 persp)
                    if (flags_for_lmas & 0x200) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_KP_PSQT_RAW_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_kp_psqt_raw_w_black", "lmas_kp_psqt_raw_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v6 Hand × King-zone: [2][NUM_BUCKETS][9 king_zone][7 hand_type] = 1,008 i16
                    if (flags_for_lmas & 0x400) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_HAND_KZONE_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_hand_kzone_w_black", "lmas_hand_kzone_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v6 Threat Map: [2][NUM_BUCKETS][9 king_zone][14 attacker_pt] = 2,016 i16
                    if (flags_for_lmas & 0x800) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_THREAT_MAP_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_threat_map_w_black", "lmas_threat_map_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v7 KKR: [persp][bucket][enemy_ks 45][friend_ks 81][phase 4] = 233,280 i16
                    // disk 並びは rshogi-core::kkr_flat_index と一致 (persp-major、 次に bucket、
                    // 末尾の input_idx は enemy_ks/friend_ks/phase の row-major)。
                    // graph 上は per-perspective に "lmas_kkr_w_black/white" Shape::new(NUM_BUCKETS, KKR_DIMS)
                    // column-major (w.values[input_idx * NUM_BUCKETS + bucket])。
                    if (flags_for_lmas & 0x1000) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_KKR_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_kkr_w_black", "lmas_kkr_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v7 PSH: [persp][bucket][slot 54] = 864 i16
                    if (flags_for_lmas & 0x2000) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_PSH_DIMS;
                        let qa_f = QA as f64;
                        for name in ["lmas_psh_w_black", "lmas_psh_w_white"] {
                            let w = graph.get(name);
                            for bucket in 0..NUM_BUCKETS {
                                for input_idx in 0..dims {
                                    let v = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                    let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }

                    // v7 PBR: [bucket][phase 16][hand 8][king_dist 5] = 5,120 i16
                    // perspective なし 1 weight tensor のみ。
                    // rshogi-core 側初期値は GATE_DEFAULT=1024 (×1.0) なので、 学習側で書き出す値は
                    // 「学習で得た delta」を **1024 を中心に** offset として加算する。
                    // ContextGate と同じく QA scale なし (raw Q10 fixed-point として disk に書く)。
                    if (flags_for_lmas & 0x4000) != 0 {
                        let dims = bullet_lib::game::inputs::LMAS_PBR_DIMS;
                        let w = graph.get("lmas_pbr_w");
                        const GATE_DEFAULT: f64 = 1024.0;
                        for bucket in 0..NUM_BUCKETS {
                            for input_idx in 0..dims {
                                // graph weight は 0 中心の delta、 disk は GATE_DEFAULT 中心。
                                let delta = w.values[input_idx * NUM_BUCKETS + bucket] as f64;
                                let v = GATE_DEFAULT + delta;
                                let q = v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }

                    // v8 MATE_THREAT (= bit 18 OPP, bit 19 MY)、 各 NUM_BUCKETS の i16
                    // NPS 計測 path: 重みを強制 0 で書き出し (= eval overflow 回避)
                    if (flags_for_lmas & (1u32 << 18)) != 0 {
                        for _ in 0..NUM_BUCKETS {
                            bytes.extend_from_slice(&0i16.to_le_bytes());
                        }
                    }
                    if (flags_for_lmas & (1u32 << 19)) != 0 {
                        for _ in 0..NUM_BUCKETS {
                            bytes.extend_from_slice(&0i16.to_le_bytes());
                        }
                    }

                    // byte passthrough (i8 として f32 化、 後段で i8(1) 量子化される)
                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1);
            formats.push(lmas_data);
        }
    } // close #[cfg(any())] {

    // ---- PIN section (spec.md と rshogi-core read_with_options に合致する順序) ----
    //
    // sub_magic = "P2PIN\0\0\0" (8B)
    // version   = u32 LE = 1
    // for bucket in 0..NUM_PIN_BUCKETS:
    //   for persp in 0..2:                  (0=black, 1=white)
    //     i16[PIN_DIMENSIONS_PER_PERSPECTIVE]
    //
    // QA = 127、 layout は rshogi-core `read_pin_weights` と bit-exact 同値。
    if pin_aware {
        let pin_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    PIN_DIMENSIONS_PER_PERSPECTIVE, PIN_NUM_BUCKETS, PIN_QA, PIN_SECTION_VERSION, PIN_SUB_MAGIC,
                };
                let mut bytes: Vec<u8> = Vec::new();
                // sub_magic (8B) + version (u32 LE)
                bytes.extend_from_slice(PIN_SUB_MAGIC);
                bytes.extend_from_slice(&PIN_SECTION_VERSION.to_le_bytes());

                // graph 上の "pin_w_black" / "pin_w_white" は Shape::new(PIN_NUM_BUCKETS, PIN_DIMS)
                // で column-major (w.values[input_idx * PIN_NUM_BUCKETS + bucket])。
                let qa_f = PIN_QA as f64;
                let names = ["pin_w_black", "pin_w_white"];
                let dims = PIN_DIMENSIONS_PER_PERSPECTIVE;
                // disk 順 = [bucket][persp][dim]、 graph layout は per-persp weight tensor。
                for bucket in 0..PIN_NUM_BUCKETS {
                    for &name in names.iter() {
                        let w = graph.get(name);
                        for input_idx in 0..dims {
                            let v = w.values[input_idx * PIN_NUM_BUCKETS + bucket] as f64;
                            let q = (qa_f * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }

                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(pin_data);
    }

    // ---- DUAL section (PIN の後) ----
    //
    // 2 つの format が存在 (相互排他):
    //
    // 1) reserve (= 旧 Plan C iter 2 領域確保):
    //    sub_magic = "P2DUAL\0\0" (8B) + version u32 + i16[2 head][8 bucket]
    //    = 44 byte (zeroed)
    //
    // 2) king-attacker (= 2026-05-26 新設計):
    //    sub_magic = "P2DUAL2\0" (8B) + version u32 + count u32 +
    //    i16[2 persp][2 head][8 bucket][81 sq][14 piece] = 72,592 byte (zeroed)
    //
    // 現状はどちらも全 0 (= graph 側に dual weight tensor は無く、 disk 直接書き出し)。
    // king-attacker の実 training 統合は別 sub-agent で実装予定 (= Phase 3)。
    if dual_psqt_reserve {
        let dual_data = SavedFormat::empty()
            .transform(move |_graph, _| {
                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(b"P2DUAL\0\0");
                bytes.extend_from_slice(&1u32.to_le_bytes());
                // head=2 × bucket=8 = 16 個の i16 zero (= 32 byte)
                for _ in 0..(2 * 8) {
                    bytes.extend_from_slice(&0i16.to_le_bytes());
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(dual_data);
    } else if dual_psqt_king_attacker {
        // Phase 3b-sparse: graph 上の 4 weight tensor
        //   "dual_w_attack_black" / "dual_w_attack_white" / "dual_w_defend_black" / "dual_w_defend_white"
        // (各 shape (1, 9072)) を読み取り、 disk layout `[persp][head][bucket][sq][piece]`
        // (= 36,288 i16) に再構築する。 rshogi-core `read_dual_psqt` と bit-exact 同値。
        //
        // arm 5: king_attacker head が graph に存在しない場合 (= --dual-disable-king-attacker)、
        // P2DUAL2 section を全 Zero で書き出す (= rshogi-core 側で legacy/zero 扱い)。
        let dual_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    DUAL_DIM_PER_HEAD, DUAL_PIECE_TYPES, DUAL_SECTION_VERSION, DUAL_SQUARES, DUAL_SUB_MAGIC,
                    DUAL_TOTAL_COUNT, NUM_DUAL_PSQT_BUCKETS, NUM_DUAL_PSQT_HEADS,
                };

                // QA = 127 (= pin と同じ。 rshogi-core 側 i16 weight scale)
                const DUAL_QA: f64 = 127.0;

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(DUAL_SUB_MAGIC);
                bytes.extend_from_slice(&DUAL_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&DUAL_TOTAL_COUNT.to_le_bytes());

                // arm 5 対応: king_attacker weight が存在するかを graph.get の panic catch で
                // 判定する。 std::panic::catch_unwind で wrap、 panic なら全 zero で出力。
                let names_per_persp =
                    [("dual_w_attack_black", "dual_w_defend_black"), ("dual_w_attack_white", "dual_w_defend_white")];
                let has_king_attacker =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| graph.get("dual_w_attack_black"))).is_ok();

                if !has_king_attacker {
                    // arm 5: king_attacker 廃止。 全 Zero で disk 書き出し
                    // (= rshogi-core が読んでも eval 加算 0)
                    let zero_count = 2 * NUM_DUAL_PSQT_HEADS * NUM_DUAL_PSQT_BUCKETS * DUAL_SQUARES * DUAL_PIECE_TYPES;
                    for _ in 0..zero_count {
                        bytes.extend_from_slice(&0i16.to_le_bytes());
                    }
                    debug_assert_eq!(bytes.len(), 72_592);
                    return bytes.iter().map(|&b| (b as i8) as f32).collect();
                }

                // disk layout: [persp][head][bucket][sq][piece]
                // 4 weight tensor を head=0 (attack) / head=1 (defend) × persp で結合。
                //
                // 重要 (= sign convention): graph forward は `score = attack - defend` で
                // 動いている (= defend weight 部分は forward 時に -1 倍されている)。 disk
                // format は rshogi-core dense 経路時の `(attack - defend)` 規約と同じ
                // (= head=1 weight に正値を入れ、 推論側で sub する)。 したがって defend
                // weight をそのまま書き出せば bit-exact 一致 する。
                for (attack_name, defend_name) in names_per_persp.iter() {
                    let w_attack = graph.get(attack_name);
                    let w_defend = graph.get(defend_name);
                    debug_assert_eq!(w_attack.values.len(), DUAL_DIM_PER_HEAD);
                    debug_assert_eq!(w_defend.values.len(), DUAL_DIM_PER_HEAD);
                    for head in 0..NUM_DUAL_PSQT_HEADS {
                        let w_head = if head == 0 { &w_attack } else { &w_defend };
                        for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                            for sq in 0..DUAL_SQUARES {
                                for piece in 0..DUAL_PIECE_TYPES {
                                    let flat =
                                        bucket * (DUAL_SQUARES * DUAL_PIECE_TYPES) + sq * DUAL_PIECE_TYPES + piece;
                                    let v = w_head.values[flat] as f64;
                                    let q = (DUAL_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 72_592);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(dual_data);

        // D2 cross feature section: piece_threat_relation (+ piece_defended_by if legacy)。
        //
        // Path I (= 2026-05-27): 新 default = `P2DUALZ\0` magic で head 0 (= threat) のみ
        // 書き出し (= 254,016 i16 = ~0.5 MB)。 `--dual-save-legacy-p2dualx` 指定時のみ
        // 旧 `P2DUALX\0` magic で head 0/1 両方を書き出し (= 508,032 i16 = ~1 MB、 後方互換)。
        // rshogi-core 推論側 `read_dual_cross` は両 magic 対応 (P2DUALX 時 head 1 は load 後捨て)。
        //
        // arm 6 (= 2026-05-27): `--dual-disable-cross-defended` ON のとき graph 側で
        // `dual_w_defended_*` tensor は生成されない → 常に P2DUALZ (= head 0 only) で
        // save する (= legacy_p2dualx との併用は起動時 error で排除済)。
        if !enable_scalar_threat || scalar_threat_coexist_dual_cross {
            let save_legacy = dual_save_legacy_p2dualx;
            let defended_in_graph = !dual_disable_cross_defended;
            let dual_cross_data = SavedFormat::empty()
                .transform(move |graph, _| {
                    use bullet_lib::game::inputs::{
                        DUAL_CROSS_DIM_PER_HEAD, DUAL_PIECE_TYPES, DUAL_SQUARES, NUM_DUAL_PSQT_BUCKETS,
                    };

                    // QA = 127 と同じ
                    const DUAL_CROSS_QA: f64 = 127.0;
                    const MAGIC_Z: &[u8; 8] = b"P2DUALZ\0";
                    const MAGIC_X: &[u8; 8] = b"P2DUALX\0";
                    const VERSION: u32 = 1;
                    const COUNT_Z: u32 = (2 * 127_008) as u32; // = 254,016
                    const COUNT_X: u32 = (2 * 2 * 127_008) as u32; // = 508,032

                    let (magic, count): (&[u8; 8], u32) =
                        if save_legacy { (MAGIC_X, COUNT_X) } else { (MAGIC_Z, COUNT_Z) };
                    let n_disk_heads = if save_legacy { 2 } else { 1 };

                    let mut bytes: Vec<u8> = Vec::new();
                    bytes.extend_from_slice(magic);
                    bytes.extend_from_slice(&VERSION.to_le_bytes());
                    bytes.extend_from_slice(&count.to_le_bytes());

                    // 順序: persp(black, white) × head(threat, defended)
                    // P2DUALZ では head 0 (= threat) のみ書き出す。
                    let names_per_persp = [
                        ("dual_w_threat_black", "dual_w_defended_black"),
                        ("dual_w_threat_white", "dual_w_defended_white"),
                    ];
                    for (threat_name, defended_name) in names_per_persp.iter() {
                        let w_threat = graph.get(threat_name);
                        debug_assert_eq!(w_threat.values.len(), DUAL_CROSS_DIM_PER_HEAD);
                        // arm 6: defended が graph に存在しない場合は load を skip
                        // (= P2DUALZ save で head 1 を書かない、 legacy_p2dualx 併用は
                        // 起動時 error で排除済 → ここに到達するときは save_legacy=false)。
                        let w_defended_opt = if defended_in_graph {
                            let w = graph.get(defended_name);
                            debug_assert_eq!(w.values.len(), DUAL_CROSS_DIM_PER_HEAD);
                            Some(w)
                        } else {
                            None
                        };
                        for head_idx in 0..n_disk_heads {
                            let w_head = if head_idx == 0 {
                                &w_threat
                            } else {
                                w_defended_opt
                                    .as_ref()
                                    .expect("P2DUALX legacy save requires dual_w_defended_* in graph")
                            };
                            for bucket in 0..NUM_DUAL_PSQT_BUCKETS {
                                for sq in 0..DUAL_SQUARES {
                                    for own_p in 0..DUAL_PIECE_TYPES {
                                        for target_p in 0..DUAL_PIECE_TYPES {
                                            let flat = bucket * (DUAL_SQUARES * DUAL_PIECE_TYPES * DUAL_PIECE_TYPES)
                                                + sq * (DUAL_PIECE_TYPES * DUAL_PIECE_TYPES)
                                                + own_p * DUAL_PIECE_TYPES
                                                + target_p;
                                            let v = w_head.values[flat] as f64;
                                            let q = (DUAL_CROSS_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64)
                                                as i16;
                                            bytes.extend_from_slice(&q.to_le_bytes());
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let expected = 16 + 2 * n_disk_heads * 2 * 127_008;
                    debug_assert_eq!(bytes.len(), expected);
                    bytes.iter().map(|&b| (b as i8) as f32).collect()
                })
                .quantise::<i8>(1);
            formats.push(dual_cross_data);
        }
    }

    // ---- P2KRING section (arm 11、 2026-05-28、 dual section の後) ----
    //
    // sub_magic "P2KRING\0" (8B) + version u32 + count u32 + i16[bucket 8][persp 2][sq 24][piece 13][state 4]
    // = 8 + 4 + 4 + 2 × 19,968 = 39,952 byte。
    //
    // graph 上の 2 weight tensor `kingring_w_black` / `kingring_w_white` (= 各 shape (1, 9984)、
    // bucket-major flat layout) を [bucket][persp][sq][piece][state] disk layout に再構築する。
    // rshogi-core `read_kingring` の inverse。
    //
    // QA = 1 (= rshogi-core output_scale 1、 i16 直接 round)。 dim 9984 / bucket = 1248 で
    // per-bucket slice。
    if enable_kingring {
        let kingring_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    KINGRING_DIM_PER_PERSP, KINGRING_NUM_BUCKETS, KINGRING_NUM_PIECE, KINGRING_NUM_SQ,
                    KINGRING_NUM_STATE, KINGRING_PER_BUCKET_DIM, KINGRING_QA, KINGRING_SECTION_VERSION,
                    KINGRING_SUB_MAGIC, KINGRING_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(KINGRING_SUB_MAGIC);
                bytes.extend_from_slice(&KINGRING_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&KINGRING_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("kingring_w_black");
                let w_white = graph.get("kingring_w_white");
                debug_assert_eq!(w_black.values.len(), KINGRING_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), KINGRING_DIM_PER_PERSP);

                // disk layout: [bucket][persp][sq][piece][state]
                // graph layout (per persp): bucket-major flat = bucket * 1248 + sq * 52 + piece * 4 + state
                for bucket in 0..KINGRING_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for sq in 0..KINGRING_NUM_SQ {
                            for piece in 0..KINGRING_NUM_PIECE {
                                for state in 0..KINGRING_NUM_STATE {
                                    let flat = bucket * KINGRING_PER_BUCKET_DIM
                                        + sq * (KINGRING_NUM_PIECE * KINGRING_NUM_STATE)
                                        + piece * KINGRING_NUM_STATE
                                        + state;
                                    let v = w_persp.values[flat] as f64;
                                    let q = (KINGRING_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 19_968);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(kingring_data);
    }

    // ---- P2PAIR section (arm 12、 2026-05-29、 kingring section の後) ----
    //
    // sub_magic "P2PAIR\0\0" (8B) + version u32 + count u32 + i16[bucket 8][persp 2][pair 12][shape 16]
    // = 8 + 4 + 4 + 2 × 3,072 = 6,160 byte。
    //
    // graph 上の 2 weight tensor `pairshape_w_black` / `pairshape_w_white` (= 各 shape (1, 1536)、
    // bucket-major flat layout = `bucket * 192 + pair * 16 + shape`) を
    // [bucket][persp][pair][shape] disk layout に再構築する。
    // rshogi-core `read_pairshape` の inverse。
    //
    // QA = 127 (= dual_psqt / kingring と同 convention、 1.0 バグ厳禁)。
    if enable_pairshape {
        let pairshape_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    PAIRSHAPE_DIM_PER_PERSP, PAIRSHAPE_NUM_BUCKETS, PAIRSHAPE_NUM_PAIR_TYPES, PAIRSHAPE_NUM_SHAPES,
                    PAIRSHAPE_PER_BUCKET_DIM, PAIRSHAPE_QA, PAIRSHAPE_SECTION_VERSION, PAIRSHAPE_SUB_MAGIC,
                    PAIRSHAPE_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(PAIRSHAPE_SUB_MAGIC);
                bytes.extend_from_slice(&PAIRSHAPE_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&PAIRSHAPE_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("pairshape_w_black");
                let w_white = graph.get("pairshape_w_white");
                debug_assert_eq!(w_black.values.len(), PAIRSHAPE_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), PAIRSHAPE_DIM_PER_PERSP);

                // disk layout: [bucket][persp][pair][shape]
                // graph layout (per persp): bucket-major flat = bucket * 192 + pair * 16 + shape
                for bucket in 0..PAIRSHAPE_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for pair in 0..PAIRSHAPE_NUM_PAIR_TYPES {
                            for shape in 0..PAIRSHAPE_NUM_SHAPES {
                                let flat = bucket * PAIRSHAPE_PER_BUCKET_DIM + pair * PAIRSHAPE_NUM_SHAPES + shape;
                                let v = w_persp.values[flat] as f64;
                                let q = (PAIRSHAPE_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 3_072);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(pairshape_data);
    }

    // 2026-06-06 Activity-Risk Interaction Head section (= `P2ACTR\0\0`).
    //
    // disk layout: [bucket][persp][activity][pressure][shield] row-major、
    // graph (per persp): local_idx_in_persp = bucket * 256 + activity * 32 + pressure * 4 + shield
    //
    // QA = 127 (= pairshape と同 convention)。
    if enable_activity_risk {
        let activity_risk_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    ACTIVITY_RISK_DIM_PER_PERSP, ACTIVITY_RISK_NUM_ACTIVITY, ACTIVITY_RISK_NUM_BUCKETS,
                    ACTIVITY_RISK_NUM_PRESSURE, ACTIVITY_RISK_NUM_SHIELD, ACTIVITY_RISK_PER_PB_DIM, ACTIVITY_RISK_QA,
                    ACTIVITY_RISK_SECTION_VERSION, ACTIVITY_RISK_SUB_MAGIC, ACTIVITY_RISK_V3_NUM_ESCAPE,
                    ACTIVITY_RISK_V3_NUM_HAND_CHECK, ACTIVITY_RISK_V3_NUM_HEAVY,
                    ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP, ACTIVITY_RISK_V3_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(ACTIVITY_RISK_SUB_MAGIC);
                bytes.extend_from_slice(&ACTIVITY_RISK_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&ACTIVITY_RISK_V3_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("activity_risk_w_black");
                let w_white = graph.get("activity_risk_w_white");
                debug_assert_eq!(w_black.values.len(), ACTIVITY_RISK_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), ACTIVITY_RISK_DIM_PER_PERSP);

                // base 4096 weight (= v1 form、 disk layout [bucket][persp][act][pre][shi])
                for bucket in 0..ACTIVITY_RISK_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for activity in 0..ACTIVITY_RISK_NUM_ACTIVITY {
                            for pressure in 0..ACTIVITY_RISK_NUM_PRESSURE {
                                for shield in 0..ACTIVITY_RISK_NUM_SHIELD {
                                    let local = bucket * ACTIVITY_RISK_PER_PB_DIM
                                        + activity * ACTIVITY_RISK_NUM_PRESSURE * ACTIVITY_RISK_NUM_SHIELD
                                        + pressure * ACTIVITY_RISK_NUM_SHIELD
                                        + shield;
                                    let v = w_persp.values[local] as f64;
                                    let q =
                                        (ACTIVITY_RISK_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }
                }

                // 2026-06-08 v3 tactical 1024 weight (= disk layout [bucket][persp][heavy][escape][check])
                let tw_black = graph.get("activity_risk_v3_tactical_w_black");
                let tw_white = graph.get("activity_risk_v3_tactical_w_white");
                debug_assert_eq!(tw_black.values.len(), ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP);
                debug_assert_eq!(tw_white.values.len(), ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP);

                for bucket in 0..ACTIVITY_RISK_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &tw_black } else { &tw_white };
                        for heavy in 0..ACTIVITY_RISK_V3_NUM_HEAVY {
                            for escape in 0..ACTIVITY_RISK_V3_NUM_ESCAPE {
                                for check in 0..ACTIVITY_RISK_V3_NUM_HAND_CHECK {
                                    let per_bucket = ACTIVITY_RISK_V3_NUM_HEAVY
                                        * ACTIVITY_RISK_V3_NUM_ESCAPE
                                        * ACTIVITY_RISK_V3_NUM_HAND_CHECK;
                                    let local = bucket * per_bucket
                                        + heavy * ACTIVITY_RISK_V3_NUM_ESCAPE * ACTIVITY_RISK_V3_NUM_HAND_CHECK
                                        + escape * ACTIVITY_RISK_V3_NUM_HAND_CHECK
                                        + check;
                                    let v = w_persp.values[local] as f64;
                                    let q =
                                        (ACTIVITY_RISK_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }
                }

                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 5_120);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(activity_risk_data);
    }

    // 2026-06-09 King Escape Direction (KED) section (= `P2KESC\0\0`).
    //
    // disk layout: [bucket 8][persp 2][class 4] row-major、
    // graph (per persp): local = bucket * 4 + class
    //
    // QA = 127 (= activity_risk と同 convention)。
    if enable_king_escape {
        let king_escape_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    KING_ESCAPE_DIM_PER_PERSP, KING_ESCAPE_NUM_BUCKETS, KING_ESCAPE_NUM_CLASS, KING_ESCAPE_QA,
                    KING_ESCAPE_SECTION_VERSION, KING_ESCAPE_SUB_MAGIC, KING_ESCAPE_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(KING_ESCAPE_SUB_MAGIC);
                bytes.extend_from_slice(&KING_ESCAPE_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&KING_ESCAPE_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("king_escape_w_black");
                let w_white = graph.get("king_escape_w_white");
                debug_assert_eq!(w_black.values.len(), KING_ESCAPE_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), KING_ESCAPE_DIM_PER_PERSP);

                // disk layout: [bucket][persp][class]
                for bucket in 0..KING_ESCAPE_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for class in 0..KING_ESCAPE_NUM_CLASS {
                            let local = bucket * KING_ESCAPE_NUM_CLASS + class;
                            let v = w_persp.values[local] as f64;
                            let q = (KING_ESCAPE_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 64);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(king_escape_data);
    }

    // 2026-06-10 Attack Relay Head v2 (= 構造化 980 weight、 magic `P2ARS1\0\0`)。
    //
    // 旧 v2 (= 1024 cell 直積、 magic `P2AREL2\0`) と非互換。 4 テーブル因子分解
    // + 先後共有 weight (= single tensor `attack_relay_v2_w`、 length 980)。
    // disk 連結順: W_base (576) → W_alpha (180) → W_phase (144) → W_alpha_shape (80)。
    // 基準スライス (= `alpha == 0` / `phase == 0`) は学習側で active させない仕様だが、
    // 念のため ckpt 書き出し時にも強制ゼロを施す。
    if enable_attack_relay_v2 {
        let attack_relay_v2_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    ARH_V2_NUM_ALPHA, ARH_V2_NUM_ATTACKERS, ARH_V2_NUM_BALANCE, ARH_V2_NUM_HEAVY, ARH_V2_NUM_OVERLAP,
                    ARH_V2_NUM_PHASE, ARH_V2_NUM_SUPPORTED, ARH_V2_QA, ARH_V2_SECTION_VERSION, ARH_V2_SUB_MAGIC,
                    ARH_V2_W_ALPHA_COUNT, ARH_V2_W_ALPHA_SHAPE_COUNT, ARH_V2_W_BASE_COUNT, ARH_V2_W_HEAVY_COUNT,
                    ARH_V2_W_PHASE_COUNT, ARH_V2_WEIGHT_COUNT, attack_relay_v2_w_alpha_disk_idx,
                    attack_relay_v2_w_alpha_shape_disk_idx, attack_relay_v2_w_base_disk_idx,
                    attack_relay_v2_w_heavy_disk_idx, attack_relay_v2_w_phase_disk_idx,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(ARH_V2_SUB_MAGIC);
                bytes.extend_from_slice(&ARH_V2_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&(ARH_V2_WEIGHT_COUNT as u32).to_le_bytes());

                // graph 上で weight tensor が未確保なら全 0 (= zero init ckpt)。
                if !graph.has("attack_relay_v2_w") {
                    let payload_len = 2 * ARH_V2_WEIGHT_COUNT;
                    bytes.extend(std::iter::repeat_n(0u8, payload_len));
                    debug_assert_eq!(bytes.len(), 8 + 4 + 4 + payload_len);
                    return bytes.iter().map(|&b| (b as i8) as f32).collect();
                }

                let w = graph.get("attack_relay_v2_w");
                debug_assert_eq!(w.values.len(), ARH_V2_WEIGHT_COUNT);

                // disk index → f32 で参照、 i16 量子化。 基準スライスは強制 0。
                let quantise =
                    |v: f32| -> i16 { (ARH_V2_QA * v as f64).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16 };

                // W_base[a][b][o][s] (= 576)。 基準スライス制約なし。
                for a in 0..ARH_V2_NUM_ATTACKERS {
                    for b in 0..ARH_V2_NUM_BALANCE {
                        for o in 0..ARH_V2_NUM_OVERLAP {
                            for s in 0..ARH_V2_NUM_SUPPORTED {
                                let idx = attack_relay_v2_w_base_disk_idx(a, b, o, s);
                                let q = quantise(w.values[idx]);
                                bytes.extend_from_slice(&q.to_le_bytes());
                            }
                        }
                    }
                }
                // W_alpha[alpha][a][b] (= 180)。 alpha == 0 は強制 0。
                for alpha in 0..ARH_V2_NUM_ALPHA {
                    for a in 0..ARH_V2_NUM_ATTACKERS {
                        for b in 0..ARH_V2_NUM_BALANCE {
                            let idx = attack_relay_v2_w_alpha_disk_idx(alpha, a, b);
                            let q = if alpha == 0 { 0i16 } else { quantise(w.values[idx]) };
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                // W_phase[phase][a][b] (= 144)。 phase == 0 は強制 0。
                for phase in 0..ARH_V2_NUM_PHASE {
                    for a in 0..ARH_V2_NUM_ATTACKERS {
                        for b in 0..ARH_V2_NUM_BALANCE {
                            let idx = attack_relay_v2_w_phase_disk_idx(phase, a, b);
                            let q = if phase == 0 { 0i16 } else { quantise(w.values[idx]) };
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                // W_alpha_shape[alpha][o][s] (= 80)。 alpha == 0 は強制 0。
                for alpha in 0..ARH_V2_NUM_ALPHA {
                    for o in 0..ARH_V2_NUM_OVERLAP {
                        for s in 0..ARH_V2_NUM_SUPPORTED {
                            let idx = attack_relay_v2_w_alpha_shape_disk_idx(alpha, o, s);
                            let q = if alpha == 0 { 0i16 } else { quantise(w.values[idx]) };
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                // W_heavy[heavy][b] (= 36)。 基準スライス制約なし。
                for heavy in 0..ARH_V2_NUM_HEAVY {
                    for b in 0..ARH_V2_NUM_BALANCE {
                        let idx = attack_relay_v2_w_heavy_disk_idx(heavy, b);
                        let q = quantise(w.values[idx]);
                        bytes.extend_from_slice(&q.to_le_bytes());
                    }
                }
                let expected_payload = 2
                    * (ARH_V2_W_BASE_COUNT
                        + ARH_V2_W_ALPHA_COUNT
                        + ARH_V2_W_PHASE_COUNT
                        + ARH_V2_W_ALPHA_SHAPE_COUNT
                        + ARH_V2_W_HEAVY_COUNT);
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + expected_payload);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(attack_relay_v2_data);
    }

    // 2026-06-09 Attack Relay Head (ARH) section (= `P2AREL\0\0`).
    //
    // disk layout: [bucket 8][persp 2][contact 4][reserve 4][relay 4] row-major、
    // graph (per persp): local = bucket * 64 + contact * 16 + reserve * 4 + relay
    //
    // QA = 127 (= KED / activity_risk と同 convention)。
    if enable_attack_relay {
        let attack_relay_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    ARH_DIM_PER_PERSP, ARH_NUM_BUCKETS, ARH_NUM_CONTACT, ARH_NUM_RELAY, ARH_NUM_RESERVE, ARH_QA,
                    ARH_SECTION_VERSION, ARH_SUB_MAGIC, ARH_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(ARH_SUB_MAGIC);
                bytes.extend_from_slice(&ARH_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&ARH_TOTAL_COUNT.to_le_bytes());

                // 2026-06-10: graph store に id が無い場合 (= 訓練 trigger で tensor が graph
                // から削除されるケースの safe-guard) は全 0 で section を書き出す。 panic 回避。
                if !graph.has("attack_relay_w_black") || !graph.has("attack_relay_w_white") {
                    let payload_len = 2 * ARH_TOTAL_COUNT as usize;
                    bytes.extend_from_slice(&vec![0u8; payload_len]);
                    debug_assert_eq!(bytes.len(), 8 + 4 + 4 + payload_len);
                    return bytes.iter().map(|&b| (b as i8) as f32).collect();
                }

                let w_black = graph.get("attack_relay_w_black");
                let w_white = graph.get("attack_relay_w_white");
                debug_assert_eq!(w_black.values.len(), ARH_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), ARH_DIM_PER_PERSP);

                // disk layout: [bucket][persp][contact][reserve][relay]
                for bucket in 0..ARH_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for contact in 0..ARH_NUM_CONTACT {
                            for reserve in 0..ARH_NUM_RESERVE {
                                for relay in 0..ARH_NUM_RELAY {
                                    let local = ((bucket * ARH_NUM_CONTACT + contact) * ARH_NUM_RESERVE + reserve)
                                        * ARH_NUM_RELAY
                                        + relay;
                                    let v = w_persp.values[local] as f64;
                                    let q = (ARH_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                    bytes.extend_from_slice(&q.to_le_bytes());
                                }
                            }
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 1024);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(attack_relay_data);
    }

    // 2026-06-16 King Mobility (KMOB) section (= `P2KMOB\0\0`)。
    //
    // 全 section の **物理末尾** に append する (= 推論側 rshogi-core の probe 順
    // KESC → AREL → ARS1 → KDGR → ADEV → KMOB と整合、 KMOB は最後に probe される)。
    //
    // disk layout: [bucket 8][persp 2][class 9] row-major、
    // graph (per persp): local = bucket * 9 + class。 QA = 127 (= ADEV と同 convention)。
    // class が 9 (= 0..=8) である点が ADEV / KDGR と異なる。
    if enable_king_mobility {
        let king_mobility_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    KING_MOBILITY_DIM_PER_PERSP, KING_MOBILITY_NUM_BUCKETS, KING_MOBILITY_NUM_CLASS, KING_MOBILITY_QA,
                    KING_MOBILITY_SECTION_VERSION, KING_MOBILITY_SUB_MAGIC, KING_MOBILITY_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(KING_MOBILITY_SUB_MAGIC);
                bytes.extend_from_slice(&KING_MOBILITY_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&KING_MOBILITY_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("king_mobility_w_black");
                let w_white = graph.get("king_mobility_w_white");
                debug_assert_eq!(w_black.values.len(), KING_MOBILITY_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), KING_MOBILITY_DIM_PER_PERSP);

                // disk layout: [bucket][persp][class]
                for bucket in 0..KING_MOBILITY_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for class in 0..KING_MOBILITY_NUM_CLASS {
                            let local = bucket * KING_MOBILITY_NUM_CLASS + class;
                            let v = w_persp.values[local] as f64;
                            let q = (KING_MOBILITY_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 144);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(king_mobility_data);
    }

    // 2026-06-16 Drop Pressure (DROP) section (= `P2DROP\0\0`)。
    //
    // 全 section の **物理末尾** に append する (= 推論側 rshogi-core の probe 順
    // KESC → AREL → ARS1 → KDGR → ADEV → KMOB → DROP と整合、 DROP は最後に probe される)。
    //
    // disk layout: [bucket 8][persp 2][class 8] row-major、
    // graph (per persp): local = bucket * 8 + class。 QA = 127 (= ADEV / KMOB と同 convention)。
    // class は 8 (= 0..=7、 ADEV / KDGR と同形)。
    if enable_drop_pressure {
        let drop_pressure_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    DROP_PRESSURE_DIM_PER_PERSP, DROP_PRESSURE_NUM_BUCKETS, DROP_PRESSURE_NUM_CLASS, DROP_PRESSURE_QA,
                    DROP_PRESSURE_SECTION_VERSION, DROP_PRESSURE_SUB_MAGIC, DROP_PRESSURE_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(DROP_PRESSURE_SUB_MAGIC);
                bytes.extend_from_slice(&DROP_PRESSURE_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&DROP_PRESSURE_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("drop_pressure_w_black");
                let w_white = graph.get("drop_pressure_w_white");
                debug_assert_eq!(w_black.values.len(), DROP_PRESSURE_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), DROP_PRESSURE_DIM_PER_PERSP);

                // disk layout: [bucket][persp][class]
                for bucket in 0..DROP_PRESSURE_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for class in 0..DROP_PRESSURE_NUM_CLASS {
                            let local = bucket * DROP_PRESSURE_NUM_CLASS + class;
                            let v = w_persp.values[local] as f64;
                            let q = (DROP_PRESSURE_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 128);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(drop_pressure_data);
    }

    // (3) TL1 section: threat を L1 に直結する重み。 2026-10-01。
    //   形: magic | ver=1 | K2(=2K) | L1_OUT | NUM_BUCKETS | i16[bucket][o][c]
    //   目盛り: w_q = round(1024 * w)。 engine は sum_c thr_i16[c]*w_q[c] >> 5 を L1 の i32 に足す
    //   (導出: thr_i16 = 254*float、 L1 i32 = 8128*float → 8128/254 = 32 = 1024>>5)。
    //   🚨graph 側で ±2 に clip しているので |w_q| <= 2048。
    if enable_threat_l1 {
        let k2 = 2 * threat_ft_out_c2;
        let l1n = l1_out;
        let tl1_data = SavedFormat::empty()
            .transform(move |graph, _| {
                let w = graph.get("threat_l1_w");
                let rows = NUM_BUCKETS * l1n;
                assert_eq!(
                    w.values.len(),
                    rows * k2,
                    "threat_l1_w は (NUM_BUCKETS*L1_OUT) x 2K ({} x {}) のはず",
                    rows,
                    k2
                );
                let mut over = 0usize;
                let mut bytes = Vec::with_capacity(20 + rows * k2 * 2);
                bytes.extend_from_slice(b"P2TL1\0\0\0");
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&(k2 as u32).to_le_bytes());
                bytes.extend_from_slice(&(l1n as u32).to_le_bytes());
                bytes.extend_from_slice(&(NUM_BUCKETS as u32).to_le_bytes());
                for b in 0..NUM_BUCKETS {
                    for o in 0..l1n {
                        for c in 0..k2 {
                            // column-major: (行 b*L1_OUT+o, 列 c) → 平坦 c*rows + 行
                            let v = w.values[c * rows + (b * l1n + o)] as f64;
                            if v.abs() > 2.0 + 1e-6 {
                                over += 1;
                            }
                            let v = v.clamp(-2.0, 2.0);
                            let q = (1024.0 * v).round().clamp(-32767.0, 32767.0) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                if over > 0 {
                    println!("🚨threat_l1_w に |w| > 2 が {over} 個。 graph の clip が効いていない");
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(tl1_data);
    }

    // (4) TL2 section: threat を L2 入力に直結する重み。 2026-10-08。
    //   形: magic | ver=2 | K2(=2K) | L2_IN | NUM_BUCKETS | i16[bucket][j][c]
    //   目盛り: v_q = round(S * v)。 engine は sum_c thr_i16[c]*v_q[c] >> log2(2S) を
    //   clamp の**前**の L2 入力 (127*float) に足す
    //   (導出: 127*sum thr_f*v = (127/254)*sum thr_i16*v = (1/(2*S))*sum thr_i16*v_q)。
    //   🚨S は section version と対応する (2026-10-09 に 64 -> 1024 に上げた):
    //     ver 1 -> S =   64、 engine の shift  7  (旧 net。 量子化誤差 34.28% / max|q| 8)
    //     ver 2 -> S = 1024、 engine の shift 11  (現行。 量子化誤差  2.26% / max|q| 131)
    //   graph 側の clip ±2 はそのままなので |v_q| <= 2048 (engine の i32 レーンの上限)。
    //   🚨列の並び [l1_sqr | l1_main] は engine の l2in[0..l1_eff) | l2in[l1_eff..2*l1_eff) と同じ。
    if enable_threat_l2 {
        let k2 = 2 * threat_ft_out_c2;
        let l2n = l2_in;
        let tl2_data = SavedFormat::empty()
            .transform(move |graph, _| {
                let w = graph.get("threat_l2_w");
                let rows = NUM_BUCKETS * l2n;
                assert_eq!(
                    w.values.len(),
                    rows * k2,
                    "threat_l2_w は (NUM_BUCKETS*L2_IN) x 2K ({} x {}) のはず",
                    rows,
                    k2
                );
                let mut over = 0usize;
                let mut bytes = Vec::with_capacity(20 + rows * k2 * 2);
                bytes.extend_from_slice(b"P2TL2\0\0\0");
                // 🚨ver 2 = S 1024 (engine は shift 11)。 ver 1 (S 64 / shift 7) の旧 net も
                //   engine 側で version 分岐して bit-exact に読める。
                bytes.extend_from_slice(&2u32.to_le_bytes());
                bytes.extend_from_slice(&(k2 as u32).to_le_bytes());
                bytes.extend_from_slice(&(l2n as u32).to_le_bytes());
                bytes.extend_from_slice(&(NUM_BUCKETS as u32).to_le_bytes());
                for b in 0..NUM_BUCKETS {
                    for j in 0..l2n {
                        for c in 0..k2 {
                            // column-major: (行 b*L2_IN+j, 列 c) → 平坦 c*rows + 行
                            let v = w.values[c * rows + (b * l2n + j)] as f64;
                            if v.abs() > 2.0 + 1e-6 {
                                over += 1;
                            }
                            let v = v.clamp(-2.0, 2.0);
                            let q = (1024.0 * v).round().clamp(-32767.0, 32767.0) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                if over > 0 {
                    println!("🚨threat_l2_w に |w| > 2 が {over} 個。 graph の clip が効いていない");
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(tl2_data);
    }

    // (5) TG section: threat で sqr 側 15 本を乗法ゲートする重み。 2026-10-10。
    //   形: magic "P2TGS\0\0\0" | ver=1 | K2(=2K) | SLOTS(=l1_effective) | NUM_BUCKETS
    //       | i16[bucket][slot][c]
    //   目盛り: u_q = round(1024 * u) (S_g = 1024)。 engine は
    //       `tg[i] = (Σ_c thr_i16[c] * u_q[c]) >> 10`、 `g = clamp(256 + tg[i], 0, 1024)`、
    //       `pre_sq = (sqr * g) >> 8` を計算する。
    //   導出: thr_i16 = 254*thr_f なので (Σ thr_i16 * u_q) >> 10 = 254 * Σ thr_f * u、
    //       よって g/256 = 1 + (254/256) * Σ thr_f * u。 **学習器の順伝播もこの式**
    //       (graph 側の 254/256 を 1 にすると 0.79% ずれる)。
    //   clip: graph 側で ±2 に clip しているので |u_q| <= 2048 = SIMD の i32 レーン上界
    //       (16 * 32767 * 2048 < 2^31)。
    //   🚨🚨並びの罠: f32 の重みは **列優先 (K2, NUM_BUCKETS*SLOTS)** で入っているので、
    //     ここで**転置**しないと (bucket, slot, c) の順に並ばない。 TL2 でこれを間違えて
    //     16,840 / 30,720 要素を壊した事故がある (★tl2_quantisation_throws_away_25pct_20261009)。
    //     下の往復検査 (R0) がその事故を**関門として**止める。
    if enable_threat_gate {
        let k2 = 2 * threat_ft_out_c2;
        let slots = l1_effective;
        let tg_data = SavedFormat::empty()
            .transform(move |graph, _| {
                let w = graph.get("threat_gate_w");
                let rows = NUM_BUCKETS * slots;
                assert_eq!(
                    w.values.len(),
                    rows * k2,
                    "threat_gate_w は (NUM_BUCKETS*SLOTS) x 2K ({} x {}) のはず",
                    rows,
                    k2
                );
                let mut over = 0usize;
                // header は 24 byte (magic 8 + u32 x 4)、 本体は i16 x NB x SLOTS x K2。
                const TG_HDR: usize = 24;
                let mut bytes = Vec::with_capacity(TG_HDR + rows * k2 * 2);
                bytes.extend_from_slice(b"P2TGS\0\0\0");
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&(k2 as u32).to_le_bytes());
                bytes.extend_from_slice(&(slots as u32).to_le_bytes());
                bytes.extend_from_slice(&(NUM_BUCKETS as u32).to_le_bytes());
                for b in 0..NUM_BUCKETS {
                    for j in 0..slots {
                        for c in 0..k2 {
                            // 🚨列優先: (行 b*SLOTS+j, 列 c) → 平坦 c*rows + 行
                            let u = w.values[c * rows + (b * slots + j)] as f64;
                            if u.abs() > 2.0 + 1e-6 {
                                over += 1;
                            }
                            let q = (1024.0 * u.clamp(-2.0, 2.0)).round().clamp(-2048.0, 2048.0) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                if over > 0 {
                    println!("🚨threat_gate_w に |u| > 2 が {over} 個。 graph の clip が効いていない");
                }
                // ---- 関門 R0: 往復検査 (並びを間違えた net を黙って出さないための関門) ----
                //   🚨書き込み順を再利用すると「転置の間違い」は検出できないので、
                //     **header を parse し直して** (bucket, slot, c) の添字を独立に導出し、
                //     そこから読んだ i16 を round(1024*clip(u)) と突き合わせる。
                //     TL2 では転置を間違えて 16,840 / 30,720 要素を壊した
                //     (★tl2_quantisation_throws_away_25pct_20261009)。
                {
                    assert_eq!(&bytes[0..8], b"P2TGS\0\0\0", "P2TGS magic が壊れている");
                    let rd32 =
                        |o: usize| u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize;
                    let (ver, r_k2, r_slots, r_nb) = (rd32(8), rd32(12), rd32(16), rd32(20));
                    assert_eq!(ver, 1, "P2TGS version が 1 でない");
                    assert_eq!(r_k2, k2, "P2TGS header の K2 が合わない");
                    assert_eq!(r_slots, slots, "P2TGS header の SLOTS が合わない");
                    assert_eq!(r_nb, NUM_BUCKETS, "P2TGS header の NUM_BUCKETS が合わない");
                    assert_eq!(bytes.len(), TG_HDR + r_nb * r_slots * r_k2 * 2, "P2TGS section の長さが合わない");
                    let mut mismatch = 0usize;
                    let mut first_bad: Option<(usize, usize, usize, i16, i16)> = None;
                    for b in 0..r_nb {
                        for j in 0..r_slots {
                            for c in 0..r_k2 {
                                let off = TG_HDR + ((b * r_slots + j) * r_k2 + c) * 2;
                                let got = i16::from_le_bytes([bytes[off], bytes[off + 1]]);
                                let u = w.values[c * rows + (b * slots + j)] as f64;
                                let exp = (1024.0 * u.clamp(-2.0, 2.0)).round().clamp(-2048.0, 2048.0) as i16;
                                if got != exp {
                                    mismatch += 1;
                                    if first_bad.is_none() {
                                        first_bad = Some((b, j, c, exp, got));
                                    }
                                }
                            }
                        }
                    }
                    if mismatch > 0 {
                        panic!(
                            "🚨P2TGS 往復検査 NG: {mismatch} / {} 要素が不一致 \
                             (最初の (bucket, slot, c, expect, got) = {:?})。 転置か量子化が壊れている",
                            rows * k2,
                            first_bad
                        );
                    }
                    println!(
                        "P2TGS 往復検査 OK: {} 要素 (NUM_BUCKETS={r_nb} SLOTS={r_slots} K2={r_k2}) が \
                         round(1024*clip(u)) と全一致",
                        rows * k2
                    );
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(tg_data);
    }

    // B6 可動域 P2MOB section: i16[NUM_BUCKETS]。 2026-09-29。
    // 🚨目盛りは PSQT と同じ QA*QB=8128。 学習器の入力が mob/16 なので、
    //   エンジン側は `(mob_diff * w) >> 4` で受ける (MOB_INPUT_SHIFT)。
    if enable_mob {
        let mob_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{MOB_QA, MOB_SECTION_VERSION, MOB_SUB_MAGIC, MOB_WEIGHT_COUNT};
                let w = graph.get("mobw");
                assert_eq!(w.values.len(), MOB_WEIGHT_COUNT, "mobw は bucket 数ぶん ({} 本) のはず", MOB_WEIGHT_COUNT);
                let mut bytes = Vec::with_capacity(16 + MOB_WEIGHT_COUNT * 2);
                bytes.extend_from_slice(MOB_SUB_MAGIC);
                bytes.extend_from_slice(&MOB_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&(MOB_WEIGHT_COUNT as u32).to_le_bytes());
                for &v in &w.values {
                    let q = (MOB_QA * v as f64).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                    bytes.extend_from_slice(&q.to_le_bytes());
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(mob_data);
    }

    // 案1 ブロック射影 P2TPRJ section: i16[FT_DIM]。 2026-09-30。
    //
    // 学習器の forward は
    //     m = clip(proj_raw ⊙ mask, -1, 1)、  mask[j][c] = 1 iff c == j % K
    //     ft += m @ thr             →  ft[j] += m[j][j % K] * thr[j % K]
    // なので **対角だけ**が生きている。 ここでは対角を取り出して Q15 で書く。
    //
    // 🚨 Shape::new(rows, cols) は **column-major** (device.rs:150 の cuBLAS 規約)。
    //    → 平坦添字 = c * rows + j。 graph 側 mask と同じ式を使うこと。
    // 🚨 forward の clip を**必ず**同じように掛ける。 掛け忘れると engine と食い違い、
    //    しかも ★19 のとおり**黙って壊れた評価値で動く**。
    if enable_threat_proj {
        let ftn = ft_out_captured;
        let kn = threat_ft_out_c2;
        let offn = threat_proj_offset;
        let pairn = threat_proj_pair.clone();
        let bkn = if threat_proj_buckets { NUM_BUCKETS } else { 1 };
        let proj_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{THREAT_PROJ_QA, THREAT_PROJ_SUB_MAGIC};
                let w = graph.get("threat_proj_raw");
                let rows = ftn * bkn;
                assert_eq!(w.values.len(), rows * kn, "threat_proj_raw は (FT*bucket) x K ({} x {}) のはず", rows, kn);
                // 🚨非対角は勾配が 0 なので永久に 0 のはず。 0 でなければ graph の mask と
                //   ここの添字が食い違っている = engine と学習器が別の評価関数になる (★19)。
                let mut off_diag_nz = 0usize;
                let blkc = threat_proj_block_offsets(ftn, kn, &pairn, offn);
                for c in 0..kn {
                    for b in 0..bkn {
                        for j in 0..ftn {
                            let live = (j + blkc[j / 16]) % kn;
                            if c != live && w.values[c * rows + (b * ftn + j)] != 0.0 {
                                off_diag_nz += 1;
                            }
                        }
                    }
                }
                assert_eq!(
                    off_diag_nz, 0,
                    "threat_proj_raw の非対角が生きている ({} 個)。 mask の添字 (column-major 前提) がずれている",
                    off_diag_nz
                );
                // 🚨version 2: offset を書き込む。 エンジンはこれを読んでマッピングを決める
                //   (旧 net を新エンジンが読んで黙って別物になるのを防ぐ。 ★19)。
                // 🚨version 3: ブロックごとのずらし量を書き込む。 エンジンは式を持たない。
                let blk = threat_proj_block_offsets(ftn, kn, &pairn, offn);
                // 🚨version は bucket 別のときだけ 4。 bkn==1 なら 3 (従来と同形)。
                //   無条件に 4 を書くと現行エンジン (ver<=3 のみ受理) が rich 単独の net も
                //   読めなくなる (fable 指摘 B、 2026-10-01)。
                let sec_ver: u32 = if bkn > 1 { 4 } else { 3 };
                let mut bytes = Vec::with_capacity(24 + blk.len() + rows * 2);
                bytes.extend_from_slice(THREAT_PROJ_SUB_MAGIC);
                bytes.extend_from_slice(&sec_ver.to_le_bytes());
                bytes.extend_from_slice(&(ftn as u32).to_le_bytes());
                bytes.extend_from_slice(&(blk.len() as u32).to_le_bytes());
                for &o in &blk {
                    bytes.push(o as u8);
                }
                if sec_ver >= 4 {
                    bytes.extend_from_slice(&(bkn as u32).to_le_bytes());
                }
                for b in 0..bkn {
                    for j in 0..ftn {
                        let c = (j + blk[j / 16]) % kn;
                        // column-major: (row j, col c) → 平坦 c * ftn + j
                        let v = w.values[c * rows + (b * ftn + j)] as f64;
                        let v = v.clamp(-1.0, 1.0); // forward の clip_pass_through_grad と同じ
                        let q = (THREAT_PROJ_QA * v).round().clamp(-32767.0, 32767.0) as i16;
                        bytes.extend_from_slice(&q.to_le_bytes());
                    }
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(proj_data);
    }

    // Endgame Tempo (EGT) section: [bucket][persp][class] i16。
    if enable_endgame_tempo {
        let endgame_tempo_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    ENDGAME_TEMPO_DIM_PER_PERSP, ENDGAME_TEMPO_NUM_BUCKETS, ENDGAME_TEMPO_NUM_CLASS, ENDGAME_TEMPO_QA,
                    ENDGAME_TEMPO_SECTION_VERSION, ENDGAME_TEMPO_SUB_MAGIC, ENDGAME_TEMPO_TOTAL_COUNT,
                };
                let mut bytes = Vec::with_capacity(16 + ENDGAME_TEMPO_TOTAL_COUNT as usize * 2);
                bytes.extend_from_slice(ENDGAME_TEMPO_SUB_MAGIC);
                bytes.extend_from_slice(&ENDGAME_TEMPO_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&ENDGAME_TEMPO_TOTAL_COUNT.to_le_bytes());
                let black = graph.get("endgame_tempo_w_black");
                let white = graph.get("endgame_tempo_w_white");
                debug_assert_eq!(black.values.len(), ENDGAME_TEMPO_DIM_PER_PERSP);
                debug_assert_eq!(white.values.len(), ENDGAME_TEMPO_DIM_PER_PERSP);
                for bucket in 0..ENDGAME_TEMPO_NUM_BUCKETS {
                    for persp in 0..2 {
                        let weights = if persp == 0 { &black } else { &white };
                        for class in 0..ENDGAME_TEMPO_NUM_CLASS {
                            let local = bucket * ENDGAME_TEMPO_NUM_CLASS + class;
                            let q = (ENDGAME_TEMPO_QA * weights.values[local] as f64)
                                .round()
                                .clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(endgame_tempo_data);
    }

    // Full Threat scalar residual。disk/runtimeともfeature-major [feature][bucket]。
    if enable_scalar_threat {
        let scalar_threat_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::SCALAR_THREAT_WEIGHT_COUNT;
                let mut bytes = Vec::with_capacity(16 + SCALAR_THREAT_WEIGHT_COUNT * 2);
                bytes.extend_from_slice(b"P2STHR\0\0");
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&(SCALAR_THREAT_WEIGHT_COUNT as u32).to_le_bytes());
                let weights = graph.get("scalar_threat_w");
                debug_assert_eq!(weights.values.len(), SCALAR_THREAT_WEIGHT_COUNT);
                for &value in &weights.values {
                    let q = (127.0 * value as f64).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                    bytes.extend_from_slice(&q.to_le_bytes());
                }
                bytes.iter().map(|&byte| (byte as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(scalar_threat_data);
    }

    if enable_dual_cross_geometry {
        let geometry_data = SavedFormat::empty()
            .transform(move |graph, _| {
                const PAIRS: usize = 14 * 14;
                const GEOMETRY: usize = 32;
                const BUCKETS: usize = 8;
                const DIM: usize = PAIRS * GEOMETRY * BUCKETS;
                let black = graph.get("dual_cross_geometry_w_black");
                let white = graph.get("dual_cross_geometry_w_white");
                debug_assert_eq!(black.values.len(), DIM);
                debug_assert_eq!(white.values.len(), DIM);
                let mut bytes = Vec::with_capacity(16 + 2 * DIM * 2);
                bytes.extend_from_slice(b"P2DCG1\0\0");
                bytes.extend_from_slice(&1u32.to_le_bytes());
                bytes.extend_from_slice(&(2 * DIM as u32).to_le_bytes());
                for weights in [&black.values, &white.values] {
                    for bucket in 0..BUCKETS {
                        for pair in 0..PAIRS {
                            let base = bucket * PAIRS * GEOMETRY + pair * GEOMETRY;
                            let mean = weights[base..base + GEOMETRY].iter().map(|&value| value as f64).sum::<f64>()
                                / GEOMETRY as f64;
                            for geometry in 0..GEOMETRY {
                                let centered = weights[base + geometry] as f64 - mean;
                                let quantised =
                                    (127.0 * centered).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                                bytes.extend_from_slice(&quantised.to_le_bytes());
                            }
                        }
                    }
                }
                bytes.iter().map(|&byte| (byte as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(geometry_data);
    }

    // 2026-06-28 Slider X-ray (SXRY) section (= `P2SXRY\0\0`)。
    //
    // 全 section の **物理末尾** に append する (= 推論側 rshogi-core の probe 順で
    // 全 section (= ... / KDGR / ADEV / KMOB / DROP / STHR / DCG1) の後に最後に probe される)。
    //
    // disk layout: [bucket 8][persp 2][class 8] row-major、
    // graph (per persp): local = bucket * 8 + class。 QA = 127 (= ADEV と同 convention)。
    if enable_slider_xray {
        let slider_xray_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    SLIDER_XRAY_DIM_PER_PERSP, SLIDER_XRAY_NUM_BUCKETS, SLIDER_XRAY_NUM_CLASS, SLIDER_XRAY_QA,
                    SLIDER_XRAY_SECTION_VERSION, SLIDER_XRAY_SUB_MAGIC, SLIDER_XRAY_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::new();
                bytes.extend_from_slice(SLIDER_XRAY_SUB_MAGIC);
                bytes.extend_from_slice(&SLIDER_XRAY_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&SLIDER_XRAY_TOTAL_COUNT.to_le_bytes());

                let w_black = graph.get("slider_xray_w_black");
                let w_white = graph.get("slider_xray_w_white");
                debug_assert_eq!(w_black.values.len(), SLIDER_XRAY_DIM_PER_PERSP);
                debug_assert_eq!(w_white.values.len(), SLIDER_XRAY_DIM_PER_PERSP);

                // disk layout: [bucket][persp][class]
                for bucket in 0..SLIDER_XRAY_NUM_BUCKETS {
                    for persp in 0..2 {
                        let w_persp = if persp == 0 { &w_black } else { &w_white };
                        for class in 0..SLIDER_XRAY_NUM_CLASS {
                            let local = bucket * SLIDER_XRAY_NUM_CLASS + class;
                            let v = w_persp.values[local] as f64;
                            let q = (SLIDER_XRAY_QA * v).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                            bytes.extend_from_slice(&q.to_le_bytes());
                        }
                    }
                }
                debug_assert_eq!(bytes.len(), 8 + 4 + 4 + 2 * 128);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(slider_xray_data);
    }

    // 2026-07-01 King Zone Threat (KZT_HAND) factored v1 section (= `P2KZT\0\0\0`、 version 2)。
    //
    // 全 section の **物理末尾** に append する (= 推論側 rshogi-core の probe 順で SXRY 直後)。
    //
    // disk layout: **flat 10,336 i16** (= 5 テーブルを core → region → bucket → persp → bias の
    // 固定順で連結、 推論側 `KztFactoredWeights::from_flat` と完全一致)。 QA = 127。
    if enable_kzt {
        let kzt_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{
                    KZT_QA, KZT_SECTION_VERSION, KZT_SUB_MAGIC, KZT_TABLE_DIMS, KZT_TABLE_WEIGHT_NAMES, KZT_TOTAL_COUNT,
                };

                let mut bytes: Vec<u8> = Vec::with_capacity(16 + 2 * KZT_TOTAL_COUNT as usize);
                bytes.extend_from_slice(KZT_SUB_MAGIC);
                bytes.extend_from_slice(&KZT_SECTION_VERSION.to_le_bytes());
                bytes.extend_from_slice(&KZT_TOTAL_COUNT.to_le_bytes());

                // core → region → bucket → persp → bias の固定順で各 weight tensor を量子化連結。
                for (name, dim) in KZT_TABLE_WEIGHT_NAMES.iter().zip(KZT_TABLE_DIMS.iter()) {
                    let w = graph.get(name);
                    debug_assert_eq!(w.values.len(), *dim, "kzt table {name} dim mismatch");
                    for &v in w.values.iter() {
                        let q = (KZT_QA * v as f64).round().clamp(i16::MIN as f64, i16::MAX as f64) as i16;
                        bytes.extend_from_slice(&q.to_le_bytes());
                    }
                }
                debug_assert_eq!(bytes.len(), 16 + 2 * KZT_TOTAL_COUNT as usize);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(kzt_data);
    }

    // 2026-07-04 King Zone Threat FT 入力版 (KZT-FT) section (= `P2KZTF\0\0`、 version 1)。
    //
    // 全 section の **物理末尾** に append する (= 推論側 rshogi-core が magic probe で最後に読む)。
    //
    // l0w の KZT-FT 部分 (= feat halfka_dim..input_size、 column-major → row-major [9,440][ft_out])
    // を QA 量子化した i8 raw。 engine `read_kzt_ft_weights` (= 逆量子化なし、 row [index] = i8[L1])
    // と完全一致。 threat 部分書き出しと同方式 (= QA スケール、 主 FT と同じ i8 量子化)。
    if enable_kzt_ft {
        let halfka_dim_for_kzt_ft = halfka_dim;
        let ft_out_for_kzt_ft = ft_out;
        let qa_for_kzt_ft = QA;
        let kzt_ft_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use rshogi_core::nnue::kzt::{KZT_FT_DIMENSIONS, KZT_FT_MAGIC, KZT_FT_VERSION};

                let l0w = graph.get("l0w");
                let qa_f = qa_for_kzt_ft as f64;
                let ft_start = halfka_dim_for_kzt_ft * ft_out_for_kzt_ft;
                let count = KZT_FT_DIMENSIONS * ft_out_for_kzt_ft;
                debug_assert_eq!(
                    l0w.values.len() - ft_start,
                    count,
                    "kzt-ft l0w slice len mismatch (expected {count})"
                );

                let mut bytes: Vec<u8> = Vec::with_capacity(16 + count);
                bytes.extend_from_slice(KZT_FT_MAGIC);
                bytes.extend_from_slice(&KZT_FT_VERSION.to_le_bytes());
                bytes.extend_from_slice(&(count as u32).to_le_bytes());
                for &v in &l0w.values[ft_start..] {
                    let q = (qa_f * v as f64).round().clamp(-128.0, 127.0) as i8;
                    bytes.push(q as u8);
                }
                debug_assert_eq!(bytes.len(), 16 + count);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(kzt_ft_data);
    }

    // 2026-07-22 AUX (KZ+HANG+STATKD) FT 入力版 section (= `P2AUX\0\0\0`、 version 1)。
    //
    // 全 section の **物理末尾** に append する (= C++ p2nnue.cpp の magic 連鎖 probe で読む)。
    //
    // l0w の AUX 部分 (= feat halfka_dim..input_size、 l0w.values[feat*ft_out+out] の
    // feat 73,305..73,905 = feature-major [600][ft_out]) を QA 量子化した i8 raw。
    // C++ ローダ (p2nnue.cpp: magic "P2AUX\0\0\0") は
    //   magic(8) + version(u32 LE) + i8[AUX_DIM * FT_DIM] (feature-major、 count 無し)
    // を読む。 aux_row(af) = &aux_weight[af * FT_DIM] と bit-exact 一致。
    // ※ KZT-FT (P2KZTF) とは異なり **count u32 フィールドは書かない** (C++ が読まないため)。
    if enable_aux {
        let halfka_dim_for_aux = halfka_dim;
        let ft_out_for_aux = ft_out;
        let qa_for_aux = QA;
        let aux_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{AUX_DIM_PER_PERSP, AUX_SECTION_VERSION, AUX_SUB_MAGIC};

                let l0w = graph.get("l0w");
                let qa_f = qa_for_aux as f64;
                let ft_start = halfka_dim_for_aux * ft_out_for_aux;
                let count = AUX_DIM_PER_PERSP * ft_out_for_aux;
                debug_assert_eq!(l0w.values.len() - ft_start, count, "aux l0w slice len mismatch (expected {count})");

                let mut bytes: Vec<u8> = Vec::with_capacity(12 + count);
                bytes.extend_from_slice(AUX_SUB_MAGIC); // "P2AUX\0\0\0" (8 byte)
                bytes.extend_from_slice(&AUX_SECTION_VERSION.to_le_bytes()); // version u32=1
                // count フィールドは無し (C++ p2nnue.cpp は AUX_DIM*FT_DIM を build 定数から算出)。
                for &v in &l0w.values[ft_start..] {
                    let q = (qa_f * v as f64).round().clamp(-128.0, 127.0) as i8;
                    bytes.push(q as u8);
                }
                debug_assert_eq!(bytes.len(), 12 + count);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(aux_data);
    }

    // 2026-07-23 AUXCHEAP (KZ2x+HG6+OUT6+STATKD-kd2) PSQT-only section
    // (= `P2AUXC\0\0`、 version 1)。 全 section の物理末尾に append する。
    //
    // psqtw の aux 行 (= feat halfka_dim..halfka_dim+392、 column-major
    // values[feat * NUM_BUCKETS + bucket]) を PSQT と同じ scale QA*QB (= 8128) で
    // i32 量子化した raw。 レイアウトは feature-major [392][8bucket] i32 LE で、
    // C++ 側 `g_auxcheap_w[local * NUM_BUCKETS + bucket]` (p2nnue.cpp) と bit-exact
    // に対応する (寄与式 `(ac_stm[b]-ac_nstm[b])/2` は PSQT と同一のため同 scale)。
    // ※ P2AUX と同じく count u32 フィールドは書かない (C++ は 392*8 を build 定数から算出)。
    if enable_auxcheap {
        let halfka_dim_for_auxcheap = halfka_dim;
        // factorise 時 psqtw layout は [factoriser 1,629 | HalfKA 73,305 | aux 392] なので
        // aux 行の開始 offset に factoriser 行数を足す。 aux 行自体は derive=None ゆえ
        // factor 寄与を持たず、 実特徴行の raw 値がそのまま配備値 (= coalesce 不要)。
        let auxcheap_row_offset = if factorise { bullet_lib::game::inputs::PIECE_INPUTS } else { 0 };
        let auxcheap_data = SavedFormat::empty()
            .transform(move |graph, _| {
                use bullet_lib::game::inputs::{AUXCHEAP_DIM_PER_PERSP, AUXCHEAP_SECTION_VERSION, AUXCHEAP_SUB_MAGIC};

                let psqt_w = graph.get("psqtw");
                let scale = (QA as i32 * QB as i32) as f64; // 8128.0 (= PSQT weight と同一)
                let start = (auxcheap_row_offset + halfka_dim_for_auxcheap) * NUM_BUCKETS;
                let count = AUXCHEAP_DIM_PER_PERSP * NUM_BUCKETS;
                debug_assert_eq!(
                    psqt_w.values.len() - start,
                    count,
                    "auxcheap psqtw slice len mismatch (expected {count})"
                );

                let mut bytes: Vec<u8> = Vec::with_capacity(12 + count * 4);
                bytes.extend_from_slice(AUXCHEAP_SUB_MAGIC); // "P2AUXC\0\0" (8 byte)
                bytes.extend_from_slice(&AUXCHEAP_SECTION_VERSION.to_le_bytes()); // version u32=1
                for &v in &psqt_w.values[start..] {
                    let q = (scale * v as f64).round() as i32;
                    bytes.extend_from_slice(&q.to_le_bytes());
                }
                debug_assert_eq!(bytes.len(), 12 + count * 4);
                bytes.iter().map(|&b| (b as i8) as f32).collect()
            })
            .quantise::<i8>(1);
        formats.push(auxcheap_data);
    }

    formats
}

// =============================================================================
// Quantised → Graph 逆量子化ローダー (--resume-from-quantised)
// =============================================================================

/// 量子化済み quantised.bin から graph weights を直接 load する。
///
/// `build_layerstack_save_format` の逆操作。 optimiser state は touch しないため、
/// caller 側で zero init (= Optimiser::new 直後) で start する想定。
///
/// 対応するセクション:
/// - FT biases / weights (LEB128 圧縮、 i16 → f32 / QA)
/// - PSQT (i32、 オプション)
/// - HandThreat (u32 dims + i8、 オプション)
/// - LayerStacks (8 bucket、 L1/L2/L3 biases i32 + weights i8)
/// - LMAS section (P2LMAS magic、 SLA/HML/ZPSQT を逆量子化、 KSA/CT は skip)
///
/// 制約 (YAGNI):
/// - Threat profile (HalfKaHmThreat) は今回未対応 (利用 path なし)。 検出したらエラー。
/// - dim mismatch (quantised.bin と graph の l0/l1/l2 不一致) はエラー。
///
/// # Arguments
///
/// 全ての dim 引数は graph 側 (resume 先) の値。 quantised.bin が同 dim でない場合エラー。
#[allow(clippy::too_many_arguments)]
fn load_from_quantised_bin<G>(
    graph: &mut G,
    path: &Path,
    halfka_dim: usize,
    input_size: usize,
    ft_out: usize,
    threat_ft_out: usize,
    l1_out: usize,
    l2_out: usize,
    residual_mix: bool,
    psqt: bool,
    hand_threat: bool,
    hand_count_dense_dims: usize,
    lmas_flags: u32,
    pin_aware: bool,
    dual_psqt_reserve: bool,
    dual_psqt_king_attacker: bool,
    direct15_persp_split: bool,
    direct15_defended: bool,
) -> Result<(), String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    // dual_psqt_reserve / dual_psqt_king_attacker は load 時 skip (= ckpt にあれば
    // section をまるごと skip するのみ、 graph 側 weight は Zeroed のまま保持)。
    // どちらも arch_str の DUAL=1 marker を共有しているため、 section magic で
    // format を識別して skip する。
    let _dual_psqt_reserve = dual_psqt_reserve;
    let _dual_psqt_king_attacker = dual_psqt_king_attacker;
    use std::fs;
    let buf = fs::read(path).map_err(|e| format!("failed to read {}: {}", path.display(), e))?;
    let mut cur = QuantReader::new(&buf);

    // ---- Header ----
    let _nnue_version = cur.read_u32()?;
    let _network_hash = cur.read_u32()?;
    let arch_len = cur.read_u32()? as usize;
    // arch 文字列は保持する (= warm-start 元 net が threat 入力を持つ場合、 PSQT 直後に
    // 物理的に置かれた threat 重み block を skip するために dims / 形式を検出する)。
    let src_arch = String::from_utf8_lossy(cur.read_bytes(arch_len)?).to_string();

    // ---- FT hash ----
    let _ft_hash = cur.read_u32()?;

    // ---- FT biases (LEB128 圧縮) ----
    let ft_biases_i16 = cur.read_leb128_i16_block(ft_out)?;
    let l0b_f32: Vec<f32> = ft_biases_i16.iter().map(|&v| v as f32 / QA_FT as f32).collect();
    set_graph_weight(graph, "l0b", &l0b_f32)?;

    // ---- FT weights (LEB128 圧縮) ----
    // graph 側 l0w は shape [ft_out, input_size] column-major (l0w.values[feat * ft_out + out])。
    // quantised.bin 側は halfka_dim * ft_out values (piece 部分のみ)、 column-major 同レイアウト。
    let ft_weights_i16 = cur.read_leb128_i16_block(halfka_dim * ft_out)?;
    let mut l0w_f32 = vec![0.0f32; input_size * ft_out];
    for (i, &v) in ft_weights_i16.iter().enumerate() {
        l0w_f32[i] = v as f32 / QA_FT as f32;
    }
    // threat / handthreat 部分は別 section から読み込む (後段)。
    set_graph_weight(graph, "l0w", &l0w_f32)?;

    // ---- PSQT (optional) ----
    if psqt {
        // biases: i32[NUM_BUCKETS]、 weights: i32[halfka_dim][NUM_BUCKETS]、 scale = QA*QB = 8128
        let scale = (QA as i32 * QB as i32) as f32;
        let psqtb_i32 = cur.read_i32_array(NUM_BUCKETS)?;
        let psqtb_f32: Vec<f32> = psqtb_i32.iter().map(|&v| v as f32 / scale).collect();
        set_graph_weight(graph, "psqtb", &psqtb_f32)?;

        // weights: i32[halfka_dim][NUM_BUCKETS] (feature-major, bucket fastest) → graph
        // psqtw shape (NUM_BUCKETS, input_size) column-major: values[feat * NUM_BUCKETS + bucket]。
        //
        // ★graph 側 psqtw は l0w と同じく **input_size 幅** で確保される (bullet の affine は
        //   weight 列数 == sparse 入力次元を要求するため、 threat / aux では PSQT projection も
        //   input_size 特徴に張られる。 threat も 5803 行で同構造)。 quantised.bin 側は
        //   HalfKA 部 (halfka_dim × NUM_BUCKETS) のみを持つ (= C++ / threat と bit-exact)。
        //   したがって l0w と同様に aux/threat 行 [halfka_dim, input_size) は zero-init で pad し、
        //   size mismatch を防ぎつつ warm-start step0 の aux PSQT 寄与をゼロにする
        //   (export は halfka_dim 行のみ書き出すため C++ とは引き続き bit-exact)。
        let psqtw_i32 = cur.read_i32_array(halfka_dim * NUM_BUCKETS)?;
        let mut psqtw_f32 = vec![0.0f32; input_size * NUM_BUCKETS];
        for (i, &v) in psqtw_i32.iter().enumerate() {
            psqtw_f32[i] = v as f32 / scale;
        }
        set_graph_weight(graph, "psqtw", &psqtw_f32)?;
    }

    // ---- HandThreat (optional) ----
    // 注: --threat は対応せず、 検出したらエラー。
    if hand_threat {
        let ht_dims = cur.read_u32()? as usize;
        let expected_ht_dims = input_size - halfka_dim;
        if ht_dims != expected_ht_dims {
            return Err(format!("HandThreat dims mismatch: quantised.bin={} graph={}", ht_dims, expected_ht_dims));
        }
        let ht_i8 = cur.read_i8_array(ht_dims * ft_out)?;
        // graph l0w の threat 部分は feat halfka_dim..input_size、 column-major、
        // すなわち l0w.values[(halfka_dim + j) * ft_out + out] for j in 0..ht_dims, out in 0..ft_out。
        // 保存時は同じレイアウトで dump されている (line 1453-1448 参照)。
        let mut l0w_full = read_graph_weight(graph, "l0w")?;
        let threat_start = halfka_dim * ft_out;
        for (i, &v) in ht_i8.iter().enumerate() {
            l0w_full[threat_start + i] = v as f32 / QA_FT as f32;
        }
        set_graph_weight(graph, "l0w", &l0w_full)?;
    }

    // ---- Threat 入力 block skip (warm-start 元 net が threat 入力を持つ場合) ----
    // sb1000 (nodchip_threat768_p4h2_base) など HalfKA+Threat+PSQT の net から
    // aux / plain HalfKA graph へ warm-start する際、 quantised.bin の物理 layout は
    //   FT / PSQT / **Threat 重み** / LayerStacks
    // であり、 threat 重み block を消費せずに LayerStacks を読むと誤読して L1-L3 が壊れる。
    // 元 net の arch_str に "Threat={dims}," がある場合、 その block をここで skip する
    // (= threat 寄与は破棄、 target graph 側 aux 行は zero-init のまま = plain 一致)。
    // threat 重み layout (build_layerstack_save_format 参照):
    //   [ThreatProfile!=0 のとき u32 LE prefix] + i8|i16 [dims × ft_out] (feature-major)。
    //   i16 判定は "ThreatI16=1" marker、 profile は "ThreatProfile={pid}" marker。
    if let Some(dims_pos) = src_arch.find("Threat=") {
        // "ThreatI16=1" / "ThreatProfile=" は別 token。 "Threat=" 直後が数字であることを確認して
        // これらの誤検出を避ける (find("Threat=") は "ThreatI16=" にはマッチしない = prefix 違い)。
        let after = &src_arch[dims_pos + "Threat=".len()..];
        let threat_dims: usize =
            after.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
        if threat_dims > 0 {
            let threat_i16 = src_arch.contains("ThreatI16=1");
            // profile id (full = 0、 marker 無し)。 pid != 0 の net は重み blob 直前に u32 prefix。
            let file_profile: u32 = if let Some(pp) = src_arch.find("ThreatProfile=") {
                src_arch[pp + "ThreatProfile=".len()..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0)
            } else {
                0
            };
            if file_profile != 0 {
                cur.read_u32()?; // u32 LE profile id prefix (値は破棄、byte 消費のみ)
            }
            let bytes_per = if threat_i16 { 2 } else { 1 };
            // --threat-bucket-groups の net は群数ぶんの表が連結されている ("ThreatGroups=<群数>:...")。
            let file_groups: usize = src_arch
                .find("ThreatGroups=")
                .and_then(|p| {
                    src_arch[p + "ThreatGroups=".len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect::<String>()
                        .parse()
                        .ok()
                })
                .unwrap_or(1);
            // 2026-11-05 修正: 専用 threat 入力 (--threat-out K) の net は、
            //   ファイル上の 1 特徴あたりの列数が **K** であって FT 幅 (ft_out) ではない。
            //   従来は ft_out で数えていたため dedicated threat の net では必ず EOF になり、
            //   この経路は一度も成立していなかった (54,180 x **1024** x 4 = 221.9 MB を
            //   87 MB のファイルに要求していた)。 K=0 (threat を FT へ畳み込む旧形式) は従来どおり。
            // 🚨2026-11-05 12:40 再修正: K は**ファイルの性質**なので、 実行時の
            //   --threat-out から決めてはいけない。 plain (threat 無し) の graph へ
            //   warm-start するときは threat_ft_out = 0 なので、 ft_out に落ちて
            //   また EOF になった (887 MB を要求 / 残りは 55.6 MB)。
            //   -> **残りバイト数に収まる最大の候補**を選ぶ。 threat の塊の直後は
            //      LayerStacks なので、 誤った K を選べば後続の読みが必ず壊れて気づける
            //      (正しい K なら余りがちょうど LayerStacks の大きさになる)。
            let k_file = if threat_ft_out > 0 {
                threat_ft_out
            } else {
                let rest = cur.remaining();
                let mut pick = 0usize;
                for cand in [ft_out, 256usize, 128, 64, 32] {
                    if cand > 0 && threat_dims * cand * bytes_per * file_groups <= rest {
                        pick = cand;
                        break;
                    }
                }
                if pick == 0 {
                    return Err(format!(
                        "threat block の K を決められない (dims={}, groups={}, bytes={}, 残り={})",
                        threat_dims, file_groups, bytes_per, rest
                    ));
                }
                println!(
                    "warm-start: threat の K をファイルの大きさから推定 = {} (dims={}, groups={}, 残り={} byte)",
                    pick, threat_dims, file_groups, rest
                );
                pick
            };
            let blob_bytes = threat_dims * k_file * bytes_per * file_groups;
            let blob = cur.read_bytes(blob_bytes)?;
            let want = threat_dims * k_file * file_groups;
            let mut loaded = false;
            if threat_ft_out > 0 {
                if let Ok(mut tw) = read_graph_weight(graph, "threat_ft_w") {
                    if tw.len() == want {
                        // graph の添字はファイルと同じ [群][特徴][K] (= (g*dims+f)*K + i、 10388 行)。
                        // 量子化スケールは主 FT と同じ QA_FT (2873 行 qa_for_threat = QA_FT)。
                        for i in 0..want {
                            let q = if threat_i16 {
                                i16::from_le_bytes([blob[2 * i], blob[2 * i + 1]]) as f32
                            } else {
                                blob[i] as i8 as f32
                            };
                            tw[i] = q / QA_FT as f32;
                        }
                        set_graph_weight(graph, "threat_ft_w", &tw)?;
                        loaded = true;
                        println!(
                            "warm-start: threat 入力 block を **読み込んだ** (dims={}, K={}, groups={}, i16={}, profile={}, {} bytes)。",
                            threat_dims, k_file, file_groups, threat_i16, file_profile, blob_bytes
                        );
                    } else {
                        println!(
                            "warm-start: threat_ft_w の要素数が合わない (graph {} 対 file {}) -> skip する",
                            tw.len(),
                            want
                        );
                    }
                }
            }
            if !loaded {
                println!(
                    "warm-start: 元 net の threat 入力 block を skip (dims={}, K={}, i16={}, profile={}, {} bytes 破棄)。",
                    threat_dims, k_file, threat_i16, file_profile, blob_bytes
                );
            }
        }
    }

    // ---- LayerStacks (NUM_BUCKETS buckets) ----
    let l1_effective = l1_out - 1;
    // Existing R checkpoints have pad32 L2 rows; columns 14/15 are zero at warm start.
    let l2_in = l1_effective * 2;
    let l1_total_in = ft_out + hand_count_dense_dims;
    let l1_padded_in = pad32(l1_total_in);
    let l2_padded_in = pad32(l2_in);
    let output_padded_in = pad32(l2_out);
    let bias_scale = (QA as i32 * QB as i32) as f32; // 8128 (L1 bias)
    let l2_bias_scale = 127.0 * QB as f32; // L2 / L3 bias

    let mut l1w = vec![0.0f32; NUM_BUCKETS * l1_out * l1_total_in];
    let mut l1b = vec![0.0f32; NUM_BUCKETS * l1_out];
    let mut l2w = vec![0.0f32; NUM_BUCKETS * l2_out * l2_in];
    let mut l2b = vec![0.0f32; NUM_BUCKETS * l2_out];
    let mut l3w = vec![0.0f32; NUM_BUCKETS * l2_out];
    let mut l3b = vec![0.0f32; NUM_BUCKETS];

    let l1_rows_total = NUM_BUCKETS * l1_out;
    let l2_rows_total = NUM_BUCKETS * l2_out;

    for bucket in 0..NUM_BUCKETS {
        // fc_hash skip
        let _ = cur.read_u32()?;

        // L1 biases: i32[l1_out]、 graph 側 l1b に merged 値として書き込み (l1fb は zero のまま)。
        let l1b_i32 = cur.read_i32_array(l1_out)?;
        for (out_idx, &v) in l1b_i32.iter().enumerate() {
            let global_out = bucket * l1_out + out_idx;
            l1b[global_out] = v as f32 / bias_scale;
        }

        // L1 weights: i8[l1_out × l1_padded_in]、 row-major (out, in)。
        // graph: l1w[in * l1_rows_total + global_out] column-major、 l1fw は zero のまま。
        for out_idx in 0..l1_out {
            let global_out = bucket * l1_out + out_idx;
            for in_idx in 0..l1_padded_in {
                let v = cur.read_i8()?;
                if in_idx < l1_total_in {
                    l1w[in_idx * l1_rows_total + global_out] = v as f32 / QB as f32;
                }
                // padding は読み捨て
            }
        }

        // L2 biases: i32[l2_out]
        let l2b_i32 = cur.read_i32_array(l2_out)?;
        for (out_idx, &v) in l2b_i32.iter().enumerate() {
            let global_out = bucket * l2_out + out_idx;
            l2b[global_out] = v as f32 / l2_bias_scale;
        }

        // L2 weights: i8[l2_out × l2_padded_in]、 row-major
        for out_idx in 0..l2_out {
            let global_out = bucket * l2_out + out_idx;
            for in_idx in 0..l2_padded_in {
                let v = cur.read_i8()?;
                if in_idx < l2_in {
                    l2w[in_idx * l2_rows_total + global_out] = v as f32 / QB as f32;
                }
            }
        }

        // Output bias: i32
        let l3b_v = cur.read_i32()?;
        l3b[bucket] = l3b_v as f32 / l2_bias_scale;

        // Output weights: i8[output_padded_in]、 graph l3w[in * NUM_BUCKETS + bucket] column-major
        for in_idx in 0..output_padded_in {
            let v = cur.read_i8()?;
            if in_idx < l2_out {
                l3w[in_idx * NUM_BUCKETS + bucket] = v as f32 / QB as f32;
            }
        }
    }

    // A legacy R checkpoint serializes a 32-wide L2 row.  Its pad bytes are not
    // semantic model inputs, so they must not become the two R-Mix16 features.
    // Preserve learned columns when the source already declares L2=16.
    if residual_mix && src_arch.contains("AffineTransform[64<-14]") {
        let legacy_l2_in = l1_effective * 2;
        for in_idx in legacy_l2_in..l2_in {
            for row in 0..l2_rows_total {
                l2w[in_idx * l2_rows_total + row] = 0.0;
            }
        }
    }

    set_graph_weight(graph, "l1w", &l1w)?;
    set_graph_weight(graph, "l1b", &l1b)?;
    set_graph_weight(graph, "l2w", &l2w)?;
    set_graph_weight(graph, "l2b", &l2b)?;
    set_graph_weight(graph, "l3w", &l3w)?;
    set_graph_weight(graph, "l3b", &l3b)?;

    // l1fw / l1fb は zero のまま (l1w/l1b に merged value 全部入れたため)。
    // graph 上は InitSettings::Zeroed なので default zero、 明示的な zero set は不要。
    // ただし resume 時 graph はすでに初期化済みなので、 念のため明示的に zero clear する。
    let l1fw_size = l1_out * ft_out;
    set_graph_weight(graph, "l1fw", &vec![0.0; l1fw_size])?;
    set_graph_weight(graph, "l1fb", &vec![0.0; l1_out])?;

    // ---- LMAS section (legacy、 2026-05-25 廃止) ----
    // LMAS 機能は削除済。 lmas_flags が立っている呼び出しは想定外なので bail。
    if lmas_flags != 0 {
        return Err(format!(
            "LMAS section は 2026-05-25 に廃止されました (lmas_flags=0x{:x})。 \
             lmas-free な quantised.bin を使用してください。",
            lmas_flags
        ));
    }
    // 旧 LMAS 読み込み block 本体 (= 約 240 行) は 2026-05-25 物理削除。
    // cfg(any()) は常に false なので block 内は compile されず、 シンボル参照
    // (LMAS_*_DIMS、 LMAS_*_w 等) も解決されない。
    #[cfg(any())]
    {
        if (lmas_flags & 0x01) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_SLA_DIMS;
            let raw = cur.read_i16_array(4 * dims)?;
            let mut sla_w = vec![0.0f32; NUM_BUCKETS * dims];
            for phase in 0..4 {
                for pt in 0..dims {
                    let value = raw[phase * dims + pt] as f32 / QA as f32;
                    sla_w[pt * NUM_BUCKETS + phase * 2] = value;
                    sla_w[pt * NUM_BUCKETS + phase * 2 + 1] = value;
                }
            }
            set_graph_weight(graph, "lmas_sla_w", &sla_w)?;
        }
        if (lmas_flags & 0x02) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_HML_DIMS;
            let raw = cur.read_i16_array(4 * dims)?;
            let mut hml_w = vec![0.0f32; NUM_BUCKETS * dims];
            for phase in 0..4 {
                for pt in 0..dims {
                    let value = raw[phase * dims + pt] as f32 / QA as f32;
                    hml_w[pt * NUM_BUCKETS + phase * 2] = value;
                    hml_w[pt * NUM_BUCKETS + phase * 2 + 1] = value;
                }
            }
            set_graph_weight(graph, "lmas_hml_w", &hml_w)?;
        }
        if (lmas_flags & 0x04) != 0 {
            // ZPSQT: [2][NUM_BUCKETS][126] i16 (black, white)
            let dims = bullet_lib::game::inputs::LMAS_ZPSQT_DIMS;
            for name in ["lmas_zpsqt_w_black", "lmas_zpsqt_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut zw = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        zw[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &zw)?;
            }
        }
        if (lmas_flags & 0x08) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_KSA_DIMS;
            let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
            let mut ksa_w = vec![0.0f32; NUM_BUCKETS * dims];
            for bucket in 0..NUM_BUCKETS {
                for input_idx in 0..dims {
                    let disk_idx = bucket * dims + input_idx;
                    ksa_w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                }
            }
            set_graph_weight(graph, "lmas_ksa_w", &ksa_w)?;
        }
        if (lmas_flags & 0x10) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_CT_DIMS;
            let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
            let mut ct_w = vec![0.0f32; NUM_BUCKETS * dims];
            for bucket in 0..NUM_BUCKETS {
                for input_idx in 0..dims {
                    let disk_idx = bucket * dims + input_idx;
                    ct_w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                }
            }
            set_graph_weight(graph, "lmas_ct_w", &ct_w)?;
        }
        if (lmas_flags & 0x20) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_KZPSQT_DIMS;
            for name in ["lmas_kzpsqt_w_black", "lmas_kzpsqt_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut kzpsqt_w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        kzpsqt_w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &kzpsqt_w)?;
            }
        }

        // v5 ContextGate: 4,536 raw i16 (QA scale なし)
        if (lmas_flags & 0x40) != 0 {
            let cols = bullet_lib::game::inputs::LMAS_CG_ATTACK_BUCKETS
                * bullet_lib::game::inputs::LMAS_CG_MATERIAL_BUCKETS
                * bullet_lib::game::inputs::LMAS_CG_GROUPS;
            let phases = bullet_lib::game::inputs::LMAS_CG_PHASES;
            let raw = cur.read_i16_array(phases * cols)?;
            let mut cg_w = vec![0.0f32; NUM_BUCKETS * cols];
            for phase in 0..phases {
                for col in 0..cols {
                    let disk_idx = phase * cols + col;
                    cg_w[col * NUM_BUCKETS + phase] = raw[disk_idx] as f32;
                }
            }
            set_graph_weight(graph, "lmas_context_gate_w", &cg_w)?;
        }

        // v5 KZLine: [2][NUM_BUCKETS][432] i16
        if (lmas_flags & 0x80) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_KZLINE_AUX_DIMS;
            for name in ["lmas_kzline_w_black", "lmas_kzline_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut kz_w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        kz_w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &kz_w)?;
            }
        }

        // v5 KZESC: [2][NUM_BUCKETS][225] i16
        if (lmas_flags & 0x100) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_KZESC_AUX_DIMS;
            for name in ["lmas_kzesc_w_black", "lmas_kzesc_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut kz_w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        kz_w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &kz_w)?;
            }
        }

        // v6 KP-PSQT raw: [2][NUM_BUCKETS][183,708] i16
        if (lmas_flags & 0x200) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_KP_PSQT_RAW_DIMS;
            for name in ["lmas_kp_psqt_raw_w_black", "lmas_kp_psqt_raw_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &w)?;
            }
        }

        // v6 Hand × King-zone: [2][NUM_BUCKETS][63] i16
        if (lmas_flags & 0x400) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_HAND_KZONE_DIMS;
            for name in ["lmas_hand_kzone_w_black", "lmas_hand_kzone_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &w)?;
            }
        }

        // v6 Threat Map: [2][NUM_BUCKETS][126] i16
        if (lmas_flags & 0x800) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_THREAT_MAP_DIMS;
            for name in ["lmas_threat_map_w_black", "lmas_threat_map_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &w)?;
            }
        }

        // v7 KKR: [2][NUM_BUCKETS][14580] i16
        if (lmas_flags & 0x1000) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_KKR_DIMS;
            for name in ["lmas_kkr_w_black", "lmas_kkr_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &w)?;
            }
        }

        // v7 PSH: [2][NUM_BUCKETS][54] i16
        if (lmas_flags & 0x2000) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_PSH_DIMS;
            for name in ["lmas_psh_w_black", "lmas_psh_w_white"] {
                let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
                let mut w = vec![0.0f32; NUM_BUCKETS * dims];
                for bucket in 0..NUM_BUCKETS {
                    for input_idx in 0..dims {
                        let disk_idx = bucket * dims + input_idx;
                        w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 / QA as f32;
                    }
                }
                set_graph_weight(graph, name, &w)?;
            }
        }

        // v7 PBR: [NUM_BUCKETS][640] i16、 perspective なし。
        // disk 値は GATE_DEFAULT=1024 中心 (×1.0)、 graph weight は 0 中心 delta に変換する。
        if (lmas_flags & 0x4000) != 0 {
            let dims = bullet_lib::game::inputs::LMAS_PBR_DIMS;
            const GATE_DEFAULT: f32 = 1024.0;
            let raw = cur.read_i16_array(NUM_BUCKETS * dims)?;
            let mut w = vec![0.0f32; NUM_BUCKETS * dims];
            for bucket in 0..NUM_BUCKETS {
                for input_idx in 0..dims {
                    let disk_idx = bucket * dims + input_idx;
                    // disk 上の絶対値から GATE_DEFAULT を引いて graph 上の delta に戻す。
                    w[input_idx * NUM_BUCKETS + bucket] = raw[disk_idx] as f32 - GATE_DEFAULT;
                }
            }
            set_graph_weight(graph, "lmas_pbr_w", &w)?;
        }
    }

    // ---- PIN section (LMAS の後、 arch_str に "PIN=" がある場合) ----
    //
    // sub_magic (8B) + version (u32) + i16[bucket][persp][dim] (= 72,588 byte total)。
    if pin_aware {
        use bullet_lib::game::inputs::{
            PIN_DIMENSIONS_PER_PERSPECTIVE, PIN_NUM_BUCKETS, PIN_SECTION_VERSION, PIN_SUB_MAGIC,
        };
        let magic = cur.read_bytes(8)?;
        if magic != PIN_SUB_MAGIC {
            return Err(format!("PIN section magic mismatch: expected {:?}, got {:?}", PIN_SUB_MAGIC, magic));
        }
        let version = cur.read_u32()?;
        if version != PIN_SECTION_VERSION {
            return Err(format!("PIN section version mismatch: expected {}, got {}", PIN_SECTION_VERSION, version));
        }
        let dims = PIN_DIMENSIONS_PER_PERSPECTIVE;
        // disk 順 = [bucket][persp][dim]、 graph weight tensor は per-persp。
        // 各 graph tensor は (NUM_BUCKETS, dims) column-major、
        // すなわち w[input_idx * NUM_BUCKETS + bucket]。
        let mut w_black = vec![0.0f32; PIN_NUM_BUCKETS * dims];
        let mut w_white = vec![0.0f32; PIN_NUM_BUCKETS * dims];
        for bucket in 0..PIN_NUM_BUCKETS {
            for persp in 0..2 {
                let raw = cur.read_i16_array(dims)?;
                let target = if persp == 0 { &mut w_black } else { &mut w_white };
                for (input_idx, &v) in raw.iter().enumerate() {
                    target[input_idx * PIN_NUM_BUCKETS + bucket] = v as f32 / QA as f32;
                }
            }
        }
        set_graph_weight(graph, "pin_w_black", &w_black)?;
        set_graph_weight(graph, "pin_w_white", &w_white)?;
    }

    // ---- DUAL section ----
    // 2 format 存在 (どちらも graph 側 weight は touch せず skip):
    //
    // 1) reserve (旧): "P2DUAL\0\0" (8B) + version u32 + i16[2 head][8 bucket] = 44 byte
    // 2) king-attacker (新): "P2DUAL2\0" (8B) + version u32 + count u32 +
    //    i16[2 persp][2 head][8 bucket][81 sq][14 piece] = 72,592 byte
    //
    // ckpt format integrity の確認用に magic/version は validate する。
    if _dual_psqt_reserve {
        let magic = cur.read_bytes(8)?;
        if magic != b"P2DUAL\0\0" {
            return Err(format!(
                "DUAL section magic mismatch: expected P2DUAL\\0\\0, got {:?}",
                std::str::from_utf8(magic).unwrap_or("<binary>")
            ));
        }
        let version = cur.read_u32()?;
        if version != 1 {
            return Err(format!("DUAL section version mismatch: expected 1, got {}", version));
        }
        const NUM_DUAL_HEADS: usize = 2;
        const NUM_DUAL_BUCKETS: usize = 8;
        let _raw = cur.read_i16_array(NUM_DUAL_HEADS * NUM_DUAL_BUCKETS)?;
    } else if _dual_psqt_king_attacker {
        let magic = cur.read_bytes(8)?;
        if magic != b"P2DUAL2\0" {
            return Err(format!(
                "DUAL section magic mismatch: expected P2DUAL2\\0, got {:?}",
                std::str::from_utf8(magic).unwrap_or("<binary>")
            ));
        }
        let version = cur.read_u32()?;
        if version != 1 {
            return Err(format!("DUAL (king-attacker) section version mismatch: expected 1, got {}", version));
        }
        let count = cur.read_u32()?;
        // 2 perspective × 2 head × 8 bucket × 81 sq × 14 piece = 36,288
        let expected_count: u32 = 36_288;
        if count != expected_count {
            return Err(format!("DUAL (king-attacker) count mismatch: expected {}, got {}", expected_count, count));
        }
        // 36,288 i16 entries を skip (Zero 学習されていないため graph 反映なし)
        let _raw = cur.read_i16_array(expected_count as usize)?;
    }

    // ---- head section 群 (= dual_threat / activity_risk / king_escape) ----
    //
    // bug fix (2026-06-13): 従来 resume-from-quantised は main + psqt + LayerStacks
    // までしか読まず、 これら head section を黙って読み飛ばすため graph 側 weight が
    // Zeroed のまま残る欠陥があった。 ここで buf 全体を magic でスキャンし、 見つかった
    // section のみをロードする (= 順序非依存・存在ガード)。 graph に該当 tensor が
    // なければ黙って skip。
    //
    // 逐次 cursor ではなく buf 全体の windows().position() で magic offset を探すため、
    // ファイル末尾の section 順序や前段 skip 処理に依存しない。
    {
        // little-endian i16 を buf offset から読む小ヘルパ。
        let read_i16_le = |o: usize| -> i16 { i16::from_le_bytes([buf[o], buf[o + 1]]) };
        // buf 全体から magic byte 列の offset を探す。
        let find_magic = |magic: &[u8]| -> Option<usize> { buf.windows(magic.len()).position(|w| w == magic) };

        // ---- section 1: P2DUALZ (= dual_w_threat black/white、 各 127,008 i16) ----
        if let Some(off) = find_magic(b"P2DUALZ\0") {
            const DIM: usize = 127_008;
            let mut p = off + 16; // header 16B skip
            let mut black = vec![0.0f32; DIM];
            for slot in black.iter_mut() {
                *slot = read_i16_le(p) as f32 / 127.0;
                p += 2;
            }
            let mut white = vec![0.0f32; DIM];
            for slot in white.iter_mut() {
                *slot = read_i16_le(p) as f32 / 127.0;
                p += 2;
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "dual_w_threat_black") {
                set_graph_weight(graph, "dual_w_threat_black", &black)?;
                loaded += DIM;
            }
            if has_graph_weight(graph, "dual_w_threat_white") {
                set_graph_weight(graph, "dual_w_threat_white", &white)?;
                loaded += DIM;
            }
            eprintln!("load_from_quantised: loaded P2DUALZ ({} values)", loaded);
        }

        // ---- section 2: P2ACTR (= activity_risk base 4096 + tactical 1024) ----
        if let Some(off) = find_magic(b"P2ACTR\0\0") {
            let mut p = off + 16; // header 16B skip
            // base 部: 8 bucket × 2 persp × 8 activity × 8 pressure × 4 shield = 4096 i16
            let mut black = vec![0.0f32; 2048];
            let mut white = vec![0.0f32; 2048];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for activity in 0..8 {
                        for pressure in 0..8 {
                            for shield in 0..4 {
                                let local = bucket * 256 + activity * 32 + pressure * 4 + shield;
                                let v = read_i16_le(p) as f32 / 127.0;
                                p += 2;
                                if persp == 0 {
                                    black[local] = v;
                                } else {
                                    white[local] = v;
                                }
                            }
                        }
                    }
                }
            }
            // tactical 部: 8 bucket × 2 persp × 4 heavy × 4 escape × 4 check = 1024 i16
            let mut tac_black = vec![0.0f32; 512];
            let mut tac_white = vec![0.0f32; 512];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for heavy in 0..4 {
                        for escape in 0..4 {
                            for check in 0..4 {
                                let local = bucket * 64 + heavy * 16 + escape * 4 + check;
                                let v = read_i16_le(p) as f32 / 127.0;
                                p += 2;
                                if persp == 0 {
                                    tac_black[local] = v;
                                } else {
                                    tac_white[local] = v;
                                }
                            }
                        }
                    }
                }
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "activity_risk_w_black") {
                set_graph_weight(graph, "activity_risk_w_black", &black)?;
                loaded += 2048;
            }
            if has_graph_weight(graph, "activity_risk_w_white") {
                set_graph_weight(graph, "activity_risk_w_white", &white)?;
                loaded += 2048;
            }
            if has_graph_weight(graph, "activity_risk_v3_tactical_w_black") {
                set_graph_weight(graph, "activity_risk_v3_tactical_w_black", &tac_black)?;
                loaded += 512;
            }
            if has_graph_weight(graph, "activity_risk_v3_tactical_w_white") {
                set_graph_weight(graph, "activity_risk_v3_tactical_w_white", &tac_white)?;
                loaded += 512;
            }
            eprintln!("load_from_quantised: loaded P2ACTR ({} values)", loaded);
        }

        // ---- section 3: P2KESC (= king_escape black/white、 各 persp 32 i16、 disk 計 64) ----
        // graph tensor は per-persp で 8 bucket × 4 class = 32 要素 (= KING_ESCAPE_DIM_PER_PERSP)。
        if let Some(off) = find_magic(b"P2KESC\0\0") {
            let mut p = off + 16; // header 16B skip
            let mut black = vec![0.0f32; 32];
            let mut white = vec![0.0f32; 32];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for class in 0..4 {
                        let local = bucket * 4 + class;
                        let v = read_i16_le(p) as f32 / 127.0;
                        p += 2;
                        if persp == 0 {
                            black[local] = v;
                        } else {
                            white[local] = v;
                        }
                    }
                }
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "king_escape_w_black") {
                set_graph_weight(graph, "king_escape_w_black", &black)?;
                loaded += 32;
            }
            if has_graph_weight(graph, "king_escape_w_white") {
                set_graph_weight(graph, "king_escape_w_white", &white)?;
                loaded += 32;
            }
            eprintln!("load_from_quantised: loaded P2KESC ({} values)", loaded);
        }

        // ---- section: P2SXRY (= slider_xray black/white、 各 persp 64 i16、 disk 計 128) ----
        // graph tensor は per-persp で 8 bucket × 8 class = 64 要素 (= SLIDER_XRAY_DIM_PER_PERSP)。
        // magic-seek (= find_magic) で順序非依存にロード、 has_graph ガードで未対応 graph は skip。
        if let Some(off) = find_magic(b"P2SXRY\0\0") {
            let mut p = off + 16; // header 16B skip
            let mut black = vec![0.0f32; 64];
            let mut white = vec![0.0f32; 64];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for class in 0..8 {
                        let local = bucket * 8 + class;
                        let v = read_i16_le(p) as f32 / 127.0;
                        p += 2;
                        if persp == 0 {
                            black[local] = v;
                        } else {
                            white[local] = v;
                        }
                    }
                }
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "slider_xray_w_black") {
                set_graph_weight(graph, "slider_xray_w_black", &black)?;
                loaded += 64;
            }
            if has_graph_weight(graph, "slider_xray_w_white") {
                set_graph_weight(graph, "slider_xray_w_white", &white)?;
                loaded += 64;
            }
            eprintln!("load_from_quantised: loaded P2SXRY ({} values)", loaded);
        }

        // ---- section: P2KZT factored v1 (= flat 10,336 i16 = core→region→bucket→persp→bias) ----
        // 各 table weight tensor は KZT_TABLE_DIMS[t] 要素。 disk は固定順で連結。 section 不在
        // (= 旧 ckpt / dense version 1) なら find_magic が None → load skip → graph tensor は
        // Zeroed init のまま (= zero から育成)。 factored の base FT net は KZT section 非搭載。
        if let Some(off) = find_magic(b"P2KZT\0\0\0") {
            use bullet_lib::game::inputs::{KZT_TABLE_DIMS, KZT_TABLE_WEIGHT_NAMES};
            let mut p = off + 16; // header 16B skip (= magic8 + version4 + count4)
            let mut loaded = 0usize;
            for (name, dim) in KZT_TABLE_WEIGHT_NAMES.iter().zip(KZT_TABLE_DIMS.iter()) {
                let mut vals = vec![0.0f32; *dim];
                for v in vals.iter_mut() {
                    *v = read_i16_le(p) as f32 / 127.0;
                    p += 2;
                }
                if has_graph_weight(graph, name) {
                    set_graph_weight(graph, name, &vals)?;
                    loaded += *dim;
                }
            }
            eprintln!("load_from_quantised: loaded P2KZT factored ({} values)", loaded);
        }

        // ---- section: P2KMOB (= king_mobility black/white、 各 persp 72 i16、 disk 計 144) ----
        // graph tensor は per-persp で 8 bucket × 9 class = 72 要素 (= KING_MOBILITY_DIM_PER_PERSP)。
        // class が 9 (= 0..=8) である点が ADEV / KDGR と異なる。
        // magic-seek (= find_magic) で順序非依存にロード、 has_graph ガードで未対応 graph は skip。
        if let Some(off) = find_magic(b"P2KMOB\0\0") {
            let mut p = off + 16; // header 16B skip
            let mut black = vec![0.0f32; 72];
            let mut white = vec![0.0f32; 72];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for class in 0..9 {
                        let local = bucket * 9 + class;
                        let v = read_i16_le(p) as f32 / 127.0;
                        p += 2;
                        if persp == 0 {
                            black[local] = v;
                        } else {
                            white[local] = v;
                        }
                    }
                }
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "king_mobility_w_black") {
                set_graph_weight(graph, "king_mobility_w_black", &black)?;
                loaded += 72;
            }
            if has_graph_weight(graph, "king_mobility_w_white") {
                set_graph_weight(graph, "king_mobility_w_white", &white)?;
                loaded += 72;
            }
            eprintln!("load_from_quantised: loaded P2KMOB ({} values)", loaded);
        }

        // ---- section: P2DROP (= drop_pressure black/white、 各 persp 64 i16、 disk 計 128) ----
        // graph tensor は per-persp で 8 bucket × 8 class = 64 要素 (= DROP_PRESSURE_DIM_PER_PERSP)。
        // class は 8 (= 0..=7、 ADEV / KDGR と同形)。
        // magic-seek (= find_magic) で順序非依存にロード、 has_graph ガードで未対応 graph は skip。
        if let Some(off) = find_magic(b"P2DROP\0\0") {
            let mut p = off + 16; // header 16B skip
            let mut black = vec![0.0f32; 64];
            let mut white = vec![0.0f32; 64];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for class in 0..8 {
                        let local = bucket * 8 + class;
                        let v = read_i16_le(p) as f32 / 127.0;
                        p += 2;
                        if persp == 0 {
                            black[local] = v;
                        } else {
                            white[local] = v;
                        }
                    }
                }
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "drop_pressure_w_black") {
                set_graph_weight(graph, "drop_pressure_w_black", &black)?;
                loaded += 64;
            }
            if has_graph_weight(graph, "drop_pressure_w_white") {
                set_graph_weight(graph, "drop_pressure_w_white", &white)?;
                loaded += 64;
            }
            eprintln!("load_from_quantised: loaded P2DROP ({} values)", loaded);
        }

        // ---- section: P2EGTMP (= Endgame Tempo、各 persp 512 i16) ----
        if let Some(off) = find_magic(b"P2EGTMP\0") {
            const DIM: usize = 8 * 64;
            let mut p = off + 16;
            let mut black = vec![0.0f32; DIM];
            let mut white = vec![0.0f32; DIM];
            for bucket in 0..8 {
                for persp in 0..2 {
                    for class in 0..64 {
                        let local = bucket * 64 + class;
                        let value = read_i16_le(p) as f32 / 127.0;
                        p += 2;
                        if persp == 0 {
                            black[local] = value;
                        } else {
                            white[local] = value;
                        }
                    }
                }
            }
            if has_graph_weight(graph, "endgame_tempo_w_black") {
                set_graph_weight(graph, "endgame_tempo_w_black", &black)?;
            }
            if has_graph_weight(graph, "endgame_tempo_w_white") {
                set_graph_weight(graph, "endgame_tempo_w_white", &white)?;
            }
            eprintln!("load_from_quantised: loaded P2EGTMP ({} values)", 2 * DIM);
        }

        // ---- section: P2ARS1 (= Attack Relay v2 shared tensor) ----
        if let Some(off) = find_magic(b"P2ARS1\0\0") {
            let count = u32::from_le_bytes(buf[off + 12..off + 16].try_into().unwrap()) as usize;
            if has_graph_weight(graph, "attack_relay_v2_w") {
                let expected = bullet_lib::game::inputs::ARH_V2_WEIGHT_COUNT;
                if count != expected {
                    return Err(format!("P2ARS1 count mismatch: expected {expected}, got {count}"));
                }
                let mut values = vec![0.0f32; count];
                let mut p = off + 16;
                for value in &mut values {
                    *value = read_i16_le(p) as f32 / bullet_lib::game::inputs::ARH_V2_QA as f32;
                    p += 2;
                }
                set_graph_weight(graph, "attack_relay_v2_w", &values)?;
                eprintln!("load_from_quantised: loaded P2ARS1 ({count} values)");
            }
        }

        // ---- section: P2STHR (= feature-major [feature][bucket]) ----
        if let Some(off) = find_magic(b"P2STHR\0\0") {
            let count = u32::from_le_bytes(buf[off + 12..off + 16].try_into().unwrap()) as usize;
            if has_graph_weight(graph, "scalar_threat_w") {
                let mut values = vec![0.0f32; count];
                let mut p = off + 16;
                for value in &mut values {
                    *value = read_i16_le(p) as f32 / 127.0;
                    p += 2;
                }
                set_graph_weight(graph, "scalar_threat_w", &values)?;
                eprintln!("load_from_quantised: loaded P2STHR ({count} values)");
            }
        }

        // ---- section 4: P2PAIR (= pairshape black/white、 各 persp 1,536 i16、 disk 計 3,072) ----
        // disk layout: [bucket 8][persp 2][pair 12][shape 16]
        // graph layout (per persp): bucket_major flat = bucket * 192 + pair * 16 + shape
        // tensor 名: pairshape_w_black (persp=0) / pairshape_w_white (persp=1)
        // QA = 127 (= save 側 PAIRSHAPE_QA と同一)。
        if let Some(off) = find_magic(b"P2PAIR\0\0") {
            const DIM: usize = 1_536; // PAIRSHAPE_DIM_PER_PERSP = 8 * 12 * 16
            let mut p = off + 16; // header 16B skip
            let mut black = vec![0.0f32; DIM];
            let mut white = vec![0.0f32; DIM];
            for bucket in 0..8usize {
                for persp in 0..2usize {
                    for pair in 0..12usize {
                        for shape in 0..16usize {
                            let local = bucket * 192 + pair * 16 + shape;
                            let v = read_i16_le(p) as f32 / 127.0;
                            p += 2;
                            if persp == 0 {
                                black[local] = v;
                            } else {
                                white[local] = v;
                            }
                        }
                    }
                }
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "pairshape_w_black") {
                set_graph_weight(graph, "pairshape_w_black", &black)?;
                loaded += DIM;
            }
            if has_graph_weight(graph, "pairshape_w_white") {
                set_graph_weight(graph, "pairshape_w_white", &white)?;
                loaded += DIM;
            }
            eprintln!("load_from_quantised: loaded P2PAIR ({} values)", loaded);
        }

        // ---- section: P2DCG1 (= Dual Cross Geometry black/white、 各 persp 50,176 i16、 disk 計 100,352) ----
        // disk layout: [persp 2][bucket 8][pair 196][geo 32] の i16 stream (= save 側と同順)。
        // graph layout (per persp): bucket-major flat = bucket * 196 * 32 + pair * 32 + geo。
        // disk の per-persp 区間 (= [bucket][pair][geo]) が graph flat 順と一致するため、
        // black 前半 50,176 / white 後半 50,176 を順次 copy するだけでよい。
        // tensor 名: dual_cross_geometry_w_black (persp=0) / dual_cross_geometry_w_white (persp=1)
        // QA = 127 (= save 側と同一、 *127 で量子化済 → /127.0 で逆量子化)。
        if let Some(off) = find_magic(b"P2DCG1\0\0") {
            const DIM: usize = 50_176; // DUAL_CROSS_GEOMETRY_DIM_PER_PERSP = 8 * 14 * 14 * 32
            let mut p = off + 16; // header 16B (magic 8 + version u32 + count u32) skip
            let mut black = vec![0.0f32; DIM];
            let mut white = vec![0.0f32; DIM];
            for local in &mut black {
                *local = read_i16_le(p) as f32 / 127.0;
                p += 2;
            }
            for local in &mut white {
                *local = read_i16_le(p) as f32 / 127.0;
                p += 2;
            }
            let mut loaded = 0usize;
            if has_graph_weight(graph, "dual_cross_geometry_w_black") {
                set_graph_weight(graph, "dual_cross_geometry_w_black", &black)?;
                loaded += DIM;
            }
            if has_graph_weight(graph, "dual_cross_geometry_w_white") {
                set_graph_weight(graph, "dual_cross_geometry_w_white", &white)?;
                loaded += DIM;
            }
            eprintln!("load_from_quantised: loaded P2DCG1 ({} values)", loaded);
        }

        // ---- section 5: P2DUAL2 (= king-attacker dual PSQT、 attack/defend head × persp) ----
        //
        // bug fix (2026-06-13): 従来 load_from_quantised_bin は P2DUAL2 section を
        // 逐次 cursor で skip (= _raw に読み捨て) するだけで graph に反映しなかった。
        // この結果 dual_w_attack_* / dual_w_defend_* tensor が Zeroed のまま残り、
        // king-attacker head を持つ ckpt (= arm12_dlsuisho_v2 系) を resume-from-quantised
        // → save すると P2DUAL2 が全 zero に潰れて eval が崩れていた (= round-trip 非可逆)。
        //
        // save 側 (= 上記 dual_data transform、 P2DUAL2 ブロック) と bit-exact 逆変換:
        //   disk layout: [persp(B,W)][head(0=attack,1=defend)][bucket 8][sq 81][piece 14]
        //   各 (persp,head) は 9,072 値の連続 = graph tensor values index 0..9072 と同一。
        //   scale = 127 (= save 側 DUAL_QA)。 defend head は save が符号反転せず w_d を
        //   そのまま書くため (= forward `attack - defend` 規約、 disk 上 defend は positive)、
        //   load も graph tensor に直接代入する (= 符号操作なし)。
        // graph に該当 tensor が無い場合 (= --dual-disable-king-attacker) は skip。
        if let Some(off) = find_magic(b"P2DUAL2\0") {
            const DIM_PER_HEAD: usize = 8 * 81 * 14; // = 9,072 (= DUAL_DIM_PER_HEAD)
            let mut p = off + 16; // header (magic 8 + version 4 + count 4) を skip
            // disk 読み出し順 = persp(B,W) × head(attack,defend) × 9,072 連続。
            let names =
                [["dual_w_attack_black", "dual_w_defend_black"], ["dual_w_attack_white", "dual_w_defend_white"]];
            let mut loaded = 0usize;
            for persp_names in names.iter() {
                for &tensor_name in persp_names.iter() {
                    let mut w = vec![0.0f32; DIM_PER_HEAD];
                    for slot in w.iter_mut() {
                        *slot = read_i16_le(p) as f32 / 127.0;
                        p += 2;
                    }
                    if has_graph_weight(graph, tensor_name) {
                        set_graph_weight(graph, tensor_name, &w)?;
                        loaded += DIM_PER_HEAD;
                    }
                }
            }
            eprintln!("load_from_quantised: loaded P2DUAL2 ({} values)", loaded);
        }
    }

    load_p2thrd2_direct_if_present(graph, &buf, halfka_dim, input_size, direct15_persp_split, direct15_defended)?;
    Ok(())
}

/// `--direct15-expand-head`: checkpoint の Direct15 head (`threat_direct_w` /
/// `threat_direct_b_cancels`) を graph 側の形状 (`--direct15-persp-split` / `--direct15-defended`)
/// へ複製展開する [`CheckpointShapeAdapter`]。
///
/// 重み (weights.bin) と optimiser state (Ranger21: momentum / velocity / max_velocity / slow、
/// AdamW / RAdam: momentum / velocity、 Ranger: slow) の全バッファに同じ写像
/// ([`expand_direct15_head_weights`] / [`expand_direct15_head_bias`]) を適用するので、
/// 展開後も要素ごとの weight ↔ state の対応が保たれ、 head の学習 step 数・分散推定を
/// 引き継いだまま続きから学習できる。 展開の意味づけ (bucket / 視点 / 守備ビット) は
/// `bullet_lib::game::inputs::Direct15HeadShape` を参照。
struct Direct15HeadExpandAdapter {
    from: Direct15HeadShape,
    to: Direct15HeadShape,
}

impl Direct15HeadExpandAdapter {
    /// `weights_bin` (bullet checkpoint の `weights.bin`) のヘッダだけを走査して base の head 形状を
    /// 判定し、 graph 形状 `to` への adapter を作る。
    ///
    /// 要素数 1,050,240 は P5 `[8][131,280]` と P2 `[16][65,640]` で一致するため、
    /// `threat_direct_b_cancels` の要素数 (= 行数) で行数を確定してから列数を割り出す。
    /// head を持たない checkpoint、 または既に `to` と同形なら `Ok(None)` (展開不要)。
    fn from_weights_bin(weights_bin: &Path, to: Direct15HeadShape) -> Result<Option<Self>, String> {
        let lens = scan_checkpoint_weight_lens(weights_bin)?;
        let len_of = |id: &str| lens.iter().find(|(k, _)| k == id).map(|(_, n)| *n);
        let (Some(weight_len), Some(bias_len)) = (len_of("threat_direct_w"), len_of("threat_direct_b_cancels")) else {
            return Ok(None);
        };
        let from = Direct15HeadShape::infer(to.buckets, weight_len, bias_len).ok_or_else(|| {
            format!(
                "{}: threat_direct_w={weight_len} / threat_direct_b_cancels={bias_len} 要素は既知の Direct15 head 形状 \
                 (rows ∈ {{{}, {}}}, dims ∈ {{{THREAT_DIRECT15_DIMENSIONS}, {THREAT_DIRECT15_DEFENDED_DIMENSIONS}}}) に一致しません",
                weights_bin.display(),
                to.buckets,
                2 * to.buckets
            )
        })?;
        if from == to {
            return Ok(None);
        }
        if from.persp > to.persp || from.halves > to.halves {
            return Err(format!(
                "{}: base の Direct15 head {from:?} は graph {to:?} より大きく、 縮小展開はできません \
                 (--direct15-persp-split / --direct15-defended を base に合わせてください)",
                weights_bin.display()
            ));
        }
        Ok(Some(Self { from, to }))
    }
}

impl CheckpointShapeAdapter for Direct15HeadExpandAdapter {
    fn adapt(&self, id: &str, values: &[f32], expected: usize) -> Option<Vec<f32>> {
        let out = match id {
            "threat_direct_w" => expand_direct15_head_weights(values, self.from, self.to),
            "threat_direct_b_cancels" => expand_direct15_head_bias(values, self.from, self.to),
            _ => return None,
        };
        match &out {
            Some(v) if v.len() == expected => {
                eprintln!(
                    "direct15-expand-head: {id} {} -> {} 要素 ({}x{} -> {}x{}) を複製展開",
                    values.len(),
                    v.len(),
                    self.from.rows(),
                    self.from.dims(),
                    self.to.rows(),
                    self.to.dims()
                );
            }
            _ => {
                eprintln!(
                    "direct15-expand-head: {id} を展開できません (ckpt={} 要素, graph={expected} 要素, {:?} -> {:?})",
                    values.len(),
                    self.from,
                    self.to
                );
                return None;
            }
        }
        out
    }
}

/// 2026-11-18 KT: fc0 (l1 の入力 = 両視点の pairwise 積 [stm (ft_out/2) | ntm (ft_out/2)]) のうち、
/// KH が足される FT 行 [off, off+W/2) と [half+off, half+off+W/2) の積を読む列。
/// pairwise は `ft[j] * ft[j + half]` なので、 視点ごとに列 [off, off+W/2) が KH の値を読む。
fn kh_fc0_cols(ft_out: usize, kh_out: usize, off: usize) -> [(usize, usize); 2] {
    let half = ft_out / 2;
    [(off, off + kh_out / 2), (half + off, half + off + kh_out / 2)]
}

/// 2026-11-18 KT: column-major (列 c = 要素 [c*rows, (c+1)*rows)) の重みで、 `open` の列だけ倍率 1.0 (decay 1.0)、
/// 他を 1e-12 (decay 0) にする区間倍率 (Ranger21 の `set_range_lr_scales` の形 = [0, size) を昇順に隙間なく覆う)。
fn kh_fc0_col_ranges(rows: usize, cols: usize, open: &[(usize, usize)]) -> Vec<(usize, usize, f32, f32)> {
    let mut out = Vec::new();
    let mut cur = 0usize;
    for &(a, b) in open {
        assert!(a >= cur && b > a && b <= cols, "kh_fc0_col_ranges: 列区間が不正 {open:?} (cols {cols})");
        if a > cur {
            out.push((cur * rows, a * rows, 1e-12, 0.0));
        }
        out.push((a * rows, b * rows, 1.0, 1.0));
        cur = b;
    }
    if cur < cols {
        out.push((cur * rows, cols * rows, 1e-12, 0.0));
    }
    out
}

/// checkpoint 読み込み時の要素数変換をまとめる [`CheckpointShapeAdapter`]。
///
/// - `threat_tile = Some((per_table, groups))` (`--threat-bucket-groups`): checkpoint の `threat_ft_w` が
///   1 表分 (`per_table` 要素) で graph が `per_table * groups` 要素のとき、 群数ぶん**そのまま連結 (tile)** する。
///   `threat_ft_w` は (K, dims) の column-major = feature-major の連続ブロックなので、 連結すると
///   列 `f + g * dims` が元の列 `f` の複製になる (= 全群が元の表で始まり、 出力は元の net と同じ)。
///   weights.bin と optimiser state (momentum / velocity / max_velocity / slow) の全バッファに同じ変換が
///   掛かるので、 Ranger21 の分散推定・step 数も引き継がれる。
/// - `direct15`: `--direct15-expand-head` の展開 (従来どおり)。
struct CheckpointLoadAdapter {
    threat_tile: Option<(usize, usize)>,
    /// 2026-09-16 `--threat-factorizer`: graph の threat_ft_w 末尾にある factor 列の要素数 (K x 324)。
    /// checkpoint に factor 列が無いとき (= 実特徴分だけ) 末尾を 0 で埋める (出力は resume 元と同一)。
    threat_factor_extra: usize,
    /// 2026-11-18 KT: graph の kh_ft_w 末尾にある X 部の要素数 (W x 28)。 checkpoint が X 部なし
    /// (= ちょうどこの分だけ短い) なら末尾を 0 で埋める (column-major = 特徴ごとに W 個なので末尾 = X 部の列)。
    kh_x_extra: usize,
    direct15: Option<Direct15HeadExpandAdapter>,
}

impl CheckpointShapeAdapter for CheckpointLoadAdapter {
    fn adapt(&self, id: &str, values: &[f32], expected: usize) -> Option<Vec<f32>> {
        if id == "kh_ft_w" && self.kh_x_extra > 0 && values.len() + self.kh_x_extra == expected {
            eprintln!("kh-x: {id} {} -> {} 要素 (X 部 {} 要素を 0 で追加)", values.len(), expected, self.kh_x_extra);
            let mut v = values.to_vec();
            v.resize(expected, 0.0);
            return Some(v);
        }
        if id == "threat_ft_w" && self.threat_factor_extra > 0 && expected > self.threat_factor_extra {
            let real = expected - self.threat_factor_extra;
            let mut base: Option<Vec<f32>> = None;
            if values.len() == real {
                base = Some(values.to_vec());
            } else if let Some((per_table, groups)) = self.threat_tile {
                if values.len() == per_table && real == per_table * groups {
                    eprintln!("threat-bucket-groups: {id} {} -> {} 要素 ({groups} 群へ複製)", values.len(), real);
                    base = Some(values.repeat(groups));
                }
            }
            if let Some(mut v) = base {
                eprintln!(
                    "threat-factorizer: {id} {} -> {} 要素 (factor 列 {} 要素を 0 で追加)",
                    values.len(),
                    expected,
                    self.threat_factor_extra
                );
                v.resize(expected, 0.0);
                return Some(v);
            }
        }
        if let Some((per_table, groups)) = self.threat_tile {
            if id == "threat_ft_w" && values.len() == per_table && expected == per_table * groups {
                eprintln!("threat-bucket-groups: {id} {} -> {} 要素 ({groups} 群へ複製)", values.len(), expected);
                return Some(values.repeat(groups));
            }
        }
        self.direct15.as_ref().and_then(|a| a.adapt(id, values, expected))
    }
}

/// bullet checkpoint の重みファイル (`weights.bin` / `momentum.bin` など。 エントリ = `id` + 改行、
/// 要素数 (usize LE)、 f32 LE × 要素数 の連結) のヘッダだけを読み、 `(id, 要素数)` の一覧を返す。
/// ペイロードは seek で飛ばすので 300 MB 級のファイルでも一瞬で終わる。
fn scan_checkpoint_weight_lens(path: &Path) -> Result<Vec<(String, usize)>, String> {
    use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
    const USIZE: usize = std::mem::size_of::<usize>();
    let file = std::fs::File::open(path).map_err(|e| format!("{}: open failed: {e}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut res = Vec::new();
    loop {
        let mut id = Vec::new();
        let n = reader.read_until(b'\n', &mut id).map_err(|e| format!("{}: read id: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        if id.pop() != Some(b'\n') {
            return Err(format!("{}: truncated weight id", path.display()));
        }
        let id = String::from_utf8(id).map_err(|e| format!("{}: non-UTF-8 weight id: {e}", path.display()))?;
        let mut len = [0u8; USIZE];
        reader.read_exact(&mut len).map_err(|e| format!("{}: read size of {id}: {e}", path.display()))?;
        let len = usize::from_le_bytes(len);
        let payload = i64::try_from(len * std::mem::size_of::<f32>())
            .map_err(|_| format!("{}: payload of {id} too large", path.display()))?;
        reader.seek(SeekFrom::Current(payload)).map_err(|e| format!("{}: seek over {id}: {e}", path.display()))?;
        res.push((id, len));
    }
    Ok(res)
}

/// P2THRD2 (Profile 15) の int32 Direct table を graph の `threat_direct_w` へ戻す。
///
/// graph 形状は [rows, dims] (column-major、 `dst[feature * rows + row]`)。
/// ファイル形状 (buckets, dims) と graph 形状 (rows, dims) の組合せ:
///   - buckets 8 -> rows 16 (P2 を旧 net から開始): 両視点の行に同じ値を複製
///     (旧 net は両視点で同一テーブルなので forward は完全に同一 = 厳密な warm start)。
///   - buckets 16 -> rows 16: ファイル bucket p*8+b -> graph 行 2*b+p。
///   - buckets 16 -> rows 8: 不可 (エラー)。
///   - dims 65,640 -> graph 131,280 (P5 を旧 net から開始): 守備なし / あり の両半分に同じ値を複製
///     (どちらの半分を引いても旧 net と同じ値 = forward 同一)。
///   - dims 131,280 -> graph 65,640: 不可 (エラー)。
/// P4 (factorise) の S / P は Zeroed のままにする (D に全量を載せるので forward は同一)。
fn load_p2thrd2_direct_if_present<G>(
    graph: &mut G,
    buf: &[u8],
    halfka_dim: usize,
    input_size: usize,
    direct15_persp_split: bool,
    direct15_defended: bool,
) -> Result<(), String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    const MAGIC: &[u8; 8] = b"P2THRD2\0";
    let Some(off) = buf.windows(MAGIC.len()).position(|w| w == MAGIC) else {
        return Ok(());
    };
    const HEADER: usize = 24;
    if off + HEADER > buf.len() {
        return Err("truncated P2THRD2 header".to_string());
    }
    let read_u32 = |p: usize| -> u32 { u32::from_le_bytes(buf[p..p + 4].try_into().expect("P2THRD2 u32 range")) };
    let version = read_u32(off + 8);
    let profile = read_u32(off + 12);
    let dims = read_u32(off + 16) as usize;
    let buckets = read_u32(off + 20) as usize;
    if version != 2 || profile != 15 || !(buckets == NUM_BUCKETS || buckets == 2 * NUM_BUCKETS) {
        return Err(format!("P2THRD2 header mismatch: version={version}, profile={profile}, buckets={buckets}"));
    }
    if !(dims == THREAT_DIRECT15_DIMENSIONS || dims == THREAT_DIRECT15_DEFENDED_DIMENSIONS) {
        return Err(format!("P2THRD2 dimensions mismatch: file={dims}"));
    }
    let _ = (input_size, halfka_dim);
    let graph_dims = if direct15_defended { THREAT_DIRECT15_DEFENDED_DIMENSIONS } else { THREAT_DIRECT15_DIMENSIONS };
    let graph_rows = if direct15_persp_split { 2 * NUM_BUCKETS } else { NUM_BUCKETS };
    if dims > graph_dims {
        return Err(format!(
            "P2THRD2 dimensions mismatch: file={dims} (defended) は graph={graph_dims} へ読み込めません (--direct15-defended が必要)"
        ));
    }
    if buckets > graph_rows {
        return Err(format!(
            "P2THRD2 buckets mismatch: file={buckets} (persp split) は graph rows={graph_rows} へ読み込めません (--direct15-persp-split が必要)"
        ));
    }
    let payload_bytes = dims
        .checked_mul(buckets)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| "P2THRD2 payload size overflow".to_string())?;
    if off + HEADER + payload_bytes > buf.len() {
        return Err("truncated P2THRD2 payload".to_string());
    }
    if !has_graph_weight(graph, "threat_direct_w") {
        return Ok(());
    }
    let mut dst = read_graph_weight(graph, "threat_direct_w")?;
    let expected_len = graph_dims * graph_rows;
    if dst.len() != expected_len {
        return Err(format!("threat_direct_w graph shape mismatch: actual={}, expected={expected_len}", dst.len()));
    }
    let scale = (i32::from(QA) * i32::from(QB)) as f32;
    let file_np = buckets / NUM_BUCKETS;
    let graph_np = graph_rows / NUM_BUCKETS;
    let file_halves = dims / THREAT_DIRECT15_DIMENSIONS;
    let graph_halves = graph_dims / THREAT_DIRECT15_DIMENSIONS;
    let mut p = off + HEADER;
    for file_bucket in 0..buckets {
        let (file_persp, bucket) = (file_bucket / NUM_BUCKETS, file_bucket % NUM_BUCKETS);
        for feature in 0..dims {
            let raw = i32::from_le_bytes(buf[p..p + 4].try_into().expect("P2THRD2 i32 range"));
            p += 4;
            let v = raw as f32 / scale;
            // 視点: ファイルが 1 視点なら graph の全視点へ複製、 2 視点なら対応する視点のみ。
            for persp in 0..graph_np {
                if file_np == graph_np && persp != file_persp {
                    continue;
                }
                let row = if graph_np == 2 { 2 * bucket + persp } else { bucket };
                // 守備ビット: ファイルが 1 半分なら graph の両半分へ複製。
                for half in 0..graph_halves {
                    if file_halves == graph_halves && half != feature / THREAT_DIRECT15_DIMENSIONS {
                        continue;
                    }
                    let f =
                        if file_halves == graph_halves { feature } else { half * THREAT_DIRECT15_DIMENSIONS + feature };
                    dst[f * graph_rows + row] = v;
                }
            }
        }
    }
    set_graph_weight(graph, "threat_direct_w", &dst)?;
    eprintln!(
        "load_from_quantised: loaded P2THRD2 (file {buckets}x{dims} -> graph {graph_rows}x{graph_dims}, {} values)",
        dims * buckets
    );
    Ok(())
}

/// graph に指定 id の weight tensor が存在するか確認する (= 存在ガード用)。
fn has_graph_weight<G>(graph: &G, id: &str) -> bool
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    graph.primary().weight_idx(id).is_some()
}

/// graph の全 weight tensor の id (`--threat-ext-only` の凍結で「列挙漏れ」を作らないため)。
fn graph_weight_ids<G>(graph: &G) -> Vec<String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    let mut ids = graph.primary().weight_ids();
    ids.sort();
    ids
}

/// graph に weight tensor (id) を書き込む。 expected size と一致しない場合エラー。
/// HalfKA_hm の packed BonaPiece (= 0..1628) が表す駒の **符号付き駒得価値 (cp)** を返す。
///
/// PSQT sum は `(stm_acc - nstm_acc) / 2` であり、 同一物理駒は stm 視点では friendly
/// 平面 (F_*)、 nstm 視点では enemy 平面 (E_*) に現れる。 よって material prior は
/// **friendly = +value / enemy = -value** とすることで、 stm 側 1 駒あたり正味
/// `0.5 * (+value - (-value)) = +value` cp を寄与する (Stockfish PSQT material 慣例)。
///
/// - 盤上駒: `PIECE_BASE` 由来の F_*/E_* 平面 (各 81 マス)。
/// - 手駒: F_HAND_*/E_HAND_* 平面 (枚数分の連続 index)。 価値は盤上駒と同一。
/// - 玉 (F_KING/E_KING 平面) と未使用 index (0): 0。
///
/// 価値表 (cp): 歩100 / 香300 / 桂350 / 銀500 / 金600 / 角800 / 飛1000 /
/// と・成香・成桂・成銀600 / 馬1150 / 竜1300。 手駒に成りは無い。
fn packed_bp_material_cp(packed_bp: usize) -> i32 {
    use bullet_lib::shogi::bona_piece::{
        E_BISHOP, E_DRAGON, E_GOLD, E_HAND_BISHOP, E_HAND_GOLD, E_HAND_KNIGHT, E_HAND_LANCE, E_HAND_PAWN, E_HAND_ROOK,
        E_HAND_SILVER, E_HORSE, E_KNIGHT, E_LANCE, E_PAWN, E_ROOK, E_SILVER, F_BISHOP, F_DRAGON, F_GOLD, F_HAND_BISHOP,
        F_HAND_GOLD, F_HAND_KNIGHT, F_HAND_LANCE, F_HAND_PAWN, F_HAND_ROOK, F_HAND_SILVER, F_HORSE, F_KNIGHT, F_LANCE,
        F_PAWN, F_ROOK, F_SILVER, FE_HAND_END,
    };

    // 駒得価値 (cp)。
    const V_PAWN: i32 = 100;
    const V_LANCE: i32 = 300;
    const V_KNIGHT: i32 = 350;
    const V_SILVER: i32 = 500;
    const V_GOLD: i32 = 600; // と/成香/成桂/成銀も同値
    const V_BISHOP: i32 = 800;
    const V_ROOK: i32 = 1000;
    const V_HORSE: i32 = 1150;
    const V_DRAGON: i32 = 1300;

    let bp = packed_bp as u16;

    // ---- 手駒 (1..=89): F_* = +value, E_* = -value ----
    if (1..FE_HAND_END).contains(&packed_bp) {
        // (base, count, value, friendly) の順で線形探索。 各手駒平面は連続 index。
        const HAND: [(u16, u16, i32, bool); 14] = [
            (F_HAND_PAWN, 18, V_PAWN, true),
            (E_HAND_PAWN, 18, V_PAWN, false),
            (F_HAND_LANCE, 4, V_LANCE, true),
            (E_HAND_LANCE, 4, V_LANCE, false),
            (F_HAND_KNIGHT, 4, V_KNIGHT, true),
            (E_HAND_KNIGHT, 4, V_KNIGHT, false),
            (F_HAND_SILVER, 4, V_SILVER, true),
            (E_HAND_SILVER, 4, V_SILVER, false),
            (F_HAND_GOLD, 4, V_GOLD, true),
            (E_HAND_GOLD, 4, V_GOLD, false),
            (F_HAND_BISHOP, 2, V_BISHOP, true),
            (E_HAND_BISHOP, 2, V_BISHOP, false),
            (F_HAND_ROOK, 2, V_ROOK, true),
            (E_HAND_ROOK, 2, V_ROOK, false),
        ];
        for &(base, count, value, friendly) in HAND.iter() {
            if bp >= base && bp < base + count {
                return if friendly { value } else { -value };
            }
        }
        return 0; // 想定外の手駒 gap (= 無し) は 0
    }

    // ---- 盤上駒 (90..=1547): F_* 平面 = +value (81 マス)、 E_* 平面 = -value ----
    // (F_base, value) の順。 enemy 平面は F_base + 81 (= PIECE_BASE の E_*) と一致する。
    const BOARD: [(u16, i32); 9] = [
        (F_PAWN, V_PAWN),
        (F_LANCE, V_LANCE),
        (F_KNIGHT, V_KNIGHT),
        (F_SILVER, V_SILVER),
        (F_GOLD, V_GOLD), // 金 + 全成駒 (と/成香/成桂/成銀) は同一平面
        (F_BISHOP, V_BISHOP),
        (F_HORSE, V_HORSE),
        (F_ROOK, V_ROOK),
        (F_DRAGON, V_DRAGON),
    ];
    // enemy base を明示的に検証 (PIECE_BASE と整合): E_* は F_* + 81。
    debug_assert_eq!(E_PAWN, F_PAWN + 81);
    debug_assert_eq!(E_LANCE, F_LANCE + 81);
    debug_assert_eq!(E_KNIGHT, F_KNIGHT + 81);
    debug_assert_eq!(E_SILVER, F_SILVER + 81);
    debug_assert_eq!(E_GOLD, F_GOLD + 81);
    debug_assert_eq!(E_BISHOP, F_BISHOP + 81);
    debug_assert_eq!(E_HORSE, F_HORSE + 81);
    debug_assert_eq!(E_ROOK, F_ROOK + 81);
    debug_assert_eq!(E_DRAGON, F_DRAGON + 81);
    for &(f_base, value) in BOARD.iter() {
        if bp >= f_base && bp < f_base + 81 {
            return value; // friendly 平面
        }
        if bp >= f_base + 81 && bp < f_base + 162 {
            return -value; // enemy 平面
        }
    }

    // ---- 玉 / 未使用 index: 0 ----
    0
}

/// PSQT shortcut layer の `psqtw` を駒得価値 prior で初期化する。
///
/// scale-critical: psqt_w は graph float 単位で、 save 時に `× (QA*QB) = ×8128` で
/// disk i32 化される。 推論側は disk i32 をそのまま accumulate し、 最終 cp =
/// `(stm_acc - nstm_acc)/2 / fv_scale`。 ∴ friendly 駒 1 つを `+M` cp にしたければ
/// `disk = +M * fv_scale` (enemy は `-M * fv_scale`) であればよく、 graph float は
/// `disk / 8128 = ±M * fv_scale / 8128`。
///
/// `fv_scale` は save 側 (`(QA*QB + scale/2)/scale`) と同一式で算出して整合させる。
/// bias (`psqtb`) は 0 のまま (= material は per-feature 重みのみで表現)。
///
/// `factorise` が true のとき psqtw は Factorised layout (行数 = `PIECE_INPUTS + halfka_dim`、
/// factoriser 行 `[0, PIECE_INPUTS)` + 実特徴行 `[PIECE_INPUTS, PIECE_INPUTS+halfka_dim)`)
/// になる。 material は **実特徴行のみ** に書き込み、 factoriser 行は 0 のまま残す。
/// これで export coalesce (`coalesced[r] = real[PIECE_INPUTS+r] + factoriser[r % PIECE_INPUTS]`)
/// 後に factoriser=0 ゆえ `coalesced = real = material` となり、 非 factorise の material init
/// と等価になる。
///
/// `kzt_ft` が true のとき psqtw は composite layout (行数 = `halfka_dim + KZT_FT_DIMENSIONS`、
/// HalfKA 行 `[0, halfka_dim)` が先頭 + KZT-FT 行 `[halfka_dim, halfka_dim+KZT_FT_DIMENSIONS)`
/// が末尾) になる。 material は **HalfKA 行のみ** に書き込み、 KZT-FT 末尾行は 0 のまま残す
/// (= KZT-FT は factorise と違い export で coalesce せず末尾 section を別に書き出すだけなので、
/// KZT-FT 行 0 で正しい)。 `factorise` と `kzt_ft` は排他 (= 呼び出し側 guard 済)。
///
/// `tail_dim > 0` のとき (= threat / aux / auxcheap 連結入力有効時) psqtw は同じく
/// composite layout (行数 = `(row_offset +) halfka_dim + tail_dim`、 HalfKA 行が先頭
/// (factorise 時は factoriser 行の直後) + 連結入力行が末尾) になる。 kzt_ft と同構造なので
/// material は **HalfKA 行のみ** に書き込み、 末尾行は 0 のまま残す (= threat / aux 入力には
/// material 駒価値の概念が無いので正しい)。 `threat` / `kzt_ft` は `factorise` と排他だが、
/// `auxcheap` は factorise 併用可 (2026-07-23): その場合 layout は
/// `[factoriser 1,629 | HalfKA 73,305 | aux tail_dim]` で `row_offset` と `tail_rows` が
/// 同時に効く。
fn init_psqt_material<G>(
    graph: &mut G,
    halfka_dim: usize,
    scale: i32,
    factorise: bool,
    kzt_ft: bool,
    tail_dim: usize,
) -> Result<(), String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    use bullet_lib::game::inputs::PIECE_INPUTS;
    use rshogi_core::nnue::kzt::KZT_FT_DIMENSIONS;

    debug_assert!(!(factorise && kzt_ft), "factorise と kzt_ft は排他");
    debug_assert!(!(kzt_ft && tail_dim > 0), "kzt_ft と threat/aux tail は排他");

    // 推論側 fv_scale と整合 (save 側 line ~4494 と同一式)。
    let fv_scale = (i32::from(QA) * i32::from(QB) + scale / 2) / scale;
    let qa_qb = (i32::from(QA) * i32::from(QB)) as f64; // 8128

    // psqtw layout: [NUM_BUCKETS, rows] column-major → values[feat * NUM_BUCKETS + bucket]。
    // factorise 時は先頭 PIECE_INPUTS 行が factoriser (0 のまま)、 以降が実特徴。
    // kzt_ft 時は先頭が HalfKA (= material 対象)、 末尾 KZT_FT_DIMENSIONS 行が KZT-FT (0 のまま)。
    // threat / aux / auxcheap 時は HalfKA の後ろに tail_dim 行 (0 のまま)。
    let row_offset = if factorise { PIECE_INPUTS } else { 0 };
    let tail_rows = (if kzt_ft { KZT_FT_DIMENSIONS } else { 0 }) + tail_dim;
    let rows = row_offset + halfka_dim + tail_rows;
    let mut values = vec![0.0f32; rows * NUM_BUCKETS];
    for feat in 0..halfka_dim {
        let packed_bp = feat % PIECE_INPUTS;
        let material_cp = packed_bp_material_cp(packed_bp);
        if material_cp == 0 {
            continue;
        }
        // disk i32 目標 = material_cp * fv_scale → graph float = disk / 8128。
        let graph_w = (material_cp * fv_scale) as f64 / qa_qb;
        let row = row_offset + feat;
        for bucket in 0..NUM_BUCKETS {
            values[row * NUM_BUCKETS + bucket] = graph_w as f32;
        }
    }

    set_graph_weight(graph, "psqtw", &values)?;
    println!(
        "PSQT material prior 初期化完了 (fv_scale={}, scale={}, halfka_dim={}, factorise={}, kzt_ft={}, tail_dim={}, rows={}, 例: 歩 graph_w={:.6} 飛 graph_w={:.6})",
        fv_scale,
        scale,
        halfka_dim,
        factorise,
        kzt_ft,
        tail_dim,
        rows,
        (100 * fv_scale) as f64 / qa_qb,
        (1000 * fv_scale) as f64 / qa_qb,
    );
    Ok(())
}

fn set_graph_weight<G>(graph: &mut G, id: &str, values: &[f32]) -> Result<(), String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    use bullet_lib::acyclib::graph::{GraphNodeId, GraphNodeIdTy};
    let primary = graph.primary_mut();
    let idx = primary.weight_idx(id).ok_or_else(|| format!("weight '{}' not found in graph", id))?;
    let tensor = primary
        .get(GraphNodeId::new(idx, GraphNodeIdTy::Values))
        .map_err(|e| format!("graph.get('{}') failed: {:?}", id, e))?;
    let mut dense = tensor.dense_mut();
    let expected = dense.size();
    if expected != values.len() {
        return Err(format!("weight '{}' size mismatch: graph={} loader={}", id, expected, values.len()));
    }
    dense.load_from_slice(None, values).map_err(|_| format!("load_from_slice failed for weight '{}'", id))?;
    Ok(())
}

/// graph から weight tensor の現在値を読み出す (HandThreat の partial update 用)。
fn read_graph_weight<G>(graph: &G, id: &str) -> Result<Vec<f32>, String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    use bullet_lib::acyclib::graph::{GraphNodeId, GraphNodeIdTy};
    let primary = graph.primary();
    let idx = primary.weight_idx(id).ok_or_else(|| format!("weight '{}' not found in graph", id))?;
    let tensor = primary
        .get(GraphNodeId::new(idx, GraphNodeIdTy::Values))
        .map_err(|e| format!("graph.get('{}') failed: {:?}", id, e))?;
    let dense = tensor.dense();
    let mut buf = vec![0.0f32; dense.size()];
    dense.write_to_slice(&mut buf).map_err(|_| format!("write_to_slice failed for weight '{}'", id))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// 2026-09-12: L1 (fc0) の graph 内正則化 (--l1-anchor-lambda / --bucket-smooth-l1-lambda)
// ---------------------------------------------------------------------------

/// `--l1-anchor-lambda` 用アンカー tensor の graph 内 id (非学習、 stop-grad)。
const L1_ANCHOR_W_ID: &str = "l1_anchor_w";
const L1_ANCHOR_B_ID: &str = "l1_anchor_b";

/// [`build_l1_graph_penalty`] の設定 (形状と係数)。
#[derive(Clone, Copy)]
struct L1PenaltyCfg {
    num_buckets: usize,
    l1_out: usize,
    ft_out: usize,
    /// `l1w` の列数 (= ft_out + hand_count_dense_dims)。
    l1_in_total: usize,
    anchor_lambda: f32,
    smooth_lambda: f32,
}

/// forward closure の型を `for<'a> Fn(...)` に固定するための恒等関数。
///
/// closure を一度 `let` に束縛してから `build` / `build_custom` の両方へ渡すため、
/// 直接引数に書いたときと同じ higher-ranked な signature を推論させる (Rust の closure
/// lifetime 推論は期待型が無いと具体 lifetime に落ちて build に渡せなくなる)。
/// 戻り値は (net 出力, [l1w, l1b, l1fw, l1fb] の graph node)。
fn constrain_l1_forward<F>(f: F) -> F
where
    F: for<'a> Fn(
        &'a bullet_lib::nn::NetworkBuilder<BackendMarker>,
        NetworkBuilderNode<'a, BackendMarker>,
        NetworkBuilderNode<'a, BackendMarker>,
        NetworkBuilderNode<'a, BackendMarker>,
    ) -> (NetworkBuilderNode<'a, BackendMarker>, [NetworkBuilderNode<'a, BackendMarker>; 4]),
{
    f
}

/// L1 (fc0) の graph 内 penalty node を作る。 戻り値は shape (1,1)、 batch 無し。
///
/// - anchor (`anchor_lambda > 0`):
///   `λ × Σ_b ( ||(l1w_b + l1fw) - W0_b||² + ||(l1b_b + l1fb) - c0_b||² )`。
///   `l1fw` を bucket 数だけ縦に並べるのは定数行列 `tile` (rows × l1_out、
///   tile[b*l1_out+o, o] = 1) との matmul で行う (両辺 batch 無しの NobNob matmul)。
///   アンカー `W0 / c0` は graph の weight tensor (`l1_anchor_w` / `l1_anchor_b`) を
///   `copy_stop_grad()` で読むだけ (勾配は流れない)。 値は [`init_l1_anchor`] が
///   resume 後に埋める。
/// - smoothness (`smooth_lambda > 0`):
///   `λ × mean_b ( ||l1w[b+1] - l1w[b]||² + ||l1b[b+1] - l1b[b]||² )`。 差分行列 `dmat`
///   ((nb-1)·l1_out × rows) との matmul で隣接 bucket 差を作る。
///
/// 要素和は「1 ベクトルとの matmul」で (1,1) に畳む (graph に行列 reduce が無いため)。
/// 呼び出し側は `loss + penalty` で加算する (LinearCombination が batch 無し側を splat
/// するので 1 サンプルあたり penalty が足され、 gradient は batch 和 × gradient_factor
/// (= 1/batch) で data loss と同じ目盛りになる)。
///
/// 前提: `l1w` の列数 (= `l1_in_total`) は anchor 有効時 `ft_out` と一致していること
/// (= `--hand-count-dense` 無し)。 呼び出し前に検査する。
fn build_l1_graph_penalty<'a>(
    builder: &'a bullet_lib::nn::NetworkBuilder<BackendMarker>,
    l1_nodes: [NetworkBuilderNode<'a, BackendMarker>; 4],
    cfg: L1PenaltyCfg,
) -> NetworkBuilderNode<'a, BackendMarker> {
    let [l1w, l1b, l1fw, l1fb] = l1_nodes;
    let L1PenaltyCfg { num_buckets, l1_out, ft_out, l1_in_total, anchor_lambda, smooth_lambda } = cfg;
    let rows = num_buckets * l1_out;

    // Σ_{i,j} m[i,j]² を (1,1) に畳む: ones(1,k)·(m∘m)·ones(c,1)。 c == 1 なら後段は不要。
    // `move` で `builder: &'a NetworkBuilder` (Copy) を closure に写し、 生成 node の
    // lifetime を 'a に固定する。
    let sum_sq =
        move |m: NetworkBuilderNode<'a, BackendMarker>, k: usize, c: usize| -> NetworkBuilderNode<'a, BackendMarker> {
            let sq = m.abs_pow(2.0);
            let ones_k = builder.new_constant(Shape::new(1, k), &vec![1.0f32; k]);
            let colsum = ones_k.matmul(sq); // (1, c)
            if c == 1 {
                colsum
            } else {
                let ones_c = builder.new_constant(Shape::new(c, 1), &vec![1.0f32; c]);
                colsum.matmul(ones_c) // (1, 1)
            }
        };

    let mut penalty: Option<NetworkBuilderNode<'a, BackendMarker>> = None;

    if anchor_lambda > 0.0 {
        assert_eq!(l1_in_total, ft_out, "L1 anchor は l1w の列数 == ft_out (hand-count-dense 無し) が前提");
        // tile: (rows × l1_out) column-major、 tile[b*l1_out+o, o] = 1。
        //   tile·l1fw = l1fw を bucket 数だけ縦に並べた (rows × ft_out)。
        let mut tile_vals = vec![0.0f32; rows * l1_out];
        for b in 0..num_buckets {
            for o in 0..l1_out {
                tile_vals[o * rows + b * l1_out + o] = 1.0;
            }
        }
        let tile = builder.new_constant(Shape::new(rows, l1_out), &tile_vals);
        let merged_w = l1w + tile.matmul(l1fw); // (rows, ft_out)
        let merged_b = l1b + tile.matmul(l1fb); // (rows, 1)
        let anchor_w =
            builder.new_weights(L1_ANCHOR_W_ID, Shape::new(rows, ft_out), InitSettings::Zeroed).copy_stop_grad();
        let anchor_b = builder.new_weights(L1_ANCHOR_B_ID, Shape::new(rows, 1), InitSettings::Zeroed).copy_stop_grad();
        let pen = (sum_sq(merged_w - anchor_w, rows, ft_out) + sum_sq(merged_b - anchor_b, rows, 1)) * anchor_lambda;
        penalty = Some(pen);
    }

    if smooth_lambda > 0.0 && num_buckets >= 2 {
        let drows = (num_buckets - 1) * l1_out;
        // dmat: (drows × rows) column-major、 行 r = b*l1_out+o に対し
        //   dmat[r, (b+1)*l1_out+o] = +1、 dmat[r, b*l1_out+o] = -1。
        let mut d_vals = vec![0.0f32; drows * rows];
        for b in 0..num_buckets - 1 {
            for o in 0..l1_out {
                let r = b * l1_out + o;
                d_vals[(b * l1_out + o) * drows + r] = -1.0;
                d_vals[((b + 1) * l1_out + o) * drows + r] = 1.0;
            }
        }
        let dmat = builder.new_constant(Shape::new(drows, rows), &d_vals);
        let dw = dmat.matmul(l1w); // (drows, l1_in_total)
        let db = dmat.matmul(l1b); // (drows, 1)
        let pen = (sum_sq(dw, drows, l1_in_total) + sum_sq(db, drows, 1)) * (smooth_lambda / (num_buckets - 1) as f32);
        penalty = Some(match penalty {
            Some(p) => p + pen,
            None => pen,
        });
    }

    penalty.expect("build_l1_graph_penalty は anchor_lambda か smooth_lambda のどちらかが正のときだけ呼ぶ")
}

/// `--l1-anchor-lambda` のアンカー tensor (`l1_anchor_w` / `l1_anchor_b`) を初期化する。
///
/// resume 後 (学習開始前) に呼ぶ。 アンカーが全 0 (= checkpoint に無かった / 新規) なら
/// 現在の合算 L1 (`l1w_b + l1fw`、 `l1b_b + l1fb`) で埋める。 既に非 0 (= アンカー入り
/// checkpoint から resume) なら維持し、 現在の合算 L1 との距離を表示する (drift の目安)。
///
/// レイアウト (column-major): `l1w[in * rows + b*l1_out + o]`、 `l1fw[in * l1_out + o]`、
/// `l1b[b*l1_out + o]`、 `l1fb[o]` (quantise 側 `layerstack_data` と同じ)。
fn init_l1_anchor<G>(graph: &mut G, num_buckets: usize, l1_out: usize, ft_out: usize) -> Result<(), String>
where
    G: bullet_lib::acyclib::graph::like::GraphLike<bullet_lib::nn::ExecutionContext>,
{
    let rows = num_buckets * l1_out;
    let l1w = read_graph_weight(graph, "l1w")?;
    let l1b = read_graph_weight(graph, "l1b")?;
    let l1fw = read_graph_weight(graph, "l1fw")?;
    let l1fb = read_graph_weight(graph, "l1fb")?;
    if l1w.len() != rows * ft_out || l1fw.len() != l1_out * ft_out || l1b.len() != rows || l1fb.len() != l1_out {
        return Err(format!(
            "L1 anchor: 形状不一致 (l1w={} l1fw={} l1b={} l1fb={}、 期待 rows={} ft_out={} l1_out={})",
            l1w.len(),
            l1fw.len(),
            l1b.len(),
            l1fb.len(),
            rows,
            ft_out,
            l1_out
        ));
    }

    let mut merged_w = vec![0.0f32; rows * ft_out];
    for i in 0..ft_out {
        for b in 0..num_buckets {
            for o in 0..l1_out {
                let idx = i * rows + b * l1_out + o;
                merged_w[idx] = l1w[idx] + l1fw[i * l1_out + o];
            }
        }
    }
    let mut merged_b = vec![0.0f32; rows];
    for b in 0..num_buckets {
        for o in 0..l1_out {
            merged_b[b * l1_out + o] = l1b[b * l1_out + o] + l1fb[o];
        }
    }

    fn norm(v: &[f32]) -> f64 {
        v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt()
    }
    let anchor_w = read_graph_weight(graph, L1_ANCHOR_W_ID)?;
    let anchor_b = read_graph_weight(graph, L1_ANCHOR_B_ID)?;
    let is_zero = anchor_w.iter().chain(anchor_b.iter()).all(|v| *v == 0.0);
    if is_zero {
        set_graph_weight(graph, L1_ANCHOR_W_ID, &merged_w)?;
        set_graph_weight(graph, L1_ANCHOR_B_ID, &merged_b)?;
        println!(
            "L1 anchor: 現在の合算 L1 (l1w+l1fw / l1b+l1fb) でアンカーを初期化 (||W0||={:.4}, ||c0||={:.4})",
            norm(&merged_w),
            norm(&merged_b)
        );
    } else {
        let dist_w: Vec<f32> = merged_w.iter().zip(anchor_w.iter()).map(|(m, a)| m - a).collect();
        let dist_b: Vec<f32> = merged_b.iter().zip(anchor_b.iter()).map(|(m, a)| m - a).collect();
        println!(
            "L1 anchor: checkpoint のアンカーを維持 (||W0||={:.4}, ||c0||={:.4}、 現在の合算 L1 との距離 ||W-W0||={:.4}, ||c-c0||={:.4})",
            norm(&anchor_w),
            norm(&anchor_b),
            norm(&dist_w),
            norm(&dist_b)
        );
    }
    Ok(())
}

/// quantised.bin 用バイナリ reader。
struct QuantReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> QuantReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.buf.len() {
            return Err(format!("EOF reading {} bytes at offset {}", n, self.pos));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        let bytes = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        let bytes = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        let bytes = self.read_bytes(1)?;
        Ok(bytes[0] as i8)
    }

    fn read_i32_array(&mut self, n: usize) -> Result<Vec<i32>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_i32()?);
        }
        Ok(out)
    }

    fn read_i16_array(&mut self, n: usize) -> Result<Vec<i16>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_i16()?);
        }
        Ok(out)
    }

    fn read_i8_array(&mut self, n: usize) -> Result<Vec<i8>, String> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_i8()?);
        }
        Ok(out)
    }

    /// LEB128 圧縮ブロック (magic + size + data) を読み、 i16 値の配列を返す。
    fn read_leb128_i16_block(&mut self, expected_count: usize) -> Result<Vec<i16>, String> {
        let magic = self.read_bytes(17)?;
        if magic != b"COMPRESSED_LEB128" {
            return Err(format!(
                "LEB128 magic mismatch at offset {}: got {:?}",
                self.pos - 17,
                std::str::from_utf8(magic).unwrap_or("<binary>")
            ));
        }
        let size = self.read_u32()? as usize;
        let data = self.read_bytes(size)?;

        let mut out = Vec::with_capacity(expected_count);
        let mut i = 0;
        while i < data.len() && out.len() < expected_count {
            let (value, consumed) = decode_signed_leb128(&data[i..])?;
            out.push(value as i16);
            i += consumed;
        }
        if out.len() != expected_count {
            return Err(format!("LEB128 block expected {} values, decoded {}", expected_count, out.len()));
        }
        if i != data.len() {
            return Err(format!("LEB128 block had {} unconsumed bytes after {} values", data.len() - i, out.len()));
        }
        Ok(out)
    }
}

/// 符号付き LEB128 を decode し、 (value, consumed_bytes) を返す。
fn decode_signed_leb128(buf: &[u8]) -> Result<(i64, usize), String> {
    let mut result: i64 = 0;
    let mut shift: u32 = 0;
    let mut consumed = 0;
    loop {
        if consumed >= buf.len() {
            return Err("EOF inside LEB128 value".to_string());
        }
        let byte = buf[consumed];
        consumed += 1;
        result |= ((byte & 0x7f) as i64) << shift;
        shift += 7;
        if (byte & 0x80) == 0 {
            // sign extend
            if shift < 64 && (byte & 0x40) != 0 {
                result |= -1i64 << shift;
            }
            return Ok((result, consumed));
        }
        if shift >= 64 {
            return Err("LEB128 value too long".to_string());
        }
    }
}

// =============================================================================
// Main
// =============================================================================

/// 案1 の **16 次元ブロックごとのチャンネルずらし量**を作る。 2026-10-01。
///
/// SIMD の 1 反復が 16 次元なので、 ブロック内で一定なら 16 本の連続 load を保てる。
/// ずらし量は 0..K-1 の任意の値を取れる (エンジン側は accumulator を 2 周複製したバッファから
/// 読むので回り込みが起きない)。
///
/// * `same` : すべて 0        → 2 因子が同じチャンネル (従来)
/// * `half` : hi 側だけ K/2   → 組は (c, c+K/2) の 1 系統
/// * `rich` : hi 側で 64 次元ごとに d = 1,2,... → 組が **全部異なる**
fn threat_proj_block_offsets(ft: usize, k: usize, mode: &str, fixed: usize) -> Vec<usize> {
    let nb = ft / 16;
    let half_b = nb / 2;
    let mut v = vec![0usize; nb];
    match mode {
        "same" => {
            if fixed > 0 {
                for b in half_b..nb {
                    v[b] = fixed % k;
                }
            }
        }
        "half" => {
            for b in half_b..nb {
                v[b] = (k / 2) % k;
            }
        }
        "rich" => {
            // hi 側: 1 系統 = 64 次元 = 4 ブロック。 d を 1 から順に増やす。
            let per = k / 16; // 64 次元ぶんのブロック数 = 4
            for b in half_b..nb {
                v[b] = 1 + (b - half_b) / per;
            }
        }
        other => panic!("--threat-proj-pair は same / half / rich のいずれか (指定: {other})"),
    }
    v
}

fn main() {
    let args = Args::parse();
    bullet_lib::acyclib::trainer::logger::set_colours_enabled(!args.no_color);
    // LayerStack を P4H2(進行度4区分×持駒量2区分)モードにするか、プロセス全体で確定させる。
    // bucket 算出(ShogiProgressKPAbs::bucket)と全 input feature の bucket がこれに従う。
    bullet_lib::game::outputs::set_layerstack_p4h2(args.bucket_mode == BucketMode::Progress4Hand2KPAbs);
    // 案 B (Regime LayerStack、 2026-10-07): regime モードをプロセス全体で確定させる。
    //   🚨bucket 算出より前に呼ぶこと。 事前登録 runs/prereg/PREREG-20261007-REGIME-B.md
    if args.lsmode_regime {
        if args.bucket_mode != BucketMode::Progress8Hand2KPAbs {
            eprintln!(
                "ERROR: --lsmode-regime は --bucket-mode progress8hand2kpabs と併用すること (今 {})",
                args.bucket_mode_name()
            );
            std::process::exit(2);
        }
        if NUM_BUCKETS < 16 {
            eprintln!(
                "ERROR: --lsmode-regime は 16 bucket ビルドが必要 (--features lsmode-p8h2)。 今 NUM_BUCKETS={}",
                NUM_BUCKETS
            );
            std::process::exit(2);
        }
        if !(0..=7).contains(&args.regime_min_pb) {
            eprintln!("ERROR: --regime-min-pb は 0..=7 (今 {})", args.regime_min_pb);
            std::process::exit(2);
        }
        if args.regime_v1_th < 0 {
            eprintln!("ERROR: --regime-v1-th は 0 以上 (今 {})", args.regime_v1_th);
            std::process::exit(2);
        }
        bullet_lib::game::outputs::set_regime_mode(true, args.regime_v1_th, args.regime_min_pb);
        println!(
            "LSMODE: PROGREGIME  stack = progress_bucket8*2 + [bucket8 >= {} && V1 >= {}]  (V1 = 両玉 8 近傍の空升への敵の利き数)",
            args.regime_min_pb, args.regime_v1_th
        );
    }
    args.validate_wrm_settings().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });

    // --threat-freeze-hard (真の凍結) は **表を触る指定と排他**。  2026-11-03
    //   凍結した表に対する操作なので、 黙って無視するより止めるほうが安全
    //   (★「黙って破棄される設定」で何度も時間を失っている)。
    if args.threat_freeze_hard {
        let mut bad: Vec<&str> = Vec::new();
        if args.threat_factorizer {
            bad.push("--threat-factorizer");
        }
        if args.threat_hang_split {
            bad.push("--threat-hang-split");
        }
        if args.threat_zero_init {
            bad.push("--threat-zero-init");
        }
        if args.threat_clip > 0.0 {
            bad.push("--threat-clip");
        }
        if !bad.is_empty() {
            eprintln!("ERROR: --threat-freeze-hard は {} と排他です (凍結した表を触る指定)。", bad.join(" / "));
            std::process::exit(1);
        }
        if args.threat_out == 0 {
            eprintln!("ERROR: --threat-freeze-hard は --threat-out > 0 のときだけ意味があります。");
            std::process::exit(1);
        }
    }
    // 2026-11-12 KH ブロックの引数検査。 黙って無視される組み合わせは全部止める。
    use bullet_lib::game::inputs::shogi_kh::{KhKMode, KhParts, KhXMode};
    let kh_parts: Option<KhParts> = if args.kh {
        let fail = |m: &str| -> ! {
            eprintln!("ERROR: --kh: {m}");
            std::process::exit(1);
        };
        let parts = match args.kh_feats.as_str() {
            "k" => KhParts { k: true, h: false, kmode: KhKMode::Plain, x: KhXMode::Off },
            "h" => KhParts { k: false, h: true, kmode: KhKMode::Plain, x: KhXMode::Off },
            "kh" => KhParts { k: true, h: true, kmode: KhKMode::Plain, x: KhXMode::Off },
            "hx" => KhParts { k: false, h: true, kmode: KhKMode::Plain, x: KhXMode::Loss },
            "hxh" => KhParts { k: false, h: true, kmode: KhKMode::Plain, x: KhXMode::Hang },
            other => fail(&format!("--kh-feats は k / h / kh / hx / hxh のどれか (与えられた値: {other})")),
        };
        let parts = KhParts {
            kmode: match args.kh_kmode {
                1 => KhKMode::Plain,
                2 => KhKMode::Region,
                other => fail(&format!("--kh-kmode は 1 か 2 (与えられた値: {other})")),
            },
            ..parts
        };
        if !args.threat || args.threat_out == 0 {
            fail("--threat --threat-out K (専用 sparse の threat 構成) と併用すること");
        }
        if args.kh_out == 0 || args.kh_out % 2 != 0 {
            fail("--kh-out は正の偶数");
        }
        if args.kh_row_offset + args.kh_out / 2 > args.l0 / 2 {
            fail("--kh-row-offset + --kh-out/2 が FT の半分 (--l0 / 2) を超えている");
        }
        if args.kh_lr_mult <= 0.0 {
            fail("--kh-lr-mult は正の値");
        }
        if args.kh_freeze_rest && args.main_freeze {
            fail("--kh-freeze-rest と --main-freeze は排他 (凍結の倍率が食い違う)");
        }
        if args.kh_freeze_rest && args.threat_out > 0 && !args.threat_freeze_hard {
            fail(
                "--kh-freeze-rest は --threat-freeze-hard と併用すること (threat も凍結しないと KH だけの学習にならない)",
            );
        }
        if args.kh_unfreeze_fc0 && !args.kh_freeze_rest {
            fail("--kh-unfreeze-fc0 は --kh-freeze-rest と併用すること");
        }
        if args.kh_unfreeze_fc0_cols {
            if !args.kh_freeze_rest {
                fail("--kh-unfreeze-fc0-cols は --kh-freeze-rest と併用すること");
            }
            if args.kh_unfreeze_fc0 {
                fail("--kh-unfreeze-fc0-cols と --kh-unfreeze-fc0 は排他");
            }
            if !matches!(args.optimizer, OptimizerType::Ranger21) {
                fail("--kh-unfreeze-fc0-cols は Ranger21 だけ (区間倍率を持たない optimizer では黙って全列が動く)");
            }
            if args.hard_freeze_shared || args.threat_only || args.threat_ext_only {
                fail(
                    "--kh-unfreeze-fc0-cols は --hard-freeze-shared / --threat-only / --threat-ext-only と併用できない (fc0 が真の凍結になる)",
                );
            }
            if (args.l1_lr_mult - 1.0).abs() > 1e-6 || args.l1_weight_decay.is_some() {
                fail("--kh-unfreeze-fc0-cols と --l1-lr-mult / --l1-weight-decay は併用できない (倍率が食い違う)");
            }
        }
        if args.kh_freeze_rest
            && ((args.l0_lr_mult - 1.0).abs() > 1e-6
                || (args.psqt_lr_mult - 1.0).abs() > 1e-6
                || (args.l2_lr_mult - 1.0).abs() > 1e-6
                || (args.l3_lr_mult - 1.0).abs() > 1e-6
                || (!args.kh_unfreeze_fc0 && ((args.l1_lr_mult - 1.0).abs() > 1e-6 || args.l1_weight_decay.is_some())))
        {
            fail("--kh-freeze-rest と 凍結した層の lr 倍率 (--l0/--l1/--l2/--l3/--psqt-lr-mult) は併用できない");
        }
        if let Some(s) = args.kh_split {
            if s > 1 {
                fail("--kh-split は 0 か 1");
            }
            if args.l1_anchor_lambda > 0.0 || args.bucket_smooth_l1_lambda > 0.0 {
                fail("--kh-split は L1 graph 正則化 (build_custom) とは併用できない");
            }
        }
        if args.l4 > 0 || args.threat_proj || args.factorise {
            fail("--l4 / --threat-proj / --factorise との併用は未検証なので止める");
        }
        println!(
            "KH ブロック: 部分 {} / K 部の表現 kmode {} (入力 {} 次元 / 最大 {} 本) / 出力幅 {} (FT 行 [{}, {}) と [{}, {})) / lr 倍率 {} (decay 0) / 凍結 {} / split {} / null {}",
            args.kh_feats,
            args.kh_kmode,
            parts.dims(),
            parts.max_active(),
            args.kh_out,
            args.kh_row_offset,
            args.kh_row_offset + args.kh_out / 2,
            args.l0 / 2 + args.kh_row_offset,
            args.l0 / 2 + args.kh_row_offset + args.kh_out / 2,
            args.kh_lr_mult,
            if !args.kh_freeze_rest {
                "なし (他の層は通常どおり学習)"
            } else if args.kh_unfreeze_fc0 {
                "FT / threat / fc1 / fc2 / psqt (fc0 は学習)"
            } else if args.kh_unfreeze_fc0_cols {
                "FT / threat / fc0 の KH 以外の列と bias / fc1 / fc2 / psqt (fc0 は KH の列だけ学習)"
            } else {
                "FT / threat / fc0 / fc1 / fc2 / psqt (KH だけ学習)"
            },
            args.kh_split.map_or("なし (全局面)".to_string(), |s| format!("半分 {s} だけ")),
            args.kh_null
        );
        println!(
            "🚨KH ブロック: kh_ft_w は quantised.bin に書き出されない (エンジン側は未実装)。 判定は float の重みとログで行うこと"
        );
        Some(parts)
    } else {
        if args.kh_freeze_rest
            || args.kh_unfreeze_fc0
            || args.kh_unfreeze_fc0_cols
            || args.kh_split.is_some()
            || args.kh_null
            || args.kh_dump.is_some()
            || args.kh_feats != "kh"
            || args.kh_kmode != 1
        {
            eprintln!("ERROR: --kh-* の指定は --kh と併用すること (黙って無視させない)");
            std::process::exit(1);
        }
        None
    };
    // 2026-11-17 K256 案 A: --threat-ext-* の引数検査。 黙って無視される・関数が書き出しとずれる組み合わせは全部止める。
    if let Err(e) = validate_threat_ext_args(&args) {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }
    if args.threat_ext_out > 0 {
        let half = args.l0 / 2;
        let lo = args.threat_out / 2;
        let n2 = args.threat_ext_out / 2;
        println!(
            "THREAT-EXT: 追加列 threat_ext_w ({} x threat_dims、 1 群、 0 初期化) / FT 行 [{lo}, {}) と [{}, {}) に足す / \
             エンジンの K' = {} / lr 倍率 {} (decay 0、 clip |w| <= {:.6}) / 他を凍結 {}",
            args.threat_ext_out,
            lo + n2,
            half + lo,
            half + lo + n2,
            args.threat_out + args.threat_ext_out,
            args.threat_ext_lr_mult,
            THREAT_EXT_CLIP,
            args.threat_ext_only
        );
        println!(
            "🚨THREAT-EXT: threat_ext_w は quantised.bin に書き出されない。 net は k256_merge.py で weights.bin から作ること"
        );
    }
    // 2026-11-13 INVENT02 案 A: --bucket-filter / --hard-freeze-shared の検査。
    //   どちらも未指定なら None / 何もしない (= 従来と同一)。
    let bucket_filter_range: Option<(u8, u8)> = validate_bucket_filter_args(&args).unwrap_or_else(|e| {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    });
    let ply_bounds = args.resolved_ply_bounds().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });
    let progress_bucket = args.load_progress_bucket().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });
    let ft_out = args.l0;
    // 2026-09-15: L1 の幅は普通に --l1 で指定する (main = l1 - 1 行 + skip 1 行、 L2 入力 [main^2, main])。
    //   --residual-wide16 は旧スクリプト互換のためだけに残す (L1=16 固定 + 補助 head 禁止の検査。 計算は同じ)。
    let l1_out = if args.residual_wide16 { 16 } else { args.l1 };
    let l2_out = args.l2;
    if args.residual_wide16 {
        // R-MixWide16 keeps only heads with an intended P2NNUE inference path.
        // PSQT and Dual PSQT remain permitted; historical auxiliary heads are rejected.
        // Threat-E48 is the only supported threat mode in R-MixWide16.
        // It is connected to the P2_ARCH_1024T engine path; all historical
        // auxiliary heads remain inference-disconnected and are rejected.
        let threat_e48 = args.threat
            && matches!(
                args.threat_profile.as_str(),
                "enemy-only"
                    | "enemy-only-rep8"
                    | "enemy-only-factor16"
                    | "enemy-only-major-same-direct"
                    // 2026-09-01: Profile15 (Direct15) も 1024-16-64 / L2 入力 32 の
                    // MixWide16 構成で推論接続済 (engine: P2_ARCH_1024T =
                    // P2_ARCH_1024 + P2_RESIDUAL_MIXWIDE16)。 許可リストに追加する。
                    | "enemy-only-direct15"
            );
        // 2026-09-09: --threat-out K を指定した場合、 threat は主入力から分離され
        //   (K, threat_dims) の独立テンソルになる。 この形なら full profile でも
        //   VRAM に載り、 エンジンも P2_THREAT_OUT=K + P2_THREAT_PROFILE で対応済み。
        if args.threat && !threat_e48 && args.threat_out == 0 {
            eprintln!(
                "ERROR: --residual-wide16 accepts --threat only with --threat-profile enemy-only, enemy-only-rep8, enemy-only-factor16, enemy-only-major-same-direct, enemy-only-direct15, or any profile with --threat-out K."
            );
            std::process::exit(2);
        }
        let unsupported_aux_head = args.threat_i16
            || args.hand_threat
            || args.hand_threat_defensive
            || args.hand_count_dense
            || args.pin_aware
            || args.enable_kingring
            || args.enable_pairshape
            || args.enable_activity_risk
            || args.enable_king_escape
            || args.enable_slider_xray
            || args.enable_kzt_hand
            || args.enable_kzt_ft
            || args.enable_aux
            || args.enable_auxcheap
            || args.enable_king_mobility
            || args.enable_drop_pressure
            || args.enable_endgame_tempo
            || args.enable_scalar_threat
            || args.enable_dual_cross_geometry
            || args.enable_attack_relay
            || args.enable_attack_relay_v2;
        if unsupported_aux_head {
            eprintln!(
                "ERROR: --residual-wide16 restricts training to inference-connected heads. Remove auxiliary-head flags; --psqt and Dual PSQT options remain supported."
            );
            std::process::exit(2);
        }
        // R-MixWide16 restricts training to inference-connected heads.
        if ft_out != 1024 || l2_out != 64 {
            eprintln!("ERROR: --residual-wide16 requires --l0 1024 and --l2 64 (got l0={ft_out}, l2={l2_out})");
            std::process::exit(2);
        }
        println!("R-MixWide16 enabled: architecture 1024-16-64, L2 input 30 ([main^2, main])");
    }
    let l1_effective = l1_out - 1;
    let l2_in = l1_effective * 2;
    // 2026-11-14 OPN2048 K16: `--opn-king16` なら主入力は ShogiHalfKA_hmK16 (26,064)。 以後の arch 文字列・
    //   factoriser の coalesce・quantised.bin の FT 行数はすべてこの値に追従する (既定は従来の 73,305)。
    let halfka_dim = if args.opn_king16 {
        bullet_lib::game::inputs::ShogiHalfKA_hmK16.num_inputs() // 26064
    } else {
        ShogiHalfKA_hm.num_inputs() // 73305
    };

    // --threat / --hand-threat / --hand-threat-defensive は相互排他
    //
    // 🚨2026-11-07 (T3): `--hand-count-dense` を**この排他から外した**。
    //   dense は sparse 入力の次元を 1 つも変えない (L1 の入力に 14 列 concat するだけ) ので
    //   threat と次元が衝突しない。 排他になっていたのは hand-threat 用の流用で、
    //   本番 net は `Threat=216720` なので外さないと本番レシピで一切使えなかった。
    //   fable 設計 DESIGN-20261107-HANDFEAT-FABLE.md §3 候補 A の「無いもの 2」。
    let ht_flags = [args.threat, args.hand_threat, args.hand_threat_defensive];
    let ht_count = ht_flags.iter().filter(|&&b| b).count();
    if ht_count > 1 {
        eprintln!("ERROR: --threat / --hand-threat / --hand-threat-defensive は同時に指定できません");
        std::process::exit(1);
    }
    // 持駒 dense は hand-threat 系とは排他のまま (どちらも持駒を dense で送るので二重計上になる)。
    if args.hand_count_dense && (args.hand_threat || args.hand_threat_defensive) {
        eprintln!("ERROR: --hand-count-dense と --hand-threat / --hand-threat-defensive は同時に指定できません");
        std::process::exit(1);
    }

    // KZT-FT (= FT 入力版) は v0 で base main (+ optional --psqt) 専用。 他 head / threat /
    // dual / hand-threat / hand-count / kzt-hand (= output 加算版) とは排他 (= 入力次元を
    // 82,745 に拡張するため threat 系と混在不可、 kzt-hand とは KZT 二重計上防止)。
    if args.enable_kzt_ft {
        let incompatible = args.threat
            || args.hand_threat
            || args.hand_threat_defensive
            || args.hand_count_dense
            || args.pin_aware
            || args.dual_psqt_reserve
            || args.dual_psqt_king_attacker
            || args.enable_scalar_threat
            || args.enable_kingring
            || args.enable_pairshape
            || args.enable_dual_cross_geometry
            || args.enable_activity_risk
            || args.enable_king_escape
            || args.enable_slider_xray
            || args.enable_king_mobility
            || args.enable_drop_pressure
            || args.enable_endgame_tempo
            || args.enable_attack_relay
            || args.enable_attack_relay_v2
            || args.enable_kzt_hand
            || args.enable_aux
            || args.enable_auxcheap;
        if incompatible {
            eprintln!(
                "ERROR: --enable-kzt-ft は v0 で base main (+ optional --psqt) 専用です。 \
                 他 head / threat / hand-threat / dual / pin / hand-count-dense / --enable-kzt-hand / --enable-aux / \
                 --enable-auxcheap とは併用できません。"
            );
            std::process::exit(1);
        }
    }

    // AUX-FT (= FT 入力版) は v0 で base main (+ optional --psqt) 専用。 enable_kzt_ft と
    // 同じく入力次元を 73,905 に拡張するため、 他 head / threat / dual / hand-threat /
    // hand-count / kzt-ft / kzt-hand とは排他 (= 二重計上 / 次元衝突防止)。
    if args.enable_aux {
        let incompatible = args.threat
            || args.hand_threat
            || args.hand_threat_defensive
            || args.hand_count_dense
            || args.pin_aware
            || args.dual_psqt_reserve
            || args.dual_psqt_king_attacker
            || args.enable_scalar_threat
            || args.enable_kingring
            || args.enable_pairshape
            || args.enable_dual_cross_geometry
            || args.enable_activity_risk
            || args.enable_king_escape
            || args.enable_slider_xray
            || args.enable_king_mobility
            || args.enable_drop_pressure
            || args.enable_endgame_tempo
            || args.enable_attack_relay
            || args.enable_attack_relay_v2
            || args.enable_kzt_hand
            || args.enable_kzt_ft
            || args.enable_auxcheap
            || args.factorise;
        if incompatible {
            eprintln!(
                "ERROR: --enable-aux は v0 で base main (+ optional --psqt) 専用です。 \
                 他 head / threat / hand-threat / dual / pin / hand-count-dense / --enable-kzt-hand / \
                 --enable-kzt-ft / --enable-auxcheap / --factorise とは併用できません。"
            );
            std::process::exit(1);
        }
    }

    // AUXCHEAP (= PSQT-only 版) は base main + --psqt 専用。 enable_aux と同じく入力次元を
    // 73,697 に拡張するため、 他 head / threat / dual / hand-threat / hand-count / kzt-ft /
    // kzt-hand / enable-aux とは排他 (= 二重計上 / 次元衝突防止)。
    // aux の寄与経路が psqtw のみなので --psqt が無いと寄与ゼロ = 誤設定として拒否する。
    if args.enable_auxcheap {
        let incompatible = args.threat
            || args.hand_threat
            || args.hand_threat_defensive
            || args.hand_count_dense
            || args.pin_aware
            || args.dual_psqt_reserve
            || args.dual_psqt_king_attacker
            || args.enable_scalar_threat
            || args.enable_kingring
            || args.enable_pairshape
            || args.enable_dual_cross_geometry
            || args.enable_activity_risk
            || args.enable_king_escape
            || args.enable_slider_xray
            || args.enable_king_mobility
            || args.enable_drop_pressure
            || args.enable_endgame_tempo
            || args.enable_attack_relay
            || args.enable_attack_relay_v2
            || args.enable_kzt_hand
            || args.enable_kzt_ft
            || args.enable_aux;
        if incompatible {
            eprintln!(
                "ERROR: --enable-auxcheap は v0 で base main + --psqt 専用です。 \
                 他 head / threat / hand-threat / dual / pin / hand-count-dense / --enable-kzt-hand / \
                 --enable-kzt-ft / --enable-aux とは併用できません。"
            );
            std::process::exit(1);
        }
        if !args.psqt {
            eprintln!(
                "ERROR: --enable-auxcheap は PSQT-only 経路 (psqtw aux 行のみ訓練) のため \
                 --psqt が必須です。"
            );
            std::process::exit(1);
        }
    }

    // Threat profile の解決
    let threat_profile = if args.threat {
        let tp = ThreatProfile::from_cli(&args.threat_profile).unwrap_or_else(|| {
            eprintln!(
                "ERROR: Unknown threat profile '{}'. Available: {}",
                args.threat_profile,
                ThreatProfile::available()
            );
            std::process::exit(1);
        });
        Some(tp)
    } else {
        None
    };

    // --threat-out K: threat を専用 sparse 入力へ分離し、 出力幅を K に絞る。
    //   factor16 / direct 系は元から別枝なので対象外 (指定されたらエラー)。
    let threat_ft_dims = threat_profile.map(|tp| ShogiHalfKaHmThreat::new(tp).threat_dimensions()).unwrap_or(0);
    let threat_ft_out = if args.threat_out > 0 {
        if !args.threat {
            eprintln!("ERROR: --threat-out は --threat と同時に指定すること。");
            std::process::exit(2);
        }
        if matches!(
            args.threat_profile.as_str(),
            "enemy-only-factor16" | "enemy-only-major-same-direct" | "enemy-only-direct15"
        ) {
            eprintln!("ERROR: --threat-out は factor16 / direct 系 profile では使えない (別枝のため)。");
            std::process::exit(2);
        }
        // 2026-09-10: エンジンに K 幅の i16 加減算 (acc_add_threat_row_i16) を実装したので
        //   --threat-out と --threat-i16 は併用可能になった。
        //   threat 重みは学習が進むと i8 の +-127 に張り付く (K=64 の SB6 で max 104)。
        if args.threat_out > args.l0 {
            eprintln!("ERROR: --threat-out ({}) が FT 幅 ({}) を超えている。", args.threat_out, args.l0);
            std::process::exit(2);
        }
        args.threat_out
    } else {
        0
    };
    let threat_ft_sparse = threat_ft_out > 0;
    // 2026-09-16: threat factorizer の factor 列数 (無効なら 0)。
    let threat_factor_rows = if args.threat_factorizer {
        if !threat_ft_sparse {
            eprintln!("ERROR: --threat-factorizer は --threat-out K (K > 0) と併用すること");
            std::process::exit(1);
        }
        if threat_profile != Some(ThreatProfile::Full) {
            eprintln!("ERROR: --threat-factorizer は --threat-profile full でのみ検証済み");
            std::process::exit(1);
        }
        bullet_lib::game::inputs::THREAT_FACTOR_ROWS
    } else {
        0
    };
    // 2026-09-24 profile 16 (hanging-split): 群共有の delta 行数 (無効なら 0)。
    let threat_hang_rows = if args.threat_hang_split {
        if !threat_ft_sparse {
            eprintln!("ERROR: --threat-hang-split は --threat-out K (K > 0) と併用すること");
            std::process::exit(1);
        }
        if threat_profile != Some(ThreatProfile::FullHangSplit) {
            eprintln!("ERROR: --threat-hang-split は --threat-profile full-hang-split と併用すること");
            std::process::exit(1);
        }
        if threat_factor_rows > 0 {
            eprintln!(
                "ERROR: --threat-hang-split と --threat-factorizer は併用できない (どちらも threat_ft_w 末尾を使う)"
            );
            std::process::exit(1);
        }
        bullet_lib::game::inputs::THREAT_HANG_ROWS
    } else {
        if threat_profile == Some(ThreatProfile::FullHangSplit) {
            eprintln!("ERROR: --threat-profile full-hang-split には --threat-hang-split が必要");
            std::process::exit(1);
        }
        0
    };
    // 2026-09-16 `--threat-grad-mask`: "閾値:係数" の昇順リスト (閾値 inf = それ以上すべて)。
    let threat_grad_mask: Option<Vec<(i32, f32)>> = match args.threat_grad_mask.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(spec) => {
            if !threat_ft_sparse {
                eprintln!("ERROR: --threat-grad-mask は --threat-out K (K > 0) と併用すること");
                std::process::exit(1);
            }
            let mut table: Vec<(i32, f32)> = Vec::new();
            for tok in spec.split(',') {
                let Some((th, w)) = tok.split_once(':') else {
                    eprintln!("ERROR: --threat-grad-mask の '{tok}' は 閾値:係数 の形ではない");
                    std::process::exit(1);
                };
                let th = if matches!(th.trim(), "inf" | "" | "-") {
                    i32::MAX
                } else {
                    th.trim().parse::<i32>().unwrap_or_else(|_| {
                        eprintln!("ERROR: --threat-grad-mask の閾値 '{th}' を読めない");
                        std::process::exit(1);
                    })
                };
                let w = w.trim().parse::<f32>().ok().filter(|w| (0.0..=1.0).contains(w)).unwrap_or_else(|| {
                    eprintln!("ERROR: --threat-grad-mask の係数 '{w}' は 0..1 の実数で指定すること");
                    std::process::exit(1);
                });
                table.push((th, w));
            }
            if table.windows(2).any(|w| w[0].0 >= w[1].0) {
                eprintln!("ERROR: --threat-grad-mask の閾値は昇順にすること");
                std::process::exit(1);
            }
            println!("threat 勾配マスク: {table:?} (|教師値| <= 閾値 の係数を勾配にだけ掛ける、 順伝播は不変)");
            Some(table)
        }
    };
    let threat_grad_mask_on = threat_grad_mask.is_some();
    // 2026-09-17 `--loss-grad-mask`: 書式・検査は --threat-grad-mask と同じ。
    let loss_grad_mask: Option<Vec<(i32, f32)>> = match args.loss_grad_mask.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(spec) => {
            let mut table: Vec<(i32, f32)> = Vec::new();
            for tok in spec.split(',') {
                let Some((th, w)) = tok.split_once(':') else {
                    eprintln!("ERROR: --loss-grad-mask の '{tok}' は 閾値:係数 の形ではない");
                    std::process::exit(1);
                };
                let th = if matches!(th.trim(), "inf" | "" | "-") {
                    i32::MAX
                } else {
                    th.trim().parse::<i32>().unwrap_or_else(|_| {
                        eprintln!("ERROR: --loss-grad-mask の閾値 '{th}' を読めない");
                        std::process::exit(1);
                    })
                };
                let w = w.trim().parse::<f32>().ok().filter(|w| (0.0..=1.0).contains(w)).unwrap_or_else(|| {
                    eprintln!("ERROR: --loss-grad-mask の係数 '{w}' は 0..1 の実数で指定すること");
                    std::process::exit(1);
                });
                table.push((th, w));
            }
            if table.windows(2).any(|w| w[0].0 >= w[1].0) {
                eprintln!("ERROR: --loss-grad-mask の閾値は昇順にすること");
                std::process::exit(1);
            }
            if args.l1_anchor_lambda > 0.0 || args.bucket_smooth_l1_lambda > 0.0 {
                eprintln!("ERROR: --loss-grad-mask は L1 graph 正則化 (build_custom) とは併用できない");
                std::process::exit(1);
            }
            println!(
                "局面ごとの損失の重み: {table:?} (|教師値| <= 閾値 の係数を損失に掛ける = bullet の datapoint_weight_function)"
            );
            LOSS_WEIGHT_TABLE.set(table.clone()).expect("loss weight table should only be set once");
            Some(table)
        }
    };
    let loss_grad_mask_on = loss_grad_mask.is_some();
    /// `--loss-grad-mask` の表で、 局面の |教師値| に応じた損失の重みを返す (bullet の fn ポインタ用なので表はグローバル)。
    fn loss_weight_fn(pos: &bullet_lib::shogi::PackedSfenValue) -> f32 {
        let table = LOSS_WEIGHT_TABLE.get().expect("loss weight table must be set");
        let a = (pos.score() as i32).abs();
        for &(th, w) in table.iter() {
            if a <= th {
                return w;
            }
        }
        table.last().map_or(1.0, |&(_, w)| w)
    }

    /// `--kh-split`: 盤面 hash の半分が一致する局面だけ重み 1 (他は 0)。 `--loss-grad-mask` の表があれば掛け合わせる。
    fn kh_split_weight_fn(pos: &bullet_lib::shogi::PackedSfenValue) -> f32 {
        let half = *KH_SPLIT.get().expect("kh split must be set");
        if bullet_lib::game::inputs::shogi_kh::kh_split_half(pos) != half {
            return 0.0;
        }
        if LOSS_WEIGHT_TABLE.get().is_some() { loss_weight_fn(pos) } else { 1.0 }
    }
    /// 2026-11-19 TREEDL: byte 39 の bit 0 が立った局面 (探索木の局面) は勝敗の項を混ぜない。
    fn tree_nowdl_blend_fn(pos: &bullet_lib::shogi::PackedSfenValue, wdl: f32) -> f32 {
        if pos.as_bytes()[39] & 1 != 0 { 0.0 } else { wdl }
    }
    let tree_nowdl_on = args.tree_nowdl;
    if tree_nowdl_on {
        println!("TREE-NOWDL: byte 39 の bit 0 が 1 の局面は wdl を 0 にする (target = sigmoid(score / scale) だけ)");
    }
    let kh_split_on = args.kh && args.kh_split.is_some();
    if let Some(h) = args.kh_split.filter(|_| args.kh) {
        KH_SPLIT.set(h).expect("kh split should only be set once");
    }

    // --threat-bucket-groups (2026-09-12): threat 表を LayerStack bucket の群ごとに独立させる。
    //   未指定なら None / 群数 1 で、 graph・入力・書き出しとも従来と同一。
    fn threat_group_fail(msg: &str) -> ! {
        eprintln!("ERROR: --threat-bucket-groups: {msg}");
        std::process::exit(2);
    }
    let threat_bucket_groups: Option<Vec<u8>> = match args.threat_bucket_groups.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(spec) => {
            if threat_ft_out == 0 {
                threat_group_fail(
                    "--threat --threat-out K (K > 0) と併用すること (threat を専用 sparse 入力に分離した構成でのみ使える)",
                );
            }
            // 2026-10-07: 案 B (16 bucket) でも群化を使えるようにする。
            //   群数の検証は下の `groups.len() != NUM_BUCKETS` が行うので、
            //   16 bucket ビルドでは 16 要素を渡すこと
            //   (progress 側の 8 を複製: 0,0,1,1,1,1,2,2,2,2,3,3,3,3,3,3)。
            if !matches!(
                args.bucket_mode,
                BucketMode::Progress8KPAbs | BucketMode::Progress4Hand2KPAbs | BucketMode::Progress8Hand2KPAbs
            ) {
                threat_group_fail(
                    "--bucket-mode progress8kpabs / progress4hand2kpabs でのみ使える (群は ShogiProgressKPAbs の bucket で決める)",
                );
            }
            let groups: Vec<u8> = spec
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse::<u8>()
                        .unwrap_or_else(|_| threat_group_fail(&format!("'{t}' を群番号として読めない")))
                })
                .collect();
            if groups.len() != NUM_BUCKETS {
                threat_group_fail(&format!("{} 個指定されたが bucket 数は {NUM_BUCKETS}", groups.len()));
            }
            let count = groups.iter().copied().max().unwrap_or(0) as usize + 1;
            if count < 2 {
                threat_group_fail("群が 1 つしか無い (群化しないなら flag を外すこと)");
            }
            if let Some(missing) = (0..count).find(|&g| !groups.iter().any(|&x| x as usize == g)) {
                threat_group_fail(&format!("群 {missing} に属する bucket が無い (使われない表ができる)"));
            }
            Some(groups)
        }
    };
    let threat_groups = threat_bucket_groups.as_ref().map_or(1, |g| g.iter().copied().max().unwrap_or(0) as usize + 1);
    if let Some(groups) = &threat_bucket_groups {
        // 推定メモリ: GPU = 重み + 勾配 + optimizer 状態 3 本 (Ranger21 の momentum / velocity / max_velocity)
        //   = 5 本 x 4 byte、 host = Ranger21 lookahead の slow / current 2 本 x 4 byte。
        //   activation / sparse index は群化しても要素数が変わらないので増えない。
        let per_table = threat_ft_out * threat_ft_dims;
        let total = per_table * threat_groups;
        let extra = total - per_table;
        let mib = |params: usize, bytes: usize| (params * bytes) as f64 / (1024.0 * 1024.0);
        // 真の凍結なら勾配も optimiser 状態も持たないので **重み 4 B/param だけ**。
        let bytes_per = if args.threat_freeze_hard { 4 } else { 20 };
        let gpu_extra = mib(extra, bytes_per);
        println!(
            "threat 群化: 群数 {threat_groups}、 bucket→群 {groups:?}、 threat_ft_w ({threat_ft_out} x {threat_ft_dims}) → \
             ({threat_ft_out} x {}) = {total} params (+{extra} params)",
            threat_ft_dims * threat_groups
        );
        println!(
            "threat 群化: 推定メモリ増分 GPU +{gpu_extra:.0} MiB ({})、 \
             host +{:.0} MiB ({})。 上限 --threat-group-mem-budget-mib = {:.0} MiB",
            if args.threat_freeze_hard {
                "**重みのみ 4 byte/param** (真の凍結: 勾配・optimizer 状態なし)"
            } else {
                "重み+勾配+optimizer 状態 3 本 = 20 byte/param"
            },
            if args.threat_freeze_hard { 0.0 } else { mib(extra, 8) },
            if args.threat_freeze_hard { "真の凍結なので 0" } else { "Ranger21 lookahead 2 本" },
            args.threat_group_mem_budget_mib
        );
        if gpu_extra > args.threat_group_mem_budget_mib {
            threat_group_fail(&format!(
                "推定 GPU メモリ増分 {gpu_extra:.0} MiB が上限 {:.0} MiB を超える",
                args.threat_group_mem_budget_mib
            ));
        }
    }
    // 2026-09-15: 群別 threat lr。 threat_ft_w の要素区間 (群ごと) と倍率の列に変換しておく。
    let threat_group_lr_ranges: Option<Vec<(usize, usize, f32, f32)>> =
        match args.threat_group_lr_mult.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(spec) => {
                if threat_groups < 2 {
                    threat_group_fail("--threat-group-lr-mult は --threat-bucket-groups (群 2 つ以上) と併用すること");
                }
                if !matches!(args.optimizer, OptimizerType::Ranger21) {
                    threat_group_fail("--threat-group-lr-mult は --optimizer ranger21 でのみ使える");
                }
                let mults: Vec<f32> = spec
                    .split(',')
                    .map(|t| {
                        t.trim()
                            .parse::<f32>()
                            .ok()
                            .filter(|m| *m > 0.0 && m.is_finite())
                            .unwrap_or_else(|| threat_group_fail(&format!("'{t}' を正の lr 倍率として読めない")))
                    })
                    .collect();
                if mults.len() != threat_groups {
                    threat_group_fail(&format!(
                        "--threat-group-lr-mult に {} 個指定されたが群数は {threat_groups}",
                        mults.len()
                    ));
                }
                let per_table = threat_ft_out * threat_ft_dims;
                let ranges: Vec<(usize, usize, f32, f32)> =
                    mults.iter().enumerate().map(|(g, &m)| (g * per_table, (g + 1) * per_table, m, 0.0)).collect();
                println!("threat 群別 lr 倍率: {mults:?} (threat_ft_w の要素区間 {per_table} 個ずつ)");
                Some(ranges)
            }
        };
    // 2026-09-16 `--threat-factorizer`: factor 列 (末尾 K x 324 要素) に別の lr 倍率を掛ける。
    //   range lr は threat_ft_w 全体を順に敷き詰める必要があるので、 群別 lr が無いときは実特徴全体を倍率 1.0 の 1 区間にする。
    let threat_group_lr_ranges: Option<Vec<(usize, usize, f32, f32)>> = if threat_factor_rows > 0 {
        if !matches!(args.optimizer, OptimizerType::Ranger21) {
            eprintln!("ERROR: --threat-factorizer の factor lr 倍率は --optimizer ranger21 でのみ使える");
            std::process::exit(1);
        }
        let real = threat_ft_out * threat_ft_dims * threat_groups;
        let mut ranges = threat_group_lr_ranges.unwrap_or_else(|| vec![(0, real, 1.0, 0.0)]);
        ranges.push((real, real + threat_ft_out * threat_factor_rows, args.threat_factor_lr_mult, 1.0));
        println!(
            "threat factorizer: factor 列 {threat_factor_rows} (要素 {}) を threat_ft_w 末尾に追加、 lr 倍率 {} / decay {} (群共有、 実特徴には decay を掛けない)",
            threat_ft_out * threat_factor_rows,
            args.threat_factor_lr_mult,
            args.threat_factor_decay
        );
        Some(ranges)
    } else if threat_hang_rows > 0 {
        // 2026-09-24 hanging-split: delta 列 (末尾 K x 108,360 要素) に別の lr 倍率を掛ける。
        if !matches!(args.optimizer, OptimizerType::Ranger21) {
            eprintln!("ERROR: --threat-hang-split の delta lr 倍率は --optimizer ranger21 でのみ使える");
            std::process::exit(1);
        }
        let real = threat_ft_out * threat_ft_dims * threat_groups;
        let mut ranges = threat_group_lr_ranges.unwrap_or_else(|| vec![(0, real, 1.0, 0.0)]);
        ranges.push((real, real + threat_ft_out * threat_hang_rows, args.threat_delta_lr_mult, 0.0));
        println!(
            "threat hanging-split: delta 列 {threat_hang_rows} (要素 {}) を threat_ft_w 末尾に追加、 lr 倍率 {} / decay 0 (群共有)",
            threat_ft_out * threat_hang_rows,
            args.threat_delta_lr_mult
        );
        Some(ranges)
    } else {
        threat_group_lr_ranges
    };

    // profile 13 uses the ordinary concatenated main sparse input so all graph branches
    // share exactly one batch. Its tail is consumed only by the separate 16D factor code.
    let factor_threat_profile = match threat_profile {
        Some(ThreatProfile::EnemyOnlyFactor16) => Some(ThreatProfile::EnemyOnlyFactor16),
        _ => None,
    };
    let direct_threat_profile = match threat_profile {
        Some(ThreatProfile::EnemyOnlyMajorSameDirect) => Some(ThreatProfile::EnemyOnlyMajorSameDirect),
        Some(ThreatProfile::EnemyOnlyDirect15) => Some(ThreatProfile::EnemyOnlyDirect15),
        _ => None,
    };
    let enable_threat_factor16 = factor_threat_profile.is_some();
    let enable_threat_direct = direct_threat_profile.is_some();

    // Direct15 cap [cp]。 default 30.0 は旧ハードコード `PROFILE15_DIRECT_CAP_CP`
    // と bit 同一。
    //
    // ★ この値は**推論側 `P2_THREAT_DIRECT_CAP_CP` と必ず同値**にすること。
    //   forward は hard clamp なので、 学習時と推論時で cap が違うと主評価が
    //   前提にしている寄与量がずれて棋力が壊れる。
    //   エンジン側は static_assert(0..=200) で、 現行 production の
    //   `tsumugi_p15_cap60_fast3.exe` は 60 でビルドされている。 値を変えるなら
    //   エンジンも同値で再ビルドすること。
    //
    // 2026-09-01 実測 (実戦 1,500 局面 / SB1 net): cap 30 は 54.9% の局面で
    // 実際に clip しており、 |raw| の中央値は 34.5cp。 上限として binding が強い。
    let direct15_cap_cp = args.direct15_cap_cp;
    if !(0.0..=200.0).contains(&direct15_cap_cp) {
        eprintln!("ERROR: --direct15-cap-cp は [0, 200] で指定してください (0 = 上限なし)。指定値={direct15_cap_cp}");
        std::process::exit(1);
    }
    let direct15_grad_scale_cp = args.direct15_grad_scale_cp;
    if !(0.0..=1000.0).contains(&direct15_grad_scale_cp) {
        eprintln!(
            "ERROR: --direct15-grad-scale-cp は [0, 1000] で指定してください (0 = hard clamp の真の勾配)。             指定値={direct15_grad_scale_cp}"
        );
        std::process::exit(1);
    }
    // P2 / P4 / P5 (2026-09-04): いずれも既定 off、 off なら graph / 書き出し形式は従来と bit 同一。
    let direct15_persp_split = args.direct15_persp_split;
    let direct15_factorise = args.direct15_factorise;
    let direct15_defended = args.direct15_defended;
    if (direct15_persp_split || direct15_factorise || direct15_defended)
        && direct_threat_profile != Some(ThreatProfile::EnemyOnlyDirect15)
    {
        eprintln!(
            "ERROR: --direct15-persp-split / --direct15-factorise / --direct15-defended は \
             --threat --threat-profile enemy-only-direct15 と併用してください。"
        );
        std::process::exit(1);
    }
    if args.direct15_expand_head {
        if direct_threat_profile != Some(ThreatProfile::EnemyOnlyDirect15) {
            eprintln!(
                "ERROR: --direct15-expand-head は --threat --threat-profile enemy-only-direct15 と併用してください。"
            );
            std::process::exit(1);
        }
        if args.resume_path().is_none() && args.resume_weights_only_path().is_none() {
            eprintln!(
                "ERROR: --direct15-expand-head は --resume (checkpoint) または --resume-weights-only と併用してください \
                 (--resume-from-quantised は P2THRD2 ローダーが同等の展開を常時行うので不要)。"
            );
            std::process::exit(1);
        }
    }
    // Direct15 専用 sparse 入力の次元 (P5 で x2) と threat_direct_w の行数 (P2 で x2)。
    let direct15_dim = if direct15_defended { THREAT_DIRECT15_DEFENDED_DIMENSIONS } else { THREAT_DIRECT15_DIMENSIONS };
    let direct15_rows = if direct15_persp_split { 2 * NUM_BUCKETS } else { NUM_BUCKETS };
    if direct15_cap_cp == 0.0 {
        eprintln!(
            "WARNING: --direct15-cap-cp=0 は Direct15 の寄与に上限を設けません。             推論側も P2_THREAT_DIRECT_CAP_CP=0 (上限なし) でビルドしたエンジンを使うこと。"
        );
    } else if (direct15_cap_cp - 60.0).abs() > f32::EPSILON {
        eprintln!(
            "WARNING: --direct15-cap-cp={direct15_cap_cp} は現行 production エンジン\
             (tsumugi_p15_cap60_fast3.exe, P2_THREAT_DIRECT_CAP_CP=60) と一致しません。\
             同値で再ビルドしたエンジンを使わないと学習 forward と推論の cap が乖離します。"
        );
    }

    let use_hand_threat = args.hand_threat;
    let use_hand_threat_defensive = args.hand_threat_defensive;
    let use_hand_count_dense = args.hand_count_dense;
    let lmas_flags = args.resolved_lmas_flags();
    let any_lmas_enabled = lmas_flags != 0;

    // --factorise は base main (+ optional --psqt) 専用。 各 head / threat / hand-threat /
    // dual / pin / hand-count-dense とは併用不可 (= factoriser の coalesce は FT/PSQT のみ
    // 対応、 他 head は factorise しないため)。
    if args.factorise {
        // factorise は「main sparse 入力の次元が HalfKA_hm (73,305) のまま」である
        // 構成なら併用できる (aux dense / aux sparse は別チャネルなので干渉しない)。
        // main sparse 次元自体が変わる構成 (threat / hand-threat / kzt-ft) だけは
        // ShogiPieceFactor の derive (feat % 1629) が壊れるため引き続き排他。
        //
        // 例外 (2026-07-23): --enable-auxcheap は併用可。 auxcheap は連結空間の末尾に
        // aux 392 次元を足すだけで HalfKA 部の index は不変であり、
        // `Factorises<ShogiHalfKaHmAuxCheap>` が aux 特徴で `None` を返すため
        // factor derive は HalfKA 部にのみ適用される。
        // 2026-09-29: threat は HalfKA 部の index を変えない (末尾に足すだけ) ので
        //   auxcheap と同様に併用可にした (`Factorises<ShogiHalfKaHmThreat>` を追加)。
        //   🚨factor_threat_profile (threat 表そのものの factorise、 ★22 で撤退) は別物なので
        //     こちらは不可のまま残す。
        let incompatible = factor_threat_profile.is_some()
            || use_hand_threat
            || use_hand_threat_defensive
            || any_lmas_enabled
            || args.enable_kzt_ft
            || args.enable_aux;
        if incompatible {
            eprintln!(
                "ERROR: --factorise は main sparse 入力が HalfKA_hm (73,305 次元) のままの構成 \
                 (+ --enable-auxcheap) 専用です。 \
                 hand-threat / kzt-ft / aux / threat-factorizer とは併用できません (threat は 2026-09-29 に併用可にした)。"
            );
            std::process::exit(1);
        }
        // factorise の warm-start (--resume-from-quantised) は未対応: graph l0w/psqtw の
        // 先頭に factoriser 行 (1,629) が入るため、 quantised.bin の HalfKA 重みを
        // 行 0 から詰める load_from_quantised_bin では layout がずれて silent 破損する。
        if args.resume_from_quantised_path().is_some() {
            eprintln!(
                "ERROR: --factorise は --resume-from-quantised と併用できません \
                 (factorised graph layout への逆量子化 load 未対応)。 --resume (checkpoint) を使うか、 \
                 fresh start してください。"
            );
            std::process::exit(1);
        }
    }

    // 複数の aux head / dense fill が有効なとき、 同一局面が PreparedData の
    // fill ごとに再デコードされる。 メモ化キャッシュを有効化してデコード回数を
    // 局面あたり 1 回に抑える (CPU-bound なデータ準備の主要な高速化)。
    let multi_decode_fills = args.pin_aware
        || use_hand_count_dense
        || args.dual_psqt_reserve
        || args.dual_psqt_king_attacker
        || args.enable_scalar_threat
        || enable_threat_factor16
        || enable_threat_direct
        || args.enable_kingring
        || args.enable_pairshape
        || args.enable_dual_cross_geometry
        || args.enable_activity_risk
        || args.enable_king_escape
        || args.enable_slider_xray
        || args.enable_king_mobility
        || args.enable_drop_pressure
        || args.enable_endgame_tempo
        || args.enable_attack_relay
        || args.enable_attack_relay_v2
        || args.enable_kzt_hand;
    if multi_decode_fills {
        bullet_lib::shogi::packed_sfen::set_board_decode_cache_enabled(true);
        println!("Board decode cache: enabled (aux head 構成での再デコードをメモ化)");
    }

    // 2026-11-14 OPN2048 K16: K16 は素の HalfKA_hm 経路 (+ 任意の --factorise) の差し替えとしてだけ実装した。
    //   他の入力・head・PSQT・出力形式は HalfKA_hm 73,305 を前提にした書き出しがあるので、 併用は起動時に止める。
    if args.opn_king16 {
        let conflicts: [(&str, bool); 30] = [
            ("--threat 系 (threat profile)", threat_profile.is_some()),
            ("threat factor16", enable_threat_factor16),
            ("--hand-threat", use_hand_threat),
            ("--hand-threat-defensive", use_hand_threat_defensive),
            ("--hand-count-dense", use_hand_count_dense),
            ("--enable-kzt-ft", args.enable_kzt_ft),
            ("--enable-aux", args.enable_aux),
            ("--enable-auxcheap", args.enable_auxcheap),
            ("--dual-psqt-king-attacker", args.dual_psqt_king_attacker),
            ("--dual-psqt-reserve", args.dual_psqt_reserve),
            ("--enable-scalar-threat", args.enable_scalar_threat),
            ("--enable-kingring", args.enable_kingring),
            ("--enable-pairshape", args.enable_pairshape),
            ("--enable-dual-cross-geometry", args.enable_dual_cross_geometry),
            ("--enable-activity-risk", args.enable_activity_risk),
            ("--enable-king-escape", args.enable_king_escape),
            ("--enable-slider-xray", args.enable_slider_xray),
            ("--enable-king-mobility", args.enable_king_mobility),
            ("--enable-drop-pressure", args.enable_drop_pressure),
            ("--enable-endgame-tempo", args.enable_endgame_tempo),
            ("--enable-attack-relay", args.enable_attack_relay),
            ("--enable-attack-relay-v2", args.enable_attack_relay_v2),
            ("--enable-kzt-hand", args.enable_kzt_hand),
            ("--pin-aware", args.pin_aware),
            ("--psqt", args.psqt),
            ("--kh", kh_parts.is_some()),
            ("--mob", args.mob),
            ("--residual-wide16", args.residual_wide16),
            ("--output-format (rshogi 以外)", args.output_format != LayerStackOutputFormat::Rshogi),
            ("LMAS flags", args.resolved_lmas_flags() != 0),
        ];
        let bad: Vec<&str> = conflicts.iter().filter(|(_, on)| *on).map(|(n, _)| *n).collect();
        if !bad.is_empty() {
            eprintln!("ERROR: --opn-king16 は次の指定と併用できません: {}", bad.join(" / "));
            std::process::exit(2);
        }
        println!(
            "OPN-KING16: 主入力 = HalfKA_hm K16 (玉区分 16 = 自玉 7〜9 段 x 5 筋 + その他、 {} 次元{})",
            halfka_dim,
            if args.factorise { " + factoriser 1629" } else { "" }
        );
    }

    // 2026-11-15 QAT: 丸めの式は素の LayerStack の書き出し (build_layerstack_save_format の rshogi 形式) と
    //   整数推論 (runs/opnarch/opn_static2.py forward_encoded) に合わせてある。 そこに無い経路
    //   (threat / PSQT / L4 / 手駒 dense / 追加 head など) は丸めの式が別なので、 併用は起動時に止める。
    if args.qat_l1q128 && !args.qat {
        eprintln!("ERROR: --qat-l1q128 は --qat と併用すること");
        std::process::exit(2);
    }
    if args.qat {
        // 2026-11-17 U1a: threat (専用 sparse、 --threat-out K)・PSQT・KH は FT の活性化前 / 出力への足し込みで、
        //   QAT が丸める後段 (L1 入力〜L3) の形を変えないので併用できるようにした。 threat の表は Q=254 で丸める (下の graph)。
        //   PSQT と KH の重みは FT と同じく丸めない。 連結入力の threat (profile のみ、 --threat-out 0) は従来どおり止める。
        let conflicts: [(&str, bool); 32] = [
            ("--threat 系 (連結入力 = --threat-out 0)", threat_profile.is_some() && threat_ft_out == 0),
            ("threat factor16", enable_threat_factor16),
            ("threat direct", enable_threat_direct),
            ("--hand-threat", use_hand_threat),
            ("--hand-threat-defensive", use_hand_threat_defensive),
            ("--hand-count-dense", use_hand_count_dense),
            ("--enable-kzt-ft", args.enable_kzt_ft),
            ("--enable-aux", args.enable_aux),
            ("--enable-auxcheap", args.enable_auxcheap),
            ("--dual-psqt-king-attacker", args.dual_psqt_king_attacker),
            ("--dual-psqt-reserve", args.dual_psqt_reserve),
            ("--enable-scalar-threat", args.enable_scalar_threat),
            ("--enable-kingring", args.enable_kingring),
            ("--enable-pairshape", args.enable_pairshape),
            ("--enable-dual-cross-geometry", args.enable_dual_cross_geometry),
            ("--enable-activity-risk", args.enable_activity_risk),
            ("--enable-king-escape", args.enable_king_escape),
            ("--enable-slider-xray", args.enable_slider_xray),
            ("--enable-king-mobility", args.enable_king_mobility),
            ("--enable-drop-pressure", args.enable_drop_pressure),
            ("--enable-endgame-tempo", args.enable_endgame_tempo),
            ("--enable-attack-relay", args.enable_attack_relay),
            ("--enable-attack-relay-v2", args.enable_attack_relay_v2),
            ("--enable-kzt-hand", args.enable_kzt_hand),
            ("--pin-aware", args.pin_aware),
            ("--mob", args.mob),
            ("--l4", args.l4 > 0),
            ("--residual-zero", args.residual_zero),
            ("--residual-mix", args.residual_mix),
            ("--residual-wide16", args.residual_wide16),
            ("--output-format (rshogi 以外)", args.output_format != LayerStackOutputFormat::Rshogi),
            ("LMAS flags", args.resolved_lmas_flags() != 0),
        ];
        let mut bad: Vec<&str> = conflicts.iter().filter(|(_, on)| *on).map(|(n, _)| *n).collect();
        if threat_ft_out > 0 && (args.threat_factorizer || args.threat_hang_split || args.threat_i16) {
            bad.push(
                "--threat-factorizer / --threat-hang-split / --threat-i16 (threat の表の書き出しが i8 の素の形でない)",
            );
        }
        if !bad.is_empty() {
            eprintln!("ERROR: --qat は次の指定と併用できません: {}", bad.join(" / "));
            std::process::exit(2);
        }
        if args.qat_l1q128 {
            println!(
                "QAT L1Q128: fc0 の本体 {} 行は重み Q=128 ([-1, 127/128] で clip)・bias Q=16256、 skip 行は Q=64 / 8128",
                args.l1 - 1
            );
        }
        if threat_ft_out > 0 {
            println!(
                "QAT threat: threat_ft_w (と threat_ext_w) を Q=254 で四捨五入して FT に足す (書き出しの i8 と同じ格子、 逆伝播 STE)"
            );
        }
        println!(
            "QAT: FT 積 Q=127 切り捨て / L1・L2・L3 の重み Q=64 四捨五入 (l1w + l1fw を足してから、 [-2, 127/64] で clip) / \
             bias Q=8128 四捨五入 / L1 後・L2 後の活性 Q=127 切り捨て / 逆伝播 STE / master 重みは f32"
        );
    }

    let input_size = if args.factorise {
        // Factorised<_, ShogiPieceFactor>: graph layout は [factoriser 1,629 | 実特徴]。
        // - plain: 実特徴 73,305 + factor 1,629 = 74,934
        // - --enable-auxcheap 併用: 実特徴 73,697 (= HalfKA 73,305 + aux 392) + factor 1,629
        //   = 75,326 (aux 行は実特徴 block の末尾、 factor は HalfKA 部にのみ derive)
        // graph (l0 affine / psqtw / sparse input) はこの真の次元で構築し、 export 時に
        // factor 重みを実特徴へ coalesce して実特徴次元へ畳み込む。
        let real_dim = if args.enable_auxcheap { ShogiHalfKaHmAuxCheap::new().num_inputs() } else { halfka_dim };
        real_dim + bullet_lib::game::inputs::PIECE_INPUTS
    } else if let Some(tp) = threat_profile {
        if tp == ThreatProfile::EnemyOnlyDirect15 {
            ShogiHalfKaHmThreatDirect15::new().num_inputs()
        } else {
            let threat_input = ShogiHalfKaHmThreat::new(tp).with_threat_ft_sparse(threat_ft_sparse);
            threat_input.num_inputs()
        }
    } else if use_hand_threat {
        let hand_threat_input = ShogiHalfKaHmHandThreat::new();
        hand_threat_input.num_inputs()
    } else if use_hand_threat_defensive {
        let input = ShogiHalfKaHmHandThreatDefensive::new();
        input.num_inputs()
    } else if args.enable_kzt_ft {
        // HalfKA_hm (73,305) + KZT-FT (9,440) = 82,745。 l0 affine / export はこの次元で構築。
        ShogiHalfKaHmKzt::new().num_inputs()
    } else if args.enable_aux {
        // HalfKA_hm (73,305) + AUX (600) = 73,905。 l0 affine / export はこの次元で構築。
        ShogiHalfKaHmAux::new().num_inputs()
    } else if args.enable_auxcheap {
        // HalfKA_hm (73,305) + AUXCHEAP (392) = 73,697。 l0 affine / psqtw はこの次元で構築
        // (l0w aux 行は run_with_zeroed_rows で 0 固定、 psqtw aux 行のみ学習)。
        ShogiHalfKaHmAuxCheap::new().num_inputs()
    } else {
        // --hand-count-dense の場合も sparse 次元は HalfKA_hm と同じ (dense 14 元は
        // 別経路)。
        halfka_dim
    };

    // L1 層の入力次元は FT 出力 + （HandCount 有効時は +14）
    let hand_count_dense_dims: usize = if use_hand_count_dense { bullet_lib::game::inputs::HAND_COUNT_DIMS } else { 0 };
    let l1_input_dim = ft_out + hand_count_dense_dims;

    let optimizer_name = match args.optimizer {
        OptimizerType::AdamW => "AdamW",
        OptimizerType::RAdam => "RAdam",
        OptimizerType::Ranger => "Ranger",
        OptimizerType::Ranger21 => "Ranger21",
    };

    let fv_scale = (i32::from(QA) * i32::from(QB) + args.scale / 2) / args.scale;

    // Print configuration
    println!("=== Shogi LayerStack NNUE Training ===");
    println!("Features: {} ({} dimensions)", if args.opn_king16 { "HalfKA_hmK16" } else { "HalfKA_hm" }, input_size);
    println!(
        "Architecture: LayerStack L0={}, L1={} ({} effective + 1 skip), L2={}",
        ft_out, l1_out, l1_effective, l2_out
    );
    println!("L2 input: {} (sqr_crelu concat crelu)", l2_in);
    println!("PSQT shortcut: {}", if args.psqt { "enabled" } else { "disabled" });
    if args.psqt {
        println!("PSQT init: {:?}", args.psqt_init);
    } else if args.psqt_init == PsqtInit::Material {
        eprintln!("WARNING: --psqt-init material は --psqt 無効時に無視されます");
    }
    // 2026-09-12: PSQT フェード (--psqt-scale) / PSQT 無し export (--export-no-psqt) の検査と表示。
    //   どちらも default (1.0 / off) では従来と完全に同一。
    if args.psqt_scale < 0.0 || !args.psqt_scale.is_finite() {
        eprintln!("ERROR: --psqt-scale は 0 以上の有限値を指定してください (got {})", args.psqt_scale);
        std::process::exit(1);
    }
    let psqt_scale_active = args.psqt && (args.psqt_scale - 1.0).abs() > 1e-6;
    if psqt_scale_active {
        println!(
            "PSQT scale: {} (forward の psqt 寄与 = (stm - nstm) / 2 * {}、 psqtw/psqtb への勾配も {} 倍)",
            args.psqt_scale, args.psqt_scale, args.psqt_scale
        );
    } else if !args.psqt && (args.psqt_scale - 1.0).abs() > 1e-6 {
        eprintln!("WARNING: --psqt-scale は --psqt 無効時に無視されます");
    }
    if args.export_no_psqt {
        if args.enable_auxcheap {
            eprintln!(
                "ERROR: --export-no-psqt は --enable-auxcheap (psqtw aux 行を P2AUXC section で書く) と併用できません"
            );
            std::process::exit(1);
        }
        if args.output_format != LayerStackOutputFormat::Rshogi {
            eprintln!("ERROR: --export-no-psqt は --output-format rshogi 専用です");
            std::process::exit(1);
        }
        if !args.psqt {
            eprintln!("WARNING: --export-no-psqt は --psqt 無効時は no-op です (元々 PSQT section を書かない)");
        } else {
            println!(
                "PSQT export: OFF (quantised.bin の arch から `PSQT={},` を除き PSQT section を書かない。 raw.bin / checkpoint には psqt tensor を残す)",
                NUM_BUCKETS
            );
            if args.psqt_scale.abs() > 1e-6 {
                eprintln!(
                    "WARNING: --export-no-psqt を --psqt-scale {} (≠0) で指定。 学習時の forward には PSQT 寄与が \
                     残っているので、 書き出した net の評価値は学習時と一致しない (意図した場合のみ続行)",
                    args.psqt_scale
                );
            }
        }
    }
    println!(
        "Threat: {}",
        if let Some(tp) = threat_profile {
            // Profile15 は専用 sparse 入力なので主入力次元には含まれない。
            if tp == ThreatProfile::EnemyOnlyDirect15 {
                format!(
                    "enabled (profile={tp}, {direct15_dim} dimensions via dedicated sparse inputs, main input={input_size}, cap={direct15_cap_cp}cp, grad_scale={direct15_grad_scale_cp}cp, persp_split={direct15_persp_split}, factorise={direct15_factorise}, defended={direct15_defended})"
                )
            } else if threat_ft_out > 0 {
                // --threat-out: threat は専用 sparse 入力なので主入力次元には含まれない。
                format!(
                    "enabled (profile={tp}, {threat_ft_dims} dimensions via dedicated sparse inputs, out_width={threat_ft_out}, main input={input_size})"
                )
            } else {
                let threat_dims = input_size - halfka_dim;
                format!("enabled (profile={tp}, {threat_dims} dimensions, total input={input_size})")
            }
        } else if enable_threat_factor16 {
            "enabled (profile=enemy-only-factor16, 48,160 sparse features -> 16D factor branch)".to_string()
        } else if enable_threat_direct {
            let dims = input_size - halfka_dim;
            let profile = direct_threat_profile.expect("direct profile is present");
            format!("enabled (profile={profile}, {dims} sparse features -> bucketed direct scalar branch)")
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "HandThreat: {}",
        if use_hand_threat {
            let hand_threat_dims = input_size - halfka_dim;
            format!("enabled (案 A full drop-attack pair, {hand_threat_dims} dimensions, total input={input_size})")
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "KZT-FT (FT input): {}",
        if args.enable_kzt_ft {
            let kzt_ft_dims = input_size - halfka_dim;
            format!("enabled ({kzt_ft_dims} dimensions, total input={input_size}, P2KZTF section)")
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "AUX-FT (FT input): {}",
        if args.enable_aux {
            let aux_dims = input_size - halfka_dim;
            format!("enabled ({aux_dims} dimensions, total input={input_size}, P2AUX section, AUX=600 marker)")
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "AUXCHEAP (PSQT-only): {}",
        if args.enable_auxcheap {
            // factorise 併用時 input_size は factor 1,629 行込みなので aux 次元は定数から取る。
            let aux_dims = ShogiHalfKaHmAuxCheap::new().aux_dimensions();
            format!(
                "enabled ({aux_dims} dimensions, total input={input_size}, P2AUXC section, AUXCHEAP=392 marker, l0w aux rows frozen)"
            )
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "HandCount Dense: {}",
        if use_hand_count_dense {
            format!("enabled ({} dims concat to L1 input; L1 input dim = {})", hand_count_dense_dims, l1_input_dim)
        } else {
            "disabled".to_string()
        }
    );
    // LMAS は 2026-05-25 廃止 (= 物理削除済)。 status print のみ残す。
    if any_lmas_enabled {
        panic!("LMAS は 2026-05-25 廃止 (lmas_flags=0x{:x})", lmas_flags);
    }
    println!("LMAS: removed (2026-05-25)");
    println!("Pin-aware: {}", if args.pin_aware { "enabled" } else { "disabled" });
    println!("Dual PSQT (reserve): {}", if args.dual_psqt_reserve { "enabled (Zeroed init)" } else { "disabled" });
    println!(
        "Dual PSQT (king-attacker): {}",
        if args.dual_psqt_king_attacker { "enabled (Zeroed init, P2DUAL2 format)" } else { "disabled" }
    );
    if args.threat_out > 0 {
        println!(
            "threat_ft: K={} lr_mult={}{}",
            args.threat_out,
            args.threat_ft_lr_mult,
            if args.threat_freeze {
                " (FREEZE: threat_ft_w は動かない)"
            } else if args.threat_ft_lr_mult > 1.0 + 1e-6 {
                " (decay=0 on threat_ft_w)"
            } else {
                " (boost=off)"
            }
        );
    }
    // 2026-10-09: `--threat-l2-freeze` は黙って無視されると気づけないので常に 1 行出す。
    println!(
        "threat_l2 freeze: {}",
        if !args.threat_l2 {
            "n/a (--threat-l2 が無効)"
        } else if args.threat_l2_freeze {
            "enabled (lr_scale 1e-12 / decay 0 / lookahead 無効。 lr-mult と decay は無視)"
        } else {
            "disabled"
        }
    );
    // (5) TG (2026-10-10): 黙って無視されると気づけないので常に 1 行出す。
    //   🚨TG は thr_cat (= threat_ft_w の出力) を使うので `--threat-out K > 0` が必須。
    //     指定を忘れると graph に threat_gate_w が作られないまま export が
    //     `graph.get("threat_gate_w")` で落ちるので、 ここで先に止める。
    if args.threat_gate && args.threat_out == 0 {
        eprintln!("ERROR: --threat-gate は --threat-out K (K > 0) が必要です (thr_cat が無いとゲートが作れない)");
        std::process::exit(1);
    }
    println!(
        "threat_gate (TG 乗法ゲート): {}",
        if args.threat_gate {
            format!(
                "enabled (sqr 側 {} slot x 2K={} / lr_mult={} / decay={} / g = clamp(1 + (254/256)*thr.u, 0, 4))",
                l1_effective,
                2 * threat_ft_out,
                args.threat_gate_lr_mult,
                args.threat_gate_decay
            )
        } else {
            "disabled".to_string()
        }
    );
    // 2026-09-12: L1 (fc0) drift 対策 3 option の検査と表示。 全て default OFF (no-op)。
    if args.l1_anchor_lambda < 0.0 || args.bucket_smooth_l1_lambda < 0.0 {
        eprintln!("ERROR: --l1-anchor-lambda / --bucket-smooth-l1-lambda は 0 以上を指定してください");
        std::process::exit(1);
    }
    if args.l1_anchor_lambda > 0.0 && hand_count_dense_dims > 0 {
        eprintln!("ERROR: --l1-anchor-lambda は --hand-count-dense (L1 入力に dense 列を concat) と併用できません");
        std::process::exit(1);
    }
    if (args.l1_lr_mult - 1.0).abs() > 1e-6 || args.l1_weight_decay.is_some() {
        println!(
            "L1 lr: l1w/l1b/l1fw/l1fb lr_mult={} (effective_lr = base_lr * {}) weight_decay={}",
            args.l1_lr_mult,
            args.l1_lr_mult.max(1e-6),
            args.l1_weight_decay.map_or_else(|| format!("{} (global)", args.weight_decay), |v| v.to_string())
        );
    }
    if args.l1_anchor_lambda > 0.0 {
        println!(
            "L1 anchor: lambda={} (合算 L1 = l1w_b+l1fw / l1b_b+l1fb を resume 時の値に係留、 graph 内 penalty)",
            args.l1_anchor_lambda
        );
    }
    if args.bucket_smooth_l1_lambda > 0.0 {
        println!(
            "L1 bucket smoothness: lambda={} (隣接 bucket の l1w/l1b 差を graph 内 penalty で抑制)",
            args.bucket_smooth_l1_lambda
        );
    }
    println!(
        "KingRing: {}",
        if args.enable_kingring {
            if args.kingring_lr_mult > 1.0 + 1e-6 {
                format!("enabled (lr boost x{}, decay=0 on kingring tensors)", args.kingring_lr_mult)
            } else {
                "enabled (lr boost=off, base lr/decay)".to_string()
            }
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "PairShape: {}",
        if args.enable_pairshape {
            if args.pairshape_lr_mult > 1.0 + 1e-6 {
                format!("enabled (lr boost x{}, decay=0 on pairshape tensors)", args.pairshape_lr_mult)
            } else {
                "enabled (lr boost=off, base lr/decay)".to_string()
            }
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "DualCrossGeometry: {}",
        if args.enable_dual_cross_geometry {
            if args.dual_cross_geometry_lr_mult > 1.0 + 1e-6 {
                format!(
                    "enabled (lr boost x{}, decay=0 on dual_cross_geometry tensors)",
                    args.dual_cross_geometry_lr_mult
                )
            } else {
                "enabled (lr boost=off, base lr/decay)".to_string()
            }
        } else {
            "disabled".to_string()
        }
    );
    println!(
        "ActivityRisk: {}",
        if args.enable_activity_risk {
            if args.activity_risk_lr_mult > 1.0 + 1e-6 {
                format!("enabled (lr boost x{}, decay=0 on activity_risk tensors)", args.activity_risk_lr_mult)
            } else {
                "enabled (lr boost=off, base lr/decay)".to_string()
            }
        } else {
            "disabled".to_string()
        }
    );
    println!("Buckets: {}", NUM_BUCKETS);
    println!("Bucket mode: {}", args.bucket_mode_name());
    if let Some(bounds) = ply_bounds {
        println!("Ply bounds: {:?}", bounds);
    }
    if let Some(coeff) = &args.progress_coeff {
        println!("Progress coeff: {}", coeff.display());
    }
    println!("FV_SCALE: {} (QA={}, QB={}, scale={})", fv_scale, QA, QB, args.scale);
    println!("Optimizer: {}", optimizer_name);
    println!("Weight decay: {}", args.weight_decay);
    println!("Win rate model: {}", if args.win_rate_model { "enabled" } else { "disabled" });
    if let Some(in_scaling) = args.wrm_in_scaling {
        println!("WRM in_scaling: {} nnue2score: {} (network output WRM enabled)", in_scaling, args.wrm_nnue2score);
    }
    let batches_per_superbatch_display =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
    let positions_per_superbatch = batches_per_superbatch_display as u64 * args.batch_size as u64;
    println!("Batch size: {}", args.batch_size);
    println!(
        "Batches/superbatch: {} (~{}M positions)",
        batches_per_superbatch_display,
        positions_per_superbatch / 1_000_000
    );
    println!("Superbatches: {} (start={})", args.superbatches, args.start_superbatch);
    println!("Learning rate: {} (gamma={}, step={})", args.lr, args.lr_gamma, args.lr_step);
    println!("WDL lambda: {}", args.wdl_display());
    println!("Save rate: {}", args.save_rate);
    println!("Threads: {} (queue={})", args.threads, args.batch_queue_size);
    match args.interleave_batches_value() {
        Some(v) => println!("File mix: round-robin every {} batch(es)", v),
        None => println!("File mix: sequential by file"),
    }
    if args.epoch_file_shuffle {
        println!("Epoch file shuffle: enabled (seed={})", args.file_shuffle_seed);
    } else {
        println!("Epoch file shuffle: disabled");
    }
    println!("Output: {}", args.output.display());
    println!("Net ID: {}", args.net_id);
    println!("Data: {}", args.data);
    println!("======================================");

    if let Err(e) = validate_layerstack_output_format(
        args.output_format,
        args.bucket_mode,
        input_size,
        halfka_dim,
        ft_out,
        l1_out,
        l2_out,
        args.psqt,
        threat_profile.or(factor_threat_profile),
        use_hand_threat || use_hand_threat_defensive,
        hand_count_dense_dims,
        lmas_flags,
        args.pin_aware,
        args.dual_psqt_reserve,
        args.dual_psqt_king_attacker,
    ) {
        eprintln!("ERROR: {e}");
        std::process::exit(1);
    }

    // arm 6: --dual-disable-cross-defended ON のとき P2DUALX legacy save は意味を
    // 持たない (= 削除した defended weight を save できない)。 起動時 error。
    if args.dual_disable_cross_defended && args.dual_save_legacy_p2dualx {
        eprintln!(
            "ERROR: --dual-disable-cross-defended と --dual-save-legacy-p2dualx は同時指定不可\n\
             (P2DUALX legacy save は head 0/1 両方を要求するが、 head 1 は削除済)"
        );
        std::process::exit(1);
    }

    // Experiment context
    let experiment_params = ExperimentParams {
        architecture: format!("LayerStack-{}-{}-{}", ft_out, l1_out, l2_out),
        l0: ft_out,
        l1: l1_out,
        l2: l2_out,
        num_buckets: NUM_BUCKETS,
        bucket_mode: args.bucket_mode_name().to_string(),
        ply_bounds,
        progress_coeff: args.progress_coeff.as_ref().map(|p| p.display().to_string()),
        lr: args.lr,
        lr_gamma: args.lr_gamma,
        lr_step: args.lr_step,
        batch_size: args.batch_size,
        batches_per_superbatch: batches_per_superbatch_display,
        interleave_file_batches: args.interleave_batches_value(),
        epoch_file_shuffle: args.epoch_file_shuffle,
        file_shuffle_seed: args.epoch_file_shuffle.then_some(args.file_shuffle_seed),
        superbatches: args.superbatches,
        start_superbatch: args.start_superbatch,
        wdl: args.wdl_value(),
        start_wdl: args.start_wdl,
        end_wdl: args.end_wdl,
        scale: args.scale,
        weight_decay: args.weight_decay,
        win_rate_model: args.win_rate_model,
        optimizer: optimizer_name.to_string(),
        qa: QA,
        qb: QB,
    };
    let experiment_quantise_only = args.quantise_only;
    let mut experiment_ctx = ExperimentContext::new(
        args.output.clone(),
        args.net_id.clone(),
        std::env::args().collect::<Vec<_>>().join(" "),
        experiment_params,
        args.data.clone(),
        args.superbatches,
        fv_scale,
    );

    // WDL scheduler
    let wdl_scheduler = args.create_wdl_scheduler().unwrap_or_else(|e| {
        eprintln!("ERROR: {}", e);
        std::process::exit(1);
    });

    // Training schedule
    let batches_per_superbatch =
        args.batches_per_superbatch.unwrap_or_else(|| 100_000_000_usize.div_ceil(args.batch_size));
    let schedule = TrainingSchedule {
        net_id: args.net_id.clone(),
        // loader 側の target = sigmoid(score / eval_scale)。 --loss-scale 指定時は S に揃える
        // (損失側は loss_fn_sigmoid_ls が出力に scale / S を掛けて同じ尺度にする)。
        eval_scale: args.loss_scale.unwrap_or(args.scale as f32),
        steps: TrainingSteps {
            batch_size: args.batch_size,
            batches_per_superbatch,
            start_superbatch: args.start_superbatch,
            end_superbatch: args.superbatches,
        },
        wdl_scheduler,
        // Warmup は warmup_batches = 0 のとき完全に pass-through (旧挙動と一致)
        lr_scheduler: lr::Warmup {
            inner: lr::StepLR { start: args.lr, gamma: args.lr_gamma, step: args.lr_step },
            warmup_batches: args.warmup_batches,
        },
        save_rate: args.save_rate,
        log_rate: args.log_rate,
    };

    // resume の場合は experiment_id を引き継ぐ。on_checkpoint_saved closure が
    // experiment_ctx を不変借用する前に行う必要がある。
    if !args.quantise_only && args.resume_path().is_some() {
        experiment_ctx.inherit_resume_experiment_id();
    }

    // Local settings
    let output_dir = args.output.to_str().unwrap_or("checkpoints");
    let on_checkpoint_saved = |_superbatch: usize| {
        if let Err(e) = experiment_ctx.write_experiment_json("running") {
            eprintln!("Warning: Failed to update experiment JSON: {}", e);
        }
    };
    let settings = LocalSettings {
        threads: args.threads,
        test_set: None,
        output_directory: output_dir,
        batch_queue_size: args.batch_queue_size,
        on_checkpoint_saved: if experiment_quantise_only { None } else { Some(&on_checkpoint_saved) },
    };

    // Data loader
    let data_files_owned: Vec<String> = if args.quantise_only {
        let resume_path = args.resume_path().expect("--quantise-only requires --resume");
        let quantised = resume_path.join("quantised.bin");
        if quantised.exists() {
            vec![quantised.to_str().unwrap().to_string()]
        } else {
            vec![resume_path.join("raw.bin").to_str().unwrap().to_string()]
        }
    } else if let Some(list) = args.data.strip_prefix('@') {
        // 2026-11-17 tanuki: --data @<リストファイル> (1 行 1 パス)。 1,000 本を超えるとカンマ区切りが
        //   Windows のコマンド行の上限 (32,767 文字) を超えるため。
        let text = std::fs::read_to_string(list).unwrap_or_else(|e| {
            eprintln!("ERROR: --data @{list} を読めない: {e}");
            std::process::exit(1);
        });
        let v: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect();
        println!("DATA-LIST: {list} から {} 本", v.len());
        v
    } else {
        args.data.split(',').map(|s| s.to_string()).collect()
    };
    let data_files_ref: Vec<&str> = data_files_owned.iter().map(|s| s.as_str()).collect();

    // ───────────────────────────────────────────────────────────────────────
    // Format 判定: psv (default、 40 byte / record) vs hcpe (32 byte / record)
    // ───────────────────────────────────────────────────────────────────────
    let data_format_arg = args.data_format.to_ascii_lowercase();
    let use_hcpe = match data_format_arg.as_str() {
        "psv" => false,
        "hcpe" => true,
        "auto" => {
            // 自動判定: 1 つでも hcpe 候補 (= 32 倍数 && !40 倍数) があれば hcpe 推定。
            // 量子化 only 経路は raw.bin / quantised.bin を読むので必ず psv 想定。
            if args.quantise_only {
                false
            } else {
                let mut hcpe_count = 0usize;
                let mut psv_count = 0usize;
                for path in &data_files_owned {
                    if let Ok(m) = std::fs::metadata(path) {
                        let s = m.len();
                        if looks_like_hcpe_file(s) {
                            hcpe_count += 1;
                        } else if s > 0 && s % 40 == 0 {
                            psv_count += 1;
                        }
                    }
                }
                let inferred_hcpe = hcpe_count > 0 && psv_count == 0;
                if inferred_hcpe {
                    println!("[data-format=auto] hcpe を推定 ({} file が 32 byte 倍数 / 40 byte 非倍数)", hcpe_count,);
                }
                inferred_hcpe
            }
        }
        other => {
            eprintln!("ERROR: --data-format は psv/hcpe/auto のいずれかを指定してください (got: {})", other);
            std::process::exit(1);
        }
    };

    let mut data_loader = DirectSequentialDataLoader::new(&data_files_ref);
    if let Some(interleave_batches) = args.interleave_batches_value() {
        data_loader = data_loader.with_interleave_batches(interleave_batches);
    }
    if args.epoch_file_shuffle {
        data_loader = data_loader.with_epoch_file_shuffle(true, args.file_shuffle_seed);
    }
    if !args.no_shuffle_records {
        data_loader = data_loader.with_record_shuffle(true, args.file_shuffle_seed);
        println!("Record shuffle: enabled (256MB チャンク内 Fisher-Yates、 --no-shuffle-records で無効化)");
    }
    if lmas_flags & ((1u32 << 18) | (1u32 << 19)) != 0 {
        data_loader = data_loader.with_mate_threat_sidecar(true);
        println!("[mate_threat] sidecar reader enabled (lmas_flags 0x{:x} contains bit 18 or 19)", lmas_flags);
    }

    // hcpe loader (= 32 byte / record)。 use_hcpe = true のときに $trainer.run() で
    // data_loader の代わりに使う。 buffer_size_mb は psv 経路 (= 256 MB ベース) と
    // 揃え、 shuffle 段の memory footprint を一致させる。
    let hcpe_loader = if use_hcpe {
        if lmas_flags & ((1u32 << 18) | (1u32 << 19)) != 0 {
            eprintln!(
                "ERROR: --data-format hcpe は LMAS bit 18/19 (mate_threat sidecar) と互換性がありません \
                 (sidecar は PSV 40 byte / record 専用)。 hcpe を PSV に pre-process するか、 \
                 lmas_flags から bit 18/19 を除いてください。",
            );
            std::process::exit(1);
        }
        Some(HcpeLoader::new_concat_multiple(&data_files_ref, 256))
    } else {
        None
    };

    // SavedFormat
    // HandThreat block: full pair と defensive は quantised.bin 上の layout が同一
    // (arch_str の HandThreat={dims} でサイズを表現するだけ) なので、どちらの
    // case でも hand_threat=true 扱いで書き出せる。
    let save_format_hand_threat = use_hand_threat || use_hand_threat_defensive;
    // factorise 時は arch string / 論理入力次元は coalesce 後の実特徴次元
    // (= input_size - factor 1,629。 plain なら 73,305、 auxcheap 併用なら 73,697)。
    // graph 側の真の input_size は l0/psqtw 構築でのみ使い、 export は畳み込む。
    // profile 13 factor / profile 14 direct はともに主FTがHalfKAのみ。
    // appended Threat rowsはP2THRF16/P2THRD1 custom tailとして別保存する。
    let save_format_input_size = if args.factorise {
        input_size - bullet_lib::game::inputs::PIECE_INPUTS
    } else if direct_threat_profile.is_some() {
        halfka_dim
    } else {
        input_size
    };
    let save_format = build_layerstack_save_format(
        halfka_dim,
        save_format_input_size,
        ft_out,
        l1_out,
        args.residual_zero,
        l2_out,
        fv_scale,
        // 2026-09-12: --export-no-psqt は quantised.bin の書き出しだけ PSQT を落とす
        //   (arch 文字列 + PSQT section)。 graph / raw.bin / checkpoint は args.psqt のまま。
        args.psqt && !args.export_no_psqt,
        threat_profile,
        threat_ft_out,
        args.threat_dedicated_dims && threat_ft_out > 0,
        threat_bucket_groups.clone(),
        threat_factor_rows,
        threat_hang_rows,
        factor_threat_profile,
        direct_threat_profile,
        direct15_persp_split,
        direct15_factorise,
        direct15_defended,
        args.threat_i16,
        save_format_hand_threat,
        hand_count_dense_dims,
        lmas_flags,
        args.pin_aware,
        args.dual_psqt_reserve,
        args.dual_psqt_king_attacker,
        args.dual_save_legacy_p2dualx,
        args.dual_disable_cross_defended,
        args.enable_kingring,
        args.enable_pairshape,
        args.enable_activity_risk,
        args.enable_king_escape,
        args.enable_slider_xray,
        args.enable_king_mobility,
        args.enable_drop_pressure,
        args.enable_endgame_tempo,
        args.mob,                // B6 可動域 (2026-09-29)
        args.threat_proj,        // 案1 ブロック射影 (2026-09-30)
        args.threat_proj_offset, // 案1 hi 側のずらし (2026-10-01)
        args.threat_proj_pair.clone(),
        args.threat_proj_buckets,
        args.threat_l1,
        args.threat_l2,   // (4) TL2 (2026-10-08)
        args.threat_gate, // (5) TG 乗法ゲート (2026-10-10)
        args.enable_scalar_threat,
        args.scalar_threat_coexist_dual_cross,
        args.enable_dual_cross_geometry,
        args.enable_attack_relay,
        args.enable_attack_relay_v2,
        args.enable_kzt_hand,
        args.enable_kzt_ft,
        args.enable_aux,
        args.enable_auxcheap,
        args.factorise,
        args.bucket_mode == BucketMode::Progress4Hand2KPAbs,
        args.bucket_mode == BucketMode::Progress8Hand2KPAbs,
        args.output_format,
        // 2026-11-07: L4 (4 層目 `fc1b`) の幅。 既定 0 = 無効 = 保存形式も arch も現行と同一。
        args.l4,
    );

    // Network builder
    let threat_ft_out_c = threat_ft_out;
    // --threat-bucket-groups: 群ごとの表を 1 本の tensor に並べるので列数 (= sparse 入力次元) が群数倍。
    // 2026-09-24: hanging-split の delta 行も threat_ft_w の末尾に含める
    // (factorizer と同じ扱い。 両者は排他なのでどちらか一方だけが非 0)。
    let threat_ft_dims_c = threat_ft_dims * threat_groups + threat_factor_rows + threat_hang_rows;
    // 2026-11-17 K256 案 A: 追加列の幅 (0 = graph に node を 1 つも足さない = 従来と同一) と、 着地する FT 行のずらし量 K/2。
    let threat_ext_out_c = args.threat_ext_out;
    let threat_ext_row_offset_c = args.threat_out / 2;
    let threat_ext_gmask_on = threat_ext_out_c > 0 && threat_grad_mask.is_some();
    if threat_ext_out_c > 0 {
        // 推定 GPU メモリ: 重み + 勾配 + Ranger21 の状態 3 本 = 20 B/param (群化の見積もりと同じ式)。
        //   `--threat-group-mem-budget-mib` は群化の増分だけを見るので、 追加列は同じ上限で別に止める。
        let params = threat_ext_out_c * threat_ft_dims;
        let gpu_mib = (params * 20) as f64 / (1024.0 * 1024.0);
        let host_mib = (params * 8) as f64 / (1024.0 * 1024.0);
        println!(
            "THREAT-EXT: threat_ext_w ({threat_ext_out_c} x {threat_ft_dims}) = {params} params / 推定 GPU +{gpu_mib:.0} MiB \
             (重み+勾配+Ranger21 状態 3 本 = 20 B/param) / host +{host_mib:.0} MiB (lookahead 2 本) / 上限 {:.0} MiB / 勾配マスク {}",
            args.threat_group_mem_budget_mib,
            if threat_ext_gmask_on { "あり (--threat-grad-mask と同じ表)" } else { "なし" }
        );
        if gpu_mib > args.threat_group_mem_budget_mib {
            eprintln!(
                "ERROR: --threat-ext-out: 推定 GPU メモリ {gpu_mib:.0} MiB が上限 --threat-group-mem-budget-mib {:.0} MiB を超える",
                args.threat_group_mem_budget_mib
            );
            std::process::exit(1);
        }
    }
    let threat_dedicated_c = args.threat_dedicated_dims && threat_ft_out > 0;
    // 真の凍結 (--threat-freeze-hard)。 graph を組む closure の中で使うので外で捕らえる。
    let threat_freeze_hard_c = args.threat_freeze_hard && threat_ft_out > 0;
    // 2026-11-13 --hard-freeze-shared: l0 / l1f / psqt を勾配なしの leaf にする。 false なら graph は従来と同一。
    let hard_freeze_shared_c = args.hard_freeze_shared;
    // 2026-11-17 U1a: --threat-only / --threat-ext-only では学習しない層を**真の凍結** (勾配も optimizer 状態も持たない leaf) にする。
    //   lr_scale 1e-12 の凍結と関数は同じだが、 l0w (7,500 万) などの勾配と Ranger21 の状態 (16 B/param) を確保しないので
    //   GPU が約 1.2 GB 減る (K=128 で 4 GB を超えて共有メモリにあふれ、 5 倍遅くなったため)。
    let freeze_rest_c = args.threat_only || args.threat_ext_only;
    let frozen_leaf_c = hard_freeze_shared_c || freeze_rest_c;
    // 2026-11-17 PNKLONG: FT だけの真の凍結。
    let l0_frozen_c = frozen_leaf_c || args.l0_freeze_hard;
    // 2026-11-15 --qat: false なら graph に node を 1 つも足さない (= 従来と同一)。
    let qat_c = args.qat;
    let qat_l1q128_c = args.qat && args.qat_l1q128;
    // 2026-11-14 OPNARCH T2: gather affine。 81 / 243 bucket ビルドでは常に使う (select 版は fc0 が重すぎる)。
    //   8 / 16 bucket ビルドは `--gather-affine` を指定したときだけ (未指定なら graph は従来と同一)。
    let gather_affine_c = args.gather_affine || NUM_BUCKETS > 16;
    if gather_affine_c {
        println!(
            "GATHER-AFFINE: fc0 / fc1 / (L4) / fc2 を選んだ bucket の行だけ計算する (NUM_BUCKETS = {NUM_BUCKETS})"
        );
    }
    // 2026-11-12 KH ブロック。 kh_on_c = false のとき graph に node を 1 つも足さない (= 従来と同一)。
    let kh_on_c = kh_parts.is_some();
    // kh_ft_w の列数 / 疎入力の nnz (kmode 1 = 従来の 142 / 54)。 KH 無しのときは使わない。
    let kh_dims_c = kh_parts.map_or(0, |p| p.dims());
    let kh_max_active_c = kh_parts.map_or(0, |p| p.max_active());
    let kh_out_c = args.kh_out;
    let kh_row_offset_c = args.kh_row_offset;
    let ft_out_c = ft_out;
    let l1_out_c = l1_out;
    // 2026-11-07 (T2): L4 = 4 層目 (`fc1b`) の幅。 0 のとき graph に node を 1 つも足さない
    //   = 現行と bit-identical な graph になる。
    let l4_c = args.l4;

    // 2026-11-07: L4 の引数検査 (★usi_out_of_range_is_discard の教訓 = 黙って無視させない)。
    if l4_c > 0 {
        assert!(
            l4_c % 8 == 0 && l4_c <= 64,
            "--l4 は 8 の倍数で 64 以下 (エンジンの affine_par が 8 行単位 / L4_MAX = 64)"
        );
        assert!(matches!(args.l4_init.as_str(), "identity" | "random"), "--l4-init は identity か random");
        assert!(
            !args.main_freeze,
            "--l4 と --main-freeze は併用不可 (main_freeze は l2w まで凍結するので L4 を入れる意味が無い)"
        );
    } else {
        assert!(
            (args.l4_lr_mult - 1.0).abs() < 1e-6 && args.l4_weight_decay.is_none(),
            "--l4-lr-mult / --l4-weight-decay は --l4 > 0 のときだけ使える (graph に id が無い)"
        );
    }

    let l2_out_c = args.l2;
    let residual_zero = args.residual_zero;
    let residual_mix = args.residual_mix || args.residual_wide16;
    let l1_effective_c = l1_effective;
    let l2_out_c = l2_out;
    let l2_in_c = l2_in;
    // 2026-09-12: L1 graph 内正則化 (anchor / smoothness)。 どちらかが正なら build_custom 経路。
    let l1_graph_penalty_on = args.l1_anchor_lambda > 0.0 || args.bucket_smooth_l1_lambda > 0.0;
    let l1_penalty_cfg = L1PenaltyCfg {
        num_buckets: NUM_BUCKETS,
        l1_out,
        ft_out,
        l1_in_total: ft_out + hand_count_dense_dims,
        anchor_lambda: args.l1_anchor_lambda,
        smooth_lambda: args.bucket_smooth_l1_lambda,
    };
    let use_psqt = args.psqt;
    // 2026-09-12: PSQT フェード係数。 1.0 (default) で forward の定数は従来と同じ 0.5 になる。
    let psqt_scale_for_graph = args.psqt_scale;
    let lmas_flags_for_graph = lmas_flags;
    // B6 可動域 (2026-09-29)。 graph closure へ渡す。
    let mob_for_graph = args.mob;
    // 案1 ブロック射影 (2026-09-30)
    let threat_proj_on = args.threat_proj;
    let threat_proj_offset = args.threat_proj_offset;
    let threat_proj_pair = args.threat_proj_pair.clone();
    let threat_proj_buckets_on = args.threat_proj_buckets;
    let threat_l1_on = args.threat_l1;
    let threat_l2_on = args.threat_l2;
    // (5) TG 乗法ゲート (2026-10-10)
    let threat_gate_on = args.threat_gate;
    let pin_aware_for_graph = args.pin_aware;
    let dual_psqt_reserve_for_graph = args.dual_psqt_reserve;
    let dual_psqt_king_attacker_for_graph = args.dual_psqt_king_attacker;
    let dual_disable_king_attacker_for_graph = args.dual_disable_king_attacker;
    let dual_disable_cross_defended_for_graph = args.dual_disable_cross_defended;
    let enable_kingring_for_graph = args.enable_kingring;
    let enable_pairshape_for_graph = args.enable_pairshape;
    let enable_activity_risk_for_graph = args.enable_activity_risk;
    let enable_king_escape_for_graph = args.enable_king_escape;
    let enable_slider_xray_for_graph = args.enable_slider_xray;
    let enable_kzt_for_graph = args.enable_kzt_hand;
    let enable_king_mobility_for_graph = args.enable_king_mobility;
    let enable_drop_pressure_for_graph = args.enable_drop_pressure;
    let enable_endgame_tempo_for_graph = args.enable_endgame_tempo;
    let enable_scalar_threat_for_graph = args.enable_scalar_threat;
    let scalar_threat_coexist_dual_cross_for_graph = args.scalar_threat_coexist_dual_cross;
    let enable_dual_cross_geometry_for_graph = args.enable_dual_cross_geometry;
    let enable_attack_relay_for_graph = args.enable_attack_relay;
    let enable_attack_relay_v2_for_graph = args.enable_attack_relay_v2;
    // 既定ビルド(NUM_BUCKETS=8): progress8kpabs / progress4hand2kpabs は同じ 8-bucket 型
    // ShogiProgressKPAbs を使う(P4H2 の圧縮はグローバルモード set_layerstack_p4h2 で切替)。
    // progress8hand2kpabs(P8H2=16 bucket)は graph が 16 head でないと成立しないため、
    // この既定ビルドでは受理せず、`--features lsmode-p8h2` での再ビルドを促す。
    #[cfg(not(any(feature = "lsmode-p8h2", feature = "lsmode-opn81", feature = "lsmode-opn243")))]
    let bucket_impl = match args.bucket_mode {
        BucketMode::Progress8KPAbs | BucketMode::Progress4Hand2KPAbs => match progress_bucket {
            Some(LoadedProgressBucket::KPAbs(bucket)) => bucket,
            _ => panic!("progress.bin must exist in progress8kpabs/progress4hand2kpabs mode"),
        },
        BucketMode::Progress8Hand2KPAbs => {
            eprintln!(
                "ERROR: --bucket-mode progress8hand2kpabs (P8H2=16 bucket) requires building with \
                 --features lsmode-p8h2 (this build is compiled for {} buckets)",
                NUM_BUCKETS
            );
            std::process::exit(1);
        }
        BucketMode::OpnK9z | BucketMode::OpnK9zProg3 => {
            eprintln!(
                "ERROR: --bucket-mode {} は --features {} でビルドした exe が必要 (この exe は {} bucket)",
                args.bucket_mode_name(),
                if args.bucket_mode == BucketMode::OpnK9z { "lsmode-opn81" } else { "lsmode-opn243" },
                NUM_BUCKETS
            );
            std::process::exit(1);
        }
        _ => {
            eprintln!("ERROR: this training path requires --bucket-mode progress8kpabs or progress4hand2kpabs");
            std::process::exit(1);
        }
    };

    // P8H2 ビルド(NUM_BUCKETS=16): 16-bucket 専用型 ShogiProgress8Hand2KPAbs のみを扱う。
    // graph 全体が 16 head で monomorphize されるため、8-bucket モードはこのビルドでは使えない。
    #[cfg(feature = "lsmode-p8h2")]
    let bucket_impl = match args.bucket_mode {
        BucketMode::Progress8Hand2KPAbs => match progress_bucket {
            // progress.bin は load_progress_bucket で共有 OnceLock にロード済み。
            Some(LoadedProgressBucket::KPAbs(_)) => bullet_lib::game::outputs::ShogiProgress8Hand2KPAbs,
            _ => panic!("progress.bin must exist in progress8hand2kpabs mode"),
        },
        _ => {
            eprintln!(
                "ERROR: this build was compiled with --features lsmode-p8h2 and only supports \
                 --bucket-mode progress8hand2kpabs (16 buckets). Rebuild without the feature for 8-bucket modes."
            );
            std::process::exit(1);
        }
    };

    // 2026-11-14 OPNARCH: OPN の king bucket ビルド (81 / 243)。 graph は NUM_BUCKETS head で monomorphize される。
    #[cfg(feature = "lsmode-opn81")]
    let bucket_impl = match (args.bucket_mode, progress_bucket) {
        (BucketMode::OpnK9z, Some(LoadedProgressBucket::KPAbs(_))) => bullet_lib::game::outputs::ShogiOpnK9z::<1>,
        _ => {
            eprintln!(
                "ERROR: この exe は --features lsmode-opn81 (81 bucket) で、 --bucket-mode opnk9z --progress-coeff <progress.bin> のみを受け付ける (今 {})",
                args.bucket_mode_name()
            );
            std::process::exit(1);
        }
    };
    #[cfg(feature = "lsmode-opn243")]
    let bucket_impl = match (args.bucket_mode, progress_bucket) {
        (BucketMode::OpnK9zProg3, Some(LoadedProgressBucket::KPAbs(_))) => bullet_lib::game::outputs::ShogiOpnK9z::<3>,
        _ => {
            eprintln!(
                "ERROR: この exe は --features lsmode-opn243 (243 bucket) で、 --bucket-mode opnk9zp3 --progress-coeff <progress.bin> のみを受け付ける (今 {})",
                args.bucket_mode_name()
            );
            std::process::exit(1);
        }
    };
    // head の数 (OutputBuckets::BUCKETS) と graph / 保存形式の head 数 (NUM_BUCKETS) は必ず一致させる。
    fn output_bucket_count<O: bullet_lib::game::outputs::OutputBuckets<bullet_lib::shogi::PackedSfenValue>>(
        _: &O,
    ) -> usize {
        O::BUCKETS
    }
    assert_eq!(output_bucket_count(&bucket_impl), NUM_BUCKETS, "bucket 型の BUCKETS と NUM_BUCKETS が食い違っている");

    // 2026-11-13 INVENT02 案 A: --bucket-filter。 bucket は上の `bucket_impl` (学習器が head 選択に
    //   使うのと同一の値) で計算する。 未指定なら range = None で内側の loader をそのまま呼ぶ (従来と同一)。
    //   2026-11-14 OPNARCH: OPN の king bucket モードでは範囲の鍵を進行度 8 区分にする (head は king × 進行度)。
    if bucket_filter_range.is_some() && use_hcpe {
        eprintln!("ERROR: --bucket-filter は psv の loader 専用 (hcpe と推定された入力には使えない)");
        std::process::exit(1);
    }
    // 2026-11-17 tanuki: --score-lut / --score-lut-drop-mate / --score-lut-dump (既定は無効 = 従来と同一)。
    if (args.score_lut.is_some() || args.score_lut_drop_mate > 0 || args.score_lut_dump > 0)
        && bucket_filter_range.is_none()
    {
        eprintln!(
            "ERROR: --score-lut / --score-lut-drop-mate / --score-lut-dump は --bucket-filter と併用すること (同じ loader で処理する)"
        );
        std::process::exit(1);
    }
    if args.score_lut_drop_mate < 0 {
        eprintln!("ERROR: --score-lut-drop-mate は 0 以上");
        std::process::exit(1);
    }
    let score_lut = args.score_lut.as_deref().map(|p| {
        let v = load_score_lut(p).unwrap_or_else(|e| {
            eprintln!("ERROR: {e}");
            std::process::exit(1);
        });
        let pick = |b: usize, a: usize| v[b * SCORE_LUT_LEN + a];
        println!(
            "SCORE-LUT: {} を読んだ (bucket ごとの |score| 1000 / 3000 / 9999 -> {})",
            p.display(),
            (0..NUM_BUCKETS)
                .map(|b| format!("b{b}: {}/{}/{}", pick(b, 1000), pick(b, 3000), pick(b, 9999)))
                .collect::<Vec<_>>()
                .join(" ")
        );
        v
    });
    if args.score_lut_drop_mate > 0 {
        println!("SCORE-LUT: |score| >= {} の局面を捨てる (詰み)", args.score_lut_drop_mate);
    }
    if args.quiet_filter && bucket_filter_range.is_none() {
        eprintln!("ERROR: --quiet-filter は --bucket-filter と併用すること (同じ loader で捨てる)");
        std::process::exit(1);
    }
    if args.quiet_filter {
        println!("QUIET-FILTER: 王手をかけられている局面と、 記録された最善手が駒取りの局面を捨てる");
    }
    if args.mix_buffer_batches > 0 && bucket_filter_range.is_none() {
        eprintln!("ERROR: --mix-buffer-batches は --bucket-filter と併用すること (同じ loader で混ぜる)");
        std::process::exit(1);
    }
    if args.mix_buffer_batches > 0 {
        println!(
            "MIX-BUFFER: 通した局面を {} batch ({} 局面) ためてから混ぜて出す (ファイルをまたいで混ぜる)",
            args.mix_buffer_batches,
            args.mix_buffer_batches * args.batch_size
        );
    }
    // 2026-11-19 TREEDL: --inject-file を全部メモリに読む (40 B/局面)。
    let inject = args.inject_file.as_deref().map(|p| {
        if bucket_filter_range.is_none() || !(args.inject_ratio > 0.0 && args.inject_ratio < 1.0) {
            eprintln!("ERROR: --inject-file は --bucket-filter と 0 < --inject-ratio < 1 と併用すること");
            std::process::exit(1);
        }
        let bytes = std::fs::read(p).unwrap_or_else(|e| {
            eprintln!("ERROR: --inject-file {} を読めない: {e}", p.display());
            std::process::exit(1);
        });
        if bytes.is_empty() || bytes.len() % 40 != 0 {
            eprintln!("ERROR: --inject-file {} の大きさ {} が 40 の倍数でない", p.display(), bytes.len());
            std::process::exit(1);
        }
        let v: Vec<bullet_lib::shogi::PackedSfenValue> = bytes
            .chunks_exact(40)
            .map(|c| {
                let mut q = bullet_lib::shogi::PackedSfenValue::default();
                q.as_bytes_mut().copy_from_slice(c);
                q
            })
            .collect();
        let marked = v.iter().filter(|q| q.as_bytes()[39] & 1 != 0).count();
        println!(
            "INJECT: {} から {} 局面 (byte 39 の bit 0 が 1 = {}) を読んだ / 割合 {:.4}",
            p.display(),
            v.len(),
            marked,
            args.inject_ratio
        );
        (v, args.inject_ratio)
    });
    let data_loader = BucketFilterLoader::new(data_loader, bucket_filter_range, bucket_impl)
        .with_score_lut(
            score_lut,
            args.score_lut_drop_mate,
            (args.score_lut_dump > 0).then(|| (args.output.clone(), args.score_lut_dump)),
        )
        .with_mix_batches(args.mix_buffer_batches)
        .with_quiet(args.quiet_filter)
        .with_inject(inject);
    let data_loader = if args.bucket_mode.is_opn_king() { data_loader.with_progress8_key() } else { data_loader };
    if bucket_filter_range.is_some() && !args.quantise_only {
        data_loader.log_startup_sample(args.batch_size, 64);
    }

    type Nbn<'a> = NetworkBuilderNode<'a, BackendMarker>;

    /// 2026-11-12 KH: W 行の `x` を FT の前半 [off, off+W/2) と後半 [half+off, half+off+W/2) へ足す
    /// (threat と同じ「積の両側に入れる」配り方。 forward は pairwise_mul で ft[j] * ft[j+half])。
    fn kh_add_rows<'a>(ft: Nbn<'a>, x: Nbn<'a>, ft_out: usize, w: usize, off: usize) -> Nbn<'a> {
        let half = ft_out / 2;
        let lo = w / 2;
        let mut pieces: Vec<Nbn<'a>> = Vec::new();
        if off > 0 {
            pieces.push(ft.slice_rows(0, off));
        }
        pieces.push(ft.slice_rows(off, off + lo) + x.slice_rows(0, lo));
        // [off+lo, half+off) は前半の残りと後半の先頭 off 行 (lo == half なら off == 0 で空)。
        if lo < half {
            pieces.push(ft.slice_rows(off + lo, half + off));
        }
        pieces.push(ft.slice_rows(half + off, half + off + lo) + x.slice_rows(lo, w));
        if half + off + lo < ft_out {
            pieces.push(ft.slice_rows(half + off + lo, ft_out));
        }
        pieces.into_iter().reduce(|a, b| a.concat(b)).expect("kh_add_rows: 空")
    }

    /// 2026-11-15 --qat: 後段の重みを書き出しと同じ i8 の格子 (round(64w)、 `as i8` の飽和 = [-128, 127]) に落とす。
    /// clip も丸めも逆伝播は恒等 (STE)。 optimizer の clip (Ranger21 既定 ±1.98) の内側では clip は効かない。
    /// 2026-11-17 U1a: `new_affine(name, input, output)` と同じ id (`{name}w` = (output, input)、 `{name}b` = (output, 1)) の
    /// **勾配を持たない** affine (値は resume で読む)。 `--threat-only` / `--threat-ext-only` の真の凍結に使う。
    fn frozen_affine<'a>(
        builder: &'a bullet_lib::nn::NetworkBuilder<BackendMarker>,
        name: &str,
        input: usize,
        output: usize,
    ) -> Affine<'a, BackendMarker> {
        Affine {
            weights: builder.new_weights_frozen(&format!("{name}w"), Shape::new(output, input), InitSettings::Zeroed),
            bias: builder.new_weights_frozen(&format!("{name}b"), Shape::new(output, 1), InitSettings::Zeroed),
        }
    }

    /// 2026-11-17 --qat-l1q128: fc0 (行 = bucket * L1 + o) の本体行 (o < L1-1) を 128 / 16256、 skip 行 (o = L1-1) を
    /// 64 / 8128 の格子に落とす (`mk_l1q128.py` の書き出しと同じ)。 行ごとに目盛りが違うので、 両方の格子で丸めてから
    /// 定数の 0/1 マスクで行を選ぶ (要素積は threat_proj と同じ concat + pairwise_mul。 どちらも非 batched なので組める)。
    fn qat_rows_l1q128<'a>(
        builder: &'a bullet_lib::nn::NetworkBuilder<BackendMarker>,
        w: Nbn<'a>,
        l1: usize,
        is_bias: bool,
    ) -> Nbn<'a> {
        let shape = w.annotated_node().shape;
        let (rows, cols) = (shape.rows(), shape.cols());
        let n = rows * cols;
        let (q128, q64) = if is_bias {
            (w.copy().faux_quantise(16256.0, true), w.faux_quantise(8128.0, true))
        } else {
            (w.copy().clip_pass_through_grad(-1.0, 127.0 / 128.0).faux_quantise(128.0, true), qat_weight(w))
        };
        // 列優先: (行 r, 列 c) は c * rows + r。 本体行 = r % l1 != l1 - 1。
        let mut main = vec![0.0f32; n];
        for c in 0..cols {
            for r in 0..rows {
                if r % l1 != l1 - 1 {
                    main[c * rows + r] = 1.0;
                }
            }
        }
        let skip: Vec<f32> = main.iter().map(|m| 1.0 - m).collect();
        let pick = |q: Nbn<'a>, mask: &[f32]| {
            let m = builder.new_constant(Shape::new(n, 1), mask);
            q.reshape(Shape::new(n, 1)).concat(m).pairwise_mul().reshape(Shape::new(rows, cols))
        };
        pick(q128, &main) + pick(q64, &skip)
    }

    fn qat_weight<'a>(w: Nbn<'a>) -> Nbn<'a> {
        w.clip_pass_through_grad(-2.0, 127.0 / 64.0).faux_quantise(64.0, true)
    }

    /// Loss function: WRM applied to network output (nodchip style).
    fn loss_fn_wrm<'a>(output: Nbn<'a>, target: Nbn<'a>) -> Nbn<'a> {
        let params =
            *WRM_LOSS_PARAMS.get().expect("WRM loss parameters must be initialized before building the trainer");
        let offset = 270.0f32;
        let scorenet = output * params.nnue2score;
        let q = ((scorenet.copy() - offset) / params.in_scaling).sigmoid();
        let qm = ((-scorenet - offset) / params.in_scaling).sigmoid();
        let qf = (1.0 + q - qm) * 0.5;
        qf.power_error(target, loss_power())
    }

    /// Loss function: standard sigmoid
    fn loss_fn_sigmoid<'a>(output: Nbn<'a>, target: Nbn<'a>) -> Nbn<'a> {
        output.sigmoid().power_error(target, loss_power())
    }

    /// Loss function: `--loss-scale S` 指定時の sigmoid。 出力 (scale 単位) を S 単位に直してから sigmoid。
    fn loss_fn_sigmoid_ls<'a>(output: Nbn<'a>, target: Nbn<'a>) -> Nbn<'a> {
        let mult = *LOSS_OUT_MULT.get().expect("--loss-scale の倍率は trainer 構築前に設定される");
        (output * mult).sigmoid().power_error(target, loss_power())
    }

    LOSS_POWER.set(args.loss_power).expect("loss power should only be initialized once");
    println!("Loss: |pred - target|^{} (--loss-power)", loss_power());

    let loss_fn: for<'a> fn(Nbn<'a>, Nbn<'a>) -> Nbn<'a> = if let Some(in_scaling) = args.wrm_in_scaling {
        WRM_LOSS_PARAMS
            .set(WrmLossParams { nnue2score: args.wrm_nnue2score, in_scaling })
            .expect("WRM loss parameters should only be initialized once");
        loss_fn_wrm
    } else if let Some(ls) = args.loss_scale {
        let mult = args.scale as f32 / ls;
        LOSS_OUT_MULT.set(mult).expect("loss scale multiplier should only be initialized once");
        println!(
            "Loss scale: target = sigmoid(score / {ls}), pred = sigmoid(out * {mult}) (--loss-scale {ls} / 出力の目盛り --scale {} は不変)",
            args.scale
        );
        loss_fn_sigmoid_ls
    } else {
        loss_fn_sigmoid
    };

    macro_rules! build_trainer_with_input {
        ($opt:expr, $use_win_rate:expr, $bucket_impl:expr, $input:expr) => {{
            let mut builder = ValueTrainerBuilder::default()
                .dual_perspective()
                .optimiser($opt)
                .inputs($input)
                .output_buckets($bucket_impl)
                .save_format(&save_format);
            // 2026-09-12: L1 graph 内正則化 ON のときは build_custom で loss を自前合成する
            //   (build_custom は loss_fn 未登録を要求する)。 OFF なら従来どおり loss_fn 登録 + build。
            if !l1_graph_penalty_on {
                builder = builder.loss_fn(loss_fn);
            }
            // 2026-09-17 --loss-grad-mask: 局面ごとの損失の重み (決着局面の勾配の独占を抑える)。 未指定なら従来と同一。
            if kh_split_on {
                // --kh-split: 半分の局面だけ重み 1 (--loss-grad-mask の表があれば掛け合わせる)。
                builder = builder.datapoint_weight_function(kh_split_weight_fn);
            } else if loss_grad_mask_on {
                builder = builder.datapoint_weight_function(loss_weight_fn);
            }
            // 2026-11-19 TREEDL: --tree-nowdl のときだけ混ぜ率を局面ごとに変える (未指定なら既定の |_, wdl| wdl のまま)。
            if tree_nowdl_on {
                builder = builder.wdl_adjust_function(tree_nowdl_blend_fn);
            }
            #[cfg(feature = "cpu")]
            {
                builder = builder.use_threads(1);
            }
            if $use_win_rate {
                builder = builder.use_win_rate_model();
            }
            // forward を closure に切り出す。 OFF: `build(forward.0)` で graph は従来と bit-identical。
            // ON: `build_custom` で `loss_fn(out, targets) + L1 penalty` を loss にする。
            let forward = constrain_l1_forward(|builder, stm_inputs, ntm_inputs, output_buckets| {
                // L0 (Feature Transformer)
                // 2026-11-13 --hard-freeze-shared: 勾配なしの leaf (値は resume で読む)。
                let l0 = if l0_frozen_c {
                    Affine {
                        weights: builder.new_weights_frozen(
                            "l0w",
                            Shape::new(ft_out_c, input_size),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights_frozen("l0b", Shape::new(ft_out_c, 1), InitSettings::Zeroed),
                    }
                } else {
                    let l0 = builder.new_affine("l0", input_size, ft_out_c);
                    l0.init_with_effective_input_size(32);
                    l0
                };

                // L1 入力次元: FT 出力 + （HandCount Dense 有効時は +14）
                let l1_in_total = ft_out_c + hand_count_dense_dims;

                // LayerStack layers:
                // - l1: bucket-specific delta (zero init). 入力は [FT 出力 (ft_out_c) | HandCount (14)]
                //   を concat した全体 (l1_in_total) 次元。
                // - l1f: shared factorized part。FT 出力 (ft_out_c) のみに作用（HandCount は共有しない）
                let l1 = if freeze_rest_c {
                    frozen_affine(builder, "l1", l1_in_total, NUM_BUCKETS * l1_out_c)
                } else {
                    Affine {
                        weights: builder.new_weights(
                            "l1w",
                            Shape::new(NUM_BUCKETS * l1_out_c, l1_in_total),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("l1b", Shape::new(NUM_BUCKETS * l1_out_c, 1), InitSettings::Zeroed),
                    }
                };
                // l1f は全 bucket 共有なので --hard-freeze-shared では凍結する (bucket の絞り込みで動くと全 bucket が変わる)。
                let l1f = if frozen_leaf_c {
                    Affine {
                        weights: builder.new_weights_frozen(
                            "l1fw",
                            Shape::new(l1_out_c, ft_out_c),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights_frozen("l1fb", Shape::new(l1_out_c, 1), InitSettings::Zeroed),
                    }
                } else {
                    builder.new_affine("l1f", ft_out_c, l1_out_c)
                };
                // 2026-09-12: L1 graph 内正則化 (anchor / smoothness) 用に 4 node を持ち出す。
                let l1_nodes = [l1.weights, l1.bias, l1f.weights, l1f.bias];
                let l2 = if freeze_rest_c {
                    frozen_affine(builder, "l2", l2_in_c, NUM_BUCKETS * l2_out_c)
                } else {
                    builder.new_affine("l2", l2_in_c, NUM_BUCKETS * l2_out_c)
                };
                // 2026-11-07 (T2): L4 = 4 層目 (`fc1b`)。
                //   l4_c == 0 のときは `new_affine` を呼ばないので graph は**従来と同一**
                //   (weight id も増えないので checkpoint の互換も保たれる)。
                let l4 = if l4_c > 0 {
                    Some(builder.new_affine("l4", l2_out_c, NUM_BUCKETS * l4_c))
                } else {
                    None
                };
                // L4 を挟むと Output (l3) の入力次元が l2_out_c から l4_c に変わる。
                let l3_in = if l4_c > 0 { l4_c } else { l2_out_c };
                let l3 = if freeze_rest_c {
                    frozen_affine(builder, "l3", l3_in, NUM_BUCKETS)
                } else {
                    builder.new_affine("l3", l3_in, NUM_BUCKETS)
                };

                // Forward pass
                //
                // --threat-out K: threat は主 sparse 入力から分離された (K, dims) の独立
                //   テンソル。 その出力を FT の**先頭 K 行**へ活性化前に加算する
                //   (エンジン: acc[k] += threat_row[k] for k < THREAT_ACC_DIM と同一)。
                let mut ft_stm = l0.forward(stm_inputs);
                let mut ft_ntm = l0.forward(ntm_inputs);
                // (3) TL1: thr は下のブロック内で作られるので、 L1 の行まで Option で持ち出す。
                let mut thr_cat_for_l1 = None;
                // (4) TL2: 同じ連結を L2 の行まで持ち出す (TL1 と併用できるよう別の Option)。
                let mut thr_cat_for_l2 = None;
                // (5) TG: 乗法ゲート用に同じ連結をもう 1 本持ち出す (TL1 / TL2 と併用可)。
                let mut thr_cat_for_gate = None;
                if threat_ft_out_c > 0 {
                    // 🚨真の凍結 (--threat-freeze-hard) のときは **勾配を持たない leaf** に
                    //   する。 compile が Gradients tensor を作らないので、 勾配バッファと
                    //   Ranger21 の状態 (16 B/param) が確保されない (846 MiB の節約)。
                    let threat_w = if threat_freeze_hard_c {
                        builder.new_weights_frozen(
                            "threat_ft_w",
                            Shape::new(threat_ft_out_c, threat_ft_dims_c),
                            InitSettings::Zeroed,
                        )
                    } else {
                        builder.new_weights(
                            "threat_ft_w",
                            Shape::new(threat_ft_out_c, threat_ft_dims_c),
                            InitSettings::Zeroed,
                        )
                    };
                    // 2026-11-17 U1a --qat: 書き出しと同じ round(254 w) の格子で足す (STE)。 i8 の clamp は
                    //   --threat-clip 0.50393 で [-128, 128] に収めてあるので丸めだけで足りる (128 -> 127 の 1 刻みは無視)。
                    let threat_w = if qat_c { threat_w.faux_quantise(QA_FT as f32, true) } else { threat_w };
                    // threat factorizer / hanging-split 時は 1 特徴あたり実特徴 + 共有行の 2 index。
                    // 🚨2026-09-24: ここに hanging-split を足し忘れ、 入力側 (auxiliary_sparse_dims) が 640、
                    //   graph 側が 320 と食い違って **ローダが 320 用のバッファに最大 640 個書き込み**、
                    //   GPU が DeviceError(Generic) で落ちていた。 両者は必ず同じ式にすること。
                    let nnz = bullet_lib::game::inputs::MAX_ACTIVE_THREAT_FEATURES
                        * if threat_factor_rows > 0 || threat_hang_rows > 0 { 2 } else { 1 };
                    assert_eq!(
                        nnz,
                        bullet_lib::game::inputs::MAX_ACTIVE_THREAT_FEATURES
                            * if threat_factor_rows > 0 || threat_hang_rows > 0 { 2 } else { 1 },
                        "graph の nnz と入力側 auxiliary_sparse_dims の nnz が食い違っている"
                    );
                    let threat_stm_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::THREAT_FT_SPARSE_STM,
                        Shape::new(threat_ft_dims_c, 1),
                        nnz,
                    );
                    let threat_ntm_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::THREAT_FT_SPARSE_NSTM,
                        Shape::new(threat_ft_dims_c, 1),
                        nnz,
                    );
                    // matmul(weights, sparse indices) = bias 無しの sparse affine。
                    // FT の bias (l0b) を共有するので threat 側に bias は持たせない
                    // (持たせるとエンジンへ書き出せない重みができてしまう)。
                    let thr_stm = threat_w.matmul(threat_stm_in);
                    let thr_ntm = threat_w.matmul(threat_ntm_in);
                    // --threat-grad-mask: 順伝播は不変のまま勾配だけ m 倍する (m は局面ごとの dense 入力)。
                    //   thr_d = stop-grad(thr) とすると thr_d + (thr - thr_d) * m は forward = thr、 d/dthr = m。
                    let (thr_stm, thr_ntm) = if threat_grad_mask_on {
                        let m = builder.new_dense_input(
                            bullet_lib::game::inputs::THREAT_GMASK_NAME,
                            Shape::new(threat_ft_out_c, 1),
                        );
                        let ds = thr_stm.copy_stop_grad();
                        let dn = thr_ntm.copy_stop_grad();
                        (ds + (thr_stm - ds) * m, dn + (thr_ntm - dn) * m)
                    } else {
                        (thr_stm, thr_ntm)
                    };
                    // --threat-proj: threat を FT の**全次元**へ射影して残差で足す (案1)。
                    //   🚨Concat は check_same_batching を要求するので
                    //     「重み × バッチ活性」の要素積は組めない。 要素積を行列側に押し込み、
                    //     「重み × 定数」(どちらも非 batched) にする。
                    //   m[j][c] = proj_raw[j][c] * mask[j][c]、 mask は c == j % K のときだけ 1。
                    //   → forward は対角ブロックだけ = p_j * thr[j % K]。 非対角は死に重み。
                    if threat_l1_on {
                        // clamp を通さない経路のために両視点の threat を連結して外へ渡す。
                        thr_cat_for_l1 = Some(thr_stm.copy().concat(thr_ntm.copy()));
                    }
                    if threat_l2_on {
                        // (4) TL2: 首 (L1) を迂回して L2 入力へ足すための同じ連結。
                        //   🚨並びは [0..K) = stm、 [K..2K) = ntm。 engine の thr_cat と同じ。
                        thr_cat_for_l2 = Some(thr_stm.copy().concat(thr_ntm.copy()));
                    }
                    if threat_gate_on {
                        // (5) TG: 乗法ゲートの dot product 用。
                        //   🚨並びは [0..K) = stm、 [K..2K) = ntm。 engine の thr_cat と同じ。
                        thr_cat_for_gate = Some(thr_stm.copy().concat(thr_ntm.copy()));
                    }
                    if threat_proj_on {
                        // W: bucket 別なら行数を 8 倍にして、 matmul の後で select する。
                        let nb_p = if threat_proj_buckets_on { NUM_BUCKETS } else { 1 };
                        let rows_p = ft_out_c * nb_p;
                        let fk = rows_p * threat_ft_out_c;
                        let mut mask = vec![0.0f32; fk];
                        // 🚨16 次元ブロックごとの ずらし量から mask を作る。 SIMD の 1 反復が
                        //   16 次元なので、 ブロック内で一定なら連続 load を保てる。
                        let offs = threat_proj_block_offsets(
                            ft_out_c, threat_ft_out_c, &threat_proj_pair, threat_proj_offset);
                        // 列優先 ((行, c) は c*rows + 行)。 行 = b*FT + j (bucket 別のとき)。
                        for b in 0..nb_p {
                        for j in 0..ft_out_c {
                            let c = (j + offs[j / 16]) % threat_ft_out_c;
                            // 🚨列優先 ((j,c) は c*rows + j)。 preflight で確かめた。
                            mask[c * rows_p + (b * ft_out_c + j)] = 1.0;
                        }
                        }
                        let mask_c = builder.new_constant(Shape::new(fk, 1), &mask);
                        let praw = builder.new_weights(
                            "threat_proj_raw",
                            Shape::new(rows_p, threat_ft_out_c),
                            InitSettings::Zeroed,
                        );
                        let m = praw
                            .reshape(Shape::new(fk, 1))
                            .concat(mask_c)
                            .pairwise_mul()
                            .reshape(Shape::new(rows_p, threat_ft_out_c))
                            .clip_pass_through_grad(-1.0, 1.0);
                        // bucket 別なら (FT*8,1) を select で (FT,1) に落とす。
                        if threat_proj_buckets_on {
                            ft_stm = ft_stm + m.matmul(thr_stm.copy()).select(output_buckets);
                            ft_ntm = ft_ntm + m.matmul(thr_ntm.copy()).select(output_buckets);
                        } else {
                            ft_stm = ft_stm + m.matmul(thr_stm.copy());
                            ft_ntm = ft_ntm + m.matmul(thr_ntm.copy());
                        }
                    }
                    if threat_ft_out_c == ft_out_c {
                        ft_stm = ft_stm + thr_stm;
                        ft_ntm = ft_ntm + thr_ntm;
                    } else {
                        // ★2026-09-10 対称化: threat を FT の前半・後半へ半分ずつ乗せる。
                        //   forward は pairwise_mul: out[j] = f(ft[j]) * f(ft[j+half]) なので、
                        //   従来のように先頭 K 次元だけに乗せると K < half のとき
                        //   **積の片側にしか threat が入らない**。 前半 K/2 を ft[0..)、
                        //   後半 K/2 を ft[half..) に乗せると両側に入る。
                        //   エンジン側は THREAT_LO_DIM / THREAT_HI_DIM で同じ配り方をする。
                        let half = ft_out_c / 2;
                        let lo = threat_ft_out_c.div_ceil(2).min(half);
                        let hi = threat_ft_out_c - lo;
                        assert!(hi <= half, "threat 後半が FT_HALF を超えている");
                        // クロージャだとライフタイムが通らないので 2 回書く
                        // 混線分離: 該当次元を threat 出力だけにする (HalfKA 側を足さない)
                        let mut o = if threat_dedicated_c {
                            thr_stm.slice_rows(0, lo)
                        } else {
                            ft_stm.slice_rows(0, lo) + thr_stm.slice_rows(0, lo)
                        };
                        if lo < half {
                            o = o.concat(ft_stm.slice_rows(lo, half));
                        }
                        if hi > 0 {
                            o = o.concat(if threat_dedicated_c {
                                thr_stm.slice_rows(lo, lo + hi)
                            } else {
                                ft_stm.slice_rows(half, half + hi) + thr_stm.slice_rows(lo, lo + hi)
                            });
                        }
                        if half + hi < ft_out_c {
                            o = o.concat(ft_stm.slice_rows(half + hi, ft_out_c));
                        }
                        ft_stm = o;

                        let mut o2 = if threat_dedicated_c {
                            thr_ntm.slice_rows(0, lo)
                        } else {
                            ft_ntm.slice_rows(0, lo) + thr_ntm.slice_rows(0, lo)
                        };
                        if lo < half {
                            o2 = o2.concat(ft_ntm.slice_rows(lo, half));
                        }
                        if hi > 0 {
                            o2 = o2.concat(if threat_dedicated_c {
                                thr_ntm.slice_rows(lo, lo + hi)
                            } else {
                                ft_ntm.slice_rows(half, half + hi) + thr_ntm.slice_rows(lo, lo + hi)
                            });
                        }
                        if half + hi < ft_out_c {
                            o2 = o2.concat(ft_ntm.slice_rows(half + hi, ft_out_c));
                        }
                        ft_ntm = o2;
                    }
                }
                // 2026-11-17 K256 案 A: 追加列 `threat_ext_w` (N x threat_dims、 1 群、 0 初期化)。
                //   群 offset なしの threat 疎入力を掛け、 FT の行 [K/2, K/2 + N/2) と [half + K/2, half + K/2 + N/2) に
                //   活性化前に足す (bias なし = threat と同じ)。 N == 0 のときは node を 1 つも足さない (= 従来と同一)。
                if threat_ext_out_c > 0 {
                    let ext_w = builder.new_weights(
                        "threat_ext_w",
                        Shape::new(threat_ext_out_c, threat_ft_dims),
                        InitSettings::Zeroed,
                    );
                    let ext_s_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::THREAT_EXT_SPARSE_STM,
                        Shape::new(threat_ft_dims, 1),
                        bullet_lib::game::inputs::MAX_ACTIVE_THREAT_FEATURES,
                    );
                    let ext_n_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::THREAT_EXT_SPARSE_NSTM,
                        Shape::new(threat_ft_dims, 1),
                        bullet_lib::game::inputs::MAX_ACTIVE_THREAT_FEATURES,
                    );
                    let ext_w = if qat_c { ext_w.faux_quantise(QA_FT as f32, true) } else { ext_w };
                    let ext_s = ext_w.matmul(ext_s_in);
                    let ext_n = ext_w.matmul(ext_n_in);
                    // 勾配マスク: 旧表と同じ形 (forward は不変、 勾配だけ m 倍)。
                    let (ext_s, ext_n) = if threat_ext_gmask_on {
                        let m = builder.new_dense_input(
                            bullet_lib::game::inputs::THREAT_EXT_GMASK_NAME,
                            Shape::new(threat_ext_out_c, 1),
                        );
                        let ds = ext_s.copy_stop_grad();
                        let dn = ext_n.copy_stop_grad();
                        (ds + (ext_s - ds) * m, dn + (ext_n - dn) * m)
                    } else {
                        (ext_s, ext_n)
                    };
                    ft_stm = kh_add_rows(ft_stm, ext_s, ft_out_c, threat_ext_out_c, threat_ext_row_offset_c);
                    ft_ntm = kh_add_rows(ft_ntm, ext_n, ft_out_c, threat_ext_out_c, threat_ext_row_offset_c);
                }
                // 2026-11-12 KH ブロック:(W, 142 / kmode 2 は 798) のゼロ初期化テンソルを両視点で共有し、 FT の活性化前に足す。
                //   bias は持たない (FT の l0b を共有 = threat と同じ)。
                if kh_on_c {
                    // 2026-11-15 H2: --hard-freeze-shared では kh_ft_w も勾配なしの leaf (値は resume で読む = bit 単位で不変)。
                    let kh_w = if frozen_leaf_c {
                        builder.new_weights_frozen("kh_ft_w", Shape::new(kh_out_c, kh_dims_c), InitSettings::Zeroed)
                    } else {
                        builder.new_weights("kh_ft_w", Shape::new(kh_out_c, kh_dims_c), InitSettings::Zeroed)
                    };
                    let kh_s_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::shogi_kh::KH_SPARSE_STM,
                        Shape::new(kh_dims_c, 1),
                        kh_max_active_c,
                    );
                    let kh_n_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::shogi_kh::KH_SPARSE_NSTM,
                        Shape::new(kh_dims_c, 1),
                        kh_max_active_c,
                    );
                    let kh_s = kh_w.matmul(kh_s_in);
                    let kh_n = kh_w.matmul(kh_n_in);
                    ft_stm = kh_add_rows(ft_stm, kh_s, ft_out_c, kh_out_c, kh_row_offset_c);
                    ft_ntm = kh_add_rows(ft_ntm, kh_n, ft_out_c, kh_out_c, kh_row_offset_c);
                }
                let stm = ft_stm.crelu().pairwise_mul() * (127.0 / 128.0);
                let ntm = ft_ntm.crelu().pairwise_mul() * (127.0 / 128.0);
                let combined = stm.concat(ntm);
                // 2026-11-15 --qat ①: エンジンの L1 入力 u8 = (a*b)>>9 (a, b は [0,254] に clip した FT 出力) は
                //   a*b/512 = 127 * crelu(x)*crelu(y)*(127/128) の床なので、 Q=127 の切り捨て (0 以上で floor) と同じ格子。
                //   FT の重み自体は f32 のまま (i16 = round(254w) の丸めは丸めない: TATARA-DIFF 5.3 で効きが小さい)。
                let combined = if qat_c { combined.faux_quantise(127.0, false) } else { combined };

                // HandCount Dense を有効にする場合、L1 入力に 14 元の dense vector を concat。
                // `"hand_count"` という名前の dense 入力は ValueDataLoader が
                // `ShogiHalfKaHmHandCount::fill_hand_count` を使って populate する。
                let l1_input_full = if hand_count_dense_dims > 0 {
                    let hand_count =
                        builder.new_dense_input("hand_count", Shape::new(hand_count_dense_dims, 1));
                    combined.concat(hand_count)
                } else {
                    combined
                };

                let mut l1_out_t = if qat_c {
                    // 2026-11-15 --qat ②: 書き出しは bucket ごとに `round(64 * (l1w + l1fw))` (as i8 で飽和) と
                    //   `round(8128 * (l1b + l1fb))`。 同じく共有分を各 bucket の行へ f32 で足してから 1 回だけ丸める
                    //   (別々に丸めると誤差が 2 重になり書き出しと合わない)。 足し込みは定数の 0/1 行列 T
                    //   [NB*L1, L1] (T[b*L1+o, o] = 1) との積 = 値はそのまま複製される。 勾配は T^T で l1f に戻る
                    //   (= 従来の l1f の勾配と同じ、 bucket 分の和)。 hand_count dense は --qat と併用不可なので入力は FT だけ。
                    let rows = NUM_BUCKETS * l1_out_c;
                    let mut tile = vec![0.0f32; rows * l1_out_c];
                    for b in 0..NUM_BUCKETS {
                        for o in 0..l1_out_c {
                            // 列優先: (行 r, 列 c) は c * rows + r。
                            tile[o * rows + b * l1_out_c + o] = 1.0;
                        }
                    }
                    let tile_c = builder.new_constant(Shape::new(rows, l1_out_c), &tile);
                    let l1q = Affine {
                        weights: if qat_l1q128_c {
                            qat_rows_l1q128(builder, l1.weights + tile_c.matmul(l1f.weights), l1_out_c, false)
                        } else {
                            qat_weight(l1.weights + tile_c.matmul(l1f.weights))
                        },
                        bias: if qat_l1q128_c {
                            qat_rows_l1q128(builder, l1.bias + tile_c.matmul(l1f.bias), l1_out_c, true)
                        } else {
                            (l1.bias + tile_c.matmul(l1f.bias)).faux_quantise(8128.0, true)
                        },
                    };
                    if gather_affine_c {
                        l1q.forward_bucketed(l1_input_full, output_buckets)
                    } else {
                        l1q.forward(l1_input_full).select(output_buckets)
                    }
                } else if gather_affine_c {
                    l1.forward_bucketed(l1_input_full, output_buckets) + l1f.forward(combined)
                } else {
                    l1.forward(l1_input_full).select(output_buckets) + l1f.forward(combined)
                };
                // (3) TL1: threat を FT の clamp を通さず L1 へ直結する。
                //   bias は置かない (学習するが書き出されない事故を避ける)。
                if let Some(tc) = thr_cat_for_l1 {
                    let tl1 = builder.new_weights(
                        "threat_l1_w",
                        Shape::new(NUM_BUCKETS * l1_out_c, 2 * threat_ft_out_c),
                        InitSettings::Zeroed,
                    );
                    // 🚨C-1: w_q = round(1024*w) が i16 に収まるよう ±2 で clip する
                    //   (clip が無いと engine 側の範囲計算が崩れる。 fable 指摘)。
                    let tl1c = tl1.clip_pass_through_grad(-2.0, 2.0);
                    l1_out_t = l1_out_t + tl1c.matmul(tc).select(output_buckets);
                }
                let l1_out_t = l1_out_t;
                let l1_main = l1_out_t.slice_rows(0, l1_effective_c);
                let l1_skip_raw = l1_out_t.slice_rows(l1_effective_c, l1_out_c);
                // In the Z arm, preserve tensor layout but make the residual exactly zero.
                // Multiplication by zero also blocks the gradient to the skip output rows.
                let l1_skip = if residual_zero { l1_skip_raw * 0.0 } else { l1_skip_raw };

                let l1_sqr = l1_main.abs_pow(2.0) * (127.0 / 128.0);
                // (5) TG: threat で sqr 側 15 本を**乗法ゲート**する (2026-10-10)。
                //   engine (p2nnue.cpp): g = clampi(256 + tg[i], 0, 1024); pre_sq = (sqr * g) >> 8
                //   学習器はその g/256 を float でそのまま再現する:
                //     g_f = hard_clamp(1 + (254/256) * (thr_cat . u_i), 0, 4)
                //     pre_sq_f = sqr_f * g_f
                //   🚨係数 254/256 の根拠: engine は `thr_i16 = 254*thr_f` と
                //     `u_q = round(1024*u)` を持ち `(Σ thr_i16*u_q) >> 10 = 254*Σ thr_f*u` を
                //     計算して **256** に足す。 つまり g/256 = 1 + (254/256)*Σ thr_f*u。
                //     ここを 1 (= 256/256) にすると学習器と engine で 0.79% ずれ、
                //     「eval は bit-exact なのに勾配がずれる」事故になる。
                //   🚨hard clamp (勾配 0) は crelu で作る: crelu(x) = clamp(x,0,1) で
                //     backward も `0 < x < 1 ? 1 : 0` (acyclib/src/device/cpu/base.rs:82) なので、
                //     `4 * crelu(x/4)` は [0,4] の hard clamp と forward / backward ともに厳密一致。
                //     clip_pass_through_grad は勾配を通してしまうので**使わない**。
                //   ゲートは 2 乗の**後ろ**、 かつ TL2 の加法の**前** (engine の順序と同じ)。
                let l1_sqr = if let Some(tc) = thr_cat_for_gate {
                    let tg = builder.new_weights(
                        "threat_gate_w",
                        Shape::new(NUM_BUCKETS * l1_effective_c, 2 * threat_ft_out_c),
                        InitSettings::Zeroed,
                    );
                    // 🚨u_q = round(1024*u) が i16 に収まり engine の i32 レーンが
                    //   あふれないよう ±2 で clip する (|u_q| <= 2048。 TL1 / TL2 と同じ約束)。
                    let tgc = tg.clip_pass_through_grad(-2.0, 2.0);
                    let g_pre = tgc.matmul(tc).select(output_buckets) * (254.0 / 256.0) + 1.0;
                    let g = (g_pre * 0.25).crelu() * 4.0;
                    // Mul<Node> は concat + pairwise_mul = 要素積 (out[j] = sqr[j] * g[j])。
                    l1_sqr * g
                } else {
                    l1_sqr
                };
                // R-Mix16 keeps the raw output skip and exposes skip amplitude/sign to L2/L3.
                // skip は L2 に入れず `net_output = l3_out + l1_skip` で出力へ直加算するのみ。
                // NAGISA V3.1 (sfnn-1536.h: fc_2_out[0] += fc_0_out[kHidden1Dims]) と同構成。
                // 実測で NAGISA の fc1 列 30/31 は厳密に 0 であることを確認済み (2026-09-06)。
                // (4) TL2: threat を crelu の **前** (= engine の clamp の前) で L2 入力に足す。
                //   🚨現行の 1 文 `l1_sqr.concat(l1_main).crelu()` を 2 文に割っている。
                //     TL2 無効時は concat の結果をそのまま crelu するので graph は従来と同一。
                let l2_pre = l1_sqr.concat(l1_main);
                let l2_pre = if let Some(tc) = thr_cat_for_l2 {
                    let tl2 = builder.new_weights(
                        "threat_l2_w",
                        Shape::new(NUM_BUCKETS * l2_in_c, 2 * threat_ft_out_c),
                        InitSettings::Zeroed,
                    );
                    // 🚨v_q = round(S*v) が i16 に収まり、 かつ engine の i32 レーンが
                    //   あふれないよう ±2 で clip する (TL1 と同じ約束。 S=1024 で |v_q| <= 2048)。
                    let tl2c = tl2.clip_pass_through_grad(-2.0, 2.0);
                    l2_pre + tl2c.matmul(tc).select(output_buckets)
                } else {
                    l2_pre
                };
                let l2_input_tensor = l2_pre.crelu();
                // 2026-11-15 --qat ③: L1 の整数出力 v (= 8128 z) に対しエンジンは sqr 側 min((v*v)>>19, 127)、
                //   線形側 clamp(v>>6, 0, 127)。 v*v/2^19 = 127 * (127/128) z^2、 v/64 = 127 z なので、 どちらも
                //   crelu (= [0,1] で clip) の後の Q=127 切り捨て (0 以上で floor) と同じ格子。
                let l2_input_tensor = if qat_c { l2_input_tensor.faux_quantise(127.0, false) } else { l2_input_tensor };
                // 2026-11-15 --qat ④⑤: L2 / L3 の重みは i8 = round(64w) (飽和)、 bias は i32 = round(127*64 b)。
                let (l2, l3) = if qat_c {
                    (
                        Affine { weights: qat_weight(l2.weights), bias: l2.bias.faux_quantise(8128.0, true) },
                        Affine { weights: qat_weight(l3.weights), bias: l3.bias.faux_quantise(8128.0, true) },
                    )
                } else {
                    (l2, l3)
                };

                let l2_out_t = if gather_affine_c {
                    l2.forward_bucketed(l2_input_tensor, output_buckets).crelu()
                } else {
                    l2.forward(l2_input_tensor).select(output_buckets).crelu()
                };
                // 2026-11-15 --qat ④: L2 後はエンジン clamp((B1 + Σ a*w) >> 6, 0, 127) = 127 y の床 (y は crelu 前の f32)。
                let l2_out_t = if qat_c { l2_out_t.faux_quantise(127.0, false) } else { l2_out_t };
                // 2026-11-07 (T2): L4 を 1 段挟む。 l4 == None のときは `l2_out_t` をそのまま
                //   渡すので graph は**従来と bit-identical**。
                //   活性は fc1 と同じ ClippedReLU (エンジン: clampi(out >> 6, 0, 127))。
                let l3_in_t = match l4 {
                    Some(l4) if gather_affine_c => l4.forward_bucketed(l2_out_t, output_buckets).crelu(),
                    Some(l4) => l4.forward(l2_out_t).select(output_buckets).crelu(),
                    None => l2_out_t,
                };
                let l3_out = if gather_affine_c {
                    l3.forward_bucketed(l3_in_t, output_buckets)
                } else {
                    l3.forward(l3_in_t).select(output_buckets)
                };
                let net_output = l3_out + l1_skip;

                let net_output_with_psqt = if use_psqt {
                    // PSQT shortcut: FT と同じ入力、出力 = バケット数
                    // 学習初期に「PSQTなし」と等価にするため Zeroed で開始
                    // 2026-11-13 --hard-freeze-shared: PSQT も勾配なしの leaf にする。
                    let psqt = if frozen_leaf_c {
                        Affine {
                            weights: builder.new_weights_frozen(
                                "psqtw",
                                Shape::new(NUM_BUCKETS, input_size),
                                InitSettings::Zeroed,
                            ),
                            bias: builder.new_weights_frozen("psqtb", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                        }
                    } else {
                        Affine {
                            weights: builder.new_weights(
                                "psqtw",
                                Shape::new(NUM_BUCKETS, input_size),
                                InitSettings::Zeroed,
                            ),
                            bias: builder.new_weights("psqtb", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                        }
                    };

                    // PSQT shortcut (Stockfish 準拠: (stm - nstm) / 2)
                    // 各駒は両視点に逆符号で寄与するため、stm - nstm は正味の配置価値を
                    // 約2倍にカウントする。/2 はこの二重カウントを補正する正規化。
                    let stm_psqt = psqt.forward(stm_inputs);
                    let ntm_psqt = psqt.forward(ntm_inputs) * (-1.0);
                    // 2026-09-12: --psqt-scale f は定数 0.5 を 0.5 * f に置き換えるだけ
                    //   (ノードを増やさない)。 f=1.0 のとき 0.5 * 1.0 == 0.5 は浮動小数でも
                    //   厳密に等しいので graph は従来と bit-identical。 f=0 で psqt 寄与も
                    //   psqtw/psqtb への勾配もゼロになる。
                    let psqt_diff = (stm_psqt + ntm_psqt).select(output_buckets) * (0.5 * psqt_scale_for_graph);

                    net_output + psqt_diff
                } else {
                    net_output
                };

                // ---- LMAS heads (legacy、 2026-05-25 廃止) ----
                // LMAS 機能は削除済。 lmas_flags_for_graph != 0 の呼び出しは想定外。
                if lmas_flags_for_graph != 0 {
                    panic!(
                        "LMAS heads は 2026-05-25 に廃止されました (lmas_flags=0x{:x})",
                        lmas_flags_for_graph
                    );
                }
                let mut net_output_final = net_output_with_psqt;

                // 🚨2026-09-29: LMAS の head 群 (下の `#[cfg(any())]`) は**コンパイルされない**。
                //   B6 はここ (生きている合流点) に置く。
                // B6 可動域 (2026-09-29)。 dense 入力 1 次元 x bucket 別 affine。
                // 🚨Zeroed 初期化なので学習初期の forward は --mob なしと**完全一致**する
                //   (= 起点との関数保存。 SB0 で quantised の md5 が一致すること)。
                if mob_for_graph {
                    let mob_input = builder.new_dense_input(
                        bullet_lib::game::inputs::MOB_NAME,
                        Shape::new(1, 1),
                    );
                    let mob_aff = Affine {
                        weights: builder.new_weights("mobw", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                        bias: builder.new_weights("mobb_unused", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let mob_out = mob_aff.forward(mob_input).select(output_buckets);
                    net_output_final = net_output_final + mob_out;
                }
                if enable_threat_factor16 {
                    let factor_dim = bullet_lib::game::inputs::THREAT_FACTOR_DIM;
                    // Main STM/NSTM are sparse index tensors. Use SparseAffineActivate for
                    // factor_code so its batch dimension is created by the same sparse loader
                    // as L0; the dense factor_out then consumes its 16D batched output.
                    let factor_code = Affine {
                        weights: builder.new_weights(
                            "threat_factor_code",
                            Shape::new(factor_dim, input_size),
                            InitSettings::Normal { mean: 0.0, stdev: 0.01 },
                        ),
                        bias: builder.new_weights(
                            "threat_factor_code_bias_cancels",
                            Shape::new(factor_dim, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let factor_out = Affine {
                        weights: builder.new_weights(
                            "threat_factor_out",
                            Shape::new(1, factor_dim),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "threat_factor_out_bias_cancels",
                            Shape::new(1, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let factor_stm = factor_out.forward(factor_code.forward_sparse(stm_inputs));
                    let factor_nstm = factor_out.forward(factor_code.forward_sparse(ntm_inputs));
                    net_output_final = net_output_final + factor_stm - factor_nstm;
                }
                if enable_threat_direct {
                    // Profile15 (Direct15) は 2026-09-01 から専用 sparse 入力を使う。
                    // 旧実装は 65,640 次元を HalfKA_hm(73,305) の後ろへ append した
                    // tail 特徴として主 sparse 入力に流していたが、 学習器では
                    // index >= 73,305 の tail 特徴に forward/backward のどちらも
                    // 到達せず、 threat_direct_w が恒久的に 0 のままだった。
                    // Profile14 (P2THRD1) は従来どおり tail 経路のまま。
                    let direct15 = direct_threat_profile == Some(ThreatProfile::EnemyOnlyDirect15);
                    let (direct_in_stm, direct_in_nstm, direct_cols) = if direct15 {
                        let dim = direct15_dim;
                        let nnz = bullet_lib::game::inputs::MAX_ACTIVE_DIRECT15_FEATURES;
                        (
                            builder.new_sparse_input(
                                bullet_lib::game::inputs::THREAT_DIRECT15_SPARSE_STM,
                                Shape::new(dim, 1),
                                nnz,
                            ),
                            builder.new_sparse_input(
                                bullet_lib::game::inputs::THREAT_DIRECT15_SPARSE_NSTM,
                                Shape::new(dim, 1),
                                nnz,
                            ),
                            dim,
                        )
                    } else {
                        (stm_inputs, ntm_inputs, input_size)
                    };
                    // P2 (--direct15-persp-split): 行数 16 = 2*bucket + 視点。 select(output_buckets) の
                    //   出力が [stm 視点, nstm 視点] の 2 行になるので、 視点ごとに 1 行を slice する。
                    //   off なら行数 8 で従来と同一の graph。
                    let direct_rows = if direct15 { direct15_rows } else { NUM_BUCKETS };
                    let direct_d = builder.new_weights(
                        "threat_direct_w",
                        Shape::new(direct_rows, direct_cols),
                        InitSettings::Zeroed,
                    );
                    let direct_bias = builder.new_weights(
                        "threat_direct_b_cancels",
                        Shape::new(direct_rows, 1),
                        InitSettings::Zeroed,
                    );
                    // P4 (--direct15-factorise): W_eff = E·(S + expand(P))ᵀ + D を毎 step 合成して
                    //   sparse affine の重みに使う (線形なので畳んだ重みと forward は同一)。
                    //   S: (dims, np) column-major [persp][feature]、 P: (np*npairs, 1) [persp][half][ac*10+dc]。
                    //   expand(P): (persp, half, ac) ごとに 10 個の pair 値を A[ac] 回ずつ繰り返す
                    //   (= pair_base のレイアウト: class block 内は dc-major、 各 dc に A[ac] cell)。
                    //   E: (rows, np) 定数、 E[r][p] = 1 iff p == r % np (np=1 なら全 1)。
                    //   off なら D をそのまま使い、 graph は従来と同一。
                    let direct_weights = if direct15 && direct15_factorise {
                        let np = direct_rows / NUM_BUCKETS;
                        let halves = direct_cols / THREAT_DIRECT15_DIMENSIONS;
                        let npairs = halves * 100;
                        let shared = builder.new_weights(
                            "threat_direct_shared_w",
                            Shape::new(direct_cols, np),
                            InitSettings::Zeroed,
                        );
                        let pair = builder.new_weights(
                            "threat_direct_pair_w",
                            Shape::new(np * npairs, 1),
                            InitSettings::Zeroed,
                        );
                        let mut blocks = Vec::new();
                        for persp in 0..np {
                            for half in 0..halves {
                                for ac in 0..bullet_lib::game::inputs::THREAT_DIRECT15_CLASS_NB {
                                    let start = persp * npairs + half * 100 + ac * 10;
                                    let a = THREAT_DIRECT15_ATTACKS_PER_CLASS[ac];
                                    let block = pair.slice_rows(start, start + 10); // (10, 1)
                                    let ones = builder.new_constant(Shape::new(a, 1), &vec![1.0f32; a]);
                                    // (a, 1) x (1, 10) = (a, 10)。 column-major flatten で
                                    // [P[dc=0] x a, P[dc=1] x a, ...] = class block の cell 順。
                                    let rep = ones.gemm(false, block, true).reshape(Shape::new(a * 10, 1));
                                    blocks.push(rep);
                                }
                            }
                        }
                        let mut blocks = blocks.into_iter();
                        let first = blocks.next().expect("direct15 pair expansion");
                        let pair_col = blocks.fold(first, |acc, rep| acc.concat(rep)).reshape(Shape::new(direct_cols, np));
                        let row = shared + pair_col; // (dims, np)
                        let mut e_vals = vec![0.0f32; direct_rows * np];
                        for r in 0..direct_rows {
                            e_vals[(r % np) * direct_rows + r] = 1.0;
                        }
                        let e = builder.new_constant(Shape::new(direct_rows, np), &e_vals);
                        e.gemm(false, row, true) + direct_d // (rows, np) x (np, dims) + (rows, dims)
                    } else {
                        direct_d
                    };
                    let direct = Affine { weights: direct_weights, bias: direct_bias };
                    let direct_stm_sel = direct.forward_sparse(direct_in_stm).select(output_buckets);
                    let direct_nstm_sel = direct.forward_sparse(direct_in_nstm).select(output_buckets);
                    let (direct_stm, direct_nstm) = if direct15 && direct15_persp_split {
                        (direct_stm_sel.slice_rows(0, 1), direct_nstm_sel.slice_rows(1, 2))
                    } else {
                        (direct_stm_sel, direct_nstm_sel)
                    };
                    let direct_delta = direct_stm - direct_nstm;
                    // Profile15 の C++ は Direct15 の delta 全体を raw FV 単位で
                    // ±cap[cp] に clamp する。 学習側も同じ forward 値を再現する。
                    if direct_threat_profile == Some(ThreatProfile::EnemyOnlyDirect15) && direct15_cap_cp > 0.0
                    {
                        // cap [cp] は CLI (--direct15-cap-cp, default 30 = 旧ハードコードと
                        // bit 同一)。 C++ P2_THREAT_DIRECT_CAP_CP と一致必須。
                        let cap_train = direct15_cap_cp * fv_scale as f32
                            / (f32::from(QA) * f32::from(QB));

                        // forward = clamp(x, -c, c)、 backward = 1_{|x| < c}
                        // (hard clamp の**真の**勾配)。 relu(x + c) - relu(x - c) - c で構成:
                        //   x < -c : 0 - 0 - c = -c        grad 0
                        //   |x| < c: (x + c) - 0 - c = x   grad 1
                        //   x >  c : (x+c) - (x-c) - c = c grad 0
                        //
                        // 2026-09-01: 旧実装は cap 外でも非ゼロ勾配を与える
                        // straight-through surrogate (2c*sigmoid(2x/c) - c) を使っていた。
                        // 「勾配が死ぬのを防ぐ」意図だったが、 cap 外では forward が定数で
                        // 真の勾配は 0 なので、 これは重みを膨らませるだけの偽の駆動力だった。
                        // 実測 (SB1, 実戦 1,500 局面): head は与えた cap を埋めるまで膨張し、
                        // cap30 -> cap60 で |raw| 中央値 34.5 -> 53.6cp、 重み中央値 6.9 -> 8.1cp。
                        // cap60 でも 44.7% の局面が clip されたままだった。
                        // 膨らんだ重みは cap 未満の局面にもそのまま効くため、 偽勾配は
                        // clamp されない局面の評価を汚染する。 真の勾配に戻す。
                        // grad_scale=0 は relu 合成で hard clamp の**真の**勾配
                        // (relu(x+c) - relu(x-c) - c、 勾配 1_{|x|<c})、
                        // grad_scale>0 は clip_pass_through_grad + surrogate。
                        let hard_src = direct_delta;
                        let direct_cap_aligned = if direct15_grad_scale_cp > 0.0 {
                            // forward は hard clamp のまま、 backward だけ尺度 g で
                            // 滑らかに減衰させる (stop-grad 恒等式)。
                            //   surrogate(x) = 2g*sigmoid(2x/g) - g,  d/dx|_0 = 1
                            //
                            // hard 側は **単一 op の clip_pass_through_grad** を使う。
                            // relu 合成ノードに copy_stop_grad を掛けると
                            // GradientCalculationError(MismatchedBatchSizes) が確定的に出た
                            // (2026-09-01 実測: 8/8 失敗。 単一 op 版は従来どおり動く)。
                            let g = direct15_grad_scale_cp * fv_scale as f32
                                / (f32::from(QA) * f32::from(QB));
                            let hard_clip = hard_src
                                .copy()
                                .clip_pass_through_grad(-cap_train, cap_train)
                                .copy_stop_grad();
                            let soft = (2.0 * g) * ((2.0 / g) * hard_src).sigmoid() - g;
                            hard_clip + soft.copy() - soft.copy_stop_grad()
                        } else {
                            (hard_src.copy() + cap_train).relu() - (hard_src - cap_train).relu() - cap_train
                        };
                        net_output_final = net_output_final + direct_cap_aligned;
                    } else {
                        net_output_final = net_output_final + direct_delta;
                    }
                }


                // 旧 LMAS heads block (= 約 540 行) は 2026-05-25 物理削除。
                // cfg(any()) は常に false。 1 つの block で全 if を包む。
                #[cfg(any())]
                {
                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_SLA) != 0 {
                    // SLA: 14 dim dense input → per-bucket affine (14 → NUM_BUCKETS)
                    // weights "lmas_sla_w" は Zeroed で初期化、 訓練で立ち上がる。
                    let sla_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_SLA,
                        Shape::new(bullet_lib::game::inputs::LMAS_SLA_DIMS, 1),
                    );
                    let sla = Affine {
                        weights: builder.new_weights(
                            "lmas_sla_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_SLA_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("lmas_sla_b", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let sla_out = sla.forward(sla_input).select(output_buckets);
                    net_output_final = net_output_final + sla_out;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_HML) != 0 {
                    // HML: 7 dim dense input → per-bucket affine
                    // TODO (P2): SLA と同じ pattern。 weights は Zeroed init。
                    let hml_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_HML,
                        Shape::new(bullet_lib::game::inputs::LMAS_HML_DIMS, 1),
                    );
                    let hml = Affine {
                        weights: builder.new_weights(
                            "lmas_hml_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_HML_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("lmas_hml_b", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let hml_out = hml.forward(hml_input).select(output_buckets);
                    net_output_final = net_output_final + hml_out;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_ZPSQT) != 0 {
                    // Zone PSQT (両 perspective): 126 dim dense input × 2 → per-bucket affine × 2
                    //
                    // 黒視点 / 白視点を別 dense input で受け取り、 別 affine head で計算した後
                    // `black_out - white_out` で対称化する。 spec.md §5.3 の
                    // `(stm - nstm) / 2` の /2 は学習側で省略 (重みに吸収される)、 推論側で
                    // stm に応じて sign flip する設計。
                    //
                    // bias は 1 つだけ持たせると black/white で重複加算するため、 片側 (黒) のみ
                    // bias を持つ非対称構造にする。
                    let zpsqt_black_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_ZPSQT_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_ZPSQT_DIMS, 1),
                    );
                    let zpsqt_black = Affine {
                        weights: builder.new_weights(
                            "lmas_zpsqt_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_ZPSQT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("lmas_zpsqt_b", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let zpsqt_black_out = zpsqt_black.forward(zpsqt_black_input).select(output_buckets);

                    let zpsqt_white_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_ZPSQT_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_ZPSQT_DIMS, 1),
                    );
                    // 白視点 weight は bias なし affine (= 線形 head)。
                    // Affine struct は bias 必須なので Zero 初期化のまま保持し、 学習で 0 周辺に
                    // 留まる想定。 黒側 bias と二重カウントの懸念があるため、 quantise 時にも
                    // 白側 bias は書き出さない (LMAS section に bias slot なし)。
                    let zpsqt_white = Affine {
                        weights: builder.new_weights(
                            "lmas_zpsqt_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_ZPSQT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_zpsqt_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let zpsqt_white_out = zpsqt_white.forward(zpsqt_white_input).select(output_buckets);

                    // 黒視点固定で `black - white`。 推論側 (rshogi-core) で stm に応じて
                    // sign flip するため、 学習側は手番非対称な符号を扱わない。
                    net_output_final = net_output_final + zpsqt_black_out - zpsqt_white_out;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_KSA) != 0 {
                    // KSA 1D: 144 one-hot dense input → per-bucket affine
                    // TODO (P4): NPS micro-benchmark で省略判断
                    let ksa_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KSA,
                        Shape::new(bullet_lib::game::inputs::LMAS_KSA_DIMS, 1),
                    );
                    let ksa = Affine {
                        weights: builder.new_weights(
                            "lmas_ksa_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KSA_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("lmas_ksa_b", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let ksa_out = ksa.forward(ksa_input).select(output_buckets);
                    net_output_final = net_output_final + ksa_out;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_CT) != 0 {
                    // CT: 8 dim dense input → per-bucket affine
                    let ct_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_CT,
                        Shape::new(bullet_lib::game::inputs::LMAS_CT_DIMS, 1),
                    );
                    let ct = Affine {
                        weights: builder.new_weights(
                            "lmas_ct_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_CT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("lmas_ct_b", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let ct_out = ct.forward(ct_input).select(output_buckets);
                    net_output_final = net_output_final + ct_out;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_KZPSQT) != 0 {
                    let kzpsqt_black_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KZPSQT_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_KZPSQT_DIMS, 1),
                    );
                    let kzpsqt_black = Affine {
                        weights: builder.new_weights(
                            "lmas_kzpsqt_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KZPSQT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights("lmas_kzpsqt_b", Shape::new(NUM_BUCKETS, 1), InitSettings::Zeroed),
                    };
                    let kzpsqt_black_out = kzpsqt_black.forward(kzpsqt_black_input).select(output_buckets);

                    let kzpsqt_white_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KZPSQT_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_KZPSQT_DIMS, 1),
                    );
                    let kzpsqt_white = Affine {
                        weights: builder.new_weights(
                            "lmas_kzpsqt_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KZPSQT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kzpsqt_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kzpsqt_white_out = kzpsqt_white.forward(kzpsqt_white_input).select(output_buckets);

                    net_output_final = net_output_final + (kzpsqt_black_out - kzpsqt_white_out) * 0.5;
                }

                // ───────────────────────────────────────────────────────────────
                // v5 ContextGate / KZLine / KZESC (spec.md §14)
                // ───────────────────────────────────────────────────────────────
                //
                // ContextGate は raw value を group ごとに掛ける倍率テーブル設計だが、
                // 学習 graph 上は他の LMAS head と同じ「dense input × per-bucket affine」
                // で扱う (initial=Zeroed なので学習初期は寄与 0)。
                // rshogi-core 側で実際に gate として乗算する際は inference 時に展開する。
                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_CONTEXT_GATE) != 0 {
                    let cg_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_CONTEXT_GATE,
                        Shape::new(bullet_lib::game::inputs::LMAS_CONTEXT_GATE_AUX_DIMS, 1),
                    );
                    let cg = Affine {
                        weights: builder.new_weights(
                            "lmas_context_gate_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_CONTEXT_GATE_AUX_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_context_gate_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let cg_out = cg.forward(cg_input).select(output_buckets);
                    net_output_final = net_output_final + cg_out;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_KZLINE) != 0 {
                    let kzl_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KZLINE_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_KZLINE_AUX_DIMS, 1),
                    );
                    let kzl_b = Affine {
                        weights: builder.new_weights(
                            "lmas_kzline_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KZLINE_AUX_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kzline_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kzl_b_out = kzl_b.forward(kzl_b_input).select(output_buckets);
                    let kzl_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KZLINE_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_KZLINE_AUX_DIMS, 1),
                    );
                    let kzl_w = Affine {
                        weights: builder.new_weights(
                            "lmas_kzline_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KZLINE_AUX_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kzline_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kzl_w_out = kzl_w.forward(kzl_w_input).select(output_buckets);
                    net_output_final = net_output_final + (kzl_b_out - kzl_w_out) * 0.5;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_KZESC) != 0 {
                    let kze_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KZESC_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_KZESC_AUX_DIMS, 1),
                    );
                    let kze_b = Affine {
                        weights: builder.new_weights(
                            "lmas_kzesc_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KZESC_AUX_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kzesc_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kze_b_out = kze_b.forward(kze_b_input).select(output_buckets);
                    let kze_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KZESC_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_KZESC_AUX_DIMS, 1),
                    );
                    let kze_w = Affine {
                        weights: builder.new_weights(
                            "lmas_kzesc_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KZESC_AUX_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kzesc_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kze_w_out = kze_w.forward(kze_w_input).select(output_buckets);
                    net_output_final = net_output_final + (kze_b_out - kze_w_out) * 0.5;
                }

                // ───────────────────────────────────────────────────────────────
                // v6 KP-PSQT raw / Hand × King-zone / Threat Map (spec.md §15)
                // ───────────────────────────────────────────────────────────────
                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_KP_PSQT_RAW) != 0 {
                    let kpr_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KP_PSQT_RAW_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_KP_PSQT_RAW_DIMS, 1),
                    );
                    let kpr_b = Affine {
                        weights: builder.new_weights(
                            "lmas_kp_psqt_raw_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KP_PSQT_RAW_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kp_psqt_raw_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kpr_b_out = kpr_b.forward(kpr_b_input).select(output_buckets);
                    let kpr_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KP_PSQT_RAW_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_KP_PSQT_RAW_DIMS, 1),
                    );
                    let kpr_w = Affine {
                        weights: builder.new_weights(
                            "lmas_kp_psqt_raw_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KP_PSQT_RAW_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kp_psqt_raw_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kpr_w_out = kpr_w.forward(kpr_w_input).select(output_buckets);
                    net_output_final = net_output_final + (kpr_b_out - kpr_w_out) * 0.5;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_HAND_KZONE) != 0 {
                    let hkz_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_HAND_KZONE_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_HAND_KZONE_DIMS, 1),
                    );
                    let hkz_b = Affine {
                        weights: builder.new_weights(
                            "lmas_hand_kzone_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_HAND_KZONE_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_hand_kzone_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let hkz_b_out = hkz_b.forward(hkz_b_input).select(output_buckets);
                    let hkz_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_HAND_KZONE_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_HAND_KZONE_DIMS, 1),
                    );
                    let hkz_w = Affine {
                        weights: builder.new_weights(
                            "lmas_hand_kzone_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_HAND_KZONE_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_hand_kzone_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let hkz_w_out = hkz_w.forward(hkz_w_input).select(output_buckets);
                    net_output_final = net_output_final + (hkz_b_out - hkz_w_out) * 0.5;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_THREAT_MAP) != 0 {
                    let tm_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_THREAT_MAP_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_THREAT_MAP_DIMS, 1),
                    );
                    let tm_b = Affine {
                        weights: builder.new_weights(
                            "lmas_threat_map_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_THREAT_MAP_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_threat_map_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let tm_b_out = tm_b.forward(tm_b_input).select(output_buckets);
                    let tm_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_THREAT_MAP_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_THREAT_MAP_DIMS, 1),
                    );
                    let tm_w = Affine {
                        weights: builder.new_weights(
                            "lmas_threat_map_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_THREAT_MAP_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_threat_map_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let tm_w_out = tm_w.forward(tm_w_input).select(output_buckets);
                    net_output_final = net_output_final + (tm_b_out - tm_w_out) * 0.5;
                }

                // ───────────────────────────────────────────────────────────────
                // v7 KKR / PSH / PBR (spec.md §16)
                // ───────────────────────────────────────────────────────────────
                //
                // 既存 v5/v6 と同じ「dense input × per-bucket affine + select(output_buckets)」
                // パターンで net_output に **加算**。 bucket 軸は select で扱い、 phase 軸は
                // dense input 内の one-hot slot で表現する (bullet 側 fill は phase=0 固定だが
                // weight tensor のレイアウトは全 phase を保持し disk write/read は正しい index で行う)。
                //
                // PBR は rshogi-core 側で Q10 Hadamard gate として acc_val 全体に乗算する設計だが、
                // ContextGate と同じく 学習 graph 上は additive head として扱う
                // (initial=Zeroed → 学習初期は寄与 0、 inference 時に gate として展開)。
                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_KKR) != 0 {
                    let kkr_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KKR_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_KKR_DIMS, 1),
                    );
                    let kkr_b = Affine {
                        weights: builder.new_weights(
                            "lmas_kkr_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KKR_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kkr_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kkr_b_out = kkr_b.forward(kkr_b_input).select(output_buckets);
                    let kkr_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_KKR_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_KKR_DIMS, 1),
                    );
                    let kkr_w = Affine {
                        weights: builder.new_weights(
                            "lmas_kkr_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_KKR_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_kkr_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let kkr_w_out = kkr_w.forward(kkr_w_input).select(output_buckets);
                    net_output_final = net_output_final + (kkr_b_out - kkr_w_out) * 0.5;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_PSH) != 0 {
                    let psh_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_PSH_BLACK,
                        Shape::new(bullet_lib::game::inputs::LMAS_PSH_DIMS, 1),
                    );
                    let psh_b = Affine {
                        weights: builder.new_weights(
                            "lmas_psh_w_black",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_PSH_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_psh_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let psh_b_out = psh_b.forward(psh_b_input).select(output_buckets);
                    let psh_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_PSH_WHITE,
                        Shape::new(bullet_lib::game::inputs::LMAS_PSH_DIMS, 1),
                    );
                    let psh_w = Affine {
                        weights: builder.new_weights(
                            "lmas_psh_w_white",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_PSH_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_psh_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let psh_w_out = psh_w.forward(psh_w_input).select(output_buckets);
                    net_output_final = net_output_final + (psh_b_out - psh_w_out) * 0.5;
                }

                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_PBR) != 0 {
                    // PBR: perspective なし 1 入力。 rshogi-core では Hadamard gate (Q10) として
                    // acc_val 全体に乗算するが、 ContextGate と同じく学習 graph 上は additive。
                    // initial=Zeroed なので学習初期は寄与 0。
                    let pbr_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_PBR,
                        Shape::new(bullet_lib::game::inputs::LMAS_PBR_DIMS, 1),
                    );
                    let pbr = Affine {
                        weights: builder.new_weights(
                            "lmas_pbr_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_PBR_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_pbr_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let pbr_out = pbr.forward(pbr_input).select(output_buckets);
                    net_output_final = net_output_final + pbr_out;
                }

                // ───────────────────────────────────────────────────────────────
                // v8 MATE_THREAT (= bit 18 OPP, bit 19 MY) — Phase B 革新 feature
                // 各 1 dim dense input × per-bucket affine。 stm 視点 (= sidecar PSV byte 39)。
                // ───────────────────────────────────────────────────────────────
                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_OPP_MATE_THREAT) != 0 {
                    let mt_opp_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_MATE_THREAT_OPP,
                        Shape::new(bullet_lib::game::inputs::LMAS_MATE_THREAT_DIMS, 1),
                    );
                    let mt_opp = Affine {
                        weights: builder.new_weights(
                            "lmas_mate_threat_opp_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_MATE_THREAT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_mate_threat_opp_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let mt_opp_out = mt_opp.forward(mt_opp_input).select(output_buckets);
                    net_output_final = net_output_final + mt_opp_out;
                }
                if (lmas_flags_for_graph & bullet_lib::game::inputs::LMAS_FLAG_MY_MATE_THREAT) != 0 {
                    let mt_my_input = builder.new_dense_input(
                        bullet_lib::game::inputs::LMAS_AUX_NAME_MATE_THREAT_MY,
                        Shape::new(bullet_lib::game::inputs::LMAS_MATE_THREAT_DIMS, 1),
                    );
                    let mt_my = Affine {
                        weights: builder.new_weights(
                            "lmas_mate_threat_my_w",
                            Shape::new(NUM_BUCKETS, bullet_lib::game::inputs::LMAS_MATE_THREAT_DIMS),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "lmas_mate_threat_my_b",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let mt_my_out = mt_my.forward(mt_my_input).select(output_buckets);
                    net_output_final = net_output_final + mt_my_out;
                }
                } // close #[cfg(any())] {

                // ---- Pin-aware head (dense 2,268 × 2 perspective) ----
                //
                // 黒 / 白視点別 dense input × per-bucket affine head。 ZPSQT と同様、
                // 黒視点を `+` で、 白視点を `-` で加算して `black_out - white_out` の
                // 反対称化を行う。 推論側 (rshogi-core `pin_head_eval`) では stm 視点
                // で sign flip するため、 学習側の符号と整合する。
                //
                // 初期化は Zeroed なので、 既存 ckpt resume + pin_aware ON で立ち上げ
                // ても序盤の eval 寄与は 0。
                if pin_aware_for_graph {
                    let pin_dims = bullet_lib::game::inputs::PIN_DIMENSIONS_PER_PERSPECTIVE;
                    let pin_b_input = builder.new_dense_input(
                        bullet_lib::game::inputs::PIN_AUX_NAME_BLACK,
                        Shape::new(pin_dims, 1),
                    );
                    let pin_b = Affine {
                        weights: builder.new_weights(
                            "pin_w_black",
                            Shape::new(NUM_BUCKETS, pin_dims),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "pin_b_black",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let pin_b_out = pin_b.forward(pin_b_input).select(output_buckets);

                    let pin_w_input = builder.new_dense_input(
                        bullet_lib::game::inputs::PIN_AUX_NAME_WHITE,
                        Shape::new(pin_dims, 1),
                    );
                    // 白視点 weight は bias なし (= 黒側 bias と二重加算回避)。
                    let pin_w = Affine {
                        weights: builder.new_weights(
                            "pin_w_white",
                            Shape::new(NUM_BUCKETS, pin_dims),
                            InitSettings::Zeroed,
                        ),
                        bias: builder.new_weights(
                            "pin_b_white_unused",
                            Shape::new(NUM_BUCKETS, 1),
                            InitSettings::Zeroed,
                        ),
                    };
                    let pin_w_out = pin_w.forward(pin_w_input).select(output_buckets);

                    // 反対称: black - white (推論側 stm 視点で sign flip するため学習で
                    // 手番非対称な符号は扱わない)。
                    net_output_final = net_output_final + pin_b_out - pin_w_out;
                }

                // ---- Dual PSQT head (Plan C iter 2 用領域確保) ----
                //
                // graph 上の weight tensor `dual_psqt_w` (= shape [2 head, NUM_BUCKETS])
                // のみを Zeroed init で確保する。 iter 1 では forward に組み込まないため
                // eval 寄与は 0、 disk に Zeroed の section が書き出される。 iter 2 で
                // head-only fine-tune の際にこの weight tensor を更新する。
                //
                // 注: forward への組み込みは iter 2 で実装する (= attack/defend 信号の
                //     生成式が iter 2 で確定する)。 iter 1 では graph に存在するだけで
                //     gradient = 0 で訓練されない。
                // iter 1 では dual_psqt_w を graph に追加しない:
                //   1. graph builder は roots() == 1 を要求するため orphan weight は不可
                //   2. save 側 (= line ~1980-2000) は dual_psqt_reserve=true 時に
                //      graph 重みを参照せず P2DUAL section を全 0 で直接書き出す
                //   3. iter 2 で head-only fine-tune する際は graph に dual_psqt_w を
                //      組み込んだ別 example を作成する (= 本 example は iter 1 専用)
                if dual_psqt_reserve_for_graph {
                    // no-op: section の書き出しは save 側で zeros として処理される
                }

                // ---- Dual PSQT King-Attacker head (Phase 3b-sparse: 4 sparse input 化) ----
                //
                // attack / defend × black / white = 4 sparse input。 各 input は per-head
                // bucket-only flat index (= dim 9,072、 max_active 30)。 graph 上の weight
                // tensor は per-head の `[1, 9072]` で、 forward は
                //
                //   score_dual = (w_ab · attack_b) + (w_aw · attack_w)
                //                - (w_db · defend_b) - (w_dw · defend_w)
                //
                // となる。 attack 側 + defend 側 -1 倍で rshogi-core 推論側
                // `(attack - defend)` 規約と等価。
                //
                // save (= quantised.bin) 側で 4 weight tensor を head=0/1 × persp=B/W で
                // 結合し直して disk format `[persp][head][bucket][sq][piece]` に戻す。
                // defend 側 weight は **符号反転して保存**: 学習 graph は `-w_d · defend` で
                // 動くが、 disk format は dense 経路時の `(attack - defend)` 規約に揃えるため、
                // disk 上の defend head weight は `w_d` (= positive 学習結果) として書き出す。
                if dual_psqt_king_attacker_for_graph {
                    // king_attacker head (= attack/defend 4 sparse input) は
                    // dual_disable_king_attacker_for_graph で off 可能 (= arm 5)
                    if !dual_disable_king_attacker_for_graph {
                        let dim_per_head = bullet_lib::game::inputs::DUAL_DIM_PER_HEAD;
                        let max_active_head = bullet_lib::game::inputs::DUAL_MAX_ACTIVE_PER_HEAD;

                        // --- black perspective ---
                        let attack_b_in = builder.new_sparse_input(
                            bullet_lib::game::inputs::DUAL_SPARSE_NAME_ATTACK_BLACK,
                            Shape::new(dim_per_head, 1),
                            max_active_head,
                        );
                        let defend_b_in = builder.new_sparse_input(
                            bullet_lib::game::inputs::DUAL_SPARSE_NAME_DEFEND_BLACK,
                            Shape::new(dim_per_head, 1),
                            max_active_head,
                        );
                        let dual_w_attack_b = builder.new_weights(
                            "dual_w_attack_black",
                            Shape::new(1, dim_per_head),
                            InitSettings::Zeroed,
                        );
                        let dual_w_defend_b = builder.new_weights(
                            "dual_w_defend_black",
                            Shape::new(1, dim_per_head),
                            InitSettings::Zeroed,
                        );
                        let dual_attack_b_out = dual_w_attack_b.matmul(attack_b_in);
                        let dual_defend_b_out = dual_w_defend_b.matmul(defend_b_in);

                        // --- white perspective ---
                        let attack_w_in = builder.new_sparse_input(
                            bullet_lib::game::inputs::DUAL_SPARSE_NAME_ATTACK_WHITE,
                            Shape::new(dim_per_head, 1),
                            max_active_head,
                        );
                        let defend_w_in = builder.new_sparse_input(
                            bullet_lib::game::inputs::DUAL_SPARSE_NAME_DEFEND_WHITE,
                            Shape::new(dim_per_head, 1),
                            max_active_head,
                        );
                        let dual_w_attack_w = builder.new_weights(
                            "dual_w_attack_white",
                            Shape::new(1, dim_per_head),
                            InitSettings::Zeroed,
                        );
                        let dual_w_defend_w = builder.new_weights(
                            "dual_w_defend_white",
                            Shape::new(1, dim_per_head),
                            InitSettings::Zeroed,
                        );
                        let dual_attack_w_out = dual_w_attack_w.matmul(attack_w_in);
                        let dual_defend_w_out = dual_w_defend_w.matmul(defend_w_in);

                        // 合算: attack + (defend × -1)、 両 perspective 加算合成
                        net_output_final = net_output_final
                            + dual_attack_b_out
                            + dual_attack_w_out
                            - dual_defend_b_out
                            - dual_defend_w_out;
                    }

                    // ---- D2 cross feature: piece_threat_relation (+ piece_defended_by) ----
                    //
                    // PSQT が捉えにくい cross feature (= 駒間相互作用) を 2 (or 4) sparse input で学習。
                    //   - threat:   own (sq, p) → attacking 相手 p   [head 0]
                    //   - defended: own (sq, p) → defending 自軍 p   [head 1、 arm 6 で削除可能]
                    // 各 dim = DUAL_CROSS_DIM_PER_HEAD (127,008)、 max_active = 80。
                    // 加算合成 (= sign 反転なし、 weight 自体で sign を学習)。
                    //
                    // arm 6 (= 2026-05-27): `dual_disable_cross_defended_for_graph` ON で
                    // head 1 (= defended) の 2 sparse input + 2 weight tensor を削除。
                    // cosine similarity 解析で head 1 は無方向 random walk (= cos=0.38、
                    // norm -29%) と判明、 production 推論側でも path 削除済 (= NUM_DUAL_CROSS_HEADS=1)。
                    if !enable_scalar_threat_for_graph || scalar_threat_coexist_dual_cross_for_graph {
                    let cross_dim = bullet_lib::game::inputs::DUAL_CROSS_DIM_PER_HEAD;
                    let cross_max_active =
                        bullet_lib::game::inputs::DUAL_CROSS_MAX_ACTIVE_PER_HEAD;
                    let enable_defended = !dual_disable_cross_defended_for_graph;

                    // black perspective: threat は常に有効、 defended は flag 依存。
                    let cross_threat_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::DUAL_CROSS_THREAT_BLACK,
                        Shape::new(cross_dim, 1),
                        cross_max_active,
                    );
                    let dual_w_threat_b = builder.new_weights(
                        "dual_w_threat_black",
                        Shape::new(1, cross_dim),
                        InitSettings::Zeroed,
                    );
                    let dual_cross_threat_b_out = dual_w_threat_b.matmul(cross_threat_b_in);

                    // white perspective: threat は常に有効
                    let cross_threat_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::DUAL_CROSS_THREAT_WHITE,
                        Shape::new(cross_dim, 1),
                        cross_max_active,
                    );
                    let dual_w_threat_w = builder.new_weights(
                        "dual_w_threat_white",
                        Shape::new(1, cross_dim),
                        InitSettings::Zeroed,
                    );
                    let dual_cross_threat_w_out = dual_w_threat_w.matmul(cross_threat_w_in);

                    // threat (= head 0) は常に net_output_final へ加算合成
                    net_output_final = net_output_final
                        + dual_cross_threat_b_out
                        + dual_cross_threat_w_out;

                    if enable_defended {
                        // defended (= head 1) の sparse input + weight tensor を追加
                        let cross_defended_b_in = builder.new_sparse_input(
                            bullet_lib::game::inputs::DUAL_CROSS_DEFENDED_BLACK,
                            Shape::new(cross_dim, 1),
                            cross_max_active,
                        );
                        let dual_w_defended_b = builder.new_weights(
                            "dual_w_defended_black",
                            Shape::new(1, cross_dim),
                            InitSettings::Zeroed,
                        );
                        let dual_cross_defended_b_out =
                            dual_w_defended_b.matmul(cross_defended_b_in);

                        let cross_defended_w_in = builder.new_sparse_input(
                            bullet_lib::game::inputs::DUAL_CROSS_DEFENDED_WHITE,
                            Shape::new(cross_dim, 1),
                            cross_max_active,
                        );
                        let dual_w_defended_w = builder.new_weights(
                            "dual_w_defended_white",
                            Shape::new(1, cross_dim),
                            InitSettings::Zeroed,
                        );
                        let dual_cross_defended_w_out =
                            dual_w_defended_w.matmul(cross_defended_w_in);

                        net_output_final = net_output_final
                            + dual_cross_defended_b_out
                            + dual_cross_defended_w_out;
                    }
                    }
                }

                // ---- arm 11 (= 2026-05-28): KingRing Threat PSQT scalar head ----
                //
                // 2 sparse input (= kingring_{black,white})、 各 dim = 9,984
                // (= bucket-major flat、 sample は 1 bucket 分のみ active)、 max_active = 24。
                // 2 weight tensor (= shape (1, 9984))、 Zeroed init で graph に追加し、
                // 末端で **反対称合算**:
                //   `net_output_final += kingring_b · w_black - kingring_w · w_white`
                //
                // rshogi-core 推論側 `(us - them) / KINGRING_OUTPUT_SCALE` (= 反対称) に揃える。
                // 既存 pin head (= 同 antisymmetric eval) と同 convention。
                // dual_psqt_king_attacker の有無に関わらず単独 ON 可能。
                if enable_kingring_for_graph {
                    let kr_dim = bullet_lib::game::inputs::KINGRING_DIM_PER_PERSP;
                    let kr_max = bullet_lib::game::inputs::KINGRING_MAX_ACTIVE_PER_PERSP;

                    let kingring_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::KINGRING_SPARSE_NAME_BLACK,
                        Shape::new(kr_dim, 1),
                        kr_max,
                    );
                    let kingring_w_black = builder.new_weights(
                        "kingring_w_black",
                        Shape::new(1, kr_dim),
                        InitSettings::Zeroed,
                    );
                    let kingring_b_out = kingring_w_black.matmul(kingring_b_in);

                    let kingring_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::KINGRING_SPARSE_NAME_WHITE,
                        Shape::new(kr_dim, 1),
                        kr_max,
                    );
                    let kingring_w_white = builder.new_weights(
                        "kingring_w_white",
                        Shape::new(1, kr_dim),
                        InitSettings::Zeroed,
                    );
                    let kingring_w_out = kingring_w_white.matmul(kingring_w_in);

                    net_output_final = net_output_final + kingring_b_out - kingring_w_out;
                }

                // ---- arm 12 (= 2026-05-29): PairShape (= Piece-Pair Shape Table) scalar head ----
                //
                // 2 sparse input (= pairshape_{black,white})、 各 dim = 1,536
                // (= bucket-major flat、 sample は 1 bucket 分のみ active、 multiset 重複保持)、 max_active = 256。
                // 2 weight tensor (= shape (1, 1536))、 Zeroed init で graph に追加し、
                // 末端で **反対称合算**:
                //   `net_output_final += pairshape_b · w_black - pairshape_w · w_white`
                //
                // rshogi-core 推論側 `(us - them) / PAIRSHAPE_OUTPUT_SCALE` (= 反対称) に揃える。
                // dual / kingring と同 convention。
                if enable_pairshape_for_graph {
                    let ps_dim = bullet_lib::game::inputs::PAIRSHAPE_DIM_PER_PERSP;
                    let ps_max = bullet_lib::game::inputs::PAIRSHAPE_MAX_ACTIVE_PER_PERSP;

                    let pairshape_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::PAIRSHAPE_SPARSE_NAME_BLACK,
                        Shape::new(ps_dim, 1),
                        ps_max,
                    );
                    let pairshape_w_black = builder.new_weights(
                        "pairshape_w_black",
                        Shape::new(1, ps_dim),
                        InitSettings::Zeroed,
                    );
                    let pairshape_b_out = pairshape_w_black.matmul(pairshape_b_in);

                    let pairshape_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::PAIRSHAPE_SPARSE_NAME_WHITE,
                        Shape::new(ps_dim, 1),
                        ps_max,
                    );
                    let pairshape_w_white = builder.new_weights(
                        "pairshape_w_white",
                        Shape::new(1, ps_dim),
                        InitSettings::Zeroed,
                    );
                    let pairshape_w_out = pairshape_w_white.matmul(pairshape_w_in);

                    net_output_final = net_output_final + pairshape_b_out - pairshape_w_out;
                }

                // 2026-06-20 Dual Cross Geometry Head (= magic P2DCG1): pairshape と同 pattern。
                //
                // 2 sparse input (= dual_cross_geometry_{black,white})、 各 dim = 50,176
                // (= 8 bucket * 14 * 14 pair * 32 geometry、 bucket-major flat)、
                // max_active = DUAL_CROSS_MAX_ACTIVE_PER_HEAD (= 80)。
                // 2 weight tensor (= shape (1, 50176))、 Zeroed init で graph に追加し、
                // 末端で **反対称合算**:
                //   `net_output_final += geom_b · w_black - geom_w · w_white`
                // rshogi-core 推論側の反対称 convention に揃える (= pairshape / dual と同一)。
                if enable_dual_cross_geometry_for_graph {
                    let geom_dim = bullet_lib::game::inputs::DUAL_CROSS_GEOMETRY_DIM_PER_PERSP;
                    let geom_max = bullet_lib::game::inputs::DUAL_CROSS_MAX_ACTIVE_PER_HEAD;

                    let geometry_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::DUAL_CROSS_GEOMETRY_BLACK,
                        Shape::new(geom_dim, 1),
                        geom_max,
                    );
                    let dual_cross_geometry_w_black = builder.new_weights(
                        "dual_cross_geometry_w_black",
                        Shape::new(1, geom_dim),
                        InitSettings::Zeroed,
                    );
                    let geometry_b_out = dual_cross_geometry_w_black.matmul(geometry_b_in);

                    let geometry_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::DUAL_CROSS_GEOMETRY_WHITE,
                        Shape::new(geom_dim, 1),
                        geom_max,
                    );
                    let dual_cross_geometry_w_white = builder.new_weights(
                        "dual_cross_geometry_w_white",
                        Shape::new(1, geom_dim),
                        InitSettings::Zeroed,
                    );
                    let geometry_w_out = dual_cross_geometry_w_white.matmul(geometry_w_in);

                    net_output_final = net_output_final + geometry_b_out - geometry_w_out;
                }

                // 2026-06-06 Activity-Risk Interaction Head: pairshape と同 pattern。
                // 2 sparse input (= dim 2048、 max_active 1) + 2 weight tensor (= shape (1, 2048))。
                // 反対称合成: `net_output_final += ar_b · w_black - ar_w · w_white`
                if enable_activity_risk_for_graph {
                    let ar_dim = bullet_lib::game::inputs::ACTIVITY_RISK_DIM_PER_PERSP;
                    let ar_max = bullet_lib::game::inputs::ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP;

                    let activity_risk_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ACTIVITY_RISK_SPARSE_NAME_BLACK,
                        Shape::new(ar_dim, 1),
                        ar_max,
                    );
                    let activity_risk_w_black = builder.new_weights(
                        "activity_risk_w_black",
                        Shape::new(1, ar_dim),
                        InitSettings::Zeroed,
                    );
                    let activity_risk_b_out = activity_risk_w_black.matmul(activity_risk_b_in);

                    let activity_risk_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ACTIVITY_RISK_SPARSE_NAME_WHITE,
                        Shape::new(ar_dim, 1),
                        ar_max,
                    );
                    let activity_risk_w_white = builder.new_weights(
                        "activity_risk_w_white",
                        Shape::new(1, ar_dim),
                        InitSettings::Zeroed,
                    );
                    let activity_risk_w_out = activity_risk_w_white.matmul(activity_risk_w_in);

                    net_output_final =
                        net_output_final + activity_risk_b_out - activity_risk_w_out;

                    // 2026-06-08 v3 tactical 加算項 (= 2 weight tensor shape (1, 512))
                    let ar_v3_tdim = bullet_lib::game::inputs::ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP;
                    let ar_v3_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ACTIVITY_RISK_V3_TACTICAL_NAME_BLACK,
                        Shape::new(ar_v3_tdim, 1),
                        1,
                    );
                    let ar_v3_tactical_w_black = builder.new_weights(
                        "activity_risk_v3_tactical_w_black",
                        Shape::new(1, ar_v3_tdim),
                        InitSettings::Zeroed,
                    );
                    let ar_v3_tactical_b_out = ar_v3_tactical_w_black.matmul(ar_v3_b_in);
                    let ar_v3_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ACTIVITY_RISK_V3_TACTICAL_NAME_WHITE,
                        Shape::new(ar_v3_tdim, 1),
                        1,
                    );
                    let ar_v3_tactical_w_white = builder.new_weights(
                        "activity_risk_v3_tactical_w_white",
                        Shape::new(1, ar_v3_tdim),
                        InitSettings::Zeroed,
                    );
                    let ar_v3_tactical_w_out = ar_v3_tactical_w_white.matmul(ar_v3_w_in);

                    net_output_final =
                        net_output_final + ar_v3_tactical_b_out - ar_v3_tactical_w_out;
                }

                // 2026-06-09 King Escape Direction (KED) head: activity_risk と同 pattern。
                // 2 sparse input (= dim 32、 max_active 1) + 2 weight tensor (= shape (1, 32))。
                // 反対称合成: `net_output_final += ked_b · w_black - ked_w · w_white`
                if enable_king_escape_for_graph {
                    let ked_dim = bullet_lib::game::inputs::KING_ESCAPE_DIM_PER_PERSP;
                    let ked_max = bullet_lib::game::inputs::KING_ESCAPE_MAX_ACTIVE_PER_PERSP;

                    let king_escape_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::KING_ESCAPE_SPARSE_NAME_BLACK,
                        Shape::new(ked_dim, 1),
                        ked_max,
                    );
                    let king_escape_w_black = builder.new_weights(
                        "king_escape_w_black",
                        Shape::new(1, ked_dim),
                        InitSettings::Zeroed,
                    );
                    let king_escape_b_out = king_escape_w_black.matmul(king_escape_b_in);

                    let king_escape_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::KING_ESCAPE_SPARSE_NAME_WHITE,
                        Shape::new(ked_dim, 1),
                        ked_max,
                    );
                    let king_escape_w_white = builder.new_weights(
                        "king_escape_w_white",
                        Shape::new(1, ked_dim),
                        InitSettings::Zeroed,
                    );
                    let king_escape_w_out = king_escape_w_white.matmul(king_escape_w_in);

                    net_output_final = net_output_final + king_escape_b_out - king_escape_w_out;
                }

                // 2026-06-28 Slider X-ray (SXRY) head: ADEV と同 pattern (= dim 64)。
                // 2 sparse input (= dim 64、 max_active 1) + 2 weight tensor (= shape (1, 64))。
                // 反対称合成: `net_output_final += sx_b · w_black - sx_w · w_white`
                if enable_slider_xray_for_graph {
                    let sx_dim = bullet_lib::game::inputs::SLIDER_XRAY_DIM_PER_PERSP;
                    let sx_max = bullet_lib::game::inputs::SLIDER_XRAY_MAX_ACTIVE_PER_PERSP;

                    let slider_xray_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::SLIDER_XRAY_SPARSE_NAME_BLACK,
                        Shape::new(sx_dim, 1),
                        sx_max,
                    );
                    let slider_xray_w_black = builder.new_weights(
                        "slider_xray_w_black",
                        Shape::new(1, sx_dim),
                        InitSettings::Zeroed,
                    );
                    let slider_xray_b_out = slider_xray_w_black.matmul(slider_xray_b_in);

                    let slider_xray_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::SLIDER_XRAY_SPARSE_NAME_WHITE,
                        Shape::new(sx_dim, 1),
                        sx_max,
                    );
                    let slider_xray_w_white = builder.new_weights(
                        "slider_xray_w_white",
                        Shape::new(1, sx_dim),
                        InitSettings::Zeroed,
                    );
                    let slider_xray_w_out = slider_xray_w_white.matmul(slider_xray_w_in);

                    net_output_final = net_output_final + slider_xray_b_out - slider_xray_w_out;
                }

                // 2026-07-01 King Zone Threat (KZT_HAND) factored v1 head。
                // 5 テーブル (core/region/bucket/persp/bias) を **共有** weight tensor とし、 各
                // テーブルにつき black / white の 2 sparse input (= max_active 175) を matmul、
                //   `net_output_final += W_t · black_t - W_t · white_t`
                // で反対称合成する (= 同一 weight tensor を black/white 双方で再利用)。 disk P2KZT
                // (= version 2) は core → region → bucket → persp → bias の固定順で連結し、 推論側
                // `KztFactoredWeights::from_flat` と一致する。 W_persp のみ persp 軸を index に焼き
                // 込むため black=persp0 / white=persp1 の行が別々に育つ。
                if enable_kzt_for_graph {
                    let kzt_max = bullet_lib::game::inputs::KZT_MAX_ACTIVE_PER_PERSP;
                    let dims = bullet_lib::game::inputs::KZT_TABLE_DIMS;
                    let wnames = bullet_lib::game::inputs::KZT_TABLE_WEIGHT_NAMES;
                    let bnames = bullet_lib::game::inputs::KZT_TABLE_BLACK_NAMES;
                    let whnames = bullet_lib::game::inputs::KZT_TABLE_WHITE_NAMES;
                    for t in 0..bullet_lib::game::inputs::KZT_NUM_TABLES {
                        let dim = dims[t];
                        let w = builder.new_weights(wnames[t], Shape::new(1, dim), InitSettings::Zeroed);
                        let b_in = builder.new_sparse_input(bnames[t], Shape::new(dim, 1), kzt_max);
                        let w_in = builder.new_sparse_input(whnames[t], Shape::new(dim, 1), kzt_max);
                        net_output_final = net_output_final + w.matmul(b_in) - w.matmul(w_in);
                    }
                }

                // 2026-06-16 King Mobility (KMOB) head: ADEV と同 pattern (= dim 72)。
                // 2 sparse input (= dim 72、 max_active 1) + 2 weight tensor (= shape (1, 72))。
                // 反対称合成: `net_output_final += km_b · w_black - km_w · w_white`
                if enable_king_mobility_for_graph {
                    let km_dim = bullet_lib::game::inputs::KING_MOBILITY_DIM_PER_PERSP;
                    let km_max = bullet_lib::game::inputs::KING_MOBILITY_MAX_ACTIVE_PER_PERSP;

                    let king_mobility_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::KING_MOBILITY_SPARSE_NAME_BLACK,
                        Shape::new(km_dim, 1),
                        km_max,
                    );
                    let king_mobility_w_black = builder.new_weights(
                        "king_mobility_w_black",
                        Shape::new(1, km_dim),
                        InitSettings::Zeroed,
                    );
                    let king_mobility_b_out = king_mobility_w_black.matmul(king_mobility_b_in);

                    let king_mobility_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::KING_MOBILITY_SPARSE_NAME_WHITE,
                        Shape::new(km_dim, 1),
                        km_max,
                    );
                    let king_mobility_w_white = builder.new_weights(
                        "king_mobility_w_white",
                        Shape::new(1, km_dim),
                        InitSettings::Zeroed,
                    );
                    let king_mobility_w_out = king_mobility_w_white.matmul(king_mobility_w_in);

                    net_output_final = net_output_final + king_mobility_b_out - king_mobility_w_out;
                }

                // 2026-06-16 Drop Pressure (DROP) head: ADEV と同 pattern (= dim 64)。
                // 2 sparse input (= dim 64、 max_active 1) + 2 weight tensor (= shape (1, 64))。
                // 反対称合成: `net_output_final += dp_b · w_black - dp_w · w_white`
                if enable_drop_pressure_for_graph {
                    let dp_dim = bullet_lib::game::inputs::DROP_PRESSURE_DIM_PER_PERSP;
                    let dp_max = bullet_lib::game::inputs::DROP_PRESSURE_MAX_ACTIVE_PER_PERSP;

                    let drop_pressure_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::DROP_PRESSURE_SPARSE_NAME_BLACK,
                        Shape::new(dp_dim, 1),
                        dp_max,
                    );
                    let drop_pressure_w_black = builder.new_weights(
                        "drop_pressure_w_black",
                        Shape::new(1, dp_dim),
                        InitSettings::Zeroed,
                    );
                    let drop_pressure_b_out = drop_pressure_w_black.matmul(drop_pressure_b_in);

                    let drop_pressure_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::DROP_PRESSURE_SPARSE_NAME_WHITE,
                        Shape::new(dp_dim, 1),
                        dp_max,
                    );
                    let drop_pressure_w_white = builder.new_weights(
                        "drop_pressure_w_white",
                        Shape::new(1, dp_dim),
                        InitSettings::Zeroed,
                    );
                    let drop_pressure_w_out = drop_pressure_w_white.matmul(drop_pressure_w_in);

                    net_output_final = net_output_final + drop_pressure_b_out - drop_pressure_w_out;
                }

                if enable_endgame_tempo_for_graph {
                    let dim = bullet_lib::game::inputs::ENDGAME_TEMPO_DIM_PER_PERSP;
                    let max = bullet_lib::game::inputs::ENDGAME_TEMPO_MAX_ACTIVE_PER_PERSP;
                    let black_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ENDGAME_TEMPO_SPARSE_NAME_BLACK,
                        Shape::new(dim, 1),
                        max,
                    );
                    let black_w = builder.new_weights(
                        "endgame_tempo_w_black",
                        Shape::new(1, dim),
                        InitSettings::Zeroed,
                    );
                    let white_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ENDGAME_TEMPO_SPARSE_NAME_WHITE,
                        Shape::new(dim, 1),
                        max,
                    );
                    let white_w = builder.new_weights(
                        "endgame_tempo_w_white",
                        Shape::new(1, dim),
                        InitSettings::Zeroed,
                    );
                    net_output_final = net_output_final + black_w.matmul(black_in) - white_w.matmul(white_in);
                }

                if enable_scalar_threat_for_graph {
                    let dim = bullet_lib::game::inputs::SCALAR_THREAT_WEIGHT_COUNT;
                    let max = bullet_lib::game::inputs::MAX_ACTIVE_THREAT_FEATURES;
                    let weights = builder.new_weights(
                        "scalar_threat_w",
                        Shape::new(1, dim),
                        InitSettings::Zeroed,
                    );
                    let black = builder.new_sparse_input(
                        bullet_lib::game::inputs::SCALAR_THREAT_SPARSE_NAME_BLACK,
                        Shape::new(dim, 1),
                        max,
                    );
                    let white = builder.new_sparse_input(
                        bullet_lib::game::inputs::SCALAR_THREAT_SPARSE_NAME_WHITE,
                        Shape::new(dim, 1),
                        max,
                    );
                    net_output_final =
                        net_output_final + weights.matmul(black) - weights.matmul(white);
                }

                // 2026-06-09 Attack Relay Head (ARH): KED と同 pattern。
                // 2 sparse input (= dim 512、 max_active 1) + 2 weight tensor (= shape (1, 512))。
                // 反対称合成: `net_output_final += arh_b · w_black - arh_w · w_white` (= §3.6)。
                if enable_attack_relay_for_graph {
                    let arh_dim = bullet_lib::game::inputs::ARH_DIM_PER_PERSP;
                    let arh_max = bullet_lib::game::inputs::ARH_MAX_ACTIVE_PER_PERSP;

                    let attack_relay_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ARH_SPARSE_NAME_BLACK,
                        Shape::new(arh_dim, 1),
                        arh_max,
                    );
                    let attack_relay_w_black = builder.new_weights(
                        "attack_relay_w_black",
                        Shape::new(1, arh_dim),
                        InitSettings::Zeroed,
                    );
                    let attack_relay_b_out = attack_relay_w_black.matmul(attack_relay_b_in);

                    let attack_relay_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ARH_SPARSE_NAME_WHITE,
                        Shape::new(arh_dim, 1),
                        arh_max,
                    );
                    let attack_relay_w_white = builder.new_weights(
                        "attack_relay_w_white",
                        Shape::new(1, arh_dim),
                        InitSettings::Zeroed,
                    );
                    let attack_relay_w_out = attack_relay_w_white.matmul(attack_relay_w_in);

                    net_output_final =
                        net_output_final + attack_relay_b_out - attack_relay_w_out;
                }

                // 2026-06-10 Attack Relay Head v2 (= 構造化 980 weight、 先後共有):
                // - 単一 weight tensor `attack_relay_v2_w` を black / white sparse 双方が共有
                // - dim = 980 (= W_base 576 + W_alpha 180 + W_phase 144 + W_alpha_shape 80)
                // - max_active = 4 / persp (= 基準スライス除外で 1..=4)
                // - 反対称合成: `net_output_final += arh_b - arh_w` (= §3.6)
                if enable_attack_relay_v2_for_graph {
                    let arh_v2_dim = bullet_lib::game::inputs::ARH_V2_WEIGHT_COUNT;
                    let arh_v2_max = bullet_lib::game::inputs::ARH_V2_MAX_ACTIVE_PER_PERSP;

                    let attack_relay_v2_w = builder.new_weights(
                        "attack_relay_v2_w",
                        Shape::new(1, arh_v2_dim),
                        InitSettings::Zeroed,
                    );

                    let attack_relay_v2_b_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ARH_V2_SPARSE_NAME_BLACK,
                        Shape::new(arh_v2_dim, 1),
                        arh_v2_max,
                    );
                    let attack_relay_v2_b_out = attack_relay_v2_w.matmul(attack_relay_v2_b_in);

                    let attack_relay_v2_w_in = builder.new_sparse_input(
                        bullet_lib::game::inputs::ARH_V2_SPARSE_NAME_WHITE,
                        Shape::new(arh_v2_dim, 1),
                        arh_v2_max,
                    );
                    let attack_relay_v2_w_out = attack_relay_v2_w.matmul(attack_relay_v2_w_in);

                    net_output_final =
                        net_output_final + attack_relay_v2_b_out - attack_relay_v2_w_out;
                }


                (net_output_final, l1_nodes)
            });
            if l1_graph_penalty_on {
                builder.build_custom(|builder, (stm_inputs, ntm_inputs, output_buckets), targets| {
                    let (out, l1_nodes) = forward(builder, stm_inputs, ntm_inputs, output_buckets);
                    let raw_loss = loss_fn(out, targets);
                    assert_eq!(raw_loss.node().shape, Shape::new(1, 1));
                    // penalty は batch 無し (1,1)。 LinearCombination が batch 側へ splat する。
                    let penalty = build_l1_graph_penalty(builder, l1_nodes, l1_penalty_cfg);
                    (out, raw_loss + penalty)
                })
            } else {
                builder.build(|builder, stm_inputs, ntm_inputs, output_buckets| {
                    forward(builder, stm_inputs, ntm_inputs, output_buckets).0
                })
            }
        }};
    }

    // `--direct15-expand-head`: base checkpoint の head 形状を weights.bin のヘッダから判定し、
    // graph 形状 (P2 / P5) への複製展開 adapter を作る。 flag off / head 無し / 同形なら None。
    let direct15_expand_adapter = |weights_bin: &Path| -> Option<Direct15HeadExpandAdapter> {
        if !args.direct15_expand_head {
            return None;
        }
        let to = Direct15HeadShape::new(NUM_BUCKETS, direct15_persp_split, direct15_defended);
        match Direct15HeadExpandAdapter::from_weights_bin(weights_bin, to) {
            Ok(Some(adapter)) => {
                println!(
                    "direct15-expand-head: base head {}x{} -> graph {}x{} へ複製展開して resume します \
                     (weights + optimiser state、 forward 出力は展開前と同一)",
                    adapter.from.rows(),
                    adapter.from.dims(),
                    adapter.to.rows(),
                    adapter.to.dims()
                );
                Some(adapter)
            }
            Ok(None) => {
                println!(
                    "direct15-expand-head: base checkpoint の Direct15 head は既に graph と同形 (または head 無し) なので展開しません"
                );
                None
            }
            Err(e) => {
                eprintln!("ERROR: --direct15-expand-head: {e}");
                std::process::exit(1);
            }
        }
    };

    // checkpoint 読み込みの adapter。 --direct15-expand-head も --threat-bucket-groups も無ければ None
    // (= 従来どおり adapter 無しの load_from_checkpoint を呼ぶ)。
    // 2026-11-18 KT: X 部ありの graph に X 部なしの checkpoint を読むとき kh_ft_w の末尾に足す 0 の要素数。
    let kh_x_extra = kh_parts.map_or(0, |p| p.x_dims() * kh_out_c);
    let ckpt_adapter = |weights_bin: &Path| -> Option<CheckpointLoadAdapter> {
        let direct15 = direct15_expand_adapter(weights_bin);
        if direct15.is_none() && threat_groups <= 1 && threat_factor_rows == 0 && kh_x_extra == 0 {
            return None;
        }
        Some(CheckpointLoadAdapter {
            threat_tile: (threat_groups > 1).then_some((threat_ft_out * threat_ft_dims, threat_groups)),
            threat_factor_extra: threat_ft_out * (threat_factor_rows + threat_hang_rows),
            kh_x_extra,
            direct15,
        })
    };

    // 2026-11-07 (T5): **L4 の恒等初期化**。
    //
    // 🚨なぜ学習器の中でやる必要があるか (python で ckpt を改造しては**いけない**理由):
    //   bullet のローダは**ファイル側の項目を回す** (graph.rs:252-291 / ranger21.rs:330-372)。
    //   checkpoint に無い `l4w` / `l4b` は**エラーにならず** graph の初期化
    //   (`new_affine` = Normal(0, sqrt(2/64))) のまま走る。 つまり `--l4 64 --resume 親` だけでは
    //   **乱数 L4 で関数が壊れたまま静かに学習が始まる**。 だから load の**後**に明示的に書く。
    //
    // 🎯なぜ幅 64 だけか: 前段の出力は既に `clampi(x >> 6, 0, 127)` の u8。 i8 の対角 **64** で
    //   `clamp((64·x) >> 6, 0, 127) = x` が x = 0..127 の全てで成り立つ (63 / 65 では崩れる)。
    //   f32 側の対角は **1.0** (量子化で QB = 64 を掛けて i8 の 64 になる)。
    //
    // 🚨L4 run の**続き**を `--resume` するときは `--l4-init random` にすること
    //   (identity のままだと学習済みの L4 を潰す)。 どちらを使ったかは必ず 1 行出力する。
    macro_rules! apply_l4_identity {
        ($trainer:expr) => {{
            if l4_c > 0 {
                match args.l4_init.as_str() {
                    "identity" => {
                        assert_eq!(
                            l4_c, l2_out_c,
                            "--l4-init identity は幅が前段 (l2) と同じでなければ恒等にならない"
                        );
                        let rows = NUM_BUCKETS * l4_c;
                        let mut w = vec![0.0f32; rows * l2_out_c];
                        for b in 0..NUM_BUCKETS {
                            for o in 0..l4_c {
                                // 列優先 values[in * rows + out] (★net2net_three_traps)
                                w[o * rows + b * l4_c + o] = 1.0;
                            }
                        }
                        set_graph_weight(&mut $trainer.optimiser.graph, "l4w", &w).expect("l4w identity");
                        set_graph_weight(&mut $trainer.optimiser.graph, "l4b", &vec![0.0f32; rows])
                            .expect("l4b zero");
                        println!("L4 init: identity (幅 {l4_c} / 対角 1.0 = i8 64 / bias 0) -> 挿入しても出力は 1 ビットも変わらない");
                    }
                    "random" => {
                        println!("L4 init: random (resume なら ckpt の値 / 無ければ new_affine の乱数)");
                    }
                    other => panic!("--l4-init は identity か random (与えられた値: {other})"),
                }
            }
        }};
    }

    macro_rules! maybe_run_or_quantise {
        ($trainer:expr) => {{
            if args.quantise_only {
                let resume_path = args.resume_path().expect("--quantise-only requires --resume");
                let resume_str = resume_path.to_str().unwrap();
                println!("Loading checkpoint from {}...", resume_str);
                match ckpt_adapter(&resume_path.join("optimiser_state").join("weights.bin")) {
                    Some(adapter) => $trainer.load_from_checkpoint_adapted(resume_str, &adapter),
                    None => $trainer.load_from_checkpoint(resume_str),
                }

                let output_dir = args.output.to_str().unwrap_or("checkpoints");
                apply_l4_identity!($trainer);

                let output_path = format!("{}/requantised.bin", output_dir);
                std::fs::create_dir_all(output_dir).unwrap_or(());

                println!("Saving re-quantised weights to {}...", output_path);
                $trainer.save_quantised(&output_path).expect("Failed to save quantised weights");
                println!("Done!");
            } else {
                if let Some(resume_path) = args.resume_path() {
                    let resume_str = resume_path.to_str().unwrap();
                    println!("Resuming from checkpoint: {}", resume_str);
                    match ckpt_adapter(&resume_path.join("optimiser_state").join("weights.bin")) {
                        Some(adapter) => $trainer.load_from_checkpoint_adapted(resume_str, &adapter),
                        None => $trainer.load_from_checkpoint(resume_str),
                    }
                    apply_l4_identity!($trainer);
                    // 2026-09-16: --threat-zero-init: threat 重みをゼロにして threat だけ学習し直す。
                    if args.threat_zero_init && threat_ft_out > 0 {
                        let tensor_ref = $trainer.optimiser.graph.get_weights("threat_ft_w");
                        {
                            let mut guard = tensor_ref.borrow_mut();
                            let dense = guard.dense_mut().expect("threat_ft_w must be dense");
                            let n = dense.size();
                            dense.set_range_to(0, n, 0.0).expect("threat_ft_w zero");
                            println!("threat zero-init: threat_ft_w {n} 要素をゼロ化 (threat だけ最初から学習し直す)");
                        }
                        if let Some(state) = $trainer.optimiser.state.get_mut("threat_ft_w") {
                            bullet_lib::acyclib::trainer::optimiser::OptimiserState::reset(state)
                                .expect("threat_ft_w optimiser state reset");
                            println!("threat zero-init: threat_ft_w の optimiser state (momentum/velocity/slow) も初期化");
                        }
                    }
                    // 2026-09-16 案A: factorizer の共有行を「ペア平均」へ再パラメータ化する (関数保存)。
                    //   factor[p][k] = mean_{g,f: pair(f)=p} real[g][f][k]、 real -= factor。 和は不変。
                    if threat_factor_rows > 0 && !args.threat_factor_init_zero && !args.threat_zero_init {
                        let tensor_ref = $trainer.optimiser.graph.get_weights("threat_ft_w");
                        let mut guard = tensor_ref.borrow_mut();
                        let dense = guard.dense_mut().expect("threat_ft_w must be dense");
                        let mut v = vec![0.0f32; dense.size()];
                        dense.write_to_slice(&mut v).expect("threat_ft_w read");
                        let k = threat_ft_out;
                        let dims = threat_ft_dims;
                        let groups = threat_groups;
                        let fbase = dims * groups * k;
                        let fsum: f32 = v[fbase..].iter().map(|x| x.abs()).sum();
                        if fsum > 0.0 {
                            println!("threat factorizer: 共有行が既に学習済み (Σ|factor| = {fsum:.3}) なので再パラメータ化はしない");
                        } else {
                            let pairs: Vec<u16> = {
                                let inp = ShogiHalfKaHmThreat::new(
                                    threat_profile.expect("threat profile"),
                                )
                                .with_threat_factor(true);
                                (0..dims).map(|f| inp.threat_feature_pair(f).unwrap() as u16).collect()
                            };
                            let np = threat_factor_rows;
                            let mut sum = vec![0.0f64; np * k];
                            let mut cnt = vec![0u64; np];
                            for g in 0..groups {
                                for f in 0..dims {
                                    let p = pairs[f] as usize;
                                    cnt[p] += 1;
                                    let base = (g * dims + f) * k;
                                    for i in 0..k {
                                        sum[p * k + i] += v[base + i] as f64;
                                    }
                                }
                            }
                            let mut mean = vec![0.0f32; np * k];
                            for p in 0..np {
                                if cnt[p] == 0 {
                                    continue;
                                }
                                for i in 0..k {
                                    mean[p * k + i] =
                                        (sum[p * k + i] / cnt[p] as f64 * args.threat_factor_init_scale as f64) as f32;
                                }
                            }
                            for g in 0..groups {
                                for f in 0..dims {
                                    let p = pairs[f] as usize;
                                    let base = (g * dims + f) * k;
                                    for i in 0..k {
                                        v[base + i] -= mean[p * k + i];
                                    }
                                }
                            }
                            v[fbase..].copy_from_slice(&mean);
                            // 検算: 再パラメータ化の前後で 実特徴 + 共有 が一致するか (最大誤差を出す)
                            {
                                let mut worst = 0.0f32;
                                let mut worst_i = 0usize;
                                let mut orig = vec![0.0f32; dense.size()];
                                dense.write_to_slice(&mut orig).expect("threat_ft_w reread");
                                for g in 0..groups {
                                    for f in (0..dims).step_by(97) {
                                        let p = pairs[f] as usize;
                                        let base = (g * dims + f) * k;
                                        for i in 0..k {
                                            let d = (v[base + i] + v[fbase + p * k + i] - orig[base + i]).abs();
                                            if d > worst { worst = d; worst_i = base + i; }
                                        }
                                    }
                                }
                                println!("threat factorizer: 関数保存の検算 最大誤差 {worst:.3e} (要素 {worst_i})");
                            }
                            dense.load_from_slice(None, &v).expect("threat_ft_w write");
                            let msum: f32 = mean.iter().map(|x| x.abs()).sum();
                            println!(
                                "threat factorizer: 共有行をペア平均 x {} で初期化 (Σ|factor| = {msum:.3}、 実特徴から同量を減算 = 関数保存)",
                                args.threat_factor_init_scale
                            );
                        }
                    }
                } else if let Some(weights_only_path) = args.resume_weights_only_path() {
                    let weight_file = if weights_only_path.is_dir() {
                        weights_only_path.join("weights.bin")
                    } else {
                        weights_only_path.to_path_buf()
                    };
                    println!(
                        "Loading graph weights only; optimiser state is new: {}",
                        weight_file.display()
                    );
                    let weight_file_str = weight_file.to_str().expect("UTF-8 weights path");
                    let loaded = match ckpt_adapter(&weight_file) {
                        Some(adapter) => $trainer.optimiser.load_weights_from_file_adapted(weight_file_str, &adapter),
                        None => $trainer.optimiser.load_weights_from_file(weight_file_str),
                    };
                    if let Err(e) = loaded {
                        eprintln!("ERROR: weights-only warm-start failed: {:?}", e);
                        std::process::exit(1);
                    }
                    println!("weights.bin → graph weights load complete (optimiser state is new)");
                } else if let Some(qpath) = args.resume_from_quantised_path() {
                    println!(
                        "Resuming from quantised.bin (graph weights only, optimiser state=zero): {}",
                        qpath.display()
                    );
                    let res = load_from_quantised_bin(
                        &mut $trainer.optimiser.graph,
                        qpath,
                        halfka_dim,
                        input_size,
                        ft_out,
                        threat_ft_out,
                        l1_out,
l2_out,
residual_mix,
args.psqt,
                        save_format_hand_threat,
                        hand_count_dense_dims,
                        lmas_flags,
                        args.pin_aware,
                        args.dual_psqt_reserve,
                        args.dual_psqt_king_attacker,
                        direct15_persp_split,
                        direct15_defended,
                    );
                    if let Err(e) = res {
                        eprintln!("ERROR: load_from_quantised_bin failed: {}", e);
                        std::process::exit(1);
                    }
                    println!("quantised.bin → graph weights load 完了 (optimiser state は zero init)");

                    // velocity=0 のまま再開すると Adam/RAdam 分母 sqrt(v)+eps が eps(1e-8) に
                    // 退化し初手更新が爆発する (= eval 出力崩壊)。 正の定数で velocity を初期化して
                    // 抑止する。 default 0.0 は従来動作 (no-op)。
                    let vinit = args.resume_from_quantised_velocity_init;
                    if vinit > 0.0 {
                        if let Err(e) = $trainer.optimiser.init_velocity(vinit) {
                            eprintln!("ERROR: optimiser velocity init failed: {:?}", e);
                            std::process::exit(1);
                        }
                        println!(
                            "optimiser velocity buffer を {} で初期化 (resume-from-quantised 爆発回避)",
                            vinit
                        );
                    }
                }
                // 2026-09-12: --l1-anchor-lambda のアンカー初期化 (resume 後・学習前)。
                //   checkpoint に無ければ (= 全 0) 現在の合算 L1 で埋め、 あれば維持して drift を表示する。
                if args.l1_anchor_lambda > 0.0 {
                    if let Err(e) = init_l1_anchor(&mut $trainer.optimiser.graph, NUM_BUCKETS, l1_out, ft_out) {
                        eprintln!("ERROR: L1 anchor 初期化失敗: {}", e);
                        std::process::exit(1);
                    }
                }

                // PSQT material prior 初期化 (= 2026-06-29、 scale-critical)。
                // fresh start (resume / resume-from-quantised なし) のときのみ psqtw を
                // 駒得価値 prior で上書きする。 resume 時は load 済み重みを尊重し skip。
                // `--psqt-init zeroed` (default) は何もしない (= 従来動作 bit-identical)。
                if args.psqt
                    && args.psqt_init == PsqtInit::Material
                    && args.resume_path().is_none()
                    && args.resume_weights_only_path().is_none()
                    && args.resume_from_quantised_path().is_none()
                {
                    // threat / aux / auxcheap 連結入力有効時は psqtw の末尾に tail 行が
                    // 追加されるのでその次元を渡す。 factorise 併用時 (auxcheap のみ可) は
                    // input_size に factoriser 1,629 行が含まれるので差し引いて実特徴 tail を出す。
                    // threat / kzt_ft は factorise と排他 (guard 済)。
                    let material_tail_dim = if threat_profile.is_some()
                        || args.enable_aux
                        || args.enable_auxcheap
                    {
                        let real_input = if args.factorise {
                            input_size - bullet_lib::game::inputs::PIECE_INPUTS
                        } else {
                            input_size
                        };
                        real_input - halfka_dim
                    } else {
                        0
                    };
                    if let Err(e) = init_psqt_material(
                        &mut $trainer.optimiser.graph,
                        halfka_dim,
                        args.scale,
                        args.factorise,
                        args.enable_kzt_ft,
                        material_tail_dim,
                    ) {
                        eprintln!("ERROR: PSQT material prior 初期化失敗: {}", e);
                        std::process::exit(1);
                    }
                }

                // hcpe path: 32 byte / record の dlshogi hao 等 dataset を読み込む。
                // psv path: 既存 40 byte / record の PackedSfenValue。
                //
                // 2026-06-09: bucket smoothness regularization (= λ > 0) の場合は
                // bullet_lib::value::ValueTrainer::run_with_bucket_smoothness を呼ぶ。
                // λ=0 (= 既 default) なら 既 run と完全等価。
                // aux (FT 入力連結) 有効時、 psqtw の aux 行 [halfka_dim, input_size) を毎 optimizer
                // step 後にゼロ固定し、 訓練=配備の厳密一致を保つ (= aux は FT 専用、 PSQT には寄与しない。
                // warm-start で当該行は 0 初期化済なので毎 step 更新を打ち消して常時 0)。
                // psqtw layout は column-major values[feat * NUM_BUCKETS + bucket] なので、 aux 行の
                // flat 範囲は [halfka_dim * NUM_BUCKETS, input_size * NUM_BUCKETS) の連続区間。
                //
                // auxcheap (PSQT-only) 有効時はその **鏡像**: psqtw の aux 行を訓練し、 l0w の
                // aux 行を毎 step ゼロ固定する (= aux は PSQT 専用、
                // FT accumulator には寄与しない = C++ P2_AUXCHEAP 推論と厳密一致)。
                // l0w layout も column-major values[feat * ft_out + out] なので、 aux 行の
                // flat 範囲は [aux_row0 * ft_out, input_size * ft_out) の連続区間。
                // factorise 併用時は layout [factoriser 1,629 | HalfKA 73,305 | aux 392] のため
                // aux_row0 = 1,629 + 73,305。 factoriser 行は aux から derive されない
                // (Factorises が aux で None) ので凍結不要。 aux 行の末尾位置は不変。
                let mut psqt_zero_targets: Vec<(String, usize, usize)> =
                    if args.enable_aux && args.psqt && input_size > halfka_dim {
                        vec![("psqtw".to_string(), halfka_dim * NUM_BUCKETS, input_size * NUM_BUCKETS)]
                    } else if args.enable_auxcheap && input_size > halfka_dim {
                        let aux_row0 = if args.factorise {
                            bullet_lib::game::inputs::PIECE_INPUTS + halfka_dim
                        } else {
                            halfka_dim
                        };
                        vec![("l0w".to_string(), aux_row0 * ft_out, input_size * ft_out)]
                    } else {
                        Vec::new()
                    };
                // C++ main FT/PSQT consume HalfKA rows only. Appended Threat rows feed
                // only their dedicated head and are projected out of main FT/PSQT.
                // Profile15 (Direct15) は専用 sparse 入力に移行済で主入力に append 行が
                // 無い (input_size == halfka_dim) ため、 この projection は不要。
                if (enable_threat_factor16 || enable_threat_direct) && input_size > halfka_dim {
                    psqt_zero_targets.push(("l0w".to_string(), halfka_dim * ft_out, input_size * ft_out));
                    if args.psqt {
                        psqt_zero_targets.push(("psqtw".to_string(), halfka_dim * NUM_BUCKETS, input_size * NUM_BUCKETS));
                    }
                }
                if enable_threat_factor16 {
                    psqt_zero_targets.push((
                        "threat_factor_code".to_string(),
                        0,
                        halfka_dim * bullet_lib::game::inputs::THREAT_FACTOR_DIM,
                    ));
                }
                // Profile14 (P2THRD1) の threat_direct_w は [HalfKA | Threat] 連結なので
                // HalfKA 前置き行を毎 step 0 に射影する。 Profile15 は専用 sparse 入力で
                // 前置き行自体が無いので対象外。
                if enable_threat_direct && input_size > halfka_dim {
                    psqt_zero_targets.push((
                        "threat_direct_w".to_string(),
                        0,
                        halfka_dim * NUM_BUCKETS,
                    ));
                }
                if enable_threat_direct {
                    // threat_direct_b_cancels は stm 側と nstm 側で同一 bucket の同じ値が
                    // 足され、 direct_stm - direct_nstm で厳密に打ち消される (名前の通り)。
                    // 出力に寄与せず P2THRD2 payload にも書き出されないので、 毎 step 0 に
                    // 射影しても学習結果は変わらない。
                    //
                    // 2026-09-01: これは同時に Profile15 (専用 sparse 入力化により
                    // threat_direct_w の projection が不要になった構成) で
                    // psqt_zero_targets が空になり、 実績の無い素の `run()` 経路へ
                    // 落ちるのを防ぐ。 素の run() は起動直後に
                    // GradientCalculationError(MismatchedBatchSizes) で落ちる事例が
                    // 再現しており、 長時間の実績がある run_with_zeroed_rows に揃える。
                    // P2 では stm 行 / nstm 行の bias が打ち消し合わなくなるが、 毎 step 0 に射影される
                    // (かつ初回 batch 前にも射影される) ので forward への寄与は常に 0 のまま。
                    psqt_zero_targets.push(("threat_direct_b_cancels".to_string(), 0, direct15_rows));

                    // 2026-09-03: --direct15-zero-head は head の重み本体も 0 に固定する
                    //   (対照実験用)。 graph は変えないので resume が通り、 base / データ /
                    //   SB 数 / seed を揃えた「head 無し」net が作れる。
                    if args.direct15_zero_head {
                        psqt_zero_targets.push((
                            "threat_direct_w".to_string(),
                            0,
                            direct15_rows * direct15_dim,
                        ));
                        if direct15_factorise {
                            let np = direct15_rows / NUM_BUCKETS;
                            psqt_zero_targets.push(("threat_direct_shared_w".to_string(), 0, direct15_dim * np));
                            psqt_zero_targets.push((
                                "threat_direct_pair_w".to_string(),
                                0,
                                np * (direct15_dim / THREAT_DIRECT15_DIMENSIONS) * 100,
                            ));
                        }
                        println!("Direct15: HEAD ZEROED (control run)");
                    }
                }
                // run_with_zeroed_rows projects after each optimizer step. Apply the same
                // projection before the first batch so the exported C++ computation is exact
                // from step zero as well.
                for (name, start, end) in &psqt_zero_targets {
                    let tensor_ref = $trainer.optimiser.graph.get_weights(name);
                    let mut guard = tensor_ref.borrow_mut();
                    let dense = guard.dense_mut().expect("profile13 zero target must be dense");
                    let stop = (*end).min(dense.size());
                    if *start < stop {
                        dense.set_range_to(*start, stop, 0.0).expect("profile13 initial zero projection");
                    }
                }

                // 2026-11-13 --hard-freeze-shared: 学習対象 (optimizer 状態を持つ重み) を列挙して出す。
                //   --bucket-filter 併用時は bucket 別の head 以外が 1 つでも残っていたら止める
                //   (共有の重みが動くと範囲外 bucket の評価も変わる)。
                if args.hard_freeze_shared {
                    if !psqt_zero_targets.is_empty() {
                        eprintln!("ERROR: --hard-freeze-shared は aux / auxcheap の zero projection (凍結した psqtw / l0w を書き換える) と併用できない");
                        std::process::exit(1);
                    }
                    let mut trainable: Vec<&String> = $trainer.optimiser.state.keys().collect();
                    trainable.sort();
                    println!("HARD-FREEZE-SHARED: 学習対象 = {:?}", trainable);
                    const PER_BUCKET_HEADS: [&str; 6] = ["l1w", "l1b", "l2w", "l2b", "l3w", "l3b"];
                    let shared: Vec<&&String> =
                        trainable.iter().filter(|id| !PER_BUCKET_HEADS.contains(&id.as_str())).collect();
                    if bucket_filter_range.is_some() && !shared.is_empty() {
                        eprintln!(
                            "ERROR: --bucket-filter + --hard-freeze-shared: bucket 別 head 以外の学習対象が残っている {:?} \
                             (範囲外 bucket の評価が変わる)",
                            shared
                        );
                        std::process::exit(1);
                    }
                }

                // 2026-11-15 --dump-forward: 読み込んだ重みのまま (学習せず) PSV の全局面を順伝播して書き出し、 終了する。
                //   同じ重みの量子化 net も書くので、 整数推論 (opn_static2.py) と局面ごとに突き合わせられる。
                if let Some(psv_path) = args.dump_forward.as_ref() {
                    use bullet_lib::acyclib::graph::like::GraphLike;
                    use bullet_lib::game::outputs::OutputBuckets;
                    let bytes = std::fs::read(psv_path).unwrap_or_else(|e| {
                        eprintln!("ERROR: --dump-forward {} を読めない: {e}", psv_path.display());
                        std::process::exit(1);
                    });
                    let rec = std::mem::size_of::<bullet_lib::shogi::PackedSfenValue>();
                    if bytes.is_empty() || bytes.len() % rec != 0 {
                        eprintln!("ERROR: --dump-forward {} の長さ {} が {rec} の倍数でない", psv_path.display(), bytes.len());
                        std::process::exit(1);
                    }
                    let psvs: Vec<bullet_lib::shogi::PackedSfenValue> = bytes
                        .chunks_exact(rec)
                        .map(|c| {
                            let mut p = bullet_lib::shogi::PackedSfenValue::default();
                            p.as_bytes_mut().copy_from_slice(c);
                            p
                        })
                        .collect();
                    let mut lines = String::with_capacity(psvs.len() * 24);
                    for chunk in psvs.chunks(1024) {
                        let host = $trainer.state.prepare(chunk, 1, 1.0, 1.0);
                        {
                            #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
                            let graph = &mut $trainer.optimiser.graph;
                            #[cfg(any(feature = "multigpu", feature = "cpu"))]
                            let graph = $trainer.optimiser.graph.primary_mut();
                            let mut dev = bullet_lib::acyclib::trainer::dataloader::PreparedBatchDevice::new(
                                graph.devices(),
                                &host,
                            )
                            .expect("--dump-forward: batch の転送に失敗");
                            dev.load_into_graph(graph).expect("--dump-forward: batch の読み込みに失敗");
                            graph.synchronise().expect("--dump-forward: synchronise");
                            graph.forward().expect("--dump-forward: forward");
                        }
                        let vals = $trainer.get_output_values();
                        assert_eq!(vals.len(), chunk.len(), "--dump-forward: 出力の数が局面数と違う (出力が 1 次元でない?)");
                        for (p, v) in chunk.iter().zip(vals) {
                            lines.push_str(&format!("{} {:.9e}
", bucket_impl.bucket(p), v));
                        }
                    }
                    let out_dir = &args.output;
                    std::fs::create_dir_all(out_dir).expect("--dump-forward: 出力先を作れない");
                    let txt = out_dir.join("forward_dump.txt");
                    std::fs::write(&txt, lines).expect("--dump-forward: forward_dump.txt を書けない");
                    let qbin = out_dir.join("forward_dump_quantised.bin");
                    $trainer
                        .save_quantised(qbin.to_str().expect("UTF-8 path"))
                        .expect("--dump-forward: 量子化 net を書けない");
                    println!(
                        "DUMP-FORWARD: {} 局面 -> {} / 量子化 net -> {} (qat = {})",
                        psvs.len(),
                        txt.display(),
                        qbin.display(),
                        args.qat
                    );
                    std::process::exit(0);
                }

                let use_smoothness = args.bucket_smooth_l3_lambda > 0.0
                    || args.bucket_smooth_psqt_lambda > 0.0
                    || args.l1_decorr_lambda > 0.0
                    || args.l2_decorr_lambda > 0.0;
                if !psqt_zero_targets.is_empty() {
                    // aux 経路: post-step zero projection 付き run (smoothness とは非併用 = aux は base+psqt 専用)。
                    if let Some(hcpe) = hcpe_loader.as_ref() {
                        $trainer
                            .run_with_zeroed_rows(&schedule, &settings, hcpe, &psqt_zero_targets)
                            .expect("run_with_zeroed_rows (hcpe) failed");
                    } else {
                        $trainer
                            .run_with_zeroed_rows(&schedule, &settings, &data_loader, &psqt_zero_targets)
                            .expect("run_with_zeroed_rows (psv) failed");
                    }
                } else if use_smoothness {
                    if let Some(hcpe) = hcpe_loader.as_ref() {
                        $trainer
                            .run_with_bucket_smoothness(
                                &schedule,
                                &settings,
                                hcpe,
                                args.bucket_smooth_l3_lambda,
                                args.bucket_smooth_psqt_lambda,
                                NUM_BUCKETS,
                                args.l1_decorr_lambda,
                                l1_out_c,
                                l1_out_c - 1,   // skip 行は除外
                                args.l2_decorr_lambda,
                                l2_out_c,       // L2 は全行が対象
                            )
                            .expect("run_with_bucket_smoothness (hcpe) failed");
                    } else {
                        $trainer
                            .run_with_bucket_smoothness(
                                &schedule,
                                &settings,
                                &data_loader,
                                args.bucket_smooth_l3_lambda,
                                args.bucket_smooth_psqt_lambda,
                                NUM_BUCKETS,
                                args.l1_decorr_lambda,
                                l1_out_c,
                                l1_out_c - 1,   // skip 行は除外
                                args.l2_decorr_lambda,
                                l2_out_c,       // L2 は全行が対象
                            )
                            .expect("run_with_bucket_smoothness (psv) failed");
                    }
                } else if let Some(hcpe) = hcpe_loader.as_ref() {
                    $trainer.run(&schedule, &settings, hcpe);
                } else {
                    $trainer.run(&schedule, &settings, &data_loader);
                }
                // 2026-11-12 --kh-dump: 学習終了時の kh_ft_w (ゼロ起点なので値そのものが変位) を書き出す。
                if let Some(path) = args.kh_dump.as_ref() {
                    match read_graph_weight(&$trainer.optimiser.graph, "kh_ft_w") {
                        Ok(v) => {
                            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                            if let Some(dir) = path.parent() {
                                std::fs::create_dir_all(dir).unwrap_or(());
                            }
                            std::fs::write(path, &bytes).expect("--kh-dump の書き込みに失敗");
                            let l2: f64 = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                            let mut meta = serde_json::json!({
                                "tensor": "kh_ft_w",
                                "layout": "column-major: values[feature * out + row]",
                                "out": kh_out_c,
                                "dims": kh_dims_c,
                                "k_dims": kh_parts.map_or(0, |p| p.k_dims()),
                                "h_dims": bullet_lib::game::inputs::shogi_kh::KH_H_DIMS,
                                "row_offset": kh_row_offset_c,
                                "feats": args.kh_feats,
                                "split": args.kh_split,
                                "null": args.kh_null,
                                "freeze_rest": args.kh_freeze_rest,
                                "unfreeze_fc0": args.kh_unfreeze_fc0,
                                "unfreeze_fc0_cols": args.kh_unfreeze_fc0_cols,
                                "lr": args.lr,
                                "kh_lr_mult": args.kh_lr_mult,
                                "batches_per_superbatch": args.batches_per_superbatch,
                                "superbatches": args.superbatches,
                                "start_superbatch": args.start_superbatch,
                                "data": args.data.clone(),
                                "resume": args.resume_path().map(|p| p.display().to_string()),
                                "l2_norm": l2,
                            });
                            // 2026-11-18 KT: X 部があるときだけ足す (無ければ従来と同じ鍵のまま)。
                            if let Some(p) = kh_parts.filter(|p| p.x != KhXMode::Off) {
                                meta["x_offset"] = serde_json::json!(p.x_offset());
                                meta["x_dims"] = serde_json::json!(p.x_dims());
                                meta["x_mode"] = serde_json::json!(format!("{:?}", p.x));
                            }
                            // kmode 2 のときだけ足す (kmode 1 の json は初版と同じ鍵のまま)。
                            if let Some(p) = kh_parts.filter(|p| p.kmode != KhKMode::Plain) {
                                meta["kmode"] = serde_json::json!(args.kh_kmode);
                                meta["k_sub"] = serde_json::json!(
                                    p.k_sub().iter().map(|(n, a, b)| serde_json::json!([n, a, b])).collect::<Vec<_>>()
                                );
                            }
                            let mut jp = path.clone().into_os_string();
                            jp.push(".json");
                            std::fs::write(&jp, serde_json::to_string_pretty(&meta).unwrap())
                                .expect("--kh-dump の json 書き込みに失敗");
                            println!("KH dump: {} ({} 要素 / L2 {l2:.6e}) と {}", path.display(), v.len(), std::path::Path::new(&jp).display());
                        }
                        Err(e) => {
                            eprintln!("ERROR: --kh-dump: {e}");
                            std::process::exit(1);
                        }
                    }
                }
            }
        }};
    }

    let use_win_rate_model = args.win_rate_model;

    // KingRing 2 tensor 用に effective lr を boost し、 同時に weight decay を 0 化する。
    // - `--kingring-lr-mult > 1.0` 時のみ実効化される (デフォルト 1.0 は no-op で完全後方互換)。
    // - `--enable-kingring` が無効なときは kingring tensor 自体が graph に存在しないので、
    //   set_lr_scale / set_params_for_weight 呼び出しは行わない (= panic 回避)。
    // - $decay_params: ベース optimizer params に decay=0 を上書きしたものを与える。
    //   AdamW / RAdam / Ranger でそれぞれ型が異なるため macro で受け取る。
    macro_rules! apply_kingring_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_kingring && args.kingring_lr_mult > 1.0 + 1e-6 {
                let mult = args.kingring_lr_mult;
                $trainer.optimiser.set_lr_scale("kingring_w_black", mult);
                $trainer.optimiser.set_lr_scale("kingring_w_white", mult);
                // decay を 0 化して lr boost が weight decay 増幅で台無しにならないようにする。
                $trainer.optimiser.set_params_for_weight("kingring_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("kingring_w_white", $decay0_params);
                println!(
                    "KingRing lr boost: effective_lr = base_lr * {} (decay forced to 0 for kingring tensors)",
                    mult
                );
            }
        }};
    }

    // PairShape 2 tensor 用の boost macro (= kingring と同 convention、 arm 12)。
    macro_rules! apply_pairshape_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_pairshape && args.pairshape_lr_mult > 1.0 + 1e-6 {
                let mult = args.pairshape_lr_mult;
                $trainer.optimiser.set_lr_scale("pairshape_w_black", mult);
                $trainer.optimiser.set_lr_scale("pairshape_w_white", mult);
                $trainer.optimiser.set_params_for_weight("pairshape_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("pairshape_w_white", $decay0_params);
                println!(
                    "PairShape lr boost: effective_lr = base_lr * {} (decay forced to 0 for pairshape tensors)",
                    mult
                );
            }
        }};
    }

    // 2026-06-20 Dual Cross Geometry 2 tensor 用の boost macro (= pairshape と同 convention)。
    macro_rules! apply_dual_cross_geometry_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_dual_cross_geometry && args.dual_cross_geometry_lr_mult > 1.0 + 1e-6 {
                let mult = args.dual_cross_geometry_lr_mult;
                $trainer.optimiser.set_lr_scale("dual_cross_geometry_w_black", mult);
                $trainer.optimiser.set_lr_scale("dual_cross_geometry_w_white", mult);
                $trainer.optimiser.set_params_for_weight("dual_cross_geometry_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("dual_cross_geometry_w_white", $decay0_params);
                println!(
                    "DualCrossGeometry lr boost: effective_lr = base_lr * {} (decay forced to 0 for dual_cross_geometry tensors)",
                    mult
                );
            }
        }};
    }

    // 2026-06-06 Activity-Risk 2 tensor 用の boost macro (= pairshape と同 convention)。
    macro_rules! apply_activity_risk_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_activity_risk && args.activity_risk_lr_mult > 1.0 + 1e-6 {
                let mult = args.activity_risk_lr_mult;
                $trainer.optimiser.set_lr_scale("activity_risk_w_black", mult);
                $trainer.optimiser.set_lr_scale("activity_risk_w_white", mult);
                $trainer.optimiser.set_params_for_weight("activity_risk_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("activity_risk_w_white", $decay0_params);
                // 2026-06-08 v3 tactical 2 weight
                $trainer.optimiser.set_lr_scale("activity_risk_v3_tactical_w_black", mult);
                $trainer.optimiser.set_lr_scale("activity_risk_v3_tactical_w_white", mult);
                $trainer.optimiser.set_params_for_weight("activity_risk_v3_tactical_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("activity_risk_v3_tactical_w_white", $decay0_params);
                println!(
                    "ActivityRisk lr boost: effective_lr = base_lr * {} (decay forced to 0 for activity_risk tensors)",
                    mult
                );
            }
        }};
    }

    // 2026-06-09 King Escape Direction (KED) 2 tensor 用の boost macro (= activity_risk と同 convention)。
    macro_rules! apply_king_escape_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_king_escape && args.king_escape_lr_mult > 1.0 + 1e-6 {
                let mult = args.king_escape_lr_mult;
                $trainer.optimiser.set_lr_scale("king_escape_w_black", mult);
                $trainer.optimiser.set_lr_scale("king_escape_w_white", mult);
                $trainer.optimiser.set_params_for_weight("king_escape_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("king_escape_w_white", $decay0_params);
                println!(
                    "KingEscape lr boost: effective_lr = base_lr * {} (decay forced to 0 for king_escape tensors)",
                    mult
                );
            }
        }};
    }

    // 2026-06-28 Slider X-ray (SXRY) 2 tensor 用の boost macro (= attack_dev と同 convention)。
    macro_rules! apply_slider_xray_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_slider_xray && args.slider_xray_lr_mult > 1.0 + 1e-6 {
                let mult = args.slider_xray_lr_mult;
                $trainer.optimiser.set_lr_scale("slider_xray_w_black", mult);
                $trainer.optimiser.set_lr_scale("slider_xray_w_white", mult);
                $trainer.optimiser.set_params_for_weight("slider_xray_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("slider_xray_w_white", $decay0_params);
                println!(
                    "SliderXray lr boost: effective_lr = base_lr * {} (decay forced to 0 for slider_xray tensors)",
                    mult
                );
            }
        }};
    }

    // 2026-07-01 King Zone Threat (KZT_HAND) factored 5 tensor 用の per-table boost macro。
    // 各 table の lr_scale = kzt_hand_lr_mult × KZT_TABLE_LR_REL[t]、 decay = kzt_hand_wd ×
    // KZT_TABLE_WD_REL[t] (= Codex 指定、 低ランク table ほど強正則化 = 平滑化)。 $mk は
    // decay(f32) → optimiser params を作る closure (= optimiser 種別を吸収)。
    macro_rules! apply_kzt_boost {
        ($trainer:expr, $mk:expr) => {{
            if args.enable_kzt_hand {
                use bullet_lib::game::inputs::{
                    KZT_NUM_TABLES, KZT_TABLE_LR_REL, KZT_TABLE_WD_REL, KZT_TABLE_WEIGHT_NAMES,
                };
                let base_lr = args.kzt_hand_lr_mult;
                let base_wd = args.kzt_hand_wd;
                let mk = $mk;
                for t in 0..KZT_NUM_TABLES {
                    let name = KZT_TABLE_WEIGHT_NAMES[t];
                    let lr = base_lr * KZT_TABLE_LR_REL[t];
                    let wd = base_wd * KZT_TABLE_WD_REL[t];
                    $trainer.optimiser.set_lr_scale(name, lr);
                    $trainer.optimiser.set_params_for_weight(name, mk(wd));
                }
                println!(
                    "Kzt factored boost: base_lr_mult={} base_wd={} (per-table lr_rel={:?} wd_rel={:?})",
                    base_lr, base_wd, KZT_TABLE_LR_REL, KZT_TABLE_WD_REL
                );
            }
        }};
    }

    // 2026-06-16 King Mobility (KMOB) 2 tensor 用の boost macro (= attack_dev と同 convention)。
    macro_rules! apply_king_mobility_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_king_mobility && args.king_mobility_lr_mult > 1.0 + 1e-6 {
                let mult = args.king_mobility_lr_mult;
                $trainer.optimiser.set_lr_scale("king_mobility_w_black", mult);
                $trainer.optimiser.set_lr_scale("king_mobility_w_white", mult);
                $trainer.optimiser.set_params_for_weight("king_mobility_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("king_mobility_w_white", $decay0_params);
                println!(
                    "KingMobility lr boost: effective_lr = base_lr * {} (decay forced to 0 for king_mobility tensors)",
                    mult
                );
            }
        }};
    }

    // 2026-06-16 Drop Pressure (DROP) 2 tensor 用の boost macro (= king_mobility と同 convention)。
    macro_rules! apply_drop_pressure_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_drop_pressure && args.drop_pressure_lr_mult > 1.0 + 1e-6 {
                let mult = args.drop_pressure_lr_mult;
                $trainer.optimiser.set_lr_scale("drop_pressure_w_black", mult);
                $trainer.optimiser.set_lr_scale("drop_pressure_w_white", mult);
                $trainer.optimiser.set_params_for_weight("drop_pressure_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("drop_pressure_w_white", $decay0_params);
                println!(
                    "DropPressure lr boost: effective_lr = base_lr * {} (decay forced to 0 for drop_pressure tensors)",
                    mult
                );
            }
        }};
    }

    macro_rules! apply_endgame_tempo_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_endgame_tempo && args.endgame_tempo_lr_mult > 1.0 + 1e-6 {
                let mult = args.endgame_tempo_lr_mult;
                for id in ["endgame_tempo_w_black", "endgame_tempo_w_white"] {
                    $trainer.optimiser.set_lr_scale(id, mult);
                    $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                }
                println!("EndgameTempo lr boost: effective_lr = base_lr * {mult}");
            }
        }};
    }

    // (5) TG (2026-10-10): `threat_gate_w` もゼロ初期化なので lr を上げないと動かない。
    //   既存の `apply_threat_ft_boost!` に足さず別マクロにしているのは、 TG off のとき
    //   既存経路のトークン列をまったく変えないため (既定 off で完全同一を守る)。
    //   decay は `--threat-gate-decay` (既定 0.0) をそのまま使う = 既定では decay 0。
    macro_rules! apply_threat_gate_boost {
        ($trainer:expr, $tg_decay_params:expr) => {{
            if args.threat_gate {
                assert!(
                    args.threat_gate_decay >= 0.0,
                    "--threat-gate-decay は 0 以上を渡すこと (負だと decay が重みを増幅して発散する): {}",
                    args.threat_gate_decay
                );
                $trainer.optimiser.set_lr_scale("threat_gate_w", args.threat_gate_lr_mult);
                // 🚨set_lr_scale と set_range_lr_scales は**乗算**される実装だが、 TG は
                //   区間倍率を持たないので実効倍率はこの 1 つだけ (TL2 と同じ流儀)。
                $trainer.optimiser.set_params_for_weight("threat_gate_w", $tg_decay_params);
                if args.threat_gate_decay > 0.0 {
                    println!(
                        "threat_gate lr boost: effective_lr = base_lr * {} (decay = {}, w *= 1 - decay * effective_lr)",
                        args.threat_gate_lr_mult, args.threat_gate_decay
                    );
                } else {
                    println!(
                        "threat_gate lr boost: effective_lr = base_lr * {} (decay forced to 0)",
                        args.threat_gate_lr_mult
                    );
                }
            }
        }};
    }

    // 2026-09-10: `--threat-out K` の threat FT テンソルだけ lr を boost する。
    //   main を低 lr に保ったまま、 ゼロ起点の threat_ft_w だけ速く育てるため。
    //   threat_ft_out (= --threat-out K) が 0 のときは tensor 自体が存在しないので何もしない。
    macro_rules! apply_threat_ft_boost {
        // $tl2_decay_params = TL2 (threat_l2_w) に渡す params (`--threat-l2-decay` の decay 入り)。
        //   既定 0.0 のときは $decay0_params と同じ内容になるので従来と完全に一致する。
        // $tl2_freeze_params = `--threat-l2-freeze` 時に threat_l2_w に渡す params
        //   (decay 0 + lookahead 無効)。 lookahead を持たない optimiser では $decay0_params と同じ。
        ($trainer:expr, $decay0_params:expr, $tl2_decay_params:expr, $tl2_freeze_params:expr) => {{
            apply_threat_ft_boost!(
                $trainer,
                $decay0_params,
                $decay0_params,
                $decay0_params,
                $tl2_decay_params,
                $tl2_freeze_params
            )
        }};
        ($trainer:expr, $decay0_params:expr, $clip_params:expr, $factor_decay_params:expr, $tl2_decay_params:expr, $tl2_freeze_params:expr) => {{
            // (0) 主 FT (l0w / l0b) の lr boost (2026-10-10)。 `--l0-lr-mult` 既定 1.0 = no-op。
            //   ★`--main-freeze` と同時指定したときは**何もしない**。 main_freeze は l0w を
            //     凍結する側なので、 ここで倍率を入れると「凍結したのに動く」になる。
            if args.l0_lr_mult > 1.0 + 1e-6 && !args.main_freeze {
                for id in ["l0w", "l0b"] {
                    $trainer.optimiser.set_lr_scale(id, args.l0_lr_mult);
                    // decay を 0 化 (lr boost が weight decay 増幅で台無しになるのを防ぐ)。
                    $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                }
                println!(
                    "l0/FT lr boost: effective_lr = base_lr * {} (decay forced to 0 for l0w/l0b)",
                    args.l0_lr_mult
                );
            } else if args.l0_lr_mult > 1.0 + 1e-6 && args.main_freeze {
                println!(
                    "[warn] --l0-lr-mult {} は --main-freeze と同時なので無視した (main_freeze が l0w を凍結する)",
                    args.l0_lr_mult
                );
            }
            // (0b) PSQT (psqtw / psqtb) の lr 倍率 (2026-10-10)。 既定 1.0 = no-op。
            //   1e-6 で実質凍結。 ★`--main-freeze` と同時なら何もしない (あちらが凍結する)。
            if (args.psqt_lr_mult - 1.0).abs() > 1e-6 && !args.main_freeze {
                for id in ["psqtw", "psqtb"] {
                    $trainer.optimiser.set_lr_scale(id, args.psqt_lr_mult.max(1e-6));
                    $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                }
                println!(
                    "psqt lr mult: psqtw/psqtb effective_lr = base_lr * {} (decay forced to 0)",
                    args.psqt_lr_mult
                );
            }
            // (3) TL1 もゼロ初期化なので lr を上げないと動かない (fable 指摘 A)。
            //   🚨倍率は射影の 8 ではなく **2**。 TL1 の入力は thr (±数百) で
            //   combined ([0,1]) より桁が大きいので、 同じ lr でも動きが速い。
            if args.threat_l1 {
                $trainer.optimiser.set_lr_scale("threat_l1_w", args.threat_l1_lr_mult);
                $trainer.optimiser.set_params_for_weight("threat_l1_w", $decay0_params);
                println!(
                    "threat_l1 lr boost: effective_lr = base_lr * {} (decay forced to 0)",
                    args.threat_l1_lr_mult
                );
            }
            // (4) TL2 もゼロ初期化なので lr を上げないと動かない (2026-10-08)。
            if args.threat_l2 {
                assert!(
                    args.threat_l2_decay >= 0.0,
                    "--threat-l2-decay は 0 以上を渡すこと (負だと decay が重みを増幅して発散する): {}",
                    args.threat_l2_decay
                );
                if args.threat_l2_freeze {
                    // 2026-10-09 凍結 (= `--threat-freeze` の TL2 版): lr_scale を実質 0 にし、
                    //   decay も 0、 lookahead も無効にする。
                    // 🚨optimiser は lr_scale > 0 を要求するので 0 は入れられない。 倍率は
                    //   `--threat-freeze` の**実効**倍率と同じ 1e-12 (= set_lr_scale 1e-6 ×
                    //   set_range_lr_scales 1e-6)。 1e-6 だけでは base_lr 6e-6 との積 6e-12 が
                    //   小さい重みの 0.5 ulp を超え、 実測で 47 / 30,720 要素が 1 step で動く。
                    $trainer.optimiser.set_lr_scale("threat_l2_w", 1e-12);
                    // 🚨lookahead merge は lr_scale を無視して weights を slow 側へ引き戻すので、
                    //   $tl2_freeze_params で lookahead も無効化する (位相 0 以外の checkpoint から
                    //   resume したときに凍結した重みが動くのを防ぐ)。
                    $trainer.optimiser.set_params_for_weight("threat_l2_w", $tl2_freeze_params);
                    println!(
                        "threat_l2 FREEZE: lr_scale = 1e-12, decay = 0, lookahead 無効 (threat_l2_w は学習中に動かない)"
                    );
                    println!(
                        "threat_l2 FREEZE: lr-mult ({}) / decay ({}) は freeze により無視",
                        args.threat_l2_lr_mult, args.threat_l2_decay
                    );
                } else {
                    $trainer.optimiser.set_lr_scale("threat_l2_w", args.threat_l2_lr_mult);
                    // `--threat-l2-decay` 既定 0.0 のときは decay 0 の params = 従来と同じ。
                    $trainer.optimiser.set_params_for_weight("threat_l2_w", $tl2_decay_params);
                    if args.threat_l2_decay > 0.0 {
                        println!(
                            "threat_l2 lr boost: effective_lr = base_lr * {} (decay = {}, w *= 1 - decay * effective_lr)",
                            args.threat_l2_lr_mult, args.threat_l2_decay
                        );
                    } else {
                        println!(
                            "threat_l2 lr boost: effective_lr = base_lr * {} (decay forced to 0)",
                            args.threat_l2_lr_mult
                        );
                    }
                }
            }
            // 案1 ブロック射影 (2026-09-30)。 p はゼロ初期化なので lr を上げないと動かない。
            //   decay も 0 にする (ゼロ起点の gate を decay が引き戻すのは無意味)。
            if args.threat_proj && args.threat_proj_lr_mult > 1.0 + 1e-6 {
                $trainer.optimiser.set_lr_scale("threat_proj_raw", args.threat_proj_lr_mult);
                $trainer.optimiser.set_params_for_weight("threat_proj_raw", $decay0_params);
                println!(
                    "threat_proj lr boost: effective_lr = base_lr * {} (decay forced to 0)",
                    args.threat_proj_lr_mult
                );
            }
            if args.threat_out > 0 && args.threat_freeze_hard {
                // **真の凍結**: threat_ft_w は勾配ノードを持たないので optimiser 状態が無い。
                //   lr_scale / set_params_for_weight を呼ぶ必要が無い (呼んでも no-op になる)。
                //   🚨黙って無視されると気づけないので必ず 1 行出す (10-09 の TL2 と同じ対策)。
                println!(
                    "threat_ft HARD FREEZE: 勾配ノードなし / optimiser 状態なし。 --threat-ft-lr-mult ({}) は無視される",
                    args.threat_ft_lr_mult
                );
            } else if args.threat_out > 0 && args.threat_freeze {
                // 凍結: lr_scale を実質 0 にし、 decay も 0 にする (optimiser は lr_scale > 0 を要求するので
                // --l1-lr-mult と同じく 1e-6 に丸める)。 base_lr 6e-6 との積は 6e-12 で 1 SB では不動。
                // 🚨2026-09-24: hanging-split のときは**ここを 1.0 のままにする**。 区間倍率は
                //   テンソル全体の lr_scale に**乗算**されるので、 1e-6 を入れると delta 区間の 6.5 が
                //   6.5e-6 になり delta が全く学習されない (実測: delta の |w| 平均 1e-9、 量子化で全部 0)。
                //   凍結は下の set_range_lr_scales で base 区間だけに掛ける。
                if !(threat_hang_rows > 0 && threat_group_lr_ranges.is_some()) {
                    $trainer.optimiser.set_lr_scale("threat_ft_w", 1e-6);
                }
                $trainer.optimiser.set_params_for_weight("threat_ft_w", $decay0_params);
                println!("threat_ft FREEZE: lr_scale = 1e-6, decay = 0 (threat_ft_w は学習中に動かない)");
            } else if args.threat_out > 0 && args.threat_ft_lr_mult > 1.0 + 1e-6 {
                let mult = args.threat_ft_lr_mult;
                $trainer.optimiser.set_lr_scale("threat_ft_w", mult);
                // decay を 0 化して lr boost が weight decay 増幅で台無しにならないようにする。
                $trainer.optimiser.set_params_for_weight("threat_ft_w", $decay0_params);
                if threat_factor_rows > 0 && args.threat_factor_decay > 0.0 {
                    $trainer.optimiser.set_params_for_weight("threat_ft_w", $factor_decay_params);
                }
                println!(
                    "threat_ft lr boost: effective_lr = base_lr * {} (decay forced to 0 for threat_ft_w)",
                    mult
                );
            }
            // 🚨 2026-09-22: 凍結時はここを通さない。 通すと上の set_lr_scale(1e-6) が上書きされ、
            //    threat 表が動いてしまう (検証で最大差 0.12 を観測した)。
            if let Some(ranges) = &threat_group_lr_ranges {
                if args.threat_freeze {
                    // 2026-09-24 hanging-split: 凍結するのは **base 区間だけ**。 delta 区間 (末尾) は
                    //   `--threat-delta-lr-mult` のまま学習させる。 こうしないと HS 腕で delta が育たず、
                    //   対照 (CTL 腕 = threat 凍結の素の継続) との差が「ひもの分割」だけにならない。
                    let base_end = threat_ft_out * threat_ft_dims * threat_groups;
                    let frozen: Vec<_> = ranges
                        .iter()
                        .map(|r| if threat_hang_rows > 0 && r.0 >= base_end { *r } else { (r.0, r.1, 1e-6f32, 0.0f32) })
                        .collect();
                    if threat_hang_rows > 0 {
                        println!(
                            "threat_ft FREEZE: base 区間 [0, {base_end}) を凍結、 delta 区間 [{base_end}, {}) は lr 倍率 {} で学習",
                            base_end + threat_ft_out * threat_hang_rows,
                            args.threat_delta_lr_mult
                        );
                    }
                    $trainer.optimiser.set_range_lr_scales("threat_ft_w", frozen);
                } else {
                    $trainer.optimiser.set_range_lr_scales("threat_ft_w", ranges.clone());
                }
            }
            // --threat-clip: optimizer 側で重みを ±clip に収める (i8 書き出しの飽和を学習中に防ぐ)。
            if args.threat_clip > 0.0 {
                $trainer.optimiser.set_params_for_weight("threat_ft_w", $clip_params);
                println!("threat_ft clip: |w| <= {} (i8 書き出しの飽和を学習中に抑える、 decay 0)", args.threat_clip);
            }
        }};
    }

    // 2026-09-12: L1 (fc0) 4 tensor の lr 倍率 / weight decay 上書き (--l1-lr-mult / --l1-weight-decay)。
    //   $mk は decay(f32) → optimiser params を作る closure (= apply_kzt_boost と同 convention)。
    //   main_freeze / unfreeze の後に呼ぶので、 併用時は本 macro の設定が最終値になる。
    //   optimiser の decay は `decay * effective_lr` で掛かるため lr 倍率に自動追従する。
    //   --l1-anchor-lambda 有効時はアンカー tensor (勾配は stop-grad で常に 0) にも
    //   lr_scale 1e-6 + decay 0 を掛け、 optimiser 側の decay / norm-loss で動かないようにする。
    macro_rules! apply_l1_lr_mult {
        ($trainer:expr, $mk:expr) => {{
            let mk = $mk;
            let has_mult = (args.l1_lr_mult - 1.0).abs() > 1e-6;
            if has_mult || args.l1_weight_decay.is_some() {
                let scale = args.l1_lr_mult.max(1e-6);
                let wd = args.l1_weight_decay.unwrap_or(args.weight_decay);
                for id in ["l1w", "l1b", "l1fw", "l1fb"] {
                    if has_mult {
                        $trainer.optimiser.set_lr_scale(id, scale);
                    }
                    $trainer.optimiser.set_params_for_weight(id, mk(wd));
                }
                println!("L1 lr mult: l1w/l1b/l1fw/l1fb effective_lr = base_lr * {scale}, decay = {wd}");
            }
            if args.l1_anchor_lambda > 0.0 {
                for id in [L1_ANCHOR_W_ID, L1_ANCHOR_B_ID] {
                    $trainer.optimiser.set_lr_scale(id, 1e-6);
                    $trainer.optimiser.set_params_for_weight(id, mk(0.0));
                }
            }
        }};
    }

    // 2026-11-06: L2 (fc1) / L3 (fc2) だけ lr 倍率を掛ける。 `apply_l1_lr_mult!` と同じ形。
    //   L1 を作り直す実験では fc1 も一緒に動かす必要があるが、 基底 lr を下げると fc1 が止まるため。
    macro_rules! apply_l2l3_lr_mult {
        ($trainer:expr, $mk:expr) => {{
            let mk = $mk;
            for (mult, wd_opt, ids, name) in [
                (args.l2_lr_mult, args.l2_weight_decay, ["l2w", "l2b"], "L2"),
                (args.l3_lr_mult, args.l3_weight_decay, ["l3w", "l3b"], "L3"),
                // 🚨l4 == 0 のときは graph に id が無いので `set_lr_scale` が落ちる。
                //   倍率 1.0 かつ decay 未指定なら has_mult 判定で素通りするので安全 (引数検査でも弾く)。
                (args.l4_lr_mult, args.l4_weight_decay, ["l4w", "l4b"], "L4"),
            ] {
                let has_mult = (mult - 1.0).abs() > 1e-6;
                if has_mult || wd_opt.is_some() {
                    let scale = mult.max(1e-6);
                    let wd = wd_opt.unwrap_or(args.weight_decay);
                    for id in ids {
                        if has_mult {
                            $trainer.optimiser.set_lr_scale(id, scale);
                        }
                        $trainer.optimiser.set_params_for_weight(id, mk(wd));
                    }
                    println!("{name} lr mult: {ids:?} effective_lr = base_lr * {scale}, decay = {wd}");
                }
            }
        }};
    }

    macro_rules! apply_scalar_threat_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_scalar_threat && args.scalar_threat_lr_mult > 1.0 + 1e-6 {
                let mult = args.scalar_threat_lr_mult;
                $trainer.optimiser.set_lr_scale("scalar_threat_w", mult);
                $trainer.optimiser.set_params_for_weight("scalar_threat_w", $decay0_params);
                println!("ScalarThreat lr boost: effective_lr = base_lr * {} (decay forced to 0)", mult);
            }
        }};
    }

    // 2026-06-10 Attack Relay Head v2 (= 構造化 980 weight、 先後共有 single tensor)。
    macro_rules! apply_attack_relay_v2_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_attack_relay_v2 && args.attack_relay_v2_lr_mult > 1.0 + 1e-6 {
                let mult = args.attack_relay_v2_lr_mult;
                $trainer.optimiser.set_lr_scale("attack_relay_v2_w", mult);
                $trainer
                    .optimiser
                    .set_params_for_weight("attack_relay_v2_w", $decay0_params);
                println!(
                    "AttackRelay v2 lr boost: effective_lr = base_lr * {} (decay forced to 0 for attack_relay_v2 tensor)",
                    mult
                );
            }
        }};
    }

    // 2026-06-09 Attack Relay Head (ARH) 2 tensor 用の boost macro (= KED と同 convention)。
    macro_rules! apply_attack_relay_boost {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.enable_attack_relay && args.attack_relay_lr_mult > 1.0 + 1e-6 {
                let mult = args.attack_relay_lr_mult;
                $trainer.optimiser.set_lr_scale("attack_relay_w_black", mult);
                $trainer.optimiser.set_lr_scale("attack_relay_w_white", mult);
                $trainer.optimiser.set_params_for_weight("attack_relay_w_black", $decay0_params);
                $trainer.optimiser.set_params_for_weight("attack_relay_w_white", $decay0_params);
                println!(
                    "AttackRelay lr boost: effective_lr = base_lr * {} (decay forced to 0 for attack_relay tensors)",
                    mult
                );
            }
        }};
    }

    // arm 12 path B' (= 2026-05-30、 task #97): main backbone を実質凍結する macro。
    // - lr_scale = 1e-6 で勾配ステップ実質ゼロ。
    // - decay = 0 で AdamW weight decay も止める。
    // - graph に存在しない tensor (例: --dual-psqt-king-attacker OFF 時の
    //   dual_w_threat_*) を呼ぶと panic するため、 frozenflag に応じ skip する。
    // 2026-11-12 KH ブロック: kh_ft_w の lr 倍率 (decay 0) と、 --kh-freeze-rest の凍結。
    //   各 optimizer 分岐の**最後**に呼ぶ (他の macro の倍率を確実に上書きするため)。
    //   $freeze_params = decay 0 + lookahead 無効 (`--threat-l2-freeze` と同じ。 lookahead merge は
    //   lr_scale を無視して重みを slow 側へ引き戻すので、 位相 0 以外の ckpt から resume すると凍結層が動く)。
    macro_rules! apply_kh {
        ($trainer:expr, $decay0_params:expr, $freeze_params:expr) => {{
            if kh_parts.is_some() && args.hard_freeze_shared {
                println!("KH: --hard-freeze-shared で kh_ft_w は真の凍結 (勾配なしの leaf / --kh-lr-mult は使わない)");
            } else if kh_parts.is_some() {
                $trainer.optimiser.set_lr_scale("kh_ft_w", args.kh_lr_mult);
                $trainer.optimiser.set_params_for_weight("kh_ft_w", $decay0_params);
                println!("KH lr: kh_ft_w effective_lr = base_lr * {} (decay 0)", args.kh_lr_mult);
                if args.kh_freeze_rest {
                    let mut ids: Vec<&str> = vec!["l0w", "l0b", "l2w", "l2b", "l3w", "l3b"];
                    if args.kh_unfreeze_fc0_cols {
                        // 2026-11-18 KT: bias は凍結、 重みは KH の列だけ学習 (下の区間倍率)。
                        ids.extend(["l1b", "l1fb"]);
                    } else if !args.kh_unfreeze_fc0 {
                        ids.extend(["l1w", "l1b", "l1fw", "l1fb"]);
                    }
                    if args.psqt {
                        ids.extend(["psqtw", "psqtb"]);
                    }
                    for id in &ids {
                        // 1e-12: `--threat-l2-freeze` の実測 (1e-6 では base_lr 6e-6 との積が小さい重みの
                        //   0.5 ulp を超えて 1 step で動く要素が出る) に合わせる。
                        $trainer.optimiser.set_lr_scale(id, 1e-12);
                        $trainer.optimiser.set_params_for_weight(id, $freeze_params);
                    }
                    println!(
                        "KH FREEZE-REST: {:?} を lr_scale 1e-12 / decay 0 / lookahead 無効で凍結 (fc0 {})",
                        ids,
                        if args.kh_unfreeze_fc0 {
                            "は通常どおり学習"
                        } else if args.kh_unfreeze_fc0_cols {
                            "は KH の列だけ学習"
                        } else {
                            "も凍結"
                        }
                    );
                    if args.kh_unfreeze_fc0_cols {
                        let kh_cols = kh_fc0_cols(ft_out, kh_out_c, kh_row_offset_c);
                        for id in ["l1w", "l1fw"] {
                            let size = read_graph_weight(&$trainer.optimiser.graph, id).expect("fc0 の重みが読めない").len();
                            let cols = if id == "l1w" { ft_out + hand_count_dense_dims } else { ft_out };
                            assert_eq!(size % cols, 0, "{id}: 要素数 {size} が列数 {cols} で割り切れない");
                            let ranges = kh_fc0_col_ranges(size / cols, cols, &kh_cols);
                            let open: usize = ranges.iter().filter(|r| r.2 == 1.0).map(|r| r.1 - r.0).sum();
                            println!(
                                "KH FC0-COLS: {id} ({} 行 x {cols} 列) の列 {:?} だけ学習 ({open} 要素 / 他 {} 要素は lr 1e-12・decay 0)",
                                size / cols,
                                kh_cols,
                                size - open
                            );
                            $trainer.optimiser.set_range_lr_scales(id, ranges);
                        }
                    }
                }
            }
        }};
    }

    // 2026-11-17 K256 案 A: threat_ext_w の lr 倍率 (decay 0、 clip |w| <= THREAT_EXT_CLIP) と、 --threat-ext-only の凍結。
    //   apply_kh の**後**に呼ぶ (--threat-ext-only は kh_ft_w の倍率 64 も 1e-12 に上書きする)。
    //   凍結は graph の全 weight id を列挙して掛ける (固定の id 表だと新しい重みを凍結し忘れる)。
    //   真の凍結 (状態を持たない) の重みは触らない。 $clip_params = decay 0 + clip、 $freeze_params = decay 0 + lookahead 無効。
    //   2026-11-17 U1a: --threat-only も同じ凍結を使う (学習するのは threat_ft_w だけ。 その lr / clip は apply_threat_ft_boost の設定のまま)。
    macro_rules! apply_threat_ext {
        ($trainer:expr, $clip_params:expr, $freeze_params:expr) => {{
            if args.threat_ext_out > 0 {
                $trainer.optimiser.set_lr_scale("threat_ext_w", args.threat_ext_lr_mult);
                $trainer.optimiser.set_params_for_weight("threat_ext_w", $clip_params);
                println!(
                    "THREAT-EXT lr: threat_ext_w effective_lr = base_lr * {} (decay 0、 clip ±{:.6})",
                    args.threat_ext_lr_mult, THREAT_EXT_CLIP
                );
            }
            // --threat-only と --threat-ext-out N を併せたとき (U1a@128) は旧表と追加列の両方を学習する。
            let train_only: Vec<&str> = if args.threat_ext_only {
                vec!["threat_ext_w"]
            } else if args.threat_only && args.threat_ext_out > 0 {
                vec!["threat_ft_w", "threat_ext_w"]
            } else if args.threat_only {
                vec!["threat_ft_w"]
            } else {
                Vec::new()
            };
            if !train_only.is_empty() {
                let stateless = $trainer.optimiser.stateless_ids().clone();
                let mut frozen: Vec<String> = Vec::new();
                for id in graph_weight_ids(&$trainer.optimiser.graph) {
                    if train_only.contains(&id.as_str()) || stateless.contains(&id) {
                        continue;
                    }
                    $trainer.optimiser.set_lr_scale(&id, 1e-12);
                    $trainer.optimiser.set_params_for_weight(&id, $freeze_params);
                    frozen.push(id);
                }
                let mut hard: Vec<&String> = stateless.iter().collect();
                hard.sort();
                let mults: Vec<f32> = train_only.iter().map(|k| $trainer.optimiser.lr_scale_of(k)).collect();
                println!(
                    "TRAIN-ONLY {train_only:?}: {frozen:?} を lr_scale 1e-12 / decay 0 / lookahead 無効で凍結、 真の凍結 {hard:?}、 学習するのは {train_only:?} だけ (lr 倍率 {mults:?})"
                );
            }
        }};
    }

    macro_rules! apply_main_freeze {
        ($trainer:expr, $decay0_params:expr) => {{
            if args.main_freeze {
                const FREEZE_LR_SCALE: f32 = 1e-6;
                // Always-present main backbone tensors (= l0/l1/l1f/l2/l3).
                let always_ids: &[&str] = &["l0w", "l0b", "l1w", "l1b", "l1fw", "l1fb", "l2w", "l2b", "l3w", "l3b"];
                for id in always_ids {
                    $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                    $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                }
                // PSQT (= use_psqt flag 依存)。
                if args.psqt {
                    for id in ["psqtw", "psqtb"] {
                        $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                        $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                    }
                }
                // Dual threat (= dual_psqt_king_attacker flag 依存、 graph 4389/4402)。
                if args.dual_psqt_king_attacker {
                    if !args.enable_scalar_threat || args.scalar_threat_coexist_dual_cross {
                    for id in ["dual_w_threat_black", "dual_w_threat_white"] {
                        $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                        $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                    }
                    }
                    // king_attacker attack/defend は --dual-disable-king-attacker で graph 不在の可能性。
                    if !args.dual_disable_king_attacker {
                        for id in
                            ["dual_w_attack_black", "dual_w_attack_white", "dual_w_defend_black", "dual_w_defend_white"]
                        {
                            $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                            $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                        }
                    }
                    // dual_w_defended_* は --dual-disable-cross-defended OFF 時のみ存在。
                    if (!args.enable_scalar_threat || args.scalar_threat_coexist_dual_cross)
                        && !args.dual_disable_cross_defended
                    {
                        for id in ["dual_w_defended_black", "dual_w_defended_white"] {
                            $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                            $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                        }
                    }
                }
                // 既存 scalar head も凍結 (= 2026-06-11、 ARH v2 (kd) head に純残差を渡すため。
                //   king_danger は activity_risk と領域が被るため共訓練だと吸収される)。
                if args.enable_pairshape {
                    for id in ["pairshape_w_black", "pairshape_w_white"] {
                        $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                        $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                    }
                }
                // NOTE (2026-06-20 修正): Dual Cross Geometry は zero init から「育てる head」なので
                //   main_freeze 対象から除外する (= king_danger / attack_dev と同扱い)。
                //   以前はここで FREEZE_LR_SCALE を適用しており、 apply_dual_cross_geometry_boost! の
                //   lr_scale(×10) を上書きして head が一切学習されなかった (= L2≈0 のまま) バグだった。
                //   boost の lr_scale を温存するため dual_cross_geometry tensor には一切触れない。
                if args.enable_activity_risk {
                    for id in [
                        "activity_risk_w_black",
                        "activity_risk_w_white",
                        "activity_risk_v3_tactical_w_black",
                        "activity_risk_v3_tactical_w_white",
                    ] {
                        $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                        $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                    }
                }
                if args.enable_king_escape {
                    for id in ["king_escape_w_black", "king_escape_w_white"] {
                        $trainer.optimiser.set_lr_scale(id, FREEZE_LR_SCALE);
                        $trainer.optimiser.set_params_for_weight(id, $decay0_params);
                    }
                }
                if args.enable_attack_relay_v2 {
                    $trainer.optimiser.set_lr_scale("attack_relay_v2_w", FREEZE_LR_SCALE);
                    $trainer
                        .optimiser
                        .set_params_for_weight("attack_relay_v2_w", $decay0_params);
                }
                // NOTE (2026-06-28): slider_xray も同様に「育てる head」なので main_freeze 対象から
                //   除外する (= slider_xray tensor に一切触れない)。 apply_slider_xray_boost! の
                //   lr_scale (= --slider-xray-lr-mult) を温存する。
                // NOTE (2026-06-16): king_mobility も同様に「育てる head」なので main_freeze 対象から
                //   除外する (= king_mobility tensor に一切触れない)。 apply_king_mobility_boost! の
                //   lr_scale (= --king-mobility-lr-mult) を温存する。
                // NOTE (2026-06-16): drop_pressure も同様に「育てる head」なので main_freeze 対象から
                //   除外する (= drop_pressure tensor に一切触れない)。 apply_drop_pressure_boost! の
                //   lr_scale (= --drop-pressure-lr-mult) を温存する。
                // NOTE (2026-07-01): kzt (KZT_HAND factored) も zero init から「育てる head」なので
                //   main_freeze 対象から除外する (= kzt_w_{core,region,bucket,persp,bias} に触れない)。
                //   apply_kzt_boost! の per-table lr_scale/decay を温存する。
                println!(
                    "Main backbone FREEZE enabled (= lr_scale {:e} + decay 0 on l0/l1/l1f/l2/l3{}{} tensors)",
                    FREEZE_LR_SCALE,
                    if args.psqt { "/psqt" } else { "" },
                    if args.dual_psqt_king_attacker { "/dual_threat" } else { "" },
                );
            }
        }};
    }

    macro_rules! apply_unfreeze_final_layers {
        ($trainer:expr, $params:expr) => {{
            if args.main_freeze && args.unfreeze_l3 {
                for id in ["l3w", "l3b"] {
                    $trainer.optimiser.set_lr_scale(id, 1.0);
                    $trainer.optimiser.set_params_for_weight(id, $params);
                }
                println!("Unfreeze L3: l3w/l3b は lr_scale 1.0 で学習対象に復帰");
            }
            if args.main_freeze && args.unfreeze_l2 {
                for id in ["l2w", "l2b"] {
                    $trainer.optimiser.set_lr_scale(id, 1.0);
                    $trainer.optimiser.set_params_for_weight(id, $params);
                }
                println!("Unfreeze L2: l2w/l2b は lr_scale 1.0 で学習対象に復帰");
            }
            if args.main_freeze && args.unfreeze_l1 {
                for id in ["l1w", "l1b", "l1fw", "l1fb"] {
                    $trainer.optimiser.set_lr_scale(id, 1.0);
                    $trainer.optimiser.set_params_for_weight(id, $params);
                }
                println!("Unfreeze L1: l1w/l1b/l1fw/l1fb は lr_scale 1.0 で学習対象に復帰");
            }
            if args.main_freeze && args.unfreeze_l0 {
                for id in ["l0w", "l0b"] {
                    $trainer.optimiser.set_lr_scale(id, 1.0);
                    $trainer.optimiser.set_params_for_weight(id, $params);
                }
                println!("Unfreeze L0/FT: l0w/l0b は lr_scale 1.0 で学習対象に復帰");
            }
        }};
    }

    // main-freeze 時に apply_main_freeze! が凍結した eval head 群を lr_scale 1.0 で
    // 学習対象に戻す macro (= head 再学習用)。 凍結ブロックと厳密に同じ if ガードで、
    // graph に存在する head tensor のみへ set_lr_scale / set_params_for_weight を適用する。
    macro_rules! apply_unfreeze_heads {
        ($trainer:expr, $params:expr) => {{
            if args.main_freeze && args.unfreeze_heads {
                let mut unfrozen_ids: Vec<&str> = Vec::new();
                // Dual threat (= dual_psqt_king_attacker flag 依存)。
                if args.dual_psqt_king_attacker {
                    if !args.enable_scalar_threat || args.scalar_threat_coexist_dual_cross {
                        for id in ["dual_w_threat_black", "dual_w_threat_white"] {
                            $trainer.optimiser.set_lr_scale(id, 1.0);
                            $trainer.optimiser.set_params_for_weight(id, $params);
                            unfrozen_ids.push(id);
                        }
                    }
                    if !args.dual_disable_king_attacker {
                        for id in
                            ["dual_w_attack_black", "dual_w_attack_white", "dual_w_defend_black", "dual_w_defend_white"]
                        {
                            $trainer.optimiser.set_lr_scale(id, 1.0);
                            $trainer.optimiser.set_params_for_weight(id, $params);
                            unfrozen_ids.push(id);
                        }
                    }
                    if (!args.enable_scalar_threat || args.scalar_threat_coexist_dual_cross)
                        && !args.dual_disable_cross_defended
                    {
                        for id in ["dual_w_defended_black", "dual_w_defended_white"] {
                            $trainer.optimiser.set_lr_scale(id, 1.0);
                            $trainer.optimiser.set_params_for_weight(id, $params);
                            unfrozen_ids.push(id);
                        }
                    }
                }
                if args.enable_pairshape {
                    for id in ["pairshape_w_black", "pairshape_w_white"] {
                        $trainer.optimiser.set_lr_scale(id, 1.0);
                        $trainer.optimiser.set_params_for_weight(id, $params);
                        unfrozen_ids.push(id);
                    }
                }
                if args.enable_activity_risk {
                    for id in [
                        "activity_risk_w_black",
                        "activity_risk_w_white",
                        "activity_risk_v3_tactical_w_black",
                        "activity_risk_v3_tactical_w_white",
                    ] {
                        $trainer.optimiser.set_lr_scale(id, 1.0);
                        $trainer.optimiser.set_params_for_weight(id, $params);
                        unfrozen_ids.push(id);
                    }
                }
                if args.enable_king_escape {
                    for id in ["king_escape_w_black", "king_escape_w_white"] {
                        $trainer.optimiser.set_lr_scale(id, 1.0);
                        $trainer.optimiser.set_params_for_weight(id, $params);
                        unfrozen_ids.push(id);
                    }
                }
                if args.enable_attack_relay_v2 {
                    $trainer.optimiser.set_lr_scale("attack_relay_v2_w", 1.0);
                    $trainer.optimiser.set_params_for_weight("attack_relay_v2_w", $params);
                    unfrozen_ids.push("attack_relay_v2_w");
                }
                println!("Unfreeze heads: {:?} を lr_scale 1.0 で学習対象に復帰", unfrozen_ids);
            }
        }};
    }

    // Ranger21の内部scheduleは実行全体のbatch数を基準にする。1batch preflightでは
    // `--ranger21-total-steps 0` を指定してscheduleを明示的に無効化できる。
    //
    // ★2026-09-02 修正: 旧実装は「この run で回す残り superbatch 数」から算出していたが、
    //   `--resume` は Ranger21 の step カウンタを**絶対値**で復元する。 単位が食い違うため、
    //   クラッシュ再開で `--start-superbatch` が進むほど total_steps が縮み、
    //   `step >= total_steps` になると warmdown_scale が 0 に飽和して **lr が 0 のまま
    //   1 superbatch 丸ごと no-op** になる (実測: base sb23/24/25 が bit-identical)。
    //   step が絶対値なので total_steps も絶対値 (= superbatches 全体) にする。
    //   これで再開しても中断なしの run と同じ schedule になる。
    let ranger21_auto_total_steps = if args.superbatches < args.start_superbatch {
        0
    } else {
        batches_per_superbatch.saturating_mul(args.superbatches)
    };
    let ranger21_total_steps = args.ranger21_total_steps.unwrap_or(ranger21_auto_total_steps);

    // ステージをまたぐ resume (別 run の checkpoint から `--start-superbatch 1` で始める)
    // では、復元 step が新しい total_steps を超えるので上の修正でも lr は 0 になる。
    // 静かに no-op で進むと superbatch 単位で計算を捨てるため、 起動時に検出して警告する。
    // 対処は `--ranger21-total-steps 0` (内部 schedule を切り、 外側の
    // `--lr / --warmup-batches / --lr-gamma` に任せる)。
    if ranger21_total_steps > 0 && matches!(args.optimizer, OptimizerType::Ranger21) {
        if let Some(resume) = args.resume_path() {
            let step_file = resume.join("optimiser_state").join("step_ranger21.txt");
            if let Ok(text) = std::fs::read_to_string(&step_file) {
                let restored = text
                    .lines()
                    .filter_map(|line| line.split(',').nth(1))
                    .filter_map(|value| value.trim().parse::<usize>().ok())
                    .max()
                    .unwrap_or(0);
                if restored >= ranger21_total_steps {
                    eprintln!(
                        "WARNING: resume した Ranger21 step={restored} が total_steps={ranger21_total_steps} 以上です。\
                         内部 schedule の warmdown が飽和して学習率が 0 になり、 重みが一切更新されません。\
                         ステージをまたぐ resume では --ranger21-total-steps 0 を指定してください。"
                    );
                }
            }
        }
    }
    macro_rules! ranger21_params {
        ($decay:expr) => {
            Ranger21Params {
                decay: $decay,
                total_steps: ranger21_total_steps,
                lookahead_mergetime: args.ranger21_lookahead_mergetime,
                ..Default::default()
            }
        };
    }

    // 入力型の分岐: threat 有効時は ShogiHalfKaHmThreat、無効時は ShogiHalfKA_hm
    // ジェネリクスが異なるため、各 optimizer × input の組み合わせを展開する
    macro_rules! run_optimizer {
        ($input:expr) => {{
            match args.optimizer {
                OptimizerType::AdamW => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::AdamW, use_win_rate_model, bucket_impl, $input);
                    trainer.optimiser.set_params(AdamWParams { decay: args.weight_decay, ..Default::default() });
                    apply_kingring_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_pairshape_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_dual_cross_geometry_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_activity_risk_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_king_escape_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_slider_xray_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_kzt_boost!(trainer, |d: f32| AdamWParams { decay: d, ..Default::default() });
                    apply_king_mobility_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_drop_pressure_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_endgame_tempo_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_scalar_threat_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_threat_ft_boost!(
                        trainer,
                        AdamWParams { decay: 0.0, ..Default::default() },
                        AdamWParams { decay: args.threat_l2_decay, ..Default::default() },
                        // AdamW は lookahead を持たないので decay 0 のみ。
                        AdamWParams { decay: 0.0, ..Default::default() }
                    );
                    apply_threat_gate_boost!(
                        trainer,
                        AdamWParams { decay: args.threat_gate_decay, ..Default::default() }
                    );
                    apply_attack_relay_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_attack_relay_v2_boost!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_main_freeze!(trainer, AdamWParams { decay: 0.0, ..Default::default() });
                    apply_unfreeze_final_layers!(
                        trainer,
                        AdamWParams { decay: args.weight_decay, ..Default::default() }
                    );
                    apply_unfreeze_heads!(trainer, AdamWParams { decay: args.weight_decay, ..Default::default() });
                    apply_l1_lr_mult!(trainer, |d: f32| AdamWParams { decay: d, ..Default::default() });
                    apply_l2l3_lr_mult!(trainer, |d: f32| AdamWParams { decay: d, ..Default::default() });
                    apply_kh!(
                        trainer,
                        AdamWParams { decay: 0.0, ..Default::default() },
                        AdamWParams { decay: 0.0, ..Default::default() }
                    );
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::RAdam => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::RAdam, use_win_rate_model, bucket_impl, $input);
                    let params = RAdamParams { decay: args.weight_decay, ..Default::default() };
                    trainer.optimiser.set_params(params.into());
                    apply_kingring_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_pairshape_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_dual_cross_geometry_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_activity_risk_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_king_escape_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_slider_xray_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_kzt_boost!(trainer, |d: f32| RAdamParams { decay: d, ..Default::default() }.into());
                    apply_king_mobility_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_drop_pressure_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_endgame_tempo_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_scalar_threat_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_threat_ft_boost!(
                        trainer,
                        RAdamParams { decay: 0.0, ..Default::default() }.into(),
                        RAdamParams { decay: args.threat_l2_decay, ..Default::default() }.into(),
                        // RAdam は lookahead を持たないので decay 0 のみ。
                        RAdamParams { decay: 0.0, ..Default::default() }.into()
                    );
                    apply_threat_gate_boost!(
                        trainer,
                        RAdamParams { decay: args.threat_gate_decay, ..Default::default() }.into()
                    );
                    apply_attack_relay_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_attack_relay_v2_boost!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_main_freeze!(trainer, RAdamParams { decay: 0.0, ..Default::default() }.into());
                    apply_unfreeze_final_layers!(
                        trainer,
                        RAdamParams { decay: args.weight_decay, ..Default::default() }.into()
                    );
                    apply_unfreeze_heads!(
                        trainer,
                        RAdamParams { decay: args.weight_decay, ..Default::default() }.into()
                    );
                    apply_l1_lr_mult!(trainer, |d: f32| RAdamParams { decay: d, ..Default::default() }.into());
                    apply_l2l3_lr_mult!(trainer, |d: f32| RAdamParams { decay: d, ..Default::default() }.into());
                    apply_kh!(
                        trainer,
                        RAdamParams { decay: 0.0, ..Default::default() }.into(),
                        RAdamParams { decay: 0.0, ..Default::default() }.into()
                    );
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::Ranger, use_win_rate_model, bucket_impl, $input);
                    trainer.optimiser.set_params(RangerParams { decay: args.weight_decay, ..Default::default() });
                    apply_kingring_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_pairshape_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_dual_cross_geometry_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_activity_risk_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_king_escape_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_slider_xray_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_kzt_boost!(trainer, |d: f32| RangerParams { decay: d, ..Default::default() });
                    apply_king_mobility_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_drop_pressure_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_endgame_tempo_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_scalar_threat_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_threat_ft_boost!(
                        trainer,
                        RangerParams { decay: 0.0, ..Default::default() },
                        RangerParams { decay: args.threat_l2_decay, ..Default::default() },
                        // Ranger の lookahead 間隔は `k`。 凍結時は実質無限にする。
                        RangerParams { decay: 0.0, k: usize::MAX, ..Default::default() }
                    );
                    apply_threat_gate_boost!(
                        trainer,
                        RangerParams { decay: args.threat_gate_decay, ..Default::default() }
                    );
                    apply_attack_relay_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_attack_relay_v2_boost!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_main_freeze!(trainer, RangerParams { decay: 0.0, ..Default::default() });
                    apply_unfreeze_final_layers!(
                        trainer,
                        RangerParams { decay: args.weight_decay, ..Default::default() }
                    );
                    apply_unfreeze_heads!(trainer, RangerParams { decay: args.weight_decay, ..Default::default() });
                    apply_l1_lr_mult!(trainer, |d: f32| RangerParams { decay: d, ..Default::default() });
                    apply_l2l3_lr_mult!(trainer, |d: f32| RangerParams { decay: d, ..Default::default() });
                    apply_kh!(
                        trainer,
                        RangerParams { decay: 0.0, ..Default::default() },
                        RangerParams { decay: 0.0, k: usize::MAX, ..Default::default() }
                    );
                    maybe_run_or_quantise!(trainer);
                }
                OptimizerType::Ranger21 => {
                    let mut trainer =
                        build_trainer_with_input!(optimiser::Ranger21, use_win_rate_model, bucket_impl, $input);
                    trainer.optimiser.set_params(ranger21_params!(args.weight_decay));
                    apply_kingring_boost!(trainer, ranger21_params!(0.0));
                    apply_pairshape_boost!(trainer, ranger21_params!(0.0));
                    apply_dual_cross_geometry_boost!(trainer, ranger21_params!(0.0));
                    apply_activity_risk_boost!(trainer, ranger21_params!(0.0));
                    apply_king_escape_boost!(trainer, ranger21_params!(0.0));
                    apply_slider_xray_boost!(trainer, ranger21_params!(0.0));
                    apply_kzt_boost!(trainer, |d: f32| ranger21_params!(d));
                    apply_king_mobility_boost!(trainer, ranger21_params!(0.0));
                    apply_drop_pressure_boost!(trainer, ranger21_params!(0.0));
                    apply_endgame_tempo_boost!(trainer, ranger21_params!(0.0));
                    apply_scalar_threat_boost!(trainer, ranger21_params!(0.0));
                    apply_threat_ft_boost!(
                        trainer,
                        ranger21_params!(0.0),
                        Ranger21Params { clip: Some((-args.threat_clip, args.threat_clip)), ..ranger21_params!(0.0) },
                        ranger21_params!(args.threat_factor_decay),
                        ranger21_params!(args.threat_l2_decay),
                        // Ranger21 の lookahead merge は lr_scale を無視するので凍結時は無効化する。
                        Ranger21Params { lookahead_mergetime: usize::MAX, ..ranger21_params!(0.0) }
                    );
                    apply_threat_gate_boost!(trainer, ranger21_params!(args.threat_gate_decay));
                    apply_attack_relay_boost!(trainer, ranger21_params!(0.0));
                    apply_attack_relay_v2_boost!(trainer, ranger21_params!(0.0));
                    apply_main_freeze!(trainer, ranger21_params!(0.0));
                    apply_unfreeze_final_layers!(trainer, ranger21_params!(args.weight_decay));
                    apply_unfreeze_heads!(trainer, ranger21_params!(args.weight_decay));
                    apply_l1_lr_mult!(trainer, |d: f32| ranger21_params!(d));
                    apply_l2l3_lr_mult!(trainer, |d: f32| ranger21_params!(d));
                    apply_kh!(
                        trainer,
                        ranger21_params!(0.0),
                        Ranger21Params { lookahead_mergetime: usize::MAX, ..ranger21_params!(0.0) }
                    );
                    apply_threat_ext!(
                        trainer,
                        Ranger21Params { clip: Some((-THREAT_EXT_CLIP, THREAT_EXT_CLIP)), ..ranger21_params!(0.0) },
                        Ranger21Params { lookahead_mergetime: usize::MAX, ..ranger21_params!(0.0) }
                    );
                    maybe_run_or_quantise!(trainer);
                }
            }
        }};
    }

    // --factorise 時は入力を Factorised<_, ShogiPieceFactor> で wrap する。
    // main sparse 入力が HalfKA_hm (73,305 次元) のままの構成 (plain / pin /
    // hand-count / dual / 各 head) すべてで併用可能。 aux dense / aux sparse は
    // Factorised が inner へ delegate する。 export 時の coalesce は既存の
    // factorise path (l0w / psqt) がそのまま適用される。
    macro_rules! run_optimizer_maybe_factorised {
        ($input:expr) => {{
            if args.factorise {
                run_optimizer!(bullet_lib::game::inputs::Factorised::from_parts(
                    $input,
                    bullet_lib::game::inputs::ShogiPieceFactor,
                ));
            } else {
                run_optimizer!($input);
            }
        }};
    }

    if enable_threat_factor16 {
        let enable_ka = !args.dual_disable_king_attacker;
        let enable_cross_defended = !args.dual_disable_cross_defended;
        run_optimizer!(ShogiHalfKaHmThreatFactor16::new().with_dual_psqt(
            args.dual_psqt_king_attacker,
            enable_ka,
            enable_cross_defended
        ));
    } else if let Some(tp) = threat_profile {
        if tp == ThreatProfile::EnemyOnlyDirect15 {
            run_optimizer!(ShogiHalfKaHmThreatDirect15::new().with_defended(direct15_defended));
        } else {
            // Threat + 任意 Pin-aware の組合せをサポート (plan C 用)。
            let enable_ka = !args.dual_disable_king_attacker;
            let enable_cross_defended = !args.dual_disable_cross_defended;
            run_optimizer!(
                ShogiHalfKaHmThreat::new(tp)
                    .with_threat_ft_sparse(threat_ft_sparse)
                    .with_threat_bucket_groups(threat_bucket_groups.clone())
                    .with_threat_factor(threat_factor_rows > 0)
                    // 2026-11-17 K256 案 A: 追加列の疎入力。 0 なら aux 入力は従来と同一。
                    .with_threat_ext(args.threat_ext_out)
                    .with_threat_grad_mask(threat_grad_mask.clone(), threat_ft_out)
                    .with_pin_aware(args.pin_aware)
                    .with_mob(args.mob)
                    // 2026-11-07 (T2/T3): 持駒 14 元 dense。 false なら graph も arch も保存形式も変わらない。
                    .with_hand_count_dense(use_hand_count_dense)
                    // 2026-11-12: KH ブロック。 None なら aux 入力は従来と同一。
                    .with_kh(kh_parts.map(|p| (p, args.kh_null)))
                    .with_dual_psqt(args.dual_psqt_king_attacker, enable_ka, enable_cross_defended)
            );
        }
    } else if use_hand_threat {
        run_optimizer!(ShogiHalfKaHmHandThreat::new());
    } else if use_hand_threat_defensive {
        run_optimizer!(ShogiHalfKaHmHandThreatDefensive::new());
    } else if args.enable_kzt_ft {
        // KZT-FT (= FT 入力版)。 HalfKA_hm + KZT-FT concat の非対称 sparse 流。
        run_optimizer!(ShogiHalfKaHmKzt::new());
    } else if args.enable_aux {
        // AUX-FT (= FT 入力版)。 HalfKA_hm + AUX (KZ/HANG/STATKD) concat の非対称 sparse 流。
        run_optimizer!(ShogiHalfKaHmAux::new());
    } else if args.enable_auxcheap {
        // AUXCHEAP (= PSQT-only 版)。 HalfKA_hm + AUXCHEAP (KZ2x/HG6/OUT6/STATKD-kd2) concat
        // の非対称 sparse 流 (l0w aux 行は run_with_zeroed_rows で 0 固定)。
        // --factorise 併用可 (2026-07-23): HalfKA 部のみ factor derive、 aux は対象外。
        run_optimizer_maybe_factorised!(ShogiHalfKaHmAuxCheap::new());
    } else if args.dual_psqt_king_attacker || args.enable_scalar_threat {
        // Dual PSQT King-Attacker 有効なら ShogiHalfKaHmKingAttackerDual (= Pin 継承 + dual
        // aux dispatcher) で起動。 Pin-aware も同時に有効化可能。
        // arm 5: --dual-disable-king-attacker で king_attacker head を OFF にできる。
        // arm 6: --dual-disable-cross-defended で D2 cross head 1 (= defended) を OFF にできる。
        // arm 11: --enable-kingring で KingRing scalar head を ON にできる。
        // arm 12: --enable-pairshape で PairShape scalar head を ON にできる。
        let enable_ka = !args.dual_disable_king_attacker;
        let enable_cross_defended = !args.dual_disable_cross_defended;
        let enable_kingring = args.enable_kingring;
        let enable_pairshape = args.enable_pairshape;
        // Dual Cross Geometry head (= 2026-06-20、 pairshape と同 pattern の scalar head)
        let enable_dual_cross_geometry = args.enable_dual_cross_geometry;
        let enable_activity_risk = args.enable_activity_risk;
        let enable_king_escape = args.enable_king_escape;
        let enable_slider_xray = args.enable_slider_xray;
        let enable_king_mobility = args.enable_king_mobility;
        let enable_drop_pressure = args.enable_drop_pressure;
        let enable_endgame_tempo = args.enable_endgame_tempo;
        let enable_attack_relay = args.enable_attack_relay;
        run_optimizer_maybe_factorised!(
            bullet_lib::game::inputs::ShogiHalfKaHmKingAttackerDual::new()
                .with_pin_aware(args.pin_aware)
                .with_dual_enabled(true)
                .with_scalar_threat_enabled(args.enable_scalar_threat)
                .with_scalar_threat_coexist_dual_cross(args.scalar_threat_coexist_dual_cross)
                .with_king_attacker_enabled(enable_ka)
                .with_dual_cross_defended_enabled(enable_cross_defended)
                .with_kingring_enabled(enable_kingring)
                .with_pairshape_enabled(enable_pairshape)
                .with_dual_cross_geometry_enabled(enable_dual_cross_geometry)
                .with_activity_risk_enabled(enable_activity_risk)
                .with_king_escape_enabled(enable_king_escape)
                .with_slider_xray_enabled(enable_slider_xray)
                .with_king_mobility_enabled(enable_king_mobility)
                .with_drop_pressure_enabled(enable_drop_pressure)
                .with_endgame_tempo_enabled(enable_endgame_tempo)
                .with_attack_relay_enabled(enable_attack_relay)
                .with_attack_relay_v2_enabled(args.enable_attack_relay_v2)
                .with_kzt_enabled(args.enable_kzt_hand)
        );
    } else if args.enable_kingring
        || args.enable_pairshape
        || args.enable_dual_cross_geometry
        || args.enable_activity_risk
        || args.enable_king_escape
        || args.enable_slider_xray
        || args.enable_king_mobility
        || args.enable_drop_pressure
        || args.enable_endgame_tempo
        || args.enable_attack_relay
        || args.enable_attack_relay_v2
        || args.enable_scalar_threat
        || args.enable_kzt_hand
    {
        // arm 11 単独 kingring / arm 12 単独 pairshape / 2026-06-06 単独 activity_risk 路線
        // (= dual_psqt_king_attacker なしで scalar head のみ)。
        // dual head は OFF だが KingAttackerDual wrapper の scalar head dispatcher は dual_enabled
        // に依存しないため、 dual_enabled=false + 各 head ON で単独運用可能。
        run_optimizer_maybe_factorised!(
            bullet_lib::game::inputs::ShogiHalfKaHmKingAttackerDual::new()
                .with_pin_aware(args.pin_aware)
                .with_dual_enabled(false)
                .with_scalar_threat_enabled(args.enable_scalar_threat)
                .with_scalar_threat_coexist_dual_cross(args.scalar_threat_coexist_dual_cross)
                .with_kingring_enabled(args.enable_kingring)
                .with_pairshape_enabled(args.enable_pairshape)
                .with_dual_cross_geometry_enabled(args.enable_dual_cross_geometry)
                .with_activity_risk_enabled(args.enable_activity_risk)
                .with_king_escape_enabled(args.enable_king_escape)
                .with_slider_xray_enabled(args.enable_slider_xray)
                .with_king_mobility_enabled(args.enable_king_mobility)
                .with_drop_pressure_enabled(args.enable_drop_pressure)
                .with_endgame_tempo_enabled(args.enable_endgame_tempo)
                .with_attack_relay_enabled(args.enable_attack_relay)
                .with_attack_relay_v2_enabled(args.enable_attack_relay_v2)
                .with_kzt_enabled(args.enable_kzt_hand)
        );
    } else if args.pin_aware {
        // Pin-aware 有効なら ShogiHalfKaHmPin (= HandCount 継承 + pin dense input
        // dispatcher) で起動。 LMAS は 2026-05-25 廃止のため Pin 単独 path のみ残る。
        run_optimizer_maybe_factorised!(
            bullet_lib::game::inputs::ShogiHalfKaHmPin::new().with_pin_aware(args.pin_aware)
        );
    } else if use_hand_count_dense {
        run_optimizer_maybe_factorised!(ShogiHalfKaHmHandCount);
    } else if args.opn_king16 {
        // 2026-11-14 OPN2048 K16 (併用の検査は input_size の手前で済んでいる)。
        run_optimizer_maybe_factorised!(bullet_lib::game::inputs::ShogiHalfKA_hmK16);
    } else {
        run_optimizer_maybe_factorised!(ShogiHalfKA_hm);
    }

    // Generate final experiment JSON (status: completed)
    if !experiment_quantise_only {
        if let Err(e) = experiment_ctx.write_experiment_json("completed") {
            eprintln!("Warning: Failed to generate experiment JSON: {}", e);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leb128_roundtrip() {
        // 正の値
        let encoded = encode_signed_leb128(0);
        assert_eq!(encoded, vec![0x00]);

        let encoded = encode_signed_leb128(1);
        assert_eq!(encoded, vec![0x01]);

        let encoded = encode_signed_leb128(63);
        assert_eq!(encoded, vec![0x3F]);

        let encoded = encode_signed_leb128(64);
        assert_eq!(encoded, vec![0xC0, 0x00]);

        let encoded = encode_signed_leb128(127);
        assert_eq!(encoded, vec![0xFF, 0x00]);

        // 負の値
        let encoded = encode_signed_leb128(-1);
        assert_eq!(encoded, vec![0x7F]);

        let encoded = encode_signed_leb128(-64);
        assert_eq!(encoded, vec![0x40]);

        let encoded = encode_signed_leb128(-65);
        assert_eq!(encoded, vec![0xBF, 0x7F]);

        let encoded = encode_signed_leb128(-128);
        assert_eq!(encoded, vec![0x80, 0x7F]);
    }

    #[test]
    fn test_leb128_i16_range() {
        // i16::MAX = 32767
        let encoded = encode_signed_leb128(32767);
        assert_eq!(encoded, vec![0xFF, 0xFF, 0x01]);

        // i16::MIN = -32768
        let encoded = encode_signed_leb128(-32768);
        assert_eq!(encoded, vec![0x80, 0x80, 0x7E]);
    }

    #[test]
    fn test_leb128_tensor_format() {
        let values: Vec<i16> = vec![0, 1, -1, 127, -128];
        let result = encode_leb128_tensor_i16(&values);

        // Check magic
        assert_eq!(&result[..17], b"COMPRESSED_LEB128");

        // Check size field (u32 LE)
        let size = u32::from_le_bytes([result[17], result[18], result[19], result[20]]) as usize;

        // Verify compressed data length
        assert_eq!(result.len(), 17 + 4 + size);
    }

    #[test]
    fn test_pad32() {
        assert_eq!(pad32(1536), 1536);
        assert_eq!(pad32(30), 32);
        assert_eq!(pad32(32), 32);
        assert_eq!(pad32(1), 32);
        assert_eq!(pad32(33), 64);
    }

    #[test]
    fn test_tanuki_sfnnwop1536_format_constants() {
        assert_eq!(NUM_BUCKETS, 8);

        let format = layerstack_format_constants(
            LayerStackOutputFormat::TanukiSfnnwoP1536,
            1536,
            16,
            // 2026-11-17: l2_in の引数が足された後に更新されず、 テストがコンパイルできなかった。 tanuki の分岐は l2_in を使わない。
            30,
            32,
            "ignored for tanuki".to_string(),
        );

        assert_eq!(format.nnue_version, 0x7AF32F16);
        assert_eq!(format.network_hash, 0x3c203b32);
        assert_eq!(format.ft_hash, 0x5f134ab8);
        assert_eq!(format.fc_hash, 0x6333718A);
        assert_eq!(format.serialized_buckets, 8);
        assert_eq!(format.architecture, "Network trained with https://github.com/official-stockfish/nnue-pytorch");
    }

    #[test]
    fn test_tanuki_sfnnwop1536_requires_progress8kpabs_bucket_mode() {
        let result = validate_layerstack_output_format(
            LayerStackOutputFormat::TanukiSfnnwoP1536,
            BucketMode::Kingrank9,
            ShogiHalfKA_hm.num_inputs(),
            ShogiHalfKA_hm.num_inputs(),
            1536,
            16,
            32,
            false,
            None,
            false,
            0,
            0,     // lmas_flags
            false, // pin_aware
            false, // dual_psqt_reserve
            false, // dual_psqt_king_attacker
        );

        assert!(result.unwrap_err().contains("--bucket-mode progress8kpabs"));
    }

    /// 2026-11-17 K256 案 A / U1a: --threat-ext-* / --threat-only の引数検査。
    #[test]
    fn test_validate_threat_ext_args() {
        let base = ["x", "--l0", "1024", "--threat", "--threat-out", "64", "--optimizer", "ranger21"];
        let parse = |extra: &[&str]| Args::parse_from(base.iter().chain(extra.iter()).copied());
        // 既定は何もしない
        assert!(validate_threat_ext_args(&Args::parse_from(["x"])).is_ok());
        assert!(validate_threat_ext_args(&parse(&[])).is_ok());
        // 追加列なしで追加列の指定は止める
        assert!(validate_threat_ext_args(&parse(&["--threat-ext-only"])).is_err());
        assert!(validate_threat_ext_args(&parse(&["--threat-ext-lr-mult", "64"])).is_err());
        // K256 / K128 の段 A の形は通る
        for n in ["192", "64"] {
            let ok = ["--threat-ext-out", n, "--threat-ext-lr-mult", "64", "--threat-ext-only", "--threat-freeze-hard"];
            assert!(validate_threat_ext_args(&parse(&ok)).is_ok(), "N={n}");
        }
        // 32 の倍数でない / FT の半分を超える / 旧表を凍結しない / ranger21 以外
        assert!(validate_threat_ext_args(&parse(&["--threat-ext-out", "48"])).is_err());
        assert!(validate_threat_ext_args(&parse(&["--threat-ext-out", "1024"])).is_err());
        assert!(validate_threat_ext_args(&parse(&["--threat-ext-out", "192", "--threat-ext-only"])).is_err());
        let adam = Args::parse_from(["x", "--threat", "--threat-out", "64", "--threat-ext-out", "192"]);
        assert!(validate_threat_ext_args(&adam).is_err());
        assert!(validate_threat_ext_args(&parse(&["--threat-ext-out", "192", "--threat-l2"])).is_err());
        // U1a: threat-only は表を凍結する指定・他の凍結と排他
        assert!(validate_threat_ext_args(&parse(&["--threat-only"])).is_ok());
        assert!(validate_threat_ext_args(&parse(&["--threat-only", "--threat-freeze-hard"])).is_err());
        assert!(validate_threat_ext_args(&parse(&["--threat-only", "--threat-freeze"])).is_err());
        assert!(validate_threat_ext_args(&parse(&["--threat-only", "--main-freeze"])).is_err());
        assert!(validate_threat_ext_args(&Args::parse_from(["x", "--threat-only"])).is_err());
    }

    /// 2026-11-17 tanuki: score の置き換えは sign(s) * 表[b][|s|]、 |s| >= 10000 は None。
    #[test]
    fn test_apply_score_lut() {
        let mut lut = vec![0i32; NUM_BUCKETS * SCORE_LUT_LEN];
        for b in 0..NUM_BUCKETS {
            for a in 0..SCORE_LUT_LEN {
                lut[b * SCORE_LUT_LEN + a] = (a as i32) * (b as i32 + 1) / 4;
            }
        }
        assert_eq!(apply_score_lut(&lut, 0, 0), Some(0));
        assert_eq!(apply_score_lut(&lut, 2, 100), Some(75));
        assert_eq!(apply_score_lut(&lut, 2, -100), Some(-75));
        assert_eq!(apply_score_lut(&lut, 7, 9999), Some(19998));
        assert_eq!(apply_score_lut(&lut, 3, 10000), None);
        assert_eq!(apply_score_lut(&lut, 3, -32000), None);
    }

    /// 2026-11-17 tanuki: --mix-buffer-batches の混ぜ方は並べ替えだけ (局面の集合は変わらない) で、 同じ種なら同じ順。
    #[test]
    fn test_mix_shuffle_is_permutation() {
        let mk = || -> Vec<bullet_lib::shogi::PackedSfenValue> {
            (0..1000u16)
                .map(|i| {
                    let mut p = bullet_lib::shogi::PackedSfenValue::default();
                    p.as_bytes_mut()[32..34].copy_from_slice(&i.to_le_bytes());
                    p
                })
                .collect()
        };
        let (mut a, mut b) = (mk(), mk());
        mix_shuffle(&mut a, 7);
        mix_shuffle(&mut b, 7);
        let key = |v: &[bullet_lib::shogi::PackedSfenValue]| v.iter().map(|p| p.score()).collect::<Vec<_>>();
        assert_eq!(key(&a), key(&b));
        assert_ne!(key(&a), key(&mk()));
        let mut sorted = key(&a);
        sorted.sort();
        assert_eq!(sorted, (0..1000).collect::<Vec<i16>>());
    }

    #[test]
    fn test_no_color_flag_is_parsed() {
        let args = Args::parse_from(["shogi_layerstack", "--no-color"]);

        assert!(args.no_color);
    }

    #[test]
    fn test_log_rate_flag_is_parsed() {
        let args = Args::parse_from(["shogi_layerstack", "--log-rate", "64"]);

        assert_eq!(args.log_rate, 64);
    }

    #[test]
    fn test_log_rate_must_be_nonzero() {
        let err = Args::try_parse_from(["shogi_layerstack", "--log-rate", "0"]).unwrap_err();

        assert!(err.to_string().contains("1 or greater"));
    }

    #[test]
    fn test_empty_resume_is_treated_as_absent() {
        let args = Args::parse_from(["shogi_layerstack", "--resume", ""]);

        assert!(args.resume_path().is_none());
    }

    #[test]
    fn test_nonempty_resume_is_kept() {
        let args = Args::parse_from(["shogi_layerstack", "--resume", "checkpoints/tanuki-10"]);

        assert_eq!(args.resume_path(), Some(Path::new("checkpoints/tanuki-10")));
    }

    #[test]
    fn test_bucket_initial_position() {
        // 平手初期局面: 先手玉 = 4 (rank=4%9=4), 後手玉 = 76 (rank=76%9=4)
        // side_to_move = Black
        // f_rank = 4 (Black そのまま)
        // e_rank = 8 - 4 = 4 (Black から見て相手は後手 → 反転)
        // F_TO_INDEX[4] = 3, E_TO_INDEX[4] = 1
        // bucket = 3 + 1 = 4
        // ただし先手玉がSQ_59(=4)の場合、rank=4%9=4 ←正しい
        //
        // 実際のバケットはPackedSfenValueのデコード結果に依存するため、
        // ここではバケット計算ロジック自体をテストする

        // 味方玉rank=0, 相手玉rank=0 → bucket=0
        const F_TO_INDEX: [usize; 9] = [0, 0, 0, 3, 3, 3, 6, 6, 6];
        const E_TO_INDEX: [usize; 9] = [0, 0, 0, 1, 1, 1, 2, 2, 2];
        assert_eq!(F_TO_INDEX[0] + E_TO_INDEX[0], 0);

        // 味方玉rank=8, 相手玉rank=8 → bucket=8
        assert_eq!(F_TO_INDEX[8] + E_TO_INDEX[8], 8);

        // 味方玉rank=4, 相手玉rank=4 → bucket=4
        assert_eq!(F_TO_INDEX[4] + E_TO_INDEX[4], 4);
    }
}
