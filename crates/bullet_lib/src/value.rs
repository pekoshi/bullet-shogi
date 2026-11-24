pub(crate) mod builder;
mod dataloader;
pub mod loader;
mod save;

use std::cell::RefCell;

pub use builder::{NoOutputBuckets, ValueTrainerBuilder};

use acyclib::{
    graph::Node,
    trainer::{self, Trainer, logger, optimiser::OptimiserState},
};

use acyclib::{
    graph::{GraphNodeId, GraphNodeIdTy, like::GraphLike, save::SavedFormat},
    trainer::dataloader::{PreparedBatchDevice, PreparedBatchHost},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    nn::{ExecutionContext, Graph},
    trainer::{
        schedule::{TrainingSchedule, lr::LrScheduler, wdl::WdlScheduler},
        settings::LocalSettings,
    },
    value::{
        dataloader::ValueDataLoader,
        loader::{DefaultDataLoader, LoadableDataType},
    },
};

use crate::value::loader::PreparedData;

/// Value network trainer, generally for training NNUE networks.
pub struct ValueTrainer<
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
>(Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>);

impl<Opt, Inp, Out> std::ops::Deref for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    type Target = Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<Opt, Inp, Out> std::ops::DerefMut for ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;
type Wgt<I> = fn(&<I as SparseInputType>::RequiredDataType) -> f32;

#[derive(Clone)]
pub struct ValueTrainerState<Inp: SparseInputType, Out> {
    input_getter: Inp,
    output_getter: Out,
    blend_getter: B<Inp>,
    weight_getter: Option<Wgt<Inp>>,
    output_node: Node,
    saved_format: Vec<SavedFormat>,
    use_win_rate_model: bool,
    wdl: bool,
}

impl<Inp: SparseInputType, Out> ValueTrainerState<Inp, Out>
where
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn prepare(
        &self,
        batch: &[Inp::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> PreparedBatchHost {
        PreparedBatchHost::from(PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            batch,
            threads,
            blend,
            scale,
        ))
    }
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    /// 隣接 bucket smoothness regularization 付き訓練 (= 2026-06-09 追加)。
    ///
    /// 既 [`Self::run`] と同等の logic、 batch_callback 内で
    /// L3 (= `l3w`, `l3b`) + PSQT (= `psqtw`, `psqtb`) の weight tensor に対して
    /// `crate::bucket_smoothness` の `smoothness_grad_add` で gradient を計算し、
    /// 直接 weight に `-= lr × grad` の form で update する (= post-step projection)。
    ///
    /// # Arguments
    /// * `schedule` / `settings` / `dataloader` - 既 [`Self::run`] と同じ
    /// * `l3_lambda` - L3 weight 用 smoothness 係数 λ (= 0 で完全 no-op、 [`Self::run`] と数値一致)
    /// * `psqt_lambda` - PSQT weight 用同上
    /// * `num_buckets` - bucket 数 (= 通常 8)
    ///
    /// # 注意
    /// - λ=0 の場合は **既 [`Self::run`] を呼ぶことを推奨** (= 数値完全一致保証)
    /// - 本 method は λ > 0 の比較実験用 path、 production 訓練では λ=0 で no-op として動作
    /// - weight 名 / shape は arm 12 LayerStack 前提 (= bucket-outer flat layout)
    ///
    /// # Returns
    /// 完了時 `Ok(())`、 weight tensor 不在等で `Err`
    pub fn run_with_bucket_smoothness(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
        l3_lambda: f32,
        psqt_lambda: f32,
        num_buckets: usize,
        decorr_lambda: f32,
        decorr_rows: usize,
        decorr_use_rows: usize,
        decorr_l2_lambda: f32,
        decorr_l2_rows: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // TODO (= 次 session): 既 Self::run の logic を inline + train_custom の batch_callback で
        // smoothness apply 実装。 graph.get_weights(...) + get_dense_vals + smoothness_grad_add +
        // load_dense_from_slice で weight update。
        //
        // 現状は λ=0 path で no-op (= 既 run と完全一致)、 λ>0 でも当面 既 run と同 logic で動作
        // (= smoothness 未適用)。 完全実装は段階的に。
        let _ = (l3_lambda, psqt_lambda, num_buckets);
        if l3_lambda == 0.0 && psqt_lambda == 0.0 && decorr_lambda == 0.0 && decorr_l2_lambda == 0.0 {
            self.run(schedule, settings, dataloader);
            return Ok(());
        }
        // 既 Self::run の logic を inline (= train_custom 直接呼び) + batch_callback で
        // smoothness apply を追加。 weight tensor name "l3w" / "l3b" / "psqtw" / "psqtb" に対して
        // 隣接 bucket smoothness gradient を計算し、 直接 weight に `-= grad` で update する
        // (= post-step projection step、 数学的に SGD + L2 reg と等価)。
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble (= bucket smoothness ON)", "34;1"));
        println!(
            "info: bucket smoothness regularization enabled (l3_lambda={l3_lambda}, \
             psqt_lambda={psqt_lambda}, num_buckets={num_buckets})"
        );
        if decorr_lambda > 0.0 {
            println!(
                "info: L1 decorrelation enabled (lambda={decorr_lambda}, rows={decorr_rows},                  use_rows={decorr_use_rows}, l2_lambda={decorr_l2_lambda}, l2_rows={decorr_l2_rows})                  -- fc0/fc1 の行間相関にペナルティを掛けて分化を促す"
            );
        }

        schedule.display();
        settings.display();

        if settings.test_set.is_some() {
            println!(
                "{}",
                logger::ansi("Warning: Validation data not currently implemented! Please bother me on discord.", "31")
            );
        }

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();
        let steps = schedule.steps;

        let error_record = std::cell::RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        let smoothness_targets: &[(&str, f32)] =
            &[("l3w", l3_lambda), ("l3b", l3_lambda), ("psqtw", psqt_lambda), ("psqtb", psqt_lambda)];

        self.train_custom(
            trainer::schedule::TrainingSchedule {
                steps,
                log_rate: schedule.log_rate,
                lr_schedule: Box::new(move |a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader {
                steps,
                threads: settings.threads,
                dataloader,
                wdl: schedule.wdl_scheduler.clone(),
            },
            |trainer, superbatch, curr_batch, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if curr_batch % 32 == 0
                    || (steps.batches_per_superbatch < 32
                        && curr_batch == steps.batches_per_superbatch)
                {
                    let normalised_loss = loss_sum
                        / f32::min(ticks_since_last, steps.batches_per_superbatch as f32);
                    error_record
                        .borrow_mut()
                        .push((superbatch, curr_batch, normalised_loss));
                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }

                // post-step projection: 各 target weight に対し smoothness gradient を計算し、
                // 直接 weight に `-= grad` で update する (= optimizer step 後の projection、
                // 数学的に SGD + L2 regularization と等価)。
                // λ=0 の target は smoothness_grad_add 内 early-return で no-op、 weight 不変。
                // L1 (fc0) / L2 (fc1) の行間 decorrelation。
                //   活性化ベースで測ると L1 の実効次元は 3.69/15、 L2 は 15.4/64 (2026-09-09 実測)。
                //   FT 1024 の情報が L1 で 3.7 次元まで絞られており、 ここが真のボトルネック。
                //   重みが違っても実局面では似た値を出しているため、 行間相関にペナルティを掛ける。
                //   smoothness と同じ post-step projection (forward は触らない)。
                for (tname, lam, rows, use_rows) in [
                    ("l1w", decorr_lambda, decorr_rows, decorr_use_rows),
                    ("l2w", decorr_l2_lambda, decorr_l2_rows, decorr_l2_rows),
                ] {
                    if lam <= 0.0 || curr_batch % 32 != 0 {
                        continue;
                    }
                    let tensor_ref = trainer.optimiser.graph.get_weights(tname);
                    if let Ok(w_vals) = tensor_ref.get_dense_vals() {
                        let total = w_vals.len();
                        if rows == 0 || total % (num_buckets * rows) != 0 {
                            eprintln!("warn: {tname} の形状が合わない (total={total}, buckets={num_buckets}, rows={rows})");
                            continue;
                        }
                        let per_row = total / (num_buckets * rows);
                        let mut grad = vec![0.0_f32; total];
                        crate::l1_decorrelation::decorr_grad_add(
                            &w_vals, &mut grad, num_buckets, rows, per_row, use_rows, lam,
                        );
                        let new_w: Vec<f32> = w_vals.iter().zip(grad.iter()).map(|(w, g)| w - g).collect();
                        let batch_size = tensor_ref.batch_size();
                        let mut guard = tensor_ref.borrow_mut();
                        if let Err(e) = guard.load_dense_from_slice(batch_size, &new_w) {
                            eprintln!("warn: {tname} decorrelation write-back failed: {e:?}");
                        }
                    }
                }

                for &(name, lambda) in smoothness_targets {
                    if lambda == 0.0 {
                        continue;
                    }
                    let tensor_ref = trainer.optimiser.graph.get_weights(name);
                    let w_vals = match tensor_ref.get_dense_vals() {
                        Ok(v) => v,
                        Err(_) => {
                            eprintln!(
                                "warn: bucket_smoothness '{name}' get_dense_vals failed at sb={superbatch} curr_batch={curr_batch}"
                            );
                            continue;
                        }
                    };
                    let total = w_vals.len();
                    if total == 0 || total % num_buckets != 0 {
                        continue;
                    }
                    let dim_per_bucket = total / num_buckets;
                    let mut grad = vec![0.0_f32; total];
                    crate::bucket_smoothness::smoothness_grad_add(
                        &w_vals,
                        &mut grad,
                        num_buckets,
                        dim_per_bucket,
                        lambda,
                    );
                    let new_w: Vec<f32> =
                        w_vals.iter().zip(grad.iter()).map(|(w, g)| w - g).collect();
                    let batch_size = tensor_ref.batch_size();
                    let mut guard = tensor_ref.borrow_mut();
                    if let Err(e) = guard.load_dense_from_slice(batch_size, &new_w) {
                        eprintln!(
                            "warn: bucket_smoothness '{name}' write-back failed at sb={superbatch} curr_batch={curr_batch}: {e:?}"
                        );
                    }
                }
            },
            |trainer, superbatch| {
                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    // bucket smoothness 係数 + 各 target tensor の最新 L_smooth を JSON で記録。
                    // λ=0 の target は loss 0、 既 experiment.json に並べて保存する独立 file。
                    let mut json = String::new();
                    json.push_str("{\n");
                    json.push_str(&format!("  \"l3_lambda\": {l3_lambda},\n"));
                    json.push_str(&format!("  \"psqt_lambda\": {psqt_lambda},\n"));
                    json.push_str(&format!("  \"num_buckets\": {num_buckets},\n"));
                    json.push_str("  \"smoothness_losses\": {\n");
                    for (idx, &(tname, lambda)) in smoothness_targets.iter().enumerate() {
                        let val = if lambda == 0.0 {
                            0.0_f32
                        } else {
                            let tensor_ref = trainer.optimiser.graph.get_weights(tname);
                            match tensor_ref.get_dense_vals() {
                                Ok(w_vals) => {
                                    let total = w_vals.len();
                                    if total == 0 || total % num_buckets != 0 {
                                        0.0_f32
                                    } else {
                                        let dim = total / num_buckets;
                                        crate::bucket_smoothness::smoothness_loss(
                                            &w_vals,
                                            num_buckets,
                                            dim,
                                            lambda,
                                        )
                                    }
                                }
                                Err(_) => 0.0_f32,
                            }
                        };
                        let comma = if idx + 1 < smoothness_targets.len() { "," } else { "" };
                        json.push_str(&format!("    \"{tname}\": {val:.6e}{comma}\n"));
                    }
                    json.push_str("  }\n");
                    json.push_str("}\n");
                    let _ = std::fs::write(
                        format!("{path}/bucket_smoothness_config.json"),
                        json,
                    );

                    println!("Saved [{}]", logger::ansi(name, 31));

                    if let Some(ref callback) = settings.on_checkpoint_saved {
                        callback(superbatch);
                    }
                }
            },
        )
        .map_err(|e| -> Box<dyn std::error::Error> { format!("train_custom failed: {e:?}").into() })?;

        Ok(())
    }

    pub fn run(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) {
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble", "34;1"));

        schedule.display();
        settings.display();

        if settings.test_set.is_some() {
            println!(
                "{}",
                logger::ansi("Warning: Validation data not currently implemented! Please bother me on discord.", "31")
            )
        }

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();

        let steps = schedule.steps;

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        self.train_custom(
            trainer::schedule::TrainingSchedule {
                steps,
                log_rate: schedule.log_rate,
                lr_schedule: Box::new(|a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader { steps, threads: settings.threads, dataloader, wdl: schedule.wdl_scheduler.clone() },
            |_, superbatch, curr_batch, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if curr_batch % 32 == 0
                    || (steps.batches_per_superbatch < 32 && curr_batch == steps.batches_per_superbatch)
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, steps.batches_per_superbatch as f32);

                    error_record.borrow_mut().push((superbatch, curr_batch, normalised_loss));

                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }
            },
            |trainer, superbatch| {
                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));

                    if let Some(ref callback) = settings.on_checkpoint_saved {
                        callback(superbatch);
                    }
                }
            },
        )
        .unwrap();
    }

    /// [`run`](Self::run) と同一だが、 各 optimizer step 後 (= per-batch) に指定した weight tensor の
    /// flat dense index 範囲 `[start, end)` を 0 に固定する (= post-step zero projection)。
    ///
    /// 用途: 連結 FT 入力 (aux / kzt-ft / threat 等) で PSQT projection の tail 行を訓練中も
    /// ゼロに保ち、 export で捨てられる行が非ゼロに育つことによる **訓練 net ≠ 配備 net** を防ぐ。
    /// weight が warm-start で当該範囲 0 初期化されている前提で、 毎 step 更新を打ち消して常時 0 を保つ。
    ///
    /// `zero_targets` は `(weight_name, start_flat, end_flat)` の列。 範囲は tensor の dense flat
    /// index (column-major layout の graph tensor `values`) に対する半開区間。 `run_with_bucket_smoothness`
    /// の post-step projection と同一機構 (get_dense_vals → 範囲ゼロ → load_dense_from_slice)。
    pub fn run_with_zeroed_rows(
        &mut self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
        zero_targets: &[(String, usize, usize)],
    ) -> Result<(), Box<dyn std::error::Error>> {
        logger::clear_colours();
        println!("{}", logger::ansi("Training Preamble", "34;1"));

        schedule.display();
        settings.display();

        if settings.test_set.is_some() {
            println!(
                "{}",
                logger::ansi("Warning: Validation data not currently implemented! Please bother me on discord.", "31")
            )
        }

        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let _ = std::fs::create_dir(settings.output_directory);

        let lr_scheduler = schedule.lr_scheduler.clone();
        let steps = schedule.steps;

        let error_record = RefCell::new(Vec::new());
        let mut loss_sum = 0.0;
        let mut ticks_since_last = 0.0;

        self.train_custom(
            trainer::schedule::TrainingSchedule {
                steps,
                log_rate: schedule.log_rate,
                lr_schedule: Box::new(move |a, b| lr_scheduler.lr(a, b)),
            },
            ValueDataLoader { steps, threads: settings.threads, dataloader, wdl: schedule.wdl_scheduler.clone() },
            |trainer, superbatch, curr_batch, error| {
                loss_sum += error;
                ticks_since_last += 1.0;

                if curr_batch % 32 == 0
                    || (steps.batches_per_superbatch < 32 && curr_batch == steps.batches_per_superbatch)
                {
                    let normalised_loss = loss_sum / f32::min(ticks_since_last, steps.batches_per_superbatch as f32);
                    error_record.borrow_mut().push((superbatch, curr_batch, normalised_loss));
                    loss_sum = 0.0;
                    ticks_since_last = 0.0;
                }

                // post-step zero projection: optimizer step 後、 指定範囲を 0 に上書きする。
                // warm-start で 0 初期化されている行の毎 step 更新を打ち消し、 常時 0 を保証する。
                //
                // 2026-07-23: device 上の部分範囲 memset (`set_range_to`) で完結させる。
                // 旧実装は全行列の GPU→CPU→GPU 往復 (get_dense_vals + load_dense_from_slice)
                // だったため、 auxcheap の l0w (73,697×L1 ≈ 数百 MB) では毎 step の PCIe 転送が
                // 律速になっていた (計測: aux-FT psqtw 2.4MB → 38k pos/s、 auxcheap l0w → 9.7k)。
                // 書かれる値はどちらも f32 0.0 で、 範囲外は不変なので weight は bit-exact に一致する。
                for (name, start, end) in zero_targets {
                    let tensor_ref = trainer.optimiser.graph.get_weights(name);
                    let mut guard = tensor_ref.borrow_mut();
                    let dense = match guard.dense_mut() {
                        Ok(d) => d,
                        Err(_) => {
                            eprintln!(
                                "warn: zeroed_rows '{name}' is not a dense tensor (sb={superbatch} curr_batch={curr_batch})"
                            );
                            continue;
                        }
                    };
                    let (s, e) = (*start, (*end).min(dense.size()));
                    if s >= e {
                        continue;
                    }
                    if let Err(err) = dense.set_range_to(s, e, 0.0) {
                        eprintln!(
                            "warn: zeroed_rows '{name}' set_range_to failed at sb={superbatch} curr_batch={curr_batch}: {err:?}"
                        );
                    }
                }
            },
            |trainer, superbatch| {
                if superbatch % schedule.save_rate == 0 || superbatch == steps.end_superbatch {
                    let name = format!("{}-{superbatch}", schedule.net_id);
                    let path = format!("{}/{name}", settings.output_directory);
                    std::fs::create_dir(path.as_str()).unwrap_or(());
                    save::save_to_checkpoint(trainer, &path);
                    save::write_losses(&format!("{path}/log.txt"), &error_record.borrow());

                    println!("Saved [{}]", logger::ansi(name, 31));

                    if let Some(ref callback) = settings.on_checkpoint_saved {
                        callback(superbatch);
                    }
                }
            },
        )
        .map_err(|e| -> Box<dyn std::error::Error> { format!("train_custom failed: {e:?}").into() })
    }

    pub fn get_output_values(&self) -> Vec<f32> {
        let id = GraphNodeId::new(self.state.output_node.idx(), GraphNodeIdTy::Values);

        #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
        {
            self.optimiser.graph.get(id).unwrap().get_dense_vals().unwrap()
        }

        #[cfg(any(feature = "multigpu", feature = "cpu"))]
        self.optimiser.graph.get_all(id).unwrap().iter().flat_map(|x| x.get_dense_vals().unwrap()).collect()
    }

    pub fn eval_raw_output(&mut self, fen: &str) -> Vec<f32>
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let pos = format!("{fen} | 0 | 0.0").parse::<Inp::RequiredDataType>().unwrap();

        let host_data = self.state.prepare(&[pos], 1, 1.0, 1.0);

        #[cfg(not(any(feature = "multigpu", feature = "cpu")))]
        let graph = &mut self.optimiser.graph;

        #[cfg(any(feature = "multigpu", feature = "cpu"))]
        let graph = self.optimiser.graph.primary_mut();

        let mut device_data = PreparedBatchDevice::new(graph.devices(), &host_data).unwrap();

        device_data.load_into_graph(graph).unwrap();

        graph.synchronise().unwrap();
        graph.forward().unwrap();

        self.get_output_values()
    }

    pub fn eval(&mut self, fen: &str) -> f32
    where
        Inp::RequiredDataType: std::str::FromStr<Err: std::fmt::Debug> + LoadableDataType,
    {
        let vals = self.eval_raw_output(fen);

        match vals[..] {
            [mut loss, mut draw, mut win] => {
                let max = win.max(draw).max(loss);
                win = (win - max).exp();
                draw = (draw - max).exp();
                loss = (loss - max).exp();

                (win + draw / 2.0) / (win + draw + loss)
            }
            [score] => score,
            _ => panic!("Invalid output size!"),
        }
    }

    pub fn measure_max_cpu_throughput(
        &self,
        schedule: &TrainingSchedule<impl LrScheduler, impl WdlScheduler>,
        settings: &LocalSettings,
        dataloader: &impl loader::DataLoader<Inp::RequiredDataType>,
    ) {
        let steps = schedule.steps;
        let threads = settings.threads;
        let wdl = schedule.wdl_scheduler.clone();
        let dataloader = DefaultDataLoader::new(
            self.state.input_getter.clone(),
            self.state.output_getter,
            self.state.blend_getter,
            self.state.weight_getter,
            self.state.use_win_rate_model,
            self.state.wdl,
            schedule.eval_scale,
            dataloader.clone(),
        );

        let dataloader = ValueDataLoader { steps, threads, dataloader, wdl };

        self.0.measure_max_cpu_throughput(dataloader, steps).unwrap()
    }
}
