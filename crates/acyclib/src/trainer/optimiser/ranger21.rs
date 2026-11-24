use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    sync::Arc,
};

use crate::{
    device::{
        Device, OperationError,
        operation::{BaseOperations, Ranger21Config},
        tensor::DenseMatrix,
    },
    graph::CheckpointShapeAdapter,
    trainer::optimiser::{OptimiserState, utils},
};

/// Ranger21驍ｵ・ｺ繝ｻ・ｧ鬯ｩ・･鬮ｦ・ｪ遶擾ｽｩ驛｢譎擾ｽｼ・ｱ・取刮・ｹ譎｢・｣・ｰ髮弱・・ｽ・｣髯ｷ莉｣繝ｻ陜滂ｽｧ驛｢・ｧ陝ｶ譏ｶ繝ｻ鬨ｾ蛹・ｽｽ・ｨ驍ｵ・ｺ陷ｷ・ｶ繝ｻ邇厄ｽｭ蜴・ｽｽ・ｴ髫ｴ繝ｻ・ｽ・ｰ髣厄ｽｴ陷･・ｲ繝ｻ・ｽ繝ｻ・ｮ驍ｵ・ｲ郢晢ｽｻ///
/// Profile15髯具ｽｻ隴惹ｸ橸ｽｱ讌｢・ｱ閧ｲ・｢謇假ｽｽ・ｼ郢晢ｽｻ邵ｲ蝣､・ｸ・ｺ繝ｻ・ｯ `None` 驍ｵ・ｺ繝ｻ・ｫ髯懈圜・ｽ・ｺ髯橸ｽｳ陞｢・ｹ隨倥・・ｹ・ｧ闕ｵ謨鳴郢晢ｽｻNUE驍ｵ・ｺ繝ｻ・ｮ鬯ｩ・･闕ｳ讖ｸ・ｽ・ｭ闔牙雀蟇・寞繧・樟遶擾ｽｩFT驍ｵ・ｺ繝ｻ・ｫ髯昴・・ｽ・ｾ驍ｵ・ｺ陷ｷ・ｶ繝ｻ繝ｻ/// unit-norm鬮ｫ・ｱ闔ｨ諛ｶ・ｽ・ｰ陟托ｽｱ繝ｻ蠑孅timizer髯晢ｽｾ繝ｻ・ｮ驍ｵ・ｺ繝ｻ・ｸ髮趣ｽｺ繝ｻ・ｷ髯懶ｽｨ繝ｻ・ｨ驍ｵ・ｺ髴域喚髮ｷ驍ｵ・ｺ繝ｻ・ｪ驍ｵ・ｺ郢晢ｽｻ隨ｳ繝ｻ・ｹ・ｧ遶丞｣ｹﾂ蝣､・ｸ・ｺ郢ｧ繝ｻ・ｽ迢暦ｽｸ・ｲ郢晢ｽｻ
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NormLossPlacement {
    #[default]
    None,
    Before,
    After,
}

/// Profile15鬨ｾ蛹・ｽｽ・ｨRanger21 core驍ｵ・ｺ繝ｻ・ｮ驛｢譏懶ｽｻ・｣・主ｸｷ・ｹ譎｢・ｽ・｡驛｢譎｢・ｽ・ｼ驛｢・ｧ繝ｻ・ｿ驍ｵ・ｲ郢晢ｽｻ///
/// 髯橸ｽｳ雋・ｽｯ繝ｻ・｣郢晢ｽｻ隨倥・・ｹ・ｧ陋ｹ・ｺ郢晢ｽｻ髯具ｽｻ郢晢ｽｻ郢晢ｽｻAdamW驍ｵ・ｲ繝ｻ謐尽itive-negative momentum (PNM)驍ｵ・ｲ邵ｲ・SGrad max
/// variance驍ｵ・ｲ邵ｲ蜩］ear warm-up/warm-down驍ｵ・ｲ・ゑｽｴookahead驍ｵ・ｺ繝ｻ・ｧ驍ｵ・ｺ郢ｧ繝ｻ・ｽ迢暦ｽｸ・ｲ郢晢ｽｻGC驍ｵ・ｲ遯ｶ蠕ｨadient
/// centralization驍ｵ・ｲ邵ｲ髢｣able weight decay驍ｵ・ｲ繝ｻ・ｽorm loss驍ｵ・ｺ繝ｻ・ｯ髴托ｽ｢繝ｻ・ｬ鬩包ｽｶ驍・私・ｽ・ｦ遶乗剌・ｱ蜀暦ｽｸ・ｺ繝ｻ・ｨ驍ｵ・ｺ陷会ｽｱ遯ｶ・ｻ髯具ｽｻ郢晢ｽｻ繝ｻ鬘伜ｴ慕ｹ晢ｽｻ繝ｻ・ｰ驛｢・ｧ闕ｵ譏ｶ陞ｺ驛｢・ｧ郢晢ｽｻ/// 驍ｵ・ｺ髦ｮ蜷ｶ繝ｻ髯具ｽｻ隴惹ｸ橸ｽｱ讌｢・ｱ閧ｲ・｢謇假ｽｽ・ｼ郢晢ｽｻ邵ｲ蝣､・ｸ・ｺ繝ｻ・ｯ髴取ｻゑｽｽ・｡髯ｷ莨夲ｽｽ・ｹ驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｺ陷ｷ・ｶ繝ｻ迢暦ｽｸ・ｲ郢晢ｽｻ
#[derive(Clone, Copy, Debug)]
pub struct Ranger21Params {
    pub decay: f32,
    pub beta0: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub alpha: f32,
    pub lookahead_mergetime: usize,
    /// Parameter value clamp. This is not AGC.
    pub clip: Option<(f32, f32)>,
    pub norm_loss_factor: f32,
    pub norm_loss_placement: NormLossPlacement,
    /// 0 disables the Ranger21-internal LR schedule.
    pub total_steps: usize,
    /// 0 selects the Ranger21 automatic warm-up rule.
    pub warmup_steps: usize,
    /// 0 selects 28% of `total_steps`.
    pub warmdown_steps: usize,
}

impl Default for Ranger21Params {
    fn default() -> Self {
        Self {
            decay: 0.0,
            beta0: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
            alpha: 0.5,
            lookahead_mergetime: 5,
            clip: Some((-1.98, 1.98)),
            norm_loss_factor: 0.0,
            norm_loss_placement: NormLossPlacement::None,
            total_steps: 0,
            warmup_steps: 0,
            warmdown_steps: 0,
        }
    }
}

const STATE_FORMAT: &str = "ranger21-profile15-core-v4-host-lookahead";

/// AdamW驍ｵ・ｲ繝ｻ・｣NM驍ｵ・ｲ邵ｲ・SGrad max variance驍ｵ・ｲ・ゑｽｴookahead驛｢・ｧ陜｣・､繝ｻ・ｵ郢晢ｽｻ遶擾ｽｩ髯ｷ・ｷ陋ｹ・ｻ繝ｻ蜀暦ｽｸ・ｺ陝ｶ蜷ｮ陞ｺRanger21 core驍ｵ・ｲ郢晢ｽｻ///
/// State file names overlap with legacy Ranger (`momentum.bin`, `velocity.bin`,
/// `slow.bin`) but the format is not compatible. `ranger21_state_format.txt` is
/// written and required so old Ranger state can never be silently misread.
pub struct Ranger21<D: Device> {
    /// m_{t-1} after every update.
    momentum: DenseMatrix<D>,
    velocity: DenseMatrix<D>,
    max_velocity: DenseMatrix<D>,
    /// Lookahead驍ｵ・ｺ繝ｻ・ｮslow weight驍ｵ・ｲ郢晢ｽｻPU髴托ｽ･繝ｻ・ｶ髫ｲ・ｷ闕ｵ譎｢・ｽ繝ｻ髫ｴ蟷｢・ｽ・ｬ髮九ｇ・ｸ蜻ｻ・ｽ閾･・ｸ・ｺ陷ｷ・ｶ隨ｳ繝ｻ・ｹ・ｧ遯ｶ謐尽t驍ｵ・ｺ繝ｻ・ｫ髣厄ｽｫ隴弱・莠憺し・ｺ陷会ｽｱ・つ郢晢ｽｻ step驍ｵ・ｺ隴∫ｵｶ繝ｻ驍ｵ・ｺ繝ｻ・ｫ髯ｷ・ｷ隴ｴ・ｧ隰斐・・ｸ・ｺ陷ｷ・ｶ繝ｻ迢暦ｽｸ・ｲ郢晢ｽｻ
    slow_params: Vec<f32>,
    current_params: Vec<f32>,
    /// Lookahead merge髫ｴ蠑ｱ・・ｹ晢ｽｻGPU readback鬨ｾ蛹・ｽｽ・ｨhost scratch驍ｵ・ｲ繝ｻ・､heckpoint驍ｵ・ｺ繝ｻ・ｫ驍ｵ・ｺ繝ｻ・ｯ髯ｷ・ｷ繝ｻ・ｫ驛｢・ｧ遶丞｣ｺ繝ｻ驍ｵ・ｺ郢晢ｽｻ・つ郢晢ｽｻ
    params: Ranger21Params,
    /// 2026-09-15 群別 lr: `(start, end, lr 倍率)` の区間列。 空なら全体を倍率 1.0 で 1 回更新 (従来)。
    /// 空でないときは [0, size) を隙間なく昇順に覆っていること (`set_range_lr_scales` で検査)。
    range_lr_scales: Vec<(usize, usize, f32, f32)>,
    step: usize,
    lookahead_step: usize,
    slow_initialised: bool,
}

impl Ranger21Params {
    fn warmup_steps(self) -> usize {
        if self.total_steps == 0 {
            return 0;
        }
        if self.warmup_steps != 0 {
            return self.warmup_steps.min(self.total_steps);
        }
        let beta_rule = (2.0 / (1.0 - self.beta2)).ceil() as usize;
        if beta_rule.saturating_mul(100) > self.total_steps.saturating_mul(45) {
            ((self.total_steps as f32) * 0.22).ceil() as usize
        } else {
            beta_rule
        }
        .max(1)
    }

    fn warmdown_steps(self) -> usize {
        if self.total_steps == 0 {
            0
        } else if self.warmdown_steps != 0 {
            self.warmdown_steps.min(self.total_steps).max(1)
        } else {
            ((self.total_steps as f32) * 0.28).ceil() as usize
        }
    }

    fn scheduled_lr(self, base_lr: f32, step: usize) -> f32 {
        if self.total_steps == 0 {
            return base_lr;
        }
        let warmup = self.warmup_steps();
        let warmdown = self.warmdown_steps();
        let warmup_scale = (step as f32 / warmup as f32).min(1.0);
        let warmdown_start = self.total_steps.saturating_sub(warmdown);
        let warmdown_scale = if step <= warmdown_start {
            1.0
        } else {
            ((self.total_steps.saturating_sub(step)) as f32 / warmdown as f32).clamp(0.0, 1.0)
        };
        base_lr * warmup_scale.min(warmdown_scale)
    }
}

impl<D: Device> OptimiserState<D> for Ranger21<D> {
    type Params = Ranger21Params;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self {
            momentum: DenseMatrix::zeroed(device.clone(), size, None)?,
            velocity: DenseMatrix::zeroed(device.clone(), size, None)?,
            max_velocity: DenseMatrix::zeroed(device, size, None)?,
            slow_params: vec![0.0; size],
            current_params: vec![0.0; size],

            params,
            range_lr_scales: Vec::new(),
            step: 0,
            lookahead_step: 0,
            slow_initialised: false,
        })
    }

    fn set_range_lr_scales(&mut self, ranges: Vec<(usize, usize, f32, f32)>) {
        let size = self.momentum.size();
        let mut cursor = 0;
        for &(start, end, scale, _decay_scale) in &ranges {
            assert!(start == cursor && end > start, "range lr scales must tile [0, {size}) in order: got {ranges:?}");
            assert!(scale > 0.0 && scale.is_finite(), "range lr scale must be positive finite: got {scale}");
            cursor = end;
        }
        assert!(ranges.is_empty() || cursor == size, "range lr scales must cover [0, {size}): got {ranges:?}");
        self.range_lr_scales = ranges;
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        assert!(weights.batch_size().is_none());
        assert_eq!(weights.size(), self.momentum.size());
        assert_eq!(weights.size(), self.velocity.size());
        assert_eq!(weights.size(), self.max_velocity.size());
        assert!(self.params.lookahead_mergetime > 0, "Ranger21 lookahead interval must be positive");
        assert!((0.0..=1.0).contains(&self.params.beta0), "Ranger21 beta0 must be in [0, 1]");
        assert!((0.0..1.0).contains(&self.params.beta1), "Ranger21 beta1 must be in [0, 1)");
        assert!((0.0..1.0).contains(&self.params.beta2), "Ranger21 beta2 must be in [0, 1)");

        // Lookahead slow weights are theta_0, not zeros. GPU state驛｢・ｧ陜｣・､繝ｻ・ｯ・つ鬩堺ｸ翫・隨倥・・ｹ・ｧ闕ｵ譏ｶ陞ｺ驛｢・ｧ遯ｶ謐尽t驍ｵ・ｺ繝ｻ・ｸreadback驍ｵ・ｺ陷ｷ・ｶ繝ｻ迢暦ｽｸ・ｲ郢晢ｽｻ
        if !self.slow_initialised {
            weights.write_to_slice(&mut self.slow_params)?;
            self.slow_initialised = true;
        }

        self.step += 1;
        let params = self.params;
        let scheduled_lr = params.scheduled_lr(learning_rate, self.step);
        apply_norm_loss(weights, scheduled_lr, params, NormLossPlacement::Before)?;

        let step = self.step as f32;
        let bias_correction1 = 1.0 - params.beta1.powf(step);
        let inv_sqrt_bias_correction2 = (1.0 - params.beta2.powf(step)).sqrt().recip();
        let pnm_noise_norm = ((1.0 + params.beta0).powi(2) + params.beta0.powi(2)).sqrt();
        let (min_weight, max_weight, clamp_weights) = match params.clip {
            Some((min, max)) => (min, max, true),
            None => (0.0, 0.0, false),
        };

        let config_for_full = |lr_scale: f32, decay_scale: f32| Ranger21Config {
            beta0: params.beta0,
            beta1_sq: params.beta1 * params.beta1,
            beta2: params.beta2,
            gradient_factor,
            step_size: scheduled_lr * lr_scale / (bias_correction1 * pnm_noise_norm),
            inv_sqrt_bias_correction2,
            eps: params.eps,
            decay: 1.0 - params.decay * decay_scale * scheduled_lr * lr_scale,
            min_weight,
            max_weight,
            clamp_weights,
        };
        if self.range_lr_scales.is_empty() {
            // 区間倍率なし = 従来と完全に同一の 1 回呼び出し。
            weights.buf.ranger21(
                &config_for_full(1.0, 1.0),
                0,
                weights.size(),
                &grads.buf,
                &mut self.momentum.buf,
                &mut self.velocity.buf,
                &mut self.max_velocity.buf,
            )?;
        } else {
            // 2026-09-15 群別 lr: 区間ごとに lr (step_size と decay) だけを変えて更新する。
            //   Ranger21 の更新は要素ごとに独立 (隣の要素を読まない) なので、 区間に分けて呼んでも
            //   倍率 1.0 の区間は分けずに呼んだときと bit 単位で同じ値になる。
            //   momentum / velocity / max_velocity の更新式は lr に依存しないので、 倍率で変わるのは
            //   重みの 1 歩の大きさだけ (= その区間の lr を scale 倍したのと同じ)。
            for &(start, end, scale, decay_scale) in &self.range_lr_scales {
                weights.buf.ranger21(
                    &config_for_full(scale, decay_scale),
                    start,
                    end - start,
                    &grads.buf,
                    &mut self.momentum.buf,
                    &mut self.velocity.buf,
                    &mut self.max_velocity.buf,
                )?;
            }
        }

        apply_norm_loss(weights, scheduled_lr, params, NormLossPlacement::After)?;
        self.lookahead_step += 1;
        if self.lookahead_step >= params.lookahead_mergetime {
            self.lookahead_step = 0;
            weights.write_to_slice(&mut self.current_params)?;
            for (slow, current) in self.slow_params.iter_mut().zip(self.current_params.iter()) {
                *slow = (1.0 - params.alpha) * *slow + params.alpha * *current;
            }
            weights.load_from_slice(None, &self.slow_params)?;
        }

        Ok(())
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.step = 0;
        self.lookahead_step = 0;
        self.slow_initialised = false;
        self.momentum.set_to(0.0)?;
        self.velocity.set_to(0.0)?;
        self.max_velocity.set_to(0.0)?;
        self.slow_params.fill(0.0);
        self.current_params.fill(0.0);
        Ok(())
    }

    fn init_velocity(&mut self, value: f32) -> Result<(), D::DeviceError> {
        self.velocity.set_to(value)?;
        self.max_velocity.set_to(value)
    }

    fn set_params(&mut self, params: Self::Params) {
        self.params = params;
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>> {
        Self::load_from_checkpoint_with_skip(map, path, old_format, adapter, &HashSet::new())
    }

    /// **勾配を持たない重み (真の凍結) を許す版**。  2026-11-03
    ///   `skip` に入っている id は optimiser 状態を持たないので、 checkpoint の file に
    ///   その状態が残っていても無視する。 `skip` が空なら従来と完全に同じ挙動。
    ///   🚨`BULLET_SKIP_UNKNOWN_WEIGHTS=1` と違い、 **名指しした id だけ**を黙らせる
    ///     (本物の不一致は従来どおり panic する)。
    fn load_from_checkpoint_with_skip(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
        skip: &HashSet<String>,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let manifest_path = format!("{path}/ranger21_state_format.txt");
        let manifest = fs::read_to_string(&manifest_path).unwrap_or_else(|_| {
            panic!(
                "Ranger21 state manifest missing at {manifest_path}. Refusing to read legacy Ranger/unknown optimiser state; use the weights-only warm-start path instead."
            )
        });
        assert_eq!(manifest.trim(), STATE_FORMAT, "Ranger21 state manifest mismatch");

        let mut momentum = utils::load_weights_from_file(&format!("{path}/momentum.bin"), old_format);
        let mut velocity = utils::load_weights_from_file(&format!("{path}/velocity.bin"), old_format);
        let mut max_velocity = utils::load_weights_from_file(&format!("{path}/max_velocity.bin"), old_format);
        let mut slow = utils::load_weights_from_file(&format!("{path}/slow.bin"), old_format);
        let skip_unknown = std::env::var("BULLET_SKIP_UNKNOWN_WEIGHTS")
            .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        momentum.sort_by_key(|(id, _)| id.clone());
        velocity.sort_by_key(|(id, _)| id.clone());
        max_velocity.sort_by_key(|(id, _)| id.clone());
        slow.sort_by_key(|(id, _)| id.clone());
        assert_eq!(momentum.len(), velocity.len(), "Ranger21 velocity entry count differs");
        assert_eq!(momentum.len(), max_velocity.len(), "Ranger21 max-velocity entry count differs");
        assert_eq!(momentum.len(), slow.len(), "Ranger21 slow entry count differs");

        for ((((id1, momentum_values), (id2, velocity_values)), (id3, max_velocity_values)), (id4, slow_values)) in
            momentum.iter().zip(velocity.iter()).zip(max_velocity.iter()).zip(slow.iter())
        {
            assert_eq!(id1, id2, "Ranger21 checkpoint IDs differ");
            assert_eq!(id1, id3, "Ranger21 checkpoint IDs differ");
            assert_eq!(id1, id4, "Ranger21 checkpoint IDs differ");
            let Some(single) = map.get_mut(id1) else {
                if skip.contains(id1) {
                    // 真の凍結された重み。 状態を持たないので checkpoint の分は捨てる。
                    println!("optimiser: skip Ranger21 state for frozen weight {id1}");
                    continue;
                }
                if skip_unknown {
                    eprintln!("warn: skip unknown Ranger21 state: {id1}");
                    continue;
                }
                panic!("unknown Ranger21 weight: {id1}");
            };
            // 要素数不一致は adapter (例: Direct15 head 展開) で weights.bin と同じ変換を
            // momentum / velocity / max_velocity / slow の 4 バッファへ適用する。
            // 変換できないときのみ従来どおり skip (BULLET_SKIP_UNKNOWN_WEIGHTS) / panic。
            let adapted = (|| -> Result<_, OperationError<D::DeviceError>> {
                Ok((
                    utils::adapt_to_size::<D>(adapter, id1, momentum_values, single.momentum.size())?,
                    utils::adapt_to_size::<D>(adapter, id1, velocity_values, single.velocity.size())?,
                    utils::adapt_to_size::<D>(adapter, id1, max_velocity_values, single.max_velocity.size())?,
                    utils::adapt_to_size::<D>(adapter, id1, slow_values, single.slow_params.len())?,
                ))
            })();
            let (momentum_values, velocity_values, max_velocity_values, slow_values) = match adapted {
                Ok(adapted) => adapted,
                Err(_) if skip_unknown => {
                    eprintln!("warn: skip Ranger21 size mismatch {id1}");
                    continue;
                }
                Err(_) => panic!("Ranger21 state size mismatch for {id1}"),
            };
            single.momentum.load_from_slice(None, &momentum_values)?;
            single.velocity.load_from_slice(None, &velocity_values)?;
            single.max_velocity.load_from_slice(None, &max_velocity_values)?;
            single.slow_params.copy_from_slice(&slow_values);
            single.slow_initialised = true;
        }

        let step_path = format!("{path}/step_ranger21.txt");
        for line in BufReader::new(File::open(step_path).expect("Ranger21 step file")).lines() {
            let line = line.expect("read Ranger21 step");
            let mut values = line.split(',');
            let id = values.next().expect("Ranger21 state id");
            let step: usize = values.next().expect("Ranger21 step").parse().expect("Ranger21 numeric step");
            let lookahead_step: usize =
                values.next().expect("Ranger21 lookahead step").parse().expect("Ranger21 numeric lookahead step");
            if let Some(single) = map.get_mut(id) {
                single.step = step;
                single.lookahead_step = lookahead_step;
            }
        }
        Ok(())
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let momentum: Vec<_> = map.iter().map(|(id, single)| (id, &single.momentum)).collect();
        let velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.velocity)).collect();
        let max_velocity: Vec<_> = map.iter().map(|(id, single)| (id, &single.max_velocity)).collect();
        let slow: Vec<_> = map.iter().map(|(id, single)| (id.as_str(), &single.slow_params)).collect();
        utils::write_weights_to_file(&momentum, &format!("{path}/momentum.bin"))?;
        utils::write_weights_to_file(&velocity, &format!("{path}/velocity.bin"))?;
        utils::write_weights_to_file(&max_velocity, &format!("{path}/max_velocity.bin"))?;
        write_host_weights_to_file(&slow, &format!("{path}/slow.bin"));
        fs::write(format!("{path}/ranger21_state_format.txt"), STATE_FORMAT).expect("write Ranger21 state manifest");

        let mut file = File::create(format!("{path}/step_ranger21.txt")).expect("create Ranger21 step file");
        for (id, single) in map {
            writeln!(file, "{id},{},{}", single.step, single.lookahead_step).expect("write Ranger21 step");
        }
        Ok(())
    }
}

fn write_host_weights_to_file(map: &[(&str, &Vec<f32>)], path: &str) {
    const CHUNK_VALUES: usize = 262_144;
    let file = File::create(path).expect("create Ranger21 host slow state");
    let mut writer = BufWriter::with_capacity(CHUNK_VALUES * std::mem::size_of::<f32>(), file);
    let mut bytes = Vec::with_capacity(CHUNK_VALUES * std::mem::size_of::<f32>());
    for (id, values) in map {
        assert!(id.is_ascii() && !id.contains('\n'), "invalid Ranger21 state id");
        writer.write_all(id.as_bytes()).expect("write Ranger21 state id");
        writer.write_all(b"\n").expect("write Ranger21 state separator");
        writer.write_all(&usize::to_le_bytes(values.len())).expect("write Ranger21 state length");
        for chunk in values.chunks(CHUNK_VALUES) {
            bytes.clear();
            for value in chunk {
                bytes.extend_from_slice(&f32::to_le_bytes(*value));
            }
            writer.write_all(&bytes).expect("write Ranger21 host slow values");
        }
    }
    writer.flush().expect("flush Ranger21 host slow state");
}

fn apply_norm_loss<D: Device>(
    weights: &mut DenseMatrix<D>,
    learning_rate: f32,
    params: Ranger21Params,
    placement: NormLossPlacement,
) -> Result<(), OperationError<D::DeviceError>> {
    if params.norm_loss_factor == 0.0 || params.norm_loss_placement != placement {
        return Ok(());
    }
    let mut values = vec![0.0; weights.size()];
    weights.write_to_slice(&mut values)?;
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    let correction = 2.0 * params.norm_loss_factor * (1.0 - 1.0 / (norm + params.eps));
    let scale = 1.0 - learning_rate * correction;
    values.iter_mut().for_each(|value| *value *= scale);
    weights.load_from_slice(None, &values)?;
    Ok(())
}
