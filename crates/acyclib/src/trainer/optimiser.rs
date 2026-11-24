pub mod adam;
pub mod clip;
pub mod decay;
pub mod radam;
pub mod ranger;
pub mod ranger21;
pub mod utils;

use std::{collections::HashMap, fmt::Debug, marker::PhantomData, sync::Arc};

use crate::{
    device::{Device, OperationError, tensor::DenseMatrix},
    graph::{CheckpointShapeAdapter, Graph, GraphNodeId, GraphNodeIdTy, NoShapeAdapter, like::GraphLike},
};

pub trait OptimiserState<D: Device>: Sized {
    type Params: Clone + Debug + Default;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError>;

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>>;

    fn reset(&mut self) -> Result<(), D::DeviceError>;

    /// velocity (= 二次モーメント) buffer を指定した正の定数で埋める。
    ///
    /// `--resume-from-quantised` のように optimiser state が zero init のまま
    /// 訓練を再開すると、 Adam/RAdam の更新分母 `sqrt(v) + eps` が `eps`(=1e-8) に
    /// 退化し、 初手の更新が `lr * grad / 1e-8` で爆発する (= eval 出力 ~10x 崩壊)。
    /// velocity を小さい正の定数 (例 1e-4) で初期化すると分母が `sqrt(1e-4)=0.01`
    /// となり爆発を防げる。
    ///
    /// velocity を持たない optimiser state では no-op。
    fn init_velocity(&mut self, _value: f32) -> Result<(), D::DeviceError> {
        Ok(())
    }

    /// checkpoint から optimiser state を読み込む。
    ///
    /// `adapter` は重みと同形のバッファ (momentum / velocity / slow など) の要素数が
    /// graph と一致しないときに呼ばれ、 graph 側の形状へ変換する (weights.bin と同じ
    /// 変換を state にも適用するため)。 変換不要なら [`NoShapeAdapter`] を渡す。
    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>>;

    /// **勾配を持たない重み (真の凍結) を許す版**。  2026-11-03
    ///
    /// `skip` に入っている id は optimiser 状態を持たないので、 checkpoint の file に
    /// その id の状態が残っていても **無視する** (panic しない)。
    /// 既定実装は `skip` を見ずに従来の `load_from_checkpoint` を呼ぶので、
    /// 上書きしない optimiser の挙動は**完全に不変**。
    /// 🚨`BULLET_SKIP_UNKNOWN_WEIGHTS=1` で代用しないこと (本物の不一致まで黙る)。
    fn load_from_checkpoint_with_skip(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
        _skip: &std::collections::HashSet<String>,
    ) -> Result<(), OperationError<D::DeviceError>> {
        Self::load_from_checkpoint(map, path, old_format, adapter)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError>;

    fn set_params(&mut self, params: Self::Params);

    /// 1 本の tensor の中で区間ごとに lr 倍率を変える (`(start, end, 倍率)` の列、 [0, size) を隙間なく覆う)。
    /// tensor 全体の倍率 (`Optimiser::set_lr_scale`) にさらに掛かる。 対応していない optimiser では panic。
    fn set_range_lr_scales(&mut self, _ranges: Vec<(usize, usize, f32, f32)>) {
        panic!("this optimiser does not support per-range lr scales")
    }
}

pub struct Optimiser<D: Device, G: GraphLike<D>, S: OptimiserState<D>> {
    phantom: PhantomData<D>,
    pub graph: G,
    pub state: HashMap<String, S>,
    pre_update: Vec<Box<dyn AdditionalUpdate<D>>>,
    post_update: Vec<Box<dyn AdditionalUpdate<D>>>,
    /// Per-tensor learning rate multiplier. 重み id ごとに lr に掛ける倍率を保持する。
    /// 未登録の id は 1.0 (= 既存挙動) として扱う。
    /// 注意: Adam/RAdam では `lr` は weight decay 項にも掛かるため、 倍率を大きくすると
    /// weight decay も同倍率で強化される。 sparse embedding を boost したい場合は
    /// `set_params_for_weight` で対象 tensor の decay を 0 にしておくこと。
    lr_scale: HashMap<String, f32>,
    /// **勾配を持たない重み** (真の凍結) の id。  2026-11-03
    ///   `requires_grad=false` の leaf は compile が Gradients tensor を作らないので、
    ///   optimiser 状態を確保しない。 その id をここに覚えておき、 setter と
    ///   checkpoint の読み込みで panic しないようにする。
    stateless: std::collections::HashSet<String>,
}

pub trait AdditionalUpdate<D: Device> {
    fn apply_update(&mut self, graph: &mut Graph<D>) -> Result<(), OperationError<D::DeviceError>>;
}

impl<D: Device, G: GraphLike<D>, S: OptimiserState<D>> Optimiser<D, G, S> {
    pub fn new(graph: G, params: S::Params) -> Result<Self, D::DeviceError> {
        let weight_ids = graph.primary().weight_ids();

        let mut state = HashMap::new();

        let mut stateless = std::collections::HashSet::new();

        for id in &weight_ids {
            let idx = graph.primary().weight_idx(id).unwrap();
            let w = graph.primary().get(GraphNodeId::new(idx, GraphNodeIdTy::Values)).unwrap();
            let w = w.dense();
            assert!(w.batch_size().is_none());
            let size = w.size();

            // **勾配ノードが無い重みは optimiser 状態を作らない** (真の凍結)。  2026-11-03
            //   既定の graph では全 weight が requires_grad=true なのでここは発火しない
            //   = 既存の挙動と bit-exact。
            if graph.primary().get(GraphNodeId::new(idx, GraphNodeIdTy::Gradients)).is_err() {
                println!("optimiser: {id} has no gradient node; no optimiser state allocated ({size} params)");
                stateless.insert(id.clone());
                continue;
            }

            let single = S::new(graph.primary().device(), size, params.clone())?;

            let old = state.insert(id.clone(), single);
            assert!(old.is_none());
        }

        Ok(Self {
            phantom: PhantomData,
            graph,
            state,
            stateless,
            pre_update: Vec::new(),
            post_update: Vec::new(),
            lr_scale: HashMap::new(),
        })
    }

    /// 指定 weight id の learning rate を `scale` 倍する。 `scale == 1.0` を指定する (または
    /// 未呼び出しの) tensor は既存挙動と完全一致。
    ///
    /// 用途例: arm 11 KingRing PSQT head のように **sparse 更新で平均的に学習が遅い** tensor を
    /// boost したい場合。 ただし RAdam の weight decay は `lr` 比例なので、 scale > 1 を入れる
    /// なら同時に `set_params_for_weight(id, RangerParams { decay: 0.0, .. })` で decay を切ること。
    pub fn set_lr_scale(&mut self, id: &str, scale: f32) {
        assert!(scale > 0.0 && scale.is_finite(), "lr_scale must be positive finite, got {scale}");
        self.lr_scale.insert(id.to_string(), scale);
    }

    /// 指定 weight id の tensor の中で、 区間ごとに lr 倍率を変える (`set_lr_scale` の倍率にさらに掛かる)。
    /// 例: threat 表を群ごとに 1 本の tensor に連結している場合の群別 lr。
    pub fn set_range_lr_scales(&mut self, id: &str, ranges: Vec<(usize, usize, f32, f32)>) {
        // 真の凍結された重みは状態を持たないので、 落とさずに 1 行出して何もしない。
        if self.stateless.contains(id) {
            println!("optimiser: set_range_lr_scales({id}) ignored (frozen weight has no state)");
            return;
        }
        self.state.get_mut(id).unwrap_or_else(|| panic!("unknown weight id {id}")).set_range_lr_scales(ranges);
    }

    /// 真の凍結された重み (勾配も optimiser 状態も持たない) の id。
    pub fn stateless_ids(&self) -> &std::collections::HashSet<String> {
        &self.stateless
    }

    /// `set_lr_scale` で登録された倍率を返す (未登録は 1.0)。 デバッグ・テスト用。
    pub fn lr_scale_of(&self, id: &str) -> f32 {
        self.lr_scale.get(id).copied().unwrap_or(1.0)
    }

    pub fn add_pre_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.pre_update.push(Box::new(additional));
    }

    pub fn add_post_update(&mut self, additional: impl AdditionalUpdate<D> + 'static) {
        self.post_update.push(Box::new(additional));
    }

    pub fn update(&mut self, gradient_factor: f32, learning_rate: f32) -> Result<(), OperationError<D::DeviceError>> {
        for additional in &mut self.pre_update {
            additional.apply_update(self.graph.primary_mut())?;
        }

        for id in &self.graph.primary().weight_ids() {
            let idx = self.graph.primary().weight_idx(id).unwrap();
            let weight_id = GraphNodeId::new(idx, GraphNodeIdTy::Values);

            let weights = &mut self.graph.primary().get(weight_id)?;
            // 🚨state の取得は**勾配の分岐の中**で行う。 真の凍結された重みは state を
            //   持たないので、 外で unwrap すると落ちる。
            let grad_id = GraphNodeId::new(idx, GraphNodeIdTy::Gradients);
            if let (Ok(grads), Some(single)) = (self.graph.primary().get(grad_id), self.state.get_mut(id)) {
                self.graph.reduce_sum_into_first(&self.graph.get_all(grad_id)?)?;
                let scale = self.lr_scale.get(id).copied().unwrap_or(1.0);
                let effective_lr = learning_rate * scale;
                single.update(&mut *weights.dense_mut(), &mut *grads.dense_mut(), gradient_factor, effective_lr)?;
                self.graph.scatter_first_into_rest(&self.graph.get_all(weight_id)?)?;
            }
        }

        for additional in &mut self.post_update {
            additional.apply_update(self.graph.primary_mut())?;
        }

        Ok(())
    }

    pub fn reset_state(&mut self) -> Result<(), D::DeviceError> {
        for single in self.state.values_mut() {
            single.reset()?;
        }

        Ok(())
    }

    /// 全 weight tensor の velocity buffer を指定した正の定数で初期化する。
    /// `--resume-from-quantised` の velocity=0 爆発回避用 (詳細は
    /// `OptimiserState::init_velocity` を参照)。
    pub fn init_velocity(&mut self, value: f32) -> Result<(), D::DeviceError> {
        for single in self.state.values_mut() {
            single.init_velocity(value)?;
        }

        Ok(())
    }

    pub fn set_params_for_weight(&mut self, id: &str, params: S::Params) {
        if self.stateless.contains(id) {
            println!("optimiser: set_params_for_weight({id}) ignored (frozen weight has no state)");
            return;
        }
        self.state.get_mut(id).unwrap().set_params(params);
    }

    pub fn set_params(&mut self, params: S::Params) {
        for id in self.graph.primary().weight_ids() {
            self.set_params_for_weight(&id, params.clone());
        }
    }

    pub fn write_to_checkpoint(&self, path: &str) -> Result<(), D::DeviceError> {
        self.graph.primary().write_to_file(&format!("{path}/weights.bin"));
        let map = self.state.iter().map(|(id, single)| (id.clone(), single)).collect();
        S::write_to_checkpoint(&map, path)
    }

    pub fn load_from_checkpoint(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, false, &NoShapeAdapter)
    }

    /// [`Self::load_from_checkpoint`] と同じだが、 要素数が graph と一致しないテンソル
    /// (重み・optimiser state とも) を `adapter` で graph 側の形状へ変換して読み込む。
    pub fn load_from_checkpoint_adapted(
        &mut self,
        path: &str,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, false, adapter)
    }

    pub fn load_weights_from_file(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, false, &NoShapeAdapter)
    }

    /// [`Self::load_weights_from_file`] と同じだが、 要素数が graph と一致しない重みを
    /// `adapter` で graph 側の形状へ変換して読み込む。
    pub fn load_weights_from_file_adapted(
        &mut self,
        path: &str,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, false, adapter)
    }

    pub fn load_from_old_format_checkpoint(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_from_checkpoint_(path, true, &NoShapeAdapter)
    }

    pub fn load_old_format_weights_from_file(&mut self, path: &str) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(path, true, &NoShapeAdapter)
    }

    fn load_weights_from_file_(
        &mut self,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.graph.primary_mut().load_from_file_adapted(path, old_format, adapter)?;

        let primary = self.graph.primary();

        for id in primary.weight_ids() {
            let idx = GraphNodeId::new(primary.weight_idx(&id).unwrap(), GraphNodeIdTy::Values);
            self.graph.scatter_first_into_rest(&self.graph.get_all(idx)?)?;
        }

        Ok(())
    }

    fn load_from_checkpoint_(
        &mut self,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.load_weights_from_file_(&format!("{path}/weights.bin"), old_format, adapter)?;
        let mut map = self.state.iter_mut().map(|(id, single)| (id.clone(), single)).collect();
        S::load_from_checkpoint_with_skip(&mut map, path, old_format, adapter, &self.stateless)
    }
}

pub struct WrapOptimiser<O, P> {
    optimiser: O,
    phantom_data: PhantomData<P>,
}

impl<D, O, P> OptimiserState<D> for WrapOptimiser<O, P>
where
    D: Device,
    O: OptimiserState<D>,
    P: Clone + Default + Debug + Into<O::Params>,
{
    type Params = P;

    fn new(device: Arc<D>, size: usize, params: Self::Params) -> Result<Self, D::DeviceError> {
        Ok(Self { optimiser: O::new(device, size, params.into())?, phantom_data: PhantomData })
    }

    fn update(
        &mut self,
        weights: &mut DenseMatrix<D>,
        grads: &mut DenseMatrix<D>,
        gradient_factor: f32,
        learning_rate: f32,
    ) -> Result<(), OperationError<D::DeviceError>> {
        self.optimiser.update(weights, grads, gradient_factor, learning_rate)
    }

    fn reset(&mut self) -> Result<(), D::DeviceError> {
        self.optimiser.reset()
    }

    fn init_velocity(&mut self, value: f32) -> Result<(), D::DeviceError> {
        self.optimiser.init_velocity(value)
    }

    fn set_params(&mut self, params: Self::Params) {
        self.optimiser.set_params(params.into());
    }

    fn load_from_checkpoint(
        map: &mut HashMap<String, &mut Self>,
        path: &str,
        old_format: bool,
        adapter: &dyn CheckpointShapeAdapter,
    ) -> Result<(), OperationError<D::DeviceError>> {
        let mut map = map.iter_mut().map(|(id, single)| (id.clone(), &mut single.optimiser)).collect();
        O::load_from_checkpoint(&mut map, path, old_format, adapter)
    }

    fn write_to_checkpoint(map: &HashMap<String, &Self>, path: &str) -> Result<(), D::DeviceError> {
        let map = map.iter().map(|(id, single)| (id.clone(), &single.optimiser)).collect();
        O::write_to_checkpoint(&map, path)
    }
}
