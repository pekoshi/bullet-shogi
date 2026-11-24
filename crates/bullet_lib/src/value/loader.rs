mod direct;
pub mod hcpe;
mod montybinpack;
mod rng;
mod sfbinpack;
pub mod shogipack;
mod text;
pub mod viribinpack;

pub use direct::{CanBeDirectlySequentiallyLoaded, DirectSequentialDataLoader};
pub use hcpe::{HCPE_RECORD_SIZE, HcpeLoader, looks_like_hcpe_file};
pub use montybinpack::MontyBinpackLoader;
pub use sfbinpack::SfBinpackLoader;
pub use shogipack::ShogiPackLoader;
pub use text::InMemoryTextLoader;
pub use viribinpack::ViriBinpackLoader;

use crate::game::{inputs::SparseInputType, outputs::OutputBuckets};
use acyclib::device::tensor::Shape;
use bulletformat::BulletFormat;

use super::Wgt;

unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::ChessBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::AtaxxBoard {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::CudADFormat {}
unsafe impl CanBeDirectlySequentiallyLoaded for bulletformat::chess::MarlinFormat {}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GameResult {
    Loss = 0,
    Draw = 1,
    Win = 2,
}

pub trait LoadableDataType: Sized {
    fn score(&self) -> i16;

    fn result(&self) -> GameResult;
}

impl<T: BulletFormat + 'static> LoadableDataType for T {
    fn result(&self) -> GameResult {
        [GameResult::Loss, GameResult::Draw, GameResult::Win][self.result_idx()]
    }

    fn score(&self) -> i16 {
        <Self as BulletFormat>::score(self)
    }
}

/// Dictates how data is read from a file into the expected datatype.
/// This allows for the file format to be divorced from the training
/// data format.
pub trait DataLoader<T>: Clone + Send + Sync + 'static {
    fn data_file_paths(&self) -> &[String];

    fn count_positions(&self) -> Option<u64> {
        None
    }

    fn map_batches<F: FnMut(&[T]) -> bool>(&self, start_batch: usize, batch_size: usize, f: F);
}

pub(crate) type B<I> = fn(&<I as SparseInputType>::RequiredDataType, f32) -> f32;

#[derive(Clone)]
pub struct DefaultDataLoader<I: SparseInputType, O, D> {
    input_getter: I,
    output_getter: O,
    blend_getter: B<I>,
    weight_getter: Option<Wgt<I>>,
    use_win_rate_model: bool,
    wdl: bool,
    scale: f32,
    loader: D,
}

impl<I: SparseInputType, O, D> DefaultDataLoader<I, O, D> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        scale: f32,
        loader: D,
    ) -> Self {
        Self { input_getter, output_getter, blend_getter, weight_getter, use_win_rate_model, wdl, scale, loader }
    }
}

impl<I, O, D> DefaultDataLoader<I, O, D>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    D: DataLoader<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    pub fn load_and_map_batches<F: FnMut(&[I::RequiredDataType]) -> bool>(
        &self,
        start_batch: usize,
        batch_size: usize,
        f: F,
    ) {
        self.loader.map_batches(start_batch, batch_size, f);
    }

    pub fn prepare(&self, data: &[I::RequiredDataType], threads: usize, blend: f32) -> PreparedData<I, O> {
        PreparedData::new(
            self.input_getter.clone(),
            self.output_getter,
            self.blend_getter,
            self.weight_getter,
            self.use_win_rate_model,
            self.wdl,
            data,
            threads,
            blend,
            self.scale,
        )
    }
}

pub(crate) struct DenseInput {
    pub value: Vec<f32>,
    pub shape: Shape,
}

#[derive(Clone)]
pub(crate) struct SparseInput {
    pub value: Vec<i32>,
    pub max_active: usize,
    pub shape: Shape,
}

/// A batch of data, in the correct format for the GPU.
pub struct PreparedData<I: SparseInputType, O> {
    pub(crate) input_getter: I,
    pub(crate) output_getter: O,
    pub(crate) batch_size: usize,
    pub(crate) stm: SparseInput,
    pub(crate) nstm: SparseInput,
    pub(crate) buckets: SparseInput,
    pub(crate) targets: DenseInput,
    pub(crate) weights: DenseInput,
    /// HandCount dense auxiliary input。`I::hand_count_dims()` が `Some` のとき設定される。
    pub(crate) hand_count: Option<DenseInput>,
    /// LMAS など hand_count 以外の auxiliary dense 入力。
    ///
    /// `I::auxiliary_dense_dims()` のうち "hand_count" を除いた entries を
    /// `(name, DenseInput)` ペアで列挙し、 graph builder で `new_dense_input(name, ...)`
    /// 宣言した dense 入力へ name 一致で渡される (dataloader.rs)。
    /// 既存 input type は default impl 経由で空 Vec を返すため、 後方互換。
    pub(crate) auxiliary: Vec<(String, DenseInput)>,
    /// 大次元 sparse aux 入力 (= Dual PSQT King-Attacker 等)。
    ///
    /// `I::auxiliary_sparse_dims()` (= (name, dim, max_active)) から構築。 dense と違い
    /// max_active 個の `i32` index を 1 sample あたり書き込む (= 末尾は -1 padding)。
    /// graph builder で `new_sparse_input(name, Shape::new(dim, 1), max_active)` 宣言
    /// した sparse 入力に対応。 既定 (= default impl) は空 Vec で後方互換。
    pub(crate) auxiliary_sparse: Vec<(String, SparseInput)>,
}

impl<I, O> PreparedData<I, O>
where
    I: SparseInputType,
    O: OutputBuckets<I::RequiredDataType>,
    I::RequiredDataType: LoadableDataType,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_getter: I,
        output_getter: O,
        blend_getter: B<I>,
        weight_getter: Option<Wgt<I>>,
        use_win_rate_model: bool,
        wdl: bool,
        data: &[I::RequiredDataType],
        threads: usize,
        blend: f32,
        scale: f32,
    ) -> Self {
        let rscale = 1.0 / scale;
        let batch_size = data.len();
        let max_active = input_getter.max_active();
        let chunk_size = batch_size.div_ceil(threads);
        let input_size = input_getter.num_inputs();
        let output_size = if wdl { 3 } else { 1 };
        let sparse_size = max_active * batch_size;
        let hand_count_dims = input_getter.hand_count_dims();

        let hand_count_init =
            hand_count_dims.map(|dims| DenseInput { value: vec![0.0; dims * batch_size], shape: Shape::new(dims, 1) });

        // auxiliary dense (LMAS など): hand_count を除く全 entry を batch 分確保。
        // 既存 input type は default impl で空 Vec を返すため、 後方互換 (この Vec も空)。
        let aux_dims_all: Vec<(String, usize)> = input_getter.auxiliary_dense_dims();
        let aux_specs: Vec<(String, usize)> =
            aux_dims_all.into_iter().filter(|(name, _)| name != "hand_count").collect();
        let auxiliary_init: Vec<(String, DenseInput)> = aux_specs
            .iter()
            .map(|(name, dims)| {
                (name.clone(), DenseInput { value: vec![0.0; *dims * batch_size], shape: Shape::new(*dims, 1) })
            })
            .collect();

        // auxiliary sparse (Dual PSQT King-Attacker 等): 大次元 + 小 max_active な aux 入力を
        // sparse で確保。 dense 297 MB / batch を回避し NPS 数倍化。 default impl は空 Vec。
        let aux_sparse_specs: Vec<(String, usize, usize)> = input_getter.auxiliary_sparse_dims();
        let auxiliary_sparse_init: Vec<(String, SparseInput)> = aux_sparse_specs
            .iter()
            .map(|(name, dim, max_active_aux)| {
                let total = max_active_aux * batch_size;
                // 末尾は -1 padding (= bullet sparse 慣例)
                (
                    name.clone(),
                    SparseInput { max_active: *max_active_aux, value: vec![-1i32; total], shape: Shape::new(*dim, 1) },
                )
            })
            .collect();

        let mut prep = Self {
            input_getter,
            output_getter,
            batch_size,
            stm: SparseInput { max_active, value: vec![0; sparse_size], shape: Shape::new(input_size, 1) },
            nstm: SparseInput { max_active, value: vec![0; sparse_size], shape: Shape::new(input_size, 1) },
            buckets: SparseInput { max_active: 1, value: vec![0; batch_size], shape: Shape::new(O::BUCKETS, 1) },
            targets: DenseInput { value: vec![0.0; output_size * batch_size], shape: Shape::new(output_size, 1) },
            weights: DenseInput { value: vec![0.0; batch_size], shape: Shape::new(1, 1) },
            hand_count: hand_count_init,
            auxiliary: auxiliary_init,
            auxiliary_sparse: auxiliary_sparse_init,
        };

        let sparse_chunk_size = max_active * chunk_size;

        // HandCount 用の並列チャンクを事前に materialise。Option は並列ループ内で扱う。
        let hand_count_dim = hand_count_dims.unwrap_or(0);
        let hand_count_chunk_size = hand_count_dim * chunk_size;
        let num_chunks = batch_size.div_ceil(chunk_size);
        let hand_count_slices: Vec<Option<&mut [f32]>> = if let Some(hc) = prep.hand_count.as_mut() {
            hc.value.chunks_mut(hand_count_chunk_size).map(Some).collect()
        } else {
            (0..num_chunks).map(|_| None).collect()
        };

        std::thread::scope(|s| {
            data.chunks(chunk_size)
                .zip(prep.stm.value.chunks_mut(sparse_chunk_size))
                .zip(prep.nstm.value.chunks_mut(sparse_chunk_size))
                .zip(prep.buckets.value.chunks_mut(chunk_size))
                .zip(prep.targets.value.chunks_mut(output_size * chunk_size))
                .zip(prep.weights.value.chunks_mut(chunk_size))
                .zip(hand_count_slices)
                .for_each(
                    |(
                        (((((data_chunk, stm_chunk), nstm_chunk), buckets_chunk), results_chunk), weights_chunk),
                        hand_count_chunk,
                    )| {
                        let inp = &prep.input_getter;
                        let out = &prep.output_getter;
                        s.spawn(move || {
                            let chunk_len = data_chunk.len();
                            let mut hand_count_chunk = hand_count_chunk;

                            for i in 0..chunk_len {
                                let pos = &data_chunk[i];

                                if let Some(hc_slice) = hand_count_chunk.as_deref_mut() {
                                    let offset = hand_count_dim * i;
                                    let end = offset + hand_count_dim;
                                    // 事前に 0 で埋め済み。fill_hand_count は
                                    // 書き込みのみで読まないので再初期化は不要。
                                    inp.fill_hand_count(pos, &mut hc_slice[offset..end]);
                                }
                                // STM と NSTM は独立カウンタで管理: 非対称 feature
                                // (HandThreat defensive 等) で |STM_active| != |NSTM_active|
                                // を許可するため。symmetric な input type は
                                // map_features_split の default impl 経由で
                                // 両側同時に進むので従来挙動と一致する。
                                let mut j_stm: usize = 0;
                                let mut j_nstm: usize = 0;
                                let sparse_offset = max_active * i;

                                inp.map_features_split(pos, |our_opt, opp_opt| {
                                    if let Some(our) = our_opt {
                                        assert!(our < input_size, "STM feature index exceeded input size!");
                                        stm_chunk[sparse_offset + j_stm] = our as i32;
                                        j_stm += 1;
                                    }
                                    if let Some(opp) = opp_opt {
                                        assert!(opp < input_size, "NSTM feature index exceeded input size!");
                                        nstm_chunk[sparse_offset + j_nstm] = opp as i32;
                                        j_nstm += 1;
                                    }
                                });

                                // STM / NSTM の未使用スロットを -1 で埋める (独立)
                                for j in j_stm..max_active {
                                    stm_chunk[sparse_offset + j] = -1;
                                }
                                for j in j_nstm..max_active {
                                    nstm_chunk[sparse_offset + j] = -1;
                                }

                                assert!(
                                    j_stm <= max_active && j_nstm <= max_active,
                                    "More inputs provided than the specified maximum!"
                                );

                                buckets_chunk[i] = i32::from(out.bucket(pos));
                                weights_chunk[i] = weight_getter.map_or(1.0, |w| w(pos));

                                if wdl {
                                    results_chunk[output_size * i + usize::from(pos.result() as u8)] = 1.0;
                                } else {
                                    let score = f32::from(pos.score());
                                    let score = if use_win_rate_model {
                                        let p = (score - 270.0) / 380.0;
                                        let pm = (-score - 270.0) / 380.0;
                                        0.5 * (1.0 + sigmoid(p) - sigmoid(pm))
                                    } else {
                                        sigmoid(rscale * score)
                                    };
                                    let result = f32::from(pos.result() as u8) / 2.0;
                                    let blend = blend_getter(pos, blend);
                                    assert!((0.0..=1.0).contains(&blend), "WDL proportion must be in [0, 1]");
                                    results_chunk[i] = blend * result + (1. - blend) * score;
                                }
                            }
                        });
                    },
                );
        });

        // auxiliary dense fill: hand_count 以外の dense 入力を name 一致で埋める。
        // 既存 hand_count 経路と独立に動かす。 entries が空 (default impl) のときは
        // この block は実質 no-op。
        //
        // 各 (name, DenseInput) は独立しているため entry 単位で並列化、 entry 内では
        // batch を chunk_size で割って並列化。
        if !prep.auxiliary.is_empty() {
            let inp_aux = &prep.input_getter;
            std::thread::scope(|s| {
                for (name, dense) in prep.auxiliary.iter_mut() {
                    let dims = dense.shape.size();
                    let aux_chunk_size = dims * chunk_size;
                    let name_ref: &str = name.as_str();
                    for (data_chunk, aux_chunk) in data.chunks(chunk_size).zip(dense.value.chunks_mut(aux_chunk_size)) {
                        s.spawn(move || {
                            for (i, pos) in data_chunk.iter().enumerate() {
                                let offset = dims * i;
                                let end = offset + dims;
                                // 事前に 0 で埋め済み。 fill_auxiliary_dense は書き込みのみ。
                                inp_aux.fill_auxiliary_dense(name_ref, pos, &mut aux_chunk[offset..end]);
                            }
                        });
                    }
                }
            });
        }

        // auxiliary sparse fill: dim が大 (= 18,144 等) で max_active が小 (= 50) な aux 入力を
        // sparse 経路で埋める。 dense ベース 297 MB/batch の DMA を回避し NPS を数倍化。
        // 既定 (= 空 Vec) では no-op。 各 sample で fill_auxiliary_sparse が active 数 n を返し、
        // n..max_active_aux の末尾 slot は init 時の -1 padding がそのまま残る。
        if !prep.auxiliary_sparse.is_empty() {
            let inp_aux = &prep.input_getter;
            std::thread::scope(|s| {
                for (name, sparse) in prep.auxiliary_sparse.iter_mut() {
                    let max_active_aux = sparse.max_active;
                    let aux_chunk_elems = max_active_aux * chunk_size;
                    let name_ref: &str = name.as_str();
                    for (data_chunk, aux_chunk) in data.chunks(chunk_size).zip(sparse.value.chunks_mut(aux_chunk_elems))
                    {
                        s.spawn(move || {
                            for (i, pos) in data_chunk.iter().enumerate() {
                                let offset = max_active_aux * i;
                                let end = offset + max_active_aux;
                                // 末尾は init 時に -1 で埋め済み。 fill_auxiliary_sparse は
                                // active index を先頭から書き込み、 戻り値の n は active 個数。
                                let _n = inp_aux.fill_auxiliary_sparse(name_ref, pos, &mut aux_chunk[offset..end]);
                            }
                        });
                    }
                }
            });
        }

        prep
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
