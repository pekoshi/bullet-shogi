mod activity_risk_features;
mod adapter;
mod ataxx147;
mod attack_dev_features;
mod attack_relay_features;
mod attack_relay_v2_features;
mod chess768;
mod chess_buckets;
mod chess_buckets_mk;
mod drop_pressure_features;
mod endgame_tempo_features;
mod factorised;
mod king_danger_features;
mod king_escape_features;
mod king_mobility_features;
mod kzt_features;
mod shogi_halfka;
mod shogi_halfka_hm_aux;
mod shogi_halfka_hm_auxcheap;
mod shogi_halfka_hm_hand_count;
mod shogi_halfka_hm_hand_threat;
mod shogi_halfka_hm_hand_threat_defensive;
mod shogi_halfka_hm_king_attacker_dual;
mod shogi_halfka_hm_kzt;
mod shogi_halfka_hm_pin;
mod shogi_halfka_hm_pin_only;
mod shogi_halfka_hm_policy;
mod shogi_halfka_hm_policy_soft;
#[path = "inputs/shogi_halfka_hm_threat_profile11split.rs"]
mod shogi_halfka_hm_threat;
mod shogi_halfka_hm_threat_direct15;
mod shogi_halfkp;
pub mod shogi_kh; // 2026-11-12 KH ブロック (玉近傍の空升 x 敵の利き / 浮き駒)
mod shogi_mob; // B6 可動域
pub mod shogi_threat_exclusion;
mod slider_xray_features;

#[allow(deprecated)]
mod legacy;

pub use activity_risk_features::{
    ACTIVITY_RISK_ARCH_MARKER, ACTIVITY_RISK_DIM_PER_PERSP, ACTIVITY_RISK_MAX_ACTIVE_PER_PERSP,
    ACTIVITY_RISK_NUM_ACTIVITY, ACTIVITY_RISK_NUM_BUCKETS, ACTIVITY_RISK_NUM_PRESSURE, ACTIVITY_RISK_NUM_SHIELD,
    ACTIVITY_RISK_PER_PB_DIM, ACTIVITY_RISK_QA, ACTIVITY_RISK_SECTION_VERSION, ACTIVITY_RISK_SPARSE_NAME_BLACK,
    ACTIVITY_RISK_SPARSE_NAME_WHITE, ACTIVITY_RISK_SUB_MAGIC, ACTIVITY_RISK_TOTAL_COUNT, ACTIVITY_RISK_V3_BASE_DIM,
    ACTIVITY_RISK_V3_NUM_ESCAPE, ACTIVITY_RISK_V3_NUM_HAND_CHECK, ACTIVITY_RISK_V3_NUM_HEAVY,
    ACTIVITY_RISK_V3_TACTICAL_DIM_PER_PERSP, ACTIVITY_RISK_V3_TACTICAL_NAME_BLACK,
    ACTIVITY_RISK_V3_TACTICAL_NAME_WHITE, ACTIVITY_RISK_V3_TOTAL_COUNT, activity_risk_v3_tactical_local_idx,
    activity_to_bucket, compute_features as compute_activity_risk_features, compute_tactical_features_v3,
    flatten_idx as activity_risk_flatten_idx, local_idx_in_persp as activity_risk_local_idx, pressure_to_bucket,
    shield_to_bucket, shogi_board_to_sfen,
};
// B6 可動域 (2026-09-29)。 3 実装照合は examples/shogi_mob_canonical.rs。
pub use shogi_mob::{mob_all, mob_diff};
// 案 B (Regime LayerStack、 2026-10-07): V1 を outputs.rs の bucket 算出から使う。
//   🚨pub(crate) の item なので `pub use` でなく `pub(crate) use` にする。
pub use adapter::MarlinFormatAdapter;
pub use ataxx147::{Ataxx98, Ataxx147};
pub use attack_dev_features::{
    ATTACK_DEV_ARCH_MARKER, ATTACK_DEV_DIM_PER_PERSP, ATTACK_DEV_MAX_ACTIVE_PER_PERSP, ATTACK_DEV_NUM_BUCKETS,
    ATTACK_DEV_NUM_CLASS, ATTACK_DEV_NUM_PERSP, ATTACK_DEV_QA, ATTACK_DEV_SECTION_VERSION,
    ATTACK_DEV_SPARSE_NAME_BLACK, ATTACK_DEV_SPARSE_NAME_WHITE, ATTACK_DEV_SUB_MAGIC, ATTACK_DEV_TOTAL_COUNT,
    compute_class_for_persp as attack_dev_compute_class, local_idx as attack_dev_local_idx,
};
pub use attack_relay_features::{
    ARH_ARCH_MARKER, ARH_DIM_PER_PERSP, ARH_MAX_ACTIVE_PER_PERSP, ARH_NUM_BUCKETS, ARH_NUM_CONTACT, ARH_NUM_PERSP,
    ARH_NUM_RELAY, ARH_NUM_RESERVE, ARH_QA, ARH_SECTION_VERSION, ARH_SPARSE_NAME_BLACK, ARH_SPARSE_NAME_WHITE,
    ARH_SUB_MAGIC, ARH_TOTAL_COUNT, compute_classes_for_persp as attack_relay_compute_classes,
    local_idx as attack_relay_local_idx,
};
pub use attack_relay_v2_features::{
    ARH_V2_ARCH_MARKER, ARH_V2_MAX_ACTIVE_PER_PERSP, ARH_V2_NUM_ALPHA, ARH_V2_NUM_ATTACKERS, ARH_V2_NUM_BALANCE,
    ARH_V2_NUM_BUCKETS, ARH_V2_NUM_HEAVY, ARH_V2_NUM_OVERLAP, ARH_V2_NUM_PERSP, ARH_V2_NUM_PHASE, ARH_V2_NUM_SUPPORTED,
    ARH_V2_OFFSET_W_ALPHA, ARH_V2_OFFSET_W_ALPHA_SHAPE, ARH_V2_OFFSET_W_BASE, ARH_V2_OFFSET_W_HEAVY,
    ARH_V2_OFFSET_W_PHASE, ARH_V2_QA, ARH_V2_SECTION_VERSION, ARH_V2_SPARSE_NAME_BLACK, ARH_V2_SPARSE_NAME_WHITE,
    ARH_V2_SUB_MAGIC, ARH_V2_W_ALPHA_COUNT, ARH_V2_W_ALPHA_SHAPE_COUNT, ARH_V2_W_BASE_COUNT, ARH_V2_W_HEAVY_COUNT,
    ARH_V2_W_PHASE_COUNT, ARH_V2_WEIGHT_COUNT, ArhV2Classes, active_indices as attack_relay_v2_active_indices,
    bucket_to_phase as attack_relay_v2_bucket_to_phase, compute_classes_for_persp as attack_relay_v2_compute_classes,
    w_alpha_disk_idx as attack_relay_v2_w_alpha_disk_idx,
    w_alpha_shape_disk_idx as attack_relay_v2_w_alpha_shape_disk_idx,
    w_base_disk_idx as attack_relay_v2_w_base_disk_idx, w_heavy_disk_idx as attack_relay_v2_w_heavy_disk_idx,
    w_phase_disk_idx as attack_relay_v2_w_phase_disk_idx,
};
pub use chess_buckets::{ChessBuckets, ChessBucketsMirrored};
pub use chess768::Chess768;
pub use drop_pressure_features::{
    DROP_PRESSURE_ARCH_MARKER, DROP_PRESSURE_DIM_PER_PERSP, DROP_PRESSURE_MAX_ACTIVE_PER_PERSP,
    DROP_PRESSURE_NUM_BUCKETS, DROP_PRESSURE_NUM_CLASS, DROP_PRESSURE_NUM_PERSP, DROP_PRESSURE_QA,
    DROP_PRESSURE_SECTION_VERSION, DROP_PRESSURE_SPARSE_NAME_BLACK, DROP_PRESSURE_SPARSE_NAME_WHITE,
    DROP_PRESSURE_SUB_MAGIC, DROP_PRESSURE_TOTAL_COUNT, compute_class_for_persp as drop_pressure_compute_class,
    local_idx as drop_pressure_local_idx,
};
pub use endgame_tempo_features::{
    ENDGAME_TEMPO_ARCH_MARKER, ENDGAME_TEMPO_DIM_PER_PERSP, ENDGAME_TEMPO_MAX_ACTIVE_PER_PERSP,
    ENDGAME_TEMPO_NUM_BUCKETS, ENDGAME_TEMPO_NUM_CLASS, ENDGAME_TEMPO_QA, ENDGAME_TEMPO_SECTION_VERSION,
    ENDGAME_TEMPO_SPARSE_NAME_BLACK, ENDGAME_TEMPO_SPARSE_NAME_WHITE, ENDGAME_TEMPO_SUB_MAGIC,
    ENDGAME_TEMPO_TOTAL_COUNT, compute_class_for_persp as endgame_tempo_compute_class,
    local_idx as endgame_tempo_local_idx,
};
pub use factorised::{Factorised, Factorises};
pub use king_danger_features::{
    KING_DANGER_ARCH_MARKER, KING_DANGER_DIM_PER_PERSP, KING_DANGER_MAX_ACTIVE_PER_PERSP, KING_DANGER_NUM_BUCKETS,
    KING_DANGER_NUM_CLASS, KING_DANGER_NUM_PERSP, KING_DANGER_QA, KING_DANGER_SECTION_VERSION,
    KING_DANGER_SPARSE_NAME_BLACK, KING_DANGER_SPARSE_NAME_WHITE, KING_DANGER_SUB_MAGIC, KING_DANGER_TOTAL_COUNT,
    compute_class_for_persp as king_danger_compute_class, local_idx as king_danger_local_idx,
};
pub use king_escape_features::{
    KING_ESCAPE_ARCH_MARKER, KING_ESCAPE_DIM_PER_PERSP, KING_ESCAPE_MAX_ACTIVE_PER_PERSP, KING_ESCAPE_NUM_BUCKETS,
    KING_ESCAPE_NUM_CLASS, KING_ESCAPE_NUM_PERSP, KING_ESCAPE_QA, KING_ESCAPE_SECTION_VERSION,
    KING_ESCAPE_SPARSE_NAME_BLACK, KING_ESCAPE_SPARSE_NAME_WHITE, KING_ESCAPE_SUB_MAGIC, KING_ESCAPE_TOTAL_COUNT,
    compute_class_for_persp as king_escape_compute_class, local_idx as king_escape_local_idx,
};
pub use king_mobility_features::{
    KING_MOBILITY_ARCH_MARKER, KING_MOBILITY_DIM_PER_PERSP, KING_MOBILITY_MAX_ACTIVE_PER_PERSP,
    KING_MOBILITY_NUM_BUCKETS, KING_MOBILITY_NUM_CLASS, KING_MOBILITY_NUM_PERSP, KING_MOBILITY_QA,
    KING_MOBILITY_SECTION_VERSION, KING_MOBILITY_SPARSE_NAME_BLACK, KING_MOBILITY_SPARSE_NAME_WHITE,
    KING_MOBILITY_SUB_MAGIC, KING_MOBILITY_TOTAL_COUNT, compute_class_for_persp as king_mobility_compute_class,
    local_idx as king_mobility_local_idx,
};
pub use kzt_features::{
    KZT_ARCH_MARKER, KZT_BIAS_DIM, KZT_BUCKET_DIM, KZT_CORE_DIM, KZT_MAX_ACTIVE_PER_PERSP, KZT_NUM_TABLES,
    KZT_PERSP_DIM, KZT_QA, KZT_REGION_DIM, KZT_SECTION_VERSION, KZT_SUB_MAGIC, KZT_TABLE_BLACK_NAMES, KZT_TABLE_DIMS,
    KZT_TABLE_LR_REL, KZT_TABLE_WD_REL, KZT_TABLE_WEIGHT_NAMES, KZT_TABLE_WHITE_NAMES, KZT_TOTAL_COUNT,
    KztFactoredActive, collect_kzt_factored_for_side,
};
pub use shogi_halfka::{
    FEATURE_HASH_HM_V2, FEATURE_HASH_NONMIRROR, HALFKA_HM_DIMENSIONS, HALFKA_HM_K16_DIMENSIONS, NUM_KING_BUCKETS_K16,
    PIECE_INPUTS, ShogiHalfKA, ShogiHalfKA_hm, ShogiHalfKA_hmK16, ShogiPieceFactor, halfka_k16_index_from_k45,
    king_bucket16,
};
pub(crate) use shogi_halfka_hm_threat::board_effect_v1;
pub use shogi_halfka_hm_threat::{
    MOB_ARCH_MARKER, MOB_INPUT_SCALE, MOB_INPUT_SHIFT, MOB_NAME, MOB_QA, MOB_SECTION_VERSION, MOB_SUB_MAGIC,
    MOB_WEIGHT_COUNT, THREAT_PROJ_ARCH_MARKER, THREAT_PROJ_QA, THREAT_PROJ_SECTION_VERSION, THREAT_PROJ_SUB_MAGIC,
};

/// HalfKA_hm + piece-square factorizer (nnue-pytorch 相当)。
///
/// 訓練時のみ玉位置非依存の 1,629 次元 factor を付与し、 export で実特徴へ coalesce
/// する。 推論形状は `ShogiHalfKA_hm` (73,305) のまま不変。
pub type ShogiHalfKAFactorised = Factorised<ShogiHalfKA_hm, ShogiPieceFactor>;
pub use shogi_halfka_hm_aux::{
    AUX_ARCH_MARKER, AUX_DIM_PER_PERSP, AUX_MAX_ACTIVE_PER_PERSP, AUX_NUM_CLASSES, AUX_QA, AUX_SECTION_VERSION,
    AUX_SUB_MAGIC, HALFKA_HM_AUX_DIMENSIONS, HANG_DIM, HANG_OFFSET, KZ_DIM, KZ_OFFSET, STATKD_DIM, STATKD_OFFSET,
    ShogiHalfKaHmAux, emit_aux_active_index,
};
pub use shogi_halfka_hm_auxcheap::{
    AUXCHEAP_ARCH_MARKER, AUXCHEAP_DIM_PER_PERSP, AUXCHEAP_HG_DIM, AUXCHEAP_HG_OFFSET, AUXCHEAP_KZ_DIM,
    AUXCHEAP_KZ_OFFSET, AUXCHEAP_MAX_ACTIVE_PER_PERSP, AUXCHEAP_NUM_CLASSES, AUXCHEAP_OUT_DIM, AUXCHEAP_OUT_OFFSET,
    AUXCHEAP_SECTION_VERSION, AUXCHEAP_STATKD_DIM, AUXCHEAP_STATKD_OFFSET, AUXCHEAP_SUB_MAGIC,
    HALFKA_HM_AUXCHEAP_DIMENSIONS, ShogiHalfKaHmAuxCheap, emit_auxcheap_active_index,
};
pub use shogi_halfka_hm_hand_count::{HAND_COUNT_DIMS, ShogiHalfKaHmHandCount};
pub use shogi_halfka_hm_hand_threat::ShogiHalfKaHmHandThreat;
pub use shogi_halfka_hm_hand_threat_defensive::ShogiHalfKaHmHandThreatDefensive;
pub use shogi_halfka_hm_king_attacker_dual::{
    DUAL_ARCH_MARKER, DUAL_AUX_NAME_BLACK, DUAL_AUX_NAME_WHITE, DUAL_CROSS_DEFENDED_BLACK, DUAL_CROSS_DEFENDED_WHITE,
    DUAL_CROSS_DIM_PER_BUCKET, DUAL_CROSS_DIM_PER_HEAD, DUAL_CROSS_GEOMETRY_BLACK, DUAL_CROSS_GEOMETRY_CLASSES,
    DUAL_CROSS_GEOMETRY_DIM_PER_PERSP, DUAL_CROSS_GEOMETRY_WHITE, DUAL_CROSS_MAX_ACTIVE_PER_HEAD,
    DUAL_CROSS_THREAT_BLACK, DUAL_CROSS_THREAT_WHITE, DUAL_DIM_PER_HEAD, DUAL_DIMENSIONS_PER_PERSPECTIVE,
    DUAL_MAX_ACTIVE_PER_HEAD, DUAL_MAX_ACTIVE_PER_SIDE, DUAL_MAX_PER_PIECE, DUAL_PIECE_TYPES, DUAL_SECTION_VERSION,
    DUAL_SPARSE_NAME_ATTACK_BLACK, DUAL_SPARSE_NAME_ATTACK_WHITE, DUAL_SPARSE_NAME_DEFEND_BLACK,
    DUAL_SPARSE_NAME_DEFEND_WHITE, DUAL_SQUARES, DUAL_SUB_MAGIC, DUAL_TOTAL_COUNT, KINGRING_ARCH_MARKER,
    KINGRING_DIM_PER_PERSP, KINGRING_MAX_ACTIVE_PER_PERSP, KINGRING_NUM_BUCKETS, KINGRING_NUM_PIECE, KINGRING_NUM_SQ,
    KINGRING_NUM_STATE, KINGRING_PER_BUCKET_DIM, KINGRING_QA, KINGRING_SECTION_VERSION, KINGRING_SPARSE_NAME_BLACK,
    KINGRING_SPARSE_NAME_WHITE, KINGRING_SUB_MAGIC, KINGRING_TOTAL_COUNT, NUM_DUAL_PSQT_BUCKETS, NUM_DUAL_PSQT_HEADS,
    PAIRSHAPE_ARCH_MARKER, PAIRSHAPE_DIM_PER_PERSP, PAIRSHAPE_MAX_ACTIVE_PER_PERSP, PAIRSHAPE_NUM_BUCKETS,
    PAIRSHAPE_NUM_PAIR_TYPES, PAIRSHAPE_NUM_SHAPES, PAIRSHAPE_PER_BUCKET_DIM, PAIRSHAPE_QA, PAIRSHAPE_SECTION_VERSION,
    PAIRSHAPE_SPARSE_NAME_BLACK, PAIRSHAPE_SPARSE_NAME_WHITE, PAIRSHAPE_SUB_MAGIC, PAIRSHAPE_TOTAL_COUNT,
    ShogiHalfKaHmKingAttackerDual, collect_dual_active_for_side, collect_dual_active_sparse_for_side,
    collect_dual_cross_for_side, collect_kingring_active_for_side, dual_aux_entries, dual_cross_index,
    dual_feature_index, dual_sparse_index, fill_dual_features_for_side, kingring_encode_local, try_fill_dual_aux,
};
pub use shogi_halfka_hm_kzt::{HALFKA_HM_KZT_DIMENSIONS, ShogiHalfKaHmKzt};
pub use shogi_halfka_hm_pin::{
    PIN_ARCH_MARKER, PIN_AUX_NAME_BLACK, PIN_AUX_NAME_WHITE, PIN_DIMENSIONS_PER_PERSPECTIVE, PIN_NUM_BUCKETS,
    PIN_PIECE_TYPES, PIN_QA, PIN_SECTION_VERSION, PIN_SQUARES, PIN_STATES, PIN_SUB_MAGIC, fill_pin_features_for_side,
    pin_aux_entries, pin_feature_index, try_fill_pin_aux,
};
pub use shogi_halfka_hm_pin_only::ShogiHalfKaHmPin;
pub use shogi_halfka_hm_policy::{POLICY_TARGET_DIMS, POLICY_TARGET_NAME, ShogiHalfKaHmPolicy, make_move_label};
pub use shogi_halfka_hm_policy_soft::{POLICY_VISIT_DIST_NAME, ShogiHalfKaHmPolicySoft};
pub use shogi_halfka_hm_threat::{
    MAX_ACTIVE_THREAT_FEATURES, SCALAR_THREAT_BUCKETS, SCALAR_THREAT_DIMENSIONS, SCALAR_THREAT_SPARSE_NAME_BLACK,
    SCALAR_THREAT_SPARSE_NAME_WHITE, SCALAR_THREAT_WEIGHT_COUNT, ShogiHalfKaHmThreat, ShogiHalfKaHmThreatFactor16,
    THREAT_EXT_GMASK_NAME, THREAT_EXT_SPARSE_NSTM, THREAT_EXT_SPARSE_STM, THREAT_FACTOR_DIM, THREAT_FACTOR_ROWS,
    THREAT_FACTOR_SPARSE_NAME_NSTM, THREAT_FACTOR_SPARSE_NAME_STM, THREAT_FT_SPARSE_NSTM, THREAT_FT_SPARSE_STM,
    THREAT_GMASK_NAME, THREAT_HANG_ROWS, fill_scalar_threat_for_perspective, threat_hang_base_map, threat_pair_layout,
    threat_tactical_stats,
};
pub use shogi_halfka_hm_threat_direct15::{
    Direct15HeadShape, MAX_ACTIVE_DIRECT15_FEATURES, ShogiHalfKaHmThreatDirect15, THREAT_DIRECT15_ATTACKS_PER_CLASS,
    THREAT_DIRECT15_CLASS_NB, THREAT_DIRECT15_DEFENDED_DIMENSIONS, THREAT_DIRECT15_DIMENSIONS,
    THREAT_DIRECT15_KING_CLASS, THREAT_DIRECT15_SPARSE_NSTM, THREAT_DIRECT15_SPARSE_STM, direct15_pair_of_feature,
    expand_direct15_head_bias, expand_direct15_head_weights,
};
pub use shogi_halfkp::{FEATURE_HASH, ShogiHalfKP};
pub use shogi_threat_exclusion::ThreatProfile;
pub use slider_xray_features::{
    SLIDER_XRAY_ARCH_MARKER, SLIDER_XRAY_DIM_PER_PERSP, SLIDER_XRAY_MAX_ACTIVE_PER_PERSP, SLIDER_XRAY_NUM_BUCKETS,
    SLIDER_XRAY_NUM_CLASS, SLIDER_XRAY_NUM_PERSP, SLIDER_XRAY_QA, SLIDER_XRAY_SECTION_VERSION,
    SLIDER_XRAY_SPARSE_NAME_BLACK, SLIDER_XRAY_SPARSE_NAME_WHITE, SLIDER_XRAY_SUB_MAGIC, SLIDER_XRAY_TOTAL_COUNT,
    compute_class_for_persp as slider_xray_compute_class, local_idx as slider_xray_local_idx,
};

#[allow(deprecated)]
pub use chess_buckets_mk::*;

#[allow(deprecated)]
pub use legacy::InputType;

#[deprecated(note = "See `examples/progression/3_input_buckets.rs` for a faster alternative to this.")]
pub type ChessBucketsFactorised = Factorised<ChessBuckets, Chess768>;

#[allow(deprecated)]
impl ChessBucketsFactorised {
    pub fn new(buckets: [usize; 64]) -> Self {
        Self::from_parts(ChessBuckets::new(buckets), Chess768)
    }
}

#[deprecated(note = "See `examples/progression/3_input_buckets.rs` for a faster alternative to this.")]
pub type ChessBucketsMirroredFactorised = Factorised<ChessBucketsMirrored, Chess768>;

#[allow(deprecated)]
impl ChessBucketsMirroredFactorised {
    pub fn new(buckets: [usize; 32]) -> Self {
        Self::from_parts(ChessBucketsMirrored::new(buckets), Chess768)
    }
}

pub trait SparseInputType: Clone + Send + Sync + 'static {
    type RequiredDataType: Send + Sync;

    /// The total number of inputs
    fn num_inputs(&self) -> usize;

    /// The maximum number of active inputs
    fn max_active(&self) -> usize;

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F);

    /// Asymmetric feature emission.
    ///
    /// Each call to `f(stm_opt, nstm_opt)` activates at most one index on each
    /// perspective independently. `None` means "no feature on this side for this
    /// call", allowing asymmetric active sets (|STM| != |NSTM|).
    ///
    /// The default implementation delegates to the symmetric `map_features`,
    /// emitting `(Some(stm), Some(nstm))` for every call. Existing symmetric
    /// input types do not need to override this method — they remain fully
    /// backward compatible.
    ///
    /// Input types that need asymmetric features (e.g. HandThreat defensive)
    /// must override this method and leave `map_features` calling a single-
    /// purpose symmetric emission (or panic).
    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.map_features(pos, |stm, nstm| f(Some(stm), Some(nstm)));
    }

    /// Shorthand for the input e.g. `768x4`
    fn shorthand(&self) -> String;

    /// Description of the input type
    fn description(&self) -> String;

    /// Dense auxiliary input (`"hand_count"`) の次元数。
    ///
    /// `Some(dims)` を返すと、`PreparedData` は `hand_count` 用の dense 領域を
    /// 確保し、サンプルごとに `fill_hand_count` を呼び出してトレーナ graph の
    /// `"hand_count"` 入力ノードに渡す。
    ///
    /// 既定は `None`（HandCount dense 入力を使用しない）。
    fn hand_count_dims(&self) -> Option<usize> {
        None
    }

    /// Dense auxiliary input (`"hand_count"`) をサンプル分埋める。
    ///
    /// `out.len()` は `hand_count_dims().unwrap_or(0)` と一致する。
    /// `hand_count_dims() == None` の input type では呼び出されないため既定は空実装。
    fn fill_hand_count(&self, _pos: &Self::RequiredDataType, _out: &mut [f32]) {}

    /// 補助 dense 入力 (name, dims) 一覧。
    ///
    /// 汎用化された API。 graph builder で `new_dense_input(name, Shape::new(dims, 1))`
    /// として参照される dense 入力をすべて列挙する。 Pin-aware のような複数 dense head
    /// を持つ input type はこれを override する。
    ///
    /// 既定は `hand_count_dims()` の値があれば `("hand_count", dims)` 単独を返す
    /// 後方互換 wrapper。 既存 input type は変更不要。
    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        match self.hand_count_dims() {
            Some(dims) => vec![("hand_count".to_string(), dims)],
            None => Vec::new(),
        }
    }

    /// 補助 dense 入力を name で分岐して fill する。
    ///
    /// 既定は `name == "hand_count"` のとき既存の `fill_hand_count` に
    /// delegate する。 Pin-aware のような複数 head は match で分岐させる。
    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        if name == "hand_count" {
            self.fill_hand_count(pos, out);
        }
    }

    /// 補助 sparse 入力 (name, dim, max_active) 一覧。
    ///
    /// 大次元 (= 18k 等) で max_active が小さい (= 50 等) input は sparse 経路で
    /// 渡すと NPS が劇的に改善する (= dense path の 297 MB / batch DMA 回避)。
    /// graph builder 側で `new_sparse_input(name, Shape::new(dim, 1), max_active)` 宣言
    /// した sparse 入力に対応する。
    ///
    /// 既定は空 Vec。 大次元 aux を持つ input type のみ override する。
    fn auxiliary_sparse_dims(&self) -> Vec<(String, usize, usize)> {
        Vec::new()
    }

    /// 補助 sparse 入力を name で分岐して fill する。
    ///
    /// `out.len()` は `max_active` (= `auxiliary_sparse_dims` 第 3 要素) と一致する。
    /// active index を先頭から書き込み、 末尾の未使用 slot は **書き換えない** (= caller
    /// 側で -1 padding 済前提)。 戻り値は実際に書き込んだ active 個数。
    ///
    /// 既定は no-op で 0 を返す。
    fn fill_auxiliary_sparse(&self, _name: &str, _pos: &Self::RequiredDataType, _out: &mut [i32]) -> usize {
        0
    }

    fn is_factorised(&self) -> bool {
        false
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        assert!(self.is_factorised());
        unmerged
    }
}

pub const fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    let mut i = 0;

    while i < N {
        if arr[i] > max {
            max = arr[i];
        }

        i += 1;
    }
    max + 1
}
