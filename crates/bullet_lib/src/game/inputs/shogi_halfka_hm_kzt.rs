//! HalfKA_hm + KZT-FT (King Zone Threat, FT 入力版) 連結特徴量
//! (= 2026-07-04、 KZT を output 加算 head から FT 入力へ昇格する trainer 側配線)。
//!
//! HalfKA_hm (73,305 次元) と KZT-FT (9,440 次元) を連結した sparse input 型を
//! 提供する。 KZT-FT 部分は **1024 accumulator へ流す FT 入力** で、 SCReLU → L1 → L2
//! を通り piece features と非線形に co-adapt する (= threat FT 入力版と同機構)。
//!
//! ## 仕様
//!
//! - HalfKA_hm: 既存 [`ShogiHalfKA_hm`] と完全同一ロジック (= 対称 emit)。
//! - KZT-FT: 推論側 `rshogi-core::nnue::kzt` の active KZT-FT index を **同一 primitive**
//!   (`collect_hand_features` + `kzt_hand_decode` + `kzt_*_index` + `KZT_FT_*_OFFSET`)
//!   で emit する。 これは engine の feature-gated 関数
//!   `for_each_active_kzt_ft_index` の body と byte 同一で、 bit-exact を保証する
//!   (= 下記 [`emit_active_kzt_ft_index`] 参照)。
//!
//! ## なぜ engine 関数を直接 import しないか
//!
//! engine の `for_each_active_kzt_ft_index` は `#[cfg(feature = "nnue-kzt-ft")]` gated。
//! bullet の rshogi-core 依存は既に `nnue-kzt-psqt` を有効化しており、 rshogi-core 側
//! `compile_error!` により `nnue-kzt-ft` と `nnue-kzt-psqt` は同時有効化できない
//! (= KZT 寄与の二重計上防止)。 output 加算 head (`kzt_features.rs`) の並行作業を壊さず、
//! かつ engine (rshogi-core) を触らないため、 gated 関数は import せず **同一 primitive を
//! 使ってその body を忠実に複製** する (= 呼ぶ関数・定数・順序が全て engine と同一なので
//! index 集合は原理的に完全一致)。
//!
//! ## 非対称 emission
//!
//! KZT-FT の active set は perspective (= attacker 側) 依存で非対称:
//! - STM accumulator: attacker = STM の持駒で敵玉 zone を攻める feature
//! - NSTM accumulator: attacker = NSTM の持駒で敵玉 zone を攻める feature
//!
//! 同一局面でも `|STM_active| != |NSTM_active|` が通常なので、 defensive HandThreat と同様
//! [`SparseInputType::map_features_split`] で片側のみを `Some` にして emit する。 これは engine
//! の `kzt_accumulation[stm]` / `kzt_accumulation[nstm]` (= 各 perspective 単一側 feature 総和)
//! と対応する。
//!
//! ## ckpt section (= export)
//!
//! l0 affine の KZT-FT 部分 (feat `[73,305, 82,745)`) を `P2KZTF\0\0` section として
//! 物理末尾に append する (= `examples/shogi_layerstack.rs`)。 engine の
//! `read_kzt_ft_weights` (= i8 raw, row-major `[dim 9,440][L1]`) と一致。

use super::SparseInputType;
use super::shogi_halfka::{HALFKA_HM_DIMENSIONS, MAX_ACTIVE_FEATURES, ShogiHalfKA_hm};
use crate::shogi::{PackedSfenValue, ShogiBoard};

use rshogi_core::nnue::kzt::{
    KZT_FT_BIAS_OFFSET, KZT_FT_CORE_OFFSET, KZT_FT_DIMENSIONS, KZT_FT_MAX_ACTIVE_PER_PERSP, KZT_FT_PERSP_OFFSET,
    KZT_FT_REGION_OFFSET, KZT_HAND_MAX_ACTIVE, KztHandFeature, collect_hand_features, kzt_bias_index, kzt_core_index,
    kzt_hand_decode, kzt_persp_index, kzt_region_index,
};
use rshogi_core::position::Position;
use rshogi_core::types::Color as CoreColor;

// =============================================================================
// 定数
// =============================================================================

/// KZT-FT 連結後の総次元数 (= 73,305 + 9,440 = 82,745)。
pub const HALFKA_HM_KZT_DIMENSIONS: usize = HALFKA_HM_DIMENSIONS + KZT_FT_DIMENSIONS;

/// 1 perspective (= 1 accumulator) あたりの max active。
/// HalfKA_hm (最大 [`MAX_ACTIVE_FEATURES`]) + KZT-FT (最大 [`KZT_FT_MAX_ACTIVE_PER_PERSP`])。
const MAX_ACTIVE_TOTAL: usize = MAX_ACTIVE_FEATURES + KZT_FT_MAX_ACTIVE_PER_PERSP;

// =============================================================================
// bit-exact KZT-FT index emission (= engine for_each_active_kzt_ft_index の複製)
// =============================================================================

/// `attacker` の持駒で敵玉 5x5 zone を攻める KZT active feature を gather し、 各 feature を
/// FT 入力の 4 テーブル成分 (= core/region/persp/bias) に射影して `f` へ **連結空間の** index
/// (`[0, KZT_FT_DIMENSIONS)`) を渡す。
///
/// engine (rshogi-core) の `nnue::kzt::for_each_active_kzt_ft_index` の body を、 同一 primitive
/// (`collect_hand_features` / `kzt_hand_decode` / `kzt_*_index` / `KZT_FT_*_OFFSET`) で忠実に
/// 複製したもの。 呼ぶ関数・定数・emit 順序が engine と完全一致するため、 emit する index 集合は
/// bit-exact に一致する (= `examples/kzt_ft_verify.rs` で 10,000 ケース検証)。
///
/// FT accumulator は bucket 非依存のため、 `collect_hand_features` の bucket 引数は 0 を渡し、
/// dense index の bucket 成分は [`kzt_hand_decode`] で捨てる。
#[inline]
fn emit_active_kzt_ft_index<F: FnMut(usize)>(pos: &Position, attacker: CoreColor, mut f: F) {
    let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
    let n = collect_hand_features(pos, attacker, 0, &mut buf);
    for feat in buf.iter().take(n) {
        let (persp, _bucket, king_region, hand_piece, target_rel, drop_state) = kzt_hand_decode(feat.index);
        f(KZT_FT_CORE_OFFSET + kzt_core_index(hand_piece, target_rel, drop_state));
        f(KZT_FT_REGION_OFFSET + kzt_region_index(king_region, target_rel, drop_state));
        f(KZT_FT_PERSP_OFFSET + kzt_persp_index(persp, hand_piece, drop_state));
        f(KZT_FT_BIAS_OFFSET + kzt_bias_index(drop_state));
    }
}

/// `PackedSfenValue` から rshogi-core `Position` を復元する (= sfen round-trip)。
///
/// KZT feature gather (`collect_hand_features`) が rshogi-core `Position` を要求するため、
/// `kzt_features::collect_kzt_factored_for_side` と同一経路 (= board → sfen → Position) で復元する。
/// 復元に失敗した場合 (= 片玉データ等) は `None` (= KZT feature 無し) を返す。
#[inline]
fn packed_to_position(pos: &PackedSfenValue) -> Option<Position> {
    let board = ShogiBoard::from_packed_sfen(pos);
    let sfen = super::shogi_board_to_sfen(&board);
    let mut position = Position::new();
    match position.set_sfen_with_pass_rights(&sfen, 0, 0) {
        Ok(()) => Some(position),
        Err(_) => None,
    }
}

// =============================================================================
// ShogiHalfKaHmKzt
// =============================================================================

/// HalfKA_hm + KZT-FT 連結特徴量。
///
/// `SparseInputType` を実装し、 HalfKA_hm 特徴量 (対称) と KZT-FT 特徴量 (非対称) を
/// 連結した sparse input として提供する。 l0 affine は `[82,745 → L1]` で構築され、 export
/// 時に KZT-FT 部分 (feat `[73,305, 82,745)`) を `P2KZTF` section へ書き出す。
#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Default)]
pub struct ShogiHalfKaHmKzt;

impl ShogiHalfKaHmKzt {
    /// 構築する (= 状態なし)。
    pub fn new() -> Self {
        Self
    }

    /// KZT-FT 部分の次元数 (= 9,440)。
    pub fn kzt_ft_dimensions(&self) -> usize {
        KZT_FT_DIMENSIONS
    }
}

impl SparseInputType for ShogiHalfKaHmKzt {
    type RequiredDataType = PackedSfenValue;

    fn num_inputs(&self) -> usize {
        HALFKA_HM_KZT_DIMENSIONS
    }

    fn max_active(&self) -> usize {
        MAX_ACTIVE_TOTAL
    }

    /// symmetric API (= 後方互換 fallback)。 KZT-FT 部分は非対称なので symmetric `f()` では
    /// 正確に emit できず、 HalfKA_hm 部分のみを best-effort で emit する。 訓練 loader は
    /// [`SparseInputType::map_features_split`] を優先呼びするため、 通常経路では呼ばれない。
    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, f: F) {
        // HalfKA_hm は対称なので既存ロジックへ委譲。 KZT-FT は非対称のためここでは emit しない。
        ShogiHalfKA_hm.map_features(pos, f);
    }

    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        // HalfKA_hm 部分 (= 対称) は両側 Some で emit。
        ShogiHalfKA_hm.map_features(pos, |stm, nstm| f(Some(stm), Some(nstm)));

        // KZT-FT 部分 (= 非対称)。 STM accumulator は attacker=STM、 NSTM accumulator は
        // attacker=NSTM の KZT feature を受け取る (= engine kzt_accumulation[stm/nstm] と対応)。
        let Some(position) = packed_to_position(pos) else {
            return;
        };
        let stm = position.side_to_move();
        let nstm = stm.opponent();
        emit_active_kzt_ft_index(&position, stm, |idx| f(Some(HALFKA_HM_DIMENSIONS + idx), None));
        emit_active_kzt_ft_index(&position, nstm, |idx| f(None, Some(HALFKA_HM_DIMENSIONS + idx)));
    }

    fn shorthand(&self) -> String {
        format!("shogi-{HALFKA_HM_KZT_DIMENSIONS}x45hm+kztft")
    }

    fn description(&self) -> String {
        format!(
            "Shogi HalfKA_hm ({HALFKA_HM_DIMENSIONS}) + KZT-FT ({KZT_FT_DIMENSIONS}) concatenated (FT input, asymmetric)"
        )
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_dimensions() {
        let input = ShogiHalfKaHmKzt::new();
        assert_eq!(input.num_inputs(), 73_305 + 9_440);
        assert_eq!(input.num_inputs(), 82_745);
        assert_eq!(input.kzt_ft_dimensions(), 9_440);
    }

    #[test]
    fn test_max_active_covers_both_parts() {
        let input = ShogiHalfKaHmKzt::new();
        assert_eq!(input.max_active(), MAX_ACTIVE_FEATURES + KZT_FT_MAX_ACTIVE_PER_PERSP);
        // KZT-FT は 1 hand feature が 4 成分を emit するため、 175 × 4 = 700 が上限。
        assert_eq!(KZT_FT_MAX_ACTIVE_PER_PERSP, KZT_HAND_MAX_ACTIVE * 4);
    }

    #[test]
    fn test_shorthand() {
        let input = ShogiHalfKaHmKzt::new();
        assert_eq!(input.shorthand(), "shogi-82745x45hm+kztft");
    }
}
