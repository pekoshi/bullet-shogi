//! Attack Relay Head v2 (= 構造化 980 weight 仕様、 2026-06-10 改訂)。
//!
//! 旧 v2 (= 1024 cell 直積、 attackers × overlap × supported × persp × bucket)
//! を **完全置換** する新仕様の sparse 表現。 推論側 `rshogi-core::nnue::
//! attack_relay_v2` と完全一致する index 契約を保つ。
//!
//! # モデル
//! ```text
//! side_score
//!     = W_base[a][b][o][s]                576
//!     + W_alpha[alpha][a][b]              180
//!     + W_phase[phase][a][b]              144
//!     + W_alpha_shape[alpha][o][s]         80
//! 合計                                   = 980 i16
//! ```
//! 先後共有 weight。 per persp で最大 4 active index (= 4 テーブル各 1)。
//! `alpha == 0` / `phase == 0` の基準スライスは activate しない (= 学習除外)。
//!
//! # 経路 (= 既存 KED / ARH v1 と同 pattern、 SFEN delegation)
//! 1. bullet 側 `ShogiBoard` を SFEN string に変換
//! 2. `rshogi-core::Position::set_sfen` で Position 構築
//! 3. `rshogi-core::nnue::attack_relay_v2::compute_arh_v2_classes` で classes
//! 4. `rshogi-core::nnue::attack_relay_v2::w_*_disk_idx` で disk 全体 index

use crate::shogi::packed_sfen::ShogiBoard;
pub use rshogi_core::nnue::attack_relay_v2::{
    ARH_V2_INTRUDE_MAX_ACTIVE, ARH_V2_NUM_ALPHA, ARH_V2_NUM_ATTACKERS, ARH_V2_NUM_BALANCE, ARH_V2_NUM_BUCKETS,
    ARH_V2_NUM_HEAVY, ARH_V2_NUM_OVERLAP, ARH_V2_NUM_PHASE, ARH_V2_NUM_SUPPORTED, ARH_V2_OFFSET_W_ALPHA,
    ARH_V2_OFFSET_W_ALPHA_SHAPE, ARH_V2_OFFSET_W_BASE, ARH_V2_OFFSET_W_HEAVY, ARH_V2_OFFSET_W_PHASE, ARH_V2_VERSION,
    ARH_V2_W_ALPHA_COUNT, ARH_V2_W_ALPHA_SHAPE_COUNT, ARH_V2_W_BASE_COUNT, ARH_V2_W_HEAVY_COUNT, ARH_V2_W_PHASE_COUNT,
    ARH_V2_WEIGHT_COUNT, ArhV2Classes, bucket_to_phase, intrude_disk_indices, w_alpha_disk_idx, w_alpha_shape_disk_idx,
    w_base_disk_idx, w_heavy_disk_idx, w_phase_disk_idx,
};

/// 先後で重みを共有するため persp 別 sparse は不要、 1 つの sparse input を共有。
pub const ARH_V2_NUM_PERSP: usize = 2;

/// 1 局面 1 perspective での最大 active 数 (= version 14 侵入 head: 6 軸 固定、 余裕含め 32)。
pub const ARH_V2_MAX_ACTIVE_PER_PERSP: usize = ARH_V2_INTRUDE_MAX_ACTIVE;

/// version 14 侵入 head: board + perspective + bucket から敵陣侵入 / 前進歩 / 持駒枚数 / 入玉残点の flat index を `out` へ。
/// 推論側 `intrude_disk_indices` と同一 (= bit-exact、 6 軸 intrusion/promo/support/adv_pawn/hand/ek)。
pub fn intrude_active(board: &ShogiBoard, persp_is_black: bool, bucket: usize, out: &mut [u32]) -> usize {
    use rshogi_core::position::Position;
    use rshogi_core::types::Color;
    let mut pos = Position::new();
    pos.set_sfen(&board.to_sfen()).expect("valid sfen");
    let persp = if persp_is_black { Color::Black } else { Color::White };
    let phase = bucket_to_phase(bucket);
    intrude_disk_indices(&pos, persp, phase, out)
}

/// ckpt sub-section magic (= 旧 P2AREL2 と非互換、 構造化 980 形式の識別子)。
pub const ARH_V2_SUB_MAGIC: &[u8; 8] = b"P2ARS1\0\0";
/// ckpt section バージョン (= 13、 推論側 ARH_V2_VERSION を再 export)。
pub const ARH_V2_SECTION_VERSION: u32 = ARH_V2_VERSION;
/// 量子化 QA (= 推論側 OUTPUT_SCALE 既定 4 と整合)。
pub const ARH_V2_QA: f64 = 127.0;
/// ckpt 内 architecture marker (= 旧 ARHv2=1 とは別文字列で識別)。
pub const ARH_V2_ARCH_MARKER: &str = "ARHv2-struct=1";
/// sparse input 名 (= bullet graph 内で参照、 stm / nstm でラベル分離)。
pub const ARH_V2_SPARSE_NAME_BLACK: &str = "attack_relay_v2_black";
pub const ARH_V2_SPARSE_NAME_WHITE: &str = "attack_relay_v2_white";

/// 局面 + 攻撃側 perspective から 5 特徴クラスを計算する。
pub fn compute_classes_for_persp(board: &ShogiBoard, persp_is_black: bool) -> ArhV2Classes {
    use rshogi_core::nnue::attack_relay_v2::compute_arh_v2_classes;
    use rshogi_core::position::Position;
    use rshogi_core::types::Color;
    let sfen = board.to_sfen();
    let mut pos = Position::new();
    pos.set_sfen(&sfen).expect("valid sfen");
    let color = if persp_is_black { Color::Black } else { Color::White };
    compute_arh_v2_classes(&pos, color)
}

/// 与えられた classes / bucket に対して active な disk index を `out` に push する。
///
/// - 戻り値: 実 push 件数 (= 1..=4)
/// - 基準スライス (= `alpha == 0` または `phase == 0`) は push しない (= 学習除外)
/// - 推論側と同一の disk index を返すため、 学習側 / 推論側で重みが直接共有される
pub fn active_indices(classes: &ArhV2Classes, bucket: usize, out: &mut [u32; ARH_V2_MAX_ACTIVE_PER_PERSP]) -> usize {
    let mut n = 0;
    // W_base は常に active。
    out[n] = w_base_disk_idx(classes.attackers, classes.balance, classes.overlap, classes.supported) as u32;
    n += 1;
    // W_alpha は alpha != 0 の場合のみ active (= 基準スライス除外)。
    if classes.alpha != 0 {
        out[n] = w_alpha_disk_idx(classes.alpha, classes.attackers, classes.balance) as u32;
        n += 1;
    }
    // W_phase は phase != 0 の場合のみ active (= 基準スライス除外)。
    let phase = bucket_to_phase(bucket);
    if phase != 0 {
        out[n] = w_phase_disk_idx(phase, classes.attackers, classes.balance) as u32;
        n += 1;
    }
    // W_alpha_shape も alpha != 0 のみ active。
    if classes.alpha != 0 {
        out[n] = w_alpha_shape_disk_idx(classes.alpha, classes.overlap, classes.supported) as u32;
        n += 1;
    }
    // W_heavy は基準スライス無し (= 常時 active、 heavy==0 も有効)。
    out[n] = w_heavy_disk_idx(classes.heavy, classes.balance) as u32;
    n += 1;
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_sanity() {
        assert_eq!(ARH_V2_WEIGHT_COUNT, 1016);
        assert_eq!(ARH_V2_W_BASE_COUNT, 576);
        assert_eq!(ARH_V2_W_ALPHA_COUNT, 180);
        assert_eq!(ARH_V2_W_PHASE_COUNT, 144);
        assert_eq!(ARH_V2_W_ALPHA_SHAPE_COUNT, 80);
        assert_eq!(ARH_V2_W_HEAVY_COUNT, 36);
        assert_eq!(ARH_V2_OFFSET_W_BASE, 0);
        assert_eq!(ARH_V2_OFFSET_W_ALPHA, 576);
        assert_eq!(ARH_V2_OFFSET_W_PHASE, 756);
        assert_eq!(ARH_V2_OFFSET_W_ALPHA_SHAPE, 900);
        assert_eq!(ARH_V2_OFFSET_W_HEAVY, 980);
        assert_eq!(ARH_V2_SUB_MAGIC, b"P2ARS1\0\0");
        assert_eq!(ARH_V2_SECTION_VERSION, 14);
    }

    #[test]
    fn active_indices_skip_baseline() {
        // alpha == 0 + phase == 0 (= bucket 0..=1) -> W_base + W_heavy (= 常時)。
        let classes = ArhV2Classes { attackers: 2, balance: 5, overlap: 1, supported: 1, alpha: 0, heavy: 1 };
        let mut out = [0u32; ARH_V2_MAX_ACTIVE_PER_PERSP];
        let n = active_indices(&classes, 0, &mut out);
        assert_eq!(n, 2);
        assert_eq!(out[0] as usize, w_base_disk_idx(2, 5, 1, 1));
        assert_eq!(out[1] as usize, w_heavy_disk_idx(1, 5));
    }

    #[test]
    fn active_indices_full() {
        // alpha != 0 + phase != 0 -> base + alpha + phase + alpha_shape + heavy = 5 active。
        let classes = ArhV2Classes { attackers: 3, balance: 7, overlap: 2, supported: 2, alpha: 2, heavy: 2 };
        let mut out = [0u32; ARH_V2_MAX_ACTIVE_PER_PERSP];
        let n = active_indices(&classes, 5, &mut out); // phase = 2
        assert_eq!(n, 5);
        assert_eq!(out[0] as usize, w_base_disk_idx(3, 7, 2, 2));
        assert_eq!(out[1] as usize, w_alpha_disk_idx(2, 3, 7));
        assert_eq!(out[2] as usize, w_phase_disk_idx(2, 3, 7));
        assert_eq!(out[3] as usize, w_alpha_shape_disk_idx(2, 2, 2));
        assert_eq!(out[4] as usize, w_heavy_disk_idx(2, 7));
    }

    #[test]
    fn active_indices_only_phase() {
        // alpha == 0、 phase != 0 -> W_base + W_phase + W_heavy = 3 active。
        let classes = ArhV2Classes { attackers: 1, balance: 4, overlap: 0, supported: 0, alpha: 0, heavy: 0 };
        let mut out = [0u32; ARH_V2_MAX_ACTIVE_PER_PERSP];
        let n = active_indices(&classes, 6, &mut out); // phase = 3
        assert_eq!(n, 3);
        assert_eq!(out[0] as usize, w_base_disk_idx(1, 4, 0, 0));
        assert_eq!(out[1] as usize, w_phase_disk_idx(3, 1, 4));
        assert_eq!(out[2] as usize, w_heavy_disk_idx(0, 4));
    }
}
