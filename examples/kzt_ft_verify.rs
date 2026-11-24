//! KZT-FT (King Zone Threat, FT 入力版) の bit-exact 相互検証 (= 2026-07-04)。
//!
//! 訓練側 `ShogiHalfKaHmKzt::map_features_split` が emit する KZT-FT の active index
//! (= HalfKA 連結 offset 73,305 を引いた連結空間 local index) の **multiset** が、 推論側
//! KZT-FT index 計算 (= `collect_hand_features` → `kzt_hand_decode` → core/region/persp/bias の
//! 4 テーブル射影 + `KZT_FT_*_OFFSET`) と perspective 別 (STM / NSTM accumulator) に
//! **完全一致** することを、 実 PSV 局面で検証する。
//!
//! engine の `for_each_active_kzt_ft_index` は `nnue-kzt-ft` feature gated で、 bullet が依存する
//! `nnue-kzt-psqt` と相互排他 (= compile_error) のため直接呼べない。 そのため trainer 側 helper と
//! この検証 reference は共に engine 関数の body を **同一 primitive** で複製している。 本検証は、
//! trainer の SparseInputType 配線 (= HalfKA offset の付与 / STM↔NSTM の attacker 割当) が正しいことを、
//! 独立に計算した reference multiset と突き合わせて確認する (= 4 成分の multiplicity も含めて一致)。
//!
//! 実行: `cargo run --release --example kzt_ft_verify -- <psv path> [N]`

use std::fs::File;
use std::io::Read;

use bullet_lib::game::inputs::{HALFKA_HM_DIMENSIONS, ShogiHalfKaHmKzt, SparseInputType, shogi_board_to_sfen};
use bullet_lib::shogi::{PackedSfenValue, ShogiBoard};

use rshogi_core::nnue::kzt::{
    KZT_FT_BIAS_OFFSET, KZT_FT_CORE_OFFSET, KZT_FT_DIMENSIONS, KZT_FT_PERSP_OFFSET, KZT_FT_REGION_OFFSET,
    KZT_HAND_MAX_ACTIVE, KztHandFeature, collect_hand_features, kzt_bias_index, kzt_core_index, kzt_hand_decode,
    kzt_persp_index, kzt_region_index,
};
use rshogi_core::position::Position;
use rshogi_core::types::Color as CoreColor;

/// reference: `attacker` の KZT-FT active index (= 連結空間 local) を multiset として集める。
/// engine `for_each_active_kzt_ft_index` の body と同一 primitive・同一順序。
fn reference_kzt_ft_indices(pos: &Position, attacker: CoreColor) -> Vec<usize> {
    let mut buf = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
    let n = collect_hand_features(pos, attacker, 0, &mut buf);
    let mut out = Vec::with_capacity(n * 4);
    for feat in buf.iter().take(n) {
        let (persp, _bucket, king_region, hand_piece, target_rel, drop_state) = kzt_hand_decode(feat.index);
        out.push(KZT_FT_CORE_OFFSET + kzt_core_index(hand_piece, target_rel, drop_state));
        out.push(KZT_FT_REGION_OFFSET + kzt_region_index(king_region, target_rel, drop_state));
        out.push(KZT_FT_PERSP_OFFSET + kzt_persp_index(persp, hand_piece, drop_state));
        out.push(KZT_FT_BIAS_OFFSET + kzt_bias_index(drop_state));
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: kzt_ft_verify <psv> [N]");
    let n_records: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(5000);

    let input = ShogiHalfKaHmKzt::new();

    let mut file = File::open(&path).expect("psv open");
    let mut buf = [0u8; 40];

    let mut checked = 0usize; // board × color の検証数
    let mut mismatches = 0usize;
    let mut nonempty = 0usize;
    let mut total_active = 0usize;
    let mut max_active = 0usize;

    for _ in 0..n_records {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        // SAFETY: PackedSfenValue は #[repr(C)] で [u8;40] 1 個。 40 byte を unaligned read。
        let psv: PackedSfenValue = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const PackedSfenValue) };
        let board = ShogiBoard::from_packed_sfen(&psv);

        // ---- trainer 側: map_features_split の KZT-FT 部分を perspective 別に収集 ----
        // stm accumulator (= 第 1 Option) / nstm accumulator (= 第 2 Option) それぞれで、
        // HalfKA (< 73,305) を除いた KZT-FT index を local (= idx - 73,305) に戻して multiset 化。
        let mut trainer_stm: Vec<usize> = Vec::new();
        let mut trainer_nstm: Vec<usize> = Vec::new();
        input.map_features_split(&psv, |stm_opt, nstm_opt| {
            if let Some(i) = stm_opt {
                if i >= HALFKA_HM_DIMENSIONS {
                    trainer_stm.push(i - HALFKA_HM_DIMENSIONS);
                }
            }
            if let Some(i) = nstm_opt {
                if i >= HALFKA_HM_DIMENSIONS {
                    trainer_nstm.push(i - HALFKA_HM_DIMENSIONS);
                }
            }
        });

        // ---- reference 側: Position を復元し、 stm/nstm attacker の KZT-FT index を計算 ----
        let sfen = shogi_board_to_sfen(&board);
        let mut pos = Position::new();
        let (mut ref_stm, mut ref_nstm) = if pos.set_sfen_with_pass_rights(&sfen, 0, 0).is_ok() {
            let stm = pos.side_to_move();
            let nstm = stm.opponent();
            (reference_kzt_ft_indices(&pos, stm), reference_kzt_ft_indices(&pos, nstm))
        } else {
            (Vec::new(), Vec::new())
        };

        // multiset 比較のため sort。 全 index は連結空間内 (`[0, KZT_FT_DIMENSIONS)`) であること。
        for v in [&mut trainer_stm, &mut trainer_nstm, &mut ref_stm, &mut ref_nstm] {
            v.sort_unstable();
            debug_assert!(v.iter().all(|&i| i < KZT_FT_DIMENSIONS));
        }

        for (side, (t, r)) in [("STM", (&trainer_stm, &ref_stm)), ("NSTM", (&trainer_nstm, &ref_nstm))] {
            checked += 1;
            if t != r {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "MISMATCH sfen='{sfen}' side={side} trainer_n={} ref_n={} (first diff)",
                        t.len(),
                        r.len()
                    );
                }
            }
            if !r.is_empty() {
                nonempty += 1;
                total_active += r.len();
                max_active = max_active.max(r.len());
            }
        }
    }

    println!("\n==== KZT-FT bit-exact verification ====");
    println!("checked (board×side)     = {checked}");
    println!("mismatches               = {mismatches}");
    println!("nonempty emit cases      = {nonempty}");
    println!("total active (ref)       = {total_active}");
    println!(
        "avg active (nonempty)    = {:.2}",
        if nonempty > 0 { total_active as f64 / nonempty as f64 } else { 0.0 }
    );
    println!("max active per (board,c) = {max_active}  (cap {})", KZT_HAND_MAX_ACTIVE * 4);
    if mismatches == 0 {
        println!("RESULT: PASS (trainer map_features_split KZT-FT == reference for all {checked} cases)");
    } else {
        println!("RESULT: FAIL ({mismatches} mismatches)");
        std::process::exit(1);
    }
}
