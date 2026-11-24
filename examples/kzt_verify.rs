//! KZT_HAND factored v1 の bit-exact 相互検証 (= 2026-06-30、 2026-07-01 factored 化)。
//!
//! 訓練側 `collect_kzt_factored_for_side` が展開する 5 テーブル (core/region/bucket/persp/bias)
//! の sparse index 集合を、 既知の乱数 weight net に対して合算した値が、 engine 側 factored
//! evaluate (= `collect_hand_features` → `kzt_hand_decode` → `KztFactoredWeights::feature_score`)
//! の per-side head 合計と **完全一致** することを、 実 PSV 局面で数値検証する。 gather
//! (`collect_hand_features`) は不変なので、 5 テーブルへの分解式の一致が検証対象。
//!
//! 実行: `cargo run --release --example kzt_verify -- <psv path> [N] [bucket]`

use std::fs::File;
use std::io::Read;

use bullet_lib::game::inputs::{collect_kzt_factored_for_side, shogi_board_to_sfen};
use bullet_lib::shogi::{PackedSfenValue, ShogiBoard};

use rshogi_core::nnue::kzt::{
    KZT_FACTORED_COUNT, KZT_HAND_MAX_ACTIVE, KztFactoredWeights, KztHandFeature, collect_hand_features, kzt_hand_decode,
};
use rshogi_core::position::Position;
use rshogi_core::types::Color as CoreColor;

/// 決定的な擬似乱数 flat weight (= 各テーブル固定順、 値域 -400..=400)。
fn make_reference_weights() -> Box<KztFactoredWeights> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut flat = vec![0i16; KZT_FACTORED_COUNT];
    for v in flat.iter_mut() {
        // xorshift64。
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *v = ((state % 801) as i32 - 400) as i16;
    }
    KztFactoredWeights::from_flat(&flat)
}

/// engine 側: collect_hand_features + decode + factored feature_score の per-side 合計。
fn engine_side_sum(w: &KztFactoredWeights, sfen: &str, color: CoreColor, bucket: usize) -> (i64, usize) {
    let mut pos = Position::new();
    pos.set_sfen_with_pass_rights(sfen, 0, 0).expect("sfen 受理");
    let mut feats = [KztHandFeature { index: 0 }; KZT_HAND_MAX_ACTIVE];
    let n = collect_hand_features(&pos, color, bucket, &mut feats);
    let mut acc: i64 = 0;
    for f in feats.iter().take(n) {
        let (persp, bk, kr, hp, tr, ds) = kzt_hand_decode(f.index);
        acc += w.feature_score(persp, bk, kr, hp, tr, ds) as i64;
    }
    (acc, n)
}

/// 訓練側: collect_kzt_factored_for_side が emit した 5 テーブル local index を weight に当てて合計。
fn trainer_side_sum(w: &KztFactoredWeights, board: &ShogiBoard, color: CoreColor, bucket: usize) -> (i64, usize) {
    let a = collect_kzt_factored_for_side(board, color, bucket);
    let mut acc: i64 = 0;
    for i in 0..a.n {
        acc += w.core[a.core[i] as usize] as i64
            + w.region[a.region[i] as usize] as i64
            + w.bucket[a.bucket[i] as usize] as i64
            + w.persp[a.persp[i] as usize] as i64
            + w.bias[a.bias[i] as usize] as i64;
    }
    (acc, a.n)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: kzt_verify <psv> [N] [bucket]");
    let n_records: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let bucket: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(4);

    let weights = make_reference_weights();

    let mut file = File::open(&path).expect("psv open");
    let mut buf = [0u8; 40];

    let mut checked = 0usize;
    let mut mismatches = 0usize;
    let mut nonempty = 0usize;
    let mut total_active = 0usize;
    let mut max_active = 0usize;
    let mut checksum: i64 = 0;
    let mut samples_printed = 0usize;

    for _ in 0..n_records {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        // SAFETY: PackedSfenValue は #[repr(C)] で [u8;40] 1 個。 40 byte を unaligned read。
        let psv: PackedSfenValue = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const PackedSfenValue) };
        let board = ShogiBoard::from_packed_sfen(&psv);
        let sfen = shogi_board_to_sfen(&board);

        for color in [CoreColor::Black, CoreColor::White] {
            let (t_sum, t_n) = trainer_side_sum(&weights, &board, color, bucket);
            let (e_sum, e_n) = engine_side_sum(&weights, &sfen, color, bucket);
            checked += 1;
            checksum = checksum.wrapping_add(e_sum);
            if t_sum != e_sum || t_n != e_n {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "MISMATCH sfen='{sfen}' color={color:?} trainer=(sum={t_sum}, n={t_n}) engine=(sum={e_sum}, n={e_n})"
                    );
                }
            }
            if e_n > 0 {
                nonempty += 1;
                total_active += e_n;
                max_active = max_active.max(e_n);
                if samples_printed < 6 {
                    println!("SAMPLE sfen='{sfen}' color={color:?} active={e_n} head_sum={e_sum}");
                    samples_printed += 1;
                }
            }
        }
    }

    println!("\n==== KZT factored bit-exact verification ====");
    println!("checked (board×color)    = {checked}");
    println!("mismatches               = {mismatches}");
    println!("nonempty emit cases      = {nonempty}");
    println!("total active features    = {total_active}");
    println!(
        "avg active (nonempty)    = {:.2}",
        if nonempty > 0 { total_active as f64 / nonempty as f64 } else { 0.0 }
    );
    println!("max active per (board,c) = {max_active}  (cap {KZT_HAND_MAX_ACTIVE})");
    println!("engine head_sum checksum = {checksum}");
    if mismatches == 0 {
        println!("RESULT: PASS (trainer factored expand == engine factored evaluate for all {checked} cases)");
    } else {
        println!("RESULT: FAIL ({mismatches} mismatches)");
        std::process::exit(1);
    }
}
