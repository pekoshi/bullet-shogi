//! 訓練済み quantised.bin から KZT_HAND factored weight の per-table / per-axis L2 を報告する
//! (= 2026-07-01)。
//!
//! P2KZT section (= version 2、 10,336 i16 = core→region→bucket→persp→bias) を seek + parse し、
//! engine 側 `rshogi-core::nnue::kzt::kzt_weight_stats` で per-table / per-axis L2 / max_abs を出力。
//!
//! 実行: `cargo run --release --example kzt_stats -- <quantised.bin>`

use std::fs;

use rshogi_core::nnue::kzt::{KZT_FACTORED_COUNT, KZT_VERSION, KztFactoredWeights, kzt_weight_stats};

fn main() {
    let path = std::env::args().nth(1).expect("usage: kzt_stats <quantised.bin>");
    let bytes = fs::read(&path).expect("read quantised.bin");

    let magic = b"P2KZT\0\0\0";
    let off = bytes.windows(8).position(|w| w == magic).unwrap_or_else(|| panic!("P2KZT section not found in {path}"));

    let mut p = off + 8;
    let version = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
    p += 4;
    let count = u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as usize;
    p += 4;
    assert_eq!(version, KZT_VERSION, "expected factored version {KZT_VERSION}, got {version}");
    assert_eq!(count, KZT_FACTORED_COUNT, "count mismatch: {count} != {KZT_FACTORED_COUNT}");

    let mut flat = vec![0i16; KZT_FACTORED_COUNT];
    for v in flat.iter_mut() {
        *v = i16::from_le_bytes([bytes[p], bytes[p + 1]]);
        p += 2;
    }
    let w = KztFactoredWeights::from_flat(&flat);
    let s = kzt_weight_stats(&w);

    let names = ["core", "region", "bucket", "persp", "bias"];
    println!("==== KZT factored weight stats: {path} ====");
    for i in 0..5 {
        println!("  table {:<7} L2={:>10.3}  max|w|={}", names[i], s.table_l2[i], s.table_max_abs[i]);
    }
    println!("  by_drop_state_l2  = {:?}", round1(&s.by_drop_state_l2));
    println!("  by_hand_piece_l2  = {:?} (歩香桂銀金角飛)", round1(&s.by_hand_piece_l2));
    println!("  by_target_rel_l2  = {:?}", round1(&s.by_target_rel_l2));
    println!("  by_bucket_l2      = {:?}", round1(&s.by_bucket_l2));
    println!("  by_king_region_l2 = {:?}", round1(&s.by_king_region_l2));
}

fn round1(a: &[f64]) -> Vec<f64> {
    a.iter().map(|v| (v * 10.0).round() / 10.0).collect()
}
