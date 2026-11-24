//! ARH v2 構造化 (= zone 拡張 version 2、 2026-06-11) の軸分布計測。
//!
//! PSV file の各 record を ShogiBoard 経由 SFEN へ変換、
//! `rshogi_core::nnue::attack_relay_v2::compute_arh_v2_classes` で
//! attackers / balance / overlap / supported / alpha の各 class 分布を集計。
//!
//! 使い方:
//!   cargo run --release --example arh_v2_alpha_dist -- <psv_file> [n_records] [ply_min]
//!
//! 例:
//!   cargo run --release --example arh_v2_alpha_dist -- /data/dataset/suisho5_entering_king/kifu.tag=... 100000 80

use std::{collections::HashMap, fs::File, io::Read};

use bullet_lib::shogi::PackedSfenValue;
use rshogi_core::nnue::attack_relay_v2::{
    ARH_V2_NUM_ALPHA, ARH_V2_NUM_ATTACKERS, ARH_V2_NUM_BALANCE, ARH_V2_NUM_OVERLAP, ARH_V2_NUM_SUPPORTED,
    compute_arh_v2_classes,
};
use rshogi_core::position::Position;
use rshogi_core::types::Color;

fn main() {
    let path = std::env::args().nth(1).expect("usage: arh_v2_alpha_dist <psv> [n] [ply_min]");
    let n_records: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let ply_min: i32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);

    let mut file = File::open(&path).expect("open PSV");
    let mut buf = [0u8; 40];

    let mut alpha = [0u64; ARH_V2_NUM_ALPHA];
    let mut attackers = [0u64; ARH_V2_NUM_ATTACKERS];
    let mut balance = [0u64; ARH_V2_NUM_BALANCE];
    let mut overlap = [0u64; ARH_V2_NUM_OVERLAP];
    let mut supported = [0u64; ARH_V2_NUM_SUPPORTED];
    let mut ply_hist: HashMap<i32, u64> = HashMap::new();
    let mut count: u64 = 0;
    let mut filtered_out: u64 = 0;
    let mut record_idx: usize = 0;

    println!("scanning up to {n_records} records from {path}, ply_min={ply_min}...");

    while record_idx < n_records {
        if file.read_exact(&mut buf).is_err() {
            println!("EOF at record {record_idx}");
            break;
        }
        let psv: PackedSfenValue = unsafe { std::ptr::read(buf.as_ptr() as *const PackedSfenValue) };
        let board = psv.board();
        let sfen: String = board.to_sfen();

        // sfen 文字列から game_ply を抽出 (= sfen 最後の token)。
        let ply: i32 = sfen.split_ascii_whitespace().next_back().and_then(|s| s.parse().ok()).unwrap_or(0);
        *ply_hist.entry(ply / 10 * 10).or_insert(0) += 1;
        if ply < ply_min {
            filtered_out += 1;
            record_idx += 1;
            continue;
        }
        let mut pos = Position::new();
        if pos.set_sfen(&sfen).is_err() {
            record_idx += 1;
            continue;
        }
        for persp in [Color::Black, Color::White] {
            let c = compute_arh_v2_classes(&pos, persp);
            alpha[c.alpha] += 1;
            attackers[c.attackers] += 1;
            balance[c.balance] += 1;
            overlap[c.overlap] += 1;
            supported[c.supported] += 1;
            count += 1;
        }
        record_idx += 1;
    }

    println!(
        "\n=== ARH v2 軸分布 (= {count} obs / {record_idx} records read, {filtered_out} filtered out by ply<{ply_min}) ==="
    );
    let pct = |x: u64| (x as f64) * 100.0 / (count as f64).max(1.0);

    println!("\nalpha (0..4):");
    for (i, &v) in alpha.iter().enumerate() {
        println!("  {}: {:8} ({:5.1}%)", i, v, pct(v));
    }
    println!("\nattackers (0..3):");
    for (i, &v) in attackers.iter().enumerate() {
        println!("  {}: {:8} ({:5.1}%)", i, v, pct(v));
    }
    println!("\nbalance (0..8):");
    for (i, &v) in balance.iter().enumerate() {
        println!("  {} (=raw {:+2}): {:8} ({:5.1}%)", i, i as i32 - 4, v, pct(v));
    }
    println!("\noverlap (0..3):");
    for (i, &v) in overlap.iter().enumerate() {
        println!("  {}: {:8} ({:5.1}%)", i, v, pct(v));
    }
    println!("\nsupported (0..3):");
    for (i, &v) in supported.iter().enumerate() {
        println!("  {}: {:8} ({:5.1}%)", i, v, pct(v));
    }

    println!("\n=== ply 分布 (= bucket 10 ごと) ===");
    let mut keys: Vec<i32> = ply_hist.keys().copied().collect();
    keys.sort();
    for k in keys {
        println!("  ply {:3}-{:3}: {:8}", k, k + 9, ply_hist[&k]);
    }
}
