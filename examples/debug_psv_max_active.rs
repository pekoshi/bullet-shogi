//! PSV file の各 record で ShogiHalfKA_hm の sparse index 数を count、
//! max_active=40 を超える record を特定するデバッグツール。
//!
//! 使い方:
//!   cargo run --release --example debug_psv_max_active -- <psv_file> [n_records]
//!
//! 例:
//!   cargo run --release --example debug_psv_max_active -- E:/shogi/p2shogi/data/bullet/aoba115.spk 100000

use std::{fs::File, io::Read};

use bullet_lib::{
    game::inputs::{ShogiHalfKA_hm, SparseInputType},
    shogi::PackedSfenValue,
};

fn main() {
    let path = std::env::args().nth(1).expect("usage: debug_psv_max_active <psv_file> [n_records]");
    let n_records: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    let mut file = File::open(&path).expect("open PSV file");
    let mut buf = [0u8; 40];
    let input = ShogiHalfKA_hm;

    let mut total = 0usize;
    let mut max_stm = 0usize;
    let mut max_nstm = 0usize;
    let mut over_count = 0usize;
    let mut histogram = std::collections::HashMap::<usize, usize>::new();
    let mut first_over_record: Option<(usize, usize, usize)> = None;

    println!("scanning {n_records} records from {path}...");

    for i in 0..n_records {
        if file.read_exact(&mut buf).is_err() {
            println!("EOF at record {i}");
            break;
        }
        // SAFETY: PackedSfenValue is #[repr(C)] [u8; 40]
        let psv: PackedSfenValue = unsafe { std::ptr::read(buf.as_ptr() as *const PackedSfenValue) };

        let mut count_stm = 0usize;
        let mut count_nstm = 0usize;
        input.map_features(&psv, |_stm, _nstm| {
            count_stm += 1;
            count_nstm += 1;
        });

        max_stm = max_stm.max(count_stm);
        max_nstm = max_nstm.max(count_nstm);
        *histogram.entry(count_stm).or_default() += 1;

        if count_stm > 40 || count_nstm > 40 {
            over_count += 1;
            if first_over_record.is_none() {
                first_over_record = Some((i, count_stm, count_nstm));
                println!("first over_max_active record: idx={i} stm={count_stm} nstm={count_nstm}");
                // PSV byte dump (最初の record だけ)
                print!("  bytes:");
                for b in &buf {
                    print!(" {b:02x}");
                }
                println!();
            }
        }

        total += 1;
    }

    println!("\n=== summary ===");
    println!("total records scanned: {total}");
    println!("max stm count: {max_stm}");
    println!("max nstm count: {max_nstm}");
    println!("records over max_active(40): {over_count} ({:.2}%)", 100.0 * over_count as f64 / total as f64);

    let mut hist_vec: Vec<(usize, usize)> = histogram.into_iter().collect();
    hist_vec.sort();
    println!("\ncount histogram:");
    for (count, freq) in &hist_vec {
        let pct = 100.0 * *freq as f64 / total as f64;
        println!("  {count:3}: {freq:8} ({pct:.2}%)");
    }
}
