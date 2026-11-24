//! P4H2 bucket 設計の測定 M2/M3。
//!
//! P4H2 bucket(progress4hand2)が progress8 に対し情報利得ゼロか検証する。
//! 核心仮説: 「hand2 分割 ≈ progress のサブビンの代理」。
//!
//! 実行:
//! ```text
//! cargo run --release --example p4h2_crosstab -- \
//!   <progress.bin> <file1.psv> <file2.psv> [per_file_sample]
//! ```

use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use bullet_lib::{
    game::outputs::{HAND_VALUE_THRESHOLDS, ShogiProgressKPAbs, weighted_hand_total},
    shogi::PackedSfenValue,
};

const REC: usize = 40;

fn median(v: &mut [i32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 { v[n / 2] as f64 } else { (v[n / 2 - 1] as f64 + v[n / 2] as f64) / 2.0 }
}

#[derive(Default, Clone)]
struct Prog4Stat {
    occ_hand0: u64,
    occ_hand1: u64,
    // bucket8 内訳: 2k(even, index0) / 2k+1(odd, index1)
    b8_even: u64,
    b8_odd: u64,
    // M3 的中: hand2 == (bucket8 が奇数=ペア後半)
    m3_hit: u64,
    m3_total: u64,
    hand_vals: Vec<i32>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: p4h2_crosstab <progress.bin> <file1.psv> <file2.psv> [per_file_sample]");
        std::process::exit(1);
    }
    let progress_bin = &args[1];
    let files = [&args[2], &args[3]];
    let per_file: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(500_000);

    ShogiProgressKPAbs::load_from_bin(Path::new(progress_bin)).expect("load progress.bin");
    let model = ShogiProgressKPAbs;

    let mut prog4: [Prog4Stat; 4] = Default::default();
    let mut stack8 = [0u64; 8]; // P4H2: prog4*2 + hand2
    let mut bucket8_occ = [0u64; 8]; // progress8 対照
    let mut total = 0u64;

    for f in files {
        let file = File::open(f).unwrap_or_else(|e| panic!("open {f}: {e}"));
        let mut r = BufReader::with_capacity(REC * 4096, file);
        let mut buf = vec![0u8; REC * 4096];
        let mut count = 0usize;
        let mut leftover = 0usize;
        'outer: loop {
            let n = r.read(&mut buf[leftover..]).expect("read");
            if n == 0 {
                break;
            }
            let avail = leftover + n;
            let nrec = avail / REC;
            for i in 0..nrec {
                let rec = &buf[i * REC..i * REC + REC];
                let mut psv = PackedSfenValue::default();
                psv.as_bytes_mut().copy_from_slice(rec);

                let p = model.progress(&psv);
                let bucket8 = (p * 8.0).floor().clamp(0.0, 7.0) as usize;
                bucket8_occ[bucket8] += 1;
                let pb4 = bucket8 / 2; // 0..3
                let wh = weighted_hand_total(&psv);
                let hand2 = if wh >= HAND_VALUE_THRESHOLDS[pb4] { 1usize } else { 0usize };

                let st = &mut prog4[pb4];
                if hand2 == 0 {
                    st.occ_hand0 += 1;
                } else {
                    st.occ_hand1 += 1;
                }
                if bucket8 % 2 == 0 {
                    st.b8_even += 1;
                } else {
                    st.b8_odd += 1;
                }
                // M3: hand2 が「ペアのどちらの半分か(=bucket8 奇数)」を復元する的中率
                let pair_half = bucket8 % 2; // 0=前半(2k), 1=後半(2k+1)
                if hand2 == pair_half {
                    st.m3_hit += 1;
                }
                st.m3_total += 1;
                st.hand_vals.push(wh);

                stack8[pb4 * 2 + hand2] += 1;
                total += 1;

                count += 1;
                if count >= per_file {
                    break 'outer;
                }
            }
            let consumed = nrec * REC;
            leftover = avail - consumed;
            buf.copy_within(consumed..avail, 0);
        }
        eprintln!("{f}: sampled {count} records");
    }

    println!("=== P4H2 crosstab measurement (total sampled = {total}) ===\n");

    // 1,2,3 per prog4
    println!("--- Per prog4 (progress_bucket4) ---");
    println!(
        "{:<6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "prog4", "hand2=0", "hand2=1", "hand_bal", "b8_even(2k)", "b8_odd(2k+1)", "M3_hit%", "wh_median"
    );
    let mut m3_hit_all = 0u64;
    let mut m3_tot_all = 0u64;
    for (i, st) in prog4.iter_mut().enumerate() {
        let occ = st.occ_hand0 + st.occ_hand1;
        let bal = if occ > 0 { st.occ_hand1 as f64 / occ as f64 * 100.0 } else { 0.0 };
        let m3 = if st.m3_total > 0 { st.m3_hit as f64 / st.m3_total as f64 * 100.0 } else { 0.0 };
        let med = median(&mut st.hand_vals);
        m3_hit_all += st.m3_hit;
        m3_tot_all += st.m3_total;
        println!(
            "{:<6} {:>10} {:>10} {:>7.1}% {:>10} {:>12} {:>9.2}% {:>10.1}",
            i, st.occ_hand0, st.occ_hand1, bal, st.b8_even, st.b8_odd, m3, med
        );
    }

    println!("\n--- 4. P4H2 8-stack occupancy (prog4*2 + hand2) ---");
    for (i, &c) in stack8.iter().enumerate() {
        let pct = if total > 0 { c as f64 / total as f64 * 100.0 } else { 0.0 };
        println!("  stack {} (prog4={}, hand2={}): {:>10}  ({:>6.2}%)", i, i / 2, i % 2, c, pct);
    }

    println!("\n--- 5. progress8 occupancy (bucket8) ---");
    for (i, &c) in bucket8_occ.iter().enumerate() {
        let pct = if total > 0 { c as f64 / total as f64 * 100.0 } else { 0.0 };
        println!("  bucket8 {}: {:>10}  ({:>6.2}%)", i, c, pct);
    }

    let m3_all = if m3_tot_all > 0 { m3_hit_all as f64 / m3_tot_all as f64 * 100.0 } else { 0.0 };
    println!("\n--- 6. Overall M3 hit rate (prog4 加重平均) = {:.2}% ---", m3_all);
    println!("    (>75% なら hand2≈progress サブビン代理 = 情報利得上限ゼロ)");
}
