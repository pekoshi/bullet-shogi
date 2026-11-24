//! progress-bucket × eval(cp) クロス集計 と progress-bucket フィルタ(psv 出力)。
//!
//! progress8kpabs bucket(0=序盤〜7=終盤)を `ShogiProgressKPAbs` で算出し、
//! (A) 計測モード: bucket × cp帯 のクロス表を出力
//! (B) フィルタモード: bucket≥N (+任意 eval band) の局面のみ psv 出力
//!
//! 決定版 bucket = floor(sigmoid(prog_sum) * 8).clamp(0, 7)
//! (`ShogiProgressKPAbs::progress` は既に sigmoid 適用済みの p∈[0,1] を返す)
//!
//! 実行例(計測モード):
//! ```text
//! cargo run --release --example bucket_cp_crosstab -- \
//!   --progress <progress.bin> \
//!   --inputs <f1.psv> <f2.psv> \
//!   [--sample <per_file>]
//! ```
//!
//! 実行例(フィルタモード = psv 出力):
//! ```text
//! cargo run --release --example bucket_cp_crosstab -- \
//!   --progress <progress.bin> \
//!   --inputs <f1.psv> <f2.psv> \
//!   --min-bucket 6 --output <out.psv> \
//!   [--eval-lo <L>] [--eval-hi <H>]
//! ```
//! --eval-lo / --eval-hi は |score| (評価値絶対値) に対する下限/上限 (両端含む、cp 単位)。

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use bullet_lib::{game::outputs::ShogiProgressKPAbs, shogi::PackedSfenValue};

const REC: usize = 40;

/// cp 絶対値の帯区分 [0,50) [50,200) [200,500) [500,∞)
const CP_EDGES: [i32; 3] = [50, 200, 500];
const CP_BANDS: usize = 4;

fn cp_band(abs_cp: i32) -> usize {
    if abs_cp < CP_EDGES[0] {
        0
    } else if abs_cp < CP_EDGES[1] {
        1
    } else if abs_cp < CP_EDGES[2] {
        2
    } else {
        3
    }
}

struct Args {
    progress: String,
    inputs: Vec<String>,
    sample: Option<usize>,
    min_bucket: Option<usize>,
    output: Option<String>,
    eval_lo: Option<i32>,
    eval_hi: Option<i32>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut progress = None;
    let mut inputs = Vec::new();
    let mut sample = None;
    let mut min_bucket = None;
    let mut output = None;
    let mut eval_lo = None;
    let mut eval_hi = None;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--progress" => {
                i += 1;
                progress = Some(argv[i].clone());
            }
            "--inputs" => {
                i += 1;
                while i < argv.len() && !argv[i].starts_with("--") {
                    inputs.push(argv[i].clone());
                    i += 1;
                }
                continue;
            }
            "--sample" => {
                i += 1;
                sample = Some(argv[i].parse().expect("--sample <usize>"));
            }
            "--min-bucket" => {
                i += 1;
                min_bucket = Some(argv[i].parse().expect("--min-bucket <0..7>"));
            }
            "--output" => {
                i += 1;
                output = Some(argv[i].clone());
            }
            "--eval-lo" => {
                i += 1;
                eval_lo = Some(argv[i].parse().expect("--eval-lo <i32>"));
            }
            "--eval-hi" => {
                i += 1;
                eval_hi = Some(argv[i].parse().expect("--eval-hi <i32>"));
            }
            other => panic!("unknown arg: {other}"),
        }
        i += 1;
    }

    Args {
        progress: progress.expect("--progress <progress.bin> required"),
        inputs,
        sample,
        min_bucket,
        output,
        eval_lo,
        eval_hi,
    }
}

/// psv レコードを順次読み、コールバックに (psv, bucket) を渡す。
/// 戻り値 false で早期停止(per-file sample 用)。
fn for_each_record<F: FnMut(&PackedSfenValue, usize) -> bool>(
    path: &str,
    model: &ShogiProgressKPAbs,
    mut f: F,
) -> usize {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut r = BufReader::with_capacity(REC * 8192, file);
    let mut buf = vec![0u8; REC * 8192];
    let mut leftover = 0usize;
    let mut count = 0usize;
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
            let bucket = (p * 8.0).floor().clamp(0.0, 7.0) as usize;
            count += 1;
            if !f(&psv, bucket) {
                break 'outer;
            }
        }
        let consumed = nrec * REC;
        leftover = avail - consumed;
        buf.copy_within(consumed..avail, 0);
    }
    count
}

fn run_measure(args: &Args, model: &ShogiProgressKPAbs) {
    // bucket × cp帯 件数
    let mut table = [[0u64; CP_BANDS]; 8];
    // bucket ごとの生 score 符号分布
    let mut sign_pos = [0u64; 8];
    let mut sign_neg = [0u64; 8];
    let mut sign_zero = [0u64; 8];
    let mut bucket_total = [0u64; 8];
    let mut total = 0u64;
    let per_file = args.sample.unwrap_or(usize::MAX);

    for path in &args.inputs {
        let mut local = 0usize;
        let sampled = for_each_record(path, model, |psv, bucket| {
            let s = psv.score() as i32;
            let ab = s.abs();
            table[bucket][cp_band(ab)] += 1;
            match s.cmp(&0) {
                std::cmp::Ordering::Greater => sign_pos[bucket] += 1,
                std::cmp::Ordering::Less => sign_neg[bucket] += 1,
                std::cmp::Ordering::Equal => sign_zero[bucket] += 1,
            }
            bucket_total[bucket] += 1;
            total += 1;
            local += 1;
            local < per_file
        });
        eprintln!("{path}: scanned {sampled} records (sampled into stats: {local})");
    }

    println!("=== bucket × cp帯 crosstab (total sampled = {total}) ===\n");

    // (1) bucket 占有率
    println!("--- bucket 占有率 ---");
    for b in 0..8 {
        let pct = if total > 0 { bucket_total[b] as f64 / total as f64 * 100.0 } else { 0.0 };
        println!("  bucket {b}: {:>12}  ({:>6.2}%)", bucket_total[b], pct);
    }

    // (2) 生 score 符号分布
    println!("\n--- 生 score 符号分布 (手番側視点) ---");
    println!("{:<8} {:>12} {:>12} {:>12}", "bucket", "score>0", "score<0", "score=0");
    for b in 0..8 {
        println!("{:<8} {:>12} {:>12} {:>12}", b, sign_pos[b], sign_neg[b], sign_zero[b]);
    }

    // (3) bucket × |cp|帯 件数 と bucket 内%
    println!("\n--- bucket × |cp|帯 件数 (各 bucket 内%) ---");
    let band_names = ["[0,50)", "[50,200)", "[200,500)", "[500,inf)"];
    print!("{:<8}", "bucket");
    for n in band_names {
        print!(" {:>20}", n);
    }
    println!("{:>14}", "row_total");
    for b in 0..8 {
        let rt = bucket_total[b];
        print!("{:<8}", b);
        for band in 0..CP_BANDS {
            let c = table[b][band];
            let pct = if rt > 0 { c as f64 / rt as f64 * 100.0 } else { 0.0 };
            print!(" {:>11} ({:>5.1}%)", c, pct);
        }
        println!("{:>14}", rt);
    }

    // (4) 各 cp帯 が どの bucket に集中するか (列方向% = band 内での bucket 分布)
    println!("\n--- 各 |cp|帯 の bucket 分布 (列内%; 「その cp帯の局面がどの bucket に居るか」) ---");
    print!("{:<10}", "cp帯\\bkt");
    for b in 0..8 {
        print!(" {:>8}", format!("b{b}"));
    }
    println!("{:>14}", "col_total");
    for band in 0..CP_BANDS {
        let col_total: u64 = (0..8).map(|b| table[b][band]).sum();
        print!("{:<10}", band_names[band]);
        for b in 0..8 {
            let c = table[b][band];
            let pct = if col_total > 0 { c as f64 / col_total as f64 * 100.0 } else { 0.0 };
            print!(" {:>7.1}%", pct);
        }
        println!("{:>14}", col_total);
    }

    // (5) 注目: cp200-500 (優勢帯) が bucket5,6,7 に集中するか
    let band = 2; // [200,500)
    let col_total: u64 = (0..8).map(|b| table[b][band]).sum();
    let late: u64 = (5..8).map(|b| table[b][band]).sum();
    let late_pct = if col_total > 0 { late as f64 / col_total as f64 * 100.0 } else { 0.0 };
    println!(
        "\n--- 注目: |cp|∈[200,500) の {} 局面のうち bucket5,6,7 に {} 件 = {:.1}% ---",
        col_total, late, late_pct
    );
}

fn run_filter(args: &Args, model: &ShogiProgressKPAbs) {
    let min_bucket = args.min_bucket.expect("filter mode requires --min-bucket");
    let out_path = args.output.clone().expect("filter mode requires --output");
    let eval_lo = args.eval_lo;
    let eval_hi = args.eval_hi;

    let out_file = File::create(&out_path).unwrap_or_else(|e| panic!("create {out_path}: {e}"));
    let mut w = BufWriter::with_capacity(REC * 8192, out_file);

    let mut total = 0u64;
    let mut kept = 0u64;
    let per_file = args.sample.unwrap_or(usize::MAX);

    for path in &args.inputs {
        let mut local = 0usize;
        for_each_record(path, model, |psv, bucket| {
            total += 1;
            local += 1;
            let mut keep = bucket >= min_bucket;
            if keep {
                let ab = (psv.score() as i32).abs();
                if let Some(lo) = eval_lo {
                    if ab < lo {
                        keep = false;
                    }
                }
                if let Some(hi) = eval_hi {
                    if ab > hi {
                        keep = false;
                    }
                }
            }
            if keep {
                w.write_all(psv.as_bytes()).expect("write");
                kept += 1;
            }
            local < per_file
        });
        eprintln!("{path}: scanned, local={local}");
    }
    w.flush().expect("flush");

    let bytes = kept * REC as u64;
    println!("=== filter done ===");
    println!("  条件: min_bucket>={min_bucket}, eval_lo={eval_lo:?}, eval_hi={eval_hi:?} (|score| 帯)");
    println!("  scanned total = {total}");
    println!("  kept          = {kept}  ({:.2}%)", kept as f64 / total.max(1) as f64 * 100.0);
    println!("  output        = {out_path}");
    println!("  output size   = {} bytes ({:.2} GB)", bytes, bytes as f64 / 1e9);
}

fn main() {
    let args = parse_args();
    assert!(!args.inputs.is_empty(), "--inputs に psv を1つ以上指定してください");

    ShogiProgressKPAbs::load_from_bin(Path::new(&args.progress)).expect("load progress.bin");
    let model = ShogiProgressKPAbs;

    if args.min_bucket.is_some() || args.output.is_some() {
        run_filter(&args, &model);
    } else {
        run_measure(&args, &model);
    }
}
