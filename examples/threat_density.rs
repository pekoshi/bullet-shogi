//! psv の各局面について threat ペア数を数え、 分布を出す / 閾値で絞った psv を書き出す。
//!
//! 「threat 学習に特化したデータ」を作るための第一歩。 局面によって threat の
//! アクティブ数に有意な差が無ければ、 特化しても意味が無いのでまずそれを測る。
//! 特徴定義は学習器と同一 (`ShogiHalfKaHmThreat` の専用 sparse 入力経路) を使う。
//!
//! usage:
//!   threat_density <in.psv> [--sample N] [--out <out.psv> --min-score M] [--profile full]

use std::{
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
};

use bullet_lib::game::inputs::{
    MAX_ACTIVE_THREAT_FEATURES, ShogiHalfKaHmThreat, SparseInputType, THREAT_FT_SPARSE_NSTM, THREAT_FT_SPARSE_STM,
    ThreatProfile, threat_tactical_stats,
};
use bullet_lib::shogi::PackedSfenValue;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: threat_density <in.psv> [--sample N] [--out <out.psv> --min-pairs M] [--profile full]");
        std::process::exit(2);
    }
    let path = args[1].clone();
    let mut sample = 0usize; // 0 = 全件
    let mut out_path: Option<String> = None;
    let mut min_pairs = 0usize;
    let mut profile = "full".to_string();
    let mut start = 0usize; // 走査開始レコード
    let mut count = 0usize; // 0 = 末尾まで
    let mut i = 2;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--sample" => sample = args[i + 1].parse().expect("sample"),
            "--out" => out_path = Some(args[i + 1].clone()),
            "--min-score" => min_pairs = args[i + 1].parse().expect("min-pairs"),
            "--profile" => profile = args[i + 1].clone(),
            "--start" => start = args[i + 1].parse().expect("start"),
            "--count" => count = args[i + 1].parse().expect("count"),
            _ => {}
        }
        i += 2;
    }

    let tp = ThreatProfile::from_cli(&profile).expect("unknown threat profile");
    let input = ShogiHalfKaHmThreat::new(tp).with_threat_ft_sparse(true);
    println!("profile={profile} threat_dims={}", input.threat_dimensions());

    let mut file = File::open(&path).expect("open psv");
    let file_records = file.metadata().expect("metadata").len() as usize / 40;
    // --start / --count で範囲を切って複数プロセスに分割できる (単スレッドで 7.8 万局面/秒)
    let total = if count > 0 { (start + count).min(file_records) } else { file_records };
    if start > 0 {
        file.seek(SeekFrom::Start((start as u64) * 40)).expect("seek start");
    }
    let step = if sample > 0 && sample < total { total / sample } else { 1 };
    println!("入力 {path}  範囲 [{start}, {total}) / 全 {file_records} 局面  (step={step})");

    let mut writer =
        out_path.as_ref().map(|p| BufWriter::with_capacity(16 << 20, File::create(p).expect("create out")));

    let mut buf = [0u8; 40];
    let mut stm = vec![0i32; MAX_ACTIVE_THREAT_FEATURES];
    let mut nstm = vec![0i32; MAX_ACTIVE_THREAT_FEATURES];
    let mut hist = vec![0u64; 129];
    let mut n = 0u64;
    let mut kept = 0u64;
    let mut sum = 0u64;
    let mut sum_h = 0u64;
    let mut sum_u = 0u64;
    let mut sum_p = 0u64;

    // step > 1 のときは seek で飛ばす (全件 read すると数十 GB を舐めることになる)
    let mut idx = start;
    while idx < total {
        if step > 1 && idx > 0 && file.seek(SeekFrom::Start((idx as u64) * 40)).is_err() {
            break;
        }
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        idx += step;
        // SAFETY: PackedSfenValue is #[repr(C)] [u8; 40]
        let psv: PackedSfenValue = unsafe { std::ptr::read(buf.as_ptr() as *const PackedSfenValue) };
        let _ = input.fill_auxiliary_sparse(THREAT_FT_SPARSE_STM, &psv, &mut stm);
        let _ = input.fill_auxiliary_sparse(THREAT_FT_SPARSE_NSTM, &psv, &mut nstm);
        let (hang, unfav, pairs) = threat_tactical_stats(&psv);
        // 戦術スコア = 浮き駒 + 不利な当たり。 threat が eval を動かすのはこの2種
        let score = (hang + unfav) as usize;
        hist[score.min(128)] += 1;
        sum += score as u64;
        sum_h += hang as u64;
        sum_u += unfav as u64;
        sum_p += pairs as u64;
        n += 1;
        if let Some(w) = writer.as_mut() {
            if score >= min_pairs {
                w.write_all(&buf).expect("write");
                kept += 1;
            }
        }
    }
    if let Some(w) = writer.as_mut() {
        w.flush().expect("flush");
    }

    println!(
        "標本 {n}  平均: 戦術スコア {:.2} (浮き駒 {:.2} + 不利な当たり {:.2})  全ペア {:.2}",
        sum as f64 / n as f64,
        sum_h as f64 / n as f64,
        sum_u as f64 / n as f64,
        sum_p as f64 / n as f64
    );
    let mut acc = 0u64;
    let mut printed = 0;
    for (p, c) in hist.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        acc += c;
        if printed < 40 {
            println!(
                "  score={p:3}  {c:9}  ({:5.2}%)  累積 {:5.1}%",
                100.0 * *c as f64 / n as f64,
                100.0 * acc as f64 / n as f64
            );
            printed += 1;
        }
    }
    if out_path.is_some() {
        println!("書き出し {kept} 局面 (min_score={min_pairs}) -> {}", out_path.unwrap());
    }
}
