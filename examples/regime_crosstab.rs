//! 案 B (Regime LayerStack) の G-0: bucket x regime の件数を訓練データで数える。
//!
//! 事前登録: `C:/project/p2shogi/runs/prereg/PREREG-20261007-REGIME-B.md`
//!
//! # なぜ要るか
//! 案 B は LayerStack を `progress_bucket8 * 2 + regime` の 16 本に割る。
//! **データが来ない側の stack を作ると害になる** ([[threat_group_starved_b01]]:
//! 「少しだけ勾配が来る」のが最悪で、双子から中途半端にずれて収束しない)。
//! 対局ログでの実測では ply 1-40 の regime 1 は 0.8% しかなかったので、
//! 序盤を割らない下限 `--regime-min-pb` を入れた。
//! **訓練データ側の分布は未測定**なので、9 時間の学習を始める前にここで数える。
//!
//! # 関門
//! 分割する各 progress bucket (>= min_pb) で **regime 1 の割合が 10% 以上**であること。
//!
//! # V1 の定義
//! 両玉 8 近傍のうち**駒が乗っていない升**への、その玉の敵の利き数の和。
//! 利きは **step + ray** (長い利きを含む)。玉の利きも含む。
//! 2026-10-07 に C++ の `P2Volm::value` と 400 局面で完全一致を確認済み。
//!
//! 実行:
//! ```text
//! cargo run --release -p bullet_lib --example regime_crosstab --features lsmode-p8h2 -- \
//!   <progress.bin> <min_pb> <v1_th> <file1.psv> [file2.psv ...] [--sample N]
//! ```
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use bullet_lib::{
    game::outputs::{OutputBuckets, ShogiProgress8Hand2KPAbs, ShogiProgressKPAbs, set_regime_mode},
    shogi::PackedSfenValue,
};

const REC: usize = 40;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut sample: usize = 2_000_000;
    let mut pos = Vec::new();
    let mut i = 1;
    while i < argv.len() {
        if argv[i] == "--sample" {
            sample = argv.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(sample);
            i += 2;
        } else {
            pos.push(argv[i].clone());
            i += 1;
        }
    }
    if pos.len() < 4 {
        eprintln!("usage: regime_crosstab <progress.bin> <min_pb> <v1_th> <file.psv> [...] [--sample N]");
        std::process::exit(1);
    }
    let progress_bin = pos[0].clone();
    let min_pb: i32 = pos[1].parse().expect("min_pb");
    let v1_th: i32 = pos[2].parse().expect("v1_th");
    let files: Vec<String> = pos[3..].to_vec();

    ShogiProgressKPAbs::load_from_bin(Path::new(&progress_bin)).expect("load progress.bin");
    // 🚨学習と同じ経路で bucket を出す (ShogiProgress8Hand2KPAbs::bucket の regime 分岐)
    set_regime_mode(true, v1_th, min_pb);
    let model = ShogiProgress8Hand2KPAbs;

    println!("regime_crosstab: min_pb={min_pb} v1_th={v1_th} sample={sample}/file");
    println!("  bucket = progress_bucket8 * 2 + [bucket8 >= min_pb && V1 >= v1_th]");

    let mut cnt = [0u64; 16];
    let mut total = 0u64;
    for f in &files {
        let file = File::open(f).unwrap_or_else(|e| panic!("open {f}: {e}"));
        let mut r = BufReader::with_capacity(REC * 4096, file);
        let mut buf = vec![0u8; REC * 4096];
        let mut n = 0usize;
        'outer: loop {
            let got = match r.read(&mut buf) {
                Ok(0) => break,
                Ok(g) => g,
                Err(e) => panic!("read {f}: {e}"),
            };
            let recs = got / REC;
            for k in 0..recs {
                let sl = &buf[k * REC..(k + 1) * REC];
                let psv: PackedSfenValue = unsafe { std::ptr::read_unaligned(sl.as_ptr().cast()) };
                let b = model.bucket(&psv) as usize;
                if b < 16 {
                    cnt[b] += 1;
                    total += 1;
                }
                n += 1;
                if n >= sample {
                    break 'outer;
                }
            }
        }
        println!("  {f}: {n} 件");
    }

    println!();
    println!("  {:<10} {:>12} {:>12} {:>10}  {}", "progress", "regime0", "regime1", "r1 の割合", "判定");
    let mut bad = 0;
    for pb in 0..8usize {
        let c0 = cnt[pb * 2];
        let c1 = cnt[pb * 2 + 1];
        let tot = c0 + c1;
        let share = if tot == 0 { 0.0 } else { 100.0 * c1 as f64 / tot as f64 };
        let split = (pb as i32) >= min_pb;
        let mark = if !split {
            if c1 > 0 { "★分割しない帯なのに regime1 がある = バグ" } else { "分割しない" }
        } else if share >= 10.0 {
            "OK"
        } else {
            bad += 1;
            "★10% 未満 = 学習データが薄い"
        };
        println!("  bucket {pb:<3} {c0:>12} {c1:>12} {share:>9.1}%  {mark}");
    }
    println!();
    println!("  合計 {total} 件");
    if bad == 0 {
        println!("  ✅G-0 合格: 分割する全 bucket で regime 1 が 10% 以上");
    } else {
        println!("  ★G-0 不合格: {bad} 個の bucket で regime 1 が 10% 未満");
        println!("     → --regime-min-pb を上げるか --regime-v1-th を下げること");
    }
}
