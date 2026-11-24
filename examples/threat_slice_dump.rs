//! psv から局面を標本抽出し、 sfen + 教師値 + 戦術量 を TSV で吐く。 2026-09-16
//!
//! 「threat の勾配を当てるデータを絞る」判断のための材料。 エンジン側で同じ sfen を eval し、
//! threat あり / なし の残差をカテゴリ別に比べる (tools/threat_slice_eval.py)。
//!
//!   threat_slice_dump <in.psv> [--sample N] [--start R] [--out out.tsv]
//!
//! 列: sfen \t score \t ply \t result \t hang \t unfav \t pairs
//!   hang  = 浮き駒 (相手の利きがあり味方の利きが無い駒) の数
//!   unfav = 不利な当たり (安い駒が高い駒に当たっている) の数
//!   pairs = threat ペア総数
use std::{
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
};

use bullet_lib::{game::inputs::threat_tactical_stats, shogi::packed_sfen::PackedSfenValue};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: threat_slice_dump <in.psv> [--sample N] [--start R] [--out out.tsv]");
        std::process::exit(2);
    }
    let path = args[1].clone();
    let mut sample = 100_000usize;
    let mut start = 0usize;
    let mut out_path: Option<String> = None;
    let mut i = 2;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--sample" => sample = args[i + 1].parse().expect("sample"),
            "--start" => start = args[i + 1].parse().expect("start"),
            "--out" => out_path = Some(args[i + 1].clone()),
            _ => {}
        }
        i += 2;
    }
    let mut file = File::open(&path).expect("open psv");
    let total = file.metadata().expect("meta").len() as usize / 40;
    let end = total;
    let step = ((end - start) / sample.max(1)).max(1);
    let mut w: Box<dyn Write> = match &out_path {
        Some(p) => Box::new(BufWriter::with_capacity(16 << 20, File::create(p).expect("create out"))),
        None => Box::new(BufWriter::new(std::io::stdout())),
    };
    writeln!(w, "sfen\tscore\tply\tresult\thang\tunfav\tpairs").unwrap();
    let mut buf = [0u8; 40];
    let mut idx = start;
    let mut n = 0usize;
    while idx < end && n < sample {
        if file.seek(SeekFrom::Start((idx as u64) * 40)).is_err() || file.read_exact(&mut buf).is_err() {
            break;
        }
        idx += step;
        // SAFETY: PackedSfenValue is #[repr(C)] [u8; 40]
        let psv: PackedSfenValue = unsafe { std::ptr::read(buf.as_ptr() as *const PackedSfenValue) };
        let (hang, unfav, pairs) = threat_tactical_stats(&psv);
        let board = psv.decode();
        writeln!(
            w,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            board.to_sfen(),
            psv.score(),
            psv.game_ply(),
            psv.game_result(),
            hang,
            unfav,
            pairs
        )
        .unwrap();
        n += 1;
    }
    w.flush().unwrap();
    eprintln!("{n} 局面を書き出し (start={start} step={step} total={total})");
}
