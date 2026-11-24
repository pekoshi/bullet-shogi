//! psv の各局面で有効な threat 特徴 index (群オフセット込み、 手番側・非手番側) を書き出す。 2026-09-17
//!
//! 「planf-42 で 0 の threat 重み (約 13%) が実戦でどれくらい引かれるか」を数えるための材料。
//! 特徴の定義・群の決め方は学習器と同一 (`ShogiHalfKaHmThreat` の専用 sparse 入力 + progress8kpabs)。
//!
//!   threat_active_dump <in.psv> <progress.bin> <out.bin> [--sample N]
//!
//! 出力: 局面ごとに u32 bucket, u32 n_stm, u32 n_nstm, u32 x (n_stm + n_nstm) (index = f + 群 x 216,720)
use std::{
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use bullet_lib::game::inputs::{
    MAX_ACTIVE_THREAT_FEATURES, ShogiHalfKaHmThreat, SparseInputType, THREAT_FT_SPARSE_NSTM, THREAT_FT_SPARSE_STM,
    ThreatProfile,
};
use bullet_lib::game::outputs::{OutputBuckets, ShogiProgressKPAbs};
use bullet_lib::shogi::PackedSfenValue;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: threat_active_dump <in.psv> <progress.bin> <out.bin> [--sample N]");
        std::process::exit(2);
    }
    let mut sample = 0usize;
    if args.len() >= 6 && args[4] == "--sample" {
        sample = args[5].parse().expect("sample");
    }
    let progress = ShogiProgressKPAbs::load_from_bin(Path::new(&args[2])).expect("load progress.bin");
    let tp = ThreatProfile::from_cli("full").expect("profile");
    let input = ShogiHalfKaHmThreat::new(tp)
        .with_threat_ft_sparse(true)
        .with_threat_bucket_groups(Some(vec![0, 1, 1, 2, 2, 3, 3, 3]));

    let mut file = File::open(&args[1]).expect("open psv");
    let total = file.metadata().expect("meta").len() as usize / 40;
    let step = if sample > 0 && sample < total { total / sample } else { 1 };
    let mut w = BufWriter::with_capacity(16 << 20, File::create(&args[3]).expect("create out"));
    let mut buf = [0u8; 40];
    let mut stm = vec![0i32; MAX_ACTIVE_THREAT_FEATURES];
    let mut nstm = vec![0i32; MAX_ACTIVE_THREAT_FEATURES];
    let mut n = 0usize;
    let mut idx = 0usize;
    while idx < total {
        file.seek(SeekFrom::Start((idx as u64) * 40)).expect("seek");
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        idx += step;
        // SAFETY: PackedSfenValue は #[repr(C)] の [u8; 40] と同じレイアウト。 buf は 40 byte 読み込み済み。
        let psv: PackedSfenValue = unsafe { std::ptr::read(buf.as_ptr() as *const PackedSfenValue) };
        let bucket = progress.bucket(&psv) as u32;
        let a = input.fill_auxiliary_sparse(THREAT_FT_SPARSE_STM, &psv, &mut stm);
        let b = input.fill_auxiliary_sparse(THREAT_FT_SPARSE_NSTM, &psv, &mut nstm);
        w.write_all(&bucket.to_le_bytes()).unwrap();
        w.write_all(&(a as u32).to_le_bytes()).unwrap();
        w.write_all(&(b as u32).to_le_bytes()).unwrap();
        for &v in stm[..a].iter().chain(nstm[..b].iter()) {
            w.write_all(&(v as u32).to_le_bytes()).unwrap();
        }
        n += 1;
    }
    w.flush().unwrap();
    eprintln!("{n} 局面 (total={total} step={step})");
}
