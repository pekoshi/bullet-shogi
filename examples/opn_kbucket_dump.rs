//! OPN の king bucket / 進行度 bucket を学習器と同じ関数で書き出す (python との添字の突き合わせ用)。 2026-11-14 OPNARCH
//!
//! 入力: PSV (40 byte / 局面)。 出力: 1 局面 1 行 `kz pb8 head243` (空白区切り、 PSV と同じ順)。
//!   kz      = `opn_kz` (k9k9z、 0..80)
//!   pb8     = `opn_progress_bucket8` (progress8kpabs と同じ進行度 8 区分、 0..7 = `--bucket-filter` の鍵)
//!   head243 = `ShogiOpnK9z::<3>::bucket` (= kz * 3 + min(pb8, 2))
//! CPU だけ・読むだけ。 GPU は使わない。
//!
//! usage: opn_kbucket_dump <psv> <progress.bin> <out.txt> [--tail N | --head N]

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::Path,
};

use bullet_lib::{
    game::outputs::{OutputBuckets, ShogiOpnK9z, opn_kz, opn_progress_bucket8},
    shogi::PackedSfenValue,
};

const RECORD: u64 = 40;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: opn_kbucket_dump <psv> <progress.bin> <out.txt> [--tail N | --head N]".into());
    }
    let (psv, prog, out) = (&args[1], &args[2], &args[3]);
    let total = std::fs::metadata(psv).map_err(|e| format!("{psv}: {e}"))?.len() / RECORD;
    let (start, count) = match args.get(4).map(String::as_str) {
        Some("--tail") => {
            let n: u64 = args.get(5).ok_or("--tail N")?.parse().map_err(|e| format!("--tail: {e}"))?;
            (total.saturating_sub(n), n.min(total))
        }
        Some("--head") => {
            let n: u64 = args.get(5).ok_or("--head N")?.parse().map_err(|e| format!("--head: {e}"))?;
            (0, n.min(total))
        }
        None => (0, total),
        Some(x) => return Err(format!("不明な引数 {x}")),
    };
    let heads = ShogiOpnK9z::<3>::load_from_bin(Path::new(prog))?;

    let mut f = File::open(psv).map_err(|e| format!("{psv}: {e}"))?;
    f.seek(SeekFrom::Start(start * RECORD)).map_err(|e| e.to_string())?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut w = BufWriter::new(File::create(out).map_err(|e| format!("{out}: {e}"))?);
    let mut pos = PackedSfenValue::default();
    let mut hist = [0u64; 8];
    for _ in 0..count {
        r.read_exact(pos.as_bytes_mut()).map_err(|e| e.to_string())?;
        let board = pos.decode();
        let kz = opn_kz(board.side_to_move, board.black_king_sq, board.white_king_sq);
        let pb8 = opn_progress_bucket8(&pos);
        let head = heads.bucket(&pos);
        hist[pb8 as usize] += 1;
        writeln!(w, "{kz} {pb8} {head}").map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    println!(
        "{psv}: 全 {total} 局面のうち 先頭から {start} 番目より {count} 局面 -> {out} / 進行度 8 区分の分布 {:?}",
        hist
    );
    Ok(())
}
