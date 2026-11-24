//! PSV 外部 sort + K-way merge dedup (CRCL-v2 改善 C+、 nnue-data_16B 全 file 対応)。
//!
//! 2-pass approach:
//!   Phase 1: 入力 file を chunk_size 単位で読み、 各 chunk を packed_sfen でソート
//!            → temp file (chunk_NNN.bin) として出力
//!   Phase 2: K-way merge from temp files、 streaming で連続同 sfen を集約 (dedup)
//!            score 平均化 + move/ply/result は最初値
//!
//! 使い方:
//!   cargo run --release --example psv_dedup_external -- input.bin output.bin [chunk_size_M] [tmp_dir]
//!
//! メモリ要件:
//!   - Phase 1: chunk_size × 40 byte (50M records = 2 GB)
//!   - Phase 2: K reader buffers + heap (small)、 数百 MB

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    env,
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::PathBuf,
    time::Instant,
};

const PSV_BYTES: usize = 40;
const SFEN_BYTES: usize = 32;

#[derive(Clone)]
#[repr(C)]
struct Record([u8; PSV_BYTES]);

impl Record {
    fn sfen(&self) -> &[u8; SFEN_BYTES] {
        // Safety: layout is [u8; 40] with first 32 bytes = packed_sfen
        unsafe { &*(self.0.as_ptr() as *const [u8; SFEN_BYTES]) }
    }
    fn score(&self) -> i16 {
        i16::from_le_bytes([self.0[32], self.0[33]])
    }
    fn move16(&self) -> u16 {
        u16::from_le_bytes([self.0[34], self.0[35]])
    }
    fn game_ply(&self) -> u16 {
        u16::from_le_bytes([self.0[36], self.0[37]])
    }
    fn game_result(&self) -> i8 {
        self.0[38] as i8
    }
}

/// K-way merge heap entry. reader_idx で source file を識別。
struct HeapEntry {
    record: Record,
    reader_idx: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.sfen() == other.record.sfen()
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.record.sfen().cmp(other.record.sfen())
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: psv_dedup_external <input.bin> <output.bin> [chunk_size_M=50] [tmp_dir=./psv_tmp]");
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let chunk_size_records: usize = if args.len() >= 4 {
        args[3].parse::<usize>().expect("chunk_size_M") * 1_000_000
    } else {
        50_000_000
    };
    let tmp_dir = if args.len() >= 5 { PathBuf::from(&args[4]) } else { PathBuf::from("./psv_tmp") };

    let input_size = fs::metadata(input_path).expect("input metadata").len() as usize;
    let total_records = input_size / PSV_BYTES;
    println!("input: {} ({} records、 {:.2} GB)", input_path, total_records, input_size as f64 / 1e9);
    println!("chunk size: {} records ({:.2} GB)", chunk_size_records, (chunk_size_records * PSV_BYTES) as f64 / 1e9);
    println!("tmp dir: {}", tmp_dir.display());

    fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let start = Instant::now();

    // ===== Phase 1: chunk sort + write temp files =====
    println!("\n=== Phase 1: chunk sort ===");
    let phase1_start = Instant::now();
    let file = File::open(input_path).expect("open input");
    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);
    let mut chunk_idx = 0usize;
    let mut total_in = 0usize;
    let mut tmp_paths: Vec<PathBuf> = Vec::new();

    loop {
        let chunk_st = Instant::now();
        let mut chunk: Vec<Record> = Vec::with_capacity(chunk_size_records);
        let mut buf = [0u8; PSV_BYTES];
        for _ in 0..chunk_size_records {
            match reader.read_exact(&mut buf) {
                Ok(()) => {
                    chunk.push(Record(buf));
                    total_in += 1;
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("read error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
        if chunk.is_empty() {
            break;
        }

        // sort by sfen
        chunk.sort_by(|a, b| a.sfen().cmp(b.sfen()));

        let tmp_path = tmp_dir.join(format!("chunk_{:04}.bin", chunk_idx));
        let mut out = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&tmp_path).expect("create tmp"));
        for r in &chunk {
            out.write_all(&r.0).expect("write tmp");
        }
        out.flush().expect("flush");
        let len = chunk.len();
        tmp_paths.push(tmp_path);
        println!("  chunk #{}: {} records sorted + written ({:.1}s)", chunk_idx, len, chunk_st.elapsed().as_secs_f64());
        chunk_idx += 1;
        if len < chunk_size_records {
            break;
        } // EOF
    }
    println!(
        "Phase 1 done: {} chunks, {} input records ({:.1}s)",
        tmp_paths.len(),
        total_in,
        phase1_start.elapsed().as_secs_f64()
    );

    // ===== Phase 2: K-way merge + streaming dedup =====
    println!("\n=== Phase 2: K-way merge + dedup ===");
    let phase2_start = Instant::now();
    let mut readers: Vec<BufReader<File>> =
        tmp_paths.iter().map(|p| BufReader::with_capacity(8 * 1024 * 1024, File::open(p).expect("open tmp"))).collect();
    let mut out_file = BufWriter::with_capacity(16 * 1024 * 1024, File::create(output_path).expect("create out"));

    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
    // 初期 record を各 reader から
    for (idx, rdr) in readers.iter_mut().enumerate() {
        let mut buf = [0u8; PSV_BYTES];
        if rdr.read_exact(&mut buf).is_ok() {
            heap.push(Reverse(HeapEntry { record: Record(buf), reader_idx: idx }));
        }
    }

    // streaming dedup state (current group の集約)
    let mut cur_key: Option<[u8; SFEN_BYTES]> = None;
    let mut cur_score_sum: i64 = 0;
    let mut cur_count: u32 = 0;
    let mut cur_move: u16 = 0;
    let mut cur_ply: u16 = 0;
    let mut cur_result: i8 = 0;
    let mut total_out = 0usize;

    let mut emit = |key: &[u8; SFEN_BYTES],
                    sum: i64,
                    cnt: u32,
                    m: u16,
                    p: u16,
                    r: i8,
                    out: &mut BufWriter<File>,
                    total: &mut usize| {
        let avg_score = (sum / cnt as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16;
        let mut buf = [0u8; PSV_BYTES];
        buf[..SFEN_BYTES].copy_from_slice(key);
        buf[32..34].copy_from_slice(&avg_score.to_le_bytes());
        buf[34..36].copy_from_slice(&m.to_le_bytes());
        buf[36..38].copy_from_slice(&p.to_le_bytes());
        buf[38] = r as u8;
        buf[39] = 0;
        out.write_all(&buf).expect("write out");
        *total += 1;
    };

    let mut iter_count = 0usize;
    while let Some(Reverse(top)) = heap.pop() {
        iter_count += 1;
        if iter_count % 10_000_000 == 0 {
            let elapsed = phase2_start.elapsed().as_secs_f64();
            println!(
                "  Phase 2: processed {} records ({:.1}s、 {:.0} records/sec)",
                iter_count,
                elapsed,
                iter_count as f64 / elapsed
            );
        }

        // 同じ sfen ならグループに集約、 違う sfen なら前 group emit + 新 group 開始
        let same_as_cur = cur_key.as_ref() == Some(top.record.sfen());
        if !same_as_cur {
            if let Some(ref key) = cur_key {
                emit(key, cur_score_sum, cur_count, cur_move, cur_ply, cur_result, &mut out_file, &mut total_out);
            }
            let mut new_key = [0u8; SFEN_BYTES];
            new_key.copy_from_slice(top.record.sfen());
            cur_key = Some(new_key);
            cur_score_sum = top.record.score() as i64;
            cur_count = 1;
            cur_move = top.record.move16();
            cur_ply = top.record.game_ply();
            cur_result = top.record.game_result();
        } else {
            cur_score_sum += top.record.score() as i64;
            cur_count += 1;
        }

        // 次 record を同 reader から
        let idx = top.reader_idx;
        let mut buf = [0u8; PSV_BYTES];
        if readers[idx].read_exact(&mut buf).is_ok() {
            heap.push(Reverse(HeapEntry { record: Record(buf), reader_idx: idx }));
        }
    }
    // 最後の group flush
    if let Some(ref key) = cur_key {
        emit(key, cur_score_sum, cur_count, cur_move, cur_ply, cur_result, &mut out_file, &mut total_out);
    }
    out_file.flush().expect("flush");
    println!("Phase 2 done: {} records out ({:.1}s)", total_out, phase2_start.elapsed().as_secs_f64());

    // cleanup tmp files
    for p in &tmp_paths {
        let _ = fs::remove_file(p);
    }
    let _ = fs::remove_dir(&tmp_dir);

    let elapsed = start.elapsed().as_secs_f64();
    let dedup_ratio = total_out as f64 / total_in.max(1) as f64;
    println!(
        "\n=== external dedup complete ({:.1}s) ===\ninput: {} records, output: {} records ({:.2}% kept、 {:.2}% removed)",
        elapsed,
        total_in,
        total_out,
        dedup_ratio * 100.0,
        (1.0 - dedup_ratio) * 100.0,
    );
}
