//! PSV (PackedSfenValue) 重複削減 preprocess tool (CRCL-v2 改善 C、 奏乗 +18.8 R 知見)。
//!
//! 同じ packed_sfen (32 byte) を持つ records を集約:
//! - score: 平均 (i16 → f32 で合計 → /count → 丸める)
//! - move16, game_ply, game_result: 最初に出現したもの
//!
//! 使い方:
//!   cargo run --release --example psv_dedup -- input.bin output.bin
//!
//! 注意:
//! - 現実装は in-memory HashMap、 chunk size 50M records (2GB) まで
//! - 全 16B records は外部 sort が必要 (次段)、 ここでは smoke + 部分適用

use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
    time::Instant,
};

const PSV_BYTES: usize = 40;
const SFEN_BYTES: usize = 32;
type SfenKey = [u8; SFEN_BYTES];

struct AggregatedRecord {
    score_sum: i64,
    count: u32,
    move16: u16,
    game_ply: u16,
    game_result: i8,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: psv_dedup <input.bin> <output.bin> [chunk_size_M]");
        std::process::exit(1);
    }
    let input_path = &args[1];
    let output_path = &args[2];
    let chunk_size_records: usize = if args.len() >= 4 {
        args[3].parse::<usize>().expect("chunk_size_M") * 1_000_000
    } else {
        50_000_000 // 50M records = 2 GB
    };

    let input_size = std::fs::metadata(input_path).expect("input metadata").len() as usize;
    let total_records = input_size / PSV_BYTES;
    println!("input: {} ({} records、 {:.1} GB)", input_path, total_records, input_size as f64 / 1e9);
    println!("chunk size: {} records ({:.1} GB)", chunk_size_records, (chunk_size_records * PSV_BYTES) as f64 / 1e9);

    let file = File::open(input_path).expect("open input");
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let out_file = File::create(output_path).expect("create output");
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, out_file);

    let mut total_input = 0usize;
    let mut total_output = 0usize;
    let mut chunk_idx = 0usize;
    let start = Instant::now();

    loop {
        let mut map: HashMap<SfenKey, AggregatedRecord> = HashMap::new();
        let chunk_start = Instant::now();
        let mut chunk_records = 0usize;
        let mut buf = [0u8; PSV_BYTES];

        for _ in 0..chunk_size_records {
            match reader.read_exact(&mut buf) {
                Ok(()) => {
                    chunk_records += 1;
                    total_input += 1;
                    let mut sfen: SfenKey = [0u8; SFEN_BYTES];
                    sfen.copy_from_slice(&buf[..SFEN_BYTES]);
                    let score = i16::from_le_bytes([buf[32], buf[33]]) as i64;
                    let move16 = u16::from_le_bytes([buf[34], buf[35]]);
                    let game_ply = u16::from_le_bytes([buf[36], buf[37]]);
                    let game_result = buf[38] as i8;

                    map.entry(sfen)
                        .and_modify(|e| {
                            e.score_sum += score;
                            e.count += 1;
                        })
                        .or_insert(AggregatedRecord { score_sum: score, count: 1, move16, game_ply, game_result });
                }
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => {
                    eprintln!("read error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }

        if chunk_records == 0 {
            break;
        }
        chunk_idx += 1;
        let unique = map.len();
        let dedup_ratio = unique as f64 / chunk_records as f64;
        println!(
            "chunk #{}: {} → {} unique ({:.2}% kept、 {:.1}s)",
            chunk_idx,
            chunk_records,
            unique,
            dedup_ratio * 100.0,
            chunk_start.elapsed().as_secs_f64()
        );

        // output unique records (score 平均化、 他は最初の値)
        let mut out_buf = vec![0u8; PSV_BYTES];
        for (sfen, agg) in map.iter() {
            let avg_score = (agg.score_sum / agg.count as i64).clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            out_buf[..SFEN_BYTES].copy_from_slice(sfen);
            out_buf[32..34].copy_from_slice(&avg_score.to_le_bytes());
            out_buf[34..36].copy_from_slice(&agg.move16.to_le_bytes());
            out_buf[36..38].copy_from_slice(&agg.game_ply.to_le_bytes());
            out_buf[38] = agg.game_result as u8;
            out_buf[39] = 0;
            writer.write_all(&out_buf).expect("write output");
            total_output += 1;
        }

        if chunk_records < chunk_size_records {
            break; // EOF
        }
    }

    writer.flush().expect("flush");
    let elapsed = start.elapsed().as_secs_f64();
    let dedup_ratio = total_output as f64 / total_input.max(1) as f64;
    println!(
        "\n=== dedup complete ({:.1}s) ===\nchunks: {}, input: {} records, output: {} records ({:.2}% kept)",
        elapsed,
        chunk_idx,
        total_input,
        total_output,
        dedup_ratio * 100.0,
    );
    println!("note: within-chunk dedup のみ、 chunk 間重複は detect しない (next step で external sort)");
    let _ = Path::new(output_path);
}
