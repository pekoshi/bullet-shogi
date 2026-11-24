use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use bullet_lib::shogi::packed_sfen::PackedSfenValue;
use rshogi_core::movegen::{MoveList, generate_legal};
use rshogi_core::position::Position;

const RECORD_SIZE: usize = 40;
const KEY_SIZE: usize = 32;
const DEFAULT_PARTITIONS: usize = 1024;
const DEFAULT_SEED: u64 = 20_260_819;
const REPORT_EVERY: u64 = 10_000_000;

#[derive(Default)]
struct Counts {
    input_records: u64,
    accepted_before_dedup: u64,
    terminal_checkmate_excluded: u64,
    sfen_rejected_excluded: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: psv_partition_filter <input.psv> <label> <output-root> [partitions_power_of_two] [seed] [append_existing:0|1] [max_records]"
    );
    std::process::exit(2);
}

fn fnv1a_partition(key: &[u8], seed: u64, mask: usize) -> usize {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for &byte in key {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    (hash as usize) & mask
}

fn terminal_checkmate(record: &[u8; RECORD_SIZE]) -> Result<bool, ()> {
    let mut psv = PackedSfenValue::default();
    psv.as_bytes_mut().copy_from_slice(record);
    let sfen = psv.board().to_sfen();
    let mut position = Position::new();
    if position.set_sfen(&sfen).is_err() {
        return Err(());
    }
    if !position.in_check() {
        return Ok(false);
    }
    let mut legal_moves = MoveList::new();
    generate_legal(&position, &mut legal_moves);
    Ok(legal_moves.is_empty())
}

fn output_path(dir: &Path, part: usize) -> PathBuf {
    dir.join(format!("part_{part:04}.psv"))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 || args.len() > 8 {
        usage();
    }
    let input_path = Path::new(&args[1]);
    let label = &args[2];
    let root = Path::new(&args[3]);
    let partitions: usize =
        args.get(4).map(|value| value.parse().expect("invalid partitions")).unwrap_or(DEFAULT_PARTITIONS);
    let seed: u64 = args.get(5).map(|value| value.parse().expect("invalid seed")).unwrap_or(DEFAULT_SEED);
    // append_existing=1 is only for resuming a confirmed interrupted run on the
    // same input/label/root. Overlap is intentionally left for the deterministic
    // key-level deduplication stage, so data integrity is preserved.
    let append_existing = args
        .get(6)
        .map(|value| match value.as_str() {
            "0" => false,
            "1" => true,
            _ => panic!("append_existing must be 0 or 1"),
        })
        .unwrap_or(false);
    let max_records: Option<u64> = args.get(7).map(|value| value.parse().expect("invalid max_records"));
    if partitions == 0 || !partitions.is_power_of_two() {
        panic!("partitions must be a positive power of two");
    }

    let input_size = fs::metadata(input_path).expect("input metadata").len();
    if input_size % RECORD_SIZE as u64 != 0 {
        panic!("input is not divisible by 40 bytes");
    }
    let total_records = input_size / RECORD_SIZE as u64;
    let out_dir = root.join("01_partitioned").join(label);
    let complete = out_dir.join("complete.json");
    if complete.exists() {
        println!("already_complete label={} manifest={}", label, complete.display());
        return;
    }
    if out_dir.exists() {
        if !append_existing {
            panic!(
                "incomplete output directory already exists: {} (use append_existing=1 only after confirming it belongs to the same input)",
                out_dir.display()
            );
        }
    } else {
        fs::create_dir_all(&out_dir).expect("create output directory");
    }

    let mut writers: Vec<BufWriter<File>> = Vec::with_capacity(partitions);
    for part in 0..partitions {
        let path = output_path(&out_dir, part);
        let file = if append_existing {
            OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap_or_else(|_| panic!("missing expected resume partition file: {}", path.display()))
        } else {
            OpenOptions::new().create_new(true).write(true).open(&path).expect("create partition file")
        };
        writers.push(BufWriter::with_capacity(8 * 1024 * 1024, file));
    }

    let mut reader = BufReader::with_capacity(32 * 1024 * 1024, File::open(input_path).expect("open input"));
    let mut raw = [0u8; RECORD_SIZE];
    let mut counts = Counts::default();
    let started = Instant::now();
    let mask = partitions - 1;

    loop {
        if max_records.is_some_and(|limit| counts.input_records >= limit) {
            break;
        }
        match reader.read_exact(&mut raw) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => panic!("input read error: {error}"),
        }
        counts.input_records += 1;
        match terminal_checkmate(&raw) {
            Ok(true) => {
                counts.terminal_checkmate_excluded += 1;
            }
            Ok(false) => {
                let part = fnv1a_partition(&raw[..KEY_SIZE], seed, mask);
                writers[part].write_all(&raw).expect("write partition record");
                counts.accepted_before_dedup += 1;
            }
            Err(()) => {
                counts.sfen_rejected_excluded += 1;
            }
        }
        if counts.input_records % REPORT_EVERY == 0 {
            // Checkpoint the partitions at every progress interval. This makes an
            // interrupted long run lose at most one interval, never its full tail.
            for writer in &mut writers {
                writer.flush().expect("checkpoint flush partition");
            }
            let elapsed = started.elapsed().as_secs_f64().max(0.001);
            println!(
                "progress\tlabel={}\tprocessed={}\tof={}\taccepted={}\tmate_excluded={}\trejected={}\trecords_per_sec={:.0}",
                label,
                counts.input_records,
                total_records,
                counts.accepted_before_dedup,
                counts.terminal_checkmate_excluded,
                counts.sfen_rejected_excluded,
                counts.input_records as f64 / elapsed,
            );
        }
    }
    for writer in &mut writers {
        writer.flush().expect("flush partition");
    }

    let elapsed = started.elapsed().as_secs_f64();
    let partition_records: u64 = (0..partitions)
        .map(|part| fs::metadata(output_path(&out_dir, part)).expect("partition metadata").len() / RECORD_SIZE as u64)
        .sum();
    let manifest = format!(
        concat!(
            "{{\n",
            "  \"label\": \"{}\",\n",
            "  \"input_path\": \"{}\",\n",
            "  \"record_size\": 40,\n",
            "  \"position_key\": \"first 32 bytes of PackedSfenValue\",\n",
            "  \"terminal_checkmate_exclusion\": \"side-to-move in check and zero legal moves\",\n",
            "  \"partitions\": {},\n",
            "  \"seed\": {},\n",
            "  \"input_records_current_pass\": {},\n",
            "  \"max_records_requested\": {},\n",
            "  \"accepted_current_pass\": {},\n",
            "  \"partition_records_including_append\": {},\n",
            "  \"append_existing\": {},\n",
            "  \"terminal_checkmate_excluded_current_pass\": {},\n",
            "  \"sfen_rejected_excluded_current_pass\": {},\n",
            "  \"elapsed_seconds\": {:.3}\n",
            "}}\n"
        ),
        label,
        input_path.display().to_string().replace('\\', "\\\\"),
        partitions,
        seed,
        counts.input_records,
        max_records.map_or("null".to_string(), |value| value.to_string()),
        counts.accepted_before_dedup,
        partition_records,
        append_existing,
        counts.terminal_checkmate_excluded,
        counts.sfen_rejected_excluded,
        elapsed,
    );
    fs::write(&complete, manifest).expect("write manifest");
    println!(
        "completed\tlabel={}\tprocessed={}\taccepted_current_pass={}\tpartition_records={}\tmate_excluded={}\trejected={}\telapsed_seconds={:.3}",
        label,
        counts.input_records,
        counts.accepted_before_dedup,
        partition_records,
        counts.terminal_checkmate_excluded,
        counts.sfen_rejected_excluded,
        elapsed,
    );
}
