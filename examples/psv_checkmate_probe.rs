use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

use bullet_lib::shogi::packed_sfen::PackedSfenValue;
use rshogi_core::movegen::{MoveList, generate_legal};
use rshogi_core::position::Position;

const RECORD_SIZE: usize = 40;

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: psv_checkmate_probe <psv> [max_records]");
    let max_records: usize = args.next().map(|s| s.parse().expect("invalid max_records")).unwrap_or(1_000_000);

    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, File::open(&path).expect("open PSV"));
    let mut raw = [0u8; RECORD_SIZE];
    let mut total = 0usize;
    let mut terminal_checkmates = 0usize;
    let mut sfen_rejected = 0usize;
    let mut checked = 0usize;

    while total < max_records {
        match reader.read_exact(&mut raw) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => panic!("read error: {error}"),
        }
        total += 1;
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&raw);
        let board = psv.board();
        let sfen = board.to_sfen();
        let mut position = Position::new();
        if position.set_sfen(&sfen).is_err() {
            sfen_rejected += 1;
            continue;
        }
        if !position.in_check() {
            continue;
        }
        checked += 1;
        let mut moves = MoveList::new();
        generate_legal(&position, &mut moves);
        if moves.is_empty() {
            terminal_checkmates += 1;
            if terminal_checkmates <= 5 {
                println!("checkmate_record={} score={} sfen={}", total - 1, psv.score(), sfen);
            }
        }
    }
    println!(
        "summary records={} in_check={} terminal_checkmates={} sfen_rejected={}",
        total, checked, terminal_checkmates, sfen_rejected
    );
}
