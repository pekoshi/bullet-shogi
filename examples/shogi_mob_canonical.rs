//! B6 可動域の **3 実装照合**用ダンプ。 2026-09-29。
//!
//! psv の各局面を `ShogiBoard` に展開し、 学習器側の `mob_all` と、
//! 照合用の sfen を tsv で吐く。 これを
//!
//!   ① Python 正典  `runs/rescore/fs_feats.py` の `mob_u` / `mob_t`
//!   ② C++ エンジン `p2mob <sfen>` (`source/eval/p2nnue/p2nnue_mob.h`)
//!
//! と**ゼロ許容で**突き合わせる。 3 つ揃わないと、 学習した重みが黙って意味を失う
//! (★19 の net×exe 不一致と同じ型)。
//!
//! 使い方:
//!   cargo run --release --example shogi_mob_canonical -- <psv> [n] > mob_trainer.tsv
//!
//! 出力 (ヘッダ 1 行 + n 行):
//!   sfen \t mob_black \t mob_white \t mob_us \t mob_them

use std::fs::File;
use std::io::{BufWriter, Read, Write};

use bullet_lib::game::inputs::{mob_all, shogi_board_to_sfen};
use bullet_lib::shogi::{PackedSfenValue, ShogiBoard};

fn main() {
    let path = std::env::args().nth(1).expect("usage: shogi_mob_canonical <psv> [n]");
    let n: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let mut file = File::open(&path).expect("psv を開けない");
    let mut buf = [0u8; 40];
    let out = std::io::stdout();
    let mut w = BufWriter::new(out.lock());
    writeln!(w, "sfen\tmob_black\tmob_white\tmob_us\tmob_them").unwrap();

    let mut i = 0usize;
    while i < n {
        if file.read_exact(&mut buf).is_err() {
            break;
        }
        // SAFETY: PackedSfenValue は #[repr(C)] で [u8;40] 1 個。 40 byte を unaligned read する。
        //   buf は 40 byte ちょうどで、 読み出し先の型も 40 byte なので範囲外参照はない。
        let psv: PackedSfenValue = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const PackedSfenValue) };
        let board = ShogiBoard::from_packed_sfen(&psv);
        let sfen = shogi_board_to_sfen(&board);
        let (mb, mw, mu, mt) = mob_all(&board);
        writeln!(w, "{sfen}\t{mb}\t{mw}\t{mu}\t{mt}").unwrap();
        i += 1;
    }
    w.flush().unwrap();
    eprintln!("{i} 局面を書いた");
}
