//! KH ブロック (shogi_kh.rs) の検算用ダンプ (2026-11-12)。
//!
//! psv の先頭から `stride` 件おきに `n` 局面を読み、 1 行 1 局面の TSV を出す:
//!   rec  za_e_b  za_e_w  hang_b  hang_w  split  feats_stm  feats_nstm  null_stm  null_nstm
//! feats_* は loader と同じ入口 (`fill_kh_sparse`) で取った index を昇順・カンマ区切りにしたもの。
//! null_* は null モードの出力 (同じ thread で順に流すので「直前の局面の特徴」になるはず)。
//! 突き合わせは `C:/project/p2shogi/runs/sibling/kh_verify.py` (sib_feats.py と独立の Python 実装) で行う。
//!
//! 使い方: kh_feat_dump <psv> <n> <stride> [kmode 1|2] [x off|hx|hxh] > out.tsv  (kmode は `--kh-kmode` と同じ。 既定 1)
//! 2026-11-18 KT: x を hx / hxh にすると X 部 (shogi_kh.rs) も立て、 末尾に X 部の生の値の列
//! `x_b` / `x_w` (色ごとの損な額の上位 2 本、 カンマ区切り) を足す。 突き合わせは
//! `C:/project/p2shogi/runs/tempo_material_20261118/kt_verify.py` (ktprobe4.py の top2 と同じ定義)。

use std::env;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use bullet_lib::game::inputs::shogi_kh::{
    KhKMode, KhParts, KhXMode, fill_kh_sparse, kh_counts, kh_split_half, kt_top2,
};
use bullet_lib::shogi::packed_sfen::PackedSfenValue;

const RECORD_SIZE: usize = 40;

fn feats(pos: &PackedSfenValue, stm: bool, null: bool, parts: KhParts) -> String {
    let mut out = vec![-1i32; parts.max_active()];
    let n = fill_kh_sparse(pos, stm, parts, null, &mut out);
    let mut v: Vec<i32> = out[..n].to_vec();
    v.sort_unstable();
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args.next().expect("usage: kh_feat_dump <psv> <n> <stride>");
    let n: usize = args.next().map(|s| s.parse().expect("n")).unwrap_or(1000);
    let stride: u64 = args.next().map(|s| s.parse().expect("stride")).unwrap_or(1);
    let kmode = match args.next().as_deref() {
        None | Some("1") => KhKMode::Plain,
        Some("2") => KhKMode::Region,
        Some(other) => panic!("kmode は 1 か 2: {other}"),
    };
    let x = match args.next().as_deref() {
        None | Some("off") => KhXMode::Off,
        Some("hx") => KhXMode::Loss,
        Some("hxh") => KhXMode::Hang,
        Some(other) => panic!("x は off / hx / hxh: {other}"),
    };
    let parts = KhParts { k: true, h: true, kmode, x };
    let mut reader = BufReader::with_capacity(1 << 20, File::open(&path).expect("open psv"));
    let mut raw = [0u8; RECORD_SIZE];
    let xcol = if x == KhXMode::Off { "" } else { "\tx_b\tx_w" };
    println!("rec\tza_e_b\tza_e_w\thang_b\thang_w\tsplit\tfeats_stm\tfeats_nstm\tnull_stm\tnull_nstm{xcol}");
    for i in 0..n as u64 {
        let rec = i * stride;
        reader.seek(SeekFrom::Start(rec * RECORD_SIZE as u64)).expect("seek");
        if reader.read_exact(&mut raw).is_err() {
            break;
        }
        let mut psv = PackedSfenValue::default();
        psv.as_bytes_mut().copy_from_slice(&raw);
        let board = psv.board();
        let c = kh_counts(&board);
        let xv = if x == KhXMode::Off {
            String::new()
        } else {
            let t2 = kt_top2(&board, x);
            format!("\t{},{}\t{},{}", t2[0][0], t2[0][1], t2[1][0], t2[1][1])
        };
        println!(
            "{rec}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}{xv}",
            c.za_e[0],
            c.za_e[1],
            c.hang[0],
            c.hang[1],
            kh_split_half(&psv),
            feats(&psv, true, false, parts),
            feats(&psv, false, false, parts),
            feats(&psv, true, true, parts),
            feats(&psv, false, true, parts),
        );
    }
}
