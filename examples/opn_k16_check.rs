//! K16 (HalfKA_hm の玉区分 45 -> 16) の添字の検査。 2026-11-14 OPN2048 (設計書 2.4 検証 1)
//!
//! 1. 全局面で `ShogiHalfKA_hmK16::map_features` の emit 列 = `ShogiHalfKA_hm::map_features` の emit 列に
//!    `halfka_k16_index_from_k45` を掛けたもの (順序込み)、 かつ factoriser 込み (`Factorised`) でも同じ関係か。
//! 2. (任意) python (`opn_static2.py k16cmp`) との突き合わせ用に、 1 局面 1 行
//!    `n_stm h_stm n_nstm h_nstm` を書く。 h = 添字を昇順に並べた多項式 hash
//!    (h = (h * 1000003 + idx + 1) mod (2^61 - 1))。 python は盤の升順に列挙するので順序ではなく集合で照合する。
//! CPU だけ・読むだけ。 GPU は使わない。
//!
//! usage: opn_k16_check <psv> [--head N] [--dump out.txt]

use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
};

use bullet_lib::{
    game::inputs::{
        Factorised, PIECE_INPUTS, ShogiHalfKA_hm, ShogiHalfKA_hmK16, ShogiPieceFactor, SparseInputType,
        halfka_k16_index_from_k45,
    },
    shogi::PackedSfenValue,
};

const RECORD: u64 = 40;
const HMOD: u128 = (1u128 << 61) - 1;

fn set_hash(v: &mut [usize]) -> u64 {
    v.sort_unstable();
    let mut h: u128 = 0;
    for &x in v.iter() {
        h = (h * 1_000_003 + x as u128 + 1) % HMOD;
    }
    h as u64
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return Err("usage: opn_k16_check <psv> [--head N] [--dump out.txt]".into());
    }
    let psv = &args[1];
    let total = std::fs::metadata(psv).map_err(|e| format!("{psv}: {e}"))?.len() / RECORD;
    let (mut count, mut dump) = (total, None);
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--head" => {
                count = args.get(i + 1).ok_or("--head N")?.parse::<u64>().map_err(|e| e.to_string())?.min(total);
                i += 2;
            }
            "--dump" => {
                dump = Some(args.get(i + 1).ok_or("--dump out.txt")?.clone());
                i += 2;
            }
            x => return Err(format!("不明な引数 {x}")),
        }
    }
    let k45 = ShogiHalfKA_hm;
    let k16 = ShogiHalfKA_hmK16;
    let f45 = Factorised::from_parts(k45, ShogiPieceFactor);
    let f16 = Factorised::from_parts(k16, ShogiPieceFactor);
    if f16.num_inputs() != 27_693 || k16.num_inputs() != 26_064 {
        return Err(format!("次元が設計と違う: K16 {} / factorise 込み {}", k16.num_inputs(), f16.num_inputs()));
    }

    let mut r = BufReader::with_capacity(1 << 20, File::open(psv).map_err(|e| format!("{psv}: {e}"))?);
    let mut w = match &dump {
        Some(p) => Some(BufWriter::new(File::create(p).map_err(|e| format!("{p}: {e}"))?)),
        None => None,
    };
    let mut pos = PackedSfenValue::default();
    let (mut bad, mut bad_fac, mut feats, mut other_persp, mut other_pos, mut empty) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let other_lo = 15 * PIECE_INPUTS;
    for _ in 0..count {
        r.read_exact(pos.as_bytes_mut()).map_err(|e| e.to_string())?;
        let (mut a, mut b) = (Vec::with_capacity(64), Vec::with_capacity(64));
        k45.map_features(&pos, |s, n| a.push((halfka_k16_index_from_k45(s), halfka_k16_index_from_k45(n))));
        k16.map_features(&pos, |s, n| b.push((s, n)));
        if a != b {
            bad += 1;
        }
        // factoriser 込み: K16 の factorised 列 = K45 の factorised 列で「実特徴 (offset 以上) だけ写像」したもの
        let (mut fa, mut fb) = (Vec::with_capacity(128), Vec::with_capacity(128));
        let off = PIECE_INPUTS;
        let mp = |x: usize| if x >= off { off + halfka_k16_index_from_k45(x - off) } else { x };
        f45.map_features(&pos, |s, n| fa.push((mp(s), mp(n))));
        f16.map_features(&pos, |s, n| fb.push((s, n)));
        if fa != fb || fb.len() != 2 * b.len() {
            bad_fac += 1;
        }
        if b.is_empty() {
            empty += 1;
        }
        feats += b.len() as u64;
        let so = b.first().is_some_and(|&(s, _)| s >= other_lo);
        let no = b.first().is_some_and(|&(_, n)| n >= other_lo);
        other_persp += so as u64 + no as u64;
        other_pos += (so || no) as u64;
        if let Some(w) = w.as_mut() {
            let mut s: Vec<usize> = b.iter().map(|p| p.0).collect();
            let mut n: Vec<usize> = b.iter().map(|p| p.1).collect();
            let (ls, ln) = (s.len(), n.len());
            writeln!(w, "{ls} {} {ln} {}", set_hash(&mut s), set_hash(&mut n)).map_err(|e| e.to_string())?;
        }
    }
    if let Some(mut w) = w {
        w.flush().map_err(|e| e.to_string())?;
    }
    println!(
        "{psv}: 先頭 {count} 局面 / K16 = 写像(K45) の不一致 {bad} / factorise 込みの不一致 {bad_fac} / 特徴なし (片玉) {empty} / \
         平均特徴 {:.2} / 「その他」区分の視点 {other_persp} ({:.3}%) / どちらかが「その他」の局面 {other_pos} ({:.3}%)",
        feats as f64 / count.max(1) as f64,
        100.0 * other_persp as f64 / (2 * count).max(1) as f64,
        100.0 * other_pos as f64 / count.max(1) as f64,
    );
    if bad != 0 || bad_fac != 0 {
        return Err(format!("🚨不一致あり (K16 {bad} / factorise {bad_fac})"));
    }
    Ok(())
}
