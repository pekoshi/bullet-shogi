//! **右玉ボーナスの焼き込みの検算**。 2026-10-18 / 2026-10-19 に 2 度直した。
//!
//! # 何を確かめるか
//!
//! 焼き込みは engine (`p2nnue.cpp` の `migigyoku_bonus_black_view` /
//! `aigakari_migi_bonus_black_view`) の式を **2 回独立に写した**ものとして実装してある:
//!   ① `C:/project/p2shogi/tools/bake_migi_bonus.py` (python / 対照 10 本)
//!   ② `crates/bullet_lib/src/shogi/packed_sfen.rs` の `migi_bonus_stm` (学習器)
//! この器は実データで両方の出力を突き合わせるためのもの。
//!
//! # 🚨🚨 2026-10-19 に踏んだ罠 2 件 (fable のレビューで発覚)
//!
//! **(a) 学習の target は `LoadableDataType::score` を通る**。 焼き込みを
//! `from_packed_sfen` (= **特徴量**の経路) だけに入れていたので、 target には
//! **1 cp も効いていなかった**。 単体テスト 21/21・実データ 386 件一致・関門 3 本を
//! すべて通していたのに、 8 superbatch 回した学習が「焼き込みなし」だった。
//! -> ✅この器で **3 つの経路を全件照合**する:
//!      `psv.score()` (生) / `from_packed_sfen().score` (特徴量側) /
//!      `<PackedSfenValue as LoadableDataType>::score()` (**学習の target**)
//!    後 2 つが食い違ったら **exit 3** で即止める。
//!
//! **(b) 龍 (成飛) を飛車として数えていた**。 engine の `pos.pieces(ROOK)` は
//! YaneuraOu の `byTypeBB[ROOK]` = **飛車のみ** (龍込みは別の合成 bitboard
//! `ROOK_DRAGON` / `position.cpp:1133`)。 python 参照も**同じ誤りを共有**していたため、
//! 「300,000 件で完全一致」は**同じ仮定を 2 回写しただけ**で検出できなかった。
//! -> 突き合わせは「2 つの実装が独立であること」が前提。 片方が他方の写しなら
//!    一致は証拠にならない (★control_must_use_production_path の変種)。
//!
//! ```text
//! usage: migi_bake_verify <pool.psv> <n>
//!   P2_MIGI_BAKE / P2_AIGA_BAKE を設定して走らせる。
//!   出力 (tsv): index, 生の score, 焼いた後の score, ply   ← **焼けた局面だけ**
//!   stderr: 読んだ件数 / 焼けた件数 / 3 経路の照合の結果
//! ```
use std::fs::File;
use std::io::Read;

use bullet_lib::shogi::{PackedSfenValue, ShogiBoard};
use bullet_lib::value::loader::LoadableDataType;

const REC: usize = 40;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: migi_bake_verify <pool.psv> <n>");
        std::process::exit(1);
    }
    let path = &args[1];
    let n: usize = args[2].parse().expect("n must be a number");

    let mut f = File::open(path).expect("cannot open pool");
    let mut buf = vec![0u8; REC * n];
    let mut got = 0usize;
    while got < buf.len() {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) => panic!("read failed: {e}"),
        }
    }
    let n = got / REC;
    eprintln!("[verify] {n} 件を読んだ");

    let mut baked = 0usize;
    let mut baked_target = 0usize;
    for i in 0..n {
        // SAFETY: PackedSfenValue は #[repr(C)] の [u8; 40] 1 個なので、 40 byte 境界の
        //   読み出しはレイアウトとして妥当。 アラインメントは保証できないので
        //   read_unaligned を使う。 buf は 40*n byte あるので範囲内。
        let psv: PackedSfenValue =
            unsafe { std::ptr::read_unaligned(buf[i * REC..].as_ptr() as *const PackedSfenValue) };
        let raw = psv.score();
        let board = ShogiBoard::from_packed_sfen(&psv).score;
        // 🚨🚨**学習の target はこの経路**。 inherent の `psv.score()` と同じ名前だが
        //   別の値を返す (trait 実装が焼き込みを掛ける)。
        let target = <PackedSfenValue as LoadableDataType>::score(&psv);
        if target != board {
            eprintln!(
                "[verify] 🚨🚨idx {i}: **target の経路と board.score が食い違う** \
                 (生 {raw} / 特徴量側 {board} / target {target})\n\
                 [verify]   = 焼き込みが学習の target に乗っていない。 ここで止める。"
            );
            std::process::exit(3);
        }
        if board != raw {
            baked += 1;
            println!("{}\t{}\t{}\t{}", i, raw, board, psv.game_ply());
        }
        if target != raw {
            baked_target += 1;
        }
    }
    eprintln!("[verify] 焼けた局面 {baked} 件 ({:.3}%)", 100.0 * baked as f64 / n as f64);
    eprintln!("[verify] ✅3 経路の照合: 特徴量側 == target が **全 {n} 件で一致**");
    eprintln!("[verify] ✅target が生の score と違う件数 {baked_target} 件 (上と同数なら正しい)");
    if baked != baked_target {
        eprintln!("[verify] 🚨件数が違う -> どちらかの経路で焼け落ちている");
        std::process::exit(4);
    }
}
