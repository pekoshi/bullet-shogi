//! auxcheap emit CPU スループット計測ベンチ (2026-07-23)。
//!
//! psv (PackedSfenValue 40B) を先頭 N 局面読み、 各 input 型の emit 経路
//! (map_features / map_features_split) を単スレッドで回して positions/sec を測る。
//! 最適化前後の比較 (bit-exact 前提の速度検証) 用。
//!
//! 実行:
//! ```
//! CARGO_TARGET_DIR=/e/shogi/bullet_aux_target2 cargo run --release -p bullet_lib \
//!   --no-default-features --features cpu --example bench_auxcheap_emit -- \
//!   E:/shogi/p2shogi/data/washiun_dl/novel_final.psv 40000
//! ```

use std::io::Read;
use std::time::Instant;

use bullet_lib::game::inputs::{ShogiHalfKA_hm, ShogiHalfKaHmAux, ShogiHalfKaHmAuxCheap, SparseInputType};
use bullet_lib::shogi::{PackedSfenValue, ShogiBoard};

fn read_psv(path: &str, count: usize) -> Vec<PackedSfenValue> {
    let mut file = std::fs::File::open(path).expect("psv open failed");
    let mut buf = vec![0u8; count * 40];
    let mut read_total = 0usize;
    while read_total < buf.len() {
        let n = file.read(&mut buf[read_total..]).expect("psv read failed");
        if n == 0 {
            break;
        }
        read_total += n;
    }
    let n_pos = read_total / 40;
    let mut out = Vec::with_capacity(n_pos);
    for i in 0..n_pos {
        let mut rec = [0u8; 40];
        rec.copy_from_slice(&buf[i * 40..(i + 1) * 40]);
        // SAFETY: PackedSfenValue は #[repr(C)] 40 byte POD。
        out.push(unsafe { std::mem::transmute::<[u8; 40], PackedSfenValue>(rec) });
    }
    out
}

fn bench<F: FnMut(&PackedSfenValue) -> usize>(name: &str, data: &[PackedSfenValue], reps: usize, mut f: F) {
    // warmup
    let mut sink = 0usize;
    for pos in data.iter().take(1000) {
        sink += f(pos);
    }
    let start = Instant::now();
    for _ in 0..reps {
        for pos in data {
            sink += f(pos);
        }
    }
    let dt = start.elapsed().as_secs_f64();
    let total = (reps * data.len()) as f64;
    println!("{name:<32} {:>10.0} pos/sec  ({:.3}s, {} pos, sink={sink})", total / dt, dt, total as u64);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("E:/shogi/p2shogi/data/washiun_dl/novel_final.psv");
    let count: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(40_000);
    let reps: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(1);

    let data = read_psv(path, count);
    println!("loaded {} positions from {path}", data.len());

    // 1) 局面 decode のみ
    bench("decode(from_packed_sfen)", &data, reps, |pos| {
        let board = ShogiBoard::from_packed_sfen(pos);
        board.side_to_move as usize
    });

    // 2) plain HalfKA_hm (対称 emit)
    bench("plain HalfKA_hm map_features", &data, reps, |pos| {
        let mut n = 0usize;
        ShogiHalfKA_hm.map_features(pos, |_, _| n += 1);
        n
    });

    // 3) aux-FT (KZ/HANG/STATKD 600) split emit
    let aux = ShogiHalfKaHmAux::new();
    bench("aux(600) map_features_split", &data, reps, |pos| {
        let mut n = 0usize;
        aux.map_features_split(pos, |a, b| n += usize::from(a.is_some()) + usize::from(b.is_some()));
        n
    });

    // 4) auxcheap (392) split emit
    let auxcheap = ShogiHalfKaHmAuxCheap::new();
    bench("auxcheap(392) map_features_split", &data, reps, |pos| {
        let mut n = 0usize;
        auxcheap.map_features_split(pos, |a, b| n += usize::from(a.is_some()) + usize::from(b.is_some()));
        n
    });

    // 5) zero-projection 1 step 分の host 側コスト simulation (L1=768 想定 l0w)。
    //    旧実装 = 全行列 get_dense_vals (alloc+zero+D2H copy) + 部分 zero + load (H2D copy)。
    //    ここでは D2H/H2D を host memcpy で代替 (= PCIe より速いので **下限**)。
    //    新実装 = device 上の部分範囲 memset のみ (ここでは host memset で代替)。
    {
        let halfka_dim = 73_305usize;
        let input_size = 73_697usize;
        let ft_out = 768usize;
        let full = input_size * ft_out;
        let (s, e) = (halfka_dim * ft_out, input_size * ft_out);
        let device_side = vec![1.0f32; full];
        let mut sink = 0.0f32;

        let t = Instant::now();
        let iters = 5;
        for _ in 0..iters {
            // 旧: get_dense_vals = vec![0.0; size] + write_to_slice(copy)
            let mut w_vals = vec![0.0f32; full];
            w_vals.copy_from_slice(&device_side); // D2H 代替
            for v in &mut w_vals[s..e] {
                *v = 0.0;
            }
            // 旧: load_dense_from_slice = H2D copy
            let mut dev2 = vec![0.0f32; full];
            dev2.copy_from_slice(&w_vals); // H2D 代替
            sink += dev2[full - 1];
        }
        let old_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

        let mut dev = device_side.clone();
        let t = Instant::now();
        let iters2 = 200;
        for _ in 0..iters2 {
            for v in &mut dev[s..e] {
                *v = 0.0;
            }
            sink += dev[s];
        }
        let new_ms = t.elapsed().as_secs_f64() * 1000.0 / iters2 as f64;
        println!(
            "zero-projection sim (l0w {input_size}x{ft_out} = {:.1} MB): old(full roundtrip, host-memcpy lower bound) {old_ms:.1} ms/step, new(range memset) {new_ms:.3} ms/step (sink={sink})",
            full as f64 * 4.0 / 1048576.0
        );
    }
}
