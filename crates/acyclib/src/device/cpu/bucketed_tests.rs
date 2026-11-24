//! gather affine (2026-11-14 OPNARCH T2) の CPU 実装の検証。
//!
//! 参照 = 従来の経路「全 bucket の affine を素朴に計算 → `select`」(順伝播) と
//! 「`select_backprop` で全 bucket 分の勾配に戻す → 素朴な転置積」(逆伝播)。
//! さらに有限差分 (中心差分) で勾配を確かめる。 乱数は固定種の xorshift (再現可能)。

use std::sync::Arc;

use super::{CpuBuffer, CpuThread};
use crate::device::{DeviceBuffer, operation::CoreDeviceOps};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// [-1, 1) の一様乱数。
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }
}

fn buf_f32(v: &[f32]) -> CpuBuffer<f32> {
    let mut b = CpuBuffer::<f32>::new(Arc::new(CpuThread), v.len()).unwrap();
    b.load_from_slice(v).unwrap();
    b
}

fn buf_i32(v: &[i32]) -> CpuBuffer<i32> {
    let mut b = CpuBuffer::<i32>::new(Arc::new(CpuThread), v.len()).unwrap();
    b.load_from_slice(v).unwrap();
    b
}

struct Case {
    bs: usize,
    i: usize,
    o: usize,
    nb: usize,
    w: Vec<f32>,
    bias: Vec<f32>,
    x: Vec<f32>,
    bk: Vec<i32>,
    /// 損失 L = Σ_b Σ_j r[b, j] * y[b, j] の重み (= dL/dy)。
    r: Vec<f32>,
}

fn make_case(seed: u64, bs: usize, i: usize, o: usize, nb: usize) -> Case {
    let mut rng = Rng(seed);
    let rows = nb * o;
    let w = (0..rows * i).map(|_| rng.f()).collect();
    let bias = (0..rows).map(|_| rng.f()).collect();
    let x = (0..bs * i).map(|_| rng.f()).collect();
    // 一部の bucket に局面を偏らせ、 使われない bucket も作る (243 の薄い head を模す)。
    let bk = (0..bs).map(|_| (((rng.next() % 7) * (rng.next() % (nb as u64 / 3 + 1))) % nb as u64) as i32).collect();
    let r = (0..bs * o).map(|_| rng.f()).collect();
    Case { bs, i, o, nb, w, bias, x, bk, r }
}

/// 参照の順伝播: 全 bucket の affine (rows 行) を計算してから `select`。
fn reference_fwd(c: &Case) -> Vec<f32> {
    let rows = c.nb * c.o;
    let mut full = vec![0.0f32; c.bs * rows];
    for b in 0..c.bs {
        for r in 0..rows {
            let mut acc = c.bias[r];
            for k in 0..c.i {
                acc += c.w[k * rows + r] * c.x[b * c.i + k];
            }
            full[b * rows + r] = acc;
        }
    }
    let full_b = buf_f32(&full);
    let mut out = buf_f32(&vec![0.0; c.bs * c.o]);
    CpuThread::select(c.bs, true, rows, c.o, &full_b, &buf_i32(&c.bk), &mut out).unwrap();
    out.buf.clone()
}

fn gather_fwd(c: &Case, w: &[f32], bias: &[f32], x: &[f32]) -> Vec<f32> {
    let mut out = buf_f32(&vec![7.0; c.bs * c.o]); // 上書きされることを確かめるため 0 以外で埋める
    CpuThread::bucketed_affine_fwd(
        c.bs,
        c.i,
        c.o,
        c.nb,
        &buf_f32(w),
        &buf_f32(bias),
        &buf_f32(x),
        &buf_i32(&c.bk),
        &mut out,
    )
    .unwrap();
    out.buf.clone()
}

fn loss(c: &Case, y: &[f32]) -> f64 {
    y.iter().zip(&c.r).map(|(&a, &b)| a as f64 * b as f64).sum()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// 参照の逆伝播: dy を `select_backprop` で全 bucket 分 (rows) に戻し、 素朴な転置積で dW / dbias / dx を作る。
fn reference_bwd(c: &Case) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = c.nb * c.o;
    let mut dfull = buf_f32(&vec![0.0; c.bs * rows]);
    CpuThread::select_backprop(c.bs, true, rows, c.o, &buf_i32(&c.bk), &buf_f32(&c.r), &mut dfull).unwrap();
    let dfull = dfull.buf;
    let mut dw = vec![0.0f32; rows * c.i];
    let mut db = vec![0.0f32; rows];
    let mut dx = vec![0.0f32; c.bs * c.i];
    for b in 0..c.bs {
        for r in 0..rows {
            let g = dfull[b * rows + r];
            db[r] += g;
            for k in 0..c.i {
                dw[k * rows + r] += g * c.x[b * c.i + k];
                dx[b * c.i + k] += c.w[k * rows + r] * g;
            }
        }
    }
    (dw, db, dx)
}

/// gather 版の逆伝播。 勾配は「加算」なので、 0 以外の初期値 (0.25) に足されることも確かめる。
fn gather_bwd(c: &Case) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = c.nb * c.o;
    let init = 0.25f32;
    let mut dw = buf_f32(&vec![init; rows * c.i]);
    let mut db = buf_f32(&vec![init; rows]);
    let mut dx = buf_f32(&vec![init; c.bs * c.i]);
    let (bk, r) = (buf_i32(&c.bk), buf_f32(&c.r));
    CpuThread::bucketed_affine_bwd_input(c.bs, c.i, c.o, c.nb, &buf_f32(&c.w), &bk, &r, &mut dx).unwrap();
    CpuThread::bucketed_affine_bwd_weights(c.bs, c.i, c.o, c.nb, &buf_f32(&c.x), &bk, &r, &mut dw).unwrap();
    CpuThread::bucketed_affine_bwd_bias(c.bs, c.o, c.nb, &bk, &r, &mut db).unwrap();
    let sub = |v: Vec<f32>| v.into_iter().map(|a| a - init).collect::<Vec<_>>();
    (sub(dw.buf), sub(db.buf), sub(dx.buf))
}

/// (batch, 入力, 1 bucket の出力, bucket 数)。 OPN の fc0 (I=1024 は重いので縮小) / fc1 / fc2 の形と、
/// 8 bucket (従来の select 版の照合) を含める。
const SHAPES: [(usize, usize, usize, usize); 5] =
    [(64, 96, 32, 243), (64, 62, 64, 243), (64, 64, 1, 243), (48, 40, 16, 8), (37, 33, 5, 81)];

#[test]
fn bucketed_affine_forward_matches_select() {
    for (n, &(bs, i, o, nb)) in SHAPES.iter().enumerate() {
        let c = make_case(0x9E37_79B9_7F4A_7C15 ^ n as u64, bs, i, o, nb);
        let d = max_abs_diff(&reference_fwd(&c), &gather_fwd(&c, &c.w, &c.bias, &c.x));
        println!("fwd shape {:?}: max|gather - select| = {d:.3e}", (bs, i, o, nb));
        assert!(d <= 1e-4, "順伝播が select 版と合わない: {d}");
    }
}

#[test]
fn bucketed_affine_backward_matches_select() {
    for (n, &(bs, i, o, nb)) in SHAPES.iter().enumerate() {
        let c = make_case(0xD1B5_4A32_D192_ED03 ^ n as u64, bs, i, o, nb);
        let (rw, rb, rx) = reference_bwd(&c);
        let (gw, gb, gx) = gather_bwd(&c);
        let (dw, db, dx) = (max_abs_diff(&rw, &gw), max_abs_diff(&rb, &gb), max_abs_diff(&rx, &gx));
        println!("bwd shape {:?}: max diff dW {dw:.3e} / dbias {db:.3e} / dx {dx:.3e}", (bs, i, o, nb));
        assert!(dw <= 1e-4 && db <= 1e-4 && dx <= 1e-4, "逆伝播が select 版と合わない");
        // 使われていない bucket の勾配は厳密に 0 (= 初期値のまま)。
        let used: std::collections::HashSet<i32> = c.bk.iter().copied().collect();
        for bkt in 0..nb {
            if !used.contains(&(bkt as i32)) {
                assert!(gb[bkt * o..(bkt + 1) * o].iter().all(|&v| v == 0.0));
            }
        }
    }
}

#[test]
fn bucketed_affine_gradient_matches_finite_difference() {
    // L は W / bias / x のそれぞれについて線形なので、 中心差分は丸め誤差を除いて厳密。
    let (bs, i, o, nb) = (24, 20, 8, 81);
    let c = make_case(42, bs, i, o, nb);
    let (gw, gb, gx) = gather_bwd(&c);
    let eps = 1e-2f32;
    let mut rng = Rng(7);
    let mut worst = 0.0f64;
    for _ in 0..200 {
        let (mut w, mut bias, mut x) = (c.w.clone(), c.bias.clone(), c.x.clone());
        // どの値を揺らすか (0 = 重み / 1 = bias / 2 = 入力)。 重み / bias は使われている bucket から選ぶ。
        let which = rng.next() % 3;
        let b = c.bk[(rng.next() % bs as u64) as usize] as usize;
        let j = (rng.next() % o as u64) as usize;
        let k = (rng.next() % i as u64) as usize;
        let xi = (rng.next() % (bs * i) as u64) as usize;
        let (slot, analytic): (&mut f32, f32) = match which {
            0 => (&mut w[k * nb * o + b * o + j], gw[k * nb * o + b * o + j]),
            1 => (&mut bias[b * o + j], gb[b * o + j]),
            _ => (&mut x[xi], gx[xi]),
        };
        let base = *slot;
        *slot = base + eps;
        let (wp, bp, xp) = (w.clone(), bias.clone(), x.clone());
        let lp = loss(&c, &gather_fwd(&c, &wp, &bp, &xp));
        let slot: &mut f32 = match which {
            0 => &mut w[k * nb * o + b * o + j],
            1 => &mut bias[b * o + j],
            _ => &mut x[xi],
        };
        *slot = base - eps;
        let lm = loss(&c, &gather_fwd(&c, &w, &bias, &x));
        let numeric = (lp - lm) / (2.0 * eps as f64);
        worst = worst.max((numeric - analytic as f64).abs());
    }
    println!("有限差分: 200 点の max|数値 - 解析| = {worst:.3e}");
    assert!(worst < 2e-3, "勾配が有限差分と合わない: {worst}");
}

#[test]
fn bucketed_affine_out_of_range_bucket_is_zero() {
    let mut c = make_case(5, 6, 4, 3, 8);
    c.bk = vec![0, 8, -1, 7, 100, 3];
    let y = gather_fwd(&c, &c.w, &c.bias, &c.x);
    for b in [1usize, 2, 4] {
        assert!(y[b * c.o..(b + 1) * c.o].iter().all(|&v| v == 0.0));
    }
    let (_, _, gx) = gather_bwd(&c);
    for b in [1usize, 2, 4] {
        assert!(gx[b * c.i..(b + 1) * c.i].iter().all(|&v| v == 0.0));
    }
}
