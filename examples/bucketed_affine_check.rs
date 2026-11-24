//! gather affine (2026-11-14 OPNARCH T2) の backend 照合と所要時間。
//!
//! CPU の素朴な参照実装 (`acyclib::device::cpu`) と、 この exe の backend (CUDA ビルドなら GPU) の
//! `bucketed_affine_{fwd,bwd_input,bwd_weights,bwd_bias}` を同じ乱数入力で比べ、 最大誤差を出す。
//! その後 OPN 案 A の形 (B=4096、 fc0 1024→32 / fc1 62→64 / fc2 64→1、 243 bucket) で 1 回あたりの時間を測る。
//!
//! 🚨GPU を使うのは数十 MB だけ (学習ではない)。 それでも GPU で学習中は走らせないこと (メモリ 4GB を共有)。
//!
//! usage (CUDA): cargo run --release --example bucketed_affine_check --target-dir target_bf
//!        (CPU) : cargo run --release --example bucketed_affine_check --no-default-features --features cpu --target-dir target_bf_cpu

use std::{sync::Arc, time::Instant};

use bullet_lib::{
    acyclib::device::{Device, DeviceBuffer, cpu::CpuThread, operation::CoreDeviceOps},
    nn::ExecutionContext,
};

type Dev = ExecutionContext;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 * 2.0 - 1.0
    }
}

/// 243 のうち 143 個に偏って入る分布 (設計書 1.1 の実効 38 / 上位 4 で 36.5% 程度を粗く模す)。
fn skewed_buckets(rng: &mut Rng, bs: usize, nb: usize) -> Vec<i32> {
    let used = (nb * 143 / 243).max(1);
    (0..bs)
        .map(|_| {
            // 2 つの一様乱数の積 = 小さい番号ほど多い。 番号を nb の中に散らす。
            let a = (rng.next() % used as u64) as usize;
            let b = (rng.next() % used as u64) as usize;
            ((a * b / used) * 37 % nb) as i32
        })
        .collect()
}

fn dev_f32(dev: &Arc<Dev>, v: &[f32]) -> <Dev as Device>::BufferF32 {
    let mut b = <Dev as Device>::BufferF32::new(dev.clone(), v.len()).unwrap();
    b.load_from_slice(v).unwrap();
    b
}

fn dev_i32(dev: &Arc<Dev>, v: &[i32]) -> <Dev as Device>::BufferI32 {
    let mut b = <Dev as Device>::BufferI32::new(dev.clone(), v.len()).unwrap();
    b.load_from_slice(v).unwrap();
    b
}

fn cpu_f32(v: &[f32]) -> <CpuThread as Device>::BufferF32 {
    let mut b = <CpuThread as Device>::BufferF32::new(Arc::new(CpuThread), v.len()).unwrap();
    b.load_from_slice(v).unwrap();
    b
}

fn cpu_i32(v: &[i32]) -> <CpuThread as Device>::BufferI32 {
    let mut b = <CpuThread as Device>::BufferI32::new(Arc::new(CpuThread), v.len()).unwrap();
    b.load_from_slice(v).unwrap();
    b
}

fn read_dev(dev: &Arc<Dev>, b: &<Dev as Device>::BufferF32, n: usize) -> Vec<f32> {
    dev.synchronise().unwrap();
    let mut v = vec![0.0f32; n];
    b.write_into_slice(&mut v, n).unwrap();
    v
}

fn read_cpu(b: &<CpuThread as Device>::BufferF32, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    b.write_into_slice(&mut v, n).unwrap();
    v
}

/// 相対誤差 max|a-b| / max(1, max|a|)。
fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let m = a.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max) / m
}

/// 1 つの形で CPU と backend を比べる。 返り値 = 4 演算の相対誤差の最大。
fn compare(dev: &Arc<Dev>, seed: u64, bs: usize, i: usize, o: usize, nb: usize) -> f32 {
    let mut rng = Rng(seed);
    let rows = nb * o;
    let w: Vec<f32> = (0..rows * i).map(|_| rng.f() * 0.1).collect();
    let bias: Vec<f32> = (0..rows).map(|_| rng.f()).collect();
    let x: Vec<f32> = (0..bs * i).map(|_| rng.f()).collect();
    let bk = skewed_buckets(&mut rng, bs, nb);
    let gy: Vec<f32> = (0..bs * o).map(|_| rng.f()).collect();
    let init_w: Vec<f32> = (0..rows * i).map(|_| rng.f()).collect();
    let init_b: Vec<f32> = (0..rows).map(|_| rng.f()).collect();
    let init_x: Vec<f32> = (0..bs * i).map(|_| rng.f()).collect();

    // CPU (参照)
    let mut y_c = cpu_f32(&vec![0.0; bs * o]);
    let mut dx_c = cpu_f32(&init_x);
    let mut dw_c = cpu_f32(&init_w);
    let mut db_c = cpu_f32(&init_b);
    let (w_c, b_c, x_c, k_c, g_c) = (cpu_f32(&w), cpu_f32(&bias), cpu_f32(&x), cpu_i32(&bk), cpu_f32(&gy));
    CpuThread::bucketed_affine_fwd(bs, i, o, nb, &w_c, &b_c, &x_c, &k_c, &mut y_c).unwrap();
    CpuThread::bucketed_affine_bwd_input(bs, i, o, nb, &w_c, &k_c, &g_c, &mut dx_c).unwrap();
    CpuThread::bucketed_affine_bwd_weights(bs, i, o, nb, &x_c, &k_c, &g_c, &mut dw_c).unwrap();
    CpuThread::bucketed_affine_bwd_bias(bs, o, nb, &k_c, &g_c, &mut db_c).unwrap();

    // backend
    let mut y_d = dev_f32(dev, &vec![0.0; bs * o]);
    let mut dx_d = dev_f32(dev, &init_x);
    let mut dw_d = dev_f32(dev, &init_w);
    let mut db_d = dev_f32(dev, &init_b);
    let (w_d, b_d, x_d, k_d, g_d) =
        (dev_f32(dev, &w), dev_f32(dev, &bias), dev_f32(dev, &x), dev_i32(dev, &bk), dev_f32(dev, &gy));
    Dev::bucketed_affine_fwd(bs, i, o, nb, &w_d, &b_d, &x_d, &k_d, &mut y_d).unwrap();
    Dev::bucketed_affine_bwd_input(bs, i, o, nb, &w_d, &k_d, &g_d, &mut dx_d).unwrap();
    Dev::bucketed_affine_bwd_weights(bs, i, o, nb, &x_d, &k_d, &g_d, &mut dw_d).unwrap();
    Dev::bucketed_affine_bwd_bias(bs, o, nb, &k_d, &g_d, &mut db_d).unwrap();

    let e = [
        rel_err(&read_cpu(&y_c, bs * o), &read_dev(dev, &y_d, bs * o)),
        rel_err(&read_cpu(&dx_c, bs * i), &read_dev(dev, &dx_d, bs * i)),
        rel_err(&read_cpu(&dw_c, rows * i), &read_dev(dev, &dw_d, rows * i)),
        rel_err(&read_cpu(&db_c, rows), &read_dev(dev, &db_d, rows)),
    ];
    println!(
        "照合 B={bs} I={i} O={o} NB={nb}: 相対誤差 fwd {:.2e} / dx {:.2e} / dW {:.2e} / dbias {:.2e}",
        e[0], e[1], e[2], e[3]
    );
    e.iter().fold(0.0f32, |m, &v| m.max(v))
}

/// 1 層ぶん (fwd + bwd 3 本) の 1 回あたりの時間 [ms]。
fn time_layer(dev: &Arc<Dev>, bs: usize, i: usize, o: usize, nb: usize, reps: usize) -> [f64; 4] {
    let mut rng = Rng(99);
    let rows = nb * o;
    let w = dev_f32(dev, &(0..rows * i).map(|_| rng.f() * 0.1).collect::<Vec<_>>());
    let bias = dev_f32(dev, &vec![0.0; rows]);
    let x = dev_f32(dev, &(0..bs * i).map(|_| rng.f()).collect::<Vec<_>>());
    let k = dev_i32(dev, &skewed_buckets(&mut rng, bs, nb));
    let g = dev_f32(dev, &(0..bs * o).map(|_| rng.f()).collect::<Vec<_>>());
    let mut y = dev_f32(dev, &vec![0.0; bs * o]);
    let mut dx = dev_f32(dev, &vec![0.0; bs * i]);
    let mut dw = dev_f32(dev, &vec![0.0; rows * i]);
    let mut db = dev_f32(dev, &vec![0.0; rows]);
    let mut t = [0.0f64; 4];
    for (n, slot) in t.iter_mut().enumerate() {
        dev.synchronise().unwrap();
        let t0 = Instant::now();
        for _ in 0..reps {
            match n {
                0 => Dev::bucketed_affine_fwd(bs, i, o, nb, &w, &bias, &x, &k, &mut y).unwrap(),
                1 => Dev::bucketed_affine_bwd_input(bs, i, o, nb, &w, &k, &g, &mut dx).unwrap(),
                2 => Dev::bucketed_affine_bwd_weights(bs, i, o, nb, &x, &k, &g, &mut dw).unwrap(),
                _ => Dev::bucketed_affine_bwd_bias(bs, o, nb, &k, &g, &mut db).unwrap(),
            }
        }
        dev.synchronise().unwrap();
        *slot = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    }
    t
}

fn main() {
    let dev = Arc::new(Dev::default());
    println!("=== gather affine: CPU 参照と backend の照合 ===");
    let shapes =
        [(4096, 1024, 32, 243), (4096, 62, 64, 243), (4096, 64, 1, 243), (4096, 256, 16, 8), (1000, 37, 5, 81)];
    let mut worst = 0.0f32;
    for (n, &(bs, i, o, nb)) in shapes.iter().enumerate() {
        worst = worst.max(compare(&dev, 0x1234_5678 + n as u64, bs, i, o, nb));
    }
    println!("最大の相対誤差 {worst:.2e} → {}", if worst < 1e-4 { "一致 (< 1e-4)" } else { "🚨不一致" });

    println!("=== 所要時間 (OPN 案 A の形、 B=4096、 243 bucket) [ms / 回] ===");
    for (name, i, o) in [("fc0 1024->32", 1024, 32), ("fc1   62->64", 62, 64), ("fc2   64->1 ", 64, 1)] {
        let t = time_layer(&dev, 4096, i, o, 243, 20);
        println!(
            "{name}: fwd {:.3} / bwd_input {:.3} / bwd_weights {:.3} / bwd_bias {:.3} / 計 {:.3}",
            t[0],
            t[1],
            t[2],
            t[3],
            t.iter().sum::<f64>()
        );
    }
    if worst >= 1e-4 {
        std::process::exit(1);
    }
}
