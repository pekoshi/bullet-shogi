//! CRCL-v2 PTQ tool: bullet raw.bin (FP32 weights) → Conv weight i8 + scale 抽出
//!
//! raw.bin format (bullet `save_unquantised`):
//!   saved_format 順に f32 bytes 直接 concat (ID なし、 size header なし)
//!   crcl_resnet_psv_train の save_format = [l0w, l0b, v_outw, v_outb, b0_w1, b0_w2, b1_w1, b1_w2, ...]
//!
//! .crcli8 format (CRCL-v2 inference 用):
//!   magic: u32 LE = 0xC2C2C201
//!   n_layers: u32 LE
//!   for each layer:
//!     name_len: u8
//!     name: [u8; name_len] (ASCII)
//!     n_elements: u32 LE
//!     scale_w: f32 LE  (= 127 / max(abs(weight)))
//!     weight_i8: [i8; n_elements]
//!
//! 使い方:
//!   cargo run --release --example crcl_ptq -- \
//!     <raw.bin> <num_blocks> <channels> <output.crcli8>

use std::{
    fs::File,
    io::{Read, Write},
};

const HALFKA_HM_DIM: usize = 73_305;

fn read_f32_block(bytes: &[u8], offset: &mut usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v = f32::from_le_bytes(bytes[*offset..*offset + 4].try_into().unwrap());
        out.push(v);
        *offset += 4;
    }
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let raw_path = args.next().expect("usage: crcl_ptq <raw.bin> <num_blocks> <channels> <out.crcli8>");
    let num_blocks: usize = args.next().expect("num_blocks").parse().expect("usize");
    let channels: usize = args.next().expect("channels").parse().expect("usize");
    let out_path = args.next().expect("out.crcli8");

    let dense_size = channels * 81;
    let l0w_size = HALFKA_HM_DIM * dense_size;
    let l0b_size = dense_size;
    let v_outw_size = dense_size;
    let v_outb_size = 1usize;
    let conv_size = channels * channels * 3 * 3;
    let pre_conv_f32 = l0w_size + l0b_size + v_outw_size + v_outb_size;
    let total_conv_f32 = num_blocks * 2 * conv_size;
    let expected_size = (pre_conv_f32 + total_conv_f32) * 4;

    println!(
        "expected raw.bin size: pre_conv {} f32 ({:.1} MB) + {} conv layers × {} f32 = {} bytes total",
        pre_conv_f32,
        (pre_conv_f32 * 4) as f64 / 1e6,
        num_blocks * 2,
        conv_size,
        expected_size,
    );

    let mut bytes = Vec::new();
    File::open(&raw_path).and_then(|mut f| f.read_to_end(&mut bytes)).expect("read raw.bin");
    println!("read {} bytes from {raw_path}", bytes.len());

    if bytes.len() < expected_size {
        eprintln!(
            "WARNING: raw.bin size {} < expected {} (差 {})、 conv weight が含まれていない可能性",
            bytes.len(),
            expected_size,
            expected_size as i64 - bytes.len() as i64
        );
    }

    // skip pre-conv (l0w + l0b + v_outw + v_outb)
    let mut offset = pre_conv_f32 * 4;
    println!("skipping pre-conv ({} bytes)、 conv weight 開始 offset = {}", offset, offset);

    // read conv weights in order
    let mut layers: Vec<(String, Vec<f32>)> = Vec::with_capacity(num_blocks * 2);
    for b in 0..num_blocks {
        for suffix in ["w1", "w2"] {
            let values = read_f32_block(&bytes, &mut offset, conv_size);
            layers.push((format!("b{b}_{suffix}"), values));
        }
    }

    // PTQ
    let mut out = File::create(&out_path).expect("create output");
    out.write_all(&0xC2C2C201u32.to_le_bytes()).unwrap();
    out.write_all(&(layers.len() as u32).to_le_bytes()).unwrap();

    for (name, values) in &layers {
        let max_abs = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale: f32 = if max_abs > 1e-9 { 127.0 / max_abs } else { 127.0 };
        let i8_vals: Vec<i8> = values
            .iter()
            .map(|&v| {
                let q = (v * scale).clamp(-127.0, 127.0).round();
                q as i8
            })
            .collect();

        // 量子化誤差統計
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq_fp32 = 0.0f64;
        for (&fp32, &i8v) in values.iter().zip(i8_vals.iter()) {
            let dequant = (i8v as f32) / scale;
            sum_sq_err += ((fp32 - dequant) as f64).powi(2);
            sum_sq_fp32 += (fp32 as f64).powi(2);
        }
        let rel_rmse = if sum_sq_fp32 > 0.0 { (sum_sq_err / sum_sq_fp32).sqrt() } else { 0.0 };
        println!(
            "  {:>10}  n={:7}  max_abs={:.5}  scale={:.2}  rel_rmse={:.2}%",
            name,
            values.len(),
            max_abs,
            scale,
            rel_rmse * 100.0
        );

        // write
        let name_bytes = name.as_bytes();
        out.write_all(&[name_bytes.len() as u8]).unwrap();
        out.write_all(name_bytes).unwrap();
        out.write_all(&(values.len() as u32).to_le_bytes()).unwrap();
        out.write_all(&scale.to_le_bytes()).unwrap();
        let i8_bytes: &[u8] = unsafe { std::slice::from_raw_parts(i8_vals.as_ptr() as *const u8, i8_vals.len()) };
        out.write_all(i8_bytes).unwrap();
    }
    out.flush().unwrap();
    drop(out);

    let out_size = std::fs::metadata(&out_path).unwrap().len();
    println!("\nwrote {out_path} ({} bytes、 {} conv layers)", out_size, layers.len());
}
