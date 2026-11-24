//! Threat profile 間の重み移植ツール (raw.bin / optimiser_state)
//!
//! 学習済み net (通常 `full` profile) の threat 重み行から、より狭い profile
//! (例: `no-pawn-target`) で残る特徴に対応する行だけを抜き出し、新 profile の
//! index 順に並べ替えた checkpoint を書き出す。
//!
//! ## 前提とするファイルレイアウト
//!
//! - `raw.bin` = `value::save::save_unquantised` の fallback 形式:
//!   `"BULLETRW"` (8B) + version u32 + weight count u32 +
//!   各 weight `[id_len u32][id bytes][size u32][f32 * size]` (weight id 昇順)
//! - `optimiser_state/{weights,momentum,velocity,slow}.bin` =
//!   `DenseMatrix::write_to_byte_buffer` 形式:
//!   各 weight `[id bytes]['\n'][single_size usize LE (8B)][f32 * size]`
//! - `l0w`   = Shape (ft_out, input_size) の **column-major** = feature ごとに
//!   `ft_out` 個の f32 が連続 (`values[feat * ft_out + out]`)
//! - `psqtw` = Shape (NUM_BUCKETS, input_size) の column-major = feature ごとに
//!   `NUM_BUCKETS` 個の f32 が連続
//! - feature index は `[0, HALFKA_HM_DIMENSIONS)` が HalfKA_hm、
//!   それ以降が threat。threat 部分のみ並べ替える。
//!
//! ## 使い方
//!
//! ```text
//! threat_profile_subset --src <ckpt_dir> --dst <out_dir> \
//!     [--from-profile full] [--to-profile no-pawn-target] [--skip-optimiser]
//! ```

use std::{
    fs::File,
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use bullet_lib::game::inputs::{HALFKA_HM_DIMENSIONS, ShogiHalfKaHmThreat, ThreatProfile, threat_pair_layout};

/// feature ごとの並べ替え計画
struct FeaturePlan {
    /// src 側の総 feature 数
    src_features: usize,
    /// dst 側の総 feature 数
    dst_features: usize,
    /// `(src_feature_start, feature_len)` を dst 側 index 昇順に並べたもの
    blocks: Vec<(usize, usize)>,
}

impl FeaturePlan {
    fn build(from: ThreatProfile, to: ThreatProfile) -> Result<Self, String> {
        let src_threat = ShogiHalfKaHmThreat::new(from).threat_dimensions();
        let dst_threat = ShogiHalfKaHmThreat::new(to).threat_dimensions();

        let src_layout = threat_pair_layout(from);
        let dst_layout = threat_pair_layout(to);

        let mut blocks = Vec::new();
        // HalfKA_hm 部分はそのまま先頭にコピー
        blocks.push((0, HALFKA_HM_DIMENSIONS));

        let mut expected_base = 0usize;
        for &(pair_idx, base, len) in &dst_layout {
            if base != expected_base {
                return Err(format!("dst layout not contiguous at pair {pair_idx}"));
            }
            expected_base += len;

            let src = src_layout
                .iter()
                .find(|&&(i, _, _)| i == pair_idx)
                .ok_or_else(|| format!("pair {pair_idx} exists in dst profile but not in src profile"))?;
            if src.2 != len {
                return Err(format!("pair {pair_idx}: block length mismatch src={} dst={len}", src.2));
            }
            blocks.push((HALFKA_HM_DIMENSIONS + src.1, len));
        }
        if expected_base != dst_threat {
            return Err(format!("dst threat dims mismatch: {expected_base} vs {dst_threat}"));
        }

        Ok(Self {
            src_features: HALFKA_HM_DIMENSIONS + src_threat,
            dst_features: HALFKA_HM_DIMENSIONS + dst_threat,
            blocks,
        })
    }
}

/// feature ごとに `stride` 個の f32 が連続する weight を並べ替えて書き出す。
fn copy_permuted(
    src: &mut File,
    data_off: u64,
    stride: usize,
    plan: &FeaturePlan,
    out: &mut BufWriter<File>,
) -> std::io::Result<()> {
    let mut buf = vec![0u8; stride * 4 * 4096];
    for &(src_start, len) in &plan.blocks {
        src.seek(SeekFrom::Start(data_off + (src_start * stride * 4) as u64))?;
        let mut remaining = len;
        while remaining > 0 {
            let chunk = remaining.min(4096);
            let bytes = chunk * stride * 4;
            src.read_exact(&mut buf[..bytes])?;
            out.write_all(&buf[..bytes])?;
            remaining -= chunk;
        }
    }
    Ok(())
}

/// weight が feature 次元を持つか (= 並べ替え対象か) を判定し、stride を返す。
fn stride_for(id: &str, size: usize, plan: &FeaturePlan) -> Option<usize> {
    if id != "l0w" && id != "psqtw" {
        return None;
    }
    if size % plan.src_features != 0 {
        panic!("weight {id}: size {size} is not divisible by src feature count {}", plan.src_features);
    }
    Some(size / plan.src_features)
}

fn convert_raw_bin(src_path: &Path, dst_path: &Path, plan: &FeaturePlan) -> std::io::Result<()> {
    let mut src = File::open(src_path)?;
    let mut out = BufWriter::with_capacity(1 << 22, File::create(dst_path)?);

    let mut hdr = [0u8; 16];
    src.read_exact(&mut hdr)?;
    assert_eq!(&hdr[..8], b"BULLETRW", "raw.bin magic mismatch");
    let count = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
    out.write_all(&hdr)?;

    for _ in 0..count {
        let mut u32buf = [0u8; 4];
        src.read_exact(&mut u32buf)?;
        let id_len = u32::from_le_bytes(u32buf) as usize;
        let mut id_bytes = vec![0u8; id_len];
        src.read_exact(&mut id_bytes)?;
        let id = String::from_utf8(id_bytes.clone()).expect("weight id utf8");
        src.read_exact(&mut u32buf)?;
        let size = u32::from_le_bytes(u32buf) as usize;
        let data_off = src.stream_position()?;

        match stride_for(&id, size, plan) {
            Some(stride) => {
                let new_size = plan.dst_features * stride;
                out.write_all(&(id_len as u32).to_le_bytes())?;
                out.write_all(&id_bytes)?;
                out.write_all(&(new_size as u32).to_le_bytes())?;
                copy_permuted(&mut src, data_off, stride, plan, &mut out)?;
                println!("  raw.bin {id}: {size} -> {new_size} f32 (stride {stride})");
            }
            None => {
                out.write_all(&(id_len as u32).to_le_bytes())?;
                out.write_all(&id_bytes)?;
                out.write_all(&(size as u32).to_le_bytes())?;
                src.seek(SeekFrom::Start(data_off))?;
                let mut buf = vec![0u8; size * 4];
                src.read_exact(&mut buf)?;
                out.write_all(&buf)?;
                println!("  raw.bin {id}: copied verbatim ({size} f32)");
            }
        }
        src.seek(SeekFrom::Start(data_off + (size * 4) as u64))?;
    }

    out.flush()
}

fn convert_optimiser_bin(src_path: &Path, dst_path: &Path, plan: &FeaturePlan) -> std::io::Result<()> {
    let mut src = File::open(src_path)?;
    let file_len = src.metadata()?.len();
    let mut out = BufWriter::with_capacity(1 << 22, File::create(dst_path)?);

    loop {
        let pos = src.stream_position()?;
        if pos >= file_len {
            break;
        }
        // id は '\n' 終端
        let mut id_bytes = Vec::new();
        loop {
            let mut b = [0u8; 1];
            src.read_exact(&mut b)?;
            if b[0] == b'\n' {
                break;
            }
            id_bytes.push(b[0]);
        }
        let id = String::from_utf8(id_bytes.clone()).expect("weight id utf8");
        let mut sizebuf = [0u8; 8];
        src.read_exact(&mut sizebuf)?;
        let size = usize::from_le_bytes(sizebuf);
        let data_off = src.stream_position()?;

        match stride_for(&id, size, plan) {
            Some(stride) => {
                let new_size = plan.dst_features * stride;
                out.write_all(&id_bytes)?;
                out.write_all(b"\n")?;
                out.write_all(&new_size.to_le_bytes())?;
                copy_permuted(&mut src, data_off, stride, plan, &mut out)?;
                println!("  {}: {id}: {size} -> {new_size} f32", src_path.file_name().unwrap().to_string_lossy());
            }
            None => {
                out.write_all(&id_bytes)?;
                out.write_all(b"\n")?;
                out.write_all(&size.to_le_bytes())?;
                src.seek(SeekFrom::Start(data_off))?;
                let mut buf = vec![0u8; size * 4];
                src.read_exact(&mut buf)?;
                out.write_all(&buf)?;
            }
        }
        src.seek(SeekFrom::Start(data_off + (size * 4) as u64))?;
    }

    out.flush()
}

fn main() {
    let mut src_dir: Option<PathBuf> = None;
    let mut dst_dir: Option<PathBuf> = None;
    let mut from = "full".to_string();
    let mut to = "no-pawn-target".to_string();
    let mut skip_optimiser = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--src" => src_dir = Some(PathBuf::from(args.next().expect("--src needs a value"))),
            "--dst" => dst_dir = Some(PathBuf::from(args.next().expect("--dst needs a value"))),
            "--from-profile" => from = args.next().expect("--from-profile needs a value"),
            "--to-profile" => to = args.next().expect("--to-profile needs a value"),
            "--skip-optimiser" => skip_optimiser = true,
            other => panic!("unknown argument: {other}"),
        }
    }

    let src_dir = src_dir.expect("--src <checkpoint dir> is required");
    let dst_dir = dst_dir.expect("--dst <output dir> is required");

    let from = ThreatProfile::from_cli(&from)
        .unwrap_or_else(|| panic!("unknown --from-profile (available: {})", ThreatProfile::available()));
    let to = ThreatProfile::from_cli(&to)
        .unwrap_or_else(|| panic!("unknown --to-profile (available: {})", ThreatProfile::available()));

    let plan = FeaturePlan::build(from, to).unwrap_or_else(|e| panic!("cannot build feature plan: {e}"));
    println!(
        "profile {from} (id {}) -> {to} (id {}): features {} -> {} (threat {} -> {})",
        from.profile_id(),
        to.profile_id(),
        plan.src_features,
        plan.dst_features,
        plan.src_features - HALFKA_HM_DIMENSIONS,
        plan.dst_features - HALFKA_HM_DIMENSIONS,
    );

    std::fs::create_dir_all(&dst_dir).expect("create dst dir");

    let raw = src_dir.join("raw.bin");
    assert!(raw.is_file(), "{} not found", raw.display());
    convert_raw_bin(&raw, &dst_dir.join("raw.bin"), &plan).expect("convert raw.bin");

    if !skip_optimiser {
        let src_opt = src_dir.join("optimiser_state");
        if src_opt.is_dir() {
            let dst_opt = dst_dir.join("optimiser_state");
            std::fs::create_dir_all(&dst_opt).expect("create optimiser_state dir");
            for name in ["weights.bin", "momentum.bin", "velocity.bin", "slow.bin"] {
                let p = src_opt.join(name);
                if p.is_file() {
                    convert_optimiser_bin(&p, &dst_opt.join(name), &plan)
                        .unwrap_or_else(|e| panic!("convert {name}: {e}"));
                }
            }
            for name in ["step.txt", "step_ranger.txt"] {
                let p = src_opt.join(name);
                if p.is_file() {
                    std::fs::copy(&p, dst_opt.join(name)).expect("copy step file");
                }
            }
        } else {
            println!("optimiser_state not found, skipped");
        }
    }

    println!("done: {}", dst_dir.display());
}
