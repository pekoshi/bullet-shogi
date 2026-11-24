use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
const REC: usize = 40;
const TARGET_CORE: u64 = 1_350_000_000;
const TARGET_TRANS: u64 = 450_000_000;
const TARGET_MID: u64 = 450_000_000;
const TARGET_LATE: u64 = 250_000_000;
// Counts confirmed at the last completed source boundary of the first pass.
const BASE_CORE: u64 = 877_762_840;
const BASE_LATE: u64 = 112_476_733;
fn cat(r: &[u8; REC]) -> usize {
    let s = i16::from_le_bytes([r[32], r[33]]) as i32;
    let a = s.abs();
    if a <= 300 {
        0
    } else if a <= 500 {
        1
    } else {
        let p = u16::from_le_bytes([r[36], r[37]]);
        if (40..=119).contains(&p) {
            2
        } else if p >= 120 {
            3
        } else {
            4
        }
    }
}
fn main() -> io::Result<()> {
    let output = r"E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact.psv";
    let mut counts = [0u64; 5];
    let mut r = [0u8; REC];
    {
        let file = File::open(output)?;
        let mut rd = BufReader::with_capacity(64 * 1024 * 1024, file);
        loop {
            match rd.read_exact(&mut r) {
                Ok(()) => counts[cat(&r)] += 1,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
        }
    }
    eprintln!("existing_counts={counts:?}");
    if counts[1] != TARGET_TRANS || counts[2] != TARGET_MID {
        return Err(io::Error::other(format!("unexpected transition/mid counts: {counts:?}")));
    }
    if counts[0] < BASE_CORE || counts[3] < BASE_LATE {
        return Err(io::Error::other(format!("existing counts before base: {counts:?}")));
    }
    let already_extra_core = counts[0] - BASE_CORE;
    let already_extra_late = counts[3] - BASE_LATE;
    let need_core = TARGET_CORE - counts[0];
    let need_late = TARGET_LATE - counts[3];
    let sources = [
        r"E:\shogi\p2shogi\data\washiun_dl\novel_ext2.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\novel_final.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\novel_endgame_bkt6.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\conv_band.psv",
    ];
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, OpenOptions::new().append(true).open(output)?);
    let mut seen_core = 0u64;
    let mut seen_late = 0u64;
    let mut add_core = 0u64;
    let mut add_late = 0u64;
    let mut scanned = 0u64;
    for source in sources {
        let file = File::open(source)?;
        let mut rd = BufReader::with_capacity(64 * 1024 * 1024, file);
        loop {
            match rd.read_exact(&mut r) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            scanned += 1;
            match cat(&r) {
                0 => {
                    if seen_core < already_extra_core {
                        seen_core += 1;
                    } else if add_core < need_core {
                        writer.write_all(&r)?;
                        add_core += 1;
                    }
                }
                3 => {
                    if seen_late < already_extra_late {
                        seen_late += 1;
                    } else if add_late < need_late {
                        writer.write_all(&r)?;
                        add_late += 1;
                    }
                }
                _ => {}
            }
            if add_core == need_core && add_late == need_late {
                break;
            }
        }
        eprintln!(
            "source={source} scanned={scanned} add_core={} add_late={} need_core={} need_late={}",
            add_core, add_late, need_core, need_late
        );
        if add_core == need_core && add_late == need_late {
            break;
        }
    }
    writer.flush()?;
    if add_core != need_core || add_late != need_late {
        return Err(io::Error::other(format!(
            "insufficient data: add_core={add_core} add_late={add_late} need_core={need_core} need_late={need_late}"
        )));
    }
    std::fs::write(
        r"E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact_manifest.json",
        format!(
            "{{\n  \"output_records\":2500000000,\n  \"output_bytes\":100000000000,\n  \"categories\":{{\"core\":1350000000,\"transition\":450000000,\"mid\":450000000,\"late\":250000000}},\n  \"existing_counts_before_final_append\":{:?},\n  \"final_append_core\":{},\n  \"final_append_late\":{},\n  \"source_scan_records_final_append\":{}\n}}\n",
            counts, add_core, add_late, scanned
        ),
    )?;
    eprintln!("completed final append core={} late={} scanned={}", add_core, add_late, scanned);
    Ok(())
}
