use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const REC: usize = 40;
const CORE: u64 = 1_350_000_000;
const TRANSITION: u64 = 450_000_000;
const MID: u64 = 450_000_000;
const LATE: u64 = 250_000_000;

fn category(r: &[u8; REC]) -> usize {
    let score = i16::from_le_bytes([r[32], r[33]]) as i32;
    let abs = score.abs();
    if abs <= 300 {
        0
    } else if abs <= 500 {
        1
    } else {
        let ply = u16::from_le_bytes([r[36], r[37]]);
        if (40..=119).contains(&ply) {
            2
        } else if ply >= 120 {
            3
        } else {
            4
        }
    }
}

fn main() -> io::Result<()> {
    let sources = [
        r"E:\shogi\p2shogi\dataset_dual3v2_dedup_simplemate\dual3v2_all_priority_dedup.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\base_mix2b.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\novel_ext2.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\novel_final.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\novel_endgame_bkt6.psv",
        r"E:\shogi\p2shogi\data\washiun_dl\conv_band.psv",
    ];
    let output = r"E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact.psv";
    let manifest =
        r"E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact_manifest.json";
    if Path::new(output).exists() || Path::new(manifest).exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output or manifest already exists"));
    }
    fs::create_dir_all(Path::new(output).parent().unwrap())?;
    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, File::create(output)?);
    let targets = [CORE, TRANSITION, MID, LATE];
    let mut selected = [0u64; 4];
    let mut eligible = [0u64; 5];
    let mut read_total = 0u64;
    let mut record = [0u8; REC];

    for source in sources {
        let file = File::open(source)?;
        let bytes = file.metadata()?.len();
        if bytes % REC as u64 != 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("not 40-byte aligned: {source}")));
        }
        let mut reader = BufReader::with_capacity(64 * 1024 * 1024, file);
        loop {
            match reader.read_exact(&mut record) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            read_total += 1;
            let c = category(&record);
            eligible[c] += 1;
            if c < 4 && selected[c] < targets[c] {
                writer.write_all(&record)?;
                selected[c] += 1;
            }
            if selected == targets {
                break;
            }
        }
        eprintln!(
            "source={source} read={read_total} selected_core={} selected_transition={} selected_mid={} selected_late={} eligible_early={}",
            selected[0], selected[1], selected[2], selected[3], eligible[4]
        );
        if selected == targets {
            break;
        }
    }
    writer.flush()?;
    if selected != targets {
        return Err(io::Error::other(format!("insufficient source data: selected={selected:?}, targets={targets:?}")));
    }
    let text = format!(
        "{{\n  \"sources\": [{}],\n  \"source_records_read\": {},\n  \"output_records\": {},\n  \"output_bytes\": {},\n  \"targets\": {{\"core\":{},\"transition\":{},\"mid\":{},\"late\":{} }},\n  \"selected\": {{\"core\":{},\"transition\":{},\"mid\":{},\"late\":{} }},\n  \"eligible_by_category_seen\": {{\"core\":{},\"transition\":{},\"mid\":{},\"late\":{},\"early\":{} }},\n  \"selection\": \"streaming first-eligible records per category from each source; no synthetic records; source order is preserved within each input\"\n}}\n",
        sources.iter().map(|s| format!("\"{}\"", s.replace('\\', "\\\\"))).collect::<Vec<_>>().join(","),
        read_total,
        targets.iter().sum::<u64>(),
        targets.iter().sum::<u64>() * REC as u64,
        CORE,
        TRANSITION,
        MID,
        LATE,
        selected[0],
        selected[1],
        selected[2],
        selected[3],
        eligible[0],
        eligible[1],
        eligible[2],
        eligible[3],
        eligible[4]
    );
    fs::write(manifest, text)?;
    eprintln!("completed read={read_total} output_records={} output={output}", targets.iter().sum::<u64>());
    Ok(())
}
