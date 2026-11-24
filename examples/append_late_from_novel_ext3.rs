use std::{
    fs::{File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
};
const REC: usize = 40;
const NEED: u64 = 15_676_409;
fn main() -> io::Result<()> {
    let src = r"E:\shogi\p2shogi\data\washiun_dl\novel_ext3.psv";
    let out = r"E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact.psv";
    eprintln!("start src={} out={}", src, out);
    let mut rd = BufReader::with_capacity(64 * 1024 * 1024, File::open(src)?);
    let mut wr = BufWriter::with_capacity(64 * 1024 * 1024, OpenOptions::new().append(true).open(out)?);
    let mut buf = [0u8; REC];
    let mut scanned = 0u64;
    let mut added = 0u64;
    loop {
        match rd.read_exact(&mut buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        scanned += 1;
        let score = i16::from_le_bytes([buf[32], buf[33]]) as i32;
        let ply = u16::from_le_bytes([buf[36], buf[37]]);
        if score.abs() > 500 && ply >= 120 {
            wr.write_all(&buf)?;
            added += 1;
            if (added % 1_000_000) == 0 {
                wr.flush()?;
                eprintln!("scanned={} added={}", scanned, added)
            }
            if added == NEED {
                break;
            }
        }
    }
    wr.flush()?;
    eprintln!("finished scanned={} added={} need={}", scanned, added, NEED);
    if added != NEED {
        return Err(io::Error::other(format!("insufficient late records: {}", added)));
    }
    Ok(())
}
