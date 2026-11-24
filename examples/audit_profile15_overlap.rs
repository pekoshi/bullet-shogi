use std::{
    collections::HashSet,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, BufWriter, Read, Write},
};
const REC: usize = 40;
const NB: usize = 4096;
fn hash(r: &[u8; REC]) -> (u64, u64) {
    let mut a = 0xcbf29ce484222325u64;
    let mut b = 0x9e3779b97f4a7c15u64;
    for &x in r {
        a = (a ^ (x as u64)).wrapping_mul(0x100000001b3);
        b = (b ^ (x as u64).wrapping_add(0x9e37)).rotate_left(13).wrapping_mul(0xff51afd7ed558ccd)
    }
    (a, b)
}
fn part(path: &str, dir: &str) -> io::Result<u64> {
    let mut ws = Vec::with_capacity(NB);
    for i in 0..NB {
        ws.push(BufWriter::new(OpenOptions::new().create(true).append(true).open(format!("{dir}\\b{i:04}.bin"))?));
    }
    let mut rd = BufReader::with_capacity(64 * 1024 * 1024, File::open(path)?);
    let mut r = [0u8; REC];
    let mut n = 0;
    loop {
        match rd.read_exact(&mut r) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let (h1, h2) = hash(&r);
        let k = (h1 as usize) & (NB - 1);
        ws[k].write_all(&h1.to_le_bytes())?;
        ws[k].write_all(&h2.to_le_bytes())?;
        n += 1;
        if n % 100_000_000 == 0 {
            eprintln!("{} hashed={}", path, n)
        }
    }
    for w in ws {
        w.into_inner()?.sync_all()?
    }
    Ok(n)
}
fn main() -> io::Result<()> {
    let root = r"E:\shogi\p2shogi\runs\profile15_overlap_audit";
    let ex = r"E:\shogi\p2shogi\dataset_profile15_rebalance_v2_2500m\profile15_rebalance_v2_2500m_exact.psv";
    let ca = r"E:\shogi\p2shogi\data\washiun_dl\novel_ext3.psv";
    fs::create_dir_all(format!("{root}\\existing"))?;
    fs::create_dir_all(format!("{root}\\candidate"))?;
    let ne = part(ex, &format!("{root}\\existing"))?;
    let nc = part(ca, &format!("{root}\\candidate"))?;
    let mut overlap = 0u64;
    let mut unique = 0u64;
    for i in 0..NB {
        let ep = format!("{root}\\existing\\b{i:04}.bin");
        let cp = format!("{root}\\candidate\\b{i:04}.bin");
        let mut set = HashSet::new();
        if let Ok(mut f) = File::open(ep) {
            let mut x = [0u8; 16];
            while f.read_exact(&mut x).is_ok() {
                set.insert(x);
            }
        }
        if let Ok(mut f) = File::open(cp) {
            let mut x = [0u8; 16];
            while f.read_exact(&mut x).is_ok() {
                if set.contains(&x) { overlap += 1 } else { unique += 1 }
            }
        }
    }
    println!(
        "existing={} candidate={} overlap_records={} candidate_hash_unique_vs_existing={}",
        ne, nc, overlap, unique
    );
    Ok(())
}
