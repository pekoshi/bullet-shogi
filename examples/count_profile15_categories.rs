use std::{
    env,
    fs::File,
    io::{self, BufReader, Read},
};
const REC: usize = 40;
fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut r = [0u8; REC];
    for p in args {
        let f = File::open(&p)?;
        let mut rd = BufReader::with_capacity(64 * 1024 * 1024, f);
        let mut c = [0u64; 5];
        loop {
            match rd.read_exact(&mut r) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e),
            }
            let s = i16::from_le_bytes([r[32], r[33]]) as i32;
            let a = s.abs();
            if a <= 300 {
                c[0] += 1
            } else if a <= 500 {
                c[1] += 1
            } else {
                let ply = u16::from_le_bytes([r[36], r[37]]);
                if (40..=119).contains(&ply) {
                    c[2] += 1
                } else if ply >= 120 {
                    c[3] += 1
                } else {
                    c[4] += 1
                }
            }
        }
        println!(
            "{} records={} core={} transition={} mid={} late={} early={}",
            p,
            c.iter().sum::<u64>(),
            c[0],
            c[1],
            c[2],
            c[3],
            c[4]
        );
    }
    Ok(())
}
