use std::{env, fs::File, io::Read};

use bullet_lib::{
    game::inputs::{SparseInputType, ShogiHalfKaHmThreatDirect15, THREAT_DIRECT15_DIMENSIONS},
    shogi::PackedSfenValue,
};

const MAGIC: &[u8; 8] = b"P2THRD2\0";
const HALFKA_DIM: usize = 73_305;
const BUCKET: usize = 0;

fn read_direct_weights(path: &str) -> Vec<i32> {
    let data = std::fs::read(path).expect("read quantised.bin");
    let offset = data.windows(MAGIC.len()).position(|w| w == MAGIC).expect("P2THRD2 section");
    let dims = u32::from_le_bytes(data[offset + 16..offset + 20].try_into().unwrap()) as usize;
    let buckets = u32::from_le_bytes(data[offset + 20..offset + 24].try_into().unwrap()) as usize;
    assert_eq!(dims, THREAT_DIRECT15_DIMENSIONS, "P2THRD2 Direct15 dims");
    assert_eq!(buckets, 8, "P2THRD2 buckets");
    let payload = offset + 24;
    (0..dims * buckets)
        .map(|i| i32::from_le_bytes(data[payload + 4 * i..payload + 4 * (i + 1)].try_into().unwrap()))
        .collect()
}

fn main() {
    let mut args = env::args().skip(1);
    let psv_path = args.next().expect("usage: profile15_direct_payload_activity <psv> <quantised.bin> [records]");
    let quantised_path = args.next().expect("missing quantised.bin");
    let requested = args.next().map(|x| x.parse::<usize>().expect("records integer")).unwrap_or(100_000);
    let weights = read_direct_weights(&quantised_path);
    let input = ShogiHalfKaHmThreatDirect15::new();
    let mut psv = PackedSfenValue::default();
    let mut record = [0u8; 40];
    let mut file = File::open(psv_path).expect("open PSV");

    let mut scanned = 0usize;
    let mut stm_active = 0usize;
    let mut nstm_active = 0usize;
    let mut stm_nonzero = 0usize;
    let mut nstm_nonzero = 0usize;
    let mut equal_delta = 0usize;
    let mut nonzero_delta = 0usize;
    let mut sum_abs_delta: i128 = 0;
    let mut max_abs_delta: i64 = 0;

    while scanned < requested {
        match file.read_exact(&mut record) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("read PSV: {e}"),
        }
        psv.as_bytes_mut().copy_from_slice(&record);
        let mut stm_sum: i64 = 0;
        let mut nstm_sum: i64 = 0;
        input.map_features_split(&psv, |stm, nstm| {
            if let Some(idx) = stm {
                if idx >= HALFKA_DIM {
                    let local = idx - HALFKA_DIM;
                    let value = weights[BUCKET * THREAT_DIRECT15_DIMENSIONS + local];
                    stm_active += 1;
                    if value != 0 { stm_nonzero += 1; }
                    stm_sum += i64::from(value);
                }
            }
            if let Some(idx) = nstm {
                if idx >= HALFKA_DIM {
                    let local = idx - HALFKA_DIM;
                    let value = weights[BUCKET * THREAT_DIRECT15_DIMENSIONS + local];
                    nstm_active += 1;
                    if value != 0 { nstm_nonzero += 1; }
                    nstm_sum += i64::from(value);
                }
            }
        });
        let delta = stm_sum - nstm_sum;
        if delta == 0 { equal_delta += 1; } else { nonzero_delta += 1; }
        let abs = delta.abs();
        sum_abs_delta += i128::from(abs);
        max_abs_delta = max_abs_delta.max(abs);
        scanned += 1;
    }

    println!(
        "{{\"records\":{scanned},\"bucket\":{BUCKET},\"stm_active\":{stm_active},\"nstm_active\":{nstm_active},\"stm_nonzero\":{stm_nonzero},\"nstm_nonzero\":{nstm_nonzero},\"stm_nonzero_pct\":{:.6},\"nstm_nonzero_pct\":{:.6},\"zero_delta_records\":{equal_delta},\"nonzero_delta_records\":{nonzero_delta},\"mean_abs_delta\":{:.6},\"max_abs_delta\":{max_abs_delta}}}",
        100.0 * stm_nonzero as f64 / stm_active.max(1) as f64,
        100.0 * nstm_nonzero as f64 / nstm_active.max(1) as f64,
        sum_abs_delta as f64 / scanned.max(1) as f64,
    );
}
