//! PSV から LayerStack の bucket が指定範囲に入る局面だけを抜き出す (2026-11-13)。
//!
//! 学習器 (`examples/shogi_layerstack.rs`) の `--bucket-filter` を学習前に済ませておくための道具。
//! bucket の正は、 学習器が head 選択に使う `bullet_lib::game::outputs::ShogiProgressKPAbs` の
//! `OutputBuckets::bucket` (= 学習器の `BucketFilterLoader` が呼ぶのと同一の関数)。
//!
//! 速さのため、 既定ではその計算を足し算の順序を変えずに書き直した `FastKPAbs` で bucket を出す
//! (ライブラリ版は駒種 x 色ごとに 81 升を舐め直すので 1 局面 5us 以上かかる)。
//! f32 の足し算の順序がライブラリ版と同一なので値は bit 単位で一致する。 念のため
//! `--check-every N` 局面に 1 回ライブラリ版でも計算して照合し、 1 件でも違えば `.done` を
//! 書かずに止まる (`--check-every 1` = 全件照合 / `0` = 照合なし)。
//!
//! 構成:
//!   読み手 1 本 (入力ファイルを 1 本ずつ先頭から順読み = HDD のシークを避ける)
//!     -> 計算 N 本 (bucket を計算して範囲内の 40 byte をそのまま出力 buffer へ)
//!     -> 書き手 1 本 (到着順に追記。 入力の順序は保たない)
//! buffer は固定数を使い回す (定常状態で heap 割り当てなし)。
//!
//! 使い方:
//!   psv_bucket_filter --in a.psv,b.psv --out out.psv --bucket-min 0 --bucket-max 2 \
//!       --bucket-mode progress8kpabs --progress-coeff progress.bin --threads 6
//! 完了すると `<out>.done` に合計を書く。 `<out>` か `<out>.done` が既にあれば何もせず終了する。

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bullet_lib::game::outputs::{
    OutputBuckets, SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS, ShogiProgressKPAbs, hand_bucket2, set_layerstack_p4h2,
};
use bullet_lib::shogi::bona_piece::FE_OLD_END;
use bullet_lib::shogi::types::{BOARD_PIECE_TYPES, HAND_PIECE_TYPES};
use bullet_lib::shogi::{BonaPiece, Color, PackedSfenValue, Piece, Square};
use clap::{Parser, ValueEnum};

const RECORD_SIZE: usize = 40;
/// 1 回の読み込みの局面数 (16,000,000 byte)。 HDD の順読みに十分大きく、 計算スレッドへの分配にも十分細かい。
const CHUNK_RECORDS: usize = 400_000;
const NUM_BUCKETS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Debug, ValueEnum)]
enum BucketMode {
    /// 学習器の `--bucket-mode progress8kpabs` と同じ (bucket = 進行度 8 区分)
    #[value(name = "progress8kpabs")]
    Progress8KPAbs,
    /// 学習器の `--bucket-mode progress4hand2kpabs` と同じ (進行度 4 区分 x 持駒量 2 区分)
    #[value(name = "progress4hand2kpabs")]
    Progress4Hand2KPAbs,
}

/// PSV から bucket が範囲内の局面だけを抜き出す。
#[derive(Parser, Debug)]
struct Args {
    /// 入力 PSV (カンマ区切りで複数可)。 書いた順に 1 本ずつ読む
    #[arg(long = "in", value_delimiter = ',', required = true)]
    inputs: Vec<PathBuf>,
    /// 出力 PSV (既にあれば中止する)
    #[arg(long)]
    out: PathBuf,
    /// 通す bucket の下限 (両端を含む)
    #[arg(long)]
    bucket_min: u8,
    /// 通す bucket の上限 (両端を含む)
    #[arg(long)]
    bucket_max: u8,
    /// bucket の計算方法 (学習器の --bucket-mode と同じ値を渡す)
    #[arg(long, value_enum)]
    bucket_mode: BucketMode,
    /// progress.bin (学習器の --progress-coeff と同じもの)
    #[arg(long)]
    progress_coeff: PathBuf,
    /// bucket を計算するスレッド数 (読み手・書き手は別に 1 本ずつ)
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// 進捗を出す間隔 (秒)
    #[arg(long, default_value_t = 30)]
    progress_secs: u64,
    /// N 局面に 1 回、 学習器と同じライブラリ版の bucket でも計算して照合する
    /// (1 = 全件照合 = ライブラリ版と同じ遅さ / 0 = 照合しない)
    #[arg(long, default_value_t = 64)]
    check_every: u64,
}

/// `ShogiProgressKPAbs::bucket` (bullet_lib/src/game/outputs.rs) の高速版。
///
/// ライブラリ版との違いは「盤上の駒の列挙」だけ: ライブラリ版は (駒種 13 x 色 2) ごとに
/// `ShogiBoard::pieces` で 81 升を舐め直す。 こちらは 81 升を 1 回舐めて (駒種, 色) ごとの
/// 升の表を作り、 同じ順序 (駒種 -> 色 -> 升の昇順 -> 先手玉側, 後手玉側) で重みを足す。
/// f32 の足し算の順序が同一なので和は bit 単位で一致する。 持駒・sigmoid・量子化はライブラリ版の写し。
struct FastKPAbs {
    /// progress.bin を f64 -> f32 に変換した重み (ライブラリ版 `load_from_bin` と同じ変換)
    weights: Box<[f32]>,
    /// `PieceType as usize` -> `BOARD_PIECE_TYPES` の中の位置 (盤上の駒種でなければ `u8::MAX`)
    pt_slot: [u8; 16],
    /// P4H2 (進行度 4 区分 x 持駒量 2 区分) か
    p4h2: bool,
}

impl FastKPAbs {
    fn load(path: &Path, p4h2: bool) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("failed to read '{}': {e}", path.display()))?;
        let expected = SHOGI_PROGRESS_KP_ABS_NUM_WEIGHTS * size_of::<f64>();
        if bytes.len() != expected {
            return Err(format!("progress.bin size mismatch: got {} bytes, expected {}", bytes.len(), expected));
        }
        let weights: Box<[f32]> = bytes
            .chunks_exact(size_of::<f64>())
            .map(|c| f64::from_le_bytes(c.try_into().expect("chunk size is checked")) as f32)
            .collect();
        let mut pt_slot = [u8::MAX; 16];
        for (i, &pt) in BOARD_PIECE_TYPES.iter().enumerate() {
            pt_slot[pt as usize] = i as u8;
        }
        Ok(Self { weights, pt_slot, p4h2 })
    }

    /// KP-absolute の重みの和 (ライブラリ版 `ShogiProgressKPAbs::progress` の sigmoid の手前)。
    fn weight_sum(&self, pos: &PackedSfenValue) -> f32 {
        let board = pos.decode();
        let mut sum = 0.0f32;
        if !board.black_king_sq.is_valid() || !board.white_king_sq.is_valid() {
            return sum;
        }
        let w = &self.weights;
        let sq_bk = board.black_king_sq.index();
        let sq_wk = board.white_king_sq.inverse().index();

        // (駒種の位置 x 2 + 色) ごとの升の表。 81 升を昇順に舐めるので表の中も昇順になる。
        const SLOTS: usize = BOARD_PIECE_TYPES.len() * 2;
        let mut sqs = [[0u8; 81]; SLOTS];
        let mut cnt = [0usize; SLOTS];
        for (i, p) in board.board.iter().enumerate() {
            let k = self.pt_slot[p.piece_type as usize];
            if k == u8::MAX {
                continue;
            }
            let slot = usize::from(k) * 2 + usize::from(p.color == Color::White);
            sqs[slot][cnt[slot]] = i as u8;
            cnt[slot] += 1;
        }
        for (k, &pt) in BOARD_PIECE_TYPES.iter().enumerate() {
            for (c, color) in [Color::Black, Color::White].into_iter().enumerate() {
                let slot = k * 2 + c;
                let piece = Piece::new(color, pt);
                for &sq in &sqs[slot][..cnt[slot]] {
                    let sq = Square(sq);
                    let bp_b = BonaPiece::from_piece_square(piece, sq, Color::Black);
                    if bp_b != BonaPiece::ZERO {
                        sum += w[sq_bk * FE_OLD_END + bp_b.value() as usize];
                    }
                    let bp_w = BonaPiece::from_piece_square(piece, sq, Color::White);
                    if bp_w != BonaPiece::ZERO {
                        sum += w[sq_wk * FE_OLD_END + bp_w.value() as usize];
                    }
                }
            }
        }

        for owner in [Color::Black, Color::White] {
            let hand = if owner == Color::Black { board.black_hand } else { board.white_hand };
            for &pt in &HAND_PIECE_TYPES {
                let count = hand.count(pt);
                for c in 1..=count {
                    let bp_b = BonaPiece::from_hand_piece(Color::Black, owner, pt, c);
                    if bp_b != BonaPiece::ZERO {
                        sum += w[sq_bk * FE_OLD_END + bp_b.value() as usize];
                    }
                    let bp_w = BonaPiece::from_hand_piece(Color::White, owner, pt, c);
                    if bp_w != BonaPiece::ZERO {
                        sum += w[sq_wk * FE_OLD_END + bp_w.value() as usize];
                    }
                }
            }
        }
        sum
    }

    /// ライブラリ版 `OutputBuckets::bucket` と同じ値 (soft bucket なし = `BUCKET_SOFT_PROB` 未設定が前提)。
    fn bucket(&self, pos: &PackedSfenValue) -> u8 {
        let sum = self.weight_sum(pos);
        let p = (1.0 / (1.0 + (-sum).exp())).clamp(0.0, 1.0);
        let scaled = p * 8.0;
        let bucket8 = (scaled.floor() as i32).clamp(0, 7) as u8;
        if self.p4h2 {
            let progress_bucket4 = usize::from(bucket8 / 2);
            (progress_bucket4 * 2 + usize::from(hand_bucket2(pos, progress_bucket4))) as u8
        } else {
            bucket8
        }
    }
}

/// 全スレッドで共有する集計。
struct Stats {
    read_records: AtomicU64,
    passed_records: AtomicU64,
    written_records: AtomicU64,
    hist: [AtomicU64; NUM_BUCKETS],
    /// ライブラリ版と照合した件数 / 食い違った件数
    checked: AtomicU64,
    mismatched: AtomicU64,
    failed: AtomicBool,
}

impl Stats {
    fn new() -> Self {
        Self {
            read_records: AtomicU64::new(0),
            passed_records: AtomicU64::new(0),
            written_records: AtomicU64::new(0),
            hist: std::array::from_fn(|_| AtomicU64::new(0)),
            checked: AtomicU64::new(0),
            mismatched: AtomicU64::new(0),
            failed: AtomicBool::new(false),
        }
    }

    fn hist_snapshot(&self) -> [u64; NUM_BUCKETS] {
        std::array::from_fn(|b| self.hist[b].load(Ordering::Relaxed))
    }
}

fn fail(msg: impl AsRef<str>) -> ! {
    eprintln!("ERROR: {}", msg.as_ref());
    std::process::exit(1);
}

fn done_path(out: &Path) -> PathBuf {
    let mut s = out.as_os_str().to_owned();
    s.push(".done");
    PathBuf::from(s)
}

fn format_hist(hist: &[u64; NUM_BUCKETS]) -> String {
    let total: u64 = hist.iter().sum();
    hist.iter()
        .enumerate()
        .map(|(b, &n)| format!("b{b}={n}({:.2}%)", 100.0 * n as f64 / total.max(1) as f64))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `buf` を EOF か満杯まで読む。 読めた byte 数を返す。
fn read_full(file: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match file.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

/// 読み手: 入力を 1 本ずつ順に読み、 `CHUNK_RECORDS` 局面ずつ計算スレッドへ渡す。
fn reader_main(inputs: Vec<PathBuf>, free_in: Receiver<Vec<u8>>, work_tx: SyncSender<Vec<u8>>, stats: Arc<Stats>) {
    for path in &inputs {
        let mut file = File::open(path).unwrap_or_else(|e| fail(format!("{} を開けない: {e}", path.display())));
        loop {
            if stats.failed.load(Ordering::Relaxed) {
                return;
            }
            let Ok(mut buf) = free_in.recv() else { return };
            buf.resize(CHUNK_RECORDS * RECORD_SIZE, 0);
            let n = read_full(&mut file, &mut buf)
                .unwrap_or_else(|e| fail(format!("{} の読み込みに失敗: {e}", path.display())));
            if n % RECORD_SIZE != 0 {
                // 事前に大きさを検査しているので、 ここに来るのは読み込み中にファイルが変わった場合だけ。
                fail(format!("{} の末尾が 40 byte の倍数でない (読めた {n} byte)", path.display()));
            }
            if n == 0 {
                break;
            }
            buf.truncate(n);
            stats.read_records.fetch_add((n / RECORD_SIZE) as u64, Ordering::Relaxed);
            if work_tx.send(buf).is_err() {
                return;
            }
            if n < CHUNK_RECORDS * RECORD_SIZE {
                break;
            }
        }
    }
    // work_tx を落とすと計算スレッドが終わる。
}

/// 計算スレッドの設定。
#[derive(Clone, Copy)]
struct WorkerCfg {
    lo: u8,
    hi: u8,
    fast: &'static FastKPAbs,
    check_every: u64,
}

/// 計算スレッド: bucket が範囲内の局面だけを元の 40 byte のまま出力 buffer へ写す。
fn worker_main(
    work_rx: Arc<Mutex<Receiver<Vec<u8>>>>,
    free_in_tx: SyncSender<Vec<u8>>,
    free_out_rx: Arc<Mutex<Receiver<Vec<u8>>>>,
    write_tx: SyncSender<Vec<u8>>,
    cfg: WorkerCfg,
    stats: Arc<Stats>,
) {
    let WorkerCfg { lo, hi, fast, check_every } = cfg;
    let library = ShogiProgressKPAbs;
    let mut psv = PackedSfenValue::default();
    loop {
        // 受け取りの間だけ lock を持つ (計算中は持たない)。
        let msg = work_rx.lock().expect("work_rx の lock が壊れた").recv();
        let Ok(input) = msg else { return };
        let Ok(mut out) = free_out_rx.lock().expect("free_out_rx の lock が壊れた").recv() else { return };
        out.clear();

        let mut hist = [0u64; NUM_BUCKETS];
        let (mut checked, mut mismatched) = (0u64, 0u64);
        for (i, rec) in input.chunks_exact(RECORD_SIZE).enumerate() {
            psv.as_bytes_mut().copy_from_slice(rec);
            let b = fast.bucket(&psv);
            if check_every > 0 && (i as u64) % check_every == 0 {
                checked += 1;
                let lib = library.bucket(&psv);
                if lib != b {
                    mismatched += 1;
                    if mismatched <= 5 {
                        eprintln!("ERROR: bucket の食い違い: 高速版 {b} / ライブラリ版 {lib} (chunk 内 {i} 局面目)");
                    }
                }
            }
            hist[usize::from(b)] += 1;
            if b >= lo && b <= hi {
                out.extend_from_slice(rec);
            }
        }

        stats.checked.fetch_add(checked, Ordering::Relaxed);
        if mismatched > 0 {
            stats.mismatched.fetch_add(mismatched, Ordering::Relaxed);
            stats.failed.store(true, Ordering::Relaxed);
        }
        let passed: u64 = hist[usize::from(lo)..=usize::from(hi)].iter().sum();
        stats.passed_records.fetch_add(passed, Ordering::Relaxed);
        for (b, n) in hist.iter().enumerate() {
            stats.hist[b].fetch_add(*n, Ordering::Relaxed);
        }
        // 読み手が先に終わっていれば返却先が無いが、 その buffer はもう要らない。
        let _ = free_in_tx.send(input);
        if write_tx.send(out).is_err() {
            return;
        }
    }
}

/// 書き手: 届いた順に出力へ追記する。
fn writer_main(mut file: File, write_rx: Receiver<Vec<u8>>, free_out_tx: SyncSender<Vec<u8>>, stats: Arc<Stats>) {
    for buf in write_rx {
        if !buf.is_empty() {
            if let Err(e) = file.write_all(&buf) {
                stats.failed.store(true, Ordering::Relaxed);
                fail(format!("出力の書き込みに失敗: {e}"));
            }
            stats.written_records.fetch_add((buf.len() / RECORD_SIZE) as u64, Ordering::Relaxed);
        }
        let _ = free_out_tx.send(buf);
    }
    if let Err(e) = file.flush().and_then(|_| file.sync_all()) {
        fail(format!("出力の flush に失敗: {e}"));
    }
}

fn main() {
    let args = Args::parse();
    let t0 = Instant::now();

    // ---- 引数の検査 ----
    if args.bucket_min > args.bucket_max || usize::from(args.bucket_max) >= NUM_BUCKETS {
        fail(format!("0 <= --bucket-min <= --bucket-max < {NUM_BUCKETS} であること"));
    }
    if args.threads == 0 {
        fail("--threads は 1 以上");
    }
    // 学習器と同じく、 soft bucket (乱数で bucket を選ぶ) とは併用できない。
    let soft = std::env::var("BUCKET_SOFT_PROB").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);
    if soft > 0.0 {
        fail(format!("BUCKET_SOFT_PROB={soft} (> 0) では bucket が乱数になる。 環境変数を外すこと"));
    }
    let done = done_path(&args.out);
    if args.out.exists() || done.exists() {
        fail(format!("出力 {} か {} が既にある (上書きしない)", args.out.display(), done.display()));
    }
    let mut total_bytes = 0u64;
    for p in &args.inputs {
        let len = std::fs::metadata(p).unwrap_or_else(|e| fail(format!("{} が読めない: {e}", p.display()))).len();
        if len % RECORD_SIZE as u64 != 0 {
            fail(format!("{} の大きさ {len} が 40 の倍数でない", p.display()));
        }
        total_bytes += len;
    }
    let total_records = total_bytes / RECORD_SIZE as u64;

    // ---- bucket 関数の用意 ----
    // ライブラリ版 (照合用) は学習器の main と同じ手順で用意する。
    let p4h2 = args.bucket_mode == BucketMode::Progress4Hand2KPAbs;
    set_layerstack_p4h2(p4h2);
    ShogiProgressKPAbs::load_from_bin(&args.progress_coeff).unwrap_or_else(|e| fail(e));
    let fast: &'static FastKPAbs =
        Box::leak(Box::new(FastKPAbs::load(&args.progress_coeff, p4h2).unwrap_or_else(|e| fail(e))));

    println!(
        "psv_bucket_filter: 入力 {} 本 / {} 局面 ({:.2} GB) / bucket {}..={} / mode {:?} / 計算スレッド {} / 照合 {} 局面に 1 回",
        args.inputs.len(),
        total_records,
        total_bytes as f64 / 1e9,
        args.bucket_min,
        args.bucket_max,
        args.bucket_mode,
        args.threads,
        args.check_every
    );
    println!("出力: {}", args.out.display());

    let out_file =
        File::create_new(&args.out).unwrap_or_else(|e| fail(format!("{} を作れない: {e}", args.out.display())));

    // ---- buffer 池と通り道 ----
    // 入力 buffer: 計算中 threads 個 + 待ち行列 threads 個 + 読み手の 1 個。
    let n_in = args.threads * 2 + 1;
    let n_out = args.threads * 2 + 1;
    let (free_in_tx, free_in_rx) = sync_channel::<Vec<u8>>(n_in);
    let (free_out_tx, free_out_rx) = sync_channel::<Vec<u8>>(n_out);
    for _ in 0..n_in {
        free_in_tx.send(Vec::with_capacity(CHUNK_RECORDS * RECORD_SIZE)).expect("池の初期化");
    }
    for _ in 0..n_out {
        free_out_tx.send(Vec::with_capacity(CHUNK_RECORDS * RECORD_SIZE)).expect("池の初期化");
    }
    let (work_tx, work_rx) = sync_channel::<Vec<u8>>(args.threads);
    let (write_tx, write_rx) = sync_channel::<Vec<u8>>(args.threads);
    let work_rx = Arc::new(Mutex::new(work_rx));
    let free_out_rx = Arc::new(Mutex::new(free_out_rx));
    let stats = Arc::new(Stats::new());

    let reader = {
        let inputs = args.inputs.clone();
        let stats = stats.clone();
        std::thread::spawn(move || reader_main(inputs, free_in_rx, work_tx, stats))
    };
    let cfg = WorkerCfg { lo: args.bucket_min, hi: args.bucket_max, fast, check_every: args.check_every };
    let workers: Vec<_> = (0..args.threads)
        .map(|_| {
            let (work_rx, free_in_tx, free_out_rx, write_tx, stats) =
                (work_rx.clone(), free_in_tx.clone(), free_out_rx.clone(), write_tx.clone(), stats.clone());
            std::thread::spawn(move || worker_main(work_rx, free_in_tx, free_out_rx, write_tx, cfg, stats))
        })
        .collect();
    drop(free_in_tx);
    drop(write_tx);
    let writer = {
        let stats = stats.clone();
        std::thread::spawn(move || writer_main(out_file, write_rx, free_out_tx, stats))
    };

    // ---- 進捗 ----
    let report = |label: &str, last: &mut (Instant, u64)| {
        let read = stats.read_records.load(Ordering::Relaxed);
        let hist = stats.hist_snapshot();
        let done_n: u64 = hist.iter().sum();
        let passed = stats.passed_records.load(Ordering::Relaxed);
        let el = t0.elapsed().as_secs_f64();
        let dt = last.0.elapsed().as_secs_f64().max(1e-9);
        let mbps_avg = done_n as f64 * RECORD_SIZE as f64 / 1e6 / el.max(1e-9);
        let mbps_now = (done_n - last.1) as f64 * RECORD_SIZE as f64 / 1e6 / dt;
        println!(
            "[{label} {:.0}s] 読んだ {read} / 計算済 {done_n} ({:.2}%) / 通した {passed} ({:.2}%) / 書いた {} / {:.1} MB/s (直近 {:.1}) / {:.2} M局面/s / 照合 {} (食い違い {})",
            el,
            100.0 * done_n as f64 / total_records.max(1) as f64,
            100.0 * passed as f64 / done_n.max(1) as f64,
            stats.written_records.load(Ordering::Relaxed),
            mbps_avg,
            mbps_now,
            done_n as f64 / el.max(1e-9) / 1e6,
            stats.checked.load(Ordering::Relaxed),
            stats.mismatched.load(Ordering::Relaxed)
        );
        println!("    分布: {}", format_hist(&hist));
        *last = (Instant::now(), done_n);
    };
    let mut last = (Instant::now(), 0u64);
    while !writer.is_finished() {
        std::thread::sleep(Duration::from_millis(200));
        if last.0.elapsed() >= Duration::from_secs(args.progress_secs) {
            report("進捗", &mut last);
        }
    }
    reader.join().expect("読み手が panic した");
    for w in workers {
        w.join().expect("計算スレッドが panic した");
    }
    writer.join().expect("書き手が panic した");

    // ---- 合計と完了の印 ----
    let hist = stats.hist_snapshot();
    let read = stats.read_records.load(Ordering::Relaxed);
    let computed: u64 = hist.iter().sum();
    let passed = stats.passed_records.load(Ordering::Relaxed);
    let written = stats.written_records.load(Ordering::Relaxed);
    let checked = stats.checked.load(Ordering::Relaxed);
    let mismatched = stats.mismatched.load(Ordering::Relaxed);
    let out_len = std::fs::metadata(&args.out).map(|m| m.len()).unwrap_or(0);
    // 合計行の「直近」は全体平均と同じにする (起点 = 開始時刻・0 局面)。
    let mut last_final = (t0, 0u64);
    report("合計", &mut last_final);
    if mismatched > 0 {
        fail(format!("ライブラリ版との照合で {mismatched} / {checked} 件が食い違った (.done は書かない)"));
    }
    if read != total_records || computed != total_records || written != passed || out_len != passed * RECORD_SIZE as u64
    {
        fail(format!(
            "件数が合わない: 入力 {total_records} / 読んだ {read} / 計算 {computed} / 通した {passed} / 書いた {written} / 出力 {out_len} byte (.done は書かない)"
        ));
    }
    let el = t0.elapsed().as_secs_f64();
    let summary = format!(
        "inputs={}\nbucket_mode={:?}\nbucket_range={}-{}\nprogress_coeff={}\nthreads={}\nread={read}\npassed={passed}\ndropped={}\nhist={}\nchecked={checked}\nmismatched={mismatched}\nelapsed_s={el:.1}\nMBps={:.1}\n",
        args.inputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(","),
        args.bucket_mode,
        args.bucket_min,
        args.bucket_max,
        args.progress_coeff.display(),
        args.threads,
        read - passed,
        hist.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(","),
        total_bytes as f64 / 1e6 / el.max(1e-9),
    );
    std::fs::write(&done, &summary).unwrap_or_else(|e| fail(format!("{} を書けない: {e}", done.display())));
    println!(
        "完了: 読んだ {read} / 通した {passed} ({:.2}%) / 捨てた {} / 照合 {checked} 件 食い違い 0 / {:.1}s / {:.1} MB/s -> {}",
        100.0 * passed as f64 / read.max(1) as f64,
        read - passed,
        el,
        total_bytes as f64 / 1e6 / el.max(1e-9),
        done.display()
    );
}
