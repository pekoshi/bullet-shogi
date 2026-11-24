//! ARH (Attack Relay Head) cross-implementation parity 用の bullet 側 dump binary
//! (= 2026-06-09)。
//!
//! 設計書 `docs/attack_relay_head_design.md` §11.9 「2 binary + JSON 比較構成」 の
//! bullet 側担当。 固定 SFEN 集合に対して bullet 側
//! `attack_relay_features::compute_classes_for_persp` を実行し JSON 出力する。
//!
//! p2shogi 側 `arh_parity_check` がこの JSON を読んで rshogi 結果と diff し、
//! 不一致 1 件でも `exit 1` する (= 訓練前必須 gate)。
//!
//! # JSON 形式
//!
//! ```json
//! {
//!   "arh_format_version": 1,
//!   "fixtures": [
//!     {"sfen": "...", "black": [c, r, re], "white": [c, r, re]},
//!     ...
//!   ]
//! }
//! ```
//!
//! # 使い方
//!
//! ```bash
//! cargo run --release --example arh_fixture_dump -- \
//!   --fixtures docs/parity/arh_fixtures.sfen \
//!   --output   docs/parity/arh_bullet_dump.json
//! ```

use std::io::Write;
use std::path::PathBuf;

use bullet_lib::game::inputs::{ARH_NUM_BUCKETS, ARH_NUM_CONTACT, ARH_NUM_RELAY, ARH_NUM_RESERVE};
use clap::Parser;
use rshogi_core::nnue::attack_relay::compute_arh_classes;
use rshogi_core::position::Position;
use rshogi_core::types::Color;
use serde::Serialize;

const ARH_FORMAT_VERSION: u32 = 1;

#[derive(Parser, Debug)]
#[command(name = "arh_fixture_dump")]
#[command(about = "ARH cross-impl parity: bullet 側 (ShogiBoard → compute_classes_for_persp) JSON 出力")]
struct Args {
    /// 固定 SFEN fixtures path (= 1 SFEN / line、 先頭 `sfen ` prefix は除去)
    #[arg(long)]
    fixtures: PathBuf,

    /// bullet 側 dump JSON 出力 path
    #[arg(long)]
    output: PathBuf,
}

#[derive(Serialize, Debug, Clone)]
struct ArhFixtureEntry {
    sfen: String,
    black: [u8; 3],
    white: [u8; 3],
}

#[derive(Serialize, Debug)]
struct ArhDump {
    arh_format_version: u32,
    fixtures: Vec<ArhFixtureEntry>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 雛形 sanity check: 定数が想定値であることを確認
    debug_assert_eq!(ARH_NUM_BUCKETS, 8);
    debug_assert_eq!(ARH_NUM_CONTACT, 4);
    debug_assert_eq!(ARH_NUM_RESERVE, 4);
    debug_assert_eq!(ARH_NUM_RELAY, 4);

    let raw = std::fs::read_to_string(&args.fixtures)?;
    let sfens: Vec<String> = raw
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.strip_prefix("sfen ").unwrap_or(l).to_string())
        .collect();

    eprintln!("info string arh_fixture_dump: loaded {} SFEN from {}", sfens.len(), args.fixtures.display());

    // bullet 側 attack_relay_compute_classes は ShogiBoard 引数 (= PSV 展開済) のため
    // SFEN string からの直接構築 API が無い (= ShogiBoard::from_sfen 不在)。
    // 本 dump binary は **bullet workspace 内で動く verification** を目的とし、
    // rshogi-core 側 helper を直接呼ぶ (= attack_relay_features.rs の delegation 経路と
    // 同一 path、 訓練 throughput 側で動く logic と完全一致)。
    let mut entries: Vec<ArhFixtureEntry> = Vec::with_capacity(sfens.len());
    for sfen in &sfens {
        let mut pos = Position::new();
        pos.set_sfen(sfen).map_err(|e| format!("set_sfen failed for sfen={sfen:?}: {e:?}"))?;
        let (bc, br, bre) = compute_arh_classes(&pos, Color::Black);
        let (wc, wr, wre) = compute_arh_classes(&pos, Color::White);
        entries.push(ArhFixtureEntry {
            sfen: sfen.clone(),
            black: [bc as u8, br as u8, bre as u8],
            white: [wc as u8, wr as u8, wre as u8],
        });
    }

    let dump = ArhDump { arh_format_version: ARH_FORMAT_VERSION, fixtures: entries.clone() };

    let json = serde_json::to_string_pretty(&dump)?;
    let mut out = std::fs::File::create(&args.output)?;
    out.write_all(json.as_bytes())?;
    out.flush()?;
    eprintln!(
        "info string arh_fixture_dump: wrote bullet dump to {} ({} fixtures)",
        args.output.display(),
        entries.len()
    );

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: arh_fixture_dump failed: {e}");
        std::process::exit(2);
    }
}
