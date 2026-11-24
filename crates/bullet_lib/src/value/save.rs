use std::{
    fs::File,
    io::{self, Write},
};

use crate::{
    nn::{ExecutionContext, Graph},
    value::ValueTrainerState,
};
use acyclib::{
    graph::{
        CheckpointShapeAdapter, GraphNodeId, GraphNodeIdTy,
        like::GraphLike,
        save::{GraphWeights, QuantTarget},
    },
    trainer::{Trainer, optimiser::OptimiserState},
};

use crate::{
    game::{inputs::SparseInputType, outputs::OutputBuckets},
    value::{ValueTrainer, loader::LoadableDataType},
};

type ValueTrainerInner<Opt, Inp, Out> = Trainer<ExecutionContext, Graph, Opt, ValueTrainerState<Inp, Out>>;

pub(super) fn write_losses(path: &str, error_record: &[(usize, usize, f32)]) {
    use std::io::Write;

    let mut writer = std::io::BufWriter::new(std::fs::File::create(path).expect("Opening log file failed!"));
    for (superbatch, batch, loss) in error_record {
        writeln!(writer, "{superbatch},{batch},{loss}",).expect("Writing to log file failed!");
    }
}

pub(super) fn save_to_checkpoint<Opt, Inp, Out>(trainer: &ValueTrainerInner<Opt, Inp, Out>, path: &str)
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    std::fs::create_dir(path).unwrap_or(());

    let optimiser_path = format!("{path}/optimiser_state");
    std::fs::create_dir(optimiser_path.as_str()).unwrap_or(());
    trainer.optimiser.write_to_checkpoint(&optimiser_path).unwrap();

    if let Err(e) = save_unquantised(trainer, &format!("{path}/raw.bin")) {
        println!("Failed to write raw network weights:");
        println!("{e}");
    }

    if let Err(e) = save_quantised(trainer, &format!("{path}/quantised.bin")) {
        println!("Failed to write quantised network weights:");
        println!("{e}");
    }
}

pub(super) fn save_unquantised<Opt, Inp, Out>(trainer: &ValueTrainerInner<Opt, Inp, Out>, path: &str) -> io::Result<()>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    let mut file = File::create(path).unwrap();

    let mut buf = Vec::new();

    let has_any_id = trainer.state.saved_format.iter().any(|fmt| fmt.get_id().is_some());

    if has_any_id {
        for fmt in &trainer.state.saved_format {
            if let Some(id) = &fmt.get_id() {
                let idx =
                    GraphNodeId::new(trainer.optimiser.graph.primary().weight_idx(id).unwrap(), GraphNodeIdTy::Values);
                let weights = trainer.optimiser.graph.primary().get(idx).unwrap();
                let weights = weights.dense();

                let mut weight_buf = vec![0.0; weights.size()];
                let written = weights.write_to_slice(&mut weight_buf).unwrap();
                assert_eq!(written, weights.size());

                let quantised = QuantTarget::Float.quantise(false, &weight_buf)?;
                buf.extend_from_slice(&quantised);
            }
        }
    } else {
        // Fallback: SavedFormat に id 付きが無い (= custom transform-only 形式、 例: rshogi format)。
        // raw.bin が 0 byte になる bug を回避するため graph 全 weight を ID 順に f32 dump。
        // フォーマット: magic "BULLETRW" (8 byte) + version u32 (=1) + weight count u32
        //              + 各 weight: id_len u32 + id bytes + size u32 + f32 weights (size * 4 byte)
        const MAGIC: &[u8; 8] = b"BULLETRW";
        const VERSION: u32 = 1;

        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());

        let mut weight_ids = trainer.optimiser.graph.primary().weight_ids();
        weight_ids.sort();
        let count = weight_ids.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());

        for id in &weight_ids {
            let idx =
                GraphNodeId::new(trainer.optimiser.graph.primary().weight_idx(id).unwrap(), GraphNodeIdTy::Values);
            let weights = trainer.optimiser.graph.primary().get(idx).unwrap();
            let weights = weights.dense();
            let size = weights.size();

            let mut weight_buf = vec![0.0f32; size];
            let written = weights.write_to_slice(&mut weight_buf).unwrap();
            assert_eq!(written, size);

            let id_bytes = id.as_bytes();
            buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(id_bytes);
            buf.extend_from_slice(&(size as u32).to_le_bytes());
            for w in &weight_buf {
                buf.extend_from_slice(&w.to_le_bytes());
            }
        }
    }

    file.write_all(&buf)?;

    Ok(())
}

pub(super) fn save_quantised<Opt, Inp, Out>(trainer: &ValueTrainerInner<Opt, Inp, Out>, path: &str) -> io::Result<()>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    let weight_store = GraphWeights::from(trainer.optimiser.graph.primary());

    let mut file = File::create(path).unwrap();
    let mut buf = Vec::new();

    // カスタムフォーマットの有無を判定（trueだとYaneuraOu形式）
    let has_custom_format = trainer.state.saved_format.iter().any(|fmt| fmt.is_custom());
    for fmt in &trainer.state.saved_format {
        buf.extend_from_slice(&fmt.write_to_byte_buffer(&weight_store)?);
    }

    // YaneuraOu形式の場合、パディング不要
    if !has_custom_format {
        let bytes = buf.len() % 64;
        if bytes > 0 {
            let chs = [b'b', b'u', b'l', b'l', b'e', b't'];

            for i in 0..64 - bytes {
                buf.push(chs[i % chs.len()]);
            }
        }
    }

    file.write_all(&buf)?;

    Ok(())
}

impl<Opt, Inp, Out> ValueTrainer<Opt, Inp, Out>
where
    Opt: OptimiserState<ExecutionContext>,
    Inp: SparseInputType,
    Inp::RequiredDataType: LoadableDataType,
    Out: OutputBuckets<Inp::RequiredDataType>,
{
    pub fn load_from_checkpoint(&mut self, path: &str) {
        let err = self.optimiser.load_from_checkpoint(&format!("{path}/optimiser_state"));
        if let Err(e) = err {
            println!();
            println!("Error loading from checkpoint:");
            println!("{e:?}");
            std::process::exit(1);
        }
    }

    /// [`Self::load_from_checkpoint`] と同じだが、 graph と要素数が一致しないテンソル
    /// (重み・optimiser state とも) を `adapter` で graph 側の形状へ変換して読み込む。
    pub fn load_from_checkpoint_adapted(&mut self, path: &str, adapter: &dyn CheckpointShapeAdapter) {
        let err = self.optimiser.load_from_checkpoint_adapted(&format!("{path}/optimiser_state"), adapter);
        if let Err(e) = err {
            println!();
            println!("Error loading from checkpoint:");
            println!("{e:?}");
            std::process::exit(1);
        }
    }

    pub fn save_to_checkpoint(&self, path: &str) {
        save_to_checkpoint(&self.0, path);
    }

    pub fn save_quantised(&self, path: &str) -> io::Result<()> {
        save_quantised(self, path)
    }

    pub fn save_unquantised(&self, path: &str) -> io::Result<()> {
        save_unquantised(self, path)
    }
}
