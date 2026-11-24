//! `.crcli8` format reader: CRCL-v2 PTQ output (i8 weight + per-tensor scale) を読む
//!
//! format (`examples/crcl_ptq.rs` で書き出し):
//!   magic: u32 LE = 0xC2C2C201
//!   n_layers: u32 LE
//!   for each layer:
//!     name_len: u8
//!     name: [u8; name_len] (ASCII)
//!     n_elements: u32 LE
//!     scale_w: f32 LE
//!     weight_i8: [i8; n_elements]
//!
//! 使い方:
//! ```ignore
//! let layers = read_crcli8("ckpt.crcli8")?;
//! for layer in &layers {
//!     println!("{}: {} elems, scale={}", layer.name, layer.weight_i8.len(), layer.scale_w);
//! }
//! ```

use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read},
    path::Path,
};

pub const MAGIC: u32 = 0xC2C2C201;

#[derive(Clone, Debug)]
pub struct Crcli8Layer {
    pub name: String,
    pub scale_w: f32,
    pub weight_i8: Vec<i8>,
}

#[derive(Debug)]
pub enum Crcli8Error {
    Io(io::Error),
    BadMagic(u32),
    Eof,
    Utf8(String),
}

impl From<io::Error> for Crcli8Error {
    fn from(e: io::Error) -> Self {
        Crcli8Error::Io(e)
    }
}

fn read_u32(r: &mut impl Read) -> Result<u32, Crcli8Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u8(r: &mut impl Read) -> Result<u8, Crcli8Error> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_f32(r: &mut impl Read) -> Result<f32, Crcli8Error> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

/// `.crcli8` file から全 layer を読み込む。
pub fn read_crcli8<P: AsRef<Path>>(path: P) -> Result<Vec<Crcli8Layer>, Crcli8Error> {
    let mut f = File::open(path)?;
    let magic = read_u32(&mut f)?;
    if magic != MAGIC {
        return Err(Crcli8Error::BadMagic(magic));
    }
    let n_layers = read_u32(&mut f)? as usize;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let name_len = read_u8(&mut f)? as usize;
        let mut name_bytes = vec![0u8; name_len];
        f.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes).map_err(|e| Crcli8Error::Utf8(e.to_string()))?;
        let n_elements = read_u32(&mut f)? as usize;
        let scale_w = read_f32(&mut f)?;
        let mut weight_bytes = vec![0u8; n_elements];
        f.read_exact(&mut weight_bytes)?;
        // SAFETY: i8 is layout-compatible with u8
        let weight_i8: Vec<i8> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(weight_bytes);
            let ptr = v.as_mut_ptr() as *mut i8;
            let len = v.len();
            let cap = v.capacity();
            Vec::from_raw_parts(ptr, len, cap)
        };
        layers.push(Crcli8Layer { name, scale_w, weight_i8 });
    }
    Ok(layers)
}

/// 読んだ layer を name → layer の HashMap に変換 (lookup 用)
pub fn layers_to_map(layers: Vec<Crcli8Layer>) -> HashMap<String, Crcli8Layer> {
    layers.into_iter().map(|l| (l.name.clone(), l)).collect()
}
