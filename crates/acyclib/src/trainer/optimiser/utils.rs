use std::borrow::Cow;

use crate::{
    device::{
        Device, OperationError,
        tensor::{DenseMatrix, read_from_byte_buffer},
    },
    graph::CheckpointShapeAdapter,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Before,
    After,
}

/// checkpoint のバッファ `values` を graph 側の要素数 `expected` に合わせる。
///
/// 要素数が一致していればそのまま借用で返し、 不一致なら `adapter` で変換する。
/// 変換できない (= `None`) 場合は size mismatch エラーを返す。 optimiser state
/// (momentum / velocity / slow など) のロードで weights.bin と同じ変換を適用するために使う。
pub fn adapt_to_size<'a, D: Device>(
    adapter: &dyn CheckpointShapeAdapter,
    id: &str,
    values: &'a [f32],
    expected: usize,
) -> Result<Cow<'a, [f32]>, OperationError<D::DeviceError>> {
    if values.len() == expected {
        return Ok(Cow::Borrowed(values));
    }
    match adapter.adapt(id, values, expected) {
        Some(adapted) if adapted.len() == expected => Ok(Cow::Owned(adapted)),
        Some(adapted) => Err(OperationError::WeightLoadingError(id.to_string(), Some((adapted.len(), expected)))),
        None => Err(OperationError::WeightLoadingError(id.to_string(), Some((values.len(), expected)))),
    }
}

/// Write a set of labelled weights from a `HashMap` into a file.
pub fn write_weights_to_file<D: Device>(
    map: &[(impl AsRef<str>, &DenseMatrix<D>)],
    path: &str,
) -> Result<(), D::DeviceError> {
    use std::{fs::File, io::Write};

    let mut buf = Vec::new();

    for (id, weights) in map {
        let this_buf = weights.write_to_byte_buffer(id.as_ref()).unwrap();
        buf.extend_from_slice(&this_buf);
    }

    let mut file = File::create(path).unwrap();
    file.write_all(&buf).unwrap();

    Ok(())
}

/// Loads a set of labelled weights from a file into a `HashMap`.
pub fn load_weights_from_file(path: &str, old_format: bool) -> Vec<(String, Vec<f32>)> {
    use std::{fs::File, io::Read};

    let mut buf = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buf).unwrap();

    let mut offset = 0;

    let mut res = Vec::new();

    while offset < buf.len() {
        let (buffer, id, bytes_read) = read_from_byte_buffer(&buf[offset..], old_format);
        res.push((id, buffer));
        offset += bytes_read;
    }

    res
}
