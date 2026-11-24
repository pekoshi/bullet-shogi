//! PackedSfenWithVisits — PSV-40 + top-K visit distribution 拡張 record。
//!
//! Phase 2 soft policy training 用。 dlshogi hcpe3 を aggregate して得た per-position
//! visit distribution を Bullet が固定長 record として読めるようにする。
//!
//! ## Record layout (168 bytes)
//!
//! ```text
//! offset  size  field
//! 0       32    packed_sfen   (YaneuraOu Huffman 256bit)
//! 32      2     score          (i16、 placeholder 0 also acceptable)
//! 34      2     move16         (u16、 top-1 by visits、 PSV best_move fallback)
//! 36      2     game_ply       (u16)
//! 38      1     result         (i8)
//! 39      1     pad            (0)
//! ─── 上記 40 byte は PackedSfenValue と完全一致 ───
//! 40      2     visit_count    (u16、 有効な MoveVisits 数 ∈ 0..32)
//! 42      2     pad            (alignment)
//! 44      4×32  visits         (u16 move + u16 visit × 32 = 128 bytes)
//!                                未使用 slot は move=0 visit=0
//! ─── 合計 172 bytes ───
//! ```
//!
//! 実装注意:
//! - `#[repr(C, packed)]` で alignment 0、 mod 4 bytes 厳守。
//! - `CanBeDirectlySequentiallyLoaded` 実装で DirectSequentialDataLoader に対応。
//! - 既存 PackedSfenValue とは layout 互換 (先頭 40 byte 同一) で、 sparse encoder は
//!   この 40 byte 部分のみ参照する。

use super::packed_sfen::{PackedSfen, PackedSfenValue};

/// 単一 MoveVisits entry。
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveVisit {
    pub move16: u16,
    pub visit: u16,
}

/// top-K (= [`MAX_VISITS_PER_POS`]) visit distribution。
pub const MAX_VISITS_PER_POS: usize = 32;

/// PackedSfen + visit distribution 拡張 record (172 bytes 固定)。
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct PackedSfenWithVisits {
    /// PSV 部分 (40 bytes、 既存 [`PackedSfenValue`] と互換 layout)。
    pub psv: PackedSfenValue,
    /// 有効な MoveVisit slot 数 (0..=MAX_VISITS_PER_POS)。
    pub visit_count: u16,
    /// alignment padding。
    pub _pad: u16,
    /// top-K MoveVisits、 visit count 降順。 未使用 slot は (0, 0)。
    pub visits: [MoveVisit; MAX_VISITS_PER_POS],
}

impl Default for PackedSfenWithVisits {
    fn default() -> Self {
        Self {
            psv: PackedSfenValue::default(),
            visit_count: 0,
            _pad: 0,
            visits: [MoveVisit::default(); MAX_VISITS_PER_POS],
        }
    }
}

// SAFETY: #[repr(C, packed)] でパディング無し、 全 field は plain old data (u8/i8/u16/i16)。
// fixed-size 172 byte record として直接 sequential load 可能。
unsafe impl crate::value::loader::CanBeDirectlySequentiallyLoaded for PackedSfenWithVisits {}

// SAFETY: PackedSfen 内部の data: [u8; 32] は POD、 全 field plain。
unsafe impl Send for PackedSfenWithVisits {}
unsafe impl Sync for PackedSfenWithVisits {}

impl PackedSfenWithVisits {
    /// PSV 部分 (40 byte) への参照。
    pub fn psv(&self) -> &PackedSfenValue {
        &self.psv
    }

    /// PSV 内 packed_sfen への参照。
    pub fn sfen(&self) -> &PackedSfen {
        self.psv.sfen()
    }

    /// 評価値 (cp、 stm 視点)。 PSV と同じ。
    pub fn score(&self) -> i16 {
        self.psv.score()
    }

    /// top-1 move16 (PSV の best_move、 visit 降順 sort の 1 位)。
    pub fn move16(&self) -> u16 {
        self.psv.move16()
    }

    /// 手数 (PSV の game_ply)。
    pub fn game_ply(&self) -> u16 {
        self.psv.game_ply()
    }

    /// 対局結果 (-1 / 0 / +1、 PSV と同じ)。
    pub fn game_result(&self) -> i8 {
        self.psv.game_result()
    }

    /// 有効な visit slot を返す iterator。
    pub fn iter_visits(&self) -> impl Iterator<Item = MoveVisit> + '_ {
        let n = (self.visit_count as usize).min(MAX_VISITS_PER_POS);
        self.visits[..n].iter().copied()
    }

    /// visit 総和 (確率正規化用)。
    pub fn total_visits(&self) -> u32 {
        self.iter_visits().map(|m| m.visit as u32).sum()
    }
}

// LoadableDataType を impl (score / result extraction、 dataloader が値教師に使う)。
impl crate::value::loader::LoadableDataType for PackedSfenWithVisits {
    fn score(&self) -> i16 {
        self.psv.score()
    }

    fn result(&self) -> crate::value::loader::GameResult {
        // PSV の game_result (-1 / 0 / +1) → GameResult enum (Loss=0 / Draw=1 / Win=2)
        let r = self.psv.game_result();
        match r {
            -1 => crate::value::loader::GameResult::Loss,
            0 => crate::value::loader::GameResult::Draw,
            _ => crate::value::loader::GameResult::Win,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_is_172() {
        // alignment 込みで 172 bytes 固定 (DirectSequentialDataLoader 要件)。
        assert_eq!(std::mem::size_of::<PackedSfenWithVisits>(), 172);
    }

    #[test]
    fn default_is_empty() {
        let p = PackedSfenWithVisits::default();
        // packed 構造のため field 直接 borrow 不可、 値コピーで比較。
        let visit_count = p.visit_count;
        assert_eq!(visit_count, 0);
        assert!(p.iter_visits().next().is_none());
    }
}
