//! 将棋固有のモジュール
//!
//! bullet を将棋 NNUE 学習に対応させるための型・関数定義。
//!
//! - `types`: 基本型 (Color, PieceType, Square, Piece)
//! - `packed_sfen`: PackedSfen/PackedSfenValue デコーダ
//! - `bona_piece`: BonaPiece 定義

pub mod bona_piece;
pub mod packed_sfen;
pub mod packed_sfen_with_visits;
pub mod types;

pub use bona_piece::BonaPiece;
pub use packed_sfen::{BitStream, PackedSfen, PackedSfenValue, ShogiBoard};
pub use packed_sfen_with_visits::{MAX_VISITS_PER_POS, MoveVisit, PackedSfenWithVisits};
pub use types::{Color, Hand, Piece, PieceType, Square};
