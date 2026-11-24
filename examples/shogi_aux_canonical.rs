//! AUX (KZ + HANG + STATKD) canonical vector 出力 / bit-exact 照合ドライバ (2026-07-22)。
//!
//! `ShogiHalfKaHmAux` の aux active feature (連結空間 local index = 絶対 index -
//! HALFKA_HM_DIMENSIONS) を、 代表局面 (平手 + 中盤 2 局面) について STM / NSTM 別に
//! ソートして出力する。 契約 `scratchpad/aux_xcheck.py` (cand_features.py/2) の出力と
//! **完全一致** することで C++ (P2NNUE) 移植の bit-exact 基準を与える。
//!
//! 実行: `cargo run --release -p bullet_lib --no-default-features --features cpu --example shogi_aux_canonical`

use bullet_lib::game::inputs::{HALFKA_HM_DIMENSIONS, emit_aux_active_index};
use bullet_lib::shogi::ShogiBoard;
use bullet_lib::shogi::types::{Color, Piece, PieceType, Square};

/// 盤面のみの SFEN パーサ (手駒は aux に非関与のため無視)。
/// SFEN のマス並びは 9 段 × 各段は 9筋→1筋。 file=8-col, rank=row。
fn parse_sfen_board(sfen: &str) -> ShogiBoard {
    let mut board = ShogiBoard::default();
    let mut it = sfen.split_whitespace();
    let placement = it.next().expect("sfen placement");
    let stm = it.next().unwrap_or("b");
    board.side_to_move = if stm == "w" { Color::White } else { Color::Black };

    for (rank, row) in placement.split('/').enumerate() {
        let mut file: i32 = 8;
        let mut promo = false;
        for ch in row.chars() {
            match ch {
                '+' => {
                    promo = true;
                }
                '1'..='9' => {
                    file -= ch.to_digit(10).unwrap() as i32;
                }
                _ => {
                    let color = if ch.is_ascii_uppercase() { Color::Black } else { Color::White };
                    let base = match ch.to_ascii_uppercase() {
                        'P' => PieceType::Pawn,
                        'L' => PieceType::Lance,
                        'N' => PieceType::Knight,
                        'S' => PieceType::Silver,
                        'B' => PieceType::Bishop,
                        'R' => PieceType::Rook,
                        'G' => PieceType::Gold,
                        'K' => PieceType::King,
                        other => panic!("bad sfen piece {other}"),
                    };
                    let pt = if promo { base.promote() } else { base };
                    promo = false;
                    let sq = Square::new(file as u8, rank as u8);
                    board.board[sq.index()] = Piece::new(color, pt);
                    if pt == PieceType::King {
                        match color {
                            Color::Black => board.black_king_sq = sq,
                            Color::White => board.white_king_sq = sq,
                        }
                    }
                    file -= 1;
                }
            }
        }
    }
    board
}

fn aux_locals(board: &ShogiBoard, persp: Color) -> Vec<usize> {
    let mut v = Vec::new();
    emit_aux_active_index(board, persp, |idx| v.push(idx - HALFKA_HM_DIMENSIONS));
    v.sort_unstable();
    v
}

fn main() {
    let positions: [(&str, &str); 3] = [
        ("startpos", "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"),
        ("mid1", "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1"),
        ("mid2", "ln1gk2nl/1r4gs1/p1ppppbpp/1p4p2/7P1/2P6/PP1PPPP1P/1SG4R1/LN1GK1SNL b Bp 1"),
    ];

    for (name, sfen) in positions {
        let board = parse_sfen_board(sfen);
        let stm = board.side_to_move;
        let nstm = stm.opponent();
        let istm = aux_locals(&board, stm);
        let instm = aux_locals(&board, nstm);
        println!("### {name}  stm={}", if stm == Color::Black { "B" } else { "W" });
        println!("STM  ({}): {:?}", istm.len(), istm);
        println!("NSTM ({}): {:?}", instm.len(), instm);
    }
}
