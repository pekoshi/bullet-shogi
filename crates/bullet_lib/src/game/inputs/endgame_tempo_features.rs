//! Endgame Tempo (EGT) scalar head 用の結合特徴量。
//!
//! Floodgate 終盤で観測した「安全な早逃げを逃す」「大駒を安易に切って、
//! 相手の持駒打ち込みを許す」を一つの局面 class にする。
//!
//! class = king urgency 4 × drop pressure 4 × major state 4 = 64。
//! LayerStack bucket 8 と組み合わせ、各 perspective で常に 1 feature を発火する。

use crate::shogi::packed_sfen::ShogiBoard;
use rshogi_core::nnue::activity_risk::compute_king_pressure;
use rshogi_core::nnue::king_mobility::king_mobility_safe_count;
use rshogi_core::position::Position;
use rshogi_core::types::{Color, PieceType};

pub const ENDGAME_TEMPO_NUM_BUCKETS: usize = 8;
pub const ENDGAME_TEMPO_NUM_CLASS: usize = 64;
pub const ENDGAME_TEMPO_DIM_PER_PERSP: usize = ENDGAME_TEMPO_NUM_BUCKETS * ENDGAME_TEMPO_NUM_CLASS;
pub const ENDGAME_TEMPO_TOTAL_COUNT: u32 = (2 * ENDGAME_TEMPO_DIM_PER_PERSP) as u32;
pub const ENDGAME_TEMPO_MAX_ACTIVE_PER_PERSP: usize = 1;
pub const ENDGAME_TEMPO_SUB_MAGIC: &[u8; 8] = b"P2EGTMP\0";
pub const ENDGAME_TEMPO_SECTION_VERSION: u32 = 1;
pub const ENDGAME_TEMPO_QA: f64 = 127.0;
pub const ENDGAME_TEMPO_ARCH_MARKER: &str = "EGTMP=1";
pub const ENDGAME_TEMPO_SPARSE_NAME_BLACK: &str = "endgame_tempo_black";
pub const ENDGAME_TEMPO_SPARSE_NAME_WHITE: &str = "endgame_tempo_white";

#[inline]
fn king_urgency(pressure: usize, safe_escapes: u32) -> usize {
    if safe_escapes == 0 {
        0
    } else if safe_escapes <= 2 {
        1
    } else if pressure < 2 {
        2
    } else {
        3
    }
}

#[inline]
fn hand_threat_4(pos: &Position, side: Color) -> usize {
    let hand = pos.hand(side.opponent());
    let weighted = 2
        * (hand.count(PieceType::Rook) as usize
            + hand.count(PieceType::Bishop) as usize
            + hand.count(PieceType::Gold) as usize
            + hand.count(PieceType::Silver) as usize)
        + hand.count(PieceType::Lance) as usize
        + hand.count(PieceType::Knight) as usize
        + hand.count(PieceType::Pawn) as usize;
    match weighted {
        0 => 0,
        1 | 2 => 1,
        3 | 4 => 2,
        _ => 3,
    }
}

#[inline]
fn is_major(pt: PieceType) -> bool {
    matches!(pt, PieceType::Bishop | PieceType::Rook | PieceType::Horse | PieceType::Dragon)
}

/// 0=盤上大駒なし、1=無傷、2=攻撃され守りあり、3=攻撃され守りなし。
fn major_state(pos: &Position, side: Color) -> usize {
    let enemy = side.opponent();
    let mut state = 0;
    for sq in pos.pieces_c(side).iter() {
        if !is_major(pos.piece_on(sq).piece_type()) {
            continue;
        }
        state = state.max(1);
        let attackers = pos.attackers_to(sq);
        if (attackers & pos.pieces_c(enemy)).is_not_empty() {
            state = state.max(if (attackers & pos.pieces_c(side)).is_not_empty() { 2 } else { 3 });
        }
    }
    state
}

/// perspective 側の EGT class (0..63) を返す。
pub fn compute_class_for_persp(board: &ShogiBoard, persp_is_black: bool) -> usize {
    let mut pos = Position::new();
    pos.set_sfen(&board.to_sfen()).expect("valid sfen");
    let side = if persp_is_black { Color::Black } else { Color::White };
    let urgency = king_urgency(compute_king_pressure(&pos, side), king_mobility_safe_count(&pos, side));
    let drop = hand_threat_4(&pos, side);
    let major = major_state(&pos, side);
    let class = (urgency * 4 + drop) * 4 + major;
    debug_assert!(class < ENDGAME_TEMPO_NUM_CLASS);
    class
}

#[inline]
pub const fn local_idx(bucket: usize, class: usize) -> usize {
    bucket * ENDGAME_TEMPO_NUM_CLASS + class
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_small_and_bounded() {
        assert_eq!(ENDGAME_TEMPO_NUM_CLASS, 64);
        assert_eq!(ENDGAME_TEMPO_DIM_PER_PERSP, 512);
        assert_eq!(ENDGAME_TEMPO_TOTAL_COUNT, 1024);
        assert_eq!(local_idx(7, 63) + 1, ENDGAME_TEMPO_DIM_PER_PERSP);
    }

    #[test]
    fn component_mapping_covers_all_classes() {
        let mut seen = [false; ENDGAME_TEMPO_NUM_CLASS];
        for urgency in 0..4 {
            for drop in 0..4 {
                for major in 0..4 {
                    seen[(urgency * 4 + drop) * 4 + major] = true;
                }
            }
        }
        assert!(seen.into_iter().all(|x| x));
    }
}
