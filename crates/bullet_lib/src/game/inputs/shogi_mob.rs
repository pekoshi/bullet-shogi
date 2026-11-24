//! B6 可動域 (mobility) — 反対称スカラー 1 本。 2026-09-29。
//!
//! 手設計の特徴量 26 ブロックが全滅したあと、 screen の器を作り直して
//! (594 局 → 33,096 局、 基準式に非線形 17 列) 裁き直したところ、 唯一生き残った量。
//!
//!   勝敗 ΔR² +0.038 (床 ±0.001 の 40 倍)、 較正 1 ΔR² = +19〜39pt → **+0.7〜1.5pt (5〜10 Elo)**
//!   関門: 集合分割で再現 / 係数の符号安定 / 手数帯で全帯正 / **信号の 99% が反対称側** /
//!         戦型 (net が既に持つ) を足しても減らない
//!
//! # 定義 (正典 = `C:/project/p2shogi/runs/rescore/fs_feats.py:28-61, 158`)
//!
//! ```text
//! mob_c = Σ(色 c の全駒の疑似合法な移動先の数)
//!   - 移動先が **自駒の升なら数えない**、 空升と敵駒の升は数える
//!   - ray 駒は遮蔽まで (敵駒の升を数えてそこで止まる)
//!   - **玉も含む**
//!   - 🚨王手放置等の合法性は見ていない (疑似合法)
//! B6 = mob(手番側) − mob(非手番側)
//! ```
//!
//! # 🚨3 実装が一致しないと黙って壊れる
//!
//! ★19「エンジン改造は eval bit-exact だけでは不十分」の一族。 Python (正典) /
//! C++ エンジン (`p2nnue_mob.h`、 `p2mob` コマンド) / ここ (学習器) の 3 つが
//! **ゼロ許容で一致**すること。 検証は `examples/shogi_mob_canonical.rs`。
//!
//! # 🚨`for_each_attack` は玉を列挙しない
//!
//! `shogi_halfka_hm_threat.rs` の `for_each_attack` は末尾が
//! `PieceType::King | PieceType::None => {}` で、 **玉の利きを出さない**
//! (threat 表に玉を含めない設計のため)。 そのまま使うと片側が 5〜8 だけ黙って
//! 小さくなるので、 ここで 8 近傍を手で足す。

use super::shogi_halfka_hm_threat::{Occupied, for_each_attack};
use crate::shogi::types::Color;
use crate::shogi::{
    ShogiBoard,
    types::{PieceType, Square},
};

/// 色 `c` の可動域 (疑似合法な移動先の総数)。
///
/// `occ` は `Occupied::from_board(board)` で作ったもの。 同じ盤面で 2 色分呼ぶので
/// 呼び出し側で 1 度だけ作って使い回す。
pub(super) fn mob_scalar(board: &ShogiBoard, occ: &Occupied, c: Color) -> i32 {
    let mut m = 0i32;
    for idx in 0..81usize {
        let p = board.board[idx];
        // 🚨`Piece::NONE` は color が Black なので、 piece_type を先に見ること。
        if p.piece_type == PieceType::None || p.color != c {
            continue;
        }
        let from = Square::from_index(idx);
        if p.piece_type == PieceType::King {
            // 🚨for_each_attack は玉を出さないので 8 近傍を手で足す。
            let f = from.file() as i8;
            let r = from.rank() as i8;
            for df in -1i8..=1 {
                for dr in -1i8..=1 {
                    if df == 0 && dr == 0 {
                        continue;
                    }
                    let (nf, nr) = (f + df, r + dr);
                    if (0..9).contains(&nf) && (0..9).contains(&nr) {
                        let t = Square::new(nf as u8, nr as u8);
                        let q = board.board[t.index()];
                        if q.piece_type == PieceType::None || q.color != c {
                            m += 1;
                        }
                    }
                }
            }
        } else {
            for_each_attack(p.piece_type, c, from, occ, |t| {
                let q = board.board[t.index()];
                if q.piece_type == PieceType::None || q.color != c {
                    m += 1;
                }
            });
        }
    }
    m
}

/// B6 本体: 手番側 − 非手番側。 **反対称**なので PSQT 頭 1 列で表現できる。
pub fn mob_diff(board: &ShogiBoard) -> i32 {
    let occ = Occupied::from_board(board);
    let us = board.side_to_move;
    let them = if us == Color::Black { Color::White } else { Color::Black };
    mob_scalar(board, &occ, us) - mob_scalar(board, &occ, them)
}

/// 先手 / 後手 / 手番側 / 非手番側 を一度に返す (照合用)。
pub fn mob_all(board: &ShogiBoard) -> (i32, i32, i32, i32) {
    let occ = Occupied::from_board(board);
    let mb = mob_scalar(board, &occ, Color::Black);
    let mw = mob_scalar(board, &occ, Color::White);
    let (mu, mt) = if board.side_to_move == Color::Black { (mb, mw) } else { (mw, mb) };
    (mb, mw, mu, mt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::Piece;

    /// 裸玉 `9/9/9/9/4k4/9/9/9/4K4 b - 1` で 先手 5 / 後手 8。
    ///
    /// 🚨これは **玉の 8 近傍を手で足した分岐だけ**を見るテスト。 `for_each_attack` は
    /// 玉を列挙しないので、 その足し込みを落とすとここが 0 / 0 になる。
    /// 正典 `mob_reference.tsv` の該当行と同じ値。
    #[test]
    fn bare_kings_5_and_8() {
        let mut board = ShogiBoard {
            board: [Piece::NONE; 81],
            black_hand: Default::default(),
            white_hand: Default::default(),
            side_to_move: Color::Black,
            black_king_sq: Square::new(4, 8),
            white_king_sq: Square::new(4, 4),
            score: 0,
            result: 0,
            ply: 1,
        };
        board.board[Square::new(4, 8).index()] = Piece::new(Color::Black, PieceType::King);
        board.board[Square::new(4, 4).index()] = Piece::new(Color::White, PieceType::King);

        let (mb, mw, mu, mt) = mob_all(&board);
        assert_eq!(mb, 5, "隅寄り (5i) の玉は 5 近傍");
        assert_eq!(mw, 8, "中央 (5e) の玉は 8 近傍");
        assert_eq!((mu, mt), (5, 8), "手番が先手なので us=先手");
        assert_eq!(mob_diff(&board), -3);
    }
}
