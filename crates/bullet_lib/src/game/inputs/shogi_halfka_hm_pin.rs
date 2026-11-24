//! Pin-aware NNUE 補助入力 (rshogi-core `nnue::pin_features` と数値一致)。
//!
//! 盤上の駒の **pin/skewer 状態** を per-perspective dense 補助入力として
//! 列挙する。 LMAS 廃止後も残る単独の革新 bit 候補。
//!
//! # 仕様 (rshogi-core SoT 準拠)
//!
//! - dim per perspective: `81 (sq) × 14 (piece_type) × 2 (state) = 2,268`
//! - state: 0=Pinned (= 絶対 pin)、 1=Skewer (= **standard X-ray skewer**、 玉無関係)
//! - 正規化: Black=identity、 White=`sq -> 80 - sq` (Half-Mirror なし)
//! - Pinned: 自軍駒のうち自玉に対する絶対 pin の駒
//! - Skewer: 敵 slider の ray 上で sq の手前に自軍 non-king 駒 (= friend1) が居り、
//!   friend1 が動くと sq が ray attack を受ける構図の **奥の駒** (= perspective 側、
//!   king を含む = 5九玉が奥に居るケースも Skewer 成立)
//!
//! # bullet 側 contract
//!
//! 本 module の `fill_pin_features_*` は、 同 SFEN で p2shogi の
//! `crate::nnue::pin_features::feature_pin_dense(pos, perspective)` と
//! **bit-exact 同値** を返す。 変更時は両側で必ず Golden Forward 確認すること。
//!
//! # bullet ↔ rshogi-core 対応
//!
//! | bullet                                  | rshogi-core                                       |
//! |-----------------------------------------|---------------------------------------------------|
//! | `PIN_DIMENSIONS_PER_PERSPECTIVE = 2268` | `PIN_DIMENSIONS_PER_PERSPECTIVE`                  |
//! | `PIN_PIECE_TYPES = 14`                  | `PIN_PIECE_TYPES`                                 |
//! | `PIN_STATES = 2`                        | `PIN_STATES`                                      |
//! | `PIN_NUM_BUCKETS = 8`                   | `NUM_PIN_BUCKETS`                                 |
//! | `pin_feature_index(sq,pt,state)`        | `pin_feature_index`                               |
//! | `fill_pin_features_black/white`         | `feature_pin_dense(pos, perspective)`             |
//! | `collect_pinned_pieces`                 | `Position::pinned_pieces`                         |
//! | `collect_skewered_pieces`               | `compute_skewered`                                |

use crate::shogi::{
    PackedSfenValue, ShogiBoard,
    types::{Color, Piece, PieceType, Square},
};

// =============================================================================
// 定数 (rshogi-core::nnue::pin_features と一致)
// =============================================================================

/// Pin/Skewer state 数。 0=Pinned, 1=Skewer。
pub const PIN_STATES: usize = 2;

/// pin feature が対象とする piece_type 数 (King 含む)。
pub const PIN_PIECE_TYPES: usize = 14;

/// 盤面マス数。
pub const PIN_SQUARES: usize = 81;

/// 1 perspective あたりの pin feature 次元数 = 81 × 14 × 2 = 2,268。
pub const PIN_DIMENSIONS_PER_PERSPECTIVE: usize = PIN_SQUARES * PIN_PIECE_TYPES * PIN_STATES;

/// LayerStack bucket 数 (rshogi-core `NUM_LAYER_STACK_BUCKETS` と一致)。
pub const PIN_NUM_BUCKETS: usize = 8;

/// 量子化スケール (LMAS と同じ QA=127)。
pub const PIN_QA: i16 = 127;

/// Pin-aware 用 dense 入力名 (黒視点)。
pub const PIN_AUX_NAME_BLACK: &str = "pin_features_black";

/// Pin-aware 用 dense 入力名 (白視点)。
pub const PIN_AUX_NAME_WHITE: &str = "pin_features_white";

/// arch_str に埋め込む marker 文字列 (rshogi-core loader が `"PIN="` を検出)。
pub const PIN_ARCH_MARKER: &str = "PIN=2268,";

/// quantised.bin section の sub_magic (8 byte)。
pub const PIN_SUB_MAGIC: &[u8; 8] = b"P2PIN\0\0\0";

/// quantised.bin section の format version。
pub const PIN_SECTION_VERSION: u32 = 1;

// =============================================================================
// piece_type indexing
// =============================================================================

/// `PieceType` enum (1..=14) を pin feature の 0-indexed (0..=13) に変換。
///
/// rshogi-core `piece_type_to_pin_idx` と完全同 mapping:
/// Pawn(1)→0, Lance(2)→1, ..., King(8)→7, ProPawn(9)→8, ..., Dragon(14)→13。
#[inline]
const fn piece_type_to_pin_idx(pt: PieceType) -> Option<usize> {
    match pt {
        PieceType::Pawn => Some(0),
        PieceType::Lance => Some(1),
        PieceType::Knight => Some(2),
        PieceType::Silver => Some(3),
        PieceType::Bishop => Some(4),
        PieceType::Rook => Some(5),
        PieceType::Gold => Some(6),
        PieceType::King => Some(7),
        PieceType::ProPawn => Some(8),
        PieceType::ProLance => Some(9),
        PieceType::ProKnight => Some(10),
        PieceType::ProSilver => Some(11),
        PieceType::Horse => Some(12),
        PieceType::Dragon => Some(13),
        PieceType::None => None,
    }
}

// =============================================================================
// 正規化
// =============================================================================

/// perspective ベースの square index 正規化。
///
/// Black: identity (`sq.index()`)
/// White: `80 - sq.index()` (color swap、 Half-Mirror なし)
#[inline]
const fn oriented_sq_idx(sq_index: usize, perspective: Color) -> usize {
    match perspective {
        Color::Black => sq_index,
        Color::White => 80 - sq_index,
    }
}

/// pin feature の flat index。 rshogi-core `pin_feature_index` と同式。
///
/// `flat = sq_idx × 28 + pt_idx × 2 + state_idx`。
#[inline]
pub const fn pin_feature_index(sq_idx: usize, pt_idx: usize, state_idx: usize) -> usize {
    sq_idx * (PIN_PIECE_TYPES * PIN_STATES) + pt_idx * PIN_STATES + state_idx
}

// =============================================================================
// pin 計算 (board → pinned set)
// =============================================================================

/// `perspective` 視点の自玉に対する absolute pin 駒を列挙する。
///
/// 戻り値 `(buf, len)` で `buf[..len]` が pinned 駒の `(sq, pt)` を含む。
/// 同一駒に pinner が複数あるケースは shogi 規則上稀 (重 pin) だが、 1 駒は
/// 1 回しか push しないように内部で重複除去する (= bitboard 集合と同じ振舞)。
///
/// rshogi-core `Position::pinned_pieces(perspective, perspective_king_sq)` の
/// 数値再現:
/// 1. enemy slider 候補を列挙 (Lance / Bishop / Horse / Rook / Dragon)。
/// 2. king から該当方向へ ray cast し、 between に enemy/友軍 1 駒のみが在り、
///    かつそれが perspective 側 (= 自軍) の駒のとき pin と判定。
/// 3. Lance は color-sensitive (前方のみ): perspective の前方に sniper 側、
///    つまり enemy lance の前方が king を向く配置のみ pin。
fn collect_pinned_pieces(board: &ShogiBoard, perspective: Color) -> ([(Square, PieceType); MAX_PIN_PIECES], usize) {
    let mut out: [(Square, PieceType); MAX_PIN_PIECES] = [(Square::NONE, PieceType::None); MAX_PIN_PIECES];
    let mut len: usize = 0;

    let king_sq = board.king_square(perspective);
    if !king_sq.is_valid() {
        return (out, len);
    }

    let enemy = perspective.opponent();
    let king_file = king_sq.file() as i8;
    let king_rank = king_sq.rank() as i8;

    // (df_step, dr_step) 8 方向 (king から見た外向き方向)。
    // direction kind 0 = orthogonal (file/rank 軸)、 1 = diagonal。
    const DIRECTIONS: [(i8, i8, u8); 8] = [
        (0, -1, 0),  // N (rank -1)
        (1, -1, 1),  // NE
        (1, 0, 0),   // E
        (1, 1, 1),   // SE
        (0, 1, 0),   // S
        (-1, 1, 1),  // SW
        (-1, 0, 0),  // W
        (-1, -1, 1), // NW
    ];

    // perspective から見て "lance forward" の dr 符号。
    // bullet/yo 流: Black の歩/香は rank-=1 (= 上に進む)、 White は rank+=1 (= 下に進む)。
    // Lance が king を pin する場合、 sniper lance は king の **enemy の前方** に居る:
    //   - perspective=Black → king は下側、 enemy lance は上から下向きに射る (= dr < 0、
    //     king の rank より小さい file=king_file の sq から rank- 方向)。
    //     i.e. sniper lance は king の N 方向 (dr=-1)。
    //   - perspective=White → 逆方向、 sniper lance は king の S 方向 (dr=+1)。
    let lance_pin_dr: i8 = match perspective {
        Color::Black => -1, // sniper enemy lance shoots from north (king-side rank-=)、 ray dir N。
        Color::White => 1,
    };

    for (df, dr, dir_kind) in DIRECTIONS.iter().copied() {
        // king から (df, dr) 方向に ray cast。 最初に見つかった駒が friend なら
        // 「pin 候補 (between)」、 その奥で最初に見つかる駒が enemy sniper の
        // 駒種・色・direction と整合すれば pin 成立。
        let mut blocker_sq: Option<Square> = None;
        let mut blocker_is_friend = false;
        let mut found_pinner = false;
        let mut ray_file = king_file + df;
        let mut ray_rank = king_rank + dr;

        while (0..9).contains(&ray_file) && (0..9).contains(&ray_rank) {
            let sq = Square::new(ray_file as u8, ray_rank as u8);
            let piece: Piece = board.piece_on(sq);
            if !piece.is_none() {
                if blocker_sq.is_none() {
                    // 最初の駒 = blocker 候補。 自軍駒なら pin 候補、 敵駒なら
                    // 即 check / discovered で pin 不成立 (ray 終了)。
                    blocker_sq = Some(sq);
                    blocker_is_friend = piece.color == perspective;
                    if !blocker_is_friend {
                        break;
                    }
                } else {
                    // 2 駒目 = sniper 候補。 driver 駒の駒種・色・方向と整合するか確認。
                    if blocker_is_friend && is_sniper_on_direction(piece, enemy, dir_kind, df, dr, lance_pin_dr) {
                        found_pinner = true;
                    }
                    break;
                }
            }
            ray_file += df;
            ray_rank += dr;
        }

        if found_pinner {
            if let Some(b_sq) = blocker_sq {
                let piece: Piece = board.piece_on(b_sq);
                if !piece.is_none() && piece.color == perspective {
                    // 重複 push 防止 (同一 sq が複数 ray 経由で見つかるケースは
                    // shogi 通常局面では発生しないが、 双方向 pin 等の極端なケースに備える)。
                    let already = out[..len].iter().any(|&(s, _)| s.index() == b_sq.index());
                    if !already && len < MAX_PIN_PIECES {
                        out[len] = (b_sq, piece.piece_type);
                        len += 1;
                    }
                }
            }
        }
    }

    (out, len)
}

/// `piece` が `enemy` 側の sniper で、 king から見て方向 `(df, dr)` に対し
/// pin を成立させる駒種であるかを判定する。
///
/// - direction kind 0 (file/rank 軸): Rook / Dragon、 および Lance (色一致 + 方向一致時)。
/// - direction kind 1 (diagonal):     Bishop / Horse。
#[inline]
fn is_sniper_on_direction(piece: Piece, enemy: Color, dir_kind: u8, df: i8, dr: i8, lance_pin_dr: i8) -> bool {
    if piece.is_none() || piece.color != enemy {
        return false;
    }
    match piece.piece_type {
        PieceType::Rook | PieceType::Dragon => dir_kind == 0,
        PieceType::Bishop | PieceType::Horse => dir_kind == 1,
        PieceType::Lance => {
            // Lance は file 軸 (df==0) かつ enemy の前方向 (= king 側 dr == lance_pin_dr) のみ。
            dir_kind == 0 && df == 0 && dr == lance_pin_dr
        }
        _ => false,
    }
}

/// 同時に pin され得る駒数の理論上限 (8 方向 × 1 駒 = 8、 安全側 16)。
const MAX_PIN_PIECES: usize = 16;

// =============================================================================
// Skewer 計算 (standard X-ray skewer、 玉と無関係)
// =============================================================================

/// 同時に skewer され得る駒数の理論上限 (8 方向 × 1 駒 = 8、 安全側 16)。
const MAX_SKEWER_PIECES: usize = 16;

/// `perspective` 視点で **standard X-ray skewer (= 玉無関係)** された駒を列挙する。
///
/// 戻り値 `(buf, len)` で `buf[..len]` が skewered 駒の `(sq, pt)` を含む。
///
/// # 定義
///
/// sq の駒が Skewer 状態 iff:
/// 1. ある敵 slider (= Lance / Bishop / Rook / Horse / Dragon) が存在し、
/// 2. その slider の attack ray 上で、 slider から見て sq よりも手前に
///    自軍の **non-king** 駒 (= friend1) が **ちょうど 1 駒** 存在し、
/// 3. friend1 のすぐ奥 (= ray の継続) の最初の駒が sq の駒であり、
///    かつ perspective 側 (= 自軍、 king を含む) である。
///
/// Pin との重複避け: friend1 = king なら Pin 側で計上、 Skewer 側では除外。
/// sq の駒は king を含む (= ユーザー仕様 5「sq の駒 (= skewered) を Skewer 状態に」
/// で「king を除外」 とは書かれていないので含めて扱う)。 これにより 5九玉が
/// friend1=5五金 の X-ray 奥にある場合に Pin=5五金 + Skewer=5九玉 となり、
/// rshogi-core の同 SFEN test と数値一致する。
///
/// rshogi-core `compute_skewered(pos, perspective)` と **bit-exact 同値**:
/// 1. 81 マス走査で敵 slider 駒を列挙
/// 2. 各 sniper について該当方向集合 (= Rook/Dragon: orthogonal、 Bishop/Horse: diagonal、
///    Lance: 前方 1 方向) を walk
/// 3. ray 上の最初の駒 = friend1 候補 (= perspective 側、 non-king)
/// 4. friend1 のすぐ奥 = skewered 候補 (= perspective 側、 king 含む) → 重複除去で push
fn collect_skewered_pieces(
    board: &ShogiBoard,
    perspective: Color,
) -> ([(Square, PieceType); MAX_SKEWER_PIECES], usize) {
    let mut out: [(Square, PieceType); MAX_SKEWER_PIECES] = [(Square::NONE, PieceType::None); MAX_SKEWER_PIECES];
    let mut len: usize = 0;

    let enemy = perspective.opponent();

    // enemy lance の前方を ray dr で表現:
    //   enemy=Black: lance は rank- 方向に進むので dr=-1
    //   enemy=White: lance は rank+ 方向に進むので dr=+1
    let enemy_lance_ray_dr: i8 = match enemy {
        Color::Black => -1,
        Color::White => 1,
    };

    // (df, dr, dir_kind) 8 方向 (sniper から外向き)。 dir_kind=0:orthogonal、 1:diagonal。
    const DIRECTIONS: [(i8, i8, u8); 8] = [
        (0, -1, 0),  // N
        (1, -1, 1),  // NE
        (1, 0, 0),   // E
        (1, 1, 1),   // SE
        (0, 1, 0),   // S
        (-1, 1, 1),  // SW
        (-1, 0, 0),  // W
        (-1, -1, 1), // NW
    ];

    for sniper_idx in 0..(PIN_SQUARES as u8) {
        let sniper_sq = Square(sniper_idx);
        let sniper_piece = board.piece_on(sniper_sq);
        if sniper_piece.is_none() || sniper_piece.color != enemy {
            continue;
        }
        match sniper_piece.piece_type {
            PieceType::Lance | PieceType::Bishop | PieceType::Rook | PieceType::Horse | PieceType::Dragon => {}
            _ => continue,
        }

        let sniper_file = sniper_sq.file() as i8;
        let sniper_rank = sniper_sq.rank() as i8;

        for (df, dr, dir_kind) in DIRECTIONS.iter().copied() {
            // sniper が本方向に attack を出さない場合は skip。
            // (collect_pinned_pieces で使う is_sniper_on_direction は king 視点の lance_pin_dr
            // を使うが、 ここは sniper 視点で「enemy lance の前方 = ray dr」 と一致する場合のみ
            // attack を出す、 = enemy_lance_ray_dr と同値判定で OK。)
            if !is_sniper_on_direction(sniper_piece, enemy, dir_kind, df, dr, enemy_lance_ray_dr) {
                continue;
            }

            // sniper の前方を ray walk: 最初の駒 = friend1 候補。
            let mut file = sniper_file + df;
            let mut rank = sniper_rank + dr;
            let mut friend1_found = false;
            while (0..9).contains(&file) && (0..9).contains(&rank) {
                let sq = Square::new(file as u8, rank as u8);
                let piece = board.piece_on(sq);
                if !piece.is_none() {
                    if piece.color == perspective && piece.piece_type != PieceType::King {
                        friend1_found = true;
                    }
                    break;
                }
                file += df;
                rank += dr;
            }
            if !friend1_found {
                continue;
            }

            // friend1 のすぐ奥 (= ray 継続) で最初の駒 = skewered 候補。
            file += df;
            rank += dr;
            while (0..9).contains(&file) && (0..9).contains(&rank) {
                let sq = Square::new(file as u8, rank as u8);
                let piece = board.piece_on(sq);
                if !piece.is_none() {
                    // 奥の駒が perspective 側 (= 自軍、 king を含む) なら Skewer 成立。
                    // (= rshogi-core compute_skewered と数値一致、 Pin 側で friend1=king を
                    // 除外しているので奥 king は Skewer 独立で扱う。)
                    if piece.color == perspective {
                        // 重複 push 防止: 異なる sniper 経由で同 sq に到達する
                        // ケースに備える (通常局面では発生し得ないが安全側で線形検索)。
                        let already = out[..len].iter().any(|&(s, _)| s.index() == sq.index());
                        if !already && len < MAX_SKEWER_PIECES {
                            out[len] = (sq, piece.piece_type);
                            len += 1;
                        }
                    }
                    break;
                }
                file += df;
                rank += dr;
            }
        }
    }

    (out, len)
}

// =============================================================================
// dense fill (training-side parity for rshogi-core feature_pin_dense)
// =============================================================================

/// `perspective` 視点で pin/skewer 状態にある自駒を dense vector に展開する。
///
/// 出力 `out.len()` は `PIN_DIMENSIONS_PER_PERSPECTIVE` (= 2,268) と一致必須。
/// rshogi-core `feature_pin_dense(pos, perspective)` と **bit-exact 同値**。
///
/// - state 0 (Pinned): `collect_pinned_pieces` の自軍駒
/// - state 1 (Skewer): `collect_skewered_pieces` = standard X-ray skewer (= 玉無関係)
pub fn fill_pin_features_for_side(pos: &PackedSfenValue, side: Color, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        PIN_DIMENSIONS_PER_PERSPECTIVE,
        "pin feature 出力スロットは {} 次元",
        PIN_DIMENSIONS_PER_PERSPECTIVE,
    );
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let board = ShogiBoard::from_packed_sfen(pos);

    // --- Pinned (state 0) ---
    let (pinned, n_pin) = collect_pinned_pieces(&board, side);
    for &(sq, pt) in &pinned[..n_pin] {
        let Some(pt_idx) = piece_type_to_pin_idx(pt) else { continue };
        let sq_idx = oriented_sq_idx(sq.index(), side);
        out[pin_feature_index(sq_idx, pt_idx, 0)] = 1.0;
    }

    // --- Skewer (state 1) — standard X-ray skewer (= 玉無関係) ---
    let (skewered, n_sk) = collect_skewered_pieces(&board, side);
    for &(sq, pt) in &skewered[..n_sk] {
        let Some(pt_idx) = piece_type_to_pin_idx(pt) else { continue };
        let sq_idx = oriented_sq_idx(sq.index(), side);
        out[pin_feature_index(sq_idx, pt_idx, 1)] = 1.0;
    }
}

// =============================================================================
// pub helpers for example/training-side glue
// =============================================================================

/// `auxiliary_dense_dims` で追加すべき pin 関連 dense input 定義。
///
/// 戻り値 `[(name, dims); 2]`。 LMAS struct から呼び出され、 `pin_aware == true`
/// のときのみ 2 entry を `auxiliary_dense_dims` に append する想定。
pub fn pin_aux_entries() -> [(&'static str, usize); 2] {
    [(PIN_AUX_NAME_BLACK, PIN_DIMENSIONS_PER_PERSPECTIVE), (PIN_AUX_NAME_WHITE, PIN_DIMENSIONS_PER_PERSPECTIVE)]
}

/// dispatch helper for `fill_auxiliary_dense`。
///
/// `name == PIN_AUX_NAME_BLACK / WHITE` の場合に pin fill を実行し、 `true` を返す。
/// それ以外の name では何もせず `false` を返す (caller が他 path に dispatch する)。
pub fn try_fill_pin_aux(name: &str, pos: &PackedSfenValue, out: &mut [f32]) -> bool {
    match name {
        PIN_AUX_NAME_BLACK => {
            fill_pin_features_for_side(pos, Color::Black, out);
            true
        }
        PIN_AUX_NAME_WHITE => {
            fill_pin_features_for_side(pos, Color::White, out);
            true
        }
        _ => false,
    }
}

// =============================================================================
// SparseInputType への薄い trait 注入は LMAS struct 側で行う (本 module は純粋関数)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shogi::types::{Color, PieceType};

    /// startpos では pinned は 0 駒のはず。
    #[test]
    fn pin_startpos_empty() {
        // PackedSfenValue::default() は完全 0、 from_packed_sfen で空盤面 (玉位置 0,0) になる。
        // startpos そのものをここでテストするには PSV encode が必要だが、
        // 任意の board に対して empty 結果を確認する目的では十分でない。
        // 代わりに startpos に近い board を直接組み立てて pin が無いことを確認する。
        let mut board = ShogiBoard::default();
        board.black_king_sq = Square::new(4, 8); // 5九
        board.white_king_sq = Square::new(4, 0); // 5一
        board.board[Square::new(4, 8).index()] = Piece::new(Color::Black, PieceType::King);
        board.board[Square::new(4, 0).index()] = Piece::new(Color::White, PieceType::King);

        for &persp in &[Color::Black, Color::White] {
            let (_pinned, n) = collect_pinned_pieces(&board, persp);
            assert_eq!(n, 0, "empty board should have no pin for {:?}", persp);
        }
    }

    /// 飛車 pin: 5筋 sniper 後手飛車 (5一) + 先手金 (5五) + 先手玉 (5九)。
    /// 黒視点で 1 駒 pin (= 5五金)、 白視点では pin 無し。
    #[test]
    fn pin_rook_pin_position() {
        let mut board = ShogiBoard::default();
        // file=4 (5筋)
        let king_b = Square::new(4, 8); // 5九 (Black king)
        let king_w = Square::new(0, 0); // 1一 (White king、 1筋に避難させて pin 無関与)
        let gold_b = Square::new(4, 4); // 5五 (Black Gold = blocker)
        let rook_w = Square::new(4, 0); // 5一 (White Rook = pinner)

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[gold_b.index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[rook_w.index()] = Piece::new(Color::White, PieceType::Rook);

        // Black 視点: 5五金 が pin
        let (pinned, n) = collect_pinned_pieces(&board, Color::Black);
        assert_eq!(n, 1, "expected 1 pinned piece for black");
        assert_eq!(pinned[0].0.index(), gold_b.index());
        assert_eq!(pinned[0].1, PieceType::Gold);

        // White 視点: pin 無し
        let (_pinned, n_w) = collect_pinned_pieces(&board, Color::White);
        assert_eq!(n_w, 0, "white should have no pin");
    }

    /// 香 pin: 先手玉 (5九) + 先手歩 (5五) + 後手香 (5一)。
    /// 後手香は前方 = rank+ 方向に射るので、 5一の香は rank=0、 dr=+1 で 5五歩を pin。
    /// 先手 perspective から見ると lance_pin_dr=-1、 ray 方向 (df=0, dr=-1) で sniper 探索:
    /// king 5九から N=(0,-1) ray、 5五 (歩) が between、 5一 (香) が sniper。
    /// White 香の前方 = rank+、 これは perspective=Black の lance_pin_dr=-1 と整合
    /// (= enemy lance の前方が king を向く配置を perspective 視点では dr=-1 方向)。
    #[test]
    fn pin_lance_pin_position() {
        let mut board = ShogiBoard::default();
        let king_b = Square::new(4, 8);
        let king_w = Square::new(0, 0);
        let pawn_b = Square::new(4, 4);
        let lance_w = Square::new(4, 0);

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[pawn_b.index()] = Piece::new(Color::Black, PieceType::Pawn);
        board.board[lance_w.index()] = Piece::new(Color::White, PieceType::Lance);

        let (pinned, n) = collect_pinned_pieces(&board, Color::Black);
        assert_eq!(n, 1, "expected 1 pinned piece (pawn) for black via lance");
        assert_eq!(pinned[0].1, PieceType::Pawn);
        assert_eq!(pinned[0].0.index(), pawn_b.index());
    }

    /// 角 pin: 先手玉 (5九 = file=4, rank=8) + 先手銀 (3七 = file=2, rank=6) +
    /// 後手角 (1五 = file=0, rank=4)。 NW 方向 (df=-1, dr=-1) で pin。
    #[test]
    fn pin_bishop_pin_position() {
        let mut board = ShogiBoard::default();
        let king_b = Square::new(4, 8);
        let king_w = Square::new(8, 0);
        let silver_b = Square::new(2, 6);
        let bishop_w = Square::new(0, 4);

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[silver_b.index()] = Piece::new(Color::Black, PieceType::Silver);
        board.board[bishop_w.index()] = Piece::new(Color::White, PieceType::Bishop);

        let (pinned, n) = collect_pinned_pieces(&board, Color::Black);
        assert_eq!(n, 1, "expected 1 pinned silver via bishop");
        assert_eq!(pinned[0].1, PieceType::Silver);
        assert_eq!(pinned[0].0.index(), silver_b.index());
    }

    /// pin_feature_index 範囲チェック (rshogi-core と同式)。
    #[test]
    fn pin_feature_index_range() {
        // 最大値
        let max_idx = pin_feature_index(PIN_SQUARES - 1, PIN_PIECE_TYPES - 1, PIN_STATES - 1);
        assert_eq!(max_idx, PIN_DIMENSIONS_PER_PERSPECTIVE - 1);
        // 最小値
        let min_idx = pin_feature_index(0, 0, 0);
        assert_eq!(min_idx, 0);
        // 期待値 2268
        assert_eq!(PIN_DIMENSIONS_PER_PERSPECTIVE, 81 * 14 * 2);
    }

    /// 新仕様 (= standard X-ray skewer) で Pinned と Skewer が **別マス** に立つ check。
    ///
    /// 5筋 sniper 5一 r + 自軍 5五 G (= friend1 non-king) + 自軍 5九 K (= 奥 king)。
    /// 新仕様: Pinned=5五金 (state 0)、 Skewer=5九玉 (state 1)、 合計 2 entry。
    #[test]
    fn fill_pin_features_pin_and_skewer_distinct() {
        let mut board = ShogiBoard::default();
        let king_b = Square::new(4, 8);
        let king_w = Square::new(0, 0);
        let gold_b = Square::new(4, 4);
        let rook_w = Square::new(4, 0);

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[gold_b.index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[rook_w.index()] = Piece::new(Color::White, PieceType::Rook);

        // 直接 dense にエミット (= fill_pin_features_for_side ロジック再現)。
        let mut out = vec![0.0f32; PIN_DIMENSIONS_PER_PERSPECTIVE];
        let (pinned, n_p) = collect_pinned_pieces(&board, Color::Black);
        for &(sq, pt) in &pinned[..n_p] {
            let Some(pt_idx) = piece_type_to_pin_idx(pt) else { continue };
            let sq_idx = oriented_sq_idx(sq.index(), Color::Black);
            out[pin_feature_index(sq_idx, pt_idx, 0)] = 1.0;
        }
        let (skewered, n_s) = collect_skewered_pieces(&board, Color::Black);
        for &(sq, pt) in &skewered[..n_s] {
            let Some(pt_idx) = piece_type_to_pin_idx(pt) else { continue };
            let sq_idx = oriented_sq_idx(sq.index(), Color::Black);
            out[pin_feature_index(sq_idx, pt_idx, 1)] = 1.0;
        }

        // Pinned = 5五金 (pt_idx=Gold=6、 state=0)
        let sq_idx_g = oriented_sq_idx(gold_b.index(), Color::Black);
        let idx_pinned_g = pin_feature_index(sq_idx_g, 6, 0);
        let idx_skewer_g = pin_feature_index(sq_idx_g, 6, 1);
        assert_eq!(out[idx_pinned_g], 1.0, "5五金 should be Pinned");
        assert_eq!(out[idx_skewer_g], 0.0, "5五金 should NOT be Skewered (= friend1)");

        // Skewer = 5九玉 (pt_idx=King=7、 state=1)
        let sq_idx_k = oriented_sq_idx(king_b.index(), Color::Black);
        let idx_pinned_k = pin_feature_index(sq_idx_k, 7, 0);
        let idx_skewer_k = pin_feature_index(sq_idx_k, 7, 1);
        assert_eq!(out[idx_pinned_k], 0.0, "5九玉 should NOT be Pinned");
        assert_eq!(out[idx_skewer_k], 1.0, "5九玉 should be Skewered (= 奥)");

        let nz = out.iter().filter(|&&v| v != 0.0).count();
        assert_eq!(nz, 2, "expected 2 active entries (Pinned 金 + Skewer 玉、 別マス)");
    }

    /// Standard X-ray skewer 検証: sniper 5一 r、 friend1 = 5五 G、 friend2 = 5七 P。
    /// 黒玉は 6九 K (= 5筋線上から外して Pinned 関係に入らない)、 白玉 1一 k。
    ///
    /// 期待: Pinned 集合 0、 Skewer 集合 1 駒 (= 5七 P)。 rshogi-core の同 test と数値一致。
    #[test]
    fn skewer_xray_position() {
        let mut board = ShogiBoard::default();
        let king_b = Square::new(5, 8); // 6九 (Black king、 5筋線外)
        let king_w = Square::new(0, 0); // 1一 (White king)
        let gold_b = Square::new(4, 4); // 5五 (Black Gold = friend1)
        let pawn_b = Square::new(4, 6); // 5七 (Black Pawn = friend2 / skewer)
        let rook_w = Square::new(4, 0); // 5一 (White Rook = sniper)

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[gold_b.index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[pawn_b.index()] = Piece::new(Color::Black, PieceType::Pawn);
        board.board[rook_w.index()] = Piece::new(Color::White, PieceType::Rook);

        // Pinned: king が 5筋線外なので 0 駒
        let (_pinned, n_pin) = collect_pinned_pieces(&board, Color::Black);
        assert_eq!(n_pin, 0, "no Pinned expected (king off 5-line)");

        // Skewer: 5七 Pawn 1 駒
        let (skewered, n_sk) = collect_skewered_pieces(&board, Color::Black);
        assert_eq!(n_sk, 1, "expected 1 Skewer entry (5七 Pawn)");
        assert_eq!(skewered[0].0.index(), pawn_b.index(), "skewer sq = 5七");
        assert_eq!(skewered[0].1, PieceType::Pawn, "skewer piece = Pawn");

        // 別マス確認 (= Pinned と Skewer が異なる sq)
        assert_ne!(gold_b.index(), pawn_b.index(), "different squares for pinned/skewer slots");
    }

    /// 奥の駒が自玉 (king) の場合も Skewer 成立 (= 新仕様、 friend1 が non-king なら奥 OK)。
    ///
    /// sniper 5一 r、 friend1 = 5五 G (= non-king)、 奥 = 5九 K (= 自玉)。
    /// 新仕様: friend1=non-king + 奥=自軍 (king 含む) なら Skewer 成立 → 5九玉も Skewer。
    /// (= rshogi-core `pin_simple_rook_pin_position` 同じ局面)
    #[test]
    fn skewer_includes_king_at_far_end() {
        let mut board = ShogiBoard::default();
        let king_b = Square::new(4, 8); // 5九 (= 奥候補 = king)
        let king_w = Square::new(0, 0); // 1一
        let gold_b = Square::new(4, 4); // 5五 (= friend1 non-king)
        let rook_w = Square::new(4, 0); // 5一 (= sniper)

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[gold_b.index()] = Piece::new(Color::Black, PieceType::Gold);
        board.board[rook_w.index()] = Piece::new(Color::White, PieceType::Rook);

        // Pinned: 5五金
        let (pinned, n_pin) = collect_pinned_pieces(&board, Color::Black);
        assert_eq!(n_pin, 1, "expected 1 Pinned (5五金)");
        assert_eq!(pinned[0].0.index(), gold_b.index());

        // Skewer: 5九玉 (= friend1 が non-king、 奥は自軍 king OK)
        let (skewered, n_sk) = collect_skewered_pieces(&board, Color::Black);
        assert_eq!(n_sk, 1, "expected 1 Skewered (5九玉、 = 奥は king OK)");
        assert_eq!(skewered[0].0.index(), king_b.index());
        assert_eq!(skewered[0].1, PieceType::King);
    }

    /// friend1 が king の場合は Skewer 不成立 (= Pin と重複しない、 新仕様)。
    ///
    /// 1筋 sniper 後手飛車 (1一) + 先手玉 (1四) + 先手銀 (1六)、 後手玉は file=7。
    /// ray: 1一飛 → 1四玉 (= friend1=king) → 1六銀。 friend1=king なので Skewer 不成立。
    #[test]
    fn skewer_pin_kingside_excluded() {
        let mut board = ShogiBoard::default();
        let king_b = Square::new(0, 3); // 1四 = friend1 候補 = king
        let king_w = Square::new(7, 8); // 2九
        let rook_w = Square::new(0, 0); // 1一 = sniper
        let silver_b = Square::new(0, 5); // 1六 = 奥候補

        board.black_king_sq = king_b;
        board.white_king_sq = king_w;
        board.board[king_b.index()] = Piece::new(Color::Black, PieceType::King);
        board.board[king_w.index()] = Piece::new(Color::White, PieceType::King);
        board.board[rook_w.index()] = Piece::new(Color::White, PieceType::Rook);
        board.board[silver_b.index()] = Piece::new(Color::Black, PieceType::Silver);

        // Skewer は 0 駒 (= friend1 が king なので X-ray ロジックで除外)
        let (_skewered, n_s) = collect_skewered_pieces(&board, Color::Black);
        assert_eq!(n_s, 0, "no Skewer should fire (= friend1 = king)");
    }
}

// SparseInputType は LMAS struct 側で実装するため、 本 module は trait 実装を持たない。
