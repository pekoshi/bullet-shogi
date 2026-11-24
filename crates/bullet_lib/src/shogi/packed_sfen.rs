//! PackedSfen / PackedSfenValue デコーダ
//!
//! YaneuraOu の教師データ形式を読み込むためのモジュール。
//! PackedSfenValue (40バイト) から局面を復元し、特徴量計算に使用する。
//!
//! YaneuraOu-latest 互換（駒箱対応）。

use super::types::{Color, Hand, Piece, PieceType, Square};

// =============================================================================
// Huffman 符号テーブル
// =============================================================================

/// Huffman 符号テーブル（YaneuraOu sfen_packer.cpp 準拠）
///
/// インデックス: 0=NO_PIECE, 1=PAWN, 2=LANCE, 3=KNIGHT, 4=SILVER, 5=BISHOP, 6=ROOK, 7=GOLD
const HUFFMAN_TABLE: [(u32, u8); 8] = [
    (0x00, 1), // NO_PIECE: 0
    (0x01, 2), // PAWN:     01
    (0x03, 4), // LANCE:    0011
    (0x0b, 4), // KNIGHT:   1011
    (0x07, 4), // SILVER:   0111
    (0x1f, 6), // BISHOP:   011111
    (0x3f, 6), // ROOK:     111111
    (0x0f, 5), // GOLD:     01111
];

/// 駒種インデックス（Huffman テーブル用）
const HUFFMAN_PAWN: usize = 1;
const HUFFMAN_LANCE: usize = 2;
const HUFFMAN_KNIGHT: usize = 3;
const HUFFMAN_SILVER: usize = 4;
const HUFFMAN_BISHOP: usize = 5;
const HUFFMAN_ROOK: usize = 6;
const HUFFMAN_GOLD: usize = 7;

// =============================================================================
// ビットストリーム
// =============================================================================

/// LSB-first ビットストリーム
pub struct BitStream<'a> {
    data: &'a [u8],
    bit_cursor: usize,
    /// ビット単位の上限（これ以上は読み出し不可）
    bit_limit: usize,
}

impl<'a> BitStream<'a> {
    /// 新しいビットストリームを作成
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, bit_cursor: 0, bit_limit: data.len() * 8 }
    }

    /// 現在のカーソル位置（ビット単位）
    #[inline]
    pub fn cursor(&self) -> usize {
        self.bit_cursor
    }

    /// 読み出し可能かどうか
    #[inline]
    pub fn can_read(&self) -> bool {
        self.bit_cursor < self.bit_limit
    }

    /// 1ビット読み出し（境界チェック付き）
    ///
    /// 境界を超えた場合は false を返す（安全なデフォルト）
    #[inline]
    pub fn read_bit(&mut self) -> bool {
        if self.bit_cursor >= self.bit_limit {
            return false;
        }
        let byte_pos = self.bit_cursor / 8;
        let bit_pos = self.bit_cursor % 8;
        let bit = (self.data[byte_pos] >> bit_pos) & 1;
        self.bit_cursor += 1;
        bit != 0
    }

    /// nビット読み出し（最大32ビット）
    #[inline]
    pub fn read_bits(&mut self, n: u8) -> u32 {
        let mut result = 0u32;
        for i in 0..n {
            if self.read_bit() {
                result |= 1 << i;
            }
        }
        result
    }

    /// カーソルを進めずに次の 8 bit を先読みする (LSB-first)。
    ///
    /// `bit_limit` を超える分は 0 として返す。 これは `read_bit` の境界挙動
    /// (= limit 超過は false を返す) と一致するため、 LUT デコードは bit-loop
    /// デコードと同じ結果を返す。
    #[inline]
    pub fn peek_bits8(&self) -> u8 {
        let byte_pos = self.bit_cursor / 8;
        let bit_pos = self.bit_cursor % 8;
        let b0 = if byte_pos < self.data.len() { self.data[byte_pos] as u16 } else { 0 };
        let b1 = if byte_pos + 1 < self.data.len() { self.data[byte_pos + 1] as u16 } else { 0 };
        let raw = (((b0 | (b1 << 8)) >> bit_pos) & 0xFF) as u8;
        let remaining = self.bit_limit.saturating_sub(self.bit_cursor);
        if remaining >= 8 { raw } else { raw & ((1u16 << remaining) - 1) as u8 }
    }

    /// カーソルを `bits` 進める (bit_limit で clamp)。
    ///
    /// `read_bit` は limit 到達後カーソルを進めないため、 clamp により
    /// bit-loop デコードとカーソル位置が完全一致する。
    #[inline]
    pub fn advance_clamped(&mut self, bits: usize) {
        self.bit_cursor = (self.bit_cursor + bits).min(self.bit_limit);
    }
}

// =============================================================================
// PackedSfen (32バイト)
// =============================================================================

/// Huffman符号化された局面データ (32バイト)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PackedSfen {
    pub data: [u8; 32],
}

impl PackedSfen {
    /// バイト配列への参照を取得
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.data
    }

    /// バイト配列への可変参照を取得
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 32] {
        &mut self.data
    }
}

// =============================================================================
// PackedSfenValue (40バイト)
// =============================================================================

/// YaneuraOu の教師データ形式 (40バイト)
///
/// これが `SparseInputType::RequiredDataType` として使用される。
///
/// メモリレイアウト:
/// - bytes 0-31:  PackedSfen (32 bytes)
/// - bytes 32-33: score (i16, little-endian)
/// - bytes 34-35: move16 (u16, little-endian)
/// - bytes 36-37: game_ply (u16, little-endian)
/// - byte 38:     game_result (i8)
/// - byte 39:     padding (u8)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PackedSfenValue {
    data: [u8; 40],
}

impl Default for PackedSfenValue {
    fn default() -> Self {
        Self { data: [0u8; 40] }
    }
}

// Safety: PackedSfenValue is a plain byte array
unsafe impl Send for PackedSfenValue {}
unsafe impl Sync for PackedSfenValue {}

impl PackedSfenValue {
    /// PackedSfen 部分への参照を取得
    pub fn sfen(&self) -> &PackedSfen {
        // Safety: PackedSfen は [u8; 32] と同じレイアウト
        unsafe { &*(self.data.as_ptr() as *const PackedSfen) }
    }

    /// 評価値を取得（手番側視点）
    /// 生の評価値 (psv に書かれている値そのまま)。
    ///
    /// 🚨🚨**同じ名前の `LoadableDataType::score` とは別の値を返す**。
    ///   こちらは inherent メソッドなので、 具体型 `&PackedSfenValue` から呼ぶと
    ///   **こちらが優先される** (trait の方は焼き込みを掛ける)。
    ///   2026-10-19 に fable のレビューで判明した、 生の score を読んでいる場所:
    ///     - `examples/shogi_layerstack.rs` の `loss_weight_fn` (|教師値| 帯の重み表)
    ///     - `shogi_halfka_hm_threat_profile11split.rs` の `threat_grad_weight`
    ///     - `packed_sfen_with_visits.rs` の `score()`
    ///     - `examples/shogi_layerstack_eval.rs` の target 表示
    ///   前 2 つは「教師の自信で重みを引く表」なので生の値で設計上は正しい。
    ///   🚨ただし**焼き込みを足すときはどちらを読んでいるか必ず確かめること**。
    ///     学習の target は `LoadableDataType::score` の方を通る。
    pub fn score(&self) -> i16 {
        i16::from_le_bytes([self.data[32], self.data[33]])
    }

    /// 指し手を取得
    pub fn move16(&self) -> u16 {
        u16::from_le_bytes([self.data[34], self.data[35]])
    }

    /// 手数を取得
    pub fn game_ply(&self) -> u16 {
        u16::from_le_bytes([self.data[36], self.data[37]])
    }

    /// 勝敗結果を取得
    /// 1=手番側の勝ち, 0=引き分け, -1=手番側の負け
    pub fn game_result(&self) -> i8 {
        self.data[38] as i8
    }

    /// バイトスライスへの参照を取得
    pub fn as_bytes(&self) -> &[u8; 40] {
        &self.data
    }

    /// バイトスライスへの可変参照を取得
    pub fn as_bytes_mut(&mut self) -> &mut [u8; 40] {
        &mut self.data
    }

    /// 局面をデコードして ShogiBoard を返す
    pub fn decode(&self) -> ShogiBoard {
        ShogiBoard::from_packed_sfen(self)
    }
}

// =============================================================================
// ShogiBoard - デコード済み局面
// =============================================================================

/// デコード済みの将棋局面
///
/// PackedSfenValue からデコードした結果を保持。
/// `map_features` で使用する。
#[derive(Clone)]
pub struct ShogiBoard {
    /// 盤面 (81マス)
    pub board: [Piece; 81],
    /// 先手の持ち駒
    pub black_hand: Hand,
    /// 後手の持ち駒
    pub white_hand: Hand,
    /// 手番
    pub side_to_move: Color,
    /// 先手玉の位置
    pub black_king_sq: Square,
    /// 後手玉の位置
    pub white_king_sq: Square,
    /// 評価値
    pub score: i16,
    /// 勝敗結果
    pub result: i8,
    /// 手数
    pub ply: u16,
}

// ---- 右玉ボーナスの焼き込み (2026-10-18)
//   ユーザー指示「右玉ボーナスを学習させて欲しかった」
//     「**少し低めに焼き付けて、ボーナス機構はそのままで**」「ボーナス機構で微調整します」
//     「**少し悪い -> 互角になるような値にしてください いい教師だと右玉は悪くないはずなので！**」
//     「dl水匠では分からない局面で、最近出てきた定跡なので！」
//     「floodgate上位陣は分かっています！定跡搭載しているので！」
//
// 🎯なぜ**学習器側**でやるか (psv を作り直す案が死んだ理由)
//   注入 pool (fgtop4000) の右玉局面は **0.97% = 41,468 局面**しかなく、
//   注入 2.4% で net が見るのは **全教師の 0.0046%** = 勾配が届かない。
//   厚くすれば届くが 10 倍で **41 回/局面** = 過学習の帯 (安全帯 4-5 回)。
//   ここで焼けば**土台 400M も含む全局面**に効くので、 ディスク 0 で信号量が 200 倍になる。
//
// 🎯焼く値の根拠 (`tools/migi_score_dist.py` / `tools/migi_color_strat.py` の実測)
//   pool の右玉局面の score は**右玉側から見て中央 -44 (Migigyoku) / -27 (Aigakari)**。
//   = ユーザーの言う「少し悪い」。 これを 0 (互角) に戻すのが既定の使い方。
//   🚨同じ実測で**実際の勝率は色を合わせても -10.7 / -7.4 pt** 残る。 つまり
//     「score の見立てを直す」話と「右玉が勝てる形か」は別問題。 前者だけを焼く。
//
// 🚨既定 (0, 0) = **完全に無効で bit-exact**。 有効時は必ず 1 回 stderr に出す
//   (★settings_audit: HIST_CONFIDENCE_SCALE が toml 未記載で暗黙 ON だった反省)。
/// この手数以降は焼かない。 engine の `g_migi_fade_end` と同じ 70。
///
/// 🚨🚨**fade は掛けない** (2026-10-18 ユーザー指示「fade外して」)。
///   engine (`migigyoku_bonus_black_view`) は ply 30->70 で 1.0->0 に減衰させるが、
///   **net は手数を入力に持たない** (`--bucket-mode progress8kpabs` は駒の進行度で
///   手数ではない) ので、 fade を掛けると**同じ盤面に違う score を教える**ことになり
///   矛盾した教師になる。 これは p2nnue.cpp:787-792 のコメント
///   「net のラベルに下駄を履かせる案は (c) net は手数を入力に持たない」が指摘していた点。
///   ✅net は「右玉の形 -> +G」を一定で覚え、 手数による調整は**engine のボーナス側**が持つ
///     (ユーザー方針「ボーナス機構はそのままで / ボーナス機構で微調整します」と整合)。
const MIGI_PLY_END: u16 = 70;

/// 焼き込みの強さ `(Migigyoku, Aigakari)` を cp で返す。 既定 `(0, 0)` = 無効。
///
/// 環境変数 `P2_MIGI_BAKE` / `P2_AIGA_BAKE` から 1 回だけ読む。 数値として読めない値は
/// 0 (無効) として扱う (学習が黙って別物になるのを避けるため、 パニックにはしない)。
fn migi_bake_cfg() -> (i32, i32) {
    static CFG: std::sync::OnceLock<(i32, i32)> = std::sync::OnceLock::new();
    *CFG.get_or_init(|| {
        let read = |k: &str| std::env::var(k).ok().and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or(0);
        let (g, ga) = (read("P2_MIGI_BAKE"), read("P2_AIGA_BAKE"));
        if g != 0 || ga != 0 {
            eprintln!(
                "[migi-bake] 右玉ボーナスをラベルに焼き込む: Migigyoku {g} cp / Aigakari {ga} cp \
                 (角換わり限定 / ply < {MIGI_PLY_END} / **fade なし** / Aigakari 優先)"
            );
        }
        (g, ga)
    })
}

impl Default for ShogiBoard {
    fn default() -> Self {
        Self {
            board: [Piece::NONE; 81],
            black_hand: Hand::EMPTY,
            white_hand: Hand::EMPTY,
            side_to_move: Color::Black,
            black_king_sq: Square::NONE,
            white_king_sq: Square::NONE,
            score: 0,
            result: 0,
            ply: 0,
        }
    }
}

impl PackedSfenValue {
    /// `ShogiBoard::from_packed_sfen(self)` の wrapper (= 2026-06-10、 既存 KED / ARH
    /// `fill_*_sparse_head` で利用)。
    #[inline]
    pub fn board(&self) -> ShogiBoard {
        ShogiBoard::from_packed_sfen(self)
    }
}

impl ShogiBoard {
    /// SFEN string にエンコード (= `crate::game::inputs::shogi_board_to_sfen` の
    /// wrapper、 2026-06-10 追加)。 KED / ARH の delegation pattern で使う。
    #[inline]
    pub fn to_sfen(&self) -> String {
        crate::game::inputs::shogi_board_to_sfen(self)
    }

    /// 右玉か (先手, 後手) を返す。 engine の閾値そのまま = **玉と飛車が同じ側**。
    ///
    /// 先手: 玉の筋 <= 4 かつ **飛車**の筋 <= 4 / 後手: どちらも筋 >= 6。
    /// 🚨`Square::file()` は 0 起点 (0 = 1筋) なので「筋 <= 4」は `file() <= 3`。
    ///
    /// 🚨🚨**龍 (成飛) は数えない**。 2026-10-19 に fable のレビューで発覚:
    ///   engine は `pos.pieces(BLACK, ROOK)` を使っており、 YaneuraOu の
    ///   `byTypeBB[ROOK]` は**飛車のみ** (龍込みは別の合成 bitboard `ROOK_DRAGON` /
    ///   `position.cpp:1133` / `types.h:566`)。
    ///   最初は `Rook || Dragon` と書いていて、 python 参照も**同じ誤りを共有**していたため
    ///   「実データ 300,000 件で 386 件一致」でも検出できなかった
    ///   (= 同じ仮定を 2 回写しただけ。 ★control_must_use_production_path の変種)。
    fn migi_sides(&self) -> (i32, i32) {
        let mut ib = 0;
        let mut iw = 0;
        // 🚨7 bit 読みなので 82-127 もあり得る。 NONE (81) だけの除外では足りない
        if self.black_king_sq.index() < 81 && self.black_king_sq.file() <= 3 {
            for (i, p) in self.board.iter().enumerate() {
                if p.piece_type == PieceType::Rook && p.color == Color::Black && Square::from_index(i).file() <= 3 {
                    ib = 1;
                    break;
                }
            }
        }
        if self.white_king_sq.index() < 81 && self.white_king_sq.file() >= 5 {
            for (i, p) in self.board.iter().enumerate() {
                if p.piece_type == PieceType::Rook && p.color == Color::White && Square::from_index(i).file() >= 5 {
                    iw = 1;
                    break;
                }
            }
        }
        (ib, iw)
    }

    /// 相掛かり = **双方が飛車先の歩を伸ばしている**か。
    ///
    /// 先手の 2 筋の歩が 2六以前 (段 <= 6) かつ 後手の 8 筋の歩が 8四以降 (段 >= 4)。
    /// 🚨`Square::rank()` は 0 起点 (0 = 1段) なので「段 <= 6」は `rank() <= 5`。
    fn is_aigakari_shape(&self) -> bool {
        let mut adv_b = false;
        let mut adv_w = false;
        for (i, p) in self.board.iter().enumerate() {
            if p.piece_type != PieceType::Pawn {
                continue;
            }
            let sq = Square::from_index(i);
            if p.color == Color::Black {
                if sq.file() == 1 && sq.rank() <= 5 {
                    adv_b = true;
                }
            } else if sq.file() == 7 && sq.rank() >= 3 {
                adv_w = true;
            }
            if adv_b && adv_w {
                return true;
            }
        }
        false
    }

    /// 右玉ボーナスを **手番側視点の cp** で返す。 無効時 (既定) は 0。
    ///
    /// engine (`p2nnue.cpp` の `migigyoku_bonus_black_view` /
    /// `aigakari_migi_bonus_black_view`) と同じ式・同じ排他・同じ fade。
    /// ```text
    /// bonus_black = (ib - iw) * G                      先手視点 cp
    ///   手数    ply >= 70 -> 0 / それ未満は**一定** (engine の fade は掛けない)
    ///   角換わり 双方が角を手駒に持つ (engine の g_migi_kakugawari = 1 が既定)
    ///   排他    Aigakari が発火した局面では Migigyoku を 0 にする (**160 側を優先**)
    /// ```
    /// 🚨整数除算は 0 方向切り捨て。 Rust も C++ も同じなので式をそのまま写せる。
    fn migi_bake_stm(&self) -> i16 {
        let (g, ga) = migi_bake_cfg();
        self.migi_bonus_stm(g, ga)
    }

    /// 強さ `(g, ga)` を与えたときの手番側視点の cp。 `migi_bake_stm` の実体。
    ///
    /// 🚨`migi_bake_cfg` は `OnceLock` なので 1 プロセスで 1 回しか読めず、
    ///   テストから強さを変えられない。 そのため式だけをここに分けてある。
    fn migi_bonus_stm(&self, g: i32, ga: i32) -> i16 {
        if g == 0 && ga == 0 {
            return 0;
        }
        if self.ply >= MIGI_PLY_END {
            return 0;
        }
        if !(self.black_hand.count(PieceType::Bishop) > 0 && self.white_hand.count(PieceType::Bishop) > 0) {
            return 0;
        }
        let (ib, iw) = self.migi_sides();
        if ib == iw {
            return 0;
        }
        let gain = if ga != 0 && self.is_aigakari_shape() {
            ga
        } else if g != 0 {
            g
        } else {
            return 0;
        };
        // 🚨fade なし = ply < MIGI_PLY_END なら一定で G を乗せる
        let black = (ib - iw) * gain;
        let stm = if self.side_to_move == Color::Black { black } else { -black };
        stm as i16
    }

    /// PackedSfenValue からデコード
    ///
    /// `set_board_decode_cache_enabled(true)` のとき、 packed sfen 32 byte を key に
    /// デコード結果をメモ化する。 aux head (Dual PSQT / PairShape / KingDanger 等) を
    /// 複数有効にした学習では同一局面が head ごとに再デコードされるため
    /// (PreparedData の aux fill が head × chunk で並列走行する)、 メモ化により
    /// デコード回数が 1/(有効 head 数) に落ちる。 score/result/ply は PSV ごとに
    /// 異なり得るので、 cache hit 時も必ず PSV から上書きする。
    pub fn from_packed_sfen(psv: &PackedSfenValue) -> Self {
        let mut board = Self::from_packed_sfen_inner(psv);
        // 右玉ボーナスの焼き込み (2026-10-18)。 既定 0 = 何もしない (bit-exact)。
        let d = board.migi_bake_stm();
        if d != 0 {
            board.score = board.score.saturating_add(d);
        }
        board
    }

    /// `from_packed_sfen` の実体 (焼き込みを掛ける前のデコード)。
    ///
    /// 🚨cache hit と uncached の **2 経路**があるので、 焼き込みは呼び出し側で
    ///   1 回だけ掛ける。 ここに書くと cache hit 経路で二重に乗る。
    fn from_packed_sfen_inner(psv: &PackedSfenValue) -> Self {
        if board_decode_cache_enabled() {
            let key = psv.sfen().data;
            let stripe = &board_decode_cache()[board_cache_stripe(&key)];

            if let Some(cached) = stripe.lock().unwrap().get(&key) {
                let mut board = cached.clone();
                board.score = psv.score();
                board.result = psv.game_result();
                board.ply = psv.game_ply();
                return board;
            }

            let board = Self::decode_packed_sfen_uncached(psv);
            let mut guard = stripe.lock().unwrap();
            if guard.len() >= BOARD_CACHE_STRIPE_CAP {
                // 満杯なら stripe ごと破棄 (再利用は batch 内 head 間のみなので
                // 使い捨てで十分。 LRU 管理のコストを避ける)。
                guard.clear();
            }
            guard.insert(key, board.clone());
            return board;
        }

        Self::decode_packed_sfen_uncached(psv)
    }

    /// PackedSfenValue からデコード (キャッシュ非経由の実体)
    fn decode_packed_sfen_uncached(psv: &PackedSfenValue) -> Self {
        let mut board =
            ShogiBoard { score: psv.score(), result: psv.game_result(), ply: psv.game_ply(), ..Default::default() };

        let mut stream = BitStream::new(&psv.sfen().data);

        // 1. 手番 (1 bit)
        board.side_to_move = if stream.read_bit() { Color::White } else { Color::Black };

        // 2. 玉の位置 (7 bit × 2)
        let black_king_idx = stream.read_bits(7) as u8;
        let white_king_idx = stream.read_bits(7) as u8;

        board.black_king_sq = Square(black_king_idx);
        board.white_king_sq = Square(white_king_idx);

        // 玉を盤面に配置
        if black_king_idx < 81 {
            board.board[black_king_idx as usize] = Piece::new(Color::Black, PieceType::King);
        }
        if white_king_idx < 81 {
            board.board[white_king_idx as usize] = Piece::new(Color::White, PieceType::King);
        }

        // 3. 盤上の駒 (Huffman符号)
        for sq_idx in 0..81u8 {
            // 玉位置はスキップ
            if sq_idx == black_king_idx || sq_idx == white_king_idx {
                continue;
            }

            let piece = decode_board_piece(&mut stream);
            board.board[sq_idx as usize] = piece;
        }

        // 4. 持ち駒・駒箱 (Huffman符号、256bitまで)
        while stream.cursor() < 256 {
            let (piece, is_piecebox) = decode_hand_piece(&mut stream);

            // 駒箱の駒は無視（駒落ち対応）
            if is_piecebox {
                continue;
            }

            // 持ち駒に追加
            let pt = piece.piece_type;
            match piece.color {
                Color::Black => board.black_hand.add(pt, 1),
                Color::White => board.white_hand.add(pt, 1),
            }
        }

        board
    }

    /// 指定マスの駒を取得
    #[inline]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.index()]
    }

    /// 指定色の玉位置を取得
    #[inline]
    pub fn king_square(&self, color: Color) -> Square {
        match color {
            Color::Black => self.black_king_sq,
            Color::White => self.white_king_sq,
        }
    }

    /// 指定色の持ち駒を取得
    #[inline]
    pub fn hand(&self, color: Color) -> &Hand {
        match color {
            Color::Black => &self.black_hand,
            Color::White => &self.white_hand,
        }
    }

    /// 盤上の指定色・駒種の駒を列挙
    pub fn pieces(&self, color: Color, pt: PieceType) -> impl Iterator<Item = Square> + '_ {
        self.board
            .iter()
            .enumerate()
            .filter(move |(_, p)| p.color == color && p.piece_type == pt)
            .map(|(i, _)| Square(i as u8))
    }
}

// =============================================================================
// 局面デコードのメモ化キャッシュ
// =============================================================================

/// stripe 数 (並列 prepare スレッドのロック競合を分散)
const BOARD_CACHE_STRIPES: usize = 64;
/// stripe あたりの最大 entry 数。 合計 64 × 2048 = 131072 局面 (> batch 16384 ×
/// 複数 chunk) を保持でき、 batch 内の head 間再利用には十分。
const BOARD_CACHE_STRIPE_CAP: usize = 2048;

static BOARD_DECODE_CACHE_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 局面デコードのメモ化キャッシュを有効/無効にする。
///
/// aux head を複数使う学習 (同一局面が head ごとに再デコードされる) では有効化で
/// CPU-bound なデータ準備が大幅に速くなる。 head なし構成ではデコードは局面あたり
/// 1 回なのでキャッシュは無駄になる — 既定は無効。
pub fn set_board_decode_cache_enabled(enabled: bool) {
    BOARD_DECODE_CACHE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn board_decode_cache_enabled() -> bool {
    BOARD_DECODE_CACHE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[allow(clippy::type_complexity)]
fn board_decode_cache() -> &'static [std::sync::Mutex<std::collections::HashMap<[u8; 32], ShogiBoard>>] {
    static CACHE: std::sync::OnceLock<Vec<std::sync::Mutex<std::collections::HashMap<[u8; 32], ShogiBoard>>>> =
        std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        (0..BOARD_CACHE_STRIPES).map(|_| std::sync::Mutex::new(std::collections::HashMap::new())).collect()
    })
}

/// packed sfen 32 byte から stripe index を導出 (FNV-1a 風の軽量 hash)。
#[inline]
fn board_cache_stripe(key: &[u8; 32]) -> usize {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for chunk in key.chunks_exact(8) {
        h ^= u64::from_le_bytes(chunk.try_into().unwrap());
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    (h >> 32) as usize % BOARD_CACHE_STRIPES
}

// =============================================================================
// Huffman デコード
// =============================================================================

/// Huffman 符号インデックスから PieceType に変換
fn huffman_index_to_piece_type(idx: usize) -> PieceType {
    match idx {
        HUFFMAN_PAWN => PieceType::Pawn,
        HUFFMAN_LANCE => PieceType::Lance,
        HUFFMAN_KNIGHT => PieceType::Knight,
        HUFFMAN_SILVER => PieceType::Silver,
        HUFFMAN_BISHOP => PieceType::Bishop,
        HUFFMAN_ROOK => PieceType::Rook,
        HUFFMAN_GOLD => PieceType::Gold,
        _ => PieceType::None,
    }
}

/// 盤上の駒をデコード (LUT 版、 bit-loop 参照実装と完全一致)
///
/// 盤上駒の Huffman 符号は最大 6 bit + 成り bit + 先後 bit = 最大 8 bit なので、
/// 次の 8 bit を先読みして 256-entry LUT を 1 回引くだけで (駒, 消費 bit 数) が
/// 確定する。 旧実装 (1 bit ずつ読みつつ 8-entry table を線形走査) 比で
/// デコードが大幅に高速化される。
#[inline]
fn decode_board_piece(stream: &mut BitStream) -> Piece {
    let lut = board_piece_lut();
    let (piece, consumed) = lut[stream.peek_bits8() as usize];
    stream.advance_clamped(consumed as usize);
    piece
}

/// 手駒をデコード (LUT 版、 bit-loop 参照実装と完全一致)
///
/// 手駒符号は最大 5 bit + 駒箱 bit + 先後 bit = 最大 7 bit ≤ 8 bit。
#[inline]
fn decode_hand_piece(stream: &mut BitStream) -> (Piece, bool) {
    let lut = hand_piece_lut();
    let (piece, is_piecebox, consumed) = lut[stream.peek_bits8() as usize];
    stream.advance_clamped(consumed as usize);
    (piece, is_piecebox)
}

/// 盤上駒 LUT: index = 次の 8 bit (LSB-first)、 value = (駒, 消費 bit 数)。
fn board_piece_lut() -> &'static [(Piece, u8); 256] {
    static LUT: std::sync::OnceLock<[(Piece, u8); 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [(Piece::NONE, 0u8); 256];
        for (bits, entry) in table.iter_mut().enumerate() {
            let data = [bits as u8];
            let mut stream = BitStream::new(&data);
            let piece = decode_board_piece_bitloop(&mut stream);
            *entry = (piece, stream.cursor() as u8);
        }
        table
    })
}

/// 手駒 LUT: index = 次の 8 bit (LSB-first)、 value = (駒, 駒箱フラグ, 消費 bit 数)。
fn hand_piece_lut() -> &'static [(Piece, bool, u8); 256] {
    static LUT: std::sync::OnceLock<[(Piece, bool, u8); 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [(Piece::NONE, false, 0u8); 256];
        for (bits, entry) in table.iter_mut().enumerate() {
            let data = [bits as u8];
            let mut stream = BitStream::new(&data);
            let (piece, is_piecebox) = decode_hand_piece_bitloop(&mut stream);
            *entry = (piece, is_piecebox, stream.cursor() as u8);
        }
        table
    })
}

/// 盤上の駒をデコード (bit-loop 参照実装、 LUT 構築とテストで使用)
///
/// YaneuraOu sfen_packer.cpp の read_board_piece_from_stream() に準拠。
/// 形式: Huffman符号 + 成りbit(金以外) + 先後bit
fn decode_board_piece_bitloop(stream: &mut BitStream) -> Piece {
    // Huffman 符号をデコードして駒種を取得
    let mut code = 0u32;
    let mut bits = 0u8;

    loop {
        code |= (stream.read_bit() as u32) << bits;
        bits += 1;

        // テーブルから一致するエントリを探す
        for (idx, &(pattern, len)) in HUFFMAN_TABLE.iter().enumerate() {
            if bits == len && code == pattern {
                if idx == 0 {
                    // NO_PIECE
                    return Piece::NONE;
                }

                let base_pt = huffman_index_to_piece_type(idx);

                // 成りフラグ（金以外）
                let promoted = if idx != HUFFMAN_GOLD { stream.read_bit() } else { false };

                // 先後フラグ
                let color = if stream.read_bit() { Color::White } else { Color::Black };

                let pt = if promoted { base_pt.promote() } else { base_pt };
                return Piece::new(color, pt);
            }
        }

        // 最大6ビット
        if bits > 6 {
            return Piece::NONE;
        }
    }
}

/// 手駒をデコード (bit-loop 参照実装、 LUT 構築とテストで使用)
///
/// YaneuraOu sfen_packer.cpp の read_hand_piece_from_stream() に準拠。
/// 形式: Huffman符号(bit0を除く) + 成りbit(金以外、駒箱判定用) + 先後bit
///
/// 戻り値: (駒, 駒箱フラグ)
/// - 駒箱フラグが true の場合、その駒は持ち駒ではなく駒箱の駒
fn decode_hand_piece_bitloop(stream: &mut BitStream) -> (Piece, bool) {
    // 手駒の Huffman 符号は盤上より1ビット少ない（bit0 を省略）
    let mut code = 0u32;
    let mut bits = 0u8;

    loop {
        code |= (stream.read_bit() as u32) << bits;
        bits += 1;

        for (idx, &(pattern, len)) in HUFFMAN_TABLE.iter().enumerate() {
            // 手駒: pattern >> 1, len - 1
            if idx == 0 {
                continue; // NO_PIECE は手駒にない
            }

            let hand_pattern = pattern >> 1;
            let hand_len = len - 1;

            if bits == hand_len && code == hand_pattern {
                let base_pt = huffman_index_to_piece_type(idx);

                // 成りフラグ（金以外）
                // これが true なら駒箱の駒
                let is_piecebox = if idx != HUFFMAN_GOLD { stream.read_bit() } else { false };

                // 先後フラグ
                let color = if stream.read_bit() { Color::White } else { Color::Black };

                let piece = Piece::new(color, base_pt);
                return (piece, is_piecebox);
            }
        }

        if bits > 5 {
            // エラー: 不正な符号
            return (Piece::NONE, false);
        }
    }
}

// =============================================================================
// DirectSequentialDataLoader 対応
// =============================================================================

/// PackedSfenValue は固定長で transmute 可能
///
/// # Safety
/// - `#[repr(C)]` でメモリレイアウトが固定
/// - 40バイト固定長
/// - 任意のビットパターンでもパニックしない
unsafe impl crate::value::loader::CanBeDirectlySequentiallyLoaded for PackedSfenValue {}

/// LoadableDataType 実装
impl crate::value::loader::LoadableDataType for PackedSfenValue {
    /// 学習の target になる評価値。
    ///
    /// 🚨🚨**ここが学習の target の経路**。 2026-10-19 に発覚した罠:
    ///   右玉ボーナスの焼き込みを `from_packed_sfen` (= **特徴量**の経路) に入れたが、
    ///   target はこの関数を通るので **焼き込みが 1 cp も効いていなかった**
    ///   (8 SB 回した学習が「BOOK2 の続きを回すだけ」になっていた)。
    ///   ✅ここでも焼く。 既定 (0, 0) のときは **psfen をデコードせずに**素の score を
    ///     返すので、 従来と **bit-exact かつ速度も不変**。
    fn score(&self) -> i16 {
        let (g, ga) = migi_bake_cfg();
        if g == 0 && ga == 0 {
            return PackedSfenValue::score(self);
        }
        // 焼き込みが有効なときだけ board を作る (board.score は焼いた後の値)
        ShogiBoard::from_packed_sfen(self).score
    }

    fn result(&self) -> crate::value::loader::GameResult {
        match self.game_result() {
            r if r > 0 => crate::value::loader::GameResult::Win,
            r if r < 0 => crate::value::loader::GameResult::Loss,
            _ => crate::value::loader::GameResult::Draw,
        }
    }
}

// =============================================================================
// テスト
// =============================================================================

#[cfg(test)]
mod tests {
    /// 右玉の盤面を手で組む。 `migi_b` / `migi_w` でどちら側を右玉にするか決める。
    ///
    /// 🚨筋段は `Square::new(file, rank)` の **0 起点**。 筋 2 = file 1 / 段 8 = rank 7。
    fn migi_board(migi_b: bool, migi_w: bool, ply: u16, stm: Color, aiga: bool) -> ShogiBoard {
        let mut b = ShogiBoard { ply, side_to_move: stm, ..Default::default() };
        // 先手玉 2八 (右玉) か 8八 (居飛車の通常位置)
        let kb = if migi_b { Square::new(1, 7) } else { Square::new(7, 7) };
        let kw = if migi_w { Square::new(7, 1) } else { Square::new(1, 1) };
        b.black_king_sq = kb;
        b.white_king_sq = kw;
        b.board[kb.index()] = Piece::new(Color::Black, PieceType::King);
        b.board[kw.index()] = Piece::new(Color::White, PieceType::King);
        // 飛車は玉と同じ側に置く (右玉の条件 = 玉と飛車が同じ側)
        let rb = if migi_b { Square::new(2, 7) } else { Square::new(6, 7) };
        let rw = if migi_w { Square::new(6, 1) } else { Square::new(2, 1) };
        b.board[rb.index()] = Piece::new(Color::Black, PieceType::Rook);
        b.board[rw.index()] = Piece::new(Color::White, PieceType::Rook);
        // 角換わり: 双方が角を手駒に持つ
        b.black_hand.set(PieceType::Bishop, 1);
        b.white_hand.set(PieceType::Bishop, 1);
        // 相掛かり: 先手の 2 筋の歩を 2六 (段 6 = rank 5) / 後手の 8 筋の歩を 8四 (段 4 = rank 3)
        if aiga {
            b.board[Square::new(1, 5).index()] = Piece::new(Color::Black, PieceType::Pawn);
            b.board[Square::new(7, 3).index()] = Piece::new(Color::White, PieceType::Pawn);
        } else {
            b.board[Square::new(1, 6).index()] = Piece::new(Color::Black, PieceType::Pawn);
            b.board[Square::new(7, 2).index()] = Piece::new(Color::White, PieceType::Pawn);
        }
        b
    }

    #[test]
    fn migi_bake_matches_engine_formula() {
        // 陰性: 双方右玉でない -> 0
        let b = migi_board(false, false, 30, Color::Black, false);
        assert_eq!(b.migi_sides(), (0, 0));
        assert_eq!(b.migi_bonus_stm(44, 27), 0);
        // 陰性: 双方が右玉 -> 0 (engine も ib == iw なら 0)
        let b = migi_board(true, true, 30, Color::Black, false);
        assert_eq!(b.migi_sides(), (1, 1));
        assert_eq!(b.migi_bonus_stm(44, 27), 0);
        // 陽性: 先手だけ右玉 / 先手番 -> +44
        let b = migi_board(true, false, 30, Color::Black, false);
        assert_eq!(b.migi_sides(), (1, 0));
        assert!(!b.is_aigakari_shape());
        assert_eq!(b.migi_bonus_stm(44, 27), 44);
        // 🚨手番で符号が変わる (psv の score は手番側視点)
        let b = migi_board(true, false, 30, Color::White, false);
        assert_eq!(b.migi_bonus_stm(44, 27), -44);
        // 陽性: 後手だけ右玉 / 後手番 -> +44 (後手から見て良い)
        let b = migi_board(false, true, 30, Color::White, false);
        assert_eq!(b.migi_sides(), (0, 1));
        assert_eq!(b.migi_bonus_stm(44, 27), 44);
        // 🚨**fade なし**: ply 50 でも ply 30 と同じ値 (2026-10-18 ユーザー指示「fade外して」)
        //   net は手数を入力に持たないので、 手数で値を変えると矛盾した教師になる。
        let b = migi_board(true, false, 50, Color::Black, false);
        assert_eq!(b.migi_bonus_stm(44, 27), 44);
        let b = migi_board(true, false, 69, Color::Black, false);
        assert_eq!(b.migi_bonus_stm(44, 27), 44);
        // 打ち切り: ply 70 以降は 0
        let b = migi_board(true, false, 70, Color::Black, false);
        assert_eq!(b.migi_bonus_stm(44, 27), 0);
        let b = migi_board(true, false, 120, Color::Black, false);
        assert_eq!(b.migi_bonus_stm(44, 27), 0);
        // 排他: 相掛かりの形では **Aigakari 側 (27)** だけが乗る
        let b = migi_board(true, false, 30, Color::Black, true);
        assert!(b.is_aigakari_shape());
        assert_eq!(b.migi_bonus_stm(44, 27), 27);
        // 角換わりでない (角を手駒に持たない) -> 0
        let mut b = migi_board(true, false, 30, Color::Black, false);
        b.white_hand.set(PieceType::Bishop, 0);
        assert_eq!(b.migi_bonus_stm(44, 27), 0);
        // 既定 (0, 0) は何もしない = bit-exact
        let b = migi_board(true, false, 30, Color::Black, true);
        assert_eq!(b.migi_bonus_stm(0, 0), 0);
    }

    /// 🚨🚨**学習の target の経路が焼き込みを通るか**。
    ///
    /// 2026-10-19: `from_packed_sfen` (特徴量の経路) だけに焼き込みを入れていたため、
    /// `LoadableDataType::score` を通る **target には 1 cp も効いていなかった**。
    /// 既定 (env 未設定) では両者が一致することを確かめる (env は OnceLock なので
    /// テストから強さを変えられない / 有効時の一致は examples/migi_bake_verify.rs で
    /// 実データ 300,000 件を突き合わせて確認する)。
    #[test]
    fn target_score_path_matches_board_score() {
        use crate::value::loader::LoadableDataType;
        let mut psv = PackedSfenValue::default();
        // 平手の packed sfen を入れる代わりに、 score だけ見る (sfen は 0 埋めでも
        // from_packed_sfen は落ちない = BitStream が範囲外を 0 として読む)
        psv.data[32] = 0x34;
        psv.data[33] = 0x12;
        assert_eq!(psv.score(), 0x1234);
        // 既定 (焼き込み無効) では target も board.score も素の score と一致する
        assert_eq!(LoadableDataType::score(&psv), 0x1234);
        assert_eq!(ShogiBoard::from_packed_sfen(&psv).score, 0x1234);
    }

    /// 🚨🚨**龍 (成飛) は飛車として数えない**。
    ///
    /// engine は `pos.pieces(BLACK, ROOK)` で、 YaneuraOu の `byTypeBB[ROOK]` は
    /// **飛車のみ** (龍込みは `ROOK_DRAGON`)。 2026-10-19 に fable のレビューで発覚し、
    /// それまでは逆の assert (龍も数える) を書いていた。
    #[test]
    fn migi_sides_ignores_promoted_rook() {
        let mut b = migi_board(true, false, 30, Color::Black, false);
        // 飛車を龍に差し替えると、 玉と同じ側に**飛車が無い**ので右玉でなくなる
        b.board[Square::new(2, 7).index()] = Piece::new(Color::Black, PieceType::Dragon);
        assert_eq!(b.migi_sides(), (0, 0));
        // 飛車に戻せば右玉
        b.board[Square::new(2, 7).index()] = Piece::new(Color::Black, PieceType::Rook);
        assert_eq!(b.migi_sides(), (1, 0));
    }

    use super::*;

    #[test]
    fn test_packed_sfen_value_size() {
        assert_eq!(std::mem::size_of::<PackedSfenValue>(), 40);
    }

    /// LUT デコードが bit-loop 参照実装と全 256 パターン × 全カーソル位置で
    /// 一致すること (駒・駒箱フラグ・カーソル進行量すべて)。
    #[test]
    fn test_lut_decode_matches_bitloop_all_patterns() {
        for hi in 0..=255u8 {
            for lo in 0..=255u8 {
                let data = [lo, hi];
                // 全カーソル開始位置 (境界 = limit 付近含む) で比較
                for start in 0..16usize {
                    // 盤上駒
                    let mut s_lut = BitStream::new(&data);
                    s_lut.advance_clamped(start);
                    let mut s_ref = BitStream::new(&data);
                    s_ref.advance_clamped(start);
                    let p_lut = decode_board_piece(&mut s_lut);
                    let p_ref = decode_board_piece_bitloop(&mut s_ref);
                    assert_eq!(p_lut, p_ref, "board piece mismatch: data={data:?} start={start}");
                    assert_eq!(s_lut.cursor(), s_ref.cursor(), "board cursor mismatch: data={data:?} start={start}");

                    // 手駒
                    let mut s_lut = BitStream::new(&data);
                    s_lut.advance_clamped(start);
                    let mut s_ref = BitStream::new(&data);
                    s_ref.advance_clamped(start);
                    let h_lut = decode_hand_piece(&mut s_lut);
                    let h_ref = decode_hand_piece_bitloop(&mut s_ref);
                    assert_eq!(h_lut, h_ref, "hand piece mismatch: data={data:?} start={start}");
                    assert_eq!(s_lut.cursor(), s_ref.cursor(), "hand cursor mismatch: data={data:?} start={start}");
                }
            }
        }
    }

    /// メモ化キャッシュ有効時もデコード結果が一致し、 score/result/ply は
    /// PSV ごとに正しく上書きされること。
    #[test]
    fn test_board_decode_cache_consistency() {
        // 疑似 PSV: ランダム風バイト列 (デコード自体は任意 bit 列で安全)
        let mut psv1 = PackedSfenValue::default();
        let mut state = 0x1234_5678_9abc_def0u64;
        for b in psv1.as_bytes_mut().iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = (state & 0xFF) as u8;
        }
        // 同一局面・異なる score/result/ply の PSV
        let mut psv2 = psv1;
        psv2.as_bytes_mut()[32] = 0x11;
        psv2.as_bytes_mut()[33] = 0x22;
        psv2.as_bytes_mut()[38] = 1;

        let ref1 = ShogiBoard::decode_packed_sfen_uncached(&psv1);
        let ref2 = ShogiBoard::decode_packed_sfen_uncached(&psv2);

        set_board_decode_cache_enabled(true);
        // 1 回目 (miss) と 2 回目 (hit) 両方を検証
        for _ in 0..2 {
            let c1 = ShogiBoard::from_packed_sfen(&psv1);
            assert_eq!(c1.board, ref1.board);
            assert_eq!(c1.score, ref1.score);
            assert_eq!(c1.result, ref1.result);
            assert_eq!(c1.ply, ref1.ply);
            let c2 = ShogiBoard::from_packed_sfen(&psv2);
            assert_eq!(c2.board, ref2.board);
            assert_eq!(c2.score, ref2.score, "cache hit は PSV の score を上書きすべき");
            assert_eq!(c2.result, ref2.result);
        }
        set_board_decode_cache_enabled(false);
    }

    #[test]
    fn test_packed_sfen_size() {
        assert_eq!(std::mem::size_of::<PackedSfen>(), 32);
    }

    #[test]
    fn test_bitstream_read_bits() {
        let data = [0b10101010, 0b11001100];
        let mut stream = BitStream::new(&data);

        // 0b10101010 の bit0-3 = 0,1,0,1 → result に bit 順に格納 → 0b1010 = 10
        assert_eq!(stream.read_bits(4), 0b1010);

        // 0b10101010 の bit4-7 = 0,1,0,1 → 0b1010 = 10
        assert_eq!(stream.read_bits(4), 0b1010);

        // 0b11001100 の bit0-3 = 0,0,1,1 → 0b1100 = 12
        assert_eq!(stream.read_bits(4), 0b1100);

        // 0b11001100 の bit4-7 = 0,0,1,1 → 0b1100 = 12
        assert_eq!(stream.read_bits(4), 0b1100);
    }

    #[test]
    fn test_bitstream_cursor() {
        let data = [0u8; 32];
        let mut stream = BitStream::new(&data);

        assert_eq!(stream.cursor(), 0);
        stream.read_bit();
        assert_eq!(stream.cursor(), 1);
        stream.read_bits(7);
        assert_eq!(stream.cursor(), 8);
        stream.read_bits(7);
        assert_eq!(stream.cursor(), 15);
    }

    #[test]
    fn test_packed_sfen_value_accessors() {
        let mut psv = PackedSfenValue::default();
        // score = 0x1234 at bytes 32-33
        psv.data[32] = 0x34;
        psv.data[33] = 0x12;
        // game_ply = 0x0056 at bytes 36-37
        psv.data[36] = 0x56;
        psv.data[37] = 0x00;
        // game_result = -1 at byte 38
        psv.data[38] = 0xFF; // -1 as i8

        assert_eq!(psv.score(), 0x1234);
        assert_eq!(psv.game_ply(), 0x0056);
        assert_eq!(psv.game_result(), -1);
    }

    #[test]
    fn test_huffman_decode_empty() {
        // 空マス: 0
        let data = [0b00000000u8; 32];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece, Piece::NONE);
        assert_eq!(stream.cursor(), 1);
    }

    #[test]
    fn test_huffman_decode_pawn() {
        // 先手歩: 01 (PAWN) + 0 (不成) + 0 (先手) = 0001
        // LSB first: bit0=1, bit1=0, bit2=0, bit3=0 → 0b0001
        let data = [0b00000001u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
    }

    #[test]
    fn test_huffman_decode_promoted_pawn() {
        // 後手と金: 01 (PAWN) + 1 (成り) + 1 (後手) = 1101
        // LSB first: bit0=1, bit1=0, bit2=1, bit3=1 → 0b1101
        let data = [0b00001101u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::ProPawn);
        assert_eq!(piece.color, Color::White);
    }

    #[test]
    fn test_huffman_decode_gold() {
        // 先手金: 01111 (GOLD) + 0 (先手) = 001111
        // LSB first: 1,1,1,1,0,0 → 0b001111
        let data = [0b00001111u8, 0u8, 0u8, 0u8];
        let mut stream = BitStream::new(&data);

        let piece = decode_board_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Gold);
        assert_eq!(piece.color, Color::Black);
    }

    #[test]
    fn test_bitstream_oob_protection() {
        // 2バイト = 16ビットのデータ
        let data = [0xFF, 0xFF];
        let mut stream = BitStream::new(&data);

        // 16ビット読み出し可能
        for _ in 0..16 {
            assert!(stream.can_read());
            stream.read_bit();
        }

        // 17ビット目以降は読み出し不可（false を返す）
        assert!(!stream.can_read());
        assert!(!stream.read_bit()); // OOB だが panic しない
        assert!(!stream.read_bit());
    }

    #[test]
    fn test_decode_hand_piece_pawn() {
        // 手駒の歩: 盤上歩の Huffman 符号 "01" から bit0 を除いた "0" + 成りbit(0) + 先後bit(0)
        // = 00 (2ビット)
        // LSB first: bit0=0, bit1=0 → 0b00
        let data = [0b00000000u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert!(!is_piecebox);
    }

    #[test]
    fn test_decode_hand_piece_gold() {
        // 手駒の金: 盤上金の Huffman 符号 "01111" から bit0 を除いた "0111" + 先後bit(0)
        // 0111 (code) = bit0=1, bit1=1, bit2=1, bit3=0
        // 先後bit = bit4 = 0 (Black)
        // バイト: 0b000_0_0111 = 0b00000111 = 7
        let data = [0b00000111u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Gold);
        assert_eq!(piece.color, Color::Black);
        assert!(!is_piecebox);
    }

    #[test]
    fn test_decode_hand_piece_piecebox_pawn() {
        // 駒箱の歩: "0" (歩の手駒符号) + 成りbit(1=駒箱) + 先後bit(0)
        // 0 (code) = bit0 = 0
        // 成りbit = bit1 = 1 (駒箱)
        // 先後bit = bit2 = 0 (Black)
        // バイト: 0b00000_0_1_0 = 0b00000010 = 2
        let data = [0b00000010u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Pawn);
        assert_eq!(piece.color, Color::Black);
        assert!(is_piecebox); // 駒箱フラグが立っている
    }

    #[test]
    fn test_decode_hand_piece_piecebox_rook() {
        // 駒箱の飛: 盤上飛の Huffman "111111" から bit0 除去 → "11111" + 成りbit(1=駒箱) + 先後bit(0)
        // 11111 (code) = bit0-4 = 1,1,1,1,1
        // 成りbit = bit5 = 1 (駒箱)
        // 先後bit = bit6 = 0 (Black)
        // バイト: 0b0_0_1_11111 = 0b00111111 = 63
        let data = [0b00111111u8; 32];
        let mut stream = BitStream::new(&data);

        let (piece, is_piecebox) = decode_hand_piece(&mut stream);
        assert_eq!(piece.piece_type, PieceType::Rook);
        assert_eq!(piece.color, Color::Black);
        assert!(is_piecebox);
    }

    #[test]
    fn test_hand_decode_cursor_256_boundary() {
        // 256ビット境界のテスト
        // 手駒ループは cursor < 256 でループするので、
        // 256ビットちょうどで終了することを確認

        let data = [0u8; 32];
        // 全て空マス符号 "0" で埋める（1ビット×256=256ビット）
        // これにより cursor が 256 に到達して終了

        let mut stream = BitStream::new(&data);

        // 256ビット読み出し
        for _ in 0..256 {
            stream.read_bit();
        }
        assert_eq!(stream.cursor(), 256);
        assert!(!stream.can_read()); // これ以上読めない

        // OOB アクセスしても panic しない
        assert!(!stream.read_bit());
    }
}
