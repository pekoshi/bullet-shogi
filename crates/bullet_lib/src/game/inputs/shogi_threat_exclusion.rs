//! Threat pair 除外 profile (runtime 選択)
//!
//! 学習・評価時に CLI オプションで profile を指定する。
//! rshogi 側は compile-time (feature flag) で同一ロジックを実装。
//!
//! ## Profile 一覧
//!
//! | id | CLI 値 | 除外内容 |
//! |----|--------|---------|
//! | 0 | `full` | なし (Baseline) |
//! | 1 | `same-class` | 同種ペア全除外 |
//! | 2 | `same-class-major-pawn` | 同種 + 大駒→歩除外 |
//! | 3 | `no-pawn-target` | 被攻撃側が歩のペアを全除外 (NPS 削減狙い) |
//! | 10 | `cross-side` | cross-side 異種ペアのみ (増やすアプローチ) |
//!
//! 仕様: rshogi `docs/threat_spec.md` Exclusion profiles セクション

/// Threat pair 除外 profile
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreatProfile {
    /// Profile 0: 全 pair (Baseline)
    Full,
    /// Profile 1: 同種ペア全除外
    SameClass,
    /// Profile 2: 同種 + 大駒→歩除外
    SameClassMajorPawn,
    /// Profile 3: 被攻撃側 (target) が歩のペアを全除外
    NoPawnTarget,
    /// Profile 10: cross-side 異種ペアのみ
    CrossSide,
    /// Profile 11: enemy attacker -> friend target, same-class pairs excluded.
    EnemyOnly,
    /// Profile 12: profile 11と同じ特徴空間で、評価時に各視点の代表8行だけを使用。
    EnemyOnlyRep8,
    /// Profile 13: EnemyOnly features through a compact factor branch.
    EnemyOnlyFactor16,
    /// Profile 14: EnemyOnly plus same-class major-piece attacks for a direct bucketed branch.
    EnemyOnlyMajorSameDirect,
    /// Profile 15: enemy attacker -> friendly target with King as an independent 10th class.
    EnemyOnlyDirect15,
    /// Profile 16: Full の pair 空間に「被攻撃駒にひもが無いか (h)」を足した hanging-split。
    /// 敵味方をまたぐ pair だけ h=1 用の別行を持つ (同色 pair は攻め駒自身がひもなので h=0 に縮退)。
    /// 除外規則は Full と同一。 次元は 216,720 (base) + 108,360 (hang) = 325,080。
    /// 設計 = C:/project/p2shogi/runs/prereg/DESIGN-20260924-THREAT-DEFENDED.md
    FullHangSplit,
    /// Profile 17: enemy attacker -> friendly target, ALL classes (same-class pairs kept).
    /// 2026-11-05 partial threat (P2_THREAT_PARTIAL=2 と同じ集合): 各視点で `as==1 && ds==0` の対だけ。
    /// 次元 54,180 = full の 1/4。 設計 runs/threat_tax/partial_20261104/。
    EnemyOnlyAll,
}

impl ThreatProfile {
    /// CLI 文字列から変換
    pub fn from_cli(s: &str) -> Option<Self> {
        match s {
            "full" => Some(Self::Full),
            "same-class" => Some(Self::SameClass),
            "same-class-major-pawn" => Some(Self::SameClassMajorPawn),
            "no-pawn-target" => Some(Self::NoPawnTarget),
            "cross-side" => Some(Self::CrossSide),
            "enemy-only" => Some(Self::EnemyOnly),
            "enemy-only-rep8" => Some(Self::EnemyOnlyRep8),
            "enemy-only-factor16" => Some(Self::EnemyOnlyFactor16),
            "enemy-only-major-same-direct" => Some(Self::EnemyOnlyMajorSameDirect),
            "enemy-only-direct15" => Some(Self::EnemyOnlyDirect15),
            "full-hang-split" => Some(Self::FullHangSplit),
            "enemy-only-all" => Some(Self::EnemyOnlyAll),
            _ => None,
        }
    }

    /// quantised.bin に書き込む profile ID
    pub fn profile_id(self) -> u32 {
        match self {
            Self::Full => 0,
            Self::SameClass => 1,
            Self::SameClassMajorPawn => 2,
            Self::NoPawnTarget => 3,
            Self::CrossSide => 10,
            Self::EnemyOnly => 11,
            Self::EnemyOnlyRep8 => 12,
            Self::EnemyOnlyFactor16 => 13,
            Self::EnemyOnlyMajorSameDirect => 14,
            Self::EnemyOnlyDirect15 => 15,
            Self::FullHangSplit => 16,
            Self::EnemyOnlyAll => 17,
        }
    }

    /// hanging-split (profile 16) か。 true のとき threat の行が base / hang の 2 段になる。
    pub const fn is_hang_split(self) -> bool {
        matches!(self, Self::FullHangSplit)
    }

    /// pair を除外すべきかどうか判定する
    ///
    /// rshogi の `threat_exclusion::is_excluded` と同一ロジック。
    ///
    /// # 引数
    /// - `as_`: attacker side (0=friend, 1=enemy)
    /// - `ac`: attacker class index (0..8)
    /// - `ds`: attacked side (0=friend, 1=enemy)
    /// - `dc`: attacked class index (0..8)
    pub fn is_excluded(self, as_: usize, ac: usize, ds: usize, dc: usize) -> bool {
        match self {
            Self::Full => false,
            Self::SameClass => ac == dc,
            Self::SameClassMajorPawn => ac == dc || (ac >= 5 && dc == 0),
            // 被攻撃側 class が Pawn (dc == 0) のペアを attacker に関わらず全除外
            Self::NoPawnTarget => dc == 0,
            Self::CrossSide => as_ == ds || ac == dc,
            Self::EnemyOnly | Self::EnemyOnlyRep8 | Self::EnemyOnlyFactor16 => !(as_ == 1 && ds == 0) || ac == dc,
            // Keep EnemyOnly's one-sided material threats.  Re-admit only same-class
            // Bishop/Rook/Horse/Dragon attacks: these encode slider batteries and
            // exchange pressure that the King-zone-only Dual PSQT does not model.
            Self::EnemyOnlyMajorSameDirect => !(as_ == 1 && ds == 0) || (ac == dc && ac < 5),
            Self::EnemyOnlyDirect15 => !(as_ == 1 && ds == 0),
            // profile 16 は除外規則が Full と同一 (ひもは別の軸なので pair の除外には関与しない)。
            Self::FullHangSplit => false,
            // profile 17: 相手の駒が自分の駒に当たっている対だけ (同クラス対も残す)。
            Self::EnemyOnlyAll => !(as_ == 1 && ds == 0),
        }
    }

    /// 利用可能な profile 名の一覧（ヘルプ表示用）
    pub fn available() -> &'static str {
        "full, same-class, same-class-major-pawn, no-pawn-target, cross-side, enemy-only, enemy-only-rep8, enemy-only-factor16, enemy-only-major-same-direct, enemy-only-direct15, full-hang-split, enemy-only-all"
    }
}

impl std::fmt::Display for ThreatProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FullHangSplit => write!(f, "full-hang-split"),
            Self::Full => write!(f, "full"),
            Self::SameClass => write!(f, "same-class"),
            Self::SameClassMajorPawn => write!(f, "same-class-major-pawn"),
            Self::NoPawnTarget => write!(f, "no-pawn-target"),
            Self::CrossSide => write!(f, "cross-side"),
            Self::EnemyOnly => write!(f, "enemy-only"),
            Self::EnemyOnlyRep8 => write!(f, "enemy-only-rep8"),
            Self::EnemyOnlyFactor16 => write!(f, "enemy-only-factor16"),
            Self::EnemyOnlyMajorSameDirect => write!(f, "enemy-only-major-same-direct"),
            Self::EnemyOnlyDirect15 => write!(f, "enemy-only-direct15"),
            Self::EnemyOnlyAll => write!(f, "enemy-only-all"),
        }
    }
}
