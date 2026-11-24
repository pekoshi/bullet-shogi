use super::SparseInputType;

pub trait Factorises<T: SparseInputType>: SparseInputType<RequiredDataType = T::RequiredDataType> {
    fn derive_feature(&self, input: &T, feat: usize) -> Option<usize>;
}

#[derive(Clone, Copy, Default)]
pub struct Factorised<A: SparseInputType, B: Factorises<A>> {
    normal: A,
    factoriser: B,
    offset: usize,
}

impl<A: SparseInputType, B: Factorises<A>> Factorised<A, B> {
    pub fn from_parts(normal: A, factoriser: B) -> Self {
        let offset = factoriser.num_inputs();
        Self { normal, factoriser, offset }
    }
}

impl<A: SparseInputType, B: Factorises<A>> SparseInputType for Factorised<A, B> {
    type RequiredDataType = <A as SparseInputType>::RequiredDataType;

    fn num_inputs(&self) -> usize {
        self.normal.num_inputs() + self.factoriser.num_inputs()
    }

    fn max_active(&self) -> usize {
        2 * self.normal.max_active()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.normal.map_features(pos, |stm, ntm| {
            f(self.offset + stm, self.offset + ntm);

            let stm = self.factoriser.derive_feature(&self.normal, stm);
            let ntm = self.factoriser.derive_feature(&self.normal, ntm);

            match (stm, ntm) {
                (Some(stm), Some(ntm)) => {
                    assert!(stm < self.offset && ntm < self.offset, "Factoriser feature exceeded factoriser size!");
                    f(stm, ntm);
                }
                (None, None) => {}
                _ => panic!("One factorised feature existed but the other did not!"),
            }
        });
    }

    /// 非対称 emission (aux 連結入力等) を factorise 対応で転送する。
    ///
    /// inner の `map_features_split` が emit する各特徴を `offset` だけずらして
    /// 実特徴として emit し、 `derive_feature` が `Some` を返す特徴 (= HalfKA 部)
    /// にのみ factor virtual feature を追加 emit する。 aux 特徴 (= king 非依存の
    /// 生特徴) は `derive_feature` が `None` を返すため factor は付かない。
    ///
    /// 対称な inner (default `map_features_split` = `map_features` 経由) では
    /// emit 順・値とも従来の `map_features` 経路と完全一致する (= 既存 factorise
    /// 構成と bit-identical)。
    fn map_features_split<F: FnMut(Option<usize>, Option<usize>)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        self.normal.map_features_split(pos, |stm, ntm| {
            f(stm.map(|v| self.offset + v), ntm.map(|v| self.offset + v));

            let stm_fac = stm.and_then(|v| self.factoriser.derive_feature(&self.normal, v));
            let ntm_fac = ntm.and_then(|v| self.factoriser.derive_feature(&self.normal, v));

            if let Some(v) = stm_fac {
                assert!(v < self.offset, "Factoriser feature exceeded factoriser size!");
            }
            if let Some(v) = ntm_fac {
                assert!(v < self.offset, "Factoriser feature exceeded factoriser size!");
            }
            if stm_fac.is_some() || ntm_fac.is_some() {
                f(stm_fac, ntm_fac);
            }
        });
    }

    fn shorthand(&self) -> String {
        self.normal.shorthand()
    }

    // ------------------------------------------------------------------
    // 補助入力 (dense / sparse) の転送
    //
    // factoriser は main sparse 入力 (FT) だけを仮想特徴で拡張するもので、
    // aux dense (hand_count / pin) や aux sparse (Dual PSQT / PairShape /
    // KingDanger 等の head 入力) は別チャネルなので、 そのまま inner へ
    // delegate すれば良い。 これにより --factorise と各 head の併用が可能になる。
    // ------------------------------------------------------------------

    fn hand_count_dims(&self) -> Option<usize> {
        self.normal.hand_count_dims()
    }

    fn fill_hand_count(&self, pos: &Self::RequiredDataType, out: &mut [f32]) {
        self.normal.fill_hand_count(pos, out);
    }

    fn auxiliary_dense_dims(&self) -> Vec<(String, usize)> {
        self.normal.auxiliary_dense_dims()
    }

    fn fill_auxiliary_dense(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [f32]) {
        self.normal.fill_auxiliary_dense(name, pos, out);
    }

    fn auxiliary_sparse_dims(&self) -> Vec<(String, usize, usize)> {
        self.normal.auxiliary_sparse_dims()
    }

    fn fill_auxiliary_sparse(&self, name: &str, pos: &Self::RequiredDataType, out: &mut [i32]) -> usize {
        self.normal.fill_auxiliary_sparse(name, pos, out)
    }

    fn description(&self) -> String {
        format!("{}, factorised by {}", self.normal.description(), self.factoriser.description().to_lowercase())
    }

    fn is_factorised(&self) -> bool {
        true
    }

    fn merge_factoriser(&self, unmerged: Vec<f32>) -> Vec<f32> {
        let src_size = self.num_inputs();

        assert_eq!(unmerged.len() % src_size, 0);
        let layer_size = unmerged.len() / src_size;
        let offset = self.factoriser.num_inputs();

        (0..self.normal.num_inputs() * layer_size)
            .map(|elem| {
                let feat = elem / layer_size;
                let idx = elem % layer_size;
                let factoriser = self
                    .factoriser
                    .derive_feature(&self.normal, feat)
                    .map(|feat| unmerged[layer_size * feat + idx])
                    .unwrap_or(0.0);

                unmerged[layer_size * (feat + offset) + idx] + factoriser
            })
            .collect()
    }
}
