# bullet-shogi (p2shogi 版)

nodchip 氏の [bullet-shogi](https://github.com/nodchip/bullet-shogi) を fork し、将棋エンジン [p2shogi](https://github.com/pekoshi/p2shogi) の評価関数 (p2nnue) を学習するために改修したものです。

- 主な追加: 利き (threat) 入力、玉周りの補助入力 (KH)、LayerStack の bucket 選択、OPN (序盤用 net)、PSV の読み込みと bucket フィルタ、Ranger21 など。
- `crates/bullet_lib` は `rshogi-core` に依存します。ラム将棋 (SH11235/rshogi) 派生の [pekoshi/rshogi](https://github.com/pekoshi/rshogi) を、このリポジトリと同じ親ディレクトリに `rshogi` として置いてください。
- ライセンスは元の bullet-shogi に従います。

詳しい学習手順と機構の解説は [pekoshi/p2shogi の docs/](https://github.com/pekoshi/p2shogi/tree/master/docs) (`TRAINING_FROM_SCRATCH.md`、`CHANGES_FROM_UPSTREAM_trainer.md`) にあります。
