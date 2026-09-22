# guide — 資料

**English**: [README.en.md](README.en.md)

PrettyPitch の仕組みを説明する資料をここにまとめています。図は `fig/` にあります。

## 全体像

| 資料 | 内容 |
|---|---|
| [svs_pipeline.md](svs_pipeline.md) | 楽譜から波形までの全体構成。どのモデルが何を担い、何が学習で何が規則か |

## F0

| 資料 | 内容 |
|---|---|
| [f0_model.md](f0_model.md) | F0 モデルの構成。baseline・plain・vibrato の 3 層と v/uv、条件付けの流れ |
| [representative_pitch.md](representative_pitch.md) | F0 代表値の推定。楽譜の代わりになる連続 MIDI 値をどう決めるか。学習のラベル作りと、生成時のピッチ補正の目標値の両方に使う |
| [vibrato_separation.md](vibrato_separation.md) | 残差から plain と vibrato を分ける手順。ソフトクリップ・帯域・ゲイン |
| [pitch_correction.md](pitch_correction.md) | 生成時のピッチ補正 2 段の原理 |

## ツール

| 資料 | 内容 |
|---|---|
| [tuner.md](tuner.md) | ビブラート抽出と F0 代表値推定のハイパーパラメータを調整する Web アプリの使い方 |

## 持続長

| 資料 | 内容 |
|---|---|
| [duration_model.md](duration_model.md) | 子音長モデルと後処理の詳細 |
