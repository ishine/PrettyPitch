# examples — 付属楽譜

`python -m svs.render examples/<name>.musicxml -o out.wav --spk_id 2` で合成できる。`example_score.json` は F0 モデル単体（`python -m infer`）の入力例で、楽譜ではなく音素列と音高の JSON。

| file | 曲 | 出典 |
|---|---|---|
| `kirakiraboshi.musicxml` / `.ust` | きらきら星 | Public domain melody |
| `kaeru_no_uta.musicxml` | かえるのうた | Traditional (German melody); public-domain Japanese lyrics. |
| `furusato.musicxml` | ふるさと | Music: Teiichi Okano (1878-1941). Lyrics: Tatsuyuki Takano (1876-1947). Public domain. |
| `momiji.musicxml` | もみじ | Music: Teiichi Okano (1878-1941). Lyrics: Tatsuyuki Takano (1876-1947). Public domain. |
| `haru_no_ogawa.musicxml` | 春の小川 | Music: Teiichi Okano (1878-1941). Lyrics: Tatsuyuki Takano (1876-1947). Public domain. |
| `ievan_polkka_{200,250,300}.musicxml` | Ievan polkka（早口テスト） | Traditional Finnish polka melody; public domain. 歌詞は意味を持たないスキャット音節で、この 3 ファイルはテンポだけが異なる。 |

`ievan_polkka_*` は早口の確認用。8 分音符が 150 / 120 / 100 ms になり、子音長モデルと圧縮規則が詰まった譜面でどう振る舞うかを聴ける。
