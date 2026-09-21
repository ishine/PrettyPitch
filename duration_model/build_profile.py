"""F0 シャードから子音長プロファイル（floor 等）を作る（score-free）。

slab は使わない。PrettyPitch と同じ preprocess のシャード（phoneme_ids / dur_sec）から
音素ごとの長さを集計する。

手法:
  子音 floor `cons_floor_sec` = 子音別 長さの **平均−1σ**（既定）を 5ms 丸め（n < min_n はベース値）。
  母音 floor `vowel_floor_sec` = 固定値 75ms（統計で決めない。--vowel_floor_ms で変更可）。
  **話者別の値はレシピ**（configs/recipes/<db>.yaml の `duration:`）から引き、`per_speaker` に
  母音 floor（vowel_floor_sec）・音素別の取り分の重み（phoneme_weight）・その DB の統計の取り方
  （cons_floor_stat / sd_cap / min_n）を反映する。
  先頭子音 floor `lead_cons_floor_sec` = 直前が無音の子音の p75（n ≥ min_lead_n, 5ms 丸め）。
  enabled はベース（既定プロファイル）から継承。

Usage:
    python -m duration_model.build_profile --data_dir data/oniku data/natsume \
        --out duration_model/profiles/<name>.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import phoneset as _ps_mod
from .profile import (ConsDurProfile, default_profile, from_dict, VOWEL_FLOOR_SEC,
                      _ratio_to_weight)

_RECIPE_DIR = Path(__file__).resolve().parent.parent / 'configs' / 'recipes'


def recipe_duration_settings(db: str) -> dict:
    """configs/recipes/<db>.yaml の `duration:` を返す（無ければ {}）。話者ごとの値はここが出どころ。"""
    p = _RECIPE_DIR / f'{db}.yaml'
    if not p.exists():
        return {}
    import yaml
    r = yaml.safe_load(open(p, encoding='utf-8')) or {}
    return dict(r.get('duration') or {})



def _collect(data_dirs: list[str], phoneset=None, spk_map=None, style_map=None):
    """シャードから子音長を集める。→ (cons, vowel, lead, per_key)

    per_key: 話者キー -> (cons, lead)。キーは "<spk>" と "<spk>:<style>" の両方を作る
        （どちらの粒度で floor を出すかは build_profile 側が n で決める）。
        spk_map が無ければ per_key は空 = 従来どおり全話者混合のみ。
    """
    ps = phoneset or _ps_mod.active()
    cons: dict[str, list[float]] = {}   # ph -> [ms]（全位置）
    vowel: list[float] = []             # [ms]（有声母音）
    lead: dict[str, list[float]] = {}   # ph -> [ms]（直前が無音の子音）
    per_key: dict[str, tuple[dict, dict]] = {}
    key_dbs: dict[str, list[str]] = {}    # 話者キー -> そのキーに寄与した DB 名（レシピ設定を引く）
    for d in data_dirs:
        d = Path(d)
        db = os.path.basename(os.path.normpath(str(d)))
        keys = []
        if spk_map and db in spk_map:
            s = int(spk_map[db])
            keys.append(str(s))
            if style_map and db in style_map:
                keys.append(f'{s}:{int(style_map[db])}')
        for k in keys:
            per_key.setdefault(k, ({}, {}))
            key_dbs.setdefault(k, []).append(db)
        meta = json.load(open(d / 'metadata.json', encoding='utf-8'))
        id2ph = {v: k for k, v in meta['phoneme_vocab'].items()}
        z = np.load(d / 'shard.npz')
        for name in meta['phrases']:
            ph_ids = z[f'{name}|phoneme_ids']
            dur = z[f'{name}|dur_sec']
            ph = [id2ph.get(int(p), 'pau') for p in ph_ids]
            for i, p in enumerate(ph):
                ms = float(dur[i]) * 1000
                if p in ps.cons_set:
                    is_lead = (i == 0 or ps.is_pau(ph[i - 1]))
                    cons.setdefault(p, []).append(ms)
                    if is_lead:
                        lead.setdefault(p, []).append(ms)
                    for k in keys:
                        kc, kl = per_key[k]
                        kc.setdefault(p, []).append(ms)
                        if is_lead:
                            kl.setdefault(p, []).append(ms)
                elif ps.normalize_ph(p) in ps.vowels:
                    vowel.append(ms)
    _collect.key_dbs = key_dbs          # build_profile が読む
    return cons, vowel, lead, per_key


def _round5_sec(ms: float) -> float:
    return round(round(ms / 5) * 5 / 1000.0, 4)


# 5ms 丸めで 0 に落ちた子音を「無」にしないための下限。統計が暴れたときの保険で、
# 通常はここに当たらない（当たったら n と σ を疑うこと）。
_MIN_CONS_FLOOR_SEC = 0.005


def build_profile(data_dirs: list[str], base: ConsDurProfile | None = None,
                  min_n: int = 30, min_lead_n: int = 10,
                  cons_stat: str = 'mean-sd', cons_pctl: float = 5.0,
                  sd_cap: float = 0.6,
                  vowel_floor_sec: float = VOWEL_FLOOR_SEC, phoneset=None,
                  spk_map=None, style_map=None) -> tuple[ConsDurProfile, dict]:
    """シャードからプロファイルを構築。返り値 = (profile, 統計サマリ)。

    cons_stat        : 子音 floor の統計量。'mean-sd'（既定）= 平均−1σ / 'pctl' = 百分位。
    cons_pctl        : cons_stat='pctl' のときの百分位。
    sd_cap           : σ を平均の何倍で頭打ちにするか（既定 0.6 = floor は平均の 40% 未満にしない）。
    vowel_floor_sec  : 母音 floor（固定値。統計では決めない。既定 75ms）。
    spk_map/style_map: DB名 -> id。渡すと **話者別 floor**（profile.per_speaker）も作る。
        統計が min_n に足りない音素はそのキーに書かないので、推論時は全話者共通値へ落ちる
        （3 段フォールバック: "<spk>:<style>" → "<spk>" → 全話者共通）。

    **平均−1σ を既定にしている理由。** p5 は少ないサンプルだと不安定（n=20 で標準偏差
    ~10ms、実質「観測最小値」）で、破裂音では 20〜40ms 低く外れて早口で子音が潰れる原因になる。
    平均−1σ は全サンプルを使うので低 n でも安定し、話者別の floor でも既定 min_n=30 で足りる。
    """
    ps = phoneset or _ps_mod.active()
    prof = base or default_profile(ps)
    cons, vowel, lead, per_key = _collect(data_dirs, ps, spk_map, style_map)

    def _stat(v, cons_stat=cons_stat, cons_pctl=cons_pctl, sd_cap=sd_cap) -> float:
        """子音 floor の統計量 [ms]。5ms 丸めは呼び出し側。負にはしない。

        既定は **平均 − min(σ, sd_cap × 平均)**。素の平均−1σ は裾の重い子音で壊れる
        （`p` のように σ が平均に近い子音では floor が平均の 2 割まで落ちる）。
        σ を平均の一定割合で頭打ちにすると「平均の (1-sd_cap) 未満にはしない」になる。
        """
        a = np.asarray(v, dtype=float)
        if cons_stat == 'pctl':
            return float(np.percentile(a, cons_pctl))
        mu = float(a.mean())
        return max(0.0, mu - min(float(a.std()), sd_cap * mu))

    def _floors(c_src: dict, l_src: dict, base_cons: dict | None, st: dict | None = None):
        """st = レシピ `duration:` の上書き（min_n / min_lead_n / cons_floor_stat / cons_pctl / sd_cap）。"""
        st = st or {}
        mn = int(st.get('min_n', min_n)); ml = int(st.get('min_lead_n', min_lead_n))
        kw = dict(cons_stat=str(st.get('cons_floor_stat', cons_stat)),
                  cons_pctl=float(st.get('cons_pctl', cons_pctl)), sd_cap=float(st.get('sd_cap', sd_cap)))
        cf = dict(base_cons) if base_cons is not None else {}
        for c in ps.consonants:
            v = c_src.get(c, [])
            if len(v) >= mn:
                cf[c] = max(_MIN_CONS_FLOOR_SEC, _round5_sec(_stat(v, **kw)))
        lf = {c: _round5_sec(float(np.percentile(vs, 75)))
              for c, vs in l_src.items() if len(vs) >= ml}
        return cf, lf

    prof.cons_floor, prof.lead_cons_floor = _floors(cons, lead, prof.cons_floor)

    # 母音 floor は固定値（早口時の最終下限）。統計で決めない。
    prof.vowel_floor = float(vowel_floor_sec)

    # 話者別は「統計が足りた音素だけ」書く（base を継承しない = 落ちたら共通値へ）。
    # 話者ごとの設定（母音 floor・配分比・統計の取り方）は **レシピの `duration:`** から引く。
    # 同じ話者キーに複数 DB が寄与するとき（ritsu の 3 トーン → "2"）は先頭 DB の設定を使い、
    # DB 間で食い違えば警告する。
    key_dbs = getattr(_collect, 'key_dbs', {})
    per_spk: dict[str, dict] = {}
    recipe_used: dict[str, dict] = {}
    for k, (kc, kl) in sorted(per_key.items()):
        dbs = key_dbs.get(k, [])
        sts = [(db, recipe_duration_settings(db)) for db in dbs]
        st = sts[0][1] if sts else {}
        for db, s2 in sts[1:]:
            if s2 != st:
                print(f'  [warn] 話者キー {k}: {sts[0][0]} と {db} の duration: 設定が違う → {sts[0][0]} を採用')
        cf, lf = _floors(kc, kl, None, st)
        e: dict = {}
        if cf:
            e['cons_floor'] = cf
        if lf:
            e['lead_cons_floor'] = lf
        if st.get('vowel_floor_sec') is not None:
            e['vowel_floor'] = float(st['vowel_floor_sec'])
        # 音素別の重み。表（dict）なら phoneme_weight、スカラーなら phoneme_weight_default。
        wt = st.get('phoneme_weight')
        if isinstance(wt, dict):
            e['phoneme_weight'] = {c: float(x) for c, x in wt.items()}
        if st.get('phoneme_weight_default') is not None:
            e['phoneme_weight_default'] = float(st['phoneme_weight_default'])
        if 'phoneme_weight' not in e and 'phoneme_weight_default' not in e:
            legacy = st.get('cons_weight', st.get('short_note_ratio'))   # 旧キー
            conv = (float if 'cons_weight' in st else _ratio_to_weight)
            if isinstance(legacy, dict):
                e['phoneme_weight'] = {c: conv(float(x)) for c, x in legacy.items()}
            elif legacy is not None:
                e['phoneme_weight_default'] = conv(float(legacy))
        if e:
            per_spk[k] = e
            recipe_used[k] = {kk: st[kk] for kk in st}
    prof.per_speaker = per_spk

    summary = {
        'n_cons': {c: len(cons.get(c, [])) for c in ps.consonants},
        'n_vowel': len(vowel),
        'n_lead': {c: len(lead.get(c, [])) for c in lead},
        'per_speaker': {k: {'n_cons_kept': len(v.get('cons_floor', {})),
                            'n_lead_kept': len(v.get('lead_cons_floor', {})),
                            'recipe': recipe_used.get(k, {})}
                        for k, v in per_spk.items()},
    }
    return prof, summary


def main() -> None:
    ap = argparse.ArgumentParser(description='F0 シャードから子音長プロファイルを作る')
    ap.add_argument('--data_dir', nargs='+', required=True,
                    help='PrettyPitch preprocess のシャードディレクトリ（複数可）')
    ap.add_argument('--out', required=True, help='出力プロファイル JSON パス')
    ap.add_argument('--base', default=None, help='継承元プロファイル JSON（enabled / 既存 floor を引き継ぐ）')
    ap.add_argument('--lang', default='ja',
                   help="言語パック名またはパス（dict/<lang>.yaml）")
    ap.add_argument('--phonemes', default=None, help='音素表を上書き（言語パックの phonemes:）')
    ap.add_argument('--nucleus', default=None, help='核音素表を上書き（言語パックの nucleus:）')
    ap.add_argument('--config', default=None,
                    help='spk_map / style_map を読む学習 config（渡すと話者別 floor も作る）')
    ap.add_argument('--min_n', type=int, default=30,
                    help='子音 floor を統計で決める最小 n（既定 30。平均−1σ なら n=32 でも実用値になる）')
    ap.add_argument('--min_lead_n', type=int, default=10, help='先頭 floor を書く最小 n')
    ap.add_argument('--cons_stat', choices=['mean-sd', 'pctl'], default='mean-sd',
                    help='子音 floor の統計量（既定 mean-sd = 平均−1σ）')
    ap.add_argument('--cons_pctl', type=float, default=5.0,
                    help='--cons_stat pctl のときの百分位（既定 p5）')
    ap.add_argument('--sd_cap', type=float, default=0.6,
                    help='σ を平均の何倍で頭打ちにするか（既定 0.6）')
    ap.add_argument('--vowel_floor_ms', type=float, default=VOWEL_FLOOR_SEC * 1000,
                    help='母音 floor（固定・統計で決めない）。既定 75ms')
    a = ap.parse_args()

    ps = _ps_mod.set_default(_ps_mod.load(a.lang, a.phonemes, a.nucleus))
    print(f'言語パック: {ps.lang}  子音{ps.n_consonants}')

    base = None
    if a.base:
        base = from_dict(json.loads(Path(a.base).read_text(encoding='utf-8')))

    spk_map = style_map = None
    if a.config:
        import yaml
        cfg = yaml.safe_load(open(a.config, encoding='utf-8')) or {}
        dcfg = cfg.get('data', {}) or {}
        spk_map = dcfg.get('spk_map') or None
        style_map = dcfg.get('style_map') or None
        print(f'話者別 floor: spk_map {len(spk_map or {})} 件 / style_map {len(style_map or {})} 件')

    if not spk_map:
        # レシピの `duration:` は per_speaker を作る経路にしか繋がっていない。spk_map が
        # 無いと黙って落ちるので、書いてあるのに効かないことをここで知らせる。
        ignored = [os.path.basename(os.path.normpath(str(d))) for d in a.data_dir
                   if recipe_duration_settings(os.path.basename(os.path.normpath(str(d))))]
        if ignored:
            print(f'  [warn] --config が無いのでレシピの duration: は無視されます: {", ".join(ignored)}')

    prof, summary = build_profile(a.data_dir, base, a.min_n, a.min_lead_n,
                                  cons_stat=a.cons_stat, cons_pctl=a.cons_pctl, sd_cap=a.sd_cap,
                                  vowel_floor_sec=a.vowel_floor_ms / 1000.0,
                                  spk_map=spk_map, style_map=style_map)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prof.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    n_written = sum(1 for c in ps.consonants if summary['n_cons'][c] >= a.min_n)
    print(f'wrote {a.out}')
    print(f'  母音 floor = {prof.vowel_floor*1000:.0f}ms (固定・統計不使用)')
    _lab = '平均−1σ' if a.cons_stat == 'mean-sd' else f'p{a.cons_pctl:g}'
    print(f'  子音 floor = {_lab} 統計・更新 {n_written}/{len(ps.consonants)} 子音（残りはベース値）')
    print(f'  先頭 floor = {len(prof.lead_cons_floor)} 子音')
    if summary['per_speaker']:
        print(f'  話者別 floor（n >= {a.min_n} の音素のみ。足りない分は全話者共通へ落ちる）:')
        for k, v in sorted(summary['per_speaker'].items()):
            rc = v.get('recipe') or {}
            print(f'    {k:>6}: 子音 {v["n_cons_kept"]:2d}/{len(ps.consonants)}  先頭 {v["n_lead_kept"]:2d}'
                  + (f'  レシピ duration: {rc}' if rc else '  （レシピ duration: 無し → 共通既定）'))


if __name__ == '__main__':
    main()
