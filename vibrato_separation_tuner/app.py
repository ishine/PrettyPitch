"""vibrato_separation_tuner — interactive tuning app for PrettyPitch's vibrato separation.

Reads preprocessed datasets (`data/<db>/shard.npz` + `metadata.json`) and shows how the
plain/vibrato decomposition reacts to the hyperparameters, phrase by phrase, as cards
(full F0 + plain overlay on top, vibrato + the zeroing-gain curve below, phoneme
boundaries marked). The decomposition runs through the exact same functions the training
dataloader uses (`prettypitch.decomp`), so what you see here is what the model will be
trained on. Once you like a setting, copy the values into `configs/ja_example.yaml`
(`decomp:` section — per-database overrides go under `decomp.per_db`). No re-preprocessing
is ever needed: the shards store only raw phonemes/F0/uv.

    pip install -e .[tuner]
    python vibrato_separation_tuner/app.py [--port 8017]   # then open http://127.0.0.1:8017
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent          # PrettyPitch repo root
if str(_REPO) not in sys.path:
    sys.path.append(str(_REPO))

from prettypitch.decomp import (make_vibrato_fir, compress_residual, extract_vibrato,  # noqa: E402
                                make_vibrato_gain, filtfilt_min_len)
from prettypitch.baseline import build_baseline, phoneme_frame_bounds                  # noqa: E402
# 代表値は **本体** の nucleus_values をそのまま使う（tuner が見せるものは学習ローダーが
# 計算するものと同一）。
from prettypitch.pitch import hz_to_midi, nucleus_values                               # noqa: E402
from preprocess.vocab import load_nucleus                                              # noqa: E402

from fastapi import FastAPI, HTTPException                                             # noqa: E402
from fastapi.responses import FileResponse                                             # noqa: E402
from pydantic import BaseModel, ConfigDict, Field                                      # noqa: E402

app = FastAPI(title='vibrato_separation_tuner')

_STATIC = Path(__file__).resolve().parent / 'static'
_NUCLEUS_NAMES = None       # None = 各DBの音素 vocab から言語パックを自動判定; --nucleus で固定


@lru_cache(maxsize=8)
def _packs() -> tuple:
    """Every bundled language pack, as (lang, phoneme set, nucleus set)."""
    from duration_model import phoneset as _ps_mod
    out = []
    for y in sorted((_REPO / 'dict').glob('*.yaml')):
        try:
            ps = _ps_mod.load(y.stem)
        except Exception:
            continue
        out.append((y.stem, frozenset(ps.phonemes), frozenset(ps.nuclei)))
    return tuple(out)


@lru_cache(maxsize=64)
def _nucleus_for(vocab_key: frozenset) -> tuple:
    """Pick the language pack whose phoneme set best covers this database's vocab.

    The vocab is the fingerprint: a Japanese database covers dict/ja.yaml 100% and
    dict/en.yaml 38%, and shares ZERO nucleus names with it (and vice versa). So the
    right pack is unambiguous and nothing has to be declared per dataset — the shards
    carry no language field, and some datasets have no recipe at all.

    Returns (lang, nucleus names). --nucleus overrides the whole thing.
    """
    if _NUCLEUS_NAMES is not None:
        return ('(--nucleus)', frozenset(_NUCLEUS_NAMES))
    best = None
    for lang, phs, nuc in _packs():
        hit = len(vocab_key & nuc)
        cov = len(vocab_key & phs) / max(len(vocab_key), 1)
        if hit and (best is None or (hit, cov) > (best[1], best[2])):
            best = (lang, hit, cov, nuc)
    if best is None:
        return (None, frozenset())
    return (best[0], best[3])


LOG2_440 = float(np.log2(440.0))
MAX_POINTS = 1600      # per-phrase decimation cap for display arrays

_shards: dict = {}     # path -> NpzFile (cached across requests)
_metas: dict = {}      # path -> dict


def _meta(db_dir: Path) -> dict:
    key = str(db_dir / 'metadata.json')
    if key not in _metas:
        with open(key, encoding='utf-8') as f:
            _metas[key] = json.load(f)
    return _metas[key]


def _shard(db_dir: Path):
    key = str(db_dir / 'shard.npz')
    if key not in _shards:
        _shards[key] = np.load(key)
    return _shards[key]


def _song_of(stem: str) -> str:
    return re.sub(r'_\d{4,}$', '', stem)


@lru_cache(maxsize=32)
def _fir(frame_rate: float, lo: float, hi: float, time_len_sec: float):
    return make_vibrato_fir(frame_rate, band=(lo, hi), time_len_sec=time_len_sec)


# ── API ──────────────────────────────────────────────────────────────────────

class Params(BaseModel):
    """Vibrato-extraction hyperparameters (mirror of configs/ja_example.yaml `decomp:`).

    These defaults are the values the shipped config uses. They are transcribed by hand:
    nothing reads the YAML from here, so when `decomp:` changes, change this too.
    """
    band_lo: float = 4.5
    band_hi: float = 7.5
    fir_time_sec: float = 0.3
    cons_margin_pre_sec: float = 0.05
    cons_margin_post_sec: float = 0.10
    boundary_margin_pre_sec: float = 0.05
    boundary_margin_post_sec: float = 0.10
    gain_sigma_sec: float = 0.05
    boundary_mode: str = 'all'            # all | nonnucleus | none
    limit_margin_to_adjacent: bool = True
    min_open_sec: float = 0.10
    comp_method: str = 'tanh'             # tanh | log | linear
    comp_thresh_st: float = 2.0
    comp_param: float = 0.5
    baseline_sigma_sec: float = 0.02


class PitchParams(BaseModel):
    """Representative-value hyperparameters (mirror of configs/ja_example.yaml `pitch:`).

    Hand-transcribed like Params above: nothing here reads the YAML.

    extra='forbid' は意図的。ブラウザは index.html を毎回ディスクから読むが、この
    モジュールは**起動時に一度だけ**読まれる。既定の extra='ignore' だと、新しいつまみを
    足した後に古いサーバへ繋いだとき、そのつまみが黙って捨てられて「UI に出ているのに
    何も起きない」になる。ここで弾けば 422 が返り、再起動が必要だと分かる。
    """
    model_config = ConfigDict(extra='forbid')
    n_iter: int = 2
    sigma_st: float = 1.0
    sg_window_ms: float = 55.0
    min_voiced: int = 3
    devoiced_fill: str = 'interp'          # interp | neighbor | none
    edge_window: str = 'tukey'             # tukey | hann | asym | asym_adaptive
    edge_lead_frac: float = 0.8            # asym only
    edge_trail_frac: float = 0.3           # asym only
    tukey_alpha: float = 0.5

    # ── 安定度重み w_d の定義（prettypitch/pitch.py と同じ。説明はそちら）───────────
    stability: str = 'blaauw'              # blaauw（Blaauw 2017 原式）| slope（SG の 1 階微分）
    slope_k: float = 0.5                   # w_d = 1/min(1 + k*|df/dt|, cap)
    slope_cap: float = 15.0
    # ── 短いノートの山検出（同上）────────────────────────────────────────────
    # 触らないための条件（ゲート）が要点: 核が crest_max_ms 未満、頂点 − 核内中央値 ≥
    # crest_depth_st、極大の卓立 ≥ crest_prom_st。全部満たす核だけ頂点付近の平均に置き換える。
    crest_max_ms: float = 0.0              # 0 = 山検出オフ
    crest_depth_st: float = 1.0            # 頂点 − 中央値 の下限 [半音]
    crest_prom_st: float = 1.0             # 極大と認める卓立の下限 [半音]
    crest_sg_ms: float = 25.0              # 頂点探索の平滑窓 [ms]（短い山を潰さないよう短め）
    # どんぴしゃの頂点は行き過ぎることが多いので、少し下げる 2 通り
    crest_band_st: float = 0.0             # 頂点−band 以内の連続領域を平均（0 = 頂点そのもの）
    crest_offset_cent: float = 0.0         # 最後に引く固定量 [cent]
    edge_adapt_lo_ms: float = 70.0         # 'asym_adaptive' のしきい値（本体の機能）
    edge_adapt_hi_ms: float = 150.0


class PitchReq(BaseModel):
    root: str
    db: str
    params: PitchParams = Field(default_factory=PitchParams)
    query: str = ''
    sort: str = 'name'
    seed: int = 0
    page: int = 0
    page_size: int = 30


class ScanReq(BaseModel):
    root: str


class RespReq(BaseModel):
    params: Params = Field(default_factory=Params)
    frame_rate: float = 172.265625
    f_max: float = 0.0            # 0 = auto (band_hi + 4 Hz, at least 12)


class DecompReq(BaseModel):
    root: str
    db: str                        # 1 データベースずつ。混ぜると言語パックも frame_rate も一意に決まらない
    params: Params = Field(default_factory=Params)
    sort: str = 'name'                    # name | shuffle
    seed: int = 0
    query: str = ''
    page: int = 0
    page_size: int = 30


@app.get('/')
async def index():
    return FileResponse(_STATIC / 'index.html')


@app.get('/config')
async def config():
    return {'default_root': str(_REPO / 'data')}


@app.post('/filter_response')
async def filter_response(req: RespReq):
    """Effective gain of the current bandpass: |H(f)|^2, because filtfilt applies the
    FIR forward AND backward (a one-way gain of 0.85 becomes 0.72)."""
    from scipy.signal import freqz
    p = req.params
    fr = req.frame_rate
    fmax = req.f_max if req.f_max > 0 else max(12.0, p.band_hi + 4.0)
    h = _fir(fr, p.band_lo, p.band_hi, p.fir_time_sec)
    w, H = freqz(h, worN=4096, fs=fr)
    sel = w <= fmax
    f, g = w[sel], np.abs(H[sel]) ** 2
    label = f'FIR filtfilt effective |H|^2 ({len(h)} taps)'
    return {'freqs': np.round(f, 3).tolist(), 'gain': np.round(g, 4).tolist(),
            'band_lo': p.band_lo, 'band_hi': p.band_hi, 'label': label}


@app.post('/scan')
async def scan(req: ScanReq):
    root = Path(req.root).expanduser()
    if not root.is_dir():
        raise HTTPException(400, f'folder not found: {root}')
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / 'shard.npz').exists() and (d / 'metadata.json').exists():
            m = _meta(d)
            fr = float(m['frame_rate'])
            hours = sum(m['phrases'].values()) / fr / 3600.0
            out.append({'db': d.name, 'n_phrases': int(m['n_phrases']),
                        'hours': round(hours, 2)})
    return {'dbs': out}


@app.post('/decompose')
async def decompose(req: DecompReq):
    root = Path(req.root).expanduser()
    p = req.params
    if not (0 < p.band_lo < p.band_hi):
        raise HTTPException(400, 'band_lo must be < band_hi')

    # phrase list: one DB -> substring filter -> sort -> page
    db = req.db
    d0 = root / db
    if not (d0 / 'shard.npz').exists():
        raise HTTPException(400, f'shard.npz not found: {d0}')
    m0 = _meta(d0)
    metas = {db: m0}
    lang0, _ = _nucleus_for(frozenset(m0['phoneme_vocab']))
    q = req.query.strip().lower()
    entries: list[tuple[str, str]] = [(db, n) for n in m0['phrases']
                                      if not q or q in n.lower()]
    if req.sort == 'shuffle':
        rng = random.Random(req.seed)
        rng.shuffle(entries)
    else:
        entries.sort(key=lambda e: (e[1], e[0]))
    total = len(entries)
    lo = max(0, req.page) * req.page_size
    page_entries = entries[lo:lo + req.page_size]

    items = []
    for db, name in page_entries:
        d = root / db
        m = metas[db]
        fr = float(m['frame_rate'])
        vocab = m['phoneme_vocab']
        id2ph = {v: k for k, v in vocab.items()}
        lang, nuc_names = _nucleus_for(frozenset(vocab))
        nuc_ids = frozenset(vocab[n] for n in nuc_names if n in vocab)
        if not nuc_ids:
            raise HTTPException(400, f"{db}: no language pack in dict/ shares a nucleus "
                                     f"phoneme with this database's vocab "
                                     f"({len(vocab)} phonemes, e.g. "
                                     f"{', '.join(sorted(vocab)[:6])}). Add a pack, or "
                                     f"pass --nucleus FILE to override.")

        z = _shard(d)
        ph_ids = z[f'{name}|phoneme_ids'].astype(np.int64)
        dur_sec = z[f'{name}|dur_sec'].astype(np.float32)
        f0 = z[f'{name}|f0_interp'].astype(np.float32)
        uv = z[f'{name}|uv'].astype(np.float32)
        T = len(f0)
        fb = phoneme_frame_bounds(dur_sec, fr, T)

        # ── same pipeline as the training dataloader (dataset.py) ──
        values = nucleus_values(ph_ids, dur_sec, f0, uv, fr, nuc_ids)
        baseline = build_baseline(values, dur_sec, T, fr, sigma_sec=p.baseline_sigma_sec)
        if baseline is None:
            baseline = np.full(T, float(np.median(np.log2(np.maximum(f0, 1.0)))),
                               dtype=np.float32)
        residual = (np.log2(np.maximum(f0, 1.0)) - baseline).astype(np.float32)
        resid_c = compress_residual(residual, method=p.comp_method,
                                    thresh_st=p.comp_thresh_st, param=p.comp_param)
        clip_frac = float(np.mean(np.abs(residual) > p.comp_thresh_st / 12.0))
        fir = _fir(fr, p.band_lo, p.band_hi, p.fir_time_sec)
        too_short = T < filtfilt_min_len(fir)       # filtfilt refuses -> vibrato all-zero
        vib_raw = extract_vibrato(resid_c, uv, fir)
        phf = np.zeros(T, dtype=np.int64)
        is_nuc_f = np.zeros(T, dtype=bool)
        for k in range(len(ph_ids)):
            phf[fb[k]:fb[k + 1]] = ph_ids[k]
            if int(ph_ids[k]) in nuc_ids:
                is_nuc_f[fb[k]:fb[k + 1]] = True
        gain = make_vibrato_gain(               # the SAME extended function training uses
            phf, is_nuc_f, dur_sec, fr, T,
            sigma_sec=p.gain_sigma_sec,
            cons_margin_pre_sec=p.cons_margin_pre_sec,
            cons_margin_post_sec=p.cons_margin_post_sec,
            boundary_margin_pre_sec=p.boundary_margin_pre_sec,
            boundary_margin_post_sec=p.boundary_margin_post_sec,
            boundary_mode=p.boundary_mode,
            limit_margin_to_adjacent=p.limit_margin_to_adjacent,
            min_open_sec=p.min_open_sec)
        vib = (vib_raw * gain).astype(np.float32)
        plain = residual - vib

        f0_midi = hz_to_midi(f0)
        plain_midi = 69.0 + 12.0 * (baseline.astype(np.float64) + plain - LOG2_440)
        vib_cents = vib * 1200.0
        v = uv > 0.5
        rms = float(np.sqrt(np.mean(vib_cents[v] ** 2))) if v.any() else 0.0

        # ── decimate for display ──
        step = max(1, int(np.ceil(T / MAX_POINTS)))
        sl = slice(None, None, step)
        bounds = []
        for k in range(len(ph_ids)):
            bounds.append({'x': int(fb[k] // step), 'x2': int(fb[k + 1] // step),
                           'ph': id2ph.get(int(ph_ids[k]), '?')})
        items.append({
            'db': db, 'name': name, 'song': _song_of(name),
            'T': T, 'frame_rate': fr, 'dur_sec': round(T / fr, 2),
            'n_points': len(f0_midi[sl]),
            'f0_midi': np.round(f0_midi[sl], 3).tolist(),
            'plain_midi': np.round(plain_midi[sl], 3).tolist(),
            'vib_cents': np.round(vib_cents[sl], 1).tolist(),
            'gain': np.round(gain[sl], 3).tolist(),
            'uv': uv[sl].astype(int).tolist(),
            'bounds': bounds,
            'vib_rms_cents': round(rms, 1),
            'clip_pct': round(100 * clip_frac, 1),
            'too_short': bool(too_short),
        })

    fr0 = float(next(iter(metas.values()))['frame_rate']) if metas else 172.265625
    fir0 = _fir(fr0, p.band_lo, p.band_hi, p.fir_time_sec)
    return {'total': total, 'page': req.page, 'page_size': req.page_size,
               'lang': lang0,      # 自動判定した言語パック。UI に出して取り違えに気づけるようにする
            'taps': len(fir0), 'min_len_sec': round(filtfilt_min_len(fir0) / fr0, 2),
            'items': items}


# ── pitch representative tab ─────────────────────────────────────────────────

def _db_ctx(root: Path, db: str):
    d = root / db
    if not (d / 'shard.npz').exists():
        raise HTTPException(400, f'shard.npz not found: {d}')
    m = _meta(d)
    lang, nuc_names = _nucleus_for(frozenset(m['phoneme_vocab']))
    nuc_ids = frozenset(m['phoneme_vocab'][n] for n in nuc_names if n in m['phoneme_vocab'])
    if not nuc_ids:
        raise HTTPException(400, f"{db}: no language pack shares a nucleus phoneme with "
                                 f"this database's vocab.")
    return d, m, lang, nuc_ids


@app.post('/pitch')
async def pitch(req: PitchReq):
    root = Path(req.root).expanduser()
    d, m, lang, nuc_ids = _db_ctx(root, req.db)
    fr = float(m['frame_rate'])
    id2ph = {v: k for k, v in m['phoneme_vocab'].items()}
    kw = req.params.model_dump()

    q = req.query.strip().lower()
    names = [n for n in m['phrases'] if not q or q in n.lower()]
    if req.sort == 'shuffle':
        random.Random(req.seed).shuffle(names)
    else:
        names.sort()
    total = len(names)
    lo = max(0, req.page) * req.page_size
    z = _shard(d)

    items = []
    for name in names[lo:lo + req.page_size]:
        ph_ids = z[f'{name}|phoneme_ids'].astype(np.int64)
        dur_sec = z[f'{name}|dur_sec'].astype(np.float32)
        f0 = z[f'{name}|f0_interp'].astype(np.float32)
        uv = z[f'{name}|uv'].astype(np.float32)
        T = len(f0)
        fb = phoneme_frame_bounds(dur_sec, fr, T)
        values, wts = nucleus_values(ph_ids, dur_sec, f0, uv, fr, nuc_ids,
                                     return_weights=True, **kw)
        # 山検出の発火判定用: 山検出だけを切った値
        values0 = (nucleus_values(ph_ids, dur_sec, f0, uv, fr, nuc_ids, **{**kw, 'crest_max_ms': 0.0})
                   if float(kw.get('crest_max_ms', 0.0)) > 0.0 else values)

        f0_midi = hz_to_midi(f0)
        step = max(1, int(np.ceil(T / MAX_POINTS)))
        sl = slice(None, None, step)
        segs, bounds = [], []
        for k in range(len(ph_ids)):
            ph = id2ph.get(int(ph_ids[k]), '?')
            bounds.append({'x': int(fb[k] // step), 'x2': int(fb[k + 1] // step), 'ph': ph})
            v = float(values[k]) if k < len(values) else float('nan')
            if np.isfinite(v):
                n_fr = int(fb[k + 1] - fb[k])
                segs.append({'x': int(fb[k] // step), 'x2': int(fb[k + 1] // step),
                             'midi': round(v, 3), 'ph': ph,
                             'ms': round(n_fr / fr * 1000.0),
                             # 山検出が発火した核か（値が山検出なしの値と違う）
                             'alpha': 1.0 if (k < len(values0) and np.isfinite(values0[k])
                                              and abs(float(values0[k]) - v) > 1e-4) else 0.0,
                             'cents_off': round((v - round(v)) * 100.0, 1)})
        vv = values[np.isfinite(values)]
        items.append({
            'db': req.db, 'name': name, 'song': _song_of(name),
            'T': T, 'dur_sec': round(T / fr, 2),
            'n_points': len(f0_midi[sl]),
            'f0_midi': np.round(f0_midi[sl], 3).tolist(),
            'uv': uv[sl].astype(int).tolist(),
            # 加重平均に実際に入った重み（核ごとに最大1で正規化・核の外は0）。
            # どのフレームが代表値を決めたかを色で見せるためだけに使う。
            'w': np.round(wts[sl], 3).tolist(),
            'bounds': bounds, 'segs': segs,
            'n_values': int(len(vv)),
            'n_nuclei': int(sum(1 for x in ph_ids if int(x) in nuc_ids)),
        })
    return {'total': total, 'page': req.page, 'page_size': req.page_size,
            'lang': lang, 'items': items}


def main():
    global _NUCLEUS_NAMES
    import uvicorn
    ap = argparse.ArgumentParser(description='vibrato separation tuner web app')
    ap.add_argument('--port', type=int, default=8017)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--nucleus', default=None, metavar='FILE',
                    help='核音素リスト（1 行 1 音素）で言語パック dict/<lang>.yaml の nucleus: を'
                         '上書きする。名前は各 DB の音素表と照合される。')
    args = ap.parse_args()
    if args.nucleus:
        _NUCLEUS_NAMES = frozenset(load_nucleus(args.nucleus, validate=False))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
