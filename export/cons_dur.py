"""子音長モデルの ONNX エクスポート。

    python -m export.cons_dur --ckpt <cons_dur ckpt> --out export/out/cons_dur.onnx

ONNX が返すのは **自然長 [秒]** だけ（`model.predict` と同じ `exp(x) − LOG_OFFSET`、
`clamp(min=0.005)` までをグラフに含める）。floor・リージョン圧縮・音節分解はホスト側
（DiffSinger の dur.onnx と同じ切り方）。

モデルは `ConsonantDurModel`（`duration_model/model.py`）の 1 種: 子音 1 つずつを、その前後
1 音素の文脈と音節の情報で見る MLP。入力は **子音ごとのレコード** [N]。

入力の組はモデルが持つ埋め込みで変わる（話者・スタイル・cluster_len・±2 文脈）。
どの名前で何を渡すかは、隣に書き出す `<out>.json` と ONNX の metadata_props に記録する。
ホストはそこから音素名 → id の表と spk_map を読む（`ps.to_id` を自前で再現しないこと。
無声化母音の正規化 A→a などは表の側で解決済み＝表は正規化後の id を持つ）。

検証: ORT と PyTorch を (a) 乱数入力 (b) 実フレーズから作った実レコード の両方で比べ、
最大差を報告する。配布前に **バイト走査** して個人情報（絶対パス等）が焼かれていない
ことを確かめる（`--scan` は既定 ON）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from duration_model.model import ConsonantDurModel, LOG_OFFSET, load_cons_dur_model  # noqa: E402

OPSET = 17


# ── ラッパー ─────────────────────────────────────────────────────────────────

class NeighborWrapper(nn.Module):
    """ConsonantDurModel を「名前付きテンソルの列 → dur_sec [N]」に平たくする。"""

    def __init__(self, model: ConsonantDurModel):
        super().__init__()
        self.m = model
        names = ['cons_id', 'prev_ph_id', 'next_ph_id', 'pitch_midi', 'prev_note_exists',
                 'prev_note_dur', 'next_note_dur', 'pitch_diff']
        if model.use_wide_context:
            names += ['prev2_ph_id', 'next2_ph_id']
        if model.use_cluster_len:
            names += ['cluster_len']
        if model.n_speakers > 1:
            names += ['spk_id']
        if model.n_styles > 0:
            names += ['style_id']
        self.input_names = names

    def forward(self, *args):
        b = dict(zip(self.input_names, args))
        return (torch.exp(self.m(b)) - LOG_OFFSET).clamp(min=0.005)

    def example(self, n: int = 7):
        n_ph = self.m.n_phonemes
        g = torch.Generator().manual_seed(0)
        ex = {
            'cons_id':          torch.randint(1, n_ph, (n,), generator=g),
            'prev_ph_id':       torch.randint(0, n_ph, (n,), generator=g),
            'next_ph_id':       torch.randint(0, n_ph, (n,), generator=g),
            'pitch_midi':       torch.randint(40, 90, (n,), generator=g),
            'prev_note_exists': torch.randint(0, 2, (n,), generator=g),
            'prev_note_dur':    torch.rand(n, generator=g) * 0.6,
            'next_note_dur':    torch.rand(n, generator=g) * 0.6,
            'pitch_diff':       torch.randn(n, generator=g) * 3.0,
            'prev2_ph_id':      torch.randint(0, n_ph, (n,), generator=g),
            'next2_ph_id':      torch.randint(0, n_ph, (n,), generator=g),
            'cluster_len':      torch.randint(1, 4, (n,), generator=g),
            'spk_id':           torch.randint(0, max(1, self.m.n_speakers), (n,), generator=g),
            'style_id':         torch.randint(0, max(1, self.m.n_styles), (n,), generator=g),
        }
        return tuple(ex[k] for k in self.input_names)

    def dynamic_axes(self):
        return {**{k: {0: 'N'} for k in self.input_names}, 'dur_sec': {0: 'N'}}


# ── 読み込み・メタデータ ─────────────────────────────────────────────────────

def load_any(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location='cpu')
    cfg = ck.get('config', {}) if isinstance(ck, dict) else {}
    model = load_cons_dur_model(ck, device='cpu')
    return model, NeighborWrapper(model), cfg


def metadata(model, wrapper, cfg: dict, ckpt_path: str) -> dict:
    ps = getattr(model, 'phoneset', None)
    phonemes = list(ps.phonemes) if ps is not None else None
    md = {
        'model': 'prettypitch-cons-dur',
        'model_description': ('ConsonantDurModel: an MLP that predicts the natural length of each '
                              'consonant from the consonant, its previous and next phoneme, the '
                              'syllable pitch and the neighbouring note lengths; one record per '
                              'consonant'),
        'inputs': list(wrapper.input_names),
        'output': 'dur_sec',
        'output_unit': 'seconds (natural consonant length; exp(x) - LOG_OFFSET, clamp >= 0.005)',
        'log_offset': LOG_OFFSET,
        'n_phonemes': int(model.n_phonemes),
        'phonemes': phonemes,                     # id = 行順。ホストはこれで記号 → id
        'lang': getattr(ps, 'lang', None) if ps is not None else None,
        'nuclei': sorted(ps.nuclei) if ps is not None else None,
        'consonants': sorted(ps.cons_set) if ps is not None else None,
        'vowel_normalize': dict(getattr(ps, 'vowel_normalize', {}) or {}) if ps is not None else None,
        'n_speakers': int(model.n_speakers),
        'n_styles': int(model.n_styles),
        'spk_map': dict(cfg.get('spk_map') or {}),
        'style_map': dict(cfg.get('style_map') or {}),
        'profile_hint': f"duration_model/profiles/{Path(ckpt_path).stem}.json",
        'source_ckpt': Path(ckpt_path).name,      # ファイル名だけ。パスは焼かない
        'opset': OPSET,
    }
    md['use_cluster_len'] = bool(model.use_cluster_len)
    md['use_wide_context'] = bool(model.use_wide_context)
    return md


# ── 検証 ───────────────────────────────────────────────────────────────────

def _to_np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def verify(wrapper, onnx_path: str, examples: list[tuple]) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    worst = 0.0
    for ex in examples:
        with torch.no_grad():
            ref = _to_np(wrapper(*ex))
        out = sess.run(['dur_sec'], {n: _to_np(t) for n, t in zip(wrapper.input_names, ex)})[0]
        assert out.shape == ref.shape, (out.shape, ref.shape)
        worst = max(worst, float(np.abs(out - ref).max()))
    return worst


def real_examples_neighbor(model) -> list[tuple]:
    """実フレーズ（言語パックの音素で組んだ列）から extract_phrase_records → records_to_batch。
    推論経路そのものを通した入力で ORT を叩く。"""
    from duration_model.features import extract_phrase_records, records_to_batch
    ps = model.phoneset
    nuc = sorted(ps.nuclei)
    cons = sorted(c for c in ps.cons_set if c not in ('cl',))[:12]
    seqs = [
        ['pau', 'k', 'a', 'r', 'a', 's', 'u', 'pau'],
        ['pau', 'sh', 'i', 'n', 'a', 'g', 'a', 'w', 'a', 'pau'],
        ['pau'] + [x for c in cons for x in (c, nuc[0])] + ['pau'],
    ]
    out = []
    for phs in seqs:
        phs = [p for p in phs if p in ps.phonemes]
        durs = np.array([0.30 if (ps.is_nucleus(p) or ps.is_pau(p)) else 0.0 for p in phs])
        vals = np.array([60.0 + (i % 5) if ps.is_nucleus(p) else np.nan for i, p in enumerate(phs)])
        recs = extract_phrase_records(phs, durs, vals, 0, 0, phoneset=ps)
        if not recs:
            continue
        for spk in range(max(1, model.n_speakers)):
            for r in recs:
                r['spk_id'] = spk; r['style_id'] = min(spk, max(0, model.n_styles - 1))
            b = records_to_batch(recs, device='cpu', phoneset=ps)
            out.append(tuple(b[k] for k in NeighborWrapper(model).input_names))
    return out


# ── 個人情報の走査 ─────────────────────────────────────────────────────────

def scan_bytes(path: str, needles: list[bytes]) -> list[str]:
    """ONNX のバイト列に焼かれてはいけない文字列（絶対パス・メール等）を探す。"""
    data = Path(path).read_bytes()
    hits = [n.decode('utf-8', 'replace') for n in needles if n and n in data]
    for n in (b'/Users/', b'/home/', b'C:\\\\Users', b'@gmail', b'CloudStorage'):
        if n in data:
            hits.append(n.decode())
    return sorted(set(hits))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True, help='書き出す .onnx。隣に <out>.json も書く')
    ap.add_argument('--opset', type=int, default=OPSET)
    ap.add_argument('--no-simplify', action='store_true', help='onnxsim を通さない')
    ap.add_argument('--no-scan', action='store_true', help='個人情報のバイト走査を省く')
    args = ap.parse_args()

    model, wrapper, cfg = load_any(args.ckpt)
    wrapper.eval()
    ex = wrapper.example()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    print(f'n_phonemes={model.n_phonemes}  n_speakers={model.n_speakers}  '
          f'n_styles={model.n_styles}  inputs={wrapper.input_names}')
    torch.onnx.export(
        wrapper, ex, str(out), opset_version=args.opset,
        input_names=list(wrapper.input_names), output_names=['dur_sec'],
        dynamic_axes=wrapper.dynamic_axes(), do_constant_folding=True, dynamo=False,
    )

    # onnxsim（任意）。失敗しても元のグラフで続ける。
    if not args.no_simplify:
        try:
            import onnx
            from onnxsim import simplify
            m, ok = simplify(onnx.load(str(out)))
            if ok:
                onnx.save(m, str(out)); print('[onnxsim] simplified')
            else:
                print('[onnxsim] check failed; keeping the unsimplified graph')
        except Exception as e:                                       # noqa: BLE001
            print(f'[onnxsim] skipped ({e.__class__.__name__}: {e})')

    # メタデータ: JSON サイドカー + ONNX metadata_props（同じ内容）。
    import onnx
    md = metadata(model, wrapper, cfg, args.ckpt)
    side = out.with_suffix('.json')
    side.write_text(json.dumps(md, ensure_ascii=False, indent=2), encoding='utf-8')
    m = onnx.load(str(out))
    m.producer_name = 'prettypitch'
    m.doc_string = 'PrettyPitch consonant-duration model: natural consonant length in seconds'
    del m.metadata_props[:]
    for k, v in md.items():
        p = m.metadata_props.add(); p.key = k
        p.value = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    onnx.save(m, str(out))
    onnx.checker.check_model(onnx.load(str(out)))

    # 検証: 乱数 + 実レコード
    rand = [wrapper.example(n) for n in (1, 3, 64, 200)]
    real = real_examples_neighbor(model)
    w_rand = verify(wrapper, str(out), rand)
    w_real = verify(wrapper, str(out), real)
    print(f'[verify] ORT vs torch  乱数 {len(rand)} 組: 最大差 {w_rand*1000:.4f} ms   '
          f'実レコード {len(real)} 組: 最大差 {w_real*1000:.4f} ms')
    if max(w_rand, w_real) > 1e-4:
        print('[verify] WARNING: 0.1 ms を超える差があります'); rc = 2
    else:
        rc = 0

    # 個人情報の走査（絶対パス・メール・作者のホーム）
    if not args.no_scan:
        needles = [os.path.expanduser('~').encode(), str(Path(args.ckpt).resolve().parent).encode()]
        hits = scan_bytes(str(out), needles)
        if hits:
            print(f'[scan] NG: 焼かれてはいけない文字列があります: {hits}'); rc = max(rc, 3)
        else:
            print('[scan] OK: 絶対パス・メールアドレスなし')

    size = out.stat().st_size / 1e6
    print(f'wrote {out} ({size:.2f} MB) + {side.name}')
    sys.exit(rc)


if __name__ == '__main__':
    main()
