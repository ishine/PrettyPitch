"""Render every clip the demo page plays, into wav.

    python tools/render_demos.py <out_dir> [--only songs|tempo|scales]

Then encode them with ``tools/make_demo_audio.py <out_dir>``, which writes
``docs/audio/*.flac`` under the names ``docs/index.html`` refers to.

Three groups, matching the three tables on the page:

``songs``   three songs × three speakers. Natsume is transposed down 5 semitones to sit in
            his range; the other two sing the score as written.
``tempo``   Ievan polkka at 200–400 BPM, sung by Ritsu, to show what happens when the
            consonants run out of room.
``scales``  the first phrase of Furusato, sung by Ritsu, with ``plain_scale`` (expression)
            and ``vib_scale`` (vibrato) each swept over 0.0 … 2.0. One seed throughout, so
            the only difference between the 25 clips is the scaling.

Re-run this whenever a model changes. It is the whole demo page in one command, so the page
never drifts away from the weights the repository ships.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import svs.render as R                                    # noqa: E402
from svs.score_io import load_score                       # noqa: E402

SEED = 0
SPEAKERS = [(0, 'oniku', 0), (1, 'natsume', -5), (2, 'ritsu', 0)]
SONGS = ['kirakiraboshi', 'furusato', 'haru_no_ogawa']
TEMPI = [200, 250, 300, 350, 400]
SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]


def _tag(x: float) -> str:
    """0.5 -> 'p05'/'v05' の数字部。0.0 -> '00', 2.0 -> '20'。"""
    return f'{int(round(x * 10)):02d}'


def _write(out: Path, name: str, wav: np.ndarray, sr: int) -> None:
    p = out / f'{name}.wav'
    sf.write(p, wav, sr)
    a = np.abs(wav.astype(np.float64))
    print(f'  {name:26s} {len(wav) / sr:6.2f}s  peak {a.max():.3f}  '
          f'RMS {np.sqrt((a ** 2).mean()):.4f}', flush=True)


def songs(out: Path) -> None:
    for spk, name, tr in SPEAKERS:
        ctx = R.load_models(spk_id=spk, device='cpu', verbose=False)
        for song in SONGS:
            wav, sr = R.render_score(REPO / 'examples' / f'{song}.musicxml',
                                     ctx=ctx, seed=SEED, transpose=tr)
            _write(out, f'{song}_{name}', wav, sr)


def tempo(out: Path) -> None:
    ctx = R.load_models(spk_id=2, device='cpu', verbose=False)
    for bpm in TEMPI:
        wav, sr = R.render_score(REPO / 'examples' / f'ievan_polkka_{bpm}.musicxml',
                                 ctx=ctx, seed=SEED)
        _write(out, f'ievan_polkka_{bpm}_ritsu', wav, sr)


def scales(out: Path) -> None:
    """25 通り。1 フレーズだけなので render_phrase を直に呼ぶ。"""
    ctx = R.load_models(spk_id=2, device='cpu', verbose=False)
    phr = load_score(REPO / 'examples' / 'furusato.musicxml', phoneset=ctx.phoneset)[0]
    for ps_ in SCALES:
        for vs_ in SCALES:
            r = R.render_phrase(ctx, phr, seed=SEED, plain_scale=ps_, vib_scale=vs_)
            _write(out, f'demo_scale_p{_tag(ps_)}_v{_tag(vs_)}', r['wav'], ctx.sample_rate)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out', help='wav の出力先')
    ap.add_argument('--only', choices=('songs', 'tempo', 'scales'), default=None,
                    help='一部だけ作り直す')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    groups = {'songs': songs, 'tempo': tempo, 'scales': scales}
    for name, fn in groups.items():
        if a.only in (None, name):
            print(f'--- {name} ---', flush=True)
            fn(out)
    n = len(list(out.glob('*.wav')))
    print(f'\n{n} clips -> {out}')
    print(f'次: python tools/make_demo_audio.py {out}')


if __name__ == '__main__':
    main()
