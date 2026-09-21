"""Collect the demo-page audio into ``docs/audio/``.

    python tools/make_demo_audio.py <wav_dir> [<wav_dir> ...]

Every ``*.wav`` in the given directories is encoded to ``docs/audio/<name>.flac`` and the
sizes are printed. The page (``docs/index.html``) refers to these names, so re-running this
after a new render is all that is needed to refresh the demo.

FLAC rather than MP3 because the page exists to demonstrate how the vocoder sounds. At
libsndfile's default quality MP3 lands at ~80 kbps and audibly dulls the top octave; even at
320 kbps it still differs. FLAC is lossless, about a quarter larger than 320 kbps MP3, and
plays in every current browser.

**The level is left exactly as rendered.** ``svs.render`` settles it on the mel, where the
vocoder is driven at the level it was trained on, and touches the waveform no further. Peak
normalising each clip here would undo that: one plosive would decide a whole clip's level,
and the page compares speakers side by side. Measured over the nine speaker clips, the
spread of the active RMS is 1.8 dB as rendered and 5.4 dB after a peak-0.95 pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "audio"


def main(dirs: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for d in dirs:
        for wav in sorted(Path(d).glob("*.wav")):
            x, sr = sf.read(wav, dtype="float32")
            peak = float(np.abs(x).max())
            if peak >= 1.0:                  # 鳴らない不具合だけは見逃さない
                print(f"  [warn] {wav.name}: peak {peak:.3f} >= 1.0")
            dst = OUT / (wav.stem + ".flac")
            sf.write(dst, x, sr, format="FLAC", subtype="PCM_16")
            kb = dst.stat().st_size / 1024
            total += kb
            print(f"{dst.relative_to(REPO)}  {len(x) / sr:5.1f}s  {kb:6.0f} KB")
    print(f"\n{total / 1024:.1f} MB in {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
