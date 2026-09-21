"""QA plot: F0 / nucleus representative values / baseline / plain / vibrato per phrase.

    python -m tools.plot_decomp --db data/ritsu --out out/decomp --limit 5

前処理済み DB の各フレーズについて、上段=F0(MIDI)+代表値マーカー+階段+平滑baseline、
下段=plain と vibrato(cents) を1枚の PNG に描く。前処理・代表値・分解が意図どおりか
目で確認するためのツール(パイプラインは目視検査できる形を保つ)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prettypitch.pitch import nucleus_values, hz_to_midi
from prettypitch.baseline import build_baseline, phoneme_frame_bounds, fill_values
from prettypitch.decomp import (make_vibrato_fir, compress_residual, extract_vibrato,
                                make_vibrato_gain)
from preprocess.vocab import load_nucleus


def main():
    ap = argparse.ArgumentParser(description="Plot F0 decomposition per phrase (QA).")
    ap.add_argument("--db", required=True, help="preprocessed dataset dir (with shard.npz)")
    ap.add_argument("--out", required=True, help="output dir for PNGs")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--nucleus", default=None)
    args = ap.parse_args()

    d = Path(args.db)
    meta = json.load(open(d / "metadata.json", encoding="utf-8"))
    z = np.load(d / "shard.npz")
    fr = float(meta["frame_rate"])
    v2i = meta["phoneme_vocab"]
    i2v = {v: k for k, v in v2i.items()}
    nuc_names = load_nucleus(args.nucleus)
    nuc_ids = frozenset(v2i[p] for p in nuc_names if p in v2i)
    fir = make_vibrato_fir(fr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(meta["phrases"])[:args.limit]:
        ph = z[f"{name}|phoneme_ids"].astype(int)
        du = z[f"{name}|dur_sec"]
        f0 = z[f"{name}|f0_interp"]
        uv = z[f"{name}|uv"]
        T = len(f0)
        fb = phoneme_frame_bounds(du, fr, T)
        vals = nucleus_values(ph, du, f0, uv, fr, nuc_ids)
        base = build_baseline(vals, du, T, fr)
        if base is None:
            print(f"[skip] {name}: no pitch value")
            continue
        filled = fill_values(vals)
        step = np.concatenate([np.full(fb[k + 1] - fb[k], filled[k]) for k in range(len(ph))])
        resid = np.log2(np.maximum(f0, 1.0)) - base
        is_nuc = np.zeros(T, bool)
        phf = np.zeros(T, int)
        for k in range(len(ph)):
            phf[fb[k]:fb[k + 1]] = ph[k]
            if int(ph[k]) in nuc_ids:
                is_nuc[fb[k]:fb[k + 1]] = True
        vib = make_vibrato_gain(phf, is_nuc, du, fr, T) * \
            extract_vibrato(compress_residual(resid), uv, fir)
        plain = resid - vib

        t = np.arange(T) / fr
        f0m = hz_to_midi(f0)
        f0m_plot = np.where(uv > 0.5, f0m, np.nan)
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(t, f0m, color="#bbddbb", lw=0.8, label="F0 interp")
        ax.plot(t, f0m_plot, color="#22aa44", lw=1.4, label="F0 voiced")
        ax.plot(t, step, color="#ccaa66", lw=1.0, ls=":", label="value staircase")
        ax.plot(t, 12 * (base - np.log2(440.0)) + 69, color="#dd4444", lw=1.6,
                label="baseline (smoothed)")
        for k in range(len(ph)):
            if np.isfinite(vals[k]):
                c = (fb[k] + fb[k + 1]) / 2 / fr
                ax.plot([c], [vals[k]], "o", color="#d000d0", ms=5)
                ax.annotate(f"{vals[k]:.2f}", (c, vals[k]), fontsize=6,
                            textcoords="offset points", xytext=(0, 6), ha="center")
            ax.axvline(fb[k] / fr, color="#eeeeee", lw=0.5)
            ax.text(fb[k] / fr, ax.get_ylim()[0], i2v[int(ph[k])], fontsize=5, va="bottom")
        ax.set_ylabel("MIDI")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(name, fontsize=9)
        ax2.plot(t, plain * 1200, color="#8888dd", lw=0.9, label="plain [cents]")
        ax2.plot(t, vib * 1200, color="#dd8888", lw=0.9, label="vibrato [cents]")
        ax2.axhline(0, color="#cccccc", lw=0.5)
        ax2.set_xlabel("Time (s)")
        ax2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", dpi=110)
        plt.close(fig)
        print(f"  {name}.png")


if __name__ == "__main__":
    main()
