"""Draw the demo page's two result figures into ``docs/fig/``.

    python tools/make_demo_figures.py

Both figures come from real renders with the bundled weights and seed 0, so they show the
same durations and F0 as the clips on the page:

    demo_<song>_<speaker>.png   the notes, the generated F0 and the generated durations of
                                every clip in the "three singers" table (nine figures)
    demo_durations.png          the same bar of Ievan polkka at 200 and 400 BPM, to scale
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

from prettypitch.pitch import hz_to_midi                          # noqa: E402
from svs import render as R                                       # noqa: E402
from svs.score_io import load_score                               # noqa: E402

OUT = REPO / "docs" / "fig"
SEED = 0
SPK = [(0, "ONIKU Kurumi", 0, "#2f6fae"),
       (1, "Natsume Yuuri  (−5)", -5, "#c2702d"),
       (2, "Namine Ritsu", 0, "#3f8f6a")]
INK, SUB, GRID = "#2b3a4d", "#6c757d", "#e4e9ee"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c7d2dc")
    ax.tick_params(colors=SUB, labelsize=8)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


SONGS = [("kirakiraboshi", "Kirakira Boshi"),
         ("furusato", "Furusato"),
         ("haru_no_ogawa", "Haru no Ogawa")]
C_NOTE, C_F0 = "#dbe4f0", "#1f6f8b"
C_CONS, C_CONS_E = "#cfdbe9", "#6b7f9b"
C_VOW, C_VOW_E = "#dff0e6", "#5f9e7f"
C_PAU, C_PAU_E = "#eef1f4", "#c3ced9"


def clip_figure(ctx, song, title, speaker, transpose, stem, row_sec: float = 13.0):
    """One clip, laid out like a score: rows of `row_sec`, each with the notes, the
    generated F0 and the phoneme timeline on one time axis."""
    phr = load_score(REPO / "examples" / f"{song}.musicxml", phoneset=ctx.phoneset)
    if transpose:
        for p in phr:
            for n in p.notes:
                n.pitch_midi = int(n.pitch_midi) + transpose

    t0 = 0.0
    f0_t, f0_m, f0_all, notes, strip = [], [], [], [], []
    for p in phr:
        r = R.render_phrase(ctx, p, seed=SEED, f0_only=True)
        seq, out = r["seq"], r["f0"]
        fr = float(out["frame_rate"])
        f0 = np.asarray(out["f0_hz"], float)
        uv = np.asarray(out["uv"], float)
        f0_t.append(t0 + np.arange(len(f0)) / fr)
        # 音響モデルとボコーダには無声区間も含む連続 F0 が入る（uv は別チャンネル）。
        f0_all.append(hz_to_midi(f0))
        f0_m.append(np.where(uv > 0.5, hz_to_midi(f0), np.nan))

        d = np.asarray(seq.dur_sec, float)
        edges = t0 + np.concatenate([[0.0], np.cumsum(d)])
        for ph, a, b in zip(seq.phonemes, edges[:-1], edges[1:]):
            strip.append((ph, a, b))
        starts = list(np.where(np.asarray(seq.note_event_starts, bool))[0]) + [len(d)]
        for i0, i1 in zip(starts[:-1], starts[1:]):
            midi = float(np.asarray(seq.note_midi, float)[i0])
            if midi > 0:
                notes.append((edges[i0], edges[i1], midi))
        t0 = float(edges[-1])

    vow = set("aiueoN")
    lo = min(float(np.nanmin(y)) for y in f0_m if np.isfinite(y).any())
    hi = max(float(np.nanmax(y)) for y in f0_m if np.isfinite(y).any())
    ylo = min(lo, min(n[2] for n in notes)) - 1.0
    yhi = max(hi, max(n[2] for n in notes)) + 1.0

    n_rows = max(1, int(np.ceil(t0 / row_sec)))
    row_sec = t0 / n_rows                       # 端数の行を作らず、全行を同じ幅にする
    h = 0.80 + 2.05 * n_rows
    fig = plt.figure(figsize=(11.2, h), dpi=170)
    outer = fig.add_gridspec(n_rows, 1, hspace=0.42, left=0.075, right=0.99,
                             top=1 - 0.38 / h, bottom=0.52 / h)
    for k in range(n_rows):
        a0, a1 = k * row_sec, min((k + 1) * row_sec, t0)
        inner = outer[k].subgridspec(2, 1, height_ratios=[3.0, 1.15], hspace=0.0)
        ax = fig.add_subplot(inner[0])
        axs = fig.add_subplot(inner[1], sharex=ax)
        _style(ax); _style(axs)
        axs.grid(False)

        for ph, a, b in strip:                      # 背景: 子音 / 母音 / 休符の領域
            if b <= a0 or a >= a1:
                continue
            pau = ph in ("pau", "br", "sil")
            bg = ("#fafbfc" if pau else ("#f1f8f4" if ph in vow else "#eef3f9"))
            ax.axvspan(max(a, a0), min(b, a1), color=bg, lw=0, zorder=0)
        for a, b, midi in notes:                    # ノート（F0 の下に敷く）
            if b <= a0 or a >= a1:
                continue
            ax.add_patch(plt.Rectangle((a, midi - 0.40), b - a, 0.80, facecolor=C_NOTE,
                                       edgecolor="#93a8c4", lw=0.7, zorder=1))
        for x, ya, yv in zip(f0_t, f0_all, f0_m):   # 生成された F0
            m = (x >= a0 - 0.05) & (x <= a1 + 0.05)
            if not m.any():
                continue
            # 点線 = 無声区間も含む連続 F0（これが音響モデルとボコーダへ入る）
            ax.plot(x[m], ya[m], lw=1.0, ls=(0, (1.6, 1.4)), color=C_F0, alpha=0.75,
                    zorder=2)
            ax.plot(x[m], yv[m], lw=1.5, color=C_F0, zorder=3, solid_capstyle="round")
        ax.set_xlim(a0, a1)
        ax.set_ylim(ylo, yhi)
        ax.tick_params(labelbottom=False)
        ax.set_ylabel("MIDI", fontsize=7.5, color=SUB)

        for ph, a, b in strip:                      # 音素の帯（全部にラベルを書く）
            if b <= a0 or a >= a1:
                continue
            pau = ph in ("pau", "br", "sil")
            fc, ec = ((C_PAU, C_PAU_E) if pau else
                      ((C_VOW, C_VOW_E) if ph in vow else (C_CONS, C_CONS_E)))
            x0, x1 = max(a, a0), min(b, a1)
            axs.add_patch(plt.Rectangle((x0, 0.0), x1 - x0, 1.0, facecolor=fc,
                                        edgecolor=ec, lw=0.5))
            if pau:
                continue
            w = x1 - x0
            rot = 0 if w > row_sec * 0.030 else 90   # 狭いものは縦書きにして必ず出す
            axs.text((x0 + x1) / 2, 0.5, ph, ha="center", va="center",
                     fontsize=6.0 if rot else 7.0, color=INK, rotation=rot)
        axs.set_xlim(a0, a1)
        axs.set_ylim(0, 1)
        axs.set_yticks([])
        axs.tick_params(labelsize=7.5)
        if k == n_rows - 1:
            axs.set_xlabel("seconds", fontsize=8.5, color=SUB)
    tag = f"  (transposed {transpose:+d})" if transpose else ""
    fig.suptitle(f"{title} — {speaker}{tag}", fontsize=10.5, color=INK, x=0.075,
                 ha="left", y=1 - 0.10 / h)
    fig.savefig(OUT / f"{stem}.png", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / f"{stem}.png").relative_to(REPO))


def clip_figures():
    for spk, name, tr, _c in SPK:
        ctx = R.load_models(spk_id=spk, device="cpu", verbose=False)
        short = name.split()[0].lower()
        for song, title in SONGS:
            clip_figure(ctx, song, title, name.split("  ")[0], tr, f"demo_{song}_{short}")


# ── figure 2: the generated durations at two tempos ──────────────────────────

def duration_figure(n_notes: int = 10):
    ctx = R.load_models(spk_id=2, device="cpu", verbose=False)
    ps = ctx.phoneset
    vow = set("aiueoN")
    rows = []
    for bpm in (200, 400):
        phr = load_score(REPO / "examples" / f"ievan_polkka_{bpm}.musicxml",
                         phoneset=ps)[0]
        seq = R.render_phrase(ctx, phr, seed=SEED, f0_only=True)["seq"]
        items = [(p, d) for p, d in zip(seq.phonemes, seq.dur_sec) if p != "pau"]
        keep, n_v = [], 0
        for p, d in items:                       # the first n_notes syllables
            keep.append((p, d))
            if p in vow:
                n_v += 1
                if n_v >= n_notes:
                    break
        rows.append((bpm, keep))

    span = max(sum(d for _, d in k) for _, k in rows) * 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 3.2), dpi=170, sharex=True)
    for ax, (bpm, keep) in zip(axes, rows):
        _style(ax)
        ax.grid(False)
        x = 0.0
        for p, d in keep:
            w = d * 1000.0
            is_v = p in vow
            ax.barh(0, w, left=x, height=0.62,
                    color=("#dff0e6" if is_v else "#cfdbe9"),
                    edgecolor=("#5f9e7f" if is_v else "#6b7f9b"), lw=1.0)
            if w > span * 0.018:
                ax.text(x + w / 2, 0, p, ha="center", va="center", fontsize=8, color=INK)
            x += w
        ax.set_xlim(0, span * 1.02)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel(f"{bpm} BPM", fontsize=9.5, color=INK, rotation=0,
                      ha="right", va="center", labelpad=12)
    axes[-1].set_xlabel("milliseconds", fontsize=9, color=SUB)
    axes[0].legend(handles=[Patch(facecolor="#cfdbe9", edgecolor="#6b7f9b", label="consonant"),
                            Patch(facecolor="#dff0e6", edgecolor="#5f9e7f", label="vowel")],
                   fontsize=8.5, ncol=2, loc="upper right", frameon=False,
                   bbox_to_anchor=(1.0, 1.55))
    axes[0].set_title("Generated durations — the same ten syllables, two tempos",
                      fontsize=11, color=INK, pad=22, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "demo_durations.png", facecolor="white")
    print("wrote", (OUT / "demo_durations.png").relative_to(REPO))


# ── the expression / vibrato matrix ──────────────────────────────────────────

def scale_figure():
    """The two knobs, one axis each: expression scales the whole contour, vibrato only the
    oscillation. Overlaying all combinations hides the vibrato, which is ±0.1 st.

    曲線はグリッドやゼロ線より前に描く（zorder）。後から引くと線が曲線を横切って見える。"""
    ctx = R.load_models(spk_id=2, device="cpu", verbose=False)
    phr = load_score(REPO / "examples" / "furusato.musicxml", phoneset=ctx.phoneset)[0]
    SC = (0.0, 0.5, 1.0, 1.5, 2.0)
    HUE = {0.0: "#c3ced9", 0.5: "#8fb4d4", 1.0: "#3d87a8",
           1.5: "#1f6f8b", 2.0: "#0e2d3a"}

    # 時間軸は共通。上段の目盛りを消して、x 軸を図の下 1 本だけにする
    # （上段の軸が残ると、下段の曲線の真上に軸が浮いて見える）。
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.2, 4.8), dpi=170, sharex=True,
                                   gridspec_kw=dict(height_ratios=[2.0, 1.0], hspace=0.42))
    for ax in (ax1, ax2):
        _style(ax)
    ax1.tick_params(labelbottom=False)
    ax1.spines["bottom"].set_visible(False)

    for k, ps_ in enumerate(SC):                       # 表情: F0 そのもの
        out = R.render_phrase(ctx, phr, seed=SEED, plain_scale=ps_, vib_scale=1.0,
                              f0_only=True)["f0"]
        fr = float(out["frame_rate"])
        f0 = np.asarray(out["f0_hz"], float)
        uv = np.asarray(out["uv"], float)
        t = np.arange(len(f0)) / fr
        if k == 0:
            b = np.asarray(out["baseline_log2"], float)
            ax1.plot(t, 69 + 12 * (b - np.log2(440.0)), color="#aebccb", lw=1.1, zorder=2,
                     ls=(0, (5, 2)), label="score pitch")
        ax1.plot(t, np.where(uv > 0.5, hz_to_midi(f0), np.nan), color=HUE[ps_], lw=1.3,
                 zorder=3 + k, label=f"expression ×{ps_}")
    ax1.set_xlim(0, float(len(f0) / fr))
    ax1.set_ylabel("pitch [MIDI]", fontsize=8.5, color=SUB)
    ax1.margins(y=0.10)                       # 曲線が題名や枠に触れないよう余白を取る
    ax1.legend(fontsize=8, ncol=6, frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    ax1.set_title("Expression — the scoops, the transitions and the note tails grow together",
                  fontsize=10, color=INK, pad=22, loc="left")

    ax2.axhline(0, color="#c3ced9", lw=0.9, zorder=2)   # 曲線より先に、背面へ
    for j, vs_ in enumerate(SC):                       # ビブラート: その寄与だけ
        out = R.render_phrase(ctx, phr, seed=SEED, plain_scale=1.0, vib_scale=vs_,
                              f0_only=True)["f0"]
        fr = float(out["frame_rate"])
        v = np.asarray(out["vib_delta"], float) * 1200.0
        t = np.arange(len(v)) / fr
        ax2.plot(t, v, color=HUE[vs_], lw=1.2, zorder=3 + j, label=f"vibrato ×{vs_}")
    ax2.set_xlim(0, float(len(v) / fr))
    ax2.set_xlabel("seconds", fontsize=8.5, color=SUB)
    ax2.set_ylabel("cents", fontsize=8.5, color=SUB)
    ax2.margins(y=0.12)
    ax2.legend(fontsize=8, ncol=5, frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0)
    ax2.set_title("Vibrato — the same oscillation, scaled; it is a separate channel",
                  fontsize=10, color=INK, pad=22, loc="left")

    fig.subplots_adjust(left=0.068, right=0.99, top=0.91, bottom=0.11)
    fig.savefig(OUT / "demo_scales.png", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / "demo_scales.png").relative_to(REPO))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    clip_figures()
    duration_figure()
    scale_figure()
