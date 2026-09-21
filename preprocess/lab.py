"""Database-agnostic lab -> per-phrase training-data pipeline (score-free, mel-free).

Input per song: a phoneme `.lab` label file (`start end phoneme`) aligned to a `.wav`.
Output: per-phrase `.npz` files with phoneme ids, durations, F0 (RMVPE) and the voiced
flag — everything the PrettyPitch F0 model trains on. Phoneme timing comes straight from
the lab; **there is no musical score anywhere in this pipeline**. Note-pitch information
is derived later, in the dataset loader, from the F0 itself (nucleus representative
values), so an out-of-tune singer is perfectly fine training data.

Frame settings come from a `FrameSpec` (default hop 256 @ 44100 Hz). F0 uses RMVPE only,
with the single-frame despike always on and the octave-error fix on by default (PrettyPitch
has no multi-estimator consensus, and teacher pitch accuracy is what the model learns).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prettypitch.config import FrameSpec
from .vocab import DEFAULT_VOCAB
from .phrase_cut import (find_phrase_spans, cut_phrases, frame_loudness_db,
                         build_sil_mask, clip_pau_gain, gain_to_samples)
from .f0_rmvpe import extract_f0_rmvpe

# Non-canonical silence tokens -> `pau`. Defined by the **language pack**
# (`dict/<lang>.yaml`, key `sil_map`) — not hardcoded here, so a new language only needs
# data files. `load_lab(sil_map=...)` still overrides per call, and a recipe can add to it.
def _default_sil_map() -> dict:
    try:
        from duration_model import phoneset as _ps_mod
        m = dict(_ps_mod.active().sil_map)
        if m:
            return m
    except Exception:                                   # noqa: BLE001
        pass
    # Last-resort fallback if the language pack cannot be read.
    return {"sil": "pau", "br": "pau", "sp": "pau", "sli": "pau", "vf": "pau"}


DEFAULT_SIL_MAP = _default_sil_map()


def read_lab(path: str, unit: str = "100ns"):
    """Read a phoneme `.lab` -> list of (start_sec, end_sec, phoneme).

    unit='100ns' (HTS integer) or 'sec' (decimal seconds).
    """
    scale = 1e-7 if unit == "100ns" else 1.0
    segs = []
    for ln in open(path, encoding="utf-8", errors="ignore"):
        t = ln.split()
        if len(t) >= 3:
            s, e = float(t[0]) * scale, float(t[1]) * scale
            if e > s:
                segs.append((s, e, t[2]))
    return segs


def load_lab(lab_path: str, *, lab_unit: str = "100ns", sil_map: dict | None = None,
             phon_norm: dict | None = None, lab_fix=None, strict: bool = True, vocab=None):
    """Read + clean a `.lab` into a gap-free `(start_sec, end_sec, phoneme)` timeline:
    normalise phonemes (phon_norm, then silence map -> `pau`), insert `pau` rows over gaps
    (continuity), and merge adjacent `pau`. No score, no note — timing is the lab's own.

    lab_fix: optional per-singer `(intervals)->intervals` hook (a database recipe correction).
    strict=True raises on any phoneme outside the vocab (no silent fallback).
    vocab: the phoneme set to validate against (default: the bundled Japanese vocab).
    """
    vocab = vocab or DEFAULT_VOCAB
    sil_map = {**_default_sil_map(), **(sil_map or {})}   # 言語パックを都度読む
                                                          # （set_default 後も追従するため）
    phon_norm = phon_norm or {}
    segs = read_lab(lab_path, lab_unit)
    if lab_fix is not None:
        segs = lab_fix(segs)
    rows, prev_e, unknown = [], 0.0, set()
    for s, e, ph in segs:
        ph = phon_norm.get(ph, sil_map.get(ph, ph))
        if ph not in vocab:
            unknown.add(ph)
            continue
        if s > prev_e + 1e-6:                                # gap -> pau row (continuity)
            rows.append((prev_e, s, "pau"))
        rows.append((s, e, ph))
        prev_e = e
    merged = []                                              # merge adjacent pau rows
    for r in rows:
        if merged and r[2] == "pau" and merged[-1][2] == "pau" and abs(merged[-1][1] - r[0]) < 1e-9:
            merged[-1] = (merged[-1][0], r[1], "pau")
        else:
            merged.append(r)
    if strict and unknown:
        raise ValueError(f"{lab_path}: phonemes not in canonical vocab: {sorted(unknown)} "
                         f"(add them to phon_norm/sil_map or the vocab)")
    return merged


def process_file(
    name: str,
    lab_path: Path,
    wav_path: Path,
    out_dir: Path,
    *,
    frames: FrameSpec | None = None,
    f0_device: str = "cpu",
    f0_octave_fix: bool = True,          # median-window octave-error fix (see f0_rmvpe.py)
    lufs_target: float = -23.0,
    exclude_spans: list | None = None,   # [[s_sec, e_sec], ...] drop overlapping phrases
    quiet: bool = False,
    lab_unit: str = "100ns",
    sil_map: dict | None = None,
    phon_norm: dict | None = None,
    lab_fix=None,
    vocab=None,                          # phoneme set (default: bundled Japanese vocab)
    silence_pau: bool = True,            # 静音区間を無音化してから F0 を抽出（既定 ON）
    silence_fade_sec: float = 0.05,      # その両端フェード長 [s]
) -> bool:
    """Turn one (lab, wav) pair into per-phrase npz files under `out_dir/phrases/`."""
    import librosa
    import pyloudnorm as pyln

    frames = frames or FrameSpec()
    vocab = vocab or DEFAULT_VOCAB
    sr, hop, frame_rate = frames.sr, frames.hop, frames.frame_rate

    phrase_dir = Path(out_dir) / "phrases"
    if list(phrase_dir.glob(f"{name}_*.npz")):
        if not quiet:
            print("  [SKIP] already done")
        return True

    # ── read + clean the .lab (phoneme timeline; no score, no note) ─────────────
    lab_rows = load_lab(str(lab_path), lab_unit=lab_unit, sil_map=sil_map,
                        phon_norm=phon_norm, lab_fix=lab_fix, vocab=vocab)
    ph_list, dur_sec_l, lab_segs = [], [], []
    for s, e, ph in lab_rows:
        ph_list.append(ph)
        dur_sec_l.append(e - s)
        lab_segs.append((s, e, ph))
    if not ph_list:
        if not quiet:
            print("  [SKIP] empty phoneme sequence")
        return False

    # ── load audio + LUFS normalise (whole song once) ───────────────────────────
    audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    meter = pyln.Meter(sr)
    cur_lufs = meter.integrated_loudness(audio.astype(float))
    gain = (10.0 ** ((lufs_target - cur_lufs) / 20.0)
            if np.isfinite(cur_lufs) and cur_lufs > -70.0 else 1.0)
    audio = (audio * gain).astype(np.float32)

    # ── phrase spans (cut on long silence) ──────────────────────────────────────
    T_frames  = int(len(audio) // hop)
    total_sec = T_frames / frame_rate
    spans = find_phrase_spans(lab_segs, total_sec, hop_size=hop, sr=sr)
    if not spans:
        spans = [(0, T_frames, 0, 0)]

    if exclude_spans:
        def _hits(sp):
            s_sec, e_sec = sp[0] / frame_rate, sp[1] / frame_rate
            return any(not (e_sec <= xs or s_sec >= xe) for xs, xe in exclude_spans)
        n0 = len(spans)
        spans = [sp for sp in spans if not _hits(sp)]
        if not quiet and len(spans) < n0:
            print(f"  [EXCLUDE] {n0 - len(spans)} phrase(s) dropped")
        if not spans:
            return True

    # ── 静音区間(pau / 長い cl)のフレームマスク ───────────────────────────────
    # ラベル上 pau でも息継ぎが入っており、抽出器がそれを有声と判定して歌唱より高い F0 を
    # 返す。**F0 を後から潰すのでは足りない**（息継ぎは抽出器の推定そのものを歪める）ので、
    # 音声を無音化してから RMVPE に渡す。静音の定義は phrase_cut の _is_silence と同一。
    sil_mask = (build_sil_mask(lab_segs, T_frames, frame_rate)
                if silence_pau else np.zeros(T_frames, dtype=bool))
    fade_frames = int(round(max(0.0, silence_fade_sec) * frame_rate))

    def _span_data(s_fr: int, e_fr: int):
        n = e_fr - s_fr
        clip = audio[s_fr * hop: e_fr * hop]
        # 無音句ゲートは **無音化前** の音量で判定する。ゲートの役目は「ラベルは歌っている
        # と言っているのに音がない」ズレの検出なので、こちらが意図的に消した分を数えると
        # 意味が変わる。
        loud_db = frame_loudness_db(clip, hop, n)         # silent-phrase gate (mel substitute)

        core = None
        if silence_pau and sil_mask[s_fr:e_fr].any():
            g = clip_pau_gain(sil_mask[s_fr:e_fr], fade_frames)
            clip = clip * gain_to_samples(g, hop, len(clip))
            core = g <= 1e-6                              # 完全に消えたフレーム

        # F0 (RMVPE only; raw 0-in-unvoiced — cut_phrases fills gaps per phrase)
        f0_clip, uv_clip = extract_f0_rmvpe(
            np.clip(clip, -1.0, 1.0), sr, hop,
            device=f0_device,
            octave_fix=f0_octave_fix, interpolate=False)
        f0_clip = (f0_clip[:n] if len(f0_clip) >= n
                   else np.pad(f0_clip, (0, n - len(f0_clip)))).astype(np.float32)
        uv_clip = (uv_clip[:n] if len(uv_clip) >= n
                   else np.pad(uv_clip, (0, n - len(uv_clip)))).astype(np.float32)
        if core is not None:
            # 完全無音に対する抽出器の出力は保証されないので、core だけは明示的に無声にする。
            # フェードの両裾は触らない（尾曳き・歌い出しは歌唱の一部）。
            f0_clip[core] = 0.0
            uv_clip[core] = 0.0
        return f0_clip, uv_clip, loud_db

    ph_ids_arr   = np.array([vocab.get(p, 0) for p in ph_list], dtype=np.int16)
    dur_sec_arr  = np.array(dur_sec_l,  dtype=np.float32)

    if not quiet:
        print(f"  {len(spans)} spans (per-phrase F0 @{sr}/{hop}) ...")
    n_written = cut_phrases(
        name=name, lab_segs=lab_segs, ph_ids=ph_ids_arr, dur_sec=dur_sec_arr,
        out_dir=phrase_dir, span_data=_span_data, total_frames=T_frames,
        hop_size=hop, sr=sr,
        pau_id=vocab.pau_id, spans=spans, verbose=True,
    )
    if not quiet:
        print(f"  -> {n_written} phrases  ({len(ph_list)} phonemes, {T_frames} frames)")
    return True
