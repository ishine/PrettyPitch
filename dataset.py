"""Dataset + batching for the PrettyPitch F0 model.

Reads the `shard.npz` produced by `preprocess` — each phrase carries the **raw**
phoneme ids, per-phoneme durations, F0 (`f0_interp`) and voiced flag. Everything
opinionated is derived here, per item:

  1. nucleus representative values (continuous MIDI, `prettypitch.pitch`)
  2. baseline = staircase of those values, Gaussian-smoothed (`prettypitch.baseline`)
  3. plain / vibrato decomposition of `log2(f0) − baseline` (`prettypitch.decomp`)
  4. model pitch input: one normalized note-pitch value, forward-filled from each nucleus
     to every phoneme (staircase; no has-flag, no nucleus/consonant split)

Speaker / style ids are assigned per-DB at train time via `spk_map` / `style_map`
(keyed by the DB folder name) — never baked into the shard. So nucleus sets, Blaauw
weights and speaker mappings can all change without re-preprocessing.

The reconstruction invariant `baseline + plain + vibrato ≡ log2(f0)` holds exactly.
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from prettypitch.pitch import nucleus_values
from prettypitch.baseline import (build_baseline, phoneme_frame_bounds, fill_values,
                                   GAUSSIAN_SIGMA_SEC)
from prettypitch.decomp import (
    make_vibrato_fir, extract_vibrato, compress_residual, make_vibrato_gain,
    filtfilt_min_len,
    VIB_COMP_METHOD, VIB_COMP_THRESH_ST, GAIN_MARGIN_SEC, GAIN_SIGMA_SEC,
)

PAU_AUG_MAX_SEC = 0.50    # train-only: extend the edge pau by up to this much (each side)

# defaults for the pitch-input normalization: input = (midi − center) / scale.
# Centered on the singing range, one unit per octave — matches the semitone geometry
# of the targets (dividing by 127 would compress the melody into a ~0.2 span).
PITCH_CENTER = 60.0
PITCH_SCALE = 12.0


def _song_of(stem: str) -> str:
    """Phrase name '{song}_{idx:04d}' -> song name (drop the trailing _NNNN)."""
    return re.sub(r"_\d{4,}$", "", stem)


def _open(path: str, cache: dict):
    """Lazily open (and cache per process) an npz shard. Fork-safe (pid-keyed)."""
    key = (os.getpid(), str(path))
    if key not in cache:
        cache[key] = np.load(path) if os.path.exists(path) else None
    return cache[key]


class PrettyPitchF0Dataset(Dataset):
    def __init__(self, dirs, split: str = "train", eval_songs: int = 2,
                 eval_dbs=None,
                 min_sec: float = 1.7, seed: int = 42, pau_aug: bool = False,
                 spk_map: dict | None = None, style_map: dict | None = None,
                 nucleus_names=None,             # iterable[str]; required
                 pitch_cfg: dict | None = None,  # nucleus_values kwargs + center/scale
                 decomp_cfg: dict | None = None,
                 reject_f0_below: float | None = None,      # Hz。この値未満の有声フレームを
                 reject_f0_per_db: dict | None = None):     # 1つでも含む句を落とす（DB別上書き）
        self.split = split
        self.pau_aug = pau_aug
        self.spk_map = spk_map
        self.style_map = style_map
        self._cache: dict = {}
        self._values_cache: dict = {}     # idx -> nucleus values (deterministic, pre-pau_aug)
        self._devib_cache: dict = {}      # idx -> representative from the vibrato-removed F0
        self.files: list = []             # (shard_path, phrase_name, db_dir)
        self.frame_counts: List[int] = []

        pc = dict(pitch_cfg or {})
        # 代表値ハイパラの DB 別上書き（pitch.per_db: {db名: {key: value}}）。レシピが書く。
        self._pitch_per_db: dict = dict(pc.pop("per_db", None) or {})
        self._pitch_resolved: dict = {}           # dbname -> (devib: bool, kwargs: dict)
        # 代表値をビブラート除去後の F0 から採る（2パス）。False = 生 F0（従来）
        self.devib_representative = bool(pc.pop("devib_representative", False))
        self.pitch_center = float(pc.pop("center", PITCH_CENTER))
        self.pitch_scale = float(pc.pop("scale", PITCH_SCALE))
        self.pitch_kwargs = pc            # n_iter / sigma_st / sg_window_ms / min_voiced / devoiced_fill

        # 分解パラメータ: 既定 + DB別上書き(decomp.per_db: {db名: {key: value}})。
        # 分解は全て __getitem__ 時にDB別設定で行うので、DB毎に調整しても再preprocess不要。
        dc = dict(decomp_cfg or {})
        self._decomp_per_db: dict = dict(dc.pop("per_db", None) or {})
        self._decomp_default: dict = dc
        self._decomp_resolved: dict = {}     # dbname -> (merged dict, FIR)
        self._min_sec = float(min_sec)

        # ── F0 下限による句の除外（既定 = 無効）─────────────────────────────
        # 抽出器はフレーズ末尾の消え際で、声が減衰していく残響へ食いついて実際より
        # 低い F0 を返すことがある。ritsu では 100Hz 未満 2 句がこれで、どちらも
        # ロングトーンの終端（210Hz → 79Hz へ 1-2 フレームで垂直に落ちる）だった。
        # 閾値は話者の音域で決まるので DB 別に持つ。女声 100Hz で落ちるのは 0-2 句、
        # 男声(natsume)に同じ 100Hz を当てると 15% が落ちるため 60Hz にする。
        # シャードには触らないので、閾値を変えても再 preprocess は要らない。
        self._reject_below = None if reject_f0_below is None else float(reject_f0_below)
        self._reject_per_db = {str(k): float(v) for k, v in (reject_f0_per_db or {}).items()}

        meta0 = None
        frame_rate = None
        n_dropped_no_nucleus = 0
        n_dropped_low_f0: dict = {}
        for d in dirs:
            d = Path(d)
            dbname = os.path.basename(os.path.normpath(str(d)))
            lo_hz = self._reject_per_db.get(dbname, self._reject_below)
            meta = json.load(open(d / "metadata.json", encoding="utf-8"))
            if meta0 is None:
                meta0 = meta
            else:
                for k in ("frame_rate", "sample_rate", "hop_size", "n_phonemes"):
                    assert meta.get(k) == meta0.get(k), f"{d}: metadata {k} mismatch"
                assert meta.get("phoneme_vocab") == meta0.get("phoneme_vocab"), \
                    f"{d}: phoneme_vocab mismatch"
            frame_rate = float(meta["frame_rate"])
            phrases = meta["phrases"]
            names = sorted(phrases)
            songs = sorted({_song_of(n) for n in names})
            # eval_dbs を指定したら、その DB からだけ eval 用の曲を抜く（他は全曲 train）。
            # 1 話者あたりの曲数が少ないコーパスで、全 DB から抜くと hold-out が
            # 大きくなりすぎるのを避けるため。未指定なら従来どおり全 DB が対象。
            if eval_dbs is not None and os.path.basename(str(d)) not in eval_dbs:
                n_hold = 0
            else:
                n_hold = min(eval_songs, max(0, len(songs) - 1))  # keep >=1 song in train
            hold = set(random.Random(seed).sample(songs, n_hold)) if n_hold else set()
            shard = str(d / "shard.npz")
            z = _open(shard, self._cache)
            nuc_ids = frozenset(meta["phoneme_vocab"][p] for p in nucleus_names
                                if p in meta["phoneme_vocab"])
            for n in names:
                is_eval = _song_of(n) in hold
                if (split == "eval") != is_eval:
                    continue
                T = int(phrases[n])
                if min_sec and T < min_sec * frame_rate:
                    continue
                # a phrase with no nucleus phoneme has no pitch teacher at all -> drop
                if not any(int(p) in nuc_ids for p in z[f"{n}|phoneme_ids"]):
                    n_dropped_no_nucleus += 1
                    continue
                if lo_hz is not None:
                    f0 = z[f"{n}|f0_interp"]
                    v = z[f"{n}|uv"] > 0.5
                    if v.any() and float(f0[v & (f0 > 0)].min(initial=np.inf)) < lo_hz:
                        n_dropped_low_f0[dbname] = n_dropped_low_f0.get(dbname, 0) + 1
                        continue
                self.files.append((shard, n, str(d)))
                self.frame_counts.append(T)

        assert meta0 is not None and self.files, f"no phrases for split={split} in {dirs}"
        self.meta = meta0
        self.frame_rate = float(frame_rate)
        self.n_phonemes = int(meta0["n_phonemes"])
        self.nucleus_ids = frozenset(meta0["phoneme_vocab"][p] for p in nucleus_names
                                     if p in meta0["phoneme_vocab"])
        self.pau_id = int(meta0["phoneme_vocab"].get("pau", 0))
        self._fir = self._decomp_for("__default__")[1]      # 既定FIR(表示用)
        for db in sorted(self._decomp_per_db):
            self._decomp_for(db)                            # 上書きDBのFIR検証を先に走らせる

        # per-phrase balancing keys: speaker id (styles of one singer collapse) / db folder
        self.speaker_ids = [int(self.spk_map.get(os.path.basename(os.path.normpath(db)), 0))
                            if self.spk_map else 0 for (_s, _n, db) in self.files]
        self.db_ids = [os.path.basename(os.path.normpath(db)) for (_s, _n, db) in self.files]
        print(f"[dataset] {split}: {len(self.files)} phrases  frame_rate={self.frame_rate:.3f}"
              f"  FIR={len(self._fir)}tap"
              + (f"  (dropped {n_dropped_no_nucleus} nucleus-free)" if n_dropped_no_nucleus else ""))
        if n_dropped_low_f0:
            # 落とした数は必ず出す。黙って捨てると「なぜ句が減ったか」が追えない。
            detail = "  ".join(f"{db} {c}句(<{self._reject_per_db.get(db, self._reject_below):.0f}Hz)"
                               for db, c in sorted(n_dropped_low_f0.items()))
            print(f"[dataset] {split}: F0下限で除外 {sum(n_dropped_low_f0.values())} 句 — {detail}")

    def __len__(self) -> int:
        return len(self.files)

    def _decomp_for(self, dbname: str):
        """DB別に解決した分解設定と FIR を返す（既定 + decomp.per_db[db] のマージ）。"""
        if dbname not in self._decomp_resolved:
            d = {**self._decomp_default, **(self._decomp_per_db.get(dbname) or {})}
            fir_kw = {}
            if d.get("band") is not None:
                fir_kw["band"] = tuple(d["band"])
            if d.get("fir_time_sec") is not None:
                fir_kw["time_len_sec"] = float(d["fir_time_sec"])
            fir = make_vibrato_fir(self.frame_rate, **fir_kw)
            min_len = filtfilt_min_len(fir) / self.frame_rate
            if self._min_sec and min_len > self._min_sec:
                print(f"[dataset] WARNING: {dbname}: filtfilt最小長 {min_len:.2f}s > "
                      f"min_sec {self._min_sec}s — 短いフレーズは vibrato 全ゼロになります")
            if self._decomp_per_db.get(dbname):
                print(f"[dataset] decomp per_db[{dbname}]: "
                      f"{self._decomp_per_db[dbname]}")
            self._decomp_resolved[dbname] = (d, fir)
        return self._decomp_resolved[dbname]

    def _pitch_for(self, dbname: str):
        """DB別に解決した代表値設定 (devib_representative, nucleus_values の kwargs) を返す。
        推論側は代表値を使わない（音高は譜面から来る）ので、DB 別にしても ckpt と矛盾しない。"""
        if dbname not in self._pitch_resolved:
            ov = dict(self._pitch_per_db.get(dbname) or {})
            devib = bool(ov.pop("devib_representative", self.devib_representative))
            ov.pop("center", None)               # 正規化は全 DB 共通（ckpt が1つだけ持つ）
            ov.pop("scale", None)
            kw = {**self.pitch_kwargs, **ov}
            if self._pitch_per_db.get(dbname):
                print(f"[dataset] pitch per_db[{dbname}]: {self._pitch_per_db[dbname]}")
            self._pitch_resolved[dbname] = (devib, kw)
        return self._pitch_resolved[dbname]

    def _nucleus_values(self, i, ph_ids, dur_sec, f0_hz, uv, pitch_kw=None):
        if i not in self._values_cache:
            self._values_cache[i] = nucleus_values(
                ph_ids, dur_sec, f0_hz, uv, self.frame_rate, self.nucleus_ids,
                **(self.pitch_kwargs if pitch_kw is None else pitch_kw))
        return self._values_cache[i]

    def _decomp_closures(self, dcfg, fir, ph_ids, dur_sec, f0_hz, uv, T, fb):
        """vibrato gain と baseline/vibrato のクロージャを返す（__getitem__ / phrase_pitch 共有）。"""
        baseline_sigma = float(dcfg.get("baseline_sigma_sec", GAUSSIAN_SIGMA_SEC))
        is_nucleus = np.zeros(T, dtype=np.float32)
        phf = np.zeros(T, dtype=np.int64)
        for k in range(len(ph_ids)):
            phf[fb[k]:fb[k + 1]] = ph_ids[k]
            if int(ph_ids[k]) in self.nucleus_ids:
                is_nucleus[fb[k]:fb[k + 1]] = 1.0
        gain = make_vibrato_gain(phf, is_nucleus > 0.5, dur_sec, self.frame_rate, T,
                                 margin_sec=float(dcfg.get("gain_margin_sec", GAIN_MARGIN_SEC)),
                                 sigma_sec=float(dcfg.get("gain_sigma_sec", GAIN_SIGMA_SEC)),
                                 cons_margin_pre_sec=dcfg.get("cons_margin_pre_sec"),
                                 cons_margin_post_sec=dcfg.get("cons_margin_post_sec"),
                                 boundary_margin_pre_sec=dcfg.get("boundary_margin_pre_sec"),
                                 boundary_margin_post_sec=dcfg.get("boundary_margin_post_sec"),
                                 boundary_mode=dcfg.get("boundary_mode", "all"),
                                 limit_margin_to_adjacent=bool(
                                     dcfg.get("limit_margin_to_adjacent", True)),
                                 min_open_sec=float(dcfg.get("min_open_sec", 0.0)))

        def baseline_of(vals):
            bl = build_baseline(vals, dur_sec, T, self.frame_rate, sigma_sec=baseline_sigma)
            if bl is None:                      # despite the init filter — no value at all
                bl = np.full(T, float(np.median(np.log2(np.maximum(f0_hz, 1.0)))), dtype=np.float32)
            return bl

        def vibrato_of(bl):                     # bl 基準の (vibrato[T], residual[T])
            resid = (np.log2(np.maximum(f0_hz, 1.0)) - bl).astype(np.float32)
            resid_c = compress_residual(resid,
                                        method=dcfg.get("comp_method", VIB_COMP_METHOD),
                                        thresh_st=float(dcfg.get("comp_thresh_st", VIB_COMP_THRESH_ST)),
                                        param=dcfg.get("comp_param", None))
            return (extract_vibrato(resid_c, uv, fir) * gain).astype(np.float32), resid

        return is_nucleus, baseline_of, vibrato_of

    def _representative(self, i, ph_ids, dur_sec, f0_hz, uv, baseline_of, vibrato_of,
                        dbname: str = "__default__"):
        """nucleus 代表値（devib_representative 時はビブラート除去後の F0 から, 2パス）。"""
        devib, pkw = self._pitch_for(dbname)
        values = self._nucleus_values(i, ph_ids, dur_sec, f0_hz, uv, pkw)      # pass 1（生 F0）
        if devib:
            if i not in self._devib_cache:
                vibrato0, _ = vibrato_of(baseline_of(values))                # 暫定ビブラート
                f0_dv = np.where(f0_hz > 1.0,
                                 np.exp2(np.log2(np.maximum(f0_hz, 1.0)) - vibrato0),
                                 f0_hz).astype(np.float32)                    # ビブラート除去 F0
                self._devib_cache[i] = nucleus_values(
                    ph_ids, dur_sec, f0_dv, uv, self.frame_rate, self.nucleus_ids,
                    **pkw)                                                    # pass 2（代表値）
            values = self._devib_cache[i]
        return values

    def phrase_pitch(self, i: int):
        """1フレーズの (ph_ids, dur_sec[sec], 代表値[continuous MIDI, NaN=非核]) を返す。
        代表値は __getitem__ と同一手順（devib_representative 対応）。F0 の代表値だけ要る
        用途（子音長モデル等）向けの軽量アクセサ。"""
        shard, name, dbdir = self.files[i]
        z = _open(shard, self._cache)
        ph_ids = z[f"{name}|phoneme_ids"].astype(np.int64)
        dur_sec = z[f"{name}|dur_sec"].astype(np.float32)
        f0_hz = z[f"{name}|f0_interp"].astype(np.float32)
        uv = z[f"{name}|uv"].astype(np.float32)
        dbname = os.path.basename(os.path.normpath(dbdir))
        T = len(f0_hz)
        fb = phoneme_frame_bounds(dur_sec, self.frame_rate, T)
        dcfg, fir = self._decomp_for(dbname)
        _, baseline_of, vibrato_of = self._decomp_closures(
            dcfg, fir, ph_ids, dur_sec, f0_hz, uv, T, fb)
        values = self._representative(i, ph_ids, dur_sec, f0_hz, uv, baseline_of, vibrato_of,
                                     dbname)
        return ph_ids, dur_sec, values

    def __getitem__(self, i: int) -> dict:
        shard, name, dbdir = self.files[i]
        z = _open(shard, self._cache)

        def g(k):
            return z[f"{name}|{k}"]

        ph_ids = g("phoneme_ids").astype(np.int64)
        dur_sec = g("dur_sec").astype(np.float32)
        f0_hz = g("f0_interp").astype(np.float32)
        uv = g("uv").astype(np.float32)
        dbname = os.path.basename(os.path.normpath(dbdir))
        spk_id = int(self.spk_map.get(dbname, 0)) if self.spk_map else 0
        style_id = int(self.style_map.get(dbname, 0)) if self.style_map else 0

        T = len(f0_hz)
        fb = phoneme_frame_bounds(dur_sec, self.frame_rate, T)
        ph_durs = np.maximum(np.diff(fb), 1).astype(np.int64)

        dcfg, fir = self._decomp_for(dbname)     # DB別の分解設定(per_db 上書き込み)

        # 代表値 + plain/vibrato 分解（invariant: baseline+plain+vibrato == log2 f0）。
        # gain/クロージャ/2パス代表値は phrase_pitch と共有（_decomp_closures/_representative）。
        is_nucleus, baseline_of, vibrato_of = self._decomp_closures(
            dcfg, fir, ph_ids, dur_sec, f0_hz, uv, T, fb)
        values = self._representative(i, ph_ids, dur_sec, f0_hz, uv, baseline_of, vibrato_of,
                                     dbname)
        baseline = baseline_of(values)
        vibrato, residual = vibrato_of(baseline)
        plain = (residual - vibrato).astype(np.float32)
        gt_f0_hz_full = f0_hz.copy()

        # 4. model pitch input: forward-filled note-pitch staircase (every phoneme, no has)
        filled = fill_values(values)
        ph_pitch = np.where(np.isfinite(filled),
                            (filled - self.pitch_center) / self.pitch_scale, 0.0).astype(np.float32)

        # pau_aug (train only): extend the edge pau by random silence — the data always
        # starts/ends with pau. baseline/f0 edge-extend; plain/vib/uv zero-pad.
        if self.pau_aug and ph_ids[0] == self.pau_id and ph_ids[-1] == self.pau_id:
            mx = int(round(PAU_AUG_MAX_SEC * self.frame_rate))
            n_pre, n_post = random.randint(0, mx), random.randint(0, mx)
            if n_pre or n_post:
                ph_durs = ph_durs.copy()
                ph_durs[0] += n_pre
                ph_durs[-1] += n_post
                ed = lambda a: np.concatenate([np.full(n_pre, a[0], a.dtype), a,
                                               np.full(n_post, a[-1], a.dtype)])
                zz = lambda a: np.concatenate([np.zeros(n_pre, a.dtype), a,
                                               np.zeros(n_post, a.dtype)])
                baseline = ed(baseline)
                gt_f0_hz_full = ed(gt_f0_hz_full)
                plain = zz(plain)
                vibrato = zz(vibrato)
                uv = zz(uv)
                is_nucleus = zz(is_nucleus)

        return {
            "ph_ids": ph_ids, "ph_durs": ph_durs,
            "ph_pitch": ph_pitch,
            "spk_id": np.int64(spk_id), "style_id": np.int64(style_id),
            "baseline": baseline.astype(np.float32),
            "plain": plain, "vibrato": vibrato,
            "is_nucleus": is_nucleus, "uv": uv,
            "gt_f0_hz_full": gt_f0_hz_full,
            "item_name": name,
        }

    def __getstate__(self):                          # drop open npz handles across fork/spawn
        s = self.__dict__.copy()
        s["_cache"] = {}
        return s


def f0_collate_fn(batch: list) -> dict:
    B = len(batch)
    max_L = max(len(b["ph_ids"]) for b in batch)
    max_T = max(len(b["plain"]) for b in batch)

    def pad_ph(key, dtype):
        o = torch.zeros(B, max_L, dtype=dtype)
        for i, b in enumerate(batch):
            t = torch.as_tensor(b[key], dtype=dtype)
            o[i, :len(t)] = t
        return o

    def pad_fr(key, dtype=torch.float32):
        o = torch.zeros(B, max_T, dtype=dtype)
        for i, b in enumerate(batch):
            t = torch.as_tensor(b[key], dtype=dtype)
            o[i, :len(t)] = t
        return o

    ph_mask = torch.ones(B, max_L, dtype=torch.bool)      # True = padding
    frame_mask = torch.ones(B, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        ph_mask[i, :len(b["ph_ids"])] = False
        frame_mask[i, :len(b["plain"])] = False

    return {
        "ph_ids": pad_ph("ph_ids", torch.long),
        "ph_durs": pad_ph("ph_durs", torch.long),
        "ph_pitch": pad_ph("ph_pitch", torch.float32),
        "spk_id": torch.tensor([int(b["spk_id"]) for b in batch], dtype=torch.long),
        "style_id": torch.tensor([int(b["style_id"]) for b in batch], dtype=torch.long),
        "baseline": pad_fr("baseline"),
        "plain": pad_fr("plain"),
        "vibrato": pad_fr("vibrato"),
        "is_nucleus": pad_fr("is_nucleus"),
        "uv": pad_fr("uv"),
        "gt_f0_hz_full": pad_fr("gt_f0_hz_full"),
        "ph_mask": ph_mask, "frame_mask": frame_mask,
        "item_name": [b["item_name"] for b in batch],
    }


class FrameBasedBatchSampler(Sampler):
    """Length-sorted batches capped by total frames (B*maxT) and item count.

    With `weights` (per-phrase sampling weights), each epoch draws a weighted sample
    WITH replacement before packing — used to balance speakers/DBs (rare voices get
    oversampled so every voice trains equally often).
    """

    def __init__(self, frame_counts, max_frames: int, max_items: int,
                 shuffle: bool = True, seed: int = 0, weights=None):
        self.frame_counts = list(frame_counts)
        self.max_frames = max_frames
        self.max_items = max_items
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.weights = None if weights is None else torch.as_tensor(weights, dtype=torch.float)
        self._batches = self._build(range(len(self.frame_counts)))

    def _build(self, indices):
        order = sorted(indices, key=lambda i: self.frame_counts[i])
        batches, cur, cur_max = [], [], 0
        for idx in order:
            fc = self.frame_counts[idx]
            nmax = max(cur_max, fc)
            if cur and (nmax * (len(cur) + 1) > self.max_frames or len(cur) + 1 > self.max_items):
                batches.append(cur)
                cur, cur_max = [], 0
            cur.append(idx)
            cur_max = max(cur_max, fc)
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        if self.weights is not None:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            idxs = torch.multinomial(self.weights, len(self.frame_counts),
                                     replacement=True, generator=g).tolist()
            batches = self._build(idxs)
        else:
            batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return len(self._batches)
