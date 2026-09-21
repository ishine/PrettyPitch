"""Slim a training checkpoint for distribution: keep the model and its metadata only.

    python -m tools.slim_ckpt --ckpt log/<run>/ckpt_010000.pt --out checkpoints/f0_3singer.pt
    python -m tools.slim_ckpt --ckpt logs/<run>/best.pt --out checkpoints/cons_dur_3singer.pt

Works for both checkpoint kinds (the F0 model and the consonant-duration model): it drops
the training state — discriminator, optimizers, schedulers, the training argparse namespace —
and keeps ``model``, the self-describing ``config`` and the small scalar metadata
(``step`` / ``epoch`` / ``arch`` / ``val_mae_ms`` …), so infer.py, svs and the exporters
rebuild the model from the slim file alone.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Training state that is never needed for inference.
DROP_KEYS = {"D", "optG", "optD", "schedG", "schedD", "optimizer", "scheduler", "args", "val_loss"}


def slim(ck: dict) -> dict:
    for key in ("model", "config"):
        if key not in ck:
            raise SystemExit(f"missing {key!r} — not a PrettyPitch checkpoint?")
    out = {}
    for k, v in ck.items():
        if k in DROP_KEYS:
            continue
        if k in ("model", "config") or isinstance(v, (int, float, str, bool, type(None))):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Slim a PrettyPitch checkpoint for distribution.")
    ap.add_argument("--ckpt", required=True, help="training checkpoint")
    ap.add_argument("--out", required=True, help="output path for the slim checkpoint")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    out_ck = slim(ck)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ck, out)
    src_mb = Path(args.ckpt).stat().st_size / 1e6
    dst_mb = out.stat().st_size / 1e6
    meta = {k: v for k, v in out_ck.items() if k not in ("model", "config")}
    print(f"{args.ckpt} ({src_mb:.2f} MB) -> {out} ({dst_mb:.2f} MB)  keys={sorted(out_ck)}  {meta}")


if __name__ == "__main__":
    main()
