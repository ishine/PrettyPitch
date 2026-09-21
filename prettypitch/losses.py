"""Losses for the 3ch F0 model: multi-resolution STFT magnitude L1 + Δ/ΔΔ kernels.

The STFT loss supervises the vibrato channel phase-insensitively — plain L1 on an
oscillation averages it away, magnitude spectra do not. Configs are designed in seconds
(0.68 / 1.37 / 2.73 s windows) and converted to the actual frame rate, rounded to powers
of two: @172.265625 fps → n_fft [128, 256, 512], hop = n_fft/4.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss-term weights for the plain channel's Δ/ΔΔ regularizers (measured so the weighted
# contributions roughly match the value L1: x0≈0.7, delta≈0.06, delta-delta≈0.06).
W_DELTA = 10.0
W_DELTA_DELTA = 12.0


def stft_configs_for(frame_rate: float):
    """Preserve the physical window lengths of the reference configs (n_fft 128/256/512
    @187.5 fps = 0.68/1.37/2.73 s), convert to `frame_rate`, round to powers of two."""
    cfgs = []
    for n_fft_ref in (128, 256, 512):
        sec = n_fft_ref / 187.5
        n_fft = 2 ** int(round(math.log2(sec * frame_rate)))
        cfgs.append({'n_fft': n_fft, 'hop_length': max(1, n_fft // 4)})
    return cfgs


class STFTMagnitudeLoss(nn.Module):
    """Multi-resolution STFT magnitude L1 for 1-D F0 residual signals."""

    def __init__(self, configs: list):
        super().__init__()
        self.configs = configs

    def _stft(self, x: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        """x: [B, T] → complex STFT [B, n_fft//2+1, n_frames]"""
        pad = n_fft // 2
        x = F.pad(x, (pad, pad))
        window = torch.hann_window(n_fft, device=x.device)
        return torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                          win_length=n_fft, window=window, return_complex=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask.float()
            target = target * mask.float()
        total = pred.new_zeros(())
        for cfg in self.configs:
            s_pred = self._stft(pred, cfg['n_fft'], cfg['hop_length'])
            s_target = self._stft(target, cfg['n_fft'], cfg['hop_length'])
            total = total + (s_pred.abs() - s_target.abs()).abs().mean()
        return total / len(self.configs)


# ── Δ / ΔΔ FIR kernels (nnmnkwii-compatible) ─────────────────────────────────
#   delta: [-0.5, 0, +0.5] (central difference), delta-delta: [+1, -2, +1].
#   'same' padding; exclude the edge frames via _delta_valid.

_KERNELS: dict = {}


def _get_delta_kernels(device):
    if device not in _KERNELS:
        _KERNELS[device] = (
            torch.tensor([-0.5, 0.0, 0.5], device=device).view(1, 1, 3),
            torch.tensor([1.0, -2.0, 1.0], device=device).view(1, 1, 3),
        )
    return _KERNELS[device]


def _apply_delta(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """x: [B, T] → conv1d along T → [B, T]"""
    return F.conv1d(x.unsqueeze(1), kernel, padding=1).squeeze(1)


def _delta_valid(valid: torch.Tensor) -> torch.Tensor:
    """Valid mask with the padded edge frames removed."""
    v = valid.clone()
    v[:, 0] = False
    v[:, -1] = False
    return v
