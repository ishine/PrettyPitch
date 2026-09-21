"""
preprocess.algorithms

Pitch-estimation algorithms. `preprocess/f0_rmvpe.py` constructs `RMVPEPitchAlgorithm`
directly; `base.py` and `rmvpe.py` are taken from pitch-benchmark (see THIRD_PARTY_NOTICES.md).
"""

from .base import PitchAlgorithm, ContinuousPitchAlgorithm, ThresholdPitchAlgorithm
from .rmvpe import RMVPEPitchAlgorithm

__all__ = [
    'PitchAlgorithm',
    'ContinuousPitchAlgorithm',
    'ThresholdPitchAlgorithm',
    'RMVPEPitchAlgorithm',
]
