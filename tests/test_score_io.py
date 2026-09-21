"""Every bundled score loads, splits into phrases and post-processes into a phoneme sequence."""
import numpy as np
import pytest

from tests.conftest import ROOT

SCORES = sorted((ROOT / "examples").glob("*.musicxml")) + [
    ROOT / "examples" / "kirakiraboshi.ust",
    ROOT / "tests" / "fixtures" / "edge_cases.musicxml",
]


@pytest.mark.parametrize("path", SCORES, ids=[p.name for p in SCORES])
def test_score_loads_and_postprocesses(path):
    from svs.score_io import load_score
    from svs.duration_postprocess import postprocess_durations, to_frames

    phrases = load_score(path)
    assert len(phrases) >= 1
    for phr in phrases:
        assert len(phr.notes) >= 1
        seq = postprocess_durations(phr, rest_before=1e9)           # fixed-length consonants
        assert len(seq.phonemes) == len(seq.dur_sec)
        assert np.all(seq.dur_sec > 0)
        ph_durs, T = to_frames(seq.dur_sec, 172.265625)
        assert T == int(ph_durs.sum()) and T > 0


def test_ust_and_musicxml_agree():
    from svs.score_io import load_score
    from svs.duration_postprocess import postprocess_durations

    ust = load_score(ROOT / "examples" / "kirakiraboshi.ust")
    xml = load_score(ROOT / "examples" / "kirakiraboshi.musicxml")
    assert len(ust) == len(xml)
    for a, b in zip(ust, xml):
        assert list(postprocess_durations(a, rest_before=1e9).phonemes) \
            == list(postprocess_durations(b, rest_before=1e9).phonemes)
