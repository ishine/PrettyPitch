"""The bundled F0 model runs on the example score and returns a sane curve."""
import json

import numpy as np
import pytest

from tests.conftest import ROOT

CKPT = ROOT / "checkpoints" / "f0_3singer.pt"
SCORE = ROOT / "examples" / "example_score.json"


@pytest.fixture(scope="module")
def model_and_cfg():
    import infer
    return infer.load_model(str(CKPT), "cpu")


def test_synthesize_f0_on_example(model_and_cfg):
    import infer
    model, cfg = model_and_cfg
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    out = infer.synthesize_f0(model, cfg, score, seed=0)

    n_frames = int(round(sum(s["dur"] for s in score["phonemes"]) * float(cfg["frame_rate"])))
    assert abs(len(out["f0_hz"]) - n_frames) <= len(score["phonemes"])   # rounding per phoneme
    for key in ("f0_hz", "uv", "baseline_log2", "plain_delta", "vib_delta"):
        assert len(out[key]) == len(out["f0_hz"]), key
        assert np.all(np.isfinite(out[key])), key
    assert set(np.unique(out["uv"])) <= {0.0, 1.0}
    voiced = out["uv"] > 0
    assert voiced.mean() > 0.5                          # the example is mostly vowels
    assert 50.0 < out["f0_hz"][voiced].min() and out["f0_hz"][voiced].max() < 2000.0


def test_same_seed_same_output(model_and_cfg):
    import infer
    model, cfg = model_and_cfg
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    a = infer.synthesize_f0(model, cfg, score, seed=3)
    b = infer.synthesize_f0(model, cfg, score, seed=3)
    assert np.array_equal(a["f0_hz"], b["f0_hz"])


def test_scales_change_only_their_channel(model_and_cfg):
    import infer
    model, cfg = model_and_cfg
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    base = infer.synthesize_f0(model, cfg, score, seed=0)
    louder = infer.synthesize_f0(model, cfg, score, seed=0, vib_scale=2.0)
    assert np.allclose(louder["vib_delta"], 2.0 * base["vib_delta"], atol=1e-6)
    assert np.array_equal(louder["plain_delta"], base["plain_delta"])
