"""Style conditioning reaches the model and the model uses it.

The effect on F0 is small by nature (the three Ritsu banks differ in timbre far more
than in pitch behaviour), so it is easy for a refactor to break style conditioning
without anything sounding obviously wrong. These two checks catch that.
"""
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


def test_one_hot_style_mix_equals_style_id(model_and_cfg):
    """A one-hot ``style_mix`` must reproduce ``style_id`` exactly, for every style.

    This is the strongest wiring check available: it fails if either path stops
    reaching ``cond_from_inputs``, or if the two disagree on the index order.
    """
    import infer
    model, cfg = model_and_cfg
    n_styles = int(cfg.get("n_styles", 0))
    if n_styles <= 0:
        pytest.skip("the bundled weight has no style embedding")

    score = json.loads(SCORE.read_text(encoding="utf-8"))
    for sid in range(n_styles):
        by_id = infer.synthesize_f0(model, cfg, {**score, "style_id": sid}, seed=0)
        by_mix = infer.synthesize_f0(model, cfg, {**score, "style_id": 0}, seed=0,
                                     style_mix=np.eye(n_styles, dtype=np.float32)[sid])
        assert np.array_equal(by_id["f0_hz"], by_mix["f0_hz"]), f"style {sid}"


def test_styles_produce_different_f0(model_and_cfg):
    """Different styles must not collapse to the same curve.

    The measured spread between style 0 and style 2 on a demo phrase is a median of
    ~3.5 cents, so the bar here is deliberately low. It only asserts that the style
    embedding still moves the output at all.
    """
    import infer
    model, cfg = model_and_cfg
    n_styles = int(cfg.get("n_styles", 0))
    if n_styles < 2:
        pytest.skip("fewer than two styles")

    score = json.loads(SCORE.read_text(encoding="utf-8"))
    outs = [infer.synthesize_f0(model, cfg, {**score, "style_id": s}, seed=0)
            for s in range(n_styles)]
    voiced = np.logical_and.reduce([o["uv"] > 0 for o in outs])
    assert voiced.sum() > 0

    for s in range(1, n_styles):
        cents = np.abs(np.log2(outs[s]["f0_hz"][voiced])
                       - np.log2(outs[0]["f0_hz"][voiced])) * 1200.0
        assert cents.mean() > 0.5, f"style {s} is indistinguishable from style 0"
