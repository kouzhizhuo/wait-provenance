import numpy as np
import pytest

from exp.pipeline import calib


def test_rogan_gladen_and_rd_scaling_with_a_precise_judge():
    d = np.column_stack([np.full(2000, 0.8), np.full(2000, 0.9)])       # sens .8, spec .9 -> J = .7
    rd = calib.corrected_rd(7.0, 3.0, 11.0, d)
    assert rd["est"] == pytest.approx(10.0, abs=0.5) and rd["share_unidentified"] == 0.0
    # observed p = p*.8 + (1-p)*.1 ; p = .5 -> .45
    rel = calib.corrected_rel(0.8 * 0.5 + (1 - 0.5) * (1 - 0.9 ** 3), d)
    assert rel["est"] == pytest.approx(0.5, abs=0.02)
    # two strata with true rates .30 / .02 and sizes 2000 / 18000 -> OBL .625
    r = lambda p: p * 0.8 + (1 - p) * 0.1
    obl = calib.corrected_obl(r(0.30), r(0.02), 2000.0, 18000.0, d)
    assert obl["est"] == pytest.approx(0.625, abs=0.02)


def test_uninformative_judge_leaves_the_magnitude_unidentified():
    d = np.column_stack([np.full(500, 0.5), np.full(500, 0.5)])         # J = 0
    out = calib.corrected_rd(7.0, 3.0, 11.0, d)
    assert out["share_unidentified"] == 1.0 and np.isnan(out["est"])


def test_posterior_draws_shape_and_range():
    d = calib.posterior_draws(tp=3, fn=2, tn=40, fp=5, n=500)
    assert d.shape == (500, 2) and 0 < d.min() and d.max() < 1


def test_gptoss_final_channel_parsing():
    from exp.pipeline.judge_audit import final_letter
    raw = "<|channel|>analysis<|message|>The sentence rechecks A vs B... maybe D.<|end|><|start|>assistant<|channel|>final<|message|>D<|return|>"
    assert final_letter(raw) == "D"
    assert final_letter("no channels, answer: C") == "C"
