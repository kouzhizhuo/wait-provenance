import pytest
import math

from exp.pipeline import keys


def test_rounding_rules():
    assert keys.rnd(0.8973, "pct") == 89.7 and keys.rnd(5.213, "dec1") == 5.2 and keys.rnd(2.5417, "dec2") == 2.54
    assert keys.rnd(float("nan"), "pct") is None and keys.rnd(3.6, "int") == 4


def test_stage_aliases_for_p0():
    assert keys.stage_keys(1900, "1900", keys.P0_STAGE_MAP) == ["step1900", "late"]
    assert keys.stage_keys(0, "base", keys.P0_STAGE_MAP) == ["base"]
    assert keys.stage_keys(1000, "1000", keys.P0_STAGE_MAP) == ["step1000"]


def test_export_h1_keys_and_src():
    est = {"est": 0.897, "lo": 0.88, "hi": 0.914, "se": 0.01}
    h1 = {"checkpoints": {"1900": {"step": 1900, "acc": 0.771, "truncated": 0.26, "n_samples": 800,
                                   "indices": {"SI": est, "n_wait": {"est": 56932.0, "lo": 1, "hi": 2, "se": 0}}}},
          "trends": {"SI": {"rho": 0.8, "p": None, "n_points": 4}}}
    ex = keys.Exporter()
    keys.export_h1(ex, h1, "s1", "exp/results/P0/h1_indices.json")
    d = {e["key"]: e for e in ex.entries}
    assert d["h1.s1.late.si"]["value"] == 89.7 and d["h1.s1.step1900.si.lo"]["value"] == 88.0 and d["h1.s1.late.si.n"]["value"] == 56932
    assert "checkpoints.1900.indices.SI" in d["h1.s1.late.si"]["src"]
    assert d["h1.s1.trend.rho_si"]["value"] == 0.8 and "h1.s1.trend.rho_si.p" not in d      # no p below K = 6
    assert d["ctl.s1.late.acc"]["value"] == 77.1


def test_corrections_export_rd_corr_and_keep_r_amp():
    import numpy as np
    d = np.column_stack([np.full(500, 0.8), np.full(500, 0.9)])
    e = lambda v: {"est": v, "lo": v - 3, "hi": v + 3, "valid_frac": 1.0}
    ins = {"outcomes": {"judge_revise": {"late": {"n_prefixes": 10, "n_per": 4, "policies": {"S1@base": {"arms": {}}, "S1@1900": {"arms": {}}},
           "estimates": {"S1@base:RD_next": e(7.0), "S1@1900:RD_next": e(10.5), "R_amp_RD_next": e(1.5), "dRD_next_last_minus_base": e(3.5)}}}}}
    ex = keys.Exporter()
    keys.export_corrections(ex, ins, {"checkpoints": {}}, d, "s1", "/nonexistent", "S1")
    v = {x["key"]: x["value"] for x in ex.entries}
    assert v["h4.s1.step1900.rd_op_corr_late"] == pytest.approx(15.0, abs=0.7)     # 10.5 / 0.7
    assert v["h4.s1.trend.r_amp_corr_late"] == 1.5                                    # ratio unchanged
    assert v["h4.s1.trend.d_amp_corr_late"] == pytest.approx(5.0, abs=0.7)
