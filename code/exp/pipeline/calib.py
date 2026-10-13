"""Corrections for judge misclassification, from sensitivity/specificity estimated against the HUMAN audit labels.

Assumption (stated wherever a corrected number is shown): NONDIFFERENTIAL misclassification, i.e. the judge's sensitivity and
specificity for the binary decision REVISE (= revise_value + restart) do not depend on the arm, the checkpoint or the marker.
Under it:
  * a risk difference is attenuated by Youden's J = sens + spec - 1:  RD_true = RD_obs / J  (so the corrected effect is LARGER);
  * a ratio of risk differences (R_amp) and every rank correlation over checkpoints are unchanged;
  * a proportion p_obs = p sens + (1-p)(1-spec) is corrected by Rogan-Gladen: p = (p_obs - (1-spec)) / J.
    REL = P(a revision event in the 3-sentence window | marker): one true event is detected with prob. sens, and with no true event
    at least one false alarm arises with prob. 1 - spec^3, so the window-level specificity is spec^3.
    OBL = P(marker in i or i-1 | R): each stratum's event rate is corrected with sens/spec, then recombined with the stratum sizes.
The intervals reflect the uncertainty of (sens, spec) from a small audit ONLY (Beta(1,1) posteriors; draws with J <= 0.05 are excluded
and their share is reported: the magnitude is then not identified). Sampling uncertainty of the observed quantity is added for RD.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

J_MIN = 0.05


def _summ(x: np.ndarray, share_bad: float) -> Dict[str, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"est": float("nan"), "lo": float("nan"), "hi": float("nan"), "share_unidentified": share_bad}
    return {"est": float(np.median(x)), "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)), "share_unidentified": share_bad}


def corrected_rd(est: float, lo: float, hi: float, draws: np.ndarray, seed: int = 0, n: int = 20000) -> Dict[str, float]:
    """RD_obs / J with the observed sampling error (normal approximation of the bootstrap CI) and the audit uncertainty of J."""
    rng = np.random.default_rng(seed)
    sens, spec = draws[rng.integers(0, len(draws), n)].T
    J = sens + spec - 1
    se = max((hi - lo) / 3.92, 1e-9)
    obs = rng.normal(est, se, n)
    ok = J > J_MIN
    out = np.where(ok, obs / np.where(ok, J, 1.0), np.nan)
    return _summ(np.clip(out, -100.0, 100.0), float(1 - ok.mean()))   # a risk difference (points) cannot leave [-100, 100]


def corrected_rel(est: float, draws: np.ndarray, window: int = 3, seed: int = 0, n: int = 20000) -> Dict[str, float]:
    """Rogan-Gladen for the window-level proportion REL (see module doc)."""
    rng = np.random.default_rng(seed)
    sens, spec = draws[rng.integers(0, len(draws), n)].T
    spec_w = spec ** window
    J = sens + spec_w - 1
    ok = J > J_MIN
    p = np.where(ok, (est - (1 - spec_w)) / np.where(ok, J, 1.0), np.nan)
    return _summ(np.clip(p, 0.0, 1.0), float(1 - ok.mean()))


def corrected_obl(r1_obs: float, r0_obs: float, N1: float, N0: float, draws: np.ndarray, seed: int = 0, n: int = 20000) -> Dict[str, float]:
    """Correct the event rate of each stratum (marker-adjacent / other), then OBL = N1 r1 / (N1 r1 + N0 r0)."""
    rng = np.random.default_rng(seed)
    sens, spec = draws[rng.integers(0, len(draws), n)].T
    J = sens + spec - 1
    ok = J > J_MIN
    Js = np.where(ok, J, 1.0)
    r1 = np.clip((r1_obs - (1 - spec)) / Js, 0.0, 1.0)
    r0 = np.clip((r0_obs - (1 - spec)) / Js, 0.0, 1.0)
    den = N1 * r1 + N0 * r0
    obl = np.where(ok & (den > 0), N1 * r1 / np.where(den > 0, den, 1.0), np.nan)
    return _summ(obl, float(1 - (ok & (den > 0)).mean()))


def posterior_draws(tp: int, fn: int, tn: int, fp: int, n: int = 4000, seed: int = 0) -> np.ndarray:
    """(n, 2) draws of (sensitivity, specificity) with Beta(1,1) priors from the audit's 2x2 table (human = truth)."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.beta(tp + 1, fn + 1, n), rng.beta(tn + 1, fp + 1, n)])
