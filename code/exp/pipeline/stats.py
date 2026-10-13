"""Small statistics toolbox: entropy, cluster bootstrap, odds ratios, signal-detection d'."""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def entropy_bits(counts: Iterable[float], miller_madow: bool = True) -> float:
    """Shannon entropy in bits of a count vector. Miller-Madow correction adds (K-1)/(2 n ln 2), K = #non-empty bins.

    Returns NaN for n = 0. The correction matters because comparisons across checkpoints have very
    different numbers of occurrences (the base model has few markers); plug-in entropy is biased low.
    """
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    n = c.sum()
    if n <= 0:
        return float("nan")
    p = c / n
    h = float(-(p * np.log2(p)).sum())
    if miller_madow and len(c) > 1:
        h += (len(c) - 1) / (2.0 * n * math.log(2))
    return h


def cluster_resample(n_clusters: int, B: int, seed: int) -> np.ndarray:
    """(B, n_clusters) integer indices for a cluster (e.g. per-problem) bootstrap."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(B, n_clusters))


def percentile_ci(samples: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    a = np.asarray([s for s in samples if s == s], dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(a, 100 * alpha / 2)), float(np.percentile(a, 100 * (1 - alpha / 2))))


def bootstrap_stat(matrix: np.ndarray, names: Sequence[str], stat_fn: Callable[[Dict[str, float]], Dict[str, float]],
                   B: int = 2000, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Cluster bootstrap of a statistic that is a function of column-sums of `matrix` (clusters x features).

    Returns {stat: {"est", "lo", "hi", "se"}}. The estimate is computed on the full sample.
    """
    tot = dict(zip(names, matrix.sum(axis=0)))
    est = stat_fn(tot)
    idx = cluster_resample(matrix.shape[0], B, seed)
    reps: Dict[str, List[float]] = {k: [] for k in est}
    for b in range(B):
        sums = matrix[idx[b]].sum(axis=0)
        r = stat_fn(dict(zip(names, sums)))
        for k in est:
            reps[k].append(r[k])
    out = {}
    for k, v in est.items():
        lo, hi = percentile_ci(reps[k])
        arr = np.asarray([x for x in reps[k] if x == x], dtype=float)
        out[k] = {"est": v, "lo": lo, "hi": hi, "se": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")}
    return out


# ----------------------------------------------------------------------------- odds ratios
def odds_ratio(k1: float, n1: float, k0: float, n0: float, correction: float = 0.5) -> float:
    """OR of success rates k1/n1 vs k0/n0 with Haldane-Anscombe correction if any cell is zero."""
    a, b, c, d = k1, n1 - k1, k0, n0 - k0
    if min(a, b, c, d) == 0:
        a, b, c, d = a + correction, b + correction, c + correction, d + correction
    return (a / b) / (c / d)


def paired_or_bootstrap(succ_a: np.ndarray, succ_b: np.ndarray, n_per: int, B: int = 2000, seed: int = 0):
    """Cluster bootstrap for OR(arm A vs arm B) where each cluster (prefix) has `n_per` continuations per arm.

    succ_a[i], succ_b[i] = number of REVISE among the n_per continuations of prefix i under arm A / B.
    Resampling whole prefixes keeps the pairing. Returns dict(or, lo, hi, pA, pB, n_prefixes).
    """
    succ_a = np.asarray(succ_a, dtype=float)
    succ_b = np.asarray(succ_b, dtype=float)
    m = len(succ_a)

    def f(ia):
        ka, kb = succ_a[ia].sum(), succ_b[ia].sum()
        n = len(ia) * n_per
        return odds_ratio(ka, n, kb, n)

    est = f(np.arange(m))
    idx = cluster_resample(m, B, seed)
    reps = [f(idx[b]) for b in range(B)]
    lo, hi = percentile_ci(reps)
    return {"or": float(est), "lo": lo, "hi": hi, "pA": float(succ_a.sum() / (m * n_per)),
            "pB": float(succ_b.sum() / (m * n_per)), "n_prefixes": int(m)}


def proportion_ci(succ: np.ndarray, n_per: int, B: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """Cluster-bootstrap CI of a rate where cluster i contributes succ[i] successes out of n_per."""
    succ = np.asarray(succ, dtype=float)
    m = len(succ)
    est = succ.sum() / (m * n_per)
    idx = cluster_resample(m, B, seed)
    reps = succ[idx].sum(axis=1) / (m * n_per)
    lo, hi = percentile_ci(reps)
    return float(est), lo, hi


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) == 0:
        return float("nan")
    labels = np.unique(np.concatenate([a, b]))
    po = float((a == b).mean())
    pe = sum(float((a == l).mean() * (b == l).mean()) for l in labels)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


# ----------------------------------------------------------------------------- SDT
def _phi_inv(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


def dprime(hits: float, n_signal: float, fas: float, n_noise: float) -> Dict[str, float]:
    """d' and criterion c with the log-linear correction (add 0.5 to counts, 1 to totals)."""
    hr = (hits + 0.5) / (n_signal + 1.0)
    fa = (fas + 0.5) / (n_noise + 1.0)
    zh, zf = _phi_inv(hr), _phi_inv(fa)
    return {"HR": hits / n_signal if n_signal else float("nan"), "FA": fas / n_noise if n_noise else float("nan"),
            "dprime": zh - zf, "c": -(zh + zf) / 2.0}


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, Optional[float]]:
    """Spearman rho with an EXACT two-sided permutation p-value for 6 <= n <= 8 (STATS_PLAN C1). For n < 6 the
    smallest attainable p is not below .05 in a meaningful way, so no p is returned (None); for n > 8 the
    asymptotic scipy p is used."""
    from scipy.stats import rankdata, spearmanr

    n = len(x)
    if n < 3:
        return (float("nan"), None)
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return (float("nan"), None)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    if n < 6:
        return (rho, None)
    if n <= 8:
        import itertools

        hits = tot = 0
        for perm in itertools.permutations(range(n)):
            r = np.corrcoef(rx, ry[list(perm)])[0, 1]
            tot += 1
            hits += abs(r) >= abs(rho) - 1e-12
        return (rho, hits / tot)
    return (rho, float(spearmanr(x, y).pvalue))


# ----------------------------------------------------------------------------- insertion estimands (STATS_PLAN C4)
def or_firth(k1: float, n1: float, k0: float, n0: float) -> float:
    """OR with 0.5 added to every cell. For a single binary predictor this equals the Firth-penalised
    (Jeffreys-prior) logistic estimate, and it stays finite under separation (STATS_PLAN C4)."""
    return ((k1 + 0.5) / (n1 - k1 + 0.5)) / ((k0 + 0.5) / (n0 - k0 + 0.5))


def rr_half(k1: float, n1: float, k0: float, n0: float) -> float:
    return ((k1 + 0.5) / (n1 + 1.0)) / ((k0 + 0.5) / (n0 + 1.0))


def _arm_counts(succ_arm: Dict[str, np.ndarray], idx: np.ndarray, n_per: int) -> Dict[str, Tuple[float, float]]:
    n = len(idx) * n_per
    return {a: (float(v[idx].sum()), float(n)) for a, v in succ_arm.items()}


def _policy_stats(c: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Contrasts of Wait, against Okay, (operator), the pooled synonyms {Hold on, Hmm} and the empty arm."""
    kw, n = c["wait"]
    syn = [c[a][0] for a in ("hold_on", "hmm") if a in c]
    ks, ns = sum(syn), n * len(syn)
    out = {"p_wait": kw / n, "RD_none": 100.0 * (kw - c["none"][0]) / n}
    if syn:
        out.update({"RD_syn": 100.0 * (kw / n - ks / ns), "OR_syn": or_firth(kw, n, ks, ns)})
    if "okay" in c:   # not every checkpoint has the Okay, arm (core arms are {wait, next, hold_on, none}, D28)
        ko = c["okay"][0]
        out.update({"p_okay": ko / n, "RD_op": 100.0 * (kw - ko) / n, "RR_op": rr_half(kw, n, ko, n), "OR_op": or_firth(kw, n, ko, n)})
    if "next" in c:  # D21: `Next,` is the neutral control of the main runs; `Okay,` stays as the comparability arm
        kn = c["next"][0]
        out.update({"RD_next": 100.0 * (kw - kn) / n, "OR_next": or_firth(kw, n, kn, n), "RR_next": rr_half(kw, n, kn, n)})
    for a in c:
        out["p_" + a] = c[a][0] / c[a][1]
    return out


def insertion_estimates(succ: Dict[str, Dict[str, np.ndarray]], n_per: int, B: int = 2000, seed: int = 20261012,
                        base: str = "S1@base", last: str = "S1@1900") -> Dict[str, Dict[str, float]]:
    """succ[policy][arm] = REVISE successes (out of n_per) per prefix, prefixes aligned across policies/arms.

    One bootstrap replicate resamples PREFIXES once and recomputes every policy's contrasts, so differences
    between policies (R_amp) inherit the pairing. RD_* are in percentage points.
      R_amp_RD = RD_op(last) / RD_op(base)     R_amp_OR = OR_op(last) / OR_op(base)     (nan if denominator <= 0)
    """
    pols = list(succ)
    m = len(next(iter(next(iter(succ.values())).values())))

    def compute(idx: np.ndarray) -> Dict[str, float]:
        res: Dict[str, float] = {}
        per = {p: _policy_stats(_arm_counts(succ[p], idx, n_per)) for p in pols}
        for p, d in per.items():
            for k, v in d.items():
                res["%s:%s" % (p, k)] = v
        if base in per and last in per:
            b, l = per[base], per[last]
            if "RD_op" in b and "RD_op" in l:
                res["R_amp_RD"] = l["RD_op"] / b["RD_op"] if b["RD_op"] > 0 else float("nan")
                res["R_amp_OR"] = l["OR_op"] / b["OR_op"]
                res["dRD_op_last_minus_base"] = l["RD_op"] - b["RD_op"]
            if "RD_next" in l and "RD_next" in b:  # D21: primary contrast is Wait vs Next
                res["R_amp_RD_next"] = l["RD_next"] / b["RD_next"] if b["RD_next"] > 0 else float("nan")
                res["R_amp_OR_next"] = l["OR_next"] / b["OR_next"]
                res["dRD_next_last_minus_base"] = l["RD_next"] - b["RD_next"]   # linear amplification D_amp
        return res

    est = compute(np.arange(m))
    idx = cluster_resample(m, B, seed)
    reps: Dict[str, List[float]] = {k: [] for k in est}
    for b in range(B):
        r = compute(idx[b])
        for k in est:
            reps[k].append(r[k])
    out = {}
    for k, v in est.items():
        lo, hi = percentile_ci(reps[k])
        valid = float(np.mean([x == x for x in reps[k]]))
        out[k] = {"est": float(v), "lo": lo, "hi": hi, "valid_frac": valid}
    return out


def crossed_interaction(succ_late: Dict[str, Dict[str, np.ndarray]], succ_base: Dict[str, Dict[str, np.ndarray]],
                        n_per: int, B: int = 2000, seed: int = 20261013, base: str = "S1@base",
                        last: str = "S1@1900") -> Dict[str, Dict[str, float]]:
    """2 x 2 crossed design {prefix source: late-derived, base-derived} x {policy: base, last}.
    Estimand: difference in policy effect on RD_op between the two prefix sources,
      DiD = [RD_op(last,late) - RD_op(base,late)] - [RD_op(last,base-src) - RD_op(base,base-src)].
    Prefix sets are different problems' traces, so the two sets are resampled independently."""
    def stats(succ, idx):
        return {p: _policy_stats(_arm_counts(succ[p], idx, n_per)) for p in (base, last)}

    def f(il, ib):
        a, b = stats(succ_late, il), stats(succ_base, ib)
        k = "RD_next" if all("RD_next" in x[p] for x in (a, b) for p in (base, last)) else "RD_op"   # reference arm: Next, (D21) if present everywhere
        dl = a[last][k] - a[base][k]
        db = b[last][k] - b[base][k]
        return {"policy_effect_late_src": dl, "policy_effect_base_src": db, "DiD": dl - db, "reference_arm_is_next": float(k == "RD_next"),
                "RD_base_on_base_src": b[base][k], "RD_base_on_late_src": a[base][k],
                "RD_last_on_base_src": b[last][k], "RD_last_on_late_src": a[last][k]}

    ml = len(next(iter(next(iter(succ_late.values())).values())))
    mb = len(next(iter(next(iter(succ_base.values())).values())))
    est = f(np.arange(ml), np.arange(mb))
    il, ib = cluster_resample(ml, B, seed), cluster_resample(mb, B, seed + 1)
    reps: Dict[str, List[float]] = {k: [] for k in est}
    for b in range(B):
        r = f(il[b], ib[b])
        for k in est:
            reps[k].append(r[k])
    out = {}
    for k, v in est.items():
        lo, hi = percentile_ci(reps[k])
        out[k] = {"est": float(v), "lo": lo, "hi": hi}
    return out
