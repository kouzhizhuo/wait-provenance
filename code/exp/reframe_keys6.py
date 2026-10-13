"""METHOD_CHECK B4 (Holm over the primary set) and B5 (Gwet's AC1 next to kappa for the human audits).  python3 exp/reframe_keys6.py -> exp/results/reframe/methodcheck2_keys.json
B4: per series with exact permutation p (K >= 6 checkpoints), primary set = {FREQ_wait_1k, ORTH, H_para_rare, REL_j (REL if no judge labels)}; Holm-Bonferroni adjusted p
    keys h1.<s>.trend.rho_<q>.p_holm (q in freq, orth, hpara, rel).
B5: Gwet's AC1 (binary and 5-way) human vs judge from the audit confusion matrices (v1) and 2x2 tables (v2); CI = multinomial bootstrap over the table cells (B=5000)."""
from __future__ import annotations

import numpy as np

from reframe_keys import OUT, RES, ROOT, Ex, common, est
from exp.pipeline import stats as st

CATS = ["continue", "revise_value", "restart", "reverify_confirm", "branch"]
REV = [1, 2]   # binary revise = revise_value or restart (R = judge letter B or C)
NOREV = [0, 3, 4]


def ac1(M):
    """Gwet's AC1 for a square agreement table M (rows rater 1, columns rater 2)."""
    M = np.asarray(M, float); n = M.sum(); Q = M.shape[0]
    pa = np.trace(M) / n
    pi = (M.sum(0) + M.sum(1)) / (2 * n)
    pe = (pi * (1 - pi)).sum() / (Q - 1)
    return (pa - pe) / (1 - pe) if pe < 1 else float("nan")


def boot(M, B=5000, seed=1):
    M = np.asarray(M, float); rng = np.random.default_rng(seed)
    flat = M.ravel(); n = int(flat.sum())
    reps = [ac1(rng.multinomial(n, flat / n).reshape(M.shape)) for _ in range(B)]
    reps = [r for r in reps if r == r]
    return st.percentile_ci(reps)


def main():
    X = Ex()
    # ---------------- B4 Holm
    for ser, key, path in (("S1", "s1", RES / "main/S1/h1_indices.json"), ("S2", "s2", RES / "main/S2/h1_indices.json"), ("S9", "s9", RES / "s9/S9/h1_indices.json")):
        t = common.read_json(path)["trends"]
        rel = "REL_j" if "REL_j" in t and t["REL_j"].get("p") is not None else "REL"
        prim = [("freq", "FREQ_wait_1k"), ("orth", "ORTH"), ("hpara", "H_para_rare"), ("rel", rel)]
        ps = [(q, t[ix]["p"]) for q, ix in prim if ix in t and t[ix].get("p") is not None]
        order = sorted(ps, key=lambda x: x[1]); m = len(order); run = 0.0
        for i, (q, p) in enumerate(order):
            run = max(run, min(1.0, (m - i) * p))   # Holm step-down with monotonicity
            X.put("h1.%s.trend.rho_%s.p_holm" % (key, q), run, "dec3", "%s::trends (exact permutation p, K=%d) Holm-adjusted over %s" % (path.relative_to(ROOT.parent), t[dict(prim)[q]]["n_points"], [q_ for q_, _ in ps]))
        X.put("h1.%s.trend.holm_m" % key, m, "int", "number of tests in the Holm family (primary set with an exact p)")
    # ---------------- B5 AC1
    for suf, f in (("", "human_audit_results.json"), ("_oss", "human_audit_results_gptoss.json"), ("_oss120", "human_audit_results_gptoss120b.json")):
        d = common.read_json(ROOT / "gold" / f)
        conf = d["confusion_human_vs_judge"]
        M5 = np.zeros((5, 5)); 
        for k, v in conf.items():
            h, j = k.split("|"); M5[CATS.index(h), CATS.index(j)] += v
        M2 = np.array([[M5[NOREV][:, NOREV].sum(), M5[NOREV][:, REV].sum()], [M5[REV][:, NOREV].sum(), M5[REV][:, REV].sum()]])   # rows human no/yes, cols judge no/yes (yes = revise_value or restart)
        for tag, M in (("ac1bin", M2), ("ac15", M5)):
            v = ac1(M); lo, hi = boot(M)
            for q, x in (("", v), (".lo", lo), (".hi", hi)):
                X.put("gold.all.%s_human_judge%s%s" % (tag, suf, q), x, "dec2", "exp/gold/%s::confusion_human_vs_judge (n=%d), Gwet AC1" % (f, int(M.sum())))
    d2 = common.read_json(ROOT / "gold" / "human_audit_v2_results.json")
    for jn, suf in (("qwen7b", ""), ("gptoss20b", "_oss")):
        if jn in d2["judges"]:
            c = d2["judges"][jn]["confusion_unweighted"]
            M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            v = ac1(M); lo, hi = boot(M)
            for q, x in (("", v), (".lo", lo), (".hi", hi)):
                X.put("gold.s1.v2_ac1bin_human_judge%s%s" % (suf, q), x, "dec2", "exp/gold/human_audit_v2_results.json::judges.%s.confusion_unweighted (n=%d, unweighted 2x2), Gwet AC1" % (jn, int(M.sum())))
    common.write_json(OUT / "methodcheck2_keys.json", {"entries": X.entries})
    print({e["key"]: e["value"] for e in X.entries if "holm" in e["key"] or "ac1" in e["key"]})


if __name__ == "__main__":
    main()
