"""OLMo-3-7B GRPO fork (controlled RL from the marker-bearing base; D42/D44) keys from exp/rl/results/olmo3_ref_s1_p/results_step*.tgz (idx/sN.json, step 0 = the base).
    python3 exp/reframe_keys13.py -> exp/results/reframe/fork_keys.json.  Series id: fork_ref1 (REF s1_p), fork_b0 (beta=0, when its archive lands).
Keys: the S1 index set via keys.export_h1 (h1.fork_ref1.<base|stepNNNN>.*), ctl.fork_ref1.<stage>.{acc,trunc}, h1.fork_ref1.trend.{a_wait,a_wait.lo,a_wait.hi} = FREQ(last step)/FREQ(step 0) (conditional-binomial CI on the counts)."""
from __future__ import annotations

import json
import tarfile

import numpy as np
from scipy.stats import beta

from reframe_keys import OUT, ROOT, Ex, common
from exp.pipeline import keys as K


def ratio_ci(n0, t0, n1, t1):
    # ratio of two Poisson rates: n1 | n0+n1 ~ Binomial(n0+n1, p), p = rate1*t1 / (rate0*t0 + rate1*t1)
    n = n0 + n1
    if n0 == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")
    lo_p = beta.ppf(0.025, n1, n - n1 + 1) if n1 > 0 else 0.0
    hi_p = beta.ppf(0.975, n1 + 1, n - n1)
    f = lambda p: (p / (1 - p)) * (t0 / t1) if p < 1 else float("inf")
    return f(n1 / n) if n1 < n else float("inf"), f(lo_p), f(hi_p)


def load(series, glob_pat):
    files = sorted((ROOT / "rl" / "results" / series[1]).glob(glob_pat))
    if not files:
        return None
    t = tarfile.open(files[-1])
    out = {}
    for m in t.getmembers():
        import re
        mm = re.fullmatch(r"(?:\./)?idx/s(\d+)\.json", m.name)
        if mm:
            out[int(mm.group(1))] = json.load(t.extractfile(m))
    return files[-1], dict(sorted(out.items()))


def main():
    ex = K.Exporter(); X = Ex(); summ = {}
    for sid, dirn in (("fork_ref1", "olmo3_ref_s1_p"), ("fork_b0", "olmo3_beta0_s1_p")):
        got = load((sid, dirn), "results*.tgz")
        if not got:
            continue
        f, idx = got
        h1 = {"checkpoints": {}, "trends": {}}
        for step, j in idx.items():
            h1["checkpoints"]["base" if step == 0 else str(step)] = {"step": step, "acc": j["acc"], "truncated": j["truncated"], "n_samples": j["n_samples"], "indices": j["indices"]}
        K.export_h1(ex, h1, sid, str(f.relative_to(ROOT.parent)) + "::idx/s<step>.json", {})
        s0, sl = idx[min(idx)], idx[max(idx)]
        n0, t0 = int(s0["totals"].get("wait:n", 0)), int(s0["totals"]["tokens"]); n1, t1 = int(sl["totals"].get("wait:n", 0)), int(sl["totals"]["tokens"])
        est, lo, hi = ratio_ci(n0, t0, n1, t1)
        for q, v in (("a_wait", est), ("a_wait.lo", lo), ("a_wait.hi", hi)):
            X.put("h1.%s.trend.%s" % (sid, q), v, "dec2", "FREQ_wait(step %d)/FREQ_wait(step 0), conditional-binomial CI on counts" % max(idx))
        X.put("h1.%s.trend.last_step" % sid, max(idx), "int", str(f.name))
        for step, j in idx.items():
            stg = "base" if step == 0 else "step%04d" % step
            X.put("ctl.%s.%s.toks" % (sid, stg), j["totals"]["tokens"] / j["n_samples"], "int", "tokens per response")
        summ[sid] = {"steps": list(idx), "a_wait": est, "ci": [lo, hi]}
    common.write_json(OUT / "fork_keys.json", {"entries": ex.entries + X.entries})
    print(summ, len(ex.entries) + len(X.entries), "keys")


if __name__ == "__main__":
    main()
