"""D48 primary quantity: A(t) = wait/1k at step t over wait/1k at the base, cluster bootstrap over problems (paired resampling of problem ids, B = 5000, seed 20261012),
all responses (a_boot) and finished-only (a_boot_fin). Series fork_ref1m (matched eval; needs matched/gen/s<N>.graded.jsonl.gz) and s1 (released RL-Zero run, same eval) at steps 100/200/400/final.
    python3 exp/reframe_keys15.py -> exp/results/reframe/d48_keys.json   keys h1.<series>.<stage>.a_boot{,.lo,.hi}, a_boot_fin{,.lo,.hi}"""
from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict

import numpy as np

from reframe_keys import OUT, ROOT, Ex, common
from pipeline.indices import trace_counts

MATCH = ROOT / "rl" / "results" / "olmo3_ref_s1_p" / "matched" / "gen"
MATCH_B0 = ROOT / "rl" / "results" / "olmo3_beta0_s1_p" / "matched" / "gen"
S1G = ROOT / "results" / "main" / "S1" / "gen"
B = 5000


def per_problem(path, fin_only):
    w, t = defaultdict(float), defaultdict(float)
    for l in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(l)
        if fin_only and r.get("finish") != "stop":
            continue
        c = trace_counts(r["text"], r["n_tokens"])
        w[r["pid"]] += c.get("wait:n", 0); t[r["pid"]] += c["tokens"]
    return w, t


def boot_ratio(base, cur, pids, rng):
    wb = np.array([base[0][p] for p in pids]); tb = np.array([base[1][p] for p in pids])
    wc = np.array([cur[0][p] for p in pids]); tc = np.array([cur[1][p] for p in pids])
    est = (wc.sum() / tc.sum()) / (wb.sum() / tb.sum())
    idx = rng.integers(0, len(pids), size=(B, len(pids)))
    r = (wc[idx].sum(1) / tc[idx].sum(1)) / (wb[idx].sum(1) / tb[idx].sum(1))
    return est, *np.percentile(r, [2.5, 97.5])


def run(X, series, base_path, steps):
    for fin_only, leaf in ((False, "a_boot"), (True, "a_boot_fin")):
        base = per_problem(base_path, fin_only)
        for stage, path in steps:
            cur = per_problem(path, fin_only)
            pids = sorted(set(base[1]) & set(cur[1]))
            est, lo, hi = boot_ratio(base, cur, pids, np.random.default_rng(20261012))
            for q, v in (("", est), (".lo", lo), (".hi", hi)):
                X.put("h1.%s.%s.%s%s" % (series, stage, leaf, q), v, "dec2", "cluster bootstrap over %d problems, B=%d, paired; %s; %s" % (len(pids), B, "finished only" if fin_only else "all responses", path.name))


def main():
    X = Ex()
    f = {int(re.match(r"s(\d+)\.", p.name).group(1)): p for p in MATCH.glob("s*.graded.jsonl.gz") if "greedy" not in p.name}
    base = S1G / "<EMAIL>"
    run(X, "fork_ref1m", base, [("step%04d" % s, p) for s, p in sorted(f.items())])
    fb = {int(re.match(r"s(\d+)\.", p.name).group(1)): p for p in MATCH_B0.glob("s*.graded.jsonl.gz") if "greedy" not in p.name}
    if fb:
        run(X, "fork_b0m", base, [("step%04d" % s, p) for s, p in sorted(fb.items())])
    run(X, "s1", base, [("step%04d" % s, S1G / ("S1@%d.graded.jsonl.gz" % s)) for s in (100, 200, 400, 700, 1000, 1400)] + [("final", S1G / "<EMAIL>")])
    common.write_json(OUT / "d48_keys.json", {"entries": X.entries})
    e = {x["key"]: x["value"] for x in X.entries}
    for k in sorted(e):
        if k.endswith(("a_boot", "a_boot_fin")) or k.endswith(("a_boot.lo", "a_boot.hi", "a_boot_fin.lo", "a_boot_fin.hi")):
            print(k, e[k])


if __name__ == "__main__":
    main()
