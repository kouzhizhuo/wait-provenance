"""Calibrated Wait analysis (D50/D50a-f): per-cell metrics, macro paired problem-cluster bootstrap (stratified by (model, benchmark) cell), success criteria S1-S3, S6 matched-budget check, keys h5.*.
Input layout (synced from the hosts):  exp/results/cw/test/<model>/<cond>.<bench>.graded.jsonl.gz (+ .jsonl.meta.json), exp/results/cw/dev/<model>/{calib,match,s4}.json
Benchmarks pooled into 4 columns: math500, aime (aime24+aime25), amc23, gsm8k.   Conditions: none (vanilla), sham, nowait, cw, random, bf_match, bf_loose, cgrs (optional).
    python -m exp.cw.cw_analyze [--models rlzero,olmo3think,r1d7,qwen3_8b] -> exp/results/reframe/cw_keys.json, exp/results/cw/cw_summary.json"""
import argparse
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CW = ROOT / "results" / "cw"
BENCH = {"math500": "math500", "aime24": "aime", "aime25": "aime", "amc23": "amc23", "gsm8k": "gsm8k"}
CONDS = ["none", "none_s1", "sham", "nowait", "cw", "random", "bf_match", "bf_loose", "cw75", "cw50", "cgrs"]


def load_cell(model, cond, bench):
    p = CW / "test" / model / ("%s.%s.graded.jsonl.gz" % (cond, bench))
    if not p.exists():
        return None
    rows = [json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")]
    meta = {}
    mp = CW / "test" / model / ("%s.%s.jsonl.meta.json" % (cond, bench))
    if mp.exists():
        meta = json.load(open(mp)).get("config", {})
    return rows, meta


def wait_count(text):
    from exp.pipeline.indices import trace_counts
    return trace_counts(text, 1).get("wait:n", 0)


def cell_table(model):
    """{(cond, col): {pid-key: (acc, toks, waits)}} with per-problem means over samples."""
    T = defaultdict(dict); wall = defaultdict(float)
    for cond in CONDS:
        for b, col in BENCH.items():
            r = load_cell(model, cond, b)
            if r is None:
                continue
            rows, meta = r
            by = defaultdict(list)
            for x in rows:
                by[(b, x["pid"])].append((float(bool(x["correct"])), x["n_tokens"], wait_count(x["text"])))
            for k, v in by.items():
                a = np.array(v, dtype=float).mean(0)
                T[(cond, col)][k] = tuple(a)
            wall[cond] += float(meta.get("gen_seconds", 0.0)); wall[(cond, col)] += float(meta.get("gen_seconds", 0.0))
    return T, wall


def macro(models_T, cond, ref, which, B=2000, seed=20261012):
    """macro over (model, column) cells of mean(acc_cond - acc_ref) [which='acc'] or 1 - toks_cond/toks_ref [which='tok']; paired cluster bootstrap over problems inside every cell."""
    rng = np.random.default_rng(seed); cells = []
    for m, T in models_T.items():
        for col in ("math500", "aime", "amc23", "gsm8k"):
            a, r = T.get((cond, col)), T.get((ref, col))
            if not a or not r:
                continue
            ks = sorted(set(a) & set(r))
            if not ks:
                continue
            cells.append((m, col, np.array([a[k] for k in ks]), np.array([r[k] for k in ks])))
    if not cells:
        return None

    def stat(idx_list):
        vals = []
        for (m, col, A, R), idx in zip(cells, idx_list):
            if which == "acc":
                vals.append(A[idx, 0].mean() - R[idx, 0].mean())
            elif which == "tokd":
                vals.append(A[idx, 1].mean() / R[idx, 1].mean() - 1.0)
            else:
                vals.append(1.0 - A[idx, 1].mean() / R[idx, 1].mean())
        return float(np.mean(vals))

    full = stat([np.arange(len(c[2])) for c in cells]); bs = []
    for _ in range(B):
        bs.append(stat([rng.integers(0, len(c[2]), len(c[2])) for c in cells]))
    return {"est": full, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n_cells": len(cells)}


def per_cell(models_T, cond):
    out = {}
    for m, T in models_T.items():
        for col in ("math500", "aime", "amc23", "gsm8k"):
            d = T.get((cond, col))
            if d:
                A = np.array(list(d.values())); rng = np.random.default_rng(1); bs = [A[rng.integers(0, len(A), len(A)), 0].mean() for _ in range(1000)]
                out[(m, col)] = {"acc.lo": float(np.percentile(bs, 2.5)), "acc.hi": float(np.percentile(bs, 97.5)), "acc": float(A[:, 0].mean()), "toks": float(A[:, 1].mean()), "wait_per_1k": float(1000 * A[:, 2].sum() / max(A[:, 1].sum(), 1)), "n": len(d)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="rlzero,olmo3think,r1d7,qwen3_8b")
    a = ap.parse_args(); models = a.models.split(",")
    MT = {}; WALL = {}
    for m in models:
        MT[m], WALL[m] = cell_table(m)
    keys = []; summ = {"models": models, "cells": {}, "criteria": {}}
    put = lambda k, v, src="exp/cw/cw_analyze.py": keys.append({"key": k, "value": v, "src": src})
    for cond in CONDS:
        pc = per_cell(MT, cond)
        for (m, col), v in pc.items():
            summ["cells"]["%s|%s|%s" % (m, cond, col)] = v
            for q in ("acc", "acc.lo", "acc.hi", "toks", "wait_per_1k", "n"):
                put("h5.%s.%s.%s.%s" % (m, cond, col, "wrate" if q == "wait_per_1k" else q), v[q])
            put("h5.%s.%s.%s.wait" % (m, cond, col), v["wait_per_1k"])
            if WALL[m].get((cond, col)):
                put("h5.%s.%s.%s.wall" % (m, cond, col), WALL[m][(cond, col)])
        cells = [v for v in pc.values()]
        if cells:
            put("h5.macro_abs.%s.acc" % cond, float(np.mean([c["acc"] for c in cells]))); put("h5.macro_abs.%s.acc.lo" % cond, float(np.mean([c["acc.lo"] for c in cells])), "mean of per-cell lower bounds (conservative-ish; use paired macro keys for inference)")
            put("h5.macro_abs.%s.acc.hi" % cond, float(np.mean([c["acc.hi"] for c in cells]))); put("h5.macro_abs.%s.toks" % cond, float(np.mean([c["toks"] for c in cells]))); put("h5.macro_abs.%s.wait" % cond, float(np.mean([c["wait_per_1k"] for c in cells])))
        for m in models:
            if WALL[m].get(cond):
                put("h5.%s.%s.wall_s" % (m, cond), WALL[m][cond]); put("h5.%s.%s.wall" % (m, cond), WALL[m][cond])
    for cond in CONDS:
        if cond == "none":
            continue
        for which in ("acc", "tok"):
            r = macro(MT, cond, "none", which)
            if r:
                summ["criteria"]["%s.vs_vanilla.%s" % (cond, which)] = r
                for q in ("est", "lo", "hi"):
                    put("h5.macro.%s.%s_vs_vanilla.%s" % (cond, "acc" if which == "acc" else "tokred", q), r[q])
    # signed contrasts (arm - ref)/ref for tokens, arm - ref for accuracy; pooled macro and per model; ref = vanilla (none) and sham (descriptive)
    for cond in CONDS:
        for ref, rn in (("none", "vanilla"), ("sham", "sham")):
            if cond == ref or (cond == "none" and ref == "sham"):
                continue
            for which, nm in (("acc", "acc"), ("tokd", "tok")):
                r = macro(MT, cond, ref, which)
                if r:
                    for q in ("est", "lo", "hi"):
                        put("h5.macro.%s.%s_vs_%s.%s" % (cond, nm, rn, q) if rn == "sham" or nm == "acc" else "h5.macro.%s.%s_vs_%s.%s" % (cond, nm, rn, q), r[q])
                        if nm == "tok":
                            put("h5.macro.%s.%s_vs_%s.%s.pct" % (cond, nm, rn, q), 100 * r[q])
                        if nm == "acc":
                            put("h5.macro.%s.%s_vs_%s.%s.pts" % (cond, nm, rn, q), 100 * r[q])
                for m in models:
                    r = macro({m: MT[m]}, cond, ref, which)
                    if r:
                        for q in ("est", "lo", "hi"):
                            put("h5.%s.%s.macro.%s_vs_%s.%s" % (m, cond, nm, rn, q), r[q])
                            if nm == "tok":
                                put("h5.%s.%s.macro.%s_vs_%s.%s.pct" % (m, cond, nm, rn, q), 100 * r[q])
                            if nm == "acc":
                                put("h5.%s.%s.macro.%s_vs_%s.%s.pts" % (m, cond, nm, rn, q), 100 * r[q])
    for name, c1, c2 in (("S2_cw_minus_nowait", "cw", "nowait"), ("S3_cw_minus_random", "cw", "random"), ("S6_cw_minus_bf_match", "cw", "bf_match")):
        # paired difference of two non-vanilla conditions: build on the fly with c2 as the reference
        r = macro(MT, c1, c2, "acc")
        if r:
            summ["criteria"][name] = r
            for q in ("est", "lo", "hi"):
                put("h5.macro.%s.%s" % (name, q), r[q])
    # S1 / S2 / S3 verdicts
    s1 = summ["criteria"].get("cw.vs_vanilla.tok"); s1a = summ["criteria"].get("cw.vs_vanilla.acc")
    if s1 and s1a:
        summ["S1"] = {"token_reduction": s1["est"], "acc_diff": s1a["est"], "acc_diff_lo": s1a["lo"], "pass": bool(s1["est"] >= 0.15 and s1a["lo"] >= -0.015)}
    for nm in ("S2_cw_minus_nowait", "S3_cw_minus_random"):
        if nm in summ["criteria"]:
            summ[nm.split("_")[0]] = {"est": summ["criteria"][nm]["est"], "lo": summ["criteria"][nm]["lo"], "pass": bool(summ["criteria"][nm]["lo"] > 0)}
    # S6: realised mean TEST tokens of the matched comparators vs CW, per model
    for m in models:
        for c in ("random", "bf_match"):
            ratios = []
            for col in ("math500", "aime", "amc23", "gsm8k"):
                a1, a0 = MT[m].get((c, col)), MT[m].get(("cw", col))
                if a1 and a0:
                    ratios.append(np.mean([v[1] for v in a1.values()]) / np.mean([v[1] for v in a0.values()]))
            if ratios:
                put("h5.%s.s6.%s_tokens_over_cw" % (m, c), float(np.mean(ratios)))
    json.dump({"entries": keys}, open(ROOT / "results" / "reframe" / "cw_keys.json", "w"), indent=0, default=float)
    json.dump(summ, open(CW / "cw_summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k in ("S1", "S2", "S3")}, default=float), len(keys), "keys")


if __name__ == "__main__":
    main()
