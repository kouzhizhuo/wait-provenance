"""OLMo-3-7B fork: (a) finished-only keys h1.<series>.<stage>.fin.* for the ORIGINAL fork eval (fork_ref1: 200 problems x 8, cap 3,072) and (b) the MATCHED eval fork_ref1m
(S1 main-eval settings: MATH-500 x 2, T 1.0, top-p 1.0, max_tokens 8,192; stage 'base' = the S1 base run itself) with the S1 index set and its own .fin.* keys.
    python3 exp/reframe_keys14.py  [needs exp/rl/results/olmo3_ref_s1_p/matched/{gen/s<N>.graded.jsonl.gz,idx/s<N>.json} synced from hostA]
-> exp/results/reframe/fork_matched_keys.json.  fin = finished-only (finish == 'stop'): nwait, ntok, nfin, nresp, freq (+exact Poisson .lo/.hi), acc (among finished)."""
from __future__ import annotations

import gzip
import io
import json
import re
import tarfile
from pathlib import Path

from reframe_keys import OUT, ROOT, Ex, common, poisson_ci
from reframe_keys13 import ratio_ci
from pipeline import keys as K
from pipeline.indices import sum_counts, trace_counts

RLR = ROOT / "rl" / "results" / "olmo3_ref_s1_p"
MATCH = RLR / "matched"
S1BASE = ROOT / "results" / "main" / "S1" / "gen" / "<EMAIL>"


def stg(step):
    return "base" if step == 0 else "step%04d" % step


def rows_of(fh):
    return [json.loads(l) for l in io.TextIOWrapper(gzip.open(fh), encoding="utf-8")]


def fin_stats(rows):
    fin = [r for r in rows if r.get("finish") == "stop"]
    c = sum_counts([trace_counts(r["text"], r["n_tokens"]) for r in fin])
    return {"nresp": len(rows), "nfin": len(fin), "nwait": int(c.get("wait:n", 0)), "ntok": int(c["tokens"]),
            "acc": sum(bool(r["correct"]) for r in fin) / max(len(fin), 1)}


def put_fin(X, series, step, st, src):
    b = "h1.%s.%s.fin." % (series, stg(step))
    lo, hi = poisson_ci(st["nwait"])
    X.put(b + "nwait", st["nwait"], "int", src); X.put(b + "ntok", st["ntok"], "int", src)
    X.put(b + "nfin", st["nfin"], "int", src); X.put(b + "nresp", st["nresp"], "int", src)
    X.put(b + "freq", 1000 * st["nwait"] / st["ntok"], "dec2", src + "; finished responses only")
    X.put(b + "freq.lo", 1000 * lo / st["ntok"], "dec2", src + "; exact Poisson 95%"); X.put(b + "freq.hi", 1000 * hi / st["ntok"], "dec2", src + "; exact Poisson 95%")
    X.put(b + "acc", st["acc"], "pct", src + "; accuracy among finished responses")
    X.put(b + "finshare", st["nfin"] / st["nresp"], "pct", src)


def trend(X, series, fin, last, what):
    a, b = fin[0], fin[last]
    est, lo, hi = ratio_ci(a["nwait"], a["ntok"], b["nwait"], b["ntok"])
    for q, v in (("", est), (".lo", lo), (".hi", hi)):
        X.put("h1.%s.trend.a_wait_fin%s" % (series, q), v, "dec2", "FREQ_wait(step %d)/FREQ_wait(step 0), finished responses only, %s" % (last, what))
    X.put("h1.%s.trend.fin_last_step" % series, last, "int", what)


def main():
    ex, X = K.Exporter(), Ex()
    summ = {}
    # (a) original eval, finished-only (from the archived raw generations)
    g = sorted((RLR).glob("gen_all_raw_step*.tgz"))[-1]
    t = tarfile.open(g)
    fin = {}
    for m in t.getmembers():
        mm = re.fullmatch(r"(?:\./)?gen/s(\d+)\.graded\.jsonl\.gz", m.name)
        if mm:
            fin[int(mm.group(1))] = fin_stats(rows_of(t.extractfile(m)))
    for s, st in sorted(fin.items()):
        put_fin(X, "fork_ref1", s, st, "%s::gen/s%d.graded.jsonl.gz" % (g.name, s))
    if 0 in fin and len(fin) > 1:
        trend(X, "fork_ref1", fin, max(fin), "original fork eval (cap 3,072, n=8)")
    summ["fork_ref1_fin"] = {s: round(1000 * v["nwait"] / v["ntok"], 2) for s, v in sorted(fin.items())}
    # (b) matched eval, for fork_ref1m (REF s1_p) and fork_b0m (beta=0)
    for series, MATCH_, base_idx in (("fork_ref1m", MATCH, "s0_from_S1.json"), ("fork_b0m", ROOT / "rl" / "results" / "olmo3_beta0_s1_p" / "matched", "s0_from_S1.json")):
        if not (MATCH_.exists() and (MATCH_ / "gen").exists()):
            continue
        idx, finm, gens = {}, {}, {}
        gens[0] = S1BASE
        for f in sorted((MATCH_ / "gen").glob("s*.graded.jsonl.gz")):
            if f.name.count(".") == 3 and "greedy" not in f.name:
                gens[int(re.match(r"s(\d+)\.", f.name).group(1))] = f
        for s, f in sorted(gens.items()):
            rows = rows_of(open(f, "rb"))
            finm[s] = fin_stats(rows)
            put_fin(X, series, s, finm[s], str(f))
            ij = MATCH_ / "idx" / ("s%d.json" % s)
            if s == 0:
                ij = MATCH / "idx" / base_idx
            if ij.exists():
                j = json.load(open(ij))
                idx[s] = {"step": s, "acc": j["acc"], "truncated": j["truncated"], "n_samples": j["n_samples"], "indices": j["indices"]}
        h1 = {"checkpoints": {("base" if s == 0 else str(s)): v for s, v in idx.items()}, "trends": {}}
        K.export_h1(ex, h1, series, "exp/rl/results/.../matched/idx", {})
        if len(finm) > 1:
            trend(X, series, finm, max(finm), "matched eval (S1 settings)")
        tot = {}
        for s in idx:
            j = json.load(open((MATCH_ / "idx" / ("s%d.json" % s)) if s else (MATCH / "idx" / base_idx)))
            tot[s] = (int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"]))
        if len(tot) > 1:
            last = max(tot)
            est, lo, hi = ratio_ci(tot[0][0], tot[0][1], tot[last][0], tot[last][1])
            for q, v in (("", est), (".lo", lo), (".hi", hi)):
                X.put("h1.%s.trend.a_wait%s" % (series, q), v, "dec2", "FREQ_wait(step %d)/FREQ_wait(base), all tokens, matched eval" % last)
            X.put("h1.%s.trend.last_step" % series, last, "int", "matched eval")
        summ[series + "_fin"] = {s: round(1000 * v["nwait"] / v["ntok"], 2) for s, v in sorted(finm.items())}
    common.write_json(OUT / "fork_matched_keys.json", {"entries": ex.entries + X.entries})
    print(summ, len(ex.entries) + len(X.entries), "keys")


if __name__ == "__main__":
    main()
