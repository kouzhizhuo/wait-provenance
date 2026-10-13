"""Second batch of reframe keys (local only): exposure-adjusted bounds, design counts, case-study counts.  python3 exp/reframe_keys2.py
Writes exp/results/reframe/{exposure,design,case}_keys.json."""
from __future__ import annotations

import glob
import json
from pathlib import Path

from reframe_keys import OUT, RES, ROOT, Ex, Ref, common, counts, est, ix, poisson_ci, read_gz_jsonl

REL = lambda p: str(Path(p).relative_to(ROOT.parent))


def main():
    fk = {e["key"]: e["value"] for e in common.read_json(OUT / "freq_keys.json")["entries"]}
    # ---------------------------------------------------------------- (1) exposure-adjusted bounds
    E = Ex()
    cells = [("s8base", "final"), ("s8dapo", "final"), ("s8orz", "final"), ("s7r1", "final"), ("s7r2", "final"), ("s7r7b", "final"), ("s3a", "late"), ("s3a", "max")]
    uppers = {}
    for s, stg in cells:
        nw, tk = fk["h1.%s.%s.nwait" % (s, stg)], fk["h1.%s.%s.ntok" % (s, stg)]
        lo, hi = poisson_ci(int(nw))
        src = "exact Poisson 95%% CI (chi-square form) on nwait=%d over ntok=%d (exp/results/reframe/freq_keys.json)" % (nw, tk)
        E.put("h1.%s.%s.freq4.lo" % (s, stg), 1000 * lo / tk, "dec4", src)
        E.put("h1.%s.%s.freq4.hi" % (s, stg), 1000 * hi / tk, "dec4", src)
        uppers[(s, stg)] = 1000 * hi / tk
    last5 = [("s7r1", "final"), ("s7r2", "final"), ("s7r7b", "final"), ("s8dapo", "final"), ("s8orz", "final")]   # the four zero-RL runs (S3a is RL from an instruct model: h1.rlfree)
    mx = max(last5, key=lambda c: uppers[c])
    E.put("h1.zerorl.upper95_max", uppers[mx], "dec4", "max over the last checkpoints of the 5 zero-RL runs of the exact Poisson 95%% upper bound per 1k tokens; attained by %s.%s" % mx)
    common.write_json(OUT / "exposure_keys.json", {"entries": E.entries})

    # ---------------------------------------------------------------- (2) design counts
    D = Ex()
    ck = {"s1": 9, "s1pre": 5, "s2": 8, "s2ctl": 5, "s3a": 7, "s8": 5, "s9": 7, "s10": 1, "s7": None}
    refs = [Ref(1), Ref(2)]
    steps = [set(r.idx().keys()) for r in refs]
    # simpler and explicit: shared base (step 0) + non-zero steps of seed 1 + non-zero steps of seed 2
    ck["s7"] = 1 + len(steps[0] - {0}) + len(steps[1] - {0})
    rule = ("distinct weights with E1 generations: S1 9 (base,100,200,400,700,1000,1400,1900,final); S1-pre 5 (6 branches, stage3-step11921 = S1 base); S2 8; S2-ctl 5; S3a 7; S8 5; S9 7; "
            "S10 1 (gpt-oss-120b, 3 effort settings, one weight set); S7 = shared step 0 + non-zero steps of each REF seed (ASSUMES both seeds start from the same initial weights)")
    for k, v in ck.items():
        D.put("design.nckpt.%s" % k, v, "int", rule)
    tot = sum(ck.values())
    D.put("design.nckpt", tot, "int", rule)
    D.put("n.ckpts", tot, "int", rule)
    # traces: MATH-500 E1 traces (one row = one response) of the main evidence base; robustness re-runs (greedy, boxed, T=0.7) and AIME excluded
    tr = {}
    for s, ser, pat in (("s1", "S1", "S1@*.graded.jsonl.gz"), ("s2", "S2", "S2@*.graded.jsonl.gz"), ("s3a", "S3", "S3@*.graded.jsonl.gz"), ("s2ctl", "S2ctl", "S2ctl@*.graded.jsonl.gz")):
        tr[s] = sum(len(read_gz_jsonl(f)) for f in (RES / "main" / ser / "gen").glob(pat))
    tr["s1pre"] = sum(len(read_gz_jsonl(f)) for f in (RES / "main" / "S1pre" / "gen").glob("S1pre@*.graded.jsonl.gz") if f.exists() and "stage3-step11921" not in f.name)
    tr["s9"] = sum(len(read_gz_jsonl(f)) for f in (RES / "s9" / "S9" / "gen").glob("S9@*.graded.jsonl.gz"))
    tr["s10"] = sum(len(read_gz_jsonl(f)) for f in (RES / "scale").glob("S10_*/gen/*math500.graded.jsonl.gz"))
    sm = common.read_json(RES / "scale" / "scale_summary.json")["models"]
    tr["s8"] = int(sum(sm[m]["E1_math500"]["totals"]["n_resp"] for m in sm if m.startswith("S8_")))
    tr["s7"] = sum(int(j["n_samples"]) for r in refs for j in r.idx().values())
    for k, v in tr.items():
        D.put("n.traces.%s" % k, v, "int", "rows of the local graded E1 MATH-500 generation files (S8: totals.n_resp in scale_summary.json; S7: n_samples of the REF idx files, both seeds); S1-pre excludes the linked S1-base branch")
    D.put("n.traces", sum(tr.values()), "int", "sum of n.traces.<series>: MATH-500 E1 traces of the main evidence base (no greedy/boxed/T=0.7 reruns, no AIME)")
    # GPU-hours: lower bound from the local *.meta.json of GPU stages
    gh, byst = 0.0, {}
    for f in glob.glob(str(ROOT / "results" / "**" / "*.meta.json"), recursive=True) + glob.glob(str(ROOT / "rl" / "results" / "**" / "*.meta.json"), recursive=True):
        if "/._" in f:
            continue
        m = json.load(open(f))
        if m.get("stage") not in ("gen", "insert_continue", "judge", "revlab", "lsc_embed", "hsub"):
            continue
        ex = m.get("extra", {}) or {}
        g = ex.get("gpu_hours")
        if g is None and m.get("wall_seconds") is not None:
            g = m["wall_seconds"] * (len(ex.get("gpus") or []) or len((m.get("env", {}).get("gpus") or [])) or 1) / 3600
        byst[m["stage"]] = byst.get(m["stage"], 0.0) + (g or 0.0)
        gh += g or 0.0
    for k, v in byst.items():
        D.put("n.gpuh.%s" % k, v, "dec1", "sum of wall_seconds x n_gpus over the local meta files of this stage")
    D.put("n.gpuh", gh, "int", "LOWER BOUND: GPU stages (gen, insert_continue, judge, revlab) from the LOCAL *.meta.json only; excludes remote-only runs not harvested and the GRPO training jobs")
    common.write_json(OUT / "design_keys.json", {"entries": D.entries})

    # ---------------------------------------------------------------- (4) case study counts (S1, one problem; base / mid / last)
    C = Ex()
    gens = {st: read_gz_jsonl(RES / "main" / "S1" / "gen" / ("S1@%s.graded.jsonl.gz" % ck_)) for st, ck_ in (("base", "base"), ("mid", "700"), ("last", "final"))}
    ok = {st: {(str(r["pid"]), str(r["sid"])): str(r["correct"]) == "True" for r in rows} for st, rows in gens.items()}
    cand = sorted((p, s) for (p, s) in ok["base"] if all(ok[st].get((p, s)) for st in ok) and s == "0")
    cand = sorted(cand, key=lambda x: int(x[0]))
    pid, sid = cand[0]
    rule_txt = ("selection rule (written before looking at the counts): the first MATH-500 problem in dataset order (smallest pid) whose sample sid=0 is correct at base, step 700 and final; "
                "no marker-based criterion. Counts are per response (sid=0): n_wait = occurrences of 'wait' (indices.trace_counts), n_rev = rule-based revision events (revisions.py), chains = reduplicated wait runs")
    C.put("fig1.case.pid", int(pid), "int", rule_txt)
    for st, rows in gens.items():
        r = next(x for x in rows if str(x["pid"]) == pid and str(x["sid"]) == sid)
        c = ix.trace_counts(r["text"], int(float(r["n_tokens"])))
        for q, v in (("nwait", c.get("wait:n", 0)), ("nrev", c.get("R:n", 0)), ("ntok", c["tokens"]), ("nchain", c.get("wait:chain_runs", 0))):
            C.put("fig1.case.%s.%s" % (st, q), v, "int", rule_txt)
    common.write_json(OUT / "case_keys.json", {"entries": C.entries})
    print({"exposure": len(E.entries), "design": len(D.entries), "case": len(C.entries)}, "pid", pid, "upper95_max", uppers[mx], mx, "n.traces", sum(tr.values()), "nckpt", tot, "gpuh", round(gh, 1))


if __name__ == "__main__":
    main()
