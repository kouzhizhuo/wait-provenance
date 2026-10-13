"""Decoding replicates (exp/_incoming/hostA/main): S1 seeds 1-5 (T=1.0), S1 T=0.6/top-p .95, S2 seed 1, and the S1 fresh-prefix E3 (seed 1).
    python3 exp/reframe_keys11.py -> exp/results/reframe/replicates_keys.json
Keys: h1.s1seed<k>.<base|final>.*, h1.s1t06.<base|final>.*, h1.s2seed1.<base|late>.* (same index set as the main series), h1.s1seeds.<base|final>.<q>_{mean,std,min,max,n} over seeds 0-5
(seed 0 = the main run), h4.s1seed1.<stage>.* (E3 on fresh prefixes, Qwen judge; *_oss for gpt-oss)."""
from __future__ import annotations

import numpy as np

from reframe_keys import OUT, ROOT, Ex, common
from exp.pipeline import keys as K, stats as st
from exp.pipeline.scale_report import succ_table

INC = ROOT / "_incoming" / "hostA" / "main"
MAIN = ROOT / "results" / "main"


def idx_obj(path):
    j = common.read_json(path)
    return {"step": None, "acc": j["acc"], "truncated": j["truncated"], "n_samples": j["n_problems"] * (2 if "S1" in str(path) else 2), "indices": j["indices"]}


def main():
    ex = K.Exporter()
    ex.put = lambda key, value, kind, src, _p=ex.put: _p(key, value, kind, src)
    out = []
    def export(series, stages, srcdir, pat):
        h1 = {"checkpoints": {}, "trends": {}}
        for st_, f in stages.items():
            p = srcdir / (pat % f)
            if p.exists():
                h1["checkpoints"][st_] = idx_obj(p)
        if h1["checkpoints"]:
            K.export_h1(ex, h1, series, str(srcdir) + "/" + pat, {s: s for s in h1["checkpoints"]})
        return h1
    # S1 seeds and T=0.6
    per_seed = {}
    for k in range(1, 6):
        if k == 1:   # wave-1 layout: <dir>/idx/<ckpt>.json
            h = export("s1seed1", {"base": "base", "final": "final"}, INC / "S1sd1" / "idx", "%s.json")
        else:
            h = export("s1seed%d" % k, {"base": "base", "final": "final"}, INC / ("S1sd%d" % k) / "gen", "S1sd%d_%%s.idx.json" % k)
        per_seed[k] = h["checkpoints"]
    export("s1t06", {"base": "base", "final": "final"}, INC / "S1T06" / "idx", "%s.json")
    export("s2seed1", {"base": "base", "late": "1375"}, INC / "S2sd1" / "gen", "S2sd1_%s.idx.json")
    # seed summary over seeds 0..5 (seed 0 = main run, from h1_indices.json)
    m0 = common.read_json(MAIN / "S1" / "h1_indices.json")["checkpoints"]
    for stage in ("base", "final"):
        vals = {}
        runs = [m0[stage]] + [per_seed[k][stage] for k in range(1, 6) if stage in per_seed[k]]
        for q in ("FREQ_wait_1k", "ORTH", "SI", "N_eff", "COMMA_DROP"):
            xs = [r["indices"][q]["est"] if isinstance(r["indices"].get(q), dict) else r["indices"].get(q) for r in runs]
            xs = [x for x in xs if x is not None and x == x]
            if len(xs) >= 3:
                vals[q] = xs
        accs = [r["acc"] for r in runs]
        kinds = {"FREQ_wait_1k": ("freq", "dec2", 1), "ORTH": ("orth", "dec1", 100), "SI": ("si", "dec1", 100), "N_eff": ("neff", "dec2", 1), "COMMA_DROP": ("commadrop", "dec1", 100)}
        for q, xs in vals.items():
            nm, kd, mul = kinds[q]
            for s_, f in (("mean", np.mean), ("std", lambda a: np.std(a, ddof=1)), ("min", np.min), ("max", np.max)):
                ex.put("h1.s1seeds.%s.%s_%s" % (stage, nm, s_), f(np.array(xs) * mul), kd, "exp/_incoming/hostA/main/S1sd{1..5} + results/main/S1 (seed 0): %s over %d runs" % (q, len(xs)))
            ex.put("h1.s1seeds.%s.%s_n" % (stage, nm), len(xs), "int", "number of decoding-seed runs")
        for s_, f in (("mean", np.mean), ("std", lambda a: np.std(a, ddof=1)), ("min", np.min), ("max", np.max)):
            ex.put("ctl.s1seeds.%s.acc_%s" % (stage, s_), 100 * f(np.array(accs)), "dec1", "MATH-500 accuracy over decoding seeds")
    # S1 fresh-prefix E3 (seed 1)
    E = INC / "S1seed1" / "insertion"
    for suf, tag in (("", ""), ("_oss", "_oss")):
        for src, fn in (("late", "judged_late%s.jsonl.gz"), ("base", "judged_basesrc%s.jsonl.gz")):
            f = E / (fn % suf)
            if not f.exists():
                f = E / ("judged_%s%s.jsonl.gz" % ("late" if src == "late" else "basesrc", suf))
            if not f.exists():
                continue
            rows = common.read_jsonl(f)
            pids, succ, n_per = succ_table(rows)
            order = [p for p in ("S1@base", "S1@1900", "S1@final") if p in succ]
            res = st.insertion_estimates({p: succ[p] for p in order}, n_per, B=2000, seed=20261012, base="S1@base", last="S1@final")
            for pol, stage in (("S1@base", "base"), ("S1@1900", "late"), ("S1@final", "final")):
                if pol not in succ:
                    continue
                for q, name, kd in (("rd_op", "RD_next", "dec1"), ("or_op", "OR_next", "dec2"), ("rd_syn", "RD_syn", "dec1")):
                    e = res.get("%s:%s" % (pol, name))
                    if e:
                        ex.put_est("h4.s1seed1.%s.%s_%s%s" % (stage, q, src, tag), e, kd, "exp/_incoming/hostA/main/S1seed1/insertion/%s (fresh prefixes, seed 1; n_prefixes=%d)" % (f.name, len(pids)))
                for arm, v in succ[pol].items():
                    p_ = st.proportion_ci(v, n_per, B=2000, seed=2)
                    ex.put_est("h4.s1seed1.%s.prev_%s_%s%s" % (stage, K.ARM_KEY.get(arm, arm), src, tag), {"est": p_[0], "lo": p_[1], "hi": p_[2]}, "pct", "same")
            for q, name, kd in (("r_amp", "R_amp_RD_next", "dec2"), ("d_amp", "dRD_next_last_minus_base", "dec1")):
                if name in res:
                    ex.put_est("h4.s1seed1.trend.%s_%s%s" % (q, src, tag), res[name], kd, "fresh-prefix E3, final vs base")
    common.write_json(OUT / "replicates_keys.json", {"entries": ex.entries})
    e = {x["key"]: x["value"] for x in ex.entries}
    print(len(e), "keys")
    print({k: v for k, v in e.items() if k.startswith(("h1.s1seeds.final.freq", "h1.s1seeds.base.freq", "h1.s1t06.final.freq", "h1.s1t06.base.freq")) and not k.endswith((".lo", ".hi"))})
    print({k: v for k, v in e.items() if k.startswith("h4.s1seed1") and ("rd_op" in k) and not k.endswith((".lo", ".hi"))})


if __name__ == "__main__":
    main()
