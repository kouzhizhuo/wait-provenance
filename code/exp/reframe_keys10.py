"""D43 seeded-marker GRPO keys from exp/rl/results/seed_<ARM><N>/results.tgz (infra/D39 archives: idx/sN.json + gen/sN.graded.jsonl.gz; s0 = the SFT-only checkpoint).
    python3 exp/reframe_keys10.py -> exp/results/reframe/seed_keys.json, seed_summary.json
Keys: h1.seed.<arm><N>.<stage>.{freq4,freq_next4,freq_nextopen4,nwait,ntok}[.lo/.hi], ctl.seed.<arm><N>.<stage>.acc, h1.seed.<arm><N>.trend.{a_wait,a_next,a_nextopen}[.lo/.hi].
Definitions: freq4 = occurrences of 'wait' per 1k response tokens (indices.trace_counts, as everywhere); freq_next4 = occurrences of the word 'next' per 1k tokens (all uses);
freq_nextopen4 = sentence-initial 'Next,' per 1k tokens (the form arm X supplies). A = FREQ(step 400) / FREQ(SFT-only), cluster bootstrap over problems (B=2000)."""
from __future__ import annotations

import io
import re
import tarfile

import numpy as np

from reframe_keys import OUT, ROOT, Ex, common, ix, mk, poisson_ci, read_gz_jsonl

STEPS = {0: "sft", 20: "step0020", 100: "step0100", 200: "step0200", 300: "step0300", 400: "step0400"}
NEXT = re.compile(r"\bnext\b", re.I)
NEXT_OPEN = re.compile(r"^[\s\"'*_>#-]*next\s*,", re.I)


def per_problem(rows):
    d = {}
    for r in rows:
        c = ix.trace_counts(r["text"], int(float(r["n_tokens"])))
        t = r["text"]
        no = sum(1 for s in mk.split_sentences(t) if NEXT_OPEN.match(s.text))
        a = d.setdefault(str(r["pid"]), np.zeros(5))
        a += np.array([c.get("wait:n", 0), len(NEXT.findall(t)), no, c["tokens"], str(r["correct"]) == "True"], float)
    return d


def tot(d):
    return np.sum(list(d.values()), axis=0)


def ratio_ci(d0, d1, k, B=2000, seed=20261004):
    p0, p1 = sorted(d0), sorted(d1)
    common_p = [p for p in p0 if p in d1]
    A0 = np.array([d0[p] for p in common_p]); A1 = np.array([d1[p] for p in common_p])
    def r(i):
        a, b = A0[i].sum(0), A1[i].sum(0)
        f0 = a[k] / a[3]; f1 = b[k] / b[3]
        return f1 / f0 if f0 > 0 else float("nan")
    full = r(np.arange(len(common_p)))
    rng = np.random.default_rng(seed)
    reps = [r(rng.integers(0, len(common_p), len(common_p))) for _ in range(B)] if False else [r(rng.integers(0, len(common_p), len(common_p))) for _ in range(2000)]
    reps = [x for x in reps if x == x]
    lo, hi = (np.percentile(reps, 2.5), np.percentile(reps, 97.5)) if reps else (float("nan"), float("nan"))
    return full, lo, hi


def main():
    X = Ex(); summ = {}
    for d in sorted((ROOT / "rl" / "results").glob("seed_*")):
        arm = d.name.replace("seed_", "")
        tg = d / "results.tgz"
        if not tg.exists():
            continue
        t = tarfile.open(tg)
        data = {}
        for step in STEPS:
            try:
                data[step] = (common.read_json(io.StringIO(t.extractfile("idx/s%d.json" % step).read().decode())) if False else __import__("json").load(t.extractfile("idx/s%d.json" % step)),
                              read_gz_jsonl(t.extractfile("gen/s%d.graded.jsonl.gz" % step).read()))
            except KeyError:
                continue
        pp = {}
        for step, (j, rows) in data.items():
            stg = STEPS[step]
            pp[step] = per_problem(rows)
            nw, nx, no, tk, ok = tot(pp[step])
            src = "exp/rl/results/seed_%s/results.tgz::gen/s%d.graded.jsonl.gz" % (arm, step)
            b = "h1.seed.%s.%s." % (arm, stg)
            for q, n_, nm in (("freq4", nw, "wait"), ("freq_next4", nx, "next (all uses)"), ("freq_nextopen4", no, "sentence-initial 'Next,'")):
                lo, hi = poisson_ci(int(n_))
                X.put(b + q, 1000 * n_ / tk, "dec4", src + "::" + nm + " per 1k tokens")
                X.put(b + q + ".lo", 1000 * lo / tk, "dec4", "exact Poisson 95%"); X.put(b + q + ".hi", 1000 * hi / tk, "dec4", "exact Poisson 95%")
                X.put(b + q.replace("4", "2"), 1000 * n_ / tk, "dec2", src + "::" + nm + " per 1k tokens (2 dp)")
            X.put(b + "nwait", nw, "int", src); X.put(b + "ntok", tk, "int", src)
            X.put("ctl.seed.%s.%s.acc" % (arm, stg), j["acc"], "pct", "exp/rl/results/seed_%s/results.tgz::idx/s%d.json::acc (MATH-500, 8 samples x 500)" % (arm, step))
            X.put("ctl.seed.%s.%s.toks_per_resp" % (arm, stg), tk / len(data[step][1]), "int", src)
            # truncation (finish == length, i.e. hitting the 3,072-token cap = reward 0) and the same indices split by finish status
            tr = [r for r in data[step][1] if r["finish"] == "length"]; fin = [r for r in data[step][1] if r["finish"] != "length"]
            srct = src + "::finish field; cap = 3072 tokens (REF context)"
            X.put("ctl.seed.%s.%s.trunc" % (arm, stg), 100 * len(tr) / len(data[step][1]), "dec1", srct)
            for nm_, rs in (("finished", fin), ("trunc", tr)):
                if rs:
                    X.put("ctl.seed.%s.%s.acc_%s" % (arm, stg, nm_), 100 * np.mean([str(r["correct"]) == "True" for r in rs]), "dec1", srct)
                    c_ = tot(per_problem(rs))
                    X.put("h1.seed.%s.%s.freq_%s4" % (arm, stg, nm_), 1000 * c_[0] / c_[3], "dec4", srct + "::wait per 1k tokens among %s responses" % nm_)
                    X.put("h1.seed.%s.%s.freq_%s2" % (arm, stg, nm_), 1000 * c_[0] / c_[3], "dec2", srct)
        if 0 in pp and 400 in pp:
            summ[arm] = {}
            for q, k, nm in (("a_wait", 0, "wait"), ("a_next", 1, "next"), ("a_nextopen", 2, "nextopen")):
                full, lo, hi = ratio_ci(pp[0], pp[400], k)
                f0 = tot(pp[0])[k] / tot(pp[0])[3]; f1 = tot(pp[400])[k] / tot(pp[400])[3]
                lo_p, hi_p = poisson_ci(int(tot(pp[400])[k]))
                hi_ratio = (1000 * hi_p / tot(pp[400])[3] / 1000) / f0 if f0 > 0 else float("nan")
                summ[arm][q] = {"est": full, "boot_lo": lo, "boot_hi": hi, "poisson_upper_of_ratio": hi_ratio, "f_sft_per_1k": 1000 * f0, "f_400_per_1k": 1000 * f1}
                src = "exp/rl/results/seed_%s/results.tgz: FREQ(400)/FREQ(SFT-only), cluster bootstrap over problems B=2000" % arm
                X.put("h1.seed.%s.trend.%s" % (arm, q), full, "dec4", src)
                X.put("h1.seed.%s.trend.%s.lo" % (arm, q), lo, "dec4", src); X.put("h1.seed.%s.trend.%s.hi" % (arm, q), hi, "dec4", src)
                X.put("h1.seed.%s.trend.%s.pois_hi" % (arm, q), hi_ratio, "dec4", "upper 95% bound of the ratio from the exact Poisson upper bound of the step-400 count")
    common.write_json(OUT / "seed_keys.json", {"entries": X.entries})
    common.write_json(OUT / "seed_summary.json", summ)
    for arm, v in summ.items():
        print(arm, {q: {k: (round(x, 4) if x == x else x) for k, x in d.items()} for q, d in v.items()})
    print(len(X.entries), "keys")


if __name__ == "__main__":
    main()
