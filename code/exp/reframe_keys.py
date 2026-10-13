"""R1 reframe keys from LOCAL files only (no remote access). Writes {"entries":[{key,value,src}]} files under exp/results/reframe/.

    python3 exp/reframe_keys.py            (from the P1 folder; needs numpy/scipy, the pipeline package, the local exp/results tree and exp/rl/results/ref_s{1,2})

Parts: freq (freq2/freq4/nwait/ntok, zero-RL run count) | ref (S7 REF seeds per step) | reflopen (scope check) | s1pre | rd_trend | crosstab | bm_late_half.
S9 SDT and the S2-ctl aggregation are run through the existing CLIs (see the end of main()). Nothing is imputed: whatever cannot be computed locally
is listed in exp/results/reframe/MISSING.md.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from exp.pipeline import common, indices as ix, keys as K, markers as mk, stats as st   # noqa: E402

RES = ROOT / "results"
OUT = RES / "reframe"
MISSING: list = []


def rnd(v, kind):
    if kind == "dec4":
        return None if v is None or v != v else round(float(v), 4)
    return K.rnd(v, kind)


class Ex(K.Exporter):
    def put(self, key, value, kind, src):
        v = rnd(value, kind)
        if v is not None:
            self.entries.append({"key": key, "value": v, "src": src})


def poisson_ci(n, alpha=0.05):
    lo = 0.0 if n == 0 else chi2.ppf(alpha / 2, 2 * n) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * (n + 1)) / 2
    return lo, hi


def read_gz_jsonl(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        f = gzip.open(io.BytesIO(path_or_bytes), "rt", encoding="utf-8")
    else:
        f = gzip.open(path_or_bytes, "rt", encoding="utf-8") if str(path_or_bytes).endswith(".gz") else open(path_or_bytes, encoding="utf-8")
    with f:
        return [json.loads(l) for l in f if l.strip()]


def counts(rows):
    c = {"wait:n": 0.0, "tokens": 0.0}
    for r in rows:
        t = ix.trace_counts(r["text"], int(float(r["n_tokens"])))
        c["wait:n"] += t.get("wait:n", 0.0)
        c["tokens"] += t["tokens"]
    return int(c["wait:n"]), int(c["tokens"])


class Ref:
    """REF GRPO seeds (S7): exp/rl/results/ref_s{1,2}/results.tgz (+ gen_all_raw.tgz)."""

    def __init__(self, n):
        self.n = n
        self.tag = "ref7b_s1" if n == 7 else "ref_s%d" % n   # n=7: the 7B negative control (Qwen2.5-Math-7B, REF recipe, seed 1)
        self.sid = "s7r7b" if n == 7 else "s7r%d" % n
        self.tgz = ROOT / "rl" / "results" / self.tag / "results.tgz"
        self.t = tarfile.open(self.tgz) if self.tgz.exists() else None

    def idx(self):
        out = {}
        if not self.t:
            return out
        for m in self.t.getmembers():
            mm = re.fullmatch(r"(?:\./)?idx/s(\d+)\.json", m.name)
            if mm:
                out[int(mm.group(1))] = json.load(self.t.extractfile(m))
        return dict(sorted(out.items()))

    def gen(self, step):
        for nm in ("gen/s%d.graded.jsonl.gz" % step, "./gen/s%d.graded.jsonl.gz" % step):
            try:
                return read_gz_jsonl(self.t.extractfile(nm).read())
            except KeyError:
                continue
        return None


def est(x):
    return x["est"] if isinstance(x, dict) else x


def gens_dir(series, pattern):
    return sorted((RES / "main" / series / "gen").glob(pattern))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A, R_ = Ex(), Ex()
    runs = {}   # distinct local zero-RL runs: id -> dict(base_freq, last_freq, dacc, ...)

    # ------------------------------------------------------------------ S7 REF seeds
    for n in (1, 2, 7):
        ref = Ref(n)
        idx = ref.idx()
        if not idx:
            MISSING.append("REF %s idx (exp/rl/results/%s/results.tgz not readable)" % (ref.sid, ref.tag))
            continue
        s = ref.sid
        src = "exp/rl/results/%s/results.tgz::idx/s<step>.json" % ref.tag
        steps, freqs, accs = [], [], []
        for step, j in idx.items():
            stg = "step%04d" % step
            fr = est(j["indices"]["FREQ_wait_1k"])
            nw = int(j["totals"].get("wait:n", 0))
            R_.put("h1.%s.%s.freq" % (s, stg), fr, "dec4", src + "::indices.FREQ_wait_1k")
            R_.put("h1.%s.%s.nwait" % (s, stg), nw, "int", src + "::totals.wait:n")
            R_.put("ctl.%s.%s.acc" % (s, stg), j["acc"], "pct", src + "::acc")
            R_.put("ctl.%s.%s.trunc" % (s, stg), j["truncated"], "pct", src + "::truncated (finish=length share; cap 3,072)")
            steps.append(step); freqs.append(fr); accs.append(j["acc"])
        R_.put("h1.%s.max.freq" % s, max(freqs), "dec4", src + "::max over steps of FREQ_wait_1k")
        smax = steps[int(np.argmax(freqs))]
        jm = idx[smax]
        nwm, tkm = int(jm["totals"].get("wait:n", 0)), int(jm["totals"]["tokens"])
        lo_, hi_ = poisson_ci(nwm)
        for q, v in (("freq4", 1000 * nwm / tkm), ("freq4.lo", 1000 * lo_ / tkm), ("freq4.hi", 1000 * hi_ / tkm)):
            R_.put("h1.%s.max.%s" % (s, q), v, "dec4", src + "::step %d (max FREQ); exact Poisson 95%%" % smax)
        R_.put("h1.%s.max.step" % s, smax, "int", src + "::step of max FREQ")
        R_.put("h1.%s.max.nwait" % s, int(max(int(j["totals"].get("wait:n", 0)) for j in idx.values())), "int", src + "::max over steps of wait:n")
        if len(steps) >= 3 and len(set(freqs)) > 1:
            rho, p = st.spearman(steps, freqs)
            R_.put("h1.%s.trend.rho_freq" % s, rho, "dec2", src + "::spearman(step, FREQ_wait_1k)")
            if p is not None:
                R_.put("h1.%s.trend.rho_freq.p" % s, p, "dec3", src + "::spearman p")
        else:
            MISSING.append("REF s%d rho_freq undefined: FREQ constant over steps (%s)" % (n, sorted(set(freqs))))
        last, first = idx[max(idx)], idx[min(idx)]
        for stg, j in (("base", first), ("final", last)):
            nw, tk = int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"])
            A.put("h1.%s.%s.freq4" % (s, stg), est(j["indices"]["FREQ_wait_1k"]), "dec4", src)
            A.put("h1.%s.%s.nwait" % (s, stg), nw, "int", src)
            A.put("h1.%s.%s.ntok" % (s, stg), tk, "int", src)
            A.put("ctl.%s.%s.acc" % (s, stg), j["acc"], "pct", src)
            A.put("ctl.%s.%s.trunc" % (s, stg), j["truncated"], "pct", src + "::truncated (finish=length share; cap 3,072)")
        A.put("ctl.%s.trend.dacc" % s, 100 * (last["acc"] - first["acc"]), "dec1", src + "::acc(last)-acc(first), points")
        runs[s] = {"base_freq": est(first["indices"]["FREQ_wait_1k"]), "dacc": 100 * (last["acc"] - first["acc"]), "label": "REF GRPO seed %d (S7)" % n}

    # ------------------------------------------------------------------ S8 (scale_summary) and S3
    sm = common.read_json(RES / "scale" / "scale_summary.json")["models"]
    sid = {"S8_base": "s8base", "S8_dapo": "s8dapo", "S8_orz": "s8orz", "S8_qwq": "s8qwq", "S8_r1distill": "s8r1d"}
    src8 = "exp/results/scale/scale_summary.json::models.%s.E1_math500"
    acc8 = {}
    for m, s in sid.items():
        e = sm[m]["E1_math500"]
        nw, tk = int(e["totals"].get("wait:n", 0)), int(e["totals"]["tokens"])
        fr = est(e["indices"]["FREQ_wait_1k"])
        acc8[m] = e["acc"]
        kind = "freq2" if m in ("S8_qwq", "S8_r1distill") else "freq4"
        A.put("h1.%s.final.%s" % (s, kind), fr, "dec2" if kind == "freq2" else "dec4", src8 % m)
        A.put("h1.%s.final.nwait" % s, nw, "int", src8 % m)
        A.put("h1.%s.final.ntok" % s, tk, "int", src8 % m)
        lo, hi = poisson_ci(nw)
        A.put("h1.%s.final.nwait.lo" % s, lo, "dec1", "Poisson exact 95%")
        A.put("h1.%s.final.nwait.hi" % s, hi, "dec1", "Poisson exact 95%")
        A.put("ctl.%s.final.acc" % s, e["acc"], "pct", src8 % m)
        if m == "S8_base":
            A.put("h1.s8base.base.freq4", fr, "dec4", src8 % m)
    for m in ("S8_dapo", "S8_orz"):
        runs[sid[m]] = {"base_freq": est(sm["S8_base"]["E1_math500"]["indices"]["FREQ_wait_1k"]), "dacc": 100 * (acc8[m] - acc8["S8_base"]), "label": m}
        A.put("ctl.%s.trend.dacc" % sid[m], 100 * (acc8[m] - acc8["S8_base"]), "dec1", src8 % m + "::acc - S8_base acc, points")
    # S3a: all checkpoints from the graded generations
    s3 = {}
    for f in gens_dir("S3", "S3@*.graded.jsonl.gz"):
        ck = f.name.split("@")[1].replace(".graded.jsonl.gz", "")
        rows = read_gz_jsonl(f)
        s3[ck] = (counts(rows), np.mean([str(r["correct"]) == "True" for r in rows]))
    if "base" in s3 and "1566" in s3:
        srcs3 = "exp/results/main/S3/gen/S3@<ckpt>.graded.jsonl.gz::indices.trace_counts"
        for stg, ck in (("base", "base"), ("late", "1566")):
            (nw, tk), acc = s3[ck]
            A.put("h1.s3a.%s.freq4" % stg, 1000 * nw / tk, "dec4", srcs3)
            A.put("h1.s3a.%s.nwait" % stg, nw, "int", srcs3)
            A.put("h1.s3a.%s.ntok" % stg, tk, "int", srcs3)
            A.put("ctl.s3a.%s.acc" % stg, acc, "pct", srcs3)
        mx = max(s3.items(), key=lambda kv: kv[1][0][0] / kv[1][0][1])
        A.put("h1.s3a.max.freq4", 1000 * mx[1][0][0] / mx[1][0][1], "dec4", srcs3 + " (max over checkpoints)")
        A.put("h1.s3a.max.nwait", mx[1][0][0], "int", srcs3)
        A.put("h1.s3a.max.ntok", mx[1][0][1], "int", srcs3)
        A.put("ctl.s3a.trend.dacc", 100 * (s3["1566"][1] - s3["base"][1]), "dec1", srcs3 + " acc(1566)-acc(base), points")
        runs["s3a"] = {"base_freq": 1000 * s3["base"][0][0] / s3["base"][0][1], "dacc": 100 * (s3["1566"][1] - s3["base"][1]), "label": "S3a Qwen2.5-1.5B GRPO"}
    else:
        MISSING.append("S3 gens (base/1566) not local")
    # S3t07 = the same S3 run decoded at T=0.7 (confirmed by its model paths: S3 base and S3@1566); not a separate run
    # S1 / S1-pre
    s1 = {}
    for ck in ("base", "1900", "final"):
        f = RES / "main" / "S1" / "gen" / ("S1@%s.graded.jsonl.gz" % ck)
        if f.exists():
            s1[ck] = counts(read_gz_jsonl(f))
    for stg, ck in (("base", "base"), ("late", "1900"), ("final", "final")):
        if ck in s1:
            nw, tk = s1[ck]
            A.put("h1.s1.%s.freq2" % stg, 1000 * nw / tk, "dec2", "exp/results/main/S1/gen/S1@%s.graded.jsonl.gz::trace_counts" % ck)
            A.put("h1.s1.%s.nwait" % stg, nw, "int", "same")
            A.put("h1.s1.%s.ntok" % stg, tk, "int", "same")
    pre = {}
    for f in sorted((RES / "main" / "S1pre" / "gen").glob("S1pre@*.graded.jsonl.gz")):
        br = f.name.split("@")[1].replace(".graded.jsonl.gz", "")
        if f.exists():   # stage3-step11921 is a symlink to the S1 base traces (same checkpoint); the link target is remote-only
            pre[br] = counts(read_gz_jsonl(f))
    if "stage3-step11921" not in pre and "base" in s1:
        pre["stage3-step11921"] = s1["base"]   # = S1@base generations (linked, identical checkpoint)
    for br, (nw, tk) in pre.items():
        s = "h1.s1pre.%s" % br
        src = "exp/results/main/S1pre/gen/S1pre@%s.graded.jsonl.gz::trace_counts" % br
        lo, hi = poisson_ci(nw)
        for q, v, kind in (("nwait", nw, "int"), ("ntok", tk, "int"), ("freq2", 1000 * nw / tk, "dec2"), ("freq4", 1000 * nw / tk, "dec4"),
                           ("nwait.lo", lo, "dec1"), ("nwait.hi", hi, "dec1"), ("freq4.lo", 1000 * lo / tk, "dec4"), ("freq4.hi", 1000 * hi / tk, "dec4")):
            A.put("%s.%s" % (s, q), v, kind, src + (" (Poisson exact 95% CI)" if "." in q else ""))
    # n_runs: zero-RL = RL from a BASE model (not instruct) whose base FREQ_wait_1k < 0.01 / 1k: REF s1, REF s2, DAPO-32B, ORZ-32B.
    # S3a is GRPO from Qwen2.5-1.5B-INSTRUCT (ARTIFACTS.md S3), so it is "RL from a model that does not use the marker" (rlfree), not zero-RL; its repo has no licence.
    free = {k: v for k, v in runs.items() if v["base_freq"] < 0.01}
    zero = {k: v for k, v in free.items() if k != "s3a"}
    A.put("h1.zerorl.n_runs", len(zero), "int", "count of distinct local zero-RL runs from marker-free BASE models (base FREQ_wait_1k < 0.01): " + ", ".join(sorted(zero)))
    A.put("h1.rlfree.n_runs", len(free), "int", "zero-RL runs plus S3a (GRPO from Qwen2.5-1.5B-Instruct): RL from models that do not use the marker: " + ", ".join(sorted(free)))
    for k, v in free.items():
        A.put("ctl.%s.%s.dacc" % ("zerorl" if k in zero else "rlfree", k), v["dacc"], "dec1", v["label"])
    if zero:
        A.put("ctl.zerorl.dacc_min", min(v["dacc"] for v in zero.values()), "dec1", "min over the zero-RL runs")
        A.put("ctl.zerorl.dacc_max", max(v["dacc"] for v in zero.values()), "dec1", "max over the zero-RL runs")
        A.put("ctl.zerorl.dacc_max_32b", max(v["dacc"] for k, v in zero.items() if k.startswith("s8")), "dec1", "max over the 32B runs")
    common.write_json(OUT / "freq_keys.json", {"entries": A.entries})
    common.write_json(OUT / "ref_keys.json", {"entries": R_.entries})

    # ------------------------------------------------------------------ scope check: sentence-initial reflection openers
    forms = [("wait", r"wait\b"), ("hmm", r"hm+\b"), ("hold_on", r"hold on\b"), ("actually", r"actually\b"), ("okay", r"(?:okay|ok)\b"),
             ("no", r"no(?=[,.!:;]|\s+(?:wait|that|this|it)\b|$)"), ("let_me_check", r"let me (?:just )?(?:check|verify|double[- ]?check|re-?check|re-?verify)\b"),
             ("alternatively", r"alternatively\b"), ("however", r"however\b")]
    pats = [(n, re.compile(r"^[\s\"'*_>#-]*" + p, re.I)) for n, p in forms]

    def opener_rate(rows, B=500, seed=0):
        per = []
        for r in rows:
            sents = mk.split_sentences(r["text"])[1:]   # the response's first sentence is not a reflection
            c = {n: 0 for n, _ in forms}
            for s_ in sents:
                for n, p in pats:
                    if p.match(s_.text):
                        c[n] += 1
                        break
            per.append((len(sents), c))
        N = np.array([p[0] for p in per], float)
        C = {n: np.array([p[1][n] for p in per], float) for n, _ in forms}
        tot = sum(C.values())
        rate = 1000 * tot.sum() / max(N.sum(), 1)
        rate_nowait = 1000 * (tot - C["wait"]).sum() / max(N.sum(), 1)
        rng = np.random.default_rng(seed)
        bs, bn = [], []
        for _ in range(B):
            i = rng.integers(0, len(per), len(per))
            bs.append(1000 * tot[i].sum() / max(N[i].sum(), 1)); bn.append(1000 * (tot[i] - C["wait"][i]).sum() / max(N[i].sum(), 1))
        sh = {n: (100 * C[n].sum() / tot.sum() if tot.sum() else float("nan")) for n, _ in forms}
        return rate, st.percentile_ci(bs), rate_nowait, st.percentile_ci(bn), sh, int(N.sum()), int(tot.sum())

    O = Ex()
    todo = []   # (series id, stage, rows, src)
    for ck, stg in (("base", "base"), ("1900", "late"), ("final", "final")):
        f = RES / "main" / "S1" / "gen" / ("S1@%s.graded.jsonl.gz" % ck)
        if f.exists():
            todo.append(("s1", stg, read_gz_jsonl(f), str(f.relative_to(ROOT.parent))))
    for ck, stg in (("base", "base"), ("1566", "late")):
        f = RES / "main" / "S3" / "gen" / ("S3@%s.graded.jsonl.gz" % ck)
        if f.exists():
            todo.append(("s3a", stg, read_gz_jsonl(f), str(f.relative_to(ROOT.parent))))
    for n in (1, 2, 7):
        ref = Ref(n)
        idx = ref.idx()
        if idx:
            for stg, step in (("base", min(idx)), ("final", max(idx))):
                g = ref.gen(step)
                if g:
                    todo.append((ref.sid, stg, g, "exp/rl/results/%s/results.tgz::gen/s%d.graded.jsonl.gz" % (ref.tag, step)))
    for s, stg, rows, src in todo:
        rate, rci, rnw, nci, sh, nsent, nop = opener_rate(rows)
        O.put("h1.%s.%s.reflopen" % (s, stg), rate, "dec1", src + "::sentence-initial openers per 1k sentences (first sentence excluded)")
        O.put("h1.%s.%s.reflopen.lo" % (s, stg), rci[0], "dec1", src)
        O.put("h1.%s.%s.reflopen.hi" % (s, stg), rci[1], "dec1", src)
        O.put("h1.%s.%s.reflopen_nowait" % (s, stg), rnw, "dec1", src + " (openers other than 'wait')")
        O.put("h1.%s.%s.reflopen_nowait.lo" % (s, stg), nci[0], "dec1", src)
        O.put("h1.%s.%s.reflopen_nowait.hi" % (s, stg), nci[1], "dec1", src)
        O.put("h1.%s.%s.reflopen_n" % (s, stg), nop, "int", src + " (opener count)")
        for n_, v in sh.items():
            O.put("h1.%s.%s.reflopen_share_%s" % (s, stg, n_), v, "dec1", src + " (share of all openers, %)")
    for m_, s_ in (("S8_base", "s8base"), ("S8_dapo", "s8dapo"), ("S8_orz", "s8orz"), ("S8_qwq", "s8qwq"), ("S8_r1distill", "s8r1d")):
        f = RES / "scale" / m_ / "gen" / ("%s_math500.graded.jsonl.gz" % m_)
        if f.exists():
            rows_ = read_gz_jsonl(f)
            rate, rci, rnw, nci, sh, nsent, nop = opener_rate(rows_)
            src = str(f.relative_to(ROOT.parent))
            O.put("h1.%s.final.reflopen" % s_, rate, "dec1", src + "::sentence-initial openers per 1k sentences (first sentence excluded)")
            O.put("h1.%s.final.reflopen.lo" % s_, rci[0], "dec1", src); O.put("h1.%s.final.reflopen.hi" % s_, rci[1], "dec1", src)
            O.put("h1.%s.final.reflopen_nowait" % s_, rnw, "dec1", src + " (openers other than 'wait')")
            O.put("h1.%s.final.reflopen_nowait.lo" % s_, nci[0], "dec1", src); O.put("h1.%s.final.reflopen_nowait.hi" % s_, nci[1], "dec1", src)
            O.put("h1.%s.final.reflopen_n" % s_, nop, "int", src + " (opener count)")
            for n_, v in sh.items():
                O.put("h1.%s.final.reflopen_share_%s" % (s_, n_), v, "dec1", src + " (share of all openers, %)")
        else:
            MISSING.append("reflopen S8 %s: gens not local" % m_)
    common.write_json(OUT / "reflopen_keys.json", {"entries": O.entries})

    # ------------------------------------------------------------------ judge-based RD_op trend, base-derived prefixes
    T = Ex()
    for suf, fn in (("", "insertion_pilot.json"), ("_oss", "insertion_pilot_oss.json")):
        f = RES / "main" / "S1" / fn
        if not f.exists():
            continue
        j = common.read_json(f)
        for src in ("base", "late"):
            e = j["outcomes"]["judge_revise"].get(src, {}).get("estimates", {})
            pts = [(s_, e["S1@%s:RD_next" % c]["est"]) for c, s_ in (("base", 0), ("100", 100), ("200", 200), ("400", 400), ("700", 700), ("1000", 1000), ("1400", 1400), ("1900", 1900)) if "S1@%s:RD_next" % c in e]
            if len(pts) >= 6:
                rho, p = st.spearman([a for a, _ in pts], [b for _, b in pts])
                T.put("h4.s1.trend.rho_rd_op_%s%s" % (src, suf), rho, "dec2", "exp/results/main/S1/%s::outcomes.judge_revise.%s.estimates S1@<ckpt>:RD_next, %d ckpts" % (fn, src, len(pts)))
                if p is not None:
                    T.put("h4.s1.trend.rho_rd_op_%s%s.p" % (src, suf), p, "dec3", "exact permutation")
                T.put("h4.s1.trend.rho_rd_op_%s%s.n" % (src, suf), len(pts), "int", "number of checkpoints")
    common.write_json(OUT / "rd_trend_keys.json", {"entries": T.entries})

    # ------------------------------------------------------------------ cross-tab: judge REVISE vs VALUE_CHANGE on the same windows
    X = Ex()
    stage_s1 = {"S1@100": "early", "S1@700": "mid", "S1@1900": "late", "S1@final": "final", "S1@base": "base"}

    def xtab(rows, B=2000, seed=1):
        pid = np.array([r["prefix_id"] for r in rows]); u, inv = np.unique(pid, return_inverse=True)
        rev = np.array([bool(r["judge_revise"]) for r in rows]); vc = np.array([bool(r["value_change"]) for r in rows])

        def stat(w):
            both = (w * (rev & vc)).sum(); r_ = (w * rev).sum(); v_ = (w * vc).sum()
            return (both / r_ if r_ else np.nan, both / v_ if v_ else np.nan)
        base = stat(np.ones(len(u))[inv])
        rng = np.random.default_rng(seed)
        bs = [stat(np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)[inv]) for _ in range(B)]
        ci = [st.percentile_ci([b[k] for b in bs if b[k] == b[k]]) for k in (0, 1)]
        return base, ci, int((rev & vc).sum()), int(rev.sum()), int(vc.sum())

    wf = RES / "main" / "S1" / "valchange" / "windows.jsonl.gz"
    if wf.exists():
        W = read_gz_jsonl(wf)
        for src in ("late", "base"):
            for tag, filt in (("", lambda r: True), ("_wait", lambda r: r["arm"] == "wait")):
                groups = {stg: [r for r in W if r["src"] == src and r["policy"] == pol and filt(r)] for pol, stg in stage_s1.items()}
                groups["trend"] = [r for r in W if r["src"] == src and filt(r)]
                for stg, rows in groups.items():
                    if len(rows) < 50:
                        continue
                    (pr, pv), ci, nb, nr, nv = xtab(rows)
                    sfx = "_%s%s" % (src, tag)
                    s_ = "h4.s1.%s." % stg
                    X.put(s_ + "p_vc_given_rev" + sfx, 100 * pr, "dec1", "exp/results/main/S1/valchange/windows.jsonl.gz")
                    X.put(s_ + "p_vc_given_rev" + sfx + ".lo", 100 * ci[0][0], "dec1", "cluster bootstrap over prefixes B=2000")
                    X.put(s_ + "p_vc_given_rev" + sfx + ".hi", 100 * ci[0][1], "dec1", "cluster bootstrap over prefixes B=2000")
                    X.put(s_ + "p_rev_given_vc" + sfx, 100 * pv, "dec1", "exp/results/main/S1/valchange/windows.jsonl.gz")
                    X.put(s_ + "p_rev_given_vc" + sfx + ".lo", 100 * ci[1][0], "dec1", "cluster bootstrap")
                    X.put(s_ + "p_rev_given_vc" + sfx + ".hi", 100 * ci[1][1], "dec1", "cluster bootstrap")
                    X.put(s_ + "n_both" + sfx, nb, "int", "windows with judge-REVISE and VALUE_CHANGE")
                    X.put(s_ + "n_rev" + sfx, nr, "int", "windows with judge-REVISE")
                    X.put(s_ + "n_vc" + sfx, nv, "int", "windows with VALUE_CHANGE")
    else:
        MISSING.append("S1 valchange windows not local")
    wf = RES / "scale" / "valchange" / "windows.jsonl.gz"
    if wf.exists():
        from exp.pipeline.scale_report import SID
        W = read_gz_jsonl(wf)
        for pol, s in SID.items():
            rows = [r for r in W if r["policy"] == pol]
            if len(rows) >= 50:
                (pr, pv), ci, nb, nr, nv = xtab(rows)
                X.put("h4.%s.final.p_vc_given_rev" % s, 100 * pr, "dec1", "exp/results/scale/valchange/windows.jsonl.gz")
                X.put("h4.%s.final.p_vc_given_rev.lo" % s, 100 * ci[0][0], "dec1", "cluster bootstrap"); X.put("h4.%s.final.p_vc_given_rev.hi" % s, 100 * ci[0][1], "dec1", "cluster bootstrap")
                X.put("h4.%s.final.p_rev_given_vc" % s, 100 * pv, "dec1", "same"); X.put("h4.%s.final.n_both" % s, nb, "int", "same")
                X.put("h4.%s.final.n_rev" % s, nr, "int", "same"); X.put("h4.%s.final.n_vc" % s, nv, "int", "same")
    common.write_json(OUT / "crosstab_keys.json", {"entries": X.entries})

    # ------------------------------------------------------------------ prior measures, late-half rho (S1)
    Bm = Ex()
    bmf = RES / "main" / "baseline_measures.json"
    if bmf.exists():
        bm = common.read_json(bmf)["series"]["S1"]["checkpoints"]
        order = [(c, s_) for c, s_ in (("base", 0), ("100", 100), ("200", 200), ("400", 400), ("700", 700), ("1000", 1000), ("1400", 1400), ("1900", 1900)) if c in bm or c == "base"]
        late = [(c, s_) for c, s_ in order if s_ >= 700]
        for q in ("r1kw", "r1wait", "orz", "orz_reflective", "simplerl"):
            pts = [(s_, est(bm[c][q])) for c, s_ in late if c in bm and q in bm[c]]
            if len(pts) >= 3:
                rho, _ = st.spearman([a for a, _ in pts], [b for _, b in pts])
                Bm.put("bm.s1.late_half.rho_%s" % q.replace("_reflective", "refl"), rho, "dec2", "exp/results/main/baseline_measures.json: Spearman(step, measure) over steps %s (K=%d, no p)" % ([a for a, _ in pts], len(pts)))
            else:
                MISSING.append("bm late-half %s: <3 checkpoints in baseline_measures.json" % q)
        h1 = common.read_json(RES / "main" / "S1" / "h1_indices.json")["checkpoints"]
        for q, nm in (("FREQ_wait_1k", "freq"), ("ORTH", "orth"), ("SI", "si"), ("H_para_rare", "hpara")):
            pts = [(s_, est(h1[c]["indices"][q])) for c, s_ in late if c in h1 and q in h1[c]["indices"]]
            if len(pts) >= 3:
                rho, _ = st.spearman([a for a, _ in pts], [b for _, b in pts])
                Bm.put("h1.s1.late_half.rho_%s" % nm, rho, "dec2", "exp/results/main/S1/h1_indices.json: Spearman over steps %s (K=%d, no p)" % ([a for a, _ in pts], len(pts)))
    common.write_json(OUT / "bm_late_half_keys.json", {"entries": Bm.entries})
    (OUT / "MISSING.md").write_text("# Not computable from local files (never imputed)\n" + "\n".join("- " + m for m in MISSING) + "\n")
    print({p.name: len(common.read_json(p)["entries"]) for p in sorted(OUT.glob("[a-z]*_keys.json"))}, "missing:", len(MISSING))


if __name__ == "__main__":
    main()
