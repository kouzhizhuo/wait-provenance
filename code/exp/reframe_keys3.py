"""Other markers (local generations only): per-1k-token rates with exact Poisson 95% CIs for hmm, hold on, alternatively, let me check/verify,
actually, "but wait" and the R1 keyword set WITHOUT "wait"; form-level reshaping indices (SI, canonical share, chain share) for hmm/alternatively where n >= 100.
    python3 exp/reframe_keys3.py     -> exp/results/reframe/forms_keys.json, forms_summary.json
Keys: h1.<s>.<stage>.form_<name>.freq4[.lo|.hi|.n] ; h1.<s>.<stage>.form_<name>.{si,orth,chain}[.lo|.hi] and .n (only when n >= 100).
"""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import beta

from reframe_keys import OUT, RES, Ex, Ref, common, mk, poisson_ci, read_gz_jsonl   # noqa: F401

FORMS = {
    "hmm": r"\bhm+\b", "holdon": r"\bhold on\b", "alternatively": r"\balternatively\b",
    "letmecheck": r"\blet me (?:just )?(?:check|verify|double[- ]?check|re-?check|re-?verify)\b", "actually": r"\bactually\b",
    "butwait": r"\bbut wait\b", "r1nowait": r"\b(?:mistake|however|but|retry|error|verify|wrong|evaluate|check)\b",
}
RX = {k: re.compile(v, re.I) for k, v in FORMS.items()}
REF_FORMS = ("hmm", "alternatively")   # form-level reshaping indices


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def analyse(rows):
    ntok = sum(int(float(r["n_tokens"])) for r in rows)
    cnt = {k: 0 for k in FORMS}
    form_stats = {k: {"n": 0, "si": 0, "orth": 0, "chain": 0} for k in REF_FORMS}
    for r in rows:
        t = r["text"]
        for k, rx in RX.items():
            ms = list(rx.finditer(t))
            cnt[k] += len(ms)
            if k in REF_FORMS and ms:
                starts = [s.start for s in mk.split_sentences(t)]
                import bisect
                for j, m in enumerate(ms):
                    form_stats[k]["n"] += 1
                    si = bisect.bisect_right(starts, m.start()) - 1
                    lead = t[starts[si]:m.start()] if si >= 0 else "x"
                    is_si = lead.strip(" \n\"'*_>#-") == ""
                    form_stats[k]["si"] += is_si
                    form_stats[k]["orth"] += bool(is_si and m.group(0)[0].isupper() and t[m.end():m.end() + 1] == ",")
                    if j + 1 < len(ms) and ms[j + 1].start() - m.end() <= 12:
                        form_stats[k]["chain"] += 1
    return ntok, cnt, form_stats


def sources():
    """(series id, stage, loader)"""
    out = []
    g = lambda ser, pat: (RES / "main" / ser / "gen" / pat)
    out += [("s1", "base", g("S1", "<EMAIL>")), ("s1", "final", g("S1", "<EMAIL>"))]
    for br in ("stage1-step141000", "stage1-step707000", "stage1-step1413814", "stage2-step47684", "stage3-step6000"):
        out.append(("s1pre", br, g("S1pre", "S1pre@%s.graded.jsonl.gz" % br)))
    out += [("s2", "base", g("S2", "<EMAIL>")), ("s2", "late", g("S2", "<EMAIL>")),
            ("s2ctl", "early", g("S2ctl", "<EMAIL>")), ("s2ctl", "late", g("S2ctl", "<EMAIL>")),
            ("s3a", "base", g("S3", "<EMAIL>")), ("s3a", "late", g("S3", "<EMAIL>")),
            ("s9", "base", RES / "s9" / "S9" / "gen" / "<EMAIL>"), ("s9", "final", RES / "s9" / "S9" / "gen" / "<EMAIL>")]
    return out


def main():
    X = Ex()
    summ = {}
    jobs = [(s, st, lambda p=p: read_gz_jsonl(p), str(p.relative_to(RES.parent.parent))) for s, st, p in sources() if p.exists()]
    for n in (1, 2, 7):
        ref = Ref(n)
        idx = ref.idx()
        for st, step in (("base", min(idx)), ("final", max(idx))) if idx else ():
            jobs.append((ref.sid, st, lambda ref=ref, step=step: ref.gen(step), "exp/rl/results/%s/results.tgz::gen/s%d.graded.jsonl.gz" % (ref.tag, step)))
    for s, st, load, src in jobs:
        rows = load()
        if not rows:
            continue
        ntok, cnt, fs = analyse(rows)
        summ["%s.%s" % (s, st)] = {"ntok": ntok, "counts": cnt}
        for k, c in cnt.items():
            lo, hi = poisson_ci(c)
            key = "h1.%s.%s.form_%s" % (s, st, k)
            X.put(key + ".freq4", 1000 * c / ntok, "dec4" if 1000 * c / ntok < 1 else "dec2", src + " (rate per 1k response tokens; regex count)")
            X.put(key + ".freq4.lo", 1000 * lo / ntok, "dec4" if 1000 * hi / ntok < 1 else "dec2", "exact Poisson 95%")
            X.put(key + ".freq4.hi", 1000 * hi / ntok, "dec4" if 1000 * hi / ntok < 1 else "dec2", "exact Poisson 95%")
            X.put(key + ".n", c, "int", src)
        for k in REF_FORMS:
            f = fs[k]
            if f["n"] >= 100:
                for q in ("si", "orth", "chain"):
                    p, lo, hi = wilson(f[q], f["n"])
                    key = "h1.%s.%s.form_%s.%s" % (s, st, k, q)
                    X.put(key, 100 * p, "dec1", src + " (share of occurrences; Wilson 95%)")
                    X.put(key + ".lo", 100 * lo, "dec1", "Wilson 95%"); X.put(key + ".hi", 100 * hi, "dec1", "Wilson 95%")
                    X.put("h1.%s.%s.form_%s.%s_n" % (s, st, k, q), f["n"], "int", src)
            else:
                summ.setdefault("form_level_skipped_n_lt_100", []).append("%s.%s.%s n=%d" % (s, st, k, f["n"]))
    common.write_json(OUT / "forms_keys.json", {"entries": X.entries})
    common.write_json(OUT / "forms_summary.json", summ)
    # zero-RL question: any form absent at base that appears at the last checkpoint?
    for s in ("s3a", "s7r1", "s7r2"):
        b, l = summ.get("%s.base" % s), summ.get("%s.final" % s) or summ.get("%s.late" % s)
        if b and l:
            print(s, {k: (b["counts"][k], l["counts"][k], round(1000 * b["counts"][k] / b["ntok"], 3), round(1000 * l["counts"][k] / l["ntok"], 3)) for k in FORMS})
    print(len(X.entries), "keys")


if __name__ == "__main__":
    main()
