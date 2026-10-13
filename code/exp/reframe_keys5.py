"""METHOD_CHECK fixes B1-B3 on local generations.  python3 exp/reframe_keys5.py -> exp/results/reframe/methodcheck_keys.json
B1 rates per response and per 1k WORDS next to per 1k tokens (h1.<s>.<stage>.freq_resp / freq_word).
B2 accuracy with vs without >=1 'wait' in the response (h1.<s>.<stage>.acc_wait / acc_nowait / n_resp_wait / n_resp_nowait; Wilson 95%).
B3 share of 'wait' occurrences located after the first candidate answer (first \\boxed{ or 'Answer:' line) (h1.<s>.<stage>.wait_after_ans; responses with an answer only)."""
from __future__ import annotations

import re

from reframe_keys import OUT, RES, Ex, Ref, common, ix, mk, read_gz_jsonl
from reframe_keys3 import wilson
from exp.pipeline import keys as K

ANS = re.compile(r"\\boxed\s*\{|^\s*\**answer\**\s*:", re.I | re.M)
S1C = ["base", "100", "200", "400", "700", "1000", "1400", "1900", "final"]


def sources():
    g = lambda ser, f: RES / "main" / ser / "gen" / f
    out = []
    for c in S1C:
        out.append(("s1", c, {"early": "100", "mid": "700", "late": "1900", "final": "final"}, g("S1", "S1@%s.graded.jsonl.gz" % c)))
    for c in ("base", "25", "100", "250", "500", "750", "1000", "1375"):
        out.append(("s2", c, {"early": "25", "mid": "500", "late": "1375"}, g("S2", "S2@%s.graded.jsonl.gz" % c)))
    for c in ("base", "100", "300", "700", "1200", "1800", "2300"):
        out.append(("s9", c, {"early": "100", "mid": "700", "late": "1800", "final": "2300"}, RES / "s9" / "S9" / "gen" / ("S9@%s.graded.jsonl.gz" % c)))
    for m, s in (("S8_base", "s8base"), ("S8_dapo", "s8dapo"), ("S8_orz", "s8orz"), ("S8_qwq", "s8qwq"), ("S8_r1distill", "s8r1d")):
        out.append((s, "final", {}, RES / "scale" / m / "gen" / ("%s_math500.graded.jsonl.gz" % m)))
    for c, st in (("base", "base"), ("1566", "late")):
        out.append(("s3a", c, {"late": "1566"}, g("S3", "S3@%s.graded.jsonl.gz" % c)))
    for br in ("stage1-step141000", "stage1-step707000", "stage1-step1413814", "stage2-step47684", "stage3-step6000"):
        out.append(("s1pre", br, {}, g("S1pre", "S1pre@%s.graded.jsonl.gz" % br)))
    for c in ("step5000", "step15000", "step25000", "step35000", "step43000"):
        out.append(("s2ctl", c, {"early": "step5000", "mid": "step25000", "late": "step43000"}, g("S2ctl", "S2ctl@%s.graded.jsonl.gz" % c)))
    return out


def stages(ckpt, smap):
    if ckpt == "base" or ckpt == "final":
        k = [ckpt]
    elif ckpt.isdigit():
        k = ["step%04d" % int(ckpt)]
    else:
        k = [ckpt]
    return k + [st for st, c in smap.items() if c == ckpt and st not in k]


def main():
    X = Ex()
    jobs = [(s, c, sm, p) for s, c, sm, p in sources() if p.exists()]
    for n in (1, 2, 7):
        ref = Ref(n); idx = ref.idx()
        for stg, step in (("base", min(idx)), ("final", max(idx))) if idx else ():
            jobs.append((ref.sid, stg, {}, ("REF", ref, step)))
    for s, c, sm, p in jobs:
        rows = read_gz_jsonl(p) if not isinstance(p, tuple) else p[1].gen(p[2])
        src = str(p.relative_to(RES.parent.parent)) if not isinstance(p, tuple) else "exp/rl/results/%s/results.tgz::gen/s%d" % (p[1].tag, p[2])
        nresp, ntok, nw, nwords = len(rows), 0, 0, 0
        accw, accn = [0, 0], [0, 0]   # [correct, n]
        after = [0, 0]                # [waits after answer, waits in responses with an answer]
        for r in rows:
            t = r["text"]
            cnt = ix.trace_counts(t, int(float(r["n_tokens"])))
            w = int(cnt.get("wait:n", 0))
            ntok += int(cnt["tokens"]); nw += w; nwords += len(t.split())
            ok = str(r["correct"]) == "True"
            g_ = accw if w >= 1 else accn
            g_[0] += ok; g_[1] += 1
            m = ANS.search(t)
            if m and w:
                pos = [x.start for x in mk.find_markers(t) if x.form == "wait"]
                after[0] += sum(1 for x in pos if x > m.start()); after[1] += len(pos)
        for stg in stages(c, sm):
            base = "h1.%s.%s." % (s, stg)
            X.put(base + "freq_resp", nw / nresp, "dec2", src + "::wait occurrences per response")
            X.put(base + "freq_word", 1000 * nw / max(nwords, 1), "dec3" if nw / max(nwords, 1) * 1000 < 1 else "dec2", src + "::wait per 1,000 whitespace-delimited words")
            for tag, g_ in (("acc_wait", accw), ("acc_nowait", accn)):
                if g_[1]:
                    p_, lo, hi = wilson(g_[0], g_[1])
                    X.put(base + tag, 100 * p_, "dec1", src + "::MATH-500 accuracy of responses " + ("with >=1 wait" if tag == "acc_wait" else "without wait") + " (Wilson 95%)")
                    X.put(base + tag + ".lo", 100 * lo, "dec1", "Wilson 95%"); X.put(base + tag + ".hi", 100 * hi, "dec1", "Wilson 95%")
            X.put(base + "n_resp_wait", accw[1], "int", src); X.put(base + "n_resp_nowait", accn[1], "int", src)
            if after[1] >= 20:
                p_, lo, hi = wilson(after[0], after[1])
                X.put(base + "wait_after_ans", 100 * p_, "dec1", src + "::share of wait occurrences after the first candidate answer (n=%d waits)" % after[1])
                X.put(base + "wait_after_ans.lo", 100 * lo, "dec1", "Wilson 95%"); X.put(base + "wait_after_ans.hi", 100 * hi, "dec1", "Wilson 95%")
                X.put(base + "wait_after_ans_n", after[1], "int", src)
    common.write_json(OUT / "methodcheck_keys.json", {"entries": X.entries})
    print(len(X.entries), "keys")


if __name__ == "__main__":
    main()
