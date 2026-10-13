"""bb extras: freq2/freq4 (+ exact Poisson .lo/.hi) for every non-popular backbone endpoint, and the opener scope keys h1.bb_<id>.final.open_<form> (per 1k sentences).
    python3 exp/reframe_keys9.py -> exp/results/reframe/bb/bb_extra_keys.json"""
from __future__ import annotations

import re

from reframe_keys import OUT, RES, Ex, common, mk, poisson_ci, read_gz_jsonl
from pipeline.bb_report import FAM, POP, series_stage   # noqa: E402

from reframe_keys8 import SZ_SERIES   # noqa: E402

BB = RES / "scale" / "bb"
INVALID = {"mistral_base", "qwen_math15", "r1d1p5", "r1d_llama8_old", "qwen35_0p8b", "qwen35_0p8b_base", "qwen35_2b", "qwen35_2b_base", "qwen35_4b", "qwen35_9b", "qwen35_4b_base", "qwen35_9b_base"}   # last eight: 32k-cap rerun, keys in qwen35_c32k_keys.json (reframe_keys16.py)
FORMS = [("wait", r"wait\b"), ("hmm", r"hm+\b"), ("but", r"but\b"), ("so", r"so\b"), ("perhaps", r"perhaps\b"), ("alternatively", r"alternatively\b"),
         ("let_me_check", r"let me (?:just )?(?:check|verify|double[- ]?check|re-?check|re-?verify)\b")]
PATS = [(n, re.compile(r"^[\s\"'*_>#-]*" + p, re.I)) for n, p in FORMS]
OPEN_IDS = ("orn7", "r1d7", "mimo_base", "smollm3_base", "qwen7b")


def main():
    X = Ex()
    for mids in FAM.values():
        for mid in mids:
            if mid in INVALID or mid in POP or mid.startswith("S8_"):
                continue
            d = BB / ("BB_" + mid)
            f = d / "idx" / "math500.json"
            if not f.exists():
                continue
            s, stage = series_stage(mid)
            j = common.read_json(f)
            nw, tk = int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"])
            lo, hi = poisson_ci(nw)
            b = "h1.%s.%s." % (s, stage)
            sfx = "_t10" if s in SZ_SERIES else ""   # old decode (reconstructed T=1.0 setting) of the SimpleRL-Zoo/ORZ/Oat rows: only under _t10; unsuffixed = authors' own eval (sz_keys.json)
            src = "exp/results/scale/bb/BB_%s/idx/math500.json::totals (wait:n=%d, tokens=%d)" % (mid, nw, tk)
            X.put(b + "freq2" + sfx, 1000 * nw / tk, "dec2", src)
            X.put(b + "freq4" + sfx, 1000 * nw / tk, "dec4", src)
            X.put(b + "freq4" + sfx + ".lo", 1000 * lo / tk, "dec4", src + "; exact Poisson 95%")
            X.put(b + "freq4" + sfx + ".hi", 1000 * hi / tk, "dec4", src + "; exact Poisson 95%")
    for mid in OPEN_IDS:
        p = next(iter((BB / ("BB_" + mid) / "gen").glob("*_math500.graded.jsonl.gz")))
        rows = read_gz_jsonl(p)
        ns = 0; c = {n: 0 for n, _ in FORMS}
        for r in rows:
            ss = mk.split_sentences(r["text"])[1:]
            ns += len(ss)
            for s_ in ss:
                for n, pt in PATS:
                    if pt.match(s_.text):
                        c[n] += 1   # forms are checked independently ("so"/"but" never overlap with the others at the sentence start)
        s, stage = series_stage(mid)
        for n, _ in FORMS:
            X.put("h1.%s.%s.open_%s" % (s, stage, n), 1000 * c[n] / ns, "dec1", "exp/results/scale/bb/BB_%s/gen (sentence-initial form per 1k sentences, first sentence excluded)" % mid)
        X.put("h1.%s.%s.open_nsent" % (s, stage), ns, "int", "sentences counted (first sentence of each response excluded)")
    szk = {e["key"] for e in common.read_json(OUT / "bb" / "sz_keys.json")["entries"]} if (OUT / "bb" / "sz_keys.json").exists() else set()
    X.entries = [e for e in X.entries if e["key"] not in szk]   # the _t10 copy already lives in sz_keys.json
    common.write_json(OUT / "bb" / "bb_extra_keys.json", {"entries": X.entries})
    dupe = {e["key"] for e in X.entries} & {e["key"] for f in list(OUT.glob("[a-z]*_keys*.json")) + list((OUT / "bb").glob("bb_keys*.json")) for e in common.read_json(f)["entries"]}
    print(len(X.entries), "keys; collisions with other files:", sorted(dupe)[:5])
    e = {x["key"]: x["value"] for x in X.entries}
    print({k: e[k] for k in ("h1.bb_orn7.final.freq2", "h1.bb_orn7.final.freq4", "h1.bb_orn7.final.freq4.lo", "h1.bb_orn7.final.freq4.hi", "h1.bb_orn7.final.open_but", "h1.bb_orn7.final.open_perhaps", "h1.bb_orn7.final.open_wait")})


if __name__ == "__main__":
    main()
