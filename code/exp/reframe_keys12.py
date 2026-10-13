"""SimpleRL-Zoo / ORZ-7B / Oat-Zero rows (D47/D47a): PRIMARY decode = the authors' own v1 eval setting (hkust-nlp/simpleRL-reason branch v1: qwen-boxed, T=1.0, top_p 0.95, max_tokens 16000, n=1; see notes/SZ_EVAL_META.md),
ROBUSTNESS decodes beside it: _t0 (older main-branch eval: T=0, 3000 tokens), _t10 (our harness T=1.0, 200 x 4), _t06 (T=0.6 x4), _greedy.   python3 exp/reframe_keys12.py -> exp/results/reframe/bb/sz_keys.json
Keys: h1.bb_<id>[_base].final.{nwait,ntok,freq2,freq4[.lo/.hi],si,orth} and ctl.bb_<id>[_base].final.{acc,trunc} (primary); robustness as quantity suffixes _t10 / _t06 / _greedy."""
from __future__ import annotations

from reframe_keys import OUT, RES, Ex, common, poisson_ci

BB = RES / "scale" / "bb"
IDS = ["gr3_4_s1", "gr3_4_s2", "gr3_4_s3", "gr7_base", "gr7", "gr14_base", "gr14", "gr3_4_base", "gr3_4", "gr3_14_base", "gr3_14", "os15_base", "os15", "oat15_base", "oat15", "orz32_base", "orz32", "sz_q32_base", "sz_q32", "sz_m24_base", "sz_m24", "orz7b_base", "oat7_base", "sz_q05_base", "sz_q05", "sz_q15_base", "sz_q15", "sz_q7_base", "sz_q7", "orz7b", "sz_qm7_base", "sz_qm7", "oat7", "sz_q14_base", "sz_q14", "sz_m7_base", "sz_m7", "sz_ds7_base", "sz_ds7", "sz_l8"]
ALIAS = {"sz_ds7": "sz_dsm7", "sz_ds7_base": "sz_dsm7_base"}
SERIES = {"bb_" + ALIAS.get(i, i) for i in IDS}


def main():
    X = Ex(); rows = {}
    for mid in IDS:
        s = "bb_" + ALIAS.get(mid, mid)
        runs = {"": BB / ("BB_" + mid) / "gen_v1" / ("BB_%s_v1.idx.json" % mid), "_t0": BB / ("BB_" + mid) / "gen_exact" / ("BB_%s_exact.idx.json" % mid), "_t10": BB / ("BB_" + mid) / "idx" / "math500.json",
                "_t06": BB / ("BB_" + mid) / "gen_t06" / ("BB_%s_t06.idx.json" % mid), "_greedy": BB / ("BB_" + mid) / "gen_greedy" / ("BB_%s_greedy.idx.json" % mid)}
        for suf, f in runs.items():
            if not f.exists():
                continue
            j = common.read_json(f)
            nw, tk = int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"])
            lo, hi = poisson_ci(nw)
            b = "h1.%s.final." % s
            src = "exp/results/scale/bb/%s::%s" % (f.relative_to(BB), "primary: authors' own eval setting" if not suf else "robustness decode " + suf[1:])
            X.put(b + "nwait" + suf, nw, "int", src); X.put(b + "ntok" + suf, tk, "int", src)
            if not suf:
                X.put(b + "freq2", 1000 * nw / tk, "dec2", src)
            X.put(b + "freq4" + suf, 1000 * nw / tk, "dec4", src); X.put(b + "freq4" + suf + ".lo", 1000 * lo / tk, "dec4", "exact Poisson 95%"); X.put(b + "freq4" + suf + ".hi", 1000 * hi / tk, "dec4", "exact Poisson 95%")
            if not suf and nw >= 20:
                for q, nm in (("si", "SI"), ("orth", "ORTH")):
                    e = j["indices"].get(nm)
                    if isinstance(e, dict):
                        for s_, x in (("", e["est"]), (".lo", e["lo"]), (".hi", e["hi"])):
                            X.put(b + q + s_, 100 * x, "dec1", src + "::indices." + nm)
            X.put("ctl.%s.final.acc%s" % (s, suf), j["acc"], "pct", src); X.put("ctl.%s.final.trunc%s" % (s, suf), j["truncated"], "pct", src)
            rows.setdefault(mid, {})[suf or "primary"] = {"acc": j["acc"], "trunc": j["truncated"], "wait_per_1k": 1000 * nw / tk, "nwait": nw, "ntok": tk, "lo": 1000 * lo / tk, "hi": 1000 * hi / tk}
    common.write_json(OUT / "bb" / "sz_keys.json", {"entries": X.entries, "series": sorted(SERIES)})
    common.write_json(OUT / "bb" / "sz_rows.json", rows)
    for m, v in rows.items():
        p = v.get("primary")
        print(m.ljust(13), ("primary acc %.3f trunc %.3f wait %.4f [%.4f,%.4f] n=%d" % (p["acc"], p["trunc"], p["wait_per_1k"], p["lo"], p["hi"], p["nwait"])) if p else "primary not landed", "| other:", [k for k in v if k != "primary"])
    print(len(X.entries), "keys")


if __name__ == "__main__":
    main()
