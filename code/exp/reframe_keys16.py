"""Qwen3.5-4B / 9B (post-trained) re-evaluated at max_tokens 32,768 (MATH-500[:200] x 4, T 0.6, top-p 0.95, qwen_boxed): REPLACES the primary h1/ctl keys of bb_qwen35_{4b,9b} (the 8k-cap run is no longer used).
    python3 exp/reframe_keys16.py -> exp/results/reframe/bb/qwen35_c32k_keys.json  (h1.bb_qwen35_<4b|9b>.final.* S1 index set, freq2/freq4 (+Poisson CI), .fin.* finished-only, ctl.*.final.{acc,trunc})
reframe_keys8.py drops the old h1./ctl. keys of these two series from bb_keys/bb_extra/bb_fin so that no stale 8k-cap value survives."""
from __future__ import annotations

import json

from reframe_keys import OUT, ROOT, Ex, common, poisson_ci
from reframe_keys14 import fin_stats, rows_of
from pipeline import keys as K

BB = ROOT / "results" / "scale" / "bb"
IDS = ("pangu_e7v11", "pangu_e7m", "qwen35_0p8b", "qwen35_0p8b_base", "qwen35_2b", "qwen35_2b_base", "qwen35_4b", "qwen35_9b", "qwen35_4b_base", "qwen35_9b_base")   # bases: added when their 32k reruns have landed


def main():
    ex, X = K.Exporter(), Ex()
    done = []
    for mid in IDS:
        d = BB / ("BB_" + mid) / "gen_cap32k"
        if not (d.exists() and list(d.glob("*.idx.json"))):
            print(mid, "32k run not landed yet; old keys kept")
            continue
        done.append(mid)
        idxf = next(d.glob("*.idx.json")); gf = next(d.glob("*.graded.jsonl.gz"))
        j = json.load(open(idxf)); s = "bb_" + mid
        h1 = {"checkpoints": {"final": {"step": None, "acc": j["acc"], "truncated": j["truncated"], "n_samples": j["n_samples"], "indices": j["indices"]}}, "trends": {}}
        src = "exp/results/scale/bb/BB_%s/gen_cap32k/%s" % (mid, idxf.name)
        K.export_h1(ex, h1, s, src, {"final": "final"})
        nw, tk = int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"]); lo, hi = poisson_ci(nw)
        b = "h1.%s.final." % s
        X.put(b + "nwait", nw, "int", src); X.put(b + "ntok", tk, "int", src)
        X.put(b + "freq2", 1000 * nw / tk, "dec2", src); X.put(b + "freq4", 1000 * nw / tk, "dec4", src)
        X.put(b + "freq4.lo", 1000 * lo / tk, "dec4", src + "; exact Poisson 95%"); X.put(b + "freq4.hi", 1000 * hi / tk, "dec4", src + "; exact Poisson 95%")
        st = fin_stats(rows_of(open(gf, "rb"))); flo, fhi = poisson_ci(st["nwait"])
        X.put(b + "fin.nwait", st["nwait"], "int", gf.name); X.put(b + "fin.ntok", st["ntok"], "int", gf.name); X.put(b + "fin.n", st["nfin"], "int", gf.name + " (finished responses)")
        X.put(b + "fin.freq2", 1000 * st["nwait"] / st["ntok"], "dec2", gf.name); X.put(b + "fin.freq4", 1000 * st["nwait"] / st["ntok"], "dec4", gf.name)
        X.put(b + "fin.freq4.lo", 1000 * flo / st["ntok"], "dec4", gf.name); X.put(b + "fin.freq4.hi", 1000 * fhi / st["ntok"], "dec4", gf.name)
        print(mid, "acc %.3f trunc %.3f freq %.2f | finished %d/%d freq_fin %.2f" % (j["acc"], j["truncated"], 1000 * nw / tk, st["nfin"], st["nresp"], 1000 * st["nwait"] / st["ntok"]))
    for fn in ("bb_extra_keys.json", "bb_fin_keys.json"):   # drop the stale 8k-cap copies (idempotent)
        p = OUT / "bb" / fn; dd = common.read_json(p)
        dd["entries"] = [e for e in dd["entries"] if not e["key"].startswith(tuple("h1.bb_%s." % m for m in done))]
        common.write_json(p, dd)
    common.write_json(OUT / "bb" / "qwen35_c32k_keys.json", {"entries": ex.entries + X.entries})
    print(len(ex.entries) + len(X.entries), "keys")


if __name__ == "__main__":
    main()
