"""Dataset panel keys: h1.ds_<dataset>.<series>.<stage>.* (S1 index set from the indices json) + ctl.ds_<dataset>.<series>.<stage>.{acc,trunc} + nwait/ntok/freq4 (+Poisson CI) + .fin.* finished-only.
Inputs exp/results/scale/ds/<dataset>/<model>/<dataset>_<model>.{graded.jsonl.gz,idx.json} (rsync from hostA runs/P1/scale/DS).  Model -> (series, stage): s1base=(s1, base), s1final=(s1, final), szq7base=(szq7, base), szq7=(szq7, final), s2base=(s2, base), s2final=(s2, final), r1d7=(r1d7, final).
    python3 exp/reframe_keys17.py -> exp/results/reframe/ds_keys.json"""
from __future__ import annotations

import json

from reframe_keys import OUT, ROOT, Ex, common, poisson_ci
from reframe_keys14 import fin_stats, rows_of
from pipeline import keys as K

DS = ROOT / "results" / "scale" / "ds"
MAP = {"s1base": ("s1", "base"), "s1final": ("s1", "final"), "szq7base": ("szq7", "base"), "szq7": ("szq7", "final"), "s2base": ("s2", "base"), "s2final": ("s2", "final"), "r1d7": ("r1d7", "final"), "szq32base": ("szq32", "base"), "szq32": ("szq32", "final"), "szm24base": ("szm24", "base"), "szm24": ("szm24", "final"), "gr34base": ("gr34", "base"), "gr34": ("gr34", "final")}
DATASETS = ("aime24", "aime25", "amc23", "gsm8k", "mmlupro")


def main():
    ex, X = K.Exporter(), Ex()
    got = []
    for dn in DATASETS:
        for mid, (ser, stage) in MAP.items():
            d = DS / dn / mid
            idxf, gf = d / ("%s_%s.idx.json" % (dn, mid)), d / ("%s_%s.graded.jsonl.gz" % (dn, mid))
            if not (idxf.exists() and gf.exists()):
                continue
            j = json.load(open(idxf)); s = "ds_%s.%s" % (dn, ser); src = "exp/results/scale/ds/%s/%s/%s" % (dn, mid, idxf.name)
            h1 = {"checkpoints": {stage: {"step": None, "acc": j["acc"], "truncated": j["truncated"], "n_samples": j["n_samples"], "indices": j["indices"]}}, "trends": {}}
            K.export_h1(ex, h1, s, src, {stage: stage})
            nw, tk = int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"]); lo, hi = poisson_ci(nw)
            b = "h1.%s.%s." % (s, stage)
            X.put(b + "nwait", nw, "int", src); X.put(b + "ntok", tk, "int", src); X.put(b + "nresp", j["n_samples"], "int", src)
            X.put(b + "freq4", 1000 * nw / tk, "dec4", src); X.put(b + "freq4.lo", 1000 * lo / tk, "dec4", src + "; exact Poisson 95%"); X.put(b + "freq4.hi", 1000 * hi / tk, "dec4", src + "; exact Poisson 95%")
            st = fin_stats(rows_of(open(gf, "rb")))
            if st["ntok"]:
                flo, fhi = poisson_ci(st["nwait"])
                X.put(b + "fin.nwait", st["nwait"], "int", gf.name); X.put(b + "fin.ntok", st["ntok"], "int", gf.name); X.put(b + "fin.n", st["nfin"], "int", gf.name)
                X.put(b + "fin.freq4", 1000 * st["nwait"] / st["ntok"], "dec4", gf.name); X.put(b + "fin.freq4.lo", 1000 * flo / st["ntok"], "dec4", gf.name); X.put(b + "fin.freq4.hi", 1000 * fhi / st["ntok"], "dec4", gf.name)
            got.append("%s/%s acc %.3f trunc %.3f wait %.2f/1k" % (dn, mid, j["acc"], j["truncated"], 1000 * nw / tk))
    common.write_json(OUT / "ds_keys.json", {"entries": ex.entries + X.entries})
    print("\n".join(got)); print(len(ex.entries) + len(X.entries), "keys,", len(got), "runs")


if __name__ == "__main__":
    main()
