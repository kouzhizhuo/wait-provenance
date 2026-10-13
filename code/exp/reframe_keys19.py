"""Finished-only keys h1.bb_<id>.final.fin.{nwait,ntok,n,freq2,freq4[.lo/.hi]} for the wave-6 RL-after-distillation / trace-SFT rows and their R1-Distill bases (MATH-500[:200] x 4, cap as run).
-> exp/results/reframe/bb/wave6_fin_keys.json"""
from reframe_keys import OUT, ROOT, Ex, common, poisson_ci
from reframe_keys14 import fin_stats, rows_of

IDS = ("r1d1p5b", "deepscaler", "r1d14", "light14", "r1d7", "openthinker7", "bespoke7")


def main():
    X = Ex()
    for mid in IDS:
        gf = next((ROOT / "results" / "scale" / "bb" / ("BB_" + mid) / "gen").glob("*_math500.graded.jsonl.gz"), None)
        if gf is None:
            print(mid, "missing"); continue
        st = fin_stats(rows_of(open(gf, "rb"))); lo, hi = poisson_ci(st["nwait"]); b = "h1.bb_%s.final.fin." % mid
        X.put(b + "nwait", st["nwait"], "int", gf.name); X.put(b + "ntok", st["ntok"], "int", gf.name); X.put(b + "n", st["nfin"], "int", gf.name + " (finished responses)")
        X.put(b + "freq2", 1000 * st["nwait"] / st["ntok"], "dec2", gf.name); X.put(b + "freq4", 1000 * st["nwait"] / st["ntok"], "dec4", gf.name)
        X.put(b + "freq4.lo", 1000 * lo / st["ntok"], "dec4", gf.name); X.put(b + "freq4.hi", 1000 * hi / st["ntok"], "dec4", gf.name)
        print(mid.ljust(13), "finished %d/%d, wait %.2f/1k finished-only [%.2f, %.2f]" % (st["nfin"], st["nresp"], 1000 * st["nwait"] / st["ntok"], 1000 * lo / st["ntok"], 1000 * hi / st["ntok"]))
    common.write_json(OUT / "bb" / "wave6_fin_keys.json", {"entries": X.entries})
    print(len(X.entries), "keys")


if __name__ == "__main__":
    main()
