"""Finished-only keys h1.bb_<id>.final.fin.{nwait,ntok,n,freq2,freq4,freq4.lo,freq4.hi} for the Qwen3.5 0.8B / 2B pairs from their EXISTING 8k-cap MATH-500 generations
(until 32k reruns replace them). -> exp/results/reframe/bb/qwen35_fin8k_keys.json"""
from reframe_keys import OUT, ROOT, Ex, common, poisson_ci
from reframe_keys14 import fin_stats, rows_of

IDS = ("qwen35_0p8b", "qwen35_0p8b_base", "qwen35_2b", "qwen35_2b_base")


def main():
    X = Ex()
    for mid in IDS:
        gf = ROOT / "results" / "scale" / "bb" / ("BB_" + mid) / "gen" / ("BB_%s_math500.graded.jsonl.gz" % mid)
        if list((ROOT / "results" / "scale" / "bb" / ("BB_" + mid) / "gen_cap32k").glob("*.idx.json")):
            print(mid, "32k run landed: fin8k keys superseded (reframe_keys16)"); continue
        if not gf.exists():
            print(mid, "missing"); continue
        st = fin_stats(rows_of(open(gf, "rb"))); lo, hi = poisson_ci(st["nwait"]); b = "h1.bb_%s.final.fin." % mid
        X.put(b + "nwait", st["nwait"], "int", gf.name); X.put(b + "ntok", st["ntok"], "int", gf.name); X.put(b + "n", st["nfin"], "int", gf.name + " (finished responses)")
        X.put(b + "freq2", 1000 * st["nwait"] / st["ntok"], "dec2", gf.name); X.put(b + "freq4", 1000 * st["nwait"] / st["ntok"], "dec4", gf.name)
        X.put(b + "freq4.lo", 1000 * lo / st["ntok"], "dec4", gf.name); X.put(b + "freq4.hi", 1000 * hi / st["ntok"], "dec4", gf.name)
        print(mid, "finished %d/%d, wait %.2f/1k finished-only" % (st["nfin"], st["nresp"], 1000 * st["nwait"] / st["ntok"]))
    common.write_json(OUT / "bb" / "qwen35_fin8k_keys.json", {"entries": X.entries})
    print(len(X.entries), "keys")


if __name__ == "__main__":
    main()
