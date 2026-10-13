"""D46 length-control arm keys from exp/rl/results/seed_<ARM>/lenwait.json (per step: 'all' / 'finished' blocks with tokens, wait_words, mean_tokens, acc; truncation_share).  Arms: W512LY, X512LY (when landed).
Keys: h1.seed.<ARM>.<stage>.{nwait,ntok,freq4[.lo/.hi],trunc,toks,acc} and .fin.{nwait,ntok,freq4[.lo/.hi],toks,acc}, stage = sft (step 0) | stepNNNN;  h1.seed.<ARM>.trend.{a_wait,a_wait_fin}{,.lo,.hi} = FREQ(last step)/FREQ(step 0)."""
import json

from reframe_keys import OUT, ROOT, Ex, common, poisson_ci
from reframe_keys13 import ratio_ci


def main():
    X = Ex(); summ = {}
    for d in sorted((ROOT / "rl" / "results").glob("seed_*LY")):
        arm = d.name.replace("seed_", ""); f = d / "lenwait.json"
        if not f.exists():
            continue
        L = json.load(open(f)); steps = sorted(int(k) for k in L if k.isdigit()); last = max(steps); src = "exp/rl/results/%s/lenwait.json" % d.name
        for s in steps:
            stg = "sft" if s == 0 else "step%04d" % s; b = "h1.seed.%s.%s." % (arm, stg); r = L[str(s)]
            for blk, pre in (("all", ""), ("finished", "fin.")):
                x = r[blk]; lo, hi = poisson_ci(x["wait_words"])
                X.put(b + pre + "nwait", x["wait_words"], "int", src); X.put(b + pre + "ntok", x["tokens"], "int", src)
                X.put(b + pre + "freq4", 1000 * x["wait_words"] / x["tokens"], "dec4", src); X.put(b + pre + "freq4.lo", 1000 * lo / x["tokens"], "dec4", src + "; exact Poisson"); X.put(b + pre + "freq4.hi", 1000 * hi / x["tokens"], "dec4", src + "; exact Poisson")
                X.put(b + pre + "toks", x["mean_tokens"], "int", src); X.put(b + pre + "acc", x["acc"], "pct", src)
            X.put(b + "trunc", r["truncation_share"], "pct", src)
        for blk, nm in (("all", "a_wait"), ("finished", "a_wait_fin")):
            a, z = L["0"][blk], L[str(last)][blk]
            est, lo, hi = ratio_ci(a["wait_words"], a["tokens"], z["wait_words"], z["tokens"])
            for q, v in (("", est), (".lo", lo), (".hi", hi)):
                X.put("h1.seed.%s.trend.%s%s" % (arm, nm, q), v, "dec4", "FREQ(step %d)/FREQ(step 0), %s responses, conditional-binomial CI on counts" % (last, blk))
            summ[arm + "." + nm] = (est, lo, hi)
        X.put("h1.seed.%s.trend.last_step" % arm, last, "int", src)
    common.write_json(OUT / "seed_ly_keys.json", {"entries": X.entries}); print(summ, len(X.entries), "keys")


if __name__ == "__main__":
    main()
