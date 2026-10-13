"""D49 (+D49a: single seed q3r1) keys from exp/rl/results/q3r1/results.tgz (idx/s{0,100,..,400}.json; step 0 = Qwen3-4B-Base under the identical MATH-500[:200]x8, T1.0, cap 3072 evaluation).
Keys h1.bb_q3r1.<stage>.{freq,nwait,ntok,freq4[.lo/.hi]} and ctl.bb_q3r1.<stage>.{acc,trunc}; base = *_base.base.*; verdict keys h1.bb_q3r1.d49.*  -> exp/results/reframe/q3r1_keys.json"""
import json
import re
import tarfile

from reframe_keys import OUT, ROOT, Ex, common, poisson_ci

t = tarfile.open(ROOT / "rl" / "results" / "q3r1" / "results.tgz"); idx = {}
for m in t.getmembers():
    mm = re.fullmatch(r"(?:\./)?idx/s(\d+)\.json", m.name)
    if mm:
        idx[int(mm.group(1))] = json.load(t.extractfile(m))
X = Ex(); src = "exp/rl/results/q3r1/results.tgz::idx/s<step>.json"; R = {}
for s, j in sorted(idx.items()):
    n = int(j["totals"]["wait:n"]); tk = int(j["totals"]["tokens"]); lo, hi = poisson_ci(n); R[s] = (n, tk, lo, hi, j["acc"], j["truncated"])
    h, c = ("h1.bb_q3r1_base.base.", "ctl.bb_q3r1_base.base.") if s == 0 else ("h1.bb_q3r1.step%04d." % s, "ctl.bb_q3r1.step%04d." % s)
    X.put(h + "nwait", n, "int", src); X.put(h + "ntok", tk, "int", src)
    X.put(h + "freq4", 1000 * n / tk, "dec4", src); X.put(h + "freq4.lo", 1000 * lo / tk, "dec4", src + "; exact Poisson"); X.put(h + "freq4.hi", 1000 * hi / tk, "dec4", src + "; exact Poisson")
    X.put(h + "freq", j["indices"]["FREQ_wait_1k"]["est"], "dec4", src + "::indices.FREQ_wait_1k")
    X.put(c + "acc", j["acc"], "pct", src); X.put(c + "trunc", j["truncated"], "pct", src + " (finish=length share; cap 3,072)")
b, z = R[0], R[max(R)]; rate = lambda r: 1000 * r[0] / r[1]
valid = (z[4] - b[4] >= 0.10) and b[5] < 0.20 and z[5] < 0.20; base_free = rate(b) < 0.05
coins = rate(z) >= 0.05 and 1000 * z[2] / z[1] > 1000 * b[3] / b[1]
d = "h1.bb_q3r1.d49."
X.put(d + "dacc_pts", 100 * (z[4] - b[4]), "dec1", "step %d minus base" % max(R)); X.put(d + "valid", int(valid), "int", "acc +>=10 pts and trunc <20% at base and last step")
X.put(d + "base_marker_free", int(base_free), "int", "base < 0.05/1k"); X.put(d + "coins_wait", int(coins), "int", "step-400 >= 0.05/1k AND exact lower > base exact upper (D47 rule)")
common.write_json(OUT / "q3r1_keys.json", {"entries": X.entries})
print({s: (round(rate(r), 4), round(1000 * r[2] / r[1], 4), round(1000 * r[3] / r[1], 4), r[4], r[5]) for s, r in R.items()}, "valid", valid, "base_free", base_free, "coins", coins)
