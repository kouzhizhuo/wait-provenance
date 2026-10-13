"""S6 token matching on DEV (D50c(3)): set the random-suppression scale alpha and the budget-forcing caps so that the DEV mean tokens per response of each comparator is within +-3% of CW's.
Inputs (in the DEV dir): dev_gen.jsonl (vanilla, 1 sample/problem), dev_cwgen.jsonl + .cwstats.json (CW at tau), dev_rand_<alpha>.jsonl (random at alpha x CW's per-bin rates).
  python -m exp.cw.dev_match --dir <devdir> [--pbins]       (--pbins prints the per-bin CW suppression rates used to build the random runs; without it, writes match.json)"""
import argparse
import json
import os

import numpy as np

EDGES = [0, 500, 1000, 2000, 3000, 4000, 6000, 8000, 12000, 16000]


def mean_tokens(p):
    r = [json.loads(l) for l in open(p)]
    return float(np.mean([x["n_tokens"] for x in r])), [x["n_tokens"] for x in r]


def cw_pbins(stats_path):
    d = json.load(open(stats_path)); rows = list(d["done"].values()) + list(d["live"].values())
    b = np.zeros(len(EDGES)); s = np.zeros(len(EDGES))
    for r in rows:
        b += np.array(r["bins_b"]); s += np.array(r["bins_s"])
    return [float(s[i] / b[i]) if b[i] > 0 else 0.0 for i in range(len(EDGES))], int(b.sum()), int(s.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--pbins", action="store_true")
    a = ap.parse_args(); D = a.dir
    if a.pbins:
        p, nb, ns = cw_pbins(D + "/dev_cwgen.jsonl.cwstats.json"); print(json.dumps({"p_bins": p, "n_boundary": nb, "n_suppressed": ns})); return
    m_van, L = mean_tokens(D + "/dev_gen.jsonl"); m_cw, _ = mean_tokens(D + "/dev_cwgen.jsonl")
    p, nb, ns = cw_pbins(D + "/dev_cwgen.jsonl.cwstats.json")
    alphas, means = [], []
    for fn in sorted(os.listdir(D)):
        if fn.startswith("dev_rand_") and fn.endswith(".jsonl"):
            alphas.append(float(fn[len("dev_rand_"):-len(".jsonl")])); means.append(mean_tokens(D + "/" + fn)[0])
    o = np.argsort(alphas); alphas = [alphas[i] for i in o]; means = [means[i] for i in o]
    best = min(range(len(alphas)), key=lambda i: abs(means[i] - m_cw)) if alphas else None
    # BF: tokens(B) = mean(min(L, B)) + 45 per truncated response (forced suffix + <= 80 answer tokens, ~45 on average)
    Lv = np.array(L, dtype=float)
    def tok(B): return float(np.mean(np.minimum(Lv, B) + 45.0 * (Lv > B)))
    lo, hi = 64, int(Lv.max()) + 1
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tok(mid) < m_cw else (lo, mid)
    B_match = int(round(hi))
    out = {"m_vanilla": m_van, "m_cw": m_cw, "cw_ratio": m_cw / m_van, "n_boundary_cw": nb, "p_bins_cw": p, "bin_edges": EDGES, "rand_alphas": alphas, "rand_means": means,
           "alpha_star": alphas[best] if best is not None else None, "rand_dev_ratio_to_cw": means[best] / m_cw if best is not None else None, "bf_cap_match": B_match, "bf_cap_loose": int(round(B_match * 1.5)), "bf_dev_tokens_match": tok(B_match)}
    json.dump(out, open(D + "/match.json", "w"), indent=1); print(json.dumps(out)[:900])


if __name__ == "__main__":
    main()
