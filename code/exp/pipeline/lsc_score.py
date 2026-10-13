"""LSC metrics (EXPERIMENT_PLAN §2.2) from the occurrence embeddings of lsc_embed.py (fixed encoder = the series' base model).

    python -m exp.pipeline.lsc_score --dir exp/results/main/S1/lsc --series S1 --keys_series s1 --ckpts base:0,100:100,...,final: --stage_map early=100,mid=700,late=1900,final=final --out exp/results/main/S1/lsc.json
APD(t0,t) = mean pairwise cosine distance between occurrences of t0 and t; PRT = 1 - cos(prototype_t0, prototype_t) (prototype = mean vector);
JSD = Jensen-Shannon divergence (bits) between the usage-cluster distributions (k-means on the embeddings pooled over ALL checkpoints, k by silhouette in 2..8).
Bootstrap (B=1000, seed 20261012) resamples occurrences within each checkpoint; cluster centroids are fixed. Layering: the cluster whose share grows most from t0 to the last
checkpoint, with its share of lexical ("wait for ...") occurrences. Keys h2.<s>.<stage>.{apd,prt,jsd}_base[.lo|.hi] (suffix _base = encoder is the base model) and h2.<s>.trend.rho_<q>_base.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import common, keys as K, stats as st


def unit(x):
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def apd(a, b):
    return float(1.0 - (a @ b.T).mean())


def prt(a, b):
    ma, mb = a.mean(0), b.mean(0)
    return float(1.0 - ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-12))


def jsd(p, q):
    m = 0.5 * (p + q)

    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist(labels, k):
    return np.bincount(labels, minlength=k) / max(len(labels), 1)


def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--series", required=True)
    ap.add_argument("--keys_series", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--stage_map", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args()
    ck = [c.split(":") for c in a.ckpts.split(",")]
    steps = {c: (int(s) if s != "" else None) for c, s in ck}
    sm = dict(x.split("=") for x in a.stage_map.split(",")) if a.stage_map else {}
    E, LEX = {}, {}
    for c, _ in ck:
        f = Path(a.dir) / ("emb_%s.npz" % c)
        if f.exists():
            z = np.load(f)
            E[c] = unit(z["emb"]); LEX[c] = z["lexical"].astype(bool)
    base = ck[0][0]
    pooled = np.concatenate([E[c] for c in E])
    best = (-2, None, None)
    rng = np.random.default_rng(20261012)
    sub = pooled[rng.choice(len(pooled), min(len(pooled), 4000), replace=False)]
    for k in range(2, 9):
        km = KMeans(k, n_init=5, random_state=0).fit(pooled)
        s = silhouette_score(sub, km.predict(sub))
        if s > best[0]:
            best = (s, k, km)
    _, k, km = best
    lab = {c: km.predict(E[c]) for c in E}
    ex = K.Exporter()
    res = {"series": a.series, "k": int(k), "silhouette": float(best[0]), "n_occ": {c: int(len(E[c])) for c in E}, "checkpoints": {}}

    def stat_all(t0, t, i0=None, i1=None):
        a0, a1 = E[t0][i0], E[t][i1]
        l0, l1 = lab[t0][i0], lab[t][i1]
        return {"apd": apd(a0, a1), "prt": prt(a0, a1), "jsd": jsd(dist(l0, k), dist(l1, k))}
    pts = {q: [] for q in ("apd", "prt", "jsd")}
    for c, _ in ck[1:]:
        if c not in E:
            continue
        full = stat_all(base, c, slice(None), slice(None))
        reps = {q: [] for q in full}
        for _ in range(a.B):
            i0 = rng.integers(0, len(E[base]), len(E[base])); i1 = rng.integers(0, len(E[c]), len(E[c]))
            s = stat_all(base, c, i0, i1)
            for q in reps:
                reps[q].append(s[q])
        res["checkpoints"][c] = {q: {"est": full[q], "lo": st.percentile_ci(reps[q])[0], "hi": st.percentile_ci(reps[q])[1]} for q in full}
        res["checkpoints"][c]["cluster_share"] = [float(x) for x in dist(lab[c], k)]
        for q in full:
            e = res["checkpoints"][c][q]
            kind = "dec3"
            for stage in K.stage_keys(steps[c], c, sm):
                ex.put_est("h2.%s.%s.%s_base" % (a.keys_series, stage, q), e, kind, "%s::checkpoints.%s.%s (encoder = series base model; B=%d)" % (a.out, c, q, a.B))
            if steps[c] is not None:
                pts[q].append((steps[c], e["est"]))
    res["checkpoints"][base] = {"cluster_share": [float(x) for x in dist(lab[base], k)]}
    # noise floor: APD / PRT / JSD between two random halves of the base occurrences (mean of 50 splits)
    fl = {"apd": [], "prt": [], "jsd": []}
    for _ in range(50):
        perm = rng.permutation(len(E[base])); h0, h1 = perm[: len(perm) // 2], perm[len(perm) // 2:]
        s_ = stat_all(base, base, h0, h1)
        for q in fl:
            fl[q].append(s_[q])
    res["noise_floor_base_split_half"] = {q: float(np.mean(v)) for q, v in fl.items()}
    for q, v in fl.items():
        ex.put("h2.%s.base.%s_floor" % (a.keys_series, q), float(np.mean(v)), "dec3", a.out + ": split-half of the base occurrences (mean of 50 random splits)")
    for q, v in pts.items():
        if len(v) >= 3:
            rho, p = st.spearman([x for x, _ in v], [y for _, y in v])
            ex.put("h2.%s.trend.rho_%s_base" % (a.keys_series, q), rho, "dec2", "%s: Spearman(step, %s vs base) over %d checkpoints" % (a.out, q, len(v)))
            res.setdefault("trend", {})[q] = {"rho": rho, "p": p, "n": len(v)}
    last = ck[-1][0] if ck[-1][0] in E else None
    if last:
        g = dist(lab[last], k) - dist(lab[base], k)
        j = int(np.argmax(g))
        lex = lambda c, j_: float(LEX[c][lab[c] == j_].mean()) if (lab[c] == j_).any() else float("nan")
        res["layering"] = {"cluster_growing_most": j, "share_base": float(dist(lab[base], k)[j]), "share_last": float(dist(lab[last], k)[j]),
                           "lexical_share_in_cluster_last": lex(last, j), "lexical_share_all_last": float(LEX[last].mean())}
        ex.put("h2.%s.trend.layer_share_last" % a.keys_series, 100 * res["layering"]["share_last"], "dec1", a.out + ": share of occurrences in the fastest-growing cluster at the last checkpoint")
        ex.put("h2.%s.trend.layer_share_base" % a.keys_series, 100 * res["layering"]["share_base"], "dec1", a.out + ": same at the base")
        ex.put("h2.%s.trend.layer_lex" % a.keys_series, 100 * res["layering"]["lexical_share_in_cluster_last"], "dec1", a.out + ": lexical ('wait for') share inside that cluster")
    common.write_json(a.out, res)
    common.write_json(str(a.out).replace(".json", "_keys.json"), {"pilot": False, "entries": ex.entries})
    print("[lsc_score] %s k=%d silhouette=%.3f, %d keys" % (a.series, k, best[0], len(ex.entries)))


if __name__ == "__main__":
    main()
