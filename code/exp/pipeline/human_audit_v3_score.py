"""Score human audit v3 by the rules PRE-DECLARED in exp/gold/HUMAN_AUDIT_V3_meta.json (labels used as given; 'unsure' = No in the primary analysis, excluded in the sensitivity).
    python -m exp.pipeline.human_audit_v3_score --labels exp/gold/HUMAN_AUDIT_V3_LABELS.jsonl --out exp/gold/human_audit_v3_results.json
Part A: human-vs-detector binary kappa and Gwet AC1 (item bootstrap, B=5000) for the rule detector, Qwen2.5-7B-Instruct and gpt-oss-20b; human-labelled revision rate at base and
step 1900 weighted by the inverse sampling fractions (overall, marker-initial, other); decision rule A (i)/(ii)/(iii). Part B: yes-share per group with exact binomial CIs; decision rule B."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from . import common, stats as st

ROOT = Path(__file__).resolve().parents[2]


def kappa_ac1(h, d):
    h, d = np.asarray(h, int), np.asarray(d, int)
    a = ((h == 1) & (d == 1)).sum(); b = ((h == 0) & (d == 1)).sum(); c = ((h == 1) & (d == 0)).sum(); e = ((h == 0) & (d == 0)).sum(); n = a + b + c + e
    po = (a + e) / n
    pe = ((a + b) * (a + c) + (c + e) * (b + e)) / n ** 2
    kap = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    pi = ((a + b) + (a + c)) / (2 * n)
    pe2 = 2 * pi * (1 - pi)
    return float(kap), float((po - pe2) / (1 - pe2)), float(po)


def cp(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def rate(items, h):
    w = np.array([it["weight"] for it in items]); y = np.array([h[it["id"]] for it in items], float)
    return float((w * y).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    gold = ROOT / "exp" / "gold"
    if "LLM_GPT6" in str(a.labels) or "GPT6" in str(a.labels):
        raise SystemExit("refusing to score LLM (GPT-6) labels as human labels")
    key = common.read_json(gold / "HUMAN_AUDIT_V3_key.json")["items"]
    lab = {r["item_id"]: r["label"].strip().lower() for r in common.read_jsonl(a.labels)}
    judge = {r["item_id"]: r for r in common.read_jsonl(gold / "judge_v3.jsonl")}
    res = {"labels_file": str(a.labels), "n_labelled": len(lab), "label_counts": {k: sum(v == k for v in lab.values()) for k in ("yes", "no", "unsure")}}
    A = [dict(id=i, **k) for i, k in key.items() if k["part"] == "A" and i in lab]
    rng = np.random.default_rng(20261003)
    out_A = {"n": len(A)}
    for sens in ("unsure_as_no", "unsure_excluded"):
        items = [it for it in A if sens == "unsure_as_no" or lab[it["id"]] != "unsure"]
        h = {it["id"]: int(lab[it["id"]] == "yes") for it in items}
        blk = {"n": len(items), "detectors": {}, "rates": {}}
        dets = {"rule": {it["id"]: int(it["rule_detector_revision"]) for it in items}, "qwen7b": {it["id"]: int(judge[it["id"]]["revise_qwen"]) for it in items},
                "gptoss20b": {it["id"]: int(judge[it["id"]]["revise_oss"]) for it in items}}
        ids = [it["id"] for it in items]
        for name, d in dets.items():
            H = [h[i] for i in ids]; D = [d[i] for i in ids]
            k0, c0, po = kappa_ac1(H, D)
            ks, cs = [], []
            for _ in range(a.B):
                ix = rng.integers(0, len(ids), len(ids)); kk, cc, _ = kappa_ac1([H[j] for j in ix], [D[j] for j in ix]); ks.append(kk); cs.append(cc)
            ks = [x for x in ks if x == x]
            blk["detectors"][name] = {"kappa": k0, "kappa_ci": st.percentile_ci(ks), "ac1": c0, "ac1_ci": st.percentile_ci(cs), "agreement": po, "n_human_yes": int(sum(H)), "n_detector_yes": int(sum(D))}
        for ck in ("base", "1900"):
            for g, f in (("all", lambda it: True), ("marker_initial", lambda it: it["marker_initial"]), ("other", lambda it: not it["marker_initial"])):
                sub = [it for it in items if it["checkpoint"] == ck and f(it)]
                if not sub:
                    continue
                r0 = rate(sub, h); rs = []
                for _ in range(a.B):
                    ix = rng.integers(0, len(sub), len(sub)); rs.append(rate([sub[j] for j in ix], h))
                blk["rates"]["%s|%s" % (ck, g)] = {"est": r0, "ci": st.percentile_ci(rs), "n": len(sub)}
        blk["judge_rates"] = {}
        for name, d in dets.items():
            for ck in ("base", "1900"):
                sub = [it for it in items if it["checkpoint"] == ck]
                r0 = rate(sub, d); rs = []
                for _ in range(a.B):
                    ix = rng.integers(0, len(sub), len(sub)); rs.append(rate([sub[j] for j in ix], d))
                blk["judge_rates"]["%s|%s" % (name, ck)] = {"est": r0, "ci": st.percentile_ci(rs), "n": len(sub)}
        out_A[sens] = blk
    prim = out_A["unsure_as_no"]
    r_b, r_l = prim["rates"]["base|all"]["est"], prim["rates"]["1900|all"]["est"]
    best = max(prim["detectors"].items(), key=lambda kv: kv[1]["kappa"])
    falls = (r_l - r_b) < -0.01
    flat_or_up = (r_l - r_b) >= -0.01
    verdict = "(iii) human direction disagrees or is flat within 1 point: 'revisions become rarer' and the revision-opener paradigm move to the appendix" if flat_or_up else (
        "(i) claim stays, human-weighted rates reported" if best[1]["kappa"] >= 0.4 else "(ii) sign-level statement only: direction confirmed by human labels; size not identified")
    out_A["decision_rule_A"] = {"human_weighted_rate_base": r_b, "human_weighted_rate_1900": r_l, "delta_points": 100 * (r_l - r_b), "best_detector": best[0], "best_kappa": best[1]["kappa"], "verdict": verdict}
    res["part_A"] = out_A
    res["provenance"] = ("Scored from the label file passed on the command line. GUARD: the V3 sheet returned on 2026-10-02 was filled by a GPT-6/Codex agent (final ruling), so it must NOT be passed here as human labels; only a label file that passes the provenance check of HUMAN_AUDIT_V4_meta.json qualifies.")
    B = [dict(id=i, **k) for i, k in key.items() if k["part"] == "B" and i in lab]
    out_B = {}
    for g in ("stage1", "stage2", "no_marker_runs"):
        sub = [it for it in B if it["group"] == g]
        yes = sum(lab[it["id"]] == "yes" for it in sub); uns = sum(lab[it["id"]] == "unsure" for it in sub)
        lo, hi = cp(yes, len(sub))
        out_B[g] = {"n": len(sub), "yes": int(yes), "unsure": int(uns), "share_yes": yes / max(len(sub), 1), "ci": [lo, hi]}
    s1, s2 = out_B["stage1"]["share_yes"], out_B["stage2"]["share_yes"]
    out_B["decision_rule_B"] = {"stage1_yes_le_25pct": s1 <= 0.25, "stage2_yes_ge_75pct": s2 >= 0.75,
                                "verdict": "characterisation stays (shares reported)" if (s1 <= 0.25 and s2 >= 0.75) else "report counts only and drop the 'ordinary English' / 'already reasoning use' characterisation"}
    res["part_B"] = out_B
    common.write_json(a.out, res)
    from . import keys as K
    ex = K.Exporter()
    src = str(a.out).split("exp/")[-1] if "exp/" in str(a.out) else str(a.out)
    for sens, suf in (("unsure_as_no", ""), ("unsure_excluded", "_exunsure")):
        blk = res["part_A"][sens]
        for dn, tag in (("qwen7b", "qwen"), ("gptoss20b", "oss"), ("rule", "rule")):
            v = blk["detectors"][dn]
            for q, key_ in (("kappa", "kappa"), ("ac1", "ac1")):
                ex.put("gold.v3.%s_human_%s%s" % (key_, tag, suf), v[q], "dec2", "exp/gold/human_audit_v3_results.json::part_A.%s.detectors.%s.%s" % (sens, dn, q))
                ex.put("gold.v3.%s_human_%s%s.lo" % (key_, tag, suf), v[q + "_ci"][0], "dec2", "bootstrap over items (B=%d)" % a.B)
                ex.put("gold.v3.%s_human_%s%s.hi" % (key_, tag, suf), v[q + "_ci"][1], "dec2", "bootstrap over items (B=%d)" % a.B)
        for ck, ckn in (("base", "base"), ("1900", "1900")):
            r = blk["rates"].get("%s|all" % ck)
            if r:
                for q, x in (("", r["est"]), (".lo", r["ci"][0]), (".hi", r["ci"][1])):
                    ex.put("gold.v3.revrate_%s%s%s" % (ckn, suf, q), 100 * x, "dec1", "exp/gold/human_audit_v3_results.json::part_A.%s.rates (human-weighted by inverse sampling fractions; %s)" % (sens, "unsure=No" if not suf else "unsure excluded"))
            for dn, tag in (("qwen7b", "judge"), ("gptoss20b", "judge_oss"), ("rule", "rule")):
                jr = blk["judge_rates"]["%s|%s" % (dn, ck)]
                for q, x in (("", jr["est"]), (".lo", jr["ci"][0]), (".hi", jr["ci"][1])):
                    ex.put("gold.v3.revrate_%s_%s%s%s" % (tag, ckn, suf, q), 100 * x, "dec1", "same items and weights, detector %s" % dn)
        ex.put("gold.v3.n_A%s" % suf, blk["n"], "int", "part A items scored")
    for g, tag in (("stage1", "stage1"), ("stage2", "stage2"), ("no_marker_runs", "nomarker")):
        v = res["part_B"][g]
        ex.put("gold.v3.B.%s_yes" % tag, v["yes"], "int", "exp/gold/human_audit_v3_results.json::part_B.%s" % g)
        ex.put("gold.v3.B.n_%s" % tag, v["n"], "int", "part B items in the group")
        ex.put("gold.v3.B.%s_unsure" % tag, v["unsure"], "int", "unsure labels in the group")
        ex.put("gold.v3.B.%s_share" % tag, 100 * v["share_yes"], "dec1", "yes-share, exact binomial CI")
        ex.put("gold.v3.B.%s_share.lo" % tag, 100 * v["ci"][0], "dec1", "Clopper-Pearson 95%"); ex.put("gold.v3.B.%s_share.hi" % tag, 100 * v["ci"][1], "dec1", "Clopper-Pearson 95%")
    common.write_json(str(Path(a.out).parent / "gold_keys_v3.json"), {"pilot": False, "entries": ex.entries})
    print(json.dumps({"A_rates": {k: round(v["est"], 3) for k, v in prim["rates"].items()}, "A_detectors": {k: (round(v["kappa"], 2), round(v["ac1"], 2)) for k, v in prim["detectors"].items()}, "rule_A": out_A["decision_rule_A"]["verdict"],
                      "B": {g: (v["yes"], v["n"], v.get("unsure")) for g, v in out_B.items() if g != "decision_rule_B"}, "rule_B": out_B["decision_rule_B"]["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
