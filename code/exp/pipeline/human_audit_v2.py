"""Human audit v2 (E3 insertion outcome REVISE) vs the judges, by the PRE-DECLARED rules in exp/gold/HUMAN_AUDIT_V2_meta.json + D35.

    python -m exp.pipeline.human_audit_v2 --gold exp/gold --judged exp/results/P0/insertion/judged.jsonl.gz \
        --oss exp/gold/judge_v2_oss.jsonl.gz --out exp/gold/human_audit_v2_results.json

Design facts used: 40 items from the P0 E3 judged windows (policies S1@1900 and S1@base, 20 each), sampled 10 judge-REVISE + 10 judge-not-REVISE
per policy (stratified on the PRIMARY judge, Qwen2.5-7B-Instruct), one item per prefix, arm-blind stripped windows. Because the sample is enriched on
the judge label, the plain 2x2 table is not representative: we report (a) UNWEIGHTED agreement/kappa on the 40 items as labelled (the statistic the
pre-declared decision rule names) and (b) WEIGHTED sensitivity/specificity/PPV/NPV/kappa with inverse-sampling weights w = N_stratum / n_stratum
(population of judged windows per policy and judge label), which is what calibrates a rate. Stratified bootstrap over items (within policy x judge label).
'Unsure' items (comment column) are INCLUDED in the primary numbers and EXCLUDED in the sensitivity block; no other exclusion, no relabelling.
Human labels are used exactly as given. Judges: Qwen (primary) and gpt-oss-20b (co-reported, D35).

Decision rule (pre-declared in the meta): human-vs-judge binary kappa >= 0.6 -> judge-based RD_op/R_amp stay primary; 0.4-0.6 -> estimates
corrected with the human-calibrated sensitivity/specificity become primary; < 0.4 -> E3 label-based claims are exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import common, stats as st


def table(h: np.ndarray, j: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    a = float((w * ((h == 1) & (j == 1))).sum()); b = float((w * ((h == 0) & (j == 1))).sum())
    c = float((w * ((h == 1) & (j == 0))).sum()); d = float((w * ((h == 0) & (j == 0))).sum())
    T = a + b + c + d
    po = (a + d) / T
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (T * T)
    sens = a / (a + c) if a + c else float("nan"); spec = d / (b + d) if b + d else float("nan")
    return {"agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else float("nan"), "sensitivity": sens, "specificity": spec,
            "youden_J": sens + spec - 1, "PPV": a / (a + b) if a + b else float("nan"), "NPV": d / (c + d) if c + d else float("nan")}


def strat_boot(strata: np.ndarray, h, j, w, B: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    return [np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]) for _ in range(B)]


def summarize(h, j, w, strata, B=5000) -> dict:
    out = {"n": int(len(h)), "n_human_revise": int(h.sum()), "n_judge_revise": int(j.sum())}
    for name, ww in (("unweighted", np.ones_like(w)), ("weighted", w)):
        est = table(h, j, ww)
        reps = [table(h[i], j[i], ww[i]) for i in strat_boot(strata, h, j, ww, B)]
        out[name] = {k: {"est": v, "lo": st.percentile_ci([r[k] for r in reps])[0], "hi": st.percentile_ci([r[k] for r in reps])[1]} for k, v in est.items()}
    return out


def band(k: dict) -> str:
    e, lo, hi = k["est"], k["lo"], k["hi"]
    b = "primary (>= 0.6)" if e >= 0.6 else ("corrected-primary (0.4-0.6)" if e >= 0.4 else "exploratory (< 0.4)")
    return "%s; 95%% CI [%.2f, %.2f]%s" % (b, lo, hi, "" if (lo >= (0.6 if e >= 0.6 else 0.4 if e >= 0.4 else -9)) and (hi <= (9 if e >= 0.6 else 0.6 if e >= 0.4 else 0.4)) else " (CI crosses a band boundary)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--judged", required=True, help="exp/results/P0/insertion/judged.jsonl.gz: the population the items were drawn from")
    ap.add_argument("--oss", default=None, help="gpt-oss-20b judgments of the same 40 windows (judge_v2_oss.jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    g = Path(a.gold)
    meta = json.load(open(g / "HUMAN_AUDIT_V2_meta.json"))
    o2l = meta["options_to_labels"]
    lab = {r["id"]: r for r in common.read_jsonl(g / "HUMAN_AUDIT_V2_LABELS.jsonl")}
    items = meta["id_map"]
    pop: Dict[str, Dict[bool, int]] = {}
    for r in common.read_jsonl(a.judged):
        if r["policy"] in ("S1@1900", "S1@base"):
            pop.setdefault(r["policy"], {True: 0, False: 0})[bool(r["judge_revise"])] += 1
    n_str: Dict[tuple, int] = {}
    for it in items:
        n_str[(it["policy"], bool(it["judge_revise"]))] = n_str.get((it["policy"], bool(it["judge_revise"])), 0) + 1
    H, Q, W, S, P, U, ids = [], [], [], [], [], [], []
    for it in items:
        r = lab.get(it["id"])
        if not r or r.get("label") not in o2l:
            continue
        H.append(int(o2l[r["label"]] == "revise")); Q.append(int(it["judge_revise"]))
        W.append(pop[it["policy"]][bool(it["judge_revise"])] / n_str[(it["policy"], bool(it["judge_revise"]))])
        S.append("%s|%s" % (it["policy"], it["judge_revise"])); P.append(it["policy"]); U.append(int("unsure" in (r.get("comment") or "").lower())); ids.append(it["id"])
    H, Q, W, S, P, U = np.array(H), np.array(Q), np.array(W), np.array(S), np.array(P), np.array(U)
    judges = {"qwen7b": Q}
    if a.oss and Path(a.oss).exists():
        okey = {(r["policy"], r["prefix_id"], r["arm"], r["j"]): int(bool(r["judge_revise"])) for r in common.read_jsonl(a.oss)}
        O = np.array([okey.get((it["policy"], it["prefix_id"], it["arm"], it["j"]), -1) for it in items if it["id"] in set(ids)])
        if (O >= 0).all():
            judges["gptoss20b"] = O
    res: dict = {"scope": "human = the user (author), used as given; audit v2 = E3 outcome REVISE; decision rule pre-declared in HUMAN_AUDIT_V2_meta.json",
                 "n_labelled": int(len(H)), "n_unsure": int(U.sum()), "human_revise_count": int(H.sum()),
                 "sampling_weights": {k: {"N": v} for k, v in pop.items()}, "judges": {}}
    draws: Dict[str, list] = {}
    for jn, J in judges.items():
        block = {"primary_unsure_included": summarize(H, J, W, S, a.B)}
        keep = U == 0
        block["sensitivity_unsure_excluded"] = summarize(H[keep], J[keep], W[keep], S[keep], a.B) if keep.sum() > 4 else {"n": int(keep.sum())}
        block["by_policy_unweighted"] = {p: summarize(H[P == p], J[P == p], W[P == p], S[P == p], a.B // 5) for p in ("S1@1900", "S1@base")}
        block["decision_band"] = band(block["primary_unsure_included"]["unweighted"]["kappa"])
        block["decision_band_weighted_kappa"] = band(block["primary_unsure_included"]["weighted"]["kappa"])
        block["confusion_unweighted"] = {"tp": int(((H == 1) & (J == 1)).sum()), "fn": int(((H == 1) & (J == 0)).sum()),
                                         "fp": int(((H == 0) & (J == 1)).sum()), "tn": int(((H == 0) & (J == 0)).sum())}
        res["judges"][jn] = block
        sd = []
        for i in strat_boot(S, H, J, W, 2000, seed=1):
            t = table(H[i], J[i], W[i])
            if t["sensitivity"] == t["sensitivity"] and t["specificity"] == t["specificity"]:
                sd.append([min(max(t["sensitivity"], 0.001), 0.999), min(max(t["specificity"], 0.001), 0.999)])
        draws[jn] = sd
    res["posterior_draws_sens_spec"] = draws.get("qwen7b", [])            # primary judge: read by report.py --calib
    res["posterior_draws_sens_spec_oss"] = draws.get("gptoss20b", [])
    # keys
    keys = []
    for jn, suf in (("qwen7b", ""), ("gptoss20b", "_oss")):
        if jn not in res["judges"]:
            continue
        for variant, vn in (("primary_unsure_included", ""), ("sensitivity_unsure_excluded", "_excl")):
            b = res["judges"][jn][variant]
            if "unweighted" not in b:
                continue
            for wname, wsuf, quants in (("unweighted", "", (("kappa", "v2_kappabin", "dec2"), ("agreement", "v2_agreebin", "pct"))),
                                        ("weighted", "_w", (("sensitivity", "v2_sens", "pct"), ("specificity", "v2_spec", "pct"), ("youden_J", "v2_j", "dec2"),
                                                            ("PPV", "v2_ppv", "pct"), ("kappa", "v2_kappabin", "dec2")))):
                for q, kname, kind in quants:
                    v = b[wname][q]
                    for stat, ss in (("est", ""), ("lo", ".lo"), ("hi", ".hi")):
                        x = v[stat]
                        if x == x:
                            keys.append({"key": "gold.s1.%s_human_judge%s%s%s%s" % (kname, suf, wsuf, vn, ss),
                                         "value": round(100 * x, 1) if kind == "pct" else round(x, 2),
                                         "src": "exp/gold/human_audit_v2_results.json::judges.%s.%s.%s.%s.%s" % (jn, variant, wname, q, stat)})
    common.write_json(g / "gold_keys_v2.json", {"entries": keys})
    common.write_json(a.out, res)
    for jn, b in res["judges"].items():
        u, w = b["primary_unsure_included"]["unweighted"], b["primary_unsure_included"]["weighted"]
        print("%s: n=%d human-revise=%d | unweighted kappa %.2f [%.2f,%.2f] agree %.2f | weighted sens %.2f [%.2f,%.2f] spec %.2f [%.2f,%.2f] | band: %s"
              % (jn, b["primary_unsure_included"]["n"], b["primary_unsure_included"]["n_human_revise"], u["kappa"]["est"], u["kappa"]["lo"], u["kappa"]["hi"], u["agreement"]["est"],
                 w["sensitivity"]["est"], w["sensitivity"]["lo"], w["sensitivity"]["hi"], w["specificity"]["est"], w["specificity"]["lo"], w["specificity"]["hi"], b["decision_band"]))


if __name__ == "__main__":
    main()
