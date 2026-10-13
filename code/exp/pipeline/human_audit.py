"""Human audit vs judge vs LLM reference labels, and what judge error does to the label-dependent results (TEAM_RULES §1.7).

    python -m exp.pipeline.human_audit --gold exp/gold --insertion exp/results/P0/insertion_pilot.json --out exp/gold/human_audit_results.json

The human labels (the user, an author, 59 of 60 items) are used EXACTLY as given: nothing is substituted, relabelled or filtered.
Items: exp/gold/HUMAN_AUDIT_meta.json id_map (H-xx -> item_id, stratum = series). The judge is Qwen2.5-7B-Instruct with prompt
v1.0 on the STRIPPED sentence and the full context (`judge_gold_*.jsonl`); the human saw the UNSTRIPPED sentence with two preceding
and one following sentence; the LLM reference labels come from labels_adjudicated.jsonl (S1) and labels_analyst_s23.jsonl (S2, S3).

Outputs
  agreement: 5-way and binary (REVISE = revise_value + restart) agreement and Cohen kappa for human-judge, human-reference,
             judge-reference on the same items; 95% percentile bootstrap over items (B = 5000); overall and per series.
  judge accuracy against the human (binary): sensitivity, specificity, Youden J with Beta(1,1) posterior draws.
  correction: with nondifferential misclassification a risk difference is attenuated by J = sens + spec - 1
             (RD_true = RD_obs / J) while a RATIO OF RISK DIFFERENCES (R_amp) and every RANK trend (rho over checkpoints) are unchanged;
             we propagate the posterior of J into the observed E3 risk differences and report the P(J <= 0.2) that makes
             the magnitude unidentifiable. Differential misclassification (judge accuracy that differs between base-like and RL-like
             text) is checked on the reference pool, by checkpoint.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import calib, common, stats as st

REV = {"revise_value", "restart"}
CATS = ["continue", "revise_value", "restart", "reverify_confirm", "branch"]


def rev(x: str) -> int:
    return int(x in REV)


def block(a: List[str], b: List[str], B: int = 5000, seed: int = 0) -> dict:
    n = len(a)
    if n == 0:
        return {"n": 0}
    A, Bv = np.array(a), np.array(b)
    ra, rb = np.array([rev(x) for x in a]), np.array([rev(x) for x in b])
    def stats(idx):
        return (float((A[idx] == Bv[idx]).mean()), st.cohen_kappa(list(A[idx]), list(Bv[idx])),
                float((ra[idx] == rb[idx]).mean()), st.cohen_kappa(list(ra[idx]), list(rb[idx])))
    est = stats(np.arange(n))
    rng = np.random.default_rng(seed)
    reps = np.array([stats(rng.integers(0, n, n)) for _ in range(B)], dtype=float)
    out = {"n": n, "n_revise_a": int(ra.sum()), "n_revise_b": int(rb.sum())}
    for k, name in enumerate(("agreement_5way", "kappa_5way", "agreement_binary", "kappa_binary")):
        lo, hi = st.percentile_ci(reps[:, k])
        out[name] = {"est": est[k], "lo": lo, "hi": hi}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--insertion", default=None, help="insertion_pilot.json (P0) to correct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--judge_files", nargs="+", default=["judge_gold_s1.jsonl", "judge_gold_s23.jsonl"], help="judge label files under --gold")
    ap.add_argument("--tag", default="qwen7b", help="judge tag; non-default tags do not overwrite kappa.json / gold_keys.json")
    ap.add_argument("--compare_with", nargs="*", default=None, help="second judge's label files: paired bootstrap of kappa5 (selection rule, exp/gold/JUDGE_SELECTION.md)")
    a = ap.parse_args()
    g = Path(a.gold)
    meta = json.load(open(g / "HUMAN_AUDIT_meta.json"))
    opt2lab = meta["options_to_labels"]
    idmap = {r["id"]: r for r in meta["id_map"]}
    human = {}
    for r in common.read_jsonl(g / "HUMAN_AUDIT_LABELS.jsonl"):
        if r.get("label") and r["label"] in opt2lab:
            human[idmap[r["id"]]["item_id"]] = opt2lab[r["label"]]
    strat = {v["item_id"]: v["stratum"] for v in idmap.values()}
    judge = {}
    for f in a.judge_files:
        if (g / f).exists():
            judge.update({r["item_id"]: r["judge_cat"] for r in common.read_jsonl(g / f)})
    ref = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_adjudicated.jsonl")}
    ref.update({r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_analyst_s23.jsonl")})
    ids = [i for i in human if i in judge and i in ref]
    res: dict = {"scope": "human = the user (author), used as given; reference = Claude LLM raters; judge = Qwen2.5-7B-Instruct v1.0",
                 "n_human_labels": len(human), "n_matched": len(ids), "missing_judge": sorted(i for i in human if i not in judge)}
    pairs = {"human_vs_judge": (human, judge), "human_vs_reference": (human, ref), "judge_vs_reference": (judge, ref)}
    for name, (x, y) in pairs.items():
        res[name] = {"all": block([x[i] for i in ids], [y[i] for i in ids], a.B)}
        for s in ("S1", "S2", "S3"):
            sub = [i for i in ids if strat[i] == s]
            res[name][s] = block([x[i] for i in sub], [y[i] for i in sub], a.B)
    res["confusion_human_vs_judge"] = {"%s|%s" % (h, j): c for (h, j), c in Counter((human[i], judge[i]) for i in ids).items()}
    res["human_label_counts"] = dict(Counter(human[i] for i in ids))
    # ---- judge accuracy for the binary REVISE decision against the human, Beta(1,1) posteriors
    tp = sum(rev(human[i]) and rev(judge[i]) for i in ids); fn = sum(rev(human[i]) and not rev(judge[i]) for i in ids)
    tn = sum((not rev(human[i])) and (not rev(judge[i])) for i in ids); fp = sum((not rev(human[i])) and rev(judge[i]) for i in ids)
    draws = calib.posterior_draws(tp, fn, tn, fp, n=4000)
    sens, spec = draws[:, 0], draws[:, 1]
    J = sens + spec - 1
    q = lambda x: [float(np.percentile(x, p)) for p in (2.5, 50, 97.5)]   # noqa: E731
    res["judge_binary_vs_human"] = {"tp": tp, "fn": fn, "tn": tn, "fp": fp, "sensitivity": q(sens), "specificity": q(spec), "youden_J": q(J),
                                    "P_J_below_0.2": float((J <= 0.2).mean()), "P_J_below_0": float((J <= 0).mean()),
                                    "note": "with %d human-REVISE items sensitivity is barely identified" % (tp + fn)}
    res["posterior_draws_sens_spec"] = np.round(draws[:2000], 4).tolist()   # reused by report.py --calib
    # ---- differential misclassification: judge vs reference by checkpoint on the S1 reference pool (base-like vs RL-like text)
    pool = []
    for f in ("s1_sample.jsonl", "s1_sample_lead_extra.jsonl"):
        pool += common.read_jsonl(g / f)
    key = json.load(open(g / "to_label_lead_key.json"))["items"]
    lead = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_lead.jsonl")}
    refS1 = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_analyst_d27.jsonl")}
    for k in key:
        if k["part"] == "lead_only":
            refS1[k["item_id"]] = lead[k["id"]]
    refS1.update({r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_adjudicated.jsonl")})
    ck = {p["item_id"]: p["ckpt"] for p in pool}
    res["judge_vs_reference_by_ckpt"] = {}
    for c in sorted(set(ck.values())):
        sub = [i for i in refS1 if i in judge and ck.get(i) == c]
        tp_ = sum(rev(refS1[i]) and rev(judge[i]) for i in sub); fn_ = sum(rev(refS1[i]) and not rev(judge[i]) for i in sub)
        tn_ = sum((not rev(refS1[i])) and (not rev(judge[i])) for i in sub); fp_ = sum((not rev(refS1[i])) and rev(judge[i]) for i in sub)
        res["judge_vs_reference_by_ckpt"][c] = {"n": len(sub), "sens": tp_ / max(1, tp_ + fn_), "spec": tn_ / max(1, tn_ + fp_), "n_ref_revise": tp_ + fn_,
                                                "agreement_5way": float(np.mean([refS1[i] == judge[i] for i in sub])) if sub else None}
    # ---- correct the observed E3 risk differences (P0), reference arm Next,
    if a.insertion and Path(a.insertion).exists():
        ins = json.load(open(a.insertion))["outcomes"]["judge_revise"]["late"]["estimates"]
        rows = {}
        for pol in sorted({k.split(":")[0] for k in ins if ":RD_next" in k}):
            e = ins[pol + ":RD_next"]
            rows[pol] = {"RD_obs": e["est"], "RD_obs_ci": [e["lo"], e["hi"]], **{"RD_corr_" + k: v for k, v in calib.corrected_rd(e["est"], e["lo"], e["hi"], draws).items()}}
        res["e3_correction"] = {"rule": "RD_true = RD_obs / J (nondifferential misclassification); R_amp and rank trends are unchanged", "policies": rows}
    # ---- gold keys (RESULTS_KEYS gold.*) and the kappa.json update
    keys = []
    sid = {"all": "all", "S1": "s1", "S2": "s2", "S3": "s3a"}
    for scope, sk in sid.items():
        for pair, nm in (("human_vs_judge", "human_judge"), ("human_vs_reference", "human_llm"), ("judge_vs_reference", "judge_llm")):
            b_ = res[pair][scope]
            if not b_.get("n"):
                continue
            for k, q_ in (("agreement_5way", "agree5"), ("kappa_5way", "kappa5"), ("agreement_binary", "agreebin"), ("kappa_binary", "kappabin")):
                v = b_[k]
                for stat, suf in (("est", ""), ("lo", ".lo"), ("hi", ".hi")):
                    keys.append({"key": "gold.%s.%s_%s%s" % (sk, q_, nm, suf), "value": round(100 * v[stat], 1) if q_.startswith("agree") else round(v[stat], 2),
                                 "src": "exp/gold/human_audit_results.json::%s.%s.%s.%s" % (pair, scope, k, stat)})
            keys.append({"key": "gold.%s.n_%s" % (sk, nm), "value": b_["n"], "src": "exp/gold/human_audit_results.json::%s.%s.n" % (pair, scope)})
    if a.tag != "qwen7b":
        common.write_json(a.out, res)
        print("[human_audit:%s] human_vs_judge all: %s" % (a.tag, {k: res["human_vs_judge"]["all"].get(k) for k in ("n", "agreement_5way", "kappa_5way", "agreement_binary", "kappa_binary")}))
        return
    common.write_json(g / "gold_keys.json", {"note": "human = the user (author), used as given; llm = Claude reference raters; judge = Qwen2.5-7B-Instruct", "entries": keys})
    if (g / "kappa.json").exists():
        kj = json.load(open(g / "kappa.json"))
        kj["human_audit"] = {k: res[k] for k in ("scope", "n_human_labels", "n_matched", "human_vs_judge", "human_vs_reference", "judge_vs_reference", "judge_binary_vs_human", "confusion_human_vs_judge")}
        common.write_json(g / "kappa.json", kj)
    if a.compare_with:   # paired bootstrap over items: is the second judge better than the first vs the human?
        j2 = {}
        for f in a.compare_with:
            j2.update({r["item_id"]: r["judge_cat"] for r in common.read_jsonl(g / f)})
        ids2 = [i for i in ids if i in j2]
        H = np.array([human[i] for i in ids2]); A_ = np.array([judge[i] for i in ids2]); B_ = np.array([j2[i] for i in ids2])
        rng2 = np.random.default_rng(1); diffs = []
        for _ in range(a.B):
            ix_ = rng2.integers(0, len(ids2), len(ids2))
            diffs.append(st.cohen_kappa(list(H[ix_]), list(B_[ix_])) - st.cohen_kappa(list(H[ix_]), list(A_[ix_])))
        d = np.array(diffs)
        res["selection"] = {"n": len(ids2), "kappa5_first": st.cohen_kappa(list(H), list(A_)), "kappa5_second": st.cohen_kappa(list(H), list(B_)),
                            "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "P_second_better": float((d > 0).mean()),
                            "rule": "switch only if kappa5_second > kappa5_first and P_second_better >= 0.80 (exp/gold/JUDGE_SELECTION.md)"}
    common.write_json(a.out, res)
    hv = res["human_vs_judge"]["all"]
    print(json.dumps({"n_matched": res["n_matched"], "human_vs_judge_all": {k: hv.get(k) for k in ("n", "agreement_5way", "kappa_5way", "agreement_binary", "kappa_binary")},
                      "judge_binary_vs_human": res["judge_binary_vs_human"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
