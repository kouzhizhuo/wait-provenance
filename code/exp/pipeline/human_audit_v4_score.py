"""Score human audit v4 ALONE by the rules PRE-DECLARED in exp/gold/HUMAN_AUDIT_V4_meta.json (= the v3 decision rules A(i)-(iii) and B; pooling with v3 is off: v3 was filled by an agent).
    python -m exp.pipeline.human_audit_v4_score --labels exp/gold/HUMAN_AUDIT_V4_LABELS.jsonl --out exp/gold/human_audit_v4_results.json
Provenance check (c) first: if the human binary vector (unsure = No) agrees with ANY detector vector (rule detector, Qwen, gpt-oss) on >= 95% of the part-A+ items the script STOPS and writes nothing.
Then: kappa / Gwet AC1 human vs {rule, qwen, oss} (item bootstrap B=5000; unsure=No primary, unsure excluded as sensitivity); human-weighted revision rate at base and 1900
(weights N_cell / n_cell) and the same rate under each detector; Part B yes-share per group (exact binomial). Keys gold.v4.* -> exp/gold/gold_keys_v4.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import common, keys as K, stats as st
from .human_audit_v3_score import cp, kappa_ac1, rate

ROOT = Path(__file__).resolve().parents[2]


def norm(label: str) -> str:
    t = label.strip().lower()
    return "yes" if t.startswith("yes") else "no" if t.startswith("no") else "unsure" if t.startswith("unsure") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=5000)
    a = ap.parse_args()
    if "GPT6" in str(a.labels) or "LLM_" in str(a.labels):
        raise SystemExit("refusing to score LLM labels as human labels")
    gold = ROOT / "exp" / "gold"
    key = common.read_json(gold / "HUMAN_AUDIT_V4_key.json")["items"]
    rows = common.read_jsonl(a.labels)
    lab = {r.get("id") or r.get("item_id"): norm(r["label"]) for r in rows}
    judge = {r["item_id"]: r for r in common.read_jsonl(gold / "judge_v4.jsonl")}
    A = [dict(id=i, **k) for i, k in key.items() if k["part"] == "A" and i in lab]
    dets_all = {"rule": {it["id"]: int(it["rule_detector_revision"]) for it in A}, "qwen7b": {it["id"]: int(judge[it["id"]]["revise_qwen"]) for it in A},
                "gptoss20b": {it["id"]: int(judge[it["id"]]["revise_oss"]) for it in A}}
    h_all = {it["id"]: int(lab[it["id"]] == "yes") for it in A}
    # ---- provenance check (c)
    chk = {n: float(np.mean([h_all[i] == d[i] for i in h_all])) for n, d in dets_all.items()}
    if any(v >= 0.95 for v in chk.values()):
        print("STOP: provenance check (c) FAILED: human vector agrees with", {n: round(v, 3) for n, v in chk.items()})
        raise SystemExit(2)
    res = {"labels_file": str(a.labels), "provenance_check_c_agreement_with_detectors": chk, "n_labelled": len(lab), "label_counts": {k: sum(v == k for v in lab.values()) for k in ("yes", "no", "unsure")},
           "provenance": "User attestation 2026-10-02 (via the coordinator): 'I decided every label myself without seeing any suggested answer; an agent only typed my choices'. Entry mechanism: the sheet was written by openpyxl and the choices typed in by an agent; reported as human labels on the author's attestation."}
    rng = np.random.default_rng(20261004)
    out_A = {"n": len(A)}
    for sens in ("unsure_as_no", "unsure_excluded"):
        items = [it for it in A if sens == "unsure_as_no" or lab[it["id"]] != "unsure"]
        h = {it["id"]: h_all[it["id"]] for it in items}
        ids = [it["id"] for it in items]
        blk = {"n": len(items), "detectors": {}, "rates": {}, "judge_rates": {}}
        for name, d in dets_all.items():
            H = [h[i] for i in ids]; D = [d[i] for i in ids]
            k0, c0, po = kappa_ac1(H, D); ks, cs = [], []
            for _ in range(a.B):
                ix = rng.integers(0, len(ids), len(ids)); kk, cc, _ = kappa_ac1([H[j] for j in ix], [D[j] for j in ix]); ks.append(kk); cs.append(cc)
            ks = [x for x in ks if x == x]
            blk["detectors"][name] = {"kappa": k0, "kappa_ci": st.percentile_ci(ks), "ac1": c0, "ac1_ci": st.percentile_ci(cs), "agreement": po, "n_human_yes": int(sum(H)), "n_detector_yes": int(sum(D))}
        for ck in ("base", "1900"):
            sub = [it for it in items if it["checkpoint"] == ck]
            for g, f in (("all", lambda it: True), ("marker_initial", lambda it: it["marker_initial"]), ("other", lambda it: not it["marker_initial"])):
                ss = [it for it in sub if f(it)]
                if ss:
                    r0 = rate(ss, h); rs = [rate([ss[j] for j in rng.integers(0, len(ss), len(ss))], h) for _ in range(a.B)]
                    blk["rates"]["%s|%s" % (ck, g)] = {"est": r0, "ci": st.percentile_ci(rs), "n": len(ss)}
            for name, d in dets_all.items():
                r0 = rate(sub, d); rs = [rate([sub[j] for j in rng.integers(0, len(sub), len(sub))], d) for _ in range(a.B)]
                blk["judge_rates"]["%s|%s" % (name, ck)] = {"est": r0, "ci": st.percentile_ci(rs), "n": len(sub)}
        out_A[sens] = blk
    prim = out_A["unsure_as_no"]
    r_b, r_l = prim["rates"]["base|all"]["est"], prim["rates"]["1900|all"]["est"]
    best = max(prim["detectors"].items(), key=lambda kv: kv[1]["kappa"])
    if (r_l - r_b) >= -0.01:
        verdict = "(iii) human direction disagrees or is flat within 1 point: 'revisions become rarer' and the revision-opener paradigm result move to the appendix"
    elif best[1]["kappa"] >= 0.4:
        verdict = "(i) claim stays; human-weighted rates reported next to the detector rates"
    else:
        verdict = "(ii) sign-level statement only: direction confirmed by human labels; size not identified"
    out_A["decision_rule_A"] = {"human_weighted_rate_base": r_b, "human_weighted_rate_1900": r_l, "delta_points": 100 * (r_l - r_b), "best_detector": best[0], "best_kappa": best[1]["kappa"], "verdict": verdict}
    res["part_A"] = out_A
    B = [dict(id=i, **k) for i, k in key.items() if k["part"] == "B" and i in lab]
    out_B = {}
    for g in ("stage1", "stage2"):
        sub = [it for it in B if it["group"] == g]
        yes = sum(lab[it["id"]] == "yes" for it in sub); uns = sum(lab[it["id"]] == "unsure" for it in sub)
        lo, hi = cp(yes, len(sub))
        out_B[g] = {"n": len(sub), "yes": int(yes), "unsure": int(uns), "share_yes": yes / max(len(sub), 1), "ci": [lo, hi]}
    s1, s2 = out_B["stage1"]["share_yes"], out_B["stage2"]["share_yes"]
    out_B["decision_rule_B"] = {"stage1_yes_le_25pct": s1 <= 0.25, "stage2_yes_ge_75pct": s2 >= 0.75,
                                "verdict": "characterisation stays (shares reported)" if (s1 <= 0.25 and s2 >= 0.75) else "report counts only and drop the 'ordinary English' / 'already reasoning use' characterisation"}
    res["part_B"] = out_B
    common.write_json(a.out, res)
    ex = K.Exporter()
    for sens, suf in (("unsure_as_no", ""), ("unsure_excluded", "_exunsure")):
        blk = res["part_A"][sens]
        for dn, tag in (("qwen7b", "qwen"), ("gptoss20b", "oss"), ("rule", "rule")):
            v = blk["detectors"][dn]
            for q in ("kappa", "ac1"):
                for s_, x in (("", v[q]), (".lo", v[q + "_ci"][0]), (".hi", v[q + "_ci"][1])):
                    ex.put("gold.v4.%s_human_%s%s%s" % (q, tag, suf, s_), x, "dec2", "exp/gold/human_audit_v4_results.json::part_A.%s.detectors.%s.%s (item bootstrap B=%d)" % (sens, dn, q, a.B))
        for ck in ("base", "1900"):
            r = blk["rates"].get("%s|all" % ck)
            for s_, x in (("", r["est"]), (".lo", r["ci"][0]), (".hi", r["ci"][1])):
                ex.put("gold.v4.revrate_%s%s%s" % (ck, suf, s_), 100 * x, "dec1", "human-weighted rate (weights N_cell/n_cell); %s" % ("unsure = No" if not suf else "unsure excluded"))
            for dn, tag in (("qwen7b", "judge"), ("gptoss20b", "judge_oss"), ("rule", "rule")):
                jr = blk["judge_rates"]["%s|%s" % (dn, ck)]
                for s_, x in (("", jr["est"]), (".lo", jr["ci"][0]), (".hi", jr["ci"][1])):
                    ex.put("gold.v4.revrate_%s_%s%s%s" % (tag, ck, suf, s_), 100 * x, "dec1", "same items and weights, detector %s" % dn)
        ex.put("gold.v4.n_A%s" % suf, blk["n"], "int", "part A+ items scored")
    for g in ("stage1", "stage2"):
        v = res["part_B"][g]
        ex.put("gold.v4.B.%s_yes" % g, v["yes"], "int", "exp/gold/human_audit_v4_results.json::part_B.%s" % g)
        ex.put("gold.v4.B.n_%s" % g, v["n"], "int", "part B+ items in the group")
        ex.put("gold.v4.B.%s_unsure" % g, v["unsure"], "int", "unsure labels")
        for s_, x in (("", 100 * v["share_yes"]), (".lo", 100 * v["ci"][0]), (".hi", 100 * v["ci"][1])):
            ex.put("gold.v4.B.%s_share%s" % (g, s_), x, "dec1", "yes-share, Clopper-Pearson 95%")
    common.write_json(Path(a.out).parent / "gold_keys_v4.json", {"pilot": False, "entries": ex.entries})
    print(json.dumps({"provenance_c": chk, "counts": res["label_counts"], "rates": {k: [round(v["est"], 3), [round(x, 3) for x in v["ci"]]] for k, v in prim["rates"].items()},
                      "detectors": {k: [round(v["kappa"], 2), round(v["ac1"], 2)] for k, v in prim["detectors"].items()}, "judge_rates": {k: round(v["est"], 3) for k, v in prim["judge_rates"].items()},
                      "A": verdict, "B": {g: [res["part_B"][g]["yes"], res["part_B"][g]["n"], res["part_B"][g]["unsure"]] for g in ("stage1", "stage2")}, "B_verdict": res["part_B"]["decision_rule_B"]["verdict"]}, indent=1))


if __name__ == "__main__":
    main()
