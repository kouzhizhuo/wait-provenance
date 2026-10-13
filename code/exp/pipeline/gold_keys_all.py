"""Consolidate every gold.* key the paper reads into exp/gold/gold_keys.json (RESULTS_KEYS; values pre-rounded, each with src).

Sources: human_audit_results.json (v1, Qwen), human_audit_results_gptoss.json (v1, gpt-oss-20b), human_audit_v2_results.json (v2, E3 outcome),
kappa.json (inter-LLM overlap). Wording: human = the user (author); llm = Claude reference raters; judge = Qwen2.5-7B-Instruct (primary);
oss = gpt-oss-20b (co-reported). v2 sensitivity/specificity are the inverse-sampling-WEIGHTED population estimates; v2 kappa is the UNWEIGHTED kappa on the 40
audited items (the statistic of the pre-declared decision rule); the weighted kappa is exported as kappabin_w.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import common

SID = {"all": "all", "S1": "s1", "S2": "s2", "S3": "s3a"}


def put(keys, key, v, kind, src):
    if v is None or v != v:
        return
    keys.append({"key": key, "value": round(100 * v, 1) if kind == "pct" else round(v, 2), "src": src})


def est3(keys, key, d, kind, src):
    for stat, ss in (("est", ""), ("lo", ".lo"), ("hi", ".hi")):
        put(keys, key + ss, d[stat], kind, "%s.%s" % (src, stat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    a = ap.parse_args()
    g = Path(a.gold)
    keys = []
    # ---- v1 (sentence-level revision judge): Qwen -> *_judge, gpt-oss-20b -> *_oss
    for fname, jtag in (("human_audit_results.json", "judge"), ("human_audit_results_gptoss.json", "oss"), ("human_audit_results_gptoss120b.json", "oss120")):
        if not (g / fname).exists():
            continue
        r = json.load(open(g / fname))
        for scope, sk in SID.items():
            for pair, nm in (("human_vs_judge", "human_" + jtag), ("judge_vs_reference", jtag + "_llm")):
                b = r[pair][scope]
                if not b.get("n"):
                    continue
                for k, q in (("agreement_5way", "agree5"), ("kappa_5way", "kappa5"), ("agreement_binary", "agreebin"), ("kappa_binary", "kappabin")):
                    est3(keys, "gold.%s.%s_%s" % (sk, q, nm), b[k], "pct" if q.startswith("agree") else "dec2", "%s::%s.%s.%s" % (fname, pair, scope, k))
                put(keys, "gold.%s.n_%s" % (sk, nm), b["n"], "raw", fname)
        if jtag == "judge":   # human vs the Claude reference is judge-independent: exported once
            for scope, sk in SID.items():
                b = r["human_vs_reference"][scope]
                if b.get("n"):
                    for k, q in (("agreement_5way", "agree5"), ("kappa_5way", "kappa5"), ("agreement_binary", "agreebin"), ("kappa_binary", "kappabin")):
                        est3(keys, "gold.%s.%s_human_llm" % (sk, q), b[k], "pct" if q.startswith("agree") else "dec2", "%s::human_vs_reference.%s.%s" % (fname, scope, k))
        jb = r["judge_binary_vs_human"]
        for q, name in (("sensitivity", "v1_sens"), ("specificity", "v1_spec"), ("youden_J", "v1_j")):
            lo, med, hi = jb[q]
            est3(keys, "gold.s1.%s_%s" % (name, jtag), {"est": med, "lo": lo, "hi": hi}, "dec2" if q == "youden_J" else "pct", "%s::judge_binary_vs_human.%s (posterior median, 95%% interval)" % (fname, q))
    # ---- inter-LLM (Opus vs Fable on the 60-item overlap)
    if (g / "kappa.json").exists():
        o = json.load(open(g / "kappa.json"))["overlap"]
        for k, q, kind in (("agreement_5way", "agree5", "pct"), ("kappa_5way", "kappa5", "dec2"), ("agreement_revise", "agreebin", "pct"), ("kappa_revise", "kappabin", "dec2")):
            put(keys, "gold.s1.%s_llm_llm" % q, o[k], kind, "kappa.json::overlap.%s" % k)
        put(keys, "gold.s1.n_llm_llm", o["n"], "raw", "kappa.json::overlap.n")
    # ---- v2 (E3 outcome judge)
    if (g / "human_audit_v2_results.json").exists():
        r = json.load(open(g / "human_audit_v2_results.json"))
        put(keys, "gold.v2.n", r["n_labelled"], "raw", "human_audit_v2_results.json::n_labelled")
        put(keys, "gold.v2.n_unsure", r["n_unsure"], "raw", "human_audit_v2_results.json::n_unsure")
        put(keys, "gold.v2.n_human_revise", r["human_revise_count"], "raw", "human_audit_v2_results.json::human_revise_count")
        for jn, jt in (("qwen7b", "judge"), ("gptoss20b", "oss")):
            if jn not in r["judges"]:
                continue
            for variant, vs in (("primary_unsure_included", ""), ("sensitivity_unsure_excluded", "_excl_unsure")):
                b = r["judges"][jn][variant]
                if "unweighted" not in b:
                    continue
                src = "human_audit_v2_results.json::judges.%s.%s" % (jn, variant)
                est3(keys, "gold.v2.kappabin_human_%s%s" % (jt, vs), b["unweighted"]["kappa"], "dec2", src + ".unweighted.kappa")
                est3(keys, "gold.v2.agreebin_human_%s%s" % (jt, vs), b["unweighted"]["agreement"], "pct", src + ".unweighted.agreement")
                est3(keys, "gold.v2.kappabin_w_human_%s%s" % (jt, vs), b["weighted"]["kappa"], "dec2", src + ".weighted.kappa")
                for q, nm, kind in (("sensitivity", "sens", "pct"), ("specificity", "spec", "pct"), ("PPV", "ppv", "pct"), ("youden_J", "j", "dec2")):
                    est3(keys, "gold.v2.%s_%s%s" % (nm, jt, vs), b["weighted"][q], kind, src + ".weighted." + q)
            put(keys, "gold.v2.band_code_%s" % jt, {"p": 2, "c": 1, "e": 0}[r["judges"][jn]["decision_band"][0]], "raw",
                "human_audit_v2_results.json::judges.%s.decision_band (2 primary >= .6, 1 corrected-primary .4-.6, 0 exploratory < .4)" % jn)
    common.write_json(g / "gold_keys.json", {"note": "human = the user (author), used as given; llm = Claude reference raters; judge = Qwen2.5-7B-Instruct; oss = gpt-oss-20b", "entries": keys})
    print("[gold_keys_all] %d keys" % len(keys))


if __name__ == "__main__":
    main()
