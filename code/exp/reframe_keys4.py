"""D40 detector check against the open-weight co-rater (Qwen3-32B-FP8, thinking off, question in pipeline/valchange_check.py): precision of VALUE_CHANGE=1 windows and
recall-proxy on judge-REVISE=1 windows. Keys gold.s1.vc_* in exp/results/reframe/valchange_check_keys.json.   python3 exp/reframe_keys4.py"""
import json

from reframe_keys import OUT, RES, ROOT, Ex, common
from reframe_keys3 import wilson


def main():
    rate = {r["id"]: r["answer"] for r in common.read_jsonl(RES / "main" / "S1" / "valchange" / "valchange_check_qwen3.jsonl")}
    key = common.read_jsonl(ROOT / "gold" / ".valchange_check_KEY.jsonl")
    X = Ex()
    src = "exp/results/main/S1/valchange/valchange_check_qwen3.jsonl (Qwen3-32B-FP8, no-think, T=0) vs blind key"
    vc = [k for k in key if k["sample"] == "VC1"]
    rv = [k for k in key if k["sample"] == "REV1"]
    yes_vc = sum(rate[k["id"]] == "YES" for k in vc)
    p, lo, hi = wilson(yes_vc, len(vc))
    for q, v in (("", p), (".lo", lo), (".hi", hi)):
        X.put("gold.s1.vc_precision_qwen3" + q, 100 * v, "dec1", src + ": share of 50 VALUE_CHANGE=1 windows the co-rater calls a value change/contradiction")
    X.put("gold.s1.vc_n_checked", len(vc), "int", src)
    X.put("gold.s1.vc_n_yes", yes_vc, "int", src)
    yes_rv = [k for k in rv if rate[k["id"]] == "YES"]
    p2, lo2, hi2 = wilson(sum(k["value_change"] for k in yes_rv), max(len(yes_rv), 1))
    for q, v in (("", p2), (".lo", lo2), (".hi", hi2)):
        X.put("gold.s1.vc_recall_proxy_qwen3" + q, 100 * v, "dec1", src + ": among the 50 random judge-REVISE=1 windows, share of those the co-rater calls a value change that the detector flags")
    X.put("gold.s1.rev_qwen3_yes_share", 100 * len(yes_rv) / len(rv), "dec1", src + ": share of judge-REVISE=1 windows the co-rater calls a value change")
    X.put("gold.s1.rev_qwen3_yes_n", len(yes_rv), "int", src)
    X.put("gold.s1.rev_n_checked", len(rv), "int", src)
    common.write_json(OUT / "valchange_check_keys.json", {"entries": X.entries, "note": "co-rater only; the Claude/human rating of the same 100 items is separate"})
    print({e["key"]: e["value"] for e in X.entries})


main()
