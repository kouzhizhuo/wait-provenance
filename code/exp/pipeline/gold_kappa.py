"""Agreement statistics for the reference labels (LLM raters; never called hand/human labels).

    python -m exp.pipeline.gold_kappa --gold exp/gold [--judge exp/gold/judge_gold_s1.jsonl] --out exp/gold/kappa.json
(1) Inter-LLM kappa on the 60-item overlap: labels_analyst.jsonl (independent, Opus) vs labels_lead.jsonl (Fable), joined by
    to_label_lead_key.json; 5-way and REVISE-collapsed (revise_value + restart). Disagreements -> overlap_disagreements.jsonl.
(2) Judge vs reference (if --judge): reference = labels_adjudicated (60 overlap) + labels_analyst_d27 (the other 90 of the
    150) + labels_lead (90 lead-only items); kappa 5-way and REVISE-collapsed, confusion matrix.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import common, stats as st

REV = {"revise_value", "restart"}


def rev(x: str) -> int:
    return int(x in REV)


def kappa_block(a, b) -> dict:
    n = len(a)
    return {"n": n, "agreement_5way": sum(x == y for x, y in zip(a, b)) / n if n else float("nan"), "kappa_5way": st.cohen_kappa(a, b),
            "agreement_revise": sum(rev(x) == rev(y) for x, y in zip(a, b)) / n if n else float("nan"),
            "kappa_revise": st.cohen_kappa([rev(x) for x in a], [rev(y) for y in b])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = Path(a.gold)
    key = json.load(open(g / "to_label_lead_key.json"))["items"]
    ana = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_analyst.jsonl")}
    lead = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_lead.jsonl")}
    sent = {r["item_id"]: r for r in common.read_jsonl(g / "s1_sample.jsonl")}
    ov = [k for k in key if k["part"] == "overlap"]
    A = [ana[k["item_id"]] for k in ov]
    L = [lead[k["id"]] for k in ov]
    res = {"scope": "inter-LLM agreement on the overlap (Opus 5.5 vs Fable 5.1), independent labels; NOT human labels",
           "overlap": kappa_block(A, L), "confusion_overlap": {"%s|%s" % (x, y): c for (x, y), c in Counter(zip(A, L)).items()}}
    dis = []
    for k, x, y in zip(ov, A, L):
        if x != y:
            s = sent.get(k["item_id"], {})
            dis.append({"item_id": k["item_id"], "lead_id": k["id"], "analyst": x, "lead": y, "marker_initial": s.get("marker_initial"),
                        "sentence": s.get("sentence"), "next": s.get("next")})
    common.write_jsonl(g / "overlap_disagreements.jsonl", dis)
    res["n_disagreements"] = len(dis)
    if a.judge and Path(a.judge).exists():
        J = {r["item_id"]: r["judge_cat"] for r in common.read_jsonl(a.judge)}
        adj = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_adjudicated.jsonl")}
        d27 = {r["item_id"]: r["label"] for r in common.read_jsonl(g / "labels_analyst_d27.jsonl")}
        ref = {}
        for k in key:
            if k["part"] == "lead_only":
                ref[k["item_id"]] = lead[k["id"]]
        for iid, lab in d27.items():
            ref.setdefault(iid, lab)
        ref.update(adj)   # adjudicated labels override on the overlap
        ids = [i for i in ref if i in J]
        res["judge_vs_reference"] = {**kappa_block([ref[i] for i in ids], [J[i] for i in ids]),
                                     "confusion": {"%s|%s" % (x, y): c for (x, y), c in Counter((ref[i], J[i]) for i in ids).items()},
                                     "reference_sources": {"adjudicated_overlap": len(adj), "analyst_d27": len(d27), "lead_only": sum(k["part"] == "lead_only" for k in key)},
                                     "rows": "reference (Opus/Fable, adjudicated) | judge (Qwen2.5-7B-Instruct, prompts v1.0, stripped sentence)"}
        for part in ("marker_initial", "other"):
            sub = [i for i in ids if bool(sent.get(i, {}).get("marker_initial", False)) == (part == "marker_initial")]
            if sub:
                res["judge_vs_reference"]["by_" + part] = kappa_block([ref[i] for i in sub], [J[i] for i in sub])
    common.write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k in ("overlap", "n_disagreements")}, indent=1))


if __name__ == "__main__":
    main()
