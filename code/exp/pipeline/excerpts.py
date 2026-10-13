"""Verbatim Figure-1 excerpts (figures/specs/P1_fig1.md, selection rule by the P1 lead).

A qualifying prefix has: the source trace correct with no revision/marker up to the cut (by construction of the
prefix selector, checked again with the rule detector); ALL four S1@1900 continuations of the empty arm judged
`continue` (not REVISE); at least one S1@1900 `Wait,` continuation that the judge AND the rule call REVISE and
(the prefix was correct by construction, so any revision is a false revision; whether the final answer recovers is recorded, not required); and ALL four S1@base `Wait,`
continuations not REVISE. The shortest qualifying examples (prefix last sentence + both continuations) win.

    python -m exp.pipeline.excerpts --res <results dir> --data MATH-500/test.jsonl --out exp/P0/fig1_excerpts.json
"""
from __future__ import annotations

import argparse
import glob
from typing import Dict, List

from . import common, insert as ins, markers as mk, prompts


def _cont_text(raw_rows: Dict[tuple, dict], policy: str, prefix_id: str, arm: str, j: int) -> str:
    return raw_rows[(policy, prefix_id, arm, j)]["text"]


def _verbatim(form: str, text: str, n: int = 3) -> str:
    """The inserted form + generated text, cut after the first n sentences (unstripped, as shown to a reader)."""
    full = form + text
    sents = mk.split_sentences(full)[:n]
    return full[:sents[-1].end] if sents else full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+", required=True)
    ap.add_argument("--conts", nargs="+", required=True, help="raw conts_S1@*.jsonl.gz (with the full generated text)")
    ap.add_argument("--prefixes", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_examples", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    uid = {p["pid"]: p["unique_id"] for p in prompts.load_problems(a.data)}
    pre = {p["prefix_id"]: p for f in a.prefixes for p in common.read_jsonl(f)}
    judged = [r for f in a.judged for r in common.read_jsonl(f)]
    raw = {(r["policy"], r["prefix_id"], r["arm"], r["j"]): r for f in a.conts for r in common.read_jsonl(f)}
    by: Dict[tuple, List[dict]] = {}
    for r in judged:
        by.setdefault((r["policy"], r["prefix_id"], r["arm"]), []).append(r)

    cands = []
    for pid_, p in pre.items():
        rl_none = by.get(("S1@1900", pid_, "none"), [])
        rl_wait = by.get(("S1@1900", pid_, "wait"), [])
        b_wait = by.get(("S1@base", pid_, "wait"), [])
        if not (rl_none and rl_wait and b_wait):
            continue
        if any(r["judge_revise"] or r["rule_revise"] for r in rl_none) or any(r["judge_revise"] or r["rule_revise"] for r in b_wait):
            continue
        false_rev = [r for r in rl_wait if r["judge_revise"] and r["rule_revise"]]   # judge and rule agree on the revision
        if not false_rev:
            continue
        last = mk.split_sentences(p["prefix"])[-1].text
        best = None
        for r in false_rev:
            rl_txt = _verbatim("Wait,", _cont_text(raw, "S1@1900", pid_, "wait", r["j"]))
            b_txt = _verbatim("Wait,", _cont_text(raw, "S1@base", pid_, "wait", b_wait[0]["j"]))
            cost = len(last) + len(rl_txt) + len(b_txt)
            if best is None or cost < best[0]:
                best = (cost, r, rl_txt, b_txt)
        cost, r, rl_txt, b_txt = best
        cands.append({"cost": cost, "problem_id": uid[p["pid"]], "prefix_id": pid_, "prefix_src": p["src"], "k": p["k"],
                      "prefix_last_sentence": last, "inserted": "Wait,", "base_continuation": b_txt, "rl_continuation": rl_txt,
                      "rl_first_sentence": mk.split_sentences(rl_txt)[0].text, "base_first_sentence": mk.split_sentences(b_txt)[0].text,
                      "rl_judge_category": r["judge_cat"], "rl_final_answer": r["pred"], "rl_final_correct": bool(r["correct"]), "gold": p["gold"],
                      "base_model": "allenai/Olmo-3-1025-7B", "rl_checkpoint": "allenai/Olmo-3-7B-RL-Zero-Math@step_1900",
                      "arm_counts": {"n_continuations_per_arm": len(rl_wait),
                                     "rl_wait_revised": sum(bool(x["judge_revise"]) for x in rl_wait),
                                     "rl_none_revised": sum(bool(x["judge_revise"]) for x in rl_none),
                                     "base_wait_revised": sum(bool(x["judge_revise"]) for x in b_wait),
                                     "rl_wait_false_rev": sum(bool(x.get("false_rev")) for x in rl_wait)},
                      "n_qualifying_prefixes": None})
    cands.sort(key=lambda c: c["cost"])
    for c in cands:
        c["n_qualifying_prefixes"] = len(cands)
    top = cands[0]["arm_counts"] if cands else {}
    out = {"selection_rule": __doc__.strip().split("\n\n")[0], "examples": cands[:a.n_examples],
           "fig1": {"n": top.get("n_continuations_per_arm"), "rl_revised": top.get("rl_wait_revised"),
                    "base_revised": top.get("base_wait_revised"), "rl_none_revised": top.get("rl_none_revised")},
           "source_paths": {"judged": a.judged, "conts": a.conts, "prefixes": a.prefixes},
           "prefix_code_hash": common.code_hash()}
    common.write_json(a.out, out)
    print("[excerpts] %d qualifying prefixes; wrote %d examples to %s" % (len(cands), len(out["examples"]), a.out))


if __name__ == "__main__":
    main()
