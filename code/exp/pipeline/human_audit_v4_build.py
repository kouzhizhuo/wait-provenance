"""Build human audit v4 (power extension of v3; design, decision rules and the pooling rule PRE-DECLARED in exp/gold/HUMAN_AUDIT_V4_meta.json BEFORE any v4 label exists).
    python -m exp.pipeline.human_audit_v4_build   -> exp/gold/HUMAN_AUDIT_V4_TODO.jsonl (blank answers; fields id, instruction, text, options), HUMAN_AUDIT_V4_key.json (hidden), .v4_judge_items.jsonl
Part A+ (60): same pool/cells as v3 (S1 E1, base and step 1900; marker-initial x rule-detector flag), 30 per checkpoint (8/8/7/7 over the four cells), responses used in v3 excluded.
Part B+ (54): 27 'wait' hits from the stage-1 pretraining branches and 27 from the stage-2 mid-training branch, v3 hits excluded. Seed 20261004."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "exp"))
from exp.pipeline import common   # noqa: E402
from exp.pipeline import human_audit_v3_build as b3   # noqa: E402

SEED = 20261004
GOLD = ROOT / "exp" / "gold"
NOTE = "Please label PERSONALLY, without AI assistance: do not use any pre-filled, suggested or AI-written answers or comments, and do not copy labels from another file."
INSTR_A = ("Read the CONTEXT (the end of a model's step-by-step maths solution) and the marked TARGET sentence that follows. Does the TARGET change or correct something the solution already stated "
           "(a number, an expression or a claim), negate an earlier statement, or abandon the approach and start over? Re-checking a step and keeping it, or adding a new step, counts as No. "
           "Judge the text as written; you do not need to check the maths. " + NOTE)
INSTR_B = ("Is 'wait' in the marked TARGET used by the writer to interrupt and reconsider their own reasoning (a reflection marker)? Other uses, such as the verb ('wait for'), quoted speech, a name, "
           "or dialogue in a story, count as No. Some texts are noisy early-pretraining text by design; judge them as written. " + NOTE)
OPTS_A = ["Yes: it changes, corrects or negates an earlier value or claim, or starts over", "No: it continues, re-checks without changing anything, or adds a new step", "Unsure"]
OPTS_B = ["Yes: a reflection marker (the writer interrupts to reconsider their own reasoning)", "No: another use (verb, quoted speech, name, story dialogue, ...)", "Unsure"]


def main():
    v3 = common.read_json(GOLD / "HUMAN_AUDIT_V3_key.json")["items"]
    exA = {(k["response"]["pid"], k["response"]["sid"], k["checkpoint"]) for k in v3.values() if k["part"] == "A"}
    exB = {(k["source"], k["response"]["pid"], k["response"]["sid"], k["sentence_index"]) for k in v3.values() if k["part"] == "B"}
    rng = random.Random(SEED)
    alloc = {(1, 1): 8, (1, 0): 8, (0, 1): 7, (0, 0): 7}
    ia, ka = b3.part_a(rng, alloc=alloc, exclude=exA)
    ib, kb = b3.part_b(rng, n_per={"stage1": 27, "stage2": 27, "no_marker_runs": 0}, exclude=exB, groups_only={"stage1", "stage2"})
    allitems = list(zip(ia + ib, ka + kb))
    rng.shuffle(allitems)
    todo, key, judge_items = [], {}, []
    for n, (it, k) in enumerate(allitems, 1):
        iid = "V4-%03d" % n
        if it["part"] == "A":
            todo.append({"id": iid, "instruction": INSTR_A, "text": "CONTEXT: %s\n\n>>> TARGET: %s <<<" % (it["context"], it["target"]), "options": OPTS_A})
            judge_items.append({"item_id": iid, **it["_judge"]})
        else:
            todo.append({"id": iid, "instruction": INSTR_B, "text": "%s\n\n>>> TARGET: %s <<<" % (it["context"], it["target"]), "options": OPTS_B})
        key[iid] = k
    common.write_jsonl(GOLD / "HUMAN_AUDIT_V4_TODO.jsonl", todo)
    common.write_jsonl(GOLD / ".v4_judge_items.jsonl", judge_items)
    cnt = {g: sum(1 for k in key.values() if k["part"] == "B" and k["group"] == g) for g in ("stage1", "stage2")}
    common.write_json(GOLD / "HUMAN_AUDIT_V4_key.json", {"seed": SEED, "meta": "exp/gold/HUMAN_AUDIT_V4_meta.json", "items": key, "part_B_counts": cnt,
                       "v3_items_excluded": {"A_responses": len(exA), "B_hits": len(exB)}})
    cells = {}
    for k in key.values():
        if k["part"] == "A":
            c = (k["checkpoint"], k["marker_initial"], k["rule_detector_revision"]); cells[c] = cells.get(c, 0) + 1
    print("items", len(todo), "A", sum(k["part"] == "A" for k in key.values()), "B", cnt, cells)


if __name__ == "__main__":
    main()
