"""Build human audit v3 (design PRE-DECLARED in exp/gold/HUMAN_AUDIT_V3_meta.json, seed 20261003). No scoring here.
    python -m exp.pipeline.human_audit_v3_build            -> exp/gold/HUMAN_AUDIT_V3_TODO.jsonl (blind), exp/gold/HUMAN_AUDIT_V3_key.json (hidden),
                                                              exp/gold/.v3_judge_items.jsonl (inputs for the judges; part A only)
Part A (48): S1 E1 generations at base and step 1900; per checkpoint 12 marker-initial + 12 other sentences; within each cell half flagged by the rule detector
(revisions.detect_events on the stripped sentences) and half not (where available); one sentence per response; sentence index >= 1; 3 preceding sentences as context.
Weights = N_cell / n_cell (N_cell = all index>=1 sentences of the checkpoint in the cell).
Part B (24): regex hits of 'wait' in local E1 generations: 8 from the three stage-1 pretraining branches (pooled), 8 from the stage-2 mid-training branch, 8 from runs that do
not use the marker (REF s1/s2/7B base+last, Qwen2.5-1.5B GRPO S3 base+last; Qwen2.5-32B runs are NOT local) uniformly at random within each group. If a group has fewer hits
than required, ALL are taken and the key says so (never resampled from another group).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "exp"))
from exp.pipeline import common, markers as mk, revisions as rv   # noqa: E402
from reframe_keys import Ref, read_gz_jsonl                       # noqa: E402

SEED = 20261003
RES = ROOT / "exp" / "results"
GOLD = ROOT / "exp" / "gold"
QA = ("Does this sentence change or correct something the solution already stated (a number, an expression or a claim), negate an earlier statement, or abandon the approach and start over? "
      "Re-checking a step and keeping it, or adding a new step, counts as No. Answer yes / no / unsure. Judge the text as written; you do not need to check the maths.")
QB = "Is 'wait' here used by the writer to interrupt and reconsider their own reasoning (a reflection marker)? Other uses, such as the verb ('wait for'), quoted speech, or dialogue in a story, count as No. Answer yes / no / unsure."


def part_a(rng, alloc=None, exclude=frozenset()):
    alloc = alloc or {(1, 1): 6, (1, 0): 6, (0, 1): 6, (0, 0): 6}   # (marker_initial, rule-flag) -> items per checkpoint
    items, key = [], []
    for ck in ("base", "1900"):
        path = RES / "main" / "S1" / "gen" / ("S1@%s.graded.jsonl.gz" % ck)
        rows = read_gz_jsonl(path)
        cells = {}
        for ri, r in enumerate(rows):
            sents = mk.split_sentences(r["text"])
            st = [mk.strip_initial_forms(s.text) for s in sents]
            ev = rv.detect_events([x for x, _ in st])
            for i in range(1, len(sents)):
                mi = int(bool(st[i][1]))
                fl = int(rv.is_revise_event(ev[i]))
                cells.setdefault((mi, fl), []).append((ri, i))
        used = set()
        for mi in (1, 0):
            take = {}
            pool = {fl: rng.sample(cells.get((mi, fl), []), len(cells.get((mi, fl), []))) for fl in (0, 1)}
            for fl in (1, 0):
                take[fl] = []
                for ri, i in pool[fl]:
                    if len(take[fl]) >= alloc[(mi, fl)]:
                        break
                    if ri in used or (str(rows[ri]["pid"]), str(rows[ri]["sid"]), ck) in exclude:
                        continue
                    used.add(ri); take[fl].append((ri, i))
            for fl in (1, 0):
                N = len(cells.get((mi, fl), []))
                for ri, i in take[fl]:
                    r = rows[ri]; sents = mk.split_sentences(r["text"])
                    stripped, forms = mk.strip_initial_forms(sents[i].text)
                    ctx = r["text"][sents[max(0, i - 3)].start:sents[i].start].strip()
                    nxt = " ".join(s.text for s in sents[i + 1:i + 3])[:600]
                    key.append({"part": "A", "checkpoint": ck, "marker_initial": bool(mi), "rule_detector_revision": bool(fl), "response": {"pid": str(r["pid"]), "sid": str(r["sid"])},
                                "sentence_index": i, "source": str(path.relative_to(ROOT)), "N_cell": N, "n_cell": len(take[fl]), "weight": N / max(len(take[fl]), 1)})
                    items.append({"part": "A", "context": ctx, "target": sents[i].text.strip(), "question": QA,
                                  "_judge": {"context": r["text"][:sents[i].start][-2500:], "sentence": sents[i].text, "sentence_stripped": stripped, "next": nxt}})
    return items, key


def part_b(rng, n_per=None, exclude=frozenset(), groups_only=None):
    n_per = n_per or {"stage1": 8, "stage2": 8, "no_marker_runs": 8}
    def hits(rows, src, label):
        out = []
        for r in rows:
            sents = mk.split_sentences(r["text"])
            seen = set()
            for m in re.finditer(r"\bwait\b", r["text"], re.I):
                si = next((j for j, s in enumerate(sents) if s.start <= m.start() < s.start + len(s.text) + 2), None)
                if si is None or si in seen:
                    continue
                seen.add(si)
                prev = sents[si - 1].text if si > 0 else ""
                nxt = sents[si + 1].text if si + 1 < len(sents) else ""
                out.append({"group": label, "source": src, "pid": str(r["pid"]), "sid": str(r["sid"]), "sentence_index": si, "prev": prev.strip(), "sent": sents[si].text.strip(), "next": nxt.strip()})
        return out
    groups = {"stage1": [], "stage2": [], "no_marker_runs": []}
    for br in ("stage1-step141000", "stage1-step707000", "stage1-step1413814"):
        p = RES / "main" / "S1pre" / "gen" / ("S1pre@%s.graded.jsonl.gz" % br)
        groups["stage1"] += hits(read_gz_jsonl(p), str(p.relative_to(ROOT)), "stage1")
    p = RES / "main" / "S1pre" / "gen" / "<EMAIL>"
    groups["stage2"] += hits(read_gz_jsonl(p), str(p.relative_to(ROOT)), "stage2")
    for n in (1, 2, 7):
        ref = Ref(n); idx = ref.idx()
        for step in (min(idx), max(idx)):
            groups["no_marker_runs"] += hits(ref.gen(step), "exp/rl/results/%s/results.tgz::gen/s%d" % (ref.tag, step), "no_marker_runs")
    for ck in ("base", "1566"):
        p = RES / "main" / "S3" / "gen" / ("S3@%s.graded.jsonl.gz" % ck)
        groups["no_marker_runs"] += hits(read_gz_jsonl(p), str(p.relative_to(ROOT)), "no_marker_runs")
    items, key = [], []
    for g, pool in groups.items():
        if groups_only and g not in groups_only:
            continue
        pool = [h for h in pool if (h["source"], h["pid"], h["sid"], h["sentence_index"]) not in exclude]
        take = rng.sample(pool, min(n_per[g], len(pool)))
        for h in take:
            key.append({"part": "B", "group": g, "source": h["source"], "response": {"pid": h["pid"], "sid": h["sid"]}, "sentence_index": h["sentence_index"], "N_group_hits": len(pool),
                        "n_taken": len(take), "note": ("fewer hits available than requested: all taken" if len(pool) < n_per[g] else "")})
            items.append({"part": "B", "context": (h["prev"] + " [[ " + h["sent"] + " ]] " + h["next"]).strip(), "target": h["sent"], "question": QB})
    return items, key


def main():
    rng = random.Random(SEED)
    ia, ka = part_a(rng)
    ib, kb = part_b(rng)
    allitems = list(zip(ia + ib, ka + kb))
    rng.shuffle(allitems)
    todo, key_items, judge_items = [], {}, []
    for n, (it, k) in enumerate(allitems, 1):
        iid = "V3-%02d" % n
        todo.append({"item_id": iid, "part": it["part"], "context": it["context"], "target": it["target"], "question": it["question"]})
        key_items[iid] = k
        if it["part"] == "A":
            judge_items.append({"item_id": iid, **it["_judge"]})
    common.write_jsonl(GOLD / "HUMAN_AUDIT_V3_TODO.jsonl", todo)
    common.write_jsonl(GOLD / ".v3_judge_items.jsonl", judge_items)
    nB = {g: sum(1 for k in key_items.values() if k["part"] == "B" and k["group"] == g) for g in ("stage1", "stage2", "no_marker_runs")}
    common.write_json(GOLD / "HUMAN_AUDIT_V3_key.json", {"seed": SEED, "meta": "exp/gold/HUMAN_AUDIT_V3_meta.json", "items": key_items, "part_B_counts": nB,
                       "note_B": "Qwen2.5-32B base/DAPO/ORZ generations are not local (remote only): not in the no_marker_runs pool; stated per the pre-declared 'if local' clause"})
    print("items", len(todo), "A", sum(k["part"] == "A" for k in key_items.values()), "B counts", nB)
    cells = {}
    for k in key_items.values():
        if k["part"] == "A":
            c = (k["checkpoint"], k["marker_initial"], k["rule_detector_revision"]); cells[c] = cells.get(c, 0) + 1
    print(cells)


if __name__ == "__main__":
    main()
