"""Detector check for VALUE_CHANGE (D40): 50 random VALUE_CHANGE=1 windows and 50 random judge-REVISE=1 windows, BLIND (no labels, no arm, no
detector/judge flags in the item file), rated by humans/Claude and by an open-weight co-rater (Qwen3-32B). The rules are frozen; this is a report, not tuning.

    python -m exp.pipeline.valchange_check draw --windows windows.jsonl.gz --prefixes P.jsonl [P2.jsonl] --out_items exp/gold/valchange_check.jsonl --out_key exp/gold/valchange_check_KEY.jsonl
    python -m exp.pipeline.valchange_check rate --items exp/gold/valchange_check.jsonl --model <Qwen3-32B dir> --out exp/gold/valchange_check_qwen3.jsonl     (GPU, env gen)
Question (identical for every item, wording below): did the continuation change or contradict the value established in the last step?
"""
from __future__ import annotations

import argparse
import random
import re
import time

from . import common, markers as mk

QUESTION = ("A student is solving a math problem. Below are the END OF THE SOLUTION SO FAR and the NEXT FEW SENTENCES the student wrote.\n\n"
            "SOLUTION SO FAR (last part):\n{prefix_tail}\n\nNEXT SENTENCES:\n{window}\n\n"
            "Question: Do the next sentences CHANGE or CONTRADICT a value, result or claim that the last step of the solution so far established "
            "(i.e. does the student now assign a different value to the same quantity, or say that the last step was wrong)? "
            "Re-checking or restating the same value, continuing to a new step, or introducing a different quantity does NOT count.\n"
            "Answer with one word: YES or NO.")


def draw(a):
    rows = common.read_jsonl(a.windows)
    pre = {p["prefix_id"]: p for f in a.prefixes for p in common.read_jsonl(f)}
    rng = random.Random(20261012)
    vc = [r for r in rows if r["value_change"]]
    rv = [r for r in rows if r["judge_revise"]]
    items, key = [], []
    for name, pool in (("VC1", vc), ("REV1", rv)):
        for r in rng.sample(pool, min(a.n, len(pool))):
            key.append({"sample": name, "policy": r["policy"], "prefix_id": r["prefix_id"], "arm": r["arm"], "j": r["j"], "value_change": r["value_change"], "judge_revise": r["judge_revise"]})
    order = list(range(len(key)))
    rng.shuffle(order)
    for new_id, i in enumerate(order):
        k = key[i]
        sents = [s.text for s in mk.split_sentences(pre[k["prefix_id"]]["prefix"])]
        items.append({"id": "VC%03d" % new_id, "prefix_tail": " ".join(sents[-4:])[-900:], "window": " ".join(next(r for r in rows if (r["policy"], r["prefix_id"], r["arm"], r["j"]) == (k["policy"], k["prefix_id"], k["arm"], k["j"]))["units"])})
        k["id"] = "VC%03d" % new_id
    common.write_jsonl(a.out_items, items)
    common.write_jsonl(a.out_key, sorted(key, key=lambda k: k["id"]))
    print("[valchange_check] %d items (blind) -> %s ; key -> %s" % (len(items), a.out_items, a.out_key))


def rate(a):
    from vllm import LLM, SamplingParams

    t0 = time.time()
    common.guard_gpus()
    items = common.read_jsonl(a.items)
    llm = LLM(a.model, gpu_memory_utilization=0.9, max_model_len=8192, seed=0, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": QUESTION.format(prefix_tail=it["prefix_tail"], window=it["window"])}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for it in items]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4, logprobs=10))
    res = []
    for it, o in zip(items, outs):
        t = o.outputs[0].text.strip().upper()
        res.append({"id": it["id"], "answer": "YES" if t.startswith("YES") else "NO" if t.startswith("NO") else "?", "raw": o.outputs[0].text})
    common.write_jsonl(a.out, res)
    common.write_meta(a.out + ".meta.json", "valchange_check_rate", {"model": a.model, "question": QUESTION, "temperature": 0.0, "thinking": False}, {"engine_seed": 0},
                      inputs=[a.items], outputs=[a.out], extra={"date": time.strftime("%Y-%m-%d"), "n": len(res)}, t_start=t0)
    print("[valchange_check] rated %d in %.0fs" % (len(res), time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("draw"); d.add_argument("--windows", required=True); d.add_argument("--prefixes", nargs="+", required=True)
    d.add_argument("--out_items", required=True); d.add_argument("--out_key", required=True); d.add_argument("--n", type=int, default=50)
    r = sp.add_parser("rate"); r.add_argument("--items", required=True); r.add_argument("--model", required=True); r.add_argument("--out", required=True)
    a = ap.parse_args()
    draw(a) if a.cmd == "draw" else rate(a)
