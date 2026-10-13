"""gpt-oss-20b (harmony template, reasoning effort low, greedy) on the human-audit v1 items with the revision prompt v1.0.

    python -m exp.pipeline.judge_audit --model <gpt-oss-20b dir> --gold exp/gold --out judge_audit_gptoss.jsonl
Same items and same prompt text as the Qwen judge (stripped target sentence + full context + next sentences); the answer letter is read
from the final channel. GPU 6/7 only (vLLM, env gen)."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from . import common, judge as J


def final_letter(raw: str) -> str:
    """Letter after the last '<|channel|>final<|message|>' (special tokens kept); fallback: last standalone A-E."""
    m = list(re.finditer(r"<\|channel\|>final<\|message\|>(.*)", raw, re.S))
    tail = m[-1].group(1) if m else raw[-200:]
    ls = re.findall(r"\b([A-E])\b", tail)
    return ls[-1] if ls else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--mode", default="harmony", choices=["harmony", "letter"], help="harmony = gpt-oss reasoning model (final-channel letter); letter = one-token answer of a non-reasoning chat model (Qwen3-32B: thinking off)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    t0 = time.time()
    gpus = common.guard_gpus()
    g = Path(a.gold)
    meta = json.load(open(g / "HUMAN_AUDIT_meta.json"))
    ids = [r["item_id"] for r in meta["id_map"]]
    pool = {}
    for f in ("s1_sample.jsonl", "s1_sample_lead_extra.jsonl", "s2_sample.jsonl", "s3_sample.jsonl"):
        if (g / f).exists():
            pool.update({r["item_id"]: r for r in common.read_jsonl(g / f)})
    system, template, phash, pver = J.load_prompt("revision.md")
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, seed=0, tensor_parallel_size=a.tp)
    tok = llm.get_tokenizer()
    texts = []
    for i in ids:
        it = pool[i]
        user = template.replace("{context}", it["context"][-2500:]).replace("{sentence}", it["sentence_stripped"] or it["sentence"]).replace("{next}", it["next"])
        kw = {"reasoning_effort": a.effort} if a.mode == "harmony" else {"enable_thinking": False}
        texts.append(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False,
                                             add_generation_prompt=True, **kw))
    outs = llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=(a.max_tokens if a.mode == "harmony" else 4), skip_special_tokens=(a.mode == "letter")))
    rows = []
    for i, o in zip(ids, outs):
        raw = o.outputs[0].text
        L = final_letter(raw) if a.mode == "harmony" else J.parse_letter(raw)
        rows.append({"item_id": i, "judge_letter": L, "judge_cat": J.CATS.get(L, "invalid"), "raw_tail": raw[-160:], "n_tokens": len(o.outputs[0].token_ids),
                     "finish": o.outputs[0].finish_reason, "prompt_version": pver, "prompt_hash": phash, "judge_model": a.model, "effort": a.effort})
    common.write_jsonl(a.out, rows)
    common.write_meta(a.out + ".meta.json", "judge_audit", {"model": a.model, "effort": a.effort, "n": len(rows), "prompt_version": pver}, {"engine_seed": 0},
                      inputs=[str(g / "HUMAN_AUDIT_meta.json")], outputs=[a.out], extra={"gpus": gpus, "invalid": sum(r["judge_cat"] == "invalid" for r in rows),
                      "truncated": sum(r["finish"] == "length" for r in rows)}, t_start=t0)
    print("[judge_audit] %d items in %.0fs (invalid: %d)" % (len(rows), time.time() - t0, sum(r["judge_cat"] == "invalid" for r in rows)))


if __name__ == "__main__":
    main()
