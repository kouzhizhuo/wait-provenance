"""Judges (Qwen2.5-7B-Instruct letter mode; gpt-oss-20b harmony, effort low) on the 48 part-A items of human audit v3 with revision.md v1.0, exactly as in the
earlier audits (stripped target, 2500-char context, next two sentences). No scoring against labels here.
    CUDA_VISIBLE_DEVICES=6 python -m exp.pipeline.human_audit_v3_judge --items exp/gold/.v3_judge_items.jsonl --qwen <dir> --oss <dir> --out exp/gold/judge_v3.jsonl"""
from __future__ import annotations

import argparse
import time

from . import common, judge as J
from .judge_audit import final_letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--qwen", required=True)
    ap.add_argument("--oss", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    t0 = time.time()
    common.guard_gpus()
    items = common.read_jsonl(a.items)
    system, template, phash, pver = J.load_prompt("revision.md")
    users = [template.replace("{context}", it["context"][-2500:]).replace("{sentence}", it["sentence_stripped"] or it["sentence"]).replace("{next}", it["next"]) for it in items]
    rows = [{"item_id": it["item_id"]} for it in items]
    for tag, path, harmony in (("qwen", a.qwen, False), ("oss", a.oss, True)):
        llm = LLM(path, gpu_memory_utilization=0.85, max_model_len=8192, seed=0, **({} if harmony else {"dtype": "bfloat16"}))
        tok = llm.get_tokenizer()
        kw = {"reasoning_effort": "low"} if harmony else {}
        texts = [tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, **kw) for u in users]
        sp = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False) if harmony else SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outs = llm.generate(texts, sp)
        for r, o in zip(rows, outs):
            raw = o.outputs[0].text
            let = final_letter(raw) if harmony else J.parse_letter(raw)
            r["letter_" + tag] = let
            r["revise_" + tag] = let in ("B", "C")
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    common.write_jsonl(a.out, rows)
    common.write_meta(a.out + ".meta.json", "human_audit_v3_judge", {"prompt": "revision.md", "prompt_version": pver, "prompt_hash": phash, "qwen": a.qwen, "oss": a.oss, "effort": "low"},
                      {"engine_seed": 0}, inputs=[a.items], outputs=[a.out], extra={"n": len(rows)}, t_start=t0)
    print("[v3 judge] %d items, %.0fs" % (len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
