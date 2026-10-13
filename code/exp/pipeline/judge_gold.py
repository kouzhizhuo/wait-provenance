"""Run the revision judge (revision.md v1.0, Qwen2.5-7B-Instruct) on the gold-label sentences, on CPU (one forward pass, first-token
logits over A-E), so the GPUs stay with the experiment queue.

    python -m exp.pipeline.judge_gold --model <dir> --items exp/gold/s1_sample.jsonl exp/gold/s1_sample_lead_extra.jsonl --out judge_gold_s1.jsonl
The judge sees the STRIPPED target sentence (sentence-initial discourse forms removed, D15); labellers saw the unstripped one.
"""
from __future__ import annotations

import argparse
import math
import time

from . import common, judge as J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=96)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch.set_num_threads(a.threads)
    system, template, phash, pver = J.load_prompt("revision.md")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).eval()
    letter_ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in J.LETTERS}
    items = [it for f in a.items for it in common.read_jsonl(f)]
    out = []
    with torch.no_grad():
        for it in items:
            user = (template.replace("{context}", it["context"][-2500:]).replace("{sentence}", it["sentence_stripped"] or it["sentence"])
                    .replace("{next}", it["next"]))
            text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids
            lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)
            lps = {c: float(lp[i]) for c, i in letter_ids.items()}
            letter = max(lps, key=lps.get)
            z = sum(math.exp(v) for v in lps.values())
            out.append({"item_id": it["item_id"], "judge_letter": letter, "judge_cat": J.CATS[letter], "judge_conf": math.exp(lps[letter]) / z,
                        "letter_logprobs": lps, "prompt_file": "revision.md", "prompt_version": pver, "prompt_hash": phash, "device": "cpu-bf16"})
    common.write_jsonl(a.out, out)
    common.write_meta(a.out + ".meta.json", "judge_gold", {"model": a.model, "items": a.items, "prompt_version": pver}, {}, inputs=a.items, outputs=[a.out],
                      extra={"n": len(out)}, t_start=t0)
    print("[judge_gold] %d items in %.0fs" % (len(out), time.time() - t0))


if __name__ == "__main__":
    main()
