"""D43(a): fresh OLMo-3-7B-RL-Zero-Math traces on MATH TRAIN L3-5 (the REF GRPO pool, data/math_verl/train.parquet; never MATH-500).
Prompt = P1's `olmo_rlzero` template (exp.pipeline.prompts.render; checked against the model's own chat template); T=1.0, top_p=1, 1 sample/problem.
One process per GPU:  CUDA_VISIBLE_DEVICES=g python gen_seed_traces.py --model M --shard i/K --out raw_i.jsonl   (env gen)"""
import argparse, json, os, sys, time
from pathlib import Path
R = os.environ.get("ARR_ROOT", "/<CLUSTER>/<USER>/arr_oct")
sys.path.insert(0, R + "/code/P1")
import pandas as pd
from exp.pipeline import prompts
from exp.pipeline.grade import extract_answer

INST = "\nPlease reason step by step, and put your final answer within \\boxed{}."   # prep_math_verl.py default PROMPT_INST

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", default=R + "/data/math_verl/train.parquet")
ap.add_argument("--shard", default="0/1"); ap.add_argument("--samples", type=int, default=1)
ap.add_argument("--max_tokens", type=int, default=3072); ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--sid_offset", type=int, default=0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out", required=True)
a = ap.parse_args()
si, sk = (int(x) for x in a.shard.split("/"))
df = pd.read_parquet(a.data)
rows = []
for pid, r in enumerate(df.to_dict("records")):
    if pid % sk != si or (a.limit and pid >= a.limit): continue
    c = r["prompt"][0]["content"]; assert c.endswith(INST), pid
    rows.append({"pid": pid, "problem": c[:-len(INST)], "prompt_messages": [dict(m) for m in r["prompt"]],
                 "gold": r["reward_model"]["ground_truth"], "level": r["extra_info"].get("level"), "type": r["extra_info"].get("type")})
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(a.model, gpu_memory_utilization=0.88, seed=a.seed + si, max_model_len=a.max_tokens + 1024, dtype="bfloat16", enable_prefix_caching=True)
tok = llm.get_tokenizer()
tmpl_ok = prompts.verify_template(tok) if getattr(tok, "chat_template", None) else None
assert tmpl_ok in (True, None), "olmo_rlzero string != the model chat template"
outs = llm.generate([prompts.render("olmo_rlzero", r["problem"]) for r in rows for _ in range(a.samples)],
                    SamplingParams(n=1, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens))
k = 0; ntok = 0
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        for j in range(a.samples):
            c = outs[k].outputs[0]; k += 1; ntok += len(c.token_ids)
            f.write(json.dumps({**r, "sid": j + a.sid_offset, "text": c.text, "n_tokens_olmo": len(c.token_ids), "finish": c.finish_reason}, ensure_ascii=False) + "\n")
json.dump({"model": a.model, "template": "olmo_rlzero", "template_matches_tokenizer": tmpl_ok, "temperature": a.temperature, "top_p": 1.0,
           "max_tokens": a.max_tokens, "samples": a.samples, "shard": a.shard, "n_records": k, "gen_tokens": ntok, "seconds": round(time.time() - t0),
           "data": a.data}, open(a.out + ".meta.json", "w"), indent=1)
print("[gen_seed] shard %s: %d samples, %d tokens, %.0fs" % (a.shard, k, ntok, time.time() - t0))
