"""Checkpoint sampler: vLLM generation of on-policy traces for one checkpoint.

    export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
    python -m exp.pipeline.gen --model $R/models/S1/1900 --tag S1@1900 --data $R/data/MATH-500/test.jsonl \
        --n 200 --samples 4 --max_tokens 6144 --temperature 1.0 --template olmo_rlzero --seed 0 \
        --out $R/runs/P1/P0/gen/<EMAIL>

Runs ONLY on GPUs 6/7 of hostA (common.guard_gpus). One record per sample; per-request seeds are
derived from (seed, pid, sid) so reruns and different checkpoints see the same random streams.
`--shard i/K` processes problems with pid % K == i (used to split a checkpoint over two GPUs).
"""
from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

from . import common, prompts


def build_llm(model: str, max_model_len: int, gpu_util: float, seed: int, tp: int = 1, kv_dtype: str = "auto"):
    """kv_dtype='fp8' doubles the KV-cache capacity (OLMo-3-7B is MHA: ~110k tokens per GPU in bf16). It must pass the
    numerics check of DECISIONS D28 before use and must then be applied to ALL checkpoints; it is logged in the meta."""
    from vllm import LLM

    return LLM(model, tensor_parallel_size=tp, gpu_memory_utilization=gpu_util, seed=seed, kv_cache_dtype=kv_dtype,
               max_model_len=max_model_len, dtype="bfloat16", enable_prefix_caching=True, trust_remote_code=common.trust_remote(),
               **({"max_num_seqs": int(os.environ["P1_MAX_NUM_SEQS"])} if os.environ.get("P1_MAX_NUM_SEQS") else {}))   # hybrid (Mamba/GDN) models: max_num_seqs must not exceed the cache blocks


def run_requests(llm, prompt_strs, seeds, temperature, top_p, max_tokens, store_logprobs=True, seeded=False):
    """One request per prompt (n=1). Returns vLLM outputs in order.

    seeded=True gives each request its own seed (bit-reproducible per request, but vLLM then samples through per-request
    generators, which cost ~5x throughput at batch 256 on our H800s: measured 1.6k tok/s vs >8k). The default draws from
    the engine RNG only (LLM(seed=...)): distributionally identical, reproducible up to the batch schedule.
    """
    from vllm import SamplingParams

    stop = [x for x in os.environ.get("P1_STOP", "").split(",") if x] or None   # e.g. the SimpleRL-Zoo eval stop words "</s>,<|im_end|>,<|endoftext|>"
    extra = {}
    if os.environ.get("P1_KEEP_SPECIAL") == "1":   # openPangu: keep the [unused16]...[unused17] thinking delimiters in the text
        extra["skip_special_tokens"] = False
    if os.environ.get("P1_STOP_IDS"):
        extra["stop_token_ids"] = [int(x) for x in os.environ["P1_STOP_IDS"].split(",") if x]
    sps = [SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop, **extra,
                          seed=(s if seeded else None), logprobs=0 if store_logprobs else None) for s in seeds]
    return llm.generate(prompt_strs, sps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="checkpoint name, e.g. S1@1900")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200, help="first n problems")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--template", default="olmo_rlzero", choices=prompts.TEMPLATES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.88)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seeded", action="store_true", help="per-request seeds (slow, see run_requests)")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size (cluster: 4 for 32B)")
    ap.add_argument("--effort", default=None, help="gpt-oss reasoning effort: low|medium|high")
    ap.add_argument("--kv_dtype", default="auto", help="KV-cache dtype ('fp8' only after the D28 numerics check)")
    ap.add_argument("--no_tokens", action="store_true", help="do not store token ids / logprobs")
    ap.add_argument("--greedy_n", type=int, default=0, help="also decode greedily (T=0, 1 sample) the first N problems -> <out>.greedy.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    gpus = common.guard_gpus()
    shard_i, shard_k = (int(x) for x in a.shard.split("/"))
    probs = [p for p in prompts.load_problems(a.data, a.n) if p["pid"] % shard_k == shard_i]
    llm = build_llm(a.model, a.max_tokens + 1024, a.gpu_util, a.seed, tp=a.tp, kv_dtype=a.kv_dtype)
    tok = llm.get_tokenizer()
    tmpl_ok = prompts.verify_template(tok) if a.template == "olmo_rlzero" and getattr(tok, "chat_template", None) else None
    prompt_of = {p["pid"]: prompts.render(a.template, p["problem"], tok, a.effort) for p in probs}

    reqs = [(p, j) for p in probs for j in range(a.samples)]
    seeds = [common.derive_seed(a.seed, "gen", p["pid"], j) for p, j in reqs]
    outs = run_requests(llm, [prompt_of[p["pid"]] for p, _ in reqs], seeds, a.temperature, a.top_p, a.max_tokens,
                        store_logprobs=not a.no_tokens, seeded=a.seeded)
    rows = []
    for (p, j), s, o in zip(reqs, seeds, outs):
        c = o.outputs[0]
        r = {"ckpt": a.tag, "pid": p["pid"], "sid": j, "unique_id": p["unique_id"], "seed": s, "text": c.text,
             "n_tokens": len(c.token_ids), "finish": c.finish_reason}
        if not a.no_tokens:
            r["token_ids"] = list(c.token_ids)
            r["logprobs"] = [float(d[t].logprob) for t, d in zip(c.token_ids, c.logprobs)]
        rows.append(r)
    n = common.write_jsonl(a.out, rows)
    if a.greedy_n:  # same model load; decoding-robustness subset (EXPERIMENT_PLAN §7.6)
        gp = [p for p in probs if p["pid"] < a.greedy_n]
        gseeds = [common.derive_seed(a.seed, "greedy", p["pid"]) for p in gp]
        gouts = run_requests(llm, [prompt_of[p["pid"]] for p in gp], gseeds, 0.0, 1.0, a.max_tokens, store_logprobs=False, seeded=a.seeded)
        grows = [{"ckpt": a.tag, "pid": p["pid"], "sid": 0, "unique_id": p["unique_id"], "seed": s_, "text": o.outputs[0].text,
                  "n_tokens": len(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason} for p, s_, o in zip(gp, gseeds, gouts)]
        common.write_jsonl(a.out.replace(".jsonl", ".greedy.jsonl"), grows)
    ntok = sum(r["n_tokens"] for r in rows)
    dt = time.time() - t0
    cfg = {"model": a.model, "tag": a.tag, "data": a.data, "n_problems": a.n, "samples": a.samples,
           "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "template": a.template,
           "shard": a.shard, "kv_cache_dtype": a.kv_dtype, "tp": a.tp, "effort": a.effort, "prompt_example": prompt_of[probs[0]["pid"]] if probs else None}
    model_id = {}
    idx = Path(a.model) / "model.safetensors.index.json"
    if idx.exists():
        model_id["safetensors_index_hash"] = common.file_hash(idx)
    prompt_ids = [list(map(int, tok.encode(prompt_of[q["pid"]]))) for q in probs[:3]]
    cfg["prompts_hash"] = common.git_blob_hash("\n".join(prompt_of[q["pid"]] for q in probs).encode())
    common.write_meta(a.out + ".meta.json", "gen", cfg, {"engine_seed": a.seed, "sampling_seed_mode": "per-request derive_seed(seed,'gen',pid,sid)" if a.seeded else "engine RNG only (reproducible up to batch schedule)"},
                      inputs=[a.data], outputs=[a.out],
                      extra={"n_records": n, "gen_tokens": ntok, "tok_per_s": round(ntok / dt, 1),
                             "template_matches_tokenizer": tmpl_ok, "gpus": gpus, "model_id": model_id,
                             "prompt_token_ids_first3": prompt_ids, "gpu_hours": round(dt * max(1, len(gpus)) / 3600, 3)}, t_start=t0)
    print("[gen] %s: %d samples, %d tokens in %.0fs (%.0f tok/s incl. load); template_ok=%s" % (a.tag, n, ntok, dt, ntok / dt, tmpl_ok))


if __name__ == "__main__":
    main()
