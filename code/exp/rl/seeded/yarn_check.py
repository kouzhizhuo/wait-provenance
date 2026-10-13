"""Verify a (YaRN-)scaled checkpoint dir: vLLM sampled-token logprobs vs HF teacher-forced logprobs (D46a: mean |dlogp| must be < 0.03).
  python yarn_check.py gen  --model DIR --out x.jsonl --n 24 --max_tokens 6000     (vLLM; env rl = the verl rollout stack)
  python yarn_check.py hf   --model DIR --gen x.jsonl --out x.json                 (HF transformers; env rl = the verl actor stack)"""
import argparse, json, os, sys, time
R = os.environ.get("ARR_ROOT", "/<CLUSTER>/<USER>/arr_oct"); sys.path.insert(0, R + "/code/P1")
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["gen", "hf"]); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--gen"); ap.add_argument("--n", type=int, default=24); ap.add_argument("--max_tokens", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
from exp.pipeline import prompts
if a.mode == "gen":
    from vllm import LLM, SamplingParams
    probs = prompts.load_problems(R + "/data/MATH-500/test.jsonl", a.n)
    llm = LLM(a.model, gpu_memory_utilization=0.6, seed=a.seed, max_model_len=a.max_tokens + 1024, dtype="bfloat16", enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    ps = [prompts.render("qwen_boxed", p["problem"], tok) for p in probs]
    outs = llm.generate(ps, SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, logprobs=0))
    with open(a.out, "w") as f:
        for p, pr, o in zip(probs, ps, outs):
            c = o.outputs[0]
            f.write(json.dumps({"pid": p["pid"], "prompt": pr, "prompt_ids": list(o.prompt_token_ids), "token_ids": list(c.token_ids),
                                "logprobs": [float(d[t].logprob) for t, d in zip(c.token_ids, c.logprobs)], "finish": c.finish_reason}) + "\n")
    print("[yarn_check gen] done", a.out)
else:
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    print("[hf] rope:", getattr(m.config, "rope_parameters", None) or getattr(m.config, "rope_scaling", None), "max_pos", m.config.max_position_embeddings)
    rows = [json.loads(l) for l in open(a.gen)]
    per = []; late_d = []; early_d = []
    for r in rows:
        ids = torch.tensor([r["prompt_ids"] + r["token_ids"]]).cuda(); P = len(r["prompt_ids"])
        with torch.no_grad():
            lg = m(ids).logits[0, P - 1:-1].float()
        lp = torch.log_softmax(lg, -1).gather(1, ids[0, P:].unsqueeze(1)).squeeze(1).cpu()
        d = (lp - torch.tensor(r["logprobs"])).abs()
        pos = torch.arange(len(d)) + P
        per.append({"pid": r["pid"], "n_resp": len(d), "mean_abs_dlogp": float(d.mean()), "max_pos": int(pos[-1]),
                    "mean_abs_dlogp_pos_gt4096": float(d[pos >= 4096].mean()) if (pos >= 4096).any() else None})
        early_d.append(d[pos < 4096]); late_d.append(d[pos >= 4096])
    e = torch.cat(early_d); l = torch.cat(late_d)
    res = {"model": a.model, "n_seqs": len(rows), "mean_abs_dlogp_all": float(torch.cat([e, l]).mean()), "mean_abs_dlogp_pos_lt4096": float(e.mean()),
           "mean_abs_dlogp_pos_ge4096": float(l.mean()) if len(l) else None, "n_tokens_ge4096": int(len(l)), "per_seq": per, "pass_lt_0.03": bool(float(torch.cat([e, l]).mean()) < 0.03)}
    json.dump(res, open(a.out, "w"), indent=1)
    print("[yarn_check hf] mean|dlogp| all %.4f  <4096 %.4f  >=4096 %s (n=%d) -> %s" % (res["mean_abs_dlogp_all"], res["mean_abs_dlogp_pos_lt4096"],
          res["mean_abs_dlogp_pos_ge4096"], res["n_tokens_ge4096"], "PASS" if res["pass_lt_0.03"] else "FAIL"))
