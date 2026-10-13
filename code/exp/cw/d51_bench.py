"""D51 feasibility benchmark (one 7-8B model, one GPU).  Modes:
  vllm       plain vLLM (cudagraphs, no processor) - the D50 implementation's reference throughput
  vllm_hook  vLLM enforce_eager + forward hook on layer L (probe score S per request) + the CW processor in sham mode (the D51 implementation)
  hf         HF transformers generate with a forward hook on layer L (batch B), the fallback implementation
  check      correctness of the hook: 1 request, greedy, 64 tokens; probe score of the last step vs the HF recomputation on the same sequence (should agree to bf16 noise)
    python -m exp.cw.d51_bench --mode vllm|vllm_hook|hf|check --model <dir> --tables <cw_tables.json> --n 128 --max_tokens 4096 --layer 24 --bench <aime24 parquet> """
import argparse
import json
import os
import time

import torch


def prompts_(tok, bench, n, template):
    from exp.pipeline import prompts
    ps = prompts.load_problems(bench, 30)
    return [prompts.render(template, ps[i % len(ps)]["problem"], tok, None) for i in range(n)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tables"); ap.add_argument("--bench", required=True)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--max_tokens", type=int, default=4096); ap.add_argument("--layer", type=int, default=24); ap.add_argument("--template", default="qwen_boxed")
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--every", default="50"); ap.add_argument("--gpu_util", type=float, default=0.88); ap.add_argument("--out", required=True)
    a = ap.parse_args(); res = {"mode": a.mode, "model": a.model, "n": a.n, "max_tokens": a.max_tokens, "layer": a.layer}
    if a.mode in ("vllm", "vllm_hook", "check"):
        from vllm import LLM, SamplingParams
        kw = {}
        if a.mode != "vllm":
            T = json.load(open(a.tables)); cfg = {"mode": "sham", "seed": 0, "stats_path": a.out + ".stats.json", **{k: T[k] for k in ("marker_ids", "boundary_ids", "connector_ids", "digit_end_ids", "dot_ids")}}
            json.dump(cfg, open(a.out + ".cfg.json", "w")); os.environ["CW_CONFIG"] = a.out + ".cfg.json"; os.environ["CW_PROBE_LOG"] = a.out + ".probelog.jsonl"
            kw = {"logits_processors": ["exp.cw.cw_proc:CalibratedWaitProcessor"], "async_scheduling": False, "enforce_eager": True, "worker_extension_cls": "exp.cw.probe_hook.WorkerExt"}
        llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=a.max_tokens + 2048, dtype="bfloat16", enable_prefix_caching=True, max_num_seqs=256, **kw)
        tok = llm.get_tokenizer()
        if a.mode != "vllm":
            from exp.cw import probe_hook
            import numpy as np
            D = llm.llm_engine.model_config.hf_config.hidden_size
            wl = (np.random.RandomState(0).randn(D) / np.sqrt(D)).tolist() if a.mode == "check" else None
            res["install"] = llm.collective_rpc("install_probe", args=(a.layer, wl, 0.0))
        P = prompts_(tok, a.bench, a.n if a.mode != "check" else 1, a.template)
        if a.mode == "check":
            sp = SamplingParams(temperature=0.0, max_tokens=64, extra_args={"cw_id": 0})
            o = llm.generate(P, sp)[0].outputs[0]; res["text_head"] = o.text[:120]; res["ids"] = list(o.token_ids)
            res["probe_log"] = [json.loads(l) for l in open(a.out + ".probelog.jsonl")][-3:] if os.path.exists(a.out + ".probelog.jsonl") else None
            res["prompt"] = P[0]
        else:
            sps = [SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, extra_args=({"cw_id": i} if a.mode == "vllm_hook" else None)) for i in range(len(P))]
            t0 = time.time(); outs = llm.generate(P, sps, use_tqdm=False); dt = time.time() - t0
            nt = sum(len(o.outputs[0].token_ids) for o in outs); res.update({"tokens": nt, "seconds": round(dt, 1), "tok_per_s": round(nt / dt, 1)})
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        sink = {}
        model.model.layers[a.layer].register_forward_hook(lambda m, i, o: sink.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].float()))
        P = prompts_(tok, a.bench, a.n, a.template); nt = 0; t0 = time.time()
        for b in range(0, len(P), a.batch):
            enc = tok(P[b:b + a.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_tokens, do_sample=True, temperature=0.6, top_p=0.95)
            nt += int((out[:, enc["input_ids"].shape[1]:] != tok.pad_token_id).sum())
        dt = time.time() - t0; res.update({"tokens": nt, "seconds": round(dt, 1), "tok_per_s": round(nt / dt, 1), "batch": a.batch, "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)})
    json.dump(res, open(a.out, "w"), indent=1, default=str); print(json.dumps({k: v for k, v in res.items() if k not in ("prompt", "ids")}, default=str)[:700])


if __name__ == "__main__":
    main()
