"""D51 gate (i): VALUE-level check of the hooked hidden state / probe score.  Step A (vLLM, batch size 1, greedy, 20 prompts x 48 tokens, hook on layer L with a fixed random w): log S for every decoding step.
Step B (HF, same weights): recompute S at the same positions from output_hidden_states (layer OUTPUT, hidden_states[L+1]) and compare.  Relative error = |S_vllm - S_hf| / max(|S_hf|, 1e-6) and also on the pre-sigmoid logit.
    python -m exp.cw.d51_check2 --stage A|B --model <dir> --tables T.json --bench B --layer L --out DIR [--gpu_util 0.05]"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--model", required=True); ap.add_argument("--tables"); ap.add_argument("--bench", required=True)
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--gpu_util", type=float, default=0.05); ap.add_argument("--template", default="qwen_boxed"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--dtype", default="bfloat16"); ap.add_argument("--new", type=int, default=48)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from exp.pipeline import prompts
    if a.stage == "A":
        T = json.load(open(a.tables)); cfg = {"mode": "sham", "seed": 0, **{k: T[k] for k in ("marker_ids", "boundary_ids", "connector_ids", "digit_end_ids", "dot_ids")}}
        json.dump(cfg, open(a.out + "/cfg.json", "w")); os.environ["CW_CONFIG"] = a.out + "/cfg.json"; os.environ["CW_PROBE_LOG"] = a.out + "/probelog.jsonl"; os.environ["CW_PROBE_EVERY"] = "1"
        open(a.out + "/probelog.jsonl", "w").close()
        from vllm import LLM, SamplingParams
        llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=2048, dtype=a.dtype, enable_prefix_caching=False, max_num_seqs=1, enforce_eager=True,
                  logits_processors=["exp.cw.cw_proc:CalibratedWaitProcessor"], async_scheduling=False, worker_extension_cls="exp.cw.probe_hook.WorkerExt")
        tok = llm.get_tokenizer(); D = llm.llm_engine.model_config.hf_config.hidden_size
        w = (np.random.RandomState(0).randn(D) / np.sqrt(D)).tolist(); llm.collective_rpc("install_probe", args=(a.layer, w, 0.0))
        ps = prompts.load_problems(a.bench, a.n); P = [prompts.render(a.template, p["problem"], tok, None) for p in ps]
        res = []
        for i, p in enumerate(P):
            o = llm.generate([p], SamplingParams(temperature=0.0, max_tokens=a.new, extra_args={"cw_id": i}), use_tqdm=False)[0].outputs[0]
            res.append({"i": i, "prompt": p, "ids": list(o.token_ids)})
        json.dump(res, open(a.out + "/seqs.json", "w"))
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype), device_map="cuda").eval()
        seqs = json.load(open(a.out + "/seqs.json")); log = [json.loads(l) for l in open(a.out + "/probelog.jsonl")]
        D = model.config.hidden_size; w = torch.tensor(np.random.RandomState(0).randn(D) / np.sqrt(D), dtype=torch.float32, device="cuda")
        H = {}; rel, absd, rel_logit = [], [], []; zv_all, zh_all = [], []
        for s in seqs:
            ids_p = tok(s["prompt"], add_special_tokens=False)["input_ids"]; ids = torch.tensor([ids_p + s["ids"]], device="cuda")
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states[a.layer + 1][0].float()
            z = (hs @ w); S = torch.sigmoid(z).cpu().numpy(); z = z.cpu().numpy()
            for r in log:
                if r["cw_id0"] == s["i"]:
                    pos = len(ids_p) + r["n_out0"] - 1
                    if 0 <= pos < len(S):
                        v = r["S_head"][0]; rel.append(abs(v - S[pos]) / max(abs(S[pos]), 1e-6)); absd.append(abs(v - S[pos]))
                        zv = float(np.log(max(v, 1e-9) / max(1 - v, 1e-9))); zv_all.append(zv); zh_all.append(float(z[pos])); rel_logit.append(abs(zv - z[pos]) / max(abs(z[pos]), 1e-3))
        out = {"n_compared": len(rel), "max_rel_S": float(max(rel)), "mean_rel_S": float(np.mean(rel)), "max_abs_S": float(max(absd)), "max_rel_logit": float(max(rel_logit)), "median_rel_logit": float(np.median(rel_logit)), "n_sequences": len(seqs), "pearson_logit": float(np.corrcoef(zv_all, zh_all)[0, 1]), "rmse_logit_over_std": float(np.sqrt(np.mean((np.array(zv_all) - np.array(zh_all)) ** 2)) / np.std(zh_all)), "frac_abs_S_below_0.01": float(np.mean(np.array(absd) < 0.01)), "p95_abs_S": float(np.percentile(absd, 95)), "std_logit": float(np.std(zh_all)), "max_abs_logit_over_std": float(np.max(np.abs(np.array(zv_all) - np.array(zh_all))) / np.std(zh_all)), "median_abs_logit_over_std": float(np.median(np.abs(np.array(zv_all) - np.array(zh_all))) / np.std(zh_all))}
        json.dump(out, open(a.out + "/result.json", "w")); print(json.dumps(out))


if __name__ == "__main__":
    main()
