"""CGRS baseline (Huang et al. 2025, arXiv 2508.05337, Alg. 1), reimplemented on vLLM for the dense primary models (D50d).
At every '\\n\\n' checkpoint: probe a tentative answer (append '**Final Answer: \\boxed', greedy, <= 12 tokens), certainty C = 1 - mean_i H(p_{a_i}) / log|V|, suppression probability p = max(0, (C - delta)/(1 - delta)), delta = 0.9 (paper, Sec. 4.1);
until the next checkpoint every sampling step masks the trigger tokens (Wait/But/Alternatively/Hmm variants) with probability p (processor mode 'cgrs').  T 0.6 / top-p 0.95 as in the paper (we use each model's own decode).
DEVIATIONS (logged in the meta): (i) token entropies in the probe are computed from the top-20 logprobs plus the residual mass spread uniformly over the other |V|-20 tokens (vLLM returns top-k only); (ii) the probe stops at 12 tokens;
(iii) segments are generated in synchronous rounds; response length counts the main trajectory only, probe tokens are reported separately and ALL wall-clock is charged.
    python -m exp.cw.cgrs_run --model <dir> --benches '[{name,path,n,samples,cap}]' --template <t> --tables <cw_tables.json> --out <prefix> --temperature .6 --top_p .95 [--delta .9] [--trust]"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

from exp.pipeline import common, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--benches", required=True); ap.add_argument("--template", required=True); ap.add_argument("--tables", required=True)
    ap.add_argument("--delta", type=float, default=0.9); ap.add_argument("--seg_cap", type=int, default=1024); ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu_util", type=float, default=0.9); ap.add_argument("--max_num_seqs", type=int, default=256); ap.add_argument("--trust", action="store_true")
    ap.add_argument("--tag", default="cgrs"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time(); common.guard_gpus()
    T = json.load(open(a.tables)); C = json.load(open(a.tables.replace(".json", ".cgrs.json")))
    cfg = {"mode": "cgrs", "seed": a.seed, "cgrs_ids": C["cgrs_ids"], **{k: T[k] for k in ("marker_ids", "boundary_ids", "connector_ids", "digit_end_ids", "dot_ids")}}
    cp = a.out + ".cwconfig.json"; json.dump(cfg, open(cp, "w")); os.environ["CW_CONFIG"] = cp
    from vllm import LLM, SamplingParams
    benches = json.loads(a.benches); maxcap = max(b["cap"] for b in benches)
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, seed=a.seed, max_model_len=maxcap + 1536, dtype="bfloat16", enable_prefix_caching=True, trust_remote_code=a.trust or common.trust_remote(),
              max_num_seqs=a.max_num_seqs, logits_processors=["exp.cw.cw_proc:CalibratedWaitProcessor"], async_scheduling=False, max_logprobs=20)
    tok = llm.get_tokenizer(); V = llm.llm_engine.model_config.get_vocab_size() if hasattr(llm.llm_engine, "model_config") else len(tok); logV = math.log(V)
    for b in benches:
        probs = prompts.load_problems(b["path"], b["n"])
        reqs = [(p, j) for p in probs for j in range(b["samples"])]
        pr = {p["pid"]: prompts.render(a.template, p["problem"], tok, None) for p in probs}
        N = len(reqs); text = [""] * N; ntok = [0] * N; pp = [0.0] * N; done = [False] * N; fin = ["length"] * N; probes = [0] * N; ptok = [0] * N; kid = 0
        t1 = time.time(); rounds = 0
        while not all(done):
            act = [i for i in range(N) if not done[i]]; rounds += 1
            sps = []
            for i in act:
                sps.append(SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=max(1, min(a.seg_cap, b["cap"] - ntok[i])), stop=["\n\n"], include_stop_str_in_output=True, extra_args={"cw_id": kid, "p": pp[i]})); kid += 1
            outs = llm.generate([pr[reqs[i][0]["pid"]] + text[i] for i in act], sps, use_tqdm=False)
            chk = []
            for i, o in zip(act, outs):
                c = o.outputs[0]; text[i] += c.text; ntok[i] += len(c.token_ids)
                if c.finish_reason == "stop" and c.stop_reason is None:
                    done[i] = True; fin[i] = "stop"
                elif ntok[i] >= b["cap"]:
                    done[i] = True; fin[i] = "length"
                elif c.stop_reason == "\n\n":
                    chk.append(i)
            if chk:
                psp = SamplingParams(n=1, temperature=0.0, max_tokens=12, logprobs=20, stop=["}"])
                pouts = llm.generate([pr[reqs[i][0]["pid"]] + text[i] + "**Final Answer: \\boxed" for i in chk], [psp] * len(chk), use_tqdm=False)
                for i, o in zip(chk, pouts):
                    c = o.outputs[0]; H = []
                    for d in c.logprobs or []:
                        ps = [math.exp(v.logprob) for v in d.values()]; top = -sum(p * math.log(p) for p in ps if p > 0); r = max(1.0 - sum(ps), 1e-12)
                        H.append(top - r * math.log(r / max(V - len(ps), 1)))
                    Cc = 1.0 - (sum(H) / len(H)) / logV if H else 0.0
                    pp[i] = max(0.0, (Cc - a.delta) / (1.0 - a.delta)); probes[i] += 1; ptok[i] += len(c.token_ids)
        dt = time.time() - t1
        rows = [{"ckpt": a.tag, "pid": p["pid"], "sid": j, "unique_id": p["unique_id"], "seed": a.seed, "text": text[k], "n_tokens": ntok[k], "finish": fin[k], "n_probes": probes[k], "probe_tokens": ptok[k]} for k, (p, j) in enumerate(reqs)]
        fn = "%s.%s.jsonl" % (a.out, b["name"]); common.write_jsonl(fn, rows)
        common.write_meta(fn + ".meta.json", "cgrs_run", {**vars(a), "bench": b, "gen_seconds": round(dt, 1), "rounds": rounds, "probe_tokens": sum(ptok), "main_tokens": sum(ntok), "n_probes": sum(probes), "V": V,
                          "deviations": "top-20 entropy + uniform residual; probe 12 tokens; synchronous rounds; probe tokens not in response length; wall-clock includes probes", "tables_hash": T["tables_hash"], "prompt_example": pr[probs[0]["pid"]]},
                          {"engine_seed": a.seed}, inputs=[b["path"]], outputs=[fn], extra={"n_records": len(rows)}, t_start=t0)
        print("[cgrs_run] %s/%s: %d samples, %d main tokens (%.0f/resp), %d probes (%d tok), %d rounds, %.0fs" % (a.tag, b["name"], N, sum(ntok), sum(ntok) / N, sum(probes), sum(ptok), rounds, dt))


if __name__ == "__main__":
    main()
