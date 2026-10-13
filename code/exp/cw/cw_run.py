"""Run one (model, benchmark, condition) cell of the Calibrated Wait matrix (D50/D50a) with vLLM + the CW logits processor.
    python -m exp.cw.cw_run --model <dir> --bench <path> --template <t> --mode none|sham|nowait|random|late|soft|cw --tables <cw_tables.json> [--tau x] [--lam x] [--t_late n] [--p_bins json]
        --n N --samples S --max_tokens M --temperature T --top_p P --seed 0 --out <x.jsonl> [--trust]
mode none = vanilla (no processor loaded).  Output rows have the gen.py schema + wall-clock in the .meta.json (cfg + tables_hash + processor config).  Stats of the processor: <out>.cwstats.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from exp.pipeline import common, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--bench"); ap.add_argument("--benches", default=None, help="json list of {name,path,n,samples,cap}: all benchmarks in ONE engine load; outputs <out>.<name>.jsonl"); ap.add_argument("--template", required=True); ap.add_argument("--tag", default="cw")
    ap.add_argument("--mode", default="none"); ap.add_argument("--tables"); ap.add_argument("--tau", type=float, default=0.0); ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--t_late", type=int, default=1 << 30)
    ap.add_argument("--p_bins", default=None, help="json list: random-suppression probability per position bin"); ap.add_argument("--bin_edges", default=None)
    ap.add_argument("--n", type=int, default=500); ap.add_argument("--samples", type=int, default=1); ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sync", action="store_true", help="async_scheduling=False also for vanilla (identical engine config in all conditions)"); ap.add_argument("--gpu_util", type=float, default=0.9); ap.add_argument("--max_num_seqs", type=int, default=256); ap.add_argument("--trust", action="store_true"); ap.add_argument("--bf_cap", type=int, default=0, help="budget forcing: stage-1 cap; unfinished responses get --bf_suffix appended and 80 more tokens"); ap.add_argument("--bf_suffix", default="\n\nAnswer: \\boxed{"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time(); common.guard_gpus()
    cfg = {"mode": a.mode, "tau": a.tau, "lam": a.lam, "t_late": a.t_late, "seed": a.seed, "stats_path": a.out + ".cwstats.json"}
    kw = {}
    tables_hash = None
    if a.mode != "none":
        T = json.load(open(a.tables)); tables_hash = T["tables_hash"]
        cfg.update({k: T[k] for k in ("marker_ids", "boundary_ids", "connector_ids", "digit_end_ids", "dot_ids")})
        if a.p_bins: cfg["p_bins"] = json.loads(a.p_bins)
        if a.bin_edges: cfg["bin_edges"] = json.loads(a.bin_edges)
        cpath = a.out + ".cwconfig.json"; json.dump(cfg, open(cpath, "w")); os.environ["CW_CONFIG"] = cpath
        kw = {"logits_processors": ["exp.cw.cw_proc:CalibratedWaitProcessor"], "async_scheduling": False}
    elif a.sync:
        kw = {"async_scheduling": False}
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, seed=a.seed, max_model_len=(max(b["cap"] for b in json.loads(a.benches)) if a.benches else a.max_tokens) + 2048, dtype="bfloat16", enable_prefix_caching=True, trust_remote_code=a.trust or common.trust_remote(),
              max_num_seqs=a.max_num_seqs, **kw)
    tok = llm.get_tokenizer()
    benches = json.loads(a.benches) if a.benches else [{"name": "main", "path": a.bench, "n": a.n, "samples": a.samples, "cap": a.max_tokens}]
    kid = 0
    for b in benches:
        probs = prompts.load_problems(b["path"], b["n"])
        reqs = [(p, j) for p in probs for j in range(b["samples"])]
        pr = {p["pid"]: prompts.render(a.template, p["problem"], tok, None) for p in probs}
        cap = min(b["cap"], a.bf_cap) if a.bf_cap else b["cap"]
        sps = [SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=cap, extra_args=({"cw_id": kid + k} if a.mode != "none" else None)) for k, _ in enumerate(reqs)]
        kid += len(reqs)
        t1 = time.time()
        outs = llm.generate([pr[p["pid"]] for p, _ in reqs], sps)
        dt = time.time() - t1
        rows = [{"ckpt": a.tag, "pid": p["pid"], "sid": j, "unique_id": p["unique_id"], "seed": a.seed, "text": o.outputs[0].text, "n_tokens": len(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason}
                for (p, j), o in zip(reqs, outs)]
        if a.bf_cap:
            todo = [k for k, r in enumerate(rows) if r["finish"] == "length"]
            suf_n = len(tok.encode(a.bf_suffix, add_special_tokens=False))
            if todo:
                sps2 = [SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=80) for _ in todo]
                outs2 = llm.generate([pr[reqs[k][0]["pid"]] + rows[k]["text"] + a.bf_suffix for k in todo], sps2)
                for k, o2 in zip(todo, outs2):
                    rows[k]["text"] = rows[k]["text"] + a.bf_suffix + o2.outputs[0].text; rows[k]["n_tokens"] += suf_n + len(o2.outputs[0].token_ids); rows[k]["forced"] = True
            dt = time.time() - t1
        fn = a.out if not a.benches else "%s.%s.jsonl" % (a.out[:-6] if a.out.endswith(".jsonl") else a.out, b["name"])
        common.write_jsonl(fn, rows)
        ntok = sum(r["n_tokens"] for r in rows)
        common.write_meta(fn + ".meta.json", "cw_run", {**{k: v for k, v in vars(a).items()}, "bench": b, "tables_hash": tables_hash, "prompt_example": pr[probs[0]["pid"]], "gen_seconds": round(dt, 1), "load_seconds": round(t1 - t0, 1), "gen_tokens": ntok, "tok_per_s": round(ntok / dt, 1),
                          "n_forced": sum(1 for r in rows if r.get("forced"))}, {"engine_seed": a.seed}, inputs=[b["path"]], outputs=[fn], extra={"n_records": len(rows)}, t_start=t0)
        print("[cw_run] %s/%s mode=%s bf_cap=%s: %d samples, %d tokens (%.0f/resp), gen %.0fs (%.0f tok/s)" % (a.tag, b["name"], a.mode, a.bf_cap, len(rows), ntok, ntok / len(rows), dt, ntok / dt))


if __name__ == "__main__":
    main()
