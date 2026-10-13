"""HF-transformers generation fallback with the SAME output schema as gen.py (rows: ckpt, pid, sid, unique_id, seed, text, n_tokens, finish) for architectures vLLM cannot run
(openPangu custom code).  Batched, left-padded, sampling with temperature/top_p; finish='stop' iff an EOS id was produced, else 'length'.
    python -m exp.pipeline.gen_hf --model <dir> --tag <t> --data <MATH-500> --n 200 --samples 4 --max_tokens 8192 --temperature 1.0 --top_p 0.8 --template pangu_slow --eos 45892 --out <x.jsonl>
"""
from __future__ import annotations

import argparse
import time

from . import common, prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=200); ap.add_argument("--samples", type=int, default=4); ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_p", type=float, default=1.0); ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--template", default="pangu_slow", choices=prompts.TEMPLATES); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--eos", type=int, nargs="*", default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time(); common.guard_gpus()
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True).eval()
    eos = a.eos or ([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int) else list(model.generation_config.eos_token_id))
    probs = prompts.load_problems(a.data, a.n)
    reqs = [(p, j) for p in probs for j in range(a.samples)]
    texts = {p["pid"]: prompts.render(a.template, p["problem"], tok, None) for p in probs}
    order = sorted(range(len(reqs)), key=lambda i: len(texts[reqs[i][0]["pid"]]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    tok.padding_side = "left"
    rows = [None] * len(reqs)
    for b in range(0, len(order), a.batch):
        idx = order[b:b + a.batch]
        seeds = [common.derive_seed(a.seed, "gen", reqs[i][0]["pid"], reqs[i][1]) for i in idx]
        torch.manual_seed(common.derive_seed(a.seed, "hfbatch", b))
        enc = tok([texts[reqs[i][0]["pid"]] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_tokens, do_sample=a.temperature > 0, temperature=a.temperature if a.temperature > 0 else None, top_p=a.top_p if a.temperature > 0 else None,
                                 top_k=a.top_k, eos_token_id=eos, pad_token_id=pad)
        gen = out[:, enc["input_ids"].shape[1]:]
        for k, i in enumerate(idx):
            ids = gen[k].tolist(); stop = False
            for t, x in enumerate(ids):
                if x in eos:
                    ids = ids[:t + 1]; stop = True; break
            else:
                while ids and ids[-1] == pad:
                    ids.pop()
            p, j = reqs[i]
            rows[i] = {"ckpt": a.tag, "pid": p["pid"], "sid": j, "unique_id": p["unique_id"], "seed": seeds[k], "text": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids), "finish": "stop" if stop else "length"}
        print("[gen_hf] %d/%d done, %.0fs" % (min(b + a.batch, len(order)), len(order), time.time() - t0), flush=True)
    n = common.write_jsonl(a.out, rows)
    common.write_meta(a.out + ".meta.json", "gen_hf", {"model": a.model, "tag": a.tag, "samples": a.samples, "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "template": a.template, "eos": eos,
                      "prompt_example": texts[probs[0]["pid"]], "engine": "transformers generate (bf16, sdpa/eager per model code)"}, {"seed": a.seed}, inputs=[a.data], outputs=[a.out], extra={"n_records": n}, t_start=t0)
    print("[gen_hf] %s: %d samples in %.0fs" % (a.tag, n, time.time() - t0))


if __name__ == "__main__":
    main()
