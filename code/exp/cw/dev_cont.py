"""DEV continuations of the SDT clean/corrupted prefixes (n samples x 300 tokens) with mode none (vanilla) | sham | cw (needs tau, tables, and the teacher-forced s0 of every prefix).
    python -m exp.cw.dev_cont --model <dir> --prefixes P.jsonl --scores S.jsonl --mode none|sham|cw --tables T.json --tau x --samples 4 --temperature .. --top_p .. --out C.jsonl [--trust]"""
import argparse
import json
import os

from exp.pipeline import common
from exp.pipeline import sdt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--prefixes", required=True); ap.add_argument("--scores", required=True); ap.add_argument("--mode", default="none"); ap.add_argument("--tables")
    ap.add_argument("--tau", type=float, default=0.0); ap.add_argument("--samples", type=int, default=4); ap.add_argument("--max_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.85); ap.add_argument("--trust", action="store_true"); ap.add_argument("--sync", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    common.guard_gpus()
    kw = {}
    if a.mode != "none":
        T = json.load(open(a.tables))
        cfg = {"mode": a.mode, "tau": a.tau, "seed": a.seed, "stats_path": a.out + ".cwstats.json", **{k: T[k] for k in ("marker_ids", "boundary_ids", "connector_ids", "digit_end_ids", "dot_ids")}}
        cp = a.out + ".cwconfig.json"; json.dump(cfg, open(cp, "w")); os.environ["CW_CONFIG"] = cp
        kw = {"logits_processors": ["exp.cw.cw_proc:CalibratedWaitProcessor"], "async_scheduling": False}
    elif a.sync:
        kw = {"async_scheduling": False}
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, seed=a.seed, max_model_len=8192, dtype="bfloat16", enable_prefix_caching=True, trust_remote_code=a.trust or common.trust_remote(), max_num_seqs=256, **kw)
    pre = [json.loads(l) for l in open(a.prefixes)]
    sc = {r["prefix_id"]: r for r in map(json.loads, open(a.scores))}
    pre = [p for p in pre if p["prefix_id"] in sc]
    prompts_, sps, meta = [], [], []
    for k, p in enumerate(pre):
        for j in range(a.samples):
            ea = {"cw_id": len(meta)}
            if a.mode != "none":
                ea["s0"] = sc[p["prefix_id"]]["s_ent"]
            prompts_.append(p["prompt"] + p["prefix"] + p["sep"]); sps.append(SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, extra_args=ea if a.mode != "none" else None)); meta.append((p, j))
    outs = llm.generate(prompts_, sps)
    rows = []
    for (p, j), o in zip(meta, outs):
        text = o.outputs[0].text
        any_, wait_ = sdt.mark_in_window(text)
        rows.append({"prefix_id": p["prefix_id"], "pid": p["pid"], "src": p["src"], "kind": (p.get("corruption") or {}).get("kind"), "sid": j, "text": text, "mark_any": any_, "mark_wait": wait_, "s0": sc[p["prefix_id"]]["s_ent"]})
    common.write_jsonl(a.out, rows)
    print("[dev_cont] %s mode=%s: %d continuations; MARK corrupted %.3f clean %.3f" % (a.out, a.mode, len(rows),
          sum(r["mark_any"] for r in rows if r["src"] == "corrupted") / max(1, sum(r["src"] == "corrupted" for r in rows)), sum(r["mark_any"] for r in rows if r["src"] == "clean") / max(1, sum(r["src"] == "clean" for r in rows))))


if __name__ == "__main__":
    main()
