"""S5 (D50): shared-prefix branching on NATURAL marker events from held-out DEV-B problems (odd pid) of one model's vanilla DEV generations.
For every event (a sentence-initial paradigm-F marker in a vanilla DEV response; prefix = response text before the marker):
  gate score s = mean token entropy of the last sentence of the prefix (dev_scores), label SUPPRESS iff s < tau (D50f tau from calib.json), else KEEP;
  ALLOW branch    = prefix + the original marker string, then free decoding to the cap (x2 samples);
  SUPPRESS branch = one token sampled with every marker token masked at that position only (logit_bias), then free decoding to the cap (x2); later markers are allowed.
  gain_e = acc(allow) - acc(suppress);  S5 compares mean gain of SUPPRESS-labelled vs KEEP events (problem-cluster bootstrap CI of the difference; prediction: smaller for SUPPRESS).
    python -m exp.cw.s5 events --dev_dir D --data DEV.jsonl --template T --tokenizer M --out D/s5_events.jsonl [--max_events 300]
    (GPU) python -m exp.cw.dev_scores --model M --prefixes D/s5_events.jsonl --out D/s5_scores.jsonl
    (GPU) python -m exp.cw.s5 run --model M --dev_dir D --tables TAB --template T --temperature .. --top_p .. --cap 8192 --out D/s5_branches.jsonl
    python -m exp.pipeline.grade --gen D/s5_branches.jsonl --data DEV.jsonl --out_dir D
    python -m exp.cw.s5 analyze --dev_dir D --out D/s5.json"""
import argparse
import gzip
import json
import random

import numpy as np


def events(a):
    from transformers import AutoTokenizer

    from exp.pipeline import markers as mk
    from exp.pipeline import prompts
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=a.trust, local_files_only=True)
    probs = {p["pid"]: p for p in prompts.load_problems(a.data, None)}
    rows = [json.loads(l) for l in gzip.open(a.dev_dir + "/dev_gen.graded.jsonl.gz", "rt")]
    rng = random.Random(20261012); ev = []
    for r in rows:
        if r["pid"] % 2 != 1:
            continue
        text = r["text"]; sents = mk.split_sentences(text); cand = []
        for m in mk.find_markers(text, sents):
            if m.lexical or m.word_idx != 0 or m.start < 300 or m.form not in ("wait", "hold_on", "hmm", "actually", "alternatively"):
                continue
            cand.append(m)
        rng.shuffle(cand)
        for m in cand[:a.per_problem]:
            pre = text[:m.start]; lead = ""
            if pre.endswith(" "):
                pre = pre.rstrip(" "); lead = " "
            if not pre.strip():
                continue
            prompt = prompts.render(a.template, probs[r["pid"]]["problem"], tok, None)
            ev.append({"prefix_id": "s5:%d:%d" % (r["pid"], m.start), "pid": r["pid"], "src": "natural", "prompt": prompt, "prefix": pre, "lead": lead, "marker": m.surface, "form": m.form, "n_prefix_tok": len(tok.encode(pre, add_special_tokens=False))})
    rng.shuffle(ev); ev = ev[:a.max_events]
    with open(a.out, "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    print("events", len(ev), "problems", len({e["pid"] for e in ev}), "forms", {k: sum(e["form"] == k for e in ev) for k in sorted({e["form"] for e in ev})})


def run(a):
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    from exp.pipeline import common
    common.guard_gpus()
    T = json.load(open(a.tables)); ids = sorted(set(T["marker_ids"]))
    ev = [json.loads(l) for l in open(a.dev_dir + "/s5_events.jsonl")]
    sc = {r["prefix_id"]: r["s_ent"] for r in map(json.loads, open(a.dev_dir + "/s5_scores.jsonl"))}
    ev = [e for e in ev if e["prefix_id"] in sc]
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, seed=a.seed, max_model_len=a.cap + 2048, dtype="bfloat16", enable_prefix_caching=True, trust_remote_code=a.trust or common.trust_remote(), max_num_seqs=256, async_scheduling=False)
    rem = lambda e: max(64, a.cap - e["n_prefix_tok"])
    # allow: prefix + lead + marker, continue freely
    aprompts = [e["prompt"] + e["prefix"] + e["lead"] + e["marker"] for e in ev]
    asp = [SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=rem(e)) for e in ev]
    aouts = llm.generate(aprompts, asp)
    # suppress: step 1 with markers masked at this position only, then free
    bias = {int(i): -100.0 for i in ids}
    s1 = llm.generate([e["prompt"] + e["prefix"] for e in ev], [SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, max_tokens=1, logit_bias=bias) for e in ev])
    p2, sp2, back = [], [], []
    for k, (e, o) in enumerate(zip(ev, s1)):
        for j, c in enumerate(o.outputs):
            p2.append(e["prompt"] + e["prefix"] + c.text); sp2.append(SamplingParams(n=1, temperature=a.temperature, top_p=a.top_p, max_tokens=max(1, rem(e) - len(c.token_ids)))); back.append((k, j, c.text, len(c.token_ids)))
    souts = llm.generate(p2, sp2)
    rows = []; sid = 0
    for k, e in enumerate(ev):
        for j, c in enumerate(aouts[k].outputs):
            rows.append({"ckpt": "s5", "pid": e["pid"], "sid": sid, "unique_id": e["prefix_id"], "event": e["prefix_id"], "arm": "allow", "rep": j, "s": sc[e["prefix_id"]], "text": e["prefix"] + e["lead"] + e["marker"] + c.text, "n_tokens": e["n_prefix_tok"] + len(c.token_ids)}); sid += 1
    for (k, j, t1, n1), o in zip(back, souts):
        e = ev[k]; c = o.outputs[0]
        rows.append({"ckpt": "s5", "pid": e["pid"], "sid": sid, "unique_id": e["prefix_id"], "event": e["prefix_id"], "arm": "suppress", "rep": j, "s": sc[e["prefix_id"]], "text": e["prefix"] + t1 + c.text, "n_tokens": e["n_prefix_tok"] + n1 + len(c.token_ids)}); sid += 1
    common.write_jsonl(a.out, rows); print("branches", len(rows))


def analyze(a, B=4000, seed=20261012):
    tau = json.load(open(a.dev_dir + "/calib.json"))["tau"]
    g = [json.loads(l) for l in gzip.open(a.dev_dir + "/s5_branches.graded.jsonl.gz", "rt")]
    ev = {}
    for r in g:
        ev.setdefault(r["event"], {"pid": r["pid"], "s": r["s"], "allow": [], "suppress": [], "ta": [], "ts": []})
        ev[r["event"]][r["arm"]].append(float(bool(r["correct"]))); ev[r["event"]]["t" + ("a" if r["arm"] == "allow" else "s")].append(r["n_tokens"])
    E = [{"pid": v["pid"], "sup": v["s"] < tau, "gain": np.mean(v["allow"]) - np.mean(v["suppress"]), "dtok": np.mean(v["ta"]) - np.mean(v["ts"]), "acc_allow": np.mean(v["allow"]), "acc_sup": np.mean(v["suppress"])} for v in ev.values() if v["allow"] and v["suppress"]]
    pids = sorted({e["pid"] for e in E}); byp = {p: [e for e in E if e["pid"] == p] for p in pids}

    def stat(ps):
        sel = [e for p in ps for e in byp[p]]; a1 = [e["gain"] for e in sel if e["sup"]]; a0 = [e["gain"] for e in sel if not e["sup"]]
        return (np.mean(a1) if a1 else np.nan, np.mean(a0) if a0 else np.nan)
    rng = np.random.default_rng(seed); full = stat(pids); bs = np.array([stat(list(rng.choice(pids, len(pids)))) for _ in range(B)])
    d = bs[:, 0] - bs[:, 1]; d = d[~np.isnan(d)]
    pc = lambda x, q: float(np.nanpercentile(x, q))
    out = {"tau": tau, "n_events": len(E), "n_problems": len(pids), "n_suppress_labelled": int(sum(e["sup"] for e in E)), "n_keep": int(sum(not e["sup"] for e in E)),
           "gain_suppress_labelled": {"est": float(full[0]), "lo": pc(bs[:, 0], 2.5), "hi": pc(bs[:, 0], 97.5)}, "gain_keep": {"est": float(full[1]), "lo": pc(bs[:, 1], 2.5), "hi": pc(bs[:, 1], 97.5)},
           "diff_sup_minus_keep": {"est": float(full[0] - full[1]), "lo": pc(d, 2.5), "hi": pc(d, 97.5)}, "acc_allow": float(np.mean([e["acc_allow"] for e in E])), "acc_suppress": float(np.mean([e["acc_sup"] for e in E])),
           "dtok_allow_minus_suppress": float(np.mean([e["dtok"] for e in E])), "pass": bool(pc(d, 97.5) < 0)}
    json.dump(out, open(a.out, "w"), indent=1); print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("events"); e.add_argument("--dev_dir", required=True); e.add_argument("--data", required=True); e.add_argument("--template", required=True); e.add_argument("--tokenizer", required=True)
    e.add_argument("--out", required=True); e.add_argument("--max_events", type=int, default=300); e.add_argument("--per_problem", type=int, default=3); e.add_argument("--trust", action="store_true")
    r = sub.add_parser("run"); r.add_argument("--model", required=True); r.add_argument("--dev_dir", required=True); r.add_argument("--tables", required=True); r.add_argument("--template")
    r.add_argument("--temperature", type=float, default=0.6); r.add_argument("--top_p", type=float, default=0.95); r.add_argument("--cap", type=int, default=8192); r.add_argument("--n", type=int, default=2)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--gpu_util", type=float, default=0.9); r.add_argument("--trust", action="store_true"); r.add_argument("--out", required=True)
    an = sub.add_parser("analyze"); an.add_argument("--dev_dir", required=True); an.add_argument("--out", required=True)
    a = ap.parse_args()
    {"events": events, "run": run, "analyze": analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
