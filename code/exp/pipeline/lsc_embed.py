"""Occurrence embeddings for the LSC metrics (EXPERIMENT_PLAN §2.2): one vector per occurrence of `wait`.

A FIXED encoder is used for every checkpoint of a series (primary: the series' base model) so that APD / PRT / JSD
between checkpoints measure change in USAGE, not drift of the encoder. The occurrence vector is the mean of the
hidden states at layer round(2/3 * n_layers) over the marker's tokens, teacher-forced on prompt + response.

    python -m exp.pipeline.lsc_embed --encoder <hf dir> --gen <graded jsonl.gz> --data <MATH-500> --template qwen_boxed \
        --n 500 --seed 0 --out emb.npz            (GPU; env gen)

Output npz: emb (n, d) float16, pid, sid, char_start, lexical (bool), and a JSON side-car with the selection rule.
Occurrences are sampled uniformly at random, stratified by problem (round-robin over problems), so N is equal
across checkpoints (LSC scores are frequency sensitive).
"""
from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--template", default="olmo_rlzero")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--layer_frac", type=float, default=2.0 / 3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import common, markers as mk, prompts

    t0 = time.time()
    gpus = common.guard_gpus()
    tok = AutoTokenizer.from_pretrained(a.encoder)
    model = AutoModelForCausalLM.from_pretrained(a.encoder, torch_dtype=torch.bfloat16).cuda().eval()
    n_layers = model.config.num_hidden_layers
    layer = int(round(a.layer_frac * n_layers))
    probs = {p["pid"]: p for p in prompts.load_problems(a.data)}
    rows = common.read_jsonl(a.gen)
    occ = defaultdict(list)  # pid -> [(sid, char_start, char_end, lexical)]
    for r in rows:
        for m in mk.find_markers(r["text"]):
            if m.form == "wait":
                occ[r["pid"]].append((r["sid"], m.start, m.end, m.lexical, r["text"]))
    rng = random.Random(common.derive_seed(a.seed, "lsc"))
    pool = {p: rng.sample(v, len(v)) for p, v in occ.items()}
    chosen = []
    while len(chosen) < a.n and any(pool.values()):  # round-robin over problems = stratified by problem
        for p in sorted(pool):
            if pool[p] and len(chosen) < a.n:
                chosen.append((p,) + pool[p].pop())
    embs, meta = [], []
    with torch.no_grad():
        for pid, sid, s, e, lex, text in chosen:
            prompt = prompts.render(a.template, probs[pid]["problem"], tok)
            full = prompt + text[:e]
            enc = tok(full, return_offsets_mapping=True, add_special_tokens=True)
            ids = enc["input_ids"]
            if len(ids) > a.max_ctx:
                continue
            lo = len(prompt) + s
            span = [i for i, (x, y) in enumerate(enc["offset_mapping"]) if y > lo and x < len(full) and y > x and x >= lo - 1]
            if not span:
                continue
            out = model(torch.tensor([ids]).cuda(), output_hidden_states=True)
            h = out.hidden_states[layer][0, span].float().mean(0)
            embs.append(h.cpu().numpy().astype(np.float16))
            meta.append((pid, sid, s, lex))
    arr = np.stack(embs) if embs else np.zeros((0, model.config.hidden_size), dtype=np.float16)
    np.savez_compressed(a.out, emb=arr, pid=np.array([m[0] for m in meta]), sid=np.array([m[1] for m in meta]),
                        char_start=np.array([m[2] for m in meta]), lexical=np.array([m[3] for m in meta]))
    common.write_meta(a.out + ".meta.json", "lsc_embed", {"encoder": a.encoder, "gen": a.gen, "layer": layer, "n_layers": n_layers,
                      "n_requested": a.n, "template": a.template}, {"seed": a.seed}, inputs=[a.gen], outputs=[a.out],
                      extra={"n_embedded": len(meta), "n_available": sum(len(v) for v in occ.values()), "gpus": gpus}, t_start=t0)
    print("[lsc_embed] %d/%d occurrences embedded (layer %d/%d)" % (len(meta), a.n, layer, n_layers))


if __name__ == "__main__":
    main()
