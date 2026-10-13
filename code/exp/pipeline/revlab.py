"""Judge-based revision labels for OBL / REL / revision rate (D30: the judge is the PRIMARY revision-event labeller;
the rule detector of revisions.py stays as the robustness row).

For each checkpoint (one graded generation file each) we label a stratified sample of sentences with the blinded revision
prompt (exp/judge_prompts/revision.md v1.0: stripped target sentence, context, next two sentences; Qwen2.5-7B-Instruct;
one output token A-E). A revision event R is judge letter B (revise_value) or C (restart).

  OBL_j = P(a discourse marker in sentence i or i-1 | R at i)   two strata by m_i = [marker in i or i-1]; n_s1 and n_s0
          sentences sampled uniformly from each; weights W_s = N_s / n_s (all sentences of the stratum in the checkpoint's
          traces); OBL_j = W1*r1 / (W1*r1 + W0*r0) with r_s = #R among sampled.
  REL_j = P(R in sentences i..i+2 | marker sentence i)   n_rel marker sentences sampled uniformly, their three sentences
          labelled; REL_strict_j uses i+1..i+2 only.
  R_per_100s_j = 100 * (W1 r1 + W0 r0) / (all sentences).
Cluster bootstrap over problems (resampling problem ids; strata sizes fixed). Sentences whose stripped text is empty
("Wait.") get no judge call and count as non-events.

    python -m exp.pipeline.revlab --gen a.graded.jsonl.gz b.graded.jsonl.gz --model <judge dir> --out_dir DIR
writes DIR/<gen stem>.revlab.json (+ .labels.jsonl.gz). GPU 6/7 only (vLLM), env gen.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import common, judge as J, markers as mk, stats as st

REV_LETTERS = ("B", "C")


def build_universe(rows: List[dict]):
    """All (response, sentence) targets with stratum, plus the marker sentences."""
    tgt, msent = [], []
    for ri, r in enumerate(rows):
        sents = mk.split_sentences(r["text"])
        ms = {m.sent_idx for m in mk.find_markers(r["text"], sents) if m.kind == "disc"}
        r["_sents"] = sents
        for i in range(1, len(sents)):
            m_i = int(i in ms or (i - 1) in ms)
            tgt.append((ri, i, m_i))
            if i in ms:
                msent.append((ri, i))
    return tgt, msent


def target_prompt_fields(row: dict, i: int) -> Tuple[str, str, str]:
    sents = row["_sents"]
    stripped, _ = mk.strip_initial_forms(sents[i].text)
    nxt = " ".join(s.text for s in sents[i + 1:i + 3])[:600]
    return row["text"][:sents[i].start][-2500:], stripped, nxt


def estimate(pid: np.ndarray, stratum: np.ndarray, isr: np.ndarray, N1: float, N0: float, rel_pid: np.ndarray, rel_any: np.ndarray,
             rel_strict: np.ndarray, total_sent: float, B: int = 2000, seed: int = 20261012) -> Dict[str, Dict[str, float]]:
    """OBL_j, REL_j, REL_strict_j, R_per_100s_j with cluster bootstrap over problem ids."""
    pids = np.unique(np.concatenate([pid, rel_pid]))
    pos = {p: k for k, p in enumerate(pids)}
    a, b = np.array([pos[p] for p in pid]), np.array([pos[p] for p in rel_pid])
    n1, n0 = float((stratum == 1).sum()), float((stratum == 0).sum())

    def stat(w: np.ndarray) -> Dict[str, float]:
        wa, wb = w[a], w[b]
        r1 = (wa * isr * (stratum == 1)).sum(); r0 = (wa * isr * (stratum == 0)).sum()
        m1 = (wa * (stratum == 1)).sum(); m0 = (wa * (stratum == 0)).sum()
        # weights: the stratum totals are fixed, the sampled fractions come from the resample
        R1 = N1 * r1 / m1 if m1 > 0 else 0.0
        R0 = N0 * r0 / m0 if m0 > 0 else 0.0
        nb = wb.sum()
        return {"OBL_j": R1 / (R1 + R0) if R1 + R0 > 0 else float("nan"),
                "REL_j": (wb * rel_any).sum() / nb if nb > 0 else float("nan"),
                "REL_strict_j": (wb * rel_strict).sum() / nb if nb > 0 else float("nan"),
                "R_per_100s_j": 100.0 * (R1 + R0) / total_sent}

    est = stat(np.ones(len(pids)))
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in est}
    for _ in range(B):
        w = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids)).astype(float)
        s = stat(w)
        for k in reps:
            reps[k].append(s[k])
    out = {}
    for k, v in est.items():
        lo, hi = st.percentile_ci(reps[k])
        out[k] = {"est": float(v), "lo": lo, "hi": hi}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_s1", type=int, default=700)
    ap.add_argument("--n_s0", type=int, default=700)
    ap.add_argument("--n_rel", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--gpu_util", type=float, default=0.9)
    ap.add_argument("--mode", default="letter", choices=["letter", "harmony"])
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    t0 = time.time()
    gpus = common.guard_gpus()
    system, template, phash, pver = J.load_prompt("revision.md")
    plan = []   # per gen file: dict with samples
    prompts, keys = [], []
    for g in a.gen:
        rows = common.read_jsonl(g)
        tgt, msent = build_universe(rows)
        rng = random.Random(common.derive_seed(a.seed, "revlab", Path(g).name))
        s1 = [t for t in tgt if t[2] == 1]; s0 = [t for t in tgt if t[2] == 0]
        samp1 = rng.sample(s1, min(a.n_s1, len(s1))); samp0 = rng.sample(s0, min(a.n_s0, len(s0)))
        relm = rng.sample(msent, min(a.n_rel, len(msent)))
        rel_targets = [(ri, j) for ri, i in relm for j in (i, i + 1, i + 2) if j < len(rows[ri]["_sents"])]
        need = {(ri, i) for ri, i, _ in samp1 + samp0} | set(rel_targets)
        for ri, i in sorted(need):
            ctx, tgt_s, nxt = target_prompt_fields(rows[ri], i)
            if tgt_s.strip():
                prompts.append((system, template.replace("{context}", ctx).replace("{sentence}", tgt_s).replace("{next}", nxt)))
                keys.append((g, ri, i))
        plan.append({"g": g, "rows": rows, "samp1": samp1, "samp0": samp0, "relm": relm, "N1": len(s1), "N0": len(s0),
                     "total_sent": sum(len(r["_sents"]) for r in rows)})
    harmony = a.mode == "harmony"
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, seed=0, enable_prefix_caching=True, tensor_parallel_size=a.tp,
              **({} if harmony else {"dtype": "bfloat16"}))
    tok = llm.get_tokenizer()
    kw = {"reasoning_effort": a.effort} if harmony else {}
    texts, n_clipped = [], 0
    limit = 8192 - (512 if harmony else 2) - 64
    for s, u in prompts:
        t = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, **kw)
        if len(tok(t, add_special_tokens=False)["input_ids"]) > limit:   # degenerate over-long sentence/context: clip the user text from the middle-front until it fits
            n_clipped += 1
            keep = len(u)
            while keep > 200:
                keep = int(keep * 0.8)
                t = tok.apply_chat_template([{"role": "system", "content": s}, {"role": "user", "content": u[:keep // 2] + " [...] " + u[-keep // 2:]}], tokenize=False, add_generation_prompt=True, **kw)
                if len(tok(t, add_special_tokens=False)["input_ids"]) <= limit:
                    break
        texts.append(t)
    sp = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False) if harmony else SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate(texts, sp)
    if harmony:
        from .judge_audit import final_letter
        letter = {k: final_letter(o.outputs[0].text) for k, o in zip(keys, outs)}
    else:
        letter = {k: J.parse_letter(o.outputs[0].text) for k, o in zip(keys, outs)}
    for pl in plan:
        g, rows = pl["g"], pl["rows"]
        isR = lambda ri, i: int(letter.get((g, ri, i), "A") in REV_LETTERS)   # unlabelled (empty after stripping) = not an event
        items = pl["samp1"] + pl["samp0"]
        pid = np.array([rows[ri]["pid"] for ri, _, _ in items]); stratum = np.array([m for _, _, m in items])
        r = np.array([isR(ri, i) for ri, i, _ in items], float)
        rel_pid = np.array([rows[ri]["pid"] for ri, _ in pl["relm"]])
        rel_any = np.array([max(isR(ri, j) for j in (i, i + 1, i + 2) if j < len(rows[ri]["_sents"])) for ri, i in pl["relm"]], float)
        rel_strict = np.array([max([isR(ri, j) for j in (i + 1, i + 2) if j < len(rows[ri]["_sents"])] or [0]) for ri, i in pl["relm"]], float)
        est = estimate(pid, stratum, r, pl["N1"], pl["N0"], rel_pid, rel_any, rel_strict, pl["total_sent"], B=a.B)
        cats = {c: sum(1 for k, v in letter.items() if k[0] == g and J.CATS.get(v) == c) for c in J.CATS.values()}
        stem = Path(g).name.replace(".graded.jsonl.gz", "")
        common.write_json(Path(a.out_dir) / (stem + ".revlab.json"),
                          {"indices": est, "N_stratum1": pl["N1"], "N_stratum0": pl["N0"], "n_sampled": len(items), "n_marker_sentences_sampled": len(pl["relm"]),
                           "n_judged_calls": sum(1 for k in letter if k[0] == g), "judge_category_counts": cats, "prompt_version": pver,
                           "prompt_hash": phash, "judge_model": a.model, "n_prompts_clipped_overlong": n_clipped, "stratum_R_rate": {"marker_adjacent": float(r[stratum == 1].mean()), "other": float(r[stratum == 0].mean())}})
    common.write_meta(Path(a.out_dir) / "revlab.meta.json", "revlab", {"model": a.model, "gen": a.gen, "n_s1": a.n_s1, "n_s0": a.n_s0, "n_rel": a.n_rel, "B": a.B,
                      "prompt_version": pver}, {"seed": a.seed, "rng_seed": 20261012}, inputs=a.gen, outputs=[],
                      extra={"gpus": gpus, "n_prompts": len(prompts), "gpu_hours": round((time.time() - t0) * max(1, len(gpus)) / 3600, 3)}, t_start=t0)
    print("[revlab] %d gens, %d judge calls, %.0fs" % (len(a.gen), len(prompts), time.time() - t0))


if __name__ == "__main__":
    main()
