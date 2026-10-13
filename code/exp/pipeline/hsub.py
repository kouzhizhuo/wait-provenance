"""Policy-side paradigm entropy H_sub (EXPERIMENT_PLAN §2.1): what does the CHECKPOINT expect at a revision slot?

At each revision slot (a sentence the rule detector marks as a revision event) the checkpoint is teacher-forced on
prompt + response up to the END of the previous sentence. We read the next-token distribution and, for every form f
in F, log-sum-exp the probability of the first token over the surface variants (leading space / newline / blank line
x capitalised sentence-initial spelling). Renormalised over F this gives a distribution whose entropy is H_sub.
We also keep the un-normalised mass P_F (how much probability the policy puts on ANY paradigm member): a
policy-side obligatoriness proxy that needs no revision labels other than the slot positions.

    python -m exp.pipeline.hsub --model <hf dir> --gen <graded jsonl.gz> --template qwen_boxed --data <MATH-500> \
        --n_slots 300 --seed 0 --out hsub.json         (GPU; env gen)

Slots come from the checkpoint's OWN traces (--gen), so H_sub is on-policy for that checkpoint.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

# first words of the forms, sentence-initial spelling (multi-token forms use their first token only)
FORM_WORDS: Dict[str, List[str]] = {
    "wait": ["Wait"], "hold_on": ["Hold"], "hmm": ["Hmm", "Hm"], "actually": ["Actually"], "alternatively": ["Alternatively"],
    "but": ["But"], "however": ["However"], "oh": ["Oh"], "no": ["No"], "recheck": ["Let"], "mistake": ["I"],
}


def variant_token_ids(tokenizer, lead: str) -> Dict[str, List[int]]:
    """First-token ids per form. lead=" " when the slot follows an in-line space (GPT-style BPE merges the space into
    the word token), lead="" when the context already ends in a newline (the separator is teacher-forced)."""
    out: Dict[str, List[int]] = {}
    for f, words in FORM_WORDS.items():
        ids = set()
        for w in words:
            enc = tokenizer.encode(lead + w, add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        out[f] = sorted(ids)
    return out


def form_distribution(logprobs: np.ndarray, variants: Dict[str, Sequence[int]]) -> Tuple[Dict[str, float], float]:
    """logprobs: next-token log-probabilities (V,). Returns ({form: renormalised prob over F}, P_F mass)."""
    lse = {f: float(np.logaddexp.reduce(logprobs[list(ids)])) if len(ids) else -np.inf for f, ids in variants.items()}
    mass = {f: math.exp(v) for f, v in lse.items()}
    tot = sum(mass.values())
    if tot <= 0:
        return {f: 0.0 for f in mass}, 0.0
    return {f: m / tot for f, m in mass.items()}, tot


def entropy_of(dist: Dict[str, float]) -> float:
    return float(-sum(p * math.log2(p) for p in dist.values() if p > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--template", default="olmo_rlzero")
    ap.add_argument("--n_slots", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_ctx", type=int, default=6144)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import common, markers as mk, prompts, revisions as rv

    t0 = time.time()
    gpus = common.guard_gpus()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
    probs = {p["pid"]: p for p in prompts.load_problems(a.data)}
    rows = common.read_jsonl(a.gen)
    slots = []
    for r in rows:
        sents = mk.split_sentences(r["text"])
        ev = rv.detect_events([mk.strip_initial_forms(s.text)[0] for s in sents])
        for i, e in enumerate(ev):
            if e and i >= 1:
                sep = r["text"][sents[i - 1].end:sents[i].start]
                # in-line space: stop at the sentence end and score " Word"; newline: teacher-force the separator, score "Word"
                ctx = r["text"][:sents[i - 1].end] if "\n" not in sep else r["text"][:sents[i].start]
                slots.append((r["pid"], r["sid"], i, e, ctx, " " if "\n" not in sep else ""))
    rng = random.Random(common.derive_seed(a.seed, "hsub"))
    rng.shuffle(slots)
    slots = slots[:a.n_slots]
    variants = {" ": variant_token_ids(tok, " "), "": variant_token_ids(tok, "")}
    res = []
    with torch.no_grad():
        for pid, sid, i, e, ctx, lead in slots:
            prompt = prompts.render(a.template, probs[pid]["problem"], tok)
            ids = tok(prompt + ctx, return_tensors="pt", add_special_tokens=True).input_ids[:, -a.max_ctx:].cuda()
            lp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1).cpu().numpy()
            dist, mass = form_distribution(lp, variants[lead])
            res.append({"pid": pid, "sid": sid, "sent": i, "event": e, "P_F": mass, "H_sub": entropy_of(dist),
                        "p_wait": dist["wait"], "argmax": max(dist, key=dist.get)})
    H = [r["H_sub"] for r in res if r["P_F"] > 0]
    summary = {"n_slots": len(res), "H_sub_mean": float(np.mean(H)) if H else float("nan"),
               "P_F_mean": float(np.mean([r["P_F"] for r in res])) if res else float("nan"),
               "p_wait_mean": float(np.mean([r["p_wait"] for r in res])) if res else float("nan")}
    common.write_json(a.out, {"summary": summary, "slots": res, "variants": variants})
    common.write_meta(a.out + ".meta.json", "hsub", {"model": a.model, "gen": a.gen, "template": a.template, "n_slots": a.n_slots},
                      {"seed": a.seed}, inputs=[a.gen], outputs=[a.out], extra={"gpus": gpus, **summary}, t_start=t0)
    print("[hsub]", summary)


if __name__ == "__main__":
    main()
