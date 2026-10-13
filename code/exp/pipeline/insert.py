"""Insertion-intervention harness (EXPERIMENT_PLAN §3): prefix -> insert form -> continue N samples -> outcome.

Stages (each a sub-command; the CPU ones run in any env with numpy, `score` also needs math_verify):
  prefixes  CPU  pick counterfactual cut points in the late checkpoint's *correct*, marker-free traces
  continue  GPU  for one policy checkpoint, continue every prefix under every arm (n samples each)
  score     CPU  final-answer correctness + marker-independent rule REVISE on the first 3 sentence units
  (judge)        see judge.py: arm-blind local LLM judge on the same units

Design choices that matter for the estimand:
  * The SAME prefixes are continued by every checkpoint (incl. the base), so policies differ only in the
    continuing model. Prefixes come from the late checkpoint's traces.
  * A prefix is used only if the source trace finished, is correct, and has NO revision event and NO
    marker (any form in F) up to and including the sentence after the cut, so "the model's own next
    sentence is a plain continuation" (§3.1b) and steps 1..k are correct by construction (a correct
    final answer with no revision event; the PRM/judge third vote of §3.1a is not used in P0).
  * Cut k is drawn uniformly from the middle 60% of the trace's sentences (k >= 1, i.e. >= 2 sentences,
    and the next sentence must precede the final `Answer:` line).
  * The judge never sees the inserted form (arm-blind), only the continuation text.
"""
from __future__ import annotations

import argparse
import random
import re
import time
from typing import Dict, List, Optional, Tuple

from . import common, grade, markers as mk, prompts, revisions as rv

ARMS: Dict[str, str] = {"wait": "Wait,", "hold_on": "Hold on,", "hmm": "Hmm,", "okay": "Okay,", "next": "Next,", "so": "So,", "none": ""}
N_UNITS = 3


def arm_suffix(arm: str, sep: str) -> str:
    """String appended to the prefix for an arm. With an in-line separator the arm text is preceded by one
    space; the empty arm ends exactly at the sentence end (no dangling space, so token boundaries stay natural)."""
    form = ARMS[arm]
    if "\n" in sep:
        return sep + form
    return (" " + form) if form else ""


def _math_balanced(prefix: str) -> bool:
    stripped = prefix.replace("\\$", "")
    return stripped.count("$$") % 2 == 0 and (stripped.replace("$$", "")).count("$") % 2 == 0 and \
        prefix.count("\\[") == prefix.count("\\]") and prefix.count("\\(") == prefix.count("\\)")


def candidate_cuts(text: str, lo: float = 0.2, hi: float = 0.8, allow_prior_markers: bool = True) -> List[int]:
    """Indices k of the last prefix sentence that satisfy the protocol constraints (EXPERIMENT_PLAN §3.1.1):
      (a) steps 1..k are correct: the source trace is correct (checked by the caller) and NO content-based revision
          event (value change / negation / restart, detected on stripped sentences) occurs in sentences <= k+1;
      (b) the model's own next sentence k+1 is a plain continuation: no discourse marker and no revision event in it;
      (c) k is in the middle 60% of the trace (lo <= k/n <= hi), k >= 1, and k+1 precedes the final `Answer:` line.
    Earlier discourse markers in the prefix ARE allowed (allow_prior_markers=True): RL checkpoints emit `wait`
    every few sentences, so a marker-free prefix would select only atypical traces. P0's first version required a
    marker-free prefix and found 1 prefix in 800 traces of S1@1900; `allow_prior_markers=False` reproduces that.
    """
    sents = mk.split_sentences(text)
    n = len(sents)
    if n < 6:
        return []
    events = rv.detect_events([mk.strip_initial_forms(s.text)[0] for s in sents])
    marker_sents = {m.sent_idx for m in mk.find_markers(text, sents) if m.kind == "disc"} | \
                   {m.sent_idx for m in mk.find_markers(text, sents)}
    first_event = min((i for i, e in enumerate(events) if e), default=n + 1)
    first_marker = min(marker_sents, default=n + 1)
    ans = next((i for i, s in enumerate(sents) if re.search(r"answer\s*:", s.text, re.I)), n)
    cands = []
    for k in range(1, n - 1):
        if k + 1 >= ans:
            break
        if not (lo <= k / n <= hi):
            continue
        if first_event <= k + 1:
            break  # a content revision already happened: steps 1..k are no longer known-correct
        if (k + 1) in marker_sents or (not allow_prior_markers and first_marker <= k + 1):
            continue
        last = sents[k].text.rstrip()
        nxt = sents[k + 1].text.lstrip()
        if not last or last[-1] not in ".!?$:)" or not nxt or nxt[0].islower():
            continue
        if len(mk.words(mk.mask_math(last))) < 3 or k in marker_sents and len(mk.words(mk.mask_math(last))) < 5:
            continue  # do not cut right after a bare interjection
        if not _math_balanced(text[:sents[k].end]):
            continue
        cands.append(k)
    return cands


def select_prefixes(rows: List[dict], n_prefixes: int, seed: int, render_prompt, src: str = "late") -> List[dict]:
    """rows: graded gen records of the source checkpoint (need pid, sid, text, finish, correct, gold, problem)."""
    by_pid: Dict[int, List[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r)
    out = []
    for pid in sorted(by_pid):
        if len(out) >= n_prefixes:
            break
        rng = random.Random(common.derive_seed(seed, "prefix", src, pid))
        samples = [r for r in by_pid[pid] if r.get("correct") and r.get("finish") == "stop"]
        rng.shuffle(samples)
        for r in samples:
            cands = candidate_cuts(r["text"])
            if not cands:
                continue
            k = rng.choice(cands)
            sents = mk.split_sentences(r["text"])
            prefix = r["text"][:sents[k].end]
            out.append({"prefix_id": "%s:%d" % (src, pid), "src": src, "pid": pid, "sid": r["sid"], "k": k, "n_sent": len(sents), "prefix": prefix,
                        "sep": r["text"][sents[k].end:sents[k + 1].start], "next_sentence": sents[k + 1].text,
                        "gold": r["gold"], "problem": r["problem"], "prompt": render_prompt(r["problem"]),
                        "src_ckpt": r["ckpt"], "prefix_hash": common.git_blob_hash(prefix.encode())})
            break
    return out


def window_units(form: str, generated: str, n: int = N_UNITS) -> List[str]:
    """Outcome window (EXPERIMENT_PLAN §3.1.5): the first n sentences of `form + generated`, split by ONE fixed
    splitter, so the inserted form and the model's first clause are one sentence and the empty arm (form = "")
    takes the same code path. The inserted form is then removed from sentence 1 and the remainder capitalised
    (blinding, D15). A sentence that consists only of the form (generation started with a newline) becomes empty
    and is dropped, but it still counted towards the n sentences."""
    sents = mk.split_sentences(form + generated)[:n]
    units = [x.text for x in sents]
    if form and units and units[0].startswith(form):
        rest = units[0][len(form):].lstrip()
        units[0] = rest[:1].upper() + rest[1:] if rest else ""
    return [u for u in units if u]


def rule_revise(prefix: str, window: List[str]) -> Tuple[bool, Optional[str]]:
    """Marker-independent rule REVISE on the stripped window, judged against all prefix sentences. Sentence-initial
    discourse forms are stripped mechanically first (the rule never uses them anyway; the strip is logged upstream)."""
    p = [mk.strip_initial_forms(x.text)[0] for x in mk.split_sentences(prefix)]
    u = [mk.strip_initial_forms(x)[0] for x in window]
    ev = rv.detect_events(p + u)[len(p):]
    first = next((e for e in ev if e), None)
    return (first is not None, first)


# ----------------------------------------------------------------------------- CLI
def cmd_prefixes(a):
    t0 = time.time()
    rows = common.read_jsonl(a.gen)  # already graded (grade.py CLI): has 'correct'
    probs = {p["pid"]: p for p in prompts.load_problems(a.data)}
    for r in rows:
        r["gold"] = probs[r["pid"]]["gold"]
        r["problem"] = probs[r["pid"]]["problem"]
    tok = None
    if a.template != "olmo_rlzero":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=common.trust_remote())
    pre = select_prefixes(rows, a.n_prefixes, a.seed, lambda q: prompts.render(a.template, q, tok), src=a.src)
    n = common.write_jsonl(a.out, pre)
    cfg = {"gen": a.gen, "src": a.src, "n_prefixes": a.n_prefixes, "template": a.template, "cut_range": [0.2, 0.8],
           "n_units": N_UNITS, "arms": ARMS}
    common.write_meta(a.out + ".meta.json", "prefixes", cfg, {"seed": a.seed}, inputs=[a.gen, a.data], outputs=[a.out],
                      extra={"n_prefixes": n, "n_problems_with_prefix": len({p["pid"] for p in pre}),
                             "mean_k": sum(p["k"] for p in pre) / max(1, n)}, t_start=t0)
    print("[prefixes] %d prefixes from %d source samples" % (n, len(rows)))


def cmd_continue(a):
    from .gen import build_llm, run_requests

    t0 = time.time()
    gpus = common.guard_gpus()
    pre = common.read_jsonl(a.prefixes)
    arms = a.arms.split(",")
    reqs = [(p, arm, j) for p in pre for arm in arms for j in range(a.samples)]
    seeds = [common.derive_seed(a.seed, "ins", p["prefix_id"], arm, j) for p, arm, j in reqs]
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(a.model, trust_remote_code=common.trust_remote())
    if a.template:   # re-render each prefix's problem in THIS model's own prompt format (scale panel: prefixes come from another model)
        for p in pre:
            p["prompt"] = prompts.render(a.template, p["problem"], tk, a.effort)
    prompt_strs = [p["prompt"] + p["prefix"] + arm_suffix(arm, p["sep"]) for p, arm, j in reqs]
    longest = max(len(tk(x, add_special_tokens=False)["input_ids"]) for x in {x for x in prompt_strs})
    llm = build_llm(a.model, longest + a.max_tokens + 64, a.gpu_util, a.seed, tp=a.tp, kv_dtype=a.kv_dtype)   # context = longest prompt + generation (never drops a prefix)
    outs = run_requests(llm, prompt_strs, seeds, a.temperature, 1.0, a.max_tokens, store_logprobs=False, seeded=a.seeded)
    rows = []
    for (p, arm, j), s, o in zip(reqs, seeds, outs):
        c = o.outputs[0]
        rows.append({"policy": a.tag, "prefix_id": p["prefix_id"], "src": p["src"], "pid": p["pid"], "arm": arm, "j": j, "seed": s, "text": c.text,
                     "n_tokens": len(c.token_ids), "finish": c.finish_reason})
    n = common.write_jsonl(a.out, rows)
    cfg = {"model": a.model, "tag": a.tag, "prefixes": a.prefixes, "arms": {k: ARMS[k] for k in arms},
           "samples": a.samples, "max_tokens": a.max_tokens, "temperature": a.temperature, "kv_cache_dtype": a.kv_dtype, "tp": a.tp}
    common.write_meta(a.out + ".meta.json", "insert_continue", cfg, {"engine_seed": a.seed, "sampling_seed_mode": "per-request derive_seed(seed,'ins',prefix_id,arm,j)" if a.seeded else "engine RNG only (reproducible up to batch schedule)"},
                      inputs=[a.prefixes], outputs=[a.out], extra={"n_records": n, "gpus": gpus}, t_start=t0)
    print("[continue] %s: %d continuations in %.0fs" % (a.tag, n, time.time() - t0))


def cmd_score(a):
    t0 = time.time()
    pre = {p["prefix_id"]: p for p in common.read_jsonl(a.prefixes)}
    rows = common.read_jsonl(a.conts)
    out = []
    for r in rows:
        p = pre[r["prefix_id"]]
        full = p["prefix"] + arm_suffix(r["arm"], p["sep"]) + r["text"]
        pred = grade.extract_answer(r["text"]) or grade.extract_answer(full)
        u = window_units(ARMS[r["arm"]], r["text"])
        rev, ev = rule_revise(p["prefix"], u)
        out.append({"policy": r["policy"], "prefix_id": r["prefix_id"], "src": r["src"], "prefix_source": r["src"], "pid": r["pid"], "arm": r["arm"], "j": r["j"],
                    "stripped_token": ARMS[r["arm"]], "finish": r["finish"],
                    "n_tokens": r["n_tokens"], "pred": pred, "correct": grade.is_correct(pred, p["gold"]),
                    "rule_revise": bool(rev), "rule_event": ev, "units": u,
                    "starts_with_marker": bool(mk.find_markers(r["text"][:80])) if r["text"] else False})
    n = common.write_jsonl(a.out, out)
    common.write_meta(a.out + ".meta.json", "insert_score", {"conts": a.conts, "prefixes": a.prefixes, "n_units": N_UNITS},
                      {}, inputs=[a.conts, a.prefixes], outputs=[a.out], extra={"n_records": n}, t_start=t0)
    print("[score] %d continuations scored" % n)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefixes")
    p.add_argument("--gen", required=True, help="graded jsonl of the source (late) checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--n_prefixes", type=int, default=100)
    p.add_argument("--template", default="olmo_rlzero")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tokenizer", default=None, help="model dir whose chat template renders qwen_boxed/generic prompts")
    p.add_argument("--src", default="late", choices=["late", "base"], help="checkpoint family the traces come from")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prefixes)
    c = sub.add_parser("continue")
    c.add_argument("--model", required=True)
    c.add_argument("--tag", required=True)
    c.add_argument("--prefixes", required=True)
    c.add_argument("--arms", default=",".join(ARMS))
    c.add_argument("--samples", type=int, default=4)
    c.add_argument("--max_tokens", type=int, default=3072)
    c.add_argument("--temperature", type=float, default=1.0)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--gpu_util", type=float, default=0.88)
    c.add_argument("--seeded", action="store_true", help="per-request seeds (slow)")
    c.add_argument("--tp", type=int, default=1)
    c.add_argument("--template", default=None, help="re-render prompts with this template and the model's tokenizer (default: use the stored prompt)")
    c.add_argument("--effort", default=None)
    c.add_argument("--kv_dtype", default="auto", help="'fp8' after the numerics check (D28)")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_continue)
    s = sub.add_parser("score")
    s.add_argument("--conts", required=True)
    s.add_argument("--prefixes", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
