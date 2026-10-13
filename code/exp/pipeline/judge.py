"""Arm-blind local judge (vLLM, open weights, GPU 6/7 only). Prompts: exp/judge_prompts/*.md (v1.0, wording by the analyst).

    python -m exp.pipeline.judge --model $R/models/Qwen2.5-7B-Instruct --prefixes prefixes.jsonl \
        --scored <EMAIL> [more ...] --out judged.jsonl.gz

Call 1 (`insertion_outcome.md`): one output token, the letter A-E; we store the log-probabilities of the letters
from the top-20 first-token candidates. The judge sees the last 2500 characters of the prefix and the first three
sentence units of the continuation with the inserted form REMOVED (insert.window_units); it is never told the arm
or that the prefix is correct.
Call 2 (`revision_detail.md`, only when call 1 answered B): extraction `STEP=<n> OLD=<..> NEW=<..>`, also arm-blind.
FALSE_REV (B only) = STEP is the last prefix step and NEW is not math-equivalent to OLD; STEP < last is logged
separately (`earlier_step_rev`). Steps are sentences of the prefix (P0). Restarts (C) have no step information.
Every row logs judge model (+ index hash), prompt file/version/hash, date, letter, letter log-probs, stripped token, parse_ok.
"""
from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import common, grade, markers as mk

LETTERS = "ABCDE"
CATS = {"A": "continue", "B": "revise_value", "C": "restart", "D": "reverify_confirm", "E": "branch"}
PROMPT_DIR = Path(__file__).resolve().parent.parent / "judge_prompts"
_DETAIL_RE = re.compile(r"STEP\s*=\s*(\d+)\s+OLD\s*=\s*(.*?)\s+NEW\s*=\s*(.*)", re.S)


def load_prompt(name: str = "insertion_outcome.md", section: Optional[str] = None) -> Tuple[str, str, str, str]:
    """(system, user_template, file_hash, prompt_version). `section` ('M1'/'M3') selects one prompt of error_presence.md."""
    txt = (PROMPT_DIR / name).read_text(encoding="utf-8")
    vm = re.search(r"prompt_version:\s*([^\s|]+)", txt)
    body = re.sub(r"<!--.*?-->", "", txt, flags=re.S)
    if section:
        parts = re.split(r"(?m)^# (M\d)\s*$", body)
        body = dict(zip(parts[1::2], parts[2::2]))[section]
    m = re.search(r"##\s*SYSTEM\s*\n(.*?)\n##\s*USER\s*\n(.*)$", body, flags=re.S)
    if not m:
        raise ValueError("bad judge prompt file %s" % name)
    return m.group(1).strip(), m.group(2).strip(), common.git_blob_hash(txt.encode()), (vm.group(1) if vm else "unversioned")


def build_user(template: str, prefix: str, unit_texts: List[str]) -> str:
    cont = " ".join(unit_texts) if unit_texts else "(empty)"
    return template.replace("{prefix_tail}", prefix[-2500:]).replace("{continuation}", cont)


def numbered_steps(prefix: str) -> Tuple[str, int]:
    """'[1] ...\\n[2] ...' with one sentence per step, and the number of steps."""
    sents = [s.text for s in mk.split_sentences(prefix)]
    return "\n".join("[%d] %s" % (i + 1, t) for i, t in enumerate(sents)), len(sents)


_LABEL2LETTER = {"continue": "A", "revise-value": "B", "revise_value": "B", "restart": "C", "re-verify-confirm": "D",
                 "reverify_confirm": "D", "branch": "E"}


def parse_letter(text: str) -> str:
    """Letter A-E from a one-letter answer (JSON `label` answers are also accepted for robustness)."""
    t = text.strip()
    m = re.search(r'"label"\s*:\s*"([a-z_\-]+)"', t)
    if m:
        return _LABEL2LETTER.get(m.group(1), "?")
    m = re.search(r"\b([A-E])\b", t)
    return m.group(1) if m else "?"


def parse_detail(text: str) -> Optional[Tuple[int, str, str]]:
    m = _DETAIL_RE.search(text.strip())
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip(), (m.group(3).strip().splitlines() or [""])[0].strip()


def letter_logprobs(first_token_logprobs) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {c: None for c in LETTERS}
    for _tid, d in (first_token_logprobs or {}).items():
        tt = (d.decoded_token or "").strip()
        if tt in out and (out[tt] is None or d.logprob > out[tt]):
            out[tt] = float(d.logprob)
    return out


def conf_of(lp: Dict[str, Optional[float]], letter: str) -> float:
    """P(letter) renormalised over the letters that appear in the top-20 candidates."""
    ps = {c: math.exp(v) for c, v in lp.items() if v is not None}
    z = sum(ps.values())
    return ps.get(letter, float("nan")) / z if z > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefixes", required=True)
    ap.add_argument("--scored", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="insertion_outcome.md")
    ap.add_argument("--no_detail", action="store_true")
    ap.add_argument("--mode", default="letter", choices=["letter", "harmony"], help="letter: one output token (Qwen); harmony: gpt-oss reasoning model, letter of the final channel")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--no_think", action="store_true", help="Qwen3: chat template with enable_thinking=False")
    ap.add_argument("--limit", type=int, default=0, help="random subsample of the scored rows (seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    t0 = time.time()
    gpus = common.guard_gpus()
    system, template, phash, pver = load_prompt(a.prompt)
    dsys, dtemplate, dhash, dver = load_prompt("revision_detail.md")
    pre = {p["prefix_id"]: p for p in common.read_jsonl(a.prefixes)}
    rows: List[dict] = []
    for f in a.scored:
        rows += common.read_jsonl(f)
    if a.limit and len(rows) > a.limit:
        import random
        rows = random.Random(a.seed).sample(rows, a.limit)
    harmony = a.mode == "harmony"
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_util, max_model_len=8192, seed=0, enable_prefix_caching=True, tensor_parallel_size=a.tp,
              **({} if harmony else {"dtype": "bfloat16"}))
    tok = llm.get_tokenizer()
    if harmony:
        a.no_detail = True   # second label column: the outcome letter only

    def chat(sysm, user):
        kw = {"reasoning_effort": a.effort} if harmony else ({"enable_thinking": False} if a.no_think else {})
        return tok.apply_chat_template([{"role": "system", "content": sysm}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True, **kw)

    sp = SamplingParams(temperature=0.0, max_tokens=512, skip_special_tokens=False) if harmony else SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    outs = llm.generate([chat(system, build_user(template, pre[r["prefix_id"]]["prefix"], r["units"])) for r in rows], sp)
    idx_file = Path(a.model) / "model.safetensors.index.json"
    model_id = {"path": a.model, "index_hash": common.file_hash(idx_file) if idx_file.exists() else None}
    date = time.strftime("%Y-%m-%d", time.gmtime())
    for r, o in zip(rows, outs):
        c = o.outputs[0]
        if harmony:
            from .judge_audit import final_letter
            letter = final_letter(c.text)
            lp = {k: None for k in LETTERS}
        else:
            letter = parse_letter(c.text)
            lp = letter_logprobs(c.logprobs[0] if c.logprobs else None)
        r.update({"judge_letter": letter, "judge_cat": CATS.get(letter, "invalid"), "judge_revise": letter in ("B", "C"),
                  "judge_conf": conf_of(lp, letter), "letter_logprobs": lp, "parse_ok": letter in LETTERS,
                  "judge_model": model_id, "prompt_file": a.prompt, "prompt_version": pver, "prompt_hash": phash, "date": date})
    n_detail = 0
    if not a.no_detail:
        need = [r for r in rows if r["judge_letter"] == "B"]
        dprompts = []
        for r in need:
            steps, n_steps = numbered_steps(pre[r["prefix_id"]]["prefix"])
            r["n_prefix_steps"] = n_steps
            dprompts.append(chat(dsys, dtemplate.replace("{prefix_numbered}", steps).replace("{continuation}", " ".join(r["units"]))))
        first = llm.generate(dprompts, SamplingParams(temperature=0.0, max_tokens=96)) if dprompts else []
        parsed = [parse_detail(o.outputs[0].text) for o in first]
        retry = [i for i, p in enumerate(parsed) if p is None]  # re-query once (sampled), then `unknown`
        if retry:
            second = llm.generate([dprompts[i] for i in retry], SamplingParams(temperature=0.3, max_tokens=96, seed=1))
            for i, o in zip(retry, second):
                parsed[i] = parse_detail(o.outputs[0].text)
        for r, p in zip(need, parsed):
            r["detail_parse_ok"] = p is not None
            if p:
                step, old, new = p
                r.update({"detail_step": step, "detail_old": old, "detail_new": new, "detail_prompt_version": dver,
                          "false_rev": bool(step == r["n_prefix_steps"] and not grade.is_correct(new, old)),
                          "earlier_step_rev": bool(step < r["n_prefix_steps"])})
        n_detail = len(need)
    n = common.write_jsonl(a.out, rows)
    common.write_meta(a.out + ".meta.json", "judge",
                      {"model": model_id, "prompt_file": a.prompt, "prompt_version": pver, "prompt_hash": phash,
                       "detail_prompt_hash": dhash, "max_prefix_chars": 2500, "n_units": 3, "temperature": 0.0,
                       "max_tokens": 512 if harmony else 1, "mode": a.mode, "arm_blind": True},
                      {"engine_seed": 0}, inputs=[a.prefixes] + list(a.scored), outputs=[a.out],
                      extra={"n_records": n, "gpus": gpus, "n_detail_calls": n_detail,
                             "parse_failures": sum(not r["parse_ok"] for r in rows),
                             "gpu_hours": round((time.time() - t0) * max(1, len(gpus)) / 3600, 3)}, t_start=t0)
    print("[judge] %d labelled (%d detail calls) in %.0fs" % (n, n_detail, time.time() - t0))


if __name__ == "__main__":
    main()
