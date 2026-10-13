"""D43 seeded-marker arms: W (as generated), X (every sentence-initial opener -> "Next,"), D (every opener deleted).

Pure string logic (unit-tested in test_seeded_arms.py) + a `build` CLI that turns raw OLMo-3 RL-Zero generations on MATH train into
the pool, the nested N in {512, 64} selections and the three arms.  Sentence boundaries = the E3 splitter (exp.pipeline.markers.split_sentences).

Rules (D43 + lead rulings of 2026-10-02):
  * opener      = a RUN of consecutive sentence-initial wait forms ("Wait,", "Wait, wait,", "Wait. Wait,"); a run counts as ONE opener.
                  Wait form = [Ww]ait + punctuation, followed by spaces and a non-space on the same line.
  * X           = the whole run replaced by a single "Next,".   D = the whole run and the spaces after it deleted; the next word keeps its case.
  * eligible    = >= 1 run, every run contains an exact "Wait," form, and EVERY case-insensitive word 'wait' of the trace lies inside a run
                  (so X and D contain no 'wait' at all).  "but wait", "wait for", mid-sentence wait, "Wait" before a newline => trace dropped.
  * same traces for W/X/D; N=64 is the prefix of the shuffled N=512 selection; distinct problems preferred.
Usage:  python seeded_arms.py build --gen raw1.jsonl [raw2.jsonl] --out_dir DIR [--n_high 512 | env N_HIGH] [--fallback_min 256 | env FALLBACK_MIN] [--tokenizer QWEN_DIR] [--reward_check]
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[3], Path(os.environ.get("ARR_ROOT", "/<CLUSTER>/<USER>/arr_oct")) / "code" / "P1"):   # repo root holding exp/pipeline
    if (_p / "exp" / "pipeline").is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from exp.pipeline.markers import split_sentences  # noqa: E402

OPENER = "Wait,"
REPL = "Next,"
WAIT_WORD = re.compile(r"\bwait\b", re.I)
_ELEM = re.compile(r"([Ww]ait)([,.!?]*)[ \t]+(?=\S)")      # one wait form + its trailing spaces; the next char must be on the same line
_W = re.compile(r"\w+")


def find_runs(text: str) -> Tuple[List[dict], int]:
    """Sentence-initial runs of wait forms ("Wait,", "Wait, wait,", "Wait. Wait,") = ONE opener each (lead ruling on D43).
    Returns (runs, n_uncovered) where each run = {a: start, e: end of the last wait form incl. its punctuation, k: end of the spaces
    after it, n_elem, has_comma: contains an exact "Wait," form} and n_uncovered = number of wait words (case-insensitive, \\bwait\\b)
    that lie outside every run (non-initial 'but wait', 'wait for', a 'Wait' followed by a newline, ...)."""
    runs, cursor = [], 0
    for st in sorted({s.start for s in split_sentences(text)}):
        if st < cursor:
            continue
        pos, elems = st, []
        while True:
            m = _ELEM.match(text, pos)
            if not m:
                break
            elems.append(m)
            pos = m.end()
        if elems:
            runs.append({"a": st, "e": elems[-1].end(2), "k": pos, "n_elem": len(elems),
                         "has_comma": any(m.group(1) == "Wait" and m.group(2) == "," for m in elems)})
            cursor = pos
    n_uncovered = len(WAIT_WORD.findall(text)) - sum(r["n_elem"] for r in runs)
    return runs, n_uncovered


def make_arms(text: str):
    """(arms, reason). arms = {"W","X","D","n_openers"} or None with reason in {no_wait_opener, non_initial_wait, run_without_comma}.
    Eligible trace (lead ruling): >= 1 run, EVERY wait word of the trace lies in a run, and every run contains an exact "Wait," form.
    X: each run -> one "Next,"; D: each run (and the spaces after it) deleted, next word keeps its case."""
    runs, unc = find_runs(text)
    if not runs:
        return None, "no_wait_opener"
    if unc:
        return None, "non_initial_wait"
    if not all(r["has_comma"] for r in runs):
        return None, "run_without_comma"
    x, d = text, text
    for r in reversed(runs):
        x = x[:r["a"]] + REPL + x[r["e"]:]
        d = d[:r["a"]] + d[r["k"]:]
    return {"W": text, "X": x, "D": d, "n_openers": len(runs)}, None


def _words(s: str) -> List[str]:
    return _W.findall(s)


def verify_arms(w: str, x: str, d: str) -> Dict[str, int]:
    """Exact verification (AssertionError otherwise): (i) X and D contain no 'wait' word; (ii) W->X and W->D reconstruct exactly from the W runs;
    (iii) independent word-level diff: W->D = k deletions, every deleted word is 'wait'; W->X = k replacements of wait-words by the single word
    'Next'; (iv) X has exactly k more 'Next,' than W."""
    runs, unc = find_runs(w)
    k = len(runs)
    assert k >= 1 and unc == 0, (k, unc)
    assert not WAIT_WORD.search(x) and not WAIT_WORD.search(d), "residual wait word in X or D"
    xr, dr, pos = [], [], 0
    for r in runs:
        xr += [w[pos:r["a"]], REPL, w[r["e"]:r["k"]]]
        dr.append(w[pos:r["a"]])
        pos = r["k"]
    xr.append(w[pos:]); dr.append(w[pos:])
    assert "".join(xr) == x and "".join(dr) == d, "arms differ from the run-wise reconstruction"
    tw, tx, td = _words(w), _words(x), _words(d)
    ops = [o for o in difflib.SequenceMatcher(None, tw, td, autojunk=False).get_opcodes() if o[0] != "equal"]
    assert len(ops) == k, "W->D: %d edits, expected %d" % (len(ops), k)
    for tag, i1, i2, j1, j2 in ops:
        assert tag == "delete" and all(t.lower() == "wait" for t in tw[i1:i2]), (tag, tw[i1:i2])
    ops = [o for o in difflib.SequenceMatcher(None, tw, tx, autojunk=False).get_opcodes() if o[0] != "equal"]
    assert len(ops) == k, "W->X: %d edits, expected %d" % (len(ops), k)
    for tag, i1, i2, j1, j2 in ops:
        assert tag == "replace" and all(t.lower() == "wait" for t in tw[i1:i2]) and tx[j1:j2] == ["Next"], (tag, tw[i1:i2], tx[j1:j2])
    assert x.count(REPL) == w.count(REPL) + k
    return {"n_openers": k}


def arm_stats(texts: List[str]) -> Dict[str, float]:
    """Counts over a list of traces: sentence-initial 'Wait,'/'Next,', all case-insensitive word 'wait'/'next'."""
    st = {"n": len(texts), "init_wait_comma": 0, "init_next_comma": 0, "wait_word_any": 0, "next_word_any": 0}
    for t in texts:
        for s in split_sentences(t):
            st["init_wait_comma"] += s.text.startswith("Wait,")
            st["init_next_comma"] += s.text.startswith("Next,")
        st["wait_word_any"] += len(WAIT_WORD.findall(t))
        st["next_word_any"] += len(re.findall(r"\bnext\b", t, re.I))
    return st


def md5_file(p) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _wjson(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build(a) -> dict:
    from exp.pipeline.grade import extract_answer, is_correct
    rows = [json.loads(l) for f in a.gen for l in open(f, encoding="utf-8") if l.strip()]
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rep = {"n_generated": len(rows), "drop": {}, "dropped_verify_pids": [], "reward_check": bool(a.reward_check)}
    def drop(k):
        rep["drop"][k] = rep["drop"].get(k, 0) + 1
    pool = []
    rep["n_correct_with_initial_wait_comma"] = 0      # for reference: traces that would pass the ORIGINAL D43 rule (>=1 sentence-initial "Wait,")
    for r in rows:
        if r["finish"] != "stop":
            drop("not_finished"); continue
        t = r["text"]
        if "\\boxed" not in t:
            drop("no_boxed"); continue
        if not is_correct(extract_answer(t), r["gold"]):
            drop("incorrect"); continue
        rep["n_correct_with_initial_wait_comma"] += any(sn.text.startswith(OPENER) for sn in split_sentences(t))
        arms, why = make_arms(t)
        if arms is None:
            drop(why); continue
        if tok is not None:
            L = [len(tok(arms[k], add_special_tokens=False).input_ids) + 1 for k in "WXD"]   # +1 = EOS
            r["qwen_tokens"] = {"W": L[0], "X": L[1], "D": L[2]}
            if max(L) > a.max_resp_tokens:
                drop("too_long_qwen_tokens"); continue
            ptxt = tok.apply_chat_template(r["prompt_messages"], add_generation_prompt=True, tokenize=False)
            if len(tok(ptxt, add_special_tokens=False).input_ids) > a.max_prompt_tokens:
                drop("prompt_too_long"); continue
        try:
            verify_arms(arms["W"], arms["X"], arms["D"])
        except AssertionError as e:
            drop("verify_failed"); rep["dropped_verify_pids"].append([r["pid"], str(e)[:120]]); continue
        r["arms"] = arms
        pool.append(r)
    if a.reward_check:   # the REF reward function itself (verl Math-Verify wrapper) must also score every W trace 1.0
        from verl.utils.reward_score import math_verify
        keep = []
        for r in pool:
            if float(math_verify.compute_score(r["text"], r["gold"])) == 1.0:
                keep.append(r)
            else:
                drop("verl_reward_not_1")
        pool = keep
    rep["n_pool"] = len(pool)
    rep["n_pool_distinct_problems"] = len({r["pid"] for r in pool})
    rnd = random.Random(a.seed)
    order = sorted(pool, key=lambda r: (r["pid"], r.get("sid", 0))); rnd.shuffle(order)    # one fixed order; N=64 is the prefix of N=512 (nested)
    seen, uniq, rest = set(), [], []
    for r in order:                                                                         # distinct problems first, repeats only as a fallback
        (rest if r["pid"] in seen else uniq).append(r); seen.add(r["pid"])
    order = uniq + rest
    out = Path(a.out_dir); (out / "arms").mkdir(parents=True, exist_ok=True)
    rep["pool_stats"] = arm_stats([r["text"] for r in pool])
    _wjson(out / "pool.jsonl", [{k: v for k, v in r.items() if k != "arms"} for r in pool])
    ns = [a.n_high, a.n_low]
    rep["requested_N"] = list(ns)
    print("POOL_SIZE=%d (distinct problems %d); required N_high=%d N_low=%d" % (len(order), rep["n_pool_distinct_problems"], a.n_high, a.n_low), flush=True)
    if len(order) < a.n_high:
        if a.fallback_min and len(order) >= a.fallback_min:       # lead's fallback (D43 amendment): largest N >= fallback_min the pool supports, same for W/X/D
            new_high = (len(order) // a.n_low) * a.n_low
            rep["fallback"] = {"triggered": True, "n_high_requested": a.n_high, "n_high_used": new_high, "pool": len(order), "fallback_min": a.fallback_min,
                               "rule": "N_high = pool size rounded down to a multiple of N_low (64); low dose unchanged; identical for W/X/D"}
            print("FALLBACK: N_high %d -> %d (pool %d)" % (a.n_high, new_high, len(order)), flush=True)
            ns = [new_high, a.n_low]
        else:
            json.dump(rep, open(out / "build_report.json", "w"), indent=1)
            msg = "BUILD FAILED: eligible pool = %d (%d distinct problems) < N_high = %d (fallback_min=%s); report: %s" % (
                len(order), rep["n_pool_distinct_problems"], a.n_high, a.fallback_min, out / "build_report.json")
            print(msg, flush=True)
            raise SystemExit(msg)
    a.n = ns
    rep["files"], rep["arms"] = {}, {}
    md5s = []
    for N in a.n:
        sel = order[:N]
        for arm in "WXD":
            recs = [{"pid": r["pid"], "prompt": r["prompt_messages"], "response": r["arms"][arm], "arm": arm, "N": N,
                     "gold": r["gold"], "n_openers": r["arms"]["n_openers"]} for r in sel]
            f = out / "arms" / ("%s%d.jsonl" % (arm, N)); _wjson(f, recs)
            st = arm_stats([r["response"] for r in recs])
            if tok is not None:
                st["mean_qwen_tokens"] = sum(r["qwen_tokens"][arm] for r in sel) / N
                st["mean_qwen_tokens_delta_vs_W"] = sum(r["qwen_tokens"][arm] - r["qwen_tokens"]["W"] for r in sel) / N
                st["max_qwen_tokens"] = max(r["qwen_tokens"][arm] for r in sel)
            st["mean_chars"] = sum(len(r["response"]) for r in recs) / N
            st["n_openers_total"] = sum(r["n_openers"] for r in recs)
            rep["arms"]["%s%d" % (arm, N)] = st
            rep["files"]["%s%d.jsonl" % (arm, N)] = {"md5": md5_file(f), "n": N}
            md5s.append("%s  %s" % (md5_file(f), f))
        rep["files"]["pids_%d" % N] = [r["pid"] for r in sel] if N <= 64 else hashlib.md5(",".join(str(r["pid"]) for r in sel).encode()).hexdigest()
    rep["files"]["pool.jsonl"] = {"md5": md5_file(out / "pool.jsonl"), "n": len(pool)}
    md5s.append("%s  %s" % (md5_file(out / "pool.jsonl"), out / "pool.jsonl"))
    open(out / "MD5SUMS", "w").write("\n".join(md5s) + "\n")                                   # written here, before any caller looks for it
    json.dump(rep, open(out / "build_report.json", "w"), indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gen", nargs="+", required=True); b.add_argument("--out_dir", required=True)
    b.add_argument("--n_high", type=int, default=int(os.environ.get("N_HIGH", 512))); b.add_argument("--n_low", type=int, default=64)
    b.add_argument("--fallback_min", type=int, default=int(os.environ.get("FALLBACK_MIN", 0)), help="0 = off. If the pool < n_high but >= this, N_high := pool rounded down to a multiple of n_low (lead fallback, use only after the Oct 4 00:00 CST deadline)")
    b.add_argument("--n", nargs="+", type=int, default=None, help="legacy: --n HIGH LOW (the already-queued job 407194 passes --n 512 64)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tokenizer", default=None, help="Qwen2.5-Math-1.5B dir: enforces response <= --max_resp_tokens (default 3072 = REF max_response_length)")
    b.add_argument("--max_resp_tokens", type=int, default=3072); b.add_argument("--max_prompt_tokens", type=int, default=1024)
    b.add_argument("--reward_check", action="store_true", help="env rl only: require verl math_verify.compute_score == 1 on W")
    a = ap.parse_args()
    if a.n:
        a.n_high, a.n_low = max(a.n), min(a.n)
    rep = build(a)
    print(json.dumps({k: rep[k] for k in ("n_generated", "drop", "n_correct_with_initial_wait_comma", "n_pool", "n_pool_distinct_problems", "arms")}, indent=1))


if __name__ == "__main__":
    main()
