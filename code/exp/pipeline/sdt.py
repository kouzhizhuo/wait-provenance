"""E2: signal-detection analysis of bleaching with INJECTED errors (EXPERIMENT_PLAN §2.3, STATS_PLAN C2).

Per checkpoint t and problem p: take an on-policy trace of t, cut after a sentence k whose steps 1..k are known-correct
(the source trace is correct, finished, and has no content revision event up to k), and build
    clean      steps 1..k as written,
    corrupted  the same, with ONE numeric error injected into step k (rule-based: number +-1, adjacent-digit swap, sign flip;
               the value must change and the edit is <= 6 characters; the 'drop a term' kind of the plan needs the judge
               template and is not used in P0).
Both prefixes are continued by the SAME checkpoint (n = 4, short: 300 tokens are enough for 2 sentences).
MARK = a form of F occurs in the first s = 2 sentences of the continuation (regex, no judge).
HR = P(MARK | corrupted), FA = P(MARK | clean); d' and c with the log-linear correction (stats.dprime); cluster bootstrap over problems.

    python -m exp.pipeline.sdt prefixes --gen <EMAIL> --data MATH-500 --n_problems 300 --out sdt_prefixes.jsonl
    python -m exp.pipeline.insert continue --model <ckpt> --tag S1@1900 --prefixes sdt_prefixes.jsonl --arms none --samples 4 --max_tokens 300 --out sdt_conts.jsonl.gz
    python -m exp.pipeline.sdt score --conts sdt_conts_S1@*.jsonl.gz --out sdt.json
"""
from __future__ import annotations

import argparse
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import common, insert as ins, markers as mk, prompts, stats as st

_NUM = re.compile(r"(?<![\w.\\])-?\d+(?:\.\d+)?(?![\w])")
_EQ_RHS = re.compile(r"=\s*(-?\d+(?:\.\d+)?)(?![\w.^{])")
_IS_RHS = re.compile(r"\b(?:is|are|equals|gives|get|gets|becomes)\s+(-?\d+(?:\.\d+)?)(?![\w.^{])", re.I)
S_WINDOW = 2


def edit_distance(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[-1]


def _perturb(num: str, kind: str) -> Optional[str]:
    neg = num.startswith("-")
    core = num.lstrip("-")
    if kind == "pm1":
        if "." in core:
            ip, fp = core.split(".")
            v = str(int(ip) + 1)
            out = v + "." + fp
        else:
            out = str(int(core) + 1) if int(core) != 0 else "1"
        return ("-" if neg else "") + out
    if kind == "swap":
        digs = [i for i in range(len(core) - 1) if core[i].isdigit() and core[i + 1].isdigit() and core[i] != core[i + 1]]
        if not digs:
            return None
        i = digs[-1]
        out = core[:i] + core[i + 1] + core[i] + core[i + 2:]
        if out.startswith("0") and len(out) > 1 and "." not in out:
            return None
        return ("-" if neg else "") + out
    if kind == "sign":
        return core if neg else "-" + core
    return None


def corrupt_sentence(sentence: str, rng: random.Random) -> Optional[Dict[str, str]]:
    """One numeric error in a computed RESULT: the number after the last '=' or after 'is/are/equals/gives/get'.
    Sentences without such a result are not corruptible (incidental numbers such as 'between 0 and 2 pi' are not touched)."""
    m = list(_EQ_RHS.finditer(sentence)) or list(_IS_RHS.finditer(sentence))
    if not m:
        return None
    target, span = m[-1].group(1), (m[-1].start(1), m[-1].end(1))
    kinds = ["pm1", "swap", "sign"] if _EQ_RHS.search(sentence) else ["pm1", "swap"]
    rng.shuffle(kinds)
    for kind in kinds:
        new = _perturb(target, kind)
        if new is None or new == target:
            continue
        out = sentence[:span[0]] + new + sentence[span[1]:]
        if edit_distance(sentence, out) <= 6:
            return {"sentence": out, "kind": kind, "old": target, "new": new}
    return None


def _build_pair(r: dict, rng: random.Random, render_prompt) -> Optional[List[dict]]:
    """Clean + corrupted prefix records from one correct finished trace, or None if no corruptible cut exists."""
    sents = mk.split_sentences(r["text"])
    n = len(sents)
    if n < 6:
        return None
    events = rv_events(sents)
    first_event = min((i for i, e in enumerate(events) if e), default=n + 1)
    ans = next((i for i, s in enumerate(sents) if re.search(r"answer\s*:", s.text, re.I)), n)
    ks = [k for k in range(1, n - 1) if 0.2 <= k / n <= 0.8 and k + 1 < ans and k < first_event
          and sents[k].text.rstrip()[-1:] in ".!?$:)" and re.search(r"\d", sents[k].text)]
    rng.shuffle(ks)
    for k in ks:
        c = corrupt_sentence(sents[k].text, rng)
        if c is None:
            continue
        clean = r["text"][:sents[k].end]
        corr = r["text"][:sents[k].start] + c["sentence"]
        base = {"pid": r["pid"], "sid": r["sid"], "k": k, "n_sent": n, "sep": r["text"][sents[k].end:sents[k + 1].start],
                "gold": r["gold"], "problem": r["problem"], "prompt": render_prompt(r["problem"]), "src_ckpt": r["ckpt"]}
        return [{**base, "prefix_id": "clean:%d" % r["pid"], "src": "clean", "prefix": clean,
                 "prefix_hash": common.git_blob_hash(clean.encode())},
                {**base, "prefix_id": "corrupted:%d" % r["pid"], "src": "corrupted", "prefix": corr,
                 "corruption": {kk: c[kk] for kk in ("kind", "old", "new")}, "prefix_hash": common.git_blob_hash(corr.encode())}]
    return None


def select(rows: List[dict], n_problems: int, seed: int, render_prompt) -> List[dict]:
    """Two prefix records (clean, corrupted) per selected problem, from a correct finished trace."""
    by_pid: Dict[int, List[dict]] = {}
    for r in rows:
        by_pid.setdefault(r["pid"], []).append(r)
    out: List[dict] = []
    for pid in sorted(by_pid):
        if len(out) >= 2 * n_problems:
            break
        rng = random.Random(common.derive_seed(seed, "sdt", pid))
        samples = [r for r in by_pid[pid] if r.get("correct") and r.get("finish") == "stop"]
        rng.shuffle(samples)
        for r in samples:
            pair = _build_pair(r, rng, render_prompt)
            if pair:
                out += pair
                break
    return out


def rv_events(sents) -> List[Optional[str]]:
    from . import revisions as rv

    return rv.detect_events([mk.strip_initial_forms(s.text)[0] for s in sents])


def mark_in_window(continuation: str, s: int = S_WINDOW) -> Tuple[bool, bool]:
    """(any form of F, `wait`) within the first s sentences of the continuation."""
    sents = mk.split_sentences(continuation)[:s]
    if not sents:
        return (False, False)
    text = continuation[:sents[-1].end]
    ms = mk.find_markers(text)
    return (bool(ms), any(m.form == "wait" for m in ms))


def sdt_from_rows(rows: List[dict], key: str = "mark_any", B: int = 2000, seed: int = 20261012) -> dict:
    """Cluster bootstrap over problems of HR, FA, d', c from continuation rows with `src` in {clean, corrupted}."""
    pids = sorted({r["pid"] for r in rows})
    pos = {p: i for i, p in enumerate(pids)}
    M = np.zeros((len(pids), 4))  # hits, n_corr, fas, n_clean
    for r in rows:
        i = pos[r["pid"]]
        if r["src"] == "corrupted":
            M[i, 0] += bool(r[key])
            M[i, 1] += 1
        else:
            M[i, 2] += bool(r[key])
            M[i, 3] += 1

    def f(t: dict) -> Dict[str, float]:
        d = st.dprime(t["h"], t["nc"], t["f"], t["nn"])
        return {"HR": d["HR"], "FA": d["FA"], "dprime": d["dprime"], "criterion": d["c"]}

    names = ["h", "nc", "f", "nn"]
    res = st.bootstrap_stat(M, names, f, B=B, seed=seed)
    res["n_error"] = {"est": float(M[:, 1].sum())}
    res["n_clean"] = {"est": float(M[:, 3].sum())}
    return res


def cmd_prefixes(a):
    t0 = time.time()
    rows = common.read_jsonl(a.gen)
    probs = {p["pid"]: p for p in prompts.load_problems(a.data)}
    for r in rows:
        r["gold"], r["problem"] = probs[r["pid"]]["gold"], probs[r["pid"]]["problem"]
    tok = None
    if a.template != "olmo_rlzero":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=common.trust_remote())
    pre = select(rows, a.n_problems, a.seed, lambda q: prompts.render(a.template, q, tok))
    n = common.write_jsonl(a.out, pre)
    kinds: Dict[str, int] = {}
    for p in pre:
        if p["src"] == "corrupted":
            kinds[p["corruption"]["kind"]] = kinds.get(p["corruption"]["kind"], 0) + 1
    common.write_meta(a.out + ".meta.json", "sdt_prefixes", {"gen": a.gen, "n_problems": a.n_problems, "template": a.template,
                      "error_kinds": ["pm1", "swap", "sign"], "max_edit_distance": 6, "window_sentences": S_WINDOW},
                      {"seed": a.seed}, inputs=[a.gen, a.data], outputs=[a.out],
                      extra={"n_problems_selected": n // 2, "corruption_kinds": kinds}, t_start=t0)
    print("[sdt prefixes] %d problems (%s)" % (n // 2, kinds))


def cmd_score(a):
    t0 = time.time()
    out_ck = []
    for f in a.conts:
        rows = common.read_jsonl(f)
        pol = rows[0]["policy"]
        for r in rows:
            r["mark_any"], r["mark_wait"] = mark_in_window(r["text"])
        ck = pol.split("@")[1] if "@" in pol else pol
        ent = {"ckpt": ck, "step": None}
        for key, name in (("mark_any", ""), ("mark_wait", "_wait")):
            res = sdt_from_rows(rows, key, B=a.B)
            for k, dst in (("HR", "hr"), ("FA", "fa"), ("dprime", "dprime"), ("criterion", "criterion")):
                ent[dst + name] = {x: res[k][x] for x in ("est", "lo", "hi")}
            ent["n_error"], ent["n_clean"] = int(res["n_error"]["est"]), int(res["n_clean"]["est"])
        out_ck.append(ent)
    steps = {"base": 0}
    for e in out_ck:
        e["step"] = steps.get(e["ckpt"], int(e["ckpt"]) if e["ckpt"].isdigit() else None)
    out_ck.sort(key=lambda e: (e["step"] is None, e["step"]))
    common.write_json(a.out, {"series": a.series, "pilot": a.pilot, "outcome": "MARK = any form of F (and `wait` only: *_wait keys) in the first 2 sentences",
                              "checkpoints": out_ck})
    if a.keys_series:   # RESULTS_KEYS h2.* (hr, fa in %, dprime, crit; stage aliases from --stage_map)
        from . import keys as K

        sm = {"base": "base", **(dict(kv.split("=") for kv in a.stage_map.split(",")) if a.stage_map else {})}
        ex = K.Exporter()
        for e in out_ck:
            for stage in K.stage_keys(e["step"], e["ckpt"], sm):
                for q, kk, kind in (("hr", "hr", "pct"), ("fa", "fa", "pct"), ("dprime", "dprime", "dec2"), ("criterion", "crit", "dec2"),
                                    ("hr_wait", "hr_wait", "pct"), ("fa_wait", "fa_wait", "pct")):
                    ex.put_est("h2.%s.%s.%s" % (a.keys_series, stage, kk), e[q], kind, "%s::checkpoints.%s.%s" % (a.out, e["ckpt"], q))
        common.write_json(str(a.out).replace(".json", "_keys.json"), {"entries": ex.entries})
    common.write_meta(a.out + ".meta.json", "sdt_score", {"conts": a.conts, "B": a.B}, {"rng_seed": 20261012}, inputs=a.conts, outputs=[a.out], t_start=t0)
    print("[sdt score] %d checkpoints -> %s" % (len(out_ck), a.out))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prefixes")
    p.add_argument("--gen", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--n_problems", type=int, default=300)
    p.add_argument("--template", default="olmo_rlzero")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prefixes)
    s = sub.add_parser("score")
    s.add_argument("--conts", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--series", default="S1")
    s.add_argument("--pilot", action="store_true")
    s.add_argument("--B", type=int, default=2000)
    s.add_argument("--keys_series", default=None)
    s.add_argument("--stage_map", default=None)
    s.set_defaults(fn=cmd_score)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
