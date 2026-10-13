"""Prior keyword measures of marker change, re-run on OUR E1 generations (plan §5b; never copied). CPU only.

  r1kw      DeepSeek-R1 (Nature suppl. C.2 / arXiv 2501.12948v2 App. C.2) reflection-word count: the 10 words
            wait, mistake, however, but, retry, error, verify, wrong, evaluate, check (case-insensitive whole words; per 1k tokens).
            r1wait = the count of "wait" alone (their Fig. 9b).
  orz       Open-Reasoner-Zero (check_reflection_pattern): the 5 CASE-SENSITIVE regexes  wait,  recheck[,\\s]  retry  alternatively,  however,
            per-response score = number of matches (mean per response); orz_reflective = share of responses with >= 1 match (the binary rule is UNVERIFIED).
            Because of case sensitivity "Wait," / "However," at a sentence start are NOT counted: the measure is orthography-dependent.
  simplerl  SimpleRL-Zoo collect_results.py: lower-cased SUBSTRING counts of recheck, rethink, try again, wait, alternatively, retry, however; mean per response.
Cluster bootstrap over problems (B = 2000). Judged-behaviour baselines (SimpleRL-Zoo / Gandhi et al. judge prompts) are separate (they need a judge).

    python -m exp.pipeline.baselines --spec spec.json --out baseline_measures.json
spec.json: [{"dir": "<.../S1/gen>", "series": "S1", "keys_series": "s1", "ckpts": ["base", "100", ..., "final"], "steps": {"base": 0, ...}, "stage_map": {"early": "100", ...}}]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import common, keys as K, stats as st

R1_WORDS = ["wait", "mistake", "however", "but", "retry", "error", "verify", "wrong", "evaluate", "check"]
_R1 = re.compile(r"\b(" + "|".join(R1_WORDS) + r")\b", re.I)
ORZ_PATTERNS = [r"wait,", r"recheck[,\s]", r"retry", r"alternatively,", r"however,"]
_ORZ = [re.compile(p) for p in ORZ_PATTERNS]
SIMPLERL_KW = ["recheck", "rethink", "try again", "wait", "alternatively", "retry", "however"]
NAMES = ["tokens", "n_resp", "r1", "r1wait", "orz", "orz_refl", "simplerl"]


def response_counts(text: str, n_tokens: int) -> List[float]:
    low = text.lower()
    r1 = _R1.findall(text)
    orz = sum(len(p.findall(text)) for p in _ORZ)
    return [float(n_tokens), 1.0, float(len(r1)), float(sum(1 for w in r1 if w.lower() == "wait")), float(orz), float(orz > 0),
            float(sum(low.count(k) for k in SIMPLERL_KW))]


def measures(t: Dict[str, float]) -> Dict[str, float]:
    return {"r1kw": 1000.0 * t["r1"] / t["tokens"], "r1wait": 1000.0 * t["r1wait"] / t["tokens"], "orz": t["orz"] / t["n_resp"],
            "orz_reflective": t["orz_refl"] / t["n_resp"], "simplerl": t["simplerl"] / t["n_resp"]}


def for_file(path: Path, B: int = 2000, seed: int = 20261012) -> Dict[str, Dict[str, float]]:
    rows = common.read_jsonl(path)
    pids = sorted({r["pid"] for r in rows})
    pos = {p: i for i, p in enumerate(pids)}
    M = np.zeros((len(pids), len(NAMES)))
    for r in rows:
        M[pos[r["pid"]]] += response_counts(r["text"], r["n_tokens"])
    return st.bootstrap_stat(M, NAMES, measures, B=B, seed=seed)


QTY = {"r1kw": ("r1kw", "dec2"), "r1wait": ("r1wait", "dec2"), "orz": ("orz", "dec2"), "orz_reflective": ("orzrefl", "pct"), "simplerl": ("simplerl", "dec2")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    out: dict = {"definitions": {"r1kw": R1_WORDS, "orz_patterns": ORZ_PATTERNS, "simplerl_keywords": SIMPLERL_KW}, "series": {}}
    ex = K.Exporter()
    for s in spec:
        res = {}
        for ck in s["ckpts"]:
            f = Path(s["dir"]) / ("%s@%s.graded.jsonl.gz" % (s["series"], ck))
            if not f.exists():
                continue
            res[ck] = for_file(f, a.B)
            stage_keys = K.stage_keys(s["steps"].get(ck), ck, {"base": "base", **s.get("stage_map", {})})
            for m, (q, kind) in QTY.items():
                for stage in stage_keys:
                    ex.put_est("bm.%s.%s.%s" % (s["keys_series"], stage, q), res[ck][m], kind, "%s::%s" % (f, m))
        # rank trend vs step over checkpoints that have a numeric step
        steps = [(ck, s["steps"][ck]) for ck in res if s["steps"].get(ck) is not None]
        tr = {}
        for m, (q, kind) in QTY.items():
            if len(steps) >= 3:
                rho, p = st.spearman([x[1] for x in steps], [res[x[0]][m]["est"] for x in steps])
                tr[m] = {"rho": rho, "p": p, "n_points": len(steps)}
                ex.put("bm.%s.trend.rho_%s" % (s["keys_series"], q), rho, "dec2", "spearman over %d checkpoints" % len(steps))
        out["series"][s["series"]] = {"checkpoints": res, "trends": tr}
    common.write_json(a.out, out)
    common.write_json(str(a.out).replace(".json", "_keys.json"), {"entries": ex.entries})
    print("[baselines] %d series, %d keys" % (len(out["series"]), len(ex.entries)))


if __name__ == "__main__":
    main()
