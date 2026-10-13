"""Answer extraction and grading (math-verify, with a normalised string fallback).

Graders read both the `Answer: ...` convention of the OLMo RL-Zero prompt and `\\boxed{...}`
(reasoning-cues README: graders must accept both). Needs `math_verify` (env `rl`, or `pip install math-verify`).
"""
from __future__ import annotations

import re
from typing import List, Optional

_ANSWER_RE = re.compile(r"(?im)^[ \t>*_#-]*\**answer\**\s*:\s*\**\s*(.+?)\s*$")


def last_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} with balanced braces."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def _clean(ans: str) -> str:
    a = ans.strip()
    b = last_boxed(a)
    if b is not None:
        a = b
    a = a.strip().strip("$").strip()
    a = re.sub(r"^\\\(|\\\)$", "", a).strip()
    a = re.sub(r"^\\\[|\\\]$", "", a).strip()
    a = a.rstrip(".").strip()
    return a


def extract_answer(text: str) -> Optional[str]:
    """Final answer of a response, plan §4 grader rule: the last \\boxed{} first (RL-Zero-Math commits with \\boxed{}),
    the last `Answer:` line as fallback; None if absent."""
    b = last_boxed(text)
    if b is not None and _clean(b):
        return _clean(b)
    ms = list(_ANSWER_RE.finditer(text))
    if ms:
        a = _clean(ms[-1].group(1))
        if a:
            return a
    return None


def _norm(s: str) -> str:
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return re.sub(r"\s+", "", s).lower().rstrip(".")


def is_correct(pred: Optional[str], gold: str) -> bool:
    if pred is None or pred == "":
        return False
    if _norm(pred) == _norm(gold):
        return True
    try:
        from math_verify import parse, verify

        g = parse("$" + gold + "$")
        p = parse("$" + pred + "$")
        return bool(verify(g, p))
    except Exception:
        return False


# ---- MMLU-Pro answer extraction: VERBATIM logic of TIGER-AI-Lab/MMLU-Pro evaluate_from_local.py (extract_answer -> extract_again -> extract_final); only the debug prints are dropped
def mc_extract_answer(text: str) -> Optional[str]:
    import re
    m = re.search(r"answer is \(?([A-J])\)?", text)
    if m:
        return m.group(1)
    m = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if m:
        return m.group(1)
    m = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return m.group(0) if m else None


def grade_texts(texts: List[str], golds: List[str]) -> List[dict]:
    out = []
    for t, g in zip(texts, golds):
        a = extract_answer(t)
        out.append({"pred": a, "correct": is_correct(a, g)})
    return out


def main():
    """CLI: add pred/correct to a gen jsonl and drop token ids/logprobs -> light graded jsonl."""
    import argparse

    from . import common, prompts

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mc", action="store_true", help="multiple-choice letter grading (MMLU-Pro official extraction)")
    a = ap.parse_args()
    gold = {p["pid"]: p["gold"] for p in prompts.load_problems(a.data)}
    for g in a.gen:
        rows = common.read_jsonl(g)
        for r in rows:
            r.pop("token_ids", None)
            r.pop("logprobs", None)
            r["pred"] = mc_extract_answer(r["text"]) if a.mc else extract_answer(r["text"])
            r["correct"] = (r["pred"] == gold[r["pid"]]) if a.mc else is_correct(r["pred"], gold[r["pid"]])
            r["gold"] = gold[r["pid"]]
        name = g.rsplit("/", 1)[-1].replace(".jsonl", "") + ".graded.jsonl.gz"
        out = a.out_dir.rstrip("/") + "/" + name
        n = common.write_jsonl(out, rows)
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        print("[grade] %s: %d rows, acc=%.3f" % (name, n, acc))


if __name__ == "__main__":
    main()
