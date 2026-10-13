"""MMLU-Pro panel data (expansion round; replaces gated GPQA-Diamond, D-note in BROADEN_EXPERIMENTS.md).
Sample: 500 test questions, 125 each from the categories physics, chemistry, biology, engineering, random.Random(20261012) over question_id-sorted rows (fixed before any result).
Prompt: the OFFICIAL 5-shot CoT prompt of TIGER-AI-Lab/MMLU-Pro (evaluate_from_local.py: cot_prompt_lib/initial_prompt.txt + the first 5 validation examples of the question's category with their cot_content,
then 'Question:\\n<q>\\nOptions:\\nA. ...\\nAnswer: Let's think step by step.'); 'N/A' options dropped as in preprocess().  Rendered with template mmlupro_raw (raw completion, stop 'Question:'); graded with grade.py --mc.
    python3 exp/build_mmlupro.py <parquet_dir> <out.jsonl>        # parquet_dir holds test-*.parquet and validation-*.parquet from TIGER-Lab/MMLU-Pro (data/)"""
import glob
import json
import random
import sys
from pathlib import Path

import pandas as pd

CATS = ["physics", "chemistry", "biology", "engineering"]
SEED = 20261012
LET = "ABCDEFGHIJKLMNOP"
HERE = Path(__file__).resolve().parent


def clean(r):
    r = dict(r)
    r["options"] = [o for o in list(r["options"]) if o != "N/A"]
    return r


def fmt(ex, with_answer):
    s = "Question:\n" + ex["question"] + "\nOptions:\n"
    for i, o in enumerate(ex["options"]):
        s += "{}. {}\n".format(LET[i], o)
    if with_answer:
        return s + ex["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.") + "\n\n"
    return s + "Answer: Let's think step by step."


def main():
    src, out = Path(sys.argv[1]), sys.argv[2]
    test = [clean(r) for r in pd.read_parquet(glob.glob(str(src / "test-*.parquet"))[0]).to_dict("records")]
    val = [clean(r) for r in pd.read_parquet(glob.glob(str(src / "validation-*.parquet"))[0]).to_dict("records")]
    head = (HERE / "mmlupro_initial_prompt.txt").read_text()
    rng = random.Random(SEED)
    rows = []
    for c in CATS:
        pool = sorted([r for r in test if r["category"] == c], key=lambda r: int(r["question_id"]))
        pick = sorted(rng.sample(pool, 125), key=lambda r: int(r["question_id"]))
        shots = [r for r in val if r["category"] == c][:5]
        for r in pick:
            prompt = head.replace("{$}", c) + "\n" + "".join(fmt(e, True) for e in shots) + fmt(r, False)
            rows.append({"problem": prompt, "answer": r["answer"], "unique_id": "mmlupro-%s" % r["question_id"], "subject": c})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "questions;", {c: sum(r["subject"] == c for r in rows) for c in CATS}, "answers", pd.Series([r["answer"] for r in rows]).value_counts().to_dict())


if __name__ == "__main__":
    main()
