"""DEV set for CW calibration: 300 MATH-train problems, levels 3-5, seed 20261012 -> jsonl {problem, answer, unique_id, level}.   python -m exp.cw.build_dev <ARR_ROOT> <out.jsonl>"""
import glob
import json
import random
import sys

import pandas as pd


def last_boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, d = i + 7, 1
    while j < len(s) and d:
        d += (s[j] == "{") - (s[j] == "}"); j += 1
    return s[i + 7:j - 1] if d == 0 else None


root, out = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rows = []
for f in sorted(glob.glob(root + "/data/hendrycks_math/*/train-*.parquet")):
    for _, r in pd.read_parquet(f).iterrows():
        if r["level"] in ("Level 3", "Level 4", "Level 5"):
            gt = last_boxed(r["solution"])
            if gt is not None:
                rows.append({"problem": r["problem"], "answer": gt, "unique_id": "train/%s/%d" % (r["type"], len(rows)), "level": r["level"], "subject": r["type"]})
random.Random(20261012).shuffle(rows)
with open(out, "w") as f:
    for r in rows[:N]:
        f.write(json.dumps(r) + "\n")
print(len(rows), "L3-5 train problems; wrote N to", out)
