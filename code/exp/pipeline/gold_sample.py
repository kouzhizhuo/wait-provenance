"""Draw the sentences for the gold revision labels (T-W3; EXPERIMENT_PLAN §2.1).

    python -m exp.pipeline.gold_sample --gen exp/results/P0/gen/<EMAIL> exp/results/P0/gen/<EMAIL> \
        --n 150 --marker_frac 0.4 --seed 0 --out exp/gold/s1_sample.jsonl

Each ckpt contributes n/len(gen) sentences. Stratified: `marker_frac` of them are marker-initial (the sentence starts with a
discourse form of the strip list, i.e. strip_initial_forms removes something), the rest are other sentences. Sentences with
< 3 words, the final `Answer:` line and markdown-only lines are excluded. One sentence per (problem, sample) at most, so
labels are spread over traces. Labellers see the UNSTRIPPED sentence (they are the gold standard); the judge sees the
stripped one (gold_labels compare `label` to the judge letter). Fields are what the labelling sheet needs.
"""
from __future__ import annotations

import argparse
import random
import re
from typing import List

from . import common, markers as mk


def candidates(row: dict) -> List[dict]:
    sents = mk.split_sentences(row["text"])
    out = []
    for i, s in enumerate(sents):
        words = mk.words(mk.mask_math(mk.normalise_quotes(s.text)))
        if len(words) < 3 or re.match(r"\W*answer\s*:", s.text, re.I) or i == 0:
            continue
        stripped, removed = mk.strip_initial_forms(s.text)
        out.append({"ckpt": row["ckpt"], "pid": row["pid"], "sid": row["sid"], "sent_idx": i, "n_sent": len(sents),
                    "context": row["text"][:s.start][-2500:], "sentence": s.text, "sentence_stripped": stripped,
                    "stripped_token": removed, "marker_initial": bool(removed),
                    "next": " ".join(x.text for x in sents[i + 1:i + 3])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--marker_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    per = a.n // len(a.gen)
    out = []
    for g in a.gen:
        rows = common.read_jsonl(g)
        rng.shuffle(rows)
        pool_m, pool_o, used = [], [], set()
        for r in rows:
            for c in candidates(r):
                (pool_m if c["marker_initial"] else pool_o).append(c)
        rng.shuffle(pool_m)
        rng.shuffle(pool_o)
        want_m = int(round(per * a.marker_frac))
        picked = []
        for c in pool_m + pool_o:  # fill strata; at most one sentence per (pid, sid)
            key = (c["pid"], c["sid"])
            is_m = c["marker_initial"]
            if key in used:
                continue
            if is_m and sum(p["marker_initial"] for p in picked) >= want_m:
                continue
            if not is_m and sum(not p["marker_initial"] for p in picked) >= per - want_m:
                continue
            used.add(key)
            picked.append(c)
            if len(picked) == per:
                break
        out += picked
    rng.shuffle(out)
    for k, c in enumerate(out):
        c["item_id"] = "s1-%03d" % k
    n = common.write_jsonl(a.out, out)
    common.write_meta(a.out + ".meta.json", "gold_sample", {"gen": a.gen, "n": a.n, "marker_frac": a.marker_frac},
                      {"seed": a.seed}, inputs=a.gen, outputs=[a.out],
                      extra={"n_items": n, "n_marker_initial": sum(c["marker_initial"] for c in out)})
    print("[gold_sample] %d items (%d marker-initial)" % (n, sum(c["marker_initial"] for c in out)))


if __name__ == "__main__":
    main()
