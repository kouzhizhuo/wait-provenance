"""openPangu: restrict the wait counting to the THINKING span. The model emits [unused16] <thinking> [unused17] <answer> [unused10](eos); infra/vendor: count 'wait' only inside the thinking.
Reads a graded jsonl.gz (full text incl. the special markers), writes
    <name>.full.graded.jsonl.gz   (the untouched full-text file)
    <name>.graded.jsonl.gz        (text = thinking span only; n_tokens = tokens of the thinking span (model tokenizer); correct/finish/pred unchanged: graded on the full text)
A truncated response has no [unused17]: everything after [unused16] is thinking.  A response without [unused16] (fast/empty thinking) gets an empty thinking text (0 tokens).
    python -m exp.pipeline.pangu_think --graded <x.graded.jsonl.gz> --model <dir>"""
import argparse
import gzip
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True); ap.add_argument("--model", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=False, trust_remote_code=True, local_files_only=True)
    full = a.graded.replace(".graded.jsonl.gz", ".full.graded.jsonl.gz")
    if not os.path.exists(full):
        shutil.copy(a.graded, full)
    rows = [json.loads(l) for l in gzip.open(full, "rt", encoding="utf-8")]
    out, n_nothink = [], 0
    for r in rows:
        t = r["text"]
        if "[unused16]" in t:
            th = t.split("[unused16]", 1)[1].split("[unused17]", 1)[0]
        else:
            th = ""; n_nothink += 1
        th = th.strip()
        r2 = dict(r); r2["text"] = th; r2["n_tokens_total"] = r["n_tokens"]; r2["n_tokens"] = len(tok.encode(th, add_special_tokens=False)) if th else 0
        out.append(r2)
    with gzip.open(a.graded, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("[pangu_think] %d rows, %d without a [unused16] span; thinking tokens %d of %d" % (len(rows), n_nothink, sum(r["n_tokens"] for r in out), sum(r["n_tokens_total"] for r in out)))


if __name__ == "__main__":
    main()
