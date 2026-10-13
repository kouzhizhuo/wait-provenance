"""D46a metrics on a gen jsonl (exp.pipeline.gen output, sampled): share of responses > 4000 tokens, share hitting the cap, loop rate
(last 1000 chars contain a 50-char block that occurs >= 2 times), accuracy as generated and re-scored under shorter caps
(a response with n_tokens > cap counts as unfinished = wrong; valid because T=1 prefixes of the same sample).
  python len_metrics.py --gen smoke.jsonl --data MATH-500/test.jsonl [--pid_max 48] [--cap 8192]"""
import argparse, json, sys, os
R = os.environ.get("ARR_ROOT", "/<CLUSTER>/<USER>/arr_oct"); sys.path.insert(0, R + "/code/P1")
from exp.pipeline.grade import extract_answer, is_correct

def loop(text, blk=50, tail=1000):
    t = text[-tail:]
    return any(t.count(t[i:i + blk]) >= 2 for i in range(0, max(0, len(t) - blk)))

ap = argparse.ArgumentParser(); ap.add_argument("--gen", required=True); ap.add_argument("--data", default=R + "/data/MATH-500/test.jsonl")
ap.add_argument("--pid_max", type=int, default=10**9); ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--out", default=None)
a = ap.parse_args()
gold = {i: str(json.loads(l)["answer"]) for i, l in enumerate(open(a.data))}
rows = [json.loads(l) for l in open(a.gen)]
rows = [r for r in rows if r["pid"] < a.pid_max]
for r in rows:
    r["ok"] = is_correct(extract_answer(r["text"]), gold[r["pid"]])
n = len(rows)
out = {"n": n, "cap": a.cap, "mean_tokens": sum(r["n_tokens"] for r in rows) / n,
       "share_gt4000": sum(r["n_tokens"] > 4000 for r in rows) / n,
       "share_at_cap": sum(r["finish"] == "length" for r in rows) / n,
       "loop_rate": sum(loop(r["text"]) for r in rows) / n,
       "loop_rate_among_truncated": (lambda t: sum(loop(r["text"]) for r in t) / max(1, len(t)))([r for r in rows if r["finish"] == "length"]),
       "acc_as_generated": sum(r["ok"] for r in rows) / n}
for cap in (3072, 4000, a.cap):
    out["acc_at_cap_%d" % cap] = sum(r["ok"] and r["finish"] != "length" and r["n_tokens"] <= cap for r in rows) / n
    out["share_unfinished_at_cap_%d" % cap] = sum(r["n_tokens"] > cap or r["finish"] == "length" for r in rows) / n
print(json.dumps(out, indent=1))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
