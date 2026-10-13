"""D46b / D46a statistics on generations (MATH-500, sampled T=1).  Shared by (1) the per-step table of the analysis tail and (2) the three-condition SFT-only table.
 per set of responses: n, truncation share (finish == length), share > 4000 tokens, loop rate (a 50-char block repeated in the last 1000 chars),
 tokens per response / per finished response, wait & next words per 1k tokens for ALL responses and for FINISHED responses only (counts + token totals are stored,
 so exact Poisson CIs can be computed downstream), sentence-initial 'Wait,' per 1k, accuracy (math-verify via exp.pipeline.grade when `correct` is absent).
  python lenwait_stats.py <gen_dir> <out.json> [sft_conditions.json]     per-step table from gen/s<N>.graded.jsonl.gz
  python lenwait_stats.py conditions <out.json> NAME=path[:pid_max[:sid_max]] ...   side-by-side conditions (jsonl / jsonl.gz generations)"""
import glob, gzip, json, os, re, sys
R = os.environ.get("ARR_ROOT", "/<CLUSTER>/<USER>/arr_oct")
WAIT = re.compile(r"\bwait\b", re.I); NEXT = re.compile(r"\bnext\b", re.I)
INIT = re.compile(r"(?:^|(?<=[.!?:])|(?<=\n))\s*Wait,", re.M)

def loop(text, blk=50, tail=1000):
    t = text[-tail:]
    return any(t.count(t[i:i + blk]) >= 2 for i in range(0, max(0, len(t) - blk)))

def agg(rs):
    toks = sum(r["n_tokens"] for r in rs)
    w = sum(len(WAIT.findall(r["text"])) for r in rs); nx = sum(len(NEXT.findall(r["text"])) for r in rs); iw = sum(len(INIT.findall(r["text"])) for r in rs)
    n = len(rs)
    return {"n": n, "tokens": toks, "wait_words": w, "next_words": nx, "init_wait_comma": iw,
            "wait_per_1k": 1000.0 * w / toks if toks else None, "next_per_1k": 1000.0 * nx / toks if toks else None,
            "init_wait_comma_per_1k": 1000.0 * iw / toks if toks else None, "mean_tokens": toks / n if n else None,
            "share_gt4000": sum(r["n_tokens"] > 4000 for r in rs) / n if n else None,
            "loop_rate": sum(loop(r["text"]) for r in rs) / n if n else None,
            "acc": sum(bool(r["correct"]) for r in rs) / n if n else None}

def summarize(rows):
    fin = [r for r in rows if r["finish"] != "length"]
    return {"all": agg(rows), "finished": agg(fin), "truncation_share": 1 - len(fin) / len(rows)}

def read(path):
    op = gzip.open if path.endswith(".gz") else open
    return [json.loads(l) for l in op(path, "rt")]

def grade_missing(rows):
    if rows and "correct" not in rows[0]:
        sys.path.insert(0, R + "/code/P1")
        from exp.pipeline.grade import extract_answer, is_correct
        gold = {i: str(json.loads(l)["answer"]) for i, l in enumerate(open(R + "/data/MATH-500/test.jsonl"))}
        for r in rows:
            r["correct"] = is_correct(extract_answer(r["text"]), gold[r["pid"]])
    return rows

if __name__ == "__main__":
    if sys.argv[1] == "conditions":
        out = {}
        for spec in sys.argv[3:]:
            name, rest = spec.split("=", 1); parts = rest.split(":"); rows = grade_missing(read(parts[0]))
            if len(parts) > 1 and parts[1]: rows = [r for r in rows if r["pid"] < int(parts[1])]
            if len(parts) > 2 and parts[2]: rows = [r for r in rows if r.get("sid", 0) < int(parts[2])]
            out[name] = summarize(rows); out[name]["definition"] = spec
        json.dump(out, open(sys.argv[2], "w"), indent=1)
        for k, o in out.items():
            print("%-34s n=%4d acc %.3f trunc %.3f >4000 %.3f loop %.3f tok/resp %.0f wait/1k all %.2f fin %.2f" % (
                k, o["all"]["n"], o["all"]["acc"], o["truncation_share"], o["all"]["share_gt4000"], o["all"]["loop_rate"], o["all"]["mean_tokens"],
                o["all"]["wait_per_1k"], o["finished"]["wait_per_1k"]))
        sys.exit(0)
    out = {}
    for f in sorted(glob.glob(sys.argv[1] + "/s*.graded.jsonl.gz")):
        name = f.rsplit("/", 1)[-1]
        if ".greedy." in name: continue
        out[int(name[1:].split(".")[0])] = summarize(read(f))
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        out["sft_conditions"] = json.load(open(sys.argv[3]))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    for s in sorted(k for k in out if isinstance(k, int)):
        o = out[s]; print("step %4d  trunc %.3f  wait/1k all %.3f finished %.3f  tokens/finished %.0f  acc %.3f" % (
            s, o["truncation_share"], o["all"]["wait_per_1k"] or 0, o["finished"]["wait_per_1k"] or 0, o["finished"]["mean_tokens"] or 0, o["all"]["acc"]))
