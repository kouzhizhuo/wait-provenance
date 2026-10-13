"""Popular-models panel keys (local): h1.pop_<id>.final.{freq2,freq4[.lo/.hi],nwait,ntok,si,orth,reflopen,reflopen_nowait} and ctl.pop_<id>.final.acc  ->  exp/results/reframe/pop_keys.json
Sources: exp/results/scale/bb/BB_<id>/{gen,idx} (hostA, <=24B) and exp/results/scale/bb/pop_<id>/ (Slurm >=30B archives extracted from infra's results.tgz).
Mistral-Small-24B-Base: the first run (BB_mistral_base, boxed_raw prompt at T=1.0) is NOT used (acc 2.5%, diagnosed as prompt/temperature); the valid run is BB_mistral_base_ps (problem_solution prompt, T=0.15)."""
from __future__ import annotations

import re

import numpy as np

from reframe_keys import OUT, RES, Ex, common, est, ix, mk, poisson_ci, read_gz_jsonl

BB = RES / "scale" / "bb"
MODELS = [  # (key id, directory, label for src)
    ("pop_phi4", "BB_phi4", ""), ("pop_phi4r", "BB_phi4r", ""), ("pop_phi4rp", "BB_phi4rp", ""),
    ("pop_mistral_base", "BB_mistral_base_ps", " (Mistral-Small-24B-Base-2501, problem_solution prompt, T=0.15; the boxed_raw T=1.0 run is invalid)"),
    ("pop_qwen3_8b_base", "BB_qwen3_8b_base", ""), ("pop_qwen3_8b", "BB_qwen3_8b", ""),
    ("pop_qwen3_14b_base", "BB_qwen3_14b_base", ""), ("pop_qwen3_14b", "BB_qwen3_14b", ""),
    ("pop_r1d_llama8", "BB_r1d_llama8", ""),
    ("pop_qwen3b30", "pop_qwen3b30", " (Slurm, model commit in model_commit.txt)"),
    ("pop_r1dllama70b", "pop_r1dllama70b", " (Slurm, R1-Distill-Llama-70B; model commit unrecorded; rerun 407448 after the tokenizer fix)"),
    ("pop_qwen3t30", "pop_qwen3t30", " (Slurm, Qwen3-30B-A3B thinking mode, job 407443)"),
]
FORMS = [("wait", r"wait\b"), ("hmm", r"hm+\b"), ("hold_on", r"hold on\b"), ("actually", r"actually\b"), ("okay", r"(?:okay|ok)\b"), ("no", r"no(?=[,.!:;]|\s+(?:wait|that|this|it)\b|$)"),
         ("let_me_check", r"let me (?:just )?(?:check|verify|double[- ]?check|re-?check|re-?verify)\b"), ("alternatively", r"alternatively\b"), ("however", r"however\b")]
PATS = [(n, re.compile(r"^[\s\"'*_>#-]*" + p, re.I)) for n, p in FORMS]


def openers(rows):
    n_sent = n_op = n_nowait = 0
    for r in rows:
        sents = mk.split_sentences(r["text"])[1:]
        n_sent += len(sents)
        for s in sents:
            for n, p in PATS:
                if p.match(s.text):
                    n_op += 1; n_nowait += (n != "wait")
                    break
    return 1000 * n_op / max(n_sent, 1), 1000 * n_nowait / max(n_sent, 1)


def main():
    X = Ex(); done = []
    for sid, d, note in MODELS:
        dd = BB / d
        gf = next(iter((dd / "gen").glob("*_math500.graded.jsonl.gz")), None)
        idxf = dd / "idx" / "math500.json"
        if gf is None or not idxf.exists():
            continue
        rows = read_gz_jsonl(gf)
        if any(("Ġ" in r["text"][:300]) for r in rows[:50]):
            print("SKIP (undecoded byte-level text):", sid); continue
        j = common.read_json(idxf)
        nw, tk = int(j["totals"].get("wait:n", 0)), int(j["totals"]["tokens"])
        src = "exp/results/scale/bb/%s/idx/math500.json%s" % (d, note)
        lo, hi = poisson_ci(nw)
        b = "h1.%s.final." % sid
        X.put(b + "freq2", 1000 * nw / tk, "dec2", src); X.put(b + "freq4", 1000 * nw / tk, "dec4", src)
        X.put(b + "freq4.lo", 1000 * lo / tk, "dec4", src + "; exact Poisson 95%"); X.put(b + "freq4.hi", 1000 * hi / tk, "dec4", src + "; exact Poisson 95%")
        X.put(b + "nwait", nw, "int", src); X.put(b + "ntok", tk, "int", src)
        for q, nm in (("si", "SI"), ("orth", "ORTH")):
            e = j["indices"].get(nm)
            if isinstance(e, dict) and nw >= 20:
                for s_, x in (("", e["est"]), (".lo", e["lo"]), (".hi", e["hi"])):
                    X.put(b + q + s_, 100 * x, "dec1", src + "::indices." + nm)
        X.put("ctl.%s.final.acc" % sid, j["acc"], "pct", src + "::acc")
        X.put("ctl.%s.final.trunc" % sid, j["truncated"], "pct", src + "::truncated")
        ro, ron = openers(rows)
        X.put(b + "reflopen", ro, "dec1", src + " (sentence-initial reflection openers per 1k sentences, first sentence excluded)")
        X.put(b + "reflopen_nowait", ron, "dec1", src)
        done.append(sid)
    common.write_json(OUT / "pop_keys.json", {"entries": X.entries})
    print("ids with keys:", done, len(X.entries), "keys")


if __name__ == "__main__":
    main()
