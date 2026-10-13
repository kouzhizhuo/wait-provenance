"""(1) one source per key: split bb_keys{,_oss}.json into bb (non-popular) and pop (h2/h4 of the popular panel); invalid runs dropped; (2) provenance group count keys; (3) opener scope check data.
    python3 exp/reframe_keys8.py"""
from __future__ import annotations

import collections
import json
import re

from reframe_keys import OUT, RES, Ex, common, mk, read_gz_jsonl

INVALID_SERIES = {"pop_mistral_base"}
C32K_SERIES = {"bb_qwen35_" + x for x in ("0p8b", "0p8b_base", "2b", "2b_base", "4b", "9b", "4b_base", "9b_base")} | {"bb_pangu_e7v11", "bb_pangu_r7", "bb_pangu_e7m"}   # primary h1/ctl keys come from qwen35_c32k_keys.json (32k-cap rerun)
SZ_SERIES = {"bb_" + {"sz_ds7": "sz_dsm7", "sz_ds7_base": "sz_dsm7_base"}.get(i, i) for i in ["gr3_4_s1", "gr3_4_s2", "gr3_4_s3", "gr7_base", "gr7", "gr14_base", "gr14", "gr3_4_base", "gr3_4", "gr3_14_base", "gr3_14", "os15_base", "os15", "oat15_base", "oat15", "orz32_base", "orz32", "sz_q32_base", "sz_q32", "sz_m24_base", "sz_m24", "orz7b_base", "oat7_base", "sz_q05_base", "sz_q05", "sz_q15_base", "sz_q15", "sz_q7_base", "sz_q7", "orz7b", "sz_qm7_base", "sz_qm7", "oat7", "sz_q14_base", "sz_q14", "sz_m7_base", "sz_m7", "sz_ds7_base", "sz_ds7", "sz_l8"]}   # the boxed_raw T=1.0 run (BB_mistral_base): invalid, acc 2.5%
RENAME = {"pop_mistral_base_ps": "pop_mistral_base"}   # the valid Mistral-Small-24B-Base run carries the pop_mistral_base id


def ser(key):
    return key.split(".")[1]


def main():
    seen = {}
    for suf in ("", "_oss"):
        d = common.read_json(OUT / "bb" / ("bb_keys%s.json" % suf))
        bb, pop = [], []
        for e in d["entries"]:
            s = ser(e["key"])
            if s in INVALID_SERIES:
                continue
            if s in C32K_SERIES and e["key"].startswith(("h1.", "ctl.", "n.")):
                continue
            if s in SZ_SERIES and e["key"].startswith(("h1.", "ctl.")):   # primary h1/ctl keys of the SimpleRL-Zoo rows come from sz_keys.json (their own eval decode)
                continue
            if s in RENAME:
                p = e["key"].split("."); p[1] = RENAME[s]; e = {**e, "key": ".".join(p)}; s = RENAME[s]
            (pop if s.startswith("pop_") else bb).append(e)
        common.write_json(OUT / "bb" / ("bb_keys%s.json" % suf), {**{k: v for k, v in d.items() if k != "entries"}, "entries": bb, "note": "non-popular backbones only; popular-panel E2/E3 keys are in pop_e23_keys%s.json; h1/ctl keys of the popular panel are in pop_keys.json" % suf})
        # popular panel: h1/ctl/n keys already live in pop_keys.json -> keep only h2/h4 here
        common.write_json(OUT / ("pop_e23_keys%s.json" % suf), {"entries": [e for e in pop if e["key"].startswith(("h2.", "h4."))]})
    # global duplicate check across every key file in reframe/ and bb/
    dup = collections.defaultdict(list)
    for f in list(OUT.glob("[a-z]*_keys*.json")) + [x for x in (OUT / "bb").glob("*_keys*.json") if not x.name.startswith("._")]:
        for e in common.read_json(f)["entries"]:
            dup[e["key"]].append(f.name)
    clashes = {k: v for k, v in dup.items() if len(set(v)) > 1}
    print("keys:", len(dup), "| cross-file duplicates:", len(clashes), list(clashes.items())[:5])
    # (2) provenance group counts
    pm = [r for r in common.read_json(OUT / "provenance_map.json")["rows"] if r.get("counted", True)]   # rows flagged counted=false (no licence text) are secondary only
    slug = {"pretrained_base": "base", "zero_rl": "zerorl", "reasoning_sft_distill": "sft_distill", "rl_on_marker_model": "rl_marker", "undisclosed": "undisclosed", "instruct": "instruct", "rl_from_instruct": "rl_instruct"}
    X = Ex()
    for g, sg in slug.items():
        rs = [r for r in pm if r["group"] == g]
        if not rs:
            continue
        src = "exp/results/reframe/provenance_map.json::rows[group=%s]" % g
        X.put("h1.prov.%s.n" % sg, len(rs), "int", src)
        X.put("h1.prov.%s.n_ge05" % sg, sum(r["wait_per_1k"] >= 0.5 for r in rs), "int", src + " (wait per 1k >= 0.5)")
        X.put("h1.prov.%s.min" % sg, min(r["wait_per_1k"] for r in rs), "dec4", src); X.put("h1.prov.%s.max" % sg, max(r["wait_per_1k"] for r in rs), "dec4", src)
    sft = [r for r in pm if r["group"] == "reasoning_sft_distill"]
    free = [r for r in sft if r.get("base_wait_per_1k") is not None and r["base_wait_per_1k"] < 0.05]
    marked = [r for r in sft if r.get("base_wait_per_1k") is not None and r["base_wait_per_1k"] >= 0.05]
    miss = [r for r in sft if r.get("base_wait_per_1k") is None]
    srcb = "exp/results/reframe/provenance_map.json::rows[group=reasoning_sft_distill].base_wait_per_1k (base from the official card; < 0.05 = marker-free)"
    X.put("h1.prov.sft_free.n", len(free), "int", srcb); X.put("h1.prov.sft_free.n_ge05", sum(r["wait_per_1k"] >= 0.5 for r in free), "int", srcb + "; own wait per 1k >= 0.5")
    X.put("h1.prov.sft_free.n_basemissing", len(miss), "int", srcb + "; base not measured / not released (not in n)")
    X.put("h1.prov.sft_marked.n", len(marked), "int", srcb); X.put("h1.prov.sft_marked.n_ge05", sum(r["wait_per_1k"] >= 0.5 for r in marked), "int", srcb + "; own wait per 1k >= 0.5")
    print("sft_free n=%d n_ge05=%d basemissing=%d | marked n=%d n_ge05=%d" % (len(free), sum(r["wait_per_1k"] >= 0.5 for r in free), len(miss), len(marked), sum(r["wait_per_1k"] >= 0.5 for r in marked)))
    print("FREE:", [(r["model"], round(r["wait_per_1k"], 3), round(r["base_wait_per_1k"], 4)) for r in free]); print("MISSING:", [r["model"] for r in miss]); print("MARKED:", [(r["model"], round(r["base_wait_per_1k"], 2)) for r in marked])
    X.entries.append({"key": "h1.prov.sft_distill.exception_model", "value": "OpenReasoning-Nemotron-7B", "src": "exp/results/reframe/provenance_map.json::rows (the only reasoning_sft_distill row with wait per 1k < 0.5)"})
    X.put("h1.prov.n_rows", len(pm), "int", "provenance_map.json rows (PRELIM)")
    common.write_json(OUT / "prov_count_keys.json", {"entries": X.entries})
    # (3) opener scope check on the three models
    forms = [("wait", r"wait\b"), ("hmm", r"hm+\b"), ("hold_on", r"hold on\b"), ("actually", r"actually\b"), ("okay", r"(?:okay|ok)\b"), ("no", r"no(?=[,.!:;]|\s+(?:wait|that|this|it)\b|$)"),
             ("let_me_check", r"let me (?:just )?(?:check|verify|double[- ]?check|re-?check|re-?verify)\b"), ("alternatively", r"alternatively\b"), ("however", r"however\b"), ("but", r"but\b"), ("let_me", r"let me\b"), ("so", r"so\b")]
    pats = [(n, re.compile(r"^[\s\"'*_>#-]*" + p, re.I)) for n, p in forms]
    out = {}
    for mid in ("orn7", "mimo_base", "smollm3_base", "r1d7", "qwen7b"):
        p = next(iter((RES / "scale" / "bb" / ("BB_" + mid) / "gen").glob("*_math500.graded.jsonl.gz")))
        rows = read_gz_jsonl(p)
        c = collections.Counter(); first = collections.Counter(); ns = 0
        for r in rows:
            ss = mk.split_sentences(r["text"])[1:]
            ns += len(ss)
            for s in ss:
                w = re.findall(r"[A-Za-z']+", s.text[:40])
                first[(w[0].lower() if w else "")] += 1
                for n, pt in pats:
                    if pt.match(s.text):
                        c[n] += 1
        out[mid] = {"n_sentences": ns, "rate_per_1k_sentences": {n: round(1000 * c[n] / ns, 1) for n, _ in forms}, "top_sentence_initial_words": [(w, round(1000 * v / ns, 1)) for w, v in first.most_common(14)]}
    common.write_json(OUT / "opener_scope_orn7_mimo_smollm3.json", out)
    for k, v in out.items():
        print(k, v["rate_per_1k_sentences"]); print("   top:", v["top_sentence_initial_words"][:10])


if __name__ == "__main__":
    main()
