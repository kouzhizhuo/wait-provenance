"""P0 report: H1 indices (rarefied entropies, cluster bootstrap), insertion-pilot estimands, plots, tables.

    python -m exp.pipeline.report --res exp/results/P0 --figdir figures

Inputs (under --res):  gen/S1@<ckpt>.graded.jsonl.gz, insertion/{prefixes,prefixes_base}.jsonl, insertion/judged*.jsonl.gz
Outputs (under --res): h1_indices.json/.csv, insertion_pilot.json/.csv, tables.md, run_meta.json (+ PDF/PNG plots in --figdir).
Every number in the tables comes from these files; nothing is typed by hand. Intervals: 95% percentile
cluster-bootstrap over problems (H1) or over prefixes (insertion); B = 2000 (pilot), seed 20261012 (STATS_PLAN §0).
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import common, indices as ix, markers as mk, stats as st

SERIES = "S1"   # tag used in file names / policy names: gen/<SERIES>@<ckpt>.graded.jsonl.gz
CKPTS = [("base", 0), ("100", 100), ("500", 500), ("1000", 1000), ("1900", 1900)]   # (name, step); step None = 'final', plotted off-axis
LAST = "1900"   # the 'last checkpoint' of the amplification statistics (P0: 1900; main runs: final)
KEYS_SERIES, STAGE_MAP, PILOT = "s1", None, True
SUF = ""   # "_oss" when reporting the second judge (gpt-oss-20b): inputs judged*_oss*, revlab_oss/, outputs *_oss.*, keys with the _oss suffix (D35)


def configure(series: str, ckpts: str, last: str, keys_series: str, stage_map: Optional[str], pilot: bool):
    """ckpts = 'base:0,100:100,...,final:' (empty step = final). stage_map = 'early=100,mid=700,late=1900,final=final'."""
    global SERIES, CKPTS, LAST, KEYS_SERIES, STAGE_MAP, PILOT
    SERIES, LAST, KEYS_SERIES, PILOT = series, last, keys_series, pilot
    CKPTS = [(t.split(":")[0], (int(t.split(":")[1]) if t.split(":")[1] else None)) for t in ckpts.split(",")]
    if stage_map:
        STAGE_MAP = {"base": "base", **dict(kv.split("=") for kv in stage_map.split(","))}


def xpos(step: Optional[int]) -> float:
    """x position on the step axis; 'final' (step None) is drawn just right of the last numeric step."""
    nums = [s for _, s in CKPTS if s is not None]
    return float(step) if step is not None else float(max(nums) * 1.12 if nums else 1.0)
ARM_ORDER = ["none", "okay", "next", "so", "hmm", "hold_on", "wait"]
ARM_LABEL = {"none": "∅", "okay": "Okay,", "next": "Next,", "so": "So,", "hmm": "Hmm,", "hold_on": "Hold on,", "wait": "Wait,"}
H1_KEYS = ["FREQ_wait_1k", "FREQ_wait_100s", "FREQ_disc_1k", "SI", "SI_conn", "PI", "ORTH", "ORTH_max", "COMMA_DROP", "ORTH_comma", "ORTH_period", "ORTH_bare", "CHAIN", "CHAIN_RUNS_1k", "BUT_WAIT", "H_pos", "H_pos_rare",
           "H_para", "H_para_rare", "H_para_disc", "N_eff", "SHARE_wait", "OBL", "REL", "REL_strict", "H_next", "H_next_rare",
           "EROS", "SENTLEN_init", "CTX", "LEX_share", "R_per_100s", "n_wait", "n_R", "n_M", "n_resp", "tokens_per_resp"]
COUNT_KEYS = ("n_wait", "n_R", "n_M", "n_resp")
JUDGE_KEYS = ["OBL_j", "REL_j", "REL_strict_j", "R_per_100s_j"]   # judge-based revision indices (D30), from <res>/revlab/<SERIES>@<ckpt>.revlab.json
FIN_KEYS = ["FREQ_wait_1k", "SI", "ORTH", "CHAIN", "H_pos", "H_para", "OBL", "REL", "H_next", "EROS", "tokens_per_resp", "n_wait"]
B = 2000
SEED = 20261012
MIN_OCC = 5  # a checkpoint with fewer occurrences does not define the rarefaction size n*


# ----------------------------------------------------------------------------- H1
def per_problem_counts(rows: List[dict]) -> List[dict]:
    by_pid: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by_pid[r["pid"]].append(r)
    return [ix.sum_counts(ix.trace_counts(r["text"], r["n_tokens"]) for r in by_pid[pid]) for pid in sorted(by_pid)]


def run_h1(res_dir: Path) -> dict:
    data, meta, raw = {}, {}, {}
    for name, step in CKPTS:
        p = res_dir / "gen" / (SERIES + "@%s.graded.jsonl.gz" % name)
        if not p.exists():
            print("[h1] missing %s" % p, file=sys.stderr)
            continue
        rows = common.read_jsonl(p)
        raw[name] = rows
        data[name] = per_problem_counts(rows)
        meta[name] = {"step": step, "acc": float(np.mean([r["correct"] for r in rows])),
                      "truncated": float(np.mean([r["finish"] == "length" for r in rows])),
                      "has_answer": float(np.mean([r["pred"] is not None for r in rows])),
                      "n_problems": len(data[name]), "n_samples": len(rows)}
    tot = {n: ix.sum_counts(v) for n, v in data.items()}

    def nstar(f):
        vals = [f(t) for t in tot.values() if f(t) >= MIN_OCC]
        return int(min(vals)) if vals else None

    para_n = lambda t: sum(t.get("para:" + x, 0) for x in mk.FORM_NAMES)  # noqa: E731
    rare = {"pos": nstar(lambda t: t.get("wait:n", 0)), "next": nstar(lambda t: t.get("wait:n", 0)), "para": nstar(para_n)}
    out = {}
    for name, step in CKPTS:
        if name not in data:
            continue
        res = ix.bootstrap_indices(data[name], B=B, seed=SEED, rare=rare)
        strip = {k[6:]: v for k, v in tot[name].items() if k.startswith("strip:")}
        # robustness: responses that finished before the token cap (truncation rises with RL: 14% -> 26% at 6,144)
        fin = [r for r in raw[name] if r["finish"] == "stop"]
        fin_res = ix.bootstrap_indices(per_problem_counts(fin), B=B, seed=SEED, rare=None)
        rl = res_dir / ("revlab" + SUF) / ("%s@%s.revlab.json" % (SERIES, name))
        jud = {k: {**v, "se": float("nan")} for k, v in common.read_json(rl)["indices"].items()} if rl.exists() else {}
        out[name] = {"step": step, **meta[name], "indices": {**{k: res[k] for k in H1_KEYS}, **jud},
                     "indices_finished_only": {k: fin_res[k] for k in FIN_KEYS}, "n_finished": len(fin),
                     "stripped_initial_forms": dict(sorted(strip.items(), key=lambda kv: -kv[1])[:12]),
                     **orth_variants(raw[name])}
    return {"checkpoints": out, "rarefaction_n_star": rare, "matched_length": matched_length(raw)}


def matched_length(raw: Dict[str, List[dict]], keys=("FREQ_wait_1k", "REL", "SI", "OBL", "H_next"), n_bins: int = 5,
                   repeats: int = 50, seed: int = 0) -> dict:
    """Matched-length companion (D30, mandatory for FREQ and REL): responses are binned by length (quantiles of the pooled token
    counts) and every checkpoint is subsampled WITHOUT replacement to the same number of responses per bin (the smallest count
    over checkpoints), so all checkpoints have the same length distribution. Mean over `repeats` subsamples; the range is the
    2.5-97.5 percentile of the subsample means (sampling variability of the matching, NOT a confidence interval)."""
    rng = np.random.default_rng(seed)
    lens = np.concatenate([[r["n_tokens"] for r in rows] for rows in raw.values()])
    edges = np.quantile(lens, np.linspace(0, 1, n_bins + 1)[1:-1])
    binned = {c: [[] for _ in range(n_bins)] for c in raw}
    for c, rows in raw.items():
        for r in rows:
            binned[c][int(np.searchsorted(edges, r["n_tokens"], side="right"))].append(r)
    m = [min(len(binned[c][b]) for c in raw) for b in range(n_bins)]
    counts = {c: {id(r): ix.trace_counts(r["text"], r["n_tokens"]) for r in rows} for c, rows in raw.items()}   # once per response
    out = {"per_bin_n": m, "bin_edges_tokens": [float(e) for e in edges], "checkpoints": {}}
    for c in raw:
        vals = {k: [] for k in keys}
        for _ in range(repeats):
            sub = []
            for b in range(n_bins):
                if m[b]:
                    sub += [binned[c][b][i] for i in rng.choice(len(binned[c][b]), m[b], replace=False)]
            tot = ix.sum_counts(counts[c][id(r)] for r in sub)
            idx = ix.indices_from_totals(tot)
            for k in keys:
                vals[k].append(idx[k])
        out["checkpoints"][c] = {k: {"est": float(np.nanmean(v)), "lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))}
                                 for k, v in vals.items()}
    return out


def orth_variants(rows: List[dict], top: int = 10, n_examples: int = 10, seed: int = 0) -> dict:
    """Top orthographic variants of `wait` with shares, ORTH_max (modal-variant share; exploratory, the pre-registered
    ORTH is the canonical 'Wait,' share) and verbatim sentence-initial examples."""
    import random

    cnt: Dict[str, int] = defaultdict(int)
    pool = []
    for r in rows:
        sents = mk.split_sentences(r["text"])
        for m in mk.find_markers(r["text"], sents):
            if m.form != "wait":
                continue
            cnt[mk.wait_variant(m)] += 1
            if m.word_idx == 0:
                pool.append((r["pid"], r["sid"], sents[m.sent_idx].text.replace("\n", " ")[:160]))
    tot = sum(cnt.values())
    rng = random.Random(seed)
    rng.shuffle(pool)
    top_v = sorted(cnt.items(), key=lambda kv: -kv[1])[:top]
    return {"orth_variants": [{"variant": k, "n": v, "share": v / tot if tot else float("nan")} for k, v in top_v],
            "ORTH_max": (top_v[0][1] / tot) if tot else float("nan"), "ORTH_max_variant": top_v[0][0] if top_v else None,
            "marker_initial_examples": [{"pid": a, "sid": b, "sentence": c} for a, b, c in pool[:n_examples]]}


def detector_audit(rows: List[dict], n: int = 10, seed: int = 0) -> dict:
    """Sanity check of the rule revision detector: revision events it fires, and `wait` sentences after which it sees
    no revision event within the next two sentences."""
    import random

    from . import revisions as rv

    rng = random.Random(seed)
    events, silent = [], []
    for r in rows:
        sents = mk.split_sentences(r["text"])
        st_ = [mk.strip_initial_forms(x.text)[0] for x in sents]
        ev = rv.detect_events(st_)
        msent = {m.sent_idx for m in mk.find_markers(r["text"], sents) if m.form == "wait"}
        for i, e in enumerate(ev):
            if e:
                events.append({"pid": r["pid"], "sid": r["sid"], "type": e, "previous": sents[i - 1].text[:140] if i else "",
                               "sentence": sents[i].text[:200]})
        for i in msent:
            if not any(ev[j] for j in range(i, min(len(sents), i + 3))):
                silent.append({"pid": r["pid"], "sid": r["sid"], "sentence": sents[i].text[:200],
                               "next": " ".join(x.text for x in sents[i + 1:i + 3])[:200]})
    rng.shuffle(events)
    rng.shuffle(silent)
    types = defaultdict(int)
    for e in events:
        types[e["type"]] += 1
    return {"n_events": len(events), "event_types": dict(types), "n_wait_sentences_without_event": len(silent),
            "sample_events": events[:n], "sample_silent_wait_sentences": silent[:n]}


def h1_trends(h1: dict) -> dict:
    """Spearman rho of each index against RL step over the available checkpoints (K = 5 in P0: indicative only)."""
    cks = h1["checkpoints"]
    steps = [cks[n]["step"] for n, _ in CKPTS if n in cks and cks[n]["step"] is not None]   # 'final' has no numeric step
    tr = {}
    for k in ["FREQ_wait_1k", "SI", "SI_conn", "ORTH", "ORTH_max", "COMMA_DROP", "CHAIN_RUNS_1k", "SENTLEN_init", "H_pos_rare", "H_para_rare", "H_para",
              "OBL", "REL", "H_next_rare", "EROS"] + [j for j in JUDGE_KEYS if all(j in c["indices"] for c in cks.values())]:
        vals = [cks[n]["indices"][k]["est"] for n, _ in CKPTS if n in cks and cks[n]["step"] is not None]
        ok = [(s, v) for s, v in zip(steps, vals) if v == v]
        rho, p = st.spearman([s for s, _ in ok], [v for _, v in ok]) if len(ok) >= 3 else (float("nan"), float("nan"))
        tr[k] = {"rho": rho, "p": p, "n_points": len(ok)}
    return tr


# ----------------------------------------------------------------------------- insertion
def _succ_table(rows: List[dict], outcome: str):
    """(prefix_ids, {policy: {arm: successes per prefix}}, n_per)."""
    pids = sorted({r["prefix_id"] for r in rows})
    pos = {p: i for i, p in enumerate(pids)}
    succ: Dict[str, Dict[str, np.ndarray]] = {}
    per = defaultdict(int)
    for r in rows:
        d = succ.setdefault(r["policy"], {}).setdefault(r["arm"], np.zeros(len(pids)))
        d[pos[r["prefix_id"]]] += (not r["correct"]) if outcome == "wrong" else bool(r.get(outcome, False))
        per[(r["policy"], r["arm"], r["prefix_id"])] += 1
    n_per = max(per.values())
    return pids, succ, n_per


def analyse_set(rows: List[dict], outcome: str, seed: int) -> dict:
    pids, succ, n_per = _succ_table(rows, outcome)
    order = {SERIES + "@%s" % n: i for i, (n, _) in enumerate(CKPTS)}
    succ = {p: succ[p] for p in sorted(succ, key=lambda p: order[p])}
    res = {"n_prefixes": len(pids), "n_per": n_per, "policies": {}}
    for pol, arms in succ.items():
        res["policies"][pol] = {"arms": {a: dict(zip(("p", "lo", "hi"), st.proportion_ci(v, n_per, B=B, seed=seed)))
                                         for a, v in arms.items()}}
        syn = [arms[a] for a in ("hold_on", "hmm") if a in arms]
        if syn:  # pooled synonyms {Hold on, Hmm}: each prefix contributes 2 * n_per continuations
            res["policies"][pol]["arms"]["syn"] = dict(zip(("p", "lo", "hi"), st.proportion_ci(sum(syn), n_per * len(syn), B=B, seed=seed)))
    if all(a in arms for arms in succ.values() for a in ("wait", "none")):
        res["estimates"] = st.insertion_estimates(succ, n_per, B=B, seed=seed, base=SERIES + "@base", last=SERIES + "@" + LAST)
    return res, succ


def insertion_json(ins: dict, src: str, pilot: bool = True, series: str = "S1") -> Optional[dict]:
    """Layout proposed by the figure builder (figures/scripts/SCHEMAS.md), one file per prefix source. `syn` = pooled
    {Hold on, Hmm}; `p_revise` is the arm-blind judge outcome; `n` = continuations per arm (prefixes x continuations)."""
    o = ins["outcomes"]["judge_revise"].get(src)
    if not o:
        return None
    cks = []
    for pol, d in o["policies"].items():
        ck = pol.split("@")[1]
        cks.append({"ckpt": ck, "step": dict(CKPTS).get(ck), "n_prefixes": o["n_prefixes"],
                    "arms": {a: {"p_revise": {"est": v["p"], "lo": v["lo"], "hi": v["hi"]},
                                 "n": o["n_prefixes"] * o["n_per"] * (2 if a == "syn" else 1)} for a, v in d["arms"].items()}})
    return {"series": series, "pilot": pilot, "prefix_source": src, "outcome": "judge_revise (arm-blind %s, prompts v1.0)" % ("gpt-oss-20b" if SUF == "_oss" else "Qwen2.5-7B-Instruct"),
            "arms": sorted({a for c in cks for a in c["arms"]}), "checkpoints": cks,
            "estimates": o.get("estimates"), "reference_arms": ins.get("reference_arms")}


def run_insertion(res_dir: Path) -> Optional[dict]:
    files = sorted(glob.glob(str(res_dir / "insertion" / ("judged*%s*.jsonl.gz" % SUF if SUF else "judged*.jsonl.gz"))))
    if not SUF:
        files = [f for f in files if "_oss" not in f]
    if not files:
        return None
    rows = [r for f in files for r in common.read_jsonl(f)]
    out: dict = {"outcomes": {}, "sets": {}, "reference_arms": {
        "RD_op": "okay (Okay, = reasoning-cues comparability arm)", "RD_next": "next (Next, = primary neutral control, D21)",
        "RD_syn": "pooled synonyms {hold_on, hmm}", "RD_none": "none (empty insertion)",
        "note": "RD = 100 * (P(REVISE | wait) - P(REVISE | reference arm)), percentage points"}}
    for outcome in ("judge_revise", "rule_revise", "wrong", "false_rev"):
        o: dict = {}
        succs = {}
        for src in ("late", "base"):
            rs = [r for r in rows if r["src"] == src]
            if rs:
                o[src], succs[src] = analyse_set(rs, outcome, SEED)
        if "late" in succs and "base" in succs and all(p in succs["base"] for p in (SERIES + "@base", SERIES + "@" + LAST)):
            pp = (SERIES + "@base", SERIES + "@" + LAST)
            o["crossed"] = st.crossed_interaction({p: succs["late"][p] for p in pp}, {p: succs["base"][p] for p in pp},
                                                  o["late"]["n_per"], B=B, seed=SEED + 1, base=pp[0], last=pp[1])
        out["outcomes"][outcome] = o
    out["n_prefixes"] = {s: out["outcomes"]["judge_revise"][s]["n_prefixes"] for s in ("late", "base")
                         if s in out["outcomes"]["judge_revise"]}
    # judge diagnostics
    diag = {}
    for pol in sorted({r["policy"] for r in rows}):
        prs = [r for r in rows if r["policy"] == pol]
        diag[pol] = {"kappa_rule_vs_judge": st.cohen_kappa([int(bool(r["rule_revise"])) for r in prs],
                                                            [int(bool(r["judge_revise"])) for r in prs]),
                     "cats": {c: sum(r["judge_cat"] == c for r in prs) / len(prs) for c in
                              ("continue", "revise_value", "restart", "reverify_confirm", "branch", "invalid")},
                     "acc_by_arm": {a: float(np.mean([r["correct"] for r in prs if r["arm"] == a])) for a in ARM_ORDER
                                    if any(r["arm"] == a for r in prs)},
                     "trunc_by_arm": {a: float(np.mean([r["finish"] == "length" for r in prs if r["arm"] == a])) for a in ARM_ORDER
                                      if any(r["arm"] == a for r in prs)}}
    out["diagnostics"] = diag
    late = out["outcomes"]["judge_revise"].get("late", {}).get("estimates", {})
    rdk = "RD_next" if all(("%s@%s:RD_next" % (SERIES, n)) in late for n, st_ in CKPTS if st_ is not None) else "RD_op"   # D21: Next, is the reference
    pols = [p for p in (SERIES + "@%s" % n for n, st_ in CKPTS if st_ is not None) if "%s:%s" % (p, rdk) in late]
    steps = [dict(CKPTS)[p.split("@")[1]] for p in pols]
    out["trend_RD_op"] = dict(zip(("rho", "p"), st.spearman(steps, [late["%s:%s" % (p, rdk)]["est"] for p in pols]))) if len(pols) >= 3 else {}
    out["trend_reference_arm"] = rdk
    return out


# ----------------------------------------------------------------------------- writers
def write_h1_csv(path: Path, h1: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ckpt", "step", "index", "est", "lo", "hi", "se"])
        for name, step in CKPTS:
            c = h1["checkpoints"].get(name)
            if not c:
                continue
            for k, v in c["indices"].items():
                w.writerow([name, step, k, v["est"], v["lo"], v["hi"], v["se"]])
            for k in ("acc", "truncated", "has_answer"):
                w.writerow([name, step, k, c[k], "", "", ""])


def write_ins_csv(path: Path, ins: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outcome", "set", "policy", "name", "est", "lo", "hi"])
        for oc, o in ins["outcomes"].items():
            for s in ("late", "base"):
                if s not in o:
                    continue
                for pol, d in o[s]["policies"].items():
                    for a, v in d["arms"].items():
                        w.writerow([oc, s, pol, "P_" + a, v["p"], v["lo"], v["hi"]])
                for k, v in o[s].get("estimates", {}).items():
                    w.writerow([oc, s, "", k, v["est"], v["lo"], v["hi"]])
            for k, v in o.get("crossed", {}).items():
                w.writerow([oc, "crossed", "", k, v["est"], v["lo"], v["hi"]])


def fmt(v, d: int = 3) -> str:
    return "n/a" if v is None or v != v else ("%.*f" % (d, v))


def ci(x: dict, d: int = 3) -> str:
    return "%s [%s, %s]" % (fmt(x["est"], d), fmt(x["lo"], d), fmt(x["hi"], d))


def ci_p(x: dict, d: int = 3) -> str:
    return "%s [%s, %s]" % (fmt(x["p"], d), fmt(x["lo"], d), fmt(x["hi"], d))


def tables_md(h1: dict, trends: dict, ins: Optional[dict]) -> str:
    L = []
    cks = h1["checkpoints"]
    cols = [n for n, _ in CKPTS if n in cks]
    L.append("### H1 indices per checkpoint (estimate [95%% cluster-bootstrap CI over problems, B=%d]); rarefaction n* = %s\n" %
             (B, h1["rarefaction_n_star"]))
    L.append("| index | " + " | ".join("S1@%s" % c for c in cols) + " | Spearman rho vs step (K=%d) |" % len(cols))
    L.append("|---|" + "---|" * (len(cols) + 1))
    L.append("| acc | " + " | ".join(fmt(cks[c]["acc"]) for c in cols) + " | |")
    L.append("| truncated at 6144 tok | " + " | ".join(fmt(cks[c]["truncated"]) for c in cols) + " | |")
    L.append("| has answer | " + " | ".join(fmt(cks[c]["has_answer"]) for c in cols) + " | |")
    for k in H1_KEYS + [j for j in JUDGE_KEYS if all(j in cks[c]["indices"] for c in cols)]:
        if k in COUNT_KEYS:
            row = " | ".join(fmt(cks[c]["indices"][k]["est"], 0) for c in cols)
        else:
            row = " | ".join(ci(cks[c]["indices"][k], 2 if k == "tokens_per_resp" else 3) for c in cols)
        t = trends.get(k)
        L.append("| %s | %s | %s |" % (k, row, ("%s%s" % (fmt(t["rho"], 2), (" (exact p=%s)" % fmt(t["p"], 3)) if t["p"] is not None else " (K<6: no p)")) if t else ""))
    L.append("")
    L.append("### Robustness: indices on responses that finished before the token cap (stop only)\n")
    L.append("| index | " + " | ".join("S1@%s" % c for c in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    L.append("| n finished / 800 | " + " | ".join(str(cks[c]["n_finished"]) for c in cols) + " |")
    for k in FIN_KEYS:
        L.append("| %s | %s |" % (k, " | ".join(fmt(cks[c]["indices_finished_only"][k]["est"], 0 if k == "n_wait" else 3) if k in ("n_wait",) else ci(cks[c]["indices_finished_only"][k], 2 if k == "tokens_per_resp" else 3) for c in cols)))
    L.append("")
    ml = h1.get("matched_length")
    if ml:
        L.append("### Matched-length companion (5 length bins, %s responses per bin per checkpoint; mean over 50 subsamples [2.5, 97.5 percentile of subsample means; matching variability, not a CI])\n" % ml["per_bin_n"])
        L.append("| index | " + " | ".join("S1@%s" % c for c in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        for k in ("FREQ_wait_1k", "REL", "SI", "OBL", "H_next"):
            L.append("| %s (matched) | %s |" % (k, " | ".join(ci(ml["checkpoints"][c][k]) for c in cols)))
        L.append("")
    L.append("### Orthographic variants of `wait` (top 10 per checkpoint; share of all occurrences). ORTH = share of `Wait,` (pre-registered); ORTH_max = modal-variant share (exploratory)\n")
    for c in cols:
        L.append("**S1@%s**: ORTH_max = %s (%s). " % (c, fmt(cks[c]["ORTH_max"]), cks[c]["ORTH_max_variant"]) +
                 "; ".join("`%s` %s" % (v["variant"], fmt(v["share"], 3)) for v in cks[c]["orth_variants"]))
        L.append("")
    L.append("### Verbatim marker-initial sentences (10 random per checkpoint; `\\n` = line break)\n")
    for c in cols:
        L.append("**S1@%s**" % c)
        for e in cks[c]["marker_initial_examples"]:
            L.append("- (pid %d, sample %d) %s" % (e["pid"], e["sid"], e["sentence"]))
        L.append("")
    if not ins:
        return "\n".join(L) + "\n"
    for oc, title in (("judge_revise", "judge (Qwen2.5-7B-Instruct, arm-blind)"), ("rule_revise", "marker-independent rule")):
        o = ins["outcomes"][oc]
        for s, sname in (("late", "late-derived prefixes (S1@1900 traces)"), ("base", "base-derived prefixes (S1@base traces)")):
            if s not in o:
                continue
            d = o[s]
            L.append("### P(REVISE) by arm, %s outcome, %s: %d prefixes x %d continuations\n" % (title, sname, d["n_prefixes"], d["n_per"]))
            L.append("| policy | " + " | ".join(ARM_LABEL[a] for a in ARM_ORDER) + " |")
            L.append("|---|" + "---|" * len(ARM_ORDER))
            for pol, v in d["policies"].items():
                L.append("| %s | " % pol + " | ".join(ci_p(v["arms"][a]) if a in v["arms"] else "n/a" for a in ARM_ORDER) + " |")
            L.append("")
            if "estimates" in d:
                e = d["estimates"]
                L.append("Estimands (%s, %s; RD in percentage points; OR with 0.5 cell correction = Firth for a binary predictor):\n" % (title, s))
                cie = lambda pol, key, dp: ci(e[pol + ":" + key], dp) if (pol + ":" + key) in e else "n/a"   # noqa: E731  (arms differ by checkpoint)
                L.append("| policy | RD (Wait \u2212 Next) | OR (Wait vs Next) | RD (Wait \u2212 Okay) | OR (Wait vs Okay) | RD_syn (Wait vs {Hold on, Hmm}) | OR_syn | RD (Wait \u2212 \u2205) |")
                L.append("|---|---|---|---|---|---|---|---|")
                for pol in d["policies"]:
                    L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (pol, cie(pol, "RD_next", 1), cie(pol, "OR_next", 2), cie(pol, "RD_op", 1), cie(pol, "OR_op", 2),
                                                                      cie(pol, "RD_syn", 1), cie(pol, "OR_syn", 2), cie(pol, "RD_none", 1)))
                for tag, rd, orr, dd in (("Next (primary, D21)", "R_amp_RD_next", "R_amp_OR_next", "dRD_next_last_minus_base"),
                                         ("Okay", "R_amp_RD", "R_amp_OR", "dRD_op_last_minus_base")):
                    if rd in e:
                        L.append("\nAmplification vs %s: R_amp = RD(last)/RD(base) = %s (valid replicates %.2f); R_amp(OR) = %s; D_amp = RD(last) \u2212 RD(base) = %s points.\n" %
                                 (tag, ci(e[rd], 2), e[rd]["valid_frac"], ci(e[orr], 2), ci(e[dd], 1)))
        if "crossed" in o:
            L.append("Crossed 2x2 (%s outcome), RD_op in points:\n" % title)
            L.append("| quantity | estimate [95% CI] |\n|---|---|")
            for k, v in o["crossed"].items():
                L.append("| %s | %s |" % (k, ci(v, 1)))
            L.append("")
    L.append("### Diagnostics\n")
    L.append("| policy | judge category mix (continue / revise_value / restart / reverify_confirm / branch / invalid) | kappa(rule, judge) |")
    L.append("|---|---|---|")
    for pol, d in ins["diagnostics"].items():
        L.append("| %s | %s | %s |" % (pol, " / ".join(fmt(d["cats"][c], 2) for c in ("continue", "revise_value", "restart", "reverify_confirm", "branch", "invalid")), fmt(d["kappa_rule_vs_judge"], 2)))
    L.append("\nSpearman rho(RD_op judge, step) over policies: %s\n" % ins.get("trend_RD_op"))
    return ("\n".join(L) + "\n").replace("S1@", SERIES + "@")


# ----------------------------------------------------------------------------- plots
def make_plots(h1: dict, ins: Optional[dict], figdir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "figures" / "style"))
    import matplotlib
    matplotlib.use("Agg")
    import palette as P

    P.use()
    plt = matplotlib.pyplot
    cks = h1["checkpoints"]
    cols = [n for n, _ in CKPTS if n in cks]
    x = np.array([xpos(cks[c]["step"]) for c in cols], dtype=float)

    def series(k):
        e = [cks[c]["indices"][k] for c in cols]
        return tuple(np.array([v[f] for v in e], dtype=float) for f in ("est", "lo", "hi"))

    def line(ax, k, color, label, ls="-"):
        y, lo, hi = series(k)
        if np.isfinite(y).sum() == 0:
            return
        ax.plot(x, y, color=color, ls=ls, label=label, **P.marker_kw(color, markersize=3))
        P.band(ax, x, lo, hi, color)

    panels = [("(a) frequency of wait", [("FREQ_wait_1k", P.BLUE, "per 1k tokens")], "per 1k tokens"),
              ("(b) positional fixation", [("SI", P.BLUE, "SI"), ("SI_conn", P.SKY, "SI+But"), ("ORTH", P.VERMILION, "ORTH")], "share"),
              ("(c) entropy (rarefied)", [("H_para_rare", P.BLUE, "H_para"), ("H_pos_rare", P.ORANGE, "H_pos")], "bits"),
              ("(d) obligatoriness / reliability", [("OBL", P.BLUE, "OBL"), ("REL", P.VERMILION, "REL")], "probability"),
              ("(e) erosion", [("EROS", P.BLUE, "EROS")], "tokens to first math token"),
              ("(f) bonding (rarefied)", [("H_next_rare", P.BLUE, "H_next")], "bits")]
    fig, axes = P.subplots("text", aspect=0.42, nrows=2, ncols=3)
    for ax, (title, ks, ylab) in zip(axes.ravel(), panels):
        for k, col, lab in ks:
            line(ax, k, col, lab)
        P.panel_title(ax, title)
        ax.set_xlabel("RL step (0 = base)")
        ax.set_ylabel(ylab)
        if len(ks) > 1:
            ax.legend(loc="best", ncols=1, fontsize=6.5, handlelength=1.2)
    P.save(fig, figdir / "P0_h1")
    plt.close(fig)

    if not ins or "late" not in ins["outcomes"]["judge_revise"]:
        return
    o = ins["outcomes"]["judge_revise"]
    late = o["late"]
    pols = [p for p in (SERIES + "@%s" % n for n, _ in CKPTS) if p in late["policies"]]
    xs = np.array([xpos(dict(CKPTS)[p.split("@")[1]]) for p in pols], dtype=float)
    fig, (a1, a2) = P.subplots("text", aspect=0.36, ncols=2)
    colors = {"wait": P.BLUE, "hold_on": P.ORANGE, "hmm": P.GREEN, "okay": P.GREY, "next": P.SKY, "none": P.INK_2}
    for arm in ("wait", "next", "hold_on", "hmm", "okay", "none"):
        have = [p for p in pols if arm in late["policies"][p]["arms"]]   # arms differ by checkpoint (core vs full battery)
        if not have:
            continue
        xa = np.array([xpos(dict(CKPTS)[p.split("@")[1]]) for p in have], dtype=float)
        v = [late["policies"][p]["arms"][arm] for p in have]
        a1.plot(xa, [d["p"] for d in v], color=colors[arm], ls="--" if arm == "none" else "-", label=ARM_LABEL[arm],
                **P.marker_kw(colors[arm], markersize=3))
        P.band(a1, xa, [d["lo"] for d in v], [d["hi"] for d in v], colors[arm])
    P.panel_title(a1, "(a) P(REVISE) after insertion")
    a1.set_xlabel("continuing policy: RL step (0 = base)")
    a1.set_ylabel("P(revise the last step)")
    a1.legend(loc="upper left", ncols=2, fontsize=6.5, handlelength=1.2)
    est = late["estimates"]
    rdk = "RD_next" if all(("%s:RD_next" % p) in est for p in pols) else "RD_op"
    pols = [p for p in pols if ("%s:%s" % (p, rdk)) in est]
    xs = np.array([xpos(dict(CKPTS)[p.split("@")[1]]) for p in pols], dtype=float)
    y = [est["%s:%s" % (p, rdk)] for p in pols]
    a2.plot(xs, [d["est"] for d in y], color=P.BLUE, label="late-derived prefixes", **P.marker_kw(P.BLUE, markersize=3))
    P.band(a2, xs, [d["lo"] for d in y], [d["hi"] for d in y], P.BLUE)
    if "base" in o:
        be = o["base"]["estimates"]
        bp = [p for p in (SERIES + "@base", SERIES + "@" + LAST) if "%s:%s" % (p, rdk) in be]
        bx = [xpos(dict(CKPTS)[p.split("@")[1]]) for p in bp]
        a2.errorbar(bx, [be["%s:%s" % (p, rdk)]["est"] for p in bp],
                    yerr=[[be["%s:%s" % (p, rdk)]["est"] - be["%s:%s" % (p, rdk)]["lo"] for p in bp],
                          [be["%s:%s" % (p, rdk)]["hi"] - be["%s:%s" % (p, rdk)]["est"] for p in bp]],
                    fmt="s", color=P.VERMILION, mfc="white", ms=3.5, label="base-derived prefixes")
    a2.axhline(0.0, color=P.GREY, lw=0.6, ls=":")
    P.panel_title(a2, "(b) operator strength (RD)")
    a2.set_xlabel("continuing policy: RL step (0 = base)")
    a2.set_ylabel("P(Wait,) − P(Okay,)  [points]")
    a2.legend(loc="upper left", fontsize=6.5, handlelength=1.2)
    P.save(fig, figdir / "P0_insertion")
    plt.close(fig)


def collect_run_meta(root: Path, out: Path):
    """Consolidate every stage meta (config hash, code hash, seeds, gpu-hours, versions) into one run_meta.json."""
    stages = {}
    for f in sorted(glob.glob(str(root / "**" / "*.meta.json"), recursive=True)):
        m = common.read_json(f)
        stages[Path(f).name.replace(".meta.json", "")] = {
            "stage": m["stage"], "config_hash": m["config_hash"], "code_hash": m["code_hash"], "seeds": m["seeds"],
            "wall_seconds": m.get("wall_seconds"), "gpu_hours": (m.get("extra") or {}).get("gpu_hours"),
            "n_records": (m.get("extra") or {}).get("n_records"), "packages": m["env"]["packages"], "gpus": m["env"].get("gpus"),
            "template_matches_tokenizer": (m.get("extra") or {}).get("template_matches_tokenizer"),
            "prompt_token_ids_first3": (m.get("extra") or {}).get("prompt_token_ids_first3")}
    total_gpu_h = sum(v["gpu_hours"] or 0 for v in stages.values())
    common.write_json(out, {"stages": stages, "total_gpu_hours_generation_stages": round(total_gpu_h, 2)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--excerpts", default=None, help="exp/P0/fig1_excerpts.json (for the fig1.* keys)")
    ap.add_argument("--series", default="S1")
    ap.add_argument("--keys_series", default="s1", help="series id in RESULTS_KEYS (s1, s2, s3a, ...)")
    ap.add_argument("--ckpts", default=None, help="'base:0,100:100,...,final:' (empty step = final)")
    ap.add_argument("--last", default="1900")
    ap.add_argument("--stage_map", default=None, help="'early=100,mid=700,late=1900,final=final'")
    ap.add_argument("--main", action="store_true", help="main run (not pilot)")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--judge_tag", default=None, help="e.g. oss: report the second judge (inputs judged*_oss*, revlab_oss; outputs and keys suffixed _oss)")
    ap.add_argument("--calib_e3", default=None, help="exp/gold/human_audit_v2_results.json: sens/spec draws for the E3 outcome judge (audit v2); --calib (audit v1) then only calibrates REL/OBL")
    ap.add_argument("--calib", default=None, help="exp/gold/human_audit_results.json: adds *_corr keys (judge-misclassification corrected)")
    a = ap.parse_args()
    if a.ckpts or a.series != "S1" or a.main:
        configure(a.series, a.ckpts or ",".join("%s:%s" % (n, s_) for n, s_ in CKPTS), a.last, a.keys_series, a.stage_map, not a.main)
    global SUF
    t0 = time.time()
    res = Path(a.res)
    SUF = "_" + a.judge_tag if a.judge_tag else ""
    h1 = run_h1(res)
    trends = h1_trends(h1)
    common.write_json(res / ("h1_indices%s.json" % SUF), {**h1, "trends": trends})
    write_h1_csv(res / ("h1_indices%s.csv" % SUF), h1)
    ins = run_insertion(res)
    if ins:
        common.write_json(res / ("insertion_pilot%s.json" % SUF), ins)
        for src in ("late", "base"):
            j = insertion_json(ins, src, pilot=PILOT, series=SERIES)
            if j:
                common.write_json(res / ("insertion_%s%s.json" % (src, SUF)), j)
        write_ins_csv(res / ("insertion_pilot%s.csv" % SUF), ins)
    (res / ("tables%s.md" % SUF)).write_text(tables_md(h1, trends, ins), encoding="utf-8")
    if not a.no_plots:
        make_plots(h1, ins, Path(a.figdir))
    if not SUF:
        collect_run_meta(res, res / "run_meta.json")
    # json2tex export under the RESULTS_KEYS scheme (pre-rounded values + src); the writer wraps P0 values in \pilot{}
    from . import keys as K

    ex = K.Exporter()
    sm = STAGE_MAP or K.P0_STAGE_MAP
    K.export_h1(ex, {**h1, "trends": trends}, KEYS_SERIES, str(res / ("h1_indices%s.json" % SUF)), sm)
    if ins:
        K.export_e3(ex, ins, KEYS_SERIES, str(res / ("insertion_pilot%s.json" % SUF)), sm)
    if a.calib and Path(a.calib).exists():
        import numpy as _np
        dkey = "posterior_draws_sens_spec_oss" if a.judge_tag == "oss" else "posterior_draws_sens_spec"
        d_e3 = _np.array(common.read_json(a.calib_e3)[dkey]) if a.calib_e3 and Path(a.calib_e3).exists() else None
        d_v1 = _np.array(common.read_json(a.calib)[dkey]) if dkey in common.read_json(a.calib) else _np.array(common.read_json(a.calib)["posterior_draws_sens_spec"])
        K.export_corrections(ex, ins, h1, d_v1, KEYS_SERIES, res, SERIES, sm, a.calib, draws_e3=d_e3, src_calib_e3=a.calib_e3 or "exp/gold/human_audit_v2_results.json")
    exc = Path(a.excerpts) if a.excerpts else None
    if exc and exc.exists():
        K.export_fig1(ex, common.read_json(exc), str(exc))
    if SUF:   # second judge: keep only judge-dependent keys, suffix the quantity (h4.s1.late.rd_op_late -> rd_op_late_oss)
        judge_q = ("obl", "rel", "relstrict", "revrate")
        kept = []
        for e in ex.entries:
            parts = e["key"].split(".")
            if parts[0] == "h4" or (parts[0] == "h1" and parts[3].split("_")[0] in judge_q and "rule" not in parts[3]):
                parts[3] = parts[3] + SUF
                kept.append({**e, "key": ".".join(parts)})
        ex.entries = kept
    common.write_json(res / ("results_keys%s.json" % SUF), {"pilot": PILOT, "stage_map": sm, "judge_tag": a.judge_tag, "entries": ex.entries})
    common.write_meta(res / "report.meta.json", "report", {"bootstrap_B": B, "ckpts": CKPTS, "arms": ARM_ORDER, "min_occ_rarefaction": MIN_OCC},
                      {"rng_seed": SEED}, inputs=list(res.glob("gen/*.graded.jsonl.gz")),
                      outputs=[res / ("h1_indices%s.json" % SUF), res / ("insertion_pilot%s.json" % SUF), res / ("tables%s.md" % SUF)], t_start=t0)
    print("[report] done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
