#!/usr/bin/env python3
"""Provenance tables: the compact main-body table (tab:main) and the full appendix table (tab:provenance).

    sh tools/regen_results.sh && python3 tools/make_prov_tables.py     # writes tables/tab_main.tex, tables/tab_provenance.tex

Every cell is computed here from numbers/results.tex (itself generated from the producers' key files) or, for the two
OLMo Think trajectories whose key files carry one-decimal rates, from their index files; nothing is typed by hand.
Format (lead, 2026-10-04): wait per 1k with 2 decimals, "0" = no occurrence, "<.01" = non-zero below .01;
95% CIs are exact Poisson on the occurrence count (as in the hero figure); OLMo-3 RL-Zero shows mean +- s.d. over its
six decoding seeds. Bold / underline: largest / second "after" value among rows whose CI excludes the "before" value
and lies above it (the marker installed or amplified); grey: the CI includes the "before" value (no detectable change).
Grey = the before and after intervals overlap. A row whose keys are missing is DROPPED (and listed on stdout), never shown with a placeholder.
"""
import json
import re
from pathlib import Path

from scipy.stats import chi2

RES = Path("numbers/results.tex")
IDX = {"s2": Path("../exp/results/main/S2/h1_indices.json"), "s9": Path("../exp/results/s9/S9/h1_indices.json")}
K = {}
for line in RES.read_text().splitlines():
    m = re.match(r"\\resdef\{([^}]*)\}\{(.*)\}(\s+%.*)?$", line)
    if m:
        K[m.group(1)] = m.group(2)


class Missing(Exception):
    pass


def num(key):
    if key not in K:
        raise Missing(key)
    v = K[key].replace("{,}", "").replace(r"\ensuremath{-}", "-")
    return float(v)


def poisson(n, tok):
    """Rate per 1k tokens with an exact 95% Poisson interval."""
    lo = 0.0 if n == 0 else chi2.ppf(0.025, 2 * n) / 2
    hi = chi2.ppf(0.975, 2 * n + 2) / 2
    s = 1000.0 / tok
    return {"est": n * s, "lo": lo * s, "hi": hi * s, "n": n}


def cnt(prefix):
    return poisson(num(prefix + ".nwait"), num(prefix + ".ntok"))


def idx(series, ck):
    x = json.load(open(IDX[series]))["checkpoints"][ck]["indices"]
    n = x["n_wait"]["est"]
    return poisson(n, n * 1000.0 / x["FREQ_wait_1k"]["est"])


def seeds(prefix):
    return {"est": num(prefix + "_mean"), "sd": num(prefix + "_std")}


def kv(key, ci=True):
    d = {"est": num(key)}
    if ci and key + ".lo" in K:
        d.update(lo=num(key + ".lo"), hi=num(key + ".hi"))
    return d


MINUS = r"\ensuremath{-}"


def f_rate(x):
    if x == 0:
        return "0"
    if x < 0.01:
        return r"$<$.01"
    return f"{x:.2f}"


def f_num(x, d=2):
    s = f"{x:.{d}f}"
    if s.startswith("-"):
        s = s[1:] if float(s) == 0 else MINUS + s[1:]
    return s


def cell(v, fmt, mark=""):
    s = fmt(v["est"])
    if mark in ("best", "second"):
        s = r"\%s{%s}" % (mark, s)
    if "sd" in v:
        s += r"\std{%s}" % fmt(v["sd"])
    elif "lo" in v:
        s += r"\ci{%s}{%s}" % (fmt(v["lo"]), fmt(v["hi"]))
    return r"\nsig{%s}" % s if mark == "ns" else s


def iv(v):
    """Interval of a cell: mean +- 2 s.d. over seeds, else its 95% CI, else the point."""
    if "sd" in v:
        return v["est"] - 2 * v["sd"], v["est"] + 2 * v["sd"]
    return v.get("lo", v["est"]), v.get("hi", v["est"])


def changed_up(before, after):
    """True if the after interval lies entirely above the before interval."""
    return before is not None and after is not None and iv(after)[0] > iv(before)[1]


def no_change(before, after):
    """True if the before and after intervals overlap."""
    if before is None or after is None:
        return False
    (b0, b1), (a0, a1) = iv(before), iv(after)
    return a0 <= b1 and b0 <= a1


QWEN35 = {"Qwen3.5-0.8B": "0p8b", "Qwen3.5-2B": "2b", "Qwen3.5-4B": "4b", "Qwen3.5-9B": "9b"}
# A Qwen3.5 pair with >= 20% truncated responses on either side is flagged and left unmarked (coordinator rule, Oct 4);
# its direction is stated in the text only from finished-only rates.
TRUNC = {n for n, t in QWEN35.items()
         if any(float(K.get(f"ctl.bb_qwen35_{t}{s}.final.trunc", "0")) >= 20 for s in ("", "_base"))}


def mark_column(rows, field):
    """Bold / underline the two largest detectable increases; grey the rows with no detectable change."""
    ups = sorted([r for r in rows if r["name"] not in TRUNC and changed_up(r[field][0], r[field][1])], key=lambda r: -r[field][1]["est"])
    for r in rows:
        r.setdefault("marks", {})[field] = "ns" if no_change(*r[field]) else ""
    for r, m in zip(ups[:2], ("best", "second")):
        r["marks"][field] = m
    for r in rows:
        if r["name"] in TRUNC:
            r["marks"][field] = ""


# ------------------------------------------------------------------------------------------------------------- rows
# (group, model, recipe/base, acc (before, after), wait (before, after), d' (before, after) | None, si/orth key prefix | None)
def orz32_id():
    """D47b: the own-template ORZ-32B pair supersedes the S8-harness pair once it exists."""
    return "orz32" if "h1.zr.orz32.final.nwait" in K else "s8orz"


def A(b, a):
    return (lambda: (kv(b, False), kv(a, False)))


MAIN = [
    ("Zero-RL from a base without the marker", [
        ("ORZ-32B", "PPO", lambda: (kv(f"ctl.zr.{orz32_id()}.base.acc", False), kv(f"ctl.zr.{orz32_id()}.final.acc", False)),
         lambda: (cnt(f"h1.zr.{orz32_id()}.base"), cnt(f"h1.zr.{orz32_id()}.final")), None),
        ("Qwen2.5-Math-7B", "GRPO (ours)", A("ctl.zr.s7r7b.base.acc", "ctl.zr.s7r7b.final.acc"),
         lambda: (cnt("h1.zr.s7r7b.base"), cnt("h1.zr.s7r7b.final")), None),
        ("REF-Qwen3-4B (ours)", "GRPO (ours)", A("ctl.zr.q3r1.base.acc", "ctl.zr.q3r1.final.acc"),
         lambda: (cnt("h1.zr.q3r1.base"), cnt("h1.zr.q3r1.final")), None),
        ("SimpleRL-Zoo-Mistral-24B", "GRPO", A("ctl.zr.sz_m24.base.acc", "ctl.zr.sz_m24.final.acc"),
         lambda: (cnt("h1.zr.sz_m24.base"), cnt("h1.zr.sz_m24.final")), None),
        ("General-Reasoner-Qwen3-4B (exception)", "GRPO", A("ctl.zr.gr3_4.base.acc", "ctl.zr.gr3_4.final.acc"),
         lambda: (cnt("h1.zr.gr3_4.base"), cnt("h1.zr.gr3_4.final")), None)]),
    ("Fine-tuned on reasoning traces", [
        ("R1-Distill-Qwen-32B", "SFT", A("ctl.s8base.final.acc", "ctl.s8r1d.final.acc"),
         lambda: (cnt("h1.s8base.final"), cnt("h1.s8r1d.final")), None),
        ("Phi-4-reasoning", "SFT", A("ctl.pop_phi4.final.acc", "ctl.pop_phi4r.final.acc"),
         lambda: (cnt("h1.pop_phi4.final"), cnt("h1.pop_phi4r.final")), None),
        ("MiMo-7B-SFT", "SFT", A("ctl.bb_mimo_base.final.acc", "ctl.bb_mimo_sft.final.acc"),
         lambda: (cnt("h1.bb_mimo_base.final"), cnt("h1.bb_mimo_sft.final")), None)]),
    ("Qwen3: SFT and RL on its own base (before = its base)", [
        ("Qwen3-8B", "SFT+RL", A("ctl.pop_qwen3_8b_base.final.acc", "ctl.pop_qwen3_8b.final.acc"),
         lambda: (cnt("h1.pop_qwen3_8b_base.final"), cnt("h1.pop_qwen3_8b.final")), None),
        ("Qwen3-14B", "SFT+RL", A("ctl.pop_qwen3_14b_base.final.acc", "ctl.pop_qwen3_14b.final.acc"),
         lambda: (cnt("h1.pop_qwen3_14b_base.final"), cnt("h1.pop_qwen3_14b.final")), None),
        ("Qwen3-30B-A3B", "SFT+RL", A("ctl.pop_qwen3b30.final.acc", "ctl.pop_qwen3t30.final.acc"),
         lambda: (cnt("h1.pop_qwen3b30.final"), cnt("h1.pop_qwen3t30.final")), None)]),
    ("Training stages of OLMo-3 7B (before = previous stage)", [
        ("Pretraining", "stage 1", A("ctl.s1pre.stage1-step141000.acc", "ctl.s1pre.stage1-step1413814.acc"),
         lambda: (cnt("h1.s1pre.stage1-step141000"), cnt("h1.s1pre.stage1-step1413814")), None),
        ("Mid-training", "stage 2", A("ctl.s1pre.stage1-step1413814.acc", "ctl.s1pre.stage2-step47684.acc"),
         lambda: (cnt("h1.s1pre.stage1-step1413814"), cnt("h1.s1pre.stage2-step47684")), None),
        ("Base", "stage 3", A("ctl.s1pre.stage2-step47684.acc", "ctl.s1pre.stage3-step11921.acc"),
         lambda: (cnt("h1.s1pre.stage2-step47684"), cnt("h1.s1pre.stage3-step11921")), None)]),
    ("RL from a model that uses the marker", [
        ("OLMo-3 7B RL-Zero", "RL-Zero",
         lambda: (seeds("ctl.s1seeds.base.acc"), seeds("ctl.s1seeds.final.acc")),
         lambda: (seeds("h1.s1seeds.base.freq"), seeds("h1.s1seeds.final.freq")),
         lambda: (kv("h2.s1.base.dprime"), kv("h2.s1.final.dprime"))),
        ("OLMo-3 7B Think", "SFT, DPO, RL", A("ctl.s2.base.acc", "ctl.s2.late.acc"),
         lambda: (idx("s2", "base"), idx("s2", "1375")),
         lambda: (kv("h2.s2.base.dprime"), kv("h2.s2.late.dprime"))),
        (r"OLMo-3.1 32B Think", "SFT, DPO, RL", A("ctl.s9.base.acc", "ctl.s9.late.acc"),
         lambda: (idx("s9", "base"), idx("s9", "2300")),
         lambda: (kv("h2.s9.step0100.dprime"), kv("h2.s9.late.dprime"))),
        ("MiMo-7B-RL-Zero", "RL-Zero", A("ctl.bb_mimo_base.final.acc", "ctl.bb_mimo_rlzero.final.acc"),
         lambda: (cnt("h1.bb_mimo_base.final"), cnt("h1.bb_mimo_rlzero.final")), None),
        ("MiMo-7B-RL", "SFT, then RL", A("ctl.bb_mimo_sft.final.acc", "ctl.bb_mimo_rl.final.acc"),
         lambda: (cnt("h1.bb_mimo_sft.final"), cnt("h1.bb_mimo_rl.final")), None)]),
    ("Newest generation: Qwen3.5 and Qwen3.8 (before = its base)", [
        ("Qwen3.5-0.8B", "post-training", A("ctl.bb_qwen35_0p8b_base.final.acc", "ctl.bb_qwen35_0p8b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_0p8b_base.final"), cnt("h1.bb_qwen35_0p8b.final")), None),
        ("Qwen3.5-2B", "post-training", A("ctl.bb_qwen35_2b_base.final.acc", "ctl.bb_qwen35_2b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_2b_base.final"), cnt("h1.bb_qwen35_2b.final")), None),
        ("Qwen3.5-4B", "post-training", A("ctl.bb_qwen35_4b_base.final.acc", "ctl.bb_qwen35_4b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_4b_base.final"), cnt("h1.bb_qwen35_4b.final")), None),
        ("Qwen3.5-9B", "post-training", A("ctl.bb_qwen35_9b_base.final.acc", "ctl.bb_qwen35_9b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_9b_base.final"), cnt("h1.bb_qwen35_9b.final")), None),
        ("Qwen3.8-27B", "post-training", lambda: (None, kv("ctl.bb_qwen38_27b.final.acc", False)),
         lambda: (None, cnt("h1.bb_qwen38_27b.final")), None)]),
    ("Post-training not documented, no base released", [
        ("openPangu-Embedded-7B-V1.1", "post-training", lambda: (None, kv("ctl.bb_pangu_e7v11.final.acc", False)),
         lambda: (None, cnt("h1.bb_pangu_e7v11.final")), None),
        ("openPangu-Embedded-7B", "post-training", lambda: (None, kv("ctl.bb_pangu_e7m.final.acc", False)),
         lambda: (None, cnt("h1.bb_pangu_e7m.final")), None)]),
]
# Qwen3.8-27B: add to the newest-generation group when its keys land; rows with missing keys are dropped.

FULL = [
    ("RL from models that do not use the marker (zero-RL from a base, except the last row)", [
        ("DAPO-Qwen-32B", "DAPO, Qwen2.5-32B", A("ctl.zr.s8dapo.base.acc", "ctl.zr.s8dapo.final.acc"),
         lambda: (cnt("h1.zr.s8dapo.base"), cnt("h1.zr.s8dapo.final")), None, "h1.s8dapo.final"),
        ("ORZ-32B", "PPO, Qwen2.5-32B", lambda: (kv(f"ctl.zr.{orz32_id()}.base.acc", False), kv(f"ctl.zr.{orz32_id()}.final.acc", False)),
         lambda: (cnt(f"h1.zr.{orz32_id()}.base"), cnt(f"h1.zr.{orz32_id()}.final")), None, "h1.s8orz.final"),
        ("REF-s1 (ours)", "GRPO, Qwen2.5-Math-1.5B", A("ctl.zr.s7r1.base.acc", "ctl.zr.s7r1.final.acc"),
         lambda: (cnt("h1.zr.s7r1.base"), cnt("h1.zr.s7r1.final")), None, None),
        ("REF-s2 (ours)", "GRPO, Qwen2.5-Math-1.5B", A("ctl.zr.s7r2.base.acc", "ctl.zr.s7r2.final.acc"),
         lambda: (cnt("h1.zr.s7r2.base"), cnt("h1.zr.s7r2.final")), None, None),
        ("REF-7B (ours)", "GRPO, Qwen2.5-Math-7B", A("ctl.zr.s7r7b.base.acc", "ctl.zr.s7r7b.final.acc"),
         lambda: (cnt("h1.zr.s7r7b.base"), cnt("h1.zr.s7r7b.final")), None, None),
        ("SimpleRL-Zoo-0.5B", "GRPO, Qwen2.5-0.5B", A("ctl.zr.sz_q05.base.acc", "ctl.zr.sz_q05.final.acc"),
         lambda: (cnt("h1.zr.sz_q05.base"), cnt("h1.zr.sz_q05.final")), None, None),
        ("SimpleRL-Zoo-1.5B", "GRPO, Qwen2.5-1.5B", A("ctl.zr.sz_q15.base.acc", "ctl.zr.sz_q15.final.acc"),
         lambda: (cnt("h1.zr.sz_q15.base"), cnt("h1.zr.sz_q15.final")), None, None),
        ("SimpleRL-Zoo-7B", "GRPO, Qwen2.5-7B", A("ctl.zr.sz_q7.base.acc", "ctl.zr.sz_q7.final.acc"),
         lambda: (cnt("h1.zr.sz_q7.base"), cnt("h1.zr.sz_q7.final")), None, None),
        ("ORZ-7B", "PPO, Qwen2.5-7B", A("ctl.zr.orz7b.base.acc", "ctl.zr.orz7b.final.acc"),
         lambda: (cnt("h1.zr.orz7b.base"), cnt("h1.zr.orz7b.final")), None, None),
        ("SimpleRL-Zoo-Math-7B", "GRPO, Qwen2.5-Math-7B", A("ctl.zr.sz_qm7.base.acc", "ctl.zr.sz_qm7.final.acc"),
         lambda: (cnt("h1.zr.sz_qm7.base"), cnt("h1.zr.sz_qm7.final")), None, None),
        ("Oat-Zero-7B", "Dr.\\ GRPO, Qwen2.5-Math-7B", A("ctl.zr.oat7.base.acc", "ctl.zr.oat7.final.acc"),
         lambda: (cnt("h1.zr.oat7.base"), cnt("h1.zr.oat7.final")), None, None),
        ("SimpleRL-Zoo-14B", "GRPO, Qwen2.5-14B", A("ctl.zr.sz_q14.base.acc", "ctl.zr.sz_q14.final.acc"),
         lambda: (cnt("h1.zr.sz_q14.base"), cnt("h1.zr.sz_q14.final")), None, None),
        ("SimpleRL-Zoo-32B", "GRPO, Qwen2.5-32B", A("ctl.zr.sz_q32.base.acc", "ctl.zr.sz_q32.final.acc"),
         lambda: (cnt("h1.zr.sz_q32.base"), cnt("h1.zr.sz_q32.final")), None, None),
        ("SimpleRL-Zoo-Mistral-7B", "GRPO, Mistral-7B-v0.1", A("ctl.zr.sz_m7.base.acc", "ctl.zr.sz_m7.final.acc"),
         lambda: (cnt("h1.zr.sz_m7.base"), cnt("h1.zr.sz_m7.final")), None, None),
        ("SimpleRL-Zoo-Mistral-24B", "GRPO, Mistral-Small-24B", A("ctl.zr.sz_m24.base.acc", "ctl.zr.sz_m24.final.acc"),
         lambda: (cnt("h1.zr.sz_m24.base"), cnt("h1.zr.sz_m24.final")), None, None),
        ("SimpleRL-Zoo-DSMath-7B", "GRPO, DeepSeek-Math-7B", A("ctl.zr.sz_dsm7.base.acc", "ctl.zr.sz_dsm7.final.acc"),
         lambda: (cnt("h1.zr.sz_dsm7.base"), cnt("h1.zr.sz_dsm7.final")), None, None),
        ("SimpleRL-Zoo-Llama-8B$^\\S$", "GRPO, Llama-3.1-8B", lambda: (None, kv("ctl.zr.sz_l8.final.acc", False)),
         lambda: (None, cnt("h1.zr.sz_l8.final")), None, None),
        ("General-Reasoner-7B", "GRPO, Qwen2.5-7B", A("ctl.zr.gr7.base.acc", "ctl.zr.gr7.final.acc"),
         lambda: (cnt("h1.zr.gr7.base"), cnt("h1.zr.gr7.final")), None, None),
        ("General-Reasoner-14B", "GRPO, Qwen2.5-14B", A("ctl.zr.gr14.base.acc", "ctl.zr.gr14.final.acc"),
         lambda: (cnt("h1.zr.gr14.base"), cnt("h1.zr.gr14.final")), None, None),
        ("General-Reasoner-Qwen3-4B", "GRPO, Qwen3-4B", A("ctl.zr.gr3_4.base.acc", "ctl.zr.gr3_4.final.acc"),
         lambda: (cnt("h1.zr.gr3_4.base"), cnt("h1.zr.gr3_4.final")), None, None),
        ("General-Reasoner-Qwen3-14B", "GRPO, Qwen3-14B", A("ctl.zr.gr3_14.base.acc", "ctl.zr.gr3_14.final.acc"),
         lambda: (cnt("h1.zr.gr3_14.base"), cnt("h1.zr.gr3_14.final")), None, None),
        ("One-Shot-RLVR-1.5B", "GRPO (1 example), Qwen2.5-Math-1.5B", A("ctl.zr.os15.base.acc", "ctl.zr.os15.final.acc"),
         lambda: (cnt("h1.zr.os15.base"), cnt("h1.zr.os15.final")), None, None),
        ("Oat-Zero-1.5B", "Dr.\\ GRPO, Qwen2.5-Math-1.5B", A("ctl.zr.oat15.base.acc", "ctl.zr.oat15.final.acc"),
         lambda: (cnt("h1.zr.oat15.base"), cnt("h1.zr.oat15.final")), None, None),
        ("REF-Qwen3-4B (ours)", "GRPO, Qwen3-4B", A("ctl.zr.q3r1.base.acc", "ctl.zr.q3r1.final.acc"),
         lambda: (cnt("h1.zr.q3r1.base"), cnt("h1.zr.q3r1.final")), None, None),
        (r"open run$^\dagger$", "GRPO, Qwen2.5-1.5B-Inst.", A("ctl.s3a.base.acc", "ctl.s3a.late.acc"),
         lambda: (cnt("h1.s3a.base"), cnt("h1.s3a.late")), None, None)]),
    ("Fine-tuned on reasoning traces (distillation or cold start)", [
        ("QwQ-32B", "cold start+RL, Qwen2.5-32B", A("ctl.s8base.final.acc", "ctl.s8qwq.final.acc"),
         lambda: (cnt("h1.s8base.final"), cnt("h1.s8qwq.final")), None, "h1.s8qwq.final"),
        ("R1-Distill-Qwen-32B", "SFT, Qwen2.5-32B", A("ctl.s8base.final.acc", "ctl.s8r1d.final.acc"),
         lambda: (cnt("h1.s8base.final"), cnt("h1.s8r1d.final")), None, "h1.s8r1d.final"),
        ("Phi-4-reasoning", "SFT, Phi-4", A("ctl.pop_phi4.final.acc", "ctl.pop_phi4r.final.acc"),
         lambda: (cnt("h1.pop_phi4.final"), cnt("h1.pop_phi4r.final")), None, "h1.pop_phi4r.final"),
        ("Phi-4-reasoning-plus", "SFT+RL, Phi-4", A("ctl.pop_phi4.final.acc", "ctl.pop_phi4rp.final.acc"),
         lambda: (cnt("h1.pop_phi4.final"), cnt("h1.pop_phi4rp.final")), None, "h1.pop_phi4rp.final"),
        ("Qwen3-8B", "SFT+RL, Qwen3-8B", A("ctl.pop_qwen3_8b_base.final.acc", "ctl.pop_qwen3_8b.final.acc"),
         lambda: (cnt("h1.pop_qwen3_8b_base.final"), cnt("h1.pop_qwen3_8b.final")), None, "h1.pop_qwen3_8b.final"),
        ("Qwen3-14B", "SFT+RL, Qwen3-14B", A("ctl.pop_qwen3_14b_base.final.acc", "ctl.pop_qwen3_14b.final.acc"),
         lambda: (cnt("h1.pop_qwen3_14b_base.final"), cnt("h1.pop_qwen3_14b.final")), None, "h1.pop_qwen3_14b.final"),
        ("Qwen3-30B-A3B", "SFT+RL, Qwen3-30B-A3B", A("ctl.pop_qwen3b30.final.acc", "ctl.pop_qwen3t30.final.acc"),
         lambda: (cnt("h1.pop_qwen3b30.final"), cnt("h1.pop_qwen3t30.final")), None, "h1.pop_qwen3t30.final"),
        ("OpenThinker-7B", "SFT, Qwen2.5-7B-Inst.", A("ctl.bb_qwen7i.final.acc", "ctl.bb_openthinker7.final.acc"),
         lambda: (cnt("h1.bb_qwen7i.final"), cnt("h1.bb_openthinker7.final")), None, "h1.bb_openthinker7.final"),
        ("Bespoke-Stratos-7B", "SFT, Qwen2.5-7B-Inst.", A("ctl.bb_qwen7i.final.acc", "ctl.bb_bespoke7.final.acc"),
         lambda: (cnt("h1.bb_qwen7i.final"), cnt("h1.bb_bespoke7.final")), None, "h1.bb_bespoke7.final"),
        (r"R1-Distill-Llama-8B$^\ddagger$", "SFT, Llama-3.1-8B", lambda: (None, kv("ctl.pop_r1d_llama8.final.acc", False)),
         lambda: (None, cnt("h1.pop_r1d_llama8.final")), None, "h1.pop_r1d_llama8.final")]),
    ("Training stages of OLMo-3 7B (before = previous stage)", [
        ("Pretraining", "stage 1, step 141k", lambda: (None, kv("ctl.s1pre.stage1-step141000.acc", False)),
         lambda: (None, cnt("h1.s1pre.stage1-step141000")), None, "h1.s1pre.stage1-step141000"),
        ("Pretraining", "stage 1, step 707k", A("ctl.s1pre.stage1-step141000.acc", "ctl.s1pre.stage1-step707000.acc"),
         lambda: (cnt("h1.s1pre.stage1-step141000"), cnt("h1.s1pre.stage1-step707000")), None, "h1.s1pre.stage1-step707000"),
        ("Pretraining", "stage 1, step 1.41M", A("ctl.s1pre.stage1-step707000.acc", "ctl.s1pre.stage1-step1413814.acc"),
         lambda: (cnt("h1.s1pre.stage1-step707000"), cnt("h1.s1pre.stage1-step1413814")), None, "h1.s1pre.stage1-step1413814"),
        ("Mid-training", "stage 2", A("ctl.s1pre.stage1-step1413814.acc", "ctl.s1pre.stage2-step47684.acc"),
         lambda: (cnt("h1.s1pre.stage1-step1413814"), cnt("h1.s1pre.stage2-step47684")), None, "h1.s1pre.stage2-step47684"),
        ("Base", "stage 3", A("ctl.s1pre.stage2-step47684.acc", "ctl.s1pre.stage3-step11921.acc"),
         lambda: (cnt("h1.s1pre.stage2-step47684"), cnt("h1.s1pre.stage3-step11921")), None, "h1.s1pre.stage3-step11921")]),
    ("RL from a model that uses the marker (after = last checkpoint)", [
        ("OLMo-3 7B RL-Zero", "RL-Zero",
         lambda: (seeds("ctl.s1seeds.base.acc"), seeds("ctl.s1seeds.final.acc")),
         lambda: (seeds("h1.s1seeds.base.freq"), seeds("h1.s1seeds.final.freq")), None, "h1.s1.final"),
        ("OLMo-3 7B Think", "SFT, DPO, RL", A("ctl.s2.base.acc", "ctl.s2.late.acc"),
         lambda: (idx("s2", "base"), idx("s2", "1375")), None, "h1.s2.late"),
        ("OLMo-3.1 32B Think", "SFT, DPO, RL", A("ctl.s9.base.acc", "ctl.s9.late.acc"),
         lambda: (idx("s9", "base"), idx("s9", "2300")), None, "h1.s9.late"),
        ("MiMo-7B-RL-Zero", "RL-Zero, MiMo-7B", A("ctl.bb_mimo_base.final.acc", "ctl.bb_mimo_rlzero.final.acc"),
         lambda: (cnt("h1.bb_mimo_base.final"), cnt("h1.bb_mimo_rlzero.final")), None, "h1.bb_mimo_rlzero.final"),
        ("DeepScaleR-1.5B", "RL, R1-Distill-1.5B", A("ctl.bb_r1d1p5b.final.acc", "ctl.bb_deepscaler.final.acc"),
         lambda: (cnt("h1.bb_r1d1p5b.final"), cnt("h1.bb_deepscaler.final")), None, "h1.bb_deepscaler.final"),
        ("Light-R1-14B-DS", "RL, R1-Distill-14B", A("ctl.bb_r1d14.final.acc", "ctl.bb_light14.final.acc"),
         lambda: (cnt("h1.bb_r1d14.final"), cnt("h1.bb_light14.final")), None, "h1.bb_light14.final")]),
    ("Newest generation: Qwen3.5 (before = its base)", [
        ("Qwen3.5-0.8B", "post-training", A("ctl.bb_qwen35_0p8b_base.final.acc", "ctl.bb_qwen35_0p8b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_0p8b_base.final"), cnt("h1.bb_qwen35_0p8b.final")), None, "h1.bb_qwen35_0p8b.final"),
        ("Qwen3.5-2B", "post-training", A("ctl.bb_qwen35_2b_base.final.acc", "ctl.bb_qwen35_2b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_2b_base.final"), cnt("h1.bb_qwen35_2b.final")), None, "h1.bb_qwen35_2b.final"),
        ("Qwen3.5-4B", "post-training", A("ctl.bb_qwen35_4b_base.final.acc", "ctl.bb_qwen35_4b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_4b_base.final"), cnt("h1.bb_qwen35_4b.final")), None, "h1.bb_qwen35_4b.final"),
        ("Qwen3.5-9B", "post-training", A("ctl.bb_qwen35_9b_base.final.acc", "ctl.bb_qwen35_9b.final.acc"),
         lambda: (cnt("h1.bb_qwen35_9b_base.final"), cnt("h1.bb_qwen35_9b.final")), None, "h1.bb_qwen35_9b.final"),
        ("Qwen3.8-27B", "post-training", lambda: (None, kv("ctl.bb_qwen38_27b.final.acc", False)),
         lambda: (None, cnt("h1.bb_qwen38_27b.final")), None, "h1.bb_qwen38_27b.final")]),
    ("OLMo-2 7B lineage (before = the base)", [
        ("OLMo-2-7B-SFT", "SFT, OLMo-2-7B", A("ctl.bb_olmo2_base.final.acc", "ctl.bb_olmo2_sft.final.acc"),
         lambda: (cnt("h1.bb_olmo2_base.final"), cnt("h1.bb_olmo2_sft.final")), None, None),
        ("OLMo-2-7B-DPO", "DPO after SFT", A("ctl.bb_olmo2_base.final.acc", "ctl.bb_olmo2_dpo.final.acc"),
         lambda: (cnt("h1.bb_olmo2_base.final"), cnt("h1.bb_olmo2_dpo.final")), None, None),
        ("OLMo-2-7B-Instruct", "RLVR after DPO", A("ctl.bb_olmo2_base.final.acc", "ctl.bb_olmo2_instruct.final.acc"),
         lambda: (cnt("h1.bb_olmo2_base.final"), cnt("h1.bb_olmo2_instruct.final")), None, None)]),
    ("Post-training not documented (openPangu; no base released)", [
        ("openPangu-Embedded-7B-V1.1", "post-trained release", lambda: (None, kv("ctl.bb_pangu_e7v11.final.acc", False)),
         lambda: (None, cnt("h1.bb_pangu_e7v11.final")), None, "h1.bb_pangu_e7v11.final"),
        ("openPangu-Embedded-7B", "post-trained release", lambda: (None, kv("ctl.bb_pangu_e7m.final.acc", False)),
         lambda: (None, cnt("h1.bb_pangu_e7m.final")), None, "h1.bb_pangu_e7m.final")]),
]


def build(groups):
    out, dropped = [], []
    for g, rows in groups:
        rr = []
        for row in rows:
            name, recipe, facc, fwait, fd = row[:5]
            form = row[5] if len(row) > 5 else None
            try:
                r = {"name": name, "recipe": recipe, "acc": facc(), "wait": fwait(), "d": fd() if fd else None}
                if form:
                    r["form"] = (kv(form + ".si", False), kv(form + ".orth", False)) if form + ".si" in K else None
            except (Missing, KeyError) as e:
                dropped.append(f"{name}: missing {e}")
                continue
            rr.append(r)
        if rr:
            out.append((g, rr))
    allrows = [r for _, rr in out for r in rr]
    mark_column(allrows, "wait")
    drows = [r for r in allrows if r["d"]]
    mark_column(drows, "d")
    return out, dropped


def acc(v):
    return r"\na" if v is None else cell(v, lambda x: f"{x:.1f}")


def wait(v, mark=""):
    return r"\na" if v is None else cell(v, f_rate, mark)


def cue(after, before=None):
    """Colour cue for an after-training wait cell: orange shading (log scale) + arrow when intervals do not overlap."""
    if after is None:
        return "", ""
    import math
    lvl = round(36 * math.log10(1 + 10 * after["est"]) / math.log10(1 + 150))
    shade = r"\cellcolor{cCmpA!%d}" % max(0, min(40, lvl)) if lvl > 0 else ""
    arrow = ""
    if before is not None and not no_change(before, after):
        arrow = r"\,\textcolor{cInterv}{$\uparrow$}" if after["est"] > before["est"] else r"\,\textcolor{cOurs}{$\downarrow$}"
    return shade, arrow

CUE_NOTE = r"Shading: \form{wait} per 1k after training (log scale, darker = more); \textcolor{cInterv}{$\uparrow$}/\textcolor{cOurs}{$\downarrow$}: rise/fall with non-overlapping intervals; grey rows: our runs."


def main_table():
    groups, dropped = build(MAIN)
    L = [r"% AUTO-GENERATED by tools/make_prov_tables.py -- do not edit by hand.",
         r"\begin{table*}[t]", r"  \centering", r"  \tabstyle", r"  \setlength{\tabcolsep}{2.5pt}",
         r"  \caption{\textbf{Zero-RL rarely installs \form{wait}; training data does, and released RL-Zero recipes amplify it.} "
          r"Bold, underline: largest, second increase; grey text: intervals overlap; $\pm$: s.d.\ over six decoding seeds, else [95\% CI]. " + CUE_NOTE + "}",
          r"  \label{tab:main}",
         r"  \fitwidth{\begin{tabular}{@{}ll cc cc@{}}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{\textbf{MATH-500 (\%)}} & \multicolumn{2}{c}{\textbf{\form{wait} per 1k tokens}} \\",
         r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
         r"    \textbf{Model} & \textbf{Training} & before & after & before & after \\"]
    nobase = []
    for i, (g, rows) in enumerate(groups):
        nobase += [r for r in rows if r["acc"][0] is None and r["wait"][0] is None]
        rows = [r for r in rows if not (r["acc"][0] is None and r["wait"][0] is None)]
        if not rows:
            continue
        L.append(r"    \addlinespace[0.5mm]\midrule" if i else r"    \midrule")
        L.append(r"    \multicolumn{6}{@{}l}{\textsc{%s}} \\" % g.lower().replace("olmo-3 7b", "olmo-3-1025-7b"))
        for r in rows:
            d = r["d"]
            dcells = [r"\na", r"\na"] if not d else [cell(d[0], f_num) + (r"$^\dagger$" if "32B Think" in r["name"] else ""),
                                                    cell(d[1], f_num, r["marks"]["d"])]
            nb = r"\textit{no base}" if "no base" in g else (r"\textit{n.m.}" if r["acc"][0] is None and r["wait"][0] is None else None)
            L.append(("    \\rowcolor{gray!10}" if "ours" in (r["name"] + r["recipe"]) else "    ") + "%s & %s & %s & %s & %s & %s \\\\" % (
                disp(r["name"]), r["recipe"], nb or acc(r["acc"][0]), acc(r["acc"][1]), nb or wait(r["wait"][0]),
                cue(r["wait"][1], r["wait"][0])[0] + wait(r["wait"][1], r["marks"]["wait"]) + (r"$^\ddagger$" if r["name"] in TRUNC else "") + cue(r["wait"][1], r["wait"][0])[1]))
    L += [r"    \bottomrule", r"  \end{tabular}}",

          r"  \par\vspace{2pt}\begin{minipage}{\textwidth}\footnotesize "
          r"Before: the starting model, or the previous OLMo-3 stage. CIs: exact Poisson; 0: no occurrence; $<$.01: non-zero below .01. ORZ-32B: own template, under which its base scores low. $^\ddagger$At least 20\% of responses reach the length limit; direction from finished responses (\Cref{sec:exp:source}). Full table: \Cref{tab:provenance}.\end{minipage}",
          r"\end{table*}"]
    Path("tables/tab_main.tex").write_text("\n".join(L) + "\n")
    N = [r"% AUTO-GENERATED by tools/make_prov_tables.py -- do not edit by hand.", r"\begin{table}[t]", r"  \centering", r"  \tabstyle",
         r"  \caption{\textbf{Models released without a base also write \form{wait}.} MATH-500 accuracy and rate per 1k tokens of the released model; 95\% CI; shading: rate (log scale, darker = more). openPangu: vendor's Chinese system prompt, rate per 1k thinking tokens.}",
         r"  \label{tab:nobase}", r"  \fitwidth{\begin{tabular}{@{}llcc@{}}", r"    \toprule",
         r"    \textbf{Model} & \textbf{Training} & \textbf{MATH-500 (\%)} & \textbf{\form{wait} per 1k} \\", r"    \midrule"]
    N += ["    %s & %s & %s & %s \\\\" % (disp(r["name"]), r["recipe"], acc(r["acc"][1]), cue(r["wait"][1])[0] + wait(r["wait"][1])) for r in nobase]
    N += [r"    \bottomrule", r"  \end{tabular}}", r"\end{table}"]
    Path("tables/tab_main_nobase.tex").write_text("\n".join(N) + "\n")
    return dropped, sum(len(r) for _, r in groups)


ZR_ID = {"DAPO-Qwen-32B": "s8dapo", "ORZ-32B": "s8orz", "REF-s1 (ours)": "s7r1", "REF-s2 (ours)": "s7r2",
         "REF-7B (ours)": "s7r7b", "SimpleRL-Zoo-0.5B": "sz_q05", "SimpleRL-Zoo-1.5B": "sz_q15", "SimpleRL-Zoo-7B": "sz_q7",
         "ORZ-7B": "orz7b", "SimpleRL-Zoo-Math-7B": "sz_qm7", "Oat-Zero-7B": "oat7", "SimpleRL-Zoo-14B": "sz_q14",
         "SimpleRL-Zoo-32B": "sz_q32", "SimpleRL-Zoo-Mistral-7B": "sz_m7", "SimpleRL-Zoo-Mistral-24B": "sz_m24",
         "SimpleRL-Zoo-DSMath-7B": "sz_dsm7", r"SimpleRL-Zoo-Llama-8B$^\S$": "sz_l8",
         "General-Reasoner-7B": "gr7", "General-Reasoner-14B": "gr14", "General-Reasoner-Qwen3-4B": "gr3_4",
         "General-Reasoner-Qwen3-14B": "gr3_14", "One-Shot-RLVR-1.5B": "os15", "Oat-Zero-1.5B": "oat15", "REF-Qwen3-4B (ours)": "q3r1", "REF-Qwen3-4B-s2 (ours)": "q3r2"}


def not_reproduced(name):
    """D47a: a zero-RL row whose accuracy is below half its published value, or >= 20% truncated."""
    g = K.get("h1.zr.%s.gate" % ZR_ID.get(name, ""), "")
    return g.startswith("not reproduced")


def no_reference(name):
    """D47a: a zero-RL row whose authors publish no MATH-500 accuracy (counted neither way)."""
    return K.get("h1.zr.%s.gate" % ZR_ID.get(name, ""), "") == "no published reference"


# Official release names (repo names) for display; internal row names stay short because the gate/flag lookups use them.
OFFICIAL = {
    "ORZ-32B": "Open-Reasoner-Zero-32B", "ORZ-7B": "Open-Reasoner-Zero-7B",
    "SimpleRL-Zoo-Mistral-24B": "Mistral-Small-24B-SimpleRL-Zoo", "SimpleRL-Zoo-Mistral-7B": "Mistral-7B-v0.1-SimpleRL-Zoo",
    "SimpleRL-Zoo-0.5B": "Qwen-2.5-0.5B-SimpleRL-Zoo", "SimpleRL-Zoo-1.5B": "Qwen-2.5-1.5B-SimpleRL-Zoo",
    "SimpleRL-Zoo-7B": "Qwen-2.5-7B-SimpleRL-Zoo", "SimpleRL-Zoo-Math-7B": "Qwen-2.5-Math-7B-SimpleRL-Zoo",
    "SimpleRL-Zoo-14B": "Qwen-2.5-14B-SimpleRL-Zoo", "SimpleRL-Zoo-32B": "Qwen-2.5-32B-SimpleRL-Zoo",
    "SimpleRL-Zoo-DSMath-7B": "DeepSeek-Math-7B-SimpleRL-Zoo", r"SimpleRL-Zoo-Llama-8B$^\S$": r"Llama-3.1-8B-SimpleRL-Zoo$^\S$",
    "Oat-Zero-7B": "Qwen2.5-Math-7B-Oat-Zero", "Oat-Zero-1.5B": "Qwen2.5-Math-1.5B-Oat-Zero",
    "General-Reasoner-7B": "General-Reasoner-Qwen2.5-7B", "General-Reasoner-14B": "General-Reasoner-Qwen2.5-14B",
    "One-Shot-RLVR-1.5B": "One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1",
    "R1-Distill-Qwen-32B": "DeepSeek-R1-Distill-Qwen-32B", r"R1-Distill-Llama-8B$^\ddagger$": r"DeepSeek-R1-Distill-Llama-8B$^\ddagger$",
    "DeepScaleR-1.5B": "DeepScaleR-1.5B-Preview",
    "OLMo-3 7B RL-Zero": "Olmo-3-7B-RL-Zero-Math", "OLMo-3 7B Think": "Olmo-3-7B-Think", "OLMo-3.1 32B Think": "Olmo-3.1-32B-Think",
    "OLMo-2-7B-SFT": "OLMo-2-1124-7B-SFT", "OLMo-2-7B-DPO": "OLMo-2-1124-7B-DPO", "OLMo-2-7B-Instruct": "OLMo-2-1124-7B-Instruct",
    r"open run$^\dagger$": r"Qwen2.5-1.5B-Open-R1-GRPO-math-v1$^\dagger$",
}


def disp(name):
    return OFFICIAL.get(name, name)


def full_table():
    groups, dropped = build(FULL)
    L = [r"% AUTO-GENERATED by tools/make_prov_tables.py -- do not edit by hand.",
         r"\begin{table*}[t]", r"  \centering", r"  \tabstyle\footnotesize\renewcommand{\arraystretch}{0.97}", r"  \setlength{\tabcolsep}{2pt}",
         r"  \fitwidth{\begin{tabular}{@{}ll cc cc@{}}", r"    \toprule",
         r"    & & \multicolumn{2}{c}{\textbf{MATH-500 (\%)}} & \multicolumn{2}{c}{\textbf{\form{wait} per 1k tokens}} \\",
         r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
         r"    \textbf{Model} & \textbf{Training step} & before & after & before & after \\"]
    for g, rows in groups:
        rows = [r for r in rows if r["acc"][0] is not None and r["wait"][0] is not None]
        if not rows:
            continue
        L.append(r"    \midrule")
        L.append(r"    \multicolumn{6}{@{}l}{\textsc{%s}} \\" % g.lower().replace("olmo-3 7b", "olmo-3-1025-7b"))
        for r in rows:
            f = r.get("form")
            nw = r["wait"][1]
            show_form = f and nw is not None and nw.get("n", 1e9) >= 100
            fc = [f"{f[0]['est']:.1f}", f"{f[1]['est']:.1f}"] if show_form else [r"\na", r"\na"]
            L.append("    %s & %s & %s & %s & %s & %s \\\\" % (
                disp(r["name"]) + (r"$^\diamond$" if r["name"].startswith("openPangu") else "") + (r"$^\ast$" if r["name"] == "ORZ-32B" and orz32_id() == "orz32" else "") + (r"$^\dagger$" if K.get("h1.zr.%s.gate" % ZR_ID.get(r["name"], ""), "").startswith("no licence") else "") + (r"$^\circ$" if not_reproduced(r["name"]) else "") + (r"\textsuperscript{n.r.}" if no_reference(r["name"]) else ""), r["recipe"], acc(r["acc"][0]), acc(r["acc"][1]), wait(r["wait"][0]),
                cue(r["wait"][1], r["wait"][0])[0] + wait(r["wait"][1], r["marks"]["wait"]) + (r"$^\P$" if r["name"] in TRUNC else "") + cue(r["wait"][1], r["wait"][0])[1]))
    L += [r"    \bottomrule", r"  \end{tabular}}",
          r"  \caption{\textbf{Full provenance table: all but one reproduced zero-RL run stay without \form{wait}; documented routes to it otherwise pass through training data.} "
          r"Format as in \Cref{tab:main}. Models without a measured starting point are in \Cref{tab:nobase}. " + CUE_NOTE + r"}",
          r"  \label{tab:provenance}",
          r"  \par\vspace{2pt}\begin{minipage}{\textwidth}\footnotesize "
          r"$^\dagger$Secondary, not counted: the release has no licence text (the open run also starts from an instruct model). " + (r"$^\diamond$Run with the vendor's required Chinese system prompt (its evaluation note says the system prompt is empty; without it the model degenerates); \form{wait} per 1k thinking tokens. " if any(r["name"].startswith("openPangu") for _, rr in groups for r in rr) else "") + (r"$^\ast$Base and final use the ORZ prompt template, under which the base scores low; the accuracy gain partly reflects the template. " if orz32_id() == "orz32" else "") + (r"$^\circ$Not reproduced: accuracy below half the published value or at least 20\% truncated; not counted (D47a, \Cref{app:prereg}). " if any(not_reproduced(r["name"]) for _, rr in groups for r in rr) else "") +
          (r"\textsuperscript{n.r.}No MATH-500 reference published by its authors; counted neither way (D47a, \Cref{app:prereg}). " if any(no_reference(r["name"]) for _, rr in groups for r in rr) else "") +
          r"$^\ddagger$Base gated (manual approval); not used. $^\S$Base gated, not measured. $^\P$At least 20\% of responses reach the length limit; not marked, direction from finished responses in \Cref{sec:exp:source}. QwQ-32B's cold-start data are not public \citep{qwen2025qwq}.\end{minipage}",
          r"\end{table*}"]
    Path("tables/tab_provenance.tex").write_text("\n".join(L) + "\n")
    return dropped, sum(len(r) for _, r in groups)


DS = [("aime24", "AIME24"), ("aime25", "AIME25"), ("amc23", "AMC23"), ("gsm8k", "GSM8K"), ("mmlupro", "MMLU-Pro")]
DS_ROWS = [("s1", "OLMo-3 7B RL-Zero", True), ("s2", "OLMo-3 7B Think", True),
           ("szq7", "Qwen2.5-7B $\\rightarrow$ SimpleRL-Zoo", True), ("r1d7", "R1-Distill-Qwen-7B", False)]


def dataset_table():
    """Appendix table: wait per 1k among FINISHED responses, before -> after, per benchmark. Written only when every
    cell exists (no placeholders); otherwise the missing keys are listed and the file is left untouched."""
    missing, L = [], []
    for sid, name, has_base in DS_ROWS:
        cells = []
        for ds, _ in DS:
            try:
                a = cnt(f"h1.ds_{ds}.{sid}.final.fin")
                if has_base:
                    b = cnt(f"h1.ds_{ds}.{sid}.base.fin")
                    cell = "%s $\\rightarrow$ %s" % (f_rate(b["est"]), f_rate(a["est"]))
                    cell = r"\nsig{%s}" % cell if no_change(b, a) else cell
                else:
                    cell = f_rate(a["est"])
                cells.append(cell)
            except Missing as e:
                missing.append(f"{sid}/{ds}: {e}")
        L.append("    %s & %s \\\\" % (name, " & ".join(cells)))
    if missing:
        return missing
    T = [r"% AUTO-GENERATED by tools/make_prov_tables.py -- do not edit by hand.",
         r"\begin{table*}[t]", r"  \centering", r"  \tabstyle", r"  \fitwidth{\begin{tabular}{@{}l ccccc@{}}", r"    \toprule",
         r"    \textbf{Model (before $\rightarrow$ after)} & " + " & ".join(r"\textbf{%s}" % d for _, d in DS) + r" \\", r"    \midrule"]
    T += L + [r"    \bottomrule", r"  \end{tabular}}",
              r"  \caption{\textbf{Zero-RL from Qwen2.5-7B leaves \form{wait} absent on all five benchmarks; models that inherited it keep it on the mathematical ones.} Rate per 1k tokens of finished responses, before $\rightarrow$ after training; grey: 95\% intervals overlap; MMLU-Pro: official few-shot prompt.}",
              r"  \label{tab:datasets}", r"\end{table*}"]
    Path("tables/tab_datasets.tex").write_text("\n".join(T) + "\n")
    return []


if __name__ == "__main__":
    for fn in (main_table, full_table):
        dropped, n = fn()
        print(fn.__name__, n, "rows;", "dropped:", dropped or "none")
    miss = dataset_table()
    print("dataset_table:", "written" if not miss else f"not written, {len(miss)} cells missing (first: {miss[0]})")
