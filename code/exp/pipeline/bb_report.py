"""Backbone-breadth / popular-models panel report (the same machinery as scale_report.py; one 'final' stage per model).

    python -m exp.pipeline.bb_report --root <.../runs/P1/scale> --out <dir> [--judge_tag oss] [--models id ...]
Per model directory <root>/BB_<id>/ (written by run_scale.sh, judged by the bbjudge tasks): idx/math500.json (+aime24), ins/judged_shared[_oss].jsonl.gz on the family-shared base prefixes,
sdt/conts.jsonl.gz. Amplification is paired against the FAMILY BASE on the shared prefixes (R_amp = RD_next(model) / RD_next(base)).
Series ids: popular panel `pop_<id>` for the ids in POP, otherwise `bb_<id>`; Tulu S6 -> `s6` with stages `dpo` and `stepNNNN`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import common, keys as K, sdt, stats as st
from .scale_report import succ_table, read, st_seed

B = 2000
POP = {"pop_qwen3b30", "pop_r1dllama70b", "pop_qwen3t30", "phi4", "phi4r", "phi4rp", "mistral_base", "mistral_base_ps", "magistral", "qwen3_8b_base", "qwen3_8b", "r1d_llama8", "qwen3_14b_base", "qwen3_14b"}
FAM = {"phi": ["phi4", "phi4r", "phi4rp"], "mistral": ["mistral_base"], "qwen3": ["qwen3_8b_base", "qwen3_8b"], "r1dl": ["r1d_llama8"],
       "qwen7": ["qwen7b", "orn7", "r1d7", "acer", "qwen7i", "openthinker7", "bespoke7"], "mimo": ["mimo_base", "mimo_rlzero", "mimo_sft", "mimo_rl"], "glm": ["glm4", "glmz1"],
       "qwen14": ["qwen14b", "r1d14", "light14"], "dsc15": ["r1d1p5b", "deepscaler"], "qwen3_14": ["qwen3_14b_base", "qwen3_14b", "gr3_14m"], "mistral_ps": ["mistral_base_ps"], "r1d15": ["qwen_math15", "r1d1p5"], "qwen3_1p7": ["qwen3_1p7_base", "qwen3_1p7"], "qwen3_4": ["qwen3_4b_base", "qwen3_4b", "gr3_4m"],
       "olmo2": ["olmo2_base", "olmo2_sft", "olmo2_dpo", "olmo2_instruct"], "smollm3": ["smollm3_base", "smollm3"], "granite": ["granite_base", "granite_instruct"], "q35_0p8": ["qwen35_0p8b_base", "qwen35_0p8b"], "q35_2b": ["qwen35_2b_base", "qwen35_2b"], "q35_4b": ["qwen35_4b_base", "qwen35_4b"], "q35_9b": ["qwen35_9b_base", "qwen35_9b"], "q38": ["qwen38_27b"],
       "sz_q05": ["sz_q05_base", "sz_q05"], "sz_q15": ["sz_q15_base", "sz_q15"], "sz_q7": ["sz_q7_base", "sz_q7", "orz7b"], "sz_qm7": ["sz_qm7_base", "sz_qm7", "oat7"], "sz_q14": ["sz_q14_base", "sz_q14"],
       "sz_m7": ["sz_m7_base", "sz_m7"], "sz_ds7": ["sz_ds7_base", "sz_ds7"], "sz_l8": ["sz_l8"], "yi": ["yi_base", "yi_chat"],
       "base32": ["S8_base", "pop_qwen3b30", "pop_r1dllama70b", "pop_qwen3t30"],   # Slurm >=30B popular panel: E3 on the S8 (Qwen2.5-32B base) shared prefixes, paired against S8_base (not re-exported)
       "s6": ["s6_dpo", "s6_step40", "s6_step200", "s6_step600", "s6_step1200", "s6_step1800", "s6_step2440"]}


ALIAS = {"sz_ds7": "sz_dsm7", "sz_ds7_base": "sz_dsm7_base"}   # key names agreed with the lead (h1.bb_sz_dsm7...)


def series_stage(mid: str):
    mid = ALIAS.get(mid, mid)
    if mid.startswith("s6_"):
        return "s6", ("dpo" if mid == "s6_dpo" else "step%04d" % int(mid.split("step")[1]))
    return ("pop_" if mid in POP else "bb_") + mid, "final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_tag", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    suf = "_" + a.judge_tag if a.judge_tag else ""
    ex = K.Exporter()
    summary = {"judge_tag": a.judge_tag, "models": {}}
    allm = a.models or [m for f in FAM.values() for m in f]
    rows_by = {}
    for mid in allm:
        d = root / ("BB_" + mid)
        if not d.exists():
            d = root / mid   # S8_* / pop_* directories carry no BB_ prefix
        if not d.exists():
            continue
        s, stage = series_stage(mid)
        base_only = mid.startswith("S8_")   # used only as the paired base of the base32 family
        res: dict = {}
        for ds in ("math500", "aime24"):
            f = d / "idx" / ("%s.json" % ds)
            if f.exists():
                j = common.read_json(f)
                res["E1_" + ds] = {"acc": j["acc"], "truncated": j["truncated"], "n_problems": j["n_problems"], "indices": j["indices"],
                                   "totals": {k: v for k, v in j["totals"].items() if not k.startswith(("wait:next", "wait:var", "strip:", "form:", "para:"))}}
        if "E1_math500" in res and not a.judge_tag and not base_only:
            c = res["E1_math500"]
            h1 = {"checkpoints": {stage: {"step": None, "acc": c["acc"], "truncated": c["truncated"], "n_samples": c["n_problems"] * 4, "indices": c["indices"]}}, "trends": {}}
            K.export_h1(ex, h1, s, str(d / "idx/math500.json"), {stage: stage})
            ex.put("ctl.%s.%s.aime_acc" % (s, stage), res["E1_aime24"]["acc"], "pct", str(d / "idx/aime24.json") + "::acc") if "E1_aime24" in res else None
            nw = int(c["totals"].get("wait:n", 0)); tk = int(c["totals"]["tokens"])
            ex.put("h1.%s.%s.nwait" % (s, stage), nw, "int", str(d / "idx/math500.json") + "::totals.wait:n")
            ex.put("h1.%s.%s.ntok" % (s, stage), tk, "int", str(d / "idx/math500.json") + "::totals.tokens")
        cont = read(d / "sdt" / "conts.jsonl.gz")
        if cont and not a.judge_tag and not base_only:
            for r in cont:
                r["mark_any"], r["mark_wait"] = sdt.mark_in_window(r["text"])
            rs = sdt.sdt_from_rows(cont, "mark_any", B=B)
            for q, kk in (("HR", "hr"), ("FA", "fa"), ("dprime", "dprime"), ("criterion", "crit")):
                ex.put_est("h2.%s.%s.%s" % (s, stage, kk), rs[q], "pct" if kk in ("hr", "fa") else "dec2", str(d / "sdt/conts.jsonl.gz") + "::" + q)
        rows = read(d / "ins" / ("judged_shared%s.jsonl.gz" % suf))
        if rows:
            for r in rows:
                r["policy"] = mid
            rows_by[mid] = rows
        summary["models"][mid] = res
    for fam, mids in FAM.items():
        have = [m for m in mids if m in rows_by]
        if not have:
            continue
        base = mids[0] if mids[0] in rows_by else None
        for m in have:
            pids, succ, n_per = succ_table(rows_by[m])
            arms = succ[m]
            s, stage = series_stage(m)
            if m.startswith("S8_"):
                continue
            pr = {arm: dict(zip(("p", "lo", "hi"), st.proportion_ci(v, n_per, B=B, seed=2))) for arm, v in arms.items()}
            entry = {"n_prefixes": len(pids), "n_per": n_per, "arms": pr}
            if all(x in arms for x in ("wait", "next", "none")):
                both = {m: arms}
                if base and m != base:
                    bp, bsucc, _ = succ_table(rows_by[base])
                    if bp == pids:
                        both = {base: bsucc[base], m: arms}
                entry["estimates"] = st.insertion_estimates(both, n_per, B=B, seed=st_seed(m), base=base or m, last=m)
            summary["models"].setdefault(m, {})["E3_shared"] = entry
            e = entry.get("estimates", {})
            for arm, v in pr.items():
                ex.put_est("h4.%s.%s.prev_%s" % (s, stage, K.ARM_KEY.get(arm, arm)), {"est": v["p"], "lo": v["lo"], "hi": v["hi"]}, "pct", "bb_report::E3_shared.%s.%s" % (m, arm))
            for q, name, kind in (("rd_op", "RD_next", "dec1"), ("or_op", "OR_next", "dec2"), ("rd_syn", "RD_syn", "dec1"), ("rd_none", "RD_none", "dec1")):
                if m + ":" + name in e:
                    ex.put_est("h4.%s.%s.%s" % (s, stage, q), e[m + ":" + name], kind, "bb_report::E3_shared.%s.%s" % (m, name))
            for q, name, kind in (("r_amp", "R_amp_RD_next", "dec2"), ("d_amp", "dRD_next_last_minus_base", "dec1")):
                if name in e and base and m != base:
                    ex.put_est("h4.%s.%s.%s" % (s, stage, q), e[name], kind, "bb_report::paired vs family base %s" % base)
    if a.judge_tag:
        ex.entries = [{**e, "key": ".".join(e["key"].split(".")[:3] + [e["key"].split(".")[3] + suf] + e["key"].split(".")[4:])} for e in ex.entries if e["key"].startswith("h4.")]
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / ("bb_summary%s.json" % suf), summary)
    common.write_json(out / ("bb_keys%s.json" % suf), {"pilot": False, "judge_tag": a.judge_tag, "entries": ex.entries})
    print("[bb_report] %d models, %d keys" % (len(summary["models"]), len(ex.entries)))


if __name__ == "__main__":
    main()
