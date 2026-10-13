"""S8 scale panel (32B endpoints): E1 indices, E3 on the shared base-derived prefixes (+ the crossed design for DAPO), E2 SDT, and RESULTS_KEYS.

    python -m exp.pipeline.scale_report --root <.../runs/P1/scale> --out <dir> [--judge_tag oss]

Per model directory <root>/<name>/ (written by run_scale.sh; judged by judge_scale.sh):
    idx/math500.json, idx/aime24.json                 E1 (indices CLI)
    ins/judged_shared.jsonl.gz                        E3 on the SHARED 100 base-derived prefixes (arms wait, next, hold_on, none x 4)
    ins/judged_own.jsonl.gz, judged_own_by_cross.jsonl.gz   (DAPO) own-derived prefixes by the model and by the base -> crossed 2 x 2
    sdt/conts.jsonl.gz                                E2 injected-error continuations (MARK computed here)
Series ids (RESULTS_KEYS): S8_base=s8base, S8_dapo=s8dapo, S8_qwq=s8qwq, S8_orz=s8orz, S8_r1distill=s8r1d; one 'final' stage per model.
Amplification vs the base is paired on the shared prefixes: R_amp = RD_next(model) / RD_next(base), D_amp = difference.
With --judge_tag oss the second judge's files (judged_shared_oss.jsonl.gz ...) are used and judge-dependent keys get the _oss suffix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import common, keys as K, sdt, stats as st

SID = {"S8_base": "s8base", "S8_dapo": "s8dapo", "S8_qwq": "s8qwq", "S8_orz": "s8orz", "S8_r1distill": "s8r1d",
       "S10_low": "s10low", "S10_med": "s10med", "S10_high": "s10high"}
B = 2000


def succ_table(rows: List[dict], policy_of=lambda r: r["policy"]):
    pids = sorted({r["prefix_id"] for r in rows})
    pos = {p: i for i, p in enumerate(pids)}
    succ: Dict[str, Dict[str, np.ndarray]] = {}
    per: Dict[tuple, int] = {}
    for r in rows:
        pol = policy_of(r)
        d = succ.setdefault(pol, {}).setdefault(r["arm"], np.zeros(len(pids)))
        d[pos[r["prefix_id"]]] += bool(r["judge_revise"])
        per[(pol, r["arm"], r["prefix_id"])] = per.get((pol, r["arm"], r["prefix_id"]), 0) + 1
    return pids, succ, max(per.values())


def read(path: Path):
    return common.read_jsonl(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_tag", default=None)
    ap.add_argument("--models", nargs="*", default=list(SID))   # models without a directory are skipped
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    suf = "_" + a.judge_tag if a.judge_tag else ""
    summary: dict = {"judge_tag": a.judge_tag, "models": {}}
    ex = K.Exporter()
    shared = {}
    for m in a.models:
        d = root / m
        if not d.exists():
            continue
        res: dict = {}
        for ds in ("math500", "aime24"):
            f = d / "idx" / ("%s.json" % ds)
            if f.exists():
                j = common.read_json(f)
                res["E1_" + ds] = {"acc": j["acc"], "truncated": j["truncated"], "n_problems": j["n_problems"], "indices": j["indices"], "totals": {k: v for k, v in j["totals"].items() if not k.startswith(("wait:next", "wait:var", "strip:", "form:", "para:"))}}
        s = SID[m]
        if "E1_math500" in res and not a.judge_tag:
            c = res["E1_math500"]
            h1 = {"checkpoints": {"final": {"step": None, "acc": c["acc"], "truncated": c["truncated"], "n_samples": c["n_problems"] * 4, "indices": c["indices"]}}, "trends": {}}
            K.export_h1(ex, h1, s, str(root / m / "idx/math500.json"), {"final": "final"})
            ds = res.get("E1_aime24")
            if ds:
                ex.put("ctl.%s.final.aime_acc" % s, ds["acc"], "pct", str(root / m / "idx/aime24.json") + "::acc")
        # ---- E3 on the shared prefixes
        rows = read(d / "ins" / ("judged_shared%s.jsonl.gz" % suf))
        if rows:
            for r in rows:
                r["policy"] = m
            shared[m] = rows
        # ---- E2 SDT
        cont = read(d / "sdt" / "conts.jsonl.gz")
        if cont and not a.judge_tag:
            for r in cont:
                r["mark_any"], r["mark_wait"] = sdt.mark_in_window(r["text"])
            rs = sdt.sdt_from_rows(cont, "mark_any", B=B)
            res["E2_sdt"] = {k: rs[k] for k in ("HR", "FA", "dprime", "criterion")}
            res["E2_sdt"]["n_error"], res["E2_sdt"]["n_clean"] = int(rs["n_error"]["est"]), int(rs["n_clean"]["est"])
            for q, kk in (("HR", "hr"), ("FA", "fa"), ("dprime", "dprime"), ("criterion", "crit")):
                ex.put_est("h2.%s.final.%s" % (s, kk), rs[q], "pct" if kk in ("hr", "fa") else "dec2", str(root / m / "sdt/conts.jsonl.gz") + "::" + q)
        summary["models"][m] = res
    # ---- E3 per model (+ paired amplification vs the base on the same prefixes)
    base = shared.get("S8_base")
    for m, rows in shared.items():
        pids, succ, n_per = succ_table(rows)
        arms = succ[m]
        pr = {arm: dict(zip(("p", "lo", "hi"), st.proportion_ci(v, n_per, B=B, seed=2))) for arm, v in arms.items()}
        if "hold_on" in arms:
            pr["syn"] = dict(zip(("p", "lo", "hi"), st.proportion_ci(arms["hold_on"], n_per, B=B, seed=2)))
        entry = {"n_prefixes": len(pids), "n_per": n_per, "arms": pr}
        if all(x in arms for x in ("wait", "next", "none")):
            both = {m: arms}
            if base is not None and m != "S8_base":
                bp, bsucc, _ = succ_table(base)
                if bp == pids:   # identical prefix set: a paired contrast
                    both = {"S8_base": bsucc["S8_base"], m: arms}
            entry["estimates"] = st.insertion_estimates(both, n_per, B=B, seed=st_seed(m), base="S8_base", last=m)
        summary["models"].setdefault(m, {})["E3_shared"] = entry
        s = SID[m]
        e = entry.get("estimates", {})
        for arm, v in pr.items():
            ex.put_est("h4.%s.final.prev_%s" % (s, K.ARM_KEY.get(arm, arm)), {"est": v["p"], "lo": v["lo"], "hi": v["hi"]}, "pct", "scale_report::E3_shared.%s.%s" % (m, arm))
        for q, name, kind in (("rd_op", "RD_next", "dec1"), ("or_op", "OR_next", "dec2"), ("rr_op", "RR_next", "dec2"), ("rd_syn", "RD_syn", "dec1"), ("rd_none", "RD_none", "dec1")):
            if m + ":" + name in e:
                ex.put_est("h4.%s.final.%s" % (s, q), e[m + ":" + name], kind, "scale_report::E3_shared.%s.%s" % (m, name))
        for q, name, kind in (("r_amp", "R_amp_RD_next", "dec2"), ("d_amp", "dRD_next_last_minus_base", "dec1")):
            if name in e:
                ex.put_est("h4.%s.trend.%s" % (s, q), e[name], kind, "scale_report::E3_shared paired vs S8_base (%s)" % name)
    # ---- crossed design for every model with own-derived prefixes: own prefixes by the model and by the base vs the shared (base-derived) prefixes by both
    for m in a.models:
        d = root / m / "ins"
        own, own_x = read(d / ("judged_own%s.jsonl.gz" % suf)), read(d / ("judged_own_by_cross%s.jsonl.gz" % suf))
        if own and own_x and base is not None and m in shared:
            for r in own:
                r["policy"] = m
            for r in own_x:
                r["policy"] = "S8_base"
            pids_o, s_own, n1 = succ_table(own + own_x)
            _, s_sh, n2 = succ_table(base + shared[m])
            summary["models"].setdefault(m, {})["E3_own"] = {"n_prefixes": len(pids_o), "arms_model": {ar: dict(zip(("p", "lo", "hi"), st.proportion_ci(v, n1, B=B, seed=3))) for ar, v in s_own[m].items()},
                                                               "arms_base": {ar: dict(zip(("p", "lo", "hi"), st.proportion_ci(v, n1, B=B, seed=3))) for ar, v in s_own["S8_base"].items()}}
            if n1 == n2:
                cr = st.crossed_interaction(s_own, s_sh, n1, B=B, seed=2026, base="S8_base", last=m)
                summary["models"][m]["E3_crossed"] = {"own_derived_vs_shared": cr}
                for k, v in cr.items():
                    ex.put_est("h4.%s.trend.crossed_%s" % (SID[m], k.lower()), v, "dec1", "scale_report::crossed.%s" % k)
            rdm = st.insertion_estimates({"S8_base": s_own["S8_base"], m: s_own[m]}, n1, B=B, seed=4, base="S8_base", last=m)
            for q, nm, kind in (("rd_op_own", m + ":RD_next", "dec1"), ("r_amp_own", "R_amp_RD_next", "dec2"), ("d_amp_own", "dRD_next_last_minus_base", "dec1")):
                if nm in rdm:
                    ex.put_est("h4.%s.%s.%s" % (SID[m], "final" if q.startswith("rd") else "trend", q), rdm[nm], kind, "scale_report::own-derived prefixes (model vs base, paired)")
    if a.judge_tag:   # keep only judge-dependent keys, suffixed
        ex.entries = [{**e, "key": ".".join(e["key"].split(".")[:3] + [e["key"].split(".")[3] + suf] + e["key"].split(".")[4:])} for e in ex.entries if e["key"].startswith("h4.")]
    out.mkdir(parents=True, exist_ok=True)
    common.write_json(out / ("scale_summary%s.json" % suf), summary)
    common.write_json(out / ("scale_keys%s.json" % suf), {"pilot": False, "judge_tag": a.judge_tag, "entries": ex.entries})
    print("[scale_report] %d models, %d keys" % (len(summary["models"]), len(ex.entries)))


def st_seed(m: str) -> int:
    return 20261012 + sum(ord(c) for c in m)


if __name__ == "__main__":
    main()
