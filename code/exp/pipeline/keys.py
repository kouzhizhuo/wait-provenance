"""Export result JSON under the json2tex key scheme of notes/RESULTS_KEYS.md.

    key = <claim>.<series>.<stage>.<quantity>[.<stat>]     stat in {est (omitted), lo, hi, n, p, min, max, std}

Every entry: {"key", "value" (PRE-ROUNDED to paper precision, STYLE_GUIDE §5.2), "src" ("file::field")}. The P0 tree is exported
under the same keys as the main runs (the writer wraps them in \\pilot{}); swapping to main-run numbers is a path change.
Stage map for P0 (S1): base = base, early = step_100, mid = step_500 (the main-run mid is 700), late = step_1900.
NaN values are not exported (a missing key is a loud error in the paper build, not a silent blank).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

P0_STAGE_MAP = {"base": "base", "early": "100", "mid": "500", "late": "1900"}

# quantity -> (index name in h1_indices.json, kind)
H1_MAP = {
    "freq": ("FREQ_wait_1k", "dec1"), "freq100": ("FREQ_wait_100s", "dec1"), "fdisc": ("FREQ_disc_1k", "dec1"),
    "si": ("SI", "pct"), "siconn": ("SI_conn", "pct"), "pi": ("PI", "pct"), "orth": ("ORTH", "pct"),
    "orthmax": ("ORTH_max", "pct"), "commadrop": ("COMMA_DROP", "pct"), "chain": ("CHAIN_RUNS_1k", "dec1"),
    "chainshare": ("CHAIN", "pct"), "hpos": ("H_pos_rare", "dec2"), "hpara": ("H_para_rare", "dec2"),
    "neff": ("N_eff", "dec2"), "sharewait": ("SHARE_wait", "pct"), "obl": ("OBL", "pct"), "rel": ("REL", "pct"),
    "hnext": ("H_next_rare", "dec2"), "eros": ("EROS", "dec1"), "sentlen": ("SENTLEN_init", "dec1"), "ctx": ("CTX", "dec2"),
    "lex": ("LEX_share", "pct"), "revrate": ("R_per_100s", "dec2"), "toks": ("tokens_per_resp", "int"),
}
TREND_MAP = {"FREQ_wait_1k": "freq", "SI": "si", "SI_conn": "siconn", "ORTH": "orth", "ORTH_max": "orthmax", "COMMA_DROP": "commadrop",
             "CHAIN_RUNS_1k": "chain", "H_pos_rare": "hpos", "H_para_rare": "hpara", "OBL": "obl", "REL": "rel",
             "H_next_rare": "hnext", "EROS": "eros", "SENTLEN_init": "sentlen"}
N_STAT = {"si": "n_wait", "siconn": "n_wait", "orth": "n_wait", "orthmax": "n_wait", "commadrop": "n_wait", "hpos": "n_wait",
          "hnext": "n_wait", "obl": "n_R", "rel": "n_M", "hpara": "n_R"}
ARM_KEY = {"wait": "wait", "next": "next", "okay": "okay", "hold_on": "holdon", "hmm": "hmm", "none": "none", "so": "so", "syn": "syn"}


def rnd(v: Any, kind: str) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    x = float(v)
    if kind == "pct":
        return round(100.0 * x, 1)
    if kind == "dec1":
        return round(x, 1)
    if kind == "dec2":
        return round(x, 2)
    if kind == "dec3":
        return round(x, 3)
    if kind == "int":
        return int(round(x))
    raise ValueError(kind)


class Exporter:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def put(self, key: str, value: Any, kind: str, src: str):
        v = rnd(value, kind)
        if v is not None:
            self.entries.append({"key": key, "value": v, "src": src})

    def put_est(self, base_key: str, est: Dict[str, float], kind: str, src: str):
        for stat, suffix in (("est", ""), ("lo", ".lo"), ("hi", ".hi")):
            if stat in est:
                self.put(base_key + suffix, est[stat], kind, "%s.%s" % (src, stat))


def stage_keys(step: Optional[int], ckpt: str, stage_map: Dict[str, str]) -> List[str]:
    """Every stage label a checkpoint carries: stepNNNN (or base) and the stage-table alias if it is one."""
    keys = ["base" if ckpt == "base" else "step%04d" % int(ckpt)] if (ckpt == "base" or str(ckpt).isdigit()) else [ckpt]
    keys += [st for st, c in stage_map.items() if c == ckpt and st not in keys]
    return keys


def export_h1(ex: Exporter, h1: dict, series: str, src_file: str, stage_map: Dict[str, str] = P0_STAGE_MAP):
    H1_MAP_ = dict(H1_MAP)
    trend_map = dict(TREND_MAP)
    if all("OBL_j" in c["indices"] for c in h1["checkpoints"].values()):   # D30: judge-based revision indices are primary
        H1_MAP_.update({"obl": ("OBL_j", "pct"), "rel": ("REL_j", "pct"), "relstrict": ("REL_strict_j", "pct"), "revrate": ("R_per_100s_j", "dec2"),
                        "obl_rule": ("OBL", "pct"), "rel_rule": ("REL", "pct"), "revrate_rule": ("R_per_100s", "dec2")})
        trend_map.update({"OBL_j": "obl", "REL_j": "rel", "OBL": "obl_rule", "REL": "rel_rule"})
    for ckpt, c in h1["checkpoints"].items():
        for stage in stage_keys(c["step"], ckpt, stage_map):
            for q, (idx, kind) in H1_MAP_.items():
                if idx in c["indices"]:
                    ex.put_est("h1.%s.%s.%s" % (series, stage, q), c["indices"][idx], kind, "%s::checkpoints.%s.indices.%s" % (src_file, ckpt, idx))
                    if q in N_STAT and N_STAT[q] in c["indices"]:
                        ex.put("h1.%s.%s.%s.n" % (series, stage, q), c["indices"][N_STAT[q]]["est"], "int", "%s::checkpoints.%s.indices.%s" % (src_file, ckpt, N_STAT[q]))
            ex.put("n.%s.%s.resp" % (series, stage), c["n_samples"], "int", "%s::checkpoints.%s.n_samples" % (src_file, ckpt))
            ex.put("ctl.%s.%s.acc" % (series, stage), c["acc"], "pct", "%s::checkpoints.%s.acc" % (src_file, ckpt))
            ex.put("ctl.%s.%s.trunc" % (series, stage), c["truncated"], "pct", "%s::checkpoints.%s.truncated" % (src_file, ckpt))
            for q in ("freq", "si", "orth", "obl", "rel", "hnext", "eros", "toks"):
                idx, kind = H1_MAP[q]
                fo = c.get("indices_finished_only", {}).get(idx)
                if fo:
                    ex.put_est("ctl.%s.%s.finished_%s" % (series, stage, q), fo, kind, "%s::checkpoints.%s.indices_finished_only.%s" % (src_file, ckpt, idx))
    for ckpt, m in h1.get("matched_length", {}).get("checkpoints", {}).items():
        for stage in stage_keys(None, ckpt, stage_map):
            for idx, q, kind in (("FREQ_wait_1k", "freq", "dec1"), ("REL", "rel", "pct"), ("SI", "si", "pct"), ("OBL", "obl", "pct"), ("H_next", "hnext", "dec2")):
                ex.put_est("ctl.%s.%s.matched_%s" % (series, stage, q), m[idx], kind, "%s::matched_length.checkpoints.%s.%s" % (src_file, ckpt, idx))
    for idx, q in trend_map.items():
        t = h1.get("trends", {}).get(idx)
        if t:
            ex.put("h1.%s.trend.rho_%s" % (series, q), t["rho"], "dec2", "%s::trends.%s.rho" % (src_file, idx))
            if t.get("p") is not None:
                ex.put("h1.%s.trend.rho_%s.p" % (series, q), t["p"], "dec3", "%s::trends.%s.p (exact permutation)" % (src_file, idx))
    ex.put("n.%s.ckpts" % series, len(h1["checkpoints"]), "int", "%s::checkpoints" % src_file)


def export_e3(ex: Exporter, ins: dict, series: str, src_file: str, stage_map: Dict[str, str] = P0_STAGE_MAP, outcome: str = "judge_revise"):
    o = ins["outcomes"][outcome]
    n_cont = 0
    for source in ("late", "base"):
        if source not in o:
            continue
        sfx = "_" + source
        d = o[source]
        n_cont += sum(d["n_prefixes"] * d["n_per"] * len(v["arms"]) for v in d["policies"].values())
        ex.put("n.%s.prefixes%s" % (series, sfx), d["n_prefixes"], "int", "%s::outcomes.%s.%s.n_prefixes" % (src_file, outcome, source))
        est = d.get("estimates", {})
        for pol, v in d["policies"].items():
            ckpt = pol.split("@")[1]
            for stage in stage_keys(None, ckpt, stage_map):
                for arm, a in v["arms"].items():
                    ex.put_est("h4.%s.%s.prev_%s%s" % (series, stage, ARM_KEY.get(arm, arm), sfx), {"est": a["p"], "lo": a["lo"], "hi": a["hi"]}, "pct",
                               "%s::outcomes.%s.%s.policies.%s.arms.%s" % (src_file, outcome, source, pol, arm))
                # D21: the primary neutral control is Next, (rd_op = Wait - Next); Okay, stays as rd_op_okay. Fall back to Okay if no Next arm.
                ref = "next" if "%s:RD_next" % pol in est else "op"   # also robust to checkpoints without an Okay, arm
                for q, name, kind in (("rd_op", "RD_" + ref, "dec1"), ("rr_op", "RR_" + ref, "dec2"), ("or_op", "OR_" + ref, "dec2"),
                                      ("rd_op_okay", "RD_op", "dec1"), ("or_op_okay", "OR_op", "dec2"),
                                      ("rd_syn", "RD_syn", "dec1"), ("or_syn", "OR_syn", "dec2"), ("rd_none", "RD_none", "dec1")):
                    e = est.get("%s:%s" % (pol, name))
                    if e:
                        ex.put_est("h4.%s.%s.%s%s" % (series, stage, q, sfx), e, kind, "%s::outcomes.%s.%s.estimates.%s:%s" % (src_file, outcome, source, pol, name))
        fr = ins["outcomes"].get("false_rev", {}).get(source)
        if fr:  # FALSE_REV: judge B, STEP = last prefix step, value changed (restarts carry no step, so they are not counted)
            for pol, v in fr["policies"].items():
                for stage in stage_keys(None, pol.split("@")[1], stage_map):
                    for arm in ("wait", "next", "hold_on", "none"):
                        if arm in v["arms"]:
                            ex.put_est("h4.%s.%s.falserev_%s%s" % (series, stage, ARM_KEY[arm], sfx), {"est": v["arms"][arm]["p"], "lo": v["arms"][arm]["lo"], "hi": v["arms"][arm]["hi"]},
                                       "pct", "%s::outcomes.false_rev.%s.policies.%s.arms.%s" % (src_file, source, pol, arm))
        ref = "_next" if "R_amp_RD_next" in est else ""
        for q, name, kind in (("r_amp", "R_amp_RD" + ref, "dec2"), ("r_amp_or", "R_amp_OR" + ref, "dec2"),
                              ("d_amp", ("dRD_next_last_minus_base" if ref else "dRD_op_last_minus_base"), "dec1")):
            if name in est:
                ex.put_est("h4.%s.trend.%s%s" % (series, q, sfx), est[name], kind, "%s::outcomes.%s.%s.estimates.%s" % (src_file, outcome, source, name))
    if "crossed" in o:
        for k, v in o["crossed"].items():
            ex.put_est("h4.%s.trend.crossed_%s" % (series, k.lower()), v, "dec1", "%s::outcomes.%s.crossed.%s" % (src_file, outcome, k))
    tr = ins.get("trend_RD_op") or {}
    if tr.get("rho") is not None:
        ex.put("h4.%s.trend.rho_rd_op" % series, tr["rho"], "dec2", "%s::trend_RD_op.rho" % src_file)
        if tr.get("p") is not None:
            ex.put("h4.%s.trend.rho_rd_op.p" % series, tr["p"], "dec3", "%s::trend_RD_op.p (exact permutation)" % src_file)
    ex.put("n.%s.interventions" % series, n_cont, "int", "%s::outcomes.%s (prefixes x continuations x arms)" % (src_file, outcome))


def export_fig1(ex: Exporter, exc: dict, src_file: str):
    for k in ("n", "rl_revised", "base_revised", "rl_none_revised"):
        if exc.get("fig1", {}).get(k) is not None:
            ex.put("fig1.%s" % k, exc["fig1"][k], "int", "%s::fig1.%s" % (src_file, k))


def export_corrections(ex: Exporter, ins: Optional[dict], h1: dict, draws, series: str, res_dir, ser_tag: str, stage_map: Dict[str, str] = P0_STAGE_MAP,
                       src_calib: str = "exp/gold/human_audit_results.json", draws_e3=None, src_calib_e3: str = "exp/gold/human_audit_v2_results.json"):
    """`*_corr` keys: E3 risk differences and judge-based REL/OBL corrected for judge misclassification (calib.py; nondifferential
    misclassification, sens/spec from the human audit). Intervals = audit uncertainty of J (+ observed sampling error for RD)."""
    from pathlib import Path

    from . import calib

    de3 = draws if draws_e3 is None else draws_e3   # audit v2 validates the E3 OUTCOME judge; audit v1 validates the sentence-level (REL/OBL) judge
    if ins:
        o = ins["outcomes"]["judge_revise"]
        for source in ("late", "base"):
            if source not in o:
                continue
            sfx, est = "_" + source, o[source].get("estimates", {})
            for pol in o[source]["policies"]:
                ck = pol.split("@")[1]
                name = "RD_next" if pol + ":RD_next" in est else "RD_op"
                if pol + ":" + name not in est:
                    continue
                e = est[pol + ":" + name]
                c = calib.corrected_rd(e["est"], e["lo"], e["hi"], de3)
                for stage in stage_keys(None, ck, stage_map):
                    ex.put_est("h4.%s.%s.rd_op_corr%s" % (series, stage, sfx), c, "dec1", "calib.corrected_rd(%s::%s, %s; sens/spec from %s)" % (src_calib, name, pol, src_calib_e3))
                    ex.put("h4.%s.%s.rd_op_corr%s.unid" % (series, stage, sfx), 100 * c["share_unidentified"], "dec1", "calib: share of J draws <= %.2f" % calib.J_MIN)
            for q, name in (("d_amp", "dRD_next_last_minus_base" if "dRD_next_last_minus_base" in est else "dRD_op_last_minus_base"),
                            ("r_amp", "R_amp_RD_next" if "R_amp_RD_next" in est else "R_amp_RD")):
                if name not in est:
                    continue
                e = est[name]
                if q == "d_amp":   # a difference of risk differences is attenuated by J like RD itself
                    ex.put_est("h4.%s.trend.d_amp_corr%s" % (series, sfx), calib.corrected_rd(e["est"], e["lo"], e["hi"], de3), "dec1", "calib.corrected_rd(%s)" % name)
                else:              # a ratio of risk differences is invariant to nondifferential misclassification
                    ex.put_est("h4.%s.trend.r_amp_corr%s" % (series, sfx), e, "dec2", "%s (unchanged by nondifferential misclassification)" % name)
    steps, rels = [], []
    for ck, c in h1["checkpoints"].items():
        f = Path(res_dir) / "revlab" / ("%s@%s.revlab.json" % (ser_tag, ck))
        if not f.exists():
            continue
        import json

        rl = json.load(open(f))
        rel = calib.corrected_rel(rl["indices"]["REL_j"]["est"], draws)
        obl = calib.corrected_obl(rl["stratum_R_rate"]["marker_adjacent"], rl["stratum_R_rate"]["other"], rl["N_stratum1"], rl["N_stratum0"], draws)
        for stage in stage_keys(c["step"], ck, stage_map):
            ex.put_est("h1.%s.%s.rel_corr" % (series, stage), rel, "pct", "calib.corrected_rel(%s)" % f)
            ex.put_est("h1.%s.%s.obl_corr" % (series, stage), obl, "pct", "calib.corrected_obl(%s)" % f)
        if c["step"] is not None and rel["est"] == rel["est"]:
            steps.append(c["step"]); rels.append(rel["est"])
    if len(steps) >= 3:
        from .stats import spearman

        rho, p = spearman(steps, rels)
        ex.put("h1.%s.trend.rho_rel_corr" % series, rho, "dec2", "spearman of corrected REL medians (rank trend is invariant to nondifferential misclassification)")
