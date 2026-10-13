"""VALUE_CHANGE: the label-free insertion outcome pre-declared in notes/DECISIONS.md D40 (implemented as written, not tuned).

Unit: one E3 continuation (inserted form already stripped); window = its first 3 sentence units (judge `units`).
Reference: the last `LHS = RHS` equation of step k (the last sentence of the prefix; LaTeX-normalised); if step k has none, the last
equation of step k-1; if none, the PREFIX is EXCLUDED (counted per checkpoint, never imputed).
VALUE_CHANGE = 1 iff the window contains an equation whose normalised LHS equals LHS_k and whose RHS is not math-verify-equivalent to v_k.
A restatement with an equivalent value is 0. Implementation notes (frozen with the code): (i) a chain `a = b = c` is one equation with LHS a and
RHS c (the final value); (ii) the "same single variable symbol" clause is subsumed by LHS equality; (iii) an equation whose RHS (or v_k)
cannot be parsed by math-verify is skipped, not counted as a change (conservative; the number of skipped equations is reported).
CONTRADICT (secondary, reported only): a window sentence that mentions LHS_k or v_k and contains an explicit negation word.

    python -m exp.pipeline.valchange --judged J.jsonl.gz [..] --prefixes P.jsonl [..] --series S1 --keys_series s1 \
        --ckpts base:0,100:100,...,final: --last final --stage_map early=100,mid=700,late=1900,final=final --out DIR/valchange.json
    python -m exp.pipeline.valchange --scale_root <.../runs/P1/scale> --out DIR        (S8/S10 panels: shared prefixes, paired vs S8_base)
Needs math_verify (env rl). Cluster bootstrap over prefixes, B = 10,000, seed 20261012.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import common, keys as K, markers as mk, stats as st

B_DEFAULT = 10000
_FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "sqrt", "pi", "exp", "min", "max", "gcd", "lcm", "det"}
_MATHTOK = re.compile(r"^[\w\\{}^_().+\-*/|\[\]<>,;:!'&]+$")
_WORD = re.compile(r"^[A-Za-z]{2,}$")
_NEG = re.compile(r"\b(not|isn't|is not|incorrect|wrong|mistake|error|doesn't|does not|wasn't|was not|didn't|did not)\b", re.I)


def clean(s: str) -> str:
    s = s.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", " ").replace("\\;", " ").replace("\\ ", " ")
    s = s.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ").replace("$", " ")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\boxed\{([^{}]*)\}", r"\1", s)
    return re.sub(r"[:;]", " ; ", s)   # a colon/semicolon closes a math run ("k=3:7*3=21" is two statements)


def _is_math(tok: str) -> bool:
    t = tok.strip(".,;:!?")
    if not t:
        return bool(tok.strip()) and tok.strip() in {",", ";"}
    if _WORD.match(t) and t.lower() not in _FUNCS:
        return False
    return bool(_MATHTOK.match(tok))


def _bal(s: str) -> int:
    return s.count("(") + s.count("[") + s.count("{") - s.count(")") - s.count("]") - s.count("}")


def _stop_after(tok: str, acc: List[str]) -> bool:
    """A token ending in , ; : . with balanced brackets closes a math run."""
    return tok[-1:] in ",;:." and _bal(" ".join(acc + [tok])) <= 0


def math_head(part: str) -> Tuple[str, bool]:
    """(leading math run, whether the whole part is math)."""
    toks, acc = part.split(), []
    for i, t in enumerate(toks):
        if not _is_math(t):
            return " ".join(acc), False
        acc.append(t)
        if _stop_after(t, acc[:-1]):
            return " ".join(acc), i == len(toks) - 1
    return " ".join(acc), True


def math_suffix(part: str) -> str:
    toks, acc = part.split(), []
    for t in reversed(toks):
        if not _is_math(t):
            break
        if acc and _stop_after(t, []):   # previous run ended with , ; : . (balanced)
            break
        acc.insert(0, t)
    return " ".join(acc)


def _tidy(s: str) -> str:
    return re.sub(r"\s+", "", s).rstrip(".,;:")


def equations(sentence: str) -> List[Tuple[str, str]]:
    """[(LHS, RHS)] normalised, chains a = b = c collapsed to (a, c)."""
    t = clean(sentence)
    parts = re.split(r"(?<![<>!=\\])=(?!=)", t)
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(parts) - 1:
        lhs = _tidy(math_suffix(parts[i]))
        j = i + 1
        head, whole = math_head(parts[j])
        while whole and j + 1 < len(parts):   # chain continues only when the whole part is a math run
            j += 1
            head, whole = math_head(parts[j])
        rhs = _tidy(head)
        if lhs and rhs:
            out.append((lhs, rhs))
        i = j
    return out


def reference(prefix: str) -> Optional[Tuple[str, str]]:
    """(LHS_k, v_k) from the last step of the prefix; fall back to step k-1; None -> excluded."""
    sents = [s.text for s in mk.split_sentences(prefix)]
    for s in reversed(sents[-2:]):
        eq = equations(s)
        if eq:
            return eq[-1]
    return None


def _parse(x: str):
    try:
        from math_verify import parse
        r = parse("$" + x + "$")
        return r if r else None
    except Exception:
        return None


def equivalent(a: str, b: str) -> Optional[bool]:
    """True/False, or None when either side cannot be parsed (the equation is then skipped)."""
    if re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b):
        return True
    pa, pb = _parse(a), _parse(b)
    if pa is None or pb is None:
        return None
    try:
        from math_verify import verify
        return bool(verify(pa, pb))
    except Exception:
        return None


def window_flags(ref: Tuple[str, str], units: List[str]) -> Dict[str, int]:
    lhs_k, v_k = ref
    vc, skipped, contra = 0, 0, 0
    for u in units:
        for lhs, rhs in equations(u):
            if lhs == lhs_k:
                eq = equivalent(rhs, v_k)
                if eq is None:
                    skipped += 1
                elif not eq:
                    vc = 1
        nu = re.sub(r"\s+", "", clean(u))
        if (lhs_k in nu or v_k in nu) and _NEG.search(u):
            contra = 1
    return {"value_change": vc, "contradict": contra, "skipped_unparsable": skipped}


def annotate(rows: List[dict], prefixes: Dict[str, dict]) -> Tuple[List[dict], Dict[str, dict]]:
    """Adds value_change/contradict to the judged rows; returns (kept rows, {prefix_id: excluded?})."""
    refs: Dict[str, Optional[Tuple[str, str]]] = {pid: reference(p["prefix"]) for pid, p in prefixes.items()}
    kept = []
    for r in rows:
        ref = refs.get(r["prefix_id"])
        if ref is None:
            continue
        r.update(window_flags(ref, r["units"]))
        kept.append(r)
    return kept, {pid: ref is None for pid, ref in refs.items()}


def succ_table(rows: List[dict], key: str = "value_change"):
    pids = sorted({r["prefix_id"] for r in rows})
    pos = {p: i for i, p in enumerate(pids)}
    succ: Dict[str, Dict[str, np.ndarray]] = {}
    per: Dict[tuple, int] = {}
    for r in rows:
        d = succ.setdefault(r["policy"], {}).setdefault(r["arm"], np.zeros(len(pids)))
        d[pos[r["prefix_id"]]] += bool(r[key])
        per[(r["policy"], r["arm"], r["prefix_id"])] = per.get((r["policy"], r["arm"], r["prefix_id"]), 0) + 1
    return pids, succ, max(per.values())


def analyse(rows: List[dict], order: List[str], base: str, last: str, B: int, seed: int, key: str = "value_change") -> dict:
    pids, succ, n_per = succ_table(rows, key)
    succ = {p: succ[p] for p in order if p in succ}
    res = {"n_prefixes": len(pids), "n_per": n_per, "policies": {}}
    for pol, arms in succ.items():
        res["policies"][pol] = {"arms": {a: dict(zip(("p", "lo", "hi"), st.proportion_ci(v, n_per, B=B, seed=seed))) for a, v in arms.items()}}
    if base in succ and last in succ and all(a in arms for arms in succ.values() for a in ("wait", "none")):
        res["estimates"] = st.insertion_estimates(succ, n_per, B=B, seed=seed, base=base, last=last)
    return res


def export(ex: K.Exporter, res_by_src: dict, series: str, stage_of: Dict[str, List[str]], steps: Dict[str, int], src_file: str, excl: Dict[str, Dict[str, int]]):
    for src, d in res_by_src.items():
        est = d.get("estimates", {})
        for pol, v in d["policies"].items():
            for stage in stage_of.get(pol, []):
                for arm, a in v["arms"].items():
                    ex.put_est("h4.%s.%s.prevval_%s_%s" % (series, stage, K.ARM_KEY.get(arm, arm), src), {"est": a["p"], "lo": a["lo"], "hi": a["hi"]}, "pct", "%s::%s.%s.%s" % (src_file, src, pol, arm))
                e = est.get("%s:RD_next" % pol)
                if e:
                    ex.put_est("h4.%s.%s.rdval_op_%s" % (series, stage, src), e, "dec1", "%s::%s.estimates.%s:RD_next" % (src_file, src, pol))
                for q, nm in (("orval_op", "OR_next"), ("rdval_syn", "RD_syn"), ("rdval_none", "RD_none")):
                    e = est.get("%s:%s" % (pol, nm))
                    if e:
                        ex.put_est("h4.%s.%s.%s_%s" % (series, stage, q, src), e, "dec2" if nm.startswith("OR") else "dec1", "%s::%s.estimates.%s:%s" % (src_file, src, pol, nm))
                ex.put("h4.%s.%s.nexcl_val_%s" % (series, stage, src), excl[src]["n_excluded"], "int", "%s::%s.n_excluded" % (src_file, src))
        for q, nm, kind in (("rampval", "R_amp_RD_next", "dec2"), ("dampval", "dRD_next_last_minus_base", "dec1")):
            if nm in est:
                ex.put_est("h4.%s.trend.%s_%s" % (series, q, src), est[nm], kind, "%s::%s.estimates.%s" % (src_file, src, nm))
        tr = d.get("trend")
        if tr and tr.get("rho") is not None:
            ex.put("h4.%s.trend.rho_rdval_%s" % (series, src), tr["rho"], "dec2", "%s::%s.trend.rho" % (src_file, src))
            if tr.get("p") is not None:
                ex.put("h4.%s.trend.rho_rdval_%s.p" % (series, src), tr["p"], "dec3", "%s::%s.trend.p" % (src_file, src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", nargs="+")
    ap.add_argument("--prefixes", nargs="+")
    ap.add_argument("--series", default="S1")
    ap.add_argument("--keys_series", default="s1")
    ap.add_argument("--ckpts", default=None, help="'base:0,100:100,...,final:' as in report.py")
    ap.add_argument("--last", default="final")
    ap.add_argument("--stage_map", default=None)
    ap.add_argument("--scale_root", default=None, help="S8/S10 panel root (runs/P1/scale): shared prefixes, models vs S8_base")
    ap.add_argument("--judge_files_suffix", default="", help="'' for the Qwen judged files (outcome is label-free; units identical across judges)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=B_DEFAULT)
    ap.add_argument("--dump_windows", default=None, help="write annotated windows (jsonl.gz) for the detector check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = 20261012
    ex = K.Exporter()
    dump: List[dict] = []
    if a.scale_root:
        root = Path(a.scale_root)
        from .scale_report import SID
        rows_all, pre = [], {}
        for m in SID:
            d = root / m / "ins"
            f = d / "judged_shared.jsonl.gz"
            if not f.exists():
                continue
            rs = common.read_jsonl(f)
            for r in rs:
                r["policy"] = m
            rows_all += rs
        for pf in list(root.glob("shared/*prefixes*.jsonl")) + list(root.glob("*/ins/prefixes*.jsonl")):
            if pf.name.startswith("prefixes_own"):
                continue
            for p in common.read_jsonl(pf):
                pre.setdefault(p["prefix_id"], p)
        kept, ex_ids = annotate(rows_all, pre)
        n_excl = sum(ex_ids.values())
        order = [m for m in SID if any(r["policy"] == m for r in kept)]
        summary = {"n_prefixes_excluded": n_excl, "n_prefixes": len(pre), "models": {}}
        base_rows = [r for r in kept if r["policy"] == "S8_base"]
        for m in order:
            if m == "S8_base":
                continue
            d = analyse(base_rows + [r for r in kept if r["policy"] == m], ["S8_base", m], "S8_base", m, a.B, seed)
            summary["models"][m] = d
            sid = SID[m]
            e = d.get("estimates", {})
            for pol in ("S8_base", m):
                for arm, v in d["policies"][pol]["arms"].items():
                    ex.put_est("h4.%s.final.prevval_%s" % (SID[pol], K.ARM_KEY.get(arm, arm)), {"est": v["p"], "lo": v["lo"], "hi": v["hi"]}, "pct", "valchange::scale.%s.%s" % (pol, arm))
            if m + ":RD_next" in e:
                ex.put_est("h4.%s.final.rdval_op" % sid, e[m + ":RD_next"], "dec1", "valchange::scale.%s.RD_next" % m)
            for q, nm, kind in (("rampval", "R_amp_RD_next", "dec2"), ("dampval", "dRD_next_last_minus_base", "dec1")):
                if nm in e:
                    ex.put_est("h4.%s.trend.%s" % (sid, q), e[nm], kind, "valchange::scale.paired vs S8_base")
            ex.put("h4.%s.final.nexcl_val" % sid, n_excl, "int", "valchange::scale.n_excluded")
        common.write_json(out / "valchange_scale.json", summary)
        common.write_json(out / "valchange_scale_keys.json", {"pilot": False, "entries": ex.entries})
        dump = kept
    else:
        prefixes = {p["prefix_id"]: p for f in a.prefixes for p in common.read_jsonl(f)}
        rows = [r for f in a.judged for r in common.read_jsonl(f)]
        kept, ex_ids = annotate(rows, prefixes)
        ck = [c.split(":") for c in a.ckpts.split(",")]
        order = ["%s@%s" % (a.series, c) for c, _ in ck]
        steps = {"%s@%s" % (a.series, c): int(s) for c, s in ck if s != ""}
        sm = dict(x.split("=") for x in a.stage_map.split(",")) if a.stage_map else {}
        stage_of: Dict[str, List[str]] = {}
        for stage, c in sm.items():
            stage_of.setdefault("%s@%s" % (a.series, c), []).append(stage)
        base, last = "%s@base" % a.series, "%s@%s" % (a.series, a.last)
        by_src, excl = {}, {}
        for src in ("late", "base"):
            rs = [r for r in kept if r["src"] == src]
            if not rs:
                continue
            d = analyse(rs, order, base, last, a.B, seed)
            est = d.get("estimates", {})
            pols = [p for p in order if p in steps and "%s:RD_next" % p in est]
            d["trend"] = dict(zip(("rho", "p"), st.spearman([steps[p] for p in pols], [est["%s:RD_next" % p]["est"] for p in pols]))) if len(pols) >= 3 else {}
            n_ex = sum(1 for pid, e in ex_ids.items() if e and prefixes[pid]["src"] == src)
            excl[src] = {"n_excluded": n_ex, "n_prefixes_total": sum(1 for p in prefixes.values() if p["src"] == src)}
            d["n_excluded"] = n_ex
            d["secondary"] = {pol: {"contradict_rate": float(np.mean([r["contradict"] for r in rs if r["policy"] == pol])),
                                    "skipped_unparsable_eq_per_window": float(np.mean([r["skipped_unparsable"] for r in rs if r["policy"] == pol]))} for pol in order if any(r["policy"] == pol for r in rs)}
            by_src[src] = d
        export(ex, by_src, a.keys_series, stage_of, steps, "valchange::" + a.series, excl)
        common.write_json(out / "valchange.json", {"series": a.series, "definition": "notes/DECISIONS.md D40", "B": a.B, "seed": seed, "sources": by_src, "exclusions": excl})
        common.write_json(out / "valchange_keys.json", {"pilot": False, "entries": ex.entries})
        dump = kept
    if a.dump_windows:
        slim = [{k: r.get(k) for k in ("policy", "prefix_id", "src", "arm", "j", "units", "judge_revise", "judge_letter", "value_change", "contradict")} for r in dump]
        common.write_jsonl(a.dump_windows, slim)
    print("[valchange] %d windows kept, %d keys" % (len(dump), len(ex.entries)))


if __name__ == "__main__":
    main()
