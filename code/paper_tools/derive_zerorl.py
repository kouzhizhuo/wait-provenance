#!/usr/bin/env python3
"""Zero-RL summary keys over every base -> final pair, with the D47 verdict per run (notes/DECISIONS.md, D47).

    python3 tools/derive_zerorl.py      # reads numbers/results.tex, writes numbers/keys/zerorl_keys.json

A run is a COUNTEREXAMPLE iff its base is marker-free (< 0.05 per 1k), its final rate is >= 0.05 per 1k and the exact
95% lower bound of the final rate exceeds the exact 95% upper bound of the base rate. A run whose base is not measured
counts as a counterexample if its final rate is >= 0.05 per 1k. Runs from a base >= 0.05 are not zero-RL from a
marker-free base and are excluded here (they belong to rl_on_marker_model).
"""
import json
import re
from pathlib import Path

from scipy.stats import chi2

K = {}
for line in Path("numbers/results.tex").read_text().splitlines():
    m = re.match(r"\\resdef\{([^}]*)\}\{(.*)\}(\s+%.*)?$", line)
    if m:
        K[m.group(1)] = m.group(2)


def num(k):
    return float(K[k].replace("{,}", "").replace(r"\ensuremath{-}", "-"))


def rate(prefix):
    n, tok = num(prefix + ".nwait"), num(prefix + ".ntok")
    lo = 0.0 if n == 0 else chi2.ppf(0.025, 2 * n) / 2
    hi = chi2.ppf(0.975, 2 * n + 2) / 2
    return n * 1000 / tok, lo * 1000 / tok, hi * 1000 / tok


# (run id, base label, base wait prefix | None, final wait prefix, base acc key | None, final acc key)
PAIRS = [
    ("s8dapo", "Qwen2.5-32B", "h1.s8base.final", "h1.s8dapo.final", "ctl.s8base.final.acc", "ctl.s8dapo.final.acc"),
    ("s8orz", "Qwen2.5-32B", "h1.s8base.final", "h1.s8orz.final", "ctl.s8base.final.acc", "ctl.s8orz.final.acc"),
    ("s7r1", "Qwen2.5-Math-1.5B", "h1.s7r1.base", "h1.s7r1.final", "ctl.s7r1.base.acc", "ctl.s7r1.final.acc"),
    ("s7r2", "Qwen2.5-Math-1.5B", "h1.s7r2.base", "h1.s7r2.final", "ctl.s7r2.base.acc", "ctl.s7r2.final.acc"),
    ("s7r7b", "Qwen2.5-Math-7B", "h1.s7r7b.base", "h1.s7r7b.final", "ctl.s7r7b.base.acc", "ctl.s7r7b.final.acc"),
    ("sz_q05", "Qwen2.5-0.5B", "h1.bb_sz_q05_base.final", "h1.bb_sz_q05.final", "ctl.bb_sz_q05_base.final.acc", "ctl.bb_sz_q05.final.acc"),
    ("sz_q15", "Qwen2.5-1.5B", "h1.bb_sz_q15_base.final", "h1.bb_sz_q15.final", "ctl.bb_sz_q15_base.final.acc", "ctl.bb_sz_q15.final.acc"),
    ("sz_q7", "Qwen2.5-7B", "h1.bb_sz_q7_base.final", "h1.bb_sz_q7.final", "ctl.bb_sz_q7_base.final.acc", "ctl.bb_sz_q7.final.acc"),
    # ORZ-7B / Oat-Zero-7B: the primary decode uses each final's own template, so its base is run with that template
    # (bb_orz7b_base, bb_oat7_base); the earlier T=1.0 decode used the SimpleRL-Zoo base runs (qwen-boxed). Base per decode.
    ("orz7b", "Qwen2.5-7B", {"": "h1.bb_orz7b_base.final", "_t10": "h1.bb_sz_q7_base.final"}, "h1.bb_orz7b.final",
     {"": "ctl.bb_orz7b_base.final.acc", "_t10": "ctl.bb_sz_q7_base.final.acc"}, "ctl.bb_orz7b.final.acc"),
    ("sz_qm7", "Qwen2.5-Math-7B", "h1.bb_sz_qm7_base.final", "h1.bb_sz_qm7.final", "ctl.bb_sz_qm7_base.final.acc", "ctl.bb_sz_qm7.final.acc"),
    ("oat7", "Qwen2.5-Math-7B", {"": "h1.bb_oat7_base.final", "_t10": "h1.bb_sz_qm7_base.final"}, "h1.bb_oat7.final",
     {"": "ctl.bb_oat7_base.final.acc", "_t10": "ctl.bb_sz_qm7_base.final.acc"}, "ctl.bb_oat7.final.acc"),
    ("sz_q14", "Qwen2.5-14B", "h1.bb_sz_q14_base.final", "h1.bb_sz_q14.final", "ctl.bb_sz_q14_base.final.acc", "ctl.bb_sz_q14.final.acc"),
    # Pending (rows appear once their keys exist): SimpleRL-Zoo on Mistral-7B, DeepSeek-Math-7B, Mistral-Small-24B,
    # Qwen2.5-32B, and Llama-3.1-8B (base not measured: gated).
    ("sz_m7", "Mistral-7B-v0.1", "h1.bb_sz_m7_base.final", "h1.bb_sz_m7.final", "ctl.bb_sz_m7_base.final.acc", "ctl.bb_sz_m7.final.acc"),
    ("sz_dsm7", "DeepSeek-Math-7B", "h1.bb_sz_dsm7_base.final", "h1.bb_sz_dsm7.final", "ctl.bb_sz_dsm7_base.final.acc", "ctl.bb_sz_dsm7.final.acc"),
    ("sz_m24", "Mistral-Small-24B", "h1.bb_sz_m24_base.final", "h1.bb_sz_m24.final", "ctl.bb_sz_m24_base.final.acc", "ctl.bb_sz_m24.final.acc"),
    # base must share the SimpleRL-Zoo eval setting (D47); the S8 base was evaluated differently and is not paired here
    ("sz_q32", "Qwen2.5-32B", "h1.bb_sz_q32_base.final", "h1.bb_sz_q32.final", "ctl.bb_sz_q32_base.final.acc", "ctl.bb_sz_q32.final.acc"),
    ("sz_l8", "Llama-3.1-8B", None, "h1.bb_sz_l8.final", None, "ctl.bb_sz_l8.final.acc"),
    # D47b breadth families (pre-registered before any result)
    ("gr7", "Qwen2.5-7B", "h1.bb_gr7_base.final", "h1.bb_gr7.final", "ctl.bb_gr7_base.final.acc", "ctl.bb_gr7.final.acc"),
    ("gr14", "Qwen2.5-14B", "h1.bb_gr14_base.final", "h1.bb_gr14.final", "ctl.bb_gr14_base.final.acc", "ctl.bb_gr14.final.acc"),
    ("gr3_4", "Qwen3-4B", "h1.bb_gr3_4_base.final", "h1.bb_gr3_4.final", "ctl.bb_gr3_4_base.final.acc", "ctl.bb_gr3_4.final.acc"),
    ("gr3_14", "Qwen3-14B", "h1.bb_gr3_14_base.final", "h1.bb_gr3_14.final", "ctl.bb_gr3_14_base.final.acc", "ctl.bb_gr3_14.final.acc"),
    ("os15", "Qwen2.5-Math-1.5B", "h1.bb_os15_base.final", "h1.bb_os15.final", "ctl.bb_os15_base.final.acc", "ctl.bb_os15.final.acc"),
    ("oat15", "Qwen2.5-Math-1.5B", "h1.bb_oat15_base.final", "h1.bb_oat15.final", "ctl.bb_oat15_base.final.acc", "ctl.bb_oat15.final.acc"),
    ("orz32", "Qwen2.5-32B", "h1.bb_orz32_base.final", "h1.bb_orz32.final", "ctl.bb_orz32_base.final.acc", "ctl.bb_orz32.final.acc"),
    # D49: our REF GRPO from Qwen3-4B-Base, 2 seeds (own runs; valid iff acc +>= 10 points and truncation < 20%)
    ("q3r1", "Qwen3-4B", "h1.bb_q3r1_base.base", "h1.bb_q3r1.step0400", "ctl.bb_q3r1_base.base.acc", "ctl.bb_q3r1.step0400.acc"),
    # q3r2 (D49 seed 2) stopped for compute (D49a)
]
# D47b: a released model counts once; the own-template ORZ-32B pair supersedes the S8-harness pair when it exists.
SUPERSEDES = {"orz32": "s8orz"}

out, runs, missing = [], [], []
E = lambda k, v, src: out.append({"key": k, "value": v, "src": src})
SRC = "tools/derive_zerorl.py over numbers/results.tex (D47)"
TRUNC = {"s8dapo": "ctl.s8dapo.final.trunc", "s8orz": "ctl.s8orz.final.trunc", "s7r1": "ctl.s7r1.final.trunc",
         "s7r2": "ctl.s7r2.final.trunc", "s7r7b": "ctl.s7r7b.final.trunc"}


def leaf(key, sfx):
    return key + sfx


def pick(x, sfx):
    """A base prefix / acc key may differ per decode (dict keyed by suffix)."""
    return x.get(sfx) if isinstance(x, dict) else x


def has(rid, bp, fp, ba, fa, sfx):
    bp, ba = pick(bp, sfx), pick(ba, sfx)
    if bp is None and ba is None and isinstance(PAIRS_BY_ID[rid][2], dict):
        return False
    tk = TRUNC.get(rid, fa.rsplit(".", 1)[0] + ".trunc")
    need = [fp + ".nwait" + sfx, fp + ".ntok" + sfx, fa + sfx, tk + sfx]
    if bp:
        need += [bp + ".nwait" + sfx, bp + ".ntok" + sfx]
    return all(k in K for k in need)


def rate_s(prefix, sfx):
    n, tok = num(prefix + ".nwait" + sfx), num(prefix + ".ntok" + sfx)
    lo = 0.0 if n == 0 else chi2.ppf(0.025, 2 * n) / 2
    hi = chi2.ppf(0.975, 2 * n + 2) / 2
    return n * 1000 / tok, lo * 1000 / tok, hi * 1000 / tok


# Decode per pair (D47 / D47a): the primary decode (unsuffixed keys; for SimpleRL-Zoo rows = the authors' own evaluation) when base AND final have it; otherwise
# our earlier decode (suffix _t10) for BOTH. Never mixed within a pair. Normalised keys h1.zr.<id>.* feed the tables.
DECODES = [("", "primary"), ("_t10", "earlier decode, T = 1.0")]
PAIRS_BY_ID = {p[0]: p for p in PAIRS}
for rid, base, bp0, fp, ba0, fa in PAIRS:
    choice = next(((sfx, lab) for sfx, lab in DECODES if has(rid, bp0, fp, ba0, fa, sfx)), None)
    if choice is None:
        missing.append(rid)
        continue
    sfx, lab = choice
    bp, ba = pick(bp0, sfx), pick(ba0, sfx)
    f = rate_s(fp, sfx)
    b = rate_s(bp, sfx) if bp else None
    if b and b[0] >= 0.05:
        verdict = "rl_on_marker"
    elif b is None:
        verdict = "counterexample" if f[0] >= 0.05 else "confirm_base_not_measured"
    else:
        verdict = "counterexample" if (f[0] >= 0.05 and f[1] > b[2]) else "confirm"
    acc_f = num(fa + sfx)
    acc_b = num(ba + sfx) if (ba and ba + sfx in K) else None
    tk = TRUNC.get(rid, fa.rsplit(".", 1)[0] + ".trunc") + sfx
    trunc = num(tk)
    dacc = acc_f - acc_b if acc_b is not None else None
    runs.append(dict(id=rid, base=base, verdict=verdict, final=f, base_rate=b, dacc=dacc, acc=acc_f, trunc=trunc, decode=lab))
    src = f"{SRC}; decode: {lab}"
    for stage, pre in (("final", fp), ("base", bp)):
        if pre:
            E(f"h1.zr.{rid}.{stage}.nwait", int(num(pre + ".nwait" + sfx)), src)
            E(f"h1.zr.{rid}.{stage}.ntok", int(num(pre + ".ntok" + sfx)), src)
    E(f"ctl.zr.{rid}.final.acc", acc_f, src)
    if acc_b is not None:
        E(f"ctl.zr.{rid}.base.acc", acc_b, src)
    E(f"ctl.zr.{rid}.final.trunc", trunc, src)
    E(f"h1.zr.{rid}.decode", lab, src)

# D47a reproduction gate: third-party rows need accuracy >= half the published MATH-500 number AND truncation < 20%;
# our own runs (s7*) need truncation < 20% only. Active once notes/published_acc.json exists ({run id: acc or null}).
import os
PUB_FILE = Path(os.environ.get("PUBLISHED_ACC", "../notes/published_acc.json"))  # env override: dry runs only
GATE = PUB_FILE.exists()
PUB = json.load(open(PUB_FILE)) if GATE else {}
OWN = {"s7r1", "s7r2", "s7r7b"}
D49 = {"q3r1"}
# Releases without licence text are used for robustness only, never for a headline number (as the S3a series;
# appendix "Checkpoint provenance"; notes/LICENCES.md Flag 1).
NO_LICENCE = {"sz_q14"}
for r in runs:
    if r["id"] in NO_LICENCE:
        r["gate"] = "no licence (secondary only)"
        continue
    if not GATE:
        r["gate"] = "not applied"
    elif r["id"] in D49:
        ok = r["trunc"] < 20 and r["dacc"] is not None and r["dacc"] >= 10
        r["gate"] = "pass" if ok else "not reproduced (D49 invalid)"
    elif r["id"] in OWN:
        r["gate"] = "pass" if r["trunc"] < 20 else "not reproduced"
    elif PUB.get(r["id"]) is None:
        r["gate"] = "no published reference" if r["id"] in PUB else "missing published number"
    else:
        r["gate"] = "pass" if (r["acc"] >= PUB[r["id"]] / 2 and r["trunc"] < 20) else "not reproduced"
present = {r["id"] for r in runs}
for r in runs:
    for new_id, old_id in SUPERSEDES.items():
        if r["id"] == old_id and new_id in present:
            r["gate"] = f"superseded by {new_id} (robustness)"
counted = [r for r in runs if r["gate"] in ("pass", "not applied")]
for r in runs:
    if r not in counted:
        print(f"  D47a: {r['id']} {r['gate']}")
zr = [r for r in counted if r["verdict"] != "rl_on_marker"]
for r in runs:
    if r not in counted and r["verdict"] == "counterexample":
        print(f"  !! D47 COUNTEREXAMPLE in a row that fails D47a ({r['id']}): report it to the coordinator and in the appendix")
ce = [r for r in zr if r["verdict"] == "counterexample"]
fam = sorted({r["base"].split("-")[0] for r in zr})
E("h1.zr.n_runs", len(zr), SRC)
E("h1.zr.n_bases", len({r["base"] for r in zr}), SRC)
E("h1.zr.n_counter", len(ce), SRC)
E("h1.zr.n_notrepro", sum(1 for r in runs if r["gate"].startswith("not reproduced")), SRC)
E("h1.zr.n_noref", sum(1 for r in runs if r["gate"] == "no published reference"), SRC)
E("h1.zr.gate_active", "yes" if GATE else "no", SRC)
conf = [r for r in zr if r["verdict"] != "counterexample"]   # summaries describe the CONFIRMING runs; counterexamples get their own keys
E("h1.zr.n_confirm", len(conf), SRC)
import statistics
_med = statistics.median(r["final"][0] for r in zr)
E("h1.zr.final_median", "0" if _med == 0 else (r"$<$.01" if _med < 0.01 else f"{_med:.2f}"), SRC + ": median final rate over counted runs")
E("h1.zr.final_max", round(max(r["final"][0] for r in conf), 4), SRC)
E("h1.zr.upper95_max", round(max(r["final"][2] for r in conf), 3), SRC)
for r in ce:
    E(f"h1.zr.ce.{r['id']}.rate", f"{r['final'][0]:.2f}", SRC)
    E(f"h1.zr.ce.{r['id']}.lo", f"{r['final'][1]:.2f}", SRC)
    E(f"h1.zr.ce.{r['id']}.hi", f"{r['final'][2]:.2f}", SRC)
E("h1.zr.base_min_size", "0.5B", SRC)
d = [r["dacc"] for r in conf if r["dacc"] is not None]
E("ctl.zr.dacc_min", round(min(d), 1), SRC)
E("ctl.zr.dacc_max", round(max(d), 1), SRC)
import statistics
E("ctl.zr.dacc_median", f"{statistics.median(d):.1f}", SRC + ": median accuracy gain over the confirming runs")
for r in runs:
    E(f"h1.zr.{r['id']}.verdict", r["verdict"].replace("_", " "), SRC)
    E(f"h1.zr.{r['id']}.gate", r["gate"], SRC)
fams = []
for r in zr:
    f = r["base"].split("-")[0]
    if f not in fams:
        fams.append(f)
E("h1.zr.families", fams[0] if len(fams) == 1 else ", ".join(fams[:-1]) + " and " + fams[-1], SRC)
E("h1.zr.n_families", len(fams), SRC)
sizes = sorted({float(m.group(1)) for r in zr for m in [re.search(r"(\d+(?:\.\d+)?)B", r["base"])] if m})
fmt_b = lambda x: (f"{x:g}") + "B"
E("h1.zr.size_min", fmt_b(sizes[0]), SRC)
E("h1.zr.size_max", fmt_b(sizes[-1]), SRC)
# Rate per 1k from the counts for every prefix with nwait and ntok (2 decimals; "<.01"-style formatting stays in the
# tables). Text cites <prefix>.r2 so it always agrees with the count-based tables.
for k in list(K):
    if k.endswith(".nwait") and k[:-6] + ".ntok" in K:
        pre = k[:-6]
        try:
            x = num(k) * 1000 / num(pre + ".ntok")
            E(pre + ".r2", "0" if x == 0 else (r"$<$.01" if x < 0.01 else f"{x:.2f}"), f"{SRC}: 1000*nwait/ntok")
        except (ValueError, ZeroDivisionError):
            pass
# Two-decimal copies of ratio keys quoted in the text (format rule: 2 decimals in prose).
for k in list(K):
    if re.search(r"\.trend\.a_(nextopen|wait)$", k):
        try:
            E(k + ".r2", f"{num(k):.2f}", f"{SRC}: rounded copy of {k}")
        except ValueError:
            pass
import math
E("h1.zr.upper95_ceil", f"{math.ceil(max(r['final'][2] for r in conf) * 100) / 100:.2f}", SRC + ": exact upper bound, rounded UP to 2 decimals")
Path("numbers/keys").mkdir(exist_ok=True)
# Sensitivity of the zero-RL count to analysis choices (descriptive; the registered D47 count above is the result).
SSRC = "tools/derive_zerorl.py: sensitivity of the D47 count (descriptive, not registered)"
def verdict_v(f, b, thr, ci=True):
    if b and b[0] >= thr:
        return "rl_on_marker"
    if b is None:
        return "counterexample" if f[0] >= thr else "confirm"
    return "counterexample" if (f[0] >= thr and (f[1] > b[2] or not ci)) else "confirm"
def sens(tag, rows, thr=0.05, ci=True, sfx=None):
    n = c = 0
    for r in rows:
        rid, base, bp0, fp, ba0, fa = PAIRS_BY_ID[r["id"]]
        if sfx is None:
            f, b = r["final"], r["base_rate"]
        else:
            if not has(rid, bp0, fp, ba0, fa, sfx):
                continue
            f = rate_s(fp, sfx); b = rate_s(pick(bp0, sfx), sfx) if bp0 else None
        v = verdict_v(f, b, thr, ci)
        if v == "rl_on_marker":
            continue
        n += 1; c += v == "confirm"
    E(f"sens.zr.{tag}.n_runs", n, SSRC); E(f"sens.zr.{tag}.n_confirm", c, SSRC)
gated = [r for r in counted]
sens("reg", gated)
sens("thr10", gated, thr=0.1)
sens("thr02", gated, thr=0.02)
sens("point", gated, ci=False)
sens("t10", gated, sfx="_t10")
sens("released", [r for r in gated if r["id"] not in OWN | D49])
sens("nogate", [r for r in runs if r["gate"] != "no licence (secondary only)"])
sens("withsec", runs)
# Sensitivity of the OLMo-3 RL-Zero rise (S1, step 400 vs base) to the rate unit, decoding and sampling seed.
def ratio(a, b):
    try:
        return f"{num(a) / num(b):.2f}"
    except (ValueError, ZeroDivisionError, KeyError):
        return None
for tag, a, b in [("sent", "h1.s1.step0400.freq100", "h1.s1.base.freq100"),
                  ("greedy", "h1.s1greedy.step0400.freq", "h1.s1greedy.base.freq"),
                  ("greedy_sent", "h1.s1greedy.step0400.freq100", "h1.s1greedy.base.freq100")]:
    v = ratio(a, b)
    if v:
        E(f"sens.s1.{tag}.ratio", v, f"{SSRC}: {a} / {b}")
sb = [num(k) for k in ["h1.s1.base.freq"] + [f"h1.s1seed{i}.base.freq" for i in range(1, 6)] if k in K]
if sb:
    E("sens.s1.seedbase.min", f"{min(sb):.1f}", SSRC); E("sens.s1.seedbase.max", f"{max(sb):.1f}", SSRC)
    E("sens.s1.seedbase.n", len(sb), SSRC)
json.dump({"entries": out}, open("numbers/keys/zerorl_keys.json", "w"), indent=1)
print(f"{len(zr)} zero-RL runs from marker-free bases on {len({r['base'] for r in zr})} bases ({', '.join(fam)}); "
      f"counterexamples: {len(ce)} {[r['id'] for r in ce]}; rl_on_marker: {[r['id'] for r in runs if r['verdict']=='rl_on_marker']}; pending: {missing}")
for r in runs:
    b = "n/m" if r["base_rate"] is None else "%.4f" % r["base_rate"][0]
    print(f"  {r['id']:8s} {r['base']:18s} base {b:>7s} -> final {r['final'][0]:.4f} [{r['final'][1]:.4f}, {r['final'][2]:.4f}]  acc {r['acc']:.1f} trunc {r['trunc']:.1f}  {r['verdict']} / {r['gate']} / {r['decode']}")
