"""Rounded, signed copies of the Calibrated Wait contrast keys for prose (h5.r.*), from numbers/results.tex.
acc in points, tokens in % change (negative = fewer); one decimal; source keys are the coder's paired problem-cluster bootstrap."""
import json, re
from pathlib import Path
K = dict(re.findall(r"\\resdef\{([^}]*)\}\{(.*?)\}[ \t]*(?:%.*)?$", Path("numbers/results.tex").read_text(), re.M))
def num(k):
    return float(K[k].replace(r"\ensuremath{-}", "-").replace("{,}", ""))
def sg(x):
    return (r"\ensuremath{-}" if x < 0 else "+") + f"{abs(x):.1f}"
out = []
E = lambda k, v, src: out.append({"key": k, "value": v, "src": "tools/derive_cw.py: " + src})
for arm in ["sham", "nowait", "random", "bf_match", "bf_loose", "cgrs", "cw75", "cw50", "cw"]:
    for ref in ["sham", "vanilla"]:
        for met, unit in [("acc", "pts"), ("tok", "pct")]:
            b = f"h5.macro.{arm}.{met}_vs_{ref}"
            if all(f"{b}.{p}.{unit}" in K for p in ("est", "lo", "hi")):
                for p in ("est", "lo", "hi"):
                    E(f"h5.r.{arm}.{met}_vs_{ref}.{p}", sg(num(f"{b}.{p}.{unit}")), b)
for c in ["S2_cw_minus_nowait", "S3_cw_minus_random", "S6_cw_minus_bf_match"]:
    b = f"h5.macro.{c}"
    if all(f"{b}.{p}" in K for p in ("est", "lo", "hi")):
        for p in ("est", "lo", "hi"):
            E(f"h5.r.{c}.{p}", sg(100 * num(f"{b}.{p}")), b + " x100")
b = "h5.rlzero.none_s1.macro.acc_vs_vanilla"
if all(f"{b}.{p}.pts" in K for p in ("est", "lo", "hi")):
    for p in ("est", "lo", "hi"):
        E(f"h5.r.noise.acc.{p}", sg(num(f"{b}.{p}.pts")), b)
for m in ["rlzero", "olmo3think", "r1d7", "qwen3_8b"]:
    b = f"h5.{m}.s5.diff"
    if all(f"{b}.{p}" in K for p in ("est", "lo", "hi")):
        for p in ("est", "lo", "hi"):
            x = num(f"{b}.{p}")
            E(f"h5.r.{m}.s5.diff.{p}", (r"\ensuremath{-}" if x < 0 else "+") + f"{abs(x):.2f}", b)
json.dump({"entries": out}, open("numbers/keys/cw_derived2_keys.json", "w"), indent=1)
print(f"derive_cw: {len(out)} keys")
