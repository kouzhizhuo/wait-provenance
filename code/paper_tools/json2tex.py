#!/usr/bin/env python3
"""Turn experiment result JSON files into LaTeX \\resdef macros.

Every number in the paper must trace to a file under exp/. Instead of typing
numbers, write \\res{key} in the .tex and regenerate numbers/results.tex:

    python3 tools/json2tex.py ../exp/h1/summary.json:h1 ../exp/h3/effects.json:h3 \\
        -o numbers/results.tex --digits 2

Nested keys are flattened with '.', e.g. {"qwen7b": {"slope": 0.4213}} with
prefix h1 becomes \\res{h1.qwen7b.slope} -> 0.42. Each output line records its
source file. A dict with numeric est/lo/hi also gets hw = max(hi-est, est-lo)
(the table convention of notes/STATS_PLAN.md sec. 4). --pct REGEX adds
<key>.pct = 100*value with one decimal for proportions shown in percent. Integers are printed as-is; floats with --digits decimals
(override per key with a "<key>__digits" entry in the JSON).
"""
import argparse
import json
import re
import sys
from pathlib import Path

KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _decimals(x):
    t = repr(float(x))
    return 0 if "e" in t else len(t.split(".")[1].rstrip("0")) if "." in t else 0


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        if all(_num(obj.get(k)) for k in ("est", "lo", "hi")) and "hw" not in obj:
            # Conservative symmetric summary of a percentile CI (STATS_PLAN sec. 4): the larger half-width,
            # rounded to the precision of its inputs (the coder pre-rounds; RESULTS_KEYS sec. 2).
            nd = max(_decimals(obj[k]) for k in ("est", "lo", "hi"))
            obj = dict(obj, hw=round(max(obj["hi"] - obj["est"], obj["est"] - obj["lo"]), nd))
        if "est" in obj and prefix and not isinstance(obj["est"], (dict, list)):
            # RESULTS_KEYS: the bare key is the estimate (h1.s1.late.si == h1.s1.late.si.est)
            yield prefix, obj["est"]
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}.{i}" if prefix else str(i))
    else:
        yield prefix, obj


MINUS = r"\ensuremath{-}"  # a true minus sign in text and in math (typography rule, 2026-10-04); never a hyphen


def signed(s):
    """'-0.0' -> '0.0'; '-1.2' -> MINUS + '1.2'."""
    if not s.startswith("-"):
        return s
    return s[1:] if float(s.replace("{,}", "")) == 0 else MINUS + s[1:]


def fmt(value, digits):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return signed(f"{value:,}".replace(",", "{,}"))
    if isinstance(value, float):
        if digits < 0:  # --asis: keep the producer's rounding
            return signed(repr(value) if "e" not in repr(value) else f"{value:.3g}")
        return signed(f"{value:.{digits}f}")
    s = str(value)
    return s.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="file.json[:prefix]")
    ap.add_argument("-o", "--output", default="numbers/results.tex")
    ap.add_argument("--digits", type=int, default=2)
    ap.add_argument("--asis", action="store_true", help="print floats exactly as rounded by the producer (RESULTS_KEYS sec. 2)")
    ap.add_argument("--pct", action="append", default=[], metavar="REGEX",
                    help="also emit <key>.pct = 100*value (1 decimal) for numeric keys matching REGEX (repeatable)")
    ap.add_argument("--skip", action="append", default=[], metavar="REGEX",
                    help="drop keys matching REGEX (e.g. diagnostic sub-dicts whose keys are not valid macro names)")
    args = ap.parse_args()
    if args.asis:
        args.digits = -1
    ap_skip = [re.compile(r) for r in args.skip]
    pct_res = [re.compile(r) for r in args.pct]

    lines, seen, seen_val = [], {}, {}
    for spec in args.inputs:
        path, _, prefix = spec.partition(":")
        data = json.loads(Path(path).read_text())
        srcs = {}
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            # RESULTS_KEYS export format: {"entries": [{"key", "value", "src"}, ...]}; values pre-rounded by the producer
            flat = {(prefix + "." if prefix else "") + e["key"]: e["value"] for e in data["entries"]}
            srcs = {(prefix + "." if prefix else "") + e["key"]: e.get("src", path) for e in data["entries"]}
            for e in data["entries"]:
                k = (prefix + "." if prefix else "") + e["key"]
                lo, hi = flat.get(k + ".lo"), flat.get(k + ".hi")
                if _num(e["value"]) and _num(lo) and _num(hi) and k + ".hw" not in flat:
                    nd = max(_decimals(v) for v in (e["value"], lo, hi))
                    flat[k + ".hw"] = round(max(hi - e["value"], e["value"] - lo), nd)
                    srcs[k + ".hw"] = srcs[k] + " (hw = max CI half-width)"
        else:
            flat = dict(flatten(data, prefix))
        for key, value in sorted(flat.items()):
            if key.endswith("__digits") or value is None or any(r.search(key) for r in ap_skip):
                continue
            if not KEY_RE.match(key):
                sys.exit(f"invalid key {key!r} in {path}: use [A-Za-z0-9._-] only")
            if key in seen:
                if seen_val.get(key) == value:  # identical value from a second producer: keep the first
                    continue
                sys.exit(f"duplicate key {key!r} in {path} and {seen[key]} with different values")
            seen[key] = path
            seen_val[key] = value
            digits = int(flat.get(f"{key}__digits", args.digits))
            if value == "":
                value = "{}"  # empty significance mark
            lines.append(f"\\resdef{{{key}}}{{{fmt(value, digits)}}}  % src: {srcs.get(key, path)}")
            if _num(value) and any(r.search(key) for r in pct_res):
                lines.append(f"\\resdef{{{key}.pct}}{{{fmt(100.0 * value, 1)}}}  % src: {path} (x100)")

    header = [
        "% AUTO-GENERATED by tools/json2tex.py -- do not edit by hand.",
        "% Regenerate:  python3 tools/json2tex.py " + " ".join(args.inputs) + f" -o {args.output} --digits {args.digits}",
    ]
    Path(args.output).write_text("\n".join(header + lines) + "\n")
    print(f"wrote {len(lines)} macros to {args.output}")


if __name__ == "__main__":
    main()
