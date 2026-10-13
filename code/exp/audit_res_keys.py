"""Mechanical \\res-key audit (coder side of /paper-claim-audit): every \\res{key} used in paper/**/*.tex must be defined in paper/numbers/*.tex,
every definition must be traceable to a key in exp/results/**/*keys*.json with the same value, and pilot-sourced definitions are listed.
    python3 exp/audit_res_keys.py            -> prints a report, writes exp/results/reframe/res_audit.json"""
import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
used = {}
for f in glob.glob(str(ROOT / "paper" / "**" / "*.tex"), recursive=True):
    if "/numbers/" in f or "/build/" in f or "/acl_template/" in f:
        continue
    for i, line in enumerate(open(f, encoding="utf-8", errors="ignore"), 1):
        if line.lstrip().startswith("%"):
            continue
        for k in re.findall(r"\\res\{([^}]+)\}", line):
            used.setdefault(k, []).append("%s:%d" % (Path(f).name, i))
defs = {}
for f in sorted(glob.glob(str(ROOT / "paper" / "numbers" / "*.tex")), key=lambda x: (x.endswith("results.tex"), x)):   # pilot.tex first, results.tex last (main.tex load order: later wins)
    for line in open(f, encoding="utf-8"):
        m = re.match(r"\\resdef\{([^}]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}\s*(?:%\s*src:\s*(.*))?", line)
        if m:
            defs[m.group(1)] = (m.group(2).replace("{,}", ""), m.group(3) or "", Path(f).name)
vals = {}
for f in glob.glob(str(ROOT / "exp" / "**" / "*keys*.json"), recursive=True):
    if "/._" in f:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for e in d.get("entries", []) if isinstance(d, dict) else []:
        vals.setdefault(e["key"], set()).add(str(e["value"]))
undefined = sorted(k for k in used if k not in defs)
pilot_only = sorted(k for k in used if k in defs and defs[k][2] == "pilot.tex")
mismatch = sorted(k for k, (v, _, fn) in defs.items() if k in used and k in vals and v.strip() not in vals[k] and not any(abs(float(v) - float(x)) < 1e-9 for x in vals[k] if re.fullmatch(r"-?[\d.]+", x) and re.fullmatch(r"-?[\d.]+", v.strip())))
untraceable = sorted(k for k in used if k in defs and k not in vals)
rep = {"n_used": len(used), "n_defs": len(defs), "undefined": {k: used[k] for k in undefined}, "pilot_only": pilot_only, "value_mismatch_vs_results_files": mismatch, "defined_but_no_key_in_results_files": untraceable}
Path(ROOT / "exp" / "results" / "reframe" / "res_audit.json").write_text(json.dumps(rep, indent=1))
print({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in rep.items()})
for k in ("undefined", "pilot_only", "value_mismatch_vs_results_files", "defined_but_no_key_in_results_files"):
    print(k, list(rep[k])[:15])
