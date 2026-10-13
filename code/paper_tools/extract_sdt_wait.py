#!/usr/bin/env python3
"""Keys for the `wait`-only SDT outcome (dprime_wait, criterion_wait) that the producers wrote into sdt.json but not into
their key files. The registered outcome (h2.*.dprime / .crit) counts any form of the revision paradigm F; these keys
report `wait` alone next to it. Read-only extraction, src recorded per key.

    python3 tools/extract_sdt_wait.py      # writes numbers/keys/sdt_wait_keys.json
"""
import json
from pathlib import Path

R = Path("../exp/results")
SRC = {"s1": (R / "main/S1/sdt.json", {"base": "base", "final": "final", "late": "1900"}),
       "s2": (R / "main/S2/sdt.json", {"base": "base", "late": "1375"}),
       "s9": (R / "reframe/s9_sdt.json", {"step0100": "100", "late": "1800"})}
out = []
for ser, (path, stages) in SRC.items():
    cks = {str(c["ckpt"]): c for c in json.load(open(path))["checkpoints"]}
    for stage, ck in stages.items():
        for field, name, scale, dig in (("dprime_wait", "dprime_wait", 1, 2), ("criterion_wait", "crit_wait", 1, 2)):
            v = cks[ck][field]
            for part in ("est", "lo", "hi"):
                key = f"h2.{ser}.{stage}.{name}" + ("" if part == "est" else f".{part}")
                out.append({"key": key, "value": round(v[part] * scale, dig), "src": f"{path}::checkpoints[{ck}].{field}.{part}"})
Path("numbers/keys").mkdir(parents=True, exist_ok=True)
json.dump({"entries": out}, open("numbers/keys/sdt_wait_keys.json", "w"), indent=1)
print(len(out), "keys")
