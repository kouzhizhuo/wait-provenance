"""Frozen tables for Calibrated Wait (D50): marker first-tokens (paradigm F), sentence-boundary tokens, connectors, digit-ending tokens, for ONE tokenizer.
    python -m exp.cw.cw_prep --tokenizer <dir> --out <cw_tables_<model>.json> [--trust]
The RULES below are frozen before the first run; the sha256 of this file's RULES block + the produced id lists goes into every run meta ('tables_hash').
RULES (frozen 2026-10-05):
  marker token  : decoded text, after stripping leading whitespace / newline / markdown chars (* # > - _ ` ( [ "), lower-cased, matches  ^(wait|hmm+|hold|actually|alternatively)[,.!:;]*$   (first token of: Wait, Hold on, Hmm, Actually, Alternatively; 'But wait' = connector + 'wait')
  boundary token: decoded text rstripped ends with '.', '!' or '?'  OR contains a newline;  EXCEPT a token whose text is '.' and whose predecessor ends in a digit (decimal point)
  connector     : decoded text stripped lower-cased in {but, so, and, okay, ok, oh, ah, now, then}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re

MARK = re.compile(r"^(wait|hmm+|hold|actually|alternatively)[,.!:;]*$")
STRIP = " \t\n\r*#>-_`(['\"▁ĠĊ"
CONN = {"but", "so", "and", "okay", "ok", "oh", "ah", "now", "then"}


CGRS = re.compile(r"^(wait|but|alternatively|hmm+)[,.!:;]*$")   # CGRS trigger set (Huang et al. 2025, Sec. 3.3): Wait, But, Alternatively, Hmm and their case/space variants; separate file, NOT part of the frozen CW hash


def build_cgrs(tok):
    ids = []
    for i in range(len(tok)):
        try:
            s = tok.decode([i], skip_special_tokens=False)
        except Exception:
            continue
        if s and CGRS.match(s.lstrip(STRIP).lower()):
            ids.append(i)
    return ids


def build(tok):
    V = len(tok)
    marker, boundary, connector, digit_end, dots = [], [], [], [], []
    for i in range(V):
        try:
            s = tok.decode([i], skip_special_tokens=False)
        except Exception:
            continue
        if not s:
            continue
        core = s.lstrip(STRIP).lower()
        if MARK.match(core):
            marker.append(i)
        t = s.rstrip(" \t")
        if t.endswith((".", "!", "?")) or "\n" in s:
            boundary.append(i)
        if s.strip(STRIP).lower() in CONN and s.strip():
            connector.append(i)
        if s.rstrip()[-1:].isdigit():
            digit_end.append(i)
        if s.strip() == ".":
            dots.append(i)
    return {"marker_ids": marker, "boundary_ids": boundary, "connector_ids": connector, "digit_end_ids": digit_end, "dot_ids": dots}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True); ap.add_argument("--trust", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=a.trust, local_files_only=True)
    t = build(tok)
    rules = open(__file__).read().split('"""')[1]
    t["rules"] = rules; t["tables_hash"] = hashlib.sha256((rules + json.dumps({k: v for k, v in t.items() if k.endswith("_ids")}, sort_keys=True)).encode()).hexdigest()
    json.dump(t, open(a.out, "w"))
    json.dump({"cgrs_ids": build_cgrs(tok)}, open(a.out.replace(".json", ".cgrs.json"), "w"))
    ex = [tok.decode([i]) for i in t["marker_ids"][:40]]
    print("markers %d (e.g. %s) | boundary %d | connectors %d | digit_end %d | hash %s" % (len(t["marker_ids"]), ex, len(t["boundary_ids"]), len(t["connector_ids"]), len(t["digit_end_ids"]), t["tables_hash"][:12]))


if __name__ == "__main__":
    main()
