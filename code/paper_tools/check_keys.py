#!/usr/bin/env python3
"""Fail if any \\res/\\resT/\\rpm/\\rcell/\\rci key used in a file reachable from main.tex is missing from numbers/results.tex
(lead 064961b7, 2026-10-04). Run from paper/ after tools/regen.sh; exit code 1 on missing keys."""
import re, sys
from pathlib import Path
R = set(re.findall(r'\\resdef\{([^}]*)\}', Path('numbers/results.tex').read_text()))
used = {'main.tex'}
def walk(f):
    for m in re.findall(r'\\input\{([^}]*)\}', Path(f).read_text()):
        p = m if m.endswith('.tex') else m + '.tex'
        if Path(p).exists() and p not in used:
            used.add(p); walk(p)
walk('main.tex')
bad = []
for f in sorted(used):
    s = re.sub(r'(?<!\\)%.*', '', Path(f).read_text())
    bad += [(f, k) for _, k in re.findall(r'\\(res|rt|cell|cellsd|resT|rpm|rcell|rci|rhl|rsig)\{([^}#]*)\}', s) if k not in R]
for f, k in bad:
    print(f'MISSING {k}  ({f})')
print(f'check_keys: {len(used)} files, {len(bad)} missing')
sys.exit(1 if bad else 0)
