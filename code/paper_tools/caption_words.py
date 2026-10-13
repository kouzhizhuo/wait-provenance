#!/usr/bin/env python3
"""Word counts of \\caption{...} per file (TeX commands stripped; \\res/\\rt values count as one word)."""
import re, sys
def captions(t):
    i = 0
    while (j := t.find(r"\caption{", i)) >= 0:
        k, d = j + 9, 1
        while d:
            d += {"{": 1, "}": -1}.get(t[k], 0); k += 1
        yield t[j + 9:k - 1]; i = k
def words(c):
    c = re.sub(r"\\(res|rt|Cref|cref|ref|citep|citet|label)\{[^}]*\}", " X ", c)
    c = re.sub(r"\\[a-zA-Z]+\*?", " ", c); c = re.sub(r"[{}$]", " ", c)
    return len([w for w in c.split() if re.search(r"\w", w)])
for f in sys.argv[1:]:
    t = open(f).read()
    for c in captions(t):
        lab = re.search(r"\\label\{([^}]*)\}", t[t.find(c):t.find(c) + len(c) + 200])
        print(f"{words(c):4d}  {f}  {lab.group(1) if lab else ''}")
