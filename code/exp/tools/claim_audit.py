"""List every digit in paper/sections/*.tex that is NOT inside \\res{...}/\\rt{...} (hand-typed numbers to check against \\res keys).
    python3 exp/tools/claim_audit.py [--sections 0 1 2 ...] [--all]      (default: main-text sections 0..9; --all adds the appendix)
Skips comments, labels/refs/cites/urls/includegraphics/section commands and column specs. Output: file:line  [numbers]  context."""
import argparse
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STRIP = [r"\\(?:res|rt|pilot)\{(?:[^{}]|\{[^{}]*\})*\}", r"\\(?:Cref|cref|ref|eqref|label|autoref|cite[a-z]*|nocite|input|include|url|href|includegraphics|hspace|vspace|bibliography[a-z]*|begin|end|usepackage|newcommand|renewcommand|setlength|resizebox|multicolumn|multirow|cmidrule)(?:\[[^\]]*\])?\{[^{}]*\}",
         r"\\(?:section|subsection|paragraph|caption)\*?\{", r"\$?\\(?:times|approx)\$?"]
NUM = re.compile(r"(?<![\w\\{])(\d[\d,]*\.?\d*)(\s?%|\\%)?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include the appendix")
    a = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "paper" / "sections" / "*.tex")))
    n_tot = 0
    for f in files:
        name = Path(f).name
        if not a.all and not re.match(r"[0-9]_", name):
            continue
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if line.lstrip().startswith("%"):
                continue
            body = re.split(r"(?<!\\)%", line)[0]
            for p in STRIP:
                body = re.sub(p, " ", body)
            hits = [m for m in NUM.finditer(body)]
            if hits:
                n_tot += len(hits)
                print("%s:%d  %s  | %s" % (name, i, [h.group(0).strip() for h in hits], body.strip()[:140]))
    print("TOTAL %d hand-typed number tokens (check each against a \\res key or justify: counts of items, years, equation labels)" % n_tot)


if __name__ == "__main__":
    main()
