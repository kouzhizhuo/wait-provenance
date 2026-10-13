"""Move every \\caption{...} (+ the \\label right after it) of a table/figure float to just after its tabular or graphics,
so captions sit BELOW floats (ACL convention requested 2026-10-12). Idempotent; run on hand-written and generated .tex."""
import re, sys
from pathlib import Path

def grab(s, i):  # s[i] == '{' -> index after the matching brace
    d = 0
    for j in range(i, len(s)):
        d += s[j] == '{'; d -= s[j] == '}'
        if d == 0:
            return j + 1
    raise ValueError

def fix_env(body):
    c = body.find('\\caption{')
    anchors = [m.end() for m in re.finditer(r'\\end\{tabular\*?\}\}?|\\includegraphics(\[[^\]]*\])?\{[^}]*\}', body)]
    if c < 0 or not anchors or c > anchors[0]:
        return body
    ce = grab(body, c + 8)
    m = re.match(r'\s*\\label\{[^}]*\}', body[ce:])
    end = ce + (m.end() if m else 0)
    cap = body[c:end].strip()
    rest = body[:c].rstrip() + '\n' + body[end:].lstrip('\n')
    a = [mm.end() for mm in re.finditer(r'\\end\{tabular\*?\}\}?|\\includegraphics(\[[^\]]*\])?\{[^}]*\}', rest)][-1]
    nl = rest.find('\n', a)
    nl = len(rest) if nl < 0 else nl
    return rest[:nl] + '\n  ' + cap + rest[nl:]

def fix(text):
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{(table|figure)(\*?)\}', text):
        if m.start() < pos:
            continue
        end = text.index('\\end{%s%s}' % (m.group(1), m.group(2)), m.end())
        out.append(text[pos:m.end()]); out.append(fix_env(text[m.end():end])); pos = end
    out.append(text[pos:])
    return ''.join(out)

for f in sys.argv[1:]:
    p = Path(f); t = p.read_text(); n = fix(t)
    if n != t:
        p.write_text(n); print('caption below:', f)
