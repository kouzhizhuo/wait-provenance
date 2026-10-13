"""Rule-based revision-event detector (EXPERIMENT_PLAN §2.1, rule (i)).

A *revision event* at sentence i is detected from CONTENT ONLY, never from the discourse markers in
`markers.DISC_NAMES`. Otherwise OBL(t) = P(marker | revision event) would be 1 by construction.

Event types (priority order):
  value     sentence i states `key = number` (or "the sum is number") for a key that an earlier sentence
            already assigned a *different* number.
  negation  explicit error/negation vocabulary ("mistake", "incorrect", "that's not right", ...).
  restart   "start over", "from scratch", "redo", "try again".

The rule is a stand-in for the local judge (EXPERIMENT_PLAN §2.1 (ii)); its precision/recall are
measured against the judge and the hand labels before any headline number is quoted.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Set

from .markers import normalise_quotes

_NUM = r"-?\d+(?:,\d{3})*(?:\.\d+)?"
_FRAC = r"-?\\d?frac\{\d+\}\{\d+\}"
_VAL = r"(?:%s|%s)(?:\s*/\s*\d+)?" % (_FRAC, _NUM)
_VAL_RE = re.compile(_VAL)

# words after which an equation's left-hand side starts (prose that precedes the expression)
_LHS_STOP = re.compile(r"(?:,|;|:|\$|\\\\|&|\band\b|\bis\b|\bare\b|\bso\b|\bthen\b|\bget\b|\bgives?\b|\bhave\b|\bwe\b|"
                       r"\bthus\b|\btherefore\b|\bwith\b|\bwhere\b|\bwhen\b|\bif\b|\bfor\b|\bnow\b|\bhence\b|\bequals?\b)", re.I)

_VERBAL_RE = re.compile(
    r"\b(?P<key>answer|result|sum|product|difference|quotient|total|remainder|area|perimeter|volume|probability|"
    r"slope|distance|radius|length|count|number of \w+|value of \w+)\s+(?:is|are|equals|=)\s+(?:approximately\s+)?\$?"
    r"(?P<val>" + _VAL + r")(?![\w^{(]|\.\d)", re.I)

_NEG_RE = re.compile(
    r"\b(?:mistake|(?:an?|the|my|this|possible)\s+(?:\w+\s+)?error|incorrect(?:ly)?|wrong(?:ly)?|miscalculat\w*|misread|misinterpret\w*|contradict\w*|"
    r"not\s+(?:right|correct)|(?:isn't|is\s+not|wasn't|was\s+not|doesn't|does\s+not)\s+(?:seem\s+|look\s+)?(?:right|correct|add\s+up)|"
    r"doesn't\s+make\s+sense)\b", re.I)
_RESTART_RE = re.compile(r"\b(?:start(?:ing)?\s+over|from\s+scratch|re-?do|re-?start|try\s+again|scrap\s+that)\b", re.I)


def parse_value(s: str) -> Optional[float]:
    """'1,234' -> 1234.0, '3/4' -> 0.75, '\\frac{3}{4}' -> 0.75, '-2.5' -> -2.5; None if not numeric."""
    s = s.strip().replace(" ", "")
    m = re.fullmatch(r"(-?)\\d?frac\{(\d+)\}\{(\d+)\}", s)
    if m:
        d = int(m.group(3))
        return None if d == 0 else (-1 if m.group(1) else 1) * int(m.group(2)) / d
    m = re.fullmatch(r"(%s)/(\d+)" % _NUM, s)
    if m:
        d = int(m.group(2))
        v = parse_value(m.group(1))
        return None if d == 0 or v is None else v / d
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _norm_key(lhs: str) -> str:
    k = lhs
    for a, b in ((r"\(", ""), (r"\)", ""), (r"\[", ""), (r"\]", ""), (r"\left", ""), (r"\right", ""), (r"\cdot", "*"), (r"\times", "*"), ("×", "*"), (r"\div", "/"),
                 (r"\,", ""), (r"\!", ""), (r"\ ", ""), ("$", "")):
        k = k.replace(a, b)
    k = re.sub(r"\s+", "", k).lower()
    return k.lstrip("+*/^=-(")


def _valid_key(k: str) -> bool:
    """Reject single-letter/single-digit keys (too ambiguous) and pure numbers."""
    if len(k) < 2:
        return False
    has_digit = any(c.isdigit() for c in k)
    has_op = any(c in "+-*/^" for c in k)
    word3 = re.search(r"[a-z]{3,}", k) is not None  # a real word ("sum", "area"), not a subscripted variable
    if k.replace(".", "").replace(",", "").isdigit():
        return False
    return (has_digit and has_op) or word3 or ("(" in k and has_digit)


def extract_assignments(sentence: str) -> List[tuple]:
    """(key, value) pairs asserted in a sentence: `lhs = number` and 'the sum is number'."""
    s = normalise_quotes(sentence)
    out = []
    for i, ch in enumerate(s):
        if ch != "=":
            continue
        if i > 0 and s[i - 1] in "<>!=\\":
            continue
        if i + 1 < len(s) and s[i + 1] == "=":
            continue
        rest = s[i + 1:]
        m = re.match(r"\s*(" + _VAL + r")", rest)
        if not m:
            continue
        tail = rest[m.end():].lstrip()
        if tail[:1] in ("+", "*", "/", "^", "×", "÷") or tail.startswith(("\\cdot", "\\times", "\\div", "\\sqrt")):
            continue  # rhs continues as an expression, not a plain number
        if re.match(r"[-]\s*\d", tail):
            continue
        lhs_full = s[:i]
        stops = list(_LHS_STOP.finditer(lhs_full))
        lhs = lhs_full[stops[-1].end():] if stops else lhs_full
        key = _norm_key(lhs[-60:])
        v = parse_value(m.group(1))
        if v is not None and _valid_key(key):
            out.append((key, v))
    for m in _VERBAL_RE.finditer(s):
        v = parse_value(m.group("val"))
        if v is not None:
            out.append(("V:" + re.sub(r"\s+", " ", m.group("key").lower()), v))
    return out


def detect_events(sentences: Sequence[str]) -> List[Optional[str]]:
    """Per-sentence revision event type ('value' | 'negation' | 'restart') or None."""
    seen: Dict[str, Set[float]] = {}
    events: List[Optional[str]] = []
    for s in sentences:
        ev: Optional[str] = None
        pairs = extract_assignments(s)
        for k, v in pairs:
            prev = seen.get(k)
            if prev is not None and not any(abs(v - p) <= 1e-9 * max(1.0, abs(p)) for p in prev):
                ev = "value"
                break
        if ev is None and _NEG_RE.search(normalise_quotes(s)):
            ev = "negation"
        if ev is None and _RESTART_RE.search(s):
            ev = "restart"
        for k, v in pairs:
            seen.setdefault(k, set()).add(v)
        events.append(ev)
    return events


def is_revise_event(ev: Optional[str]) -> bool:
    """REVISE outcome of the insertion protocol = revise-value or restart (plus explicit negation)."""
    return ev in ("value", "negation", "restart")
