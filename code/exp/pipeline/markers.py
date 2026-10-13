"""Sentence splitting, marker (form) extraction and position/orthography features.

Everything works on *detokenised text* with regexes, never on token ids (EXPERIMENT_PLAN §2.1, §7.3).

Terminology (Lehmann-style parameters, EXPERIMENT_PLAN §2.1):
  * form      a member of the revision paradigm F (wait, hold on, hmm, actually, ...).
  * disc      discourse-marker forms (wait, hold on, hmm, actually, alternatively, but, however, oh, no, zh).
  * content   lexical forms that describe a revision (let me re-check, I made a mistake, that's not right).
  * slot      the sentence(s) where a revision event happens (see revisions.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

MASK = "█"  # replaces characters inside math spans so they never split sentences or match forms

_MATH_RE = re.compile(r"(?<!\\)\$\$.+?(?<!\\)\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$|\\\[.+?\\\]|\\\(.+?\\\)", re.S)
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|\d+(?:\.\d+)?|[一-鿿]+|" + MASK + "+")
_PUNCT_CLOSERS = "\"')]*_"


def normalise_quotes(text: str) -> str:
    """Curly quotes -> ASCII, same length so offsets are preserved."""
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def mask_math(text: str) -> str:
    """Replace every character inside $...$, $$...$$, \\(...\\), \\[...\\] with MASK (offsets preserved)."""
    return _MATH_RE.sub(lambda m: MASK * len(m.group(0)), text)


# ----------------------------------------------------------------------------- sentences
@dataclass(frozen=True)
class Sentence:
    start: int
    end: int
    text: str  # original text of the span (not masked)

    def __len__(self) -> int:
        return len(self.text)


_SENT_END_RE = re.compile(r"[.!?。！？][" + re.escape(_PUNCT_CLOSERS) + r"]*(?=[ \t]+[^\sa-z])")
_ZH_END_RE = re.compile(r"[。！？](?=[一-鿿])")
_NL_RE = re.compile(r"[ \t]*\n\s*")


def split_sentences(text: str) -> List[Sentence]:
    """Split on newlines and on sentence-final punctuation followed by a non-lowercase character.

    Math spans are masked first so "$x = 3.$ Then" does not split inside math. Bare list markers such
    as "1." at a line start do not create a sentence of their own. Offsets refer to `text`.
    """
    masked = mask_math(normalise_quotes(text))
    cuts: List[Tuple[int, int]] = []  # (sentence_end, next_sentence_start)
    for m in _SENT_END_RE.finditer(masked):
        line_start = masked.rfind("\n", 0, m.start()) + 1
        if masked[line_start:m.start()].strip().isdigit():  # "1." list marker
            continue
        j = m.end()
        while j < len(masked) and masked[j] in " \t":
            j += 1
        cuts.append((m.end(), j))
    for m in _ZH_END_RE.finditer(masked):
        cuts.append((m.end(), m.end()))
    for m in _NL_RE.finditer(masked):
        cuts.append((m.start(), m.end()))
    cuts.sort()
    spans: List[Tuple[int, int]] = []
    pos = 0
    for e, s in cuts:
        if e < pos:
            continue
        spans.append((pos, e))
        pos = s
    spans.append((pos, len(text)))
    out: List[Sentence] = []
    for a, b in spans:
        seg = text[a:b]
        if not seg.strip():
            continue
        lead = len(seg) - len(seg.lstrip())
        trail = len(seg) - len(seg.rstrip())
        out.append(Sentence(a + lead, b - trail, text[a + lead:b - trail]))
    return out


def words(sentence_masked: str) -> List[Tuple[str, int, int]]:
    """Word tokens (lower-cased) with offsets, ignoring punctuation. Math runs count as one word."""
    return [(m.group(0).lower(), m.start(), m.end()) for m in _WORD_RE.finditer(sentence_masked)]


# ----------------------------------------------------------------------------- forms
@dataclass(frozen=True)
class FormSpec:
    name: str
    pattern: str
    kind: str  # "disc" | "content"
    initial_only: bool = False  # only counts when it is the first word of a sentence


_APOS = r"(?:'|’)"
FORMS: Tuple[FormSpec, ...] = (
    FormSpec("wait", r"\bwait\b", "disc"),
    FormSpec("hold_on", r"\bhold on\b", "disc"),
    FormSpec("hmm", r"\bhm+\b", "disc"),
    FormSpec("actually", r"\bactually\b", "disc"),
    FormSpec("alternatively", r"\balternatively\b", "disc"),
    FormSpec("but", r"\bbut\b", "disc", initial_only=True),
    FormSpec("however", r"\bhowever\b", "disc", initial_only=True),
    FormSpec("oh", r"\boh\b", "disc", initial_only=True),
    FormSpec("no", r"\bno\b(?=\s*[,.!])", "disc", initial_only=True),
    FormSpec("wait_zh", r"等等|等一下|稍等", "disc"),
    FormSpec("no_zh", r"不对", "disc"),
    FormSpec("recheck", r"\blet(?:" + _APOS + r"s|\s+me|\s+us)\s+(?:re-?check|double[- ]?check|re-?verify|verify|re-?examine|"
             r"re-?evaluate|re-?consider|re-?calculate|re-?compute|check)\b", "content"),
    FormSpec("mistake", r"\b(?:i|we)\s+(?:made|make|see|spotted|found)\s+(?:a|an)\s+(?:mistake|error)\b|\bmistake\b", "content"),
    FormSpec("not_right", r"\b(?:that|this|it)(?:" + _APOS + r"s|\s+is)\s+(?:not\s+(?:right|correct)|wrong|incorrect)\b"
             r"|\b(?:that|this|it)\s+(?:doesn" + _APOS + r"t|does not)\s+(?:seem|look)\s+right\b", "content"),
)
FORM_NAMES: Tuple[str, ...] = tuple(f.name for f in FORMS)
DISC_NAMES: Tuple[str, ...] = tuple(f.name for f in FORMS if f.kind == "disc")
CONTENT_NAMES: Tuple[str, ...] = tuple(f.name for f in FORMS if f.kind == "content")
_COMPILED = [(f, re.compile(f.pattern, re.I)) for f in FORMS]
_KIND = {f.name: f.kind for f in FORMS}

LEXICAL_NEXT = {"for", "until", "till", "time", "times", "list", "period", "line", "and"}  # "wait for ..." = lexical use
CONNECTIVES = {"but", "and", "so", "no", "oh", "okay", "ok", "hmm", "well", "yet"}


def form_kind(name: str) -> str:
    return _KIND[name]


@dataclass(frozen=True)
class Marker:
    form: str
    kind: str
    start: int  # offsets into the analysed text
    end: int
    surface: str
    sent_idx: int
    word_idx: int  # index of the marker's first word among the sentence's words (0 = sentence-initial)
    n_words: int  # number of words in the sentence
    prev_word: Optional[str]  # previous word in the sentence (lower-cased) or None
    next_word: Optional[str]  # next word after the marker, punctuation skipped (lower-cased) or None
    next_char: str  # character right after the marker in the text ("" at end)
    lexical: bool  # e.g. "wait for" (dictionary sense), only meaningful for form == "wait"


def _locate_sentence(sents: Sequence[Sentence], pos: int) -> int:
    lo, hi = 0, len(sents) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = sents[mid]
        if pos < s.start:
            hi = mid - 1
        elif pos >= s.end:
            lo = mid + 1
        else:
            return mid
    return -1


def find_markers(text: str, sents: Optional[Sequence[Sentence]] = None) -> List[Marker]:
    """All occurrences of every form in FORMS, with sentence and word positions (sorted by offset)."""
    if sents is None:
        sents = split_sentences(text)
    norm = normalise_quotes(text)
    masked = mask_math(norm)
    sent_words = [words(masked[s.start:s.end]) for s in sents]
    out: List[Marker] = []
    for spec, rx in _COMPILED:
        for m in rx.finditer(masked):
            si = _locate_sentence(sents, m.start())
            if si < 0:
                continue
            s = sents[si]
            ws = sent_words[si]
            rel = m.start() - s.start
            widx = next((k for k, (_, a, _b) in enumerate(ws) if a >= rel), len(ws))
            if spec.initial_only and widx != 0:
                continue
            prev_w = ws[widx - 1][0] if 0 < widx <= len(ws) else None
            # next word after the marker's last word: first word starting at/after marker end
            rel_end = m.end() - s.start
            nxt = next((w for (w, a, _b) in ws if a >= rel_end), None)
            nchar = norm[m.end():m.end() + 1]
            out.append(Marker(spec.name, spec.kind, m.start(), m.end(), text[m.start():m.end()], si, widx, len(ws),
                              prev_w, nxt, nchar, spec.name == "wait" and nxt in LEXICAL_NEXT))
    out.sort(key=lambda k: (k.start, k.form))
    return out


# ----------------------------------------------------------------------------- features
POS_BINS: Tuple[str, ...] = ("initial", "early", "medial", "final")


def position_class(m: Marker) -> str:
    """initial (1st word) > early (2nd-3rd word) > final (last word) > medial."""
    if m.word_idx == 0:
        return "initial"
    if m.word_idx <= 2:
        return "early"
    if m.word_idx >= m.n_words - 1:
        return "final"
    return "medial"


def is_connective_initial(m: Marker) -> bool:
    """Sentence-initial or right after a single connective ("But wait", "So wait", "No wait")."""
    return m.word_idx == 0 or (m.word_idx == 1 and m.prev_word in CONNECTIVES)


def orth_class(m: Marker) -> str:
    """canonical = capitalised + comma ("Wait,"); cap_other = capitalised, other/no punctuation; lower."""
    first = m.surface[:1]
    if first.isupper():
        return "canonical" if m.next_char == "," else "cap_other"
    return "lower"


def is_paragraph_initial(text: str, m: Marker, sents: Sequence[Sentence]) -> bool:
    """Marker starts its sentence and that sentence starts the text or follows a newline."""
    if m.word_idx != 0:
        return False
    s = sents[m.sent_idx]
    return s.start == 0 or "\n" in text[max(0, s.start - 3):s.start]


_ACTION_RE = re.compile(r"\d|=|\\boxed|\\frac|\\sqrt|\\cdot|\\times|[+*/^×÷]|(?<=\s)-(?=\s*\d)")
_TOK_RE = re.compile(r"\w+|[^\w\s]", re.U)


def tokens_to_action(text: str, m: Marker) -> Optional[int]:
    """Regex-tokens between the marker and the next mathematical action token (digit, '=', operator,
    \\boxed...). None if there is none in the rest of the response. Uses the raw text (math is not masked)."""
    rest = text[m.end:]
    a = _ACTION_RE.search(rest)
    if not a:
        return None
    return len(_TOK_RE.findall(rest[:a.start()]))


def clause_length(text: str, m: Marker, sents: Sequence[Sentence]) -> int:
    """Number of words in the clause (split at , ; :) containing the marker."""
    s = sents[m.sent_idx]
    seg = mask_math(normalise_quotes(text))[s.start:s.end]
    rel = m.start - s.start
    left = max(seg.rfind(c, 0, rel) for c in ",;:")
    right_c = [seg.find(c, rel + (m.end - m.start)) for c in ",;:"]
    right_c = [r for r in right_c if r >= 0]
    right = min(right_c) if right_c else len(seg)
    return len(words(seg[left + 1:right]))


def count_forms(text: str) -> Dict[str, int]:
    c = {n: 0 for n in FORM_NAMES}
    for m in find_markers(text):
        c[m.form] += 1
    return c


# ----------------------------------------------------------------------------- mechanical blinding (D15)
_STRIP_CORE = r"(?:wait|hold\s+on|hmm+|hm+|actually|however|alternatively|oh|okay|ok|well|alright|now|next)"
# But / So / Then carry content ("But x is odd"), so they are stripped only before a comma or before another stripped form
_STRIP_CONJ = r"(?:but|so|then)\b(?=\s*,|\s+" + _STRIP_CORE + r"\b)"
_STRIP_NO = r"no\b(?=\s*[,.!])"  # "No, ..." / "No. ..." but not "No solution exists"
_STRIP_TOKENS = r"(?:" + _STRIP_CONJ + r"|" + _STRIP_CORE + r"\b|" + _STRIP_NO + r")"
_STRIP_RE = re.compile(r"^(?P<lead>[\s\W_]*?)(?P<forms>(?:" + _STRIP_TOKENS + r"[\s,.:;!\-\u2014\u2026]*)+)", re.I)


def strip_initial_forms(sentence: str):
    """Remove sentence-initial discourse forms (wait, hold on, hmm, actually, however, alternatively, oh, no, okay, ok,
    well, alright, now, next; but / so / then only before a comma or another such form) plus the following punctuation
    before a sentence goes to the revision detector or judge (notes/JUDGE_PROMPTS_v1.md §0.3).
    Returns (stripped_sentence, removed_string). The first letter of the remainder is capitalised. If the whole
    sentence consists of such forms, the stripped sentence is empty. The removed token is meant to be logged.
    """
    m = _STRIP_RE.match(normalise_quotes(sentence))
    if not m or not m.group("forms").strip():
        return sentence, ""
    rest = sentence[m.end():]
    removed = sentence[m.start("forms"):m.end()].strip()
    rest = rest.lstrip()
    return (rest[:1].upper() + rest[1:]) if rest else "", removed


def wait_variant(m: "Marker") -> str:
    """Orthographic variant of a `wait` occurrence: its surface form plus what follows it (',' '.' ':' '!' '?', a dash,
    an ellipsis, a line break, or '_' for a bare word). Used for ORTH_max and the variant tables."""
    nc = m.next_char
    if nc in ",.:!?":
        tail = nc
    elif nc in "\u2014\u2013-":
        tail = "\u2014"
    elif nc == "\u2026":
        tail = nc
    elif nc == "\n":
        tail = "\\n"
    else:
        tail = "_"
    return m.surface + tail
