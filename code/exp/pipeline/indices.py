"""H1 (grammaticalization) indices from generated traces. EXPERIMENT_PLAN §2.1.

Design: each response is reduced to a flat dict of *additive counts* (`trace_counts`). Counts add up
over responses and over problems, so a cluster bootstrap over problems is just "resample rows of a
count matrix, sum, recompute" (`bootstrap_indices`). All indices are pure functions of the summed
counts (`indices_from_totals`), which keeps them unit-testable on hand-made examples.

Index glossary (prediction under grammaticalization in brackets):
  FREQ_wait_1k      occurrences of `wait` per 1,000 response tokens [up]
  FREQ_wait_100s    occurrences of `wait` per 100 sentences [up]
  FREQ_disc_1k      all discourse-marker forms per 1,000 tokens
  SI                P(`wait` is the first word of its sentence | wait) [up]
  SI_conn           same, also counting "But wait"/"So wait"/"No wait" [up]
  PI                P(wait starts a paragraph/line) [up]
  ORTH              share of the canonical form "Wait," (capital + comma) [up]
  H_pos             entropy (bits) of the position class of `wait` in its sentence [down]
  H_para            entropy (bits) of the forms used in revision slots, all forms in F [down]
  H_para_disc       same, discourse-marker forms only
  N_eff             2**H_para
  SHARE_wait        share of `wait` among forms in revision slots
  OBL               P(discourse marker in sentence i or i-1 | revision event at i) [up]
  REL               P(revision event in sentences i..i+2 | marker sentence i) [down]
  REL_strict        same with i+1..i+2 only
  H_next            entropy (bits) of the word following `wait` [down]
  EROS              regex-tokens between `wait` and the next mathematical action token [down]
  SENTLEN_init      mean words in sentences that start with `wait` [down]
  CTX               mean words in the clause containing `wait` [down]
  LEX_share         share of `wait` in its dictionary sense ("wait for ...")
Entropies use the Miller-Madow correction (stats.entropy_bits).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import markers as mk
from . import revisions as rv
from .stats import bootstrap_stat, entropy_bits

REL_WINDOW = 2


def trace_counts(text: str, n_tokens: int) -> Dict[str, float]:
    """Additive counts for one response."""
    c: Dict[str, float] = defaultdict(float)
    sents = mk.split_sentences(text)
    ms = mk.find_markers(text, sents)
    c["n_resp"] += 1
    c["tokens"] += n_tokens
    c["sents"] += len(sents)
    for m in ms:
        c["form:" + m.form] += 1
        if m.kind == "disc":
            c["disc_total"] += 1

    # -- the primary marker `wait`
    prev_wait_end, prev_chained = None, False
    for m in ms:
        if m.form != "wait":
            continue
        c["wait:n"] += 1
        chained = prev_wait_end is not None and m.start - prev_wait_end < 40
        if chained:
            c["wait:chained"] += 1        # reduplication: "Wait, wait, wait" (within 40 characters of the previous wait)
            if not prev_chained:
                c["wait:chain_runs"] += 1  # a maximal run of >= 2 consecutive waits
        prev_wait_end, prev_chained = m.end, chained
        c["wait:var:" + mk.wait_variant(m)] += 1
        c["wait:nc:" + (m.next_char if m.next_char in ",.:!?" else ("nl" if m.next_char == "\n" else "bare"))] += 1
        if m.lexical:
            c["wait:lex"] += 1
        c["wait:pos:" + mk.position_class(m)] += 1
        if mk.is_connective_initial(m):
            c["wait:si_conn"] += 1
        if mk.is_paragraph_initial(text, m, sents):
            c["wait:par_init"] += 1
        c["wait:orth:" + mk.orth_class(m)] += 1
        if m.prev_word == "but":
            c["wait:but_wait"] += 1
        c["wait:next:" + (m.next_word or "<eos>")] += 1
        e = mk.tokens_to_action(text, m)
        if e is not None:
            c["wait:eros_sum"] += e
            c["wait:eros_n"] += 1
        if m.word_idx == 0:
            c["wait:sentlen_sum"] += m.n_words
            c["wait:sentlen_n"] += 1
        c["wait:ctx_sum"] += mk.clause_length(text, m, sents)
        c["wait:ctx_n"] += 1

    # -- revision events and slots
    stripped = [mk.strip_initial_forms(s.text) for s in sents]
    events = rv.detect_events([st_ for st_, _ in stripped])  # revision events never see sentence-initial forms
    for _, removed in stripped:
        if removed:
            c["strip:" + removed.lower().rstrip(" ,.:;!-")] += 1
    disc_in_sent = [0] * len(sents)
    forms_in_sent: List[Dict[str, int]] = [defaultdict(int) for _ in sents]
    for m in ms:
        forms_in_sent[m.sent_idx][m.form] += 1
        if m.kind == "disc":
            disc_in_sent[m.sent_idx] += 1
    slot_sents = set()
    for i, ev in enumerate(events):
        if ev is None:
            continue
        c["R:n"] += 1
        c["R:type:" + ev] += 1
        if disc_in_sent[i] > 0 or (i > 0 and disc_in_sent[i - 1] > 0):
            c["R:disc"] += 1
        slot_sents.add(i)
        if i > 0:
            slot_sents.add(i - 1)
    for i in slot_sents:
        for f, k in forms_in_sent[i].items():
            c["para:" + f] += k
    # -- marker reliability
    for i in range(len(sents)):
        if disc_in_sent[i] == 0:
            continue
        c["M:n"] += 1
        if any(events[j] is not None for j in range(i, min(len(sents), i + REL_WINDOW + 1))):
            c["M:rel"] += 1
        if any(events[j] is not None for j in range(i + 1, min(len(sents), i + REL_WINDOW + 1))):
            c["M:rel_strict"] += 1
    return dict(c)


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _rare(counts: Sequence[float], n_star: Optional[int], draws: int, rng) -> float:
    """Mean Miller-Madow entropy over `draws` hypergeometric subsamples of size n_star (STATS_PLAN C3)."""
    c = np.asarray(counts, dtype=np.int64)
    if not n_star or c.sum() < n_star:
        return float("nan")
    vals = [entropy_bits(rng.multivariate_hypergeometric(c, int(n_star))) for _ in range(draws)]
    return float(np.mean(vals))


def indices_from_totals(t: Mapping[str, float], rare: Optional[Mapping[str, int]] = None, draws: int = 20,
                        seed: int = 0) -> Dict[str, float]:
    """rare = {"pos": n*, "para": n*, "next": n*}: subsample size (min over checkpoints) for the rarefied entropies."""
    g = lambda k: float(t.get(k, 0.0))  # noqa: E731
    n = g("wait:n")
    out: Dict[str, float] = {}
    out["FREQ_wait_1k"] = 1000.0 * _ratio(n, g("tokens"))
    out["FREQ_wait_100s"] = 100.0 * _ratio(n, g("sents"))
    out["FREQ_disc_1k"] = 1000.0 * _ratio(g("disc_total"), g("tokens"))
    out["SI"] = _ratio(g("wait:pos:initial"), n)
    out["SI_conn"] = _ratio(g("wait:si_conn"), n)
    out["PI"] = _ratio(g("wait:par_init"), n)
    out["ORTH"] = _ratio(g("wait:orth:canonical"), n)
    out["CHAIN"] = _ratio(g("wait:chained"), n)                     # share of waits that continue a chain
    out["CHAIN_RUNS_1k"] = 1000.0 * _ratio(g("wait:chain_runs"), n)   # D28: reduplicated wait runs per 1k waits
    var = [v for k, v in t.items() if k.startswith("wait:var:")]
    out["ORTH_max"] = _ratio(max(var), sum(var)) if var else float("nan")   # D28, exploratory: modal-spelling share
    out["COMMA_DROP"] = 1.0 - _ratio(g("wait:nc:,"), n) if n else float("nan")   # D28: share of wait without a following comma
    out["ORTH_comma"] = _ratio(g("wait:nc:,"), n)
    out["ORTH_period"] = _ratio(g("wait:nc:."), n)
    out["ORTH_bare"] = _ratio(g("wait:nc:bare") + g("wait:nc:nl"), n)
    out["BUT_WAIT"] = _ratio(g("wait:but_wait"), n)
    out["LEX_share"] = _ratio(g("wait:lex"), n)
    out["H_pos"] = entropy_bits([g("wait:pos:" + b) for b in mk.POS_BINS]) if n > 0 else float("nan")
    para_all = [g("para:" + f) for f in mk.FORM_NAMES]
    para_disc = [g("para:" + f) for f in mk.DISC_NAMES]
    out["H_para"] = entropy_bits(para_all) if sum(para_all) > 0 else float("nan")
    out["H_para_disc"] = entropy_bits(para_disc) if sum(para_disc) > 0 else float("nan")
    out["N_eff"] = 2.0 ** out["H_para"] if out["H_para"] == out["H_para"] else float("nan")
    out["SHARE_wait"] = _ratio(g("para:wait"), sum(para_all))
    out["OBL"] = _ratio(g("R:disc"), g("R:n"))
    out["REL"] = _ratio(g("M:rel"), g("M:n"))
    out["REL_strict"] = _ratio(g("M:rel_strict"), g("M:n"))
    nxt = [v for k, v in t.items() if k.startswith("wait:next:")]
    out["H_next"] = entropy_bits(nxt) if sum(nxt) > 0 else float("nan")
    out["EROS"] = _ratio(g("wait:eros_sum"), g("wait:eros_n"))
    out["SENTLEN_init"] = _ratio(g("wait:sentlen_sum"), g("wait:sentlen_n"))
    out["CTX"] = _ratio(g("wait:ctx_sum"), g("wait:ctx_n"))
    if rare:
        rng = np.random.default_rng(seed)
        out["H_pos_rare"] = _rare([g("wait:pos:" + b) for b in mk.POS_BINS], rare.get("pos"), draws, rng)
        out["H_para_rare"] = _rare(para_all, rare.get("para"), draws, rng)
        out["H_next_rare"] = _rare(nxt, rare.get("next"), draws, rng)
    out["R_per_100s"] = 100.0 * _ratio(g("R:n"), g("sents"))
    # raw denominators, reported next to every index so tiny cells are visible
    out["n_wait"] = n
    out["n_R"] = g("R:n")
    out["n_M"] = g("M:n")
    out["n_resp"] = g("n_resp")
    out["tokens_per_resp"] = _ratio(g("tokens"), g("n_resp"))
    return out


def to_matrix(per_problem: Sequence[Mapping[str, float]]) -> Tuple[np.ndarray, List[str]]:
    names = sorted({k for d in per_problem for k in d})
    col = {k: i for i, k in enumerate(names)}
    M = np.zeros((len(per_problem), len(names)))
    for r, d in enumerate(per_problem):
        for k, v in d.items():
            M[r, col[k]] = v
    return M, names


def sum_counts(rows: Iterable[Mapping[str, float]]) -> Dict[str, float]:
    tot: Dict[str, float] = defaultdict(float)
    for r in rows:
        for k, v in r.items():
            tot[k] += v
    return dict(tot)


def bootstrap_indices(per_problem: Sequence[Mapping[str, float]], B: int = 2000, seed: int = 0,
                      rare: Optional[Mapping[str, int]] = None):
    """{index: {est, lo, hi, se}} with a cluster bootstrap over problems."""
    M, names = to_matrix(per_problem)
    return bootstrap_stat(M, names, lambda t: indices_from_totals(t, rare=rare, draws=10, seed=seed), B=B, seed=seed)


def main():
    """CLI: per-checkpoint H1 indices with cluster bootstrap over problems -> JSON (used by the S7 in-job analysis)."""
    import argparse

    from . import common

    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="graded jsonl(.gz) with text, n_tokens, pid, correct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261012)
    a = ap.parse_args()
    rows = common.read_jsonl(a.gen)
    by: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        by[r["pid"]].append(r)
    per = [sum_counts(trace_counts(r["text"], r["n_tokens"]) for r in by[p]) for p in sorted(by)]
    res = bootstrap_indices(per, B=a.B, seed=a.seed)
    tot = sum_counts(per)
    common.write_json(a.out, {"indices": res, "totals": tot, "n_problems": len(per), "n_samples": len(rows),
                              "acc": sum(bool(r.get("correct")) for r in rows) / max(1, len(rows)),
                              "truncated": sum(r.get("finish") == "length" for r in rows) / max(1, len(rows))})
    common.write_meta(a.out + ".meta.json", "indices", {"gen": a.gen, "B": a.B}, {"rng_seed": a.seed}, inputs=[a.gen], outputs=[a.out])
    print("[indices] %d problems, FREQ_wait_1k=%.3f" % (len(per), res["FREQ_wait_1k"]["est"]))


if __name__ == "__main__":
    main()
