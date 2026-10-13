import math

import numpy as np
import pytest

from exp.pipeline import indices as ix
from exp.pipeline import stats as st


def totals(*traces):
    return ix.sum_counts(ix.trace_counts(t, n) for t, n in traces)


def test_freq_per_1k_and_per_100_sentences():
    t = "One. Two. Wait, three. Four. Wait, five. Six. Seven. Eight. Nine. Ten."  # 10 sentences, 2 waits
    idx = ix.indices_from_totals(totals((t, 500)))
    assert idx["FREQ_wait_1k"] == pytest.approx(4.0)       # 2 / 500 * 1000
    assert idx["FREQ_wait_100s"] == pytest.approx(20.0)    # 2 / 10 * 100


def test_position_orthography_and_entropy():
    # 4 waits: initial, initial, early ("But wait"), medial -> counts (2,1,1,0)
    t = "Wait, a. Wait, b. But wait, c. We see that we wait now."
    idx = ix.indices_from_totals(totals((t, 100)))
    assert idx["SI"] == pytest.approx(0.5)
    assert idx["SI_conn"] == pytest.approx(0.75)
    assert idx["ORTH"] == pytest.approx(0.5)  # only the two sentence-initial "Wait," are canonical
    h_plugin = -(0.5 * math.log2(0.5) + 2 * 0.25 * math.log2(0.25))
    assert st.entropy_bits([2, 1, 1, 0], miller_madow=False) == pytest.approx(h_plugin)
    assert idx["H_pos"] == pytest.approx(h_plugin + 2 / (2 * 4 * math.log(2)))  # Miller-Madow, K-1 = 2


def test_paradigm_entropy_uniform_over_two_forms():
    # two revision events (value changes); slots contain one "wait" and one "hold on" -> 1 bit before MM
    t = ("We have $3 + 4 = 7$. Wait, now $3 + 4 = 8$. Fine. Next $5 + 5 = 10$. "
         "Hold on, then $5 + 5 = 11$.")
    tot = totals((t, 200))
    assert tot["R:n"] == 2
    assert tot["para:wait"] == 1 and tot["para:hold_on"] == 1
    idx = ix.indices_from_totals(tot)
    assert idx["OBL"] == pytest.approx(1.0)
    assert idx["SHARE_wait"] == pytest.approx(0.5)
    assert idx["H_para"] == pytest.approx(1.0 + 1 / (2 * 2 * math.log(2)))
    assert idx["N_eff"] == pytest.approx(2 ** idx["H_para"])


def test_obligatoriness_is_below_one_with_unmarked_revision():
    t = ("We have $3 + 4 = 7$. So $3 + 4 = 8$ silently. Then $2 + 2 = 4$. "
         "Wait, so $2 + 2 = 5$.")
    idx = ix.indices_from_totals(totals((t, 100)))
    assert idx["n_R"] == 2
    assert idx["OBL"] == pytest.approx(0.5)


def test_reliability_marker_without_revision():
    # 2 marker sentences; only the first is followed by a revision event within the window
    t = ("We have $3 + 4 = 7$. Wait, check. Now $3 + 4 = 9$. "
         "Filler one. Filler two. Filler three. Hmm, okay. Filler four. Filler five. End.")
    idx = ix.indices_from_totals(totals((t, 100)))
    assert idx["REL"] == pytest.approx(0.5)


def test_h_next_and_lexical_share():
    t = "Wait, let me see. Wait, let me try. Wait, what? I will wait for it."
    idx = ix.indices_from_totals(totals((t, 100)))
    # next words: let, let, what, for -> counts (2,1,1)
    assert idx["H_next"] == pytest.approx(st.entropy_bits([2, 1, 1]))
    assert idx["LEX_share"] == pytest.approx(0.25)


def test_eros():
    t = "Wait, let me recompute 3 + 4 = 7. Wait 5 + 1 = 6."
    idx = ix.indices_from_totals(totals((t, 100)))
    assert idx["EROS"] == pytest.approx((4 + 0) / 2)  # 4 tokens, then 0 tokens before the digit


def test_no_markers_gives_nan_not_crash():
    idx = ix.indices_from_totals(totals(("Just a plain sentence. Another one.", 20)))
    assert idx["n_wait"] == 0 and math.isnan(idx["SI"]) and math.isnan(idx["H_pos"])


def test_bootstrap_indices_is_deterministic_and_covers_estimate():
    per = [ix.trace_counts("Wait, a. Then b.", 50), ix.trace_counts("Then a. Then b.", 50),
           ix.trace_counts("Wait, x. But wait, y.", 60), ix.trace_counts("Nothing here.", 10)]
    a = ix.bootstrap_indices(per, B=200, seed=1)
    b = ix.bootstrap_indices(per, B=200, seed=1)
    assert a["FREQ_wait_1k"] == b["FREQ_wait_1k"]
    e = a["FREQ_wait_1k"]
    assert e["lo"] <= e["est"] <= e["hi"]
    assert e["est"] == pytest.approx(1000 * 3 / 170)


def test_odds_ratio_and_paired_bootstrap():
    assert st.odds_ratio(30, 100, 10, 100) == pytest.approx((30 / 70) / (10 / 90))
    assert np.isfinite(st.odds_ratio(0, 100, 5, 100))  # Haldane correction
    a = np.array([4, 4, 3, 4, 2, 4, 4, 3, 4, 4])
    b = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    r = st.paired_or_bootstrap(a, b, n_per=4, B=300, seed=0)
    assert r["or"] > 10 and r["lo"] > 1 and r["pA"] == pytest.approx(36 / 40)


def test_dprime_signs():
    r = st.dprime(hits=80, n_signal=100, fas=20, n_noise=100)
    assert r["dprime"] == pytest.approx(2 * 0.8416, abs=0.05)
    assert r["c"] == pytest.approx(0.0, abs=0.05)
    liberal = st.dprime(hits=90, n_signal=100, fas=50, n_noise=100)
    assert liberal["c"] < 0  # more false alarms -> liberal criterion


def test_cohen_kappa():
    assert st.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert st.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_derive_seed_is_stable():
    from exp.pipeline.common import derive_seed, git_blob_hash
    assert derive_seed(0, "a", 1) == derive_seed(0, "a", 1) != derive_seed(0, "a", 2)
    assert git_blob_hash(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"  # `echo hello | git hash-object --stdin`


def test_revision_detection_ignores_stripped_initial_forms_and_logs_them():
    t = "We have $3 + 4 = 7$. Wait, now $3 + 4 = 8$. Fine."
    tot = totals((t, 50))
    assert tot["R:n"] == 1 and tot["strip:wait, now"] == 1  # whole opener is stripped and logged


def test_rarefied_entropy_uses_common_subsample_size():
    t = ("Wait, a. Wait, b. But wait, c. We see that we wait now. " * 5)
    tot = totals((t, 500))
    idx = ix.indices_from_totals(tot, rare={"pos": 6, "para": None, "next": 6}, draws=50)
    assert idx["H_pos_rare"] == idx["H_pos_rare"]  # finite
    assert math.isnan(idx["H_para_rare"])           # no size given -> nan, not a crash
    assert idx["H_pos_rare"] <= math.log2(4) + 0.5


def test_or_firth_finite_under_separation_and_matches_half_correction():
    assert st.or_firth(4, 4, 0, 4) == pytest.approx((4.5 / 0.5) / (0.5 / 4.5))
    assert st.or_firth(3, 10, 3, 10) == pytest.approx(1.0)


def _succ(pw, po, ph, pn, m=40, n_per=4, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda p: rng.binomial(n_per, p, size=m).astype(float)  # noqa: E731
    return {"wait": f(pw), "okay": f(po), "hold_on": f(ph), "hmm": f(ph), "none": f(pn)}


def test_insertion_estimates_recovers_planted_effect_and_amplification():
    succ = {"S1@base": _succ(0.10, 0.08, 0.08, 0.08, seed=1), "S1@1900": _succ(0.60, 0.10, 0.12, 0.08, seed=2)}
    r = st.insertion_estimates(succ, 4, B=300, seed=1)
    assert r["S1@1900:RD_op"]["est"] > 30 and r["S1@1900:RD_op"]["lo"] > 10   # ~50 points planted
    assert r["S1@1900:OR_op"]["lo"] > 2
    assert r["S1@base:RD_op"]["lo"] < 10
    assert r["dRD_op_last_minus_base"]["lo"] > 0
    assert r["S1@1900:RD_syn"]["est"] > 30   # form-specific: synonyms do not fire


def test_crossed_interaction_zero_when_policy_effect_is_prefix_independent():
    late = {"S1@base": _succ(0.10, 0.10, 0.1, 0.1, seed=3), "S1@1900": _succ(0.5, 0.1, 0.1, 0.1, seed=4)}
    base = {"S1@base": _succ(0.10, 0.10, 0.1, 0.1, seed=5), "S1@1900": _succ(0.5, 0.1, 0.1, 0.1, seed=6)}
    r = st.crossed_interaction(late, base, 4, B=300, seed=1)
    assert r["policy_effect_late_src"]["est"] > 20 and r["policy_effect_base_src"]["est"] > 20
    assert r["DiD"]["lo"] < 0 < r["DiD"]["hi"]


def test_hsub_form_distribution_uniform_over_two_forms():
    from exp.pipeline import hsub
    V = 10
    lp = np.full(V, -20.0)
    lp[2] = math.log(0.2)          # "wait" variant token
    lp[3] = math.log(0.2)          # "hold_on" variant token
    lp[4] = math.log(0.1)          # second variant of wait
    variants = {"wait": [2, 4], "hold_on": [3], "hmm": [5]}
    dist, mass = hsub.form_distribution(lp, variants)
    assert dist["wait"] == pytest.approx(0.3 / 0.5, abs=1e-6) and dist["hold_on"] == pytest.approx(0.4, abs=1e-6)
    assert mass == pytest.approx(0.5, abs=1e-6)
    assert hsub.entropy_of({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)


def test_chain_and_orthography_breakdown():
    t = "We get 5. Wait, wait, wait. Then we continue for a while with plenty of filler words in between. Wait. Fine."
    idx = ix.indices_from_totals(totals((t, 100)))
    assert idx["n_wait"] == 4 and idx["CHAIN"] == pytest.approx(2 / 4)   # 2nd and 3rd wait are chained
    assert idx["ORTH_comma"] == pytest.approx(2 / 4) and idx["ORTH_period"] == pytest.approx(2 / 4)


def test_spearman_exact_p_and_no_p_below_six_points():
    assert st.spearman([0, 1, 2, 3], [1, 2, 3, 4]) == (1.0, None)                        # K = 4: no p printed
    rho, p = st.spearman([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    assert rho == pytest.approx(1.0) and p == pytest.approx(2 / 720)                     # exact two-sided permutation p
    assert st.spearman([1, 2, 3, 4, 5, 6], [3, 1, 2, 6, 4, 5])[1] == pytest.approx(0.175, abs=1e-3)


def test_attrition_indices_d28():
    # waits: "Wait," "wait," (chained pair -> 1 run), then "Wait." far away, then bare "Wait" far away
    t = ("We get 5. Wait, wait, so the result differs. " + "Filler words continue for quite a while here. " * 3 +
         "Wait. More filler text follows and follows again and again here. Wait a moment.")
    idx = ix.indices_from_totals(totals((t, 100)))
    assert idx["n_wait"] == 4
    assert idx["CHAIN_RUNS_1k"] == pytest.approx(1000 * 1 / 4)
    assert idx["COMMA_DROP"] == pytest.approx(2 / 4)            # "Wait." and "Wait a" have no comma
    assert idx["ORTH_max"] == pytest.approx(1 / 4)              # four different variants: modal share 1/4


def test_revlab_estimator_recovers_planted_obl_and_rel():
    from exp.pipeline import revlab
    rng = np.random.default_rng(0)
    N1, N0 = 2000.0, 18000.0                       # marker-adjacent vs other sentences in the traces
    p1, p0 = 0.30, 0.02                            # planted P(R) in each stratum -> OBL = 2000*.3 / (2000*.3 + 18000*.02) = .625
    n1 = n0 = 700
    pid = rng.integers(0, 100, n1 + n0)
    stratum = np.array([1] * n1 + [0] * n0)
    isr = np.where(stratum == 1, rng.random(n1 + n0) < p1, rng.random(n1 + n0) < p0).astype(float)
    rel_pid = rng.integers(0, 100, 500)
    rel_any = (rng.random(500) < 0.4).astype(float)
    out = revlab.estimate(pid, stratum, isr, N1, N0, rel_pid, rel_any, rel_any * 0.5, 20000.0, B=200)
    assert out["OBL_j"]["est"] == pytest.approx(0.625, abs=0.06) and out["OBL_j"]["lo"] < 0.625 < out["OBL_j"]["hi"]
    assert out["REL_j"]["est"] == pytest.approx(0.4, abs=0.06)
    assert out["R_per_100s_j"]["est"] == pytest.approx(100 * (2000 * .3 + 18000 * .02) / 20000, abs=0.5)


def test_estimators_work_when_checkpoints_have_different_arm_sets():
    # main design: core arms {wait,next,hold_on,none} at most checkpoints, +{okay,hmm} at base/mid/late/final
    core = lambda seed: {k: v for k, v in _succ(0.5, 0.1, 0.1, 0.1, seed=seed).items() if k in ("wait", "hold_on", "none")}
    def with_next(d, seed):
        d = dict(d); d["next"] = _succ(0.1, 0.1, 0.1, 0.1, seed=seed)["okay"]; return d
    full = lambda seed: with_next(_succ(0.5, 0.1, 0.1, 0.1, seed=seed), seed + 50)
    succ = {"S1@base": with_next({k: v for k, v in _succ(0.1, .1, .1, .1, seed=1).items()}, 60), "S1@100": with_next(core(2), 61), "S1@final": full(3)}
    r = st.insertion_estimates(succ, 4, B=100, seed=1, base="S1@base", last="S1@final")
    assert "S1@100:RD_next" in r and "S1@100:RD_op" not in r and "S1@final:RD_op" in r and "R_amp_RD_next" in r
    late = {p: succ[p] for p in ("S1@base", "S1@final")}
    x = st.crossed_interaction(late, late, 4, B=50, seed=1, base="S1@base", last="S1@final")
    assert x["reference_arm_is_next"]["est"] == 1.0 and "DiD" in x


def test_baseline_keyword_measures_are_orthography_sensitive():
    from exp.pipeline import baselines as bl
    txt = "Wait, this is wrong. But wait, let me recheck. However, we retry and check it. wait, ok"
    c = dict(zip(bl.NAMES, bl.response_counts(txt, 100)))
    assert c["r1"] == 8 and c["r1wait"] == 3                      # Wait, wrong, But, wait, However, check + wait ... (whole words, case-insensitive)
    assert c["orz"] == 3 and c["orz_refl"] == 1.0                 # case-sensitive: only the final "wait," , "recheck," (space) and "retry"; "Wait,"/"However," are missed
    assert c["simplerl"] == 6                                     # wait x3 (substring), recheck, however, retry
