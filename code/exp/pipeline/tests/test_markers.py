import pytest

from exp.pipeline import markers as mk


def texts(sents):
    return [s.text for s in sents]


def test_split_basic_and_decimals():
    t = "The sum is 3.5. Then we double it. So the answer is 7."
    assert texts(mk.split_sentences(t)) == ["The sum is 3.5.", "Then we double it.", "So the answer is 7."]


def test_split_newlines_and_list_marker():
    t = "Step 1\n\n1. Compute x.\nWait, that is wrong."
    assert texts(mk.split_sentences(t)) == ["Step 1", "1. Compute x.", "Wait, that is wrong."]


def test_split_does_not_cut_inside_math_or_before_lowercase():
    t = "We have $x = 3. y = 4$ so it works, e.g. the case is fine. Next."
    s = texts(mk.split_sentences(t))
    assert s == ["We have $x = 3. y = 4$ so it works, e.g. the case is fine.", "Next."]


def test_offsets_roundtrip():
    t = "First.  Second one!\nThird?"
    for s in mk.split_sentences(t):
        assert t[s.start:s.end] == s.text


def test_find_markers_positions():
    t = "We get 5. Wait, that is wrong. But wait, maybe it is fine. I will wait for it."
    ms = [m for m in mk.find_markers(t) if m.form == "wait"]
    assert [m.word_idx for m in ms] == [0, 1, 2]  # "I will wait": third word
    assert [mk.position_class(m) for m in ms] == ["initial", "early", "early"]
    assert [m.lexical for m in ms] == [False, False, True]  # "wait for" is the dictionary sense
    assert ms[0].next_word == "that"
    assert mk.is_connective_initial(ms[1]) and not mk.is_connective_initial(ms[2])


def test_orthography_classes():
    t = "Wait, no. Wait. wait a moment"
    ms = [m for m in mk.find_markers(t) if m.form == "wait"]
    assert [mk.orth_class(m) for m in ms] == ["canonical", "cap_other", "lower"]


def test_initial_only_forms_need_sentence_start():
    t = "It is small but nice. But that fails. However, this holds."
    f = mk.count_forms(t)
    assert f["but"] == 1 and f["however"] == 1


def test_hold_on_hmm_actually_alternatively_and_content_forms():
    t = "Hold on. Hmmm, actually, let me double-check that. I made a mistake. Alternatively, use x."
    f = mk.count_forms(t)
    assert f["hold_on"] == 1 and f["hmm"] == 1 and f["actually"] == 1
    assert f["recheck"] == 1 and f["mistake"] == 1 and f["alternatively"] == 1


def test_curly_apostrophes():
    assert mk.count_forms("That’s not right.")["not_right"] == 1


def test_forms_inside_math_are_ignored():
    assert mk.count_forms("Let $\\text{wait}$ be a label.")["wait"] == 0


def test_tokens_to_action_and_clause_length():
    t = "Wait, let me recompute 3 + 4 = 7."
    m = [x for x in mk.find_markers(t) if x.form == "wait"][0]
    # tokens between "Wait" and the first digit: "," "let" "me" "recompute" -> 4
    assert mk.tokens_to_action(t, m) == 4
    sents = mk.split_sentences(t)
    assert mk.clause_length(t, m, sents) == 1  # the clause "Wait" alone (cut at the comma)


def test_paragraph_initial():
    t = "We get 5.\nWait, no. So maybe. Wait again"
    sents = mk.split_sentences(t)
    ms = [m for m in mk.find_markers(t, sents) if m.form == "wait"]
    assert [mk.is_paragraph_initial(t, m, sents) for m in ms] == [True, False]


def test_strip_initial_forms_logs_removed_token():
    assert mk.strip_initial_forms("Wait, that is wrong.") == ("That is wrong.", "Wait,")
    assert mk.strip_initial_forms("But wait, maybe it holds.") == ("Maybe it holds.", "But wait,")
    assert mk.strip_initial_forms("Hold on. Let me see") == ("Let me see", "Hold on.")
    assert mk.strip_initial_forms("Okay, so we get 5.")[0] == "So we get 5."   # "so" without a comma keeps its content
    assert mk.strip_initial_forms("Okay, so, we get 5.")[0] == "We get 5."
    assert mk.strip_initial_forms("But x is odd.") == ("But x is odd.", "")     # bare conjunction is content
    assert mk.strip_initial_forms("Now we compute 3 + 4.")[0] == "We compute 3 + 4."
    assert mk.strip_initial_forms("Then, we add.")[0] == "We add."
    assert mk.strip_initial_forms("No solution exists.")[1] == ""
    assert mk.strip_initial_forms("Then we get 5.") == ("Then we get 5.", "")
    assert mk.strip_initial_forms("Wait,") == ("", "Wait,")
    assert mk.strip_initial_forms("Nobody knows.")[1] == ""  # \b guards "no" inside a word
