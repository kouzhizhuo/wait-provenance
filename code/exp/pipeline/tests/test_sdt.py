import random

import numpy as np
import pytest

from exp.pipeline import sdt


def test_corrupt_sentence_changes_the_result_with_a_small_edit():
    rng = random.Random(0)
    for s in ["So $3 + 4 = 7$.", "Thus the total is 128 apples.", "We get $x = 12 \\cdot 5 = 60$."]:
        for seed in range(6):
            c = sdt.corrupt_sentence(s, random.Random(seed))
            assert c is not None and c["sentence"] != s and sdt.edit_distance(s, c["sentence"]) <= 6
    c = sdt.corrupt_sentence("So $3 + 4 = 7$.", random.Random(1))
    assert c["old"] == "7" and c["new"] != "7"                       # the computed RESULT is what gets corrupted
    assert sdt.corrupt_sentence("No numbers here.", rng) is None
    assert sdt.corrupt_sentence("We need theta between 0 and 2 pi.", rng) is None     # incidental numbers are not corrupted
    c = sdt.corrupt_sentence("The total is 128 apples.", random.Random(3))
    assert c["old"] == "128"


def test_perturbations():
    assert sdt._perturb("7", "pm1") == "8" and sdt._perturb("-3", "pm1") == "-4"   # +1 on the magnitude: the value changes
    assert sdt._perturb("128", "swap") == "182" and sdt._perturb("11", "swap") is None
    assert sdt._perturb("5", "sign") == "-5" and sdt._perturb("-5", "sign") == "5"
    assert sdt.edit_distance("kitten", "sitting") == 3


def test_mark_in_window_uses_first_two_sentences_only():
    assert sdt.mark_in_window("Wait, that is wrong. Then more. More.") == (True, True)
    assert sdt.mark_in_window("We add. Then we multiply. Wait, no.") == (False, False)  # third sentence is outside the window
    assert sdt.mark_in_window("Hmm. We add. Then more.") == (True, False)


def test_sdt_recovers_planted_hit_and_false_alarm_rates():
    rng = np.random.default_rng(0)
    rows = []
    for pid in range(150):
        for j in range(4):
            rows.append({"pid": pid, "src": "corrupted", "mark_any": rng.random() < 0.7})
            rows.append({"pid": pid, "src": "clean", "mark_any": rng.random() < 0.2})
    r = sdt.sdt_from_rows(rows, B=200)
    assert r["HR"]["est"] == pytest.approx(0.7, abs=0.05) and r["FA"]["est"] == pytest.approx(0.2, abs=0.05)
    assert r["dprime"]["est"] == pytest.approx(1.37, abs=0.25)      # z(.7) - z(.2) = 0.524 + 0.842
    assert r["HR"]["lo"] < 0.7 < r["HR"]["hi"] and r["n_error"]["est"] == 600
