import pytest

from exp.pipeline import valchange as V


def test_equations_basic():
    assert V.equations("So $x = 3 + 4$.") == [("x", "3+4")]
    assert V.equations("Then 3 + 4 = 7.") == [("3+4", "7")]
    assert V.equations("Thus $x = 3 + 4 = 7$ units.") == [("x", "7")]          # chain -> final value
    assert V.equations("We get x = 3, y = 4.") == [("x", "3"), ("y", "4")]
    assert V.equations("so the area is A = \\frac{1}{2} b h, which is large") == [("A", "\\frac{1}{2}bh")]


def test_equations_none():
    assert V.equations("Let me think about this.") == []
    assert V.equations("Is x <= 3 true?") == []


def test_reference_and_exclusion():
    assert V.reference("First, a = 2. Next, b = a + 3 = 5. ") == ("b", "5")
    assert V.reference("First, a = 2. Now I will think. ") == ("a", "2")      # fall back to step k-1
    assert V.reference("Let me think. Now, more thinking here. ") is None      # excluded


def test_window_flags_changed_and_restated():
    pytest.importorskip("math_verify")
    ref = ("b", "5")
    assert V.window_flags(ref, ["Hmm, b = 6 actually."])["value_change"] == 1
    assert V.window_flags(ref, ["So b = 10/2 holds."])["value_change"] == 0      # equivalent restatement
    assert V.window_flags(ref, ["And c = 7."])["value_change"] == 0              # other LHS
    assert V.window_flags(ref, ["That is not right, b is wrong."])["contradict"] == 1


def test_equal_strings_need_no_mathverify():
    assert V.equivalent("5", "5") is True
