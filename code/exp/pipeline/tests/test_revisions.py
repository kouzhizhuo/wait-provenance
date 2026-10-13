from exp.pipeline import revisions as rv


def test_parse_value():
    assert rv.parse_value("1,234") == 1234.0
    assert rv.parse_value("3/4") == 0.75
    assert rv.parse_value("\\frac{3}{4}") == 0.75
    assert rv.parse_value("-2.5") == -2.5
    assert rv.parse_value("abc") is None


def test_assignments():
    a = rv.extract_assignments("So $3 + 4 = 7$.")
    assert ("3+4", 7.0) in a
    a = rv.extract_assignments("The sum is 15.")
    assert ("V:sum", 15.0) in a
    # single-letter keys are ignored, rhs expressions are not plain values
    assert rv.extract_assignments("$x = 5$") == []
    assert rv.extract_assignments("$3 + 4 = 7 + 1$") == []


def test_value_change_is_a_revision_and_repeat_is_not():
    s = ["First $3 + 4 = 7$.", "Then $3 + 4 = 7$ again.", "So $3 + 4 = 8$."]
    assert rv.detect_events(s) == [None, None, "value"]


def test_negation_and_restart_are_content_based():
    s = ["We get a result.", "That was a mistake.", "Let us start over."]
    assert rv.detect_events(s) == [None, "negation", "restart"]


def test_discourse_markers_alone_are_not_events():
    # this independence is what keeps OBL from being 1 by construction
    s = ["Wait, let me think.", "Hmm, okay.", "Actually, fine."]
    assert rv.detect_events(s) == [None, None, None]


def test_verbal_value_change():
    s = ["The sum is 10.", "The sum is 12."]
    assert rv.detect_events(s) == [None, "value"]


def test_regression_case_analysis_is_not_a_revision():
    # different cases of n, LaTeX delimiters, capitalised "If": must not collide on a key
    s = [r"- If \(n = 1\), then \(2 \times 1 = 2\) (not divisible by 3)",
         r"- If \(n = 2\), then \(2 \times 2 = 4\) (not divisible by 3)"]
    assert rv.detect_events(s) == [None, None]


def test_trial_and_error_is_not_a_negation():
    assert rv.detect_events(["By trial and error we find a root."]) == [None]
    assert rv.detect_events(["I made an error in the setup."]) == ["negation"]
