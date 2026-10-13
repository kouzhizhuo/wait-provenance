import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytest
from seeded_arms import make_arms, find_runs, verify_arms, arm_stats

T = "First, 2+3=5. Wait, let me re-check the sum. It is 5.\n\nWait, we need $x=3$. Wait, that is odd. Hence \\boxed{5}."


def ok(t):
    a, why = make_arms(t)
    assert a is not None, why
    return a


def test_arms_basic():
    a = ok(T)
    assert a["n_openers"] == 3
    assert a["X"] == T.replace("Wait,", "Next,")
    assert a["D"] == "First, 2+3=5. let me re-check the sum. It is 5.\n\nwe need $x=3$. that is odd. Hence \\boxed{5}."
    verify_arms(a["W"], a["X"], a["D"])


def test_chain_is_one_opener():            # the failing case on the cluster data: "Wait, wait, ..." (lead ruling 1)
    t = "We get 4. Wait, wait, wait, that is odd. Fine. Wait. Wait, no, 4 is right. Done."
    runs, unc = find_runs(t)
    assert [r["n_elem"] for r in runs] == [3, 2] and unc == 0
    a = ok(t)
    assert a["n_openers"] == 2
    assert a["X"] == "We get 4. Next, that is odd. Fine. Next, no, 4 is right. Done."
    assert a["D"] == "We get 4. that is odd. Fine. no, 4 is right. Done."
    assert verify_arms(a["W"], a["X"], a["D"]) == {"n_openers": 2}


def test_non_initial_wait_drops_trace():  # lead ruling 2
    for t in ["Wait, ok. But wait, is it? Yes.", "Wait, ok. We wait for the sum.", "Wait, ok. Hmm, wait, no.", "Wait, ok. Then Wait."]:
        a, why = make_arms(t)
        assert a is None and why == "non_initial_wait", (t, why)


def test_ineligible_reasons():
    assert make_arms("No marker here. Waiting is not a wait word, nor await.")[1] == "no_wait_opener"
    assert make_arms("Fine. Wait,\nnext line")[0] is None                   # 'Wait' before a newline cannot be edited cleanly
    assert make_arms("Wait, ok. Fine. Wait,\nnext line")[1] == "non_initial_wait"
    assert make_arms("Fine. Wait.\nNext line. Wait, ok.")[1] == "non_initial_wait"
    assert make_arms("Fine. Wait. Then more. Wait, ok.")[1] == "run_without_comma"   # a lone "Wait." run has no exact "Wait," form
    assert make_arms("Fine. Wait. Wait, ok.")[0] is not None                 # but "Wait. Wait," is one run
    assert make_arms("Wait,")[1] == "no_wait_opener"


def test_start_of_text_and_after_newline_and_math():
    t = "Wait, start.\nWait, line. $a. Wait, b$. Wait, c."
    runs, unc = find_runs(t)
    assert len(runs) == 3 and runs[0]["a"] == 0 and unc == 1   # the 'Wait, b' inside $...$ is a non-initial wait -> trace dropped
    assert make_arms(t)[1] == "non_initial_wait"
    t2 = "Wait, start.\nWait, line. $a$. Wait, c."
    a = ok(t2)
    verify_arms(a["W"], a["X"], a["D"])
    assert a["D"] == "start.\nline. $a$. c."


def test_x_and_d_have_no_wait():
    a = ok(T)
    for arm in "XD":
        assert arm_stats([a[arm]])["wait_word_any"] == 0
    assert arm_stats([a["W"]])["wait_word_any"] == 3
    assert arm_stats([a["X"]])["init_next_comma"] == 3


def test_verify_catches_extra_edit():
    a = ok(T)
    with pytest.raises(AssertionError):
        verify_arms(a["W"], a["X"].replace("sum", "total"), a["D"])
    with pytest.raises(AssertionError):
        verify_arms(a["W"], a["X"], a["D"].replace("It is", "It's"))
    with pytest.raises(AssertionError):
        verify_arms(a["W"], a["X"], a["D"].replace("let", "Let"))       # re-capitalising the next word is a second edit
    with pytest.raises(AssertionError):                                  # a residual wait word in D
        verify_arms(a["W"], a["X"], a["D"].replace("that is odd", "wait that is odd"))
    with pytest.raises(AssertionError):                                  # D keeping half of a chain
        c = ok("Ok. Wait, wait, stop.")
        verify_arms(c["W"], c["X"], "Ok. wait, stop.")


def test_build_fallback_and_fail(tmp_path):
    pytest.importorskip("math_verify")
    import argparse, json
    from seeded_arms import build
    rows = []
    for i in range(300):
        rows.append({"pid": i, "sid": 0, "finish": "stop", "gold": "7", "prompt_messages": [{"role": "user", "content": "q%d" % i}],
                     "text": "We compute. Wait, let me check. So the answer is \\boxed{7}."})
    g = tmp_path / "raw.jsonl"; g.write_text("\n".join(json.dumps(r) for r in rows))
    def args(**kw):
        d = dict(gen=[str(g)], out_dir=str(tmp_path / "o"), n_high=512, n_low=64, fallback_min=0, seed=0, tokenizer=None,
                 max_resp_tokens=3072, max_prompt_tokens=1024, reward_check=False)
        d.update(kw); return argparse.Namespace(**d)
    with pytest.raises(SystemExit) as e:
        build(args())
    assert "pool = 300" in str(e.value)
    rep = build(args(fallback_min=256))
    assert rep["fallback"]["n_high_used"] == 256 and sorted(rep["files"]) and (tmp_path / "o" / "arms" / "W256.jsonl").exists()
    assert (tmp_path / "o" / "MD5SUMS").exists() and (tmp_path / "o" / "arms" / "D64.jsonl").exists()
