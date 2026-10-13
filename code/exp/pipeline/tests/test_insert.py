from exp.pipeline import insert as ins
from exp.pipeline import grade

TRACE = ("We need the sum of the first ten integers. The formula is n(n+1)/2. Plug in n = 10. "
         "That gives $10 \\cdot 11 / 2 = 55$. So the total is 55. Let us also check the parity of it. "
         "The number 55 is odd. Therefore the result is odd. Now we state the final result. "
         "The sum of the first ten integers is 55.\n\nAnswer: 55")


def test_candidate_cuts_are_in_the_middle_and_before_answer():
    ks = ins.candidate_cuts(TRACE)
    assert ks, "expected some valid cuts"
    from exp.pipeline import markers as mk
    n = len(mk.split_sentences(TRACE))
    assert all(0.2 <= k / n <= 0.8 for k in ks)


def test_no_cuts_after_a_marker_or_revision():
    t = TRACE.replace("Plug in n = 10.", "Wait, plug in n = 10.")
    from exp.pipeline import markers as mk
    first_bad = min(m.sent_idx for m in mk.find_markers(t))
    ks = ins.candidate_cuts(t, allow_prior_markers=False)      # strict variant: nothing marker-like before k+1
    assert all(k + 1 < first_bad for k in ks)
    # relaxed variant (used for P0): earlier markers are allowed, but the NEXT sentence must not be a marker
    marker_sents = {m.sent_idx for m in mk.find_markers(t)}
    assert all((k + 1) not in marker_sents for k in ins.candidate_cuts(t, allow_prior_markers=True))


def test_arm_suffix_keeps_token_boundaries_natural():
    assert ins.arm_suffix("none", " ") == ""            # ends exactly at the sentence end
    assert ins.arm_suffix("wait", " ") == " Wait,"
    assert ins.arm_suffix("wait", "\n\n") == "\n\nWait,"
    assert ins.arm_suffix("none", "\n\n") == "\n\n"


def test_rule_revise_on_units():
    prefix = "We compute $3 + 4 = 7$. Then we double it."
    w1 = ins.window_units("Wait,", " let me redo this: $3 + 4 = 8$. Then more.")
    assert ins.rule_revise(prefix, w1)[0] is True
    w2 = ins.window_units("Okay,", " we get $14$. Then we finish. Done.")
    assert ins.rule_revise(prefix, w2)[0] is False


def test_window_is_form_plus_generation_split_once_and_form_stripped():
    w = ins.window_units("Wait,", " let me check 3.5 again. It is fine. Next step. Fourth sentence.")
    assert w == ["Let me check 3.5 again.", "It is fine.", "Next step."]   # form stripped, remainder capitalised, 3 sentences
    # empty arm takes the same path and is not altered
    assert ins.window_units("", "So we get 5. Then 6. Then 7. Then 8.") == ["So we get 5.", "Then 6.", "Then 7."]
    # generation that starts on a new line: the bare form is its own (dropped) sentence but still counts as one of three
    assert ins.window_units("Wait,", "\nNow x. Then y. Then z.") == ["Now x.", "Then y."]


def test_extract_answer_and_grading():
    assert grade.extract_answer("blah\nAnswer: $\\frac{1}{2}$") == "\\frac{1}{2}"
    assert grade.extract_answer("so \\boxed{12}.") == "12"
    assert grade.extract_answer("nothing") is None
    assert grade.is_correct("\\dfrac{1}{2}", "\\frac{1}{2}")
    assert not grade.is_correct("3", "4")


def test_judge_prompt_loading_parsing_and_detail_regex():
    from exp.pipeline import judge
    sysm, user, h, ver = judge.load_prompt("insertion_outcome.md")
    assert ver == "v1.0" and "{prefix_tail}" in user and "{continuation}" in user
    assert "Wait" not in user and "Hmm" not in user and "Okay" not in user     # arm words are not named inside the prompt
    s1, u1, _, _ = judge.load_prompt("error_presence.md", "M1")
    s3, u3, _, _ = judge.load_prompt("error_presence.md", "M3")
    assert "{marker_sentence}" in u1 and "{marker_sentence}" not in u3
    assert judge.parse_letter("B") == "B" and judge.parse_letter("The answer is C.") == "C" and judge.parse_letter("zzz") == "?"
    assert judge.parse_detail("STEP=3 OLD=7 NEW=8") == (3, "7", "8")
    assert judge.parse_detail("no format here") is None
    steps, n = judge.numbered_steps("We add. Then we double. Done.")
    assert n == 3 and steps.startswith("[1] We add.") and "[3] Done." in steps


def test_grader_prefers_boxed_over_answer_line():
    assert grade.extract_answer("so \\boxed{12}\nAnswer: 13") == "12"
    assert grade.extract_answer("Answer: 13") == "13"
    from exp.pipeline import prompts
    assert prompts.render("boxed_raw", "What is 1+1?") == "What is 1+1?\nPlease reason step by step, and put your final answer within \\boxed{}."


def test_scale_templates_render_the_model_specific_user_text():
    from exp.pipeline import prompts

    class Tok:   # minimal chat-template stand-in: returns the user text and the kwargs so we can inspect them
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            return "USER:" + msgs[0]["content"] + "|" + ",".join("%s=%s" % kv for kv in sorted(kw.items()))
    t = Tok()
    d = prompts.render("dapo", "1+1?", t)
    assert d.startswith("USER:Solve the following math problem step by step. The last line of your response should be of the form Answer:")
    assert d.endswith('Remember to put your answer on its own line after "Answer:".|')
    assert prompts.render("orz", "1+1?", t) == "USER:1+1?|"
    assert "reasoning_effort=high" in prompts.render("harmony", "1+1?", t, "high") and "boxed" in prompts.render("harmony", "1+1?", t, "low")
    assert prompts.render("harmony", "1+1?", t, "low").endswith("<|channel|>analysis<|message|>")      # chain of thought continues inside the analysis channel
