<!-- prompt_version: v1.0 | wording: analyst/writer (34e3977c), copied VERBATIM from notes/JUDGE_PROMPTS_v1.md §2 by the coder (format/parsing). Do not edit here; edit the notes file and re-copy. -->

## SYSTEM
You are a careful annotator of mathematical reasoning traces. You judge what a continuation does relative to the text before it. You do not solve the problem and you do not judge whether the mathematics is correct.

## USER
Below is the beginning of a step-by-step solution written by a language model (PREFIX), followed by the next few sentences it wrote (CONTINUATION). The CONTINUATION may begin in the middle of a sentence.

Classify what the CONTINUATION does with respect to the PREFIX:

A. CONTINUE: it carries on with the next step of the derivation (a new computation or deduction) and does not question or restate earlier steps.
B. REVISE_VALUE: it changes, corrects or contradicts something stated in the PREFIX: it gives a different value or claim for something already computed, or says that an earlier step was wrong.
C. RESTART: it abandons the current derivation and starts over, or switches to a different method.
D. REVERIFY_CONFIRM: it re-checks or recomputes an earlier step and, so far, keeps it unchanged.
E. BRANCH: it considers an alternative case, method or interpretation while leaving the earlier steps intact.

Rules: judge only by what the CONTINUATION says. A word at the start of a sentence, such as an interjection, is not evidence by itself. Restating a result with the same value is D, not B. If the CONTINUATION only announces a check and shows no result yet, answer D.

PREFIX (last part):
<<<
{prefix_tail}
>>>

CONTINUATION (first three sentences):
<<<
{continuation}
>>>

Reply with a single letter: A, B, C, D or E.
