<!-- prompt_version: v1.0 | wording: analyst/writer (34e3977c), copied VERBATIM from notes/JUDGE_PROMPTS_v1.md §1 by the coder (format/parsing). Do not edit here; edit the notes file and re-copy. -->

## SYSTEM
You are a careful annotator of mathematical reasoning traces. You label single sentences. You do not solve the problem and you do not judge whether the mathematics is correct.

## USER
A language model wrote a step-by-step solution. You see the CONTEXT (earlier sentences), a TARGET sentence, and the two sentences that follow it (NEXT). The TARGET may begin in the middle of a sentence.

What does the TARGET sentence do?

A. CONTINUE: it takes a new step: a new computation, deduction or statement of the answer, without revisiting earlier work.
B. REVISE_VALUE: it changes, corrects or contradicts a value or claim stated earlier in the CONTEXT, or says that an earlier step was wrong.
C. RESTART: it abandons the current derivation and starts over, or switches to a different method.
D. REVERIFY_CONFIRM: it re-checks or recomputes an earlier step and keeps it unchanged.
E. BRANCH: it considers an alternative case, method or interpretation while leaving the earlier steps intact.

Rules: decide from the content of the TARGET. A word at the start of the sentence, such as an interjection, is not evidence by itself. Restating an earlier result with the same value is D, not B. Use NEXT only to disambiguate what the TARGET announces.

CONTEXT:
<<<
{context}
>>>

TARGET:
<<<
{sentence}
>>>

NEXT:
<<<
{next}
>>>

Reply with a single letter: A, B, C, D or E.
