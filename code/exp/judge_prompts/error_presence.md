<!-- prompt_version: v1.0 | wording: analyst/writer (34e3977c), copied VERBATIM from notes/JUDGE_PROMPTS_v1.md §4 by the coder (format/parsing). Do not edit here; edit the notes file and re-copy. -->

# M1
## SYSTEM
You are monitoring a language model while it solves a math problem step by step.

## USER
The model has written the steps below and has just written the sentence marked LATEST.

STEPS:
<<<
{prefix}
>>>

LATEST:
<<<
{marker_sentence}
>>>

Question: with the LATEST sentence, has the model detected an actual error in its earlier steps? Answer from what the model wrote.

A. Yes, it has detected an actual error.
B. No.

Reply with a single letter: A or B.

# M3
## SYSTEM
You are checking a partial solution to a math problem.

## USER
Below are the first steps of a step-by-step solution.

STEPS:
<<<
{prefix}
>>>

Question: do these steps contain a mathematical error (a wrong value, a sign error, a dropped term or an invalid deduction)?

A. Yes, there is an error.
B. No, the steps are correct so far.

Reply with a single letter: A or B.
