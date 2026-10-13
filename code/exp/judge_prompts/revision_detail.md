<!-- prompt_version: v1.0 | wording: analyst/writer (34e3977c), copied VERBATIM from notes/JUDGE_PROMPTS_v1.md §3 by the coder (format/parsing). Do not edit here; edit the notes file and re-copy. -->

## SYSTEM
You extract information from mathematical reasoning traces. You do not solve the problem.

## USER
The CONTINUATION below changes something that was stated in the numbered STEPS. Identify the change.

STEPS:
<<<
{prefix_numbered}
>>>

CONTINUATION:
<<<
{continuation}
>>>

Answer on one line in exactly this form:
STEP=<number of the step whose value or claim is changed> OLD=<the value or claim as written in that step> NEW=<the value or claim the continuation gives instead>
If several steps are changed, give the earliest one.
