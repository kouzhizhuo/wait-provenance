# P1 human audit — coordinator quick stats (2026-09-30)
Source: exp/gold/HUMAN_AUDIT_LABELS.jsonl (59/60; H-33 blank; 3 comments), annotator = the user (an author).
The user reported low confidence ("the question is a little tricky").
Vs LLM reference labels (labels_adjudicated.jsonl + labels_analyst_s23.jsonl), matched 59/59:
- 5-way agreement 0.64, Cohen kappa 0.40
- binary revise{revise_value,restart} vs not: agreement 0.88, kappa 0.20 (low prevalence)
- main confusions (human -> ref): reverify_confirm->continue 7; revise_value->continue 4; continue->reverify_confirm 3;
  reverify_confirm->branch 3; revise_value->reverify_confirm 3
The same re-check/continue boundary drove the Opus-vs-Fable disagreements (D27 convention), so the guideline is ambiguous there.
