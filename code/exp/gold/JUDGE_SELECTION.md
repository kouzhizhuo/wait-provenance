# Judge selection rule (written 2026-09-30 20:23 CST, BEFORE any gpt-oss-20b audit result exists)

Purpose: choose the revision/outcome judge from the human audit **v1 only**, so that audit v2 (E3 outcomes, 40 fresh items) validates the chosen judge without having steered the choice. Human labels are used as given.

Candidates (open weights, no API): the current judge `Qwen2.5-7B-Instruct` (prompt `exp/judge_prompts/revision.md` v1.0, one output token) and `openai/gpt-oss-20b` (different family; same prompt text; harmony chat template, reasoning effort low, greedy decoding, final-channel letter). Escalation candidates only if gpt-oss-20b is clearly better: `Qwen3-32B-FP8` (P2's GPU), `gpt-oss-120b` (Slurm).

Data: the 59 labelled v1 audit items (exp/gold/HUMAN_AUDIT_LABELS.jsonl), the same rendering of the sentence for both judges.

Rule (fixed now):
1. Criterion: Cohen's kappa (5-way) of judge vs human on the 59 items; secondary: binary REVISE kappa and Youden J.
2. Switch away from Qwen2.5-7B-Instruct only if the candidate's 5-way kappa is higher AND a paired bootstrap over items (B = 5000) gives P(kappa_candidate > kappa_Qwen) >= 0.80. Otherwise keep Qwen2.5-7B-Instruct (cheaper, already used for all queued labels).
3. Escalate to a larger judge only if the winner's kappa lifts above 0.40 AND the same rule beats Qwen; else no escalation.
4. Whatever is chosen is fixed for audit v2 and all main-run labels; both judges' v1 numbers are reported either way.
5. No item is re-labelled, dropped or re-weighted after seeing judge outputs.

## Amendment 2026-09-30 20:24 CST — D34 (lead) is BINDING; still before any gpt-oss-20b result exists
- The judge is chosen on audit **v1 only**; audit v2 (E3 outcomes, 40 fresh items) stays a held-out validation. gpt-oss-20b is run on the 60 v1 items only; **v2 items are not touched until the choice is logged in DECISIONS.**
- Report human–judge **binary REVISE kappa, sensitivity, specificity and Youden J** for both judges side by side (5-way kappa as secondary).
- "Clearly better" = non-overlapping 95% kappa CIs, or delta J >= 0.2. Only then run the escalation candidate on v1 (gpt-oss-120b in a free Slurm slot, or Qwen3-32B-FP8 via the P2 coder on hostA). Otherwise Qwen2.5-7B-Instruct stays.
- Rules 2-3 above (paired-bootstrap probability, kappa > 0.40) are superseded by this amendment; the paired bootstrap is still printed for information.
