# Published MATH-500 accuracy of released zero-RL models (for the D47a reproduction gate)

P1 support `3fde0795`, 2026-10-04. Every number is copied from the cited table; nothing is estimated or converted.
Sources were read in the arXiv HTML full text, the HF model cards and the GitHub raw files on 2026-10-04.

**Machine-readable file:** `notes/published_acc.json`. It contains **only the model authors' own numbers**;
third-party numbers appear below but not in the JSON.

## 1. SimpleRL-Zoo (Zeng et al., arXiv 2503.18892v3, **Table 1**; identical table in the repo README, branch `v1`)
- **Decoding:**
  - **SimpleRL-Zoo models:** temperature 1.0, top-p 0.95, single run (pass@1), max generation length **16K**
    tokens, with the training prompt template.
  - **Baselines (the bases):** **greedy** decoding.
  - Sources: Table 1 caption "baselines use greedy decoding, and models with SimpleRL-Zoo use temperature=1.0 and
    top-p=0.95"; App. B: "using a temperature of 1.0 and a maximum generation length of 16K tokens … we adopt the
    same prompt template used during training … we report pass@1".
- **Prompts:** simpler prompts were used for Llama-3.1-8B, Mistral-v0.1-7B and Qwen-2.5-0.5B/1.5B (§2.1; examples
  in Fig. 10). The README's example eval uses `--template qwen-boxed` for Qwen-2.5-32B.
- **Model cards:** the hkust-nlp `*-SimpleRL-Zoo` cards report **no** numbers.

| Run id | Model | Base MATH-500 (greedy) | **SimpleRL-Zoo MATH-500** |
|---|---|---|---|
| `sz_q05` | Qwen-2.5-0.5B | 15.8 | **34.4** |
| `sz_q15` | Qwen-2.5-1.5B | 29.6 | **59.0** |
| `sz_q7` | Qwen-2.5-7B | 64.6 | **78.2** |
| `sz_q14` | Qwen-2.5-14B | 65.4 | **80.2** |
| `sz_q32` | Qwen-2.5-32B | 68.6 | **82.4** |
| `sz_qm7` | Qwen-2.5-Math-7B | 63.6 | **80.2** |
| `sz_m7` | Mistral-7B-v0.1 ("Mistral-v0.1-7B") | 4.2 | **15.8** |
| `sz_m24` | Mistral-Small-24B | 43.6 | **70.6** |
| `sz_dsm7` | DeepSeek-Math-7B | 19.4 | **39.6** |
| `sz_l8` | Llama-3.1-8B | 13.6 | **23.0** |

## 2. Other released zero-RL models
| Run id | Model | **Published MATH-500** | Source | Decoding the authors report |
|---|---|---|---|---|
| `s8dapo` | `BytedTsinghua-SIA/DAPO-Qwen-32B` | **not published** by its authors | DAPO paper (arXiv 2503.14476v2) reports **AIME 2024 only** (avg@32, "temperature 1.0 and topp 0.7"). The model card has no MATH-500 number; its usage example sets `temperature=1.0, top_p=0.7, max_tokens=20480` | — |
| `s8orz` | `Open-Reasoner-Zero/Open-Reasoner-Zero-32B` | **92.2** | ORZ paper, arXiv 2503.24290v2, **Table 1** and **Table 4** (App. A) | Accuracy averaged over 16 responses per question (Fig. 1 and Fig. 6 captions; Table 1: "our evaluation metric"). **The eval temperature and maximum length are not stated** in the paper text (training: temperature and top-p 1.0). The card repeats "averaged on 16 responses" |
| `orz7b` | `Open-Reasoner-Zero/Open-Reasoner-Zero-7B` | **81.4** | ORZ paper, **Table 4** (App. A; also lists ORZ-0.5B 31.0, ORZ-1.5B 58.0) | Same as above |
| `oat7` | `sail/Qwen2.5-Math-7B-Oat-Zero` | **80.0** | Dr. GRPO paper (Liu et al.), arXiv 2503.20783v2, **App. B Table 4**. The base Qwen2.5-Math-7B is 50.6 with its template and 69.0 with no template (marked *) | **3k-token generation budget** (App. B). §2.2: "Following common practice, we use greedy decoding and limit the sampling budget to 3000 tokens". The model card example uses `temperature=0, top_p=1, max_tokens=3000` |

**Third-party numbers (published, but not by the model's authors; not in the JSON):**
- **DAPO-Qwen-32B:** **71.8**, measured by the ORZ authors on the released checkpoint "using our evaluation metric"
  (ORZ Table 1, row "DAPO-Qwen-32B*"). The same table's row for DAPO's own paper has "–" for MATH500. If the gate
  needs a reference for `s8dapo`, this is the only published one; the lead decides.
- **ORZ-7B:** 79.2 at a 3k budget and 82.4 at an 8k budget, measured by the Dr. GRPO authors (App. B Table 4).
- **SimpleRL-Zero-7B** (the older SimpleRL, not Zoo): 78.2 in Dr. GRPO App. B Table 4.

## 3. SimpleRL-Zoo eval settings in the released code (`hkust-nlp/simpleRL-reason`)
- **Use branch `v1` (the Zoo release), not `main`.** `main` still carries the January 2025 SimpleRL eval:
  - `eval/sh/eval.sh`: `--max_tokens_per_call 3000 --temperature 0 --n_sampling 1 --top_p 1`.
  - `eval/sh/run.sh`: `--template qwen25-math-cot`.
- **`v1` entry point:** `eval_math_nodes.sh` (repo root).
  - Defaults: `temperature=0.0`, `max_tokens=16000`, `top_p=1`, `n_sampling=1`, benchmarks
    `gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23`.
  - The README's "Evaluate" example overrides to `--template qwen-boxed --temperature 1.0 --top_p 0.95
    --max_tokens 16000 --n_sampling 1 --benchmarks aime24,amc23,math500,olympiadbench,gsm8k,minerva_math`.
- **Evaluator:** `examples/simplelr_math_eval/`.
  - `sh/eval.sh` calls `math_eval.py --prompt_type … --max_tokens_per_call … --temperature … --n_sampling … --top_p … --seed 0`.
    MATH-500 runs as a "regular" benchmark with `n_sampling=1`; aime24/amc23 are "special" (comment:
    `n_sampling=8`) unless temperature is 0.
- **Templates:** `examples/simplelr_math_eval/utils.py` defines them. `qwen-boxed` =
  `<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by
  step, and put your final answer within \boxed{}.<|im_end|>\n<|im_start|>assistant\n`.
  - Other candidates there: `abel` = `Question:\n{input}\nAnswer:\nLet's think step by step.\n`;
    `mistral` = `[INST] {input}[/INST]`.
  - **Which template each of the simple-prompt models used is not stated** in the README or paper text beyond
    "the same prompt template used during training" (Fig. 10). **UNVERIFIED** mapping; check the training data
    files before matching it.

## 4. Added 2026-10-05 (checked against the arXiv HTML table text; all values equal the coder's readings)
| Run id | Model | **Published MATH-500** | Exact source | Authors' decoding |
|---|---|---|---|---|
| `gr3_4` | General-Reasoner-4B (Qwen3-4B-Base) | **80.6** | Ma et al., **arXiv 2505.14652v5, Table 3**, row "General-Reasoner-4B" (base Qwen3-4B-Base 68.2) | Table 3 caption: "Most evaluations are based on **greedy decoding** except for AIME24 and AIME25 (averaged over 32 sampling runs…)"; §4.2: simple-evals framework with GPT-4o checking answer equivalence. **Max eval tokens are not stated** in the text read |
| `gr7` | General-Reasoner-7B (Qwen2.5-7B-Base) | **76.0** | same table, row "General-Reasoner-7B" (base 60.2; SimpleRL-Qwen2.5-7B-Zoo 74.0 under *their* eval) | same |
| `gr14` | General-Reasoner-Qw2.5-14B | **78.6** | same table, row "General-Reasoner-Qw2.5-14B" (base 65.4) | same |
| `gr3_14` | General-Reasoner-Qw3-14B | **83.8** | same table, row "General-Reasoner-Qw3-14B" (base 74.6) | same |
| `os15` | One-Shot RLVR {π₁}, Qwen2.5-Math-1.5B | **74.0** | Wang et al., **arXiv 2504.20571v3, Table 3**, row "{π₁}", Size 1, **Step 1860** (base 36.0) | §3.1: MATH500 at **temperature 0**; AIME24/AMC23 avg@8 at T = 0.6. Eval sets from the Qwen2.5-Math evaluation repo (App. B.3). Training response limit 3,072 tokens. **Caveat:** Table 3 reports "the best model performance on MATH500", i.e. the checkpoint chosen on the test metric (step 1860) |
| `oat15` | Oat-Zero-1.5B (Qwen2.5-Math-1.5B) | **74.2** | Liu et al., **arXiv 2503.20783v2, App. B Table 4**, row "Oat-Zero-1.5B" (base 33.0 with template, 61.8 no-template*) | 3k-token budget (App. B); greedy (§2.2) |
| `orz32` | Open-Reasoner-Zero-32B | **92.2** | = `s8orz` (arXiv 2503.24290v2, Tables 1 and 4) | as `s8orz` |

Note on `gr7`: General-Reasoner's own baseline row for SimpleRL-Qwen2.5-7B-Zoo is **74.0** (greedy, their eval), while
SimpleRL-Zoo's paper reports **78.2** (T = 1.0, 16K). This is a published example of how much the evaluation protocol
alone moves the same checkpoint.
