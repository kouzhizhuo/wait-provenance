# P1 artifacts: checkpoints, judges, PRM, datasets (ARR checklist B)

Owner: P1 support `3fde0795` (this one file inside the coder's `exp/` tree). Last updated **2026-10-01 05:15 CST** (hostA clock).
Sources: the HF API queried 2026-09-29 (`/api/models|datasets/<id>/revision/<rev>` → `sha`, `gated`, `cardData.license`;
`/tree/<rev>` → file sizes); on-disk commit ids from `<dir>/.cache/huggingface/download/*.metadata` on hostA.
`ROOT=$ARR_ROOT` (hostA). Downloads use the hf-mirror pattern of `infra/INFRA.md` §3.

## 0. Rules that every entry satisfies
- **Official repos only; no gated models; no unofficial re-uploads or mirrors of gated weights** (D17 + amendment;
  TEAM_RULES §1.8). Every repo below is the publisher's own repo and has HF `gated = False`, `private = False`
  (checked 2026-09-29).
- **Not a P1 asset:** `ROOT/models/Llama-3.2-3B` (an `unsloth/*` mirror of a gated Meta repo) and anything else under
  `ROOT/models/` that is not listed here.
- **S6 decision:** `allenai/Llama-3.1-Tulu-3.1-8B` and its DPO base `allenai/Llama-3.1-Tulu-3-8B-DPO` are official
  allenai repos with `gated = False` (llama3.1 licence), so **S6 stays eligible** under D17. It is **not queued until E1 on S1–S3 is done** (lead,
  2026-09-29; reconfirmed 2026-09-30).
- **Storage caps (coordinator, relayed by the P1 lead 2026-09-29):**
  - P1 on hostA `/work`: **≤ 1.5 TB** (`models/S*` + `runs/P1`). Planned peak ≈ 0.8 TB; sweep checkpoints stay on the
    cluster.
  - `/<CLUSTER>`: **≤ 600 GB** at any time.
  - Keep the base and the last checkpoint of every series permanently. Delete intermediates only after their
    analysis outputs exist (D14), and record each deletion in §7.
- **Licence use (plan §12):** od2961 and max-rl have no licence field, so they are secondary/robustness only and never
  carry a headline number. caiyuchen is apache-2.0 but its index→step mapping is unknown (D9), so it is reported on
  the index axis.

## 1. Status at 2026-09-30 09:15 CST (what is on disk)
**Verification (09:05–09:15), for every row marked "verified":**
- The on-disk commit (`.cache/huggingface/download/*.metadata`) equals the HF sha pinned in §2–§4.
- Total bytes equal the HF tree total after the ignore patterns, **exactly** (0 mismatches over 29 snapshots).
- There are 0 `.incomplete` files.
- `chat_template.jinja` md5s are identical within each series (S1 `33ee21d9`, S2 `294f1704`).

| Item | Local path (`ROOT/…`) | Status | Bytes on disk (excl. `.cache`) |
|---|---|---|---|
| S1 base | `models/S1/base` | complete, verified | 14,605,881,703 |
| S1 steps 100, 200, 400, 500, 700, 1000, 1400, 1900 | `models/S1/<step>` | complete, verified (8 snapshots) | 14,605,766,611 each |
| **S1 final** = HF `main` `dee26a40` (**S1 endpoint, D24**) | `models/S1/final` | complete, verified | 14,605,863,881 |
| S2 DPO (t₀) | `models/S2/dpo` | complete, verified (`.bin` duplicates skipped) | 14,605,901,452 |
| S2 steps 25, 100, 250, 500, 750, 1000, 1375 | `models/S2/<step>` | complete, verified (7 snapshots) | 14,605,768,042 each |
| S3 1.5B steps 100, 300, 600, 900, 1200, 1566 | `models/S3/1.5B/<step>` | complete, verified (6 snapshots; template in `tokenizer_config.json`) | 3,103,348,763 each |
| S3 7B steps 100, 300, 550 | `models/S3/7B/{100,300,550}` | complete, verified (100 needed a retry after an hf-mirror HTTP 429 at 01:38; OK 09:17. 550: 91 GB `global_step500/` skipped) | 15,247,181,238; 15,247,181,238; 15,247,360,184 |
| S3 bases | `models/Qwen2.5-1.5B-Instruct`, `models/Qwen2.5-7B-Instruct` | complete (commits `989aa798`, `a09a3545`) | 3,098,977,543; 15,242,811,366 |
| Judge `Qwen2.5-7B-Instruct` | `models/Qwen2.5-7B-Instruct` | complete | (as above) |
| Judge `gpt-oss-20b` | `models/gpt-oss-20b` | complete, verified (commit `6cee5e81`; the 3 root safetensors + index are byte-exact vs HF). **Pruned 09:10:** `metal/model.bin` (13.75 GB Apple-Metal copy of the weights; vLLM loads the root safetensors via `model.safetensors.index.json`, which maps only `model-0000{0,1,2}-of-00002.safetensors`). No `original/` dir. Infra informed | 13,789,264,674 |
| Judge `Qwen3-8B` | `models/Qwen3-8B` | complete (commit `b968826d`, 5/5 shards) | 16,397,465,362 |
| PRM `Qwen2.5-Math-PRM-7B` | `models/Qwen2.5-Math-PRM-7B` | complete, verified (commit `06107400`) | 15,268,589,070 |
| reasoning-cues base rollouts | `data/reasoning_cues/rollouts-olmo7b` | complete (commit `c782ebb6`; 901 files = HF tree) | 8,810,406,762 |
| reasoning-cues RL rollouts | `data/reasoning_cues/rollouts-olmo7b-rlzero` | complete (commit `5d6af392`; 470 files = HF tree) | 4,866,486,107 |
| S6 Tülu 3.1 | — | **not queued** (lead: after E1 on S1–S3) | — |
| **S1-pre** stage1-step141000, stage1-step707000, stage1-step1413814, stage2-step47684, stage3-step6000 | `models/S1pre/<ref>` (hostA) | **complete, verified** (done 2026-09-30 22:02; commits = HF refs, bytes exact, 3/3 shards, 0 incomplete) | 14,605,778,005 each (stage3-step6000: 14,605,774,601) |
| **S1-pre** stage3-step11921 (= `main`, identical LFS oids) | `models/S1pre/stage3-step11921` → symlink to `../S1/base` | complete (no duplicate download) | — |
| **S8_base** `Qwen/Qwen2.5-32B@1818d358` | cluster `models/scale/S8_base` | **complete, VERIFIED by infra 19:45** (pinned sha, exact bytes, 0 incomplete) | 65,539,410,004 |
| S8_dapo … S10 (12 targets) | cluster `models/scale/<name>` | queued. S8_dapo is on SPACE-WAIT: the fullest OST has 84 GB < the 100 GB guard (infra 19:45) | — |

Queue log (`runs/P1/support/dl_queue.log`): started 22:12, done 02:31, 24 targets, 23 OK (5 of them needed a 2nd–4th
try), 1 FAIL (S3_7B_100 on HTTP 429). It was retried separately via `runs/P1/support/dl_s3_7b_100.sh`: OK 09:17, first try. S1 final was a
separate job (`dl_s1_final.sh`), OK 00:53.

Queue script: `ROOT/runs/P1/support/dl_p1_queue.sh` (jobs in `runs/P1/support/jobs.txt`, per-job logs in
`runs/P1/support/logs/`). Ignore patterns: `*.bin`, `original/*`, `metal/*`, `global_step*/*`, `*.pt`, `*.pth`, `*.gguf`.
Two of these matter:
- `Olmo-3-7B-Think-DPO` ships duplicate `pytorch_model-*.bin` next to its safetensors (29.2 GB → 14.6 GB).
- od2961-7B `@253aa0ef` (step 550) carries a 91 GB DeepSpeed `global_step500/` dir (→ 15.25 GB).

## 2. Checkpoint series (EXPERIMENT_PLAN §4)
Resolved commit = `sha` returned by the HF API for that revision (branch or commit). Sizes are HF weight totals after
the ignore patterns.

### S1 — `allenai/Olmo-3-7B-RL-Zero-Math` (zero-RL from base; primary)
- **Base:** `allenai/Olmo-3-1025-7B@main`, commit `a81bae42db3975be1671e27b9c9a56da1a9f980f`, apache-2.0, 14.61 GB →
  `models/S1/base`.
- **Licence:** apache-2.0 (main card; branch revisions carry no `cardData`). Gated False. Training data
  `allenai/Dolci-RLZero-Math-7B` (odc-by).
- **Template (D2 amendment):** the repo's `chat_template.jinja`, md5 `33ee21d94560`, byte-identical in all 8 steps and `final`
  (checked 2026-09-30).
  Applied with `apply_chat_template([{"role":"user","content":problem}], tokenize=False, add_generation_prompt=True)`.
  - **`models/S1/base` has no `chat_template.jinja`** (base repos don't ship one). Load the template from
    `models/S1/1900/chat_template.jinja` when rendering the base, so base and steps get identical prompts.
  - Rendered text: `Solve the following math problem step by step. The last line of your response should be the
    answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the
    problem.\n\n{problem}\n\nRemember to put your answer on its own line after "Answer:"` (no system prompt, no BOS).
- **Temperature:** T = 1.0 (zero-RL; plan §4). Greedy subset T = 0.

| Step | Revision | Resolved commit | Size | Local | Status |
|---|---|---|---|---|---|
| 0 (base) | `Olmo-3-1025-7B@main` | `a81bae42db3975be1671e27b9c9a56da1a9f980f` | 14.61 GB | `models/S1/base` | complete |
| 100 | `step_100` | `3315e80ceb281ae2e6a20bd09e8594ba52d4f312` | 14.61 GB | `models/S1/100` | complete |
| 200 | `step_200` | `ea88b2fe587e4d98f526b2d74a5c925487c5be97` | 14.61 GB | `models/S1/200` | complete |
| 400 | `step_400` | `a16da265d7d1eeb6340d2fe1b25708cb479af0cc` | 14.61 GB | `models/S1/400` | complete |
| 500 | `step_500` | `e5280adc8e0de0719e336ec50e095ccbd2577ab4` | 14.61 GB | `models/S1/500` | complete |
| 700 | `step_700` | `c23e8ebda0c59b21db2cb9747739998ffbd71430` | 14.61 GB | `models/S1/700` | complete |
| 1000 | `step_1000` | `a00ee5fac82420ca6af9b3bb943f7541be20130e` | 14.61 GB | `models/S1/1000` | complete |
| 1400 | `step_1400` | `eb14838d240ab693e15a08c0e3bfeeb112c92c39` | 14.61 GB | `models/S1/1400` | complete |
| 1900 | `step_1900` | `8182367150cef52ddf00dd5259ea94eaa330918e` | 14.61 GB | `models/S1/1900` | complete (numbered step; deletable after its dumps, D14) |
| **final (endpoint, D24)** | `main` | `dee26a4074b88215cd1326472e1ab14d0259d892` | 14.61 GB | `models/S1/final` | complete; **never deleted** |

**`main` is a distinct, later checkpoint.** The card says "fine-tuning can be done from the final checkpoint (the `main`
revision of this model)". Its safetensors LFS oids differ from `step_1900`'s (`5d43c3f9…` vs `2b87f4d3…` for shard 1);
its `config.json` is identical. The HF branch list gives no step number for `main`.
- **D24 (lead, 2026-09-30):** keep both.
  - **`final` (HF `main`, `dee26a40`) is the S1 endpoint** for all "last checkpoint" statistics and E3 prefixes.
  - `step_1900` stays a numbered step.
  - In plots, `final` is a labelled last point; its step count is **UNVERIFIED** (> 1900).
- **Never delete** `models/S1/base` or `models/S1/final`.

### S1′ — `allenai/Olmo-3.1-7B-RL-Zero-Math` (extension; not queued)
- apache-2.0; gated False.
- Steps 300 / 900 / 1500 / 2100 / 2800 (branches `step_0300` … `step_2800`).
- `step_0300` = `d5cfc7c513368723f38b36e0783aed11f26ff341`; `step_2800` = `966de0160b10f47eddcc08ea8a7b8133da3328ba`
  (main == step_2800). Same template family (present in the branch tree).
- **Not queued**; the lead can ask for it (5 × 14.61 GB).

### S2 — `allenai/Olmo-3-7B-Think` (RL on an SFT+DPO thinker; primary)
- **Base (t₀):** `allenai/Olmo-3-7B-Think-DPO@main` (card `base_model` = `allenai/Olmo-3-7B-Think-SFT`), commit `7b18bf927b430ff06376fdfa5610eb3b1b6a5c38`, apache-2.0,
  bf16 safetensors 14.6 GB (`.bin` duplicates skipped) → `models/S2/dpo`.
- **Licence:** apache-2.0 (main card). Gated False. RL data `allenai/Dolci-Think-RL-7B`.
- **Template:** the repo's `chat_template.jinja`, md5 `294f1704dc15`. It is **byte-identical in Think-DPO main,
  step_0025 and step_1375** (checked via HF raw files 22:25).
  - Default system prompt: `You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is
    November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have
    access to any functions. <functions></functions>`
  - User turn `<|im_start|>user\n{content}<|im_end|>\n`. `add_generation_prompt=True` appends
    `<|im_start|>assistant\n<think>`.
  - Measure markers inside the think block. The md5 was re-checked in all 8 downloaded S2 snapshots (dpo + 7 steps):
    all `294f1704` (2026-09-30).
- **Temperature:** T = 0.6, top-p 0.95; max 16k tokens (plan §4).

| Step | Revision | Resolved commit | Size | Local | Status |
|---|---|---|---|---|---|
| 0 (DPO) | `Olmo-3-7B-Think-DPO@main` | `7b18bf927b430ff06376fdfa5610eb3b1b6a5c38` | 14.6 GB (st only) | `models/S2/dpo` | complete |
| 25 | `step_0025` | `817b9d38d9cf462c90fcf56cad08563abd0054a0` | 14.61 GB | `models/S2/25` | complete |
| 100 | `step_0100` | `0999040b167a42bc782f4806ff45d2885c0ad65f` | 14.61 GB | `models/S2/100` | complete |
| 250 | `step_0250` | `30e2d2e14fb590de4866f5eb3f5e9e92750f53ac` | 14.61 GB | `models/S2/250` | complete |
| 500 | `step_0500` | `9b90e9b32d2428bd9cf88e0ffbe16e21a97ba078` | 14.61 GB | `models/S2/500` | complete |
| 750 | `step_0750` | `d23b8f098ebb035c2d30b92b3524c58e80f258a7` | 14.61 GB | `models/S2/750` | complete |
| 1000 | `step_1000` | `4a3b0ac613cbe5dd25eb2d26b2a324c7e30a04e4` | 14.61 GB | `models/S2/1000` | complete |
| 1375 (last) | `step_1375` | `031240693eb33d302cfa8d2df76af15d2da4b579` | 14.61 GB | `models/S2/1375` | complete |

### S2-ctl — `allenai/Olmo-3-7B-Think-SFT` (SFT trajectory; anti-claim b; not queued)
- apache-2.0; gated False. main = `6ff857587e040d6d523a3d5f3a56e918f5401d66`.
- 43 SFT branches `step1000` … `step43000`. The branch tree has no `chat_template.jinja` at main.
  **VERIFY** before use: SFT-stage checkpoints may need the S2 template file.
- The 5 points are to be chosen by the lead; not queued.

### S3 — `od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-v1` (+ 7B), GRPO from an instruct model (secondary)
- **Provenance:** "The Illusion of Insight" (arXiv 2601.00514). The repo holds several runs in one git history.
  **Pinned to the 2025-09-04…09-10 run** (steps 50…1566; SHAs from `survey/T2_checkpoint_inventory.md` App. B,
  re-resolved via the HF API 2026-09-29).
- **Licence:** none (no `license` field). Gated False. Training data `open-r1/OpenR1-Math-220k` (apache-2.0) per the
  paper.
- **Base:** `Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (apache-2.0) at
  `models/Qwen2.5-1.5B-Instruct` (already on disk; not duplicated).
- **Template:** the Qwen2.5 chat template embedded in `tokenizer_config.json` (present at the pinned SHAs; no
  `chat_template.jinja`), with `{problem} Please reason step by step, and put your final answer within \boxed{}.`
  (plan §4). The od2961 *training* system prompt is in their paper App. A.2 and is not used here; it is
  **UNVERIFIED** whether it matches.
- **Temperature (D23, lead 2026-09-29):** T = 1.0 (primary; TRL training temperature, same as the zero-RL series);
  T = 0.7 comparison run on base and the last checkpoint (comparability with Illusion-of-Insight traces); greedy
  subset T = 0.

| Step | Resolved commit (pinned) | Commit date | Size | Local | Status |
|---|---|---|---|---|---|
| 0 (base) | `Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | — | 3.1 GB | `models/Qwen2.5-1.5B-Instruct` | complete |
| 100 | `2474b88f07be992360e97a083048cce9c2703b88` | 2025-09-04 | 3.1 GB | `models/S3/1.5B/100` | complete |
| 300 | `3de1d4d1d646f2b3ad3b1da63a3e917c2f0df4c8` | 2025-09-05 | 3.1 GB | `models/S3/1.5B/300` | complete |
| 600 | `3fa19912af87d2f881632b856b2fd2afc35277b6` | 2025-09-06 | 3.1 GB | `models/S3/1.5B/600` | complete |
| 900 | `65e77bba6d26d450c3708f4a09cd2d04c9a910e8` | 2025-09-07 | 3.1 GB | `models/S3/1.5B/900` | complete |
| 1200 | `24e72064f9c1b17a349d0c48d0a94b8e835b37e1` | 2025-09-09 | 3.1 GB | `models/S3/1.5B/1200` | complete |
| 1566 (last) | `cd442e7b653fc74afe50ccc9645e163b5eb946db` | 2025-09-10 | 3.1 GB | `models/S3/1.5B/1566` | complete |

S3-7B `od2961/Qwen2.5-7B-Open-R1-GRPO-math-7b` (plan §4: base, 100, 300, 550). Base = `Qwen/Qwen2.5-7B-Instruct`
(on disk). Queued last:

| Step | Resolved commit | Commit date | Size (after ignore) | Local | Status |
|---|---|---|---|---|---|
| 100 | `b11276089ca54a18c2b988b78b763a5ad035985d` | 2025-09-05 | 15.25 GB | `models/S3/7B/100` | complete |
| 300 | `bfb2363829e9df4186dc251d3ae34d9e6a45f01a` | 2025-09-10 | 15.25 GB | `models/S3/7B/300` | complete |
| 550 (last) | `253aa0ef1cab8009d0f7aa6cecdfa269f816f7d9` | 2025-10-03 | 15.25 GB (+91 GB `global_step500/` skipped) | `models/S3/7B/550` | complete |

### S4 — `caiyuchen/{DAPO,PPO}-step-k` (zero-RL from `Qwen/Qwen3-8B-Base`; robustness; not queued)
- apache-2.0; gated False. Separate repos per index. Examples: `DAPO-step-0@85f4876c54c939f085fcc7fee383fc7c6b379490`,
  `DAPO-step-27@112487b123f7c87ac4d46d1c8c01c4213810e1b5`, `PPO-step-0@441c5273951865caeb8086502b9850fa6f8f238e`
  (≈16.4 GB each).
- Base `Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4` (apache-2.0).
- Template: Qwen chat template + `{question} Please reason step by step, and put your final answer within boxed{}` (card; D9).
- **Index axis only** (D9). T = 1.0 (zero-RL).

### S5 — `max-rl/{grpo,maxrl,rloo}_qwen3_1p7B_base_polaris_{checkpoints,rollouts}` (appendix curves; not queued)
- Licence: none. Gated False. Base Qwen3-1.7B-Base (per repo name only, D10).
- `grpo_…_checkpoints@453286360d5906282fe1a393d8e4fc7078691dc1` is 347 GB of verl FSDP shards (needs the merger;
  pull individual `global_step_N/` folders only).
- `grpo_…_rollouts@c260e6171f282cf376aba84d6b673b0cd0a993cb` is 27.5 GB (rollouts for H1 indices without generation).

### S6 — `allenai/Llama-3.1-Tulu-3.1-8B` (control: non-reasoner; eligible, not queued)
- Official allenai repo, **gated False**, licence llama3.1. main = `46239c2d07db76b412e1f1b0b4542f65b81fe01f`,
  16.08 GB. 61 branches `step_40` … `step_2440`.
- Base `allenai/Llama-3.1-Tulu-3-8B-DPO@a7beb67e33ffd01cc87ac3b46cadc1000985b8db` (gated False, llama3.1, 32.1 GB incl.
  duplicates).
- Template: the Tülu chat template (tokenizer_config) + the RLVR math instruction (plan §4). Temperature: lead to set.
- **Not queued until E1 on S1–S3 is done** (lead, 2026-09-29; disk/time).

### S1-pre — `allenai/Olmo-3-1025-7B` pretraining branches (control: "was it RL?"; plan §4)
- Same repo as the S1 base: apache-2.0, gated False. There are 1,487 refs: 1,421 `stage1-*`, 52 `stage2-*`,
  13 `stage3-*`, plus `main`.
- **Template / temperature:** as S1 (the RL-Zero-Math `chat_template.jinja` from `models/S1/final`; T = 1.0).
- Picks (lead: 6 points; refs resolved via `/api/models/allenai/Olmo-3-1025-7B/refs`, 2026-09-30):

| Point | Ref | Commit | Local (`ROOT/models/…`) | Status |
|---|---|---|---|---|
| stage1 early (10 % of 1,413,814) | `stage1-step141000` | `e4c463323a…` | `S1pre/stage1-step141000` | complete |
| stage1 mid (50 %) | `stage1-step707000` | `d1f1ebe50f…` | `S1pre/stage1-step707000` | complete |
| stage1 last | `stage1-step1413814` | `373bad2500…` | `S1pre/stage1-step1413814` | complete |
| stage2 last | `stage2-step47684`: the plain branch. The three suffixed `step47684-mix-*-from-2T-ckpt` branches are data-mix ablations, and none equals `main` | `c3c800dc90…` | `S1pre/stage2-step47684` | complete |
| stage3 mid (6,000 of 11,921) | `stage3-step6000` | `615fc46713…` | `S1pre/stage3-step6000` | complete |
| stage3 last | `stage3-step11921` (**weights = `main` = `models/S1/base`**, commit `4b32e100e6…`) | — | symlink → `S1/base` | complete |

### S8 — 32B same-base endpoint panel (cluster; plan §4b)
All official, gated False (HF API 2026-09-30). All bf16 Qwen2 32B; ~65.5 GB after the ignore patterns. Templates live in
`tokenizer_config.json` (no `.jinja`).

| Name | Repo @ pinned sha | Licence | Bytes (after ignore) | Temperature (lead / card) | Prompt convention |
|---|---|---|---|---|---|
| `S8_base` | `Qwen/Qwen2.5-32B@1818d35814b8319459f4bd55ed1ac8709630f003` | apache-2.0 | 65,539,410,004 | T = 1.0 | as S3/S4: Qwen chat template + `{problem} Please reason step by step, and put your final answer within \boxed{}.` |
| `S8_dapo` | `BytedTsinghua-SIA/DAPO-Qwen-32B@ad0c933346957f0fa7f05ba3db409567a069f4f7` | apache-2.0 | 65,543,790,772 | T = 1.0 (card example: T = 1.0, **top-p 0.7**, max 20,480) | card: `apply_chat_template([{"role":"user","content":question}], add_generation_prompt=True)`, where `question` is the DAPO-Math-17k prompt (its instruction wording is **UNVERIFIED**; check a dataset row before use) |
| `S8_qwq` | `Qwen/QwQ-32B@976055f8c83f394f35dbd3ab09a285a984907bd0` | apache-2.0 | 65,539,736,672 | T = 0.6, top-p 0.95 | Qwen chat template (card quickstart); \boxed instruction as S8_base. The card requires YaRN for prompts > 8,192 tokens |
| `S8_orz` | `Open-Reasoner-Zero/Open-Reasoner-Zero-32B@e61e9a8ec0c97baa61ba0471b36786f78949eee1` | mit | 65,543,537,036 (**policy only**; see the note below) | T = 1.0 | **ORZ template, embedded in `tokenizer_config.json`** (card: "we have put the chat template jinja in the tokenizer"). Default system: `A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. ` then `User: {content}\n`; the generation prompt is `Assistant: <think>` |
| `S8_r1distill` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B@711ad2ea6aa40cfca18895e8aca02ab92df1a746` | mit | 65,535,737,017 | T = 0.6, top-p 0.95 (card) | card: no system prompt; put `Please reason step by step, and put your final answer within \boxed{}.` in the user turn |

**ORZ note:** the repo root holds **two** shard sets. `model.safetensors.index.json` maps only `model-0000{1..5}-of-00005.safetensors`
(65.5 GB, the policy). A second set, `model-000NN-of-00014.safetensors` (≈ 65.5 GB), is not referenced by the index.
It is probably the critic (card: "critic models can be loaded … like in the training code"; **UNVERIFIED**). The queue skips it
with the extra ignore pattern `*-of-00014.safetensors`.

### S9 — 32B RL trajectory `allenai/Olmo-3.1-32B-Think` (cluster; plan §4b)
- apache-2.0, gated False. 47 refs (main + `step_0050` … `step_2300`, every 50). All six requested branches exist.
- bf16, 14 shards, 64.48 GB each. Base (t₀) = `allenai/Olmo-3-32B-Think-DPO@97604023f2602a7dfe6d31b1fc229985e0002022`
  (64,477,026,002 B after skipping the 64.5 GB duplicate `.bin` weights).
- **Template:** `chat_template.jinja` md5 `bbbf8428`, the same in `step_0100` and `step_2300`.
  - **The DPO base template differs** (md5 `54d678b2`), unlike 7B S2, where DPO and Think matched. The *only*
    difference is the default system prompt: DPO `You are a helpful AI assistant.` vs Think `You are Olmo, a helpful AI
    assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at
    https://huggingface.co/allenai.`
  - **Recommendation:** render all S9 points, t₀ included, with the Think template (copy it from the step_2300
    snapshot) so that t₀ and the steps get identical prompts. **Lead to confirm.**
- **Temperature:** T = 0.6, top-p 0.95 (as S2).

| Name | Revision | Pinned sha | Bytes |
|---|---|---|---|
| `S9_dpo` | `Olmo-3-32B-Think-DPO@main` | `97604023f2602a7dfe6d31b1fc229985e0002022` | 64,477,026,002 |
| `S9_100` | `step_0100` | `e7b80281ff56aa55231daae1d2d69f0b06f45492` | 64,476,965,402 |
| `S9_300` | `step_0300` | `b0301624fc01ef96aad699b5acd19b5f51fcb430` | 64,476,963,593 |
| `S9_700` | `step_0700` | `09de5c25fc6a77dfb140743fc7612aa47578a077` | 64,476,963,593 |
| `S9_1200` | `step_1200` | `9ca68c2080d077fa25a343a8d3bbb48cc103a724` | 64,476,965,402 |
| `S9_1800` | `step_1800` | `cc238b7791aadf09e483829fa34b343dfd0aec69` | 64,476,965,402 |
| `S9_2300` (last) | `step_2300` | `29884259e8c87406883e817e0628fa5ce9b529c3` | 64,476,965,402 |

### S10 — `openai/gpt-oss-120b` (frontier-scale open reasoner; endpoint only; cluster)
- `@b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, apache-2.0, gated False. mxfp4, 15 root shards, 65,276,859,410 B after
  ignore (`original/` and `metal/`, 65.2 GB each, are skipped). Template: `chat_template.jinja` (harmony format).
- Reasoning effort is set in the system prompt ("Reasoning: high" per the card) at {low, medium, high}. The card gives
  no sampling temperature in the lines grepped; **lead to set** (plan §4b).
- vLLM 0.28 registers `GptOssForCausalLM` (checked on hostA for the 20B).

### Scale-track download spec (all S8–S10)
- **Script:** `ROOT/runs/P1/support/dl_scale_queue.sh` (the copy on hostA; infra places it on the cluster).
  - `next` downloads + verifies the next pending job and exits. `all` loops. `verify <name>` re-verifies.
  - Destination `/<CLUSTER>/<USER>/arr_oct/models/scale/<name>`; log `…/runs/P1/support/dl_scale.log`.
  - Direct huggingface.co (no mirror). Pinned shas. Same ignore patterns plus the ORZ extra.
- **D30: one model live at a time.** A job waits until no other model directory exists under `models/scale/`.
- **Verification:** VERIFIED requires the on-disk commit to equal the pinned sha, total bytes to equal the table
  above exactly, and 0 `.incomplete` files.
- **Order (slot plan):** S8_base → S8_dapo → S8_qwq → S8_orz → S8_r1distill → S9_dpo → S9_100 … S9_2300 → S10.
- Every deletion is recorded in §7 (infra tells support).

### S7 — our GRPO sweep (C5; owner `b8a856c2`, `exp/rl/`)
- Bases `Qwen/Qwen2.5-Math-1.5B@4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2` (apache-2.0) and conditionally
  `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539` (apache-2.0). Both are on hostA under `models/`.
- Checkpoints stay on the cluster (`infra/slurm_jobs.md`).

### Borrowing controls (final only; not queued)
| Model | Commit | Licence | Gated | Size | Base | On hostA |
|---|---|---|---|---|---|---|
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `916b56a44061fd5cd7d6a8fb632557ed4f724f60` | mit | False | 15.24 GB | `Qwen/Qwen2.5-Math-7B` (card text; no `base_model` field — **UNVERIFIED** via API) | yes (`models/DeepSeek-R1-Distill-Qwen-7B`, 15 GB; completeness not re-checked) |
| `simplescaling/s1.1-7B` | `ff0270ab653ee96811d034acba1a880e48612cc1` | none in card | False | 30.48 GB (7 shards; probably fp32, **UNVERIFIED**) | `Qwen/Qwen2.5-7B-Instruct` (card `base_model`) | no |

## 3. Judges and PRM (plan §2.3, §6; D3, D6)
All are local and open-weight; there are **no API models** (D6 amendment 2). Judges follow `notes/JUDGE_PROMPTS_v1.md`:
`## SYSTEM` / `## USER` format, one capital letter as the answer, T = 0, `max_tokens = 1`, and the letter's
log-probabilities over {A..E} give the confidence.

| Role | Repo | Commit | Licence | Gated | Size | Local | Status | vLLM 0.28 (env `gen`) |
|---|---|---|---|---|---|---|---|---|
| Default literal monitor + verifier (GPU 7) | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | apache-2.0 | False | 15.24 GB | `models/Qwen2.5-7B-Instruct` | **complete** | yes (`Qwen2ForCausalLM`) |
| Second, stronger literal monitor + 5 % audit judge (low reasoning effort) | `openai/gpt-oss-20b` | `6cee5e81ee83917806bbde320786a8fb61efebee` | apache-2.0 | False | 13.8 GB root weights (`original/`, `metal/` skipped) | `models/gpt-oss-20b` | **complete** (`metal/` pruned 09:10, §1) | **yes**: `GptOssForCausalLM` is in the registry of vLLM 0.28.0 (checked 22:08) |
| Fallback second judge (non-thinking) | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | apache-2.0 | False | 16.40 GB | `models/Qwen3-8B` | **complete** | yes |
| PRM, third vote for step labels (D3) | `Qwen/Qwen2.5-Math-PRM-7B` | `0610740060112df12585d00a1c5f4624d2f59051` | **other** (Qwen licence) | False | 15.27 GB | `models/Qwen2.5-Math-PRM-7B` | **complete** | yes: `Qwen2ForProcessRewardModel` registered |

## 4. Datasets
| Dataset | Commit | Licence (HF card) | Gated | Use | Local |
|---|---|---|---|---|---|
| `HuggingFaceH4/MATH-500` | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` | none in card (the licence of the parent MATH set must be checked before the checklist) | False | E1–E3 prompts | `data/` (infra) |
| `math-ai/aime24`, `math-ai/aime25` | `83a7f387…`, `563bb840…` | apache-2.0 | False | prompts | `data/aime24`, `data/aime25` |
| `math-ai/amc23` | `80815d37005feb82cd7f8fbc6901d5d3eff43057` | none in card | False | prompts | `data/amc23` |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | mit | False | easy regime | `data/gsm8k` |
| `allenai/Dolci-RLZero-Math-7B` | `93b5e498577804a93fb11bfe8821428f8535f2d8` | odc-by | False | S1 training data (provenance only) | — |
| `allenai/Dolci-Think-RL-7B` | `0fb6466d31ef3a9dd16985ef635e6429e05a6491` | none in card | False | S2 training data (provenance only) | — |
| `open-r1/OpenR1-Math-220k` | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | apache-2.0 | False | S3 training data (provenance only) | — |
| `reasoning-cues/rollouts-olmo7b` | `c782ebb6970995d1c5149080c6e06717f0a3a71b` | **cc-by-4.0** | False | E1 endpoint cross-check (base) | `data/reasoning_cues/rollouts-olmo7b` (complete; 8.81 GB) |
| `reasoning-cues/rollouts-olmo7b-rlzero` | `5d6af39277caf7aa92bcddfc388c8bdf7e5e1a7d` | **cc-by-4.0** | False | E1 endpoint cross-check (RL-Zero-Math final) | `data/reasoning_cues/rollouts-olmo7b-rlzero` (complete; 4.87 GB) |

## 5. reasoning-cues cross-check: what exists, prompt and grader (HF cards + tree + summary JSONs, 22:25)
**What their rollouts contain (MATH-500; checked file-by-file via the HF tree API):**

| Dataset | Model (card) | Prompt style | Arm | Run files (math500) | Summary |
|---|---|---|---|---|---|
| `rollouts-olmo7b` | `allenai/Olmo-3-1025-7B`, snapshot `a81bae42` (**= our `models/S1/base`**) | `rlzero` | `none` | 40 | not published (no `summary/none.json`) |
| | | `rlzero` | `p2_okay`, `sp_okay_bare` (cue arms) | 40 each (+ `run_regraded`) | yes |
| | | `boxed` | `none` | **16 (partial)** | no |
| | | `boxed` | `sp_nn_okay` | 40 | yes |
| | | `minerva4` | `none`, `nn_okay` | 40 each | yes |
| `rollouts-olmo7b-rlzero` | `allenai/Olmo-3-7B-RL-Zero-Math` (summary `tokenizer` field; no revision → **`main` = our `models/S1/final`, not `step_1900`**) | `boxed` **only** | `none` | 40 | yes: 500 × 32, pass@1 0.5276, cap rate 0.107, budget 31,744 |
| | | `boxed` | `sp_nn_okay` | 40 | yes |

**Consequences for the E1 endpoint cross-check (plan §4 "their raw prompt at base and step_1900"):**
1. **Base endpoint: exact cross-check (D24).** Their `rlzero`/`none` base rollouts use the same base commit and
   the documented RL-Zero raw prompt (below). Regenerate our base with that exact string at T = 0.6, top-p 0.95 and
   compare the E1 indices.
2. **RL endpoint: no exact cross-check against step_1900.** Their RL-Zero-Math rollouts exist only under the
   **`boxed`** prompt. That prompt's exact string lives in their private repo (`data/bank`, `sophicle/reason` → 404)
   and is **not documented** in any public card (all 31 reasoning-cues READMEs grepped). They were also generated
   from `main`, which is not `step_1900`.
   - **D24 (lead, 2026-09-30): option (a).** Compare their `boxed` RL rollouts with our generic-boxed anti-claim (c)
     run on `models/S1/final`. Label it "different prompt string, direction/magnitude only", **appendix only**.
   - Their `boxed` base arm is partial (16 of 40 files), so a same-prompt base→RL contrast *within their data* is
     also incomplete. **Skip that arm** (D24).
3. **Files to use (D24).**
   - Base end: `data/reasoning_cues/rollouts-olmo7b/rlzero/math500/run/rollouts_base_none_p*.jsonl`, all 40 files
     (20 + 20 `*_b2`, the two seeded batches). 16,000 rows = 500 × 32.
   - RL end: `data/reasoning_cues/rollouts-olmo7b-rlzero/boxed/math500/run/rollouts_base_none_p*.jsonl`, 40 files,
     16,000 rows = 500 × 32 (counted on hostA, 2026-09-30).
- **Row schema:** `problem_key, rollout_index, arm, steer, label, committed, correct, hit_token_cap, finish_reason,
  output_tokens, max_tokens_used, completion`. **No prompt field** in the rows.
- **Protocol (cards):** 32 rollouts per problem (two seeded batches of 16), T = 0.6, top-p 0.95, budget 31,744
  generated tokens, seed 20260819. Benchmarks: aime2024, aime2025, amc23, gpqa_diamond, gsm8k, humaneval
  (+ `_execgraded`), math500, olympiadbench.
- **RL-Zero raw prompt (their `PROMPT`, documented in the `reasoning-cues/smollm3-rl` README; raw text, no chat
  template, no BOS, ends right after `"Answer:"` with no trailing newline):**
  ```
  Solve the following math problem step by step.
  The last line of your response should be the answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem.
  {q}
  Remember to put your answer on its own line after "Answer:"
  ```
  This differs in whitespace from the repo `chat_template.jinja` we use for S1 primary (§2): single `\n` instead of
  `\n\n` around `{q}`. A forced opener is appended directly, e.g. `PROMPT.format(q=q) + ".\n\nOkay"`. The README
  documents it for SmolLM3; that the OLMo `rlzero/` cells use the same `PROMPT` is stated only as "prompts and grader:
  the paper's repository" (**UNVERIFIED** for OLMo).
- **Grader (card):** `data/bank_rlzero` takes the last `Answer:` line when one exists and otherwise falls back to the
  last `\boxed{}`; `data/bank` reads `\boxed{}`. Stop strings: their training and evals stop on
  `"\nSolve the following math problem"`, while their paper harness stops on `"\nProblem:"`. They report that the
  models commit with `\boxed{}` far more often than with `Answer:`, **so our grader must read both**
  (`pipeline/grade.py` does). Their per-row `committed`/`correct` flags can be re-graded with our grader for
  consistency.
- **Licence:** cc-by-4.0. Cite as unpublished concurrent work ("reasoning registers", S. Wang et al., MIT; see
  `notes/RELATED_reasoning_cues.md`).

## 6. Checks (support)
- [x] All 24 queue jobs + S1 final `OK` (S3_7B_100 after one separate retry, 09:17).
- [x] Shards and bytes re-counted against HF: 29/29 byte-exact. On-disk commit ids = pinned shas (2026-09-30 09:10; S3-7B/100 at 13:52).
- [x] `chat_template.jinja` md5: `33ee21d9` in all S1 steps + `final`; `294f1704` in all S2 snapshots (dpo + 7 steps).
- [x] reasoning-cues: the RL model revision is `main` (summary `tokenizer` field + card; no revision pinned); RL
  rollouts are boxed-only (§5).
- [x] Lead decisions recorded: D23 (S3 temperature), D24 (S1 endpoint = `final`; cross-check option (a)).
- [x] gpt-oss-20b complete; unused `metal/` pruned.

## 7. Deletion log (D14; caps ≤ 1.5 TB hostA `/work`, ≤ 600 GB `/<CLUSTER>`)
The base and last checkpoint of every series are **never deleted**. Intermediates are deleted only after the coder
confirms their E1/E2/E3 generations, hidden-state dumps (marker positions) and teacher-forced `H_sub` numbers exist
under `exp/` / `runs/P1/`.

| Date | Path (`ROOT/…`) | Size freed | Reason | Analysis outputs that exist (paths) | Confirmed by |
|---|---|---|---|---|---|
| 2026-10-01 ~02:45 HKT (infra `b8a856c2`) | cluster `/<CLUSTER>/<USER>/arr_oct/models/S7/ref_s1/` HF exports of steps {5,10,15,30,40,60,80,120,140,160,180,220,240,260,280,320,340,360,380} (19 checkpoints). **Kept:** {20,100,200,300,400} (`model.safetensors` present) | (not reported) | S7 REF s1 intermediates after analysis; `/<CLUSTER>` OST pressure (D30) | `runs/P1/S7/ref_s1/{results.tgz, gen_all_raw.tgz}` on the cluster; Mac copies in `exp/rl/results/ref_s1/` (sha256 `gen_all_raw` f79b2956…9018, `results` 82deba83…7eb6, verified by infra); rollouts `runs/P1/ref_s1/rollouts` (cluster only). Regenerable only by re-running (~11.6 h) | lead + coder `6fa6ac3a` (per infra) |
| 2026-10-01 05:11 HKT (infra `b8a856c2`, auto-prune after a successful analysis, exit 0) | cluster `/<CLUSTER>/<USER>/arr_oct/models/S7/ref_s2/` HF exports of steps {5,10,15,30,40,60,80,120,140,160,180,220,240,260,280,320,340,360,380} (19 checkpoints). **Kept:** {20,100,200,300,400} | (not reported) | S7 REF s2 intermediates after analysis (D30 OST pressure) | `runs/P1/S7/ref_s2/{results.tgz (17.3 MB, all 25 steps incl. graded + greedy gens), gen_all_raw.tgz (33.9 MB)}`; Mac copies in `exp/rl/results/ref_s2/` (sha256 `gen_all_raw` e803bb02…a6c2, `results` edb746cd…7a2c, verified by infra); rollouts `runs/P1/ref_s2/rollouts` (cluster only). Regenerable only by re-running (~14 h) | auto-prune rule (per infra) |

## 8. Disk ledger (P1 on hostA `/work`; cap 1.5 TB)
| Time | `models/S1` | `models/S2` | `models/S3` | `runs/P1` | P1 total (capped items) | Shared P1 judges/PRM/data (not in cap) | `/work` free |
|---|---|---|---|---|---|---|---|
| 2026-09-29 22:05 | 69 GB | — | — | (not measured) | ≈ 0.07 TB | — | 3.1 TB |
| 2026-09-30 09:12 | 146.1 GB (10 snapshots) | 116.8 GB (8) | 53.7 GB (8 complete + 7B/100 partial) | 19.8 GB | **≈ 336 GB (22 % of 1.5 TB)** | PRM 15.3, gpt-oss 13.8, Qwen2.5-7B-Instruct 15.2, Qwen3-8B 16.4, reasoning-cues 13.7 GB | 2.8 TB |
| 2026-09-30 13:52 | 146.1 GB | 116.8 GB | 64.4 GB (all 9 S3 snapshots) | (not re-measured) | ≈ 347 GB + `runs/P1` growth | (as above) | 2.7 TB |

`/<CLUSTER>` (cap 600 GB) is not mounted on hostA, so support cannot measure it from there. It is the Slurm cluster's
filesystem (infra's ledger: `infra/slurm_jobs.md`).

## D43 seeded-marker runs (2026-10-04, coder helper)
- 2026-10-04 09:09 pruned/verified `models/S7/seed_W512` (HF steps {20,100,200,300,400} kept; step 0 = SFT-only ckpt `models/S7seed/W512_sft`, not an S7 step). Mac copies `exp/rl/results/seed_W512/{results.tgz 24,354,657 B, gen_all_raw.tgz 42,506,733 B}` sha256 verified (e9442129…, f0c1e48d…). Contents: idx/s{0,20,100,200,300,400}.json, gen/s*.graded.jsonl.gz (+greedy).
- 2026-10-04 09:11 same for `seed_X512` (`results.tgz 23,404,352 B` a9c649ce…, `gen_all_raw.tgz 44,577,026 B` b4b5b031…).
- Seed data (cluster): `runs/P1/seeded/data/{arms/*.jsonl,pool.jsonl,build_report.json,MD5SUMS}`; SFT-only ckpts `models/S7seed/{W512,X512,D512,W64}_sft` (3 GB each, kept).
- 2026-10-04 14:05 pruned/verified `models/S7/seed_D512` (kept {20,100,200,300,400}); Mac `exp/rl/results/seed_D512/{results.tgz 23,106,327 B, gen_all_raw.tgz 43,438,515 B}` sha256 verified (505e3216…, 7a9604f0…).
- 2026-10-04 14:06 same for `seed_W64` (`results.tgz 17,955,485 B`, `gen_all_raw.tgz 32,543,663 B`); see /tmp prune output / ARTIFACTS line in the prune log. SFT-only ckpts stay in models/S7seed/.
