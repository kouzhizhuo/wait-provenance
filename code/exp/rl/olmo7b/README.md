# OLMo-3-1025-7B controlled GRPO fork (D42 / D43 item 4)

Base `allenai/Olmo-3-1025-7B` (S1 base) + OLMo RL-Zero chat template, MATH L3-5 train (5584 problems, never MATH-500), P1-grader/Math-Verify reward,
REF knobs (beta 0.001, G=8, batch 128, lr 1e-6, T=1.0, 3072 tokens, 400 steps), save grid {10,...,100,150,...,400} = 16 HF checkpoints (D42 says "14"; every-10-to-100 plus every-50-to-400 is 10 + 6 = 16), analysis tail, prune-after-copy (D39).
Runs (suffix `_p` = patched rope; unsuffixed tags were the invalid pre-fix runs): `olmo3_ref_s1_p` (seed 1, beta .001), `olmo3_beta0_s1_p` (seed 1, beta 0), `olmo3_ref_s2_p` (seed 2, beta .001).

## IMPORTANT: why `olmo3_rope_fix.py` (do not drop it)
transformers 5.9.0 (pinned by verl 0.9.1 in `envs/rl`) applies the YaRN rope (attention_factor 1.208) to ALL 32 layers of Olmo3; OLMo-core and vLLM apply it only to the 8
full-attention layers (the 24 sliding-window layers use plain RoPE). Unpatched, the FSDP actor and the vLLM rollout disagree under identical weights
(step-1 `rollout_corr/kl` 0.57, Pearson 0.68, `actor/kl_loss` 1e-4 -> 0.68 in 10 steps; run `olmo3_ref_s1.INVALID_rope_bug`, stopped 2026-10-02 ~09:25).
The patch is loaded in every verl worker through `actor_rollout_ref.model.external_lib=olmo3_rope_fix` (PYTHONPATH=$ROOT/code/olmo7b). Verified with `check_rope_fix.py`
(48 MATH-500 prompts, 27,164 sampled tokens, vLLM 0.24 vs HF 5.9.0 sdpa): mean |dlogp| 0.1747 -> **0.0055**, corr(p) 0.951 -> 1.000. envs/gen (transformers 5.18) is correct without a patch
(analysis/LSC/H_sub unaffected). Any other HF-5.9 Olmo3 forward in envs/rl needs the same patch.

## Files (this dir; deployed copies in `$ROOT/code/olmo7b/` on hostA)
| file | role |
|---|---|
| `olmo3_rope_fix.py` | the monkey patch (stateless, no new buffers) |
| `check_rope_fix.py` | vLLM-vs-HF log-prob check (`gen` then `hf nofix|fix`) |
| `grpo_olmo7b.sh` | one verl phase (copy of infra `grpo_p1.sh` + OLMo defaults + external_lib) |
| `run_olmo7b.sh` | one full run: phase A (1-100, save/10) -> phase B (to 400, save/50), exporter, disk guard, analysis tail, no in-job prune. Resumable, exit 0 iff 400/400 |
| `export_ckpts.sh` | copy of infra's exporter (GRID by env) + one change: restores the base `config.json` after every merge (the merger writes a flat yarn `rope_parameters` that transformers>=5.18 / vLLM 0.28 read as default rope on full-attention layers; checked identical to the base after the fix) |
| `run_olmo_analysis.sh` | copy of `exp/run_s7_analysis.sh` with template `olmo_rlzero`, step 0 = Olmo-3-1025-7B |
| `chain_driver.sh` | back-to-back chain REF s1 -> beta0 s1 -> REF s2 on hostA GPUs 0-3, resumable |
| `olmo7b.sbatch` | same runner inside a Slurm job (4 GPU, 3.5 d) |
| `prune_after_copy_hostA.sh` | D39 from the Mac: fetch archives, sha256 verify, then prune HF steps to {20,100,200,300,400} |

## Commands (hostA, ROOT=<WORK>/arr_oct; GPUs 0-3 only, check `nvidia-smi` first)
```bash
# deploy after editing (never print/copy keys): for f in *.sh *.py; do ssh hostA "cat > $ROOT/code/olmo7b/$f" < $f; done
# rehearsal (4 steps, both phases + resume + exporter, ~50 min):
TAG=olmo3_smoke_fix SEED=1 BETA=0.001 STEPS=4 A_STEPS=2 A_SAVE=2 B_SAVE=2 GRID="2 4" ANALYSIS=0 TEST_FREQ=4 setsid nohup bash $ROOT/code/olmo7b/run_olmo7b.sh </dev/null >$ROOT/runs/P1/_logs/olmo3_smoke_fix.out 2>&1 & disown
# the three runs back to back (unattended; re-run the same line to resume after any interruption):
mkdir -p $ROOT/runs/P1/_logs; setsid nohup bash $ROOT/code/olmo7b/chain_driver.sh </dev/null >$ROOT/runs/P1/_logs/olmo7b_chain.out 2>&1 & disown
# one run only:      TAG=olmo3_beta0_s1_p SEED=1 BETA=0 GPUS=0,1,2,3 setsid nohup bash $ROOT/code/olmo7b/run_olmo7b.sh </dev/null >$ROOT/runs/P1/_logs/olmo3_beta0_s1_p.out 2>&1 & disown
# give a run to Slurm: touch $ROOT/runs/P1/<tag>.ON_SLURM (driver skips it), then  sbatch --export=ALL,TAG=<tag>,SEED=<s>,BETA=<b> --job-name=p1-<tag> olmo7b.sbatch   (log in infra/slurm_jobs.md)
# stop the chain between runs: touch $ROOT/runs/P1/olmo7b_chain.STOP
```
Monitor: `tail -f $ROOT/runs/P1/<tag>/train.log` (metrics lines `step:N - ...`), `$ROOT/runs/P1/<tag>/export.log`, `df -h <MNT>`. Rollouts of every step: `runs/P1/<tag>/rollouts/<step>.jsonl`.
Outputs: raw resume ckpt `runs/P1/<tag>/ckpt` (latest only, ~82 GB), HF checkpoints `models/S7/<tag>/step_N` (14.6 GB each), analysis `runs/P1/S7/<tag>/{results.tgz,gen_all_raw.tgz}`.
After a run's COMPLETE marker: `bash prune_after_copy_hostA.sh <tag>` from the Mac (D39), then add the ARTIFACTS.md line it prints.

## Safety
GPUs outside P1_ALLOWED_GPUS (default 0,1,2,3) are refused; the runner will not start on a busy GPU or with < MIN_FREE_GB (300) free on <MNT>, and a guard thread SIGTERMs the trainer if free space drops below it
(raw 82 GB checkpoints; the exporter deletes every raw step except the latest once exported). If zhuzhenghao needs the GPUs: kill the `main_ppo` PID of the run (it resumes from the last checkpoint with the same command).

## Measured (rehearsal olmo3_smoke_fix + the invalid run, same compute graph)
Per-step wall time on 4 H800 (gen ~200 s, old-logprob ~70 s, ref ~80 s, actor update ~235 s): 594, 660, 701, 669 s (rehearsal) and 613-693 s (steps 1-10 of the invalid run), mean ~655 s = ~5.5 steps/h
=> 400 steps ~73 h + 21 validations (~1.5 min) + 16 saves + restart ~ 74 h (3.1 days). Response length grows during RL, so treat the ETA as +-10%. Peak memory 37.7 GB/GPU (MICRO=2, GMU=0.5).
Patched rollout/actor agreement at step 1 (rehearsal): probs_diff_mean 0.0015, Pearson 0.99995, ppl_ratio 1.0001, rollout_corr/kl 4.6e-5 (unpatched: 0.18, 0.68, 1.48, 0.57). Validation MATH-500[:200] acc 0.67 (before) -> 0.675 (step 4); AIME24 0.13-0.17.
