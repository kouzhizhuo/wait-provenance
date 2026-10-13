# D43 seeded-marker GRPO (exp/rl/seeded)

Pre-registered control (DECISIONS D43): does GRPO from a marker-free base (Qwen2.5-Math-1.5B) amplify a *Wait,* that SFT supplied?
Runs W512, X512, D512, W64, D64. Nothing outside this folder was edited; REF GRPO = `infra/scripts/p1_sweep.sbatch` + `grpo_p1.sh` verbatim.

## Files
| file | role |
|---|---|
| `gen_seed_traces.py`, `seed_gen.sbatch` | (a) OLMo-3 RL-Zero traces on MATH **train** L3-5 (`data/math_verl/train.parquet`, 5584 problems, 1 sample each, T=1, top_p=1, max 3072 tokens, `olmo_rlzero` template, 4 GPU shards), then (b) the build below |
| `seeded_arms.py` (+ `test_seeded_arms.py`, 7 tests) | opener detection (E3 splitter), arms W/X/D, token-level `verify_arms`, `build` CLI |
| `sft_seeded.py` | SFT stage; saves the SFT-only checkpoint |
| `seeded_run.sbatch` | SFT -> keep SFT-only -> unchanged `p1_sweep.sbatch` (REF GRPO + export + analysis tail) |

## Exact commands
```
# tests (Mac): cd P1_grammaticalization && python3 -m pytest exp/rl/seeded -q
# code to cluster: rsync -a --exclude __pycache__ exp/rl/seeded slogin:/<CLUSTER>/<USER>/arr_oct/code/P1/exp/rl/
# slogin (cd $ARR_ROOT/code/P1/exp/rl/seeded):
sbatch --qos=llm seed_gen.sbatch                                                       # -> runs/P1/seeded/{raw_all.jsonl,data/}
sbatch --qos=llm --dependency=afterok:<GEN> --export=ALL,ARM=W,NDOSE=512 --job-name=p1-seed-W512 seeded_run.sbatch   # likewise X512 D512 W64 D64
infra/scripts/prune_after_copy.sh seed_W512                                            # from the Mac after the job exits 0 (D39)
```
Run TAG = `seed_<ARM><N>`; SFT-only ckpt = `models/S7seed/<ARM><N>_sft`; GRPO exports = `models/S7/seed_<ARM><N>/step_{20,100,200,300,400}`; analysis = `runs/P1/S7/seed_<ARM><N>/results.tgz`.

## Decisions where D43 is silent (most literal reading)
1. **Checkpoint:** `allenai/Olmo-3-7B-RL-Zero-Math` `main` @ `dee26a4074b88215cd1326472e1ab14d0259d892` (final release, 2026-06-25). NOT `step_1900` (@8182367; different weights, last step branch). Shards verified by size + sha256 prefix against the Hub LFS oids.
2. **Opener (lead ruling 2026-10-02):** a RUN of consecutive sentence-initial wait forms (`Wait,` / `Wait, wait,` / `Wait. Wait,`; E3 splitter; form = `[Ww]ait` + punctuation + spaces + non-space on the same line) counts as ONE opener. X replaces the whole run by a single `Next,`; D deletes the whole run and the spaces after it (next word keeps its case, no re-capitalisation).
3. **Strict pool (lead ruling):** a trace is kept only if EVERY case-insensitive word `wait` in it lies inside such a run and every run contains an exact `Wait,` form; any `but wait`, `wait for`, mid-sentence wait, `Wait.` alone, or wait before a newline drops the trace. Hence X and D contain zero `wait` words (asserted).
4. **Eligible trace:** finished (EOS within 3072 OLMo tokens), contains `\boxed`, correct (P1 grader + math-verify AND the REF reward `verl math_verify.compute_score == 1`), strict-pool rule above, every arm <= 3072 Qwen tokens (+EOS; = REF max_response_length), prompt <= 1024, `verify_arms` passes (exact reconstruction + independent word-level diff: D = k deletions of wait words, X = k replacements by `Next`).
5. **Selection:** one trace per problem; one fixed shuffled order (seed 0); N=64 is the prefix of N=512 (nested), W/X/D use the same traces in the same order.
6. **SFT format:** prompt = `tokenizer.apply_chat_template([user: problem + REF instruction], add_generation_prompt=True)` of Qwen2.5-Math-1.5B (what verl feeds in REF; default system prompt included); response = trace + `<|endoftext|>` (id 151643 = the REF stop token). Loss on response tokens only.
7. **SFT hyper-parameters (not fixed by D43; identical across arms):** batch 8 sequences, AdamW (0.9, 0.999, wd 0), lr 1e-5 constant after a 10% linear warm-up, grad-clip 1, fp32 master weights + bf16 autocast, seed 0, 1 epoch (N=64 -> 8 updates, N=512 -> 64). Saved as **bf16** (same dtype as the base and as every GRPO export), so the GRPO start and the evaluated "SFT-only" model are the same bits. `sft_meta.json` stores the response NLL before/after (bf16 reload) as a check that the update survived the cast.
8. **REF GRPO knobs:** untouched (beta 0.001, G=8, batch 128, lr 1e-6, T=1, 3072, seed 1, 400 steps). Deltas that are bookkeeping only: `A_SAVE=20` (save every 20 from step 1), exporter `GRID="20 100 200 300 400"` (the D39 keep set; intermediate raw ckpts are dropped), analysis `BASE_MODEL=<SFT-only ckpt>` so "step 0" = SFT-only (note: the LSC encoder in that script is also BASE_MODEL), `S7_N=500` = full MATH-500 instead of the first 200, in-job prune OFF (D39).
9. **Slots:** chain A (W512 -> D512 -> D64) runs one job at a time; chain B (X512 -> W64) is submitted HELD. The generation job counts as a P1 job while it runs.

## Status (2026-10-02 ~11:00 HKT)
- Round 1 (407147): 5584 traces generated (25 min, 3.87M tokens/shard), build failed on the old chain-counting rule. Strict pool on these = **102** (`runs/P1/seeded/build_report_pass1_strict.json`): 3690 unfinished at 3072 tokens, 1571 non-initial wait, 203 run without exact `Wait,`, 13 incorrect, 2 reward mismatch.
- Round 2 (407194, `PREFIX=raw2`, 6 samples/problem, seed 100) + in-job build over raw+raw2; N is NOT lowered; distinct problems are preferred, repeats of a problem are used only if distinct ones run out (reported as `n_pool_distinct_problems`). Chain A/B depend on afterok:407194 (407151 held).
- If the pool is still < 512, `build` exits 1 and writes `build_report.json`; generate more rounds with another `PREFIX`/`--seed`.
- Counts, mean tokens, token deltas of D/X vs W and md5s: `runs/P1/seeded/data/{build_report.json,MD5SUMS}` after 407194.

## Fallback for the high dose (lead, 2026-10-02; DECISIONS amendment)
- If the strict pool is < 512 after round 2 (407194), ONE more generation round (same filter, new seed/PREFIX, sized from the observed yield). If it is still < 512 by **Oct 4 00:00 CST**, N_high = the pool size rounded down to a multiple of 64 (must be >= 256; low dose stays 64), applied identically to W/X/D: run the build with `FALLBACK_MIN=256` (or `--fallback_min 256`); `N_HIGH=<n>` / `--n_high` forces a value. The decision is stored in `build_report.json` under `fallback`; arm files and run names then carry that N (e.g. `W448`, `ARM=W NDOSE=448`).
- Failure path: `build` prints `POOL_SIZE=<n> ...` and `BUILD FAILED: eligible pool = <n> ...` and exits 1 after writing `build_report.json`.
