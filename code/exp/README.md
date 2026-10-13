# P1 experiment code (`exp/`)

Owner: P1 coder (Sonnet, pane `6fa6ac3a`). `exp/rl/**` belongs to the RL/infra pane (`b8a856c2`). Plan: `notes/EXPERIMENT_PLAN.md` (v2), statistics: `notes/STATS_PLAN.md` (binding for `pipeline/stats.py`).

## Layout
| Path | What |
|---|---|
| `pipeline/markers.py` | LaTeX-aware sentence splitter, form regexes (F), position/orthography features, mechanical stripping of sentence-initial discourse forms (D15) |
| `pipeline/revisions.py` | marker-independent revision-event rule (value change / negation / restart); stand-in for the judge until gold labels exist |
| `pipeline/indices.py` | additive per-trace counts -> H1 indices (FREQ, SI, PI, ORTH, H_pos, H_para, N_eff, OBL, REL, H_next, EROS, CTX); rarefied Miller-Madow entropies; CLI `python -m exp.pipeline.indices` |
| `pipeline/stats.py` | entropy, cluster bootstrap, Firth-style OR, RD/RR, R_amp, crossed 2x2 interaction, d', kappa |
| `pipeline/gen.py` | vLLM checkpoint sampler (per-request seeds, prompt-token ids in meta, GPU guard) |
| `pipeline/grade.py` | `Answer:` / `\boxed{}` extraction + math-verify |
| `pipeline/insert.py` | insertion harness: `prefixes` (late-derived / base-derived), `continue`, `score` (stripped 3-sentence window, rule REVISE) |
| `pipeline/judge.py` | arm-blind local judge (Qwen2.5-7B-Instruct); prompts in `judge_prompts/` |
| `pipeline/hsub.py`, `lsc_embed.py` | policy-side paradigm entropy; fixed-encoder occurrence embeddings for APD/PRT/JSD |
| `pipeline/report.py`, `excerpts.py` | tables, plots, `run_meta.json`; Figure-1 verbatim excerpts |
| `pipeline/tests/` | unit tests on hand-made examples: `cd P1_grammaticalization && python3 -m pytest exp/pipeline/tests` |
| `run_p0.sh`, `run_p0b.sh` | pilot P0 on hostA GPUs 6-7 (main set; crossed base-derived set) |
| `run_s7_analysis.sh` | in-job analysis of one GRPO sweep run (any host; everything via `ARR_ROOT`) |
| `configs/series.json` | checkpoint manifest (HF ids, revisions, templates) |
| `results/<experiment>/` | JSON/CSV + `run_meta.json` per stage (config, `git blob` hashes of config and code, seeds, package versions, GPU-hours) |

## Reproducibility contract
Each stage writes `<output>.meta.json` (`common.write_meta`): full config, `config_hash` and `code_hash` (git blob ids, checkable with
`git hash-object`), seeds (engine seed; per-request seeds are `derive_seed(seed, stage, ids...)`), input/output hashes, package versions, GPU ids and wall time.
`report.py` folds them into `results/<exp>/run_meta.json`. Token ids and logprobs of the generations stay on hostA (`runs/P1/...`).

## Running on hostA (env `gen`; GPUs 6-7 only)
```
rsync -a --delete --exclude '._*' --exclude __pycache__ --exclude results exp/ $ARR_ROOT/code/P1/exp/
ssh hostA 'cd $ARR_ROOT/code/P1 && setsid nohup bash exp/run_p0.sh full > ../../runs/P1/P0_driver.log 2>&1 &'
bash exp/sync_back_p0.sh && python3 -m exp.pipeline.report --res exp/results/P0 --figdir figures
```
