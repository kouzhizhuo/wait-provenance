# Where Does *Wait* Come From? Reasoning Markers Are Inherited, Rarely Coined, in Open RL Trajectories

Anonymous code and data release for an ACL Rolling Review submission.

Reinforcement learning (RL) is often credited with teaching reasoning models to write *wait* before they re-check a step. We trace this marker through the training histories of open models with label-free indices (frequency, position, form, usage) and an injected-error alarm test, all computed from sampled outputs. Zero-RL from bases that do not use the marker almost never coins it; the marker arrives with training data (mid-training, reasoning-rich pretraining, or fine-tuning on reasoning traces); where it already exists, released RL-Zero recipes multiply it and reshape it, while other recipes leave it unchanged or erase it. This repository contains the measurement pipeline, the scripts that turn result files into every number, table and figure, the pre-registration log, and the anonymised human-audit labels.

## Repository map

| Path | Contents |
|---|---|
| `code/exp/pipeline/` | sampling (vLLM), answer grading, marker indices, injected-error alarm test |
| `code/exp/rl/` | our GRPO runs (verl): zero-RL, transplant/seeded arms, controlled OLMo-3 fork |
| `code/exp/cw/` | the pre-registered Calibrated Wait decoder (reported in the appendix as not passing its criteria) |
| `code/exp/judge_prompts/` | LLM-judge prompts (used only as measured systems, never as labels for claims) |
| `code/exp/*.py` | task generators and key builders per experiment family |
| `code/paper_tools/` | `json2tex`-style key export, table generators (`make_prov_tables.py`, `make_route_table.py`, `make_cw_tables.py`), derivations (`derive_zerorl.py`, `derive_cw.py`) |
| `code/figure_scripts/` | figure scripts (matplotlib) |
| `results/keys/` | the small JSON result files every number in the paper is read from |
| `results/human_audit/` | blind revision-audit labels (single annotator, anonymised as `A1`) and agreement statistics |
| `prereg/` | the dated pre-registration and decision log (`DECISIONS.md`, D-numbers cited in the paper) and licence notes |
| `data/` | MATH-500 (MIT) and a download script for the other benchmarks |

## Install

```bash
pip install -r requirements.txt     # Python 3.10, CUDA 12 for generation
```

## Reproducing the tables and figures

Every number is a key in `results/keys/**.json`. `code/paper_tools/regen_results.sh` exports them to LaTeX macros; the generators below rebuild each table or figure from those keys.

| Paper item | Script | Input |
|---|---|---|
| Table 1 (training routes) | `code/paper_tools/make_route_table.py` | `results/keys/paper/numbers/keys/route_keys.json`, `zerorl_keys.json` |
| Table 2 (main results), Table 4 (models without a base), App. provenance table | `code/paper_tools/make_prov_tables.py` | `results/keys/exp/results/reframe/**`, `zerorl_keys.json` (`derive_zerorl.py`) |
| Table 3 (benchmarks beyond MATH-500) | `code/paper_tools/make_prov_tables.py` (`dataset_table`) | `results/keys/exp/results/reframe/bb/*` |
| Marker-index trajectories (OLMo-3 RL-Zero / Think) | `code/figure_scripts/plot_h1_clines.py` | `results/keys/exp/results/main/S1,S2/results_keys*.json` |
| Alarm (signal detection) figure | `code/figure_scripts/plot_sdt.py` | `results/keys/exp/results/main/S1,S2/sdt_keys.json` |
| Provenance strip / full map, per-model figure | `code/figure_scripts/hero_strip.py`, `hero_provenance.py`, `plot_major_models.py` | provenance map + result keys |
| Calibrated Wait (appendix) | `code/paper_tools/make_cw_tables.py`, `derive_cw.py` | `cw_keys.json`, `cw_dev_keys.json` |
| Human audits | `code/exp/` gold pipeline | `results/human_audit/` |

Raw generations (several GB) and model checkpoints are not included; the generation commands are in `code/exp/` and the checkpoints are public Hugging Face models listed in `code/exp/ARTIFACTS.md`.

## Data and licences

- MATH-500 (MIT), AIME 2024/2025, AMC23, GSM8K (MIT), MMLU-Pro (MIT): public benchmarks; see `data/download.sh`.
- Models: public checkpoints under their own licences (Apache-2.0, MIT, Llama 3.1 Community, OpenPangu Model License v1.0, …); the full licence table is in the paper's appendix.
- Our code: MIT (`LICENSE`). Result files and audit labels: CC-BY-4.0.

## Citation

Anonymous submission under review.
