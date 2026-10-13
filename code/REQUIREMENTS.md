# Environment

- Python 3.10; vLLM 0.10 (V1 engine) for generation; transformers, torch (CUDA 12) for scoring;
- numpy, scipy, pandas, scikit-learn for analysis; matplotlib for figures; TRL/verl-style GRPO for our RL runs (exp/rl/).
- LaTeX: Tectonic or pdfLaTeX with the ACL style (paper/acl.sty).

Entry points: `exp/pipeline/` (sampling, marker indices, alarm test), `exp/rl/` (our GRPO runs), `exp/cw/` (pre-registered Calibrated Wait, reported negative),
`paper_tools/regen_results.sh` (all result keys -> numbers/results.tex -> tables).
