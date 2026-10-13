"""Queue tasks (hostA engine) for the backbone-breadth / popular-models panel: per model run_scale.sh (E1 MATH-500[:200]x4 + AIME24x4, E3 on the family-shared
base-derived prefixes, E2 SDT) on ONE GPU, then both judges on the E3 continuations.   python3 exp/gen_tasks_bb.py exp/tasks_d
Families share one prefix file (runs/P1/scale/shared/prefixes_<fam>.jsonl, built from the family's first (base) model's traces: MAKE_PREFIXES=1)."""
import sys
from pathlib import Path

R = "$ARR_ROOT"
S = R + "/runs/P1/scale"
Q = "qwen_boxed"
M = lambda x: (R + "/models/BB/" + x) if not x.startswith("/") else x
# (family, [(id, model path, template, T, top_p, trust_remote)])  first model of a family = base (MAKE_PREFIXES)
FAMS = [
    ("phi", [("phi4", M("phi4"), Q, 0.8, 1.0, 0), ("phi4r", M("phi4r"), Q, 0.8, 0.95, 0), ("phi4rp", M("phi4rp"), Q, 0.8, 0.95, 0)]),
    ("mistral", [("mistral_base", M("mistral_base"), "boxed_raw", 1.0, 1.0, 0)]),
    ("qwen3", [("qwen3_8b_base", M("qwen3_8b_base"), "boxed_raw", 1.0, 1.0, 0), ("qwen3_8b", R + "/models/Qwen3-8B", Q, 0.6, 0.95, 0)]),
    ("r1dl", [("r1d_llama8", M("r1d_llama8"), Q, 0.6, 0.95, 0)]),
    ("qwen7", [("qwen7b", R + "/models/Qwen2.5-7B", "boxed_raw", 1.0, 1.0, 0), ("orn7", M("orn7"), Q, 0.6, 0.95, 0), ("r1d7", R + "/models/DeepSeek-R1-Distill-Qwen-7B", Q, 0.6, 0.95, 0),
               ("acer", M("acer"), Q, 0.6, 0.95, 0)]),
    ("mimo", [("mimo_base", M("mimo_base"), "boxed_raw", 1.0, 1.0, 1), ("mimo_rlzero", M("mimo_rlzero"), Q, 0.6, 0.95, 1), ("mimo_sft", M("mimo_sft"), Q, 0.6, 0.95, 1),
              ("mimo_rl", M("mimo_rl"), Q, 0.6, 0.95, 1)]),
    ("glm", [("glm4", M("glm4"), Q, 0.6, 0.95, 0), ("glmz1", M("glmz1"), Q, 0.6, 0.95, 0)]),
    ("s6", [("s6_dpo", M("s6_dpo"), Q, 1.0, 1.0, 0)] + [("s6_step%d" % k, M("s6_step%d" % k), Q, 1.0, 1.0, 0) for k in (40, 200, 600, 1200, 1800, 2440)]),
    # --- added 2026-10-02 (second wave): 14B pairs and the Mistral-Small base at its card temperature with a base-style prompt
    ("qwen14", [("qwen14b", M("qwen14b"), "boxed_raw", 1.0, 1.0, 0), ("r1d14", R + "/models/DeepSeek-R1-Distill-Qwen-14B", Q, 0.6, 0.95, 0)]),
    ("qwen3_14", [("qwen3_14b_base", M("qwen3_14b_base"), "boxed_raw", 1.0, 1.0, 0), ("qwen3_14b", R + "/models/Qwen3-14B", Q, 0.6, 0.95, 0)]),
    ("mistral_ps", [("mistral_base_ps", M("mistral_base"), "problem_solution", 0.15, 1.0, 0)]),
    # --- third wave (2026-10-02 15:40): zero-download local pairs, the OLMo-2 7B lineage (base -> SFT -> DPO -> Instruct/RLVR) and SmolLM3
    ("r1d15", [("qwen_math15", R + "/models/Qwen2.5-Math-1.5B", "boxed_raw", 1.0, 1.0, 0), ("r1d1p5", R + "/models/DeepSeek-R1-Distill-Qwen-1.5B", Q, 0.6, 0.95, 0)]),
    ("qwen3_1p7", [("qwen3_1p7_base", R + "/models/Qwen3-1.7B-Base", "boxed_raw", 1.0, 1.0, 0), ("qwen3_1p7", R + "/models/Qwen3-1.7B", Q, 0.6, 0.95, 0)]),
    ("qwen3_4", [("qwen3_4b_base", R + "/models/Qwen3-4B-Base", "boxed_raw", 1.0, 1.0, 0), ("qwen3_4b", R + "/models/Qwen3-4B", Q, 0.6, 0.95, 0)]),
    ("olmo2", [("olmo2_base", M("olmo2_base"), "boxed_raw", 1.0, 1.0, 0), ("olmo2_sft", M("olmo2_sft"), Q, 1.0, 1.0, 0), ("olmo2_dpo", M("olmo2_dpo"), Q, 1.0, 1.0, 0),
               ("olmo2_instruct", M("olmo2_instruct"), Q, 1.0, 1.0, 0)]),
    ("smollm3", [("smollm3_base", M("smollm3_base"), "boxed_raw", 1.0, 1.0, 0), ("smollm3", M("smollm3"), Q, 0.6, 0.95, 0)]),
    # --- fourth wave (2026-10-02 18:40): Granite-3.3-8B and Yi-1.5-9B base -> chat; Qwen2.5-7B-Instruct as an extra endpoint of the qwen7 family (shares its prefixes)
    ("granite", [("granite_base", M("granite_base"), "boxed_raw", 1.0, 1.0, 0), ("granite_instruct", M("granite_instruct"), Q, 1.0, 1.0, 0)]),
    ("yi", [("yi_base", M("yi_base"), "boxed_raw", 1.0, 1.0, 0), ("yi_chat", M("yi_chat"), Q, 1.0, 1.0, 0)]),
    ("@qwen7", [("qwen7i", R + "/models/Qwen2.5-7B-Instruct", Q, 1.0, 1.0, 0)]),
]
