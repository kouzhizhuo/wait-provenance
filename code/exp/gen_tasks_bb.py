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
    # --- fifth wave (2026-10-04): newest generation (user request): Qwen3.5 Base -> post-trained pairs (thinking on by default) and Qwen3.8-27B
    ("q35_0p8", [("qwen35_0p8b_base", M("qwen35_0p8b_base"), "boxed_raw", 1.0, 1.0, 0), ("qwen35_0p8b", M("qwen35_0p8b"), Q, 0.6, 0.95, 0)]),
    ("q35_2b", [("qwen35_2b_base", M("qwen35_2b_base"), "boxed_raw", 1.0, 1.0, 0), ("qwen35_2b", M("qwen35_2b"), Q, 0.6, 0.95, 0)]),
    ("q35_4b", [("qwen35_4b_base", M("qwen35_4b_base"), "boxed_raw", 1.0, 1.0, 0), ("qwen35_4b", M("qwen35_4b"), Q, 0.6, 0.95, 0)]),
    ("q35_9b", [("qwen35_9b_base", M("qwen35_9b_base"), "boxed_raw", 1.0, 1.0, 0), ("qwen35_9b", M("qwen35_9b"), Q, 0.6, 0.95, 0)]),
    ("q38", [("qwen38_27b", M("qwen38_27b"), Q, 0.6, 0.95, 0)]),
    # --- expansion round (2026-10-04, D47): SimpleRL-Zoo / ORZ-7B / Oat-Zero zero-RL models and their exact bases, same template + decoding for base and final
    ("sz_q05", [("sz_q05_base", M("sz_q05_base"), Q, 1.0, 1.0, 0), ("sz_q05", M("sz_q05"), Q, 1.0, 1.0, 0)]),
    ("sz_q15", [("sz_q15_base", R + "/models/Qwen2.5-1.5B", Q, 1.0, 1.0, 0), ("sz_q15", M("sz_q15"), Q, 1.0, 1.0, 0)]),
    ("sz_q7", [("sz_q7_base", R + "/models/Qwen2.5-7B", Q, 1.0, 1.0, 0), ("sz_q7", M("sz_q7"), Q, 1.0, 1.0, 0), ("orz7b", M("orz7b"), "orz", 1.0, 1.0, 0)]),
    ("sz_qm7", [("sz_qm7_base", M("sz_qm7_base"), Q, 1.0, 1.0, 0), ("sz_qm7", M("sz_qm7"), Q, 1.0, 1.0, 0), ("oat7", M("oat7"), Q, 1.0, 1.0, 0)]),
    ("sz_q14", [("sz_q14_base", M("sz_q14_base"), Q, 1.0, 1.0, 0), ("sz_q14", M("sz_q14"), Q, 1.0, 1.0, 0)]),
    ("sz_m7", [("sz_m7_base", M("sz_m7_base"), "sz_mistral", 1.0, 1.0, 0), ("sz_m7", M("sz_m7"), "sz_mistral", 1.0, 1.0, 0)]),
    ("sz_ds7", [("sz_ds7_base", M("sz_ds7_base"), "sz_dsmath", 1.0, 1.0, 0), ("sz_ds7", M("sz_ds7"), "sz_dsmath", 1.0, 1.0, 0)]),
    ("sz_l8", [("sz_l8", M("sz_l8"), "sz_llama", 1.0, 1.0, 0)]),
]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
n = 0
def wr(fn, name, pre, cmd):
    global n
    (out / (fn + ".task")).write_text("%s\n%s\n%s\n" % (name, pre, cmd)); n += 1
k = 0
for fam, models in FAMS:
    has_base = not fam.startswith("@")
    fam = fam.lstrip("@")
    pref = "%s/shared/prefixes_%s.jsonl" % (S, fam)
    for j, (mid, path, tpl, T, P, trust) in enumerate(models):
        base_here = (j == 0 and has_base)
        k += 1
        ready = path + "/.dl_ok" if "/models/BB/" in path else path + "/config.json"
        pre = ready if base_here else ready + " " + pref
        capenv = " E1_CAP=3000 E3_CAP=400" if mid.startswith(("olmo2", "yi_", "sz_qm7", "oat7", "sz_ds7")) else ""
        capenv += " P1_MAX_NUM_SEQS=128" if mid.startswith(("qwen35", "qwen38")) else ""   # 4096-token context models
        env = "ARR_ROOT=%s PREFIXES_BASE=%s%s%s%s" % (R, pref, " MAKE_PREFIXES=1" if base_here else "", " P1_TRUST_REMOTE=1" if trust else "", capenv)
        cmd = ("cd %s/code/P1 && env %s CUDA_VISIBLE_DEVICES=$G P1_ALLOWED_GPUS=6,7 bash exp/run_scale.sh BB_%s %s %s %s %s > %s/BB_%s.run.log 2>&1" % (R, env, mid, path, tpl, T, P, S, mid))
        wr("8_%02d_a_bb_%s" % (k, mid), "bb_" + mid, pre, cmd)
        jd = "%s/BB_%s/ins/scored_shared.jsonl.gz" % (S, mid)
        cmd = ("cd %s/code/P1 && export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=%s/code/P1 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=6,7 CUDA_VISIBLE_DEVICES=$G && "
               "%s/envs/gen/bin/python -m exp.pipeline.judge --model %s/models/Qwen2.5-7B-Instruct --prefixes %s --scored %s --gpu_util 0.9 --out %s/BB_%s/ins/judged_shared.jsonl.gz > %s/BB_%s.judge_qwen.log 2>&1 && "
               "%s/envs/gen/bin/python -m exp.pipeline.judge --model %s/models/gpt-oss-20b --mode harmony --effort low --prefixes %s --scored %s --gpu_util 0.9 --out %s/BB_%s/ins/judged_shared_oss.jsonl.gz > %s/BB_%s.judge_oss.log 2>&1"
               % (R, R, R, R, pref, jd, S, mid, S, mid, R, R, pref, jd, S, mid, S, mid))
        wr("8_%02d_b_bbjudge_%s" % (k, mid), "bbjudge_" + mid, jd, cmd)
print("wrote", n, "tasks")
