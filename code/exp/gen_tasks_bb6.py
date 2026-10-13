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
    # --- sixth wave (2026-10-05, expansion items 5/6): RL after distillation and trace-SFT endpoints. Each family's FIRST model is its base (builds the shared prefixes).
    ("dsc15", [("r1d1p5b", R + "/models/DeepSeek-R1-Distill-Qwen-1.5B", Q, 0.6, 0.95, 0), ("deepscaler", M("deepscaler"), Q, 0.6, 0.95, 0)]),
    ("@qwen7", [("openthinker7", M("openthinker7"), Q, 0.6, 0.95, 0), ("bespoke7", M("bespoke7"), Q, 0.6, 0.95, 0)]),
    ("@qwen14", [("light14", M("light14"), Q, 0.6, 0.95, 0)]),
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
        cmd = ("cd %s/code/P1 && env %s CUDA_VISIBLE_DEVICES=$G P1_ALLOWED_GPUS=5,6,7 bash exp/run_scale.sh BB_%s %s %s %s %s > %s/BB_%s.run.log 2>&1" % (R, env, mid, path, tpl, T, P, S, mid))
        wr("8_%03d_a_bb_%s" % (k + 100, mid), "bb_" + mid, pre, cmd)
        jd = "%s/BB_%s/ins/scored_shared.jsonl.gz" % (S, mid)
        cmd = ("cd %s/code/P1 && export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=%s/code/P1 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=5,6,7 CUDA_VISIBLE_DEVICES=$G && "
               "%s/envs/gen/bin/python -m exp.pipeline.judge --model %s/models/Qwen2.5-7B-Instruct --prefixes %s --scored %s --gpu_util 0.9 --out %s/BB_%s/ins/judged_shared.jsonl.gz > %s/BB_%s.judge_qwen.log 2>&1 && "
               "%s/envs/gen/bin/python -m exp.pipeline.judge --model %s/models/gpt-oss-20b --mode harmony --effort low --prefixes %s --scored %s --gpu_util 0.9 --out %s/BB_%s/ins/judged_shared_oss.jsonl.gz > %s/BB_%s.judge_oss.log 2>&1"
               % (R, R, R, R, pref, jd, S, mid, S, mid, R, R, pref, jd, S, mid, S, mid))
        wr("8_%03d_b_bbjudge_%s" % (k + 100, mid), "bbjudge_" + mid, jd, cmd)
print("wrote", n, "tasks")
