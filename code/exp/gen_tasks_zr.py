"""Zero-RL breadth wave (coordinator list 2026-10-05): own-eval pairs (base + final, SAME template/decoding per release) for
  General-Reasoner (TIGER-Lab, apache-2.0; Qwen2.5-7B/14B, Qwen3-4B/14B bases; release eval = greedy, template and max tokens not stated -> ours: qwen-boxed, T 0, 8,192),
  One-Shot-RLVR Qwen2.5-Math-1.5B-pi1 (apache-2.0; Qwen2.5-Math eval: greedy, 3,072 tokens, qwen25-math-cot template),
  Oat-Zero-1.5B (apache-2.0; same decode as the Oat-Zero-7B primary row: qwen25-math-cot template, T 1.0, top-p .95, 16,000; 4,096-position models run on 32k-position config copies),
  Open-Reasoner-Zero-32B (MIT; orz template, the v1 decode).   AZR is excluded: no model card / licence on its HF repos.
    python3 exp/gen_tasks_zr.py exp/tasks_zr       -> rsync to hostA runs/P1/main/tasks.d
Outputs: runs/P1/scale/BB_<id>/gen_v1/BB_<id>_v1.{jsonl,graded.jsonl.gz,idx.json}  (picked up by reframe_keys12.py as the PRIMARY decode of the row)."""
import sys
from pathlib import Path

R = "$ARR_ROOT"
M = lambda x: R + "/models/BB/" + x
SQ, SM = "szx_qwen_boxed", "szx_qwen25_math"
# id, model path, template, T, top_p, max_tokens, gpu_util, ctx32k_source (None or the original model dir to copy a 32k-position config from)
GR = [("gr7", M("gr7"), R + "/models/Qwen2.5-7B"), ("gr14", M("gr14"), M("sz_q14_base")), ("gr3_4", M("gr3_4"), R + "/models/Qwen3-4B-Base"), ("gr3_14", M("gr3_14"), M("qwen3_14b_base"))]
RUNS = []
for i, fin, base in GR:
    RUNS += [(i + "_base", base, SQ, 0.0, 0.95, 8192, 0.9, None), (i, fin, SQ, 0.0, 0.95, 8192, 0.9, None)]
RUNS += [("os15_base", R + "/models/Qwen2.5-Math-1.5B", SM, 0.0, 0.95, 3072, 0.9, None), ("os15", M("os15"), SM, 0.0, 0.95, 3072, 0.9, None),
         ("oat15_base", M("qm15base_ctx32k"), SM, 1.0, 0.95, 16000, 0.9, R + "/models/Qwen2.5-Math-1.5B"), ("oat15", M("oat15_ctx32k"), SM, 1.0, 0.95, 16000, 0.9, M("oat15")),
         ("orz32_base", M("sz_q32_base"), "orz", 1.0, 0.95, 16000, 0.95, None), ("orz32", M("orz32"), "orz", 1.0, 0.95, 16000, 0.95, None)]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for k, (i, mp, tm, T, tp, cap, gu, src) in enumerate(RUNS):
    o = "%s/runs/P1/scale/BB_%s" % (R, i); b = "%s/gen_v1/BB_%s_v1" % (o, i)
    pre_ok = (mp + "/.dl_ok") if "/models/BB/" in mp and not mp.endswith("_ctx32k") else (mp + "/config.json")
    prelude = ""
    if src:   # 32k-position config copy (identical maths below 4k): symlink every file except config.json
        if mp.endswith("oat15_ctx32k"):
            pre_ok = src + "/.dl_ok"
        else:
            pre_ok = src + "/config.json"
        prelude = ("mkdir -p {d} && for x in {s}/*; do b=$(basename $x); [ $b = config.json ] || ln -sfn $x {d}/$b; done && python3 -c \"import json;c=json.load(open('{s}/config.json'));c['max_position_embeddings']=32768;json.dump(c,open('{d}/config.json','w'),indent=1)\" && touch {d}/.dl_ok && ").format(d=mp, s=src)
    env = ("cd {R}/code/P1 && export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH={R}/code/P1 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=5,6,7 && "
           "export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 P1_STOP='</s>,<|im_end|>,<|endoftext|>' && ").format(R=R)
    cmd = (env + prelude + "mkdir -p {o}/gen_v1 {o}/logs && CUDA_VISIBLE_DEVICES=$G {R}/envs/gen/bin/python -m exp.pipeline.gen --model {mp} --tag BB_{i}@v1 --data {R}/data/MATH-500/test.jsonl --n 500 --samples 1 --max_tokens {cap} "
           "--temperature {T} --top_p {tp} --template {tm} --seed 0 --gpu_util {gu} --no_tokens --out {b}.jsonl > {o}/logs/gen_BB_{i}_v1.log 2>&1 && {R}/envs/rl/bin/python -m exp.pipeline.grade --gen {b}.jsonl "
           "--data {R}/data/MATH-500/test.jsonl --out_dir {o}/gen_v1 >> {o}/logs/grade.log 2>&1 && {R}/envs/gen/bin/python -m exp.pipeline.indices --gen {b}.graded.jsonl.gz --out {b}.idx.json >> {o}/logs/idx.log 2>&1").format(
        R=R, o=o, mp=mp, i=i, cap=cap, T=T, tp=tp, tm=tm, gu=gu, b=b)
    (out / ("9_zx_%02d_exactv1_%s.task" % (k, i))).write_text("exactv1_%s\n%s\n%s\n" % (i, pre_ok, cmd))
print("wrote", len(RUNS), "tasks to", out)
