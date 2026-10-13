"""openPangu rows (user decision 2026-10-05: INCLUDE; official names in every table): E1 MATH-500 at max_tokens 32,768 (card's generate.py limit; matches our Qwen3.5 rows), slow thinking (default), vLLM 0.28 (PanguEmbeddedForCausalLM, trust_remote_code for the tokenizer).
Decode per card: openPangu-Embedded-7B-V1.1 / -7B-model: generation_config T 1.0, top-p 0.8 -> MATH-500[:200] x4.  openPangu-R-7B-2512 is NOT runnable on our CUDA stack (merge_conv / param-sink weights need torch_npu).
Prompt (infra 2026-10-05): the vendor's mandatory Chinese safety system prompt from inference/generate.py (without it 7B-V1.1 degenerates) + user turn = problem + our boxed instruction (template pangu_slow; exact string in the .meta.json 'prompt_example'); stop token id 45892, skip_special_tokens=False; wait counted ONLY inside [unused16]...[unused17] (pipeline/pangu_think.py rewrites the graded file: text = thinking span, n_tokens = thinking tokens; the full text is kept as *.full.graded.jsonl.gz).
    python3 exp/gen_tasks_pangu.py exp/tasks_pangu -> rsync to hostA runs/P1/main/tasks.d.   Keys: h1/ctl.bb_pangu_{r7,e7v11,e7m}.final.* via reframe_keys16.py (+ .fin.*)."""
import sys
from pathlib import Path

R = "$ARR_ROOT"
B = R + "/models/B/"
RUNS = [("pangu_e7v11", B + "openPangu-Embedded-7B-V1.1", 200, 4, 1.0, 0.8), ("pangu_e7m", B + "openPangu-Embedded-7B-model", 200, 4, 1.0, 0.8)]   # openPangu-R-7B-2512: NOT runnable on our CUDA stack (merge_conv / param-sink weights need torch_npu) -> dropped
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for k, (i, mp, n, ns, T, tp) in enumerate(RUNS):
    o = "%s/runs/P1/scale/BB_%s" % (R, i); b = "%s/gen_cap32k/BB_%s_c32k" % (o, i)
    env = ("cd {R}/code/P1 && export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH={R}/code/P1 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=5,6,7 P1_TRUST_REMOTE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 P1_KEEP_SPECIAL=1 P1_STOP_IDS=45892 && ").format(R=R)
    cmd = (env + "mkdir -p {o}/gen_cap32k {o}/logs && CUDA_VISIBLE_DEVICES=$G {R}/envs/gen/bin/python -m exp.pipeline.gen --model {mp} --tag BB_{i}@c32k --data {R}/data/MATH-500/test.jsonl --n {n} --samples {ns} --max_tokens 32768 "
           "--temperature {T} --top_p {tp} --template pangu_slow --seed 0 --gpu_util 0.9 --no_tokens --out {b}.jsonl > {o}/logs/gen_BB_{i}_c32k.log 2>&1 && {R}/envs/rl/bin/python -m exp.pipeline.grade --gen {b}.jsonl "
           "--data {R}/data/MATH-500/test.jsonl --out_dir {o}/gen_cap32k >> {o}/logs/grade.log 2>&1 && {R}/envs/gen/bin/python -m exp.pipeline.pangu_think --graded {b}.graded.jsonl.gz --model {mp} >> {o}/logs/think.log 2>&1 && {R}/envs/gen/bin/python -m exp.pipeline.indices --gen {b}.graded.jsonl.gz --out {b}.idx.json >> {o}/logs/idx.log 2>&1").format(R=R, o=o, mp=mp, i=i, n=n, ns=ns, T=T, tp=tp, b=b)
    (out / ("5_pangu_%d_%s.task" % (k, i))).write_text("pangu_%s\n%s\n%s\n" % (i, mp + "/.dl_ok", cmd))
print("wrote", len(RUNS), "tasks")
