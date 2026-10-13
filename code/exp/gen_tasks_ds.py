"""Dataset panel (expansion round): the wait-index of paired models on datasets other than MATH-500 (AIME24, AIME25, AMC23, GSM8K[:500]); MMLU-Pro (500 stratified questions, official 5-shot CoT prompt and answer extraction, letter grader) is in the panel.  One task per (dataset, model): gen -> grade -> indices into $R/runs/P1/scale/DS/<ds>/<model>/.
    python3 exp/gen_tasks_ds.py exp/tasks_ds     # then copy to hostA runs/P1/main/tasks.d (queue picks them up; P1_ALLOWED_GPUS=5,6,7)
Models (pairs): S1 base/final (olmo_rlzero T1.0), Qwen2.5-7B base / SimpleRL-Zoo-7B (qwen-boxed v1 decode T1.0 top-p .95), OLMo-3 Think-recipe SFT/DPO base ('dpo') and final (step 1375) of S2 + R1-Distill-Qwen-7B (T .6 top-p .95).
Keys later: h1.ds_<dataset>.<series>.final.* (reframe_keys16.py)."""
import sys
from pathlib import Path

R = "$ARR_ROOT"
PY = R + "/envs/gen/bin/python"
ENV = ("cd %s/code/P1 && export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=%s/code/P1 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=5,6,7 "
       "&& export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 P1_STOP='</s>,<|im_end|>,<|endoftext|>' && " % (R, R))
D = R + "/data"
DS = [("aime24", D + "/aime24/test-00000-of-00001.parquet", None, 16, 16384), ("aime25", D + "/aime25/test.jsonl", None, 16, 16384),
      ("amc23", D + "/amc23/test-00000-of-00001.parquet", None, 8, 12288), ("gsm8k", D + "/gsm8k/main/test-00000-of-00001.parquet", 500, 2, 8192),
      ("mmlupro", D + "/MMLU-Pro/mmlupro_500.jsonl", None, 2, 8192)]   # MMLU-Pro: official 5-shot CoT raw prompt (template mmlupro_raw, stop 'Question:'), letter grading (grade.py --mc), 500 stratified questions (build_mmlupro.py)
MODELS = [("s1base", R + "/models/S1/base", "olmo_rlzero", 1.0, 1.0), ("s1final", R + "/models/S1/final", "olmo_rlzero", 1.0, 1.0),
          ("szq7base", R + "/models/Qwen2.5-7B", "szx_qwen_boxed", 1.0, 0.95), ("szq7", R + "/models/BB/sz_q7", "szx_qwen_boxed", 1.0, 0.95),
          ("s2base", R + "/models/S2/dpo", "olmo_think", 0.6, 0.95), ("s2final", R + "/models/S2/1375", "olmo_think", 0.6, 0.95), ("r1d7", R + "/models/DeepSeek-R1-Distill-Qwen-7B", "qwen_boxed", 0.6, 0.95),
          ("szq32base", R + "/models/BB/sz_q32_base", "szx_qwen_boxed", 1.0, 0.95), ("szq32", R + "/models/BB/sz_q32", "szx_qwen_boxed", 1.0, 0.95),
          ("szm24base", R + "/models/BB/sz_m24_base", "szx_qwen_boxed", 1.0, 0.95), ("szm24", R + "/models/BB/sz_m24", "szx_qwen_boxed", 1.0, 0.95),
          ("gr34base", R + "/models/Qwen3-4B-Base", "szx_qwen_boxed", 0.6, 0.95), ("gr34", R + "/models/BB/gr3_4", "szx_qwen_boxed", 0.6, 0.95)]   # gr34: General-Reasoner-Qwen3-4B counterexample pair
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
k = 0
for dn, dp, n, ns, cap in DS:
    for mid, mp, tmpl, T, tp in MODELS:
        o = "%s/runs/P1/scale/DS/%s/%s" % (R, dn, mid); b = "%s/%s_%s" % (o, dn, mid)
        nflag = (" --n %d" % n) if n else ""
        mc = dn == "mmlupro"
        tm_ = "mmlupro_raw" if mc else tmpl
        envx = ENV.replace("</s>,<|im_end|>,<|endoftext|>'", "</s>,<|im_end|>,<|endoftext|>,Question:'") if mc else ENV
        cmd = (envx + "mkdir -p {o}/logs && CUDA_VISIBLE_DEVICES=$G {py} -m exp.pipeline.gen --model {mp} --tag DS_{dn}_{mid} --data {dp}{nf} --samples {ns} --max_tokens {cap} --temperature {T} --top_p {tp} "
               "--template {tm} --seed 0 --gpu_util 0.9 --no_tokens --out {b}.jsonl > {o}/logs/gen.log 2>&1 && {rl} -m exp.pipeline.grade --gen {b}.jsonl --data {dp} --out_dir {o}{mcf} >> {o}/logs/grade.log 2>&1 && "
               "{py} -m exp.pipeline.indices --gen {b}.graded.jsonl.gz --out {b}.idx.json >> {o}/logs/idx.log 2>&1").format(o=o, py=PY, rl=R + "/envs/rl/bin/python", mp=mp, dn=dn, mid=mid, dp=dp, nf=nflag, ns=ns, cap=cap, T=T, tp=tp, tm=tm_, b=b, mcf=" --mc" if mc else "")
        pre = (mp + "/.dl_ok") if "/models/BB/" in mp else (mp + "/config.json")
        (out / ("9_ds_%s_%s.task" % (dn, mid))).write_text("ds_%s_%s\n%s\n%s\n" % (dn, mid, pre, cmd)); k += 1
print("wrote", k, "tasks to", out)
