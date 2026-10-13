#!/bin/bash
# S9 (Olmo-3.1-32B-Think RL trajectory, 7 points) on hostA GPUs 0-3 (TP=4), unattended and resumable (a task is skipped when its output exists).
# Layout mirrors the S1 main run: $OUT/S9/{gen,insertion,sdt,revlab,revlab_oss}. DPO base = 'base'. E1 500 problems x 2, T=0.6 top-p .95, cap 12288, Think template (= S2).
# E3: 200 prefixes from the LAST checkpoint's traces (late-derived) + 100 from the DPO base's traces (base-derived, crossed); core arms x 4, cap 1536; E2 SDT; judges Qwen2.5-7B (primary) + gpt-oss-20b.
set -uo pipefail
W=${ARR_ROOT:?}; cd $W/code/P1; export PYTHONPATH=$PWD ARR_ROOT=$W P1_ALLOWED_GPUS=0,1,2,3 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
PY=$W/envs/gen/bin/python; M=$W/models/scale; DATA=$W/data/MATH-500/test.jsonl; O=$W/runs/P1/main_s9; INS=$O/S9/insertion
mkdir -p $O/S9/{gen,insertion,sdt,revlab,revlab_oss} $O/logs
log() { echo "[$(date +%H:%M:%S)] $*"; }
CK="2300 base 100 300 700 1200 1800"
mdir() { [ "$1" = base ] && echo $M/S9_dpo || echo $M/S9_$1; }
G() { echo $O/S9/gen/S9@$1.graded.jsonl.gz; }
run() { # name  outputfile  command...   (skip if the output exists)
  local n=$1 f=$2; shift 2
  [ -e "$f" ] && { return 0; }
  log "START $n"; if "$@"; then log "OK    $n"; else log "FAIL  $n"; echo "$n" >> $O/logs/FAILED; fi
}
export CUDA_VISIBLE_DEVICES=0,1,2,3
# GPUs 0-3 are shared with the S10 ladder (run_s10_hostA.sh): never start before it has finished, and always require the four GPUs to be empty
until [ -f $W/runs/P1/scale/S10_DONE ]; do sleep 60; done
gpus_free() { local g u; for g in 0 1 2 3; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$u" -lt 3000 ] || return 1; done; }
until gpus_free; do sleep 30; done
for c in $CK; do
  until [ -f $(mdir $c)/.dl_ok ]; do sleep 60; done
  run e1_$c $(G $c) bash -c "$PY -m exp.pipeline.gen --model $(mdir $c) --tag S9@$c --data $DATA --n 500 --samples 2 --max_tokens 12288 --temperature 0.6 --top_p 0.95 --template olmo_think --tp 4 --seed 0 --gpu_util 0.9 --no_tokens --out $O/S9/gen/S9@$c.jsonl > $O/logs/gen_$c.log 2>&1 && $PY -m exp.pipeline.grade --gen $O/S9/gen/S9@$c.jsonl --data $DATA --out_dir $O/S9/gen >> $O/logs/grade.log 2>&1"
  if [ "$c" = 2300 ]; then   # late-derived prefixes need only the last checkpoint's traces
    run pre_late $INS/prefixes.jsonl $PY -m exp.pipeline.insert prefixes --gen $(G 2300) --data $DATA --n_prefixes 200 --src late --template olmo_think --tokenizer $(mdir 2300) --seed 0 --out $INS/prefixes.jsonl
  fi
  if [ "$c" = base ]; then
    run pre_base $INS/prefixes_base.jsonl $PY -m exp.pipeline.insert prefixes --gen $(G base) --data $DATA --n_prefixes 100 --src base --template olmo_think --tokenizer $(mdir base) --seed 0 --out $INS/prefixes_base.jsonl
  fi
done
for c in $CK; do
  for src in late base; do
    pre=$INS/prefixes.jsonl; cn=conts_S9@$c; sn=scored_S9@$c; [ $src = base ] && { pre=$INS/prefixes_base.jsonl; cn=conts_base_S9@$c; sn=scored_base_S9@$c; }
    run e3_${c}_$src $INS/$sn.jsonl.gz bash -c "$PY -m exp.pipeline.insert continue --model $(mdir $c) --tag S9@$c --prefixes $pre --arms wait,next,hold_on,none --samples 4 --max_tokens 1536 --template olmo_think --tp 4 --gpu_util 0.9 --seed 0 --out $INS/$cn.jsonl.gz > $O/logs/$cn.log 2>&1 && $PY -m exp.pipeline.insert score --conts $INS/$cn.jsonl.gz --prefixes $pre --out $INS/$sn.jsonl.gz >> $O/logs/score.log 2>&1"
  done
  run e2_$c $O/S9/sdt/conts_$c.jsonl.gz bash -c "$PY -m exp.pipeline.sdt prefixes --gen $(G $c) --data $DATA --n_problems 300 --template olmo_think --tokenizer $(mdir $c) --out $O/S9/sdt/prefixes_$c.jsonl >> $O/logs/e2.log 2>&1 && $PY -m exp.pipeline.insert continue --model $(mdir $c) --tag S9@$c --prefixes $O/S9/sdt/prefixes_$c.jsonl --arms none --samples 4 --max_tokens 300 --template olmo_think --tp 4 --gpu_util 0.9 --seed 0 --out $O/S9/sdt/conts_$c.jsonl.gz > $O/logs/e2_$c.log 2>&1"
done
sc() { for c in $CK; do echo $INS/$1S9@$c.jsonl.gz; done | tr '\n' ' '; }
JQ=$W/models/Qwen2.5-7B-Instruct; JO=$W/models/gpt-oss-20b
# judges (primary + co-reported) in parallel on 4 GPUs, then revision labels
( CUDA_VISIBLE_DEVICES=0 run judge_late $INS/judged.jsonl.gz $PY -m exp.pipeline.judge --model $JQ --prefixes $INS/prefixes.jsonl --scored $(sc scored_) --gpu_util 0.9 --out $INS/judged.jsonl.gz ) > $O/logs/jl.log 2>&1 &
( CUDA_VISIBLE_DEVICES=1 run judge_base $INS/judged_base.jsonl.gz $PY -m exp.pipeline.judge --model $JQ --prefixes $INS/prefixes_base.jsonl --scored $(sc scored_base_) --gpu_util 0.9 --out $INS/judged_base.jsonl.gz ) > $O/logs/jb.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2 run judge_late_oss $INS/judged_oss.jsonl.gz $PY -m exp.pipeline.judge --model $JO --mode harmony --prefixes $INS/prefixes.jsonl --scored $(sc scored_) --gpu_util 0.9 --out $INS/judged_oss.jsonl.gz ) > $O/logs/jlo.log 2>&1 &
( CUDA_VISIBLE_DEVICES=3 run judge_base_oss $INS/judged_base_oss.jsonl.gz $PY -m exp.pipeline.judge --model $JO --mode harmony --prefixes $INS/prefixes_base.jsonl --scored $(sc scored_base_) --gpu_util 0.9 --out $INS/judged_base_oss.jsonl.gz ) > $O/logs/jbo.log 2>&1 &
wait
GENS=$(for c in $CK; do G $c; done | tr '\n' ' ')
( CUDA_VISIBLE_DEVICES=0,1 run revlab $O/S9/revlab/revlab.meta.json $PY -m exp.pipeline.revlab --gen $GENS --model $JQ --out_dir $O/S9/revlab --gpu_util 0.9 --tp 2 ) > $O/logs/rl.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2,3 run revlab_oss $O/S9/revlab_oss/revlab.meta.json $PY -m exp.pipeline.revlab --gen $GENS --model $JO --mode harmony --out_dir $O/S9/revlab_oss --gpu_util 0.9 --tp 2 ) > $O/logs/rlo.log 2>&1 &
wait
CKS="base:0,100:100,300:300,700:700,1200:1200,1800:1800,2300:2300"
$PY -m exp.pipeline.sdt score --conts $(for c in $CK; do echo $O/S9/sdt/conts_$c.jsonl.gz; done | tr '\n' ' ') --out $O/S9/sdt.json --series S9 --keys_series s9 --stage_map early=100,mid=700,late=2300 > $O/logs/sdt_score.log 2>&1
$PY -m exp.pipeline.report --res $O/S9 --figdir $O/S9/plots --series S9 --keys_series s9 --ckpts $CKS --last 2300 --stage_map early=100,mid=700,late=2300 --main --no_plots > $O/logs/agg_S9.log 2>&1
$PY -m exp.pipeline.report --res $O/S9 --figdir $O/S9/plots --series S9 --keys_series s9 --ckpts $CKS --last 2300 --stage_map early=100,mid=700,late=2300 --main --no_plots --judge_tag oss > $O/logs/agg_S9_oss.log 2>&1
touch $O/S9_DONE; log "S9 DONE"
