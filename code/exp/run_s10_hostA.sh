#!/bin/bash
# S10: gpt-oss-120b at three reasoning efforts (E1 + E3 on the shared prefixes + E2), then both judges and the S8/S10 report. hostA GPUs 0-3 (TP=4).
set -uo pipefail
W=${ARR_ROOT:?}; cd $W/code/P1; export PYTHONPATH=$PWD ARR_ROOT=$W P1_ALLOWED_GPUS=0,1,2,3 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
S=$W/runs/P1/scale; PY=$W/envs/gen/bin/python; MODEL=$W/models/gpt-oss-120b
log() { echo "[$(date +%H:%M:%S)] $*"; }
for e in low med high; do
  [ "$e" = med ] && eff=medium || eff=$e
  [ -f $S/S10_$e/DONE ] && continue
  log "START S10_$e (effort $eff)"
  ( export CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4; bash exp/run_scale.sh S10_$e $MODEL harmony 1.0 1.0 $eff ) > $S/S10_$e.queue.log 2>&1
  rc=$?; [ $rc = 0 ] && touch $S/S10_$e/DONE; log "END S10_$e rc=$rc"
done
i=0
for e in low med high; do
  g=$((i % 4)); i=$((i+1))
  ( export CUDA_VISIBLE_DEVICES=$g
    f=shared; m=S10_$e; pre=$S/shared/prefixes_base32.jsonl
    [ -f $S/$m/ins/scored_$f.jsonl.gz ] && {
      $PY -m exp.pipeline.judge --model $W/models/Qwen2.5-7B-Instruct --prefixes $pre --scored $S/$m/ins/scored_$f.jsonl.gz --gpu_util 0.9 --out $S/$m/ins/judged_$f.jsonl.gz > $S/$m/judge3.log 2>&1
      $PY -m exp.pipeline.judge --model $W/models/gpt-oss-20b --mode harmony --effort low --prefixes $pre --scored $S/$m/ins/scored_$f.jsonl.gz --gpu_util 0.9 --out $S/$m/ins/judged_${f}_oss.jsonl.gz > $S/$m/judge3_oss.log 2>&1; }
    log "judged $m" ) &
done
wait
$PY -m exp.pipeline.scale_report --root $S --out $S/out > $W/runs/P1/scale_report3.log 2>&1
$PY -m exp.pipeline.scale_report --root $S --out $S/out --judge_tag oss >> $W/runs/P1/scale_report3.log 2>&1
touch $S/S10_DONE; log "S10 DONE"
