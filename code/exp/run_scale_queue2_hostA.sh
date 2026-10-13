#!/bin/bash
# hostA GPUs 0-3, second batch (after run_scale_queue.sh + run_scale_judge_hostA.sh):
#  1. gpt-oss-20b weights (hf-mirror)          2. model-specific operator tests: prefixes from each model's OWN traces (QwQ, ORZ, R1-Distill), continued by the model and by the base
#  3. judges: Qwen (primary) on the new files; gpt-oss-20b (harmony, co-reported) on ALL S8 E3 outputs     4. scale_report (Qwen) and --judge_tag oss
set -uo pipefail
W=${ARR_ROOT:?}; cd $W/code/P1; export PYTHONPATH=$PWD ARR_ROOT=$W P1_ALLOWED_GPUS=0,1,2,3 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
S=$W/runs/P1/scale; M=$W/models/scale; PY=$W/envs/gen/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }
( export HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30
  [ -f $W/models/gpt-oss-20b/.dl_ok ] || for a in 1 2 3 4 5; do $PY -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b', local_dir='$W/models/gpt-oss-20b', max_workers=8, ignore_patterns=['original/*','metal/*','*.gguf'])" && touch $W/models/gpt-oss-20b/.dl_ok && break; done ) &
for m in S8_qwq S8_orz S8_r1distill; do
  case $m in S8_qwq|S8_r1distill) tpl=qwen_boxed; T=0.6; P=0.95;; S8_orz) tpl=orz; T=1.0; P=1.0;; esac
  log "START own-prefix E3 $m"
  ( export CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 SKIP_SHARED=1 RUN_E2=0 OWN_PREFIXES=1 CROSS_MODEL=$M/S8_base CROSS_TEMPLATE=qwen_boxed
    bash exp/run_scale.sh $m $M/$m $tpl $T $P ) > $S/$m.queue2.log 2>&1
  log "END own-prefix E3 $m rc=$?"
done
wait; until [ -f $W/models/gpt-oss-20b/.dl_ok ]; do sleep 30; done
export P1_ALLOWED_GPUS=0,1,2,3
i=0
for m in S8_base S8_dapo S8_qwq S8_orz S8_r1distill; do
  g=$((i % 4)); i=$((i+1))
  ( export CUDA_VISIBLE_DEVICES=$g
    for f in shared own own_by_cross; do
      [ -f $S/$m/ins/scored_$f.jsonl.gz ] || continue
      pre=$S/$m/ins/prefixes_own.jsonl; [ $f = shared ] && pre=$S/shared/prefixes_base32.jsonl
      [ -f $S/$m/ins/judged_$f.jsonl.gz ] || $PY -m exp.pipeline.judge --model $W/models/Qwen2.5-7B-Instruct --prefixes $pre --scored $S/$m/ins/scored_$f.jsonl.gz --gpu_util 0.9 --out $S/$m/ins/judged_$f.jsonl.gz > $S/$m/judge2_$f.log 2>&1
      $PY -m exp.pipeline.judge --model $W/models/gpt-oss-20b --mode harmony --effort low --prefixes $pre --scored $S/$m/ins/scored_$f.jsonl.gz --gpu_util 0.9 --out $S/$m/ins/judged_${f}_oss.jsonl.gz > $S/$m/judge2_${f}_oss.log 2>&1
    done; log "judged $m (GPU $g)" ) &
  [ $((i % 4)) = 0 ] && wait
done
wait
$PY -m exp.pipeline.scale_report --root $S --out $S/out > $W/runs/P1/scale_report2.log 2>&1
$PY -m exp.pipeline.scale_report --root $S --out $S/out --judge_tag oss >> $W/runs/P1/scale_report2.log 2>&1
touch $S/QUEUE2_DONE; log "SCALE QUEUE 2 DONE"
