#!/bin/bash
# Unattended 32B scale-endpoint queue for a multi-GPU dxm host (hostA: workspace <WORK>/arr_oct, nothing in /home).
#   ARR_ROOT=<WORK>/arr_oct GPU_SETS='0,1,2,3;6,7' WAIT_OTHERS=1 P1_ALLOWED_GPUS=0,1,2,3,6,7 nohup bash exp/run_scale_queue.sh > $ARR_ROOT/runs/P1/scale_queue.log 2>&1 &
# Runs exp/run_scale.sh per model in D32 order (base -> DAPO -> QwQ -> ORZ -> R1-Distill), TP = number of GPUs in $GPUS. A model starts only when its
# weights are complete ($ARR_ROOT/models/scale/<name>/.dl_ok) and ALL its GPUs are empty (so it yields to the other users of the node between models).
# Resumable: a finished model (results.tgz + DONE marker) is skipped. Judging is separate (exp/judge_scale.sh, needs Qwen2.5-7B-Instruct on the node).
set -uo pipefail
ARR_ROOT=${ARR_ROOT:?set ARR_ROOT}; export ARR_ROOT
# GPU_SETS: candidate GPU sets separated by ';' (first one that is completely empty is used; TP = its size), e.g. "0,1,2,3;6,7". GPUS = a single set (old form).
GPU_SETS=${GPU_SETS:-${GPUS:?set GPUS or GPU_SETS, e.g. 0,1,2,3}}
# WAIT_OTHERS=1: start only after other users' jobs are visibly running (coordinator order: P2 on 0-3 first, P3 on 4-7, then P1).
CODE=$ARR_ROOT/code/P1; SC=$ARR_ROOT/runs/P1/scale; M=$ARR_ROOT/models/scale; mkdir -p $SC; cd $CODE
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
busy() { local g u; for g in "$@"; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -ge 3000 ] && return 0; done; return 1; }
set_free() { local g u; for g in ${1//,/ }; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); [ -n "$u" ] && [ "$u" -lt 3000 ] || return 1; done; }
pick_set() { local st; IFS=';' read -ra SS <<< "$GPU_SETS"; for st in "${SS[@]}"; do set_free "$st" && { echo "$st"; return 0; }; done; return 1; }
# the 120b judge audit (run_judge_audit_120b.sh) has priority on GPUs 0-3: wait while its weights are present and its result is not
T0=$(date +%s); GRACE=$(( ${AUDIT_GRACE_MIN:-30} * 60 ))
audit_pending() {
  [ -f $ARR_ROOT/runs/P1/gold_items/judge_audit_gptoss120b.jsonl.meta.json ] && return 1                 # audit done
  [ -f $M/../gpt-oss-120b/model.safetensors.index.json ] && return 0                                        # weights present, audit not yet done
  [ $(( $(date +%s) - T0 )) -lt $GRACE ] && return 0                                                        # grace period: give the weights a chance to arrive
  return 1
}
# name | template | temperature | top_p | extra environment
MODELS=(
 "S8_base|qwen_boxed|1.0|1.0|MAKE_PREFIXES=1"
 "S8_dapo|dapo|1.0|0.7|OWN_PREFIXES=1 CROSS_MODEL=$M/S8_base CROSS_TEMPLATE=qwen_boxed"
 "S8_qwq|qwen_boxed|0.6|0.95|"
 "S8_orz|orz|1.0|1.0|"
 "S8_r1distill|qwen_boxed|0.6|0.95|"
)
if [ "${WAIT_OTHERS:-0}" = 1 ]; then until busy 0 1 2 3 && busy 4 5 6 7; do sleep 60; done; log "other users' jobs are running: P1 scale queue may start"; fi
log "scale queue on GPU sets '$GPU_SETS', ${#MODELS[@]} models"
for spec in "${MODELS[@]}"; do
  IFS='|' read -r name tpl temp topp extra <<< "$spec"
  [ -f $SC/$name/DONE ] && { log "$name already done"; continue; }
  until [ -f $M/$name/.dl_ok ]; do sleep 120; done          # weights complete
  while true; do                                               # yield while anyone else is on the GPUs, and to the pending 120b audit
    audit_pending && { sleep 60; continue; }
    GPUS=$(pick_set) && { sleep 20; set_free "$GPUS" && break; }   # debounce: the owner may be between two jobs
    sleep 60
  done
  NG=$(echo $GPUS | tr ',' '\n' | wc -l)
  log "START $name ($tpl, T=$temp, top_p=$topp) on GPUs $GPUS (TP=$NG)"
  ( export CUDA_VISIBLE_DEVICES=$GPUS P1_ALLOWED_GPUS=${P1_ALLOWED_GPUS:-$GPUS} NGPU=$NG; eval "export $extra"; 
    bash $CODE/exp/run_scale.sh $name $M/$name $tpl $temp $topp ) > $SC/$name.queue.log 2>&1
  rc=$?; [ $rc = 0 ] && touch $SC/$name/DONE; log "END $name rc=$rc"
done
log "SCALE QUEUE DONE"
