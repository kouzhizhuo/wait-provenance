#!/bin/bash
# gpt-oss-120b on the 60 human-audit v1 items (D35 amendment). Unattended; waits for P2's single copy of the weights and for GPUs 0-3 to be empty
# (so it only runs in a gap between P2 jobs, and finishes in minutes). Only v1 items are present in $W/runs/P1/gold_items (no v2, no labels).
#   ARR_ROOT=<WORK>/arr_oct nohup bash exp/run_judge_audit_120b.sh > $ARR_ROOT/runs/P1/judge_audit_120b.log 2>&1 &
set -uo pipefail
W=${ARR_ROOT:?set ARR_ROOT}; GPUS=${GPUS:-0,1,2,3}; MODEL=${MODEL:-$W/models/gpt-oss-120b}; G=$W/runs/P1/gold_items; OUT=$G/judge_audit_gptoss120b.jsonl
cd $W/code/P1; export PYTHONPATH=$PWD HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 P1_ALLOWED_GPUS=0,1,2,3,4,5,6,7
ready() { [ -f $MODEL/model.safetensors.index.json ] && [ -z "$(find $MODEL -name '*.incomplete' 2>/dev/null | head -1)" ]; }
free() { local g u; for g in ${GPUS//,/ }; do u=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '); [ "$u" -lt 3000 ] || return 1; done; }
until ready; do sleep 120; done
echo "[$(date +%H:%M:%S)] weights present"
until free; do sleep 30; done
sleep 20; free || { until free; do sleep 30; done; }          # debounce: P2 may be between two jobs
echo "[$(date +%H:%M:%S)] GPUs $GPUS free -> running"
NG=$(echo $GPUS | tr ',' '\n' | wc -l)
CUDA_VISIBLE_DEVICES=$GPUS timeout ${AUDIT_TIMEOUT:-1800} $W/envs/gen/bin/python -m exp.pipeline.judge_audit --model $MODEL --gold $G --out $OUT --tp $NG --gpu_util 0.85 --mode harmony
rc=$?
# The scale queue yields while this audit is pending and only checks that the meta file exists: always write it, even on failure/timeout, so it cannot block forever.
[ -f $OUT.meta.json ] || echo "{\"failed\": true, \"rc\": $rc, \"note\": \"judge_audit did not finish (see judge_audit_120b.log)\"}" > $OUT.meta.json
echo "[$(date +%H:%M:%S)] done rc=$rc"
