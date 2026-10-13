#!/bin/bash
# After run_scale_judge_hostA.sh: S8 tables and RESULTS_KEYS (Qwen primary judge) -> $ARR_ROOT/runs/P1/scale/out/{scale_summary.json, scale_keys.json}
set -uo pipefail
W=${ARR_ROOT:?}; cd $W/code/P1; export PYTHONPATH=$PWD
until [ -f $W/runs/P1/scale/JUDGE_DONE ]; do sleep 60; done
$W/envs/gen/bin/python -m exp.pipeline.scale_report --root $W/runs/P1/scale --out $W/runs/P1/scale/out > $W/runs/P1/scale_report.log 2>&1
echo "[$(date +%H:%M:%S)] scale report rc=$?" >> $W/runs/P1/scale_report.log
