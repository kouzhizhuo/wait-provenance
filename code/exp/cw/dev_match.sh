#!/bin/bash
# S6 DEV matching for ONE model after dev_run.sh: full DEV generation under CW (tau from calib.json), then random suppression at alpha x CW's per-bin rates (alpha in 0.6 1.0 1.5), then match.json.
# usage: ARR_ROOT=.. GPU=n bash dev_match.sh <key> <model_dir> <template> <T> <top_p> <cap> [--trust]
R=${ARR_ROOT:?}; g=${GPU:?}; key=$1; M=$2; TPL=$3; TEMP=$4; TOPP=$5; CAP=$6; TR=${7:-}
O=$R/runs/P1/cw/dev/$key; cd $R/code/P1
export PYTHONPATH=$R/code/P1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=$g CUDA_VISIBLE_DEVICES=$g; PY=$R/envs/gen/bin/python
TAB=$R/runs/P1/cw/tables/$key.json; DEV=$(ls $R/data/cw_dev*.jsonl | head -n 1)
[ -f $O/dev_gen.jsonl ] || { echo "no vanilla DEV gen"; exit 2; }
DEVN=$($PY -c "import json;print(json.load(open('$O/dev_gen.jsonl.meta.json'))['config']['n'])" 2>/dev/null || echo 300); DEV=$R/data/cw_dev$DEVN.jsonl
TAU=$($PY -c "import json;print(json.load(open('$O/calib.json'))['tau'])")
common="--model $M --bench $DEV --template $TPL --tables $TAB --n $DEVN --samples 1 --max_tokens $CAP --temperature $TEMP --top_p $TOPP --gpu_util 0.88 --tag dev_$key $TR"
[ -f $O/dev_cwgen.jsonl ] || $PY -m exp.cw.cw_run $common --mode cw --tau $TAU --out $O/dev_cwgen.jsonl > $O/log_cwgen.txt 2>&1
PB=$($PY -m exp.cw.dev_match --dir $O --pbins | $PY -c "import sys,json;print(json.dumps(json.load(sys.stdin)['p_bins']))")
for al in 0.6 1.0 1.5; do
  PA=$($PY -c "import json;print(json.dumps([min(1.0,x*$al) for x in json.loads('$PB')]))")
  [ -f $O/dev_rand_$al.jsonl ] || $PY -m exp.cw.cw_run $common --mode random --p_bins "$PA" --out $O/dev_rand_$al.jsonl > $O/log_rand_$al.txt 2>&1
done
$PY -m exp.cw.dev_match --dir $O > $O/log_match.txt 2>&1
echo "MATCH DONE $key $(date)" >> $O/done_match.txt
