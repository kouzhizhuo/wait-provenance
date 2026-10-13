#!/bin/bash
# DEV calibration of Calibrated Wait for ONE model (D50/D50a).  usage: ARR_ROOT=.. GPU=n bash dev_run.sh <key> <model_dir> <template> <T> <top_p> <dev_cap> [--trust]
R=${ARR_ROOT:?}; g=${GPU:?}; key=$1; M=$2; TPL=$3; TEMP=$4; TOPP=$5; CAP=$6; TR=${7:-}
O=$R/runs/P1/cw/dev/$key; mkdir -p $O $R/runs/P1/cw/tables; cd $R/code/P1
export PYTHONPATH=$R/code/P1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=$g CUDA_VISIBLE_DEVICES=$g; PY=$R/envs/gen/bin/python
DEVN=${DEVN:-300}; DEV=$R/data/cw_dev$DEVN.jsonl; [ -f $DEV ] || $PY -m exp.cw.build_dev $R $DEV $DEVN
TAB=$R/runs/P1/cw/tables/$key.json; [ -f $TAB ] || $PY -m exp.cw.cw_prep --tokenizer $M --out $TAB $TR
set -e
[ -f $O/dev_gen.graded.jsonl.gz ] || { $PY -m exp.cw.cw_run --model $M --bench $DEV --template $TPL --mode none --n $DEVN --samples 1 --max_tokens $CAP --temperature $TEMP --top_p $TOPP --gpu_util 0.88 --tag dev_$key $TR --out $O/dev_gen.jsonl > $O/log_gen.txt 2>&1
  $PY -m exp.pipeline.grade --gen $O/dev_gen.jsonl --data $DEV --out_dir $O >> $O/log_grade.txt 2>&1; }
[ -f $O/prefixes.jsonl ] || $PY -m exp.pipeline.sdt prefixes --gen $O/dev_gen.graded.jsonl.gz --data $DEV --n_problems $DEVN --template $TPL --tokenizer $M --out $O/prefixes.jsonl > $O/log_pre.txt 2>&1
[ -f $O/scores.jsonl ] || $PY -m exp.cw.dev_scores --model $M --prefixes $O/prefixes.jsonl --out $O/scores.jsonl $TR > $O/log_scores.txt 2>&1
[ -f $O/cont_none.jsonl ] || $PY -m exp.cw.dev_cont --model $M --prefixes $O/prefixes.jsonl --scores $O/scores.jsonl --mode none --samples 4 --temperature $TEMP --top_p $TOPP --out $O/cont_none.jsonl $TR > $O/log_cont_none.txt 2>&1
$PY -m exp.cw.dev_calibrate --vanilla $O/cont_none.jsonl --out $O/calib.json > $O/log_calib.txt 2>&1
TAU=$($PY -c "import json;print(json.load(open('$O/calib.json'))['tau'])")
$PY -m exp.cw.dev_cont --model $M --prefixes $O/prefixes.jsonl --scores $O/scores.jsonl --mode cw --tables $TAB --tau $TAU --samples 4 --temperature $TEMP --top_p $TOPP --out $O/cont_cw.jsonl $TR > $O/log_cont_cw.txt 2>&1
$PY -m exp.cw.dev_cont --model $M --prefixes $O/prefixes.jsonl --scores $O/scores.jsonl --mode sham --tables $TAB --tau $TAU --samples 4 --temperature $TEMP --top_p $TOPP --out $O/cont_sham.jsonl $TR > $O/log_cont_sham.txt 2>&1
$PY -m exp.cw.dev_calibrate --vanilla $O/cont_none.jsonl --cw $O/cont_cw.jsonl --calib $O/calib.json --out $O/s4.json > $O/log_s4.txt 2>&1
echo "DEV DONE $key tau=$TAU $(date)" >> $O/done.txt
