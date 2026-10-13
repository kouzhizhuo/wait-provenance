#!/bin/bash
# S5 branching for ONE model on its DEV-B natural marker events.  usage: ARR_ROOT=.. HOST=hostA|hostA GPU=n bash s5_run.sh <key> <DEVN> [--trust]   (run under gpu_lock wrap)
R=${ARR_ROOT:?}; g=${GPU:?}; H=${HOST:?}; key=$1; DEVN=$2; TR=${3:-}
cd $R/code/P1; export PYTHONPATH=$R/code/P1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=$g CUDA_VISIBLE_DEVICES=$g; PY=$R/envs/gen/bin/python
O=$R/runs/P1/cw/dev/$key; DEV=$R/data/cw_dev$DEVN.jsonl; TAB=$R/runs/P1/cw/tables/$key.json
read M TPL T TP < <($PY -c "from exp.cw.models import MODELS; m=MODELS['$key']; print('$R/'+m['dir']['$H'], m['template'], m['T'], m['top_p'])")
set -e
[ -f $O/s5_events.jsonl ] || $PY -m exp.cw.s5 events --dev_dir $O --data $DEV --template $TPL --tokenizer $M --out $O/s5_events.jsonl $TR > $O/log_s5_events.txt 2>&1
[ -f $O/s5_scores.jsonl ] || $PY -m exp.cw.dev_scores --model $M --prefixes $O/s5_events.jsonl --out $O/s5_scores.jsonl $TR > $O/log_s5_scores.txt 2>&1
[ -f $O/s5_branches.jsonl ] || $PY -m exp.cw.s5 run --model $M --dev_dir $O --tables $TAB --template $TPL --temperature $T --top_p $TP --cap 8192 --out $O/s5_branches.jsonl $TR > $O/log_s5_run.txt 2>&1
[ -f $O/s5_branches.graded.jsonl.gz ] || $R/envs/rl/bin/python -m exp.pipeline.grade --gen $O/s5_branches.jsonl --data $DEV --out_dir $O >> $O/log_grade.txt 2>&1
$PY -m exp.cw.s5 analyze --dev_dir $O --out $O/s5.json > $O/log_s5_an.txt 2>&1
echo "S5 DONE $key $(date)" >> $O/done.txt
