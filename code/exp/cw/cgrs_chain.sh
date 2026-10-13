#!/bin/bash
# CGRS arm (D50d): smoke test (4 gsm8k problems, cap 512) then the full test matrix cell set.  usage: ARR_ROOT=.. HOST=hostA|hostA GPU=n bash cgrs_chain.sh key1 key2 ...
R=${ARR_ROOT:?}; g=${GPU:?}; H=${HOST:?}; cd $R
PY=$R/envs/gen/bin/python
for key in "$@"; do
  S=$R/runs/P1/cw/test/${key}_cgrs_smoke; mkdir -p $S
  MD=$($PY -c "from exp.cw.models import MODELS; print(MODELS['$key']['dir']['$H'])" 2>/dev/null || true)
  if [ ! -f $S/ok ]; then
    PYTHONPATH=$R/code/P1 bash code/gpu_lock.sh wrap $g P1-cgrs-smoke:$key 2 -- env PYTHONPATH=$R/code/P1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=$g CUDA_VISIBLE_DEVICES=$g $PY -c "
import json,subprocess,sys
from exp.cw.models import MODELS,benches
M=MODELS['$key']; b=[x for x in benches('$R','$key') if x['name']=='gsm8k'][0]; b['n']=4; b['cap']=512
subprocess.run([sys.executable,'-m','exp.cw.cgrs_run','--model','$R/'+M['dir']['$H'],'--benches',json.dumps([b]),'--template',M['template'],'--tables','$R/runs/P1/cw/tables/$key.json','--temperature',str(M['T']),'--top_p',str(M['top_p']),'--tag','smoke','--out','$S/smoke'],check=True)
" > $S/log.txt 2>&1 && touch $S/ok
  fi
  if [ -f $S/ok ]; then
    PYTHONPATH=$R/code/P1 bash code/gpu_lock.sh wrap $g P1-cgrs:$key 30 -- $PY -m exp.cw.cw_test --key $key --root $R --gpu $g --host $H --conds cgrs
  else echo "SMOKE FAILED $key (see $S/log.txt)"; fi
done
