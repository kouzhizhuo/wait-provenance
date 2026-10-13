#!/bin/bash
# After the S8 scale queue: arm-blind judging of every model's E3 continuations on hostA (Qwen2.5-7B-Instruct, prompts v1.0), one model per GPU, then a DONE marker.
#   ARR_ROOT=<WORK>/arr_oct nohup bash exp/run_scale_judge_hostA.sh > $ARR_ROOT/runs/P1/scale_judge.log 2>&1 &
set -uo pipefail
W=${ARR_ROOT:?}; cd $W/code/P1
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=30
if [ ! -f $W/models/Qwen2.5-7B-Instruct/.dl_ok ]; then
  for a in 1 2 3 4 5; do
    $W/envs/gen/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$W/models/Qwen2.5-7B-Instruct', max_workers=8, ignore_patterns=['*.gguf','original/*'])" && touch $W/models/Qwen2.5-7B-Instruct/.dl_ok && break
  done
fi
echo "[$(date +%H:%M:%S)] judge weights ready: $(ls $W/models/Qwen2.5-7B-Instruct/.dl_ok 2>&1 | tail -1)"
until grep -q "SCALE QUEUE DONE" $W/runs/P1/scale_queue.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] scale queue done -> judging"
export P1_ALLOWED_GPUS=0,1,2,3 ARR_ROOT=$W
i=0
for m in S8_base S8_dapo S8_qwq S8_orz S8_r1distill; do
  g=$((i % 4)); i=$((i+1))
  ( GPU=$g bash exp/judge_scale.sh $m > $W/runs/P1/scale/$m/judge.log 2>&1; echo "[$(date +%H:%M:%S)] judged $m (GPU $g)" ) &
  [ $((i % 4)) = 0 ] && wait
done
wait
touch $W/runs/P1/scale/JUDGE_DONE; echo "[$(date +%H:%M:%S)] SCALE JUDGE DONE"
