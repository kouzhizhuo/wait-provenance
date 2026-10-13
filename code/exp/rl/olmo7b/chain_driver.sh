#!/bin/bash
# Back-to-back, resumable chain of the three OLMo-3-7B GRPO runs on hostA GPUs 0-3:  REF s1 -> beta=0 s1 -> REF s2 (tags with suffix _p = patched rope; the unsuffixed tags were the invalid pre-fix runs).
#   setsid nohup bash $ROOT/code/olmo7b/chain_driver.sh </dev/null > $ROOT/runs/P1/_logs/olmo7b_chain.out 2>&1 & disown
# A run is skipped when $RUN/COMPLETE exists (training + analysis done) or when $ROOT/runs/P1/<tag>.ON_SLURM exists (touch it when the run was given to a Slurm slot).
# Re-running the driver is safe: an interrupted run resumes from its last complete checkpoint. The chain stops at the first failed run (no silent skip) and when
# the disk guard fired. Waits (polls every 5 min) if GPUs 0-3 are busy, never kills anything. Stop: touch $ROOT/runs/P1/olmo7b_chain.STOP (takes effect between runs;
# to stop the current run kill its main_ppo PID - it resumes later).
export ARR_ROOT=${ARR_ROOT:-<WORK>/arr_oct}; R=$ARR_ROOT; cd $R
RUNS=${RUNS:-"olmo3_ref_s1_p:1:0.001 olmo3_beta0_s1_p:1:0 olmo3_ref_s2_p:2:0.001"}   # tag:seed:beta
say() { echo "[$(date '+%F %T')] chain: $*"; }
for spec in $RUNS; do
  IFS=: read tag seed beta <<< "$spec"; RUN=$R/runs/P1/$tag
  [ -f $R/runs/P1/olmo7b_chain.STOP ] && { say "STOP file present"; exit 0; }
  [ -f $RUN/COMPLETE ] && { say "$tag already COMPLETE - skip"; continue; }
  [ -f $R/runs/P1/$tag.ON_SLURM ] && { say "$tag is on Slurm - skip"; continue; }
  until [ "$(for g in 0 1 2 3; do nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits; done | awk '$1>1500' | wc -l)" = 0 ]; do say "GPUs 0-3 busy, waiting (5 min)"; sleep 300; done
  say "start $tag seed=$seed beta=$beta"
  TAG=$tag SEED=$seed BETA=$beta GPUS=0,1,2,3 bash $R/code/olmo7b/run_olmo7b.sh > $R/runs/P1/_logs/$tag.out 2>&1; rc=$?
  say "$tag finished rc=$rc"
  [ $rc -ne 0 ] && { say "run failed - chain stops (inspect $RUN/train.log)"; exit $rc; }
  [ -f $RUN/COMPLETE ] || { say "$tag trained but analysis tail not COMPLETE - chain continues, see $RUN/analysis.log"; }
done
say "chain done"
