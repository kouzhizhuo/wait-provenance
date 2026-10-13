#!/bin/bash
# S9: allenai/Olmo-3.1-32B-Think RL trajectory + its DPO base, pinned shas (exp/ARTIFACTS.md §S9), hf-mirror, two downloads at a time. Skips the duplicate .bin weights.
# Template: the Think chat_template.jinja of step_2300 is copied into every point (the DPO base's default system prompt differs; support's recommendation, lead to confirm).
W=${ARR_ROOT:?}; M=$W/models/scale; mkdir -p $M; cd $W
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=30
dl() { # tag repo sha
  [ -f $M/$1/.dl_ok ] && return 0
  for a in 1 2 3 4 5 6; do
    $W/envs/gen/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('$2', revision='$3', local_dir='$M/$1', max_workers=8, ignore_patterns=['*.bin','original/*','metal/*','*.gguf'])" && { touch $M/$1/.dl_raw_ok; return 0; }
  done; echo "FAIL $1"; return 1
}
fin() { [ -f $M/S9_2300/chat_template.jinja ] && cp $M/S9_2300/chat_template.jinja $M/$1/chat_template.jinja; touch $M/$1/.dl_ok; echo "OK $1 $(date +%H:%M:%S)"; }
dl S9_2300 allenai/Olmo-3.1-32B-Think 29884259e8c87406883e817e0628fa5ce9b529c3 && fin S9_2300
dl S9_dpo allenai/Olmo-3-32B-Think-DPO 97604023f2602a7dfe6d31b1fc229985e0002022 && fin S9_dpo &
for p in "S9_100 e7b80281ff56aa55231daae1d2d69f0b06f45492" "S9_300 b0301624fc01ef96aad699b5acd19b5f51fcb430"; do set -- $p; (dl $1 allenai/Olmo-3.1-32B-Think $2 && fin $1) ; done &
wait
for p in "S9_700 09de5c25fc6a77dfb140743fc7612aa47578a077" "S9_1200 9ca68c2080d077fa25a343a8d3bbb48cc103a724" "S9_1800 cc238b7791aadf09e483829fa34b343dfd0aec69"; do set -- $p; (dl $1 allenai/Olmo-3.1-32B-Think $2 && fin $1); done
echo "S9 download finished $(date)"
