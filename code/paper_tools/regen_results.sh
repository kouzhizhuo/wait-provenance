#!/bin/sh
# Regenerate numbers/results.tex from every local key file (lead 34e3977c, 2026-10-02). Run from paper/.
E=../exp/results
run() {
python3 tools/json2tex.py ../exp/gold/gold_keys.json ../exp/gold/gold_keys_v4.json ../exp/gold/gold_keys_gpt6.json \
  $E/main/S1/results_keys.json $E/main/S1/results_keys_oss.json $E/main/S1/sdt_keys.json $E/main/S1/valchange/valchange_keys.json \
  $E/main/S1pre/results_keys.json $E/main/S2/results_keys.json $E/main/S2/results_keys_oss.json $E/main/S2/sdt_keys.json $E/main/S2/valchange/valchange_keys.json $E/main/S3/results_keys.json $E/main/S1/lsc_keys.json $E/main/S2/lsc_keys.json \
  $E/scale/scale_keys.json $E/scale/scale_keys_oss.json $E/scale/valchange/valchange_scale_keys.json \
  $E/s9/S9/results_keys.json $E/s9/S9/results_keys_oss.json $E/main/baseline_measures_keys.json \
  $E/main/S1greedy/results_keys.json $E/main/S1boxed/results_keys.json $E/main/S3t07/results_keys.json \
  $E/reframe/*_keys.json $E/reframe/bb/*_keys*.json $E/main/S2ctl/results_keys.json $E/main/S2ctl/results_keys_oss.json \
  numbers/keys/*.json \
  -o numbers/results.tex --digits -1
}
python3 tools/extract_sdt_wait.py >/dev/null
run
python3 tools/derive_zerorl.py >/dev/null   # reads results.tex, so json2tex runs twice
python3 tools/derive_cw.py >/dev/null
run
python3 tools/make_prov_tables.py
python3 tools/make_cw_tables.py
python3 tools/make_route_table.py
python3 tools/caption_below.py tables/*.tex >/dev/null
