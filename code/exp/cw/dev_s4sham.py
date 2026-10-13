"""S4 on DEV-B for the SHAM arm (processor loaded, never masks): same SDT summary as dev_calibrate's B_all/B_signflip, from cont_sham.jsonl.   python -m exp.cw.dev_s4sham --dir exp/results/cw/dev/<m>"""
import argparse
import json

from exp.cw.dev_calibrate import load
from exp.pipeline import sdt

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
rows = load(a.dir + "/cont_sham.jsonl"); out = {}
for name, sel in (("B_all", lambda r: r["pid"] % 2 == 1), ("B_signflip", lambda r: r["pid"] % 2 == 1 and (r["src"] == "clean" or r["kind"] == "sign"))):
    res = sdt.sdt_from_rows([r for r in rows if sel(r)], "mark_any", B=2000)
    out["%s.sham" % name] = {k: res[k] for k in ("HR", "FA", "dprime", "criterion")}
json.dump(out, open(a.dir + "/s4_sham.json", "w"), indent=1, default=float); print(a.dir, {k: round(v["HR"]["est"], 3) for k, v in out.items()}, {k: round(v["FA"]["est"], 3) for k, v in out.items()})
