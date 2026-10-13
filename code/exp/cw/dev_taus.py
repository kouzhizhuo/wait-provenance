"""Extra operating points for the accuracy-compute frontier (ablation, not a success criterion; D50 'frontier' reporting): tau at which the retained fraction rho (point estimate, DEV-A, pm1+swap) first falls to 0.75 and 0.50.
    python -m exp.cw.dev_taus --dir <devdir>  -> <devdir>/taus.json"""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.dir + "/cont_none.jsonl")]
    A = [r for r in rows if r["pid"] % 2 == 0 and r["src"] == "corrupted" and r["kind"] in ("pm1", "swap")]
    hits = [r["s0"] for r in A if r["mark_any"]]
    out = {"n_hits": len(hits)}
    if hits:
        grid = np.quantile([r["s0"] for r in A], np.linspace(0, 1, 401))
        for tag, target in (("rho75", 0.75), ("rho50", 0.50)):
            best = float(grid[0])
            for t in grid:
                if sum(h >= t for h in hits) / len(hits) >= target:
                    best = float(t)
            out["tau_" + tag] = best
    json.dump(out, open(a.dir + "/taus.json", "w"), indent=1); print(out)


if __name__ == "__main__":
    main()
