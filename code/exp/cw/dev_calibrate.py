"""Pick tau on DEV-A (D50 + D50a) and report S4 on DEV-B.
  DEV split: problems by pid parity (A = even, B = odd).  Sign-flip corruption family is HELD OUT from calibration (A uses pm1+swap only); S4 is reported on B all-families and on B sign-flip.
  Calibration (offline approximation on vanilla continuations): a corrupted prefix is a hit under tau iff MARK_vanilla and s0 >= tau (the first marker slot is allowed).  tau = the LARGEST value whose
  lower 95% problem-cluster bootstrap bound of the retained HR is >= 0.9 x vanilla HR (point estimate).
    python -m exp.cw.dev_calibrate --vanilla C_none.jsonl --out calib.json
    python -m exp.cw.dev_calibrate --vanilla C_none.jsonl --cw C_cw.jsonl --calib calib.json --out s4.json     (S4 on DEV-B; exact: continuations generated WITH the processor)"""
import argparse
import json

import numpy as np

from exp.pipeline import stats as st
from exp.pipeline import sdt


def load(p):
    return [json.loads(l) for l in open(p)]


def hr_boot(rows_c, tau, B=2000, seed=20261012):
    """PAIRED retained fraction rho(tau) = (hits with s0 >= tau) / (all vanilla hits) on the corrupted prefixes; problem-cluster bootstrap.  Returns (retained HR, rho, lower 95% bound of rho)."""
    pids = sorted({r["pid"] for r in rows_c}); pos = {p: i for i, p in enumerate(pids)}
    hit = np.zeros(len(pids)); kept = np.zeros(len(pids)); n = np.zeros(len(pids))
    for r in rows_c:
        i = pos[r["pid"]]; n[i] += 1; h = bool(r["mark_any"]); hit[i] += h; kept[i] += h and (r["s0"] >= tau)
    rng = np.random.default_rng(seed); rho = kept.sum() / max(hit.sum(), 1)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, len(pids), len(pids)); bs.append(kept[idx].sum() / max(hit[idx].sum(), 1))
    return kept.sum() / n.sum(), rho, float(np.percentile(bs, 2.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--vanilla", required=True); ap.add_argument("--cw"); ap.add_argument("--calib"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    van = load(a.vanilla)
    A = [r for r in van if r["pid"] % 2 == 0]; Bv = [r for r in van if r["pid"] % 2 == 1]
    if not a.cw:
        cA = [r for r in A if r["src"] == "corrupted" and r["kind"] in ("pm1", "swap")]
        hr_van = sum(r["mark_any"] for r in cA) / len(cA)
        s = sorted({round(r["s0"], 4) for r in A})
        grid = np.quantile([r["s0"] for r in A], np.linspace(0.0, 1.0, 201))
        best = None
        for tau in grid:
            est, rho, lo = hr_boot(cA, float(tau))
            if lo >= 0.9:                                  # D50a(1), paired reading: lower 95% bound of retained HR / vanilla HR >= 0.9
                best = (float(tau), est, lo)
        tau = best[0] if best else float(grid[0])
        cl = [r for r in A if r["src"] == "clean" and r["kind"] is None]
        fa_van = sum(r["mark_any"] for r in cl) / max(len(cl), 1)
        fa_t = sum(bool(r["mark_any"]) and r["s0"] >= tau for r in cl) / max(len(cl), 1)
        res = {"tau": tau, "hr_vanilla_A": hr_van, "hr_retained_A": best[1] if best else None, "rho_lo95_A": best[2] if best else None, "rule": "largest tau with lower 95% problem-cluster bootstrap bound of rho = retained hits / vanilla hits >= 0.9 (paired; the literal unpaired reading cannot hold even at tau = -inf)", "fa_vanilla_A": fa_van, "fa_gated_A_approx": fa_t,
               "n_corrupted_A": len(cA), "n_clean_A": len(cl), "families_calibration": ["pm1", "swap"], "found": best is not None}
        json.dump(res, open(a.out, "w"), indent=1); print(res); return
    cw = load(a.cw); cal = json.load(open(a.calib))
    out = {"tau": cal["tau"]}
    for name, sel in (("B_all", lambda r: r["pid"] % 2 == 1), ("B_signflip", lambda r: r["pid"] % 2 == 1 and (r["src"] == "clean" or r["kind"] == "sign"))):
        for cond, rows in (("vanilla", van), ("cw", cw)):
            rr = [r for r in rows if sel(r)]
            res = sdt.sdt_from_rows(rr, "mark_any", B=2000)
            out["%s.%s" % (name, cond)] = {k: res[k] for k in ("HR", "FA", "dprime", "criterion")}
    json.dump(out, open(a.out, "w"), indent=1, default=float); print(json.dumps(out, default=float)[:1500])


if __name__ == "__main__":
    main()
