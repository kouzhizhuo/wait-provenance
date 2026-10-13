"""Cheap CW ablations scored OFFLINE on existing DEV traces (no new generation).  Inputs per model dir: cont_none.jsonl (vanilla continuations of clean/corrupted prefixes, with s0 = prefix-sentence mean entropy, mark_any, mark_wait) and scores.jsonl (s_ent, s_nll).
  cw      : gate feature = mean token entropy s_ent, all of F gated          (offline reference, same approximation as dev_calibrate)
  lp      : gate feature = mean token NLL (= - mean log-prob) of the completed sentence, all of F gated; marker allowed iff s_nll >= tau
  waitonly: gate feature = s_ent, only wait-forms gated: a vanilla hit is retained iff s >= tau OR the window has no wait-form (mark_wait False)
tau rule = D50f on DEV-A (even pid; pm1+swap): largest tau whose lower 95% cluster-bootstrap bound of paired rho = retained hits / vanilla hits >= 0.9.  HR/FA scored on DEV-B (odd pid), all corruption families.
    python -m exp.cw.dev_ablate --dir exp/results/cw/dev/<model> --out <dir>/abl.json"""
import argparse
import json

import numpy as np

from exp.cw.dev_calibrate import load


def retained(r, s, tau, arm):
    if not r["mark_any"]:
        return False
    if arm == "waitonly" and not r["mark_wait"]:
        return True
    return s >= tau


def rho_boot(rows, svals, tau, arm, B=2000, seed=20261012):
    pids = sorted({r["pid"] for r in rows}); pos = {p: i for i, p in enumerate(pids)}
    hit = np.zeros(len(pids)); kept = np.zeros(len(pids))
    for r, s in zip(rows, svals):
        i = pos[r["pid"]]; h = bool(r["mark_any"]); hit[i] += h; kept[i] += retained(r, s, tau, arm)
    rng = np.random.default_rng(seed); rho = kept.sum() / max(hit.sum(), 1); bs = []
    for _ in range(B):
        idx = rng.integers(0, len(pids), len(pids)); bs.append(kept[idx].sum() / max(hit[idx].sum(), 1))
    return rho, float(np.percentile(bs, 2.5))


def rate_boot(rows, svals, tau, arm, B=2000, seed=7):
    pids = sorted({r["pid"] for r in rows}); pos = {p: i for i, p in enumerate(pids)}
    k = np.zeros(len(pids)); n = np.zeros(len(pids))
    for r, s in zip(rows, svals):
        i = pos[r["pid"]]; n[i] += 1; k[i] += retained(r, s, tau, arm)
    rng = np.random.default_rng(seed); est = k.sum() / max(n.sum(), 1); bs = []
    for _ in range(B):
        idx = rng.integers(0, len(pids), len(pids)); bs.append(k[idx].sum() / max(n[idx].sum(), 1))
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    van = load(a.dir + "/cont_none.jsonl"); sc = {r["prefix_id"]: r for r in load(a.dir + "/scores.jsonl")}
    van = [r for r in van if r["prefix_id"] in sc]
    res = {}
    for arm, feat in (("cw", "s_ent"), ("lp", "s_nll"), ("waitonly", "s_ent")):
        S = lambda rows: [sc[r["prefix_id"]][feat] for r in rows]
        A = [r for r in van if r["pid"] % 2 == 0 and r["src"] == "corrupted" and r["kind"] in ("pm1", "swap")]
        Bc = [r for r in van if r["pid"] % 2 == 1 and r["src"] == "corrupted"]; Bl = [r for r in van if r["pid"] % 2 == 1 and r["src"] == "clean"]
        sa = S(A); grid = np.quantile(sa, np.linspace(0.0, 1.0, 201)); best = None
        for tau in grid:
            rho, lo = rho_boot(A, sa, float(tau), arm)
            if lo >= 0.9:
                best = (float(tau), rho, lo)
        tau = best[0] if best else float(grid[0]); rho, lo = (best[1], best[2]) if best else rho_boot(A, sa, tau, arm)
        hr = rate_boot(Bc, S(Bc), tau, arm); fa = rate_boot(Bl, S(Bl), tau, arm)
        hv = rate_boot(Bc, S(Bc), -1e9, arm); fv = rate_boot(Bl, S(Bl), -1e9, arm)
        res[arm] = {"tau": tau, "rho": rho, "rho_lo95": lo, "found": best is not None, "hr": hr, "fa": fa, "hr_vanilla": hv, "fa_vanilla": fv, "n_A": len(A), "n_B_corrupted": len(Bc), "n_B_clean": len(Bl), "feature": feat}
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: (v["tau"], v["rho"], v["rho_lo95"], v["hr"][0], v["fa"][0]) for k, v in res.items()}))


if __name__ == "__main__":
    main()
