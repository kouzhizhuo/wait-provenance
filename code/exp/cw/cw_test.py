"""Run the test matrix of ONE primary model on ONE GPU: conditions none, sham, nowait, cw, random (alpha*), bf_match, bf_loose over MATH-500 / AIME24 / AIME25 / AMC23 / GSM8K (cw/models.py), each condition = one engine load with all benchmarks.
Frozen inputs from the DEV directory: calib.json (tau), match.json (alpha*, p_bins, bf caps), tables/<key>.json (hash goes into every meta).  Skips cells whose outputs exist.  Then grades every cell (math-verify) into <cell>.graded.jsonl.gz.
    python -m exp.cw.cw_test --key rlzero --root <ARR_ROOT> --gpu 5 [--conds none,sham,...]"""
import argparse
import json
import os
import subprocess
import sys

from exp.cw.models import MODELS, benches


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--root", required=True); ap.add_argument("--gpu", required=True)
    ap.add_argument("--conds", default="none,sham,nowait,cw,random,bf_match,bf_loose,cw75,cw50"); ap.add_argument("--host", required=True)
    a = ap.parse_args(); R = a.root; k = a.key; M = MODELS[k]
    dev = "%s/runs/P1/cw/dev/%s" % (R, k); out = "%s/runs/P1/cw/test/%s" % (R, k); os.makedirs(out, exist_ok=True)
    cal = json.load(open(dev + "/calib.json")); mat = json.load(open(dev + "/match.json")); tab = "%s/runs/P1/cw/tables/%s.json" % (R, k)
    bj = json.dumps(benches(R, k)); mdir = "%s/%s" % (R, M["dir"][a.host]); trust = []
    env = dict(os.environ, PYTHONPATH=R + "/code/P1", HF_HUB_OFFLINE="1", VLLM_USE_FLASHINFER_SAMPLER="0", TOKENIZERS_PARALLELISM="false", P1_ALLOWED_GPUS=a.gpu, CUDA_VISIBLE_DEVICES=a.gpu)
    py = R + "/envs/gen/bin/python"
    subprocess.run([py, "-m", "exp.cw.dev_taus", "--dir", dev], env=env, stdout=open(out + "/log_taus.txt", "w"), stderr=subprocess.STDOUT)
    taus = json.load(open(dev + "/taus.json"))
    alpha = mat["alpha_star"] or 1.0; pa = [min(1.0, x * alpha) for x in mat["p_bins_cw"]]
    spec = {"none": ["--mode", "none"], "sham": ["--mode", "sham"], "nowait": ["--mode", "nowait"], "cw": ["--mode", "cw", "--tau", str(cal["tau"])], "cw75": ["--mode", "cw", "--tau", str(taus.get("tau_rho75", cal["tau"]))], "cw50": ["--mode", "cw", "--tau", str(taus.get("tau_rho50", cal["tau"]))],
            "random": ["--mode", "random", "--p_bins", json.dumps(pa), "--bin_edges", json.dumps(mat["bin_edges"])],
            "bf_match": ["--mode", "none", "--bf_cap", str(mat["bf_cap_match"]), "--bf_suffix", M["bf"]], "bf_loose": ["--mode", "none", "--bf_cap", str(mat["bf_cap_loose"]), "--bf_suffix", M["bf"]]}
    for c in a.conds.split(","):
        done = all(os.path.exists("%s/%s.%s.jsonl" % (out, c, b["name"])) for b in json.loads(bj))
        if done:
            continue
        if c == "cgrs":   # CGRS baseline (D50d): own runner, outputs cgrs.<bench>.jsonl; probes + wall-clock charged in meta
            cmd = [py, "-m", "exp.cw.cgrs_run", "--model", mdir, "--benches", bj, "--template", M["template"], "--tables", tab, "--temperature", str(M["T"]), "--top_p", str(M["top_p"]), "--gpu_util", "0.9", "--tag", "%s_cgrs" % k, "--out", "%s/cgrs" % out]
            print("RUN", c, flush=True)
            subprocess.run(cmd, env=env, stdout=open("%s/log_%s.txt" % (out, c), "w"), stderr=subprocess.STDOUT)
            continue
        cmd = [py, "-m", "exp.cw.cw_run", "--model", mdir, "--benches", bj, "--template", M["template"], "--tables", tab, "--temperature", str(M["T"]), "--top_p", str(M["top_p"]), "--gpu_util", "0.9",
               "--tag", "%s_%s" % (k, c), "--sync", "--out", "%s/%s.jsonl" % (out, c)] + spec[c]
        print("RUN", c, flush=True)
        subprocess.run(cmd, env=env, stdout=open("%s/log_%s.txt" % (out, c), "w"), stderr=subprocess.STDOUT)
    # grade
    for c in a.conds.split(","):
        for b in json.loads(bj):
            f = "%s/%s.%s.jsonl" % (out, c, b["name"])
            if os.path.exists(f) and not os.path.exists("%s/%s.%s.graded.jsonl.gz" % (out, c, b["name"])):
                subprocess.run([R + "/envs/rl/bin/python", "-m", "exp.pipeline.grade", "--gen", f, "--data", b["path"], "--out_dir", out], env=env, stdout=open(out + "/log_grade.txt", "a"), stderr=subprocess.STDOUT)
    open(out + "/done.txt", "w").write("TEST DONE %s\n" % k)


if __name__ == "__main__":
    main()
