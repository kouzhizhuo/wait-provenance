"""Correctness of the vLLM hook: recompute the probe score with HF on prompt+generated tokens (same random w, seed 0) and compare with the scores the hook produced at each decoding step (probelog, CW_PROBE_EVERY=1).
    python -m exp.cw.d51_check_hf --model <dir> --check check.json --layer 24"""
import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--check", required=True); ap.add_argument("--layer", type=int, default=24); a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    chk = json.load(open(a.check)); tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    ids_p = tok(chk["prompt"], add_special_tokens=False)["input_ids"]; ids = torch.tensor([ids_p + chk["ids"]], device="cuda")
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states[a.layer + 1][0].float()          # output of decoder layer `layer`
    D = hs.shape[-1]; w = torch.tensor(np.random.RandomState(0).randn(D) / np.sqrt(D), dtype=torch.float32, device="cuda")
    S = torch.sigmoid(hs @ w).cpu().numpy()
    log = [json.loads(l) for l in open(a.check + ".probelog.jsonl")]
    diffs = []
    for r in log:
        pos = len(ids_p) + r["step"] - 2
        if 0 <= pos < S.shape[0] and r["S_head"]:
            diffs.append((r["step"], r["S_head"][0], float(S[pos])))
    print(json.dumps({"n_compared": len(diffs), "max_abs_diff": max(abs(x[1] - x[2]) for x in diffs) if diffs else None, "examples": diffs[:5]}))


if __name__ == "__main__":
    main()
