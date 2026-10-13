"""D51 probe features: the layer-L residual-stream state (HF forward, layer OUTPUT = hidden_states[L+1]) at the LAST token of each SDT prefix (the boundary token of the last sentence).
    python -m exp.cw.probe_data --model <dir> --prefixes P.jsonl --layer 24 --out feats.npz [--trust]    -> X (n, D) float16, prefix_id, pid, src, kind"""
import argparse
import json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--prefixes", required=True); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--trust", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=a.trust, local_files_only=True).eval()
    rows = [json.loads(l) for l in open(a.prefixes)]; X = []; meta = []
    with torch.no_grad():
        for r in rows:
            ids = tok(r["prompt"] + r["prefix"], add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
            hs = model.model(input_ids=ids, output_hidden_states=True).hidden_states
            X.append(hs[a.layer + 1][0, -1].float().cpu().numpy().astype(np.float16)); meta.append((r["prefix_id"], r["pid"], r["src"], (r.get("corruption") or {}).get("kind") or ""))
    np.savez(a.out, X=np.stack(X), prefix_id=np.array([m[0] for m in meta]), pid=np.array([m[1] for m in meta]), src=np.array([m[2] for m in meta]), kind=np.array([m[3] for m in meta]))
    print("features", np.stack(X).shape)


if __name__ == "__main__":
    main()
