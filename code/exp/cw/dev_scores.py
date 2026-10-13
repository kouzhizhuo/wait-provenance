"""Teacher-forced gate score s for DEV prefixes: s = mean exact token entropy (nats) over the tokens of the LAST sentence of each prefix (the sentence the gate looks at),
distribution at position i-1 predicting token i.  HF forward (bf16) on prompt+prefix; only the last-sentence positions go through lm_head.
    python -m exp.cw.dev_scores --model <dir> --prefixes dev_prefixes.jsonl --out dev_scores.jsonl [--trust]"""
import argparse
import json

import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--prefixes", required=True); ap.add_argument("--out", required=True); ap.add_argument("--trust", action="store_true")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from exp.pipeline import markers as mk
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=a.trust, local_files_only=True).eval()
    base = model.model if hasattr(model, "model") else model.base_model
    rows = [json.loads(l) for l in open(a.prefixes)]
    out = open(a.out, "w")
    with torch.no_grad():
        for k, r in enumerate(rows):
            prompt, prefix = r["prompt"], r["prefix"]
            sents = mk.split_sentences(prefix)
            if not sents:
                continue
            c0 = len(prompt) + sents[-1].start
            enc = tok(prompt + prefix, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
            offs = enc["offset_mapping"][0].tolist()
            pos = [i for i, (s, e) in enumerate(offs) if s >= c0 and i > 0]
            if not pos:
                continue
            ids = enc["input_ids"].to("cuda")
            h = base(input_ids=ids).last_hidden_state[0]
            hp = h[[i - 1 for i in pos]]
            lp = torch.log_softmax(model.lm_head(hp).float(), dim=-1)
            ent = -(lp.exp() * lp).sum(-1)
            tokn = ids[0, pos]
            nll = -lp.gather(1, tokn[:, None]).squeeze(1)
            out.write(json.dumps({"prefix_id": r["prefix_id"], "pid": r["pid"], "src": r["src"], "kind": (r.get("corruption") or {}).get("kind"), "s_ent": ent.mean().item(), "s_nll": nll.mean().item(), "n_tok": len(pos)}) + "\n")
            if k % 100 == 0:
                print(k, "/", len(rows), flush=True)
    out.close()


if __name__ == "__main__":
    main()
