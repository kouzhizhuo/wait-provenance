"""Check HF-vs-vLLM log-prob agreement for OLMo-3 with/without olmo3_rope_fix.  Env: rl (transformers 5.9 + vllm 0.24), 1 GPU.
  python check_rope_fix.py gen  <model_dir> <val.parquet> <out.pkl>      # vLLM samples (T=1) + sampled-token logprobs
  python check_rope_fix.py hf   <model_dir> <out.pkl> [fix|nofix]         # HF teacher-forced logprobs of the same tokens
"""
import sys, pickle, numpy as np
mode = sys.argv[1]
if mode == "gen":
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    md, val, out = sys.argv[2:5]
    tok = AutoTokenizer.from_pretrained(md)
    df = pd.read_parquet(val); df = df[df.data_source == "math500_200"].head(48)
    prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for _, r in df.iterrows()]
    llm = LLM(md, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.5, enforce_eager=False)
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, max_tokens=768, logprobs=0, seed=0))
    rows = []
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        rows.append({"prompt_ids": list(o.prompt_token_ids), "ids": list(c.token_ids), "lp": [c.logprobs[i][t].logprob for i, t in enumerate(c.token_ids)]})
    pickle.dump(rows, open(out, "wb")); print("GEN_OK", len(rows), "tokens", sum(len(r["ids"]) for r in rows))
else:
    import torch
    md, pk = sys.argv[2:4]; fix = (len(sys.argv) > 4 and sys.argv[4] == "fix")
    if fix:
        sys.path.insert(0, "<WORK>/arr_oct/code/olmo7b"); import olmo3_rope_fix
        from transformers.models.olmo3 import modeling_olmo3 as M; print("patch state:", M.Olmo3Attention._p1_rope_fix)
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(md, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    rows = pickle.load(open(pk, "rb")); d = []; lp_v = []; lp_h = []
    for r in rows:
        ids = torch.tensor([r["prompt_ids"] + r["ids"]]).cuda(); n = len(r["prompt_ids"])
        with torch.no_grad():
            lg = m(ids).logits[0, n - 1:-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, ids[0, n:, None])[:, 0].cpu().numpy()
        lp_h.append(lp); lp_v.append(np.array(r["lp"]))
    a = np.concatenate(lp_h); b = np.concatenate(lp_v)
    print(f"RESULT fix={fix} tokens={len(a)} mean|dlogp|={np.abs(a-b).mean():.4f} mean(vllm-hf)={np.mean(b-a):+.4f} corr(p)={np.corrcoef(np.exp(a),np.exp(b))[0,1]:.4f} "
          f"ppl_hf={np.exp(-a.mean()):.3f} ppl_vllm={np.exp(-b.mean()):.3f}")
