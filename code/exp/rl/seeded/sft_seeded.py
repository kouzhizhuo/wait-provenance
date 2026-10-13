"""D43(c) SFT stage: Qwen2.5-Math-1.5B on one arm file (prompt = the REF GRPO chat-templated prompt, response = trace + <|endoftext|>).
1 epoch, AdamW lr 1e-5 (constant after a short linear warm-up), fp32 master weights + bf16 autocast, response-only loss, 1 GPU.
Saves the SFT-ONLY checkpoint as bf16 safetensors (+ tokenizer + base generation_config) -> used both as GRPO start and as "step 0" of the analysis.
  python sft_seeded.py --data arms/W512.jsonl --base $R/models/Qwen2.5-Math-1.5B --out $R/models/S7seed/W512_sft [--bs 8 --lr 1e-5 --seed 0]"""
import argparse, hashlib, json, math, os, random, shutil, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--base", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--warmup_frac", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max_len", type=int, default=4096)
a = ap.parse_args()
assert a.epochs == 1
random.seed(a.seed); torch.manual_seed(a.seed)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(a.base)
EOS = tok.eos_token_id                                    # 151643 <|endoftext|> = the stop token of the REF GRPO rollouts (generation_config.json)
rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
exs = []
for r in rows:
    ptxt = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False)   # exactly what verl feeds in the REF runs
    p = tok(ptxt, add_special_tokens=False).input_ids
    y = tok(r["response"], add_special_tokens=False).input_ids + [EOS]
    assert len(p) + len(y) <= a.max_len, (r["pid"], len(p), len(y))
    exs.append((p, y))
order = list(range(len(exs))); random.Random(a.seed).shuffle(order)
steps = math.ceil(len(exs) / a.bs); warm = max(1, round(a.warmup_frac * steps))
dev = "cuda"
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
model.gradient_checkpointing_enable(); model.config.use_cache = False
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)

def nll_mean(m, data):
    m.eval(); tot = n = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for p, y in data:
            ids = torch.tensor([p + y], device=dev); lab = torch.tensor([[-100] * len(p) + y], device=dev)
            lg = m(ids).logits[0, :-1].float(); tgt = lab[0, 1:]
            msk = tgt != -100
            tot += torch.nn.functional.cross_entropy(lg[msk], tgt[msk], reduction="sum").item(); n += int(msk.sum())
    return tot / n

pre = nll_mean(model, exs[:64]); model.train()
log = []
for st in range(steps):
    idx = order[st * a.bs:(st + 1) * a.bs]
    ntok = sum(len(exs[i][1]) for i in idx); tot = 0.0
    for i in idx:
        p, y = exs[i]
        ids = torch.tensor([p + y], device=dev); lab = torch.tensor([[-100] * len(p) + y], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(ids).logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(lg, lab[0, 1:], ignore_index=-100, reduction="sum") / ntok
        loss.backward(); tot += loss.item()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    for g in opt.param_groups: g["lr"] = a.lr * min(1.0, (st + 1) / warm)
    opt.step(); opt.zero_grad(set_to_none=True)
    log.append({"step": st + 1, "loss": tot, "grad_norm": gn, "lr": opt.param_groups[0]["lr"], "n_resp_tokens": ntok})
    print("[sft] step %d/%d loss %.4f gnorm %.2f" % (st + 1, steps, tot, gn), flush=True)
os.makedirs(a.out, exist_ok=True)
model = model.to(torch.bfloat16); model.config.use_cache = True
model.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
shutil.copy(os.path.join(a.base, "generation_config.json"), os.path.join(a.out, "generation_config.json"))
del model, opt; torch.cuda.empty_cache()
m2 = AutoModelForCausalLM.from_pretrained(a.out, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
post = nll_mean(m2, exs[:64])                              # reload of the SAVED bf16 checkpoint: did the SFT survive the bf16 cast?
sd = hashlib.md5(open(os.path.join(a.out, "model.safetensors"), "rb").read(1 << 26)).hexdigest() if os.path.exists(os.path.join(a.out, "model.safetensors")) else None
json.dump({"data": a.data, "n_examples": len(exs), "steps": steps, "bs": a.bs, "lr": a.lr, "warmup_steps": warm, "seed": a.seed, "base": a.base,
           "eos_id": EOS, "nll_first64_before": pre, "nll_first64_after_bf16_reload": post, "mean_resp_tokens": sum(len(y) for _, y in exs) / len(exs),
           "log": log, "seconds": round(time.time() - t0), "torch": torch.__version__}, open(os.path.join(a.out, "sft_meta.json"), "w"), indent=1)
print("[sft] done: NLL on first 64 train responses %.4f -> %.4f (saved bf16 ckpt), %.0fs" % (pre, post, time.time() - t0))
