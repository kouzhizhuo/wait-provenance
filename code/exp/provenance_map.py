"""Provenance map: one row per model/checkpoint endpoint with FREQ_wait_1k (exact Poisson 95%), counts, MATH-500 accuracy and a recipe group.
    python3 exp/provenance_map.py  ->  exp/results/reframe/provenance_map.json  (PRELIM until all backbone runs are in)
Groups (assigned from the DOCUMENTED recipe, never from the measured rate):
  pretrained_base | zero_rl | reasoning_sft_distill (SFT / distillation, incl. SFT+RL releases) | rl_on_marker_model (RL on a model that already uses the marker) |
  undisclosed | instruct (instruction/preference-tuned, non-reasoning post-training) | rl_from_instruct (RL on an instruct model)  [the last two are additions to the lead's five].
Where a documented recipe puts a model in a group the measured base rate contradicts (e.g. MiMo-7B-Base already writes 'wait'), the note says so; the group is kept by recipe, except `zero_rl`
which requires a marker-free base: MiMo-RL-Zero starts from a base that writes 'wait' (4.9 per 1k), so it is `rl_on_marker_model`."""
from __future__ import annotations

from pathlib import Path

from reframe_keys import OUT, RES, Ex, Ref, common, ix, poisson_ci, read_gz_jsonl

BB = RES / "scale" / "bb"
rows = []
excluded = []


def counts_acc(rows_):
    nw = tk = 0
    ok = 0
    for r in rows_:
        c = ix.trace_counts(r["text"], int(float(r["n_tokens"])))
        nw += int(c.get("wait:n", 0)); tk += int(c["tokens"]); ok += str(r["correct"]) == "True"
    return nw, tk, ok / max(len(rows_), 1)


def add(model, hf_id, group, rows_, src, note=""):
    if not rows_:
        excluded.append({"model": model, "reason": "no local generations"}); return
    if any("Ġ" in r["text"][:300] for r in rows_[:30]):
        excluded.append({"model": model, "reason": "undecoded byte-level text (tokenizer bug): run invalid"}); return
    nw, tk, acc = counts_acc(rows_)
    lo, hi = poisson_ci(nw)
    rows.append({"model": model, "hf_id": hf_id, "group": group, "wait_per_1k": round(1000 * nw / tk, 4), "lo": round(1000 * lo / tk, 4), "hi": round(1000 * hi / tk, 4),
                 "nwait": nw, "ntok": tk, "acc": round(100 * acc, 1), "source_key": src, "note": note})


def gz(p):
    return read_gz_jsonl(p) if Path(p).exists() else None


M = RES / "main"
# ---- OLMo-3 (S1, S1-pre), S2, S9
add("OLMo-3-7B pretraining step 141k", "allenai/Olmo-3-1025-7B-stage1@step141000", "pretrained_base", gz(M / "S1pre/gen/<EMAIL>"), "h1.s1pre.stage1-step141000", "stage-1 pretraining branch; accuracy ~0 (cannot follow the prompt)")
add("OLMo-3-7B pretraining step 707k", "allenai/Olmo-3-1025-7B-stage1@step707000", "pretrained_base", gz(M / "S1pre/gen/<EMAIL>"), "h1.s1pre.stage1-step707000")
add("OLMo-3-7B pretraining step 1.41M", "allenai/Olmo-3-1025-7B-stage1@step1413814", "pretrained_base", gz(M / "S1pre/gen/<EMAIL>"), "h1.s1pre.stage1-step1413814")
add("OLMo-3-7B mid-training", "allenai/Olmo-3-1025-7B-stage2@step47684", "pretrained_base", gz(M / "S1pre/gen/<EMAIL>"), "h1.s1pre.stage2-step47684", "stage-2 mid-training (reasoning-style data enter here); still no post-training")
add("OLMo-3-7B long-context step 6000", "allenai/Olmo-3-1025-7B-stage3@step6000", "pretrained_base", gz(M / "S1pre/gen/<EMAIL>"), "h1.s1pre.stage3-step6000")
add("OLMo-3-7B base (S1 base)", "allenai/Olmo-3-1025-7B", "pretrained_base", gz(M / "S1/gen/<EMAIL>"), "h1.s1.base")
add("OLMo-3-7B RL-Zero-Math (S1 final)", "allenai/Olmo-3-7B-RL-Zero-Math", "rl_on_marker_model", gz(M / "S1/gen/<EMAIL>"), "h1.s1.final", "RLVR directly on the OLMo-3 base, which already writes 'wait' (mid-training)")
add("OLMo-3-7B Think-DPO (S2 base)", "allenai/Olmo-3-7B-Think-DPO", "reasoning_sft_distill", gz(M / "S2/gen/<EMAIL>"), "h1.s2.base", "SFT + DPO on reasoning traces")
add("OLMo-3-7B Think (S2 last)", "allenai/Olmo-3-7B-Think@step_1375", "rl_on_marker_model", gz(M / "S2/gen/<EMAIL>"), "h1.s2.late", "RLVR after Think-SFT/DPO")
add("OLMo-3-7B Think-SFT step 5000 (S2-ctl)", "allenai/Olmo-3-7B-Think-SFT@step5000", "reasoning_sft_distill", gz(M / "S2ctl/gen/<EMAIL>"), "h1.s2ctl.step5000")
add("OLMo-3-7B Think-SFT step 43000 (S2-ctl)", "allenai/Olmo-3-7B-Think-SFT@step43000", "reasoning_sft_distill", gz(M / "S2ctl/gen/<EMAIL>"), "h1.s2ctl.step43000")
add("OLMo-3-32B Think-DPO (S9 base)", "allenai/Olmo-3-32B-Think-DPO", "reasoning_sft_distill", gz(RES / "s9/S9/gen/<EMAIL>"), "h1.s9.base", "SFT + DPO on reasoning traces")
add("OLMo-3.1-32B Think (S9 step 2300)", "allenai/Olmo-3.1-32B-Think@step_2300", "rl_on_marker_model", gz(RES / "s9/S9/gen/<EMAIL>"), "h1.s9.final")
# ---- Qwen runs: S3a (instruct base), REF (zero-RL), S8
add("Qwen2.5-1.5B-Instruct (S3a base)", "Qwen/Qwen2.5-1.5B-Instruct", "instruct", gz(M / "S3/gen/<EMAIL>"), "h1.s3a.base")
add("Qwen2.5-1.5B GRPO (S3a last)", "S3a GRPO run (od2961; repo has no licence)", "rl_from_instruct", gz(M / "S3/gen/<EMAIL>"), "h1.s3a.late", "GRPO from the Instruct model; headline numbers never rest on it")
for n, base_id in ((1, "Qwen/Qwen2.5-Math-1.5B"), (2, "Qwen/Qwen2.5-Math-1.5B"), (7, "Qwen/Qwen2.5-Math-7B")):
    ref = Ref(n); idx = ref.idx()
    nm = {1: "REF seed 1", 2: "REF seed 2", 7: "REF 7B"}[n]
    add("%s base (step 0)" % nm, base_id, "pretrained_base", ref.gen(min(idx)), "h1.%s.base" % ref.sid)
    add("%s final (step %d)" % (nm, max(idx)), "%s + GRPO (REF recipe)" % base_id, "zero_rl", ref.gen(max(idx)), "h1.%s.final" % ref.sid, "our GRPO, zero-RL from a marker-free base")
S8 = [("S8_base", "Qwen/Qwen2.5-32B", "pretrained_base", ""), ("S8_dapo", "BytedTsinghua-SIA/DAPO-Qwen-32B", "zero_rl", "zero-RL from the Qwen2.5-32B base"),
      ("S8_orz", "Open-Reasoner-Zero/Open-Reasoner-Zero-32B", "zero_rl", "zero-RL from the Qwen2.5-32B base"), ("S8_qwq", "Qwen/QwQ-32B", "reasoning_sft_distill", "cold-start SFT + RL release"),
      ("S8_r1distill", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "reasoning_sft_distill", "distillation from R1")]
for m, hid, g, note in S8:
    if m == "S8_orz" and next(iter((BB / "BB_orz32" / "gen_v1").glob("*_v1.graded.jsonl.gz")), None):   # D47b: orz32 (own-eval pair) replaces s8orz as the primary ORZ-32B row; s8orz is robustness only
        excluded.append({"model": "S8 orz (S8 protocol)", "reason": "robustness decode; the primary ORZ-32B pair is orz32 (D47b, counted once)"}); continue
    add(m.replace("S8_", "S8 "), hid, g, gz(RES / "scale" / m / "gen" / ("%s_math500.graded.jsonl.gz" % m)), "h1.%s.final" % {"S8_base": "s8base", "S8_dapo": "s8dapo", "S8_orz": "s8orz", "S8_qwq": "s8qwq", "S8_r1distill": "s8r1d"}[m], note)
for e in ("low", "med", "high"):
    add("gpt-oss-120b effort=%s" % e, "openai/gpt-oss-120b", "undisclosed", gz(RES / "scale" / ("S10_%s" % e) / "gen" / ("S10_%s_math500.graded.jsonl.gz" % e)), "h1.s10%s.final" % e, "training recipe undisclosed")
# ---- backbones and popular panel
def B(mid):   # SimpleRL-Zoo/ORZ/Oat rows: the authors'-eval generations (gen_v1) are primary; every other backbone: the harness MATH-500 generations
    for d, pat in (("gen_v1", "*_v1.graded.jsonl.gz"), ("gen", "*_math500.graded.jsonl.gz"), ("gen_cap32k", "*_c32k.graded.jsonl.gz")):
        p = next(iter((BB / ("BB_" + mid) / d).glob(pat)), None)
        if p:
            return p
    return None
def bb(mid, model, hf, group, series, note=""):
    p = B(mid)
    add(model, hf, group, gz(p) if p else None, series, note)
bb("phi4", "Phi-4", "microsoft/phi-4", "instruct", "h1.pop_phi4.final", "SFT + DPO, no reasoning-trace SFT")
bb("phi4r", "Phi-4-reasoning", "microsoft/Phi-4-reasoning", "reasoning_sft_distill", "h1.pop_phi4r.final")
bb("phi4rp", "Phi-4-reasoning-plus", "microsoft/Phi-4-reasoning-plus", "rl_on_marker_model", "h1.pop_phi4rp.final", "RL on Phi-4-reasoning")
bb("mistral_base_ps", "Mistral-Small-24B-Base-2501", "mistralai/Mistral-Small-24B-Base-2501", "pretrained_base", "h1.pop_mistral_base.final", "problem_solution prompt, T=0.15 (valid run)")
bb("qwen3_8b_base", "Qwen3-8B-Base", "Qwen/Qwen3-8B-Base", "pretrained_base", "h1.pop_qwen3_8b_base.final")
bb("qwen3_8b", "Qwen3-8B", "Qwen/Qwen3-8B", "reasoning_sft_distill", "h1.pop_qwen3_8b.final", "long-CoT SFT + RL + thinking-mode fusion")
bb("qwen3_14b_base", "Qwen3-14B-Base", "Qwen/Qwen3-14B-Base", "pretrained_base", "h1.pop_qwen3_14b_base.final")
bb("qwen3_14b", "Qwen3-14B", "Qwen/Qwen3-14B", "reasoning_sft_distill", "h1.pop_qwen3_14b.final")
bb("qwen3_4b_base", "Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", "pretrained_base", "h1.bb_qwen3_4b_base.final")
bb("qwen3_4b", "Qwen3-4B", "Qwen/Qwen3-4B", "reasoning_sft_distill", "h1.bb_qwen3_4b.final")
bb("qwen3_1p7_base", "Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B-Base", "pretrained_base", "h1.bb_qwen3_1p7_base.final")
bb("qwen3_1p7", "Qwen3-1.7B", "Qwen/Qwen3-1.7B", "reasoning_sft_distill", "h1.bb_qwen3_1p7.final")
bb("r1d_llama8", "R1-Distill-Llama-8B", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "reasoning_sft_distill", "h1.pop_r1d_llama8.final", "distillation; the Llama-3.1-8B base is gated (not run)")
p = BB / "pop_qwen3b30" / "gen" / "pop_qwen3b30_math500.graded.jsonl.gz"
add("Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B-Base", "pretrained_base", gz(p), "h1.pop_qwen3b30.final")
for _id, _nm, _hf, _g, _note in (("pop_qwen3t30", "Qwen3-30B-A3B (thinking)", "Qwen/Qwen3-30B-A3B", "reasoning_sft_distill", "long-CoT SFT + RL + thinking-mode fusion; MoE; Slurm job 407443"),
                                 ("pop_r1dllama70b", "R1-Distill-Llama-70B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "reasoning_sft_distill", "distillation; rerun 407448 after the tokenizer fix")):
    _p = BB / _id / "gen" / ("%s_math500.graded.jsonl.gz" % _id)
    if _p.exists():
        add(_nm, _hf, _g, gz(_p), "h1.%s.final" % _id, _note)
bb("mimo_base", "MiMo-7B-Base", "XiaomiMiMo/MiMo-7B-Base", "pretrained_base", "h1.bb_mimo_base.final", "pretrained with reasoning-heavy data: the BASE already writes 'wait'")
bb("mimo_rlzero", "MiMo-7B-RL-Zero", "XiaomiMiMo/MiMo-7B-RL-Zero", "rl_on_marker_model", "h1.bb_mimo_rlzero.final", "zero-RL by recipe, but its base already uses 'wait' (4.9/1k), so not a marker-free-base test")
bb("mimo_sft", "MiMo-7B-SFT", "XiaomiMiMo/MiMo-7B-SFT", "reasoning_sft_distill", "h1.bb_mimo_sft.final")
bb("mimo_rl", "MiMo-7B-RL", "XiaomiMiMo/MiMo-7B-RL", "rl_on_marker_model", "h1.bb_mimo_rl.final", "RL on MiMo-7B-SFT")
bb("glm4", "GLM-4-9B-0414", "zai-org/GLM-4-9B-0414", "instruct", "h1.bb_glm4.final")
bb("glmz1", "GLM-Z1-9B-0414", "zai-org/GLM-Z1-9B-0414", "reasoning_sft_distill", "h1.bb_glmz1.final", "cold-start SFT + RL release")
bb("qwen7b", "Qwen2.5-7B", "Qwen/Qwen2.5-7B", "pretrained_base", "h1.bb_qwen7b.final")
bb("qwen7i", "Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "instruct", "h1.bb_qwen7i.final")
bb("r1d7", "R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "reasoning_sft_distill", "h1.bb_r1d7.final")
# wave 6 (2026-10-05, expansion items 5/6): RL after distillation and trace-SFT releases
bb("r1d1p5b", "R1-Distill-Qwen-1.5B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "reasoning_sft_distill", "h1.bb_r1d1p5b.final")
bb("deepscaler", "DeepScaleR-1.5B-Preview", "agentica-org/DeepScaleR-1.5B-Preview", "rl_on_marker_model", "h1.bb_deepscaler.final", "RL on R1-Distill-Qwen-1.5B")
bb("openthinker7", "OpenThinker-7B", "open-thoughts/OpenThinker-7B", "reasoning_sft_distill", "h1.bb_openthinker7.final", "trace SFT (R1 traces) of Qwen2.5-7B-Instruct")
bb("bespoke7", "Bespoke-Stratos-7B", "bespokelabs/Bespoke-Stratos-7B", "reasoning_sft_distill", "h1.bb_bespoke7.final", "trace SFT (R1 traces) of Qwen2.5-7B-Instruct")
# openPangu (user decision 2026-10-05: include; official names; Huawei OpenPangu Model License v1.0, research use permitted, EU excluded -> run on non-EU GPUs). No post-training stages documented on the cards; no base released; E1 at 32,768 tokens.
bb("pangu_e7v11", "openPangu-Embedded-7B-V1.1", "openpangu/openPangu-Embedded-7B-V1.1", "undisclosed", "h1.bb_pangu_e7v11.final", "post-training not documented; slow thinking (default); T 1.0, top-p 0.8; run with the vendor's mandatory Chinese system prompt; wait counted inside [unused16]...[unused17] only")
bb("pangu_e7m", "openPangu-Embedded-7B-model", "openpangu/openPangu-Embedded-7B-model", "undisclosed", "h1.bb_pangu_e7m.final", "earlier (v1.0) chat release, not a base; post-training not documented; slow thinking; T 1.0, top-p 0.8")
bb("light14", "Light-R1-14B-DS", "qihoo360/Light-R1-14B-DS", "rl_on_marker_model", "h1.bb_light14.final", "RL on R1-Distill-Qwen-14B")
bb("orn7", "OpenReasoning-Nemotron-7B", "nvidia/OpenReasoning-Nemotron-7B", "reasoning_sft_distill", "h1.bb_orn7.final", "distillation-only, yet almost no 'wait' (0.017/1k): a counter-example to 'distillation installs wait'")
bb("acer", "AceReason-Nemotron-7B", "nvidia/AceReason-Nemotron-7B", "rl_on_marker_model", "h1.bb_acer.final", "RL on DeepSeek-R1-Distill-Qwen-7B")
bb("smollm3_base", "SmolLM3-3B-Base", "HuggingFaceTB/SmolLM3-3B-Base", "pretrained_base", "h1.bb_smollm3_base.final", "base already writes 'wait' (4.2/1k) but accuracy is 4% (does not finish/format): see note")
bb("smollm3", "SmolLM3-3B", "HuggingFaceTB/SmolLM3-3B", "reasoning_sft_distill", "h1.bb_smollm3.final", "SFT + APO with reasoning mode")
bb("granite_base", "Granite-3.3-8B-Base", "ibm-granite/granite-3.3-8b-base", "pretrained_base", "h1.bb_granite_base.final")
bb("granite_instruct", "Granite-3.3-8B-Instruct", "ibm-granite/granite-3.3-8b-instruct", "instruct", "h1.bb_granite_instruct.final")
bb("olmo2_base", "OLMo-2-7B", "allenai/OLMo-2-1124-7B", "pretrained_base", "h1.bb_olmo2_base.final")
bb("olmo2_sft", "OLMo-2-7B-SFT", "allenai/OLMo-2-1124-7B-SFT", "instruct", "h1.bb_olmo2_sft.final")
bb("olmo2_dpo", "OLMo-2-7B-DPO", "allenai/OLMo-2-1124-7B-DPO", "instruct", "h1.bb_olmo2_dpo.final")
bb("olmo2_instruct", "OLMo-2-7B-Instruct (RLVR)", "allenai/OLMo-2-1124-7B-Instruct", "rl_from_instruct", "h1.bb_olmo2_instruct.final", "RLVR on the DPO model")
bb("yi_base", "Yi-1.5-9B", "01-ai/Yi-1.5-9B", "pretrained_base", "h1.bb_yi_base.final")
bb("yi_chat", "Yi-1.5-9B-Chat", "01-ai/Yi-1.5-9B-Chat", "instruct", "h1.bb_yi_chat.final")
bb("s6_dpo", "Tulu-3-8B-DPO (S6 start)", "allenai/Llama-3.1-Tulu-3-8B-DPO", "instruct", "h1.s6.dpo")
for k in (40, 200, 600, 1200, 1800, 2440):
    bb("s6_step%d" % k, "Tulu-3.1-8B RLVR step %d" % k, "allenai/Llama-3.1-Tulu-3.1-8B@step_%d" % k, "rl_from_instruct", "h1.s6.step%04d" % k, "RLVR on the Tulu-3-8B-DPO model")
for sz in ("0p8b", "2b", "4b", "9b"):
    hz = sz.replace("0p8b", "0.8B").upper() if sz != "0p8b" else "0.8B"
    bb("qwen35_%s_base" % sz, "Qwen3.5-%s-Base" % hz, "Qwen/Qwen3.5-%s-Base" % hz, "pretrained_base", "h1.bb_qwen35_%s_base.final" % sz, "newest generation (2026-02)")
    bb("qwen35_%s" % sz, "Qwen3.5-%s" % hz, "Qwen/Qwen3.5-%s" % hz, "reasoning_sft_distill", "h1.bb_qwen35_%s.final" % sz, "post-trained (thinking on by default)")
bb("qwen38_27b", "Qwen3.8-27B", "Qwen/Qwen3.8-27B", "reasoning_sft_distill", "h1.bb_qwen38_27b.final", "newest generation (2026-08), thinking on by default")
# ---- expansion round (D47): zero-RL model vs its exact base, group by the base rate (< 0.05 per 1k -> zero_rl, else rl_on_marker_model)
def base_rate(mid):
    p = B(mid)
    if not p:
        return None
    r = gz(p)
    nw, tk, _ = counts_acc(r)
    return 1000 * nw / tk
SZ = [("sz_q05_base", "sz_q05", "Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", "hkust-nlp/Qwen-2.5-0.5B-SimpleRL-Zoo"), ("sz_q15_base", "sz_q15", "Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo"),
      ("sz_q7_base", "sz_q7", "Qwen2.5-7B", "Qwen/Qwen2.5-7B", "hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"), ("sz_q7_base", "orz7b", "Qwen2.5-7B", "Qwen/Qwen2.5-7B", "Open-Reasoner-Zero/Open-Reasoner-Zero-7B"),
      ("sz_qm7_base", "sz_qm7", "Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B", "hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo"), ("sz_qm7_base", "oat7", "Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B", "sail/Qwen2.5-Math-7B-Oat-Zero"),
      ("sz_q14_base", "sz_q14", "Qwen2.5-14B", "Qwen/Qwen2.5-14B", "hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo"), ("sz_q32_base", "sz_q32", "Qwen2.5-32B", "Qwen/Qwen2.5-32B", "hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo"),
      ("gr7_base", "gr7", "Qwen2.5-7B", "Qwen/Qwen2.5-7B", "TIGER-Lab/General-Reasoner-Qwen2.5-7B"), ("gr14_base", "gr14", "Qwen2.5-14B", "Qwen/Qwen2.5-14B", "TIGER-Lab/General-Reasoner-Qwen2.5-14B"),
      ("gr3_4_base", "gr3_4", "Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", "TIGER-Lab/General-Reasoner-Qwen3-4B"), ("gr3_14_base", "gr3_14", "Qwen3-14B-Base", "Qwen/Qwen3-14B-Base", "TIGER-Lab/General-Reasoner-Qwen3-14B"),
      ("os15_base", "os15", "Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-1.5B", "ypwang61/One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1"), ("oat15_base", "oat15", "Qwen2.5-Math-1.5B", "Qwen/Qwen2.5-Math-1.5B", "sail/Qwen2.5-Math-1.5B-Oat-Zero"),
      ("orz32_base", "orz32", "Qwen2.5-32B", "Qwen/Qwen2.5-32B", "Open-Reasoner-Zero/Open-Reasoner-Zero-32B"), ("sz_m24_base", "sz_m24", "Mistral-Small-24B", "mistralai/Mistral-Small-24B-Base-2501", "hkust-nlp/Mistral-Small-24B-SimpleRL-Zoo"), ("sz_m7_base", "sz_m7", "Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1", "hkust-nlp/Mistral-7B-v0.1-SimpleRL-Zoo"),
      ("sz_ds7_base", "sz_ds7", "DeepSeek-Math-7B", "deepseek-ai/deepseek-math-7b-base", "hkust-nlp/DeepSeek-Math-7B-SimpleRL-Zoo"), (None, "sz_l8", "Llama-3.1-8B", "meta-llama/Llama-3.1-8B (gated, base not measured)", "hkust-nlp/Llama-3.1-8B-SimpleRL-Zoo")]
seen_base = set()
for bid, fid, nm, bhf, fhf in SZ:
    if bid and bid not in seen_base and B(bid):
        bb(bid, "%s base (%s)" % (nm, "release own-eval setting" if bid.startswith(("gr", "os15", "oat15", "orz")) else "SimpleRL-Zoo template"), bhf, "pretrained_base", "h1.bb_%s.final" % {"sz_ds7_base": "sz_dsm7_base"}.get(bid, bid), "")
        seen_base.add(bid)
    br = base_rate(bid) if bid else None
    grp = "zero_rl" if (br is None or br < 0.05) else "rl_on_marker_model"
    bb(fid, "%s zero-RL (%s)" % (nm, fhf.split("/")[-1]), fhf, grp, "h1.bb_%s.final" % {"sz_ds7": "sz_dsm7"}.get(fid, fid), ("base not measured (gated): D47 conservative rule" if br is None else "base rate %.4f per 1k (D47: %s)" % (br, "marker-free" if br < 0.05 else "marker-bearing")))
excluded += [{"model": "openPangu-R-7B-2512", "reason": "not runnable on our CUDA stack (merge_conv / param-sink weights need torch_npu)"},
             {"model": "Mistral-Small-24B-Base (boxed_raw, T=1.0 run)", "reason": "invalid: acc 2.5% (prompt/temperature); replaced by the problem_solution T=0.15 run"},
             {"model": "Magistral-Small-2506", "reason": "dropped (tekken-only tokenizer, no valid paired base)"},
             {"model": "Skywork-OR1-7B", "reason": "not run: no licence text found on the card"}, {"model": "Qwen2.5-Math-1.5B / R1-Distill-1.5B", "reason": "4096-token context; E1 cannot run at our cap"}]
# sanity: zero_rl requires a marker-free base
import collections
NO_LICENCE = {"hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo"}   # repo has no licence text (support's check): secondary only, not counted (same rule as S3a)
for r in rows:
    if r.get("hf_id") in NO_LICENCE:
        r["counted"] = False; r["note"] = (r.get("note", "") + "; " if r.get("note") else "") + "no licence text on the repo: secondary only, not counted"
# base of each trace-trained row (from the official card's base_model); None = base not measured / not released. Used for h1.prov.sft_free / sft_marked (keys8).
BASE_OF = {"h1.s2.base": "h1.s1.base", "h1.s2ctl.step5000": "h1.s1.base", "h1.s2ctl.step43000": "h1.s1.base",   # OLMo-3-7B Think-SFT/DPO start from Olmo-3-1025-7B
           "h1.s9.base": None,                                                                                  # OLMo-3-32B base not measured
           "h1.s8qwq.final": "h1.s8base.final", "h1.s8r1d.final": "h1.s8base.final",                              # Qwen2.5-32B
           "h1.pop_phi4r.final": "h1.pop_phi4.final",                                                           # Phi-4 (itself post-trained; the card's base_model)
           "h1.pop_qwen3_8b.final": "h1.pop_qwen3_8b_base.final", "h1.pop_qwen3_14b.final": "h1.pop_qwen3_14b_base.final", "h1.bb_qwen3_4b.final": "h1.bb_qwen3_4b_base.final",
           "h1.bb_qwen3_1p7.final": "h1.bb_qwen3_1p7_base.final", "h1.pop_qwen3t30.final": "h1.pop_qwen3b30.final",
           "h1.pop_r1d_llama8.final": None, "h1.pop_r1dllama70b.final": None,                                     # Llama-3.1-8B (gated) / Llama-3.3-70B-Instruct: not measured
           "h1.bb_mimo_sft.final": "h1.bb_mimo_base.final", "h1.bb_glmz1.final": "h1.bb_glm4.final",
           "h1.bb_r1d7.final": "h1.bb_sz_qm7_base.final", "h1.bb_r1d1p5b.final": "h1.bb_os15_base.final",         # Qwen2.5-Math-7B / -1.5B (own-eval bases)
           "h1.bb_openthinker7.final": "h1.bb_qwen7i.final", "h1.bb_bespoke7.final": "h1.bb_qwen7i.final",       # Qwen2.5-7B-Instruct
           "h1.bb_orn7.final": "h1.bb_qwen7b.final", "h1.bb_smollm3.final": "h1.bb_smollm3_base.final",
           "h1.bb_qwen35_0p8b.final": "h1.bb_qwen35_0p8b_base.final", "h1.bb_qwen35_2b.final": "h1.bb_qwen35_2b_base.final",
           "h1.bb_qwen35_4b.final": "h1.bb_qwen35_4b_base.final", "h1.bb_qwen35_9b.final": "h1.bb_qwen35_9b_base.final",
           "h1.bb_qwen38_27b.final": None}                                                                      # Qwen3.8-27B: no base released
_by_key = {r["source_key"]: r for r in rows}
for r in rows:
    if r["group"] == "reasoning_sft_distill":
        bk = BASE_OF.get(r["source_key"], "UNMAPPED")
        assert bk != "UNMAPPED", "no base mapping for " + r["source_key"]
        r["base_source_key"] = bk
        r["base_wait_per_1k"] = _by_key[bk]["wait_per_1k"] if bk else None
        assert bk is None or bk in _by_key, "base row missing for " + bk
cnt = collections.Counter(r["group"] for r in rows if r.get("counted", True))
common.write_json(OUT / "provenance_map.json", {"prelim": True, "date": "2026-10-02", "wait_per_1k": "FREQ_wait_1k on MATH-500 generations at the model's documented decoding setting (see exp/results/*/ and BACKBONES.md)",
                                                 "groups_count": dict(cnt), "rows": rows, "excluded_or_pending": excluded})
print(dict(cnt), len(rows), "rows;", len(excluded), "excluded/pending")
