"""Fourth-wave filler tasks (hostA engine): extra decoding seeds on headline cells, extra E3 prefixes/samples on S1, E1 seed-1 replicates of every backbone endpoint.
    python3 exp/gen_tasks_wave4.py exp/tasks_d"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
R = "$ARR_ROOT"; O = R + "/runs/P1/main"; S = R + "/runs/P1/scale"; PY = R + "/envs/gen/bin/python"; RLPY = R + "/envs/rl/bin/python"; DATA = R + "/data/MATH-500/test.jsonl"
ENV = "cd %s/code/P1 && export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONPATH=%s/code/P1 TOKENIZERS_PARALLELISM=false P1_ALLOWED_GPUS=5,6,7 && " % (R, R)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); n = 0


def w(fn, name, pre, cmd):
    global n
    (out / (fn + ".task")).write_text("%s\n%s\n%s\n" % (name, pre, cmd)); n += 1


def gen_cmd(model, tag, gdir, nprob, samples, cap, T, P, tpl, seed, data=DATA, logs=O + "/logs"):
    return (ENV + "mkdir -p %s && CUDA_VISIBLE_DEVICES=$G %s -m exp.pipeline.gen --model %s --tag %s --data %s --n %d --samples %d --max_tokens %d --temperature %s --top_p %s --template %s --seed %d --gpu_util 0.9 --no_tokens --out %s/%s.jsonl > %s/gen_%s.log 2>&1 && "
            "%s -m exp.pipeline.grade --gen %s/%s.jsonl --data %s --out_dir %s >> %s/grade.log 2>&1 && %s -m exp.pipeline.indices --gen %s/%s.graded.jsonl.gz --out %s/%s.idx.json >> %s/idx.log 2>&1"
            % (gdir, PY, model, tag, data, nprob, samples, cap, T, P, tpl, seed, gdir, tag.replace("@", "_"), logs, tag.replace("@", "_"), RLPY, gdir, tag.replace("@", "_"), data, gdir, logs, PY, gdir, tag.replace("@", "_"), gdir, tag.replace("@", "_"), logs))


# A) more decoding seeds on S1 base/final (T=1.0, 500 x 2): seeds 2..5
for sd in (2, 3, 4, 5):
    for c in ("final", "base"):
        tag = "S1sd%d@%s" % (sd, c)
        w("9_e1_S1sd%d_%s" % (sd, c), "e1_S1sd%d_%s" % (sd, c), "%s/models/S1/%s/config.json" % (R, c), gen_cmd("%s/models/S1/%s" % (R, c), tag, "%s/S1sd%d/gen" % (O, sd), 500, 2, 8192, 1.0, 1.0, "olmo_rlzero", sd))
# B) S2 second decoding seed (Think, T=0.6, 300 x 2)
for c, path in (("base", "dpo"), ("1375", "1375")):
    tag = "S2sd1@%s" % c
    w("9_e1_S2sd1_%s" % c, "e1_S2sd1_%s" % c, "%s/models/S2/%s/config.json" % (R, path), gen_cmd("%s/models/S2/%s" % (R, path), tag, "%s/S2sd1/gen" % O, 300, 2, 12288, 0.6, 0.95, "olmo_think", 1))
# C) S1 E3 with fresh prefixes (seed 1) and fresh samples at base / 1900 / final, both prefix sources, both judges
D = O + "/S1seed1/insertion"
g1 = lambda c: "%s/S1/gen/S1@%s.graded.jsonl.gz" % (O, c)
w("9_e3s1_prefixes", "e3s1_prefixes", g1("final") + " " + g1("base"),
  ENV + "mkdir -p %s && %s -m exp.pipeline.insert prefixes --gen %s --data %s --n_prefixes 300 --src late --seed 1 --out %s/prefixes.jsonl > %s/logs/e3s1_pre.log 2>&1 && %s -m exp.pipeline.insert prefixes --gen %s --data %s --n_prefixes 150 --src base --seed 1 --out %s/prefixes_base.jsonl >> %s/logs/e3s1_pre.log 2>&1" % (D, RLPY, g1("final"), DATA, D, O, RLPY, g1("base"), DATA, D, O))
for c in ("base", "1900", "final"):
    for src in ("late", "basesrc"):
        pre = D + ("/prefixes.jsonl" if src == "late" else "/prefixes_base.jsonl")
        cn = "conts_%s_S1@%s" % (src, c); sn = "scored_%s_S1@%s" % (src, c)
        w("9_e3s1_%s_%s" % (c, src), "e3s1_%s_%s" % (c, src), pre,
          ENV + "CUDA_VISIBLE_DEVICES=$G %s -m exp.pipeline.insert continue --model %s/models/S1/%s --tag S1@%s --prefixes %s --arms wait,next,hold_on,none --samples 4 --max_tokens 1536 --seed 1 --gpu_util 0.9 --out %s/%s.jsonl.gz > %s/logs/%s.log 2>&1 && %s -m exp.pipeline.insert score --conts %s/%s.jsonl.gz --prefixes %s --out %s/%s.jsonl.gz >> %s/logs/e3s1_score.log 2>&1"
          % (PY, R, c, c, pre, D, cn, O, cn, RLPY, D, cn, pre, D, sn, O))
for src, pre in (("late", D + "/prefixes.jsonl"), ("basesrc", D + "/prefixes_base.jsonl")):
    sc = " ".join("%s/scored_%s_S1@%s.jsonl.gz" % (D, src, c) for c in ("base", "1900", "final"))
    for tag, model, extra, suf in (("qwen", R + "/models/Qwen2.5-7B-Instruct", "", ""), ("oss", R + "/models/gpt-oss-20b", "--mode harmony --effort low ", "_oss")):
        w("9_e3s1_judge_%s_%s" % (src, tag), "e3s1_judge_%s_%s" % (src, tag), sc,
          ENV + "CUDA_VISIBLE_DEVICES=$G %s -m exp.pipeline.judge --model %s %s--prefixes %s --scored %s --gpu_util 0.9 --out %s/judged_%s%s.jsonl.gz > %s/logs/e3s1_judge_%s_%s.log 2>&1" % (PY, model, extra, pre, sc, D, src, suf, O, src, tag))
# D) E1 seed-1 replicates for every valid backbone endpoint (MATH-500[:200] x 4, same template/T as the main panel)
import bb_fams as B   # noqa: E402
SKIP = {"mistral_base", "qwen_math15", "r1d1p5", "magistral"}
LOWCAP = ("olmo2", "yi_")   # 4096-token context: E1 cap 3500
for fam, models in B.FAMS:
    for mid, path, tpl, T, P, trust in models:
        if mid in SKIP:
            continue
        env = ENV + ("export P1_TRUST_REMOTE=1 && " if trust else "")
        c = gen_cmd(path, "BB_%s@s1" % mid, "%s/BB_%s/gen_seed1" % (S, mid), 200, 4, 3000 if mid.startswith(LOWCAP) else 8192, T, P, tpl, 1, logs=S + "/BB_" + mid + "/logs")
        c = c.replace(ENV, env, 1)
        w("9_z_e1s1_%s" % mid, "e1s1_" + mid, (path + "/.dl_ok") if "/models/BB/" in path else (path + "/config.json"), c)
# E) the other decoding temperature (1.0 <-> 0.6/0.95) for every valid backbone endpoint, E1 only (decoding-confound control for the breadth panel)
for fam, models in B.FAMS:
    for mid, path, tpl, T, P, trust in models:
        if mid in SKIP:
            continue
        T2, P2 = (0.6, 0.95) if T >= 0.99 else (1.0, 1.0)
        if T < 0.2:
            continue
        env = ENV + ("export P1_TRUST_REMOTE=1 && " if trust else "")
        c = gen_cmd(path, "BB_%s@alt" % mid, "%s/BB_%s/gen_altT" % (S, mid), 200, 4, 3000 if mid.startswith(LOWCAP) else 8192, T2, P2, tpl, 0, logs=S + "/BB_" + mid + "/logs").replace(ENV, env, 1)
        w("9_zy_e1alt_%s" % mid, "e1alt_" + mid, (path + "/.dl_ok") if "/models/BB/" in path else (path + "/config.json"), c)
# F) LSC occurrence embeddings (fixed encoder = the lineage's earlier model) where wait exists: SFT vs RL pairs, S1-pre stage 2/3, S2-ctl checkpoints
def lsc(name, enc, genp, tpl, out, trust=False):
    pre = genp
    cmd = ENV + ("export P1_TRUST_REMOTE=1 && " if trust else "") + "mkdir -p %s && CUDA_VISIBLE_DEVICES=$G %s -m exp.pipeline.lsc_embed --encoder %s --gen %s --data %s --template %s --n 500 --seed 0 --out %s > %s/logs/lsc_%s.log 2>&1" % (str(Path(out).parent), PY, enc, genp, DATA, tpl, out, O, name)
    w("9_zz_lsc_" + name, "lscx_" + name, pre, cmd)
for mid, enc, tpl, trust in (("phi4r", "phi4", "qwen_boxed", False), ("phi4rp", "phi4", "qwen_boxed", False), ("mimo_sft", "mimo_base", "qwen_boxed", True), ("mimo_rl", "mimo_base", "qwen_boxed", True),
                             ("mimo_rlzero", "mimo_base", "qwen_boxed", True), ("orn7", "qwen7b", "qwen_boxed", False), ("acer", "qwen7b", "qwen_boxed", False), ("qwen3_8b", "qwen3_8b_base", "qwen_boxed", False),
                             ("glmz1", "glm4", "qwen_boxed", False), ("smollm3", "smollm3_base", "qwen_boxed", False)):
    encp = R + "/models/BB/" + enc if enc != "qwen7b" else R + "/models/Qwen2.5-7B"
    lsc("bb_" + mid, encp, "%s/BB_%s/gen/BB_%s_math500.graded.jsonl.gz" % (S, mid, mid), tpl, "%s/BB_%s/lsc/emb.npz" % (S, mid), trust)
for br in ("stage2-step47684", "stage3-step6000"):
    lsc("s1pre_" + br, R + "/models/S1/base", "%s/S1pre/gen/S1pre@%s.graded.jsonl.gz" % (O, br), "olmo_rlzero", "%s/S1pre/lsc/emb_%s.npz" % (O, br))
for c in ("step5000", "step15000", "step25000", "step35000", "step43000"):
    lsc("s2ctl_" + c, R + "/models/S2/dpo", "%s/S2ctl/gen/S2ctl@%s.graded.jsonl.gz" % (O, c), "olmo_think", "%s/S2ctl/lsc/emb_%s.npz" % (O, c))
print("wrote", n, "tasks (incl. E, F)")

# G) wave 6 (2026-10-04): temperature sweep of the headline cell (S1 base / 1900 / final at T in {0.3, 0.8, 1.2}; T=0.6 and 1.0 exist) and a larger E1 (all 500 MATH-500 problems x 4) for the popular endpoints
for T in (0.3, 0.8, 1.2):
    for c in ("final", "1900", "base"):
        tag = "S1T%02d@%s" % (int(round(T * 10)), c)
        w("9_t_S1T%02d_%s" % (int(round(T * 10)), c), "e1_S1T%02d_%s" % (int(round(T * 10)), c), "%s/models/S1/%s/config.json" % (R, c),
          gen_cmd("%s/models/S1/%s" % (R, c), tag, "%s/S1T%02d/gen" % (O, int(round(T * 10))), 500, 2, 8192, T, 1.0, "olmo_rlzero", 0))
BIG = ("phi4r", "phi4rp", "qwen3_8b", "qwen3_14b", "r1d_llama8", "mimo_rl", "mimo_sft", "glmz1", "r1d7", "acer", "smollm3", "qwen35_9b", "qwen35_4b")
for fam, models in B.FAMS:
    for mid, path, tpl, T, P, trust in models:
        if mid in BIG:
            env = ENV + ("export P1_TRUST_REMOTE=1 && " if trust else "") + ("export P1_MAX_NUM_SEQS=128 && " if mid.startswith("qwen35") else "")
            c = gen_cmd(path, "BB_%s@n500" % mid, "%s/BB_%s/gen_n500" % (S, mid), 500, 4, 8192, T, P, tpl, 2, logs=S + "/BB_" + mid + "/logs").replace(ENV, env, 1)
            w("9_u_e1n500_%s" % mid, "e1n500_" + mid, (path + "/.dl_ok") if "/models/BB/" in path else (path + "/config.json"), c)
print("wrote", n, "tasks (incl. G)")
