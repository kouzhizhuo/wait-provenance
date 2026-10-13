"""Writes the appended hostA queue tasks (3 lines each: name / prerequisite files / command) for the post-interruption batch:
S2 E2 SDT, S2 E3 (late + base prefixes, Qwen + gpt-oss-20b judges), S2 aggregation, LSC occurrence embeddings for S1 and S2.

    python3 exp/gen_tasks_s2.py exp/tasks_d      # then rsync to $OUT/tasks.d on hostA (the running queue picks them up)

Names/ordering: files starting with 0 run first, then the built-in list, then the rest alphabetically (3_ < 4_ < 5_ < 6_ < 7_).
"""
import sys
from pathlib import Path

R = "$ARR_ROOT"
OUT = R + "/runs/P1/main"
PY = R + "/envs/gen/bin/python"
RLPY = R + "/envs/rl/bin/python"
DATA = R + "/data/MATH-500/test.jsonl"
JUDGE = R + "/models/Qwen2.5-7B-Instruct"
OSS = R + "/models/gpt-oss-20b"
CK2 = ["base", "1375", "500", "25", "100", "250", "750", "1000"]
S1C = ["final", "base", "1900", "700", "100", "400", "1000", "1400", "200"]
INS = OUT + "/S2/insertion"
G = "$G"


def s2m(c):
    return R + "/models/S2/" + ("dpo" if c == "base" else c)


def gr(s, c):
    return "%s/%s/gen/%s@%s.graded.jsonl.gz" % (OUT, s, s, c)


tasks = []   # (filename, name, prereq, cmd)


def add(fn, name, pre, cmd):
    tasks.append((fn, name, pre if pre else "-", cmd))


# ---- E2 SDT on S2 (cheap)
for c in CK2:
    add("3_e2_S2_%s" % c, "e2_S2_%s" % c, gr("S2", c),
        "mkdir -p {o}/S2/sdt; {rl} -m exp.pipeline.sdt prefixes --gen {g} --data {d} --n_problems 300 --template olmo_think --tokenizer {m} --out {o}/S2/sdt/prefixes_{c}.jsonl >> {o}/logs/e2.log 2>&1 && "
        "CUDA_VISIBLE_DEVICES={G} {py} -m exp.pipeline.insert continue --model {m} --tag S2@{c} --prefixes {o}/S2/sdt/prefixes_{c}.jsonl --arms none --samples 4 --max_tokens 300 --seed 0 --gpu_util 0.9 "
        "--out {o}/S2/sdt/conts_{c}.jsonl.gz > {o}/logs/e2_S2_{c}.log 2>&1".format(o=OUT, rl=RLPY, py=PY, g=gr("S2", c), d=DATA, m=s2m(c), c=c, G=G))
sd = " ".join("%s/S2/sdt/conts_%s.jsonl.gz" % (OUT, c) for c in CK2)
add("5_sdt_S2", "sdt_S2", sd,
    "{rl} -m exp.pipeline.sdt score --conts {sd} --out {o}/S2/sdt.json --series S2 --keys_series s2 --stage_map early=25,mid=500,late=1375 > {o}/logs/sdt_score_S2.log 2>&1".format(rl=RLPY, sd=sd, o=OUT))

# ---- E3 on S2: prefixes, continuations + scoring
add("3_e3pre_S2", "e3pre_S2", gr("S2", "1375") + " " + gr("S2", "base"),
    "mkdir -p {i}; {rl} -m exp.pipeline.insert prefixes --gen {g1} --data {d} --n_prefixes 300 --src late --template olmo_think --tokenizer {m1} --seed 0 --out {i}/prefixes.jsonl > {o}/logs/e3pre_S2.log 2>&1 && "
    "{rl} -m exp.pipeline.insert prefixes --gen {g0} --data {d} --n_prefixes 150 --src base --template olmo_think --tokenizer {m0} --seed 0 --out {i}/prefixes_base.jsonl >> {o}/logs/e3pre_S2.log 2>&1".format(
        i=INS, rl=RLPY, g1=gr("S2", "1375"), g0=gr("S2", "base"), d=DATA, m1=s2m("1375"), m0=s2m("base"), o=OUT))
for k, c in enumerate(CK2):
    for src in ("late", "basesrc"):
        arms = "wait,next,hold_on,none"
        if c in ("base", "500", "1375") and src == "late":
            arms += ",okay,hmm"
        pre = INS + ("/prefixes.jsonl" if src == "late" else "/prefixes_base.jsonl")
        cn = ("conts_S2@%s" if src == "late" else "conts_base_S2@%s") % c
        sn = ("scored_S2@%s" if src == "late" else "scored_base_S2@%s") % c
        add("4_%02d_e3_S2_%s_%s" % (k, c, src), "e3_S2_%s_%s" % (c, src), pre,
            "CUDA_VISIBLE_DEVICES={G} {py} -m exp.pipeline.insert continue --model {m} --tag S2@{c} --prefixes {pre} --arms {arms} --samples 4 --max_tokens 1536 --seed 0 --gpu_util 0.9 --out {i}/{cn}.jsonl.gz > {o}/logs/{cn}.log 2>&1 && "
            "{rl} -m exp.pipeline.insert score --conts {i}/{cn}.jsonl.gz --prefixes {pre} --out {i}/{sn}.jsonl.gz >> {o}/logs/score_S2.log 2>&1".format(
                G=G, py=PY, rl=RLPY, m=s2m(c), c=c, pre=pre, arms=arms, i=INS, cn=cn, sn=sn, o=OUT))
sc_late = " ".join("%s/scored_S2@%s.jsonl.gz" % (INS, c) for c in CK2)
sc_base = " ".join("%s/scored_base_S2@%s.jsonl.gz" % (INS, c) for c in CK2)
for tag, mdl, extra, suf in (("", JUDGE, "", ""), ("oss_", OSS, "--mode harmony --effort low ", "_oss")):
    add("5_%sjudge_S2_late" % tag, "%sjudge_S2_late" % tag, sc_late,
        "CUDA_VISIBLE_DEVICES={G} {py} -m exp.pipeline.judge --model {m} {x}--prefixes {i}/prefixes.jsonl --scored {s} --gpu_util 0.9 --out {i}/judged{suf}.jsonl.gz > {o}/logs/{t}judge_S2_late.log 2>&1".format(
            G=G, py=PY, m=mdl, x=extra, i=INS, s=sc_late, suf=suf, o=OUT, t=tag))
    add("5_%sjudge_S2_base" % tag, "%sjudge_S2_base" % tag, sc_base,
        "CUDA_VISIBLE_DEVICES={G} {py} -m exp.pipeline.judge --model {m} {x}--prefixes {i}/prefixes_base.jsonl --scored {s} --gpu_util 0.9 --out {i}/judged_base{suf}.jsonl.gz > {o}/logs/{t}judge_S2_base.log 2>&1".format(
            G=G, py=PY, m=mdl, x=extra, i=INS, s=sc_base, suf=suf, o=OUT, t=tag))
agg = ("{py} -m exp.pipeline.report --res {o}/S2 --figdir {o}/S2/plots --series S2 --keys_series s2 --ckpts base:0,25:25,100:100,250:250,500:500,750:750,1000:1000,1375:1375 --last 1375 "
       "--stage_map early=25,mid=500,late=1375 --main --no_plots{j} > {o}/logs/agg_S2_e3{suf}.log 2>&1")
add("6_agg_S2_e3", "agg_S2_e3", "%s/judged.jsonl.gz %s/judged_base.jsonl.gz" % (INS, INS), agg.format(py=PY, o=OUT, j="", suf=""))
add("6_agg_S2_e3_oss", "agg_S2_e3_oss", "%s/judged_oss.jsonl.gz %s/judged_base_oss.jsonl.gz" % (INS, INS), agg.format(py=PY, o=OUT, j=" --judge_tag oss", suf="_oss"))

# ---- LSC occurrence embeddings: fixed encoder = the series' base model; 500 occurrences per checkpoint
for s, cks, enc, tmpl, tokflag in (("S1", S1C, R + "/models/S1/base", "olmo_rlzero", ""), ("S2", CK2, R + "/models/S2/dpo", "olmo_think", "")):
    for c in cks:
        add("7_lsc_%s_%s" % (s, c), "lsc_%s_%s" % (s, c), gr(s, c),
            "mkdir -p {o}/{s}/lsc; CUDA_VISIBLE_DEVICES={G} {py} -m exp.pipeline.lsc_embed --encoder {e} --gen {g} --data {d} --template {t} --n 500 --seed 0 --out {o}/{s}/lsc/emb_{c}.npz > {o}/logs/lsc_{s}_{c}.log 2>&1".format(
                o=OUT, s=s, G=G, py=PY, e=enc, g=gr(s, c), d=DATA, t=tmpl, c=c))

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for fn, name, pre, cmd in tasks:
    (out / (fn + ".task")).write_text("%s\n%s\n%s\n" % (name, pre, cmd))
print("wrote %d tasks to %s" % (len(tasks), out))
