"""DEV keys for Calibrated Wait: h5.<model>.dev.* from exp/results/cw/dev/<model>/{calib,s4,match,taus}.json + prefixes meta.  -> exp/results/reframe/cw_dev_keys.json (+ copy in paper/numbers/keys/)."""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "cw" / "dev"
ent = []


def put(k, v, kind, src):
    ent.append({"key": k, "value": v, "kind": kind, "src": src})


for md in sorted(D.iterdir()):
    if not (md / "s4.json").exists():
        continue
    m = md.name; src = "exp/results/cw/dev/%s" % m
    cal = json.load(open(md / "calib.json")); s4 = json.load(open(md / "s4.json"))
    pm = json.load(open(md / "prefixes.jsonl.meta.json")).get("extra", {}) if (md / "prefixes.jsonl.meta.json").exists() else {}
    gm = json.load(open(md / "dev_gen.jsonl.meta.json")).get("config", {}) if (md / "dev_gen.jsonl.meta.json").exists() else {}
    put("h5.%s.dev.n_problems" % m, gm.get("n"), "int", src); put("h5.%s.dev.n_pairs" % m, pm.get("n_problems_selected"), "int", src)
    put("h5.%s.dev.tau" % m, cal["tau"], "dec4", src + "/calib.json"); put("h5.%s.dev.rho_lo95" % m, cal.get("rho_lo95_A"), "dec3", src + "/calib.json")
    put("h5.%s.dev.hr_vanilla_A" % m, cal["hr_vanilla_A"], "dec3", src); put("h5.%s.dev.fa_vanilla_A" % m, cal["fa_vanilla_A"], "dec3", src)
    for fam in ("B_all", "B_signflip"):
        for cond in ("vanilla", "cw"):
            r = s4["%s.%s" % (fam, cond)]
            for q, nm in (("HR", "hr"), ("FA", "fa"), ("dprime", "dprime"), ("criterion", "crit")):
                for s, sfx in (("est", ""), ("lo", ".lo"), ("hi", ".hi")):
                    put("h5.%s.dev.%s.%s.%s%s" % (m, fam, cond, nm, sfx), r[q][s], "dec3", src + "/s4.json")
    if (md / "s5.json").exists():
        s5 = json.load(open(md / "s5.json")); sa = src + "/s5.json (S5 shared-prefix branching, natural DEV-B marker events)"
        put("h5.%s.s5.n_events" % m, s5["n_events"], "int", sa); put("h5.%s.s5.n_gated" % m, s5["n_suppress_labelled"], "int", sa); put("h5.%s.s5.n_keep" % m, s5["n_keep"], "int", sa); put("h5.%s.s5.n_problems" % m, s5["n_problems"], "int", sa)
        for q in ("est", "lo", "hi"):
            put("h5.%s.s5.diff.%s" % (m, q), s5["diff_sup_minus_keep"][q], "dec3", sa); put("h5.%s.s5.gain_gated.%s" % (m, q), s5["gain_suppress_labelled"][q], "dec3", sa); put("h5.%s.s5.gain_keep.%s" % (m, q), s5["gain_keep"][q], "dec3", sa)
        put("h5.%s.s5.acc_allow" % m, s5["acc_allow"], "dec3", sa); put("h5.%s.s5.acc_suppress" % m, s5["acc_suppress"], "dec3", sa); put("h5.%s.s5.pass" % m, int(s5["pass"]), "int", sa)
    if (md / "s4_sham.json").exists():
        sh = json.load(open(md / "s4_sham.json"))
        for fam in ("B_all", "B_signflip"):
            r = sh["%s.sham" % fam]
            for q, nm in (("HR", "hr"), ("FA", "fa"), ("dprime", "dprime"), ("criterion", "crit")):
                for s_, sfx in (("est", ""), ("lo", ".lo"), ("hi", ".hi")):
                    put("h5.%s.dev.%s.sham.%s%s" % (m, fam, nm, sfx), r[q][s_], "dec3", src + "/s4_sham.json (continuations through the processor path, mode sham)")
    if (md / "match.json").exists():
        mt = json.load(open(md / "match.json"))
        for k in ("m_vanilla", "m_cw", "cw_ratio", "alpha_star", "bf_cap_match", "bf_cap_loose", "rand_dev_ratio_to_cw"):
            put("h5.%s.dev.match.%s" % (m, k), mt.get(k), "dec3", src + "/match.json")
    if (md / "taus.json").exists():
        t = json.load(open(md / "taus.json"))
        for k in ("tau_rho75", "tau_rho50"):
            put("h5.%s.dev.%s" % (m, k), t.get(k), "dec4", src + "/taus.json (exploratory, set after DEV; D50g)")
    if (md / "abl.json").exists():
        ab = json.load(open(md / "abl.json")); sa = src + "/abl.json (offline ablation on existing DEV traces; D50f rule on DEV-A, scored on DEV-B)"
        for arm in ("cw", "lp", "waitonly"):
            r = ab[arm]; p = "h5.%s.dev.abl.%s" % (m, "cw_off" if arm == "cw" else arm)
            put(p + ".tau", r["tau"], "dec4", sa); put(p + ".rho", r["rho"], "dec3", sa); put(p + ".rho.lo", r["rho_lo95"], "dec3", sa)
            for q in ("hr", "fa", "hr_vanilla", "fa_vanilla"):
                put("%s.%s" % (p, q.replace("_", ".")), r[q][0], "dec3", sa); put("%s.%s.lo" % (p, q.replace("_", ".")), r[q][1], "dec3", sa); put("%s.%s.hi" % (p, q.replace("_", ".")), r[q][2], "dec3", sa)
out = ROOT / "results" / "reframe" / "cw_dev_keys.json"
json.dump({"entries": ent}, open(out, "w"), indent=0, default=float)
shutil.copy(out, ROOT.parent / "paper" / "numbers" / "keys" / "cw_dev_keys.json")
print(len(ent), "keys for", [m.name for m in D.iterdir() if (m / "s4.json").exists()])
