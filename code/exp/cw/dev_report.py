import json, sys
D = sys.argv[1]
cal = json.load(open(D + "/calib.json")); s4 = json.load(open(D + "/s4.json"))
pre = [json.loads(l) for l in open(D + "/prefixes.jsonl")]
meta = json.load(open(D + "/dev_gen.jsonl.meta.json"))["config"]
pairs = len(pre) // 2
f = lambda x: "%.3f [%.3f, %.3f]" % (x["est"], x["lo"], x["hi"])
print("n_problems_dev=%s n_pairs=%d | tau=%.4f found=%s rho_lo95_A=%s HRvan_A=%.3f FAvan_A=%.3f FAgated_A_approx=%.3f" % (meta.get("n"), pairs, cal["tau"], cal["found"], cal.get("rho_lo95_A"), cal["hr_vanilla_A"], cal["fa_vanilla_A"], cal["fa_gated_A_approx"]))
for k in ("B_all", "B_signflip"):
    for c in ("vanilla", "cw"):
        r = s4["%s.%s" % (k, c)]
        print("%-10s %-8s HR %s FA %s d' %s c %s" % (k, c, f(r["HR"]), f(r["FA"]), f(r["dprime"]), f(r["criterion"])))
