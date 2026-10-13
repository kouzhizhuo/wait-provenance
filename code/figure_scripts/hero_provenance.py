"""HERO figure (P1): provenance map of `wait`. Reads exp/results/reframe/provenance_map.json (rows by model name; PRELIM) and nothing else; band assignment,
display labels and start->end pairs are categorical metadata (the data file's `group` field sets the recipe; mismatches are printed).
One row per model series. x = wait per 1k tokens (log; the exact-zero slot "0" sits left of an axis break). Hollow dot = starting checkpoint, filled = trained
descendant, arrow = what training did, whisker = 95% CI. Blue = after RL.
    python3 figures/scripts/hero_provenance.py [--full]   ->  figures/fig_hero_provenance.pdf (+png)
"""
import json
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
import re
from scipy.stats import chi2
import _common as C
from _common import P

BANDS = ["fine-tuned on reasoning traces",
         "post-training on a base that already writes it (Qwen3.5)",
         "RL on a model that already writes wait",
         "enters with mid-training, then RL (OLMo-3)",
         "zero-RL from a marker-free base",
         "RL from a marker-free instruct model",
         "instruct models without traces",
         "pretrained bases not shown above",
         "undisclosed training"]
T_, Q35, R_, M_, Z_, RI, IN, BA, UN = BANDS
# (band, label, start model, end model, kind[, group the data file must give the end model])
FULL = {"full": True}   # row only in the appendix version
M = [
    (T_, "Qwen2.5-32B → DeepSeek-R1-Distill-Qwen-32B", "S8 base", "S8 r1distill", "tr"),
    (T_, "Qwen2.5-32B → QwQ-32B (cold start + RL)", "S8 base", "S8 qwq", "tr"),
    (T_, "Qwen2.5-7B → DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-7B", "R1-Distill-Qwen-7B", "tr"),
    (T_, "DeepSeek-R1-Distill-Llama-8B (base gated)", None, "R1-Distill-Llama-8B", "tr"),
    (T_, "Phi-4 → Phi-4-reasoning", "Phi-4", "Phi-4-reasoning", "tr"),
    (T_, "Qwen3-8B-Base → Qwen3-8B", "Qwen3-8B-Base", "Qwen3-8B", "tr"),
    (T_, "Qwen3-14B-Base → Qwen3-14B", "Qwen3-14B-Base", "Qwen3-14B", "tr"),
    (T_, "Qwen3-4B-Base → Qwen3-4B", "Qwen3-4B-Base", "Qwen3-4B", "tr", FULL),
    (T_, "Qwen3-1.7B-Base → Qwen3-1.7B", "Qwen3-1.7B-Base", "Qwen3-1.7B", "tr", FULL),
    (T_, "Qwen2.5-7B-Instruct → OpenThinker-7B", "Qwen2.5-7B-Instruct", "OpenThinker-7B", "tr", FULL),
    (T_, "Qwen2.5-7B-Instruct → Bespoke-Stratos-7B", "Qwen2.5-7B-Instruct", "Bespoke-Stratos-7B", "tr", FULL),
    (T_, "Qwen3.8-27B (base n/a)", None, "Qwen3.8-27B", "tr"),
    (T_, "GLM-4-9B-0414 → GLM-Z1-9B-0414", "GLM-4-9B-0414", "GLM-Z1-9B-0414", "tr"),
    (T_, "MiMo-7B-Base → MiMo-7B-SFT", "MiMo-7B-Base", "MiMo-7B-SFT", "tr"),
    (T_, "SmolLM3-3B-Base → SmolLM3-3B", "SmolLM3-3B-Base", "SmolLM3-3B", "tr"),
    (T_, "OpenReasoning-Nemotron-7B", None, "OpenReasoning-Nemotron-7B", "tr"),
    (Q35, "Qwen3.5-0.8B-Base → Qwen3.5-0.8B", "Qwen3.5-0.8B-Base", "Qwen3.5-0.8B", "tr"),
    (Q35, "Qwen3.5-2B-Base → Qwen3.5-2B", "Qwen3.5-2B-Base", "Qwen3.5-2B", "tr"),
    (Q35, "Qwen3.5-4B-Base → Qwen3.5-4B", "Qwen3.5-4B-Base", "Qwen3.5-4B", "tr"),
    (Q35, "Qwen3.5-9B-Base → Qwen3.5-9B", "Qwen3.5-9B-Base", "Qwen3.5-9B", "tr"),
    (R_, "Phi-4-reasoning → Phi-4-reasoning-plus", "Phi-4-reasoning", "Phi-4-reasoning-plus", "rl"),
    (R_, "Olmo-3-7B-Think: SFT, DPO → RL", "OLMo-3-7B Think-DPO (S2 base)", "OLMo-3-7B Think (S2 last)", "rl"),
    (R_, "Olmo-3.1-32B-Think: SFT, DPO → RL", "OLMo-3-32B Think-DPO (S9 base)", "OLMo-3.1-32B Think (S9 step 2300)", "rl"),
    (R_, "MiMo-7B-SFT → MiMo-7B-RL", "MiMo-7B-SFT", "MiMo-7B-RL", "rl"),
    (R_, "MiMo-7B-Base → MiMo-7B-RL-Zero", "MiMo-7B-Base", "MiMo-7B-RL-Zero", "rl"),
    (R_, "DeepSeek-R1-Distill-Qwen-7B → AceReason-Nemotron-7B", "R1-Distill-Qwen-7B", "AceReason-Nemotron-7B", "rl"),
    (R_, "DeepSeek-R1-Distill-Qwen-1.5B → DeepScaleR-1.5B-Preview", None, "DeepScaleR-1.5B-Preview", "rl", FULL),
    (R_, "DeepSeek-R1-Distill-Qwen-14B → Light-R1-14B-DS", None, "Light-R1-14B-DS", "rl", FULL),
    (M_, "Olmo-3-1025-7B: pretraining → mid-training", "OLMo-3-7B pretraining step 1.41M", "OLMo-3-7B mid-training", "mid"),
    (M_, "Olmo-3-7B: stage-3 base → RL-Zero-Math", "OLMo-3-7B base (S1 base)", "OLMo-3-7B RL-Zero-Math (S1 final)", "rl"),
    (Z_, "Qwen2.5-32B → DAPO-Qwen-32B", "S8 base", "S8 dapo", "z", {"gate": "s8dapo"}),
    (Z_, "Qwen2.5-32B → Open-Reasoner-Zero-32B", "S8 base", "Qwen2.5-32B zero-RL (Open-Reasoner-Zero-32B)", "z", {"gate": "orz32"}),
    (Z_, "Qwen2.5-Math-7B → REF-7B (ours)", "REF 7B base (step 0)", "REF 7B final (step 400)", "z", {"gate": "s7r7b"}),
    (Z_, "Qwen2.5-Math-1.5B → REF-s1 (ours)", "REF seed 1 base (step 0)", "REF seed 1 final (step 400)", "z", {"gate": "s7r1"}),
    (Z_, "Qwen2.5-Math-1.5B → REF-s2 (ours)", "REF seed 2 base (step 0)", "REF seed 2 final (step 400)", "z", {"gate": "s7r2"}),
    (Z_, "Qwen2.5-0.5B → Qwen-2.5-0.5B-SimpleRL-Zoo", "Qwen2.5-0.5B base (SimpleRL-Zoo template)", "Qwen2.5-0.5B zero-RL (Qwen-2.5-0.5B-SimpleRL-Zoo)", "z", {"gate": "sz_q05"}),
    (Z_, "Qwen2.5-1.5B → Qwen-2.5-1.5B-SimpleRL-Zoo", "Qwen2.5-1.5B base (SimpleRL-Zoo template)", "Qwen2.5-1.5B zero-RL (Qwen-2.5-1.5B-SimpleRL-Zoo)", "z", {"gate": "sz_q15"}),
    (Z_, "Qwen2.5-7B → Qwen-2.5-7B-SimpleRL-Zoo", "Qwen2.5-7B base (SimpleRL-Zoo template)", "Qwen2.5-7B zero-RL (Qwen-2.5-7B-SimpleRL-Zoo)", "z", {"gate": "sz_q7"}),
    (Z_, "Qwen2.5-7B → Open-Reasoner-Zero-7B", "Qwen2.5-7B base (SimpleRL-Zoo template)", "Qwen2.5-7B zero-RL (Open-Reasoner-Zero-7B)", "z", {"gate": "orz7b"}),
    (Z_, "Qwen2.5-Math-7B → Qwen-2.5-Math-7B-SimpleRL-Zoo", "Qwen2.5-Math-7B base (SimpleRL-Zoo template)", "Qwen2.5-Math-7B zero-RL (Qwen-2.5-Math-7B-SimpleRL-Zoo)", "z", {"gate": "sz_qm7"}),
    (Z_, "Qwen2.5-Math-7B → Qwen2.5-Math-7B-Oat-Zero", "Qwen2.5-Math-7B base (SimpleRL-Zoo template)", "Qwen2.5-Math-7B zero-RL (Qwen2.5-Math-7B-Oat-Zero)", "z", {"gate": "oat7"}),
    (Z_, "Qwen2.5-14B → Qwen-2.5-14B-SimpleRL-Zoo", "Qwen2.5-14B base (SimpleRL-Zoo template)", "Qwen2.5-14B zero-RL (Qwen-2.5-14B-SimpleRL-Zoo)", "z", {"gate": "sz_q14"}),
    (Z_, "Qwen2.5-32B → Qwen-2.5-32B-SimpleRL-Zoo", "Qwen2.5-32B base (SimpleRL-Zoo template)", "Qwen2.5-32B zero-RL (Qwen-2.5-32B-SimpleRL-Zoo)", "z", {"gate": "sz_q32"}),
    (Z_, "Mistral-Small-24B → Mistral-Small-24B-SimpleRL-Zoo", "Mistral-Small-24B base (SimpleRL-Zoo template)", "Mistral-Small-24B zero-RL (Mistral-Small-24B-SimpleRL-Zoo)", "z", {"gate": "sz_m24"}),
    (Z_, "Mistral-7B-v0.1 → Mistral-7B-v0.1-SimpleRL-Zoo", "Mistral-7B-v0.1 base (SimpleRL-Zoo template)", "Mistral-7B-v0.1 zero-RL (Mistral-7B-v0.1-SimpleRL-Zoo)", "z", {"gate": "sz_m7"}),
    (Z_, "Qwen3-4B-Base → General-Reasoner-Qwen3-4B", "@zr:gr3_4:base", "Qwen3-4B-Base zero-RL (General-Reasoner-Qwen3-4B)", "cx", {"gate": "gr3_4"}),
    (Z_, "Qwen3-14B-Base → General-Reasoner-Qwen3-14B", "@zr:gr3_14:base", "Qwen3-14B-Base zero-RL (General-Reasoner-Qwen3-14B)", "z", {"gate": "gr3_14"}),
    (Z_, "Qwen2.5-7B → General-Reasoner-Qwen2.5-7B", "@zr:gr7:base", "Qwen2.5-7B zero-RL (General-Reasoner-Qwen2.5-7B)", "z", {"gate": "gr7"}),
    (Z_, "Qwen2.5-14B → General-Reasoner-Qwen2.5-14B", "@zr:gr14:base", "Qwen2.5-14B zero-RL (General-Reasoner-Qwen2.5-14B)", "z", {"gate": "gr14"}),
    (Z_, "Qwen2.5-Math-1.5B → One-Shot-RLVR (pi1)", "@zr:os15:base", "Qwen2.5-Math-1.5B zero-RL (One-Shot-RLVR-Qwen2.5-Math-1.5B-pi1)", "z", {"gate": "os15"}),
    (Z_, "Qwen2.5-Math-1.5B → Qwen2.5-Math-1.5B-Oat-Zero", "@zr:oat15:base", "Qwen2.5-Math-1.5B zero-RL (Qwen2.5-Math-1.5B-Oat-Zero)", "z", {"gate": "oat15"}),
    (RI, "Qwen2.5-1.5B-Instruct → GRPO (ours)", "Qwen2.5-1.5B-Instruct (S3a base)", "Qwen2.5-1.5B GRPO (S3a last)", "rl"),
    (RI, "Llama-3.1-Tulu-3-8B-DPO → Tulu-3.1-8B (RLVR)", "Tulu-3-8B-DPO (S6 start)", "Tulu-3.1-8B RLVR step 2440", "rl"),
    (RI, "OLMo-2-1124-7B-DPO → -Instruct (RLVR)", "OLMo-2-7B-DPO", "OLMo-2-7B-Instruct (RLVR)", "rl"),
    (IN, "Qwen2.5-7B → Qwen2.5-7B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "ins"),
    (IN, "Granite-3.3-8B-Base → Granite-3.3-8B-Instruct", "Granite-3.3-8B-Base", "Granite-3.3-8B-Instruct", "ins"),
    (IN, "OLMo-2-1124-7B → -SFT", "OLMo-2-7B", "OLMo-2-7B-SFT", "ins"),
    (IN, "OLMo-2-1124-7B-SFT → -DPO", "OLMo-2-7B-SFT", "OLMo-2-7B-DPO", "ins"),
    (IN, "Phi-4", None, "Phi-4", "ins"),
    (IN, "GLM-4-9B-0414", None, "GLM-4-9B-0414", "ins"),
    (BA, "Mistral-Small-24B-Base-2501", None, "Mistral-Small-24B-Base-2501", "base"),
    (BA, "Qwen3-30B-A3B-Base", None, "Qwen3-30B-A3B-Base", "base"),
    (BA, "MiMo-7B-Base (already writes it)", None, "MiMo-7B-Base", "basew"),
    (BA, "SmolLM3-3B-Base (already writes it)", None, "SmolLM3-3B-Base", "basew"),
    (BA, "OLMo-2-1124-7B (writes it, weakly)", None, "OLMo-2-7B", "basew"),
    (UN, "gpt-oss-120b (low)", None, "gpt-oss-120b effort=low", "und"),
    (UN, "gpt-oss-120b (medium)", None, "gpt-oss-120b effort=med", "und"),
    (UN, "gpt-oss-120b (high)", None, "gpt-oss-120b effort=high", "und"),
    (UN, "openPangu-Embedded-7B-V1.1", None, "openPangu-Embedded-7B-V1.1", "und"),
    (UN, "openPangu-Embedded-7B", None, "openPangu-Embedded-7B-model", "und"),
]
LGREY = "#BDBDBD"
STYLE = {"tr": (P.ORANGE, "s"), "rl": (P.BLUE, "o"), "z": (P.GREY, "o"), "mid": (P.GREEN, "D"), "ins": (LGREY, "s"), "base": (P.GREY, "o"), "cx": (P.GREY, "o"), "nr": ("white", "o"), "sec": ("white", "D"), "basew": (P.GREY, "o"), "und": (P.SKY, "o")}
SLOT, XMIN, XMAX, BRK = -3.6, -4.0, 1.5, -3.15   # the empty-set slot sits left of an axis break; right of it x = log10(wait per 1k)


def gates(root):
    """h1.zr.<id>.gate from paper/numbers/results.tex (D47a): pass | not reproduced | no published reference"""
    txt = (root / "paper" / "numbers" / "results.tex").read_text()
    return dict(re.findall(r"\\resdef\{h1\.zr\.(\w+)\.gate\}\{([^}]*)\}", txt))


def numbers(root, key):
    txt = (root / "paper" / "numbers" / "results.tex").read_text()
    m = re.search(r"\\resdef\{" + re.escape(key) + r"\}\{(.*?)\}\s*(?:%|$)", txt, re.M)
    return m.group(1).replace("{,}", "") if m else None


def zr_point(root, spec):
    """'@zr:<id>:<base|final>' -> (rate per 1k, exact Poisson 95% CI) from h1.zr.<id>.<which>.{nwait,ntok} in paper/numbers/results.tex"""
    _, i, w = spec.split(":")
    k, n = numbers(root, f"h1.zr.{i}.{w}.nwait"), numbers(root, f"h1.zr.{i}.{w}.ntok")
    if k is None or n is None:
        return None
    k, n = int(k), int(n)
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2
    return dict(wait_per_1k=1000 * k / n, lo=1000 * lo / n, hi=1000 * hi / n, group="zero_rl")


def zr_runs(root):
    """h1.zr.n_runs from paper/numbers/results.tex (number of zero-RL runs that pass D47a)"""
    txt = (root / "paper" / "numbers" / "results.tex").read_text()
    m = re.search(r"\\resdef\{h1\.zr\.n_runs\}\{(\d+)\}", txt)
    return int(m.group(1)) if m else None


def fx(x):
    return SLOT if x is None or x <= 0 else max(math.log10(x), -3.0)


def main():
    import sys
    full = "--full" in sys.argv
    if full:
        sys.argv.remove("--full")
    a = C.cli(__doc__)
    path = a.res / "reframe" / "provenance_map.json"
    d = json.load(open(path))
    by = {r["model"]: r for r in d["rows"]}
    rows = []
    gate = gates(C.ROOT)
    for row in M:
        band, name, m0, m1, kind = row[:5]
        opts = row[5] if len(row) > 5 else {}
        get = lambda m: zr_point(C.ROOT, m) if (m and m.startswith("@zr:")) else by.get(m)
        if get(m1) is None or (m0 is not None and get(m0) is None):
            print("skip (not in provenance_map.json / results.tex):", name)
            continue
        g = gate.get(opts["gate"]) if "gate" in opts else None
        if "gate" in opts and g is None:
            print("no D47a gate key for", name)
        if g is not None and g != "pass":   # D47a: not reproduced / no published reference -> appendix only, hollow grey marker
            kind, opts = ("sec" if g.startswith("no licence") else "nr"), {**opts, "full": True}
        e = get(m1)
        rows.append(dict(band=band, name=name, x0=get(m0)["wait_per_1k"] if m0 else None, x1=e["wait_per_1k"], lo=e.get("lo"), hi=e.get("hi"), kind=kind, group=e["group"], full=bool(opts.get("full"))))
    if not full:   # main-text version: the five core bands, rows that pass D47a; --full = every band and every row (appendix)
        rows = [r for r in rows if r["band"] in BANDS[:6] and not r["full"]]
    P.use()
    nb = len([b for b in BANDS if any(r["band"] == b for r in rows)])
    fig, ax = P.subplots("text", height=(0.116 if full else 0.128) * (len(rows) + 1.25 * nb) + (0.8 if full else 0.6))
    y, ticks, labels, ypos = 0.0, [], [], {}
    tr = blended_transform_factory(fig.transFigure, ax.transData)
    for bi, b in enumerate(BANDS):
        rs = [r for r in rows if r["band"] == b]
        if not rs:
            continue
        ax.text(0.012, y, b, transform=tr, ha="left", va="center", fontsize=P.SIZE["label"], fontweight="bold", color=P.INK, clip_on=False)
        y += 1
        ax.axhspan(y - 0.5, y + len(rs) - 0.5, color="#F7F7F7" if bi % 2 == 0 else "#FFFFFF", zorder=0, lw=0)
        for r in rs:
            col, mk = STYLE[r["kind"]]
            if r["x0"] is not None:
                if abs(fx(r["x1"]) - fx(r["x0"])) > 0.04:
                    ax.annotate("", xy=(fx(r["x1"]), y), xytext=(fx(r["x0"]), y), arrowprops=dict(arrowstyle="-|>", color=P.GREY, lw=0.8, shrinkA=2.5, shrinkB=3, mutation_scale=6), zorder=2,
                                annotation_clip=False)
                ax.plot([fx(r["x0"])], [y], marker="o", markersize=3.6, markerfacecolor="white", markeredgecolor=P.INK, markeredgewidth=0.5, linestyle="none", zorder=3)
            if r["lo"] is not None and r["hi"] is not None:
                ax.plot([fx(r["lo"]), fx(r["hi"])], [y, y], color=P.INK, lw=0.7, zorder=3, solid_capstyle="butt")
            ax.plot([fx(r["x1"])], [y], marker=mk, markersize=(6.2 if r["kind"] == "cx" else 4.2), markerfacecolor=col, markeredgecolor=(P.GREY if r["kind"] in ("nr", "sec") else P.VERMILION if r["kind"] == "cx" else P.INK), markeredgewidth=(0.9 if r["kind"] in ("nr", "sec") else 1.1 if r["kind"] == "cx" else 0.5), linestyle="none", zorder=4)
            if r["kind"] == "cx":
                ax.text(fx(r["hi"]) + 0.12, y, "exception", fontsize=P.SIZE["sub"] + 0.5, fontweight="bold", color=P.INK, va="center", ha="left")
            ticks.append(y); labels.append(r["name"]); ypos[r["name"]] = y
            y += 1
        y += 0.3
    # claim callouts, numbers from the file
    zr = [r for r in rows if r["band"] == Z_ and r["kind"] in ("z", "cx")]   # runs that pass D47a
    nr = zr_runs(C.ROOT)
    nc = numbers(C.ROOT, "h1.zr.n_counter")
    nc = int(nc) if nc is not None else None
    if nr is not None and nr != len(zr):
        print(f"WARNING: h1.zr.n_runs = {nr} but {len(zr)} passing zero-RL rows are plotted")
    if zr:
        ax.text(fx(0.3), ypos[zr[len(zr) // 2]["name"]], f"zero-RL coins it\nin {nc} of {nr if nr is not None else len(zr)} runs", fontsize=P.SIZE["sub"] + 0.5, fontweight="bold", color=P.INK, va="center", ha="left")
    for r in rows:
        if r["kind"] == "mid":
            ax.text(fx(0.05), ypos[r["name"]] - 0.42, f"{r['x0']:.1g} → {r['x1']:.1g} per 1k", fontsize=P.SIZE["sub"] + 0.5, color=P.INK, va="bottom", ha="left")
        if r["name"] == "OLMo-3-7B: RL start → RL-Zero":
            ax.text(fx(2.2), ypos[r["name"]], "RL-Zero amplifies what is already there", fontsize=P.SIZE["sub"] + 0.5, fontweight="bold", color=P.INK, va="center", ha="right")
        if r["name"] == "OpenReasoning-Nemotron-7B":
            ax.text(fx(r["hi"]) + 0.15, ypos[r["name"]], "← the one trace-trained exception", fontsize=P.SIZE["sub"] + 0.5, fontweight="bold", color=P.ORANGE_DK if hasattr(P, "ORANGE_DK") else P.INK, va="center", ha="left")
    ax.text(0.0, 1.0 + 0.012, "0 = no occurrence in the sampled tokens; whisker = 95% CI", transform=ax.transAxes, fontsize=P.SIZE["sub"], color=P.MUTED, va="bottom", ha="left", clip_on=False)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=P.SIZE["sub"] + 0.5)
    ax.invert_yaxis()
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks([SLOT, -2, -1, 0, 1]); ax.set_xticklabels(["0", "0.01", "0.1", "1", "10"])
    bx = (BRK - XMIN) / (XMAX - XMIN)
    ax.add_patch(plt.Rectangle((bx - 0.012, -0.03 * 0.5), 0.024, 0.03, transform=ax.transAxes, color="white", clip_on=False, zorder=5, lw=0))
    for dd in (-1, 1):
        ax.plot([bx + dd * 0.012 - 0.006, bx + dd * 0.012 + 0.006], [-0.008, 0.008], transform=ax.transAxes, color=P.INK, lw=0.6, clip_on=False, zorder=6)
    ax.set_xlabel("wait per 1k tokens (log scale)"); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    def hd(mk, fc):
        return Line2D([], [], marker=mk, ls="none", markerfacecolor=fc, markeredgecolor=P.INK, markeredgewidth=0.5, markersize=4.0)
    ent = [(hd("o", "white"), "start checkpoint"), (hd("s", P.ORANGE), "after traces")] + ([(hd("s", LGREY), "after instruct tuning")] if full else []) + \
        [(hd("D", P.GREEN), "after mid-training"), (hd("o", P.BLUE), "after RL"), (hd("o", P.GREY), "zero-RL run / base")]
    if any(r["kind"] == "nr" for r in rows):
        ent.append((Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=P.GREY, markeredgewidth=0.9, markersize=4.0), "not reproduced"))
    if any(r["kind"] == "sec" for r in rows):
        ent.append((Line2D([], [], marker="D", ls="none", markerfacecolor="white", markeredgecolor=P.GREY, markeredgewidth=0.9, markersize=3.6), "secondary: no licence"))
    fig.legend([h for h, _ in ent], [t for _, t in ent], loc="outside upper center", ncols=(4 if len(ent) > 6 else len(ent)), frameon=False, fontsize=P.SIZE["legend"])
    plt.rcParams["savefig.bbox"] = "standard"
    stem = C.ROOT / "figures" / ("fig_hero_provenance_full" if full else "fig_hero_provenance")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(stem.with_suffix("." + ext), **kw)
        print("wrote", stem.with_suffix("." + ext))


if __name__ == "__main__":
    main()
