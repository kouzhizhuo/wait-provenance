"""MAIN-TEXT hero (Fig. 3, compact): strip plot of `wait` per 1k tokens after training, 5 provenance categories, one dot per model, hollow ghost dots for the bases,
median bar per category, two big annotations whose numbers are parsed from paper/numbers/results.tex (never typed). Appendix keeps the full map (hero_provenance.py --full).
Data: exp/results/reframe/provenance_map.json (+ paper/numbers/results.tex for the D47a gate and the zero-RL base points).
    python3 figures/scripts/hero_strip.py   ->  figures/fig_hero_strip.pdf (+png)
"""
import json
import zlib
import math
import statistics
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _common as C
import hero_provenance as H
from _common import P

CATS = ["fine-tuned on traces,\\nbase without it", "base unreleased,\\nnot counted", "post-trained on a base\\nthat already writes it", "RL on a model\\nthat writes it", "OLMo-3\\ntraining stages", "zero-RL from a\\nbase without it"]
CATS = [c.replace("\\n", "\n") for c in CATS]
COL = [P.ORANGE, "#BDBDBD", P.ORANGE, P.BLUE, P.GREEN, P.GREY]
MK = ["s", "s", "s", "o", "D", "o"]
YC = [0.0, 0.75, 1.5, 2.5, 3.5, 4.75]      # row centres; the unreleased-base row is thin
HH = [0.5, 0.25, 0.5, 0.5, 0.5, 0.75]      # half-heights
SHADE = ["#F7F7F7", "#F7F7F7", "#FFFFFF", "#F7F7F7", "#FFFFFF", "#F7F7F7"]


def swarm(xs, lanes=(-0.3, -0.15, 0.0, 0.15, 0.3), tol=0.16):
    """deterministic vertical offsets so equal/close x do not overlap; x of exact zeros is spread slightly inside the zero slot"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    last = {l: -1e9 for l in lanes}
    off = [0.0] * len(xs)
    for i in order:
        for l in sorted(lanes, key=lambda l: abs(l)):
            if xs[i] - last[l] >= tol:
                off[i] = l
                last[l] = xs[i]
                break
        else:
            l = min(lanes, key=lambda l: last[l])
            off[i] = l
            last[l] = xs[i]
    return off


def main():
    a = C.cli(__doc__)
    d = json.load(open(a.res / "reframe" / "provenance_map.json"))
    by = {r["model"]: r for r in d["rows"]}
    rows = d["rows"]
    gate = H.gates(C.ROOT)
    num = lambda k: H.numbers(C.ROOT, k)
    hi_by = {}   # upper exact-Poisson 95% bound per zero-RL run (astra advice: whiskers on the zero-RL row only)
    pts = [[] for _ in CATS]   # (x1, x0 or None, name)
    for r in rows:
        if r["group"] == "reasoning_sft_distill":
            bw = r.get("base_wait_per_1k")   # base from the official card; < 0.05 = marker-free (D41)
            if bw is None:
                pts[1].append((r["wait_per_1k"], None, r["model"], "nobase"))
            elif bw < 0.05:
                pts[0].append((r["wait_per_1k"], None, r["model"], ""))
            else:
                pts[2].append((r["wait_per_1k"], bw, r["model"], ""))
        elif r["group"] == "rl_on_marker_model" and "SimpleRL" not in r["model"]:   # the Mistral-7B SimpleRL row is "not reproduced" (appendix map only)
            pts[3].append((r["wait_per_1k"], None, r["model"], ""))
    for band, name, m0, m1, kind, *rest in H.M:
        opts = rest[0] if rest else {}
        get = lambda m: H.zr_point(C.ROOT, m) if (m and m.startswith("@zr:")) else by.get(m)
        if band == H.Z_ and "gate" in opts and gate.get(opts["gate"]) == "pass" and get(m1) is not None:
            hi_by[name if kind != "cx" else "EXC"] = get(m1).get("hi")
            pts[5].append((get(m1)["wait_per_1k"], get(m0)["wait_per_1k"] if m0 and get(m0) else None, name if kind != "cx" else "EXC", ""))
    pts[4] = [(by["OLMo-3-7B mid-training"]["wait_per_1k"], by["OLMo-3-7B pretraining step 1.41M"]["wait_per_1k"], "OLMo-3-7B mid-training", "")]
    # compact-table keys (lead 2026-10-12): per route, n, n with >= 0.05 per 1k after training, median after-rate
    ids = ["sftfree", "nobase", "postmarked", "rlmarked", "olmo", "zerorl"]
    ent = []
    for cid, ps in zip(ids, pts):
        xs = sorted(x[0] for x in ps if x[0] is not None)
        if not xs:
            continue
        med = statistics.median(xs)
        ent += [{"key": f"h1.route.{cid}.n", "value": len(xs), "src": "figures/scripts/hero_strip.py (provenance_map.json)"},
                {"key": f"h1.route.{cid}.n_ge05", "value": sum(x >= 0.05 for x in xs), "src": "hero_strip.py: after-rate >= 0.05 per 1k"},
                {"key": f"h1.route.{cid}.median", "value": ("0" if med == 0 else ("$<$.01" if med < 0.01 else f"{med:.2f}")), "src": "hero_strip.py: median after-rate per 1k"}]
    json.dump({"entries": ent}, open(C.ROOT / "paper/numbers/keys/route_keys.json", "w"), indent=1)
    P.use()
    fig, ax = P.subplots("text", height=4.5)
    # annotation numbers parsed from keys
    n_ge, n_all = int(num("h1.prov.sft_distill.n_ge05")), int(num("h1.prov.sft_distill.n"))
    n_conf, n_runs, n_cnt, ceil_ = int(num("h1.zr.n_confirm")), int(num("h1.zr.n_runs")), int(num("h1.zr.n_counter")), num("h1.zr.upper95_ceil")
    if len(pts[5]) != n_runs:
        print(f"WARNING: {len(pts[5])} counted zero-RL dots but h1.zr.n_runs = {n_runs}")
    sf_n, sf_ge = num("h1.prov.sft_free.n"), num("h1.prov.sft_free.n_ge05")
    mk_n, mk_ge = num("h1.prov.sft_marked.n"), num("h1.prov.sft_marked.n_ge05")
    free = pts[0]
    if sf_n is not None and len(free) != int(sf_n):
        print(f"WARNING: {len(free)} marker-free-base trace dots but h1.prov.sft_free.n = {sf_n}")
    if sf_ge is not None and sum(1 for p in free if p[0] >= 0.5) != int(sf_ge):
        print(f"WARNING: {sum(1 for p in free if p[0] >= 0.5)} of them write it but h1.prov.sft_free.n_ge05 = {sf_ge}")
    if mk_n is not None and len(pts[2]) != int(mk_n):
        print(f"WARNING: {len(pts[2])} marked-base dots but h1.prov.sft_marked.n = {mk_n}")
    ylab = {}
    for ci, cp in enumerate(pts):
        fxs = [H.fx(p[0]) for p in cp]
        off = swarm(fxs, lanes=((-0.1, 0.0, 0.1) if ci == 1 else (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6) if ci == 5 else (-0.3, -0.15, 0.0, 0.15, 0.3)), tol=0.16)
        med = H.fx(statistics.median(p[0] for p in cp))
        y0 = YC[ci]
        ax.axhspan(y0 - HH[ci], y0 + HH[ci], color=SHADE[ci], zorder=0, lw=0)
        for (x1, x0, nm, flag), fxv, o in zip(cp, fxs, off):
            xx = fxv + ((0.06 * ((zlib.crc32(nm.encode()) % 5) - 2)) if x1 <= 0 else 0.0)
            y = y0 + o
            if x0 is not None:
                gx = H.fx(x0)
                if abs(gx - xx) > 0.05:
                    ax.annotate("", xy=(xx, y), xytext=(gx, y), arrowprops=dict(arrowstyle="-|>", color=P.GREY, lw=0.6, shrinkA=2.2, shrinkB=3.2, mutation_scale=5), zorder=2, annotation_clip=False)
                ax.plot([gx], [y], marker="o", markersize=3.2, markerfacecolor="white", markeredgecolor=P.GREY, markeredgewidth=0.6, linestyle="none", zorder=3)
            if ci == 5 and hi_by.get(nm) is not None:
                hx = H.fx(hi_by[nm])
                if hx > xx + 0.1:   # interval bar just below the dot, so it never overlaps the base arrow
                    yy = y + 0.055
                    ax.plot([xx, hx], [yy, yy], color=P.INK, lw=0.55, solid_capstyle="butt", zorder=3)
                    ax.plot([hx, hx], [yy - 0.04, yy + 0.04], color=P.INK, lw=0.55, solid_capstyle="butt", zorder=3)
            exc = nm == "EXC"
            ax.plot([xx], [y], marker=MK[ci], markersize=(6.0 if exc else 4.6), markerfacecolor=COL[ci], markeredgecolor=(P.VERMILION if exc else P.GREY if ci == 1 else P.INK), markeredgewidth=(1.2 if exc else 0.5), linestyle="none", zorder=4)
            ylab[nm] = (xx, y)
        if ci != 1:   # four models: no median
            ax.plot([med, med], [y0 - HH[ci] + 0.08, y0 + HH[ci] - 0.08], color=P.INK, lw=1.6, solid_capstyle="butt", zorder=5)
    # anchors (<= 8 labels incl. the exception)
    def tag(nm, text, dx=6, dy=0, ha="left", bold=False, color=None, leader=False):
        x, y = ylab[nm]
        kw = dict(arrowprops=dict(arrowstyle="-", color=P.GREY, lw=0.6, shrinkA=1.5, shrinkB=3.5)) if leader else {}
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=P.SIZE["sub"] + 0.5, ha=ha, va="center", fontweight="bold" if bold else "normal", color=color or P.INK, **kw)
    tag("EXC", "General-Reasoner-Qwen3-4B", dx=0, dy=-13, bold=True)
    tag("OLMo-3-7B RL-Zero-Math (S1 final)", "Olmo-3-7B-RL-Zero-Math", dx=7, dy=-1, ha="left")
    tag("OLMo-3-7B mid-training", "mid-training", dx=-6, dy=-9, ha="right")
    tag("OpenReasoning-Nemotron-7B", "OpenReasoning-\nNemotron-7B", dx=16, dy=17, leader=True)
    # big annotations (numbers parsed from keys)
    if sf_n is not None and sf_ge is not None:
        big, small = f"{int(sf_ge)} of {int(sf_n)}", "from marker-free\nbases write it"
    else:
        print("note: h1.prov.sft_free.* not in results.tex yet; using h1.prov.sft_distill")
        big, small = f"{n_ge} of {n_all}", "write it"
    ax.annotate("", xy=(2.0, -0.4), xytext=(2.0, 0.4), arrowprops=dict(arrowstyle="-", color=P.INK, lw=1.0), annotation_clip=False)
    ax.text(2.08, -0.1, big, fontsize=P.SIZE["label"] + 3, fontweight="bold", color=P.INK, va="center", ha="left")
    ax.text(2.08, 0.2, small, fontsize=P.SIZE["sub"] + 0.5, color=P.INK, va="top", ha="left", linespacing=1.0)
    if mk_n is not None and mk_ge is not None:   # the denominator's complement: models whose base already writes it
        ax.annotate("", xy=(2.0, 1.1), xytext=(2.0, 1.9), arrowprops=dict(arrowstyle="-", color=P.INK, lw=1.0), annotation_clip=False)
        ax.text(2.08, 1.38, f"{int(mk_ge)} of {int(mk_n)}", fontsize=P.SIZE["label"] + 1, fontweight="bold", color=P.INK, va="center", ha="left")
        ax.text(2.08, 1.64, "write it", fontsize=P.SIZE["sub"] + 0.5, color=P.INK, va="top", ha="left")
    # the zero-RL ceiling as a thin dashed line
    cx_ = H.fx(float(ceil_))
    ax.plot([cx_, cx_], [4.12, 5.42], color=P.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(cx_, 5.47, "max 95% upper", fontsize=P.SIZE["sub"] - 0.5, color=P.MUTED, ha="center", va="bottom")
    ax.text(-1.08, 4.24, f"zero-RL: {n_conf} of {n_runs} stay below {ceil_}\n{n_cnt} exception", fontsize=P.SIZE["label"] + 1.5, fontweight="bold", color=P.INK, va="center", ha="left")
    ax.set_yticks(YC); ax.set_yticklabels(CATS, fontsize=P.SIZE["sub"] + 0.5)
    ax.set_ylim(5.5, -0.5)
    ax.set_xlim(H.XMIN, 3.9)
    ax.set_xticks([H.SLOT, -2, -1, 0, 1]); ax.set_xticklabels(["0", "0.01", "0.1", "1", "10"])
    bx = (H.BRK - H.XMIN) / (3.9 - H.XMIN)
    ax.add_patch(plt.Rectangle((bx - 0.012, -0.015), 0.024, 0.03, transform=ax.transAxes, color="white", clip_on=False, zorder=5, lw=0))
    for dd in (-1, 1):
        ax.plot([bx + dd * 0.012 - 0.006, bx + dd * 0.012 + 0.006], [-0.016, 0.016], transform=ax.transAxes, color=P.INK, lw=0.6, clip_on=False, zorder=6)
    ax.set_xlabel("wait per 1k tokens after training (log scale)"); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    h = [Line2D([], [], marker="o", ls="none", markerfacecolor="white", markeredgecolor=P.GREY, markeredgewidth=0.6, markersize=3.4), Line2D([], [], color=P.INK, lw=1.6)]
    fig.legend(h, ["base (before training)", "median"], loc="outside upper center", ncols=2, frameon=False, fontsize=P.SIZE["legend"])
    plt.rcParams["savefig.bbox"] = "standard"
    stem = C.ROOT / "figures" / "fig_hero_strip"
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(stem.with_suffix("." + ext), **kw)
        print("wrote", stem.with_suffix("." + ext))


if __name__ == "__main__":
    main()
