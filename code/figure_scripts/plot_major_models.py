
import re, json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import chi2
import _common as C
from _common import P

def parse_numbers(path):
    d = {}
    pat = re.compile(r'\\resdef\{([^}]+)\}\{(.*?)\}[ \t]*(?:%.*)?$', re.M)
    for line in open(path, encoding='utf-8'):
        m = pat.search(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        val = val.replace('{,}', '').replace('\\ensuremath{-}', '-')
        val = val.replace('{', '').replace('}', '').replace('\\', '').strip()
        try:
            d[key] = float(val)
        except ValueError:
            pass
    return d

NUM = parse_numbers(C.ROOT / "paper/numbers/results.tex")

def rate_ci(prefix):
    if prefix is None:
        return None
    nk, tk = prefix + ".nwait", prefix + ".ntok"
    if nk not in NUM or tk not in NUM:
        print(f"warning: missing {nk} or {tk}")
        return None
    n, t = NUM[nk], NUM[tk]
    if t <= 0:
        print(f"warning: zero tokens for {prefix}")
        return None
    rate = 1000.0 * n / t
    lo = chi2.ppf(0.025, 2 * n) / 2 * 1000.0 / t if n > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * n + 2) / 2 * 1000.0 / t
    return rate, lo, hi

groups = [
    ("Qwen2.5 zero-RL", P.BLUE, [
        ("ORZ-32B (PPO)", "h1.zr.orz32.base", "h1.zr.orz32.final", "circle"),
        ("Qwen2.5-Math-7B (our GRPO)", "h1.zr.s7r7b.base", "h1.zr.s7r7b.final", "circle"),
    ]),
    ("Qwen3", P.SKY, [
        ("Qwen3-4B (our GRPO)", "h1.zr.q3r1.base", "h1.zr.q3r1.final", "circle"),
        ("General-Reasoner-Qwen3-4B", "h1.zr.gr3_4.base", "h1.zr.gr3_4.final", "circle"),
        ("Qwen3-8B", "h1.pop_qwen3_8b_base.final", "h1.pop_qwen3_8b.final", "square"),
        ("Qwen3-14B", "h1.pop_qwen3_14b_base.final", "h1.pop_qwen3_14b.final", "square"),
        ("Qwen3-30B-A3B", "h1.pop_qwen3b30.final", "h1.pop_qwen3t30.final", "square"),
    ]),
    ("Qwen3.5/3.8", P.PURPLE, [
        ("Qwen3.5-0.8B", "h1.bb_qwen35_0p8b_base.final", "h1.bb_qwen35_0p8b.final", "diamond"),
        ("Qwen3.5-2B", "h1.bb_qwen35_2b_base.final", "h1.bb_qwen35_2b.final", "diamond"),
        ("Qwen3.5-4B", "h1.bb_qwen35_4b_base.final", "h1.bb_qwen35_4b.final", "diamond"),
        ("Qwen3.5-9B", "h1.bb_qwen35_9b_base.final", "h1.bb_qwen35_9b.final", "diamond"),
        ("Qwen3.8-27B", None, "h1.bb_qwen38_27b.final", "diamond"),
    ]),
    ("MiMo", P.ORANGE, [
        ("MiMo-7B-RL-Zero", "h1.bb_mimo_base.final", "h1.bb_mimo_rlzero.final", "circle"),
        ("MiMo-7B-SFT", "h1.bb_mimo_base.final", "h1.bb_mimo_sft.final", "square"),
        ("MiMo-7B-RL", "h1.bb_mimo_sft.final", "h1.bb_mimo_rl.final", "square"),
    ]),
    ("DeepSeek / Phi", P.INK, [
        ("R1-Distill-Qwen-32B", "h1.s8base.final", "h1.s8r1d.final", "square"),
        ("Phi-4-reasoning", "h1.pop_phi4.final", "h1.pop_phi4r.final", "square"),
    ]),
    ("openPangu", P.GREY, [
        ("openPangu-Embedded-7B-V1.1", None, "h1.bb_pangu_e7v11.final", "diamond"),
        ("openPangu-Embedded-7B", None, "h1.bb_pangu_e7m.final", "diamond"),
    ]),
]

MARK = {"circle": "o", "square": "s", "diamond": "D"}

rows = []
for gname, gcolor, items in groups:
    for label, bpref, apref, shape in items:
        before = rate_ci(bpref)
        after = rate_ci(apref)
        if after is None:
            print(f"warning: skipping {label}, no after data")
            continue
        rows.append(dict(group=gname, color=gcolor, label=label,
                          before=before, after=after, shape=shape))

fig, ax = plt.subplots(figsize=(P.TEXT_W * 0.5, 3.9), constrained_layout=True)

y = 0
yticks, ylabels = [], []
band_info = []
cur_group = None
group_start_y = 0
GAP = 1.0
for r in rows:
    if r["group"] != cur_group:
        if cur_group is not None:
            band_info.append((group_start_y - 0.5, y - GAP + 0.5, cur_group, cur_color))
            y -= GAP
            y += 1.0
        cur_group = r["group"]
        cur_color = r["color"]
        group_start_y = y
    r["y"] = y
    yticks.append(y)
    ylabels.append(r["label"])
    y += 1
band_info.append((group_start_y - 0.5, y - 0.5, cur_group, cur_color))

for y0, y1, name, color in band_info:
    ax.axhspan(y0, y1, color=P.GREY_LIGHT, alpha=0.35, zorder=0)
    pass

for r in rows:
    yy = r["y"]
    ax.axhline(yy, color=P.GREY_LIGHT, lw=P.LW['hair'], zorder=0.5)
    before, after = r["before"], r["after"]
    marker = MARK[r["shape"]]
    if before is not None:
        bx = before[0]
        ax.plot([bx], [yy], marker='o', mfc='none', mec=P.GREY, ms=4.5,
                 mew=P.LW['thin'], zorder=3)
    if before is not None:
        ax.plot([before[0], after[0]], [yy, yy], color=P.GREY, lw=P.LW['hair'], zorder=1)
    ax.plot([after[1], after[2]], [yy, yy], color=r["color"], lw=P.LW['hair'], zorder=2)
    ax.plot([after[0]], [yy], marker=marker, mfc=r["color"], mec=r["color"], ms=5,
             zorder=4)
    val = after[0]
    txt = "<.01" if val < 0.01 else f"{val:.2f}"
    ax.text(after[0], yy, f"  {txt}", fontsize=P.SIZE['sub'], color=P.MUTED,
            va='center', ha='left', zorder=5,
            transform=ax.transData)

ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=P.SIZE['tick'])
ax.tick_params(axis='y', length=0)
ax.invert_yaxis()

ax.set_xscale('symlog', linthresh=0.01, linscale=0.5)
ax.set_xlim(0, 30)
ax.set_xticks([0, 0.1, 1, 10])
ax.set_xticklabels(["0", "0.1", "1", "10"], fontsize=P.SIZE['tick'])
ax.set_xlabel("wait per 1k (\u25cb before \u2192 after)", fontsize=P.SIZE['label'])

ax.axvline(0.05, color=P.GREY, lw=P.LW['hair'], ls='--', zorder=0.5)
ax.text(0.05, -0.9, "marker-free threshold", fontsize=P.SIZE['sub'],
        color=P.GREY, ha='center', va='bottom')

for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

ax.set_title("\u25cf RL from base  \u25a0 SFT(+RL)  \u25c6 undisclosed", fontsize=P.SIZE['sub'], color=P.MUTED, pad=10)

outdir = C.ROOT / "figures"
outdir.mkdir(exist_ok=True, parents=True)
pdf_path = outdir / "fig_major_models.pdf"
ax.title.set_x(0.32)
ax.xaxis.set_label_coords(0.32, -0.06)
png_path = outdir / "fig_major_models.png"
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
print(f"wrote {pdf_path}")
print(f"wrote {png_path}")

records = []
for r in rows:
    records.append(dict(
        group=r["group"], label=r["label"], shape=r["shape"],
        before=r["before"], after=r["after"]
    ))
json_path = outdir / "fig_major_models.json"
with open(json_path, 'w') as f:
    json.dump(records, f, indent=2)
print(f"wrote {json_path}")
