"""SDT bleaching panel: d' and criterion c vs RL step (+/- 95% CI), one line per series; SFT control ('-ctl') grey dashed.
Reads exp/results/main/<series>/sdt.json (+ reframe/s9_sdt.json) or exp/results/P0/sdt.json (pilot).
    python3 figures/scripts/plot_sdt.py [--mode pilot|main]
"""
import _common as C
import panels as PN
from _common import P


def main():
    a = C.cli(__doc__)
    paths = C.discover(a, "main/*/sdt.json|E2/*/sdt.json|reframe/s9_sdt.json", "P0/sdt.json")
    paths = sorted(paths, key=lambda p: C.series_of(p, C.load(p)))
    if not paths:
        print("no sdt.json found; nothing plotted")
        return
    datas = [(C.series_of(p, C.load(p)), C.load(p)) for p in paths]
    pilot = any(C.is_pilot(p, d) for p, (s, d) in zip(paths, datas))
    P.use()
    fig, axs = P.subplots("col", height=1.6, ncols=2)
    PN.draw_sdt(axs[0], axs[1], datas)
    if len(datas) > 1:   # legend instead of end labels (the reader-facing names are long): marker + name per series
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], marker=C.MARKERS[k % 5], color=P.GREY if ser.endswith("ctl") else P.BLUE, ls="--" if ser.endswith("ctl") else "-", markerfacecolor=P.GREY if ser.endswith("ctl") else P.BLUE,
                     markeredgecolor=P.INK, markeredgewidth=0.5, markersize=4.2) for k, (ser, _) in enumerate(datas)]
        fig.legend(hs, [PN.SDT_LABEL.get(ser, ser) for ser, _ in datas], loc="outside upper center", ncols=len(datas), frameon=False, fontsize=P.SIZE["legend"])
    P.panel_title(axs[0], "(a) Sensitivity")
    P.panel_title(axs[1], "(b) Criterion")
    C.finish(fig, "fig4a_sdt", pilot, note=", ".join(s for s, _ in datas))


if __name__ == "__main__":
    main()
