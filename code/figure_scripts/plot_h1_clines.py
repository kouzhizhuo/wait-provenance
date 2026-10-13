"""Fig. 3: H1 cline curves, Lehmann indices vs RL step, per series, +/- 95% cluster-bootstrap CI.
Reads exp/results/E1/<series>/indices.json (main) or exp/results/P0/h1_indices.json (pilot).
Rows = series; columns = index groups; an SFT control series (name ends '-ctl') is drawn grey dashed.
Indices absent from a file (e.g. APD before the LSC stage) are skipped.
    python3 figures/scripts/plot_h1_clines.py [--mode pilot|main]
"""
import _common as C
from _common import P

# (indices, title, y label, colours by index position). Titles are descriptive; switch to claim titles once results are in.
COLS = [   # panels built on revision labels (paradigm/H_para, OBL, REL) were dropped (human audit V4, rule A(iii)); label-free indices only
    (("FREQ_wait_1k",), "(a) Frequency", "wait per 1k tokens"),
    (("SI",), "(b) Position", "sentence-initial"),
    (("ORTH",), "(c) Comma", "canonical form"),
    (("CHAIN",), "(d) Chains", "chained share"),
    (("SENTLEN_init",), "(e) Length", "words"),
]
ROW_TITLE = {"S1": "OLMo-3 RL-Zero (RL from the base): multiplies it and erodes its form",
             "S2": "OLMo-3 Think (RL after SFT and DPO): rate changes little, comma form restored"}   # reader-facing names + the paper's wording (lead, QA)
LABEL = {"SI": "SI", "ORTH": "ORTH", "OBL": "OBL", "REL": "REL"}
SECOND = [P.BLUE, P.BLUE]   # second index of a panel: same hue, dashed, hollow squares (orange is reserved for the Hold on,/Hmm, arm)
MIN_OCC = 100   # plotting rule: form-function indices need >= 100 marker occurrences at a checkpoint (n_wait); below that they are not drawn
OCC_FREE = {"FREQ_wait_1k"}   # frequency itself is always drawn


def nwait(cp):
    v = cp["indices"].get("n_wait", 0)
    return v["est"] if isinstance(v, dict) else v


def main():
    a = C.cli(__doc__)
    paths = C.discover(a, "main/*/h1_indices.json|E1/*/indices.json", "P0/h1_indices.json")
    if not paths:
        print("no H1 index files found (exp/results/E1/*/indices.json or P0/h1_indices.json); nothing plotted")
        return
    if not a.series:   # default = primary series + the SFT control; robustness variants (S1boxed, S1greedy, S1pre, S3t07, ...) via --series
        paths = [p for p in paths if C.series_of(p, C.load(p)) in ("S1", "S2", "S3", "S2ctl", "S2-ctl")] or paths
    data = [(p, C.load(p)) for p in paths]
    pilot = any(C.is_pilot(p, d) for p, d in data)
    cols = [c for c in COLS if any(i in next(iter(d["checkpoints"].values()))["indices"] for _, d in data for i in c[0])]
    nr, nc = len(data), len(cols)
    P.use()
    fig, axs = P.plt.subplots(nr, nc, figsize=(P.TEXT_W, 1.45 * nr + 0.25 + 0.15 * (nr - 1) + (0.2 if nr > 1 else 0)), squeeze=False, sharey="col")   # one y-range per column so a big and a small change are comparable (QA 3-second test)
    if nr > 1 and fig.get_layout_engine() is not None:   # room for the row titles between the rows
        fig.get_layout_engine().set(hspace=0.30)   # room for the row titles between the rows
    for r, (p, d) in enumerate(data):
        ser = C.series_of(p, d)
        cps = C.checkpoints(d["checkpoints"])
        ctl = ser.endswith("ctl")
        for c, (idxs, title, yl) in enumerate(cols):
            ax = axs[r, c]
            drawn = False
            for k, idx in enumerate(idxs):
                if idx not in cps[0]["indices"]:
                    continue
                col = P.GREY if ctl else SECOND[k]
                def get(cp, idx=idx):
                    if idx not in OCC_FREE and nwait(cp) < MIN_OCC:
                        return None
                    return cp["indices"].get(idx)
                ok = [(x, cp) for x, cp in zip(C.xpos(cps), cps) if get(cp) is not None]
                if not ok:
                    continue
                drawn = True
                C.line_ci(ax, cps, get, col, marker="s" if k else "o", ls="--" if (ctl or k) else "-", hollow=bool(k))
                C.hollow_base(ax, [cp for _, cp in ok], [x for x, _ in ok], [get(cp)["est"] for _, cp in ok], col)
                if len(idxs) > 1:
                    P.end_label(ax, ok[-1][0], get(ok[-1][1])["est"], LABEL.get(idx, idx), dx=2)
            if not drawn:
                ax.text(0.5, 0.5, f"n(wait) < {MIN_OCC}\nat every checkpoint", transform=ax.transAxes, ha="center", va="center",
                        fontsize=P.SIZE["legend"], color=P.GREY, style="italic")
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                P.panel_title(ax, title)
            if c == 0 and nr > 1:
                if r == 0:   # first row title as a suptitle (it reserves headroom in constrained layout)
                    fig.suptitle(ROW_TITLE.get(ser, f"{ser} {C.SERIES_NAME.get(ser, '')}".strip()), x=0.0, ha="left", fontsize=P.SIZE["label"], fontweight="bold", color=P.INK)
                else:
                    _t = ax.annotate(ROW_TITLE.get(ser, f"{ser} {C.SERIES_NAME.get(ser, '')}".strip()), (0, 1), xycoords="axes fraction", xytext=(0, 8), textcoords="offset points",
                            fontsize=P.SIZE["label"], fontweight="bold", color=P.INK)
                    _t.set_in_layout(False)   # a long row title must not shrink the first column in constrained layout
            if drawn:
                ax.set_ylabel(yl, fontsize=P.SIZE["legend"])
            if drawn:
                C.set_step_axis(ax, cps, label=(r == nr - 1), nbins=2)
            if len(idxs) > 1:
                ax.margins(x=0.16)
    C.finish(fig, "fig3_h1_clines" + ("_" + "".join(a.series) if a.series else ""), pilot, note=", ".join(C.series_of(p, d) for p, d in data))


if __name__ == "__main__":
    main()
