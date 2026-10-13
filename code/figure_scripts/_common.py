"""Shared helpers for the P1 result plots. Reads ONLY exp/results/** (no typed numbers).

Data discovery: main-run files (exp/results/E1..E5/...) win over pilot files (exp/results/P0/...).
Anything read from a P0 path, or with "pilot": true, is stamped PILOT in the figure and in the file name.
Schemas: figures/scripts/SCHEMAS.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]              # P1_grammaticalization/
RES = ROOT / "exp" / "results"
PLOTS = ROOT / "figures" / "plots"
sys.path.insert(0, str(ROOT.parent / "figures" / "style"))
import palette as P  # noqa: E402

# P1 role map (FIGURE_STANDARD 5.3): ours/RL blue, base grey, cmp orange/green/sky, intervention vermilion, decision purple.
SERIES_NAME = {"S1": "OLMo-3 RL-Zero", "S2": "OLMo-3 Think", "S3": "Qwen2.5 GRPO", "S9": "OLMo-3.1 32B Think"}   # as in Fig. 2 stage 1
MARKERS = ["o", "s", "^", "D", "v"]


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def is_pilot(path: Path, data: dict) -> bool:
    """PILOT iff the file sits under exp/results/P0 (path wins). A 'pilot' flag that disagrees with the path is reported."""
    by_path = "P0" in path.parts
    if "pilot" in data and bool(data["pilot"]) != by_path:
        print(f"WARNING: {path.name} in {path.parent.name}/ has pilot={data['pilot']} but is {'a pilot' if by_path else 'a main-run'} path; using the path", file=sys.stderr)
    return by_path


def est(d) -> tuple[float, float, float]:
    """{est,lo,hi} -> (est, lo, hi)."""
    return d["est"], d.get("lo", d["est"]), d.get("hi", d["est"])


def checkpoints(x) -> list[dict]:
    """Normalise a checkpoint container (dict keyed by ckpt, or list) to a list sorted by step ('final' / null step last)."""
    if isinstance(x, dict):
        cps = [{"ckpt": k, **v} for k, v in x.items()]
    else:
        cps = list(x)
    for c in cps:
        if c.get("step") is None and c["ckpt"] not in ("final",):
            c["step"] = int(c["ckpt"]) if str(c["ckpt"]).isdigit() else (0 if c["ckpt"] == "base" else None)
    return sorted(cps, key=lambda c: (c["step"] is None, c["step"] or 0))


def xpos(cps: list[dict]) -> list[float]:
    """x positions: numeric steps as is; a step-less 'final' checkpoint is placed 12% beyond the last numeric step."""
    num = [c["step"] for c in cps if c["step"] is not None]
    xmax = max(num) if num else 1
    return [c["step"] if c["step"] is not None else xmax * 1.12 for c in cps]


def set_step_axis(ax, cps, label=True, nbins=3):
    xs = xpos(cps)
    num = [c["step"] for c in cps if c["step"] is not None]
    xmax = max(num) if num else 1
    from matplotlib.ticker import MaxNLocator
    locs = [t for t in MaxNLocator(nbins=nbins, integer=True, steps=[1, 2, 5, 10]).tick_values(0, xmax) if 0 <= t <= xmax]
    labs = [f"{int(t)}" for t in locs]
    if any(c["step"] is None for c in cps):
        locs.append(max(xs))
        labs.append("final")
        ax.axvline((xmax + max(xs)) / 2, color=P.RULE, lw=0.6, ls=":")
    ax.set_xticks(locs)
    ax.set_xticklabels(labs)
    if label:
        ax.set_xlabel("RL step")


def hollow_base(ax, cps, xs, ys, color):
    """Base model (step 0) drawn as a hollow dot with an ink ring."""
    for c, x, y in zip(cps, xs, ys):
        if c["ckpt"] == "base":
            ax.plot([x], [y], marker="o", markersize=3.8, markerfacecolor="white", markeredgecolor=P.INK,
                    markeredgewidth=0.6, linestyle="none", zorder=5)


def line_ci(ax, cps, get, color, marker="o", ls="-", lw=None, label=None, alpha_band=True, zorder=3, hollow=False):
    """Plot est line + CI band. get(cp) -> {est,lo,hi} or None (skipped)."""
    pts = [(x, get(c)) for x, c in zip(xpos(cps), cps)]
    pts = [(x, d) for x, d in pts if d is not None]
    if not pts:
        return None
    xs = [x for x, _ in pts]
    e = [d["est"] for _, d in pts]
    lo = [d.get("lo", d["est"]) for _, d in pts]
    hi = [d.get("hi", d["est"]) for _, d in pts]
    (ln,) = ax.plot(xs, e, color=color, ls=ls, lw=lw or P.LW["data"], label=label, zorder=zorder, marker=marker,
                    markerfacecolor="white" if hollow else color, markeredgecolor=P.INK, markeredgewidth=0.5, markersize=3.4)
    if alpha_band:
        P.band(ax, xs, lo, hi, color)
    return ln


def stamp(fig, pilot: bool, note: str = ""):
    if pilot:
        fig.supxlabel("PILOT data" + (f" · {note}" if note else ""), x=1.0, ha="right", fontsize=P.SIZE["legend"],
                     color=P.VERMILION, fontweight="bold")


def out_stem(name: str, pilot: bool) -> Path:
    return globals()["PLOTS"] / (name + ("_PILOT" if pilot else ""))


def finish(fig, name: str, pilot: bool, note: str = ""):
    stamp(fig, pilot, note)
    stem = out_stem(name, pilot)
    outs = P.save(fig, stem)
    for o in outs:
        print("wrote", o.relative_to(ROOT) if o.is_relative_to(ROOT) else o)
    return outs


def cli(desc: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--mode", choices=["auto", "main", "pilot"], default="auto",
                    help="auto: main-run files if present else the pilot files under exp/results/P0")
    ap.add_argument("--src", type=Path, nargs="*", help="explicit input JSON file(s), overrides discovery")
    ap.add_argument("--res", type=Path, default=RES, help="results root (default exp/results)")
    ap.add_argument("--series", nargs="*", help="restrict to these series (e.g. S1 S2)")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default figures/plots)")
    a = ap.parse_args()
    if a.out is not None:
        global PLOTS
        PLOTS = a.out
    return a


def discover(a, main_glob: str, pilot_glob: str) -> list[Path]:
    paths = _discover(a, main_glob, pilot_glob)
    if getattr(a, "series", None):
        paths = [p for p in paths if p.parent.name in a.series or any(s in p.name for s in a.series)]
    return paths


def _discover(a, main_glob: str, pilot_glob: str) -> list[Path]:
    """main_glob may hold several patterns separated by '|' (layouts: main/<series>/..., E<k>/<series>/...)."""
    if a.src:
        return list(a.src)
    main = sorted({p for g in main_glob.split("|") for p in a.res.glob(g)}) if a.mode in ("auto", "main") else []
    if main:
        return main
    return sorted(a.res.glob(pilot_glob)) if a.mode in ("auto", "pilot") else []


def series_of(path: Path, data: dict) -> str:
    return data.get("series") or (path.parent.name if path.parent.name not in ("P0",) else "S1")


def fig_legend(fig, ax, ncols: int = 2):
    """Frameless legend row at the top of the FIGURE (constrained layout reserves the space, so it never squeezes the axes
    or collides with panel titles). Handles come from `ax`; bands stay out."""
    h, l = ax.get_legend_handles_labels()
    return fig.legend(h, l, loc="outside upper center", ncols=ncols, frameon=False, fontsize=P.SIZE["legend"])
