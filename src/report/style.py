"""Visual style shared by every report figure.

Colors come from a validated categorical palette (fixed slot order, colorblind-safe for adjacent
pairs; the first three slots also for all pairs). Each entity keeps its color in every figure:
the RTX 3070 Laptop is always blue, the T4 (Kaggle or g4dn) always orange, the A10G (g5) aqua.
Text never takes a series color, gridlines are solid hairlines, marks are thin.
"""
from __future__ import annotations

import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"      # slot 1
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3
BLUE_LIGHT = "#86b6ef"  # blue ramp step 250, second shade of the same entity family
API_GRAYS = ("#52514e", "#898781", "#b5b3ab")  # reference lines (APIs), not series

GPU_COLORS = {"rtx3070": BLUE, "t4": ORANGE, "a10g": AQUA}

LINE_WIDTH = 1.6      # ~2 px at the export resolution
MARKER_SIZE = 6       # ~8 px
DPI = 200
FIG_WIDTH = 10.0      # inches, two panels side by side
SUBTITLE_WRAP = 150   # characters per subtitle line at 8.5 pt on FIG_WIDTH
SUBTITLE_LINE_IN = 0.17  # height of one subtitle line, inches
TITLE_GAP_IN = 0.08


def apply_style() -> None:
    """Global matplotlib/seaborn settings; call once before building figures."""
    sns.set_theme(style="white", context="notebook")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "xtick.labelcolor": TEXT_SECONDARY,
        "ytick.labelcolor": TEXT_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": TEXT_SECONDARY,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "svg.fonttype": "none",
    })


def new_figure(ncols: int = 2, height: float = 4.0, nrows: int = 1, sharey: bool = False) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, height), sharey=sharey, squeeze=False,
                             layout="constrained")
    return fig, list(axes.flat)


def title(fig: Figure, text: str, subtitle: str | None = None) -> None:
    """Title (what the figure shows) and subtitle (how to read it), left-aligned above the panels.

    Placed with fig.text above the plotting area so that the constrained layout and the bottom legend
    never collide with them; savefig(bbox_inches="tight") grows the canvas to include them.
    """
    lines = textwrap.wrap(subtitle, SUBTITLE_WRAP) if subtitle else []
    height = fig.get_figheight()
    top = 1.0 + (SUBTITLE_LINE_IN * len(lines) + TITLE_GAP_IN) / height
    fig.text(0.0, top, text, ha="left", va="bottom", fontsize=13, fontweight="bold", color=TEXT_PRIMARY)
    if lines:
        fig.text(0.0, top - 0.03 / height, "\n".join(lines), ha="left", va="top", fontsize=8.5,
                 color=TEXT_MUTED, linespacing=1.35)


def end_label(ax: Axes, x: float, y: float, text: str, dx: float = 6, dy: float = 0) -> None:
    """Direct label at the end of a line, in text ink (never in the series color)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=TEXT_SECONDARY)


def value_label(ax: Axes, x: float, y: float, text: str, ha: str = "left", dx: float = 4, dy: float = 0) -> None:
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", ha=ha,
                fontsize=8.5, color=TEXT_PRIMARY)


def ringed_marker_kwargs(color: str) -> dict:
    """Filled marker with a surface-colored ring, legible where it crosses other marks."""
    return {"marker": "o", "markersize": MARKER_SIZE, "markerfacecolor": color, "markeredgecolor": SURFACE,
            "markeredgewidth": 1.2}
