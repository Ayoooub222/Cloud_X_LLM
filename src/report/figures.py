"""Report figures. Each function returns a matplotlib Figure built from src.report.data."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

from src.cost import model
from src.report import data, projection
from src.report.style import (
    API_GRAYS,
    AQUA,
    BLUE,
    BLUE_LIGHT,
    GPU_COLORS,
    LINE_WIDTH,
    ORANGE,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    end_label,
    new_figure,
    ringed_marker_kwargs,
    title,
    value_label,
)

UTILIZATION_MIN = 0.01
UTILIZATION_POINTS = 200
UTILIZATION_TICKS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
HOURS_PER_MONTH = 730.0
SPOT_DASHES = (4, 2.5)
MAIN_API = {"audio": "groq-whisper-large-v3-turbo", "vision": "gemini-3.1-flash-lite"}
TASK_TITLES = {"audio": "Audio: transcription of a 66 s clip", "vision": "Vision: extraction of one quote page"}
UNITS = {"audio": "transcriptions", "vision": "pages"}
API_LABELS = {
    "groq-whisper-large-v3-turbo": "Groq (same model)",
    "openai-gpt-4o-mini-transcribe": "OpenAI gpt-4o-mini-transcribe",
    "openai-whisper": "OpenAI Whisper",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gemini-3.1-flash-lite-batch": "Gemini 3.1 Flash-Lite, batch",
}
NEVER = "never"
BAR_HEIGHT = 0.28        # keeps bars thin (about 20 px at the export resolution)
T4_POWER_CAP_W = 70


# ---------- helpers ----------
def option_color(option: str) -> str:
    if option.startswith("g5"):
        return AQUA
    if option.startswith("g4dn"):
        return ORANGE
    return BLUE


def option_label(option: str) -> str:
    if option.startswith("local"):
        return "RTX 3070 Laptop, owned"
    instance, _, pricing = option.partition(" ")
    gpu = "A10G" if instance.startswith("g5") else "T4"
    return f"{instance} ({gpu}), {pricing}"


def percent(value: float, _pos: object = None) -> str:
    return f"{value:.0%}"


def eur_tick(value: float, _pos: object = None) -> str:
    if value >= 1:
        return f"{value:,.0f}"
    if value >= 0.1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def compact(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:.0f}"


def concurrency_axis(ax: Axes, levels: tuple[int, ...]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, [str(c) for c in levels])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Concurrent requests")


def mean_sd(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.groupby(["gpu", "concurrency"])[column].agg(["mean", "std"]).reset_index()


def gpu_lines(ax: Axes, frame: pd.DataFrame, column: str, scale: float = 1.0, label_fmt: str = "{:.0f}") -> None:
    """Mean ± standard deviation per GPU and concurrency, labeled at the line end."""
    stats = mean_sd(frame, column)
    for gpu, rows in stats.groupby("gpu"):
        color = GPU_COLORS[gpu]
        x, y, err = rows["concurrency"], rows["mean"] * scale, rows["std"].fillna(0) * scale
        ax.errorbar(x, y, yerr=err, color=color, lw=LINE_WIDTH, capsize=0, elinewidth=1,
                    label=data.GPU_LABELS[gpu], **ringed_marker_kwargs(color))
        end_label(ax, x.iloc[-1], y.iloc[-1], label_fmt.format(y.iloc[-1]))


def gpu_legend(fig: Figure, extra: list[tuple[Line2D, str]] | None = None) -> None:
    handles = [Line2D([], [], color=GPU_COLORS[g], lw=LINE_WIDTH, **ringed_marker_kwargs(GPU_COLORS[g]))
               for g in data.GPU_LABELS]
    labels = list(data.GPU_LABELS.values())
    for handle, label in extra or []:
        handles.append(handle)
        labels.append(label)
    fig.legend(handles, labels, loc="outside lower center", ncols=len(labels))


# ---------- projection map over model size and volume ----------
DIVERGING = ("#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#a62b2a")
INFEASIBLE_COLORS = {projection.INFEASIBLE_VRAM: "#d3d1c9", projection.INFEASIBLE_SLO: "#e6e4dd"}
RATIO_LIMIT = 3.0                       # color scale clipped at 8x either way
PROJECTION_SIZES_B = np.geomspace(0.5, 12.0, 110)
PROJECTION_VOLUMES = np.geomspace(1e3, 1e7, 130)
REGION_MIN_SHARE = 0.03                 # label a region only if it covers 3 % of the panel
WINNER_LABELS = {"API": "API cheaper", "g5.xlarge spot": "g5.xlarge spot\n(A10G) cheaper",
                 "g5.xlarge on-demand": "g5.xlarge on-demand\ncheaper",
                 "g4dn.xlarge spot": "g4dn.xlarge spot\n(T4) cheaper",
                 "g4dn.xlarge on-demand": "g4dn.xlarge on-demand\ncheaper",
                 projection.INFEASIBLE_VRAM: "needs a GPU with more\nmemory (not priced)",
                 projection.INFEASIBLE_SLO: "30 s SLO not reachable\non one GPU"}


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("cost_ratio", DIVERGING)


def region_labels(ax: Axes, winners: np.ndarray) -> None:
    log_v, log_s = np.meshgrid(np.log10(PROJECTION_VOLUMES), np.log10(PROJECTION_SIZES_B))
    for winner in sorted(set(winners.flat)):
        mask = winners == winner
        if mask.mean() < REGION_MIN_SHARE:
            continue
        x, y = 10 ** np.median(log_v[mask]), 10 ** np.median(log_s[mask])
        ax.text(x, y, WINNER_LABELS.get(winner, winner), ha="center", va="center", fontsize=8.5,
                color=TEXT_PRIMARY, fontweight="bold" if winner != "API" else "normal",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2, "alpha": 0.75})


def projection_panel(ax: Axes, report: dict, batch: bool) -> object:
    options = projection.vision_options(report)
    api = projection.api_eur_per_1000(report, batch)
    slo = None if batch else projection.slo_s(report)
    log_ratio, winners = projection.grid(options, PROJECTION_SIZES_B, PROJECTION_VOLUMES, api, slo)
    for reason, color in INFEASIBLE_COLORS.items():
        ax.pcolormesh(PROJECTION_VOLUMES, PROJECTION_SIZES_B, np.where(winners == reason, 1.0, np.nan),
                      cmap=ListedColormap([color]), shading="nearest", rasterized=True)
    mesh = ax.pcolormesh(PROJECTION_VOLUMES, PROJECTION_SIZES_B, np.clip(log_ratio, -RATIO_LIMIT, RATIO_LIMIT),
                         cmap=diverging_cmap(), vmin=-RATIO_LIMIT, vmax=RATIO_LIMIT, shading="nearest",
                         rasterized=True)
    if np.nanmin(log_ratio) < 0 < np.nanmax(log_ratio):
        ax.contour(PROJECTION_VOLUMES, PROJECTION_SIZES_B, log_ratio, levels=[0], colors=TEXT_PRIMARY,
                   linewidths=1.3)
    region_labels(ax, winners)
    ax.axhline(projection.MEASURED_SIZE_B, color=TEXT_PRIMARY, lw=0.9)
    ax.text(PROJECTION_VOLUMES[0] * 1.15, projection.MEASURED_SIZE_B * 1.04, "measured: Qwen3-VL-2B",
            fontsize=8, color=TEXT_PRIMARY, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7], ["1k", "10k", "100k", "1M", "10M"])
    ax.set_yticks([0.5, 1, 2, 4, 8, 12], ["0.5B", "1B", "2B", "4B", "8B", "12B"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(False)
    ax.set_xlabel("Pages per month")
    ax.set_ylabel("Model size (parameters, 16-bit weights)")
    api_name = "Gemini Flash-Lite batch" if batch else "Gemini Flash-Lite"
    mode = "Batch work, no latency target" if batch else "Interactive, p95 under 30 s"
    ax.set_title(f"{mode}\nagainst {api_name}, {api:.2f} EUR per 1000 pages")
    return mesh


def figure_size_volume_map(report: dict) -> Figure:
    fig, axes = new_figure(ncols=2, height=5.4)
    mesh = None
    for ax, batch in zip(axes, (False, True), strict=True):
        mesh = projection_panel(ax, report, batch)
    bar = fig.colorbar(mesh, ax=axes, orientation="horizontal", location="bottom", shrink=0.55, aspect=40,
                       pad=0.02)
    ticks = np.arange(-RATIO_LIMIT, RATIO_LIMIT + 1)
    bar.set_ticks(ticks, labels=["8× cheaper", "4×", "2×", "same", "2×", "4×", "8× dearer"])
    bar.set_label("Cheapest self-hosted cloud option against the API (blue: self-hosting wins)",
                  color=TEXT_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0)
    title(fig, "When does self-hosting a document-extraction VLM pay off? A projection from the 2B measurement",
          "Projected, not measured: throughput scales as 1 / model size (memory-bound decoding), weights must "
          "fit in 60 % of the VRAM (T4 16 GB, A10G 24 GB), one instance runs 24/7 and extra volume pays "
          "extra instance-hours. Black line: break-even. Larger models are usually more accurate; "
          "the API price is fixed.")
    return fig


# ---------- main: break-even model size, the simple view ----------
SIZE_RANGE_B = (0.3, 200.0)
LOW_UTILIZATION = 0.25
SIDE_WASH = 0.07
MAIN_TASKS = ("vision", "audio")


def size_ticks(ax: Axes) -> None:
    ticks = [0.5, 1, 2, 5, 10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [f"{v:g}B" for v in ticks])
    ax.xaxis.set_minor_locator(NullLocator())


def break_even_mark(ax: Axes, size: float, cost: float, color: str, text: str, upper_left: bool = False) -> None:
    """Dot on a crossing. A rising line leaves room below-right of its own points (and the dashed,
    low-utilization line lies above-left of the solid one), so labels go there."""
    ax.plot([size], [cost], linestyle="none", zorder=5, **ringed_marker_kwargs(color) | {"markersize": 8})
    offset, ha, va = ((-7, 7), "right", "bottom") if upper_left else ((7, -7), "left", "top")
    ax.annotate(text, (size, cost), xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=8.5,
                color=TEXT_PRIMARY, fontweight="bold",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8, "alpha": 0.8})


def label_on_left(line: projection.SizeLine, price: float) -> bool:
    """API labels go to the edge farther from where the full-load line crosses that API (its dot and
    label sit there)."""
    crossing = line.break_even_b(price)
    if crossing is None:
        return True
    return np.log(crossing / SIZE_RANGE_B[0]) > np.log(SIZE_RANGE_B[1] / crossing)


def size_panel(ax: Axes, report: dict, task: str) -> None:
    workload = next(w for w in report["workloads"] if w["task"] == task)
    line = projection.cheapest_line(report, task)
    color = option_color(line.option)
    sizes = np.geomspace(*SIZE_RANGE_B, 200)
    main_api = workload["apis_eur_per_1000"][MAIN_API[task]]
    crossing = line.break_even_b(main_api)
    if crossing is not None:
        ax.axvspan(SIZE_RANGE_B[0], crossing, color=BLUE, alpha=SIDE_WASH, lw=0)
        ax.axvspan(crossing, SIZE_RANGE_B[1], color=DIVERGING[-2], alpha=SIDE_WASH, lw=0)
        ax.text(SIZE_RANGE_B[0] * 1.15, 0.97, "self-hosting cheaper", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=8.5, color=TEXT_SECONDARY)
        ax.text(SIZE_RANGE_B[1] / 1.15, 0.97, f"{API_LABELS[MAIN_API[task]]} cheaper",
                transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=8.5, color=TEXT_SECONDARY)
    apis = sorted(workload["apis_eur_per_1000"].items(), key=lambda item: item[1])
    for (name, price), gray in zip(apis, API_GRAYS, strict=False):
        ax.axhline(price, color=gray, lw=1.0, zorder=1)
        left = label_on_left(line, price)
        below = name == apis[0][0]          # the cheapest API is labeled under its line, away from the others
        ax.text(SIZE_RANGE_B[0] * 1.15 if left else SIZE_RANGE_B[1] / 1.15, price / 1.08 if below else price * 1.08,
                f"{API_LABELS.get(name, name)}: {price:.2f} EUR", fontsize=8, color=TEXT_SECONDARY,
                va="top" if below else "bottom", ha="left" if left else "right", zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6, "alpha": 0.85})
    ax.plot(sizes, line.cost(sizes), color=color, lw=LINE_WIDTH + 0.4, zorder=3)
    ax.plot(sizes, line.cost(sizes, LOW_UTILIZATION), color=color, lw=LINE_WIDTH, dashes=SPOT_DASHES, zorder=3)
    for _name, price in apis:
        size = line.break_even_b(price)
        if size is not None and SIZE_RANGE_B[0] < size < SIZE_RANGE_B[1]:
            break_even_mark(ax, size, price, color, f"≈ {size:.1f}B" if size < 10 else f"≈ {size:.0f}B")
    low = line.break_even_b(main_api, LOW_UTILIZATION)
    if low is not None and SIZE_RANGE_B[0] < low:
        break_even_mark(ax, low, main_api, color, f"≈ {low:.1f}B", upper_left=True)
    anchor = line.anchor_b
    ax.plot([anchor], [line.cost(anchor)], linestyle="none", zorder=6,
            **ringed_marker_kwargs(TEXT_PRIMARY) | {"markersize": 7})
    y_low, y_high = line.cost(SIZE_RANGE_B[0]), line.cost(SIZE_RANGE_B[1], LOW_UTILIZATION)
    low_point = np.log(line.cost(anchor) / y_low) < 0.3 * np.log(y_high / y_low)
    ax.annotate(f"measured:\n{projection.ANCHOR_MODEL[task]}", (anchor, line.cost(anchor)),
                xytext=(40, -6) if low_point else (34, -46),
                textcoords="offset points", ha="left", va="top", fontsize=8, color=TEXT_PRIMARY,
                arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "lw": 0.8, "shrinkB": 4})
    limit = projection.max_size_b(projection.GPU_VRAM_GB[line.option.split(".")[0]])
    ax.axvline(limit, color=TEXT_MUTED, lw=0.8)
    ax.text(limit * 1.06, 0.88, f"one {projection.GPU_NAMES[line.option.split('.')[0]]} holds\n"
            f"up to ~{limit:.0f}B; beyond,\nseveral GPUs", transform=ax.get_xaxis_transform(), fontsize=7.5,
            color=TEXT_MUTED, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*SIZE_RANGE_B)
    size_ticks(ax)
    ax.yaxis.set_major_formatter(FuncFormatter(eur_tick))
    ax.grid(True, axis="both", which="major")
    ax.set_xlabel("Model size (billions of parameters)")
    ax.set_ylabel("EUR per 1000 requests")
    ax.set_title(f"{TASK_TITLES[task]}\nself-hosted on {option_label(line.option)}")


def figure_main(report: dict) -> Figure:
    fig, axes = new_figure(ncols=2, height=5.0)
    for ax, task in zip(axes, MAIN_TASKS, strict=True):
        size_panel(ax, report, task)
    handles = [Line2D([], [], color=TEXT_SECONDARY, lw=LINE_WIDTH + 0.4),
               Line2D([], [], color=TEXT_SECONDARY, lw=LINE_WIDTH, dashes=SPOT_DASHES),
               Line2D([], [], color=API_GRAYS[1], lw=1.0),
               Line2D([], [], linestyle="none", **ringed_marker_kwargs(TEXT_SECONDARY) | {"markersize": 8})]
    fig.legend(handles, ["self-hosted, GPU busy 100 % of the time", f"self-hosted, GPU busy {LOW_UTILIZATION:.0%}",
                         "API price (flat)", "break-even model size"], loc="outside lower center", ncols=4)
    title(fig, "Up to what model size does self-hosting beat an API?",
          "Self-hosted cost per request grows with the model (memory-bound decoding: twice the parameters, half the "
          "throughput), anchored on the measured model; API prices stay flat. Cheapest cloud GPU at full load, "
          "spot pricing, latency targets ignored. A projection, not a measurement.")
    return fig


# ---------- 1. cost per 1000 requests against utilization ----------
def option_cost(option: dict, u: np.ndarray) -> np.ndarray:
    return model.REQUESTS * model.cost_per_request(option["hourly_eur"], option["requests_per_hour"], u,
                                                   option["variable_eur_per_1000"] / model.REQUESTS)


def priced_options(workload: dict) -> list[dict]:
    return [o for o in workload["options"] if "eur_per_1000" in o]


def cost_curves(ax: Axes, workload: dict) -> list[Line2D]:
    """One curve per self-hosted option; a dot where it crosses the closest API."""
    u = np.geomspace(UTILIZATION_MIN, 1.0, UTILIZATION_POINTS)
    api = MAIN_API[workload["task"]]
    handles = []
    for option in priced_options(workload):
        name = option["option"]
        style = {"color": option_color(name), "lw": LINE_WIDTH}
        if name.endswith("spot"):
            style["dashes"] = SPOT_DASHES
        (line,) = ax.plot(u, option_cost(option, u), label=option_label(name), **style)
        handles.append(line)
        break_even = option["break_even_utilization"][api]
        if break_even != NEVER:
            ax.plot([break_even], [workload["apis_eur_per_1000"][api]], linestyle="none", zorder=4,
                    **ringed_marker_kwargs(option_color(name)))
    return handles


def clearest_side(workload: dict, eur_1000: float) -> float:
    """x (left or right edge) where the API label is farthest from every cost curve."""
    def gap(x: float) -> float:
        return min(abs(np.log(option_cost(o, np.array([x]))[0] / eur_1000)) for o in priced_options(workload))
    return max((UTILIZATION_MIN, 1.0), key=gap)


def api_lines(ax: Axes, workload: dict) -> None:
    apis = sorted(workload["apis_eur_per_1000"].items(), key=lambda item: item[1])
    for (name, eur_1000), gray in zip(apis, API_GRAYS, strict=False):
        ax.axhline(eur_1000, color=gray, lw=1.0, zorder=1)
        x = clearest_side(workload, eur_1000)
        ax.text(x * (1.06 if x < 1 else 0.95), eur_1000 * 1.09, f"{API_LABELS.get(name, name)}: {eur_1000:.2f} EUR",
                fontsize=8, color=TEXT_SECONDARY, va="bottom", ha="left" if x < 1 else "right",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6, "alpha": 0.85})


def slo_note(ax: Axes, workload: dict) -> None:
    """Say on the panel which options cannot meet the SLO (their cost holds for batch work only)."""
    blocked = [o for o in priced_options(workload) if not o.get("slo_reachable", True)]
    if not blocked:
        return
    first = blocked[0]
    x = 0.3
    y = option_cost(first, np.array([x]))[0]
    ax.annotate("g4dn (T4): SLO not reachable,\none page alone takes longer than 30 s;\ncost valid for batch work only",
                (x, y), xytext=(0.97, 0.80), textcoords="axes fraction", ha="right", fontsize=8,
                color=TEXT_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": TEXT_MUTED, "lw": 0.8})


def figure_cost_vs_utilization(report: dict) -> Figure:
    fig, axes = new_figure(ncols=2, height=5.2)
    handles: dict[str, Line2D] = {}
    for ax, workload in zip(axes, report["workloads"], strict=True):
        api_lines(ax, workload)
        for line in cost_curves(ax, workload):
            handles.setdefault(line.get_label(), line)
        slo_note(ax, workload)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(UTILIZATION_MIN, 1.0)
        ax.set_xticks(UTILIZATION_TICKS)
        ax.xaxis.set_major_formatter(FuncFormatter(percent))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(eur_tick))
        ax.grid(True, axis="both", which="major")
        ax.set_xlabel("Utilization (share of paid hours spent serving at the SLO rate)")
        ax.set_ylabel("EUR per 1000 requests (log scale)")
        ax.set_title(TASK_TITLES[workload["task"]])
    fig.legend(list(handles.values()), list(handles), loc="outside lower center", ncols=3)
    title(fig, "Self-hosting gets cheaper with utilization; an API costs the same at any volume",
          "Gray lines: API prices. Solid: owned or on-demand, dashed: spot. A dot marks the break-even point "
          "against the closest API (Groq for audio, Gemini Flash-Lite for vision).")
    return fig


# ---------- 2. break-even utilization ----------
def figure_break_even(report: dict) -> Figure:
    fig, axes = new_figure(ncols=2, height=3.4)
    for ax, workload in zip(axes, report["workloads"], strict=True):
        api = MAIN_API[workload["task"]]
        api_eur = workload["apis_eur_per_1000"][api]
        rows = [o for o in workload["options"] if "eur_per_1000" in o]
        rows.sort(key=lambda o: (o["break_even_utilization"][api] == NEVER,
                                 o["break_even_utilization"][api] if o["break_even_utilization"][api] != NEVER else 0))
        labels = [option_label(o["option"]) for o in rows]
        y = np.arange(len(rows))
        for i, option in enumerate(rows):
            value = option["break_even_utilization"][api]
            if value == NEVER:
                value_label(ax, 0, i, "never cheaper", dx=2)
                continue
            ax.barh(i, value, height=BAR_HEIGHT, color=option_color(option["option"]))
            volume = option["hourly_eur"] * HOURS_PER_MONTH / (api_eur / model.REQUESTS)
            value_label(ax, value, i, f"{value:.0%}  (~{compact(volume)} {UNITS[workload['task']]}/month)")
        ax.set_yticks(y, labels)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(FuncFormatter(percent))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(f"Utilization above which self-hosting is cheaper than {API_LABELS[api]}")
        ax.set_title(TASK_TITLES[workload["task"]])
    title(fig, "Break-even utilization against the closest API",
          "Monthly volume at break-even = hourly price × 730 h / API price per request: it does not depend on "
          "throughput, which only sets the ceiling.")
    return fig


# ---------- 3. throughput against concurrency ----------
def figure_throughput() -> Figure:
    fig, axes = new_figure(ncols=2, height=3.8)
    for ax, task in zip(axes, ("audio", "vision"), strict=True):
        gpu_lines(ax, data.concurrency_frame(task), "requests_per_hour", label_fmt="{:,.0f}")
        concurrency_axis(ax, data.SERIES[task]["levels"])
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
        ax.set_ylabel(f"{UNITS[task].capitalize()} per hour (batch throughput)")
        ax.set_title(TASK_TITLES[task])
    gpu_legend(fig)
    title(fig, "Throughput grows with batching, then the GPU saturates",
          "Same inputs and server settings on both GPUs; mean of 3 runs, bars = standard deviation. "
          "Burst throughput of a batch, higher than the rate sustained under the SLO (k6).")
    return fig


# ---------- 4. T4 relative to the 3070 ----------
def figure_t4_ratio() -> Figure:
    fig, axes = new_figure(ncols=1, height=3.6)
    ax = axes[0]
    for task, marker, dashes in (("audio", "o", None), ("vision", "s", SPOT_DASHES)):
        frame = data.ratio_frame(task)
        style = ringed_marker_kwargs(ORANGE) | {"marker": marker}
        if dashes:
            style["dashes"] = dashes
        ax.plot(frame["concurrency"], frame["ratio"], color=ORANGE, lw=LINE_WIDTH, label=task.capitalize(), **style)
        end_label(ax, frame["concurrency"].iloc[-1], frame["ratio"].iloc[-1],
                  f"{task}: × {frame['ratio'].iloc[-1]:.2f} (used by the cost model)")
    ax.axhline(1.0, color=TEXT_MUTED, lw=1.0)
    ax.text(1, 1.02, "same throughput as the RTX 3070 Laptop", fontsize=8, color=TEXT_MUTED, va="bottom")
    concurrency_axis(ax, data.SERIES["audio"]["levels"])
    ax.set_xlim(0.8, 13)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("T4 throughput / RTX 3070 Laptop throughput")
    ax.legend(loc="lower left")
    title(fig, "The T4 falls further behind as the batch grows",
          "Batching hides less of the slower attention kernels on Turing (Triton without tensor cores below "
          "SM 8.0 in vLLM 0.29). The ratio at saturation scales the 3070 throughput to a g4dn.xlarge.")
    return fig


# ---------- 5. energy per request ----------
def figure_energy() -> Figure:
    fig, axes = new_figure(ncols=2, height=3.8)
    for ax, task in zip(axes, ("audio", "vision"), strict=True):
        frame = data.concurrency_frame(task)
        if task == "audio":
            gpu_lines(ax, frame, "energy_wh_per_request", scale=data.MWH_PER_WH, label_fmt="{:.1f} mWh")
            ax.set_ylabel("GPU energy per transcription (mWh)")
        else:
            gpu_lines(ax, frame, "energy_wh_per_request", label_fmt="{:.2f} Wh")
            ax.set_ylabel("GPU energy per page (Wh)")
        concurrency_axis(ax, data.SERIES[task]["levels"])
        ax.set_ylim(bottom=0)
        ax.set_title(TASK_TITLES[task])
    gpu_legend(fig)
    title(fig, "Batching divides the energy per request; the T4 spends more per request",
          "GPU energy from nvidia-smi over the batch, divided by the requests; wall power is not measured.")
    return fig


# ---------- 6. k6 load steps and the SLO ----------
def sustained_rate(frame: pd.DataFrame) -> float | None:
    ok = frame.groupby("target_rps")["slo_ok"].all()
    passing = ok[ok].index
    return float(passing.max()) if len(passing) else None


def figure_load() -> Figure:
    fig, axes = new_figure(ncols=2, height=3.9)
    for ax, task in zip(axes, ("audio", "vision"), strict=True):
        frame = data.load_frame(task)
        slo = frame["slo_p95_s"].iloc[0]
        means = frame.groupby("target_rps")["latency_p95_s"].mean()
        ax.plot(means.index, means.to_numpy(), color=BLUE, lw=LINE_WIDTH)
        ok, ko = frame[frame["slo_ok"]], frame[~frame["slo_ok"]]
        ax.plot(ok["target_rps"], ok["latency_p95_s"], linestyle="none", **ringed_marker_kwargs(BLUE))
        ax.plot(ko["target_rps"], ko["latency_p95_s"], linestyle="none", marker="o", markersize=6,
                markerfacecolor=SURFACE, markeredgecolor=BLUE, markeredgewidth=1.4)
        ax.axhline(slo, color=TEXT_SECONDARY, lw=1.0)
        ax.text(frame["target_rps"].min(), slo * 1.04, f"SLO: p95 < {slo:g} s", fontsize=8,
                color=TEXT_SECONDARY, va="bottom")
        rate = sustained_rate(frame)
        if rate is not None:
            ax.axvline(rate, color=TEXT_MUTED, lw=1.0)
            unit = "req/s" if task == "audio" else "pages/s"
            ax.annotate(f"sustained: {rate:g} {unit}\n= {rate * 3600:,.0f} per hour", (rate, ax.get_ylim()[1]),
                        xytext=(-4, -4), textcoords="offset points", ha="right", va="top", fontsize=8.5,
                        color=TEXT_PRIMARY)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Arrival rate (requests per second, constant)")
        ax.set_ylabel("p95 latency (s)")
        ax.set_title(TASK_TITLES[task] + ", RTX 3070 Laptop")
    handles = [Line2D([], [], linestyle="none", **ringed_marker_kwargs(BLUE)),
               Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=SURFACE,
                      markeredgecolor=BLUE, markeredgewidth=1.4),
               Line2D([], [], color=BLUE, lw=LINE_WIDTH)]
    fig.legend(handles, ["run meeting the SLO", "run missing the SLO", "mean of 3 runs"],
               loc="outside lower center", ncols=3)
    title(fig, "The cost model uses the highest rate at which every run meets the SLO",
          "k6 constant-arrival-rate steps, 3 runs each. Past the sustained rate, queues build up and p95 explodes.")
    return fig


# ---------- 7. quality ----------
def quality_points(workload: dict) -> list[tuple[str, float, str]]:
    q = workload["quality"]
    points = [("RTX 3070 Laptop, bf16", q["self_hosted"], BLUE)]
    points += [("Tesla T4, fp16", v, ORANGE) for v in q["variants"].values() if v is not None]
    api = MAIN_API[workload["task"]]
    if q["apis"].get(api) is not None:
        points.append((API_LABELS[api], q["apis"][api], API_GRAYS[0]))
    return points


def figure_quality(report: dict) -> Figure:
    fig, axes = new_figure(ncols=2, height=2.9)
    by_task = {w["task"]: w for w in report["workloads"]}
    specs = (("audio", "Word error rate (lower is better)", (0, 0.05)),
             ("vision", "Field accuracy (higher is better)", (0.9, 1.0)))
    for ax, (task, xlabel, limits) in zip(axes, specs, strict=True):
        points = quality_points(by_task[task])
        for i, (_label, value, color) in enumerate(points):
            ax.hlines(i, limits[0], value, color=color, lw=1.0, alpha=0.35)
            ax.plot([value], [i], linestyle="none", clip_on=False, **ringed_marker_kwargs(color) | {"markersize": 8})
            value_label(ax, value, i, f"{value:.1%}", dx=7)
        ax.set_yticks(range(len(points)), [p[0] for p in points])
        ax.invert_yaxis()
        ax.set_xlim(*limits)
        ax.set_ylim(len(points) - 0.4, -0.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.0%}"))
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlabel)
        ax.set_title(TASK_TITLES[task])
    title(fig, "fp16 on the T4 gives the same quality as bf16 on the 3070",
          "Same model on every self-hosted option. One LibriSpeech clip (148 words) and one generated quote "
          "page (92 fields): small samples. The vision axis starts at 90 %.")
    return fig


# ---------- 8. cold start ----------
def figure_coldstart() -> Figure:
    fig, axes = new_figure(ncols=1, height=3.2)
    ax = axes[0]
    stats = data.coldstart_frame().groupby(["model", "deploy"])["t_first_s"].mean().unstack("deploy")
    models = ["vision", "audio", "image"]
    shades = {"Docker Compose": BLUE_LIGHT, "k3s + Helm": BLUE}
    height = BAR_HEIGHT
    for j, (deploy, color) in enumerate(shades.items()):
        y = np.arange(len(models)) + (j - 0.5) * (height + 0.04)
        values = stats.loc[models, deploy].to_numpy()
        ax.barh(y, values, height=height, color=color, label=deploy)
        for yi, v in zip(y, values, strict=True):
            value_label(ax, v, yi, f"{v:.0f} s")
    ax.set_yticks(range(len(models)), ["Vision (Qwen3-VL-2B)", "Audio (Whisper large-v3-turbo)",
                                       "Image (stable-diffusion.cpp)"])
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Seconds from container start to the first answer")
    ax.legend(loc="lower right")
    title(fig, "Cold start: one to two minutes for the vLLM models",
          "RTX 3070 Laptop, weights on local disk, mean of 2 to 5 starts. Model loading and CUDA graph capture "
          "dominate; k3s adds 1 to 11 s over Docker Compose.")
    return fig


# ---------- 9. GPU utilization and power ----------
def figure_gpu_power() -> Figure:
    fig, axes = new_figure(ncols=2, nrows=2, height=6.2)
    for col, task in enumerate(("audio", "vision")):
        frame = data.concurrency_frame(task)
        util_ax, power_ax = axes[col], axes[2 + col]
        gpu_lines(util_ax, frame, "gpu_util_pct", label_fmt="{:.0f} %")
        gpu_lines(power_ax, frame, "gpu_power_w", label_fmt="{:.0f} W")
        util_ax.set_ylim(0, 105)
        util_ax.set_ylabel("GPU utilization (%)")
        util_ax.set_title(TASK_TITLES[task])
        power_ax.axhline(T4_POWER_CAP_W, color=TEXT_MUTED, lw=1.0)
        power_ax.set_ylim(0, 110)
        power_ax.set_ylabel("Average GPU power (W)")
        for ax in (util_ax, power_ax):
            concurrency_axis(ax, data.SERIES[task]["levels"])
    gpu_legend(fig, extra=[(Line2D([], [], color=TEXT_MUTED, lw=1.0), f"T4 power cap ({T4_POWER_CAP_W} W)")])
    title(fig, "Both GPUs saturate; the T4 runs at its power cap",
          "nvidia-smi samples during each batch (every 0.25 s for audio, 1 s for vision), mean of 3 runs.")
    return fig


def all_figures() -> dict[str, Callable[[], Figure]]:
    """Name -> builder. Builders taking the cost report get it from one shared computation."""
    report = data.cost_report()
    return {
        "main": lambda: figure_main(report),
        "size_volume_map": lambda: figure_size_volume_map(report),
        "cost_vs_utilization": lambda: figure_cost_vs_utilization(report),
        "break_even": lambda: figure_break_even(report),
        "throughput": figure_throughput,
        "t4_ratio": figure_t4_ratio,
        "energy": figure_energy,
        "load_slo": figure_load,
        "quality": lambda: figure_quality(report),
        "coldstart": figure_coldstart,
        "gpu_power": figure_gpu_power,
    }

