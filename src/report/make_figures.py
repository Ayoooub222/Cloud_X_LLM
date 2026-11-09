"""Build the report figures from the records and the cost model.

Examples:
    python -m src.report.make_figures                  # every figure into docs/figures/
    python -m src.report.make_figures --only cost_vs_utilization break_even
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display needed

import matplotlib.pyplot as plt  # noqa: E402

from src.report.figures import all_figures  # noqa: E402
from src.report.style import DPI, apply_style  # noqa: E402

DEFAULT_OUT = Path("docs/figures")
PNG_METADATA = {"Software": None}   # keeps the files byte-stable across matplotlib builds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the report figures (PNG) from results/benchmark")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: docs/figures)")
    p.add_argument("--only", nargs="+", default=None, help="figure names to build (default: all)")
    return p.parse_args(argv)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25, metadata=PNG_METADATA)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_style()
    builders = all_figures()
    names = args.only or list(builders)
    unknown = sorted(set(names) - set(builders))
    if unknown:
        raise SystemExit(f"Unknown figure(s): {', '.join(unknown)}. Known: {', '.join(builders)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = args.out / f"{name}.png"
        save(builders[name](), path)
        print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
