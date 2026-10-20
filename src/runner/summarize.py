"""Aggregate benchmark records (results/benchmark/<task>/*.json) into a table.

Rows are the numeric fields of each record's "metrics" and "gpu" sections
(gpu fields are prefixed with "gpu."). Columns are tags, or individual runs
with --runs. Failed runs (non-empty "error") are counted but excluded from
the statistics. Cells show "mean ±stdev" when a column has several runs.

Examples:
    python -m src.runner.summarize image
    python -m src.runner.summarize image --tag testA-notiling --tag testA-tiling
    python -m src.runner.summarize image --since "2026-10-19 14:00:00" --runs
    python -m src.runner.summarize image --fields time vram power energy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from collections import defaultdict
from pathlib import Path

BENCH_DIR = Path(os.getenv("BENCH_DIR", "results/benchmark"))

LARGE_VALUE = 100.0   # >= 100 -> 1 decimal  (e.g. vram 8192.0)
MEDIUM_VALUE = 1.0    # >= 1   -> 2 decimals (e.g. latency 1.09)
SMALL_DECIMALS = 4    # < 1    -> 4 decimals (e.g. rtf 0.0106)



def load_records(task: str, since: str | None, tags: list[str] | None) -> list[dict]:
    task_dir = BENCH_DIR / task
    if not task_dir.is_dir():
        raise SystemExit(f"Benchmark directory not found: {task_dir}")
    records = []
    for path in sorted(task_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping unreadable file {path.name}: {exc}")
            continue
        if since and str(record.get("timestamp", "")) < since:
            continue
        if tags and record.get("tag") not in tags:
            continue
        records.append(record)
    return records


def flatten_numeric(data: dict, prefix: str = "") -> dict[str, float]:
    """Recursively collect numeric leaves as {"a.b": value}; booleans are ignored."""
    out: dict[str, float] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_numeric(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def numeric_fields(record: dict) -> dict[str, float]:
    return flatten_numeric(record.get("metrics") or {}) | flatten_numeric(
        record.get("gpu") or {}, "gpu."
    )


def is_failure(record: dict) -> bool:
    return bool(record.get("error"))


def decimals_for(value: float) -> int:
    """Number of decimals that keeps small values (rtf, energy) readable."""
    size = abs(value)
    if size >= LARGE_VALUE:
        return 1
    if size >= MEDIUM_VALUE:
        return 2
    return SMALL_DECIMALS


def format_cell(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.fmean(values)
    digits = decimals_for(mean)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ±{statistics.stdev(values):.{digits}f}"



def build_rows(groups: dict[str, list[dict]], fields_filter: list[str] | None) -> list[list[str]]:
    values: dict[str, dict[str, list[float]]] = {label: defaultdict(list) for label in groups}
    names: list[str] = []
    for label, records in groups.items():
        for record in records:
            if is_failure(record):
                continue
            for name, value in numeric_fields(record).items():
                values[label][name].append(value)
                if name not in names:
                    names.append(name)
    if fields_filter:
        names = [n for n in names if any(f in n for f in fields_filter)]

    rows = [["runs ok/total"] + [
        f"{sum(not is_failure(r) for r in records)}/{len(records)}" for records in groups.values()
    ]]
    rows += [[name] + [format_cell(values[label].get(name, [])) for label in groups] for name in names]
    return rows


def print_table(header: list[str], rows: list[list[str]]) -> None:
    table = [header] + rows
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for index, row in enumerate(table):
        cells = [row[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(row[1:], widths[1:])]
        print("  ".join(cells))
        if index == 0:
            print("  ".join("-" * w for w in widths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON records.")
    parser.add_argument("task", help="Task folder under BENCH_DIR (image, vision, audio).")
    parser.add_argument("--tag", action="append", help="Keep only this tag (repeatable).")
    parser.add_argument("--since", help='Keep records with timestamp >= "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument("--fields", nargs="+", help="Keep only fields containing one of these substrings.")
    parser.add_argument("--runs", action="store_true", help="One column per run instead of one per tag.")
    args = parser.parse_args()

    records = load_records(args.task, args.since, args.tag)
    if not records:
        raise SystemExit("No matching records.")

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        tag = record.get("tag") or "(no tag)"
        label = f"{tag}@{str(record.get('timestamp', ''))[11:19]}" if args.runs else tag
        groups[label].append(record)

    print_table(["field", *groups], build_rows(groups, args.fields))

    for record in records:
        if is_failure(record):
            print(f"FAILED {record.get('timestamp')} {record.get('tag')}: {str(record.get('error'))[:120]}")


if __name__ == "__main__":
    main()
