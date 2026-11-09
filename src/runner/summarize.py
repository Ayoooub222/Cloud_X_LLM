"""Aggregate benchmark records (results/benchmark/<task>/*.json) into a table.

Rows are the numeric fields of each record's "metrics" and "gpu" sections
(gpu fields are prefixed with "gpu."). Columns are tags, or individual runs
with --runs. With --by-env, the environment label (BENCH_ENV) is appended to
each column, to compare compose and k3s side by side. Failed runs (non-empty
"error") are counted but excluded from the statistics. Cells show
"mean ±stdev" when a column has several runs.

Examples:
    python -m src.runner.summarize image
    python -m src.runner.summarize image --tag testA-notiling --tag testA-tiling
    python -m src.runner.summarize image --since "2026-10-19 14:00:00" --runs
    python -m src.runner.summarize image --fields time vram power energy
    python -m src.runner.summarize audio --by-env --tag single-11s
    python -m src.runner.summarize coldstart --env local-k3s-rtx3070laptop
    python -m src.runner.summarize audio --format json --out outputs/summary/audio.json
    python -m src.runner.summarize all --format json

JSON output (--format json, read by the cost model): one group per (task, tag, BENCH_ENV) with,
for every numeric field, n / mean / stdev / min / p50 / p95 / max over the successful runs, and
a "headline" block: requests_per_hour, latency_p50_s, latency_p95_s, energy_wh_per_request.
For load records (k6) the latency percentiles are per request; for the other tasks they are taken
over the per-run latency_s (a batch mean), so only load records give a true request p95.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

BENCH_DIR = Path(os.getenv("BENCH_DIR", "results/benchmark"))
NO_ENV = "(no env)"

LARGE_VALUE = 100.0   # >= 100 -> 1 decimal  (e.g. vram 8192.0)
MEDIUM_VALUE = 1.0    # >= 1   -> 2 decimals (e.g. latency 1.09)
SMALL_DECIMALS = 4    # < 1    -> 4 decimals (e.g. rtf 0.0106)
ALL_TASKS = "all"
JSON_DECIMALS = 6
SECONDS_PER_HOUR = 3600.0


def env_label(record: dict) -> str:
    """BENCH_ENV label, stored either as record["env"]["label"] or record["label"]."""
    env = record.get("env")
    if isinstance(env, dict) and env.get("label"):
        return str(env["label"])
    return str(record.get("label") or NO_ENV)


def task_dirs(task: str) -> list[Path]:
    """Folder of one task, or every task folder for "all"."""
    if task == ALL_TASKS:
        return sorted(path for path in BENCH_DIR.iterdir() if path.is_dir())
    task_dir = BENCH_DIR / task
    if not task_dir.is_dir():
        raise SystemExit(f"Benchmark directory not found: {task_dir}")
    return [task_dir]


def load_records(task: str, since: str | None, tags: list[str] | None,
                 envs: list[str] | None) -> list[dict]:
    records = []
    paths = [path for task_dir in task_dirs(task) for path in sorted(task_dir.glob("*.json"))]
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Skipping unreadable file {path.name}: {exc}")
            continue
        if since and str(record.get("timestamp", "")) < since:
            continue
        if tags and record.get("tag") not in tags:
            continue
        if envs and env_label(record) not in envs:
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


def gpu_fields(record: dict) -> dict[str, float]:
    """gpu.* fields, or nothing when no nvidia-smi sample was taken (e.g. cold start records)."""
    gpu = record.get("gpu") or {}
    if not gpu.get("samples"):
        return {}
    return flatten_numeric(gpu, "gpu.")


def numeric_fields(record: dict) -> dict[str, float]:
    return flatten_numeric(record.get("metrics") or {}) | gpu_fields(record)


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


def column_label(record: dict, per_run: bool, by_env: bool) -> str:
    label = record.get("tag") or "(no tag)"
    if per_run:
        label = f"{label}@{str(record.get('timestamp', ''))[11:19]}"
    if by_env:
        label = f"{label} [{env_label(record)}]"
    return label


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
        cells = [row[0].ljust(widths[0])] + [c.rjust(w) for c, w in zip(row[1:], widths[1:], strict=True)]

        print("  ".join(cells))
        if index == 0:
            print("  ".join("-" * w for w in widths))


# ---------- JSON output (for the cost model) ----------
def field_stats(values: list[float]) -> dict[str, float | int]:
    """n, mean, stdev, min, p50, p95, max over runs (p95 is interpolated: few runs -> close to max)."""
    ordered = sorted(values)
    stats = {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }
    return {k: v if isinstance(v, int) else round(v, JSON_DECIMALS) for k, v in stats.items()}


def percentile(ordered: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (same as numpy's default)."""
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def requests_per_hour(fields: dict[str, float]) -> float | None:
    """Throughput of one run: k6 achieved rate, else requests / wall time, else 1 / latency."""
    if "achieved_rps" in fields:
        return fields["achieved_rps"] * SECONDS_PER_HOUR
    if fields.get("wall_s"):
        return fields.get("requests", 1.0) * SECONDS_PER_HOUR / fields["wall_s"]
    if fields.get("latency_s"):
        return SECONDS_PER_HOUR / fields["latency_s"]
    return None


def energy_per_request(fields: dict[str, float]) -> float | None:
    if "wh_per_request" in fields:
        return fields["wh_per_request"]
    if "gpu.energy_wh" in fields:
        return fields["gpu.energy_wh"] / fields.get("requests", 1.0)
    return None


def with_derived(fields: dict[str, float]) -> dict[str, float]:
    derived = {"derived.requests_per_hour": requests_per_hour(fields),
               "derived.energy_wh_per_request": energy_per_request(fields)}
    return fields | {k: v for k, v in derived.items() if v is not None}


def headline(fields: dict[str, dict]) -> dict[str, float | None]:
    """Cost model inputs. Latency percentiles come from k6 when present, else from the runs."""
    def stat(name: str, key: str) -> float | None:
        return fields[name][key] if name in fields else None

    load_test = "latency_p95_s" in fields
    return {
        "requests_per_hour": stat("derived.requests_per_hour", "mean"),
        "latency_p50_s": stat("latency_p50_s", "mean") if load_test else stat("latency_s", "p50"),
        "latency_p95_s": stat("latency_p95_s", "mean") if load_test else stat("latency_s", "p95"),
        "energy_wh_per_request": stat("derived.energy_wh_per_request", "mean"),
    }


def json_groups(records: list[dict], fields_filter: list[str] | None) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        key = (str(record.get("task")), record.get("tag") or "(no tag)", env_label(record))
        groups[key].append(record)
    out = []
    for (task, tag, env), group in groups.items():
        values: dict[str, list[float]] = defaultdict(list)
        ok = [r for r in group if not is_failure(r)]
        for record in ok:
            for name, value in with_derived(numeric_fields(record)).items():
                values[name].append(value)
        stats = {name: field_stats(v) for name, v in values.items()}
        shown = {n: v for n, v in stats.items() if not fields_filter or any(f in n for f in fields_filter)}
        out.append({"task": task, "tag": tag, "env": env, "runs_ok": len(ok), "runs_total": len(group),
                    "first": min(str(r.get("timestamp")) for r in group),
                    "last": max(str(r.get("timestamp")) for r in group),
                    "headline": headline(stats), "fields": shown})
    return out


def json_report(records: list[dict], args: argparse.Namespace) -> dict:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bench_dir": str(BENCH_DIR),
        "filters": {"task": args.task, "tag": args.tag, "env": args.env, "since": args.since},
        "groups": json_groups(records, args.fields),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON records.")
    parser.add_argument("task", help="Task folder under BENCH_DIR (image, vision, audio, coldstart, load) or all.")
    parser.add_argument("--tag", action="append", help="Keep only this tag (repeatable).")
    parser.add_argument("--env", action="append", help="Keep only this BENCH_ENV label (repeatable).")
    parser.add_argument("--since", help='Keep records with timestamp >= "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument("--fields", nargs="+", help="Keep only fields containing one of these substrings.")
    parser.add_argument("--runs", action="store_true", help="One column per run instead of one per tag.")
    parser.add_argument("--by-env", action="store_true", help="Append the BENCH_ENV label to each column.")
    parser.add_argument("--format", choices=["table", "json"], default="table",
                        help="json: statistics per (task, tag, env) for the cost model.")
    parser.add_argument("--out", type=Path, help="With --format json: write to this file instead of stdout.")
    args = parser.parse_args(argv)

    records = load_records(args.task, args.since, args.tag, args.env)
    if not records:
        raise SystemExit("No matching records.")

    if args.format == "json":
        text = json.dumps(json_report(records, args), indent=2, ensure_ascii=False)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text + "\n", encoding="utf-8")
            print(f"{len(records)} records -> {args.out}")
        else:
            print(text)
        return

    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[column_label(record, args.runs, args.by_env)].append(record)

    print_table(["field", *groups], build_rows(groups, args.fields))

    for record in records:
        if is_failure(record):
            print(f"FAILED {record.get('timestamp')} {record.get('tag')}: {str(record.get('error'))[:120]}")


if __name__ == "__main__":
    main()
