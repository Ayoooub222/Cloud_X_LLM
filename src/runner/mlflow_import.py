"""Send existing benchmark JSON records to MLflow. Runs already logged are skipped,
so the script can be run again safely after an error.

Examples:
    python -m src.runner.mlflow_import --dry-run
    python -m src.runner.mlflow_import --task image --task coldstart
    python -m src.runner.mlflow_import --since "2026-10-24 00:00:00"
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

from pydantic import ValidationError

from src.bench.mlflow_log import is_enabled, log_record
from src.bench.recorder import RESULTS_DIR, BenchRecord

HEALTH_TIMEOUT_S = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send benchmark JSON records to MLflow.")
    parser.add_argument("--task", action="append",
                        help="Only this task folder (repeatable): audio, vision, image, coldstart.")
    parser.add_argument("--since", help='Only records with timestamp >= "YYYY-MM-DD HH:MM:SS".')
    parser.add_argument("--dry-run", action="store_true",
                        help="List the records that would be sent, without contacting MLflow.")
    return parser.parse_args(argv)


def server_is_up() -> bool:
    """True if the MLflow server answers on /health."""
    url = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/health"
    try:
        return urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_S).status == 200
    except OSError:  # connection refused, reset, timeout...
        return False


def find_paths(root: Path, tasks: list[str] | None) -> list[Path]:
    """JSON files of the given task folders (all folders if tasks is empty)."""
    if tasks:
        return sorted(path for task in tasks for path in (root / task).glob("*.json"))
    return sorted(root.glob("*/*.json"))


def load_record(path: Path) -> BenchRecord | None:
    """Parse one file, or return None if it is not a benchmark record."""
    try:
        return BenchRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        print(f"Skipping {path.name}: not a benchmark record ({exc.error_count()} errors)", file=sys.stderr)
        return None


def select_records(root: Path, tasks: list[str] | None, since: str | None) -> list[tuple[Path, BenchRecord]]:
    selected = []
    for path in find_paths(root, tasks):
        record = load_record(path)
        if record is None:
            continue
        if since and record.timestamp < since:
            continue
        selected.append((path, record))
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = select_records(RESULTS_DIR, args.task, args.since)

    if args.dry_run:
        for path, record in selected:
            print(f"{record.task:<10} {record.timestamp} {record.tag or '-':<20} {path.name}")
        print(f"{len(selected)} records")
        return 0

    if not is_enabled():
        print("Set MLFLOW_TRACKING_URI first (e.g. http://localhost:5000)", file=sys.stderr)
        return 1
    if not server_is_up():
        print("MLflow server is not ready (try: bash scripts/wait_ready.sh mlflow)", file=sys.stderr)
        return 1

    for index, (path, record) in enumerate(selected, start=1):
        try:
            log_record(record, path)
        except Exception as exc:  # MLflow client errors have no common base class
            print(f"Stopped at {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"[{index}/{len(selected)}] {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
