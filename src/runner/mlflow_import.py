"""Send every existing benchmark JSON to MLflow. Runs already logged are skipped,
so the script can be run again safely after an error.

Example:
    python -m src.runner.mlflow_import
"""
import os
import sys
import urllib.request

from src.bench.mlflow_log import is_enabled, log_record
from src.bench.recorder import RESULTS_DIR, BenchRecord


def server_is_up() -> bool:
    """True if the MLflow server answers on /health."""
    url = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/health"
    try:
        return urllib.request.urlopen(url, timeout=5).status == 200
    except OSError:  # connection refused, reset, timeout...
        return False


def main() -> int:
    if not is_enabled():
        print("Set MLFLOW_TRACKING_URI first (e.g. http://localhost:5000)", file=sys.stderr)
        return 1
    if not server_is_up():
        print("MLflow server is not ready (try: bash scripts/wait_ready.sh mlflow)", file=sys.stderr)
        return 1

    paths = sorted(RESULTS_DIR.glob("*/*.json"))
    for i, path in enumerate(paths, start=1):
        record = BenchRecord.model_validate_json(path.read_text(encoding="utf-8"))
        try:
            log_record(record, path)
        except Exception as e:
            print(f"Stopped at {path.name}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        print(f"[{i}/{len(paths)}] {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
