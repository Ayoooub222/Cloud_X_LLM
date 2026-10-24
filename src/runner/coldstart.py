"""Record a k3s cold start: time to Ready (switch_k8s.sh) plus latency of the first real request.

The first request is a normal runner call tagged "coldstart-first"; this script finds it,
adds its latency to the time to Ready and saves one record under results/benchmark/coldstart/.

Example:
    python -m src.runner.coldstart audio --t-ready 61 --since "2026-10-22 18:30:00"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from src.bench.recorder import BenchRecord

BENCH_DIR = Path(os.getenv("BENCH_DIR", "results/benchmark"))
FIRST_TAG = "coldstart-first"
BACKENDS = {"audio": "vllm", "vision": "vllm", "image": "sdcpp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Save a cold start record (time to Ready + first request)")
    p.add_argument("task", choices=sorted(BACKENDS))
    p.add_argument("--t-ready", type=float, required=True, help="seconds from switch to Ready")
    p.add_argument("--since", required=True, help='start of the switch, "YYYY-MM-DD HH:MM:SS"')
    return p.parse_args()


def first_request(task: str, since: str) -> dict | None:
    """Oldest record of `task` tagged FIRST_TAG written at or after `since`."""
    for path in sorted((BENCH_DIR / task).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("tag") == FIRST_TAG and str(record.get("timestamp", "")) >= since:
            return record
    return None


def main() -> int:
    args = parse_args()
    first = first_request(args.task, args.since)
    latency = (first or {}).get("metrics", {}).get("latency_s")

    record = BenchRecord(
        task="coldstart",
        backend=BACKENDS[args.task],
        model=(first or {}).get("model") or args.task,
        tag=f"coldstart-{args.task}",
        params={"target": args.task, "from_state": "none", "first_run_id": (first or {}).get("run_id")},
    )
    record.metrics = {"t_ready_s": args.t_ready}
    if first is None or first.get("error") or latency is None:
        record.status = "error"
        record.error = "first request not found or failed"
    else:
        record.metrics["first_latency_s"] = latency
        record.metrics["t_first_s"] = round(args.t_ready + latency, 2)

    path = record.save()
    m = record.metrics
    print(f"[{record.status}] coldstart {args.task}: ready={m.get('t_ready_s')}s "
          f"first={m.get('first_latency_s')}s total={m.get('t_first_s')}s -> {path}")
    return 0 if record.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
