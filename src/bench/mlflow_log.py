"""Send benchmark records to an MLflow tracking server (optional).

Enabled only if MLFLOW_TRACKING_URI is set. The JSON file stays the source of truth:
if MLflow fails, a warning is printed and the benchmark goes on.
"""
from __future__ import annotations

import os
import contextlib
import io

from datetime import datetime
from pathlib import Path


def is_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def numeric_metrics(record) -> dict[str, float]:
    """metrics + gpu stats (prefixed 'gpu.'), numbers only, like summarize.py."""
    metrics = {k: float(v) for k, v in record.metrics.items() if isinstance(v, (int, float))}
    for k, v in record.gpu.model_dump().items():
        if isinstance(v, (int, float)):
            metrics[f"gpu.{k}"] = float(v)
    return metrics


def log_record(record, json_path: Path) -> None:
    """Create one MLflow run from one BenchRecord. Records already logged are skipped."""
    from mlflow import MlflowClient  # imported here so mlflow stays optional
    from mlflow.entities import Metric, Param

    client = MlflowClient()  # reads MLFLOW_TRACKING_URI
    experiment = client.get_experiment_by_name(record.task)
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(record.task)

    already = client.search_runs([experiment_id], f"tags.bench_run_id = '{record.run_id}'", max_results=1)
    if already:
        return

    start_ms = int(datetime.fromisoformat(record.timestamp).timestamp() * 1000)
    tags = {
        "bench_run_id": record.run_id,
        "tag": record.tag,
        "status": record.status,
        "error": record.error,
        "env": record.env.get("label"),
        "host": record.env.get("host"),
        "git_commit": record.env.get("git_commit"),
    }
    run = client.create_run(
        experiment_id,
        start_time=start_ms,
        run_name=f"{record.tag or 'untagged'}-{record.run_id}",
        tags={k: str(v) for k, v in tags.items() if v is not None},
    )
    run_id = run.info.run_id

    params = {**record.params, "backend": record.backend, "model": record.model}
    client.log_batch(
        run_id,
        metrics=[Metric(k, v, start_ms, 0) for k, v in numeric_metrics(record).items()],
        params=[Param(k, str(v)) for k, v in params.items() if v is not None],
    )

    client.log_artifact(run_id, str(json_path))
    for output in record.outputs:
        if Path(output).exists():
            client.log_artifact(run_id, output, artifact_path="outputs")

    duration_ms = int((record.gpu.duration_s or 0) * 1000)
    status = "FINISHED" if record.status == "ok" else "FAILED"
    # MLflow prints "View run/experiment" links with emojis: hide them
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        client.set_terminated(run_id, status=status, end_time=start_ms + duration_ms)



def try_log_record(record, json_path: Path) -> None:
    """Log to MLflow if enabled. Never raises."""
    if not is_enabled():
        return
    try:
        log_record(record, json_path)
    except Exception as e:
        print(f"[mlflow] warning: run not logged ({type(e).__name__}: {e})")
