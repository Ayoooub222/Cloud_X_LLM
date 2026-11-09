"""Unit tests for the MLflow helpers that do not need an MLflow server."""
from src.bench import mlflow_log
from src.bench.recorder import BenchRecord, GpuStats


def make_record(gpu: GpuStats) -> BenchRecord:
    return BenchRecord(task="coldstart", backend="vllm", model="m", env={"label": "ci-test"},
                       metrics={"t_first_s": 80.0, "note": None}, gpu=gpu)


def test_numeric_metrics_without_gpu_samples():
    assert mlflow_log.numeric_metrics(make_record(GpuStats())) == {"t_first_s": 80.0}


def test_numeric_metrics_with_gpu_samples():
    metrics = mlflow_log.numeric_metrics(make_record(GpuStats(name="gpu", power_avg_w=50.0, samples=2)))
    assert metrics == {"t_first_s": 80.0, "gpu.power_avg_w": 50.0, "gpu.samples": 2.0}


def failing_log(record, json_path):
    raise RuntimeError("server down")


def test_try_log_record_disabled(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_log, "log_record", failing_log)
    mlflow_log.try_log_record(make_record(GpuStats()), None)


def test_try_log_record_never_raises(monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setattr(mlflow_log, "log_record", failing_log)
    mlflow_log.try_log_record(make_record(GpuStats()), None)
    assert "[mlflow] warning" in capsys.readouterr().out
