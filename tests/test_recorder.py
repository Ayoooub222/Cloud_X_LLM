"""Unit tests for the benchmark recorder (no GPU, no MLflow)."""
import json

from src.bench import recorder
from src.runner import summarize

NVIDIA_LINE = "NVIDIA GeForce RTX 3070 Laptop GPU, 8192, 1024, 45, [N/A]\n"


class FakeResult:
    def __init__(self, stdout):
        self.stdout = stdout


def test_to_float():
    assert recorder.to_float(" 12.5 ") == 12.5
    assert recorder.to_float("[N/A]") is None


def test_read_once_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(recorder.subprocess, "run", lambda cmd, **kwargs: FakeResult(NVIDIA_LINE))
    sampler = recorder.GpuSampler()
    sample = sampler.read_once()
    assert sample == {"total": 8192.0, "used": 1024.0, "util": 45.0, "power": None}
    assert sampler.gpu_name == "NVIDIA GeForce RTX 3070 Laptop GPU"


def test_read_once_returns_none_on_failure(monkeypatch):
    def failing_run(cmd, **kwargs):
        raise OSError("nvidia-smi not found")

    monkeypatch.setattr(recorder.subprocess, "run", failing_run)
    assert recorder.GpuSampler().read_once() is None


def test_read_once_returns_none_on_empty_output(monkeypatch):
    monkeypatch.setattr(recorder.subprocess, "run", lambda cmd, **kwargs: FakeResult(""))
    assert recorder.GpuSampler().read_once() is None


def test_stats_without_samples():
    stats = recorder.GpuSampler().stats()
    assert stats.samples == 0
    assert stats.energy_wh is None


def test_stats_aggregation():
    sampler = recorder.GpuSampler()
    sampler.gpu_name = "test-gpu"
    sampler.samples = [
        {"total": 8192.0, "used": 1000.0, "util": 30.0, "power": 40.0},
        {"total": 8192.0, "used": 2000.0, "util": 50.0, "power": 60.0},
        {"total": 8192.0, "used": 1500.0, "util": None, "power": None},
    ]
    sampler._start, sampler._end = 0.0, 3600.0
    stats = sampler.stats()
    assert stats.vram_total_mib == 8192.0
    assert stats.vram_peak_mib == 2000.0
    assert stats.util_avg_pct == 40.0
    assert stats.power_avg_w == 50.0
    assert stats.energy_wh == 50.0
    assert stats.duration_s == 3600.0
    assert stats.samples == 3


def test_sampler_thread_collects_samples(monkeypatch):
    sample = {"total": 8192.0, "used": 1.0, "util": 1.0, "power": 1.0}
    monkeypatch.setattr(recorder.GpuSampler, "read_once", lambda self: sample)
    sampler = recorder.GpuSampler(interval_s=0.01)
    with sampler:
        recorder.time.sleep(0.05)
    assert len(sampler.samples) >= 1
    assert not sampler._thread.is_alive()


def test_save_writes_file_and_calls_mlflow_hook(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(recorder, "try_log_record", lambda record, path: calls.append((record, path)))
    monkeypatch.setattr(recorder, "git_commit", lambda: "deadbee")
    monkeypatch.setenv("BENCH_ENV", "ci-test")

    record = recorder.BenchRecord(task="audio", backend="vllm", model="whisper",
                                  run_id="abc123", timestamp="2026-10-24 12:34:56")
    path = record.save(tmp_path)

    assert path == tmp_path / "audio" / "2026-10-24_12-34-56_audio_abc123.json"
    assert calls == [(record, path)]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["env"]["git_commit"] == "deadbee"
    assert summarize.env_label(data) == "ci-test"


def test_default_env_label(monkeypatch):
    monkeypatch.delenv("BENCH_ENV", raising=False)
    monkeypatch.setattr(recorder, "git_commit", lambda: None)
    record = recorder.BenchRecord(task="image", backend="sdcpp", model="sd")
    assert record.env["label"] == recorder.DEFAULT_ENV_LABEL
    assert record.status == "ok"
