"""Structured benchmark records: one run = one JSON file in results/benchmark/<task>/."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.bench.mlflow_log import try_log_record

RESULTS_DIR = Path(os.getenv("BENCH_DIR", "results/benchmark"))
DEFAULT_ENV_LABEL = "local-wsl2-rtx3070laptop"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
NVIDIA_QUERY = "name,memory.total,memory.used,utilization.gpu,power.draw"


# ---------- GPU ----------
class GpuStats(BaseModel):
    """GPU summary of one run (values come from nvidia-smi)."""
    name: str | None = None
    vram_total_mib: float | None = None
    vram_peak_mib: float | None = None
    util_avg_pct: float | None = None
    power_avg_w: float | None = None
    energy_wh: float | None = None  # power_avg_w * duration -> electricity cost of self-hosting
    duration_s: float | None = None
    samples: int = 0


def to_float(text: str) -> float | None:
    """Convert a nvidia-smi value to float ('[N/A]' on some laptop GPUs -> None)."""
    try:
        return float(text.strip())
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


class GpuSampler:
    """Poll nvidia-smi every `interval_s` seconds in a background thread.

    Usage:
        sampler = GpuSampler()
        with sampler:
            run_the_workload()
        stats = sampler.stats()
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.gpu_name: str | None = None
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._start = 0.0
        self._end = 0.0

    def read_once(self) -> dict | None:
        """Run nvidia-smi once. Return one sample, or None if the call failed."""
        cmd = ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader,nounits"]
        try:
            output = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout
            name, total, used, util, power = output.splitlines()[0].split(",")
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            return None
        self.gpu_name = name.strip()
        return {"total": to_float(total), "used": to_float(used),
                "util": to_float(util), "power": to_float(power)}

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self.read_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> GpuSampler:
        self._start = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._end = time.perf_counter()

    def values(self, key: str) -> list[float]:
        """All non-empty values of one column (total, used, util or power)."""
        return [s[key] for s in self.samples if s[key] is not None]

    def stats(self) -> GpuStats:
        if not self.samples:
            return GpuStats()
        duration = self._end - self._start
        total, used = self.values("total"), self.values("used")
        util, power = self.values("util"), self.values("power")
        return GpuStats(
            name=self.gpu_name,
            vram_total_mib=total[0] if total else None,
            vram_peak_mib=max(used) if used else None,
            util_avg_pct=round(mean(util), 1) if util else None,
            power_avg_w=round(mean(power), 1) if power else None,
            energy_wh=round(mean(power) * duration / 3600, 4) if power else None,
            duration_s=round(duration, 2),
            samples=len(self.samples),
        )


# ---------- Record ----------
def git_commit() -> str | None:
    """Short hash of the current commit (to know which code produced a result)."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def env_info() -> dict[str, Any]:
    return {
        "label": os.getenv("BENCH_ENV", DEFAULT_ENV_LABEL),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": git_commit(),
    }


class BenchRecord(BaseModel):
    """One benchmark run. Failed runs are saved too (status='error')."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:6])
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime(TIMESTAMP_FORMAT))
    timezone: str = Field(default_factory=lambda: datetime.now().astimezone().tzname() or "")
    task: str                    # "image" | "audio" | "vision"
    backend: str                 # "sdcpp" | "vllm" | ...
    model: str
    tag: str | None = None       # e.g. "warmup", "testA-notiling"
    status: str = "ok"           # "ok" | "error"
    error: str | None = None
    env: dict[str, Any] = Field(default_factory=env_info)
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    gpu: GpuStats = Field(default_factory=GpuStats)
    outputs: list[str] = Field(default_factory=list)

    def save(self, root: Path = RESULTS_DIR) -> Path:
        """Write <root>/<task>/<YYYY-MM-DD_HH-MM-SS>_<task>_<run_id>.json and return its path."""
        folder = root / self.task
        folder.mkdir(parents=True, exist_ok=True)
        file_time = datetime.strptime(self.timestamp, TIMESTAMP_FORMAT).strftime("%Y-%m-%d_%H-%M-%S")
        path = folder / f"{file_time}_{self.task}_{self.run_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        try_log_record(self, path)  # optional MLflow copy, never breaks the benchmark

        return path
