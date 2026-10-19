"""Enregistrement structuré des runs de benchmark : 1 run = 1 fichier JSON dans results/benchmark/<task>/."""
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

RESULTS_DIR = Path(os.getenv("BENCH_DIR", "results/benchmark"))


# ---------- GPU ----------
class GpuStats(BaseModel):
    name: str | None = None
    vram_total_mib: float | None = None
    vram_peak_mib: float | None = None
    util_avg_pct: float | None = None
    power_avg_w: float | None = None
    energy_wh: float | None = None      # power_avg × durée → base du coût électrique self-hosted
    duration_s: float | None = None
    samples: int = 0


def _num(x: str) -> float | None:
    try:
        return float(x.strip())
    except ValueError:                  # "[N/A]" sur certains GPU laptop
        return None


class GpuSampler:
    """Equivalent of `nvidia-smi -l 1`. Usage : with GpuSampler() as g: ... ; g.stats()"""

    QUERY = "name,memory.total,memory.used,utilization.gpu,power.draw"

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self._rows: list[tuple] = []
        self._name: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = self._t1 = 0.0

    def _query(self) -> tuple | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={self.QUERY}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            return None
        name, total, used, util, power = out.split(",")
        self._name = name.strip()
        return _num(total), _num(used), _num(util), _num(power)

    def _run(self) -> None:
        while not self._stop.is_set():
            if (row := self._query()) is not None:
                self._rows.append(row)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "GpuSampler":
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._t1 = time.perf_counter()

    def stats(self) -> GpuStats:
        if not self._rows:
            return GpuStats()

        def col(i: int) -> list[float]:
            return [r[i] for r in self._rows if r[i] is not None]

        used, util, power = col(1), col(2), col(3)
        duration = self._t1 - self._t0
        p_avg = sum(power) / len(power) if power else None
        return GpuStats(
            name=self._name,
            vram_total_mib=col(0)[0] if col(0) else None,
            vram_peak_mib=max(used) if used else None,
            util_avg_pct=round(sum(util) / len(util), 1) if util else None,
            power_avg_w=round(p_avg, 1) if p_avg is not None else None,
            energy_wh=round(p_avg * duration / 3600, 4) if p_avg is not None else None,
            duration_s=round(duration, 2),
            samples=len(self._rows),
        )


# ---------- Record ----------
def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def env_info() -> dict[str, Any]:
    return {
        "label": os.getenv("BENCH_ENV", "local-wsl2-rtx3070laptop"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_commit": _git_commit(),
    }


class BenchRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:6])
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    timezone: str = Field(default_factory=lambda: datetime.now().astimezone().tzname() or "")
    task: str                       # "image" | "audio" | "vision"
    backend: str                    # "sdcpp" | "vllm" | ...
    model: str
    tag: str | None = None          # ex: "warmup", "testA-notiling"
    status: str = "ok"              # "ok" | "error"
    error: str | None = None
    env: dict[str, Any] = Field(default_factory=env_info)
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    gpu: GpuStats = Field(default_factory=GpuStats)
    outputs: list[str] = Field(default_factory=list)

    def save(self, root: Path = RESULTS_DIR) -> Path:
        d = root / self.task
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d_%H-%M-%S")
        path = d / f"{ts}_{self.task}_{self.run_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path
