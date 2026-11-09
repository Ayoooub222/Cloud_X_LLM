"""Poll the vLLM Prometheus endpoint (/metrics) during a run: scheduler state and KV cache use.

Explains saturation: if requests wait while few run and the KV cache is far from full,
the limit is the per-step token budget (--max-num-batched-tokens), not memory.
"""
from __future__ import annotations

import re
import threading

import httpx

METRICS_PATH = "/metrics"
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
KV_CACHE = "vllm:kv_cache_usage_perc"  # 0..1; replaced gpu_cache_usage_perc
WAITING_BY_REASON = "vllm:num_requests_waiting_by_reason"  # reason="capacity" | "deferred"
TRACKED = (RUNNING, WAITING, KV_CACHE, WAITING_BY_REASON)
REASON_LABEL = re.compile(r'reason="([a-z_]+)"')
GAUGES = {RUNNING: "running", WAITING: "waiting", KV_CACHE: "kv_cache",
          f"{WAITING_BY_REASON}:capacity": "waiting_capacity", f"{WAITING_BY_REASON}:deferred": "waiting_deferred"}
REQUEST_TIMEOUT_S = 2.0


def parse_metrics(text: str, names: tuple[str, ...] = TRACKED) -> dict[str, float]:
    """Sum of each tracked gauge over its label sets (one engine -> one value).

    A gauge with a reason label is keyed "<name>:<reason>".
    """
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in names:
            continue
        reason = REASON_LABEL.search(line)
        key = f"{name}:{reason.group(1)}" if reason else name
        try:
            values[key] = values.get(key, 0.0) + float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return values


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, float | int]:
    """Max and mean of each gauge, as record metrics (sched.*)."""
    out: dict[str, float | int] = {"sched.samples": len(samples)}
    for key, label in GAUGES.items():
        series = [s[key] for s in samples if key in s]
        if series:
            out[f"sched.{label}_max"] = round(max(series), 4)
            out[f"sched.{label}_avg"] = round(sum(series) / len(series), 4)
    return out


class MetricsSampler:
    """Background poller of /metrics. Never raises: a failed poll is skipped.

    Usage:
        with MetricsSampler("http://localhost:8001", 0.5) as sampler:
            run_the_workload()
        metrics = sampler.summary()
    """

    def __init__(self, base_url: str, interval_s: float = 0.5):
        self.url = base_url.rstrip("/") + METRICS_PATH
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def read_once(self, client: httpx.Client) -> dict[str, float] | None:
        try:
            response = client.get(self.url)
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return parse_metrics(response.text) or None

    def _loop(self) -> None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            while not self._stop.is_set():
                sample = self.read_once(client)
                if sample is not None:
                    self.samples.append(sample)
                self._stop.wait(self.interval_s)

    def __enter__(self) -> MetricsSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, float | int]:
        return summarize_samples(self.samples) if self.samples else {}
