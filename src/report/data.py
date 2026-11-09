"""Tidy tables (one row per run) for the report figures, read from results/benchmark and the cost model."""
from __future__ import annotations

import pandas as pd

from src.cost.config import load_prices, load_scenarios
from src.cost.report import build_report
from src.runner.summarize import env_label, is_failure, load_records, numeric_fields, requests_per_hour

REFERENCE_ENV = "local-wsl2-rtx3070laptop"
K3S_ENV = "local-k3s-rtx3070laptop"
T4_ENV = "kaggle-t4"
GPU_OF_ENV = {REFERENCE_ENV: "rtx3070", K3S_ENV: "rtx3070", T4_ENV: "t4"}
GPU_LABELS = {"rtx3070": "RTX 3070 Laptop, bf16", "t4": "Tesla T4, fp16"}
DEPLOY_LABELS = {REFERENCE_ENV: "Docker Compose", K3S_ENV: "k3s + Helm"}

# Concurrency series measured on both GPUs with the same inputs and server settings
SERIES = {
    "audio": {"tag_prefix": "ls66-c", "levels": (1, 2, 4, 8), "quality": "wer"},
    "vision": {"tag_prefix": "conc-c", "levels": (1, 2, 4), "quality": "field_accuracy"},
}
LOAD_PREFIXES = {"audio": "k6-audio-r", "vision": "k6-vision-r"}
COLDSTART_PREFIX = "coldstart-"
MWH_PER_WH = 1000.0


def concurrency_frame(task: str) -> pd.DataFrame:
    """Successful runs of the concurrency series of one task, on the 3070 (compose) and the T4."""
    spec = SERIES[task]
    tags = [f"{spec['tag_prefix']}{c}" for c in spec["levels"]]
    rows = []
    for record in load_records(task, None, tags, [REFERENCE_ENV, T4_ENV]):
        if is_failure(record):
            continue
        fields = numeric_fields(record)
        requests = fields.get("requests", 1.0)
        rows.append({
            "env": env_label(record),
            "gpu": GPU_OF_ENV[env_label(record)],
            "concurrency": int(record["tag"].removeprefix(spec["tag_prefix"])),
            "requests_per_hour": fields.get("pages_per_hour") or requests_per_hour(fields),  # as recorded
            "energy_wh_per_request": fields["gpu.energy_wh"] / requests,
            "latency_s": fields["latency_s"],
            "quality": fields.get(spec["quality"]),
            "gpu_util_pct": fields.get("gpu.util_avg_pct"),
            "gpu_power_w": fields.get("gpu.power_avg_w"),
            "batch_rtf": fields.get("batch_rtf"),   # audio only
        })
    return pd.DataFrame(rows).sort_values(["gpu", "concurrency"], ignore_index=True)


def ratio_frame(task: str) -> pd.DataFrame:
    """T4 throughput relative to the 3070 at each concurrency, defined as in conf/cost_scenarios.yaml:
    ratio of mean time per request for audio (batch_rtf), ratio of mean pages/h for vision."""
    frame = concurrency_frame(task)
    if task == "audio":
        means = frame.groupby(["gpu", "concurrency"])["batch_rtf"].mean().unstack("gpu")
        ratio = means["rtx3070"] / means["t4"]
    else:
        means = frame.groupby(["gpu", "concurrency"])["requests_per_hour"].mean().unstack("gpu")
        ratio = means["t4"] / means["rtx3070"]
    return pd.DataFrame({"concurrency": means.index.to_numpy(), "ratio": ratio.to_numpy()})


def load_frame(task: str) -> pd.DataFrame:
    """k6 steps of one task on the 3070: one row per run (arrival rate, p95 latency, SLO)."""
    prefix = LOAD_PREFIXES[task]
    rows = []
    for record in load_records("load", None, None, [REFERENCE_ENV]):
        if is_failure(record) or not str(record.get("tag", "")).startswith(prefix):
            continue
        fields = numeric_fields(record)
        rows.append({
            "target_rps": fields["target_rps"],
            "achieved_rps": fields["achieved_rps"],
            "latency_p95_s": fields["latency_p95_s"],
            "slo_p95_s": fields["slo_p95_s"],
            "slo_ok": bool(fields["slo_ok"]),
            "wh_per_request": fields.get("wh_per_request"),
        })
    return pd.DataFrame(rows).sort_values("target_rps", ignore_index=True)


def coldstart_frame() -> pd.DataFrame:
    """Cold starts on the 3070 (compose and k3s): server ready, then first answer."""
    rows = []
    for record in load_records("coldstart", None, None, [REFERENCE_ENV, K3S_ENV]):
        if is_failure(record) or not str(record.get("tag", "")).startswith(COLDSTART_PREFIX):
            continue
        fields = numeric_fields(record)
        rows.append({
            "model": record["tag"].removeprefix(COLDSTART_PREFIX),
            "deploy": DEPLOY_LABELS[env_label(record)],
            "t_ready_s": fields["t_ready_s"],
            "t_first_s": fields["t_first_s"],
        })
    return pd.DataFrame(rows)


def cost_report() -> dict:
    """The cost model output (same numbers as python -m src.cost.report --format json)."""
    return build_report(load_prices(), load_scenarios())
