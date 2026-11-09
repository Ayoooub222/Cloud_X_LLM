"""Projection of the vision cost to other model sizes, anchored on the measured 2B model.

This is a model, not a measurement. Its rules are deliberately simple:
- decoding is memory-bound, so throughput scales as 1 / parameters and latency as parameters;
- a model fits on a GPU when its 16-bit weights take at most WEIGHT_SHARE of the VRAM (the rest
  holds the KV cache, activations and CUDA graphs);
- the latency at the sustained rate (k6 p95, measured on the 3070 at 2B) scales with the model size
  and inversely with the GPU throughput ratio; an option above the SLO is not feasible interactively;
- one instance runs 24/7; a volume beyond its capacity pays the extra instance-hours (autoscaling).
The API price does not depend on the self-hosted model size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HOURS_PER_MONTH = 730.0
REQUESTS = 1000.0
MEASURED_SIZE_B = 2.0          # Qwen3-VL-2B-Instruct
BYTES_PER_PARAM = 2.0          # bf16 / fp16 weights
WEIGHT_SHARE = 0.6             # share of the VRAM the weights may take
GPU_VRAM_GB = {"g4dn": 16.0, "g5": 24.0}
GPU_NAMES = {"g4dn": "T4", "g5": "A10G"}

INFEASIBLE_VRAM = "too large for one GPU (not priced)"
INFEASIBLE_SLO = "SLO not reachable"
API = "API"


@dataclass(frozen=True)
class Option:
    """A self-hosted cloud option at the measured model size."""
    name: str
    hourly_eur: float
    requests_per_hour: float      # at the measured size
    variable_eur: float           # per request (egress)
    latency_p95_s: float          # p95 at the sustained rate, at the measured size
    vram_gb: float


def max_size_b(vram_gb: float) -> float:
    return vram_gb * WEIGHT_SHARE / BYTES_PER_PARAM


def vision_options(report: dict) -> list[Option]:
    """Cloud options of the vision workload, with the latency each would have at the sustained rate."""
    workload = next(w for w in report["workloads"] if w["task"] == "vision")
    reference_rph = workload["sustained_step"]["rate_rps"] * 3600.0
    reference_p95 = workload["sustained_step"]["latency_p95_s"]
    options = []
    for o in workload["options"]:
        family = o["option"].split(".")[0]
        if "eur_per_1000" not in o or family not in GPU_VRAM_GB:
            continue
        slowdown = reference_rph / o["requests_per_hour"]   # 1 on the A10G, 1 / 0.33 on the T4
        options.append(Option(o["option"], o["hourly_eur"], o["requests_per_hour"],
                              o["variable_eur_per_1000"] / REQUESTS, reference_p95 * slowdown,
                              GPU_VRAM_GB[family]))
    return options


def slo_s(report: dict) -> float:
    return next(w for w in report["workloads"] if w["task"] == "vision")["sustained_step"]["slo_p95_s"]


def api_eur_per_1000(report: dict, batch: bool) -> float:
    prices = next(w for w in report["workloads"] if w["task"] == "vision")["apis_eur_per_1000"]
    return prices["gemini-3.1-flash-lite-batch" if batch else "gemini-3.1-flash-lite"]


def option_cost_per_1000(option: Option, size_b: float, monthly_requests: float) -> float:
    """Cost per 1000 requests: one instance always on, and beyond its capacity the instance-hours the
    volume needs (autoscaling at full load)."""
    requests_per_hour = option.requests_per_hour * (MEASURED_SIZE_B / size_b)
    instance_hours = max(HOURS_PER_MONTH, monthly_requests / requests_per_hour)
    monthly = instance_hours * option.hourly_eur + monthly_requests * option.variable_eur
    return monthly / monthly_requests * REQUESTS


def feasible(option: Option, size_b: float, slo: float | None) -> str | None:
    """None when the option can serve this model size, else the reason it cannot."""
    if size_b > max_size_b(option.vram_gb):
        return INFEASIBLE_VRAM
    if slo is not None and option.latency_p95_s * size_b / MEASURED_SIZE_B > slo:
        return INFEASIBLE_SLO
    return None


@dataclass(frozen=True)
class Cell:
    ratio: float | None     # best self-hosted cost / API cost; None when nothing is feasible
    winner: str             # option name, API, or an infeasibility reason


def best_cell(options: list[Option], size_b: float, monthly_requests: float, api: float,
              slo: float | None) -> Cell:
    costs, reasons = {}, []
    for option in options:
        reason = feasible(option, size_b, slo)
        if reason:
            reasons.append(reason)
            continue
        costs[option.name] = option_cost_per_1000(option, size_b, monthly_requests)
    if not costs:
        return Cell(None, INFEASIBLE_VRAM if INFEASIBLE_VRAM in reasons and len(set(reasons)) == 1
                    else INFEASIBLE_SLO)
    name = min(costs, key=costs.get)
    ratio = costs[name] / api
    return Cell(ratio, name if ratio < 1 else API)


def grid(options: list[Option], sizes_b: np.ndarray, volumes: np.ndarray, api: float,
         slo: float | None) -> tuple[np.ndarray, np.ndarray]:
    """log2(cost ratio) (NaN where infeasible) and the winner of every (size, volume) cell."""
    log_ratio = np.full((len(sizes_b), len(volumes)), np.nan)
    winners = np.empty((len(sizes_b), len(volumes)), dtype=object)
    for i, size in enumerate(sizes_b):
        for j, volume in enumerate(volumes):
            cell = best_cell(options, size, volume, api, slo)
            winners[i, j] = cell.winner
            if cell.ratio is not None:
                log_ratio[i, j] = math.log2(cell.ratio)
    return log_ratio, winners


# ---------- break-even model size (simple view) ----------
ANCHOR_SIZE_B = {"audio": 0.809, "vision": MEASURED_SIZE_B}   # whisper-large-v3-turbo, Qwen3-VL-2B
ANCHOR_MODEL = {"audio": "whisper-large-v3-turbo", "vision": "Qwen3-VL-2B"}
CLOUD_FAMILIES = tuple(GPU_VRAM_GB)


@dataclass(frozen=True)
class SizeLine:
    """Cost per 1000 requests of a cloud option as a straight function of model size."""
    option: str
    fixed_per_1000_at_anchor: float   # GPU-hours part at the anchor size and 100 % utilization
    variable_per_1000: float
    anchor_b: float

    def cost(self, size_b: np.ndarray | float, utilization: float = 1.0) -> np.ndarray | float:
        return self.fixed_per_1000_at_anchor * (size_b / self.anchor_b) / utilization + self.variable_per_1000

    def break_even_b(self, api_per_1000: float, utilization: float = 1.0) -> float | None:
        margin = api_per_1000 - self.variable_per_1000
        if margin <= 0:
            return None
        return self.anchor_b * margin * utilization / self.fixed_per_1000_at_anchor


def cheapest_line(report: dict, task: str) -> SizeLine:
    """The cloud option with the lowest cost per request at full load, measured at the anchor size."""
    workload = next(w for w in report["workloads"] if w["task"] == task)
    rows = [o for o in workload["options"] if "eur_per_1000" in o and o["option"].split(".")[0] in CLOUD_FAMILIES]
    best = min(rows, key=lambda o: o["eur_per_1000"]["1.0"])
    variable = best["variable_eur_per_1000"]
    return SizeLine(best["option"], best["eur_per_1000"]["1.0"] - variable, variable, ANCHOR_SIZE_B[task])
