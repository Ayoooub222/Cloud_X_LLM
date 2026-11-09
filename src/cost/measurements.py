"""Numbers the cost model takes from the benchmark records (results/benchmark)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from src.cost.config import Ratio, Selector
from src.runner.summarize import env_label, is_failure, load_records

LOAD_TASK = "load"


@dataclass(frozen=True)
class SustainedStep:
    """Highest k6 step whose runs all met the SLO."""
    tag: str
    runs: int
    rate_rps: float          # arrival rate of the step = served rate (no failure, no drop)
    achieved_rps: float      # served requests / k6 run time (includes the final drain)
    latency_p95_s: float
    slo_p95_s: float
    wh_per_request: float | None
    capacity_rps: float      # best achieved rate over all steps, SLO met or not


def field_value(record: dict, field: str) -> float | None:
    """metrics.<field>, or gpu.<name> for GPU stats."""
    if field.startswith("gpu."):
        return (record.get("gpu") or {}).get(field[len("gpu."):])
    return (record.get("metrics") or {}).get(field)


def select(records: list[dict], selector: Selector) -> list[dict]:
    """Successful records of the selector's tag and env, matching every `where` metric."""
    return [
        r for r in records
        if not is_failure(r) and r.get("tag") == selector.tag and env_label(r) == selector.env
        and all(field_value(r, key) == value for key, value in selector.where.items())
    ]


def mean_field(records: list[dict], field: str) -> float:
    values = [v for r in records if (v := field_value(r, field)) is not None]
    if not values:
        raise ValueError(f"no value for {field!r} in {len(records)} records")
    return statistics.fmean(values)


def selector_mean(selector: Selector, field: str | None = None) -> float:
    task = selector.task
    if task is None:
        raise ValueError(f"selector {selector.tag!r} needs a task")
    records = select(load_records(task, None, [selector.tag], [selector.env]), selector)
    return mean_field(records, field or selector.field or "")


def ratio_value(ratio: Ratio) -> float:
    return selector_mean(ratio.numerator) / selector_mean(ratio.denominator)


def group_steps(records: list[dict], prefix: str, env: str) -> dict[str, list[dict]]:
    steps: dict[str, list[dict]] = {}
    for record in records:
        tag = record.get("tag") or ""
        if tag.startswith(prefix) and env_label(record) == env:
            steps.setdefault(tag, []).append(record)
    return steps


def sustained_step(records: list[dict], prefix: str, env: str) -> SustainedStep:
    """Highest arrival rate at which every run met the SLO (failed runs disqualify the step)."""
    steps = group_steps(records, prefix, env)
    if not steps:
        raise ValueError(f"no load record with tag prefix {prefix!r} for env {env!r}")
    served = [ok for rs in steps.values() if (ok := [r for r in rs if not is_failure(r)])]
    if not served:
        raise ValueError(f"every load run of {prefix!r} failed")
    capacity = max(mean_field(ok, "achieved_rps") for ok in served)
    passing = {
        tag: rs for tag, rs in steps.items()
        if all(not is_failure(r) and field_value(r, "slo_ok") == 1 for r in rs)
    }
    if not passing:
        raise ValueError(f"no step of {prefix!r} met the SLO")
    tag, runs = max(passing.items(), key=lambda item: mean_field(item[1], "target_rps"))
    energy = [v for r in runs if (v := field_value(r, "wh_per_request")) is not None]
    return SustainedStep(
        tag=tag,
        runs=len(runs),
        rate_rps=mean_field(runs, "target_rps"),
        achieved_rps=mean_field(runs, "achieved_rps"),
        latency_p95_s=mean_field(runs, "latency_p95_s"),
        slo_p95_s=mean_field(runs, "slo_p95_s"),
        wh_per_request=statistics.fmean(energy) if energy else None,
        capacity_rps=capacity,
    )


def load_steps(prefix: str, env: str) -> SustainedStep:
    return sustained_step(load_records(LOAD_TASK, None, None, [env]), prefix, env)
