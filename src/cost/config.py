"""Typed loading of conf/prices.yaml (facts) and conf/cost_scenarios.yaml (assumptions)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICES_FILE = PROJECT_ROOT / "conf" / "prices.yaml"
SCENARIOS_FILE = PROJECT_ROOT / "conf" / "cost_scenarios.yaml"


# ---------- prices.yaml ----------
class Fx(BaseModel):
    usd_per_eur: float
    date: str
    source: str


class Instance(BaseModel):
    gpu: str
    vcpu: int
    memory_gib: int
    on_demand_hour: float
    spot_hour: float | None = None
    spot_interruption_pct: float | None = None


class AwsPrices(BaseModel):
    region: str
    hours_per_month: float
    sources: dict[str, str]
    instances: dict[str, Instance]
    ebs_gp3_gb_month: float
    s3_standard_gb_month: float
    egress_internet_gb: float


class LocalPrices(BaseModel):
    electricity_eur_kwh: float
    electricity_note: str
    electricity_source: str


class ApiPricing(BaseModel):
    kind: Literal["per_audio_minute", "per_audio_hour", "per_token"]
    usd: float | None = None
    min_billed_s: float = 0.0
    input_usd_per_m: float | None = None
    output_usd_per_m: float | None = None


class Api(BaseModel):
    task: str
    pricing: ApiPricing
    source: str
    note: str | None = None


class Prices(BaseModel):
    retrieved: str
    fx: Fx
    aws: AwsPrices
    local: LocalPrices
    apis: dict[str, Api]


# ---------- cost_scenarios.yaml ----------
class Selector(BaseModel):
    """Records of one task/tag/env, optionally filtered on metric values; `field` is averaged."""
    task: str | None = None
    tag: str
    env: str
    field: str | None = None
    where: dict[str, Any] = {}


class Ratio(BaseModel):
    numerator: Selector
    denominator: Selector


class Quality(BaseModel):
    """Quality metric of the self-hosted model and of each API, measured on the same input."""
    field: str
    self_hosted: Selector
    variants: dict[str, Selector] = {}   # same model elsewhere, e.g. fp16 on the T4
    apis: dict[str, Selector] = {}


class Workload(BaseModel):
    unit: str
    load_tag_prefix: str
    reference_env: str
    egress_kb_per_request: float
    coldstart: Selector
    audio_s_per_request: float | None = None
    tokens: Selector | None = None
    quality: Quality | None = None


class Target(BaseModel):
    kind: Literal["local", "aws"]
    instance: str | None = None
    scale: dict[str, float | dict[str, Ratio] | None] = {}
    unloaded_latency: dict[str, Selector] = {}   # one request alone on this GPU, checked against the SLO


class AwsAssumptions(BaseModel):
    ebs_root_gb: float
    s3_results_gb: float


class LocalAssumptions(BaseModel):
    hardware_eur: float
    lifetime_years: float
    system_power_factor: float
    idle_wall_w: float


class Scenarios(BaseModel):
    utilizations: list[float]
    aws: AwsAssumptions
    local: LocalAssumptions
    workloads: dict[str, Workload]
    targets: dict[str, Target]


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_prices(path: Path = PRICES_FILE) -> Prices:
    return Prices.model_validate(read_yaml(path))


def load_scenarios(path: Path = SCENARIOS_FILE) -> Scenarios:
    return Scenarios.model_validate(read_yaml(path))
