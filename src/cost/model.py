"""Cost formulas (pure functions, EUR). Hourly costs are paid whether the GPU serves or idles,
so the cost of one served request is hourly_cost / (sustained requests per hour * utilization).
"""
from __future__ import annotations

SECONDS_PER_HOUR = 3600.0
SECONDS_PER_MINUTE = 60.0
HOURS_PER_YEAR = 8760.0
KB_PER_GB = 1_000_000.0
TOKENS_PER_M = 1_000_000.0
WH_PER_KWH = 1000.0
REQUESTS = 1000  # results are given per 1000 requests


def usd_to_eur(usd: float, usd_per_eur: float) -> float:
    return usd / usd_per_eur


def aws_hourly_usd(instance_hour: float, ebs_gb: float, ebs_gb_month: float,
                   s3_gb: float, s3_gb_month: float, hours_per_month: float) -> float:
    """Instance + root volume + results bucket, per hour."""
    return instance_hour + (ebs_gb * ebs_gb_month + s3_gb * s3_gb_month) / hours_per_month


def spot_overhead(interruption_pct_month: float | None, restart_s: float, hours_per_month: float) -> float:
    """Share of paid time lost to interruptions: interruptions per hour x restart time.

    The advisor gives the share of instances interrupted per month; one instance running all
    month is taken as interrupted with that probability (upper bound of the bucket).
    """
    if interruption_pct_month is None:
        return 0.0
    interruptions_per_hour = interruption_pct_month / 100 / hours_per_month
    return interruptions_per_hour * restart_s / SECONDS_PER_HOUR


def local_hourly_eur(hardware_eur: float, lifetime_years: float, idle_wall_w: float, eur_kwh: float) -> float:
    """Paid every hour, busy or not: straight-line amortization plus idle electricity.

    The idle power is also inside the measured energy per request: a small double count,
    on the safe (expensive) side.
    """
    return hardware_eur / (lifetime_years * HOURS_PER_YEAR) + idle_wall_w / WH_PER_KWH * eur_kwh


def local_energy_eur(wh_per_request: float, system_power_factor: float, eur_kwh: float) -> float:
    """Electricity of one request: GPU energy scaled to wall power."""
    return wh_per_request * system_power_factor / WH_PER_KWH * eur_kwh


def egress_eur(kb_per_request: float, usd_per_gb: float, usd_per_eur: float) -> float:
    return usd_to_eur(kb_per_request / KB_PER_GB * usd_per_gb, usd_per_eur)


def cost_per_request(hourly_eur: float, requests_per_hour: float, utilization: float,
                     variable_eur: float = 0.0) -> float:
    """Fixed hourly cost shared by the requests actually served, plus a per-request cost."""
    return hourly_eur / (requests_per_hour * utilization) + variable_eur


def break_even_utilization(hourly_eur: float, requests_per_hour: float, api_eur: float,
                           variable_eur: float = 0.0) -> float | None:
    """Utilization above which self-hosting is cheaper than the API (None: never)."""
    margin = api_eur - variable_eur
    if margin <= 0:
        return None
    return hourly_eur / (requests_per_hour * margin)


def api_cost_usd(kind: str, *, audio_s: float | None = None, input_tokens: float | None = None,
                 output_tokens: float | None = None, usd: float | None = None, min_billed_s: float = 0.0,
                 input_usd_per_m: float | None = None, output_usd_per_m: float | None = None) -> float:
    """Price of one request on a hosted API."""
    if kind == "per_audio_minute":
        return max(audio_s, min_billed_s) / SECONDS_PER_MINUTE * usd
    if kind == "per_audio_hour":
        return max(audio_s, min_billed_s) / SECONDS_PER_HOUR * usd
    if kind == "per_token":
        return (input_tokens * input_usd_per_m + output_tokens * output_usd_per_m) / TOKENS_PER_M
    raise ValueError(f"unknown API pricing kind: {kind}")
