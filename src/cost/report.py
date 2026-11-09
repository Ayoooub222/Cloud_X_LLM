"""Cost per 1000 requests: self-hosting (laptop, AWS on-demand, AWS spot) against hosted APIs.

Inputs: conf/prices.yaml (public prices), conf/cost_scenarios.yaml (assumptions) and the
benchmark records (sustained throughput at the SLO from the k6 load steps).

Examples:
    python -m src.cost.report
    python -m src.cost.report --format json --out outputs/cost/report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.cost import model
from src.cost.config import Prices, Scenarios, Selector, Target, Workload, load_prices, load_scenarios
from src.cost.measurements import SustainedStep, load_steps, ratio_value, selector_mean

PRICING_ON_DEMAND = "on-demand"
PRICING_SPOT = "spot"
NEVER = "never"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost per 1000 requests, self-hosted against hosted APIs")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown")
    p.add_argument("--out", type=Path, help="write to this file instead of stdout")
    return p.parse_args(argv)


def target_scale(target: Target, task: str) -> float | None:
    """Throughput factor of the target against the reference env (None: not measured)."""
    if target.kind == "local":
        return 1.0
    value = target.scale.get(task)
    if value is None or isinstance(value, float | int):
        return value
    return ratio_value(value["ratio"])


def api_costs_eur(prices: Prices, task: str, workload: Workload) -> dict[str, float]:
    tokens = {}
    if workload.tokens is not None:
        tokens = {"input_tokens": selector_mean(workload.tokens, "prompt_tokens"),
                  "output_tokens": selector_mean(workload.tokens, "completion_tokens")}
    costs = {}
    for name, api in prices.apis.items():
        if api.task != task:
            continue
        usd = model.api_cost_usd(api.pricing.kind, audio_s=workload.audio_s_per_request, **tokens,
                                 **api.pricing.model_dump(exclude={"kind"}))
        costs[name] = model.usd_to_eur(usd, prices.fx.usd_per_eur)
    return costs


def quality_values(workload: Workload) -> dict:
    """Measured quality: {"field", "self_hosted", "variants": {...}, "apis": {...}}, None when not measured."""
    q = workload.quality
    if q is None:
        return {}
    return {"field": q.field, "self_hosted": selector_mean(q.self_hosted, q.field),
            "variants": optional_means(q.variants, q.field), "apis": optional_means(q.apis, q.field)}


def optional_means(selectors: dict[str, Selector], field: str) -> dict[str, float | None]:
    values = {}
    for name, selector in selectors.items():
        try:
            values[name] = selector_mean(selector, field)
        except ValueError:
            values[name] = None
    return values


def quality_line(quality: dict) -> str:
    variants = "".join(f"; {name}: {v:.4f}" for name, v in quality["variants"].items() if v is not None)
    return (f"Self-hosted {quality['field']}: {quality['self_hosted']:.4f}{variants} "
            "(same model on every self-hosted option).")


def option_row(name: str, hourly_eur: float, requests_per_hour: float, variable_eur: float,
               utilizations: list[float], apis: dict[str, float], hours_per_month: float, note: str) -> dict:
    per_1000 = {str(u): model.REQUESTS * model.cost_per_request(hourly_eur, requests_per_hour, u, variable_eur)
                for u in utilizations}
    break_even = {}
    for api, api_eur in apis.items():
        u = model.break_even_utilization(hourly_eur, requests_per_hour, api_eur, variable_eur)
        break_even[api] = u if u is not None and u <= 1 else NEVER
    return {"option": name, "requests_per_hour": requests_per_hour, "hourly_eur": hourly_eur,
            "variable_eur_per_1000": variable_eur * model.REQUESTS, "eur_per_1000": per_1000,
            "break_even_utilization": break_even, "always_on_eur_month": hourly_eur * hours_per_month,
            "note": note}


def aws_rows(prices: Prices, scenarios: Scenarios, target_name: str, target: Target, task: str,
             workload: Workload, step: SustainedStep, apis: dict[str, float]) -> list[dict]:
    scale = target_scale(target, task)
    if scale is None:
        return [{"option": target_name, "note": "not measured on this GPU"}]
    aws, fx = prices.aws, prices.fx.usd_per_eur
    instance = aws.instances[target.instance]
    requests_per_hour = step.rate_rps * model.SECONDS_PER_HOUR * scale
    egress = model.egress_eur(workload.egress_kb_per_request, aws.egress_internet_gb, fx)
    restart_s = selector_mean(workload.coldstart)
    slo_note = slo_check(target, task, step)
    rows = []
    for pricing, instance_hour in ((PRICING_ON_DEMAND, instance.on_demand_hour), (PRICING_SPOT, instance.spot_hour)):
        if instance_hour is None:
            continue
        hourly_usd = model.aws_hourly_usd(instance_hour, scenarios.aws.ebs_root_gb, aws.ebs_gp3_gb_month,
                                          scenarios.aws.s3_results_gb, aws.s3_standard_gb_month, aws.hours_per_month)
        note = f"{instance.gpu}, throughput x{scale:.2f} of the reference"
        if pricing == PRICING_SPOT:
            overhead = model.spot_overhead(instance.spot_interruption_pct, restart_s, aws.hours_per_month)
            hourly_usd *= 1 + overhead
            bucket = instance.spot_interruption_pct
            note += f", interruptions <= {bucket:g} %/month" if bucket is not None else ", interruptions unknown"
        row = option_row(f"{target_name} {pricing}", model.usd_to_eur(hourly_usd, fx), requests_per_hour,
                         egress, scenarios.utilizations, apis, aws.hours_per_month, note + (slo_note or ""))
        row["slo_reachable"] = slo_note is None
        rows.append(row)
    return rows


def slo_check(target: Target, task: str, step: SustainedStep) -> str | None:
    """Note when one request alone on the target GPU is already slower than the SLO, else None."""
    selector = target.unloaded_latency.get(task)
    if selector is None:
        return None
    latency_s = selector_mean(selector)
    if latency_s <= step.slo_p95_s:
        return None
    return (f", SLO not reachable: one request alone takes {latency_s:.1f} s for a {step.slo_p95_s:g} s SLO "
            "(cost valid without the SLO, e.g. batch processing)")


def local_row(prices: Prices, scenarios: Scenarios, target_name: str, step: SustainedStep,
              apis: dict[str, float]) -> dict:
    local = scenarios.local
    hourly = model.local_hourly_eur(local.hardware_eur, local.lifetime_years, local.idle_wall_w,
                                    prices.local.electricity_eur_kwh)
    energy = model.local_energy_eur(step.wh_per_request or 0.0, local.system_power_factor,
                                    prices.local.electricity_eur_kwh)
    note = (f"amortization {local.hardware_eur:g} EUR / {local.lifetime_years:g} years, idle "
            f"{local.idle_wall_w:g} W, electricity x{local.system_power_factor:g} of GPU energy (assumptions)")
    return option_row(target_name, hourly, step.rate_rps * model.SECONDS_PER_HOUR, energy,
                      scenarios.utilizations, apis, prices.aws.hours_per_month, note)


def workload_report(prices: Prices, scenarios: Scenarios, task: str, workload: Workload) -> dict:
    step = load_steps(workload.load_tag_prefix, workload.reference_env)
    apis = api_costs_eur(prices, task, workload)
    options = []
    for name, target in scenarios.targets.items():
        if target.kind == "local":
            options.append(local_row(prices, scenarios, name, step, apis))
        else:
            options.extend(aws_rows(prices, scenarios, name, target, task, workload, step, apis))
    return {"task": task, "unit": workload.unit, "sustained_step": step.__dict__,
            "apis_eur_per_1000": {k: v * model.REQUESTS for k, v in apis.items()},
            "quality": quality_values(workload), "options": options}


def build_report(prices: Prices, scenarios: Scenarios) -> dict:
    return {"prices_retrieved": prices.retrieved, "usd_per_eur": prices.fx.usd_per_eur,
            "utilizations": scenarios.utilizations,
            "workloads": [workload_report(prices, scenarios, task, w) for task, w in scenarios.workloads.items()]}


# ---------- markdown ----------
def eur(value: float) -> str:
    if value >= 100:
        return f"{value:,.0f}".replace(",", " ")
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def share(value: float | str) -> str:
    return value if isinstance(value, str) else f"{value:.0%}"


def markdown(report: dict) -> str:
    lines = [f"Prices retrieved {report['prices_retrieved']}, 1 EUR = {report['usd_per_eur']} USD. "
             "EUR per 1000 requests at the SLO.", ""]
    utils = report["utilizations"]
    for w in report["workloads"]:
        step = w["sustained_step"]
        apis = list(w["apis_eur_per_1000"])
        lines += [f"### {w['task']}: {w['unit']}", "",
                  f"Sustained at the SLO on the reference: {step['rate_rps']:.3f} req/s ({step['tag']}, "
                  f"p95 {step['latency_p95_s']:.2f} s for a {step['slo_p95_s']:g} s SLO, {step['runs']} runs).", "",
                  ]
        quality = w["quality"]
        if quality:
            lines += [quality_line(quality), ""]
        lines += ["| API | EUR / 1000 | " + (quality.get("field", "quality") if quality else "quality") + " |",
                  "|---|---|---|"]
        for name, v in w["apis_eur_per_1000"].items():
            measured = quality.get("apis", {}).get(name) if quality else None
            shown = f"{measured:.4f}" if measured is not None else "not measured"
            lines.append(f"| {name} | {eur(v)} | {shown} |")
        header = (["Option", "req/h", "EUR/h"] + [f"EUR/1000 at {u:.0%}" for u in utils]
                  + [f"break-even vs {a}" for a in apis] + ["24/7 EUR/month"])
        lines += ["", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for o in w["options"]:
            if "eur_per_1000" not in o:
                lines.append(f"| {o['option']} | {o['note']} |" + " |" * (len(header) - 2))
                continue
            name = o["option"] if o.get("slo_reachable", True) else f"{o['option']} (SLO not reachable)"
            cells = [name, f"{o['requests_per_hour']:.0f}", eur(o["hourly_eur"])]
            cells += [eur(o["eur_per_1000"][str(u)]) for u in utils]
            cells += [share(o["break_even_utilization"][a]) for a in apis]
            cells.append(eur(o["always_on_eur_month"]))
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "Notes: " + "; ".join(f"{o['option']}: {o['note']}" for o in w["options"]), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(load_prices(), load_scenarios())
    text = json.dumps(report, indent=2) if args.format == "json" else markdown(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"-> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
