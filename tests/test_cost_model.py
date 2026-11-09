"""Cost formulas, sustained-step selection and the full report on the repository data."""
import pytest

from src.cost import model
from src.cost.config import Selector, load_prices, load_scenarios
from src.cost.measurements import select, sustained_step
from src.cost.report import NEVER, build_report, markdown


def test_cost_per_request_scales_with_utilization():
    assert model.cost_per_request(10.0, 100.0, 1.0) == 0.1
    assert model.cost_per_request(10.0, 100.0, 0.5, variable_eur=0.01) == pytest.approx(0.21)


def test_break_even_utilization():
    assert model.break_even_utilization(10.0, 100.0, api_eur=0.2) == 0.5
    assert model.break_even_utilization(10.0, 100.0, api_eur=0.01, variable_eur=0.02) is None


def test_api_costs():
    assert model.api_cost_usd("per_audio_minute", audio_s=90.0, usd=0.006) == pytest.approx(0.009)
    assert model.api_cost_usd("per_audio_hour", audio_s=5.0, usd=0.04, min_billed_s=10.0) == pytest.approx(0.04 / 360)
    assert model.api_cost_usd("per_token", input_tokens=2e6, output_tokens=1e6,
                              input_usd_per_m=0.25, output_usd_per_m=1.5) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        model.api_cost_usd("per_page")


def test_aws_hourly_and_spot_overhead():
    assert model.aws_hourly_usd(1.0, 100.0, 0.073, 0.0, 0.0, 730.0) == pytest.approx(1.01)
    assert model.spot_overhead(None, 100.0, 730.0) == 0.0
    assert model.spot_overhead(73.0, 3600.0, 730.0) == pytest.approx(0.001)


def test_local_costs():
    assert model.local_hourly_eur(876.0, 1.0, 0.0, 0.2) == pytest.approx(0.1)
    assert model.local_hourly_eur(0.0, 1.0, 500.0, 0.2) == pytest.approx(0.1)
    assert model.local_energy_eur(10.0, 2.0, 0.2) == pytest.approx(0.004)


def load_record(tag, target_rps, slo_ok, achieved=None, error=None, env="e"):
    return {"tag": tag, "env": {"label": env}, "error": error,
            "metrics": {"target_rps": target_rps, "achieved_rps": achieved or target_rps, "slo_ok": slo_ok,
                        "latency_p95_s": 1.0, "slo_p95_s": 5.0, "wh_per_request": 0.01, "requests": 3}}


def test_sustained_step_needs_every_run_within_the_slo():
    records = [load_record("k6-a-r60", 1.0, 1), load_record("k6-a-r60", 1.0, 1),
               load_record("k6-a-r120", 2.0, 1), load_record("k6-a-r120", 2.0, 0, achieved=1.8),
               load_record("k6-a-r30", 0.5, 1, env="other")]
    step = sustained_step(records, "k6-a-r", "e")
    assert step.tag == "k6-a-r60"
    assert step.rate_rps == 1.0
    assert step.capacity_rps == pytest.approx(1.9)


def test_sustained_step_failed_run_disqualifies():
    records = [load_record("k6-a-r60", 1.0, 1, error="boom"), load_record("k6-a-r30", 0.5, 1)]
    assert sustained_step(records, "k6-a-r", "e").tag == "k6-a-r30"


def test_select_where_filter():
    records = [load_record("t", 1.0, 1), load_record("t", 1.0, 1) | {"metrics": {"requests": 8}}]
    assert len(select(records, Selector(tag="t", env="e", where={"requests": 8}))) == 1


def test_report_on_repository_data():
    report = build_report(load_prices(), load_scenarios())
    tasks = {w["task"]: w for w in report["workloads"]}
    assert set(tasks) == {"audio", "vision"}
    audio = {o["option"]: o for o in tasks["audio"]["options"]}
    assert audio["g4dn.xlarge spot"]["hourly_eur"] < audio["g4dn.xlarge on-demand"]["hourly_eur"]
    assert audio["local-3070"]["eur_per_1000"]["0.1"] > audio["local-3070"]["eur_per_1000"]["1.0"]
    vision = {o["option"]: o for o in tasks["vision"]["options"]}
    assert vision["g4dn.xlarge spot"]["eur_per_1000"]["1.0"] > vision["g5.xlarge spot"]["eur_per_1000"]["1.0"]
    assert all(v == NEVER or 0 < v <= 1 for o in audio.values() for v in o["break_even_utilization"].values())
    assert "### audio" in markdown(report)
    assert tasks["audio"]["quality"]["field"] == "wer"
    assert tasks["audio"]["quality"]["apis"]["groq-whisper-large-v3-turbo"] is not None
    assert 0 < tasks["vision"]["quality"]["self_hosted"] <= 1
    assert tasks["audio"]["quality"]["variants"]["fp16 on the T4"] is not None
    assert "fp16 on the T4" in markdown(report)
    assert audio["g4dn.xlarge spot"]["slo_reachable"]
    assert not vision["g4dn.xlarge spot"]["slo_reachable"]
    assert "SLO not reachable" in markdown(report)
