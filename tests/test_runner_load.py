"""Conversion of the k6 step summary into record metrics (no k6 needed)."""
from src.runner.runner_load import parse_args, summary_metrics


def make_summary(p95_ms: float, failed: float = 0.0, dropped: int = 0) -> dict:
    return {
        "slo_p95_ms": 5000, "rate_rpm": 60, "duration_s": 60, "test_run_duration_ms": 62000.0,
        "requests": 62, "failed_rate": failed, "dropped_iterations": dropped, "vus_max": 4,
        "latency_ms": {"avg": 900.0, "min": 500.0, "med": 800.0, "p(90)": 1500.0,
                       "p(95)": p95_ms, "p(99)": 2500.0, "max": 3000.0},
    }


def test_step_within_slo():
    m = summary_metrics(make_summary(2000.0))
    assert m["target_rps"] == 1.0
    assert m["achieved_rps"] == 1.0
    assert m["latency_p50_s"] == 0.8
    assert m["latency_p95_s"] == 2.0
    assert m["slo_p95_s"] == 5.0
    assert m["slo_ok"] == 1


def test_step_over_slo():
    assert summary_metrics(make_summary(6000.0))["slo_ok"] == 0


def test_failures_or_dropped_iterations_break_the_slo():
    assert summary_metrics(make_summary(2000.0, failed=0.05))["slo_ok"] == 0
    assert summary_metrics(make_summary(2000.0, dropped=3))["slo_ok"] == 0


def test_default_rates_per_target():
    args = parse_args(["vision"])
    assert args.rates is None and args.repeats == 1
