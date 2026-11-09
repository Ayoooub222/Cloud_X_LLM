"""Batch metrics of runner_vision (no server needed)."""
from types import SimpleNamespace

from src.runner.runner_vision import accuracy_metrics, batch_metrics

USAGE = SimpleNamespace(prompt_tokens=3000, completion_tokens=1000, total_tokens=4000)


def test_batch_metrics_throughputs():
    results = [(10.0, USAGE, {}), (20.0, USAGE, {})]
    m = batch_metrics(results, wall_s=20.0)
    assert m["requests"] == 2
    assert m["latency_s"] == 15.0
    assert m["latency_max_s"] == 20.0
    assert m["batch_tokens_per_s"] == 100.0   # 2 x 1000 tokens in 20 s
    assert m["pages_per_hour"] == 360.0       # 2 pages in 20 s


def test_accuracy_metrics_mean_and_min():
    expected = {"a": "x", "b": "y"}
    m = accuracy_metrics(expected, [{"a": "x", "b": "y"}, {"a": "x", "b": "z"}])
    assert m["field_accuracy"] == 0.75
    assert m["field_accuracy_min"] == 0.5
    assert m["fields_total"] == 2


def test_server_root_strips_openai_suffix():
    from src.runner.runner_vision import server_root

    client = SimpleNamespace(openai=SimpleNamespace(base_url="http://localhost:8000/v1/"))
    assert server_root(client) == "http://localhost:8000"
