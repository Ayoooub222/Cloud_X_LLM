"""Parsing of the vLLM Prometheus text format (no server needed)."""
from src.bench.vllm_metrics import (
    KV_CACHE,
    RUNNING,
    WAITING,
    WAITING_BY_REASON,
    parse_metrics,
    summarize_samples,
)

SAMPLE = """# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="openai/whisper-large-v3-turbo"} 4.0
vllm:num_requests_waiting{engine="0",model_name="openai/whisper-large-v3-turbo"} 20.0
vllm:kv_cache_usage_perc{engine="0",model_name="openai/whisper-large-v3-turbo"} 0.125
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 3.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="deferred"} 17.0
vllm:num_requests_running_total_not_tracked 99.0
"""


def test_parse_metrics_tracked_gauges_only():
    assert parse_metrics(SAMPLE) == {RUNNING: 4.0, WAITING: 20.0, KV_CACHE: 0.125,
                                     f"{WAITING_BY_REASON}:capacity": 3.0,
                                     f"{WAITING_BY_REASON}:deferred": 17.0}


def test_summarize_samples_max_and_mean():
    samples = [{RUNNING: 2.0, WAITING: 6.0, KV_CACHE: 0.1}, {RUNNING: 4.0, WAITING: 2.0, KV_CACHE: 0.3}]
    m = summarize_samples(samples)
    assert m["sched.samples"] == 2
    assert m["sched.running_max"] == 4.0
    assert m["sched.waiting_avg"] == 4.0
    assert m["sched.kv_cache_max"] == 0.3
