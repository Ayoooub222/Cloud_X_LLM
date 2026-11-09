"""Unit tests for the sd-server log parser (no docker, no GPU)."""
from src.bench import sdcpp_logs

LOAD_LINE = "[INFO ] model.cpp:100 - loading tensors completed, taking {}s"
ENCODE_LINE = "[INFO ] stable-diffusion.cpp:200 - get_learned_condition completed, taking {}s"
SAMPLING_LINE = "[INFO ] stable-diffusion.cpp:300 - sampling completed, taking {}s"
DECODE_LINE = "[INFO ] stable-diffusion.cpp:400 - decode_first_stage completed, taking {}s"
TOTAL_LINE = "[INFO ] stable-diffusion.cpp:500 - generate_image completed in {}s"
CACHE_LINE = "[DEBUG] conditioner.hpp:50 - conditioning cache hit"

FULL_RUN = "\n".join((
    LOAD_LINE.format("2.50"),
    ENCODE_LINE.format("0.80"),
    SAMPLING_LINE.format("10.90"),
    DECODE_LINE.format("1.20"),
    TOTAL_LINE.format("13.00"),
))


def test_full_run_is_parsed():
    times = sdcpp_logs.parse_stage_times(FULL_RUN)
    assert times["encode_s"] == 0.8
    assert times["sampling_s"] == 10.9
    assert times["vae_decode_s"] == 1.2
    assert times["server_total_s"] == 13.0
    assert times["weights_load_s"] == 2.5
    assert times["cache_hit"] == 0
    assert times["cpu_share_pct"] == 6.2


def test_last_match_wins():
    logs = "\n".join((SAMPLING_LINE.format("10.0"), SAMPLING_LINE.format("11.0")))
    assert sdcpp_logs.parse_stage_times(logs)["sampling_s"] == 11.0


def test_weights_loads_are_summed():
    logs = "\n".join((LOAD_LINE.format("1.25"), LOAD_LINE.format("1.25")))
    assert sdcpp_logs.parse_stage_times(logs)["weights_load_s"] == 2.5


def test_cache_hit_is_detected():
    logs = "\n".join((CACHE_LINE, FULL_RUN))
    assert sdcpp_logs.parse_stage_times(logs)["cache_hit"] == 1


def test_empty_logs_give_empty_dict():
    assert sdcpp_logs.parse_stage_times("") == {}


def test_no_encode_means_no_derived_fields():
    times = sdcpp_logs.parse_stage_times(TOTAL_LINE.format("13.00"))
    assert "cache_hit" not in times
    assert "cpu_share_pct" not in times


def test_docker_cmd_truncates_timestamp():
    cmd = sdcpp_logs.docker_logs_cmd(1700000000.9)
    assert cmd == ["docker", "logs", "--since", "1700000000", sdcpp_logs.CONTAINER]


def test_k8s_cmd_uses_rfc3339_utc():
    cmd = sdcpp_logs.k8s_logs_cmd(1700000000)
    assert cmd[0] == "kubectl"
    assert cmd[-2:] == ["--since-time", "2023-11-14T22:13:20Z"]


class FakeResult:
    stdout = "out "
    stderr = "err"


def test_read_logs_uses_kubectl_in_k8s_mode(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return FakeResult()

    monkeypatch.setattr(sdcpp_logs, "LOG_SOURCE", "k8s")
    monkeypatch.setattr(sdcpp_logs.subprocess, "run", fake_run)
    assert sdcpp_logs.read_logs(0) == "out err"
    assert calls[0][0] == "kubectl"


def test_read_logs_returns_empty_on_failure(monkeypatch):
    def failing_run(cmd, **kwargs):
        raise OSError("docker not found")

    monkeypatch.setattr(sdcpp_logs.subprocess, "run", failing_run)
    assert sdcpp_logs.read_logs(0) == ""
