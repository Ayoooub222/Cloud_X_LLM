"""Unit tests for the benchmark summary (pure functions and file loading)."""
import json

import pytest

from src.runner import summarize

ENV_A = "local-k3s-rtx3070laptop"
ENV_B = "local-wsl2-rtx3070laptop"


def make_record(tag="a", metrics=None, gpu=None, error=None,
                timestamp="2026-10-24 12:34:56", label=ENV_A):
    return {
        "tag": tag,
        "timestamp": timestamp,
        "env": {"label": label},
        "metrics": metrics or {},
        "gpu": gpu or {},
        "error": error,
    }


def test_env_label_sources():
    assert summarize.env_label({"env": {"label": ENV_A}}) == ENV_A
    assert summarize.env_label({"label": ENV_B}) == ENV_B
    assert summarize.env_label({}) == summarize.NO_ENV


def test_flatten_numeric_skips_bool_and_text():
    data = {"a": 1, "b": {"c": 2.5}, "d": True, "e": "x"}
    assert summarize.flatten_numeric(data) == {"a": 1.0, "b.c": 2.5}


def test_numeric_fields_prefixes_gpu():
    record = make_record(metrics={"latency_s": 1.0}, gpu={"power_w": 50, "samples": 3})
    assert summarize.numeric_fields(record) == {
        "latency_s": 1.0, "gpu.power_w": 50.0, "gpu.samples": 3.0,
    }

def test_numeric_fields_without_gpu_samples():
    record = make_record(metrics={"t_first_s": 80.0}, gpu={"samples": 0, "power_w": None})
    assert summarize.numeric_fields(record) == {"t_first_s": 80.0}


def test_is_failure():
    assert not summarize.is_failure({"error": ""})
    assert not summarize.is_failure({})
    assert summarize.is_failure({"error": "boom"})


def test_decimals_for():
    assert summarize.decimals_for(8192.0) == 1
    assert summarize.decimals_for(-150.0) == 1
    assert summarize.decimals_for(1.09) == 2
    assert summarize.decimals_for(0.0106) == 4


def test_format_cell():
    assert summarize.format_cell([]) == "-"
    assert summarize.format_cell([1.0]) == "1.00"
    assert summarize.format_cell([0.01]) == "0.0100"
    assert summarize.format_cell([1.0, 3.0]) == "2.00 ±1.41"


def test_column_label():
    record = make_record(tag="single-11s")
    assert summarize.column_label(record, False, False) == "single-11s"
    assert summarize.column_label(record, True, False) == "single-11s@12:34:56"
    assert summarize.column_label(record, False, True) == f"single-11s [{ENV_A}]"
    assert summarize.column_label(make_record(tag=None), False, False) == "(no tag)"


def test_build_rows_excludes_failures_from_stats():
    groups = {"a": (
        make_record(metrics={"latency_s": 1.0}),
        make_record(metrics={"latency_s": 3.0}),
        make_record(metrics={"latency_s": 100.0}, error="timeout"),
    )}
    rows = summarize.build_rows(groups, None)
    assert rows[0] == ["runs ok/total", "2/3"]
    assert rows[1] == ["latency_s", "2.00 ±1.41"]


def test_build_rows_field_filter_and_missing_cell():
    groups = {
        "a": (make_record(metrics={"latency_s": 1.0, "tokens": 5}),),
        "b": (make_record(metrics={"tokens": 7}),),
    }
    rows = summarize.build_rows(groups, ["lat"])
    assert [row[0] for row in rows] == ["runs ok/total", "latency_s"]
    assert rows[1] == ["latency_s", "1.00", "-"]


def write_record(folder, name, record):
    (folder / name).write_text(json.dumps(record), encoding="utf-8")


def test_load_records_filters(tmp_path, monkeypatch):
    task_dir = tmp_path / "audio"
    task_dir.mkdir()
    write_record(task_dir, "1.json", make_record(tag="a", timestamp="2026-10-23 10:00:00"))
    write_record(task_dir, "2.json", make_record(tag="a", timestamp="2026-10-24 10:00:00"))
    write_record(task_dir, "3.json", make_record(tag="b", timestamp="2026-10-24 11:00:00", label=ENV_B))
    monkeypatch.setattr(summarize, "BENCH_DIR", tmp_path)

    assert len(summarize.load_records("audio", None, None, None)) == 3
    assert len(summarize.load_records("audio", "2026-10-24", None, None)) == 2
    assert len(summarize.load_records("audio", None, ["a"], None)) == 2
    assert len(summarize.load_records("audio", None, None, [ENV_B])) == 1


def test_load_records_skips_unreadable_file(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "vision"
    task_dir.mkdir()
    (task_dir / "broken.json").write_text("{not json", encoding="utf-8")
    write_record(task_dir, "ok.json", make_record())
    monkeypatch.setattr(summarize, "BENCH_DIR", tmp_path)

    assert len(summarize.load_records("vision", None, None, None)) == 1
    assert "Skipping unreadable file broken.json" in capsys.readouterr().out


def test_load_records_missing_dir_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize, "BENCH_DIR", tmp_path)
    with pytest.raises(SystemExit):
        summarize.load_records("image", None, None, None)


def test_field_stats_percentiles():
    stats = summarize.field_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["n"] == 5
    assert stats["mean"] == 3.0
    assert stats["p50"] == 3.0
    assert stats["p95"] == 4.8
    assert (stats["min"], stats["max"]) == (1.0, 5.0)
    assert summarize.field_stats([2.0])["stdev"] == 0.0


def test_derived_throughput_and_energy():
    batch = summarize.with_derived({"requests": 4.0, "wall_s": 2.0, "gpu.energy_wh": 0.04})
    assert batch["derived.requests_per_hour"] == 7200.0
    assert batch["derived.energy_wh_per_request"] == 0.01
    single = summarize.with_derived({"latency_s": 10.0})
    assert single["derived.requests_per_hour"] == 360.0
    assert "derived.energy_wh_per_request" not in single
    load = summarize.with_derived({"achieved_rps": 0.5, "wh_per_request": 0.02, "wall_s": 9.0})
    assert load["derived.requests_per_hour"] == 1800.0
    assert load["derived.energy_wh_per_request"] == 0.02


def test_json_groups_by_tag_and_env_with_headline():
    ok = {"latency_s": 2.0, "requests": 2, "wall_s": 2.0}
    records = [
        make_record(tag="c2", metrics=ok, gpu={"energy_wh": 0.02, "samples": 3}) | {"task": "audio"},
        make_record(tag="c2", metrics=ok, gpu={"energy_wh": 0.04, "samples": 3}) | {"task": "audio"},
        make_record(tag="c2", error="boom") | {"task": "audio"},
        make_record(tag="c2", metrics=ok, label=ENV_B) | {"task": "audio"},
    ]
    groups = summarize.json_groups(records, None)
    assert [(g["tag"], g["env"], g["runs_ok"], g["runs_total"]) for g in groups] == [
        ("c2", ENV_A, 2, 3), ("c2", ENV_B, 1, 1)]
    head = groups[0]["headline"]
    assert head["requests_per_hour"] == 3600.0
    assert head["latency_p50_s"] == 2.0
    assert head["energy_wh_per_request"] == 0.015


def test_json_headline_uses_k6_percentiles():
    records = [make_record(tag="k6", metrics={"achieved_rps": 1.0, "latency_p50_s": 0.5,
                                              "latency_p95_s": 1.5, "latency_s": 9.0}) | {"task": "load"}]
    head = summarize.json_groups(records, ["latency"])[0]["headline"]
    assert (head["latency_p50_s"], head["latency_p95_s"]) == (0.5, 1.5)


def test_main_json_to_file(tmp_path, monkeypatch, capsys):
    task_dir = tmp_path / "audio"
    task_dir.mkdir()
    write_record(task_dir, "1.json", make_record(metrics={"latency_s": 1.0}) | {"task": "audio"})
    monkeypatch.setattr(summarize, "BENCH_DIR", tmp_path)
    out = tmp_path / "out" / "summary.json"
    summarize.main(["all", "--format", "json", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["groups"][0]["headline"]["requests_per_hour"] == 3600.0
    assert "1 records" in capsys.readouterr().out
