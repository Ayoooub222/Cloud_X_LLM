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
