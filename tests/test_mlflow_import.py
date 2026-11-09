"""Unit tests for the MLflow import selection (no MLflow server needed)."""
from src.bench.recorder import BenchRecord
from src.runner import mlflow_import


def write_record(root, task, run_id, timestamp):
    record = BenchRecord(task=task, backend="vllm", model="m", run_id=run_id,
                         timestamp=timestamp, env={"label": "ci-test"})
    folder = root / task
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{run_id}.json").write_text(record.model_dump_json(), encoding="utf-8")


def make_tree(root):
    write_record(root, "audio", "a1", "2026-10-23 10:00:00")
    write_record(root, "audio", "a2", "2026-10-24 10:00:00")
    write_record(root, "image", "i1", "2026-10-24 11:00:00")
    sessions = root / "sessions"
    sessions.mkdir()
    (sessions / "state.json").write_text('{"gpu": "x"}', encoding="utf-8")


def test_select_all_skips_non_records(tmp_path, capsys):
    make_tree(tmp_path)
    selected = mlflow_import.select_records(tmp_path, None, None)
    assert [record.run_id for _, record in selected] == ["a1", "a2", "i1"]
    assert "Skipping state.json" in capsys.readouterr().err


def test_select_by_task_and_since(tmp_path):
    make_tree(tmp_path)
    assert len(mlflow_import.select_records(tmp_path, ["audio"], None)) == 2
    assert len(mlflow_import.select_records(tmp_path, ["audio"], "2026-10-24")) == 1


def test_dry_run_needs_no_server(tmp_path, monkeypatch, capsys):
    make_tree(tmp_path)
    monkeypatch.setattr(mlflow_import, "RESULTS_DIR", tmp_path)
    assert mlflow_import.main(["--dry-run", "--task", "image"]) == 0
    assert "1 records" in capsys.readouterr().out
