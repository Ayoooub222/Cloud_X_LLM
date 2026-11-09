"""Console display of missing metrics and project-relative paths in records."""
from pathlib import Path

from src.bench.mlflow_log import resolve_output
from src.bench.recorder import MISSING, project_path, show


def test_show_missing_value():
    assert show(None) == MISSING
    assert show(0) == "0"
    assert show(1.5) == "1.5"


def test_project_path_inside_root(tmp_path):
    inside = tmp_path / "outputs" / "audio" / "a.json"
    assert project_path(inside, root=tmp_path) == "outputs/audio/a.json"


def test_project_path_outside_root(tmp_path):
    assert project_path("/kaggle/working/outputs/a.json", root=tmp_path) == "/kaggle/working/outputs/a.json"


def test_resolve_output_relative(tmp_path):
    target = tmp_path / "outputs" / "audio" / "a.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert resolve_output("outputs/audio/a.json", root=tmp_path) == target


def test_resolve_output_remaps_foreign_absolute(tmp_path):
    target = tmp_path / "outputs" / "audio" / "a.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert resolve_output("/kaggle/working/outputs/audio/a.json", root=tmp_path) == target


def test_resolve_output_missing(tmp_path):
    assert resolve_output("/kaggle/working/outputs/audio/none.json", root=tmp_path) is None
    assert resolve_output(str(Path("elsewhere.json")), root=tmp_path) is None
