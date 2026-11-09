"""scripts/import_kaggle.sh: copies new records, never overwrites, refuses unsafe archives."""
import io
import os
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "import_kaggle.sh"
RECORD = "out/results/benchmark/audio/2026-01-01_00-00-00_audio_abc123.json"


def make_archive(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


def run(archive: Path, extract: Path, bench: Path) -> subprocess.CompletedProcess:
    env = os.environ | {"BENCH_DIR": str(bench)}
    return subprocess.run(["bash", str(SCRIPT), str(archive), str(extract)], capture_output=True, text=True,
                          env=env, check=False)


def test_copies_new_records_and_keeps_existing_ones(tmp_path):
    archive = make_archive(tmp_path / "a.tar.gz", {RECORD: b'{"new": true}', "out/gpu.csv": b"T4"})
    bench = tmp_path / "bench"
    existing = bench / "audio" / "2026-01-01_00-00-00_audio_abc123.json"
    first = run(archive, tmp_path / "x1", bench)
    assert first.returncode == 0, first.stderr
    assert "copied: 1" in first.stdout
    existing.write_text('{"edited": true}')
    second = run(archive, tmp_path / "x2", bench)
    assert second.returncode == 0, second.stderr
    assert "copied: 0" in second.stdout
    assert existing.read_text() == '{"edited": true}'


def test_refuses_paths_outside_the_extract_folder(tmp_path):
    archive = make_archive(tmp_path / "bad.tar.gz", {"../escape.json": b"{}"})
    result = run(archive, tmp_path / "x", tmp_path / "bench")
    assert result.returncode != 0
    assert not (tmp_path / "escape.json").exists()
    assert not (tmp_path / "x").exists()


def test_refuses_an_existing_extract_folder(tmp_path):
    archive = make_archive(tmp_path / "a.tar.gz", {RECORD: b"{}"})
    (tmp_path / "x").mkdir()
    assert run(archive, tmp_path / "x", tmp_path / "bench").returncode != 0
