"""nvidia-smi command construction (GPU selection via CUDA_VISIBLE_DEVICES)."""
from src.bench.recorder import GPU_ID_VAR, nvidia_smi_cmd

ID_PREFIX = "--id="


def id_args(cmd: list[str]) -> list[str]:
    return [arg for arg in cmd if arg.startswith(ID_PREFIX)]


def test_no_id_when_variable_unset(monkeypatch):
    monkeypatch.delenv(GPU_ID_VAR, raising=False)
    assert id_args(nvidia_smi_cmd()) == []


def test_no_id_when_variable_empty(monkeypatch):
    monkeypatch.setenv(GPU_ID_VAR, "")
    assert id_args(nvidia_smi_cmd()) == []


def test_first_visible_device_is_used(monkeypatch):
    monkeypatch.setenv(GPU_ID_VAR, "1, 0")
    assert id_args(nvidia_smi_cmd()) == ["--id=1"]


def test_uuid_is_passed_through(monkeypatch):
    monkeypatch.setenv(GPU_ID_VAR, "GPU-1234abcd")
    assert id_args(nvidia_smi_cmd()) == ["--id=GPU-1234abcd"]
