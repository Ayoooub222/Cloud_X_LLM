"""Unit tests for the config and .env loader."""
import json

from src.utils.loader import API_KEY_VAR, Loader


def test_load_conf_reads_json(tmp_path):
    conf = tmp_path / "app.conf"
    conf.write_text(json.dumps({"model": "qwen"}), encoding="utf-8")
    assert Loader().load_conf(conf) == {"model": "qwen"}


def test_load_env_without_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(API_KEY_VAR, "test-key")
    assert Loader().load_env() == "test-key"
