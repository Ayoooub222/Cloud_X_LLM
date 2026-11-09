"""runner_audio exit codes and console output (no server needed)."""
import sys

import numpy as np
import soundfile

from src.bench.recorder import BenchRecord
from src.runner import runner_audio


class FailingClient:
    def __init__(self, healthy: bool = True):
        self.healthy = healthy

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def health(self) -> bool:
        return self.healthy

    def transcribe(self, request):
        raise RuntimeError("HTTP 500: boom")


def make_wav(tmp_path):
    path = tmp_path / "tone.wav"
    soundfile.write(str(path), np.zeros(1600, dtype="float32"), 16000)
    return path


def run_main(monkeypatch, tmp_path, client):
    def fake_client(api_key=None):
        client.api_key = api_key
        return client

    monkeypatch.setattr(runner_audio, "AudioClient", fake_client)
    monkeypatch.setattr(BenchRecord, "save", lambda self, root=None: tmp_path / "record.json")
    monkeypatch.setattr(sys, "argv", ["runner_audio", str(make_wav(tmp_path)), "--out", str(tmp_path),
                                      "--metrics-interval", "0"])
    return runner_audio.main()


def test_failed_request_exits_nonzero_and_prints_na(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, FailingClient()) != 0
    out = capsys.readouterr().out
    assert "[error]" in out
    assert "latency=n/a" in out
    assert "None" not in out


def test_unreachable_server_exits_nonzero(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path, FailingClient(healthy=False)) != 0


def test_api_key_is_passed_to_the_client(monkeypatch, tmp_path):
    """A server started with VLLM_API_KEY answers 401 to requests without it."""
    monkeypatch.setenv("VLLM_API_KEY", "test-key")
    client = FailingClient(healthy=False)
    run_main(monkeypatch, tmp_path, client)
    assert client.api_key == "test-key"


def test_missing_file_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["runner_audio", str(tmp_path / "missing.wav")])
    assert runner_audio.main() != 0


def test_quality_metrics_mean_max_and_counts():
    results = [runner_audio.TranscriptionResult(text="A man said to the universe.", latency_s=1.0),
               runner_audio.TranscriptionResult(text="A man said to the", latency_s=1.0)]
    m = runner_audio.quality_metrics(results, "A MAN SAID TO THE UNIVERSE")
    assert m["reference_words"] == 6
    assert m["wer_max"] == round(1 / 6, 4)
    assert m["wer"] == round(1 / 12, 4)
    assert (m["substitutions"], m["deletions"], m["insertions"]) == (0, 0, 0)
