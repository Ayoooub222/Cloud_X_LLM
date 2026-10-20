"""HTTP client for the vLLM OpenAI-compatible transcription endpoint (Whisper)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import soundfile
from pydantic import BaseModel

DEFAULT_BASE_URL = os.getenv("AUDIO_BASE_URL", "http://localhost:8001")
DEFAULT_MODEL = os.getenv("AUDIO_MODEL", "openai/whisper-large-v3-turbo")
TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
HEALTH_PATH = "/health"
REQUEST_TIMEOUT_S = 600.0


def audio_info(path: Path) -> dict:
    """Exact duration and format of an audio file (the server rounds duration to 1 s)."""
    info = soundfile.info(str(path))
    return {"duration_s": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels}


class TranscriptionRequest(BaseModel):
    audio_path: Path
    model: str = DEFAULT_MODEL
    language: str | None = None  # None = auto-detect
    temperature: float = 0.0
    response_format: str = "verbose_json"  # gives language and segments

    def form_fields(self) -> dict:
        """Multipart form fields, without the file."""
        fields = {"model": self.model,
                  "temperature": str(self.temperature),
                  "response_format": self.response_format}
        if self.language:
            fields["language"] = self.language
        return fields


class TranscriptionResult(BaseModel):
    text: str
    latency_s: float
    server_duration_s: float | None = None
    language: str | None = None
    segments: int = 0


def raise_for_error(response: httpx.Response) -> None:
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")


def server_duration(body: dict) -> float | None:
    """Audio duration reported by the server (verbose_json or json format)."""
    if "duration" in body:
        return body["duration"]
    return body.get("usage", {}).get("seconds")


class AudioClient:
    """Usage:
        with AudioClient() as client:
            result = client.transcribe(TranscriptionRequest(audio_path=Path("a.flac")))
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout_s: float = REQUEST_TIMEOUT_S):
        self.http = httpx.Client(base_url=base_url, timeout=timeout_s)

    def __enter__(self) -> "AudioClient":
        return self

    def __exit__(self, *exc) -> None:
        self.http.close()

    def health(self) -> bool:
        try:
            return self.http.get(HEALTH_PATH, timeout=5).status_code == 200
        except httpx.HTTPError:
            return False

    def transcribe(self, request: TranscriptionRequest) -> TranscriptionResult:
        """Send one file. Disk read happens before the timer: latency = network + server only."""
        audio_bytes = request.audio_path.read_bytes()
        files = {"file": (request.audio_path.name, audio_bytes)}
        start = time.perf_counter()
        response = self.http.post(TRANSCRIPTION_PATH, data=request.form_fields(), files=files)
        latency = time.perf_counter() - start
        raise_for_error(response)
        body = response.json()
        return TranscriptionResult(
            text=body.get("text", ""),
            latency_s=round(latency, 3),
            server_duration_s=server_duration(body),
            language=body.get("language"),
            segments=len(body.get("segments") or []),
        )
