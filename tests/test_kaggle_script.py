"""The checksums hard-coded in scripts/kaggle_t4_bench.sh match the versioned inputs."""
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "scripts" / "kaggle_t4_bench.sh").read_text(encoding="utf-8")


def script_var(name: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"', SCRIPT, re.MULTILINE)
    assert match, f"{name} not found in the Kaggle script"
    return match.group(1)


def test_input_checksums_match_the_files():
    for file_var, sha_var in (("VISION_PDF", "VISION_PDF_SHA256"), ("AUDIO_FILE", "AUDIO_FILE_SHA256")):
        data = (ROOT / script_var(file_var)).read_bytes()
        assert hashlib.sha256(data).hexdigest() == script_var(sha_var), file_var


def test_audio_reference_exists():
    assert (ROOT / script_var("AUDIO_REFERENCE")).is_file()


def test_audio_decoding_versions_match_the_audio_image():
    """The Kaggle server venv decodes audio with the same libraries as docker/audio.Dockerfile."""
    dockerfile = (ROOT / "docker" / "audio.Dockerfile").read_text(encoding="utf-8")
    for name in ("VLLM_VERSION", "LIBROSA_VERSION", "SOUNDFILE_VERSION"):
        match = re.search(rf"^ARG {name}=(\S+)$", dockerfile, re.MULTILINE)
        assert match, f"{name} not found in docker/audio.Dockerfile"
        assert script_var(name) == match.group(1), name
    assert '"vllm[audio]==$VLLM_VERSION"' in SCRIPT


def test_servers_start_without_api_key():
    """vllm serve reads VLLM_API_KEY and then rejects requests without it (HTTP 401 on Kaggle)."""
    serve_lines = [line for line in SCRIPT.splitlines() if "vllm\" serve" in line]
    assert serve_lines, "no vllm serve command in the Kaggle script"
    assert all("env -u VLLM_API_KEY" in line for line in serve_lines)
