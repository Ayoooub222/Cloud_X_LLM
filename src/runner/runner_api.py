"""Quality and latency of hosted APIs on the same inputs as the self-hosted models (free tiers only).

The API keys are read from .env (GROQ_API_KEY, GEMINI_API_KEY) and never printed. Requests are
sequential and spaced out to stay within the free-tier rate limits. No GPU is sampled: the local
GPU does nothing during these calls. Records use the env labels api-groq and api-gemini.

Examples:
    python -m src.runner.runner_api groq-audio --repeats 3 --tag api-c1
    python -m src.runner.runner_api gemini-vision --repeats 3 --tag api-c1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from src.bench.recorder import PROJECT_ROOT, BenchRecord, project_path, show
from src.clients.audio_client import AudioClient, TranscriptionRequest, audio_info
from src.runner.runner_audio import quality_metrics, save_transcripts
from src.runner.runner_vision import accuracy_metrics, save_outputs
from src.utils.loader import Loader

PAUSE_S = 6.0  # between requests: under 10 requests per minute
AUDIO_OUT_DIR = Path("outputs/audio")
AUDIO_FILE = PROJECT_ROOT / "inputs" / "audio" / "librispeech_1272-141231_66s.flac"
AUDIO_REFERENCE = PROJECT_ROOT / "inputs" / "audio" / "librispeech_1272-141231_66s.txt"
VISION_PDF = PROJECT_ROOT / "inputs" / "vision" / "devis_fictif.pdf"
VISION_EXPECTED = PROJECT_ROOT / "inputs" / "vision" / "devis_fictif.expected.json"
VISION_SCHEMA = "conf/schemas/extraction_page.json"
VISION_DPI = 100
SELF_HOSTED_VISION_MODEL = "Qwen3-VL-2B-Instruct"  # sampling settings reused for a fair comparison
# Gemini rejects (HTTP 400) a maxItems >= 30 on the 7-field quote lines: a decoding-grammar size
# limit. Only the constraint sent to Gemini drops maxItems; the prompt is unchanged and the answer
# is still validated against the full schema.
API_SCHEMA_DROPPED_KEYS = {"maxItems"}
# Standard OpenAI sampling fields only: top_k, min_p and repetition_penalty are vLLM extensions.
API_SAMPLING_KEYS = ("temperature", "top_p", "max_completion_tokens")


@dataclass(frozen=True)
class Provider:
    task: str
    backend: str
    env_label: str
    base_url: str
    model: str
    key_var: str


PROVIDERS = {
    "groq-audio": Provider("audio", "groq", "api-groq", "https://api.groq.com/openai",
                           "whisper-large-v3-turbo", "GROQ_API_KEY"),
    "gemini-vision": Provider("vision", "gemini", "api-gemini",
                              "https://generativelanguage.googleapis.com/v1beta/openai/",
                              "gemini-3.1-flash-lite", "GEMINI_API_KEY"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quality and latency of a hosted API on the reference inputs")
    p.add_argument("provider", choices=sorted(PROVIDERS))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tag", default="api-c1")
    p.add_argument("--model", help="override the provider model")
    return p.parse_args(argv)


def without_keys(node, keys: set[str]):
    """Copy of a JSON schema without the given keywords, at every depth."""
    if isinstance(node, dict):
        return {k: without_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [without_keys(v, keys) for v in node]
    return node


def new_record(provider: Provider, model: str, tag: str, params: dict) -> BenchRecord:
    record = BenchRecord(task=provider.task, backend=provider.backend, model=model, tag=tag, params=params)
    record.env["label"] = provider.env_label
    return record


def fail(record: BenchRecord, error: Exception) -> None:
    record.status = "error"
    record.error = f"{type(error).__name__}: {error}"[:500]


def run_audio(provider: Provider, model: str, key: str, args: argparse.Namespace) -> bool:
    reference = AUDIO_REFERENCE.read_text(encoding="utf-8")
    info = audio_info(AUDIO_FILE)
    request = TranscriptionRequest(audio_path=AUDIO_FILE, model=model, language="en")
    params = {"audio_file": project_path(AUDIO_FILE), "reference_file": project_path(AUDIO_REFERENCE),
              "language": "en", "response_format": request.response_format, "concurrency": 1}
    all_ok = True
    with AudioClient(base_url=provider.base_url, api_key=key) as client:
        for i in range(args.repeats):
            record = new_record(provider, model, args.tag, params)
            try:
                result = client.transcribe(request)
                record.metrics = {"audio_duration_s": info["duration_s"], "requests": 1,
                                  "latency_s": round(result.latency_s, 3),
                                  "rtf": round(result.latency_s / info["duration_s"], 4)}
                record.metrics |= quality_metrics([result], reference)
                record.outputs = [str(save_transcripts([result], AUDIO_OUT_DIR, record.run_id))]
            except Exception as e:  # an API error is a result too
                fail(record, e)
            all_ok &= report(record, i, args.repeats)
    return all_ok


def run_vision(provider: Provider, model: str, key: str, args: argparse.Namespace) -> bool:
    from src.clients.vision_client import VisionClient, pdf_to_pages  # heavy imports, vision only

    client = VisionClient(api_key=key, base_url=provider.base_url, schema_file=VISION_SCHEMA)
    client.connect(model)
    sampling = Loader().load_conf()["hyperparameter"][SELF_HOSTED_VISION_MODEL]
    client.hyperparameters = {k: sampling[k] for k in API_SAMPLING_KEYS if k in sampling}
    api_schema = without_keys(client.schema, API_SCHEMA_DROPPED_KEYS)
    client.response_format = {"type": "json_schema",
                              "json_schema": {"name": client.schema_name, "schema": api_schema, "strict": True}}
    page = pdf_to_pages(VISION_PDF, dpi=VISION_DPI)[0]
    expected = json.loads(VISION_EXPECTED.read_text(encoding="utf-8"))
    expected = expected[0] if isinstance(expected, list) else expected
    params = {"pdf_path": project_path(VISION_PDF), "dpi": VISION_DPI, "page": 0, "concurrency": 1,
              "schema": VISION_SCHEMA, "api_schema_dropped": sorted(API_SCHEMA_DROPPED_KEYS),
              "sampling": client.hyperparameters}
    all_ok = True
    for i in range(args.repeats):
        record = new_record(provider, model, args.tag, params)
        try:
            start = time.perf_counter()
            response = client.complete(page["b64"])
            latency = time.perf_counter() - start
            data = json.loads(response.choices[0].message.content)
            jsonschema.validate(data, client.schema)
            usage = response.usage
            record.metrics = {"requests": 1, "latency_s": round(latency, 3), "structured_ok": 1,
                              "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
            record.metrics |= accuracy_metrics(expected, [data])
            record.outputs = [save_outputs([data], record.run_id)]
        except Exception as e:
            fail(record, e)
            record.metrics.setdefault("structured_ok", 0)
        all_ok &= report(record, i, args.repeats)
    return all_ok


def report(record: BenchRecord, index: int, total: int) -> bool:
    path = record.save()
    m = record.metrics
    print(f"[{record.status}] {record.env['label']} {record.tag} {index + 1}/{total}: "
          f"latency={show(m.get('latency_s'))}s wer={show(m.get('wer'))} "
          f"accuracy={show(m.get('field_accuracy'))} prompt_tok={show(m.get('prompt_tokens'))} "
          f"completion_tok={show(m.get('completion_tokens'))} -> {Path(path).name}", flush=True)
    if record.error:
        print(f"  error: {record.error}", flush=True)
    if index + 1 < total:
        time.sleep(PAUSE_S)
    return record.status == "ok"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provider = PROVIDERS[args.provider]
    key = Loader().env_value(provider.key_var)
    if not key:
        print(f"{provider.key_var} is not set (.env)", file=sys.stderr)
        return 1
    model = args.model or provider.model
    run = run_audio if provider.task == "audio" else run_vision
    return 0 if run(provider, model, key, args) else 2


if __name__ == "__main__":
    sys.exit(main())
