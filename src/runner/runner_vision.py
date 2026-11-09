"""Extract structured JSON from each page of a PDF with vLLM and save one benchmark record per page.

With --concurrency N, each page is sent N times at once and the record describes the whole batch
(pages_per_hour = GPU cost view, latency_s = user view).

Example:
    python -m src.runner.runner_vision doc.pdf schema.json --dpi 100 --tag testA
    python -m src.runner.runner_vision inputs/vision/devis_fictif.pdf conf/schemas/extraction_page.json \
        --expected inputs/vision/devis_fictif.expected.json --concurrency 4 --tag conc-c4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonschema

from src.bench.extraction_score import score
from src.bench.recorder import BenchRecord, GpuSampler, project_path, show
from src.bench.vllm_metrics import MetricsSampler
from src.clients.vision_client import VisionClient, pdf_to_pages
from src.utils.loader import Loader

OUT_DIR = Path("outputs/vision")
METRICS_INTERVAL_S = 0.5  # vLLM scheduler state (/metrics) during each batch
SECONDS_PER_HOUR = 3600


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision with vLLM (structured output), one benchmark JSON per page")
    p.add_argument("pdf", type=Path)
    p.add_argument("schema", help="JSON schema file, relative to the project root")
    p.add_argument("--dpi", type=int, default=100, help="render resolution (changes prompt_tokens)")
    p.add_argument("--tag", default=None, help="run label (warmup, testA, ...)")
    p.add_argument("--expected", type=Path, default=None,
                   help="ground truth JSON (one object per page): adds field_accuracy to the metrics")
    p.add_argument("--concurrency", type=int, default=1,
                   help="number of identical requests sent at the same time for each page")
    return p.parse_args()


def complete_page(client: VisionClient, image_b64: str) -> tuple[float, object, dict]:
    """One request. Return (latency_s, usage, extracted data checked against the schema)."""
    start = time.perf_counter()
    response = client.complete(image_b64)
    latency_s = time.perf_counter() - start
    data = json.loads(response.choices[0].message.content)
    jsonschema.validate(data, client.schema)  # full check, nested objects included
    return latency_s, response.usage, data


def run_batch(client: VisionClient, image_b64: str, concurrency: int) -> tuple[list[tuple], float]:
    """Send `concurrency` requests at once. Return the results and the wall time of the batch."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(complete_page, client, image_b64) for _ in range(concurrency)]
        results = [future.result() for future in futures]  # re-raises the first error
    return results, time.perf_counter() - start


def batch_metrics(results: list[tuple], wall_s: float) -> dict:
    """Token counts and throughputs. tokens_per_s is per request and includes the prefill."""
    latencies = [latency for latency, _, _ in results]
    completions = [getattr(usage, "completion_tokens", None) or 0 for _, usage, _ in results]
    usage = results[0][1]
    latency_avg = sum(latencies) / len(latencies)
    completion_avg = sum(completions) / len(completions)
    return {
        "requests": len(results),
        "latency_s": round(latency_avg, 3),
        "latency_max_s": round(max(latencies), 3),
        "wall_s": round(wall_s, 3),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": round(completion_avg, 1),
        "total_tokens": getattr(usage, "total_tokens", None),
        "tokens_per_s": round(completion_avg / latency_avg, 2) if latency_avg else None,
        "batch_tokens_per_s": round(sum(completions) / wall_s, 2) if wall_s else None,
        "pages_per_hour": round(len(results) * SECONDS_PER_HOUR / wall_s, 1) if wall_s else None,
    }


def accuracy_metrics(expected: dict, outputs: list[dict]) -> dict:
    """Mean and worst field accuracy over the outputs of the batch."""
    scores = [score(expected, data) for data in outputs]
    accuracies = [s["field_accuracy"] for s in scores]
    return {
        "fields_total": scores[0]["fields_total"],
        "field_accuracy": round(sum(accuracies) / len(accuracies), 4),
        "field_accuracy_min": min(accuracies),
    }


def save_outputs(outputs: list[dict], run_id: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"vision_{run_id}.json"
    data = outputs[0] if len(outputs) == 1 else outputs
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def load_expected(path: Path | None) -> list[dict]:
    """Per-page ground truth, or an empty list when no file is given."""
    return json.loads(path.read_text(encoding="utf-8")) if path else []


def server_root(client: VisionClient) -> str:
    """vLLM base URL without the OpenAI /v1 suffix (where /metrics lives)."""
    return str(client.openai.base_url).rstrip("/").removesuffix("/v1")


def run_page(client: VisionClient, page: dict, params: dict, tag: str | None, expected: dict | None) -> bool:
    """Process one page (one batch of requests) and save its record. Return True if it succeeded."""
    record = BenchRecord(task="vision", backend="vllm", model=client.model, tag=tag, params=params)
    sampler = GpuSampler()
    scheduler = MetricsSampler(server_root(client), METRICS_INTERVAL_S)
    try:
        with sampler, scheduler:
            results, wall_s = run_batch(client, page["b64"], params["concurrency"])
        outputs = [data for _, _, data in results]
        record.metrics = {**batch_metrics(results, wall_s), "structured_ok": 1} | scheduler.summary()
        if expected is not None:
            record.metrics |= accuracy_metrics(expected, outputs)
        record.outputs = [save_outputs(outputs, record.run_id)]
    except Exception as e:  # a failure is a benchmark result too: record it
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"[:500]
        record.metrics.setdefault("structured_ok", 0)
    record.gpu = sampler.stats()

    path = record.save()
    m = record.metrics
    print(f"[{record.status}] page {params['page']} conc={params['concurrency']} latency={show(m.get('latency_s'))}s "
          f"wall={show(m.get('wall_s'))}s pages/h={show(m.get('pages_per_hour'))} "
          f"prompt_tok={show(m.get('prompt_tokens'))} "
          f"completion_tok={show(m.get('completion_tokens'))} accuracy={show(m.get('field_accuracy'))} "
          f"vram_peak={show(record.gpu.vram_peak_mib)}MiB energy={show(record.gpu.energy_wh)}Wh -> {path}")
    return record.status == "ok"


def main() -> int:
    args = parse_args()
    client = VisionClient(
        api_key=Loader().load_env(),
        base_url=Loader().load_conf()["url_api"],
        schema_file=args.schema,
    )
    try:
        client.connect()
    except Exception as e:
        print(f"vLLM unreachable or invalid config: {e}", file=sys.stderr)
        return 1

    pages = pdf_to_pages(args.pdf, dpi=args.dpi)
    expected = load_expected(args.expected)
    hp = client.hyperparameters
    common_params = {
        "pdf_path": project_path(args.pdf),
        "pdf_size_bytes": args.pdf.stat().st_size,
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "pages_total": len(pages),
        "dpi": args.dpi,
        "schema": client.schema_name,
        "prompt_chars": len(client.default_prompt),
        "max_tokens": hp.get("max_completion_tokens") or hp.get("max_tokens"),
        "temperature": hp.get("temperature"),
        "concurrency": args.concurrency,
    }

    all_ok = bool(pages)
    for i, page in enumerate(pages):
        params = {**common_params, "page": i, "width": page["width"], "height": page["height"]}
        page_expected = expected[i] if i < len(expected) else None
        all_ok &= run_page(client, page, params, args.tag, page_expected)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
