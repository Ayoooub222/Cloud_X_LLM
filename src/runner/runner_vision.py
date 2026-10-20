"""Extract structured JSON from each page of a PDF with vLLM and save one benchmark record per page.

Example:
    python -m src.runner.runner_vision doc.pdf schema.json --dpi 100 --tag testA
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jsonschema

from src.bench.recorder import BenchRecord, GpuSampler
from src.clients.vision_client import VisionClient, pdf_to_pages
from src.utils.loader import Loader

OUT_DIR = Path("outputs/vision")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision with vLLM (structured output), one benchmark JSON per page")
    p.add_argument("pdf", type=Path)
    p.add_argument("schema", help="JSON schema file, relative to the project root")
    p.add_argument("--dpi", type=int, default=100, help="render resolution (changes prompt_tokens)")
    p.add_argument("--tag", default=None, help="run label (warmup, testA, ...)")
    return p.parse_args()


def usage_metrics(usage, latency_s: float) -> dict:
    """Token counts and effective throughput. tokens_per_s includes the prefill."""
    completion = getattr(usage, "completion_tokens", None)
    return {
        "latency_s": latency_s,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": completion,
        "total_tokens": getattr(usage, "total_tokens", None),
        "tokens_per_s": round(completion / latency_s, 2) if completion and latency_s else None,
    }


def save_output(data: dict, run_id: str) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"vision_{run_id}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def run_page(client: VisionClient, page: dict, params: dict, tag: str | None) -> bool:
    """Process one page and save its record. Return True if it succeeded."""
    record = BenchRecord(task="vision", backend="vllm", model=client.model, tag=tag, params=params)
    sampler = GpuSampler()
    try:
        with sampler:
            start = time.perf_counter()
            response = client.complete(page["b64"])
            latency_s = round(time.perf_counter() - start, 3)
        record.metrics = {**usage_metrics(response.usage, latency_s), "structured_ok": 0}
        data = json.loads(response.choices[0].message.content)
        jsonschema.validate(data, client.schema)  # full check, nested objects included
        record.metrics["structured_ok"] = 1
        record.outputs = [save_output(data, record.run_id)]
    except Exception as e:  # a failure is a benchmark result too: record it
        record.status = "error"
        record.error = f"{type(e).__name__}: {e}"[:500]
        record.metrics.setdefault("structured_ok", 0)
    record.gpu = sampler.stats()

    path = record.save()
    m = record.metrics
    print(f"[{record.status}] page {params['page']} latency={m.get('latency_s')}s "
          f"tok/s={m.get('tokens_per_s')} prompt_tok={m.get('prompt_tokens')} "
          f"vram_peak={record.gpu.vram_peak_mib}MiB energy={record.gpu.energy_wh}Wh -> {path}")
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
    hp = client.hyperparameters
    common_params = {
        "pdf_path": str(args.pdf),
        "pdf_size_bytes": args.pdf.stat().st_size,
        "pages_total": len(pages),
        "dpi": args.dpi,
        "schema": client.schema_name,
        "prompt_chars": len(client.default_prompt),
        "max_tokens": hp.get("max_completion_tokens") or hp.get("max_tokens"),
        "temperature": hp.get("temperature"),
    }

    all_ok = bool(pages)
    for i, page in enumerate(pages):
        params = {**common_params, "page": i, "width": page["width"], "height": page["height"]}
        all_ok &= run_page(client, page, params, args.tag)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
