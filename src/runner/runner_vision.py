"""python -m src.runner.runner_vision <pdf> <schema.json> [--dpi 100] [--tag xxx]
1 page = 1 requête = 1 JSON dans results/bench/vision/ (créé par BenchRecord.save)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import jsonschema

from src.bench.recorder import BenchRecord, GpuSampler
from src.clients.vllmapiclient import VLLMAPICLIENT
from src.utils.loader import Loader

OUT_DIR = Path("outputs/vision")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision vLLM (structured output) + bench JSON par page")
    p.add_argument("singlepdf", type=Path)
    p.add_argument("schema_response")
    p.add_argument("--dpi", type=int, default=100, help="résolution de rendu (impacte prompt_tokens)")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def usage_metrics(usage, latency_s: float) -> dict[str, Any]:
    pt = getattr(usage, "prompt_tokens", None)       # getattr(None, ..., None) -> None : pas de if
    ct = getattr(usage, "completion_tokens", None)
    return {
        "latency_s": latency_s,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": getattr(usage, "total_tokens", None),
        "tokens_per_s": round(ct / latency_s, 2) if ct and latency_s else None,  # débit effectif (prefill inclus)
    }


def run_one(client: VLLMAPICLIENT, image_b64: str, params: dict, tag: str | None) -> bool:
    rec = BenchRecord(task="vision", backend="vllm", model=client.resolve_model(), tag=tag, params=params)
    sampler = GpuSampler()
    try:
        with sampler:
            t0 = time.perf_counter()
            response = client.complete_vision(image_b64)
            latency_s = round(time.perf_counter() - t0, 3)
        rec.metrics = {**usage_metrics(client.last_usage, latency_s), "structured_ok": 0}
        data = json.loads(response.choices[0].message.content)
        jsonschema.validate(data, client.schema)          # validation complète, objets imbriqués compris
        rec.metrics["structured_ok"] = 1
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"vision_{rec.run_id}.json"
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        rec.outputs = [str(out)]
    except Exception as e:  # un échec est un résultat de benchmark
        rec.status, rec.error = "error", f"{type(e).__name__}: {e}"[:500]
        rec.metrics.setdefault("structured_ok", 0)
    rec.gpu = sampler.stats()
    path = rec.save()
    m = rec.metrics
    print(f"[{rec.status}] p{params['page']} latency={m.get('latency_s')}s tok/s={m.get('tokens_per_s')} "
          f"prompt_tok={m.get('prompt_tokens')} vram_peak={rec.gpu.vram_peak_mib}MiB "
          f"energy={rec.gpu.energy_wh}Wh -> {path}")
    return rec.status == "ok"


def main() -> int:
    args = parse_args()
    client = VLLMAPICLIENT(
        api_key=Loader().load_env(),
        key_env=Loader().load_conf()["url_api"],
        pdf_path=str(args.singlepdf),
        schema_response=args.schema_response,
    )
    try:  # échauffement réseau + config, HORS mesure
        client.resolve_model()
        hp = client.hyperparameters()
    except Exception as e:
        print(f"vLLM injoignable ou config invalide : {e}", file=sys.stderr)
        return 1

    images = client.pdf_to_base64_images(args.singlepdf, dpi=args.dpi)
    common = {
        "pdf_path": str(args.singlepdf),
        "pdf_size_bytes": args.singlepdf.stat().st_size,
        "pages_total": len(images),
        "dpi": args.dpi,
        "schema": client.response_format["json_schema"]["name"],
        "prompt_chars": len(client.default_prompt),
        "max_tokens": hp.get("max_completion_tokens") or hp.get("max_tokens"),
        "temperature": hp.get("temperature"),
    }
    ok_all = bool(images)
    for i, (b64, (w, h)) in enumerate(zip(images, client.last_page_sizes)):
        ok_all &= run_one(client, b64, {**common, "page": i, "width": w, "height": h}, args.tag)
    return 0 if ok_all else 2


if __name__ == "__main__":
    sys.exit(main())
