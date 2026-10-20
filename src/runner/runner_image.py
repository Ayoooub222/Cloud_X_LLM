"""Generate one image with sd-server and save the benchmark record as JSON.

Example:
    python -m src.runner.runner_image "a red fox in the snow" --seed 42 --size 768 --tag testA
"""
import argparse
import os
import sys
import time
from pathlib import Path

from src.bench.recorder import BenchRecord, GpuSampler
from src.bench.sdcpp_logs import sdcpp_stage_times
from src.clients.image_client import ImageGenRequest, SdCppImageClient

DEFAULT_MODEL_FILE = "qwen-image-2.1-turbo-Q4_K.gguf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate one image with sd-server and save the run as JSON")
    p.add_argument("prompt", help="use quotes if it contains special characters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=768, help="image side in px (multiple of 64)")
    p.add_argument("--tiling", action="store_true", help="enable VAE tiling (less VRAM, slower)")
    p.add_argument("--tag", default=None, help="run label (warmup, testA, ...)")
    p.add_argument("--out", type=Path, default=Path("outputs/image/"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    request = ImageGenRequest(prompt=args.prompt, seed=args.seed,
                              width=args.size, height=args.size, vae_tiling=args.tiling)
    record = BenchRecord(
        task="image", backend="sdcpp", tag=args.tag,
        model=Path(os.getenv("IMAGE_DIFFUSION", DEFAULT_MODEL_FILE)).stem,
        params={**request.model_dump(exclude={"sigmas"}), "steps": request.steps},
    )

    with SdCppImageClient() as client:
        if not client.health():
            print("sd-server is unreachable (is the 'image' profile running?)", file=sys.stderr)
            return 1

        sampler = GpuSampler()
        start_unix = time.time()  # used to read only this run's logs
        try:
            with sampler:
                result = client.generate(request)
            record.metrics = {
                "latency_s": round(result.latency_s, 2),
                "gen_time_s": result.gen_time_s,
                "queue_time_s": result.queue_time_s,
                "images": len(result.images_b64),
            }
            record.metrics.update(sdcpp_stage_times(start_unix))
            saved = result.save(args.out, stem=f"img_{record.run_id}")
            record.outputs = [str(path) for path in saved]
        except Exception as e:  # a failure is a benchmark result too: record it
            record.status = "error"
            record.error = f"{type(e).__name__}: {e}"[:500]
        record.gpu = sampler.stats()

    path = record.save()
    m = record.metrics
    print(f"[{record.status}] latency={m.get('latency_s')}s gen={m.get('gen_time_s')}s "
          f"vram_peak={record.gpu.vram_peak_mib}MiB power={record.gpu.power_avg_w}W "
          f"energy={record.gpu.energy_wh}Wh")
    print(f"  stages: encode={m.get('encode_s')}s load={m.get('weights_load_s')}s "
          f"sampling={m.get('sampling_s')}s vae={m.get('vae_decode_s')}s cpu_share={m.get('cpu_share_pct')}%")
    print(f"-> {path}")
    return 0 if record.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
