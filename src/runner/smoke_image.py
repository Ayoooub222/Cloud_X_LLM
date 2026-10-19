"""python -m scripts.smoke_image 'a red fox in the snow' --seed 42 --size 768 --tag testA"""
import argparse
import os
import sys
import time
from pathlib import Path

from src.bench.recorder import BenchRecord, GpuSampler
from src.clients.image_client import ImageGenRequest, SdCppImageClient


def main() -> int:
    p = argparse.ArgumentParser(description="Génère une image via sd-server et enregistre le run en JSON")
    p.add_argument("prompt", help="entre guillemets simples si caractères spéciaux")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=768, help="côté en px (multiple de 64)")
    p.add_argument("--tiling", action="store_true", help="active le VAE tiling (si OOM)")
    p.add_argument("--tag", default=None, help="étiquette du run (warmup, testA, ...)")
    p.add_argument("--out", type=Path, default=Path("outputs/image/"))
    a = p.parse_args()

    req = ImageGenRequest(prompt=a.prompt, seed=a.seed, width=a.size, height=a.size, vae_tiling=a.tiling)
    rec = BenchRecord(
        task="image", backend="sdcpp", tag=a.tag,
        model=Path(os.getenv("IMAGE_DIFFUSION", "qwen-image-2.1-turbo-Q4_K.gguf")).stem,
        params={**req.model_dump(exclude={"sigmas"}), "steps": req.steps},
    )

    with SdCppImageClient() as c:
        if not c.health():
            print("sd-server injoignable (profil image lancé ?)", file=sys.stderr)
            return 1
        sampler = GpuSampler()
        try:
            with sampler:
                res = c.generate(req)
            rec.metrics = {"latency_s": round(res.latency_s, 2), "gen_time_s": res.gen_time_s,
                           "queue_time_s": res.queue_time_s, "images": len(res.images_b64)}
            rec.outputs = [str(x) for x in res.save(a.out, f"img_{rec.run_id}")]
        except Exception as e:  # un échec est un résultat de benchmark : on l'enregistre
            rec.status, rec.error = "error", f"{type(e).__name__}: {e}"[:500]
        rec.gpu = sampler.stats()

    path = rec.save()
    print(f"[{rec.status}] latency={rec.metrics.get('latency_s')}s gen={rec.metrics.get('gen_time_s')}s "
          f"vram_peak={rec.gpu.vram_peak_mib}MiB power={rec.gpu.power_avg_w}W energy={rec.gpu.energy_wh}Wh")
    print(f"-> {path}")
    return 0 if rec.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
