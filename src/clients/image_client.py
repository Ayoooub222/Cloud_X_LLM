"""Client sd-server (stable-diffusion.cpp) pour Qwen-Image-2.1-Turbo, via l'API native async /sdcpp/v1."""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
import time


import httpx
from pydantic import BaseModel, Field

TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]
_TERMINAL = {"completed", "failed", "cancelled"}


class ImageGenRequest(BaseModel):
    vae_tiling: bool = False
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    width: int = Field(768, ge=256, le=1536, multiple_of=64)
    height: int = Field(768, ge=256, le=1536, multiple_of=64)
    cfg_scale: float = Field(1.0, ge=0.0, le=15.0)  # Turbo = 1.0
    seed: int = -1
    batch_count: int = Field(1, ge=1, le=4)
    sigmas: list[float] = Field(default_factory=lambda: list(TURBO_SIGMAS), min_length=2)

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1  # 9 sigmas -> 8 steps


class ImageGenResult(BaseModel):
    images_b64: list[str]
    latency_s: float                 # vu du client : file + génération + réseau
    gen_time_s: float | None = None  # côté serveur (completed - started) = temps GPU "facturable"
    queue_time_s: float | None = None
    request: ImageGenRequest

    def save(self, out_dir: str | Path, stem: str = "img" , date: str = time.strftime("%Y-%m-%d %H:%M:%S")) -> list[Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, b64 in enumerate(self.images_b64):
            p = out / f"{stem}_{date}_{i}.png"
            p.write_bytes(base64.b64decode(b64))
            paths.append(p)
        return paths


class SdCppImageClient:
    def __init__(self, base_url: str | None = None,
                 timeout_s: float = 600.0, poll_interval_s: float = 0.5):
        base_url = base_url or os.getenv("IMAGE_API_URL", "http://localhost:8082")
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self._timeout_s = timeout_s
        self._poll = poll_interval_s

    def health(self) -> bool:
        try:
            return self._client.get("/sdcpp/v1/capabilities").status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def _payload(req: ImageGenRequest) -> dict:
        return {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "seed": req.seed,
            "batch_count": req.batch_count,
            "sample_params": {
                "sample_method": "euler",
                "sample_steps": req.steps,
                "custom_sigmas": req.sigmas,
                "guidance": {"txt_cfg": req.cfg_scale},
            },
            "vae_tiling_params": {"enabled": req.vae_tiling},
            "output_format": "png",
        }

    @staticmethod
    def _check(r: httpx.Response) -> None:
        if r.is_error:
            raise RuntimeError(f"HTTP {r.status_code} {r.request.url}: {r.text[:500]}")

    @staticmethod
    def _extract_images(job: dict) -> list[str]:
        result = job.get("result") or {}
        items = result.get("images") or result.get("data") or job.get("images") or []
        out = []
        for im in items:
            if isinstance(im, str):
                out.append(im)
            elif isinstance(im, dict):
                b64 = im.get("b64_json") or im.get("data") or im.get("image")
                if b64:
                    out.append(b64)
        return out

    def generate(self, req: ImageGenRequest) -> ImageGenResult:
        t0 = time.perf_counter()
        r = self._client.post("/sdcpp/v1/img_gen", json=self._payload(req))
        self._check(r)
        body = r.json()
        job_id = body.get("id") or body.get("job_id")
        if not job_id:
            raise RuntimeError(f"pas d'id de job dans la réponse: {body}")

        while True:
            j = self._client.get(f"/sdcpp/v1/jobs/{job_id}")
            self._check(j)
            job = j.json()
            if job.get("status") in _TERMINAL:
                break
            if time.perf_counter() - t0 > self._timeout_s:
                try:
                    self._client.post(f"/sdcpp/v1/jobs/{job_id}/cancel")
                except httpx.HTTPError:
                    pass
                raise TimeoutError(f"job {job_id} > {self._timeout_s}s")
            time.sleep(self._poll)
        latency = time.perf_counter() - t0

        if job["status"] != "completed":
            raise RuntimeError(f"job {job_id} {job['status']}: {job.get('error')}")
        images = self._extract_images(job)
        if not images:
            raise RuntimeError(f"job {job_id} terminé sans image. Clés reçues: {list(job.keys())}")

        created, started, completed = job.get("created"), job.get("started"), job.get("completed")
        num = (int, float)
        return ImageGenResult(
            images_b64=images,
            latency_s=latency,
            gen_time_s=(completed - started) if isinstance(started, num) and isinstance(completed, num) else None,
            queue_time_s=(started - created) if isinstance(started, num) and isinstance(created, num) else None,
            request=req,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
