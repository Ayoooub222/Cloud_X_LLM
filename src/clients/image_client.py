"""Client for sd-server (stable-diffusion.cpp) running Qwen-Image-2.1-Turbo.

Native async API:
    POST /sdcpp/v1/img_gen    -> returns a job id
    GET  /sdcpp/v1/jobs/{id}  -> polled until the job is finished
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

DEFAULT_URL = "http://localhost:8082"
# Sigmas for the Turbo model: 9 sigmas -> 8 sampling steps
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]
FINISHED_STATUSES = {"completed", "failed", "cancelled"}


class ImageGenRequest(BaseModel):
    """Parameters of one generation, validated before being sent."""
    vae_tiling: bool = False  # decode tile by tile: less VRAM, but slower
    prompt: str = Field(min_length=1)
    negative_prompt: str = ""
    width: int = Field(768, ge=256, le=1536, multiple_of=64)
    height: int = Field(768, ge=256, le=1536, multiple_of=64)
    cfg_scale: float = Field(1.0, ge=0.0, le=15.0)  # Turbo model: keep 1.0
    seed: int = -1
    batch_count: int = Field(1, ge=1, le=4)
    sigmas: list[float] = Field(default_factory=lambda: list(TURBO_SIGMAS), min_length=2)

    @property
    def steps(self) -> int:
        return len(self.sigmas) - 1


class ImageGenResult(BaseModel):
    images_b64: list[str]
    latency_s: float                   # client side: queue + generation + network
    gen_time_s: float | None = None    # server side (completed - started), 1 s resolution
    queue_time_s: float | None = None  # server side (started - created)
    request: ImageGenRequest

    def save(self, out_dir: str | Path, stem: str = "img") -> list[Path]:
        """Write the images as <out_dir>/<stem>_<i>.png and return their paths."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, b64 in enumerate(self.images_b64):
            path = out_dir / f"{stem}_{i}.png"
            path.write_bytes(base64.b64decode(b64))
            paths.append(path)
        return paths


def build_payload(req: ImageGenRequest) -> dict:
    """Convert a request into the JSON body expected by sd-server."""
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


def raise_for_error(response: httpx.Response) -> None:
    if response.is_error:
        raise RuntimeError(f"HTTP {response.status_code} {response.request.url}: {response.text[:500]}")


def extract_images(job: dict) -> list[str]:
    """Return the base64 images of a finished job.

    The JSON layout may differ between sd-server versions, so several keys are tried.
    """
    result = job.get("result") or {}
    items = result.get("images") or result.get("data") or job.get("images") or []
    images = []
    for item in items:
        if isinstance(item, str):
            images.append(item)
        elif isinstance(item, dict):
            b64 = item.get("b64_json") or item.get("data") or item.get("image")
            if b64:
                images.append(b64)
    return images


def seconds_between(start, end) -> float | None:
    """end - start if both server timestamps are numbers, else None."""
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return end - start
    return None


class SdCppImageClient:
    def __init__(self, base_url: str | None = None,
                 timeout_s: float = 600.0, poll_interval_s: float = 0.5):
        base_url = base_url or os.getenv("IMAGE_API_URL", DEFAULT_URL)
        self.http = httpx.Client(base_url=base_url, timeout=30.0)
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s

    def health(self) -> bool:
        try:
            return self.http.get("/sdcpp/v1/capabilities").status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, req: ImageGenRequest) -> ImageGenResult:
        """Submit a job, wait until it is finished, return the images and timings."""
        start = time.perf_counter()
        response = self.http.post("/sdcpp/v1/img_gen", json=build_payload(req))
        raise_for_error(response)
        body = response.json()
        job_id = body.get("id") or body.get("job_id")
        if not job_id:
            raise RuntimeError(f"No job id in response: {body}")

        job = self.wait_for_job(job_id, start)
        latency = time.perf_counter() - start

        if job["status"] != "completed":
            raise RuntimeError(f"Job {job_id} {job['status']}: {job.get('error')}")
        images = extract_images(job)
        if not images:
            raise RuntimeError(f"Job {job_id} finished without image. Keys received: {list(job.keys())}")

        return ImageGenResult(
            images_b64=images,
            latency_s=latency,
            gen_time_s=seconds_between(job.get("started"), job.get("completed")),
            queue_time_s=seconds_between(job.get("created"), job.get("started")),
            request=req,
        )

    def wait_for_job(self, job_id: str, start: float) -> dict:
        """Poll the job until it is finished. Cancel it and raise on timeout."""
        while True:
            response = self.http.get(f"/sdcpp/v1/jobs/{job_id}")
            raise_for_error(response)
            job = response.json()
            if job.get("status") in FINISHED_STATUSES:
                return job
            if time.perf_counter() - start > self.timeout_s:
                self.cancel(job_id)
                raise TimeoutError(f"Job {job_id} took more than {self.timeout_s}s")
            time.sleep(self.poll_interval_s)

    def cancel(self, job_id: str) -> None:
        try:
            self.http.post(f"/sdcpp/v1/jobs/{job_id}/cancel")
        except httpx.HTTPError:
            pass  # best effort: the timeout error is raised anyway

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
