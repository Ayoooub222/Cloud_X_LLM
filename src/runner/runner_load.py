"""Load test at increasing arrival rates with k6: one k6 run and one record (task "load") per step.

Each step keeps a constant arrival rate (requests per minute) for --duration seconds. The SLO
(p95 latency) is a constant at the top of each k6 script; a step meets it when the p95 is below
the target, no request failed and k6 dropped no iteration. The sustained throughput at the SLO
is the highest achieved rate among the steps that meet it.

Examples:
    python -m src.runner.runner_load audio --rates 30 60 90 --duration 60
    python -m src.runner.runner_load vision --rates 3 6 9 12 --duration 120 --repeats 3
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.bench.recorder import PROJECT_ROOT, BenchRecord, GpuSampler, project_path, show
from src.bench.vllm_metrics import MetricsSampler

K6_DIR = PROJECT_ROOT / "deploy" / "k6"
OUT_DIR = PROJECT_ROOT / "outputs" / "load"
GPU_SAMPLE_INTERVAL_S = 1.0
METRICS_INTERVAL_S = 0.5
K6_THRESHOLDS_FAILED = 99  # k6 exit code when a threshold (the SLO) is crossed: still a valid measure
MAX_FAILED_RATE = 0.01
MS_PER_S = 1000.0

AUDIO_FILE = PROJECT_ROOT / "inputs" / "audio" / "librispeech_1272-141231_66s.flac"
VISION_PDF = PROJECT_ROOT / "inputs" / "vision" / "devis_fictif.pdf"
VISION_SCHEMA = "conf/schemas/extraction_page.json"
VISION_DPI = 100
DEFAULT_RATES_RPM = {"audio": [30, 60, 90, 120, 150, 180], "vision": [3, 6, 9, 12]}
DEFAULT_DURATION_S = {"audio": 60, "vision": 120}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="k6 load test at increasing arrival rates, one record per step")
    p.add_argument("target", choices=sorted(DEFAULT_RATES_RPM))
    p.add_argument("--rates", type=int, nargs="+", help="arrival rates in requests per minute")
    p.add_argument("--duration", type=int, help="length of each step in seconds")
    p.add_argument("--repeats", type=int, default=1, help="runs of the whole ramp")
    p.add_argument("--tag-prefix", default="k6", help="tag = <prefix>-<target>-r<rate>")
    return p.parse_args(argv)


def summary_metrics(summary: dict) -> dict:
    """Record metrics from the compact k6 summary written by deploy/k6/common.js."""
    latency = summary["latency_ms"]
    requests = summary["requests"] or 0
    run_s = summary["test_run_duration_ms"] / MS_PER_S
    p95_s = latency["p(95)"] / MS_PER_S if latency["p(95)"] is not None else None
    slo_s = summary["slo_p95_ms"] / MS_PER_S
    failed = summary["failed_rate"] or 0.0
    dropped = summary["dropped_iterations"] or 0
    meets = p95_s is not None and p95_s < slo_s and failed < MAX_FAILED_RATE and dropped == 0

    def seconds(stat: str) -> float | None:
        return round(latency[stat] / MS_PER_S, 3) if latency[stat] is not None else None

    return {
        "target_rps": round(summary["rate_rpm"] / 60, 4),
        "achieved_rps": round(requests / run_s, 4) if run_s else None,
        "requests": requests,
        "failed_rate": round(failed, 4),
        "dropped_iterations": dropped,
        "vus_max": summary["vus_max"],
        "latency_avg_s": seconds("avg"),
        "latency_p50_s": seconds("med"),
        "latency_p90_s": seconds("p(90)"),
        "latency_p95_s": seconds("p(95)"),
        "latency_p99_s": seconds("p(99)"),
        "latency_max_s": seconds("max"),
        "slo_p95_s": slo_s,
        "slo_ok": int(meets),
        "test_run_s": round(run_s, 1),
    }


def vision_setup() -> tuple[dict, str, str]:
    """Request body of runner_vision for page 1 of the quote, the model name and the base URL."""
    from src.clients.vision_client import VisionClient, pdf_to_pages  # heavy imports, vision only
    from src.utils.loader import Loader

    conf = Loader().load_conf()
    client = VisionClient(api_key=Loader().load_env(), base_url=conf["url_api"], schema_file=VISION_SCHEMA)
    client.connect()
    page = pdf_to_pages(VISION_PDF, dpi=VISION_DPI)[0]
    body = {
        "model": client.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page['b64']}"}},
            {"type": "text", "text": client.default_prompt},
        ]}],
        "response_format": client.response_format,
        **client.hyperparameters,
    }
    return body, client.model, conf["url_api"]


def step_env(target: str, rate: int, duration: int, summary_file: Path, extra: dict[str, str]) -> dict[str, str]:
    return {**os.environ, "RATE_RPM": str(rate), "DURATION_S": str(duration),
            "SUMMARY_FILE": str(summary_file), **extra}


def k6_version() -> str | None:
    try:
        return subprocess.run(["k6", "version"], capture_output=True, text=True, check=True).stdout.split()[1]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def run_step(args: argparse.Namespace, rate: int, duration: int, model: str, base_url: str,
             extra_env: dict[str, str], params: dict) -> bool:
    record = BenchRecord(task="load", backend="vllm", model=model, tag=f"{args.tag_prefix}-{args.target}-r{rate}",
                         params={**params, "rate_rpm": rate, "duration_s": duration})
    summary_file = OUT_DIR / f"k6_{record.run_id}.json"
    script = K6_DIR / f"{args.target}.js"
    sampler = GpuSampler(interval_s=GPU_SAMPLE_INTERVAL_S)
    scheduler = MetricsSampler(base_url.removesuffix("/v1"), METRICS_INTERVAL_S)
    with sampler, scheduler:
        proc = subprocess.run(["k6", "run", "--quiet", str(script)], capture_output=True, text=True,
                              env=step_env(args.target, rate, duration, summary_file, extra_env))
    record.gpu = sampler.stats()
    if proc.returncode not in (0, K6_THRESHOLDS_FAILED) or not summary_file.is_file():
        record.status = "error"
        record.error = f"k6 exit {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}"
    else:
        record.metrics = summary_metrics(json.loads(summary_file.read_text(encoding="utf-8")))
        record.metrics |= scheduler.summary()
        done = record.metrics["requests"]
        if record.gpu.energy_wh is not None and done:
            record.metrics["wh_per_request"] = round(record.gpu.energy_wh / done, 5)
        record.outputs = [str(summary_file)]
    path = record.save()
    m = record.metrics
    print(f"[{record.status}] {record.tag}: achieved={show(m.get('achieved_rps'))}/s "
          f"p50={show(m.get('latency_p50_s'))}s p95={show(m.get('latency_p95_s'))}s "
          f"slo_ok={show(m.get('slo_ok'))} failed={show(m.get('failed_rate'))} "
          f"dropped={show(m.get('dropped_iterations'))} waiting_max={show(m.get('sched.waiting_max'))} "
          f"energy/request={show(m.get('wh_per_request'))}Wh -> {path.name}", flush=True)
    if record.error:
        print(f"  error: {record.error}")
    return record.status == "ok"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("k6") is None:
        print("k6 not found in PATH (install it in ~/.local/bin, see scripts/install_k6.sh)", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rates = args.rates or DEFAULT_RATES_RPM[args.target]
    duration = args.duration or DEFAULT_DURATION_S[args.target]
    params = {"target": args.target, "script": f"deploy/k6/{args.target}.js", "k6_version": k6_version()}

    if args.target == "vision":
        try:
            body, model, base_url = vision_setup()
        except Exception as e:
            print(f"vLLM vision server unreachable or invalid config: {e}", file=sys.stderr)
            return 1
        body_file = OUT_DIR / "vision_body.json"
        body_file.write_text(json.dumps(body), encoding="utf-8")
        extra = {"BODY_FILE": str(body_file), "BASE_URL": base_url}
        params |= {"pdf_path": project_path(VISION_PDF), "dpi": VISION_DPI, "schema": VISION_SCHEMA}
    else:
        from src.clients.audio_client import DEFAULT_BASE_URL, DEFAULT_MODEL

        model, base_url = DEFAULT_MODEL, DEFAULT_BASE_URL
        extra = {"AUDIO_FILE": str(AUDIO_FILE), "BASE_URL": base_url, "AUDIO_MODEL": model}
        params |= {"audio_file": project_path(AUDIO_FILE), "language": "en"}

    all_ok = True
    for _ in range(args.repeats):
        for rate in rates:
            all_ok &= run_step(args, rate, duration, model, base_url, extra, params)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
