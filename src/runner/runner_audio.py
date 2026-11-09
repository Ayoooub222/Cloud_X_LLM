"""Transcribe one audio file with vLLM (Whisper) and save the benchmark record as JSON.

Examples:
    python -m src.runner.runner_audio data/audio/jfk.flac --tag warmup
    python -m src.runner.runner_audio data/audio/jfk_x6.wav --concurrency 4 --tag conc
    python -m src.runner.runner_audio inputs/audio/librispeech_1272-141231_66s.flac \
        --reference inputs/audio/librispeech_1272-141231_66s.txt --language en --tag conc-c1
"""
import argparse
import contextlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.bench.recorder import BenchRecord, GpuSampler, project_path, show
from src.bench.vllm_metrics import MetricsSampler
from src.bench.wer import reference_words, wer_details
from src.clients.audio_client import (
    DEFAULT_BASE_URL,
    AudioClient,
    TranscriptionRequest,
    TranscriptionResult,
    audio_info,
)
from src.utils.loader import Loader

GPU_SAMPLE_INTERVAL_S = 0.25  # runs are short (< 2 s), 1 s would give a single sample
METRICS_INTERVAL_S = 0.5      # vLLM scheduler state (/metrics); 0 disables it


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe one audio file with vLLM and save the run as JSON")
    p.add_argument("audio", type=Path, help="audio file (wav, flac, mp3...)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="number of identical requests sent at the same time")
    p.add_argument("--language", default=None, help="e.g. 'en'; default = auto-detect")
    p.add_argument("--tag", default=None, help="run label (warmup, single, conc, ...)")
    p.add_argument("--reference", type=Path, default=None,
                   help="reference transcript (plain text): adds wer, wer_max and error counts")
    p.add_argument("--metrics-interval", type=float, default=METRICS_INTERVAL_S,
                   help="poll vLLM /metrics every N s during the batch (sched.* metrics); 0 = off")
    p.add_argument("--out", type=Path, default=Path("outputs/audio/"))
    return p.parse_args()


def run_batch(client: AudioClient, request: TranscriptionRequest,
              concurrency: int) -> tuple[list[TranscriptionResult], float]:
    """Send `concurrency` requests at once. Return the results and the wall time of the batch."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(client.transcribe, request) for _ in range(concurrency)]
        results = [future.result() for future in futures]  # re-raises the first error
    return results, time.perf_counter() - start


def batch_metrics(results: list[TranscriptionResult], wall_s: float, audio_s: float) -> dict:
    latencies = [r.latency_s for r in results]
    latency_avg = sum(latencies) / len(latencies)
    return {
        "audio_duration_s": audio_s,
        "server_duration_s": results[0].server_duration_s,
        "requests": len(results),
        "latency_s": round(latency_avg, 3),
        "latency_max_s": round(max(latencies), 3),
        "wall_s": round(wall_s, 3),
        "rtf": round(latency_avg / audio_s, 4),                     # user view
        "batch_rtf": round(wall_s / (audio_s * len(results)), 4),   # GPU cost view
        "segments": results[0].segments,
    }


def quality_metrics(results: list[TranscriptionResult], reference: str) -> dict:
    """Mean and worst WER of the batch, error counts of the first transcript (see src/bench/wer.py)."""
    details = [wer_details(reference, r.text) for r in results]
    wers = [d["wer"] for d in details]
    first = details[0]
    return {
        "wer": round(sum(wers) / len(wers), 4),
        "wer_max": round(max(wers), 4),
        "reference_words": reference_words(reference),
        "substitutions": first["substitutions"],
        "deletions": first["deletions"],
        "insertions": first["insertions"],
    }


def save_transcripts(results: list[TranscriptionResult], out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"audio_{run_id}.json"
    data = [r.model_dump() for r in results]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    if not args.audio.is_file():
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 1
    if args.reference and not args.reference.is_file():
        print(f"Reference file not found: {args.reference}", file=sys.stderr)
        return 1
    reference = args.reference.read_text(encoding="utf-8") if args.reference else None

    info = audio_info(args.audio)
    request = TranscriptionRequest(audio_path=args.audio, language=args.language)
    record = BenchRecord(
        task="audio", backend="vllm", model=request.model, tag=args.tag,
        params={"audio_file": project_path(args.audio), "concurrency": args.concurrency,
                "language": args.language, "response_format": request.response_format,
                "sample_rate": info["sample_rate"], "channels": info["channels"],
                "reference_file": project_path(args.reference) if args.reference else None},
    )

    with AudioClient(api_key=Loader().load_env()) as client:  # sent only when VLLM_API_KEY is set
        if not client.health():
            print("vLLM audio server is unreachable (is the 'audio' profile running?)", file=sys.stderr)
            return 1

        sampler = GpuSampler(interval_s=GPU_SAMPLE_INTERVAL_S)
        scheduler = MetricsSampler(DEFAULT_BASE_URL, args.metrics_interval) if args.metrics_interval > 0 else None
        try:
            with sampler, scheduler or contextlib.nullcontext():
                results, wall_s = run_batch(client, request, args.concurrency)
            record.metrics = batch_metrics(results, wall_s, info["duration_s"])
            if scheduler is not None:
                record.metrics |= scheduler.summary()
            if reference is not None:
                record.metrics |= quality_metrics(results, reference)
            record.outputs = [str(save_transcripts(results, args.out, record.run_id))]
        except Exception as e:  # a failure is a benchmark result too: record it
            record.status = "error"
            record.error = f"{type(e).__name__}: {e}"[:500]
        record.gpu = sampler.stats()

    path = record.save()
    m = record.metrics
    print(f"[{record.status}] audio={show(m.get('audio_duration_s'))}s conc={args.concurrency} "
          f"latency={show(m.get('latency_s'))}s wall={show(m.get('wall_s'))}s "
          f"rtf={show(m.get('rtf'))} batch_rtf={show(m.get('batch_rtf'))} wer={show(m.get('wer'))}")
    print(f"  scheduler: running_max={show(m.get('sched.running_max'))} "
          f"waiting_max={show(m.get('sched.waiting_max'))} kv_cache_max={show(m.get('sched.kv_cache_max'))}")
    print(f"  vram_peak={show(record.gpu.vram_peak_mib)}MiB power={show(record.gpu.power_avg_w)}W "
          f"energy={show(record.gpu.energy_wh)}Wh samples={show(record.gpu.samples)}")
    if record.error:
        print(f"  error: {record.error}")
    print(f"-> {path}")
    return 0 if record.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
