"""Transcribe one audio file with vLLM (Whisper) and save the benchmark record as JSON.

Examples:
    python -m src.runner.runner_audio data/audio/jfk.flac --tag warmup
    python -m src.runner.runner_audio data/audio/jfk_x6.wav --concurrency 4 --tag conc
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.bench.recorder import BenchRecord, GpuSampler
from src.clients.audio_client import (AudioClient, TranscriptionRequest,
                                      TranscriptionResult, audio_info)

GPU_SAMPLE_INTERVAL_S = 0.25  # runs are short (< 2 s), 1 s would give a single sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe one audio file with vLLM and save the run as JSON")
    p.add_argument("audio", type=Path, help="audio file (wav, flac, mp3...)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="number of identical requests sent at the same time")
    p.add_argument("--language", default=None, help="e.g. 'en'; default = auto-detect")
    p.add_argument("--tag", default=None, help="run label (warmup, single, conc, ...)")
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

    info = audio_info(args.audio)
    request = TranscriptionRequest(audio_path=args.audio, language=args.language)
    record = BenchRecord(
        task="audio", backend="vllm", model=request.model, tag=args.tag,
        params={"audio_file": str(args.audio), "concurrency": args.concurrency,
                "language": args.language, "response_format": request.response_format,
                "sample_rate": info["sample_rate"], "channels": info["channels"]},
    )

    with AudioClient() as client:
        if not client.health():
            print("vLLM audio server is unreachable (is the 'audio' profile running?)", file=sys.stderr)
            return 1

        sampler = GpuSampler(interval_s=GPU_SAMPLE_INTERVAL_S)
        try:
            with sampler:
                results, wall_s = run_batch(client, request, args.concurrency)
            record.metrics = batch_metrics(results, wall_s, info["duration_s"])
            record.outputs = [str(save_transcripts(results, args.out, record.run_id))]
        except Exception as e:  # a failure is a benchmark result too: record it
            record.status = "error"
            record.error = f"{type(e).__name__}: {e}"[:500]
        record.gpu = sampler.stats()

    path = record.save()
    m = record.metrics
    print(f"[{record.status}] audio={m.get('audio_duration_s')}s conc={args.concurrency} "
          f"latency={m.get('latency_s')}s wall={m.get('wall_s')}s "
          f"rtf={m.get('rtf')} batch_rtf={m.get('batch_rtf')}")
    print(f"  vram_peak={record.gpu.vram_peak_mib}MiB power={record.gpu.power_avg_w}W "
          f"energy={record.gpu.energy_wh}Wh samples={record.gpu.samples}")
    if record.error:
        print(f"  error: {record.error}")
    print(f"-> {path}")
    return 0 if record.status == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
