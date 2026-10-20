"""Build a long test file by repeating a short clip (converted to mono).

Write FLAC: vLLM rejects uploads above VLLM_MAX_AUDIO_CLIP_FILESIZE_MB (25 MB by default).

Example:
    python scripts/make_long_audio.py data/audio/jfk.flac 28 data/audio/jfk_x28.flac
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile


def main() -> None:
    p = argparse.ArgumentParser(description="Repeat an audio clip N times into a new file")
    p.add_argument("source", type=Path)
    p.add_argument("times", type=int)
    p.add_argument("target", type=Path, help="output file, format chosen from the extension")
    args = p.parse_args()

    data, rate = soundfile.read(str(args.source), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)  # stereo -> mono
    long_data = np.tile(data, args.times)
    soundfile.write(str(args.target), long_data, rate, subtype="PCM_16")
    size_mb = args.target.stat().st_size / 1e6
    print(f"{args.target}: {len(long_data) / rate:.1f}s, {rate} Hz, mono, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
