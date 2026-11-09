"""Rebuild inputs/audio/librispeech_1272-141231_66s.{flac,txt} from LibriSpeech dev-clean.

Source: parquet of hf-internal-testing/librispeech_asr_dummy (73 dev-clean utterances,
LibriSpeech, CC BY 4.0). Consecutive utterances of one chapter are joined with 0.3 s
of silence until the clip reaches 60 s; the reference is the joined transcripts.
pyarrow is needed (not in the requirements: one-off script, run it in a throwaway venv).

Example:
    curl -sSL -o /tmp/ls/validation.parquet "$PARQUET_URL"
    python -I scripts/make_librispeech_clip.py /tmp/ls/validation.parquet inputs/audio
"""
import io
import sys
from pathlib import Path

import numpy as np
import soundfile

PARQUET_URL = ("https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
               "resolve/main/clean/validation-00000-of-00001.parquet")
PARQUET_SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
CHAPTER_ID = 141231
TARGET_S = 60.0
GAP_S = 0.3


def chapter_rows(parquet: Path) -> list[dict]:
    import pyarrow.parquet as pq  # optional dependency, see module docstring

    rows = [r for r in pq.read_table(parquet).to_pylist() if r["chapter_id"] == CHAPTER_ID]
    return sorted(rows, key=lambda r: r["id"])


def build_clip(rows: list[dict]) -> tuple[np.ndarray, int, list[str]]:
    """Audio samples, sample rate and transcripts of the first utterances reaching TARGET_S."""
    pieces, texts, rate = [], [], None
    for row in rows:
        data, sr = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        rate = rate or sr
        if pieces:
            pieces.append(np.zeros(int(GAP_S * sr), dtype="float32"))
        pieces.append(data)
        texts.append(row["text"])
        if sum(len(p) for p in pieces) / sr >= TARGET_S:
            break
    return np.concatenate(pieces), rate, texts


def main() -> int:
    parquet, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    rows = chapter_rows(parquet)
    audio, rate, texts = build_clip(rows)
    stem = f"librispeech_{rows[0]['speaker_id']}-{CHAPTER_ID}_{round(len(audio) / rate)}s"
    out_dir.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(out_dir / f"{stem}.flac"), audio, rate, format="FLAC")
    (out_dir / f"{stem}.txt").write_text(" ".join(texts) + "\n", encoding="utf-8")
    print(f"{stem}: {len(audio) / rate:.2f} s, {len(texts)} utterances")
    return 0


if __name__ == "__main__":
    sys.exit(main())
