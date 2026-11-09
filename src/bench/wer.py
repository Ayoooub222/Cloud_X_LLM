"""Word error rate of a transcript against a reference (jiwer).

Normalisation, applied to both texts before scoring:
1. lower case;
2. punctuation removed (Unicode category P, apostrophes and hyphens included:
   "Brion's" -> "brions", "over-strained" -> "overstrained");
3. runs of whitespace collapsed, leading/trailing spaces stripped.
Numbers are NOT spelled out ("20s" stays different from "twenties") and British/American
spellings are not unified: such differences count as errors. This is simpler than the
Whisper English normaliser, so the WER is a slightly pessimistic upper bound.
"""
from __future__ import annotations

import jiwer

NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def normalise(text: str) -> str:
    """Text as seen by the scorer (for display and debugging)."""
    return " ".join(NORMALISE(text)[0])


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER in [0, +inf): (substitutions + deletions + insertions) / reference words."""
    return jiwer.wer(reference, hypothesis, reference_transform=NORMALISE, hypothesis_transform=NORMALISE)


def wer_details(reference: str, hypothesis: str) -> dict[str, int | float]:
    """WER and its error counts."""
    out = jiwer.process_words(reference, hypothesis, reference_transform=NORMALISE,
                              hypothesis_transform=NORMALISE)
    return {"wer": out.wer, "substitutions": out.substitutions,
            "deletions": out.deletions, "insertions": out.insertions}


def reference_words(reference: str) -> int:
    return len(NORMALISE(reference)[0])
