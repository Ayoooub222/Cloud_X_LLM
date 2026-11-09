"""WER normalisation and scoring."""
from pathlib import Path

from src.bench.wer import normalise, reference_words, wer_details, word_error_rate

REFERENCE = Path("inputs/audio/librispeech_1272-141231_66s.txt")


def test_normalise_case_punctuation_spaces():
    assert normalise("  Brion's over-strained,   Sir! ") == "brions overstrained sir"


def test_identical_after_normalisation():
    assert word_error_rate("A MAN SAID TO THE UNIVERSE", "A man said to the universe.") == 0.0


def test_error_counts():
    d = wer_details("one two three four", "one too three four five")
    assert (d["substitutions"], d["deletions"], d["insertions"]) == (1, 0, 1)
    assert d["wer"] == 0.5


def test_numbers_are_not_spelled_out():
    assert word_error_rate("THE TWENTIES", "the 20s") == 0.5


def test_reference_clip_word_count():
    assert reference_words(REFERENCE.read_text(encoding="utf-8")) == 148
