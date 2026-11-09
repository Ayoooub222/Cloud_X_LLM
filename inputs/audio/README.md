# Audio inputs

## librispeech_1272-141231_66s.flac / .txt

- Content: 10 consecutive utterances of LibriSpeech dev-clean, speaker 1272, chapter 141231
  (1272-141231-0000 to 1272-141231-0009), joined with 0.3 s of silence. 66.42 s, 16 kHz mono FLAC,
  148 reference words. Read speech, no repetition (unlike jfk_x6).
- Reference: the LibriSpeech transcripts of those utterances, joined with spaces (upper case, no
  punctuation, as distributed).
- Source: LibriSpeech ASR corpus, V. Panayotov, G. Chen, D. Povey, S. Khudanpur, ICASSP 2015,
  https://www.openslr.org/12, via the parquet file of the Hugging Face dataset
  hf-internal-testing/librispeech_asr_dummy (sha256 4e69a06f...27c603).
- License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The audio comes from
  LibriVox public-domain recordings. Changes: utterances concatenated with silence.
- Rebuild: scripts/make_librispeech_clip.py (deterministic, same sha256).

| file | sha256 |
|---|---|
| librispeech_1272-141231_66s.flac | a1c714aef65c1bbc36c02d676aca3905962f357ff551c37ae39b44d3639a0602 |
| librispeech_1272-141231_66s.txt | f89dc695e63b894271a75c09ec2fb860c8edf44927e7d90de12987958ef3b5c9 |

WER normalisation: see src/bench/wer.py.
