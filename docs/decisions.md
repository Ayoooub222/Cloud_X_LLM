# Decisions and accepted limitations

Each entry: the decision, its reason, and what it implies when reading the results.

## Measurement

**Proxies instead of AWS instances (zero budget).** The Kaggle T4 reproduces the GPU of a
`g4dn.xlarge`, not the instance: CPU, disk and network differ. The RTX 3070 Laptop is an
architecture proxy (Ampere, like the A10G of `g5`), not a capacity proxy (8 GB of VRAM instead of
24 GB). No `g6` (L4) instance has a proxy.

**T4 results tied to vLLM 0.29.** On Turing, vLLM only uses Triton attention, without tensor cores
below SM 8.0, and requires fp16 (no bf16). T4 figures are pessimistic; they should be measured
again if FlashInfer comes back on SM 7.5.

**fp16 on the T4 checked against bf16 on the 3070.** On the same inputs the quality is unchanged:
the 46 T4 transcripts are word for word the 3070 one (WER 3.38 %), and the field accuracy of the
quote is 96.7 % at every concurrency. fp16 overflow is not an issue for these two models, at least
on these inputs.

**Sustained throughput at the SLO rather than peak throughput.** Costs use the highest k6 step at
which every run meets the SLO (p95 < 5 s for a 66 s audio clip, < 30 s for a quote page). These
SLOs are choices: a looser SLO raises the retained throughput and lowers the cost.

**Energy measured over the whole step.** The energy per request includes the GPU idling between
requests: it is higher at low load, which reflects the real cost. `nvidia-smi` only sees the GPU:
the wall power of the laptop is an assumption (× 1.5).

**One tag = one condition, at least 3 runs, warmup excluded.** Runs of a faulty protocol are set
aside without being modified (moved out of `results/`), and the protocol is run again:
- 12 vision k6 steps with 2 pre-allocated VUs (k6 dropped one iteration for each VU it started);
- 1 step interrupted by a host suspend;
- 2 audio series on the Kaggle T4 that never reached the model (HTTP 401: the server read
  `VLLM_API_KEY`; then HTTP 400: `vllm[audio]` missing). They stay in the Kaggle output archive,
  outside `results/`.

**Laptop heating.** Under sustained load the 3070 Laptop loses throughput (vision c4: 603 then
541 pages/h); a step started with a cool GPU is the best of its series. A cooled cloud instance
does not degrade that way: local figures are rather pessimistic.

**Cold start.** It depends on the state of the vLLM compilation cache and on the mode (CUDA graphs
or `--enforce-eager`): to be recorded with every measurement.

## Inputs

**Vision on a generated fictitious quote**, with ground truth, rather than a real document: no
personal data, same bytes at every generation. Its layout is cleaner than a real scan, so the
measured accuracy is optimistic. Early measurements with a 4-field toy schema (14 times less work
per page) are discarded.

**pymupdf stays on 1.28.** Version 2.1 changes the bytes of the rendered quote, hence its SHA-256:
existing vision records would no longer be comparable. Dependabot ignores its major versions.

**Audio quality: LibriSpeech** (CC BY 4.0, 66 s, non-repetitive read speech) instead of `jfk_x6`
(one excerpt repeated six times, which trapped the decoder). WER of 3.4 %, simple normalization
documented in `src/bench/wer.py` (numbers are not spelled out: slightly pessimistic WER).

**Vision accuracy slightly lower in batches** (c2, c4) than at c1: numerical non-determinism of
batching. vLLM prefix caching means vision runs must be compared at the same cache state.

## Serving

**Audio saturation: GPU compute, not the scheduler.** Two series showed it (`--max-num-seqs 16`,
`--max-num-batched-tokens 8192`): neither changes the throughput. The compose settings are kept.

**Kubernetes resources taken from `docker stats` under load**, with no tight CPU limit (CFS
throttling would distort latencies), and memory headroom for `/dev/shm` and the page cache of the
memory-mapped weights.

**sd-server must not be PID 1**: it has no SIGTERM handler. `init: true` in compose,
`shareProcessNamespace` in the chart.

**Argo CD in its core flavor** (no UI, no Dex, ~80 MB idle), with a restricted `AppProject`. The
served model is chosen in Git (`values-gitops.yaml`); the machine-specific `modelsDir` is set on
the cluster. The imperative `switch_k8s.sh` refuses to run when Argo CD manages the release.

## Cost model

**Prices and assumptions are separate**: `conf/prices.yaml` only holds public prices with source and
date; every choice lives in `conf/cost_scenarios.yaml`.

**Measurement GPU → instance mapping**: `g4dn.xlarge` = 3070 throughput × 0.62 for audio and
× 0.33 for vision, ratios measured on the T4 and the 3070 at saturation (audio c8, vision c4 =
`--max-num-seqs`). Audio uses the LibriSpeech clip of the k6 load test, not `jfk_x6` (which gave
× 0.80): the ratio and the sustained throughput must come from the same input. `g5.xlarge` = 3070
throughput × 1.0 (a conservative assumption, not measured). The T4 falls behind more under load
(audio × 0.77 at c1, × 0.62 at c8): a ratio taken at low concurrency would flatter it.

**Throughput ratios assume the SLO still holds; it is checked where measured.** On the T4 one
quote page alone takes 37.6 s for a 30 s SLO: no load meets it. `g4dn` vision keeps its cost,
valid for batch work, but the report marks it "SLO not reachable" rather than dropping it.

**APIs compared on price and quality, free tiers only.** Groq (same Whisper model) and Gemini
Flash-Lite are measured on the same clip and page; OpenAI is priced but not measured (paid API,
zero budget). The Gemini cost uses the tokens Gemini bills for the page.

**Adapted request for Gemini.** Gemini rejects `maxItems` >= 30 on the quote-line arrays as a
decoding constraint, and does not accept vLLM sampling extensions (`top_k`, `min_p`,
`repetition_penalty`). Only the constraint and the sampling fields sent to Gemini change; the
prompt is the same and the answer is validated against the full schema.

**Not counted**: labor, availability (redundancy, on-call), home network, the 100 GB free egress
tier, Savings Plans.

## Repository

**History rewritten before publication**, immutable version tags (GitHub ruleset): a tag mistake is
fixed with a new tag. GitHub Actions pinned by commit SHA, updated by Dependabot (major versions in
separate PRs, release notes read before merging).
