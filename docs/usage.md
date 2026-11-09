# Usage guide

How to install, run and measure everything in this repository. For what the measurements show,
read the [report](report.md); for why things are done this way, the [decisions](decisions.md).

## The three models

| Profile | Modality | Backend | Model | Port |
|---|---|---|---|---|
| `vision` | PDF → structured JSON | vLLM 0.29.0 | Qwen3-VL-2B-Instruct | 8000 |
| `image` | text → image | stable-diffusion.cpp (`sd-server`) | Qwen-Image 2.1 Turbo (GGUF Q4_K) | 8082 |
| `audio` | speech → text | vLLM 0.29.0 (custom image) | whisper-large-v3-turbo | 8001 |
| `mlops` | experiment tracking | MLflow 3.16.0 (SQLite in `./mlruns`) | - | 5000 |

These are not chat LLMs but multimodal models with three different load profiles (token
generation, diffusion, transcription), closer to a real catalog of services. The vision use case
is the extraction of car repair quotes and orders into a Pydantic schema
([src/schemas/extraction.py](../src/schemas/extraction.py)).

## Repository layout

```
.
├── docker-compose.yml        # 3 GPU profiles (vision, image, audio) + mlops
├── docker/
│   ├── audio.Dockerfile      # vLLM + audio extra, pinned versions, smoke test at build time
│   └── bench.Dockerfile      # benchmark client (runners, summarize), no GPU, published on GHCR
├── conf/app.conf             # API URL and per-model hyperparameters (vision)
├── conf/schemas/             # JSON schema of the extraction, generated from Pydantic
├── conf/prices.yaml          # public prices with source and date (AWS Paris, electricity, APIs)
├── conf/cost_scenarios.yaml  # cost model assumptions
├── inputs/vision/            # fictitious quote (PDF) and its ground truth
├── inputs/audio/             # 66 s LibriSpeech clip (CC BY 4.0) and its reference transcript
├── src/
│   ├── clients/              # HTTP clients: vision_client, image_client, audio_client
│   ├── runner/               # CLI entry points (python -m src.runner.<name>)
│   │   ├── runner_vision.py / runner_image.py / runner_audio.py
│   │   ├── runner_load.py    # k6 load ramp, one "load" record per step
│   │   ├── runner_api.py     # hosted APIs (Groq, Gemini free tiers) on the same inputs
│   │   ├── coldstart.py      # cold start = time to Ready + first request
│   │   ├── summarize.py      # aggregated tables (or JSON) from the records
│   │   └── mlflow_import.py  # idempotent (re)import of the records into MLflow
│   ├── bench/                # also extraction_score.py (accuracy) and synthetic_quote.py (fictitious quote)
│   │   ├── recorder.py       # BenchRecord (1 run = 1 JSON) + GpuSampler (nvidia-smi)
│   │   ├── sdcpp_logs.py     # sd-server stage timings, read from its logs
│   │   ├── wer.py            # WER (jiwer) and its normalization
│   │   ├── vllm_metrics.py   # vLLM /metrics probe (running, waiting, KV cache)
│   │   └── mlflow_log.py     # optional copy to MLflow, never blocking
│   ├── schemas/              # Pydantic schema of the vision extraction (French names, see below)
│   ├── cost/                 # cost model: config (typed YAML), measurements, model, report
│   ├── report/               # report figures (matplotlib + seaborn)
│   └── utils/loader.py       # reads conf/app.conf and .env
├── scripts/                  # switch, wait_ready, bench, k3s, registry, check_models...
├── deploy/
│   ├── helm/ai-models/       # chart: 1 Deployment + 1 Service per model, optional MLflow
│   ├── argocd/               # light Argo CD install, AppProject and Application (GitOps)
│   ├── k6/                   # load scripts (audio, vision), SLO at the top of each script
│   ├── k3s/                  # k3s server config, local registry mirror
│   └── k8s/                  # NVIDIA device plugin values, GPU test pods
├── kaggle/                   # notebook to replay the benchmark on a free Kaggle T4
├── infra/aws/                # Terraform: GPU node, S3, IAM, budget alerts, tests (mock provider)
├── docs/                    # report, usage guide, decisions, figures
├── results/benchmark/        # benchmark JSON = source of truth (versioned)
├── tests/                    # pytest
└── .github/                  # CI workflow, Dependabot
```

Design choices:

- **One run = one JSON file** in `results/benchmark/<task>/`. Failures are recorded too (a failure
  is a benchmark result). MLflow is only an optional copy: if MLflow is down, the benchmark goes on.
- **The `BENCH_ENV` label** identifies the machine (`local-wsl2-rtx3070laptop`,
  `local-k3s-rtx3070laptop`, `kaggle-t4`...), so environments can be compared side by side.
- **Code is in English.** Prompts, JSON schemas and input documents are data, not code:
  translating them would change the measured load. That is why `src/schemas/` keeps French names.
- **The vision test document is fictitious and generated** (`python -m src.bench.synthetic_quote`):
  same bytes at every generation, ground truth from the same data, no real data. The JSON schema is
  generated from the Pydantic model (`python -m src.schemas.export_schema`); a test checks that the
  versioned files stay in sync.

Decisions and their consequences: [decisions.md](decisions.md).

---

## Setup

Requirements: Linux or WSL2, an NVIDIA GPU with a recent driver, Docker + Docker Compose with the
NVIDIA Container Toolkit, Python 3.13, [uv](https://docs.astral.sh/uv/). For k3s: `k3s`, `kubectl`,
`helm`. For AWS: Terraform (see below).

```bash
python3.13 -m venv venv
source venv/bin/activate
uv pip install -r requirements.txt       # runner dependencies
uv pip install -r requirements-dev.txt   # + pytest, ruff, and the plotting libraries

cp .env.example .env                     # then adapt the model paths
```

The `.env` file holds:

| Variable | Role |
|---|---|
| `MODELS_DIR` | weights folder, mounted read-only in the containers (`./models` by default) |
| `IMAGE_DIFFUSION`, `IMAGE_VAE`, `IMAGE_LLM` | image model files, relative to `MODELS_DIR` |
| `AUDIO_MODEL_DIR` | Whisper model folder, relative to `MODELS_DIR` |
| `VLLM_API_KEY` | key read by the vision client (`src/utils/loader.py`); **not in `.env.example`**, add it |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | optional, for `runner_api` (free tiers) |

Expected weights in `./models`: `Qwen3-VL-2B-Instruct`, `Qwen3-VL-8B-Instruct-GGUF`,
`Qwen-Image-2.1-Turbo-GGUF`, `Qwen-Image-2.1-VAE`, `whisper-large-v3-turbo`. The download is not
scripted. To check:

```bash
bash scripts/check_models.sh             # prints OK or MISSING for each file
```

Extra local audio files (in `data/audio/`, not versioned):

```bash
python scripts/make_long_audio.py data/audio/jfk.flac 6  data/audio/jfk_x6.wav
python scripts/make_long_audio.py data/audio/jfk.flac 28 data/audio/jfk_x28.flac   # FLAC: vLLM rejects > 25 MB
```

---

## Local commands (Docker Compose)

One GPU model at a time. `switch.sh` stops the three GPU services, starts the requested one
(without touching MLflow) and follows its logs.

```bash
source venv/bin/activate

# Choose the served model (stops the others)
bash scripts/switch.sh vision
bash scripts/switch.sh image
bash scripts/switch.sh audio

# Wait until the server answers: "READY after Xs" = the cold start
bash scripts/wait_ready.sh audio         # vision | image | audio | mlflow

# MLflow (CPU, runs next to the models)
docker compose --profile mlops up -d
bash scripts/wait_ready.sh mlflow        # UI on http://localhost:5000

# Stop and status
docker compose --profile vision --profile image --profile audio rm --stop --force vision image audio
docker compose ps
docker compose --profile audio logs -f

# First time for audio: build the custom image
docker compose --profile audio build
```

Do not start the whole stack at once, especially with the image profile: WSL2 RAM cannot keep up.

Under WSL2 the vLLM V2 model runner crashes (`UVA is not available`): compose forces
`VLLM_USE_V2_MODEL_RUNNER=0` (variable `VLLM_V2_RUNNER`, default `0`).

---

## Running a benchmark

### One measurement at a time (`python -m src.runner.…`)

Each command sends one request (or a concurrent batch), measures, and writes a JSON file to
`results/benchmark/<task>/`. The matching server must be running.

```bash
source venv/bin/activate

# Audio (audio profile)
python -m src.runner.runner_audio inputs/audio/librispeech_1272-141231_66s.flac \
  --reference inputs/audio/librispeech_1272-141231_66s.txt --language en --tag warmup
python -m src.runner.runner_audio inputs/audio/librispeech_1272-141231_66s.flac \
  --reference inputs/audio/librispeech_1272-141231_66s.txt --language en --concurrency 4 --tag ls66-c4

# Image (image profile)
python -m src.runner.runner_image "a red fox in the snow" --seed 42 --size 768 --tag warmup
python -m src.runner.runner_image "a red fox in the snow" --tiling --tag testA-tiling

# Vision (vision profile): a PDF and a JSON schema, one record per page
python -m src.runner.runner_vision inputs/vision/devis_fictif.pdf conf/schemas/extraction_page.json \
    --expected inputs/vision/devis_fictif.expected.json --tag warmup
python -m src.runner.runner_vision inputs/vision/devis_fictif.pdf conf/schemas/extraction_page.json \
    --expected inputs/vision/devis_fictif.expected.json --concurrency 4 --tag conc-c4
```

`--reference` adds the WER, `--expected` adds `field_accuracy`. `--dpi` changes the number of input
tokens (`prompt_tokens`), hence the load: compare two vision runs only at equal `prompt_tokens`.

### Automated series (`scripts/bench.sh`)

Same protocol in compose and in k3s, with the same tags, to compare them.

```bash
# scripts/bench.sh <k3s|compose> <audio|audio-short|vision|image> [cold_start_runs]
scripts/bench.sh compose audio-short
scripts/bench.sh compose image 0         # 0 = start the model once, without recording a cold start
scripts/bench.sh k3s audio

# Vision uses the fictitious quote, its schema and its ground truth by default
scripts/bench.sh compose vision
```

The script runs: (1) *N* cold starts (3 by default) from "no model", with the time to Ready and to
the first real request, (2) warmup requests left out of the statistics, (3) the "warm" series
(3 repetitions). GPU state, power supply and RAM are written at the start and the end in
`results/benchmark/sessions/`. The script refuses to run if the other environment (k3s or compose)
is active, or if the available RAM is too low for the image model.

### k6 load (`python -m src.runner.runner_load`)

```bash
scripts/install_k6.sh                    # k6 v2.3.0 in ~/.local/bin, checksum verified

# One step = one k6 run at a constant rate (requests/min) = one "load" record, tag k6-<task>-r<rate>
python -m src.runner.runner_load audio --repeats 3                         # 30 to 180 req/min, 60 s
python -m src.runner.runner_load vision --repeats 3                        # 3 to 12 req/min, 120 s
python -m src.runner.runner_load audio --rates 60 120 --duration 30 --tag-prefix trial
```

The runner samples the GPU and `/metrics` during each step. Warm the server up first (a short step,
or a `--tag warmup` run of the task runner).

### Hosted APIs (`python -m src.runner.runner_api`)

```bash
# Keys in .env: GROQ_API_KEY, GEMINI_API_KEY (free tiers, no billing account attached)
python -m src.runner.runner_api groq-audio --repeats 3 --tag api-c1
python -m src.runner.runner_api gemini-vision --repeats 3 --tag api-c1
```

Requests are sequential and spaced out (free-tier rate limits); records use the env labels
`api-groq` and `api-gemini`.

---

## Reading the results

```bash
# Table aggregated by tag (mean ± standard deviation)
python -m src.runner.summarize audio
python -m src.runner.summarize audio --tag single-11s --tag single-308s
python -m src.runner.summarize image --fields time vram power energy

# Compare environments side by side
python -m src.runner.summarize audio --by-env --tag conc-c8

# One column per run, or filter by date / environment
python -m src.runner.summarize audio --env kaggle-t4 --runs
python -m src.runner.summarize coldstart --env local-k3s-rtx3070laptop

# JSON export per (task, tag, BENCH_ENV): statistics of every field (n, mean, stdev, min, p50, p95,
# max) and a "headline" block (requests/h, p50/p95 latency, energy per request). For the load task,
# p50/p95 are per request (k6); elsewhere they are over runs.
python -m src.runner.summarize all --format json --out outputs/summary/all.json
python -m src.runner.summarize load --format json --tag k6-audio-r120

# Cost per 1000 requests (markdown or JSON)
python -m src.cost.report
python -m src.cost.report --format json --out outputs/cost/report.json

# Send the records to MLflow (idempotent: runs already logged are skipped)
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.runner.mlflow_import --dry-run
python -m src.runner.mlflow_import --task audio --since "2026-01-01 00:00:00"
```

Failures are counted (`runs ok/total`) but left out of the statistics.

| Metric | Meaning |
|---|---|
| `latency_s` | response time seen by the user |
| `rtf` | latency / audio duration (user view) |
| `batch_rtf` | wall time of the batch / (audio duration × number of requests): **GPU cost view** |
| `requests` | number of requests in the batch (= the concurrency) |
| `wer`, `wer_max` | word error rate (batch mean and worst), with `--reference` |
| `sched.running_max`, `sched.waiting_max`, `sched.kv_cache_max` | vLLM scheduler state (`/metrics`) during the batch |
| `gpu.energy_wh` | mean power × duration: electricity cost of self-hosting |
| `gpu.vram_peak_mib` | VRAM peak (with vLLM, the `--gpu-memory-utilization` reservation, not the actual need) |
| `t_ready_s`, `t_first_s` | cold start: time to Ready, then to the first response |
| `achieved_rps`, `latency_p95_s`, `slo_ok` | k6 load (task `load`): served rate, per-request p95, SLO met (1) or not (0) |

---

## Report figures

```bash
python -m src.report.make_figures                       # every figure into docs/figures/
python -m src.report.make_figures --only cost_vs_utilization break_even
```

The figures are built from the records and from the cost model itself (`src/report/`), so they
change with the measurements and the prices. Tests check that the plotted ratios and sustained
rates are the ones the cost model uses.

## Local commands (k3s + Helm)

The same models on Kubernetes with the `deploy/helm/ai-models` chart: the local counterpart of what
Terraform would deploy on AWS. **Never run k3s and the compose GPU profiles at the same time**
(limited RAM).

```bash
# Start / stop k3s (stops compose first, cleans up "Unknown" pods)
scripts/k3s.sh start
scripts/k3s.sh stop

# Local OCI registry (images are pulled from localhost:5001)
scripts/registry.sh start
scripts/registry.sh push cloud-x-llm/vllm-audio:v0.29.0
scripts/registry.sh list

# Imperative mode (without Argo CD): replicas 1 for the chosen model, 0 for the others
scripts/switch_k8s.sh audio              # vision | image | audio | none
scripts/switch_k8s.sh vision --mlflow    # with MLflow; --no-mlflow to turn it off
scripts/switch_k8s.sh none               # frees the GPU

# Check the GPU in the cluster
kubectl apply -f deploy/k8s/tests/gpu-smoke.yaml
kubectl -n ai-models get pods
```

CPU and memory `requests`/`limits` of each model and of MLflow come from `docker stats`
measurements under load (comments in [values.yaml](../deploy/helm/ai-models/values.yaml)): vision
4.0 GiB and ~1.1 cores, audio 2.8 GiB and ~0.5 core, MLflow ~750 MiB with one worker. CPU limits are
wide (CFS throttling would distort latencies) and the memory headroom covers `/dev/shm`, which
counts toward the pod limit on Kubernetes.

The NVIDIA device plugin (v0.20.0) is preferred to a bare `RuntimeClass`: without it the GPU is not
accounted for and nothing stops two models from competing for it. The MLflow Deployment uses
`Recreate` (SQLite, a single writer) and its PVC survives both `helm uninstall` and the deletion of
the Argo CD Application.

### Argo CD (GitOps mode)

[deploy/argocd/](../deploy/argocd/) installs the light "core" flavor of Argo CD (4 workloads, no UI,
no Dex, ~80 MB of RAM when idle), a restricted `AppProject` (this repository, the `ai-models`
namespace only) and an `Application` that tracks `main`. The served model is chosen in Git, in
[values-gitops.yaml](../deploy/helm/ai-models/values-gitops.yaml): change the replicas, commit, push,
and Argo CD syncs within ~3 minutes. `switch_k8s.sh` refuses to run in this mode, since `selfHeal`
would revert a manual `helm upgrade`.

```bash
helm uninstall ai-models -n ai-models    # once, if the release was installed by switch_k8s.sh
kubectl apply --server-side -k deploy/argocd
kubectl apply -f deploy/argocd/application.yaml
kubectl -n argocd patch application ai-models --type merge -p '{"spec":{"source":{"helm":{"parameters":[{"name":"modelsDir","value":"/opt/models"}]}}}}'
kubectl -n argocd get application ai-models
```

`modelsDir` is machine-specific and stays out of Git (set it to the weights folder of the node).

---

## Measurements on a T4 (Kaggle)

A Kaggle T4 is the GPU of a `g4dn.xlarge` instance. The notebook in [kaggle/](../kaggle/) replays the
same runners there with `BENCH_ENV=kaggle-t4`; the records are then brought back and compared with
`summarize --by-env`. Step by step: [kaggle/README.md](../kaggle/README.md).

What `scripts/kaggle_t4_bench.sh` does, in one notebook cell (~20 minutes):

1. **Server**: vLLM 0.29.0 with its audio extra in a separate `uv` venv (Python 3.12). The CUDA
   wheel (12.9 or 13) is chosen from the driver CUDA version shown by `nvidia-smi`.
2. **Client**: the cloned repository in a second venv (Python 3.13, `requirements.txt`), because
   vLLM pins its own version of the `openai` SDK.
3. **Launch**: the compose arguments plus `--dtype float16` (the T4 has no bf16), and
   `--mm-encoder-attn-backend TORCH_SDPA` for vision, a workaround for a Qwen3-VL bug on Turing.
   The servers start without `VLLM_API_KEY`, as in compose.
4. **Measurements**: vision (warmup, then 3 × c1/c2/c4 on the fictitious quote), then audio
   (warmup, then 3 × c1/c2/c4/c8 on the LibriSpeech clip with its reference). `PHASES=audio`
   replays one phase and keeps the records already written.
5. **Output**: everything in `/kaggle/working/out`, archived as `kaggle-t4.tar.gz`.

Only versioned inputs are used, so nothing private is uploaded.

---

## AWS infrastructure (Terraform)

`infra/aws/` describes an AWS GPU node that reproduces the local k3s environment. **It is validated
and tested in CI but has never been applied** (zero budget).

| File | Resource |
|---|---|
| `compute.tf`, `user_data.sh.tftpl` | GPU instance (`g4dn.xlarge` by default, optional spot) that installs k3s, the NVIDIA device plugin and the Helm chart at boot |
| `s3.tf` | results bucket (encrypted, versioned, protected against `terraform destroy`) |
| `iam.tf`, `network.tf` | instance role; no inbound port, access through SSM Session Manager |
| `budgets.tf` | e-mail alerts at 50 % and 100 % of a monthly budget (alerts only, not a cap) |
| `tests/plan.tftest.hcl` | `terraform test` with a mocked AWS provider (no credentials) |

k3s runs on the VM instead of EKS, to avoid ~73 USD/month of control plane.

Before a first apply on a new account, two account settings matter more than the code:

- **vCPU quotas**: GPU instances count against the *Running On-Demand G and VT instances* quota
  (and *All G and VT Spot Instance Requests* for spot), expressed in vCPUs and often 0 on a new
  account. A `g4dn.xlarge` needs 4; the increase is requested in Service Quotas, per region.
- **Budgets alert, they do not stop anything**: `budgets.tf` sends e-mails, the instance keeps
  running. The only hard stop is `terraform destroy` (or stopping the instance).

```bash
scripts/install_terraform.sh             # installs Terraform in ~/.local/bin
cd infra/aws
terraform init -backend=false            # no remote backend
terraform validate
terraform test
```

Main variables: `instance_type` (`g4dn.xlarge`, `g4dn.2xlarge`, `g6.xlarge`), `use_spot`, `region`
(`eu-west-3`, Paris), `git_ref` (repository tag deployed on the node, `v1.0.0`), `budget_limit_usd`
(10 USD). Known limitation: the `user_data` does not download the model weights yet, so the models
start with 0 replicas.

---

## Quality: tests, lint, CI

```bash
source venv/bin/activate
python -m pytest
python -m ruff check .

helm lint deploy/helm/ai-models --set modelsDir=/models
helm template ai-models deploy/helm/ai-models --set modelsDir=/models > /dev/null
```

The GitHub Actions CI ([ci.yml](../.github/workflows/ci.yml)) runs on every push and pull request:

| Job | Content |
|---|---|
| `python` | `ruff check` + `pytest` (Python 3.13, dependencies installed with `uv`) |
| `helm` | `helm lint` + `helm template` |
| `terraform` | `fmt -check`, `init -backend=false`, `validate`, `terraform test` with a mocked AWS provider; on a `v*` tag, checks that the tag equals the default `git_ref` |
| `image` | on push only, after the other three: builds and publishes `ghcr.io/ayoooub222/cloud_x_llm/bench-client` |

Actions are pinned by commit SHA (version in a comment). Dependabot proposes weekly updates for the
actions and the pip packages; major versions come in separate PRs.

Version tags (`v*`) are protected by a GitHub ruleset: they can be neither moved nor deleted. A
mistake is fixed with a new tag. A local pre-push hook runs the same tag check before the push:

```bash
git config core.hooksPath scripts/hooks   # once per clone
```

---

## Benchmark rules

- **Warmup**: a `--tag warmup` run after every start or long idle period, left out of the
  statistics (the first vision call after idling took 19 s instead of 2.3 s).
- **One tag = one condition** (for example `single-66s`, `conc-c4`). Never two durations or two
  settings under the same tag. At least 3 runs per condition.
- **One command at a time**, otherwise queueing distorts the latencies.
- **Image**: vary the prompts, `sd-server` caches the last 4 conditionings.
- Never run `nvidia-smi -l` during a measurement.
- Set `BENCH_ENV` to identify the machine (`bench.sh` sets it itself).
- With several visible GPUs, `CUDA_VISIBLE_DEVICES` restricts sampling to the first listed GPU; add
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that CUDA and `nvidia-smi` use the same numbering.

## Known pitfalls

- A Docker "CACHED" step does not prove the step did what was intended: the audio Dockerfile runs
  an import smoke test at build time.
- An image tag is mutable: check that the container runs the ID of the built image.
- `sd-server` installs no SIGTERM handler: as PID 1 (exec-form entrypoint) the signal was ignored
  and `docker stop` waited 10.3 s before SIGKILL (exit 137). Fixed with `init: true` in compose
  (0.3 s, exit 143) and `shareProcessNamespace` in the chart (`kubectl delete pod` in 1.0 s on k3s,
  instead of the 10 s grace period).
- Argo CD ignores `helm.sh/resource-policy: keep`: the MLflow PVC also carries
  `argocd.argoproj.io/sync-options: Prune=false,Delete=false`.
- `switch.sh` run again too quickly gives `Error while Stopping`: wait a few seconds.
- After `source ~/.bashrc`, run `source venv/bin/activate` again.
- Under WSL2, a full C: drive causes an OOM then stops k3s, without triggering `DiskPressure`.
- Images stored three times (Docker, containerd, registry) take ~75 GB for ~27 GB of content. The
  8 min pull comes from decompression, not from the network.
- Image generation time depends on the free RAM (threshold ~9-10 GiB: the GGUF weights are
  memory-mapped), which matters when choosing `g4dn.xlarge` or `g4dn.2xlarge`.
- Kaggle: without *Accelerator* and *Internet*, `nvidia-smi` is missing and `github.com` does not
  resolve. A `%%bash` cell with `set -e` must never be pasted into a terminal.
