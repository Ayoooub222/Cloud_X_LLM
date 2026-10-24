#!/usr/bin/env bash
# Usage: scripts/bench_k3s.sh <audio|vision|image> [coldstart_runs]
# S2.5 benchmark in k3s: cold starts from "none" (time to Ready + first real request),
# then warmup requests (discarded) and the warm series, same inputs as the compose reference.
# Vision needs VISION_PDF and VISION_SCHEMA (same files as the compose reference).
# Prerequisite: MLflow already Ready (scripts/switch_k8s.sh none --mlflow).
set -euo pipefail

MLFLOW_HEALTH_URL="http://localhost:5000/health"
COMPOSE_GPU_PATTERN="test-project-(vision|image|audio)-1"
MIN_AVAILABLE_MIB=9500
DEFAULT_COLD_RUNS=3
WARM_REPEATS=3
AUDIO_DIR="data/audio"
AUDIO_CONCURRENCY="1 2 4 8"

FIRST_TAG="coldstart-first"
WARMUP_TAG="warmup"
TAG_SINGLE="single"
TAG_CONC="conc-c"
TAG_VISION="vision"
TAG_IMAGE="image-new"

COLD_PROMPTS=(
  "a lighthouse on a cliff at night"
  "a red tractor in a wheat field"
  "a cat sleeping on a pile of books"
)
WARMUP_PROMPTS=(
  "a mountain lake at sunrise"
  "a vintage car in a narrow street"
)
IMAGE_PROMPTS=(
  "a fishing boat in a harbour at dawn"
  "a snowy forest with a small cabin"
  "a market stall full of oranges"
  "a robot watering plants in a greenhouse"
  "an old stone bridge over a river"
)

MODEL="${1:-}"
case "$MODEL" in
  audio|vision|image) ;;
  *) echo "Usage: $0 <audio|vision|image> [coldstart_runs]" >&2; exit 1 ;;
esac
COLD_RUNS="${2:-$DEFAULT_COLD_RUNS}"

cd "$(dirname "$0")/.."
export BENCH_ENV="${BENCH_ENV:-local-k3s-rtx3070laptop}"
export IMAGE_LOG_SOURCE="k8s"

log() { echo "=== $(date '+%H:%M:%S') $*"; }

# A failed request is recorded by the runner itself: warn and keep going.
run() {
  "$@" || echo "[warn] failed: $*" >&2
}

available_mib() {
  free -m | awk '/^Mem:/ {print $7}'
}

preflight() {
  if docker ps --format '{{.Names}}' | grep -qE "$COMPOSE_GPU_PATTERN"; then
    echo "A compose GPU service is running: stop it first." >&2
    exit 1
  fi
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "$MLFLOW_HEALTH_URL" || true)
  if [ "$code" != "200" ]; then
    echo "MLflow not ready (HTTP $code): run scripts/switch_k8s.sh none --mlflow" >&2
    exit 1
  fi
  if [ "$MODEL" = "vision" ]; then
    : "${VISION_PDF:?export VISION_PDF (reference PDF)}"
    : "${VISION_SCHEMA:?export VISION_SCHEMA (reference schema)}"
  fi
  echo "BENCH_ENV=$BENCH_ENV"
}

check_memory() {
  local avail
  avail=$(available_mib)
  echo "available RAM: ${avail} MiB"
  if [ "$MODEL" = "image" ] && [ "$avail" -lt "$MIN_AVAILABLE_MIB" ]; then
    echo "Below ${MIN_AVAILABLE_MIB} MiB: sd-server would re-read the text encoder from disk." >&2
    exit 1
  fi
}

first_request() {
  local index="$1"
  case "$MODEL" in
    audio) run python -m src.runner.runner_audio "$AUDIO_DIR/jfk.flac" --tag "$FIRST_TAG" ;;
    vision) run python -m src.runner.runner_vision "$VISION_PDF" "$VISION_SCHEMA" --tag "$FIRST_TAG" ;;
    image) run python -m src.runner.runner_image "${COLD_PROMPTS[$(( index % ${#COLD_PROMPTS[@]} ))]}" --tag "$FIRST_TAG" ;;
  esac
}

coldstarts() {
  local i since start t_ready
  for i in $(seq 0 $(( COLD_RUNS - 1 ))); do
    log "cold start $(( i + 1 ))/$COLD_RUNS: $MODEL"
    scripts/switch_k8s.sh none
    check_memory
    since=$(date '+%Y-%m-%d %H:%M:%S')
    start=$(date +%s)
    scripts/switch_k8s.sh "$MODEL"
    t_ready=$(( $(date +%s) - start ))
    first_request "$i"
    run python -m src.runner.coldstart "$MODEL" --t-ready "$t_ready" --since "$since"
  done
}

warm_audio() {
  log "audio warmup"
  run python -m src.runner.runner_audio "$AUDIO_DIR/jfk.flac" --tag "$WARMUP_TAG"
  run python -m src.runner.runner_audio "$AUDIO_DIR/jfk_x6.wav" --tag "$WARMUP_TAG"
  local rep c
  for rep in $(seq "$WARM_REPEATS"); do
    log "audio single, repeat $rep/$WARM_REPEATS"
    run python -m src.runner.runner_audio "$AUDIO_DIR/jfk.flac" --tag "${TAG_SINGLE}-11s"
    run python -m src.runner.runner_audio "$AUDIO_DIR/jfk_x6.wav" --tag "${TAG_SINGLE}-66s"
    run python -m src.runner.runner_audio "$AUDIO_DIR/jfk_x28.flac" --tag "${TAG_SINGLE}-308s"
  done
  for c in $AUDIO_CONCURRENCY; do
    for rep in $(seq "$WARM_REPEATS"); do
      log "audio concurrency $c, repeat $rep/$WARM_REPEATS"
      run python -m src.runner.runner_audio "$AUDIO_DIR/jfk_x6.wav" --concurrency "$c" --tag "${TAG_CONC}${c}"
    done
  done
}

warm_vision() {
  log "vision warmup"
  run python -m src.runner.runner_vision "$VISION_PDF" "$VISION_SCHEMA" --tag "$WARMUP_TAG"
  local rep
  for rep in $(seq "$WARM_REPEATS"); do
    log "vision, repeat $rep/$WARM_REPEATS"
    run python -m src.runner.runner_vision "$VISION_PDF" "$VISION_SCHEMA" --tag "$TAG_VISION"
  done
}

warm_image() {
  log "image warmup (new prompts)"
  local prompt
  for prompt in "${WARMUP_PROMPTS[@]}"; do
    run python -m src.runner.runner_image "$prompt" --tag "$WARMUP_TAG"
  done
  check_memory
  for prompt in "${IMAGE_PROMPTS[@]}"; do
    log "image: $prompt"
    run python -m src.runner.runner_image "$prompt" --tag "$TAG_IMAGE"
  done
}

preflight
coldstarts
case "$MODEL" in
  audio) warm_audio ;;
  vision) warm_vision ;;
  image) warm_image ;;
esac
log "done. Compare with: python -m src.runner.summarize $MODEL --by-env"
