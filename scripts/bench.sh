#!/usr/bin/env bash
# Usage: scripts/bench.sh <k3s|compose> <audio|audio-short|vision|image> [coldstart_runs]
# Same benchmark in k3s or docker compose, with the same tags, compared with summarize --by-env.
# 1. cold starts from "no model running": time to ready + first real request (coldstart.py);
#    with coldstart_runs=0 the model is started once and nothing is recorded;
# 2. warmup requests (tag "warmup", discarded), then the warm series.
# GPU and power state are written at start and end to results/benchmark/sessions/.
# MLflow: used only in k3s for audio and vision, if /health answers 200. Otherwise JSON only
# (import later: runs already logged are skipped through the bench_run_id tag).
# Vision needs VISION_PDF and VISION_SCHEMA. Image needs MIN_AVAILABLE_MIB of available RAM.
set -euo pipefail

MLFLOW_URL="http://localhost:5000"
COMPOSE_GPU_PATTERN="test-project-(vision|image|audio)-1"
ALL_GPU_PROFILES="--profile vision --profile image --profile audio"
K3S_LABEL="local-k3s-rtx3070laptop"
COMPOSE_LABEL="local-wsl2-rtx3070laptop"
MIN_AVAILABLE_MIB="${MIN_AVAILABLE_MIB:-9500}"
DEFAULT_COLD_RUNS=3
WARM_REPEATS=3
READY_TIMEOUT_S=900
READY_POLL_S=1
SESSION_DIR="results/benchmark/sessions"

AUDIO_DIR="data/audio"
AUDIO_SINGLES_FULL="11s 66s 308s"
AUDIO_SINGLES_SHORT="11s 308s"
AUDIO_CONCURRENCY_FULL="1 2 4 8"
AUDIO_CONCURRENCY_SHORT="1 8"

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

USAGE="Usage: $0 <k3s|compose> <audio|audio-short|vision|image> [coldstart_runs]"
ENV_NAME="${1:-}"
SUITE="${2:-}"
COLD_RUNS="${3:-$DEFAULT_COLD_RUNS}"

case "$ENV_NAME" in
  k3s|compose) ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac
case "$SUITE" in
  audio|audio-short) TASK="audio" ;;
  vision|image) TASK="$SUITE" ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."

# The label is forced: a BENCH_ENV left over from another session would mislabel the runs.
if [ "$ENV_NAME" = "k3s" ]; then
  export BENCH_ENV="$K3S_LABEL"
  export IMAGE_LOG_SOURCE="k8s"
else
  export BENCH_ENV="$COMPOSE_LABEL"
  export IMAGE_LOG_SOURCE="docker"
fi

# In k3s, image runs without MLflow to keep RAM for the page cache (text encoder).
SWITCH_FLAG=""
if [ "$ENV_NAME" = "k3s" ] && [ "$TASK" = "image" ]; then
  SWITCH_FLAG="--no-mlflow"
fi

SESSION_FILE="$SESSION_DIR/$(date '+%Y%m%d-%H%M%S')-${ENV_NAME}-${SUITE}.txt"

log() { echo "=== $(date '+%H:%M:%S') $*"; }

# A failed request is recorded by the runner itself: warn and keep going.
run() {
  "$@" || echo "[warn] failed: $*" >&2
}

available_mib() {
  free -m | awk '/^Mem:/ {print $7}'
}

http_code() {
  curl -s -o /dev/null -w '%{http_code}' "$1" || true
}

preflight() {
  if [ "$ENV_NAME" = "k3s" ]; then
    if ! systemctl is-active --quiet k3s; then
      echo "k3s is not running: scripts/k3s.sh start" >&2
      exit 1
    fi
    if docker ps --format '{{.Names}}' | grep -qE "$COMPOSE_GPU_PATTERN"; then
      echo "A compose GPU service is running: stop it first." >&2
      exit 1
    fi
  else
    if systemctl is-active --quiet k3s; then
      echo "k3s is running: scripts/k3s.sh stop" >&2
      exit 1
    fi
  fi
  if [ "$TASK" = "vision" ]; then
    : "${VISION_PDF:?export VISION_PDF (reference PDF)}"
    : "${VISION_SCHEMA:?export VISION_SCHEMA (reference schema)}"
    local f
    for f in "$VISION_PDF" "$VISION_SCHEMA"; do
      if [ ! -f "$f" ]; then
        echo "File not found: '$f' (check for trailing spaces)" >&2
        exit 1
      fi
    done
  fi
}

setup_mlflow() {
  if [ "$ENV_NAME" = "compose" ] || [ "$TASK" = "image" ]; then
    export MLFLOW_TRACKING_URI=""
    echo "MLflow disabled for this run: JSON only, import later."
    return
  fi
  if [ "$(http_code "$MLFLOW_URL/health")" = "200" ]; then
    export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$MLFLOW_URL}"
    echo "MLflow enabled: $MLFLOW_TRACKING_URI"
  else
    export MLFLOW_TRACKING_URI=""
    echo "MLflow not reachable: JSON only, import later."
  fi
}

on_ac_power() {
  # Win32_Battery.BatteryStatus: 2 = on AC power, 1 = on battery.
  powershell.exe -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' 2>/dev/null \
    | tr -d '\r' || echo "unknown"
}

record_state() {
  mkdir -p "$SESSION_DIR"
  {
    echo "[$1] $(date '+%Y-%m-%d %H:%M:%S') env=$BENCH_ENV suite=$SUITE cold_runs=$COLD_RUNS"
    nvidia-smi --query-gpu=driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm \
      --format=csv,noheader || echo "nvidia-smi: unavailable"
    powercfg.exe /getactivescheme 2>/dev/null | tr -d '\r' || echo "power scheme: unavailable"
    echo "battery_status=$(on_ac_power)"
    echo "available_mib=$(available_mib)"
  } >> "$SESSION_FILE"
  echo "GPU/power state ($1) written to $SESSION_FILE"
}

check_memory() {
  local avail
  avail=$(available_mib)
  echo "available RAM: ${avail} MiB"
  if [ "$TASK" = "image" ] && [ "$avail" -lt "$MIN_AVAILABLE_MIB" ]; then
    echo "Below ${MIN_AVAILABLE_MIB} MiB: sd-server would re-read the text encoder from disk." >&2
    exit 1
  fi
}

ready_url() {
  case "$TASK" in
    vision) echo "http://localhost:8000/health" ;;
    audio) echo "http://localhost:8001/health" ;;
    image) echo "http://localhost:8082/" ;;
  esac
}

# vLLM: /health must answer 200. sd-server: any HTTP answer means the server is listening.
is_ready() {
  local url code
  url=$(ready_url)
  code=$(http_code "$url")
  if test "$TASK" = "image"; then
    test "$code" != "000"
    return
  fi
  test "$code" = "200"
}

wait_compose_ready() {
  local start now
  start=$(date +%s)
  until is_ready; do
    now=$(date +%s)
    if [ $(( now - start )) -ge "$READY_TIMEOUT_S" ]; then
      echo "Timeout: $(ready_url) not ready after ${READY_TIMEOUT_S}s" >&2
      exit 1
    fi
    sleep "$READY_POLL_S"
  done
}


model_down() {
  if [ "$ENV_NAME" = "k3s" ]; then
    scripts/switch_k8s.sh none $SWITCH_FLAG
  else
    docker compose $ALL_GPU_PROFILES rm --stop --force vision image audio
  fi
}

model_up() {
  if [ "$ENV_NAME" = "k3s" ]; then
    scripts/switch_k8s.sh "$TASK" $SWITCH_FLAG
  else
    docker compose --profile "$TASK" up -d "$TASK"
    wait_compose_ready
  fi
}

first_request() {
  local index="$1"
  case "$TASK" in
    audio) run python -m src.runner.runner_audio "$AUDIO_DIR/jfk.flac" --tag "$FIRST_TAG" ;;
    vision) run python -m src.runner.runner_vision "$VISION_PDF" "$VISION_SCHEMA" --tag "$FIRST_TAG" ;;
    image) run python -m src.runner.runner_image "${COLD_PROMPTS[$(( index % ${#COLD_PROMPTS[@]} ))]}" --tag "$FIRST_TAG" ;;
  esac
}

coldstarts() {
  local i since start t_ready
  if [ "$COLD_RUNS" -eq 0 ]; then
    log "no cold start recorded: starting $TASK once"
    model_down
    check_memory
    model_up
    return
  fi
  for i in $(seq 0 $(( COLD_RUNS - 1 ))); do
    log "cold start $(( i + 1 ))/$COLD_RUNS: $TASK ($ENV_NAME)"
    model_down
    check_memory
    since=$(date '+%Y-%m-%d %H:%M:%S')
    start=$(date +%s)
    model_up
    t_ready=$(( $(date +%s) - start ))
    echo "ready after ${t_ready}s"
    first_request "$i"
    run python -m src.runner.coldstart "$TASK" --t-ready "$t_ready" --since "$since"
  done
}

audio_file() {
  case "$1" in
    11s) echo "$AUDIO_DIR/jfk.flac" ;;
    66s) echo "$AUDIO_DIR/jfk_x6.wav" ;;
    308s) echo "$AUDIO_DIR/jfk_x28.flac" ;;
  esac
}

warm_audio() {
  local singles="$1" concurrency="$2" rep name c
  log "audio warmup"
  run python -m src.runner.runner_audio "$(audio_file 11s)" --tag "$WARMUP_TAG"
  run python -m src.runner.runner_audio "$(audio_file 66s)" --tag "$WARMUP_TAG"
  for rep in $(seq "$WARM_REPEATS"); do
    for name in $singles; do
      log "audio single $name, repeat $rep/$WARM_REPEATS"
      run python -m src.runner.runner_audio "$(audio_file "$name")" --tag "${TAG_SINGLE}-${name}"
    done
  done
  for c in $concurrency; do
    for rep in $(seq "$WARM_REPEATS"); do
      log "audio concurrency $c, repeat $rep/$WARM_REPEATS"
      run python -m src.runner.runner_audio "$(audio_file 66s)" --concurrency "$c" --tag "${TAG_CONC}${c}"
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
  for prompt in "${IMAGE_PROMPTS[@]}"; do
    log "image: $prompt"
    run python -m src.runner.runner_image "$prompt" --tag "$TAG_IMAGE"
  done
}

preflight
setup_mlflow
record_state start
coldstarts
case "$SUITE" in
  audio) warm_audio "$AUDIO_SINGLES_FULL" "$AUDIO_CONCURRENCY_FULL" ;;
  audio-short) warm_audio "$AUDIO_SINGLES_SHORT" "$AUDIO_CONCURRENCY_SHORT" ;;
  vision) warm_vision ;;
  image) warm_image ;;
esac
record_state end
log "done. Compare with: python -m src.runner.summarize $TASK --by-env --since $(date '+%Y-%m-%d')"
