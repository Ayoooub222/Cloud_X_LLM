#!/usr/bin/env bash
# Kaggle T4 benchmark: vision (fictitious quote) then audio (LibriSpeech clip with its reference, WER).
# Run from a clone of the repository, in a notebook with Accelerator GPU T4 and Internet on:
#   bash /tmp/repo/scripts/kaggle_t4_bench.sh
# PHASES selects the phases (default "vision audio"), e.g. PHASES=audio to replay audio only; records
# already in /kaggle/working/out are kept.
# Everything lands in /kaggle/working/out, archived at the end into /kaggle/working/kaggle-t4.tar.gz.
# Lines starting with "==" are the progress summary; the full log is kept by the caller (tee).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="/kaggle/working/out"
ARCHIVE="/kaggle/working/kaggle-t4.tar.gz"
SERVER_ENV="/tmp/vllm-env"
CLIENT_ENV="/tmp/client-env"
VLLM_VERSION="0.29.0"
CU129_INDEX="https://wheels.vllm.ai/0.29.0/cu129"
# Audio decoding on the server, same versions as docker/audio.Dockerfile (vllm[audio] has no librosa)
LIBROSA_VERSION="1.0.0"
SOUNDFILE_VERSION="0.14.0"
BENCH_ENV_LABEL="kaggle-t4"
READY_TIMEOUT_S=600
READY_POLL_S=5
METRICS_PERIOD_S=0.5
REPEATS=3
PHASES="${PHASES:-vision audio}"

VISION_MODEL="Qwen/Qwen3-VL-2B-Instruct"
VISION_PORT=8000
VISION_PDF="inputs/vision/devis_fictif.pdf"
VISION_PDF_SHA256="e89c8a1be152a80360b4f359e77af05923d7f6b521bfd8854d98d85f0e5e60d4"
VISION_SCHEMA="conf/schemas/extraction_page.json"
VISION_EXPECTED="inputs/vision/devis_fictif.expected.json"
VISION_CONCURRENCY="1 2 4"

AUDIO_MODEL="openai/whisper-large-v3-turbo"
AUDIO_PORT=8001
AUDIO_CONCURRENCY="1 2 4 8"
AUDIO_FILE="inputs/audio/librispeech_1272-141231_66s.flac"
AUDIO_FILE_SHA256="a1c714aef65c1bbc36c02d676aca3905962f357ff551c37ae39b44d3639a0602"
AUDIO_REFERENCE="inputs/audio/librispeech_1272-141231_66s.txt"
AUDIO_TAG_PREFIX="ls66"   # same tags as the local LibriSpeech series, compared with --by-env

export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID VLLM_USE_V2_MODEL_RUNNER=0
export BENCH_ENV="$BENCH_ENV_LABEL" BENCH_DIR="$OUT_DIR/results/benchmark" VLLM_API_KEY=EMPTY
unset MLFLOW_TRACKING_URI

log() { echo "== $(date +%H:%M:%S) $*"; }
fail() { echo "== ERROR: $*" >&2; exit 1; }

check_platform() {
  command -v nvidia-smi > /dev/null || fail "no GPU: Session options > Accelerator > GPU T4"
  curl -sfI https://github.com > /dev/null || fail "no internet: Session options > Internet on"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv | tee "$OUT_DIR/gpu.csv"
  echo "$VISION_PDF_SHA256  $REPO_DIR/$VISION_PDF" | sha256sum -c --quiet || fail "unexpected PDF checksum"
  echo "$AUDIO_FILE_SHA256  $REPO_DIR/$AUDIO_FILE" | sha256sum -c --quiet || fail "unexpected audio checksum"
  log "platform ok, PDF and audio checksums ok"
}

driver_cuda_major() {
  nvidia-smi | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$'
}

install_server() {
  pip install -q uv
  rm -rf "$SERVER_ENV"
  uv venv -q "$SERVER_ENV" --python 3.12
  local py="$SERVER_ENV/bin/python" major
  major="$(driver_cuda_major)"
  log "driver CUDA major: $major"
  local pkgs=("vllm[audio]==$VLLM_VERSION" "librosa==$LIBROSA_VERSION" "soundfile==$SOUNDFILE_VERSION")
  if test "$major" -ge 13; then
    uv pip install -q --python "$py" "${pkgs[@]}" --torch-backend=auto
  else
    uv pip install -q --python "$py" "${pkgs[@]}" --extra-index-url "$CU129_INDEX" --torch-backend=cu129
  fi
  uv pip freeze --python "$py" > "$OUT_DIR/freeze-server.txt"  # written before any check can fail
  "$py" -c 'import librosa, soundfile, av, soxr' || fail "audio decoding modules missing in the server venv"
  "$py" -c 'import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))' \
    | tee "$OUT_DIR/versions-server.txt"
  log "server venv ready: $(cat "$OUT_DIR/versions-server.txt")"
}

install_client() {
  rm -rf "$CLIENT_ENV"
  uv venv -q "$CLIENT_ENV" --python 3.13
  uv pip install -q --python "$CLIENT_ENV/bin/python" -r "$REPO_DIR/requirements.txt"
  uv pip freeze --python "$CLIENT_ENV/bin/python" > "$OUT_DIR/freeze-client.txt"
  log "client venv ready"
}

wait_ready() {
  local port="$1" waited=0
  while ! curl -sf "localhost:$port/health" > /dev/null; do
    pgrep -f 'vllm serve' > /dev/null || return 1
    test "$waited" -lt "$READY_TIMEOUT_S" || return 1
    sleep "$READY_POLL_S"
    waited=$((waited + READY_POLL_S))
  done
  log "server on port $port ready after ~${waited}s"
}

start_server() {
  local name="$1" port="$2"
  shift 2
  # No API key on the server, as in compose and k3s: vllm serve reads VLLM_API_KEY and would then
  # require it (HTTP 401). The client keeps it, the OpenAI SDK refuses an empty key.
  setsid nohup env -u VLLM_API_KEY "$SERVER_ENV/bin/vllm" serve "$@" --port "$port" > "$OUT_DIR/vllm-$name.log" 2>&1 &
  if ! wait_ready "$port"; then
    tail -n 40 "$OUT_DIR/vllm-$name.log"
    stop_server
    return 1
  fi
  grep -i -E 'attention backend|KV cache|dtype' "$OUT_DIR/vllm-$name.log" | head -n 10 || true
}

stop_server() {
  pkill -f 'vllm serve' || true
  local i
  for i in $(seq 1 30); do
    pgrep -f 'vllm serve' > /dev/null || break
    sleep 2
  done
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

# Scheduler state every METRICS_PERIOD_S: explains saturation (running, waiting, KV cache use)
sample_metrics() {
  local port="$1" out="$2"
  while true; do
    curl -s "localhost:$port/metrics" \
      | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc)' \
      | sed "s/^/$(date +%s.%N) /" >> "$out" || true
    sleep "$METRICS_PERIOD_S"
  done
}

client() {
  (cd "$REPO_DIR" && "$CLIENT_ENV/bin/python" -m "$@")
}

bench_vision() {
  local args=("$VISION_PDF" "$VISION_SCHEMA" --expected "$VISION_EXPECTED") c rep
  client src.runner.runner_vision "${args[@]}" --tag warmup
  for rep in $(seq 1 "$REPEATS"); do
    for c in $VISION_CONCURRENCY; do
      client src.runner.runner_vision "${args[@]}" --concurrency "$c" --tag "conc-c$c"
    done
  done
}

bench_audio() {
  local out="$OUT_DIR/outputs/audio" c rep
  local args=("$AUDIO_FILE" --reference "$AUDIO_REFERENCE" --language en --out "$out")
  client src.runner.runner_audio "${args[@]}" --tag "$AUDIO_TAG_PREFIX-warmup"
  for rep in $(seq 1 "$REPEATS"); do
    for c in $AUDIO_CONCURRENCY; do
      client src.runner.runner_audio "${args[@]}" --concurrency "$c" --tag "$AUDIO_TAG_PREFIX-c$c"
    done
  done
}

# Weights are fetched first, so the ready timeout only covers loading and compilation
download_model() {
  "$SERVER_ENV/bin/python" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" "$1" \
    > /dev/null
}

run_phase() {
  local name="$1" port="$2" bench="$3" model="$4"
  shift 3
  local sampler
  log "$name: downloading $model"
  download_model "$model" || { log "$name: download failed, phase skipped"; return 0; }
  log "$name: starting server"
  if ! start_server "$name" "$port" "$@"; then
    log "$name: server not ready after ${READY_TIMEOUT_S}s, phase skipped"
    return 0
  fi
  sample_metrics "$port" "$OUT_DIR/metrics-$name.log" &
  sampler=$!
  log "$name: benchmark"
  $bench || log "$name: a run failed (records kept)"
  kill "$sampler" || true
  stop_server
  log "$name: done"
}

archive() {
  if test -d "$REPO_DIR/outputs/vision"; then
    mkdir -p "$OUT_DIR/outputs"
    cp -r "$REPO_DIR/outputs/vision" "$OUT_DIR/outputs/"
  fi
  tar czf "$ARCHIVE" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
  log "records: $(find "$BENCH_DIR" -name '*.json' | wc -l), archive: $(du -h "$ARCHIVE" | cut -f1) $ARCHIVE"
}

has_phase() {
  case " $PHASES " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

main() {
  mkdir -p "$OUT_DIR" "$BENCH_DIR"
  log "phases: $PHASES"
  check_platform
  install_server
  install_client
  if has_phase vision; then
    run_phase vision "$VISION_PORT" bench_vision "$VISION_MODEL" --served-model-name "$VISION_MODEL" \
      --dtype float16 --mm-encoder-attn-backend TORCH_SDPA --gpu-memory-utilization 0.75 --max-model-len 8192 \
      --max-num-seqs 4 --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"max_pixels":1003520}'
  fi
  if has_phase audio; then
    run_phase audio "$AUDIO_PORT" bench_audio "$AUDIO_MODEL" --served-model-name "$AUDIO_MODEL" \
      --dtype float16 --gpu-memory-utilization 0.6 --max-num-seqs 8
  fi
  archive
}


main "$@"
