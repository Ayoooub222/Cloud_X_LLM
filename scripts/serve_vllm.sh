#!/usr/bin/env bash
set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
NAME="vllm-qwen3vl"

# Remove any previous container
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

sudo docker run -d --name "$NAME" --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  -p 127.0.0.1:8000:8000 \
  "$IMAGE" \
  Qwen/Qwen3-VL-2B-Instruct \
  --served-model-name Qwen3-VL-2B-Instruct \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 6144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2048 \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1310720}' \
  --mm-processor-cache-gb 0 \
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4]}' \
  --generation-config vllm

echo "Starting vLLM..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "Ready after ~$((i * 2))s"
    exit 0
  fi
  sleep 2
done

echo "Timed out. Last logs:"
sudo docker logs --tail 40 "$NAME"
exit 1