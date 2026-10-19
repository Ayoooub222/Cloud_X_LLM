#!/usr/bin/env bash
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.29.0}"
VLLM_API_KEY="${VLLM_API_KEY:-token-local-dev}"
RUNNER="${RUNNER:-v2}"          # v2 = pinned memory (rapide) | v1 = secours
NAME=vllm-qwen3vl

# Nettoie un conteneur précédent (sinon conflit de nom)
sudo docker rm -f "$NAME" >/dev/null 2>&1 || true

if [[ "$RUNNER" == "v2" ]]; then
  RUNNER_ENV=(-e VLLM_WSL2_ENABLE_PIN_MEMORY=1)
else
  RUNNER_ENV=(-e VLLM_USE_V2_MODEL_RUNNER=0)
fi

HF_ENV=()
[[ -n "${HF_TOKEN:-}" ]] && HF_ENV=(-e HF_TOKEN="$HF_TOKEN")

# Pas de --rm : si le moteur crashe, `docker logs` reste disponible
sudo docker run -d --name "$NAME" --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "${HF_ENV[@]}" "${RUNNER_ENV[@]}" \
  -p 127.0.0.1:8000:8000 \
  "$VLLM_IMAGE" \
  Qwen/Qwen3-VL-2B-Instruct \
  --served-model-name qwen3-vl-2b \
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
  --generation-config vllm \
  --api-key "$VLLM_API_KEY"

# Attente active jusqu'à ce que l'API réponde (ou que le conteneur meure)
echo "Démarrage ($RUNNER)..."
for i in $(seq 1 120); do
  if ! sudo docker ps -q -f name="^${NAME}$" | grep -q .; then
    echo "Conteneur mort. Dernières lignes :"; sudo docker logs --tail 40 "$NAME"; exit 1
  fi
  if curl -sf -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "Prêt en ~$((i*2))s"; exit 0
  fi
  sleep 2
done
echo "Timeout"; sudo docker logs --tail 40 "$NAME"; exit 1
