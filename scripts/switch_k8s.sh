#!/usr/bin/env bash
# Usage: scripts/switch_k8s.sh <vision|image|audio|none> [--mlflow|--no-mlflow]
# Deploy the ai-models chart with only the requested model scaled to 1,
# then wait until it is ready. Prints the time to ready.
# MLflow: --mlflow enables it, --no-mlflow disables it, no flag keeps the current state.
# The MLflow volume is never deleted (see templates/mlflow.yaml).
set -euo pipefail

RELEASE="ai-models"
NAMESPACE="ai-models"
CHART="deploy/helm/ai-models"
MODELS="vision image audio"
MODEL_SELECTOR="app.kubernetes.io/name in (vision,image,audio)"
TIMEOUT="15m"
DELETE_TIMEOUT="120s"
USAGE="Usage: $0 <vision|image|audio|none> [--mlflow|--no-mlflow]"

TARGET="${1:-}"
case "$TARGET" in
  vision|image|audio|none) ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."
set -a
source .env
set +a

mlflow_deployed() {
  kubectl -n "$NAMESPACE" get deployment mlflow >/dev/null 2>&1
}

case "${2:-}" in
  --mlflow) MLFLOW="true" ;;
  --no-mlflow) MLFLOW="false" ;;
  "") if mlflow_deployed; then MLFLOW="true"; else MLFLOW="false"; fi ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac

ARGS=(
  --set-string "modelsDir=$(realpath "$MODELS_DIR")"
  --set-string "models.audio.env.AUDIO_MODEL_DIR=$AUDIO_MODEL_DIR"
  --set-string "models.image.env.IMAGE_DIFFUSION=$IMAGE_DIFFUSION"
  --set-string "models.image.env.IMAGE_VAE=$IMAGE_VAE"
  --set-string "models.image.env.IMAGE_LLM=$IMAGE_LLM"
  --set "mlflow.enabled=$MLFLOW"
)
for model in $MODELS; do
  replicas=0
  if [ "$model" = "$TARGET" ]; then replicas=1; fi
  ARGS+=(--set "models.$model.replicas=$replicas")
done

START=$(date +%s)
helm upgrade --install "$RELEASE" "$CHART" --namespace "$NAMESPACE" --create-namespace "${ARGS[@]}"

if [ "$TARGET" = "none" ]; then
  # Wait until model pods are gone, so the GPU is free for compose.
  kubectl -n "$NAMESPACE" wait --for=delete pod -l "$MODEL_SELECTOR" --timeout="$DELETE_TIMEOUT" 2>/dev/null || true
  echo "GPU released after $(( $(date +%s) - START ))s"
else
  kubectl -n "$NAMESPACE" rollout status "deployment/$TARGET" --timeout="$TIMEOUT"
  echo "READY after $(( $(date +%s) - START ))s"
fi

if [ "$MLFLOW" = "true" ]; then
  kubectl -n "$NAMESPACE" rollout status deployment/mlflow --timeout="$TIMEOUT"
fi
echo "mlflow.enabled=$MLFLOW"
kubectl -n "$NAMESPACE" get pods
