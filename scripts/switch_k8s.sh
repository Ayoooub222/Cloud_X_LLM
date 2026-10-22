#!/usr/bin/env bash
# Usage: scripts/switch_k8s.sh <vision|image|audio|none> [--mlflow]
# Deploy the ai-models chart with only the requested model scaled to 1,
# then wait until it is ready. Prints the time to ready.
set -euo pipefail

RELEASE="ai-models"
NAMESPACE="ai-models"
CHART="deploy/helm/ai-models"
MODELS="vision image audio"
TIMEOUT="15m"

TARGET="${1:-}"
case "$TARGET" in
  vision|image|audio|none) ;;
  *) echo "Usage: $0 <vision|image|audio|none> [--mlflow]" >&2; exit 1 ;;
esac
MLFLOW="false"
if [ "${2:-}" = "--mlflow" ]; then MLFLOW="true"; fi

cd "$(dirname "$0")/.."
set -a
source .env
set +a

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

if [ "$TARGET" != "none" ]; then
  kubectl -n "$NAMESPACE" rollout status "deployment/$TARGET" --timeout="$TIMEOUT"
  echo "READY after $(( $(date +%s) - START ))s"
fi
kubectl -n "$NAMESPACE" get pods
