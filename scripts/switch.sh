#!/usr/bin/env bash
# Usage: bash scripts/switch.sh <vision|image|audio>
# Stop the GPU services only (MLflow keeps running), start the requested one, follow its logs.
set -euo pipefail

PROFILE="${1:-}"
case "$PROFILE" in
  vision|image|audio) ;;
  *) echo "Usage: $0 <vision|image|audio>" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."
ALL_GPU="--profile vision --profile image --profile audio"

echo "Stopping GPU services..."
docker compose $ALL_GPU rm --stop --force vision image audio

echo "Starting profile '$PROFILE'..."
docker compose --profile "$PROFILE" up -d

echo "Following logs (Ctrl+C to quit, the container keeps running)..."
docker compose --profile "$PROFILE" logs -f
