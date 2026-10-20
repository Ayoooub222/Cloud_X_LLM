#!/usr/bin/env bash
# Usage: bash scripts/switch.sh <vision|image|audio>
# Stop every GPU profile, start the requested one, then follow its logs.
# Ctrl+C only stops following the logs: the container keeps running.
set -euo pipefail

PROFILE="${1:-}"
case "$PROFILE" in
  vision|image|audio) ;;
  *) echo "Usage: $0 <vision|image|audio>" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."  # project root (where docker-compose.yml is)

echo "Stopping all GPU profiles..."
docker compose --profile vision --profile image --profile audio down

echo "Starting profile '$PROFILE'..."
docker compose --profile "$PROFILE" up -d

echo "Following logs (Ctrl+C to quit, the container keeps running)..."
docker compose --profile "$PROFILE" logs -f
