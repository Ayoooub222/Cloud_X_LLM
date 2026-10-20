#!/usr/bin/env bash
# Usage: bash scripts/wait_ready.sh <vision|image|audio|mlflow>
# Poll the server health endpoint until it answers (max 10 min). Prints the startup time.
set -euo pipefail

case "${1:-}" in
  vision) URL=http://localhost:8000/health ;;
  image)  URL=http://localhost:8082/sdcpp/v1/capabilities ;;
  audio)  URL=http://localhost:8001/health ;;
  mlflow) URL=http://localhost:5000/health ;;
  *) echo "Usage: $0 <vision|image|audio|mlflow>" >&2; exit 1 ;;
esac

echo "Waiting for $URL ..."
START=$(date +%s)
for _ in $(seq 1 120); do
  if curl -sf -o /dev/null "$URL"; then
    echo "READY after $(( $(date +%s) - START ))s"
    exit 0
  fi
  sleep 5
done
echo "Server not ready after 10 min" >&2
exit 1
