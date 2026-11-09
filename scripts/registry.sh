#!/usr/bin/env bash
# Local OCI registry used by k3s to pull project images.
# Usage: scripts/registry.sh start | stop | list | push IMAGE:TAG
set -euo pipefail

NAME="registry"
REGISTRY_IMAGE="registry:3.0.0"
PORT="5001"
HOST="localhost:${PORT}"
DATA_DIR="${HOME}/.local/share/registry"

start() {
  if docker container inspect "${NAME}" >/dev/null 2>&1; then
    docker start "${NAME}" >/dev/null
  else
    mkdir -p "${DATA_DIR}"
    docker run -d --name "${NAME}" --restart unless-stopped \
      -p "127.0.0.1:${PORT}:5000" \
      -v "${DATA_DIR}:/var/lib/registry" \
      "${REGISTRY_IMAGE}" >/dev/null
  fi
  echo "registry running on ${HOST}"
}

stop() {
  docker stop "${NAME}" >/dev/null
  echo "registry stopped"
}

list() {
  curl -fsS "http://${HOST}/v2/_catalog"
  echo
}

push() {
  local src="$1"
  local dst="${HOST}/${src}"
  docker tag "${src}" "${dst}"
  time docker push "${dst}"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  list) list ;;
  push) push "${2:?missing image name}" ;;
  *) echo "usage: $0 start|stop|list|push IMAGE:TAG" >&2; exit 1 ;;
esac
