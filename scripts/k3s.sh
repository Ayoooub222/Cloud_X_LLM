#!/usr/bin/env bash
# Start or stop k3s. Never run k3s and the docker compose GPU profiles at the same time (12 GB RAM).
# Note: "systemctl stop k3s" leaves pods running; k3s-killall.sh really stops them.
set -euo pipefail

case "${1:-}" in
  start)
    docker compose rm --stop --force vision image audio
    sudo systemctl start k3s
    ;;
  stop)
    sudo systemctl stop k3s
    sudo /usr/local/bin/k3s-killall.sh
    ;;
  *)
    echo "Usage: $0 start|stop"
    exit 1
    ;;
esac
free -h
