#!/usr/bin/env bash
# Usage: scripts/k3s.sh start|stop
# Start or stop k3s. Never run k3s and the docker compose GPU profiles at the same time (12 GB RAM).
# stop: model pods are removed through Helm first, so k3s-killall.sh does not leave them in "Unknown".
# start: waits for the node, then force-deletes pods left in "Unknown" by a previous killall.
set -euo pipefail

NAMESPACE="ai-models"
NODE_TIMEOUT="120s"
ALL_GPU_PROFILES="--profile vision --profile image --profile audio"

cd "$(dirname "$0")/.."

delete_unknown_pods() {
  kubectl -n "$NAMESPACE" get pods --no-headers 2>/dev/null \
    | awk '$3 == "Unknown" {print $1}' \
    | xargs -r kubectl -n "$NAMESPACE" delete pod --force --grace-period=0 \
    || true
}

case "${1:-}" in
  start)
    docker compose $ALL_GPU_PROFILES rm --stop --force vision image audio
    sudo systemctl start k3s
    kubectl wait --for=condition=Ready node --all --timeout="$NODE_TIMEOUT"
    delete_unknown_pods
    ;;
  stop)
    if systemctl is-active --quiet k3s; then
      if kubectl -n argocd get application ai-models > /dev/null 2>&1; then
        # GitOps mode: switch_k8s.sh refuses to run; the replicas follow values-gitops.yaml.
        echo "Argo CD manages ai-models: models left as declared in Git"
      else
        scripts/switch_k8s.sh none
      fi
    fi
    sudo systemctl stop k3s
    sudo /usr/local/bin/k3s-killall.sh
    ;;
  *)
    echo "Usage: $0 start|stop" >&2
    exit 1
    ;;
esac
free -h