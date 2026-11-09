#!/usr/bin/env bash
# Install the latest Terraform CLI in ~/.local/bin (no sudo, no apt repository).
set -euo pipefail

INSTALL_DIR="$HOME/.local/bin"
ARCH="linux_amd64"
CHECKPOINT_URL="https://checkpoint-api.hashicorp.com/v1/check/terraform"
RELEASES_URL="https://releases.hashicorp.com/terraform"

version=$(curl -fsSL "$CHECKPOINT_URL" | python3 -c 'import json, sys; print(json.load(sys.stdin)["current_version"])')
zip_name="terraform_${version}_${ARCH}.zip"
sums_name="terraform_${version}_SHA256SUMS"

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
cd "$work_dir"

curl -fsSLO "$RELEASES_URL/$version/$zip_name"
curl -fsSLO "$RELEASES_URL/$version/$sums_name"
sha256sum --check --ignore-missing "$sums_name"

python3 -m zipfile -e "$zip_name" .
mkdir -p "$INSTALL_DIR"
install -m 0755 terraform "$INSTALL_DIR/terraform"
echo "Installed Terraform $version in $INSTALL_DIR"

command -v terraform > /dev/null || echo "Warning: $INSTALL_DIR is not in PATH (open a new terminal)"
