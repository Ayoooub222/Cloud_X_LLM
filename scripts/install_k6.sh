#!/usr/bin/env bash
# Install a pinned k6 release in ~/.local/bin (no sudo), checksum verified.
set -euo pipefail

K6_VERSION="v2.3.0"
INSTALL_DIR="$HOME/.local/bin"
BASE_URL="https://github.com/grafana/k6/releases/download/$K6_VERSION"
ARCHIVE="k6-$K6_VERSION-linux-amd64.tar.gz"
SUMS="k6-$K6_VERSION-checksums.txt"

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT
cd "$work_dir"

curl -fsSLO "$BASE_URL/$ARCHIVE"
curl -fsSLO "$BASE_URL/$SUMS"
grep " $ARCHIVE\$" "$SUMS" | sha256sum --check -

tar xzf "$ARCHIVE"
mkdir -p "$INSTALL_DIR"
install -m 0755 "k6-$K6_VERSION-linux-amd64/k6" "$INSTALL_DIR/k6"
"$INSTALL_DIR/k6" version
