#!/usr/bin/env bash
# Import the archive produced by the Kaggle notebook (kaggle-t4.tar.gz) into results/benchmark.
# Usage: scripts/import_kaggle.sh <archive> [extract_dir]
#   extract_dir: new folder for the raw archive content (default: outputs/kaggle-import/<archive date>)
# Records are copied without overwriting anything; set BENCH_DIR to copy them elsewhere.
set -euo pipefail

ARCHIVE="${1:?usage: scripts/import_kaggle.sh <archive> [extract_dir]}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -r "$ARCHIVE" +%Y-%m-%d_%H-%M-%S 2>/dev/null || date +%Y-%m-%d_%H-%M-%S)"
EXTRACT_DIR="${2:-$REPO_DIR/outputs/kaggle-import/$STAMP}"
BENCH_DIR="${BENCH_DIR:-$REPO_DIR/results/benchmark}"
RECORDS_SUBDIR="out/results/benchmark"

test -f "$ARCHIVE" || { echo "archive not found: $ARCHIVE" >&2; exit 1; }
test ! -e "$EXTRACT_DIR" || { echo "already exists, choose another folder: $EXTRACT_DIR" >&2; exit 1; }
if tar tzf "$ARCHIVE" | grep -E '^/|(^|/)\.\.(/|$)'; then
  echo "unsafe paths in the archive, nothing extracted" >&2
  exit 1
fi

mkdir -p "$EXTRACT_DIR"
tar xzf "$ARCHIVE" -C "$EXTRACT_DIR" --no-same-owner --no-same-permissions
find "$EXTRACT_DIR" -type l -delete   # an archive has no reason to carry links

copied=0
skipped=0
while IFS= read -r record; do
  target="$BENCH_DIR/${record#"$EXTRACT_DIR/$RECORDS_SUBDIR/"}"
  if test -e "$target"; then
    skipped=$((skipped + 1))
    continue
  fi
  mkdir -p "$(dirname "$target")"
  cp "$record" "$target"
  copied=$((copied + 1))
done < <(find "$EXTRACT_DIR/$RECORDS_SUBDIR" -type f -name '*.json' | sort)

echo "extracted to $EXTRACT_DIR"
echo "records copied: $copied, already present (left untouched): $skipped"
