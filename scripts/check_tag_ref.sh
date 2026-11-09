#!/usr/bin/env bash
# Check that a release tag matches the default git_ref deployed by Terraform.
# Usage: scripts/check_tag_ref.sh v0.4.0   (CI passes github.ref_name)
set -euo pipefail

TAG="${1:?usage: check_tag_ref.sh <tag>}"
VARIABLES_FILE="$(dirname "$0")/../infra/aws/variables.tf"

DEFAULT_REF=$(awk '/^variable "git_ref"/ { found = 1 } found && $1 == "default" { gsub(/"/, "", $3); print $3; exit }' "$VARIABLES_FILE")

if test -z "$DEFAULT_REF"; then
  echo "git_ref default not found in $VARIABLES_FILE" >&2
  exit 1
fi
if test "$TAG" != "$DEFAULT_REF"; then
  echo "Tag $TAG does not match git_ref default $DEFAULT_REF in infra/aws/variables.tf" >&2
  echo "Bump git_ref in a commit, then tag that commit." >&2
  exit 1
fi
echo "Tag $TAG matches the Terraform git_ref default"
