#!/usr/bin/env bash
# Remove verification scratch.  Evidence survives unless --all.
#   clean.sh          scratch only (fixtures, caches, crash logs)
#   clean.sh --all    scratch and every evidence run
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${LLAMA_PACKER_ROOT:-$(git -C "$HERE" rev-parse --show-toplevel)}
BASE=$ROOT/.verify
rm -rf "$BASE/.scratch"
if [ "${1:-}" = "--all" ]; then
  find "$BASE" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "removed $BASE"
else
  echo "removed $BASE/.scratch; evidence kept under $BASE"
fi
