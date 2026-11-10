#!/usr/bin/env bash
# Read-only: is this checkout worth driving?
set -uo pipefail
ROOT=${LLAMA_PACKER_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
fail=0

say() { printf '%-22s %s\n' "$1" "$2"; }

if python3 -c "import llama_packer" 2>/dev/null; then
  say "python module" "ok ($(python3 -c 'import llama_packer; print(llama_packer.__version__)'))"
else
  say "python module" "MISSING: run from the repo with PYTHONPATH=$ROOT"
  fail=1
fi

if python3 -c "import gguf" 2>/dev/null; then
  say "gguf helper" "ok"
else
  say "gguf helper" "MISSING: fixture writer needs it (uv run)"
  fail=1
fi

SWAP=${LLAMA_SWAP:-$(command -v llama-swap || true)}
[ -n "$SWAP" ] || SWAP=/opt/llama/tools/llama-swap
if "$SWAP" -version >/dev/null 2>&1; then
  say "llama-swap" "ok ($("$SWAP" -version 2>&1 | head -1))"
else
  say "llama-swap" "MISSING: set LLAMA_SWAP"
  fail=1
fi

SERVER=${LLAMA_SERVER:-$(command -v llama-server || true)}
[ -n "$SERVER" ] || SERVER=/opt/llama/tools/llama-server
FIT=$(dirname "$(readlink -f "$SERVER" 2>/dev/null || echo "$SERVER")")/llama-fit-params
if [ -x "$FIT" ]; then
  say "llama-server" "ok ($SERVER)"
  say "llama-fit-params" "ok ($FIT)"
else
  say "llama-server" "MISSING or no sibling fit-params: set LLAMA_SERVER"
  fail=1
fi

exit "$fail"
