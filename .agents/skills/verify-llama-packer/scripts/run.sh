#!/usr/bin/env bash
# Pack a fixture, validate the config with llama-swap, serve it on a free
# port, and prove the model list.  Usage: run.sh [evidence-dir]
#
# IO budget: the fixture is a few KB.  Scratch (including the packer's cache
# and fit-params failure logs) lives under .verify/.scratch and is removed on
# exit, signal, or the next run's stale sweep.  Evidence is text only.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${LLAMA_PACKER_ROOT:-$(git -C "$HERE" rev-parse --show-toplevel)}
RUN=${1:-$ROOT/.verify/$(date -u +%Y%m%dT%H%M%SZ)}
case $RUN in /*) ;; *) RUN=$PWD/$RUN ;; esac
SCRATCH_ROOT=$ROOT/.verify/.scratch
SCRATCH=$SCRATCH_ROOT/$$-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$SCRATCH" "$RUN"
# Scratch is disposable: clear anything a SIGKILLed run left behind.
rm -rf "${SCRATCH_ROOT:?}"/*

SWAP=${LLAMA_SWAP:-$(command -v llama-swap || true)}
[ -n "$SWAP" ] || SWAP=/opt/llama/tools/llama-swap
SERVER=${LLAMA_SERVER:-$(command -v llama-server || true)}
SERVER_TIMEOUT=${VERIFY_SERVER_TIMEOUT:-20}
[ -n "$SERVER" ] || SERVER=/opt/llama/tools/llama-server
PID=""
cleanup() {
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
  [ -n "$PID" ] && wait "$PID" 2>/dev/null || true
  rm -rf "$SCRATCH"
}
trap cleanup EXIT INT TERM HUP

"$HERE/doctor.sh" >"$RUN/doctor.txt"
cat "$RUN/doctor.txt"

# Keep every packer artifact, including fit-params crash logs, in scratch.
export LLAMA_PACKER_CACHE_DIR=$SCRATCH/cache
python3 "$HERE/make-fixture.py" "$SCRATCH" >"$RUN/fixture.txt"
cat "$RUN/fixture.txt"

(
  cd "$SCRATCH"
  timeout 120 env PYTHONPATH="$ROOT" python3 -m llama_packer \
    --models-dir models --output "$RUN/config.yaml" \
    --llama-server "$SERVER" --vram 24576 \
    >"$RUN/packer.out" 2>"$RUN/packer.err"
)

"$SWAP" -validate -config "$RUN/config.yaml" | tee "$RUN/validate.txt"
grep -q "config is valid: 1 model(s)" "$RUN/validate.txt"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
# The timeout bounds the server even if this script is SIGKILLed and its
# trap never runs; timeout forwards the TERM to the child.
timeout --signal=TERM --kill-after=5 "$SERVER_TIMEOUT" \
  "$SWAP" -config "$RUN/config.yaml" -listen "127.0.0.1:$PORT" \
  >"$RUN/swap.log" 2>&1 &
PID=$!
ready=0
for _ in $(seq 1 50); do
  if timeout 2 curl -sf --max-time 1 \
      "http://127.0.0.1:$PORT/v1/models" >"$RUN/models.json"; then
    ready=1; break
  fi
  sleep 0.1
done
[ "$ready" = 1 ] || { echo "llama-swap did not answer on :$PORT" >&2; exit 1; }
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
PID=""
grep -q '"id":"m"' "$RUN/models.json"

echo "evidence: $RUN"
