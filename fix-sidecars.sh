#!/usr/bin/env bash
# fix-sidecars.sh — repair sidecars after the t2t->chat/qwen3,gemma4 reorg
#
# Why this exists: the reorg deleted the symlinks into $HF_HOME/hub, but the
# sidecars still reference those links in two ways:
#   1. most have NO `model:` field — they matched their .gguf by same-stem
#      convention *in the local directory*, which never consults the hub;
#   2. `mmproj:` values name the old local link filenames, not the snapshot
#      filenames (those happen to self-heal via the *mmproj*.gguf glob
#      fallback when the snapshot holds exactly one mmproj — but Ornith's
#      repo holds two, so exact names are the robust convention).
#
# Usage:  ./fix-sidecars.sh check   # unified diffs only, nothing modified
#         ./fix-sidecars.sh apply   # apply edits
set -euo pipefail

MODE="${1:-check}"
CHAT=/mnt/ai/models/chat

run() { if [ "$MODE" = apply ]; then "$@"; else echo "[check] would: $*"; fi }

# edit FILE OLD NEW — replace first literal occurrence of OLD (no regex), show diff
edit() {
  local file="$1" old="$2" new="$3"
  if ! grep -qF "$old" "$file"; then
    echo "SKIP (pattern absent): $file :: $old" >&2
    return
  fi
  if [ "$MODE" = apply ]; then
    awk -v o="$old" -v n="$new" '
      !done && (i = index($0, o)) {
        print substr($0, 1, i-1) n substr($0, i+length(o)); done = 1; next
      } { print }' "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    echo "edited: $(basename "$file"): ${old%%$'\n'*} -> $new"
  else
    echo "--- $file"
    diff "$file" \
      <(awk -v o="$old" -v n="$new" '
        !done && (i = index($0, o)) {
          print substr($0, 1, i-1) n substr($0, i+length(o)); done = 1; next
        } { print }' "$file") || true
  fi
}

# ── 1. Declare the snapshot main file explicitly ──────────────────────────
# Same-stem convention only looks in the sidecar's local dir; hub-resident
# models must be named via `model:` (resolved from hf_repo/hf_url snapshot).
insert_model() { # insert_model FILE MODEL_FILENAME
  local file="$1" model="$2"
  if grep -q "^model:" "$file"; then echo "SKIP (has model:) $file"; return; fi
  run sed -i "/^hf_url:/a model: $model" "$file"
}

insert_model "$CHAT/qwen3/Dirk-Qwen3.8-27B-UD-Q5_K_XL.md" \
  Dirk-Qwen3.8-27B-UD-Q5_K_XL.gguf
insert_model "$CHAT/qwen3/Qwen3.8-27B-UD-Q5_K_XL.md" \
  Qwen3.8-27B-UD-Q5_K_XL.gguf
insert_model "$CHAT/qwen3/Nail-Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.md" \
  Nail-Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
insert_model "$CHAT/qwen3/Huihui-Qwable-3.6-27b-abliterated-Q4_K_M_Q8-MTP.md" \
  Huihui-Qwable-3.6-27b-abliterated-Q4_K_M_Q8-MTP.gguf
insert_model "$CHAT/qwen3/Ornith-1.5-9B-AD-Q8_0-Q6_K.md" \
  Ornith-1.5-9B-AD-Q8_0-Q6_K.gguf
insert_model "$CHAT/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.md" \
  gemma-4-12B-it-qat-UD-Q4_K_XL.gguf

# ── 2. Point mmproj:/speculative: at real snapshot filenames ──────────────
# Old values name deleted local links. They'd still resolve via the single-
# glob fallback today, but exact snapshot names are unambiguous forever
# (Ornith's repo has TWO mmproj files — its already-exact value stays).
edit "$CHAT/qwen3/Dirk-Qwen3.8-27B-UD-Q5_K_XL.md" \
  'mmproj: Dirk-Qwen3.8-27B-UD-mmproj-F16.gguf' 'mmproj: mmproj-F16.gguf'
edit "$CHAT/qwen3/Qwen3.8-27B-UD-Q5_K_XL.md" \
  'mmproj: Qwen3.8-27B-UD-mmproj-BF16.gguf'     'mmproj: mmproj-BF16.gguf'
edit "$CHAT/qwen3/Nail-Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.md" \
  'mmproj: Nail-Qwen3.6-35B-A3B-mmproj-F16.gguf' 'mmproj: mmproj-F16.gguf'
edit "$CHAT/qwen3/Huihui-Qwable-3.6-27b-abliterated-Q4_K_M_Q8-MTP.md" \
  'mmproj: Huihui-Qwable-3.6-27b-abliterated-mmproj-f16.gguf' \
  'mmproj: mmproj-model-f16.gguf'

# gemma-4-12B: hf_url pointed at the upstream safetensors repo
# (google/gemma-4-12b — no GGUFs); the quant lives at unsloth, whose
# snapshot also holds the mtp draft under its real name.
G12="$CHAT/gemma4/gemma-4-12B-it-qat-UD-Q4_K_XL.md"
edit "$G12" 'hf_url: https://huggingface.co/google/gemma-4-12b' \
            'hf_url: https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF'
edit "$G12" 'speculative: gemma-4-12B-it-mtp.gguf' \
            'speculative: mtp-gemma-4-12B-it.gguf'

# ── 3. Fix the scoped override's chat-template path ───────────────────────
# chat_template/loras resolve relative to each SIDECAR's directory (not the
# models.yaml's), and you moved the template INTO qwen3/ renamed.
MY="$CHAT/qwen3/models.yaml"
edit "$MY" '../qwen_chat_template.jinja' 'chat_template.jinja'
edit "$MY" 'so from t2t/qwen3/ the shared template is one level up.' \
           'the template lives next to the sidecars in this directory.'

echo
if [ "$MODE" = apply ]; then
  echo "── verify ──"
  cd /mnt/pool/data/tools/llama
  uv run llama-packer --dry-run --no-stubs -VV 2>&1 |
    grep -E "missing|not found|skipping" || echo "no warnings — clean."
fi
