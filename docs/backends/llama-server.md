# Backend: llama-server

The default engine: GGUF chat, embeddings and rerank via llama.cpp's
`llama-server`.

- **Engine:** llama.cpp · **Transports:** host only (`llama-server`).
- **Roles:** `chat`, `embeddings`, `rerank`.
- **Formats:** `.gguf`.
- **Binary:** `--llama-server` > `profiles.yaml llama_server.bin` >
  `$LLAMA_BIN_DIR` > `llama-server` on `PATH` (see `find_bin_dir`).
- **Proxied:** no — llama-swap manages the process and proxies by port.

## Command shape

```
<llama_bin> --port ${PORT} -m <model.gguf>
  --kv-unified-per-slot <ctx> --parallel <N>
  --cache-type-k <ct> --cache-type-v <ct>
  --n-gpu-layers (999 | 0)          # 0 when the model is CPU-resident
  [-b <batch> -ub <ubatch>]         # role-aware defaults
  [--mmproj <file>] [--image-min-tokens/--image-max-tokens]
  [--spec-type ... --spec-draft-model ...]   # MTP / speculative
  [--jinja --chat-template-file <tpl>] [--lora a,b]
  [--reasoning-format <f>] [--reasoning-preserve]   # chat only
```

`--kv-unified-per-slot X` sizes the shared KV pool to `parallel × X`, which is
byte-identical to a `-c parallel*X` pool (validated against `llama-fit-params`).

## Roles and flags

- `embeddings` → `--embedding --embd-normalize 2`; batch defaults `4096/512`.
- `rerank` → `--rerank --pooling rank`; batch defaults `4096/512`.
- `chat` → batch defaults `2048/512`.
- Reasoning flags and image-token flags are chat/vision-only and skipped
  elsewhere (with warnings when a declaration cannot apply).

## profiles.yaml

- `llama_server: {args: "...", batch: N, ubatch: M}` — fleet-wide args and the
  first-class batch keys. Precedence: `built-ins < args < role flags <
  sidecar cli_args`; the named `-b`/`-ub` win over conflicting `args`.

## VRAM

Affine fit-params model (`llama-fit-params`): `model_mib + kv_per_token × ctx ×
slots + slot_mib + compute_mib`, per `cache_type`. Measurements are persisted
in the sidecar `derived:` block; `--remeasure` re-runs them. See
`docs/new-model-pipeline.md`.

## See also

- [`SPEC.md` → Context Management](../../SPEC.md#context-management),
  [→ MTP Speculative Decoding](../../SPEC.md#mtp-speculative-decoding),
  [→ Reasoning](../../SPEC.md#reasoning)
- [Transport: host](../transports/host.md)
