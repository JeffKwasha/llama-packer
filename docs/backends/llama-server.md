# Backend: llama-server

The default engine: GGUF chat, embeddings and rerank via llama.cpp's
`llama-server`.

- **Engine:** llama.cpp · **Transports:** host only (`llama-server`).
- **Endpoints:** `openai.chat.completions`, `openai.embeddings`, `openai.rerank`.
- **Formats:** `.gguf`.
- **Binary:** `--llama-server` > `profiles.yaml llama_server.bin` >
  `$LLAMA_BIN_DIR` > the run directory's `tools/`, then the run directory,
  scanned for the highest `llama-b<NNNN>/` (see `find_bin_dir`). A `bin:` or
  `$LLAMA_BIN_DIR` may name the binary or a `llama-b*` directory; there is no
  `PATH` fallback, because a build that is not a versioned `llama-b*`
  directory is not a build the planner can size against. Its cwd scan is the
  reason the other backends' resolution gained a run-directory tier too.
  When a stable `llama-server` link sits beside the build in `tools/` (or the
  run directory), the emitted config names that link rather than the versioned
  directory, so an engine update moves the served binary without a re-pack.
  `llama-fit-params` is still resolved inside the build the link points at.
- **Proxied:** no — llama-swap manages the process and proxies by port.

## Command shape

```
<llama_bin> --port ${PORT} -m <model.gguf>
  --kv-unified-per-slot <ctx> --parallel <N>
  --cache-type-k <ct> --cache-type-v <ct>
  --n-gpu-layers 0                  # only when the model is CPU-placed;
                                    # otherwise omitted (llama-server "auto"
                                    # offloads all layers on a GPU build)
  [-b <batch> -ub <ubatch>]         # role-aware defaults
  [--mmproj <file>] [--image-min-tokens/--image-max-tokens]
  [--spec-type ... --spec-draft-model ...]   # MTP / speculative
  [--jinja --chat-template-file <tpl>] [--lora a,b]
  [--reasoning-format <f>] [--reasoning-preserve]   # chat only
```

CPU-placed entries (`--n-gpu-layers 0`) are budgeted off **host RAM** at
pack time, not freed of budgeting: the pool is `MemAvailable − sysctl
pressure zone`, capped by the llama-swap service cgroup headroom, and the
same `over_budget` refusal applies — an entry whose measured quad will not
fit is omitted from the config. See `docs/plans/cpu-residency-budget.md`.

`--kv-unified-per-slot X` sizes the shared KV pool to `parallel × X`, which is
byte-identical to a `-c parallel*X` pool (validated against `llama-fit-params`).

## Roles and flags

- `embeddings` → `--embedding --embd-normalize 2`.
- `rerank` → `--rerank --pooling rank`.
- No role declares a batch default: an unset `batch:`/`ubatch:` renders no
  flag, so llama.cpp's builtin (`-b` 2048, `-ub` 512) applies. The old
  embed/rerank `4096` was never in effect — llama-server forces
  `n_batch = n_ubatch` when embeddings are enabled.
- Reasoning flags and image-token flags are chat/vision-only and skipped
  elsewhere (with warnings when a declaration cannot apply).
- A `draft-dflash`/`draft-dspark` draft is emitted at any slot count. Those
  engines are single-sequence and aborted llama-server on a concurrent batch
  on build 11988; build 12600 passed concurrent-request tests with the
  MiniCPM-V 4.7 1B and Qwen3.8-27B drafts. See SPEC.md, "Single-sequence draft
  engines".

## profiles.yaml

- `llama_server: {args: "...", batch: N, ubatch: M}` — fleet-wide args and the
  first-class batch keys. Precedence: `built-ins < args < role flags <
  sidecar cli_args`; the named `-b`/`-ub` win over conflicting `args`. Put
  `-ub` in `ubatch:` (or `args`), not sidecar `cli_args`: the VRAM estimate
  reads the named key and `args`, so a `cli_args` override makes the served
  flags disagree with the measured shape. `-b` has no effect on the estimate.
  Logging flags (`-lv 5`, `-v`, `--log-*`) are the exception: they are
  stripped from the shape stamp by `vram.shape_key()`, so enabling verbose
  server logs fleet-wide in `args` does not invalidate `derived:` blocks or
  trigger a re-measure. Only memory-relevant flags do. Stamp tokens are
  shlex-quoted, so a value containing spaces survives the round trip.

## VRAM

Affine fit-params model (`llama-fit-params`): `model_mib + kv_per_token × ctx ×
slots + slot_mib + compute_mib`, per `cache_type`. Measurements are persisted
in the sidecar `derived:` block; `--remeasure` re-runs them. See
`SPEC.md`.

## See also

- [`SPEC.md` → Context Management](../../SPEC.md#context-management),
  [→ MTP Speculative Decoding](../../SPEC.md#mtp-speculative-decoding),
  [→ Reasoning](../../SPEC.md#reasoning)
- [Transport: host](../transports/host.md)
