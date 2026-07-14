# Llama-Swap Configuration Specification

## Overview
The tools in this directory automate the generation of `config.yaml` for llama-swap and/or llama-server router config, ensuring that `llama-server` is launched with optimal flags based on the active GGUF model and available GPU VRAM.

## Output Format: llama-swap config.yaml

The generated config follows the upstream llama-swap configuration format:
- **Docs:** <https://raw.githubusercontent.com/mostlygeek/llama-swap/refs/heads/main/docs/configuration.md>
- **Full example (always check for latest):** <https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml>

Each model entry produces:
| Key | Purpose |
|-----|---------|
| `cmd` | `llama-server` invocation (multiline `\|` scalar) |
| `name` | Human-readable model name (from `.md` frontmatter) |
| `description` | Optional model description |
| `setParamsByID` | Per-profile sampling overrides keyed by `${MODEL_ID}:profile` |
| `metadata` | Pass-through metadata from `.md` frontmatter (excluded: consumed keys) |

### Writer module

`model_cfg/output/llama_swap.py` provides:

- **`build_config()`** — builds the full llama-swap `models` dict from `Model` objects + profiles
- **`write_yaml()`** — llama-swap YAML output with literal block scalars
- **`write_ini()`** — llama-server router-mode INI output

The CLI entry is `gen-config.py` → `model_cfg.__main__.main`.

## MTP Speculative Decoding
Multi-Token Prediction (MTP) is supported for models where the draft heads are baked into the GGUF file.

### Detection Logic
The system enables MTP if:
1. The filename contains `mtp` (case-insensitive).
2. OR the sidecar `.md` file has `mtp: true` in the YAML frontmatter.

### Implementation Flags
When MTP is detected, the following flags are appended to the `llama-server` command:
- `--spec-type draft-mtp`: Enables the MTP draft-head speculative decoding.
- `--spec-draft-n-max 2`: Sets the maximum number of tokens to speculate.
- `--spec-draft-p-min 0.75`: Minimum probability for draft tokens.

### Per-Model Configuration
MTP behavior can be overridden per model via the `.md` sidecar frontmatter:

```yaml
---
name: my-model
mtp: true
mtp_spec_type: draft-mtp      # default: draft-mtp
mtp_draft_n_max: 3            # default: 2
mtp_draft_p_min: 0.7          # default: 0.75
---
```

If absent, the module-level defaults apply (see `_MTP_*` constants in `llama_swap_config.py`).

## Context Management
The tool calculates `ctx_size` dynamically based on available GPU VRAM, then caps it at the model's declared `context_length` from its `.md` sidecar:

1. Detect total VRAM via `rocminfo` (fallback: 24 GB).
2. Reserve 1 GB for system/driver/scratch buffers + 1 GB for GPU video output framebuffer.
3. Subtract model file size → remaining VRAM for context.
4. Compute `max_tokens = (remaining_bytes * 1024) // 20` (estimated ~20 bytes per token for quantized KV cache + scratch).
5. Divide by `parallel` request slots.
6. Round down to nearest 8192-token boundary (minimum 4096).
7. Cap at the frontmatter `context_length` ceiling.

This ensures maximum feasible context that fits within VRAM, never exceeding the model's inherent limit. If no sidecar exists, `context_length` defaults to 32768.

### Configurable Parameters (module-level constants in `llama_swap_config.py`)

| Constant | Default | Meaning |
|----------|---------|---------|
| `_DEFAULT_CONTEXT_LENGTH` | 32768 | Fallback when no `.md` sidecar exists |
| `_FALLBACK_VRAM_MB` | 24576 | Fallback when `rocminfo` unavailable |
| `_ROCMINFO_TIMEOUT` | 10 | Seconds before `rocminfo` probe times out |
| `_BYTES_PER_TOKEN` | 20 | Estimated byte cost per context token |
| `_CTX_ROUND_TO` | 8192 | Round context size down to nearest boundary |
| `_MIN_CTX_SIZE` | 4096 | Hard floor for context size |
| `_RESERVE_SYSTEM` | 1024 | MB reserved for OS/driver/scratch buffers |
| `_RESERVE_VIDEO` | 1024 | MB reserved for GPU video output framebuffer |

## Model Metadata
Sidecar `.md` files declare model config. The generator (`gen-config.py` → `model_cfg/`) is
**pass-through-by-default**: any frontmatter key not consumed by the builder is exposed to
clients automatically, so new descriptive fields can be added without code changes.

### Two metadata channels
1. **llama-server `meta`** — via `--override-kv` flags injected into the `cmd`. Carries
   vllm-style identity keys (and `freethought`):
   `general.license`, `general.basename`, `general.finetune`, `general.type`, `general.name`,
   `general.freethought`.
2. **llama-swap `metadata`** (→ `meta.llamaswap` in `/v1/models`) — the agent-choice
   descriptor: `capabilities`, `modalities`, `ctx_size`, `freethought`, `strengths`,
   `weaknesses`, `license`, `base_model`, `finetune`, `type`, `parameters`, `quantization`,
   `hf_url`, `context_length`, `mtp_enabled`, `mtp_draft_max`, `mtp_accuracy`,
   `throughput_factor`.

### Builder-consumed keys (NOT passed through)
`name`, `template`, `context_length`, `description`, `cli_args`, `model`, `attention`,
`kv_cache`, `tool_args`, `speculative`, `mmproj`, `mtp`, `mtp_spec_type`, `mtp_draft_n_max`,
`mtp_draft_p_min`, `targets`, `allow_profiles`, `reasoning`, `spare`, and `context_limit_*G`.

### Agent-selection fields (optional, recommended)
| Field | Type | Meaning |
|-------|------|---------|
| `capabilities` | list | `[vision, tools, reasoning, audio]`; `vision` auto-added if a companion `mmproj` exists |
| `freethought` | float 0–1 | `1.0` reasons about anything; `0.0` readily refuses "distasteful" topics. Also injected into llama-server `meta` |
| `strengths` / `weaknesses` | list | concise task phrases agents match on (e.g. `"bash tool calling"`, `"low context usage"`) |
| `license` / `base_model` / `finetune` / `type` | str | identity; mapped to `general.*` via `--override-kv` |
| `mtp_accuracy` | float | MTP draft acceptance; feeds `throughput_factor` |
| `parameters` | str | `"12B"` or MoE `"26B-A4B"` (total-active) for accurate throughput |

### Derived fields (computed, not authored)
- `modalities`: `["text"]` + `"image"` if `vision` + `"audio"` if `audio`.
- `throughput_factor`: heuristic relative speed index = `54 / (active_B · quant_bits)` × `(1 + draft_n · mtp_accuracy)` when MTP is on. Relative only — not real tok/s.

### Example
```yaml
---
name: Model-Name
context_length: 131072
mtp: true
hf_url: https://huggingface.co/...
capabilities: [vision, tools, reasoning]
freethought: 0.7
license: apache-2.0
base_model: llama-3
finetune: instruct
type: instruct
mtp_accuracy: 0.9
strengths:
  - "bash tool calling"
weaknesses:
  - "slow on 32GB"
---
```

## Future work — log-derived real throughput (Phase 6)
`throughput_factor` is a heuristic. A planned offline enrichment derives **measured**
throughput from llama-server logs and overrides the heuristic when available.

- **Script:** `scripts/parse_llama_logs.py` (run out-of-band, e.g. cron or on-demand).
- **Source:** `journalctl -u llama-swap.service` (or a `--log` file).
- **Parsing:** extract the launch cmd per request window to get the `-m` model
  path/stem; capture tok/s from llama.cpp's known lines:
  - `prompt eval time = … (N tokens per second)` → preprocessing (pp) tok/s
  - `eval time = … (N tokens per second)` → generation (tg) tok/s
  - fallbacks: `… tok/s`, `… t/s`. Tolerant regexes; multiple patterns.
- **Aggregation:** average per model stem → `models/.throughput_cache.json`
  `{stem: {tps, pp_tps, samples, updated}}`.
- **Consumption:** `gen-config` reads the cache and adds `observed_tps` /
  `observed_pp_tps` to `metadata` when present, overriding `throughput_factor`.
- **Graceful:** new/unrun models simply lack the cache entry — no change to the
  core metadata pipeline required (pure enrichment source).
