# llama-packer Specification

## Overview

`llama-packer` generates `config.yaml` for [llama-swap](https://github.com/mostlygeek/llama-swap) from GGUF model metadata. It scans model directories, detects GPU hardware, measures per-model VRAM costs via `llama-fit-params`, budgets context windows, resolves companion files, applies sampling profiles, and writes ready-to-run server configurations.

**Entry point:** `gen-config.py` → `llama_packer.__main__.main`

**Models directory guide:** `gen-config.py --agents` writes `models/AGENTS.md`
from the bundled template (`llama_packer/templates/models_AGENTS.md`) when the
file is missing — never overwriting an existing one, and logging (not aborting)
on write failure. The guide documents discovery and sidecar conventions for AI
agents. As a non-frontmatter `.md` (no leading `---`), `AGENTS.md` is skipped by
model discovery.

## Output Format

### llama-swap YAML (`write_yaml()`)

Each model entry produces:
| Key | Purpose |
|-----|---------|
| `cmd` | `llama-server` invocation (multiline `\|` scalar) |
| `name` | Human-readable model name (from `.md` frontmatter) |
| `description` | Optional model description |
| `filters.setParamsByID` | Per-request sampling overrides keyed by `${MODEL_ID}:profile` (global profiles, or a model's own `modes:` when declared). Always nested under `filters:` (see [Sampling Modes](#sampling-modes)) |
| `metadata` | Pass-through metadata from `.md` frontmatter (excluded: consumed keys) |
| `capabilities` | Native llama-swap block: `in`/`out` modalities, `tools`, `reranker`, `context` |
| `env` | Per-model GPU device pinning (e.g. `ROCR_VISIBLE_DEVICES=0`) |
| `concurrencyLimit` | Per-model concurrency cap (if declared) |

Also emits:
- **`config.env`** — sibling file with `${env.*}` variables for systemd `EnvironmentFile=` or docker `--env-file`.
- **`includeAliasesInList: true`** — presents the `${MODEL_ID}:<mode>`/`${MODEL_ID}:<profile>` aliases in `/v1/models` (llama-swap default is `false`).
- **`healthCheckTimeout`** — auto-calculated or explicit (see below).

### Writer module

`llama_packer/writer.py` provides:
- **`build_config()`** — builds the full llama-swap `models` dict from `Model` objects + profiles
- **`write_yaml()`** — YAML output with literal block scalars

## Hardware Detection

VRAM is detected via a vendor-probe chain in `llama_packer/hardware.py`:

| Priority | Method | Notes |
|----------|--------|-------|
| 1 | `amd-smi metric -m --json` | AMD discrete GPUs |
| 2 | `/sys/class/drm/card*/device/mem_info_vram_total` | AMD kernel sysfs |
| 3 | `rocminfo` | AMD ROCm fallback |
| 4 | `nvidia-smi --query-gpu=memory.total` | NVIDIA discrete GPUs |
| 5 | System RAM − `hardware.unified_system_mb` | Unified memory or no GPU tools |

If all detection fails, `SystemExit` is raised (use `--vram` to override).

Unified-memory hosts (NVIDIA GB10/DGX Spark, Apple Silicon, Intel integrated) report
`N/A` from `nvidia-smi`, so detection uses total system RAM as the pool and
reserves a fixed system slice (`hardware.unified_system_mb`, default 8 GiB) —
not 50% of RAM, since these machines exist to run models. The knob folds into
the reserve (reserve = exactly the knob, not knob + fixed 2 GiB); override per
host via `profiles.yaml` `hardware.unified_system_mb` or `--unified-system-mb`.
It is a guesstimate, not a measurement: in-use memory on a unified host already
includes the slices we reserve, so used-vs-free is not counted twice.

Precedence for VRAM value: `--vram` CLI flag > `profiles.yaml` `hardware.vram` > auto-detect.

GPU vendor also determines the device-pinning env var: `ROCR_VISIBLE_DEVICES` (AMD) or `CUDA_VISIBLE_DEVICES` (NVIDIA).

## Context Management

The tool calculates `ctx_size` dynamically based on measured VRAM costs, not static estimates.

### Measurement via `llama-fit-params`

For each (model, cache_type, parallel) combination, the system runs `llama-fit-params` to measure three VRAM cost components:

| Component | Type | Meaning |
|-----------|------|---------|
| `model_mib` | Fixed | Weight-loading cost (constant regardless of context) |
| `ctx_factor` | Linear | MiB per token (KV-cache + attention scratch) |
| `compute_mib` | Fixed | Fixed compute overhead (activations, scratch buffers) |

Values are persisted to the model's `.md` sidecar as a `fit-params` block, avoiding re-measurement across runs. Values are invalidated when `cache_type` or `parallel` changes.

**Fallback chain:** saved frontmatter → in-memory cache → `llama-fit-params` binary → safetensors header estimation.

`llama-fit-params` is always invoked with `--fit off` so it measures the explicit `-c` (or design) context rather than auto-adjusting arguments — `--fit on` (the default) would otherwise change `-ngl`/`-c` when the GPU is busy, corrupting the measurement.

### Companion (mmproj / MTP) VRAM accounting

Companion GGUFs cannot be measured by `llama-fit-params` — mmproj files fail to load as standalone models, and MTP draft heads abort on a missing `ctx_other`. Their VRAM is therefore folded into the main model's budget via a combined "effective static" pass (`llama_packer/vram.py:effective_static`):

- **Main model**: measured fit-params values.
- **MTP draft**: file-size weight plus an estimated per-token `ctx_factor` (the draft holds its own KV cache that scales with context) and a fixed compute buffer.
- **mmproj**: file-size weight plus a fixed compute buffer (`_MMPROJ_COMPUTE_MB`, ~150 MiB for the vision projection buffers); no per-token factor.

An attempt is made to run `llama-fit-params` on each companion first (cached per companion); on failure it falls back to the file-size estimate with a warning. This fixes companion VRAM being under-budgeted by raw file size alone (a Gemma4-31B + MTP + mmproj combination was measured to need ~1.8 GiB more than the sum of companion file sizes).

### Context calculation formula

```
reserve = RESERVE_SYSTEM(1024) + max(RESERVE_VIDEO(1024), baseline_mb)
available = vram_total - reserve - spare
remaining = available - model_mib - compute_mib        # model/ctx/compute now include companions
max_ctx = remaining / ctx_factor
max_ctx = floor(max_ctx / CTX_ROUND_TO) * CTX_ROUND_TO    # round down to 8192 boundary
max_ctx = max(max_ctx, MIN_CTX_SIZE)                       # floor at 4096
max_ctx = min(max_ctx, gguf_context_length)                # cap at GGUF architectural max
max_ctx = min(max_ctx, sidecar_context_length)             # cap at sidecar ceiling
max_ctx = min(max_ctx, max_context)                        # cap at CLI --max-context
```

`baseline_mb` is the VRAM already consumed by the driver/compositor/other processes. It is **opt-in**: set via `--baseline` or `profiles.yaml` `hardware.baseline_mb`, and defaults to **0**. The fixed `_RESERVE_VIDEO` (1024 MiB) already covers driver/compositor overhead, so auto-detection of the live `used_vram` is intentionally NOT performed — llama-swap keeps model servers resident, and counting that usage would make the budget assume a blank GPU and collapse every context to the minimum. The effective reserve is the fixed system reserve (1024) plus the larger of the fixed video reserve (1024) and any explicit `baseline_mb`. `--spare` is subtracted on top of this reserve. CPU-resident models (`device: cpu`) are excluded from VRAM budgeting entirely and are sized to their own architectural/sidecar context.

**Design context:** When `llama-fit-params` measures `ctx_factor`, it uses the model's GGUF architectural context length as the reference point (or sidecar `context_length`, or 32768 default). If the design context fits within the remaining budget, it is used directly without scaling down.

**Note:** `parallel` slots are accounted for inside `llama-fit-params` measurement — no separate division step.

### Minimum useful context and vision (mmproj) skipping

Chat models target a minimum useful context (`_MIN_USEFUL_CTX`, default 131072 = 128k, overridable with `--min-context`). For a chat model with an mmproj companion, `calc_ctx` is evaluated both with and without vision (at the global `--spare`):

- `ctx_with ≥ min_context` → keep vision; a single entry is emitted with `--mmproj` and the `vision` capability.
- `ctx_with < min_context` → the main entry **drops** mmproj: no `--mmproj`, the `vision` capability is removed, and `metadata.mmproj_skipped: true` is set. A companion **vision variant** entry is additionally emitted with `--mmproj` at best-effort context, id-suffixed `-vision-<N>k` where `N = ctx_with // 1000` (e.g. 92567 → `-vision-92k`), display name `[vision Nk]`, keeping vision available at reduced context.
- If even the text-only context is `< min_context`, a warning is logged (the main entry is still emitted).

Both the main and vision-variant entries honor the per-profile `spare_mb` and the matrix-solved chat context; the drop decision itself is made once per model using the global spare.

## Matrix Context Solving

When `profiles.yaml` defines a `matrix` section with `embed` and `rerank` models, the system solves a shared VRAM budget equation across all model types:

```
reserve = RESERVE_SYSTEM(1024) + max(RESERVE_VIDEO(1024), baseline_mb)
available = vram_total - reserve - spare
chat_ctx solves Σ(chat_weight + chat_factor × chat_ctx) = available - embed - rerank
```

The solver (`llama_packer/vram.py:solve_matrix_ctx`) finds the maximum chat context that coexists with fixed embed/rerank allocations (default 8192 each). All chat models share the same VRAM pool (llama-swap evicts between them), so the solver picks the largest feasible context across all chat models.

Embed/rerank models are auto-selected as the smallest model of each type, or matched by `--embed`/`--rerank` CLI selectors.

## MTP Speculative Decoding

Multi-Token Prediction (MTP) is supported for models where draft heads are baked into the GGUF or provided as a companion file.

### Detection Logic

MTP is enabled when:
1. The filename contains `mtp` (case-insensitive), **or**
2. The sidecar `.md` file has `mtp: true` in YAML frontmatter, **or**
3. The `speculative` frontmatter field points to a companion file with `mtp` in its stem.

### Implementation Flags

When MTP is detected, these flags are appended to the `llama-server` command:
- `--spec-type draft-mtp` — Enables the MTP draft-head speculative decoding (configurable via `mtp_spec_type`).
- `--spec-draft-n-max 2` — Maximum number of tokens to speculate (configurable via `mtp_draft_n_max`).
- `--spec-draft-model <path>` — (companion MTP only) Path to the draft model file.

### Per-Model Configuration

MTP behavior can be overridden per model via the `.md` sidecar frontmatter:

```yaml
---
name: my-model
mtp: true
mtp_spec_type: draft-mtp      # default: draft-mtp
mtp_draft_n_max: 3            # default: 2
---
```

If absent, the module-level defaults apply (see `llama_packer/utils.py`).

### Baked-in vs Companion MTP

- **Baked-in:** Draft heads are part of the main GGUF. Set `mtp: true` in frontmatter. No separate file needed.
- **Companion:** A separate GGUF file containing draft heads. Set `speculative: <filename>` in frontmatter. The file is resolved by fuzzy match in the model directory.

## Reasoning Variants

When `reasoning` is set to `"auto"` in frontmatter, the system auto-generates multiple config entries — one per reasoning mode:

| Mode | Suffix | Effect |
|------|--------|--------|
| `none` | `.reasoning.none` | `--reasoning` flag omitted |
| `native` | `.reasoning.native` | `--reasoning native` |
| `openai` | `.reasoning.openai` | `--reasoning openai` |

Each variant gets its own llama-swap entry ID. A specific mode (e.g. `reasoning: native`) generates a single entry with that mode.

## Sampling Modes

By default sampling parameters come **only** from `profiles.yaml` (`defaults:` + `profiles:`),
merged into `setParamsByID` keys. A model can instead declare its own full sampling profiles
per **mode** (e.g. `instruct`, `thinking`, `writing`) — for models whose recommended samplers
differ per usage, like one presence-penalty for instruction and another for reasoning.

```yaml
---
default_mode: instruct
modes:
  instruct:
    temperature: 0.6
    top_p: 0.9
    top_k: 40
    min_p: 0.05
    pres_pen: 1.5
    repeat_penalty: 1.1
  thinking:
    temperature: 1.0
    pres_pen: 0.0
    repeat_penalty: 1.0
---
```

- **`modes`**: a map of mode name → full param set. The declared block is authoritative —
  global profile sampling overrides are not applied for this model. Models may declare any
  number of modes (commonly 1–3).
- **`default_mode`**: which mode is the model's default. It is emitted under the bare
  `${MODEL_ID}` `setParamsByID` key; every other mode under `${MODEL_ID}:<mode>`. Falls back to
  the first declared mode.
- **Keys**: llama.cpp parameter names — `temperature`, `top_p`, `top_k`, `min_p`, `pres_pen`,
  `repeat_penalty`, `freq_pen` (see `SAMPLING_KEYS`). Unknown or non-numeric values are
  ignored with a warning. Emission translates to the request-body JSON names llama-server
  parses (`pres_pen` → `presence_penalty`, `freq_pen` → `frequency_penalty`).
- **Schema**: per llama-swap, `setParamsByID` is a *filter* and is always nested under
  `filters:` in each model entry — a top-level key is silently ignored.
- **Alias visibility**: each mode/profile key (`${MODEL_ID}`, `${MODEL_ID}:<mode>`,
  `${MODEL_ID}:<profile>`) auto-registers as a model alias and applies per-request without
  reloading. The generated config sets global `includeAliasesInList: true` so these aliases
  appear in `/v1/models`, letting dynamic-list clients (OpenWebUI, OpenClaw, ...) select them.
- **Metadata**: a model with `modes:` also exposes `metadata.modes` (sorted list) and
  `metadata.default_mode` for static client discovery (hermes, opencode configs).
- Models without `modes:` are unaffected and keep the global-profile `setParamsByID` behavior
  (also nested under `filters:`).

## vLLM Docker Backend

A chat model can be served with vLLM instead of llama-server by declaring
`template: vllm-docker` in its sidecar. The emitted entry is a `docker run` command that
launches the `vllm/vllm-openai` image with `vllm serve`, published to llama-swap's `${PORT}`
host macro. Everything else works identically: aliases/modes (`filters.setParamsByID`),
`metadata`, capabilities, matrix routing.

```yaml
---
name: qwen3-30b
template: vllm-docker
hf_repo: Qwen/Qwen3-30B-A3B-Instruct
context_length: 65536
---
```

### Model resolution

- `hf_repo` frontmatter wins when declared; otherwise parsed from `hf_url`
  (`https://huggingface.co/{owner}/{repo}`). If neither exists, the local GGUF path is used
  as `--model`.
- vLLM serves safetensors — the local GGUF file is *not* used unless it is also an HF checkout.

### Image precedence

The container image is resolved, highest to lowest:

1. Per-model `vllm_image:` frontmatter
2. `--vllm-image` CLI flag
3. `vllm.image` in `profiles.yaml`
4. Built-in default (`vllm/vllm-openai:latest`)

`profiles.yaml` `vllm:` also configures `docker_args`, `container_port`, and `gpu_mem_util`
(the vLLM `--gpu-memory-utilization` fraction).

### Limitations

- Scenario not yet handled: per-model VRAM planning for safetensors (vLLM memory estimator
  hook) — the emitted `--max-model-len`/`--gpu-memory-utilization` come from llama.cpp's
  fit-params budget for now.
- MTP/speculative flags are not translated for the vLLM backend yet.
- Runs one vLLM server per model per docker image; multi-image or cluster/tensor-parallel
  provisioning is future work.

## Health-Check Timeout

Auto-calculated when not explicitly set via `--health-check-timeout`:

```
hct = max(120, int(1.2 × largest_model_mb / drive_speed_mb))
```

Drive speed is detected per-model via `lsblk` (NVMe → 1500 MB/s, SATA SSD → 300 MB/s, HDD → 100 MB/s, unknown → 100 MB/s). The slowest drive among all model files bounds the timeout.

Override via `--health-check-timeout`, `--drive-speed`, or the `GEN_CONFIG_DRIVE_SPEED` environment variable.

## Env Variable Substitution

All emitted paths (binary, GGUF files, mmproj, MTP companions) are grouped by filesystem mount. The longest common path prefix per mount group becomes a `${env.*}` variable:

| Variable | Content |
|----------|---------|
| `LLAMA_DIR` | Mount group containing the llama-server binary |
| `MODELS_DIR` | Mount group containing model files |
| `MODELS_DIR_2` ... | Additional mount groups (sorted by mount path) |

Written to `config.env` alongside the YAML for systemd/docker consumption.

## Model Metadata

Sidecar `.md` files declare model config. The generator is **pass-through-by-default**: any frontmatter key not consumed by the builder is exposed to clients automatically.

### Metadata channel

Model identity and agent-selection metadata is carried entirely by the llama-swap config — no
`--override-kv` flags in `cmd`. Fields with native llama-swap support use native keys; everything
else flows into the per-model `metadata` dict (→ `meta.llamaswap` in `/v1/models`):

- **`metadata`** — the agent-choice descriptor: `freethought`, `strengths`, `weaknesses`,
  `license`, `base_model`, `finetune`, `type`, `parameters`, `quantization`, `hf_url`,
  `context_length`, `mtp_enabled`, `mtp_draft_max`, `mtp_accuracy`, `throughput_factor`.
- **Native `capabilities`** — `in`/`out` modalities, `tools`, `reranker`, `context` (derived, see below).
- **Native `name` / `description`** — display fields in `/v1/models`.

### Builder-consumed keys (NOT passed through)

`name`, `template`, `context_length`, `description`, `cli_args`, `model`, `attention`,
`kv_cache`, `tool_args`, `speculative`, `mmproj`, `mtp`, `mtp_spec_type`, `mtp_draft_n_max`,
`mtp_draft_p_min`, `role`, `allow_profiles`, `reasoning`, `spare`, `capabilities`,
`ignore`, `device`, `concurrency`, `fit-params`, `modes`, `default_mode`.

### Per-model config options

| Frontmatter Key | Type | Effect |
|-----------------|------|--------|
| `device` | int | GPU device index for multi-GPU pinning (`ROCR_VISIBLE_DEVICES=N` / `CUDA_VISIBLE_DEVICES=N`) |
| `concurrency` | int | Per-model concurrency limit → `concurrencyLimit` in config |
| `spare` | str | Additional VRAM to reserve (overrides global `--spare`) |
| `allow_profiles` | str/list/bool | Restrict which profiles apply (regex string, list, or false to disable) |
| `template` | str | Backend template override. `template: vllm-docker` serves this model with vLLM in a container instead of llama-server; omitted → derived from `role` (chat/embeddings/rerank) |
| `modes` | dict | Per-model sampling modes (full profiles): name → param dict. Replaces the global-profile sampling overrides for this model. Values use llama.cpp names; see [Sampling Modes](#sampling-modes) |
| `default_mode` | str | Which declared `modes` entry is the model's default (maps to the bare `${MODEL_ID}` `setParamsByID` key). Falls back to the first mode |
| `reasoning` | str/bool | `auto` → multi-variant generation; specific mode → single variant; `true`/absent → no change |
| `ignore` | bool | Skip this model entirely |

### Agent-selection fields (optional, recommended)

| Field | Type | Meaning |
|-------|------|---------|
| `capabilities` | list | `[vision, tools, reasoning, audio]`; `vision` auto-added if a companion `mmproj` exists. Mapped to the native llama-swap `capabilities` block |
| `freethought` | float 0–1 | `1.0` reasons about anything; `0.0` readily refuses "distasteful" topics. Carried in `metadata` |
| `strengths` / `weaknesses` | list | Concise task phrases agents match on |
| `license` / `base_model` / `finetune` / `type` | str | Identity; carried in `metadata` |
| `mtp_accuracy` | float | MTP draft acceptance rate; feeds `throughput_factor` |
| `parameters` | str | `"12B"` or MoE `"26B-A4B"` (total-active) for accurate throughput |
| `hf_url` | str | HuggingFace model URL |
| `hf_repo` | str | HF repo id for vLLM backends. Optional; parsed from `hf_url` when absent |
| `vllm_image` | str | Per-model vLLM docker image. Overrides profiles.yaml `vllm.image` and `--vllm-image` for this entry |

### Derived fields (computed, not authored)

- **`capabilities`** (native block): `in`/`out` = `["text"]` + `"image"` if `vision` in capabilities + `"audio"` if `audio`; `tools`/`reranker` boolean flags; `context` = computed `ctx_size`.
- **`throughput_factor`**: Heuristic relative speed index = `54 / (active_B × quant_bits)` × `(1 + draft_n × mtp_accuracy)` when MTP is on. Relative only — not real tok/s.
- **`ctx_size`**: The computed context size for this model, exposed natively via `capabilities.context`.

### GGUF architectural context

The GGUF header's `<architecture>.context_length` is read directly from the file and takes precedence over the sidecar `context_length` when capping context size. This ensures the architectural limit is never exceeded.

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
device: 0
concurrency: 2
strengths:
  - "bash tool calling"
weaknesses:
  - "slow on 32GB"
---
```

## Fit-Params Persistence

Measured VRAM parameters are stored in the sidecar `.md` file under a `fit-params` nested block:

```yaml
fit-params:
  model_mib: 4800
  ctx_factor: 0.0312
  compute_mib: 512
  source: fit-params
  cache_type: q8_0
  parallel: 1
```

This block is:
- Read automatically on subsequent runs (avoids re-measurement).
- Invalidated when `cache_type` or `parallel` changes.
- Updated when new values are computed.
- Preserved alongside all other frontmatter keys.

## Constants

All module-level constants are in `llama_packer/utils.py`.

| Constant | Default | Meaning |
|----------|---------|---------|
| `_DEFAULT_CONTEXT_LENGTH` | 32768 | Fallback when no `.md` sidecar or GGUF context exists |
| `_CTX_ROUND_TO` | 8192 | Round context size down to nearest boundary |
| `_MIN_CTX_SIZE` | 4096 | Hard floor for context size |
| `_MIN_USEFUL_CTX` | 131072 | Min useful chat context; mmproj dropped below this (`--min-context`) |
| `_RESERVE_SYSTEM` | 1024 | MB reserved for OS/driver/scratch buffers |
| `_RESERVE_VIDEO` | 1024 | MB reserved for GPU video output framebuffer |
| `_MMPROJ_COMPUTE_MB` | 150 | Fixed compute buffer for mmproj companions |
| `_DRAFT_COMPUTE_MB` | 64 | Fixed compute overhead for MTP draft companions |
| `_DRAFT_CTX_SAFETY` | 1.6 | Safety factor on the MTP draft per-token KV estimate |
| `_MTP_SPEC_TYPE` | `"draft-mtp"` | Default MTP speculative type |
| `_MTP_DRAFT_N_MAX` | 2 | Default max draft tokens for MTP |

## Future Work — Log-Derived Real Throughput (Phase 6)

`throughput_factor` is a heuristic. A planned offline enrichment derives **measured** throughput from llama-server logs and overrides the heuristic when available.

- **Script:** `scripts/parse_llama_logs.py` (run out-of-band, e.g. cron or on-demand).
- **Source:** `journalctl -u llama-swap.service` (or a `--log` file).
- **Parsing:** extract the launch cmd per request window to get the `-m` model path/stem; capture tok/s from llama.cpp's known lines:
  - `prompt eval time = … (N tokens per second)` → preprocessing (pp) tok/s
  - `eval time = … (N tokens per second)` → generation (tg) tok/s
  - fallbacks: `… tok/s`, `… t/s`. Tolerant regexes; multiple patterns.
- **Aggregation:** average per model stem → `models/.throughput_cache.json` `{stem: {tps, pp_tps, samples, updated}}`.
- **Consumption:** `gen-config` reads the cache and adds `observed_tps` / `observed_pp_tps` to `metadata` when present, overriding `throughput_factor`.
- **Graceful:** new/unrun models simply lack the cache entry — no change to the core metadata pipeline required (pure enrichment source).
