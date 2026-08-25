# llama-packer Specification

## Overview

`llama-packer` generates `config.yaml` for [llama-swap](https://github.com/mostlygeek/llama-swap) from GGUF model metadata. It scans model directories, detects GPU hardware, measures per-model VRAM costs via `llama-fit-params`, budgets context windows, resolves companion files, applies sampling profiles, and writes ready-to-run server configurations.

**Assumptions:** llama-packer targets the **current stable release** of every
external tool it drives (llama.cpp/llama-server, vLLM, llama-swap), and adopts
features from newer releases freely; generated commands carry no compatibility
shims or version detection. Running an older stack is the operator's trade-off —
failures surface as obvious upstream errors (e.g. a 404 on an endpoint your
vLLM doesn't have).

**Entry point:** `llama-packer` (console script) → `llama_packer.__main__.main`

**Models directory guide:** `llama-packer --agents` writes `AGENTS.md` into
each `--models-dir` from the bundled template
(`llama_packer/templates/models_AGENTS.md`) when the file is missing — never
overwriting an existing one, and logging (not aborting) on write failure. The
guide documents discovery and sidecar conventions for AI agents. As a
non-frontmatter `.md` (no leading `---`), `AGENTS.md` is skipped by model
discovery.

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
- **`config.env`** — sibling file with the path macros (`MODELS_DIR=...` etc.) for systemd `EnvironmentFile=` or docker `--env-file` (`--no-env` skips it).
- **`macros:`** — top-level block mapping each `${VAR}` path macro to its absolute directory (see [Path Macros](#path-macros-macros-block-and-configenv)).
- **`includeAliasesInList: true`** — presents the `${MODEL_ID}:<mode>`/`${MODEL_ID}:<profile>` aliases in `/v1/models` (llama-swap default is `false`).
- **`healthCheckTimeout`** — auto-calculated or explicit (see below).

### Writer module

`llama_packer/writer.py` composes the generation pipeline in three steps:

1. **`_filter_supported()`** — the validation boundary (backend format/role,
   reasoning flags, cache-type knowability); runs before any VRAM work.
2. **`Planner.plan()`** — context decisions per model: mmproj keep/drop
   pre-pass, shared matrix solve, profile grouping, bounded-context clamp.
   Returns `Variant` values.
3. **`emit_config()`** — pure rendering of plans into llama-swap entry dicts.

`build_config()` composes the three; profiles.yaml is read exclusively through
the `Profiles` value object (`llama_packer/profiles.py`). See
[docs/architecture.md](docs/architecture.md) for component ownership,
invariants, and testing seams. Also here: **`write_yaml()`** — YAML output
with literal block scalars.

## profiles.yaml

The input config, resolved from `--profiles` (default `./profiles.yaml`, falling back to the bundled `llama_packer/profiles.yaml`). Top-level keys, all optional except `profiles:`:

| Key | Purpose | Detailed in |
|-----|---------|-------------|
| `defaults` | Baseline sampling parameters (+ `cache_type`, `parallel`, `spare`) merged under every profile | [Sampling Modes](#sampling-modes), [Cache precision](#cache-precision-cache_type) |
| `profiles` | Named sampling overrides layered on `defaults`; **required** (at least one) | [Sampling Modes](#sampling-modes) |
| `overrides` | Pattern-scoped serving rules (`backend`, `chat_template`, `loras`, …) | [Override Rules](#override-rules) |
| `matrix` | Shared embed/rerank/chat VRAM budget solving (`embed`/`rerank` model refs) | [Matrix Context Solving](#matrix-context-solving) |
| `hardware` | `vram`, `baseline_mb`, `unified_system_mb`, `gpu_family` overrides | [Hardware Detection](#hardware-detection) |
| `vllm` | Backend resources: `image`, `bin`, `docker_args`, `container_port`, optional `gpu_mem_util` | [vLLM Backend](#vllm-backend) |
| `backends` | Ordered enable/prefer list of backend names (absent = all, registration order) | [Backend Selection](#backend-selection) |
| `models_dirs` | Model root directories (CLI `--models-dir` wins) | [Model Discovery and Stub Sidecars](#model-discovery-and-stub-sidecars) |
| `dirs` | Directory-name → role whitelist (e.g. `{ocr: chat}`) | [Model Discovery and Stub Sidecars](#model-discovery-and-stub-sidecars) |
| `hf_home` | HF cache root for hub snapshot resolution (CLI `--hf-home` wins) | [Model Discovery and Stub Sidecars](#model-discovery-and-stub-sidecars), [Path Macros](#path-macros-macros-block-and-configenv) |

`profiles.yaml.example` provides a commented starter with one brief example per category — copy to `profiles.yaml` (gitignored) and uncomment what you need. The bundled `llama_packer/profiles.yaml` is the fallback when no file is present.

**Profile entries** (each value under `profiles:`) may set:

- any **sampling key** from `SAMPLING_KEYS` (`temperature`, `top_p`, `top_k`,
  `min_p`, `pres_pen`, `repeat_penalty`, `freq_pen`), including
  `"base * N"` expressions resolved against `defaults`
- `cache_type` — KV precision for this variant (see
  [Cache precision](#cache-precision-cache_type)); differing values split a
  model into separate entries
- `parallel` — slot count for this variant (same splitting behavior)
- `spare` — additional VRAM reserved for this variant (overrides CLI
  `--spare`; see [Context calculation formula](#context-calculation-formula))
- `description` — free-text label (documentation only, never emitted)

All `profiles.yaml` keys are **builder-consumed** — they tune generation,
placement, or routing and are never forwarded to clients (unknown top-level keys
are silently ignored; `profiles:` itself is required and the build aborts if
empty). This contrasts with sidecar frontmatter, where any key *not* in the
[builder-consumed list](#builder-consumed-keys-not-passed-through) is
pass-through `metadata` for agents (see [Model Metadata](#model-metadata)).
Within `profiles`, sampling deltas become `filters.setParamsByID` and
`parallel`/`cache_type`/`spare` differences split a model into separate
variants/entries; `description` is the only inert key.

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

**GPU family**: `--gpu-family` or `profiles.yaml` `hardware.gpu_family` names the
chip family on the resolved `GpuProfile`. It is currently an **inert annotation**
(per-library calculation rules don't diverge today, so no behavior keys off it);
it exists as the hook for future chip-specific sizing rules.

GPU vendor also determines the device-pinning env var: `ROCR_VISIBLE_DEVICES` (AMD) or `CUDA_VISIBLE_DEVICES` (NVIDIA).

## llama.cpp Build Selection

When `--llama-server` is not given, the binary directory resolves via
`utils.find_bin_dir`:

1. `$LLAMA_BIN_DIR` — explicit build dir, wins over everything
2. `-v/--llama-version N` → `./llama-bN` if it exists
3. default (`latest`) → the highest-numbered `./llama-b[0-9]*` directory

`llama-server` and `llama-fit-params` are both taken from the resolved
directory; no match aborts with the available versions listed.

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

**Cache-type scaling.** The KV-cache component of `ctx_factor` is linear in
cache precision (per-element bytes, `utils._KV_CACHE_BYTES`), while `model_mib`
and `compute_mib` are precision-independent. So a `cache_type` change does
**not** trigger a re-measure: the persisted `ctx_factor` is scaled by the
byte ratio between the old and new precisions (`_scale_ctx_factor`), rounding
up to avoid under-reserving. A `parallel` change still invalidates (batching
buffers don't scale linearly).

**Fallback chain:** saved frontmatter → cache-type-scaled derivation → in-memory cache → `llama-fit-params` binary → safetensors header estimation.

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

- `ctx_with ≥ min_context` → keep vision; the main entry is emitted with `--mmproj` and the `vision` capability. An on-demand **text-only variant** `<id>-text` (no `--mmproj`, `vision` removed, `metadata.mmproj_skipped: true`, display name `[text]`) is emitted alongside so clients can pick the lower-memory serving.
- `ctx_with < min_context` → the main entry **drops** mmproj and is renamed `<id>-text` — the invariant is that the bare `<id>` always serves vision when the model has one; every no-mmproj entry carries the `-text` suffix and `[text]` label so a client that knows nothing of server config can tell it is text-only from `/v1/models`. A companion **vision variant** entry is additionally emitted with `--mmproj` at best-effort context, id-suffixed `-vision-<N>k` where `N = ctx_with // 1000` (e.g. 92567 → `-vision-92k`), display name `[vision Nk]`, keeping vision available at reduced context.
- `ctx_without < min_context` too → **vision is kept**: dropping the projection cannot reach the minimum either way, so sacrificing it buys nothing (a small VLM stays a full VLM). Informational log only — no warning, since no configuration can fix a design-context limit.

All emitted entries honor the per-profile `spare_mb` and the matrix-solved chat context; the drop decision itself is made once per model using the global spare. Every `<id>-text` entry joins the same matrix co-loading sets as its parent `<id>` entry, so `(c1 | … | cN) & emb & rnk` can hold a text variant together with the RAG models.

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

### Speculative decoding under vLLM

vLLM models get `--speculative-config '<json>'` (see
[vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)).
Resolution order in `backends/vllm.py:_speculative_config`:

1. **`speculative_config:`** — explicit frontmatter mapping, emitted verbatim.
   Full control over any vLLM method, e.g. a cross-vocabulary draft model:
   `{method: draft_model, model: org/smoll-draft, num_speculative_tokens: 3}`
2. **`mtp: true`** (baked-in MTP) → `{method: mtp, num_speculative_tokens: N}`
   where N comes from the **same** `mtp_draft_n_max` key and default the
   llama-server path uses (`_MTP_DRAFT_N_MAX` = 2). One configuration, one
   meaning on every backend. If a checkpoint ships a single MTP module, lower
   `mtp_draft_n_max` per sidecar — depth beyond native MTP layers is rejected
   at startup; vLLM's own "start at 1" advice is tuning onboarding, not a
   semantic difference between backends.
3. A GGUF **`speculative:` companion cannot be loaded by vLLM** (it needs an HF
   repo): warned and skipped — use `speculative_config:` with a draft HF repo.

Caveats: baked-in MTP weights are not added to the VRAM budget (same as the
llama-server path); `mtp_spec_type` is llama.cpp-only and ignored by vLLM;
metadata `mtp_enabled` / `mtp_draft_max` reflect the resolved config either way.

## Reasoning

Reasoning/thinking support is two layers: a **server-side default** that
llama-packer emits, and a **per-request** control that clients drive.

### Server-side defaults (emitted by llama-packer)

Two sidecar/override settings control how llama-server surfaces reasoning
(model's own chat template must emit thinking blocks; use the fixed Qwen
template via the `chat_template` override):

| Key | Flag | Values |
|-----|------|--------|
| `reasoning-format` | `--reasoning-format` | `none`, `deepseek`, `deepseek-legacy`, `auto` |
| `reasoning-preserve` | `--reasoning-preserve` (bool) | emit when `true` |

`--reasoning-format deepseek` is the key one: it makes llama-server extract
`<think>` blocks into `message.reasoning_content` (OpenAI-style) instead of
leaking them into `message.content` — which otherwise breaks tool-calling
agents mid-loop. `deepseek-legacy` keeps the tags in `content` *and* fills
`reasoning_content`; `none` leaves thoughts unparsed; `auto` is llama-server's
default.

These flags are **only meaningful on reasoning-capable chat models**: a model
whose `role` is not `chat`, or whose `capabilities` do not include
`reasoning`, is a non-reasoning model — declaring either key on it logs an
error and the setting is ignored. An unknown `reasoning-format` value is
likewise logged and dropped. (A `reasoning` capability in the sidecar is a
descriptive signal; it never errors.)

### Per-request control (client side)

Reasoning **effort** (`enable_thinking`, `reasoning_effort`,
`preserve_thinking`) is a *template kwarg*, sent per-request — llama-packer
does not set it. llama-server accepts it via the request body's
`chat_template_kwargs` (all builds, requires `--jinja`, which the backend
already emits) or, on newer builds, top-level `reasoning_effort`. Clients
configure it themselves:

- **opencode** (`@ai-sdk/openai-compatible` provider): per-model
  `options.reasoningEffort` (e.g. `"high"`), plus built-in variants
  `none`/`minimal`/`low`/`medium`/`high`/`xhigh`:

  ```jsonc
  {
    "provider": {
      "llama.cpp": {
        "npm": "@ai-sdk/openai-compatible",
        "options": { "baseURL": "http://127.0.0.1:8080/v1" },
        "models": {
          "qwen3.8-27b": { "name": "Qwen3.8-27B (local)",
                           "options": { "reasoningEffort": "medium" } }
        }
      }
    }
  }
  ```

- **hermes / pi / similar**: llama.cpp providers default to `reasoning: false`
  and don't forward thinking params; they need a per-model override mapping
  `chat_template_kwargs` (`enable_thinking` / `reasoning_effort` /
  `preserve_thinking`) — see the pi `modelOverrides` pattern.

The `chat_template_kwargs` frontmatter key is exposed in each entry's
`metadata` (client-facing only) so UIs know which kwargs a model's template
accepts. The fixed Qwen template defaults `reasoning_effort` to `medium`
(safe); override per-request with `chat_template_kwargs: {reasoning_effort:
high}`.

Reference: [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
recommends `--jinja --chat-template-file chat_template.jinja --reasoning-format
deepseek` for Qwen 3.5/3.6/3.8 — exactly the combo the `chat_template` +
`reasoning-format` settings produce.

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
- **Expressions**: a profile value of the form `"base * N"` (string starting with
  `base *`) is evaluated against the `defaults:` value for that key — e.g.
  `temperature: "base * 0.7"` with `defaults.temperature: 1.0` yields `0.7`.
  Evaluation is sandboxed to the single name `base`; a failed expression warns
  and falls back to the base value. Only keys present in `defaults:` can use it.
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

## vLLM Backend

A model can be served with vLLM instead of llama-server by an override
rule (see below) that sets `backend: vllm` (host binary) or `backend: vllm-docker`
(container). The emitted entry runs `vllm serve`, published to llama-swap's `${PORT}`
host macro. Everything else works identically: aliases/modes (`filters.setParamsByID`),
`metadata`, capabilities, matrix routing.

All three roles are supported, mapped onto vLLM's pooling interface:

| Role | Task flag | Endpoints |
|------|-----------|-----------|
| `chat` | *(generation; speculative decoding applies)* | `/v1/chat/completions` |
| `embeddings` | `--task embed` | `/v1/embeddings` |
| `rerank` | `--task score` | `/v1/rerank`, `/v1/score` |

Sidecars themselves carry no backend key — backend selection, chat templates and
LoRA adapters are all chosen by pattern-scoped override rules in `profiles.yaml`.

```yaml
# profiles.yaml
overrides:
  - when: {base_model: 'qwen3\\.30b'}
    backend: vllm            # or vllm-docker
    hf_repo: Qwen/Qwen3-30B-A3B-Instruct
```

### Model resolution

- `hf_repo` frontmatter wins when declared; otherwise parsed from `hf_url`
  (`https://huggingface.co/{owner}/{repo}`). If neither exists, the local model file path
  (a `.safetensors` checkout) is used as `--model`.
- vLLM serves safetensors — a local GGUF file is *not* used unless it is also an HF checkout.
- A model with only `hf_repo`/`hf_url` (no local file) is valid: `gguf_path` is optional for
  vLLM backends.

### Memory estimation

vLLM has no `llama-fit-params` analog, so VRAM params are sourced differently but flow
through the same `FitParams` pipeline (`model_mib`, `ctx_factor`, `compute_mib`), and are
persisted to the sidecar `fit-params:` block with `source:` `vllm-estimate` /
`safetensors-estimate`. Sources, in order (`vram.py:_fit_params_vllm`):

1. `vllm-memory-estimator` (optional dependency) on the `hf_repo` — reuses vLLM's own
   `ModelConfig`/`KVCacheSpec` logic; maps weights→`model_mib`, activations+workspace+
   overhead→`compute_mib`, per-token KV→`ctx_factor`.
2. Local `.safetensors` header estimate (`utils.estimate_safetensors`).

When neither is available, context falls back to the declared `context_length` (or GGUF
architectural max) and vLLM's own startup profiling bounds the actual allocation. Companion
(mmproj/MTP) folding is skipped for vLLM models — vision/draft heads live inside the HF repo.

`--gpu-memory-utilization` is derived from the same reserve/spare budget llama.cpp uses
(`available / vram`), unless `vllm.gpu_mem_util` is set explicitly in `profiles.yaml`.

### Image / binary precedence

The container image (`vllm-docker`) is resolved, highest to lowest:

1. Per-model `vllm_image:` frontmatter
2. `--vllm-image` CLI flag
3. `vllm.image` in `profiles.yaml`
4. Built-in default (`vllm/vllm-openai:latest`)

The binary (`vllm`) is resolved, highest to lowest:

1. `--vllm-server` CLI flag
2. `vllm.bin` in `profiles.yaml`
3. Built-in default (`vllm` on PATH)

`profiles.yaml` `vllm:` also configures `docker_args` and `container_port` (`vllm-docker`).

### Limitations

- Accurate sizing requires `vllm-memory-estimator` (or a local `.safetensors` file); otherwise
  context falls back to the declared `context_length`.
- LoRA adapters are emitted for llama-server only; vLLM `loras:` is warned and ignored.
- Baked-in MTP weights are not added to the VRAM budget; GGUF draft companions cannot be
  loaded by vLLM (see [Speculative decoding under vLLM](#speculative-decoding-under-vllm)).
- Runs one vLLM server per model per image/binary; multi-image or cluster/tensor-parallel
  provisioning is future work.

## Image Backend (sd-server)

A model with `role: image` (selected via `dirs: {img: image}` and/or `type: image`) is served with
`sd-server` from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (`backend: sd-server`).
The emitted entry runs `sd-server --diffusion-model …` plus `cli_args` verbatim, proxied via llama-swap's `${PORT}`.

All diffusion formats are supported: `.gguf` (flux, sdxl, sd3, wan, chroma, … — see `_DIFFUSION_ARCH_RES`) and
`.safetensors` VAE / text-encoder companions. Classification is header-only, never filename-based.

| Role | Model file | Endpoints (via llama-swap proxy) |
|------|------------|-----------------------------------|
| `image` | diffusion GGUF / safetensors | `/sdapi/v1/txt2img`, `/sdapi/v1/img2img`, compat `/v1/images/generations`, `/v1/images/edits` |

Emission (see `docs/plans/comfyui-sd.md`):

```yaml
# sidecar: img/flux-ae.safetensors + img/flux-4b.gguf + img/flux-4b.md
# flux-4b.md:
# ---
# name: flux-4b
# # llm: qwen-4b.gguf
# ---
# With dirs: {img: image} and backends: [llama-server, sd-server] the entry emits:
# cmd: sd-server --listen-port ${PORT} --listen-ip 0.0.0.0 --diffusion-model ${MODELS_DIR}/img/flux-4b.gguf  # add --vae etc. via cli_args:
# proxy: http://127.0.0.1:${PORT}
# checkEndpoint: /          # sd-server returns 200 on / only (not /health — Discussion #866)
# capabilities: {in: [text, image], out: [image]}   # txt2img + img2img editing both advertised
```

### Image model resolution

- `model:` (diffusion weights) resolved like all models: `model: file.gguf` relative to sidecar, then HF hub snapshot (`hf_repo` + `model:`).
- A model with only diffusion GGUF is valid (some architectures bake the VAE).

### Memory estimation

`sd-server` has no `llama-fit-params` analog, so VRAM is fixed overhead: `model_mib = file-size(diffusion)`, `ctx_factor = 0`, `compute_mib = 512`. `ctx_size` tracks `design_context` (sidecar `context_length` or default) but does not affect VRAM; the entry is excluded from the shared `chat + emb + rnk` matrix solve (a 40 GB diffusion model would otherwise collapse the chat budget).

### Binary precedence

`sd-server` is resolved: `--sd-server` CLI flag > `sd.bin` in `profiles.yaml` > `$SD_BIN_DIR` (file or directory containing `sd-server`) > `sd-server` on `PATH`. Absent binary disables format-based inference; an explicit `backend: sd-server` pin to a disabled setup is an error that skips the model. Docker variant is future work (tracked in `docs/plans/comfyui-sd.md`).

### Capabilities

`role: image` emits `capabilities: {in: [text, image], out: [image], context: design_context}` — image outputs, text+image inputs (so llama-swap shows both `Image Gen` and `Img→Img` badges). `vision/audio/speech` capabilities are ignored for `image` roles (those are chat-only). `proxy` / `checkEndpoint` are always emitted for `sd-server` entries.

### Limitations

- Single `sd-server` per diffusion model; no multi-UNet or distributed setup.
- VRAM sizing is fixed and deliberately conservative — large flux/sdxl models should be sized via `spare` / `baseline` or explicit `hardware.vram` tuning, not matrix sharing.
- `cli_args` pass-through works, but backend-specific flags (`--diffusion-fa`, `--offload-to-cpu`, `--lora-model-dir`) are operator-provided via sidecar `cli_args:` until first-class `sd_*` keys are added.
- ComfyUI (`comfyui-boot`) remains future work via the same `image` role — see `docs/plans/comfyui-sd.md` for `compat.ignoreWebsockets` / `upstream.ignorePaths` shape.

## Audio Backend (whisper-server)

A model with `role: s2t` (opt-in: an `s2t/` directory plus `dirs: {s2t: s2t}` in
`profiles.yaml`) is served with **whisper-server** from
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`backend: whisper-server`) —
the long-lived HTTP server from `examples/server` (the analog of `llama-server`;
`whisper-cli` is oneshot and cannot be proxied by llama-swap). Exposes
OpenAI-compatible `POST /v1/audio/transcriptions`.

```yaml
# profiles.yaml
dirs: {s2t: s2t}
backends: [llama-server, whisper-server]

# sidecar: s2t/ggml-large-v3.md (authored — .bin orphans are never stubbed)
---
name: whisper-large-v3
parameters: 1.5B
---
```

Emitted entry:

```yaml
whisper-large-v3:
  cmd: whisper-server --host 0.0.0.0 --port ${PORT} --model ${MODELS_DIR}/s2t/ggml-large-v3.bin --parallel 1
  proxy: http://127.0.0.1:${PORT}
  checkEndpoint: /        # same /health pitfall as sd-server (Discussion #866)
  capabilities: {in: [audio], out: [text]}
```

### s2t model resolution

- GGML `.bin` has no header fingerprint, so the directory is authoritative:
  `.bin` files resolve only inside an `s2t`-mapped directory, by same-stem
  sidecar convention (or frontmatter `model:`).
- `.bin` orphans are never stubbed and never auto-served — a bare `.bin` with
  no same-stem `.md` logs one info line and is skipped (few whisper models,
  low churn; authored sidecars only).
- A `.bin` beside a sidecar in any non-s2t role resolves but fails backend
  inference ("no available backend supports format '.bin'") and the model is skipped.

### Binary precedence

`--whisper-server` CLI flag > `whisper.bin` in `profiles.yaml` `whisper:` section >
`$WHISPER_BIN_DIR` (file or directory containing `whisper-server`) >
`whisper-server` on `PATH`. Absent binary disables format-based inference;
an explicit `backend: whisper-server` pin to a disabled setup is an error that
skips the model. Docker variant is future work.

### Capabilities and VRAM

`role: s2t` emits `capabilities: {in: [audio], out: [text]}` (Transcription
badge); declared `vision/audio/speech` capabilities are chat-only and ignored
for s2t roles. VRAM is fixed overhead like sd-server: `model_mib = file-size`,
`ctx_factor = 0`, `compute_mib = 512`; `ctx_size` tracks `design_context`
(sidecar `context_length` or default) without affecting VRAM, and the entry is
excluded from the shared chat matrix solve. The emitted `--parallel` maps the
sidecar/profile slot count to concurrent transcription workers.


## Override Rules

Sidecars carry model-intrinsic data. Cross-cutting serving choices —
**backend**, **HF repo**, **chat template**, **LoRA adapters**, and extra
**CLI args** — are selected by pattern-scoped rules under `overrides:` in
`profiles.yaml` (a sidecar may still pin a `backend:` for one-off exceptions,
but fleet-level policy belongs here).

Rules can also live in a **directory-scoped** `models.yaml`: any subdirectory
of a models root may carry one whose `overrides:` apply only to models in
that subtree, and whose `defaults:` seed each subtree sidecar's frontmatter
(see [Directory-scoped models.yaml](#directory-scoped-modelsyaml) below).

```yaml
# profiles.yaml
overrides:
  # All Qwen 3.6/3.8 chat models get the fixed Jinja chat template (and its
  # declared kwargs, exposed to clients for per-request control).
  - when: {base_model: 'qwen3\.[68]'}
    chat_template: qwen_chat_template.jinja
    chat_template_kwargs: {enable_thinking: true}

  # One-off: pin a specific model to the vLLM container backend.
  - when: {name: 'Nail-Qwen'}
    backend: vllm-docker
    hf_repo: peculiar-ragdoll/Nail-Qwen3.6-35B-A3B-GGUF-MTP

  # A qwen finetune (still base_model: qwen3.8) also gets a LoRA adapter.
  - when: {base_model: 'qwen3\.8', parameters: '27B'}
    loras: [my-qwen36-uncensored-lora.gguf]
```

**Matching.** `when` is **required** — a rule that isn't a mapping, has an
empty/null `when`, a non-mapping `when`, or no known settings, aborts the run
(a silently-ignored rule would compound the misconfiguration). Each `when` is
a map of `field: regex`; a model matches only if *every* field regex matches
(`re.search`). Field semantics (`overrides.py:rule_matches`):

- Fields come from the sidecar frontmatter plus the synthetic `stem` and
  `name`.
- **List-valued fields are joined with spaces** before matching, so
  `capabilities: 'reasoning'` matches `[vision, tools, reasoning]`.
- A field **absent from a model never matches** (there is nothing to search) —
  to target "has no X", match a different field or invert in rule order.
- An **invalid regex** logs a warning and that rule matches nothing (the run
  continues; only structural problems abort).
- Unknown settings keys in a rule log a warning and are ignored; if *no* key
  is known, the rule aborts the run.

Use `when: true` to match every model. Regex literals are easiest in YAML
**single-quoted** or unquoted scalars — only double quotes interpret
backslashes (`\.` stays literal in single quotes).

**Merge semantics.** Settings seed from the model's own sidecar fields, then
each matching rule is layered on top **last-match-wins per key** (CSS-like:
rules read top→bottom as increasing specificity). So a later rule that changes
`backend` does not clobber a `chat_template` set by an earlier rule.

**Precedence across scopes.** Global rules apply first, then directory-scoped
rules outermost → innermost — so a closer scope beats a broader one beats
global for the same key (the flat rule list accumulated by
`scope.ScopeStack` during discovery's walk).

### Directory-scoped models.yaml

Any subdirectory of a models root may carry a `models.yaml`. It makes the
directory itself the filter — useful when HF naming makes regexes brittle
(drop models in a folder instead of writing `when: {base_model: …}`):

```yaml
# <models-root>/chat/qwen3/models.yaml
defaults:
  context_length: 16384          # frontmatter defaults for subtree sidecars

overrides:
  - when: true                   # full filter syntax available; true = all
    chat_template: ../qwen_chat_template.jinja
    chat_template_kwargs: {enable_thinking: true}
```

- **Scope**: both keys apply only to models under that directory.
- **`defaults:`**: merged into each subtree sidecar's frontmatter, outermost →
  innermost; authored sidecar values always win. Empty stub sidecars carry no
  data, so defaults fill them naturally. The per-model identity keys `name`,
  `model`, `ignore` may not be defaulted (validation error).
- **`overrides:`**: standard rules (same validation and matching); applied
  after global rules, outer scopes first — innermost wins per key.
- **Paths**: `chat_template:` / `loras:` resolve relative to each *sidecar's*
  directory (not the models.yaml), so reference shared files with `../`.
- **Entry-id collisions** are fatal: if two models slug to the same llama-swap
  entry id, the run logs an error and exits — rename one of them.

**Settings keys** (all optional): `backend`, `hf_repo`, `chat_template`,
`chat_template_kwargs`, `loras`, `cli_args`, `reasoning-format`,
`reasoning-preserve`, plus the serving/companion choices `cache_type`,
`parallel`, `mmproj`, `speculative`. Rules setting `mmproj`/`speculative`
re-trigger companion resolution, so a rule can add or remove vision /
speculative decoding per pattern.

**Backend inference.** When neither the sidecar nor any rule declares a
`backend`, one is inferred from the model's file format (`backends.infer_backend`):
the registry walks backends in **registration order** — `llama-server`,
`vllm-docker`, `vllm` — and picks the first whose registered formats cover the
model AND whose required resources are configured (llama-server binary, vLLM
image / binary). For this purpose **a locally resolved model file's extension
wins over `hf_repo`**: an HF repo id only drives selection when the model has
no local file. Today that means `.gguf` → `llama-server` and safetensors /
`hf_repo` → `vllm-docker` (falling back to host `vllm` when only the binary is
configured). A format no available backend covers logs an error and the
model's entries are skipped; so does a rule or sidecar naming an unregistered
`backend`.

**Path resolution.** `chat_template` and `loras` values are paths resolved
relative to the sidecar's own directory (absolute refs pass through). A missing
file logs an error and the model's entries are skipped (fail loud). Symlinks
are preserved by name (not dereferenced), so a chat template symlinked into the
HF cache stays under `${MODELS_DIR}` instead of widening it. Resolved paths are
written into the generated `cmd` as `${VAR}` path macros.

**Chat templates & client kwargs.** A declared `chat_template` makes the writer
emit `--jinja --chat-template-file <path>` (llama-server) or `--chat-template
<path>` (vLLM), and records `metadata.chat_template` (the file stem). The
`chat_template_kwargs` map is **client-facing metadata only** — there is no
server-side flag for it; clients pass it per-request (e.g. Qwen's
`enable_thinking`).

**Backend support matrix.** A backend that cannot serve a model — wrong file
format (e.g. a `.gguf` under vLLM) or unsupported role (e.g. embeddings/rerank
under vLLM in this version) — logs an error and the model's entries are skipped
entirely. A backend that recognizes a setting it cannot render (e.g. `loras`
under vLLM) logs a warning and ignores that one setting; settings that simply do
not apply (e.g. `cache_type` under vLLM) are silently dropped.

| Backend | Model formats | Roles |
|---------|--------------|-------|
| `llama-server` | `.gguf` | chat, embeddings, rerank |
| `vllm` | safetensors, `hf_repo` | chat |
| `vllm-docker` | safetensors, `hf_repo` | chat |
| `sd-server` | `.gguf`, `.safetensors`, `hf_repo` | image |
| `whisper-server` | `.bin` (s2t dir only) | s2t |

## Backend Selection

profiles.yaml's ordered `backends:` list both **enables** and **prioritizes**
backends; when absent, every registered backend is usable in registration
order (`llama-server`, `vllm-docker`, `vllm`, `sd-server`):

```yaml
# profiles.yaml
backends:
  - llama-server    # tried first for everything it can serve
  - vllm-docker     # enabled, second preference
  - sd-server       # image generation (opt-in; needs dirs: img: image)
  # vllm            # absent = disabled, even with resources configured
```

Inference walks this list (availability still filters: an entry without its
binary/image configured is skipped) and picks the first backend whose formats
and roles cover the model. An explicit sidecar/override `backend:` pin to a
disabled name is an error that skips that model — pinning bypasses *inference*,
never policy. Registration order: `llama-server`, `vllm-docker`, `vllm`,
`sd-server`, `whisper-server`.

## Cache precision (`cache_type`)

A single `cache_type:` line in a sidecar selects the KV-cache precision and is
used for **both** the emitted `--cache-type-k`/`--cache-type-v` flags and the
VRAM calculation (sidecar > profile `defaults.cache_type` > `q8_0`). K and V
caches are assumed to share the same precision. `parallel` follows the same
precedence (sidecar > profile > 1).

**Valid values** are exactly the keys of `utils._KV_CACHE_BYTES` — the
precisions llama-packer can size memory for. Anything else is logged as an
error and the model is skipped (an unsizable cache would make every context
calculation a guess):

| Precision | Bytes/element (rounded up) |
|-----------|---------------------------|
| `f32` | 4.0 |
| `f16`, `bf16` | 2.0 |
| `q8_0`, `q8_1`, `q8_k` | 1.0625 |
| `q6_0`, `q6_k` | 0.8125 |
| `q5_0` | 0.6875 |
| `q5_1` | 0.75 |
| `q5_k` | 0.6875 |
| `q4_0`, `q4_k`, `iq4_nl` | 0.5625 |
| `q4_1` | 0.625 |
| `nvfp4` | 0.5625 |

**vLLM translation**: valid `--kv-cache-dtype` values pass through —
`q8_*` → `fp8`, `f16`/`bf16`/`f32` → auto (no flag), `nvfp4` →
`nvfp4` (experimental upstream; whether the serving build and hardware
support it is the operator's call). The k-quants (`q4_*`, `q5_*`, `q6_*`,
`iq4_nl`) have no vLLM equivalent: warned and the flag omitted, while VRAM
sizing still uses the declared precision (conservative).

### Cache memory math

The KV cache is linear in tokens and in cache precision:

```
kv_bytes_per_token = 2 × Σ_layers (k_proj_out_dim + v_proj_out_dim) × bytes_per_element
ctx_factor [MiB/token] = kv_bytes_per_token / 2²⁰   (+ attention scratch, measured)
```

- The per-layer K/V output dims come from a `llama-fit-params` measurement of
  the main model; the safetensors fallback (`utils.estimate_safetensors`) reads
  them from `k_proj`/`v_proj` tensor shapes in the header.
- **Cache-type scaling**: only the `bytes_per_element` term depends on
  precision, so a `cache_type` change reuses the persisted measurement and
  rescales it — `ctx_factor_new = ctx_factor_old × bytes(new) / bytes(old)`
  (`vram.py:_scale_ctx_factor`, rounding up so estimates err toward reserving
  more). `model_mib` and `compute_mib` are precision-independent and carry
  over unchanged. A `parallel` change still triggers a fresh measure.
- Example: a model measured at `ctx_factor = 0.5` MiB/token under `f16`
  serves `q8_0` at `0.5 × 1.0625/2 ≈ 0.266` MiB/token — roughly half the KV
  footprint, doubling the affordable context for the same VRAM.

## Model Discovery and Stub Sidecars

Every `--models-dir` directory is scanned independently via a depth-first
walk (`discover.discover` → `scope.ScopeStack`; role mapping via
`utils.dir_role_map`). At each level the directory's `models.yaml` scope is
pushed, its models are built, then children are visited:

- `.md` sidecar files are the entry points; each binds to the model file whose
  stem matches its own, or the file named by `model:`.
- Within a models dir the **first relative path component** selects the role
  via `dirs:` in profiles.yaml (case-insensitive). Defaults:

  | Prefix | Role | Meaning |
  |--------|------|---------|
  | `chat` | `chat` | Chat — and its mmproj/MTP companions living next to it |
  | `t2t` | `chat` | Legacy name for the chat dir |
  | `vision` | `chat` | VLMs + mmproj (same role as chat, colocated) |
  | `doc`, `ocr` | `chat` | OCR / extraction / file-format models (chat-role, organizational split; `doc` is the canonical name, `ocr` its legacy alias) |
| `embed` | `embeddings` | Embedding models; nested subdirs (e.g. `embed/jina-v5/`) keep the role |
| `rerank` | `rerank` | Reranker models |
| `s2t` | `s2t` | Speech-to-text (whisper.cpp GGML `.bin`; opt-in via `dirs: {s2t: s2t}`) |
| `img` | `image` | Diffusion / image generation (sd-server; opt-in via `dirs: {img: image}`) |

  Files at the root itself default to `chat`; files under any other
  subdirectory (`img/` when not opted in, `misc/`, `tmp/`,
  `hf_hub/`, … — and `s2t/`, `img/` when not opted in) are **skipped** — one summary line per run names the
  skipped directories so nothing disappears silently. The whitelist is
  extendable via profiles.yaml `dirs:` (e.g. `{ocr: chat, it2t: chat}`) and
  via CLI `--extra-dirs` (backcompat for `embed`/`rerank`).

  A `<models-root>/.modelignore` file excludes individual files/subtrees in
  place (no moving or deleting): one glob per line, `#` comments; a pattern
  matches the path relative to the root or any single path component, so
  `R3-rerank` hides that subtree and `adetailer*` hides everything named like
  it. Matched files are summarized in one log line.
- Orphan GGUFs next to chat models are classified as companions (mmproj /
  MTP draft), never as main models.
- Hardlinks and symlinks resolving to the same `(st_dev, st_ino)` are deduplicated
  across directories (first `models_dirs` entry wins).

`--models-dir` precedence: CLI `--models-dir` (when given) > profiles.yaml
`models_dirs:` (a list) > `./models`. The tracked `profiles.yaml.example` is
the template; the live `profiles.yaml` is machine-local (gitignored). When no
profiles file exists, llama-packer logs a warning pointing at the example and
proceeds with the bundled defaults.

**Directory-scoped config.** Any subdirectory may carry a `models.yaml`
applying only to its subtree — see [Override Rules → Directory-scoped
models.yaml](#directory-scoped-modelsyaml).

**HF hub cache resolution.** A sidecar can reference a hub-downloaded GGUF without
symlinking it into a models dir: declare both `hf_repo: org/repo` (or a
parseable `hf_url:`) **and** `model: file.gguf`. Snapshot filenames are
readable — blob hashes never appear in sidecars. Resolution:

1. `model:` relative to the sidecar's dir (and its parent), then
2. `$HF_HOME/hub/models--org--repo/snapshots/<rev>/file.gguf`, revision from
   `refs/main`, else the sole snapshot dir, else the newest by mtime.

Companions resolve the same way after the local search misses:
`mmproj:` / `speculative:` values are looked up in the sidecar's repo
snapshot, with a single-glob fallback (`mmproj*.gguf`) covering HF's naming
variants (`mmproj-F16.gguf`, `mmproj-model-f16.gguf`, …). When no `mmproj:`
is declared at all, the snapshot is fuzzy-scanned for a family-matching
`*mmproj*.gguf`, mirroring the local-directory behavior. A value may also
address another cached repo explicitly: `hub:<org>/<repo>:<file-or-glob>`.
An ambiguous glob logs a warning and does not resolve.

The HF cache root is `--hf-home` > profiles.yaml `hf_home:` >
`$HF_HOME`/`$HUGGINGFACE_HUB_CACHE` > `~/.cache/huggingface`. By default it
points at your `/mnt/ai/huggingface`. With this, `hf download org/repo`
followed by a small `.md` sidecar is sufficient — no symlink step and no
widening of `${MODELS_DIR}` (HF cache paths get their own `${HF_HOME}` macro).

**Stub sidecars.** A model file without any sidecar gets an **empty** one
written next to it — just frontmatter delimiters and a title, nothing more.
Identity falls back to the file stem, context to the built-in default, role to
the model's directory, so a stub and an authored sidecar behave identically;
the empty file exists purely as the human's editing surface ("drop in a gguf,
get a placeholder to fill in"). This makes a directory of bare models and any
new `embed/`/`rerank`/`doc/` orphans work on first run; `--no-stubs` skips
generation.

Sidecars are never written inside an HF hub `blobs/` tree (blob hashes are not
human-readable names). An orphan discovered there gets its stub beside the
human-named snapshot entry that points at the blob; if none resolves, discovery
creates its own symlink named after the repo in the category directory and puts
the stub next to it.

## Path macros and HF_HOME

Generated commands are emitted with absolute paths, then rewritten to
`${LLAMA_DIR}` / `${MODELS_DIR}` / `${MODELS_DIR_2}`… macros via
`compute_env_prefixes` (grouped by mount, longest common directory per group).
Paths under the Hugging Face cache root (`--hf-home` > profiles.yaml `hf_home:`
> `$HF_HOME` > `$HUGGINGFACE_HUB_CACHE` > `~/.cache/huggingface`, the default
`hf_home: /mnt/ai/huggingface` in your profiles.yaml) are pulled into their own
`${HF_HOME}` macro so a chat template (or LoRA) living in the HF cache never
widens `${MODELS_DIR}` up to a non-models directory.


## Health-Check Timeout

Auto-calculated when not explicitly set via `--health-check-timeout`:

```
hct = max(120, int(1.2 × largest_model_mb / drive_speed_mb))
```

Drive speed is detected per-model via `lsblk` (NVMe → 1500 MB/s, SATA SSD → 300 MB/s, HDD → 100 MB/s, unknown → 100 MB/s). The slowest drive among all model files bounds the timeout.

Override via `--health-check-timeout`, `--drive-speed`, or the `GEN_CONFIG_DRIVE_SPEED` environment variable.

## Path Macros (`macros:` block and `config.env`)

All emitted paths (binary, GGUF files, mmproj, MTP companions, chat templates,
LoRAs) are grouped by filesystem mount via `utils.compute_env_prefixes`. The
longest common directory per group becomes a llama-swap **macro**: paths in the
generated `cmd` are rewritten to `${VAR}` form, and a top-level `macros:` block
in `config.yaml` maps each macro to its absolute directory — so `-watch-config`
reloads pick up moved/updated paths without a llama-swap restart.

| Macro | Content |
|-------|---------|
| `LLAMA_DIR` | Group containing the llama-server binary |
| `HF_HOME` | Paths under the HF cache root (`--hf-home` > `$HF_HOME` > `$HUGGINGFACE_HUB_CACHE` > `~/.cache/huggingface`) |
| `MODELS_DIR` | First model mount group |
| `MODELS_DIR_2` ... | Additional groups (sorted by mount path) |

The same `NAME=value` pairs are also written to a sibling `config.env` for
systemd `EnvironmentFile=` / docker `--env-file` consumption; skip it with
`--no-env`.

## Model Metadata

Sidecar `.md` files declare model config. The generator is **pass-through-by-default**: any frontmatter key not consumed by the builder is exposed to clients automatically.

### Metadata channel

Model identity and agent-selection metadata is carried entirely by the llama-swap config — no
`--override-kv` flags in `cmd`. Fields with native llama-swap support use native keys; everything
else flows into the per-model `metadata` dict (→ `meta.llamaswap` in `/v1/models`):

- **`metadata`** — the agent-choice descriptor: `freethought`, `strengths`, `weaknesses`,
  `license`, `base_model`, `finetune`, `type`, `parameters`, `quantization`, `hf_url`,
  `ctx_size`, `mtp_enabled`, `mtp_draft_max`, `mtp_accuracy`, `throughput_factor`.
- **Native `capabilities`** — `in`/`out` modalities, `tools`, `reranker`, `context` (derived, see below).
- **Native `name` / `description`** — display fields in `/v1/models`.

### Builder-consumed keys (NOT passed through)

`name`, `context_length`, `description`, `cli_args`, `model`, `backend`, `hf_repo`,
`chat_template`, `chat_template_kwargs`, `loras`, `attention`, `kv_cache`, `tool_args`,
`speculative`, `speculative_config`, `mmproj`, `mtp`, `mtp_spec_type`, `mtp_draft_n_max`,
`mtp_draft_p_min`, `role`, `targets`, `allow_profiles`, `spare`, `capabilities`,
`ignore`, `device`, `concurrency`, `fit-params`, `vllm_image`, `modes`, `default_mode`,
`reasoning-format`, `reasoning-preserve`, `cache_type`, `parallel`.

### Per-model config options

| Frontmatter Key | Type | Effect |
|-----------------|------|--------|
| `device` | int | GPU device index for multi-GPU pinning (`ROCR_VISIBLE_DEVICES=N` / `CUDA_VISIBLE_DEVICES=N`) |
| `concurrency` | int | Per-model concurrency limit → `concurrencyLimit` in config |
| `spare` | str | Additional VRAM to reserve (overrides global `--spare`) |
| `allow_profiles` | str/list/bool | Restrict which profiles apply (regex string, list, or false to disable) |
| `modes` | dict | Per-model sampling modes (full profiles): name → param dict. Replaces the global-profile sampling overrides for this model. Values use llama.cpp names; see [Sampling Modes](#sampling-modes) |
| `default_mode` | str | Which declared `modes` entry is the model's default (maps to the bare `${MODEL_ID}` `setParamsByID` key). Falls back to the first mode |
| `reasoning-format` | str | llama-server `--reasoning-format` (`none`/`deepseek`/`deepseek-legacy`/`auto`). Chat + reasoning-capable models only; see [Reasoning](#reasoning) |
| `reasoning-preserve` | bool | Emit `--reasoning-preserve`. Chat + reasoning-capable models only |
| `cache_type` | str | KV-cache precision for `--cache-type-k/v` and VRAM sizing (sidecar > profile > `q8_0`); see [Cache precision](#cache-precision-cache_type) |
| `parallel` | int | Parallel slots for `--parallel` and VRAM sizing (sidecar > profile > 1) |
| `ignore` | bool | Skip this model entirely |

### Agent-selection fields (optional, recommended)

| Field | Type | Meaning |
|-------|------|---------|
| `capabilities` | list | `[vision, tools, reasoning, audio, speech]`; `vision` auto-added if a companion `mmproj` exists. Mapped to the native llama-swap `capabilities` block (directional, see below) |
| `freethought` | float 0–1 | `1.0` reasons about anything; `0.0` readily refuses "distasteful" topics. Carried in `metadata` |
| `strengths` / `weaknesses` | list | Concise task phrases agents match on |
| `license` / `base_model` / `finetune` / `type` | str | Identity; carried in `metadata` |
| `mtp_accuracy` | float | MTP draft acceptance rate; feeds `throughput_factor` |
| `parameters` | str | `"12B"` or MoE `"26B-A4B"` (total-active) for accurate throughput |
| `hf_url` | str | HuggingFace model URL |
| `hf_repo` | str | HF repo id for vLLM backends. Optional; parsed from `hf_url` when absent |
| `vllm_image` | str | Per-model vLLM docker image. Overrides profiles.yaml `vllm.image` and `--vllm-image` for this entry |

### Derived fields (computed, not authored)

- **`capabilities`** (native block): directional modalities, matching llama-swap's badge derivation — `in` = `["text"]` + `"image"` if `vision` + `"audio"` if `audio`; `out` = `["text"]` + `"audio"` if `speech`. Output stays text unless `speech` is declared, so a vision model never advertises text→image (`Image Gen`) or image→image (`Img→Img`). `tools`/`reranker` boolean flags; `context` = design context (the model's maximum trained context: GGUF architectural max > sidecar `context_length` > default).
- **Model-kind guard**: every candidate in a served role is classified header-only (`general.architecture` + `<arch>.context_length` for GGUF; tensor-name blocks for safetensors; cached HF card `pipeline_tag` as offline fallback). Weights classified as diffusion/image-generation are excluded with an error log instead of being served.
- **`throughput_factor`**: Heuristic relative speed index = `54 / (active_B × quant_bits)` × `(1 + draft_n × mtp_accuracy)` when MTP is on. Relative only — not real tok/s.
- **`ctx_size`**: The VRAM-served context limit (`-c` / `--max-model-len`), exposed via `metadata.ctx_size`. Distinct from `capabilities.context`, which advertises the model's max trained context rather than what the deployment can currently fit.

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

| Constant | Default | Meaning |
|----------|---------|---------|
| `_DEFAULT_CONTEXT_LENGTH` | 32768 | Fallback when no `.md` sidecar or GGUF context exists |
| `_CTX_ROUND_TO` | 8192 | Round context size down to nearest boundary |
| `_MIN_CTX_SIZE` | 4096 | Hard floor for context size |
| `_MIN_USEFUL_CTX` | 131072 | Min useful chat context; mmproj dropped below this (`--min-context`) |
| `_RESERVE_SYSTEM` | 1024 | MB reserved for OS/driver/scratch buffers |
| `_RESERVE_VIDEO` | 1024 | MB reserved for GPU video output framebuffer |

In `llama_packer/utils.py`; the companion/MTP and cache-precision constants live in `llama_packer/vram.py` / `utils.py`:

| Constant | Defined in | Meaning |
|----------|-----------|---------|
| `_MMPROJ_COMPUTE_MB` | `vram.py` | Fixed compute buffer for mmproj companions (150) |
| `_DRAFT_COMPUTE_MB` | `vram.py` | Fixed compute overhead for MTP draft companions (64) |
| `_DRAFT_CTX_SAFETY` | `vram.py` | Safety factor on the MTP draft per-token KV estimate (1.6) |
| `_KV_CACHE_BYTES` | `utils.py` | Bytes/element per KV-cache precision; the set of sizeable `cache_type` values (see [Cache precision](#cache-precision-cache_type)) |
| `_MTP_SPEC_TYPE` | `utils.py` | Default MTP speculative type (`"draft-mtp"`) |
| `_MTP_DRAFT_N_MAX` | `utils.py` | Default max draft tokens for MTP (2) |

## Future Work — Log-Derived Real Throughput (Phase 6)

`throughput_factor` is a heuristic. A planned offline enrichment derives **measured** throughput from llama-server logs and overrides the heuristic when available.

- **Script:** `scripts/parse_llama_logs.py` (run out-of-band, e.g. cron or on-demand).
- **Source:** `journalctl -u llama-swap.service` (or a `--log` file).
- **Parsing:** extract the launch cmd per request window to get the `-m` model path/stem; capture tok/s from llama.cpp's known lines:
  - `prompt eval time = … (N tokens per second)` → preprocessing (pp) tok/s
  - `eval time = … (N tokens per second)` → generation (tg) tok/s
  - fallbacks: `… tok/s`, `… t/s`. Tolerant regexes; multiple patterns.
- **Aggregation:** average per model stem → `models/.throughput_cache.json` `{stem: {tps, pp_tps, samples, updated}}`.
- **Consumption:** `llama-packer` reads the cache and adds `observed_tps` / `observed_pp_tps` to `metadata` when present, overriding `throughput_factor`.
- **Graceful:** new/unrun models simply lack the cache entry — no change to the core metadata pipeline required (pure enrichment source).
