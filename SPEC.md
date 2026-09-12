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
| `matrix` | Shared embed/rerank/chat VRAM budget solving + opportunistic s2t/image co-loads + knobs (`min_chat_ctx`, `tools_min_ctx`, …) | [Matrix Context Solving](#matrix-context-solving) |
| `hardware` | `vram`, `baseline_mb`, `unified_system_mb`, `gpu_family` overrides | [Hardware Detection](#hardware-detection) |
| `vllm` | Backend resources: `image`, `bin`, `docker_args`, `container_port`, optional `gpu_mem_util` / `hf_cache` | [vLLM Backend](#vllm-backend) |
| `backends` | Ordered enable/prefer list of backend names (absent = all, registration order) | [Backend Selection](#backend-selection) |
| `llama_server` / `vllm` / `sd` / `whisper` | Per-backend fleet-wide `args:` flags (performance tuning) | [Global backend args](#global-backend-args) |
| `models_dirs` | Model root directories (CLI `--models-dir` wins) | [Model Discovery and Stub Sidecars](#model-discovery-and-stub-sidecars) |
| `dirs` | Directory-name → role whitelist (e.g. `{ocr: chat}`) | [Model Discovery and Stub Sidecars](#model-discovery-and-stub-sidecars) |
| `hf_home` | HF_HOME root (the dir containing `hub/`) for hub snapshot resolution (CLI `--hf-home` wins) | [Model Discovery and Stub Sidecars](#model-discovery-and-stub-sidecars), [Path Macros](#path-macros-macros-block-and-configenv) |

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
2. `--llama-version N` → `./llama-bN` if it exists
3. default (`latest`) → the highest-numbered `./llama-b[0-9]*` directory

`llama-server` and `llama-fit-params` are both taken from the resolved
directory; no match aborts with the available versions listed.

## Context Management

The tool calculates `ctx_size` dynamically based on measured VRAM costs, not static estimates.

### Measurement via `llama-fit-params`

VRAM follows a validated affine law (see `llama_packer/memory_probe.py` and
docs/plans/auto-parallel.md):

```
VRAM(C, p) = model_mib + compute_mib + c*C + p*D
```

`C` is the **total** shared KV pool (llama.cpp `-c`; byte-identical to
`--kv-unified-per-slot X -np p` with pool `p*X`), `p` the slot count, and the
constants are:

| Constant | Key | Meaning |
|-----------|------|---------|
| `model_mib` | weight cost | Constant regardless of context or slots |
| `kv_per_token_mib` | `c` | Shared full-attention KV per token (the pool is `X * p` tokens for per-slot context `X`) |
| `slot_mib` | `D` | Fixed cost per parallel slot (SWA ring buffers + slot overhead) |
| `compute_mib` | fixed | Compute/workspace buffers (max over the p=1/p=2 pair) |

A single measurement cannot separate `c` from `D`, so the system measures a
**(p=1, p=2) pair** at the design context: `D = ctx(2) − ctx(1)`,
`c = (ctx(1) − D)/C`. The pair pins the law exactly (validated residuals
< 0.1% across dense/MoE/SWA families and q8_0/f16/bf16/q4_0) and replaces the
former per-parallel sweeps.

Values are persisted to the model's `.md` sidecar `derived:` block — **per
`cache_type` and per measured flag `shape`** (the exact measurement flag
string: global args + the resolved `-ub`; the compute buffer depends on
batch size and flash attention, so blocks are never reused across shapes —
a profiles/batch-key change re-measures instead of serving stale compute
terms).  **Parallel-independent by construction** (the `parallel` and
legacy `ctx_factor` keys of the pre-affine schema are removed on rewrite).
A `cache_type` change invalidates the block and re-measures: `c` would scale
with KV precision (`_KV_CACHE_BYTES` byte ratios), but `D`'s precision
behavior is arch-dependent (SWA ring buffers scale, fixed slot overhead does
not), so blocks are never derived across cache types — re-measuring costs
~1 s. Pre-affine blocks (no `slot_mib`) and pre-shape blocks (no `shape`)
are stale by definition and are re-measured + rewritten on the next run.
`--remeasure` skips the saved-block path for one run (never a server).

**Fallback chain:** saved frontmatter → in-memory cache → `llama-fit-params` (p=1/p=2 pair) → safetensors header estimation (`c` only, `D=0`).

`llama-fit-params` is always invoked with `--fit off` so it measures the explicit `-c` (or design) context rather than auto-adjusting arguments — `--fit on` (the default) would otherwise change `-ngl`/`-c` when the GPU is busy, corrupting the measurement.

**Validation probe.** `llama-packer --probe-memory [ARCH...]` sweeps one
representative per GGUF arch family at p ∈ {1,2,4,8}, derives `(c, D)`, and
reports the max residual against the law (FAIL above 0.5%) — the fast
sanity check when a new architecture family appears. The underlying helpers
(`llama_packer.memory_probe`) are importable by other tools.

### Companion (mmproj / MTP) VRAM accounting

Companion GGUFs cannot be measured by `llama-fit-params` — mmproj files fail to load as standalone models, and MTP draft heads abort on a missing `ctx_other`. Their VRAM is therefore folded into the main model's budget via a combined "effective static" pass (`llama_packer/vram.py:effective_static`):

- **Main model**: measured affine constants.
- **MTP draft**: file-size weight plus estimated affine terms (both `c` and
  `D` scaled from the main model by the size ratio, padded by
  `_DRAFT_CTX_SAFETY` — the draft holds its own KV cache that scales with
  context) and a fixed compute buffer.
- **mmproj**: file-size weight plus a fixed compute buffer (`_MMPROJ_COMPUTE_MB`, ~150 MiB for the vision projection buffers); no KV terms — projection cost is independent of context and slots.

Companion GGUFs are never measured: `llama-fit-params` cannot load them (mmproj fails as a standalone model; MTP draft heads abort on a missing `ctx_other`), so the file-size estimate is used directly (cached per companion). This fixes companion VRAM being under-budgeted by raw file size alone (a Gemma4-31B + MTP + mmproj combination was measured to need ~1.8 GiB more than the sum of companion file sizes).

### Image token budget (vision sidecars)

Dynamic-resolution vision models (Qwen-VL family) convert each image to a resolution-dependent number of LLM tokens: 1 token ≈ 28×28 px for Qwen2.5-VL (14×14 patches, 2×2 merge) and ≈ 32×32 px for Qwen3-VL (16×16 patches). A sidecar can declare `image_min_tokens` / `image_max_tokens` (positive ints) on a vision model; both are pass-through to `llama-server --image-min-tokens/--image-max-tokens`, emitted only on variants that serve the `--mmproj` (text variants drop them silently). There are no packer-side defaults — unset keys mean llama-server reads the model's own metadata (which for e.g. Qwen2.5-VL tops out at 16384 tokens/image).

Semantics:

- **min**: low-resolution images are upscaled so they occupy at least this many tokens — the control for analysis quality. 1024 tokens ≈ 1 MP of effective detail, an adequate floor for art/artifact critique (dense images under-sampled below ~1k tokens tend to get answered from the model's prior instead of the pixels).
- **max**: large images are downscaled to this token ceiling — the control for bounding KV cost and prompt size. Diminishing quality returns beyond ~4k tokens.
- **Static-resolution archs** (Gemma 3/4, SigLIP: fixed resize, ~256 tokens/image) ignore both flags; declaring them there logs a one-time warning and skips emission, as does declaring them on a model with no mmproj companion.

VRAM accounting: image tokens are ordinary tokens inside the slot's context — they consume KV already budgeted by the context solve, not memory beyond it. The budget therefore guarantees **fit**, not extra headroom: the solved per-slot context is never allowed to drop below `image_max_tokens` when the VRAM affords it (`calc_ctx` raises the rounded-down context to that floor; the matrix solve warns when a model's floor exceeds the shared solution). When even the affordable context cannot hold the floor, a warning is logged and oversized images fail at request time instead of the server failing at load time. Declared bounds are advertised to clients as `metadata.image_min_tokens` / `metadata.image_max_tokens` on vision variants.

### Context calculation formula

The solve returns the **per-slot** context `X`; the emitted llama-server flag
is `--kv-unified-per-slot X` (shared pool `parallel * X`) and
`metadata.ctx_size` advertises `X`. vLLM's `--max-model-len` is per-sequence
already, so no flag translation is needed there.

```
reserve = RESERVE_SYSTEM(1024) + max(RESERVE_VIDEO(1024), baseline_mb)
available = (vram_total - reserve - spare) / (1 + memory_margin)
remaining = available - model_mib - compute_mib        # includes companions
X_max = (remaining - p*D) / (c*p)                      # affine solve
X = floor(X_max / CTX_ROUND_TO) * CTX_ROUND_TO             # round down to 8192 boundary
X = max(X, MIN_CTX_SIZE)                                   # floor at 4096
X = min(X, gguf_context_length)                            # cap at GGUF architectural max
X = min(X, sidecar_context_length)                         # cap at sidecar ceiling
X = min(X, max_context)                                    # cap at CLI --max-context
```

If the design context fits (`(c*design + D) * p ≤ remaining`), it is used
directly without scaling down. `memory_margin` (profiles.yaml
`hardware.memory_margin`, default 0.01) inflates every measured term so
measurement residual errs toward reserving more.

`baseline_mb` is the VRAM already consumed by the driver/compositor/other processes. It is **opt-in**: set via `--baseline` or `profiles.yaml` `hardware.baseline_mb`, and defaults to **0**. The fixed `_RESERVE_VIDEO` (1024 MiB) already covers driver/compositor overhead, so auto-detection of the live `used_vram` is intentionally NOT performed — llama-swap keeps model servers resident, and counting that usage would make the budget assume a blank GPU and collapse every context to the minimum. The effective reserve is the fixed system reserve (1024) plus the larger of the fixed video reserve (1024) and any explicit `baseline_mb`. `--spare` is subtracted on top of this reserve. CPU-resident models (`device: cpu`) are excluded from VRAM budgeting entirely and are sized to their own architectural/sidecar context.

**Design context:** When the affine pair is measured, it uses the model's GGUF architectural context length as the reference point (or sidecar `context_length`, or 32768 default). If the design context fits within the remaining budget, it is used directly without scaling down.

**Note:** parallel slots cost the fixed `p*D` term plus their share of the pool (`c * p * X`) — both scale with the slot count, so the solve charges them exactly.

### Minimum useful context and vision (mmproj) skipping

Chat models target a minimum useful context (`_MIN_AGENTIC_CTX`, default 131072 = 128k, overridable with `--min-context`). For a chat model with an mmproj companion, `calc_ctx` is evaluated both with and without the companion (at the global `--spare`) — the companion-on variant serving the block merged over the frontmatter, the companion-off variant serving the base frontmatter:

- `ctx_with ≥ min_context` → keep vision; the main entry is emitted with `--mmproj` and the `image` capability. An on-demand **text-only variant** `<id>-text` (no `--mmproj`, `image` removed, `metadata.mmproj_skipped: true`, display name `[text]`) is emitted alongside so clients can pick the lower-memory serving.
- `ctx_with < min_context` → the main entry **drops** mmproj and is renamed `<id>-text` — the invariant is that the bare `<id>` always serves vision when the model has one; every no-mmproj entry carries the `-text` suffix and `[text]` label so a client that knows nothing of server config can tell it is text-only from `/v1/models`. A companion **vision variant** entry is additionally emitted with `--mmproj` at best-effort context, id-suffixed `-vision-<N>k` where `N = ctx_with // 1000` (e.g. 92567 → `-vision-92k`), display name `[vision Nk]`, keeping vision available at reduced context.
- `ctx_without < min_context` too → **vision is kept**: dropping the projection cannot reach the minimum either way, so sacrificing it buys nothing (a small VLM stays a full VLM). Informational log only — no warning, since no configuration can fix a design-context limit.

All emitted entries honor the per-profile `spare_mb` and the matrix-solved chat context; the drop decision itself is made once per model using the global spare. Every `<id>-text` entry joins the same matrix co-loading sets as its parent `<id>` entry, so `(c1 | … | cN) & emb & rnk` can hold a text variant together with the RAG models.

## Matrix Context Solving

When `profiles.yaml` defines a `matrix` section with `embed` and `rerank` models, the system solves a shared VRAM budget equation across all model types:

```
reserve = RESERVE_SYSTEM(1024) + max(RESERVE_VIDEO(1024), baseline_mb)
available = vram_total - reserve - spare
chat_ctx solves Σ(chat_weight + chat_factor × chat_ctx) = available - embed - rerank - coloads
```

The solver (`llama_packer/vram.py:solve_matrix_ctx`) finds the maximum chat context that coexists with fixed embed/rerank allocations (at their declared contexts). All chat models share the same VRAM pool (llama-swap evicts between them), so the solver picks the largest feasible context across all chat models. Smaller chat models are never raised above their own design context — they are only clamped down to it. Unestimable chat participants ("riders", zero-cost placeholders) ride the measurable models' solve: they never set the shared bar (their native max would inflate `chat_ctx` for everyone and suppress tools demotion) and are flagged `estimated: false`.

Embed/rerank models are auto-selected as the smallest model of each type, or matched by `--embed`/`--rerank` CLI selectors. **They always serve single-slot**: a declared `parallel:` on an embeddings/rerank model is ignored with a note — resident parallelism must never buy context away from the main chat it serves.

### Knobs (matrix section keys)

| key | default | meaning |
| --- | --- | --- |
| `min_chat_ctx` | 65536 | Co-load decision floor: a co-load is only included while chat stays at or above this |
| `tools_min_ctx` | 131072 | Tools advertisement threshold; also the floor when a tools chat model can still keep it |
| `coload_min_ctx` | 20480 | emb/rerank squeeze floor |
| `ctx_gain_min` | 4096 | Minimum chat-context gain for a squeeze to be adopted |
| `estimate_headroom` | 1.25 | Padding applied to *estimated* (not measured/pinned) co-load overheads |
| `auto_parallel` | true | Spend leftover VRAM on concurrent chat slots: unpinned chat models on llama-server/vLLM get a value-function (ctx, slots) solve — score `(ctx/floor)^0.5 × slots^parallel_power`, best feasible pair wins. Set `false` to disable fleet-wide; a sidecar `parallel:` pin opts a single model out |
| `auto_parallel_max` | 8 | Hard cap on auto-parallel slots |
| `parallel_power` | 0.75 | Slot exponent B in the auto-parallel score: what each extra simultaneous chat is worth |

Invalid values warn and fall back to the default; the solve never fails on a bad knob.

### emb/rerank squeeze

When the baseline solve puts chat below `tools_min_ctx`, the solver re-solves with embed/rerank contexts clamped to `coload_min_ctx`. The squeeze is adopted only when it buys chat at least `ctx_gain_min`; otherwise design contexts are kept. An adopted squeeze is *realized*: the reduced context is emitted into the embed/rerank commands (that emit is what frees the VRAM), and `metadata.ctx_size` on those entries reflects it.

### Opportunistic co-loads

After the squeeze pass, enabled `s2t` and `image` models (not `t2s` — containerized, separate pool; not `embeddings`/`rerank` — unconditional residents) are included smallest-fixed-overhead-first while the chat solve stays at or above the floor:

- floor = `tools_min_ctx` when a chat model declares `tools` and the baseline still keeps it; otherwise `min_chat_ctx`.
- A candidate that would drop chat below the floor is skipped with a warning naming model and MB; it does not block smaller candidates later in the list.
- Fixed overhead = weights + fixed compute (zero KV terms for these backends). An operator-pinned `vram_mb` sidecar field is authoritative (`source: config`); otherwise the file-size + per-backend-buffer estimate applies, padded by `estimate_headroom` when no measurement exists. CPU-resident candidates cost 0.
- Shared process overhead is counted once per process, not per model — a multi-model entry (e.g. a speech server hosting ASR + VAD + diarization) is budgeted as Σ(weights + per-model activations) + one shared constant; pin the entry with `vram_mb` to encode the sum directly.

Included co-loads appear in the matrix routing: `_build_matrix_vars` adds one role-prefixed var per included model (`s2t`, `img`; numbered on collision), and set expressions may reference the `__COLOAD_VARS__` placeholder (expanded like `__CHAT_VARS__` to a parenthesized OR-list of var names; dropped from the expression when no co-loads were included). Co-loads whose entry ids are not referenced by any set stay outside the co-loading groups (independent eviction).

### tools demotion

A chat model declaring `capabilities: [tools]` that the matrix solve serves below `tools_min_ctx` gets the capability demoted in its emitted entries: `capabilities.tools: false` and `metadata.tools_demoted: true` (per entry; the sidecar declaration is untouched, so re-packing at a higher served context restores it). This keeps `/v1/models` consumers (hermes, opencode, UIs) from sending tool calls to a window too small to serve them. Note `capabilities.context` reports the max *trained* context (`design_context`) regardless; the VRAM-served limit lives in `metadata.ctx_size`.

## Main-Chat Context Determination

The full context ladder, smallest to largest:

1. `_MIN_CTX_SIZE` (4096) — hard emit floor for any served entry.
2. `matrix.coload_min_ctx` (20K) — emb/rerank squeeze floor (above).
3. `matrix.min_chat_ctx` (64K) — co-load decision floor: co-loads may not push chat below it. emb/rerank, being unconditional, may — at which point tools demotion is the signal.
4. `matrix.tools_min_ctx` (128K) — tools advertisement threshold (above).
5. `design_context` / `--max-context` — never raised, only clamped down.

Without a matrix section, each model's context is solved independently by `calc_ctx` against the shared budget minus fixed co-load overheads (none included) — unchanged historical behavior.

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

vLLM models use a different key for the same job: `reasoning_parser` →
`--reasoning-parser` (e.g. `nemotron_v3`). The two keys are unrelated —
different server, different flag — and llama.cpp's `--reasoning-format` is
stripped from vLLM commands (see [vLLM Backend](#vllm-backend)).

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

- **`modes`**: a map of mode name → param set. Each declared mode layers over the
  same-named resolved profile (or the fleet `defaults:` when no profile shares the
  name) with the [single merge rule](#layer-merge-rule) — unspecified keys inherit
  from below instead of being dropped, so `modes: {coding: {temperature: 0.2}}`
  keeps the profile's `top_p`/`min_p`. The layered modes replace the global profile
  sampling overrides for this model. Models may declare any number of modes
  (commonly 1–3).
- **`default_mode`**: which mode is the model's default. It is emitted under the bare
  `${MODEL_ID}` `setParamsByID` key; every other mode under `${MODEL_ID}:<mode>`. Falls back to
  the first declared mode.
- **Keys**: llama.cpp parameter names — `temperature`, `top_p`, `top_k`, `min_p`, `pres_pen`,
  `repeat_penalty`, `freq_pen` (see `SAMPLING_KEYS`). Unknown or non-numeric values are
  ignored with a warning. Emission translates to the request-body JSON names llama-server
  parses (`pres_pen` → `presence_penalty`, `freq_pen` → `frequency_penalty`).
- **Expressions**: a value of the form `"base * N"` (string starting with
  `base *`) on a sampling key is evaluated against the merged value from the
  layer below — e.g. `temperature: "base * 0.7"` over a below-value of `1.0`
  yields `0.7`, so a high-temperature model stays high (only less so) while a
  low one never increases. Evaluation is sandboxed to the single name `base`;
  with no numeric value below it warns and the key is skipped (never emitted
  raw).
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

### Model recipe keys (opt-in, verbatim)

Every key is a serving choice: settable per-sidecar or by override rule, emitted only when
declared (absent = vLLM auto-detects from the checkpoint — do not set a default). Precedence
is the usual sidecar > override rule > fleet default.

| Key | Flag(s) | Notes |
|-----|---------|-------|
| `vllm_quantization` | `--quantization <v>` | weight-quant *method* (e.g. `modelopt_mixed`). **Not** the metadata `quantization` field (bits-per-weight) — the separate key exists exactly to avoid that collision |
| `moe_backend` | `--moe-backend <v>` | e.g. `marlin`, `cutlass`, `triton` |
| `mamba` | five flags, one per sub-key | hybrid/Mamba models (see below) |
| `tool_call_parser` | `--enable-auto-tool-choice --tool-call-parser <v>` | chat role only; orthogonal to the `tools` capability / matrix demotion (declare `capabilities: [tools]` in the sidecar to advertise it) |
| `reasoning_parser` | `--reasoning-parser <v>` | chat role only; e.g. `nemotron_v3`, `deepseek_r1` |

The `mamba:` mapping — each present sub-key emits exactly one flag:

```yaml
mamba:
  backend: flashinfer            # -> --mamba-backend flashinfer
  ssm_cache_dtype: float16       # -> --mamba-ssm-cache-dtype float16
  stochastic_rounding: true      # -> --enable-mamba-cache-stochastic-rounding
  philox_rounds: 5               # -> --mamba-cache-philox-rounds 5
  cache_mode: align              # -> --mamba-cache-mode align
```

These are experimental, vLLM-version-specific flags — validate names/values against the
served image's `vllm serve --help` (the sub-key→flag map lives in one place,
`backends/vllm.py:_MAMBA_FLAG_MAP`).

`reasoning_parser` is **not** llama.cpp's `reasoning-format` (`Model.reasoning_format`,
emitted as `--reasoning-format`): different server, different flag, different values. The
llama.cpp `--reasoning-format`/`--reasoning-budget` are stripped from vLLM commands; the
`--reasoning-parser` string is unaffected.

### Memory estimation

vLLM has no `llama-fit-params` analog, so VRAM params are sourced differently but flow
through the same `FitParams` pipeline (`model_mib`, `kv_per_token_mib`,
`slot_mib=0` — vLLM's paged KV pool is shared, `--max-num-seqs` does not
change KV size — and `compute_mib`), and are
persisted to the sidecar `derived:` block with `source:` `vllm-estimate` /
`safetensors-estimate`. Sources, in order (`vram.py:_fit_params_vllm`):

1. `vllm-memory-estimator` (optional dependency) on the `hf_repo` — reuses vLLM's own
   `ModelConfig`/`KVCacheSpec` logic; maps weights→`model_mib`, activations+workspace+
   overhead→`compute_mib`, per-token KV→`kv_per_token_mib`.
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

`profiles.yaml` `vllm:` also configures `docker_args`, `container_port` and `hf_cache`
(`vllm-docker`).

### Container (vllm-docker)

llama-swap has no native container abstraction: a dockerized backend is a normal entry
whose `cmd:` is `docker run --name ${MODEL_ID} … <image> <vllm serve flags>` — server
flags are argv after the image, container env is `-e` inside `cmd` (an entry's `env:` list
reaches only the docker client process). Two upstream-documented lifecycle fields are
emitted with every vllm-docker entry:

- `cmdStop: docker stop ${MODEL_ID}` — an unload (swap, manual, or TTL) stops the
  *container*; without it llama-swap can only kill the `docker run` client, leaving the
  container running with its VRAM held.
- `unloadTimeout: 30` — must exceed the stop grace ("docker stop is slow").
- `proxy: http://127.0.0.1:${PORT}` — upstream's container guidance ("the single most
  common configuration error"). `checkEndpoint` keeps llama-swap's default `/health`
  (correct for vLLM).

References: llama-swap kb `guides/model-runtime/ttl-and-unloading.md` and
`guides/model-runtime/writing-cmd.md`.

**Mounts and path mapping.** Every path-shaped value inside `cmd` is rewritten to a valid
in-container path. Model refs that are repo ids stay verbatim (resolved offline through
the mounted hub); local files resolve in this order (after `Path.resolve()`, so symlinked
layouts map by real location):

1. Under `vllm.hf_cache` (host HF_HOME root — the dir containing `hub/`; falls
   back to top-level `hf_home:`) →
   `/root/.cache/huggingface/<rel>` — the whole root is bind-mounted, so HF snapshot blob
   symlinks resolve. Applies to the model ref, `speculative_config` path values (`model`,
   `draft_model`), and the chat template.
2. Under any `models_dir` (e.g. `~/models`) → `/models`, `/models2`, … (already bound).
3. Else → read-only parent bind (`-v <parent>:/extN`) and an `/extN/<name>` ref.

Every vllm-docker entry also gets `-e HF_HOME=/root/.cache/huggingface` and
`-e HF_HUB_OFFLINE=1`: **llama-packer never downloads**. A repo-id model that is not
pre-staged in the mounted hub fails fast at startup — that is the cache-miss case, not a
bug; llama-packer warns at pack time when `hf_cache` is unset for a repo-id model.

`docker_args` (default `--runtime=nvidia --gpus all --shm-size=16g`) is the operator's
flexibility point for container-runtime specifics: GPU device selection
(`--gpus device=N` / `-e CUDA_VISIBLE_DEVICES=N`), `--ipc=host` vs `--shm-size`, and
extra read-only binds (e.g. vLLM/flashinfer/triton JIT caches — without them every start
pays compile cost, since container caches are ephemeral).

The `healthCheckTimeout` budget assumes vLLM loads at 100 MB/s
(`healthCheckTimeout >= largest_vllm_model_mb / 100`; 300 s floor for repo-only models) —
a ~60 GB NVFP4 model gets ≥ 600 s.

### Limitations

- Accurate sizing requires `vllm-memory-estimator` (or a local `.safetensors` file); otherwise
  context falls back to the declared `context_length`.
- LoRA adapters are emitted for llama-server only; vLLM `loras:` is warned and ignored.
- Baked-in MTP weights are not added to the VRAM budget; GGUF draft companions cannot be
  loaded by vLLM (see [Speculative decoding under vLLM](#speculative-decoding-under-vllm)).
- Runs one vLLM server per model per image/binary; multi-image or cluster/tensor-parallel
  provisioning is future work.
- `HF_HUB_OFFLINE=1` forbids downloads in docker mode: models must be pre-staged in the
  `hf_cache` hub (or served by local path under `models_dirs`).
- vLLM recipe flags (`mamba:`, MoE/quant/parsers) are experimental and version-specific;
  validate against the served image before relying on them.

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

`sd-server` has no `llama-fit-params` analog, so VRAM is fixed overhead: `model_mib = file-size(diffusion)`, zero KV terms, `compute_mib = 512`. `ctx_size` tracks `design_context` (sidecar `context_length` or default) but does not affect VRAM; the entry is excluded from the shared chat budget solve itself, but (unlike before) is a candidate for the opportunistic co-load pass when a matrix section is configured — included only while chat keeps its floor (a 40 GB diffusion model simply won't fit after the RAG set).

### Binary precedence

`sd-server` is resolved: `--sd-server` CLI flag > `sd.bin` in `profiles.yaml` > `$SD_BIN_DIR` (file or directory containing `sd-server`) > `sd-server` on `PATH`. Absent binary disables format-based inference; an explicit `backend: sd-server` pin to a disabled setup is an error that skips the model. Docker variant is future work (tracked in `docs/plans/comfyui-sd.md`).

### Capabilities

`role: image` emits `capabilities: {in: [text, image], out: [image], context: design_context}` — image outputs, text+image inputs (so llama-swap shows both `Image Gen` and `Img→Img` badges) — or `out: [video]` when `capabilities` contains `video` or `architecture` is `wan`, `hunyuan-video`, `h3`, `mochi` etc (video diffusion). `image/audio/speech/video` capabilities are ignored for `image` roles except `video` for output selection (chat-only otherwise). `proxy` / `checkEndpoint` are always emitted for `sd-server` entries.

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
badge); declared `image/audio/speech` capabilities are chat-only and ignored
for s2t roles. VRAM is fixed overhead like sd-server: `model_mib = file-size`,
zero KV terms, `compute_mib = 100` (measured on Vulkan: process footprint
≈ Σ model files — activation memory is small; sd-server keeps its 512 MiB
diffusion buffer); a sidecar `vram_mb` pins the total outright.
`ctx_size` tracks `design_context` (sidecar `context_length` or default)
without affecting VRAM. s2t models are candidates for the *opportunistic
co-load pass* (see Matrix Context Solving): with a matrix section configured,
the smallest ones join the shared resident set while chat keeps its floor.
The emitted `--parallel` maps the sidecar/profile slot count to concurrent
transcription workers.

## Audio Backend (kokoro-podman)

A model with `role: t2s` (opt-in: a `t2s/` directory plus `dirs: {t2s: t2s}` in
`profiles.yaml`) is served with **kokoro-podman** — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
text-to-speech via [remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
in **rootless podman** (OpenAI-compatible `POST /v1/audio/speech`,
`GET /v1/audio/voices`, health on `/`, container port 8880).

```yaml
# profiles.yaml
dirs: {t2s: t2s}
backends: [llama-server, kokoro-podman]
```

```yaml
# sidecar: t2s/kokoro-v1.md — weights and ~50 voicepacks are baked into the
# image, so hf_repo alone identifies the model; no local file required.
---
name: kokoro-v1
hf_repo: hexgrad/Kokoro-82M
description: "Kokoro-82M text-to-speech"
---
```

Emitted entry (NVIDIA example):

```yaml
kokoro-v1:
  cmd: podman run --init --rm --name ${MODEL_ID} -p ${PORT}:8880 --device nvidia.com/gpu=all ghcr.io/remsky/kokoro-fastapi-gpu:latest
  proxy: http://127.0.0.1:${PORT}
  checkEndpoint: /
  capabilities: {in: [text], out: [audio]}
```

### Vendor selection

The GPU vendor picks both the default image tag and device pass-through flags:

| vendor | default image | podman flags |
|--------|--------------|--------------|
| nvidia | `ghcr.io/remsky/kokoro-fastapi-gpu:latest` | `--device nvidia.com/gpu=all` |
| amd | `ghcr.io/remsky/kokoro-fastapi-rocm:latest` | `--device /dev/kfd --device /dev/dri --group-add video --group-add render` |
| cpu | `ghcr.io/remsky/kokoro-fastapi-cpu:latest` | *(none)* |

Detection probes `amd-smi`/`rocminfo` then `nvidia-smi`. Precedence for the
image: CLI `--kokoro-image` > profiles.yaml `t2s.image:` > vendor default.
`t2s.vendor:` (`auto|nvidia|amd|cpu`) overrides detection for tag *and* flags;
`t2s.podman_args:` replaces the auto flags entirely; `t2s.container_port`
overrides 8880; `t2s.voices_dir:` bind-mounts a persistent voicepack directory
(read-write — the server loads `.pt` packs per request and saves combined
voices back). Pin to an upstream release tag rather than `:latest` for
stability (`gpu:-cu128` for RTX 50-series / Blackwell).

### Voices

No per-model configuration: voices live server-side in the image (~50 packs),
selected per request via the JSON body (`"voice": "af_heart"`, weighted mixes
like `"af_bella(2)+af_sky(1)"`) and listed at `/v1/audio/voices`.

### Capabilities and VRAM

`role: t2s` emits `capabilities: {in: [text], out: [audio]}` (Speech badge).
VRAM is fixed overhead: weights are baked into the image so `model_mib` is 0
unless a local file resolves, plus a conservative 3072 MiB runtime buffer (the
PyTorch/CUDA floor is ~2.4 GiB, peaks near 4 GiB under load — upstream
`/dev/unload` benchmarks). The entry is excluded from the shared chat matrix
solve. A local `.onnx` copy may resolve by same-stem sidecar convention inside
the `t2s/` dir.



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
  `capabilities: 'reasoning'` matches `[image, tools, reasoning]`.
- A field **absent from a model never matches** (there is nothing to search) —
  to target "has no X", match a different field or invert in rule order.
- An **invalid regex** logs a warning and that rule matches nothing (the run
  continues; only structural problems abort).
- Unknown settings keys in a rule log a warning and are ignored; if *no* key
  is known, the rule aborts the run.

Use `when: true` to match every model. Regex literals are easiest in YAML
**single-quoted** or unquoted scalars — only double quotes interpret
backslashes (`\.` stays literal in single quotes).

**Merge semantics.** One merge rule governs every layer
(`utils.merge_layer`) — `defaults:` → profiles → sidecar frontmatter →
override rules → companion block (see [Layer merge rule](#layer-merge-rule)).
Settings seed from the model's own sidecar fields, then each matching rule is
layered on top (CSS-like: rules read top→bottom as increasing specificity).
So a later rule that changes `backend` does not clobber a `chat_template` set
by an earlier rule — and a rule setting `loras:` appends to (rather than
replacing) the sidecar's list.

### Layer merge rule

Layers bottom-to-top: directory/global `defaults:` → profiles → sidecar
frontmatter → override rules (global, then directory-scoped outermost →
innermost) → companion block. At each boundary the upper layer wins:

- **Scalars** are replaced by the upper value.
- **Dicts** merge recursively, per sub-key, upper winning
  (`modes: {coding: {temperature: 0.2}}` keeps the below layer's `top_p`).
- **Lists** union-append: order-preserving, de-duplicated
  (`capabilities: [tools]` below + `[image]` above → `[tools, image]`).
  String items starting with `-` remove instead (`[-image]` drops `image`);
  removing an absent item is a no-op.
- **`None` deletes the key outright** (`top_p: None` removes an inherited
  `top_p`).
- **`"base * N"`** strings on sampling keys evaluate against the merged
  numeric value from the layer below (see [Sampling Modes](#sampling-modes)).

A rule therefore never needs to restate a full list or dict for every model
it applies to — it declares only the delta. To *remove* inherited list items
use `-item`; to drop a key entirely use `None`.

**Intrinsics.** Some keys are properties of the model itself, not serving
choices: `capabilities`, `parameters`, `quantization`, `architecture`,
`base_model`, `family`, `finetune`, `type` (`scope.INTRINSIC_KEYS`). They
belong in sidecars — a scope's `defaults:` setting one warns (legal but
questionable: it papers over an incomplete sidecar), and a sidecar *removing*
an inherited capability (`capabilities: [-tools]`) warns too. Sidecars add;
conditional serving belongs in a companion block, which appends exactly while
served. No replace semantics exist anywhere: nothing in the codebase or its
history needed lists to replace, so the rule has no replace case.

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
  innermost, with the [single merge rule](#layer-merge-rule) — authored
  sidecar values win per key, append to default lists, and delete with `None`.
  Empty stub sidecars carry no data, so defaults fill them naturally. The
  per-model identity keys `name`, `model`, `ignore` may not be defaulted
  (validation error).
- **`overrides:`**: standard rules (same validation and matching); applied
  after global rules, outer scopes first — innermost wins per key, same
  merge rule (a rule declares only its delta).
- **Paths**: `chat_template:` / `loras:` resolve relative to each *sidecar's*
  directory (not the models.yaml), so reference shared files with `../`.
- **Entry-id collisions** are fatal: if two models slug to the same llama-swap
  entry id, the run logs an error and exits — rename one of them.

**Settings keys** (all optional): `backend`, `hf_repo`, `chat_template`,
`chat_template_kwargs`, `loras`, `cli_args`, `reasoning-format`,
`reasoning-preserve`, plus the serving/companion choices `cache_type`,
`parallel`, `mmproj`, `speculative`. Rules setting `mmproj`/`speculative`
re-trigger companion resolution, so a rule can add or remove vision /
speculative decoding per pattern — an `mmproj` mapping block in a rule works
exactly like one in a sidecar (same merge rule, same validation).

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
| `kokoro-podman` | `.onnx`, `hf_repo` | t2s |

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
`sd-server`, `whisper-server`, `kokoro-podman`.

## Global backend args

Each llama.cpp-family backend section in profiles.yaml accepts a free-form
`args:` string of flags appended to **every** command that backend renders —
the place for fleet-wide performance tuning:

```yaml
# profiles.yaml
llama_server:
  args: "--flash-attn on -b 512 -ub 512"
vllm:
  args: "--max-num-batched-tokens 512"   # vLLM spelling of -ub (FA is always on)
whisper:
  args: "--flash-attn on"                # whisper.cpp shares the option
sd:
  args: "--diffusion-fa"                 # stable-diffusion.cpp variant
```

The same *intent* maps to different flags per engine, so each backend owns its
own `args` (kokoro is a containerized service — `t2s.podman_args` plays that
role). Values must be a string of flags (a non-string value aborts the run)
and are validated with shlex at build time (bad quoting aborts the run);
they don't feed the VRAM estimator — they're operator responsibility, like
sidecar `cli_args`.

Precedence per flag is most-specific-wins, implemented via the ordered
flag→value map in `render_command` (a flag can only appear once; later
sources overwrite earlier values). The same order applies to every backend:

1. backend built-in flags (`-c`, `--parallel`, …) — including per-model
   feature flags (mmproj, loras, MTP, reasoning)
2. global `<section>.args` — so e.g. `-b 512` tunes chat models only
3. per-role flags (embed/rerank mode flags) **+ the resolved named batch
   keys** — `batch:`/`ubatch:` (sidecar > profile > `llama_server:` fleet
   section > role defaults: chat 2048/512, embed/rerank 4096/512) render
   explicitly on every llama-server command and win over conflicting
   `args` values; `cli_args` `-b`/`-ub` is warned about (shadowed, and
   invisible to the VRAM measurement)
4. per-model sidecar `cli_args:`

Note that the map keys on exact flag spelling: `-fa` and `--flash-attn` are
distinct keys — use the spelling you want in the final command.

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
kv_per_token_mib [MiB/token] = kv_bytes_per_token / 2²⁰
```

- The per-layer K/V output dims come from a `llama-fit-params` measurement of
  the main model; the safetensors fallback (`utils.estimate_safetensors`) reads
  them from `k_proj`/`v_proj` tensor shapes in the header. The safetensors
  path yields only `c` (per-token KV); the per-slot `D` is unknown there and
  conservatively zero.
- **Cache-type**: measured blocks are stored per `cache_type` and never
  derived across types. `c` *would* scale by the byte ratio
  (`_KV_CACHE_BYTES`), but `D`'s precision behavior is arch-dependent (SWA
  ring buffers scale with KV precision; fixed slot overhead does not), and
  re-measuring the (p=1, p=2) pair costs ~1 s.
- Example: a model measured under `f16` serves `q8_0` with roughly half the
  KV footprint (`1.0625/2`), doubling the affordable context for the same
  VRAM — verified empirically (×0.529–0.531 across families).

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
| `t2s` | `t2s` | Text-to-speech (kokoro via podman; opt-in via `dirs: {t2s: t2s}`) |
| `img` | `image` | Diffusion / image generation (sd-server; opt-in via `dirs: {img: image}`) |

  Files at the root itself default to `chat`; files under any other
  subdirectory (`img/` when not opted in, `misc/`, `tmp/`,
  `hf_hub/`, … — and `s2t/`, `t2s/`, `img/` when not opted in) are **skipped** — one summary line per run names the
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

Companions resolve the same way after the local search misses: the block's
`file:` (or the `speculative:` filename) is looked up in the sidecar's repo
snapshot, with a single-glob fallback (`*mmproj*.gguf`) covering HF's naming
variants (`mmproj-F16.gguf`, `mmproj-model-f16.gguf`, …). When no `mmproj:`
is declared at all, an HF snapshot that contains the model's own gguf is
fuzzy-scanned for a family-matching `*mmproj*.gguf` (preferring
`mmproj-(bf|fp|f)16`); local models require an explicit block (`false`
silences). A value may also address another cached repo explicitly:
`hub:<org>/<repo>:<file-or-glob>`. An ambiguous glob logs a warning and does
not resolve.

**Companion blocks.** `mmproj:` accepts `false` (disabled) or a mapping —
bare filenames are an error, since a filename alone records no purpose:

```yaml
mmproj:
  file: gemma-4-mmproj-F16.gguf
  capabilities: [image]          # what serving this file adds
  image_max_tokens: 4096         # only meaningful while served
```

`file` locates the companion; every other key is a conditional overlay
merged over the frontmatter **only while the companion is served** (the same
[layer merge rule](#layer-merge-rule) as every layer — lists append, `None`
deletes). The companion-off variant serves the base frontmatter, so purpose
is emergent: dropping the file strips exactly what the block contributed
(`image` above), never hardcoded vision assumptions. A second serving form
declares a draft purpose instead of vision:

```yaml
mmproj:
  file: qwen3-mtp.gguf
  mtp: true
  mtp_spec_type: draft-mtp
```

Block keys must be builder-consumed serving keys; identity, placement, and
backend selection (`name`, `model`, `ignore`, `mmproj`, `hf_repo`, `backend`,
`role`) stay model-level and are rejected in the block. The standalone
`speculative:` key (a separate draft file) is unchanged; a model may not
declare MTP serving from both at once.

The writer enforces the advertised-purpose invariant: a block declaring no
`capabilities` warns (the companion's serving difference is unadvertised),
and `image`/`video` claimed at top level but absent from the block warns too
— the companion-off variant would advertise it without the file, so move it
into the block.

The HF cache **root** (the dir *containing* `hub/`) is `--hf-home` >
profiles.yaml `hf_home:` > `$HF_HOME` > `~/.cache/huggingface`. `--hf-home`,
`hf_home:` and `$HF_HOME` always name this root — never `<root>/hub` itself.
`$HUGGINGFACE_HUB_CACHE`, when set, names the hub directory directly. Point
`hf_home:` at that root; then `hf download org/repo` followed by a small `.md`
sidecar is sufficient — no symlink step and no widening of `${MODELS_DIR}` (HF
cache paths get their own `${HF_HOME}` macro).

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
Paths under the Hugging Face cache **root** (the dir containing `hub/`:
`--hf-home` > profiles.yaml `hf_home:` > `$HF_HOME` > `~/.cache/huggingface`) are pulled into
their own `${HF_HOME}` macro so a chat template (or LoRA) living in the HF cache
never widens `${MODELS_DIR}` up to a non-models directory.


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
| `HF_HOME` | Paths under the HF cache root — the dir containing `hub/` (`--hf-home` > `$HF_HOME` > `~/.cache/huggingface`; `$HUGGINGFACE_HUB_CACHE` is the hub, not the root) |
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
  `ctx_size`, `mtp_enabled`, `mtp_draft_max`, `mtp_accuracy`, `throughput_factor`,
  `image_min_tokens`, `image_max_tokens` (vision variants only).
- **Native `capabilities`** — `in`/`out` modalities, `tools`, `reranker`, `context` (derived, see below).
- **Native `name` / `description`** — display fields in `/v1/models`.

### Builder-consumed keys (NOT passed through)

`name`, `context_length`, `description`, `cli_args`, `model`, `backend`, `hf_repo`,
`chat_template`, `chat_template_kwargs`, `loras`, `attention`, `kv_cache`, `tool_args`,
`speculative`, `speculative_config`, `mmproj`, `mtp`, `mtp_spec_type`, `mtp_draft_n_max`,
`mtp_draft_p_min`, `role`, `targets`, `allow_profiles`, `spare`, `capabilities`,
`ignore`, `device`, `concurrency`, `fit-params`, `vllm_image`, `modes`, `default_mode`,
`reasoning-format`, `reasoning-preserve`, `cache_type`, `parallel`,
`image_min_tokens`, `image_max_tokens`.

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
| `parallel` | int | Parallel slots for `--parallel` and VRAM sizing (sidecar > profile > 1). Declaring it opts out of auto-parallel for this model |
| `min_context` | int | Per-model minimum useful context (tokens): explicit floor of the auto-parallel search and its score normalizer. Default cascade: pinned ctx → tool callers 131072 → half the max context (`--min-context` overrides the last two when explicitly set, never this key or a pin) |
| `mmproj` | `false` \| mapping | Companion block: `file:` locates the projector/draft file (required); all other keys form a conditional overlay served only with the file (same merge rule as every layer). Bare filenames are an error. See [Companion blocks](#model-discovery-and-stub-sidecars) |
| `image_min_tokens` / `image_max_tokens` | int | Vision (mmproj) only: floor/cap on image tokens per image → `--image-min-tokens`/`--image-max-tokens`. Dynamic-resolution archs only (Qwen-VL family; Gemma/SigLIP is fixed ~256 and warned+skipped). The cap also floors the solved context (`parallel × max` must fit); see [Image token budget](#image-token-budget-vision-sidecars) |
| `ignore` | bool | Skip this model entirely |

### Agent-selection fields (optional, recommended)

| Field | Type | Meaning |
|-------|------|---------|
| `capabilities` | list | `[image, video, tools, reasoning, audio, speech]`; explicit (mmproj does NOT imply image/video). Mapped to the native llama-swap `capabilities` block (directional, see below). `vision` was removed — use `image` |
| `freethought` | float 0–1 | `1.0` reasons about anything; `0.0` readily refuses "distasteful" topics. Carried in `metadata` |
| `strengths` / `weaknesses` | list | Concise task phrases agents match on |
| `license` / `base_model` / `architecture` / `finetune` / `type` | str | Identity; `architecture` is one step more generic than `base_model` and informs backends (e.g. `qwen3`, `qwen3-vl`, `flux`, `sdxl`, `wan`, `hunyuan-video`, `h3`, `mochi`, `omni`); carried in `metadata` |
| `mtp_accuracy` | float | MTP draft acceptance rate; feeds `throughput_factor` |
| `parameters` | str | `"12B"` or MoE `"26B-A4B"` (total-active) for accurate throughput |
| `hf_url` | str | HuggingFace model URL |
| `hf_repo` | str | HF repo id for vLLM backends. Optional; parsed from `hf_url` when absent |
| `vllm_image` | str | Per-model vLLM docker image. Overrides profiles.yaml `vllm.image` and `--vllm-image` for this entry |
| `vllm_quantization` | str | vLLM `--quantization` method (e.g. `modelopt_mixed`). Not the metadata `quantization` field; absent = vLLM auto-detects. See [vLLM Backend](#vllm-backend) |
| `moe_backend` | str | vLLM `--moe-backend` (e.g. `marlin`) |
| `mamba` | dict | Hybrid/Mamba recipe → `--mamba-backend`, `--mamba-ssm-cache-dtype`, `--enable-mamba-cache-stochastic-rounding`, `--mamba-cache-philox-rounds`, `--mamba-cache-mode` (one flag per sub-key; absent sub-keys emit nothing) |
| `tool_call_parser` | str | vLLM `--enable-auto-tool-choice --tool-call-parser <v>` (chat role); pair with `capabilities: [tools]` |
| `reasoning_parser` | str | vLLM `--reasoning-parser <v>` (chat role). Distinct from llama.cpp's `reasoning-format` |

### Derived fields (computed, not authored)

- **`capabilities`** (native block): directional modalities, matching llama-swap's badge derivation — `in` = `["text"]` + `"image"` if `image` + `"video"` if `video` + `"audio"` if `audio`; `out` = `["text"]` + `"audio"` if `speech` + `"video"` if `video` + omni/video-arch (architecture `omni`/`video`/`wan`/`h3` etc). Output stays text unless `speech`/`video` output is declared, so a vision model never advertises text→image (`Image Gen`) unless `role:image`. `tools`/`reranker` boolean flags; `context` = design context (the model's maximum trained context: GGUF architectural max > sidecar `context_length` > default). `architecture` guides image vs video output for `role:image` (`out:[video]` when `video` in capabilities or `architecture` contains `video`/`wan`/`hunyuan-video`/`h3`).
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
capabilities: [image, tools, reasoning]
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

## Measured-Block Persistence

Machine-written values live in the sidecar `.md` file under a single
`derived` nested block — the only dynamically generated branch
(llama-packer's own calculation cache: never hand-edit, delete to force
re-evaluation; legacy `fit-params`/`measured` blocks are located but
rewritten as `derived` on next persist):

```yaml
derived:
  model_mib: 4800
  kv_per_token_mib: 0.0312
  slot_mib: 12.5
  compute_mib: 512
  source: fit-estimate
  cache_type: q8_0
  shape: "--flash-attn on -ub 512"
  ts: 2026-09-08T18:11:17-0400
  file:
    size_mb: 4848
    size_bytes: 5084001234
    mtime_ns: 1788804721239187032
    arch: qwen3
    context_length: 131072
    kind: text
```

**Serve corrections.** Per-arch family rows measured by the opt-in
`--probe-memory` (fit-params estimate vs real llama-server truth) live in
the durable machine-local `serve-corrections.yaml` beside `profiles.yaml`
(gitignored; `LLAMA_PACKER_CORRECTIONS` relocates it; the retired
`~/.cache` JSON stays readable as a fallback). Each row records the
witness `shape` + `ts`; rows measured under a different shape still apply
(the deltas are mostly allocator-level bias), with a one-time per-arch
note. Uncalibrated arches estimate uncorrected — one note per arch, never
an error, and never a probe requirement.

This block is:
- Read automatically on subsequent runs (avoids re-measurement and
  re-reading weight headers).
- The VRAM constants are invalidated when `cache_type` changes (re-measured;
  blocks are per cache type and never derived across precisions), and any
  pre-affine block (missing `slot_mib`) is re-measured + rewritten on the
  next run; the `file:` intrinsics are invalidated when the weight file's
  size or mtime changes (then re-read once and re-persisted).
- Updated when new values are computed.
- Preserved alongside all other frontmatter keys.

## Constants

| Constant | Default | Meaning |
|----------|---------|---------|
| `_DEFAULT_CONTEXT_LENGTH` | 32768 | Fallback when no `.md` sidecar or GGUF context exists |
| `_CTX_ROUND_TO` | 8192 | Round context size down to nearest boundary |
| `_MIN_CTX_SIZE` | 4096 | Hard floor for context size |
| `_MIN_AGENTIC_CTX` | 131072 | Min useful chat context; mmproj dropped below this (`--min-context`) |
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
