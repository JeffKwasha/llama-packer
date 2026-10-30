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
- **`globalTTL`** — only when `--idle-unload SECONDS` is passed; llama-swap's top-level idle-unload default (see [VRAM gating and reservation notes](#vram-gating-and-reservation-notes)).

### Writer module

`llama_packer/writer.py` composes the generation pipeline in three steps:

1. **`_filter_supported()`** — the validation boundary (backend format/role,
   reasoning flags, cache-type knowability); runs before any VRAM work.
2. **`Planner.plan()`** — context decisions per (model, category): vision rung,
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
| `matrix` | Resident RAG classes at a fixed small context, plus the knobs that allocate the remainder to chat (`embed_context`, `rerank_context`, `auto_parallel*`, `parallel_power`) | [VRAM Allocation](#vram-allocation) |
| `hardware` | `vram`, `baseline_mb`, `unified_system_mb`, `gpu_family` overrides | [Hardware Detection](#hardware-detection) |
| `vllm` | Backend resources: `image`, `bin`, `container_args` (legacy `docker_args`/`podman_args`), `container_port`, `container_vendor`, optional `gpu_mem_util` / `hf_cache` | [vLLM Backend](#vllm-backend) |
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

A passing probe also files a per-arch serve correction (serve truth minus the
fit-params estimate) in the machine-local `serve-corrections.yaml`, keyed
`arch|cache_type|mtp`. **The `mtp` dimension is the baked-in-draft shape only**
(`Model.mtp_baked_in`): a model with a `speculative:` companion is costed from
file size by `effective_static` instead, so a companion witness's delta — which
carries the companion's own weights and KV — would double-count there and
over-reserve every baked-in model of the same arch. Such a probe reports the
measurement but files no row, and the arch stays uncorrected for that shape.

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

**Design context:** When the affine pair is measured, it uses the model's GGUF architectural context length as the reference point (or sidecar `context_length`, or 32768 default). Sidecar `context_length` is the model's **design maximum** — a ceiling applied when serving (line 255), never a floor. The served size is solved from the VRAM budget, the usage category and `parallel_power`; `min_context:` is the key that states a floor. Reading `context_length` as a pin (its own floor) made a 262144 design max an unreachable floor, forced the model to parallel 1, and left the only remedy editing the sidecar — rewriting a claim about the model to express an operating decision. If the design context fits within the remaining budget, it is used directly without scaling down.

**Note:** parallel slots cost the fixed `p*D` term plus their share of the pool (`c * p * X`) — both scale with the slot count, so the solve charges them exactly.

### Vision is a rung, not a second entry

For a chat model with an mmproj companion, whether the projector is attached is
decided per category, by `solve_rung` walking the category's `sacrifice:` order:

- Shedding vision **buys context** → it is shed, and the entry loses its `i`/`v`
  flags. `coding` is configured this way, so a `.coding` entry is text-only and
  large, which is the trade it asks for.
- Shedding vision **buys nothing** — the model is at its design cap, or some
  other term binds — → it is kept. A capability is not thrown away for no gain.
- Holding vision **does not fit at all** → that rung is skipped and the next one
  along the declared order is tried.

There is no companion entry and no vision-less twin. A model emits **one entry
per satisfiable category**, and vision is a property of that entry rather than a
second serving alongside it. Two entries for one model is two categories in
sidecars — a `coding` entry and a `coding-vision` one, the latter declaring
`require: [vision]` — not a hardcoded pairing.

On the live fleet, the same 31B at the same budget:

```
gemma4-31b-hauhaucs-balanced.ti        155648  vision kept   (fallback: capability first)
gemma4-31b-hauhaucs-balanced.coding.t  180224  vision shed   (coding: context first)
```

### Entry ids name the task, not the solve

An id is `<template_id>.<category>.<flags>` and is **stable across repacks**:

- **`<category>`** is the category the entry serves, read straight off the
  variant — a rung answers exactly one category, so there is no "heaviest
  satisfied" search to run. A category that cannot be met emits **nothing**
  rather than a weaker name, so an id never under-claims. A name with nothing
  left after slugging drops the segment rather than the entry. The **fallback**
  entry carries no category segment at all: the bare stem.
- **`<flags>`** is the *served* capability set as first letters, in fixed order `t`,`i`,`v`,`a`,`s` (tools, image input, video input, audio input, speech output). It is omitted entirely for a plain text-in/text-out serving. A demoted serving (context below `tools_min_ctx`) drops `t`; a no-mmproj serving drops `i`/`v`. So a vision-less variant is simply the same id without `i`.

A model with several profile groups can land two servings on the same category+flags; the first keeps the stable name and later duplicates append their context label (`…coding.ti-136k`), so ids stay unique without making the common case churn.

Why not put the context in the id? Because opencode/hermes resolve and cache model ids at startup and do not re-pull them per request — an id that tracks the solve would force a restart on every repack. The context is still exact and client-visible in `metadata.ctx_size`, in `metadata.usage` (the satisfied categories), and in the display `name` (e.g. `qwen38-27B-Dirk · coding 156k [ti]`); only the id is frozen. This is a deliberate priority order: **category > stability > fine-grained accuracy**, and a category dropping off under memory pressure is correct — that serving genuinely cannot meet the task.

Anything that *references* an entry must resolve through `entry_ids_by_stem` (stem → emitted ids) rather than `Model.template_id`. A matrix var pointing at a `template_id` makes llama-swap reject the whole config. Validate with `llama-swap -config <file> -validate`.

### VRAM gating and reservation notes

When an explicit reservation (`--spare`, `--baseline`, or their `profiles.yaml` counterparts) shrinks the budget, the planner degrades visibly instead of emitting entries that OOM at load time:

- **Over-budget means omitted.** `calc_ctx` records `over_budget_reason` when
  weights + compute do not fit at any context of a rung (or the reserve/spare
  leaves ≤ 0 available). `solve_rung` skips that rung and tries the next along
  the declared order; if none fits, that **category emits nothing** — and a model
  whose every rung is over budget, fallback included, produces no entry at all.
  Skipped stems never reach the matrix vars, so routing auto-prunes; if *every*
  chat model is gated away, matrix routing is omitted entirely (warning logged)
  rather than emitting a degenerate empty OR-list.
- **Vision is a rung, not a fallback.** There is no "vision → text" retry and
  no companion to suppress: the shed order already decides whether dropping the
  projector is worth it, and a rung that does not fit is simply not a candidate.
- **Squeeze description note.** When the solve flags `vram_squeezed` and serves strictly below the model's design context *and* the reservation is user-declared — `spare_mb > 0` (pool-adjusted, before any resident charge) or `baseline_mb > _RESERVE_VIDEO` (1024 MiB floor) — the entry description gains a trailing note: `(over configured VRAM limits — serving at X of Y design tokens; re-run without --spare/--baseline for full context)`. Equal-to-design solves stay silent (no "serving at N of N"), and a baseline at or below the fixed floor never fires (it is already covered by the reserve). With no reservation, no note is emitted.
- **Degenerate refusal.** If `vram_total − reserve − spare ≤ 0`, the run fatals *before* any VRAM work (exit 1) and does **not** overwrite an existing `config.yaml`. Same guard applies when the reservation leaves zero model entries.

`--idle-unload SECONDS` emits a top-level `globalTTL: SECONDS` (llama-swap: unload any model after that many seconds of inactivity; `0` = never, the llama-swap default; llama-packer omits the key when the flag is absent). Intended for co-residency with ComfyUI/games that need VRAM on short notice without re-packing.

### Usage categories (`usage:`)

The global `min_context` is one floor for the whole fleet, which cannot say "coding wants 128k, Q&A is fine with 32k". `profiles.yaml` gains a `usage:` section — the single **blessed vocabulary** of what models are used for. llama-packer is not a server: it pre-sizes one variant per satisfiable category and writes them into the config, which llama-swap routes at request time:

```yaml
usage:
  coding: {min_context: 128k, min_parallel: 1, sacrifice: [vision]}
  chat:   {min_context: 64k,  min_parallel: 2, sacrifice: [draft, vision]}
  qa:     {min_context: 32k,  min_parallel: 2, require: [vision]}
```

A sidecar tags itself with `usage: [coding, chat]`. **The category is the operator's declaration, not an inference** — nothing about the model or the GPU chooses it or overrides it, exactly as `hf_url` and `context_length` are declarations. A tag not declared in `profiles.yaml` is warned about and ignored, so the categories in force are exactly the ones the operator wrote. That warning **names its sidecar** — and so does every log line about one sidecar: a diagnostic leads with the sidecar's name, so an operator can triage it by name without cross-referencing. Each satisfiable category emits its **own** entry; two entries for one model is therefore just two categories in sidecars, not a separate mechanism. Every model additionally gets a **fallback** entry with bare minimums and no category name in the id, so an untagged model still reaches clients.

Two things follow from this.

**Shedding is described per category, and resolved by `solve_rung()`.** Each category declares `sacrifice:` (an ordered preference for what to give up), `keep:` (do not drop what this model has) and `require:` (must have it, or the category does not apply). `shed_levels()` turns those into the ladder of rungs to try — one per prefix of the declared order — and `solve_rung()` asks the budget what each one affords, keeping the level that buys the **most context**, and among levels buying the same context the **least shed**.

That tie rule is the whole of "coding drops vision to maximise context, and only adds vision when it is free": shedding usually buys context, so the projector goes; when VRAM holds design context either way, nothing is bought and the projector stays attached rather than being discarded for no gain.

**A category and the null category want different things from the same rung list**, so `solve_rung` takes `prefer_capability`. A category takes the rung buying the **most context** — coding would rather have 262144 tokens text-only than 131072 with a projector. The null category takes the **first** rung that qualifies, because it has no goals and handing up a capability to buy context nobody asked for is a pointless trade. On the live fleet that shows as `gemma4-31b-hauhaucs-balanced.ti` (fallback, vision kept, 155648) beside `gemma4-31b-hauhaucs-balanced.coding.t` (vision shed, 180224).

`require` and `keep` are different and both are honoured: `require` gates existence (a model lacking the capability does not satisfy the category, so no entry claims it), while `keep` only forbids shedding — a coding model with no MTP draft is still coding. A category that cannot meet `min_context` at any rung emits **nothing**. Capabilities named by neither list are shed last, in the order from `shed:` in profiles.yaml, which defaults to vision before draft.

`min_parallel` is honoured when a candidate qualifies and yields to the context floor when it does not — concurrency is a throughput preference, the context is what the display name advertises.

## VRAM Allocation

Two tiers, and only two.

### Residents

Pinned for as long as the service is up, because losing them means losing
the KV cache and reloading on every lookup. Embeddings and rerank are the
clear case; speech (STT/TTS) are optional residents.

Each resident serves a **fixed small context** — `embed_context` /
`rerank_context`, default 4096. Two reasons, and the second is what makes
the rest of this section work:

- clients standardise on one RAG context and chunk to it, so the embed size
  is a stable contract rather than a function of what else is loaded;
- **a fixed small context makes the resident tier's VRAM a number you can
  compute up front**, which is what makes the remainder meaningful. A
  context that flexed with load would make the budget move under you.

The cap is not a concession to revisit when VRAM is tight. If VRAM binds,
context in main chat is worth more than context in RAG.

A sidecar declaring `device: cpu` pins that resident (or any model) to
**host RAM instead**: it charges the host pool, never VRAM — one of the
two budgets, never both.

Residents are selected by role, or by `--embed` / `--rerank` CLI selectors.
**They always serve single-slot**: resident parallelism must never buy
context away from the chat it serves. A declared `parallel:` on an
embeddings/rerank model is ignored with a note.

### Host pool (`device: cpu`)

A `device: cpu` sidecar places the serving entirely in host RAM: it
competes for no VRAM — charged by neither the resident tier nor a chat
set, it cannot be evicted on VRAM grounds, and once warm it stays.
Its price is a host-RAM budget with the same decision quality
(`docs/plans/cpu-residency-budget.md`):

- **detection** finds the pool at pack time — `MemAvailable` minus a sysctl
  pressure zone (per-zone high-watermark sum; declared fallback 25% of
  total), capped by the llama-swap service cgroup headroom
  (`memory.high`/`memory.max` − `memory.current`). Every term is
  overridable from `profiles.yaml hardware:`; each chosen value is logged
  with the arm that picked it.
- **the charge** is the serving's own measured quad at its design context
  — the same `derived:` block family the VRAM budget uses, measured
  under a `--n-gpu-layers 0` shape stamp. A declared `parallel:` on a CPU
  chat model is honoured and multiplies the pool terms; residents stay
  single-slot.
- **the check is go/no-go**: a quad that will not fit the pool remaining
  after committed residents logs the shortfall and emits no id — no
  solve-down, no automatic offload. `device: cpu` is the only gate.

**Moving a whole usecase off VRAM** takes one declaration —
`profiles.yaml device_roles: [embeddings, rerank]` (or the mapping
`{embeddings: cpu, rerank: cpu}`) places every model of those roles on the
host, exactly as if each sidecar carried `device: cpu`. The key is a role
and the value is a device; there is no `role: cpu` field. A CPU helper entry
also joins every routing set and serves with `ttl: 0`: a request names
it, it loads once, and no later swap evicts it — evicting it would
reclaim no device memory. Set subsets mean nothing is preloaded.

`metadata.packing` names the placement (`device: cpu|gpu`) and the decided
byte facts (`ram_mib`/`vram_mib`, `kv_per_token`) so clients can filter
without parsing ids.

### Remainder

```
reserve        = RESERVE_SYSTEM(1024) + max(RESERVE_VIDEO(1024), baseline_mb)
available      = vram_total - reserve - spare
resident_vram  = Σ VRAM(resident, at its own fixed context)
chat_budget    = available - resident_vram
```

The matrix solve turns `chat_budget` into **one shared per-slot
`chat_ctx`** — the target every co-resident chat model is sized against.
Each chat model then resolves its own serving against that target, through
its category:

```
for each chat model:
    for each declared category, and the fallback:
        solve (shed, ctx, parallel) from chat_ctx
```

Two chat models can therefore serve different contexts — a 27B stops at its
design window, a 4B takes the full target — but they are priced against the
same number. The worst set branch is charged once, before `chat_ctx` is
solved; there is no per-set re-solve.

A model smaller than its own category's `min_context` at every rung emits
**nothing** for that category. The fallback entry is the floor on that: every
model gets one, and it too can fail.

### Knobs (matrix section keys)

| key | default | meaning |
| --- | --- | --- |
| `embed_context` | 4096 | Context the resident embeddings model serves at |
| `rerank_context` | 4096 | Context the resident rerank model serves at |
| `auto_parallel` | true | Spend leftover VRAM on concurrent chat slots: unpinned chat models on llama-server/vLLM get a value-function (ctx, slots) solve — score `(ctx/floor)^0.5 × slots^parallel_power`, best feasible pair wins. Set `false` to disable fleet-wide; a sidecar `parallel:` pin opts a single model out |
| `auto_parallel_max` | 8 | Hard cap on auto-parallel slots |
| `parallel_power` | 0.75 | Slot exponent B in the auto-parallel score: what each extra simultaneous chat is worth. See "What `parallel_power` actually controls" below — the neutral point is model-dependent, so a fleet-wide value is a *bias*, not a target |

### What is left from the old design

Two different things, and they should not be confused:

`sets:`, `evict_costs:` and `categories:` are **still live** — they are
forwarded verbatim to llama-swap's matrix router, which is their consumer.
What changed is that admission is now per entry rather than per model: a
`.coding` entry and a `.aux` entry of the same model get independent
verdicts, because they are independently sized servings. A `.aux` entry
that cannot hold its 80k minimum beside the 5.7 GB reranker is dropped from
`ragz` while the `.coding` entry stays in it.

`sets:` is also the **budget contract**. A set is a role composition
(`rag: __CHAT_VARS__ & emb & rnk` is embed *and* rerank; `talk:
__CHAT_VARS__ & stt & tts` is 2-way speech), and the chat solve reserves the
worst branch of the most expensive chat-capable set it can be requested
under: `&` sums, `|` maxes, `+set` expands, nesting prices recursively. It is
not the sum of every declared category, and a category in no chat set
reserves nothing (llama-swap can only hold it alone). A second category of
the same role is an alternative, never an addition — one model per role per
set. The rule is the sets' own: summing every declared category would make
every chat solve pay for a bucket even when the set holding it is an
alternative to a cheaper one.

`tools_min_ctx`, `min_chat_ctx`, `coload_min_ctx`, `ctx_gain_min` and
`estimate_headroom` shape the shared solve, in that order: the RAG pair is
squeezed only when chat falls below `tools_min_ctx`, no further than
`coload_min_ctx`, and only if the gain reaches `ctx_gain_min`; a co-load is
admitted only while chat stays at `min_chat_ctx` (or `tools_min_ctx` when
the chat model still has tools); an estimated, not measured, overhead is
padded by `estimate_headroom`. Defaults: 128k, 64k, 20k, 4k, 1.25.

### What `parallel_power` actually controls

The score is `(ctx/floor)^0.5 × slots^B`. The context exponent is fixed; `B`
is the single knob, and it decides **when to trade context for concurrency.**

Slots and context compete for the same KV pool, so doubling the slot count
costs context in proportion to how expensive that model's slots are. Writing
`r = ctx(p)/ctx(2p)` for that loss, a slot count only wins when:

```
2^B > r^0.5     ⟺     B > 0.5 · log2(r)
```

So the neutral point is a property of the model, not a constant:

| slots cost | context lost per doubling | neutral `B` | at the default 0.75 |
|---|---|---|---|
| nearly free | ×2 | 0.50 | always takes more slots |
| moderate | ×4 | 1.00 | stops around 2 slots |
| expensive | ×8 | 1.50 | strongly prefers context |

That is the intended shape, and it lines up with how the fleet is used. A
large agentic model carries both a high `floor` and an expensive per-slot
cost, so it is pushed to `parallel: 1` — its `floor` comes from its category
(128k for `coding`), and any second slot would drop it below a window that
cannot drive a tool call. A small quick model has a low floor and cheap
slots, so the same `0.75` walks it up to the cap. `floor` does most of the
work; `B` sets the rate at which the other side gives way.

The null category runs the same solve with exponent **0** — slots worth
nothing in the score, so the choice is purely "can it hold the context at
more of them", which is what a plentiful-VRAM small model should do. Ties go
to *more* slots; with a positive exponent a capped model scores strictly
higher at every slot count and never ties, so this only affects the flat
case.

Two consequences worth knowing before turning it:

- **Raise `B` to protect context**, lower it to buy concurrency. There is
  no value of `B` that is correct for every model at once — that is the
  price of one fleet-wide knob.
- **The floor gates first.** A candidate below `floor` is rejected outright,
  so `B` only arbitrates between candidates that already clear it.

Invalid values warn and fall back to the default; the solve never fails on a bad knob.

### tools demotion

A chat entry declaring `capabilities: [tools]` that its own solve serves
below `tools_min_ctx` gets the capability demoted: `capabilities.tools:
false` and `metadata.tools_demoted: true` (the sidecar declaration is
untouched, so re-packing at a higher served context restores it). This keeps
`/v1/models` consumers (hermes, opencode, UIs) from sending tool calls to a
window too small to serve them. `capabilities.context` reports the VRAM-served
per-slot window, the same number as `metadata.ctx_size`; the trained maximum
stays in `metadata.context_max` when the entry serves below it.

## Context Tiers

Smallest to largest, for any served entry:

1. `_MIN_CTX_SIZE` (4096) — hard emit floor for any served entry.
2. The category's `min_context`, or the null category's 4096 — the floor the
   `(ctx, parallel)` search will not trade below.
3. `design_context` / `--max-context` — never raised, only clamped down.

A context size is written as a positive integer or as a `128k` / `1M`
string; anything else is not a context and is rejected rather than guessed
at.

`sidecar context_length:` is the model's **design maximum**, a ceiling applied
when serving. It is never a floor: reading it as one made a 262144 design max
an unreachable floor, forced the model to one slot, and left the only remedy
editing the sidecar — rewriting a claim about the model to express an
operating decision. `min_context:` is the key that states a floor, and it was
already separate.

`--min-context` overrides the floor fleet-wide when set explicitly, never an
explicit sidecar `min_context:`.

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
- `--draft-p-min 0.75` — Minimum draft-token acceptance probability (configurable via `mtp_draft_p_min`).
- `--spec-draft-model <path>` — (companion MTP only) Path to the draft model file.

### Per-Model Configuration

MTP behavior can be overridden per model via the `.md` sidecar frontmatter:

```yaml
---
name: my-model
mtp: true
mtp_spec_type: draft-mtp      # default: draft-mtp
mtp_draft_n_max: 3            # default: 2
mtp_draft_p_min: 0.8          # default: 0.75
---
```

If absent, the module-level defaults apply (see `llama_packer/consts.py`).

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
rule (see below) that sets `backend: vllm` (host binary), `backend: vllm-podman`
or `backend: vllm-docker` (container). The emitted entry runs `vllm serve`,
published to llama-swap's `${PORT}` host macro. Everything else works
identically: aliases/modes (`filters.setParamsByID`), `metadata`, capabilities,
matrix routing.

All three endpoints are supported, mapped onto vLLM's pooling interface:

| task | vLLM flag | endpoint |
|------|-----------|----------|
| `chat` | *(generation; speculative decoding applies)* | `/v1/chat/completions` |
| `embeddings` | `--task embed` | `/v1/embeddings` |
| `rerank` | `--task score` | `/v1/rerank`, `/v1/score` |

A sidecar declares the engines that can load it with `engines:` (P2; until
that lands the same field is spelled `backend:`). Without a declaration,
selection is inferred from file format, gated by the profiles allow list; chat
templates and LoRA adapters come from pattern-scoped override rules.

```yaml
# profiles.yaml
overrides:
  - when: {base_model: 'qwen3\\.30b'}
    backend: vllm            # or vllm-podman / vllm-docker
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

The container image (`vllm-podman`/`vllm-docker`) is resolved, highest to lowest:

1. Per-model `vllm_image:` frontmatter
2. `--vllm-image` CLI flag
3. `vllm.image` in `profiles.yaml`
4. Built-in default (`vllm/vllm-openai:latest`)

The binary (`vllm`) is resolved, highest to lowest:

1. `--vllm-server` CLI flag
2. `vllm.bin` in `profiles.yaml`
3. `$VLLM_BIN_DIR`
4. the run directory
5. Built-in default (`vllm` on PATH)

See [Host binary resolution](#host-binary-resolution) — steps 1-4 are the
chain every host backend shares.

`profiles.yaml` `vllm:` also configures `container_args` (legacy `docker_args` /
`podman_args`), `container_port`, `container_vendor` and `hf_cache` for the
container pairs.

### Container (vllm-podman / vllm-docker)

llama-swap has no native container abstraction: a containerized backend is a normal entry
whose `cmd:` is `<runtime> run --name ${MODEL_ID} … <image> <vllm serve flags>` — server
flags are argv after the image, container env is `-e` inside `cmd` (an entry's `env:` list
reaches only the container client process). docker and podman share one implementation
(see [docs/transports/](docs/transports/)); two upstream-documented lifecycle fields are
emitted with every container entry:

- `cmdStop: <runtime> stop ${MODEL_ID}` — an unload (swap, manual, or TTL) stops the
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

Every vLLM container entry also gets `-e HF_HOME=/root/.cache/huggingface` and
`-e HF_HUB_OFFLINE=1`: **llama-packer never downloads**. A repo-id model that is not
pre-staged in the mounted hub fails fast at startup — that is the cache-miss case, not a
bug; llama-packer warns at pack time when `hf_cache` is unset for a repo-id model.

`container_args` (default: vendor-detected device flags + `--shm-size=16g`;
legacy keys `docker_args`/`podman_args`) is the operator's
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

`sd-server` has no `llama-fit-params` analog, so VRAM is fixed overhead: `model_mib = file-size(diffusion)`, zero KV terms, `compute_mib = 512`. `ctx_size` tracks `design_context` (sidecar `context_length` or default) but does not affect VRAM. A 40 GB diffusion model will not fit after the resident tier, and the category solve says so rather than the config pretending otherwise: if no rung fits, no entry is emitted.

### Binary precedence

`sd-server` is resolved: `--sd-server` CLI flag > `sd.bin` in `profiles.yaml` > `$SD_BIN_DIR` (file or directory containing `sd-server`) > the run directory > `sd-server` on `PATH`. Absent binary disables format-based inference; an explicit `backend: sd-server` pin to a disabled setup is an error that skips the model. Docker variant is future work (tracked in `docs/plans/comfyui-sd.md`).

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
the run directory > `whisper-server` on `PATH`. Absent binary disables format-based inference;
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
without affecting VRAM. s2t models can be declared
**residents** (see [VRAM Allocation](#vram-allocation)), in which case they
are reserved alongside RAG and served at a fixed small context so repeated
lookups do not reload them. The emitted `--parallel` maps the sidecar/profile
slot count to concurrent transcription workers.

## Audio Backend (audio-cpp)

A model with `role: t2s` (text-to-speech) or `role: s2t` (ASR) is served
through the native **audio.cpp** engine (`audiocpp_server`). The full contract
— roles, sidecar keys (`audio_cpp: {family, task, options, voice, voice_ref}`),
profiles.yaml keys (`audio_cpp: {bin, backend, device, …}`), the emitted
`sh -c` heredoc, classification, VRAM and matrix categories — lives in
[docs/backends/audio-cpp.md](docs/backends/audio-cpp.md).

```yaml
# profiles.yaml
dirs: {t2s: t2s, s2t: s2t}
backends: [llama-server, audio-cpp]
```

GGML whisper `.bin` models remain on `whisper-server` (see the section above) —
audio.cpp has no whisper family and cannot load them.

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
the registry walks backends in **registration order** — `llama-server`, `vllm`,
`vllm-podman`, `vllm-docker`, `sd-server`, `whisper-server`, `audio-cpp` — and
picks the first whose registered formats cover the model AND whose required
resources are configured (llama-server binary, vLLM image / binary, container
runtime on `PATH`). For this purpose **a locally resolved model file's extension
wins over `hf_repo`**: an HF repo id only drives selection when the model has
no local file. Today that means `.gguf` → `llama-server`; safetensors /
`hf_repo` → the vLLM pairs (host first, then podman/docker); `.bin` under
`s2t/` → `whisper-server`; audio GGUFs under `t2s/`/`s2t/` → `audio-cpp`. A
format no available backend covers logs an error and the
model's entries are skipped; so does a rule or sidecar naming an unregistered
`backend`.

**Path resolution.** `chat_template` and `loras` values are paths resolved
relative to the sidecar's own directory (absolute refs pass through). A missing
file logs an error and the model's entries are skipped (fail loud). Symlinks
are preserved by name (not dereferenced), so a chat template symlinked into the
HF cache stays under `${MODELS_DIR}` instead of widening it. Resolved paths are
written into the generated `cmd` as `${VAR}` path macros.

**File refs.** Every key that names a file (`model:`, `mmproj.file:`,
`speculative:`, `chat_template:`, `loras:` entries) accepts either a string
or a mapping. `{file: foo.bar}` is equivalent to the bare string `foo.bar`;
`{hf_repo: org/repo, file: foo.bar}` names a file inside the local HF hub
cache (tracked across `hf download` updates via `refs/main`, like weights —
no pinned copy needed). `hf_repo` falls back to the model's own `hf_repo`
when omitted (handy for LoRAs shipped in the weight repo); the legacy
`hub:org/repo:file` string form is accepted everywhere too. An optional
`pick:` token string (or list) disambiguates multiples: `newest` / `oldest`
select the snapshot revision by mtime (default follows `refs/main`), and
`top` restricts matching to files at the snapshot root (ignoring
subdirectory copies, e.g. a template repo's `archive/` versions).
Ambiguity still fails loud — `pick` only narrows the candidate set, it
never silently takes the first hit. Local files always win: an absolute
`file:` with `hf_repo:` warns and ignores the repo. Mappings merge per key
across layers, so a sidecar `{file: x}` plus a rule `{hf_repo: R}` combine
to `{file: x, hf_repo: R}`.

**Chat templates & client kwargs.** A declared `chat_template` makes the writer
emit `--jinja --chat-template-file <path>` (llama-server) or `--chat-template
<path>` (vLLM), and records `metadata.chat_template` (the file stem). The
`chat_template_kwargs` map is **client-facing metadata only** — there is no
server-side flag for it; clients pass it per-request (e.g. Qwen's
`enable_thinking`).

**Backend support matrix.** An engine that cannot serve a model — wrong file
format (e.g. a `.gguf` under vLLM) or an endpoint its task needs (e.g.
`openai.embeddings` under sd-server) — logs an error and the model's entries
are skipped entirely. A backend that recognizes a setting it cannot render (e.g. `loras`
under vLLM) logs a warning and ignores that one setting; settings that simply do
not apply (e.g. `cache_type` under vLLM) are silently dropped.

| Engine | Model formats | Endpoints |
|--------|--------------|-----------|
| `llama-server` | `.gguf` | `openai.chat.completions`, `openai.embeddings`, `openai.rerank` |
| `vllm` / `vllm-podman` / `vllm-docker` | safetensors, `hf_repo` | `openai.chat.completions`, `openai.embeddings`, `openai.rerank` |
| `sd-server` | `.gguf`, `.safetensors`, `hf_repo` | `openai.images.generations` |
| `whisper-server` | `.bin` (s2t dir only) | `openai.audio.transcriptions` |
| `audio-cpp` | `.gguf`, `.safetensors`, `hf_repo` | `openai.audio.transcriptions`, `openai.audio.speech` |

Endpoint ids are a closed set (`llama_packer/endpoints.py`): a task maps to
exactly one, an engine declares the ones it serves, and compatibility is that
intersection. An id outside the set is not an endpoint.

## Backend Selection

profiles.yaml's ordered `backends:` list both **enables** and **prioritizes**
backends; when absent, every registered backend is usable in registration
order: engines in declaration order (`llama-server`, `vllm`, `sd-server`,
`whisper-server`, `audio-cpp`) and, within an engine, transports in
`host` > `podman` > `docker` order:

```yaml
# profiles.yaml
backends:
  - llama-server    # tried first for everything it can serve
  - vllm            # vLLM host binary
  - vllm-podman     # container pair
  - sd-server       # image generation (opt-in; needs dirs: img: image)
  # vllm-docker     # absent = disabled, even with resources configured
```

Inference walks this list (availability still filters: a pair without its
binary/image configured, or without its container runtime on `PATH`, is
skipped) and picks the first engine whose formats cover the model and whose
endpoints cover its tasks. A sidecar declares that set explicitly with
`engines:` (P2; until it lands the same field is spelled `backend:`), and a
declared engine that is disabled is an error that skips that model — the
declaration bypasses *inference*, never policy. Registration
order: `llama-server`, `vllm`, `vllm-podman`, `vllm-docker`, `sd-server`,
`whisper-server`, `audio-cpp`.

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
own `args` (audio.cpp is a config-file service — its knobs live under
`audio_cpp:`). Values must be a string of flags (a non-string value aborts the run)
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
   section > backend role defaults) render explicitly on every
   llama-server command and win over conflicting `args` values; `cli_args`
   `-b`/`-ub` is warned about (shadowed, and invisible to the VRAM
   measurement)
4. per-model sidecar `cli_args:`

**A batch key that is unset anywhere in the cascade renders no flag at all**,
so llama.cpp's own builtin applies (`-b` 2048, `-ub` 512) — the defaults are
not copied into the command, which would pin them against upstream. No
llama-server role declares a default: the old embed/rerank `batch` of 4096
was never in effect (llama-server forces `n_batch = n_ubatch` when
embeddings are enabled), so it was removed instead of copied. A malformed
value is warned about and falls through to the next tier. `ubatch` is the
one half that reaches the VRAM measurement shape, and it is stamped with its
**effective** value: an unset
key still serves at the builtin, so `measurement_args()` substitutes it and
`derived: shape` keys stay stable across an upstream default change.

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
| `t2s` | `t2s` | Text-to-speech (audio.cpp via audio-cpp; opt-in via `dirs: {t2s: t2s}`) |
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
new `embed/`/`rerank`/`doc/` orphans work on first run.

Stub generation is **opt-in** (`--create-stubs`). A pack does not write into
the models tree unless asked to: a stub is a file in someone's hand-curated
collection, and creating one beside every orphan is a change to that
collection, not a read of it. Models without a sidecar are still discovered
and served with built-in defaults either way — the stub is an editing
surface, not a requirement.

Sidecars are never written inside an HF hub `blobs/` tree (blob hashes are not
human-readable names). An orphan discovered there gets its stub beside the
human-named snapshot entry that points at the blob; if none resolves, discovery
creates its own symlink named after the repo in the category directory and puts
the stub next to it.

## Host binary resolution

Every backend whose engine is a single host binary resolves it by one chain,
highest first:

1. `--<engine>-server` (`--sd-server`, `--whisper-server`,
   `--audio-cpp-server`, `--vllm-server`)
2. `profiles.yaml <engine>.bin` (`sd:`, `whisper:`, `audio_cpp:`, `vllm:`)
3. `$SD_BIN_DIR` / `$WHISPER_BIN_DIR` / `$AUDIOCPP_BIN_DIR` / `$VLLM_BIN_DIR`
4. the **run directory** — the directory llama-packer is invoked from
5. `PATH`

Steps 2-3 accept a *file or a directory*; a directory expands to
`<dir>/<engine>`, mirroring `LLAMA_BIN_DIR`.

Step 4 is what makes llama.cpp and the rest agree. llama.cpp is found by
scanning the working directory for `llama-b[0-9]*` (`find_bin_dir`), so a
deploy that installs every binary beside one config already resolved
llama.cpp while the other engines went unseen — `shutil.which()` never
looks at the cwd. Each deployment then needed its own `*_BIN_DIR`. One
chain removes the special case rather than adding a mechanism.

An explicit setting that does not exist is **kept as given**, not replaced
by a lower-precedence hit: a stale `bin:` should leave the backend
disabled so `is_available` reports False and the operator gets a message
naming the real problem, rather than silently running a different binary.

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
  `ctx_size`, `context_max`, `max_output_tokens`, `mtp_enabled`, `mtp_draft_max`, `mtp_accuracy`, `throughput_factor`,
  `image_min_tokens`, `image_max_tokens` (vision variants only).
- **Native `capabilities`** — `in`/`out` modalities, `tools`, `reranker`, `context` (derived, see below).
- **Native `name` / `description`** — display fields in `/v1/models`.

### Builder-consumed keys (NOT passed through)

`name`, `context_length`, `description`, `cli_args`, `model`, `backend`, `hf_repo`,
`chat_template`, `chat_template_kwargs`, `loras`,
`speculative`, `speculative_config`, `mmproj`, `mtp`, `mtp_spec_type`, `mtp_draft_n_max`,
`mtp_draft_p_min`, `role`, `allow_profiles`, `capabilities`,
`ignore`, `device`, `concurrency`, `fit-params`, `vllm_image`, `modes`, `default_mode`,
`reasoning-format`, `reasoning-preserve`, `cache_type`, `parallel`,
`image_min_tokens`, `image_max_tokens`.

### Per-model config options

| Frontmatter Key | Type | Effect |
|-----------------|------|--------|
| `device` | int | GPU device index for multi-GPU pinning (`ROCR_VISIBLE_DEVICES=N` / `CUDA_VISIBLE_DEVICES=N`) |
| `concurrency` | int | Per-model concurrency limit → `concurrencyLimit` in config |
| `allow_profiles` | str/list/bool | Restrict which profiles apply (regex string, list, or false to disable) |
| `modes` | dict | Per-model sampling modes (full profiles): name → param dict. Replaces the global-profile sampling overrides for this model. Values use llama.cpp names; see [Sampling Modes](#sampling-modes) |
| `default_mode` | str | Which declared `modes` entry is the model's default (maps to the bare `${MODEL_ID}` `setParamsByID` key). Falls back to the first mode |
| `reasoning-format` | str | llama-server `--reasoning-format` (`none`/`deepseek`/`deepseek-legacy`/`auto`). Chat + reasoning-capable models only; see [Reasoning](#reasoning) |
| `reasoning-preserve` | bool | Emit `--reasoning-preserve`. Chat + reasoning-capable models only |
| `cache_type` | str | KV-cache precision for `--cache-type-k/v` and VRAM sizing (sidecar > profile > `q8_0`); see [Cache precision](#cache-precision-cache_type) |
| `parallel` | int | Parallel slots for `--parallel` and VRAM sizing (sidecar > profile > 1). Declaring it opts out of auto-parallel for this model |
| `min_context` | int | Per-model minimum useful context (tokens): explicit floor of the auto-parallel search and its score normalizer. Cascade, highest first: this key → pinned ctx (`--max-context`) → the heaviest declared `usage:` category → tool callers 131072 → half the design context (`--min-context` overrides the last two when explicitly set, never this key or a pin) |
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

- **`capabilities`** (native block): directional modalities, matching llama-swap's badge derivation — `in` = `["text"]` + `"image"` if `image` + `"video"` if `video` + `"audio"` if `audio`; `out` = `["text"]` + `"audio"` if `speech` + `"video"` if `video` + omni/video-arch (architecture `omni`/`video`/`wan`/`h3` etc). Output stays text unless `speech`/`video` output is declared, so a vision model never advertises text→image (`Image Gen`) unless `role:image`. `tools`/`reranker` boolean flags; `context` = the served per-slot window (`ctx_size`, the same number clients size compaction and output room against). The trained maximum stays in `metadata.context_max` when the entry serves below it. `architecture` guides image vs video output for `role:image` (`out:[video]` when `video` in capabilities or `architecture` contains `video`/`wan`/`hunyuan-video`/`h3`).
- **Model-kind guard**: every candidate in a served role is classified header-only (`general.architecture` + `<arch>.context_length` for GGUF; tensor-name blocks for safetensors; cached HF card `pipeline_tag` as offline fallback). Weights classified as diffusion/image-generation are excluded with an error log instead of being served.
- **`throughput_factor`**: Heuristic relative speed index = `54 / (active_B × quant_bits)` × `(1 + draft_n × mtp_accuracy)` when MTP is on. Relative only — not real tok/s.
- **`ctx_size`**: The VRAM-served per-slot context limit (`-c` / `--max-model-len`), exposed via `metadata.ctx_size`. `capabilities.context` repeats it because that is what `/v1/models` consumers read; `metadata.context_max` keeps the trained maximum when the deployment serves below it.
- **`max_output_tokens`**: The client-facing generation ceiling for chat entries. Defaults to the served window (`ctx_size`, the true llama-server limit); a sidecar may set a lower positive integer for a model whose long replies degrade, and the emitted value is clamped to the window.

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
`derived` nested block — and `--dry-run` writes none of them — the only dynamically generated branch
(llama-packer's own calculation cache: never hand-edit, delete to force
re-evaluation; legacy `fit-params`/`measured` blocks are located but
rewritten as `derived` on next persist).

`--dry-run` is contractually read-only: it creates, modifies and deletes
**no** files. Discovery would otherwise write a `derived: file:` fingerprint
into every sidecar it merely looked at, and stub sidecars beside every
orphan model file, so both are gated behind the same flag as the config
output.

Every withheld operation is then **reported**, so a preview is evidence
rather than silence — silence being what made the three writers invisible in
the first place:

```
$ llama-packer --dry-run --agents --create-stubs ...
would write: /mnt/ai/models/AGENTS.md (AGENTS.md guide)
would update derived: in: /mnt/ai/models/chat/qwen3-30b.md (qwen3-30b)
would create stub sidecar: /mnt/ai/models/chat/orphan.gguf
would write: config.yaml
would write: config.env
```

One flag (`utils.DRY_RUN`) and one reporting function
(`utils.note_fs_op`) serve all of it, so a new writer cannot silently bypass
the report. `--dry-run` lifts the log floor to INFO, since the report is the
point of the run (see [Log Levels](#log-levels)). It also composes with
`--agents` and `--create-stubs`: those previews report what they would write
rather than being refused.

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

## Log Levels

The pack's output is `config.yaml`; the log on stderr says what happened on
the way there. The default level is **WARNING** — a run that goes well says
nothing, and anything that went wrong is reported without a flag.

`-v` lifts the level to INFO, `-vv` to DEBUG, `-vvv` to whatever the
libraries log. It only ever lifts: asking for more detail never silences a
warning.

`--dry-run` lifts the floor to INFO whatever the verbosity, because its
report — every write it withheld — *is* the output of the run.

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
