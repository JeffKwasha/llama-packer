# Plan: Opportunistic co-loading of small models + main-chat context

Status: **proposal — not scheduled**. Recorded 2026-08-30.
Related: `docs/plans/matrix-categories.md` (declarative categories — this plan is the automatic counterpart), `SPEC.md` Matrix Context Solving, `llama_packer/{vram,writer,__main__}.py`.

## Current state

**Budget math** (`vram.solve_matrix_ctx:680`, `writer._solve_matrix_context:531`):
the matrix solve reserves fixed overhead for exactly two non-chat models
(embed, rerank, each at its `design_context`) and solves one shared chat
context for the **largest** chat model:

```
reserve     = 1024 + max(1024, baseline_mb)
available   = vram_total − reserve − spare
chat_ctx    = solve Σ over largest chat:  weight + factor×ctx
              ≤ available − embed_overhead − rerank_overhead
```

Per chat model the solved ctx is then clamped to its own `design_context`
(`writer._bounded_ctx:383`) and floored at `_MIN_CTX_SIZE`. So "how the
main-chat context is determined" today = *solve once for the biggest chat
model, clamp per model, floor at 4k* — small chat models already effectively
get their design context via the clamp.

**Exclusion of s2t/image**: `_solve_matrix_context` skips every
`NON_CHAT_ROLES` model outside emb/rnk ("Including a 40 GB diffusion model
would collapse the chat budget"). Consequence: whisper/image models co-reside
with chat *unbudgeted* — an operator who runs `s2t` + RAG can OOM at load
time with no warning.

**Why they're excluded — no sizing**: `vram.fit_params_static:268` measures
via `llama-fit-params` (llama.cpp GGUF only) or the vLLM/safetensors
estimators. Whisper `.bin` (GGML format) and sd.cpp GGUF (diffusion arch)
have no path → `FitParams = None` → cannot be budgeted → excluded.

## Goal

1. **Size the unmeasurable.** Give s2t/image models an estimation path so
   they have `(model_mib, ctx_factor, compute_mib)` triples like everyone else.
2. **Co-load opportunistically.** Any non-chat model that *fits* alongside the
   chat floor joins the resident set automatically — smallest-first, budgeted,
   warned when skipped — instead of only emb/rnk by rule.
3. **Keep chat-context determination explicit** (documented, unchanged in
   shape: fixed co-load overheads out of `available`, solve for largest chat,
   clamp/floor per model).

## Design sketch

### 1. Sizing: `source: estimate` file-size fallback

Generalize `_estimate_safetensors` (`vram.py:373`) into `_estimate_file_size`:

- Trigger: `llama-fit-params` fails/unavailable AND the file is not
  safetensors — i.e. whisper GGML (`.bin`) and GGUF with a non-llama-family
  arch (arch is already read for `gguf_context_length`).
- Triple: `model_mib = file_size`, `ctx_factor = 0` (whisper-server takes no
  context flag; sd.cpp neither), `compute_mib = small per-kind constant`
  (e.g. ~512 MB for s2t encoder activation, ~1 GB for diffusion) — declared
  in one table.
- **Fixed size in config beats estimation**: since `ctx_factor = 0` for s2t,
  the whole fit is a constant anyway. A sidecar field (`vram_mb`) pins the
  process VRAM outright — when present it *is* the sizing (skip file-size
  estimation entirely); the file-size path then only fills models lacking
  the field. Measure once with `nvidia-smi` (server warm, after a long
  transcription) and write the number in.
- `source: "estimate"` persisted to the fit-params block like today, so the
  provenance of the guess is visible in the sidecar (a pinned `vram_mb`
  marks `source: "config"`).

Estimated models are never asked to *solve* a context — they are fixed
overhead, exactly like `on_cpu` models cost 0 today.

### 2. Opportunistic inclusion pass

In `_solve_matrix_context`, after the emb/rnk fixed overheads:

1. Candidate pool: enabled non-chat models not already budgeted
   (s2t, image — t2s is a containerized service, separate pool, excluded).
2. Order by estimated fixed overhead ascending (smallest first).
3. Include each candidate whose overhead still leaves the chat solve at or
   above a **floor**; skip (with a warning naming model and MB) otherwise.
   A skipped candidate does not block smaller ones later in the list.
4. Emit a matrix var per included model and extend the built-in set
   expression (`__CHAT_VARS__ & emb & rnk & s2t & img`) so llama-swap keeps
   the group resident together. Not-included non-chat models stay outside
   the sets (independent eviction), same as today. A combined multi-model
   process (nemo-speech serving ASR + VAD + diarization) is *one*
   llama-swap entry → one var; its overhead is the sum of all loaded model
   files plus one compute constant per model.

Floor guardrail: chat must retain at least `matrix.min_chat_ctx`
(new, default: **64K**) or the candidate is skipped. This is what keeps a
4 GB flux from "collapsing the chat budget" *automatically* — the rule
replaces the blanket role exclusion. The floor is per-chat-model: a model
declaring the `tools` capability floors at **128K** (`matrix.tools_min_ctx`,
default 128K) — tool calling needs the larger window; if solving can't reach
128K even with zero co-loads, the model's `tools` capability is demoted (see
Capabilities below) rather than the co-loads force-dropped.

Estimate headroom: a candidate sized by `source: estimate` must fit with
**1.25× headroom** (`matrix.estimate_headroom`, default 1.25) since its
sizing is a guess; measured (`source: fit-params`) candidates need 1.0×.

Backward compat: emb/rnk remain **unconditional** residents (the RAG
contract, current behavior — even if they push chat below the floor); only
s2t/image are opportunistic. When no matrix section is configured, nothing
changes (co-residency stays unmanaged, as today).

### 2b. emb/rerank context yield (squeeze)

emb/rnk are unconditional but not untouchable: when the baseline solve puts
chat below `matrix.tools_min_ctx` (128K), the solver may **squeeze** their
contexts down to `matrix.coload_min_ctx` (default **20K**) and hand the
reclaimed KV back to chat:

1. Solve baseline with emb/rnk at their `design_context` → `chat_ctx₀`.
2. If `chat_ctx₀ < tools_min_ctx`, re-solve with each of emb/rnk at
   `min(design_context, coload_min_ctx)` → `chat_ctx₁`. KV reclaimed is
   `(design − 20K) × factor` per model.
3. Adopt the squeeze **only if** `chat_ctx₁ − chat_ctx₀ ≥ matrix.ctx_gain_min`
   (default **4096**); otherwise keep design contexts (not worth perturbing
   embed/rerank quality for crumbs).

The squeeze must be *realized*, not just solved: the reduced ctx is emitted
into the embed/rerank commands (today they run at design ctx — that emit is
what actually frees the VRAM), and the squeeze result feeds the
opportunistic pass (co-load candidates are evaluated against the adopted
baseline). Order: baseline solve → optional squeeze → opportunistic adds.

### 3. Main-chat context determination (documentation + one knob)

Unchanged in shape, now with co-loads accounted:

```
available   = vram_total − reserve − spare − Σ(coload fixed overheads)
chat_ctx    = max ctx where largest chat model fits
per entry   = clamp(chat_ctx, model.design_context, --max-context),
              floor _MIN_CTX_SIZE
```

Worth documenting in SPEC.md because it is currently implicit: the shared
solve targets the *largest* chat model and smaller models are clamped down
to their own design context — they are never *raised*, and they never get a
worse ctx than the big model's solve. The context tiers, smallest to
largest: `_MIN_CTX_SIZE` 4096 (hard emit floor) → `matrix.coload_min_ctx`
20K (emb/rnk squeeze floor) → `matrix.min_chat_ctx` 64K (co-load *decision*
floor — emb/rnk may still push chat below it, at which point tools demotion
is the signal) → `matrix.tools_min_ctx` 128K (tools advertisement threshold)
→ `design_context` / `--max-context` clamp. The emb/rnk squeeze (2b)
operates in the gap between the 4096 emit floor and the 128K trigger.
A future refinement (per-chat-model
solving, so a 4B chat model keeps design ctx while a 70B scales) is out of
scope here; the clamp already delivers most of that benefit.

### 4. Capabilities: `tools` vs served context (analysis)

**What goes here.** llama-swap's native capabilities block (`writer.py:284`)
advertises `tools: true/false` per entry. Today it is a *static passthrough*
of the sidecar declaration — a model declared `capabilities: [tools]`
advertises tool calling even when the planner serves it at a solved ctx far
below what tool calling needs. Clients (`/v1/models` consumers: hermes,
opencode, UIs) cannot tell the difference, because `capabilities.context`
reports the **max trained** context (`model.py:296` design_context) while the
actually served limit lives separately in `metadata.ctx_size` (`writer.py:276`).

**Where from.** The chain is: sidecar frontmatter `capabilities: [...]`
(hand-authored; template at `templates/models_AGENTS.md:113`) →
`Model.capabilities` (`model.py:507`, vision auto-added with mmproj) →
writer maps to llama-swap's block (`in`/`out`/`tools`/`reranker`/`context`).
The `reasoning` capability already gates server-flag emission
(`writer.py:45`) — so capability → emission coupling has precedent; `tools`
is the one capability with no serving-constraint coupling.

**Who fixes it.** The **builder (writer)**, not the sidecar: the sidecar
stays the source of truth for the model's *intrinsic* capability; the writer
applies *serving* constraints at emit time:

```
emitted tools = declared tools AND (planned ctx_size ≥ tools_min_ctx)
```

with `matrix.tools_min_ctx` default 128K and a warning when demoted
("model X declares tools but is served at 8k — demoting"). The `-text`
variant and per-profile variants each get their own evaluation (a text-only
serving may clear 128K where the vision serving does not). Optionally expose
`metadata.tools_demoted: true` so clients that care can distinguish
"not tool capable" from "demoted for context". No llama-swap change needed —
it just reflects what we emit.

### 5. Sequencing vs matrix-categories.md

This plan is the **automatic default**; categories (that proposal) are the
**declarative override**. They compose: the opportunistic pass computes the
default var/set set, and a future `matrix.categories` would let operators
pin/extend it. Ship this first — it needs no new config surface.

## Resolved decisions (2026-08-30)

* Co-load floor: `matrix.min_chat_ctx` default **64K**; per-model 128K when
  the model declares `tools` (`matrix.tools_min_ctx`).
* Estimate headroom: **1.25×** for `source: estimate` candidates
  (`matrix.estimate_headroom`), configurable.
* emb/rnk stay **unconditional** residents regardless of the floor (revisit
  when `matrix-categories.md` lands), but their *context* yields: squeeze to
  20K (`matrix.coload_min_ctx`) when chat is below 128K, adopted only if it
  buys chat ≥ 4K (`matrix.ctx_gain_min`).
* Compute constants: whisper.cpp's own table puts runtime at disk +
  ~200 MB (tiny) … ~1 GB (large). The measured nemo-speech/Vulkan data
  (below: process footprint ≈ Σ files) suggests the shared runtime
  overhead is small and per-model activation memory is modest for s2t —
  a small shared constant + ~0–100 MB per model, not a big per-model
  constant. Diffusion needs more. **Roadmap item: measure both once on
  real hardware** and bake the constants into the per-kind table.
  whisper-server also runs acceptably on CPU/system-RAM, so a CPU-resident
  s2t co-load can be budgeted at 0 GPU cost.

## Research: nemo-speech as the preferred s2t backend

Researched 2026-08-30 (HF model card, `NVIDIA/NeMo-Speech.cpp`, whisper.cpp
README). The big open question — runtime story — is **resolved**:

### Findings

- **NeMo-Speech.cpp** (github.com/NVIDIA/NeMo-Speech.cpp) is NVIDIA's
  official local inference runtime for the Nemotron Speech family, built on
  ggml — it submodules both `ggml` and `llama.cpp`. GGUF-native, same
  ecosystem as whisper.cpp. Apache-2.0.
- **Official GGUF artifacts confirm the Q8_0 assumption**: the model card's
  quickstart downloads `parakeet-tdt-0.6b-v3.q8_0.gguf` (0.6B × ~1.0625 B/w
  ≈ **~640 MiB** file). Community quantizations also exist (72 on HF).
- **It has a server**: `nemo-speech serve` — HTTP with
  `POST /v1/audio/transcriptions` (OpenAI-compatible subset),
  `GET /v1/models`, `/health`, `/ready`, plus realtime WebSocket
  `/v1/realtime` (PCM16). YAML/env/CLI dotted config (`asr.model.path`,
  `asr.backend.gpu: 0` — GPU pinning). This is a drop-in shape for a
  `nemo-speech-server` backend: a near-clone of our whisper-server backend
  (proxied HTTP; use `/ready` or `/` as llama-swap `checkEndpoint`).
- **ASR model menu**: Parakeet TDT 0.6B v3 (25 languages, auto language
  detect), Nemotron 3.5 ASR Streaming 0.6B, Nemotron Speech Streaming EN
  0.6B, Parakeet CTC 1.1B. Diarization (Sortformer 4-spk) standalone or
  fused into ASR results.
- **Multi-model in one process** — `serve` takes separate model flags per
  capability (`--asr-model`, `--diar-model`, `--tts-model`, `--codec-model`)
  and "capabilities auto-enable when their required model paths are present";
  the docs ship a "combined ASR, diarization, NMT, and TTS server" example.
  "Loads each configured model once" = per request, not per process.
  This **supersedes the multi-entry budget assumption**: the planned
  deployment is **one llama-swap entry, one process, three models** — main
  s2t (`--asr-model`), dead-air detection (VAD, `asr.vad.*` engine keys),
  speaker separation (`--diar-model sortformer`; standalone via
  `POST /v1/audio/diarizations`, or fused into ASR results via `asr.diar.*`).
  Co-load budget for the entry = **sum of the loaded model files** + a
  per-model compute constant — no ×N multiplication.
- **No context knob** — ASR memory doesn't scale with a `-c` analog; long
  audio (24 min full attention / 3 h local attention) is a model property.
  Confirms `ctx_factor = 0`, fixed overhead = file size + compute constant.
- **Backends**: CUDA (`cuda-server` preset; A10/A100/L4/L40/T4 verified),
  CPU, Metal, Vulkan. HF card: "at least 2 GB RAM to load" (F32 view; the
  Q8_0 GGUF path is ~1 GB runtime footprint).
- **Bonus — native TTS**: MagpieTTS Multilingual 357M + NanoCodec run under
  the same server (`/v1/audio/speech`, OpenAI subset). A future
  `nemo-speech` TTS route could replace kokoro-podman and drop the container
  dependency entirely.

### Sizing inputs

- **Combined nemo-speech process (measured artifacts, Vulkan backend):
  ~1.2 GB total.** parakeet-tdt-0.6b-v3 Q8_0 = **740 MB** file,
  Sortformer 4spk v2 = **471 MB**, Silero VAD = **2 MB**; the whole
  process footprint lands around the sum of the files — Vulkan's runtime
  overhead is far smaller than a CUDA context, and the ggml compute
  buffers are shared. Suggest `vram_mb: 1280` for the combined entry.
- **Shared-overhead rule**: process-level costs (runtime libs, compute
  buffers, scheduler) are counted **once per process, not per model** —
  only weights and per-model activation memory scale with the number of
  models loaded. Applies to the compute-constant table and to any
  multi-model entry (nemo-speech, future combined backends): budget
  Σ(weights + per-model activations) + one shared constant.
- whisper.cpp official memory table (Disk → Mem): tiny 75→273 MB,
  base 142→388 MB, small 466→852 MB, medium 1.5→2.1 GB, large 2.9→3.9 GB —
  i.e. runtime ≈ disk + **200 MB … 1 GB** depending on model size. The
  nemo-speech/Vulkan measurement (Σ files ≈ footprint) sits well under
  this — CUDA-context overhead likely dominates whisper.cpp's larger
  deltas; treat the whisper table as CUDA-era upper bounds.

### Remaining questions (much shorter than before)

- Vulkan release binaries: are Linux Vulkan artifacts shipped, or
  source-build only? (Operator runs the Vulkan backend, not CUDA;
  install.sh "prefers a verified native release" — verify Vulkan artifact
  availability.)
- Endpoint parity: does `/v1/audio/transcriptions` accept the same request
  fields we emit against whisper-server (file upload + response_format)?
  Documented subset suggests yes.
- Identify the 4th planned model slot: candidates are the VAD gguf (dead-air
  detection itself — Silero-class, tiny) vs the NanoCodec decoder
  (TTS-only; not needed for the ASR+diarization process). User plans main +
  dead-air + sometimes diarization = 3 resident; the 4th may be optional.
- Streaming ASR variants (Nemotron 3.5) vs batch Parakeet TDT: pick default
  per role `s2t` (dictation vs file transcription).
- Whether to plan a MagpieTTS-based `t2s` backend to retire kokoro-podman.

## Remaining open questions

* Should the tools demotion also *skip* opportunistic co-loads to preserve
  128K (co-load pass yields before the tools floor), or is advertising the
  demotion sufficient? (Plan: demotion only — the floor logic above already
  prefers 128K for tools models; demotion is the last resort.)
* t2s (kokoro/podman) GPU usage is unaccounted today; keep out of scope?
  (Note: the nemo-speech MagpieTTS finding above offers a native t2s path
  that would remove the container entirely — a separate plan.)

## References

* `llama_packer/vram.py:268` `fit_params_static` (sizing cascade; where the
  file-size fallback hooks in at step 4)
* `llama_packer/vram.py:373` `_estimate_safetensors` (pattern to generalize)
* `llama_packer/vram.py:680` `solve_matrix_ctx` (budget equation)
* `llama_packer/writer.py:531` `_solve_matrix_context` (inclusion pass site;
  `NON_CHAT_ROLES` skip at :566)
* `llama_packer/writer.py:383` `_bounded_ctx` (per-model clamp)
* `llama_packer/__main__.py:230` `_build_matrix_vars` (var emission to extend)
* `llama_packer/model.py:296` `design_context`
* `docs/plans/matrix-categories.md` (declarative categories, phase 2)
* https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 (model card; q8_0
  GGUF quickstart, "≥2 GB RAM" note, 24-min/3-h audio limits)
* https://github.com/NVIDIA/NeMo-Speech.cpp (ggml runtime; server.md:
  OpenAI-compatible `/v1/audio/transcriptions`, `/health`, `/ready`,
  dotted engine config, `asr.backend.gpu`)
* https://github.com/ggml-org/whisper.cpp (Memory usage table:
  disk 75 MiB→~273 MB … 2.9 GiB→~3.9 GB)
