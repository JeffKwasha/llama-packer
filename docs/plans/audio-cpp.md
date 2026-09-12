# Plan: `audio-cpp` backend (TTS / ASR / VC via a native ggml engine)

Status: **research complete — findings below; implementation still pending** (2026-09-11).
This file is a *design brief* built on upstream research: what llama-packer needs,
the interface (validated against a real llama-swap config), and the resolved open
questions. Requirement IDs and resolved questions are kept so a future
implementation can trace each decision back to evidence.

Related: [`audio.md`](audio.md) (whisper-server + kokoro-podman — the existing
audio backends this complements), [`audio-cpp-vs-whisper.md`](audio-cpp-vs-whisper.md)
(feature comparison: whisper.cpp is ASR-only, audio.cpp is a broad multi-family
engine), [`matrix-categories.md`](matrix-categories.md)
(the `(EMB+RERANK) | (TTS&/or STT)` co-load goal), [`comfyui-sd.md`](comfyui-sd.md)
(the `sd-server` precedent for a non-llama.cpp engine), `SPEC.md` Backend
Selection + vLLM Backend, `docs/architecture.md`.

Seed research (external, provided separately to the researcher — **do not
hardcode its path in repo docs**): a ROCm/RDNA4 build-and-run survey of
`0xShug0/audio.cpp` + Chatterbox, covering binaries, `server.json`, model sizes,
and a TTS-family inventory.

## Sources (all consulted 2026-09-11/12)

- `0xShug0/audio.cpp` `app/server/README.md` — `audiocpp_server` config schema,
  endpoints, `voice_dir`, `max_loaded_models`/`idle_unload_ms`/`busy_timeout_ms`/
  `min_free_memory_mb`, streaming routes.
- `0xShug0/audio.cpp` README + `docs/gguf.md` — 62 families / 85+ variants,
  task tags, package-spec/GGUF layout, build flags.
- `0xShug0/audio.cpp` `docs/build/linux.md` — CMake configure/build per backend.
- **llama-swap issue #36, comment by `@dkruyt` (2026-07-03)** — a real, working
  `pocket-tts` + `qwen3-asr` config driving `audiocpp_server` through llama-swap.
  This is the empirical proof for the proxy path (Q1/Q2/Q7).
- `mostlygeek/llama-swap` README — lists `audio.cpp` as a supported server and
  adds the `/audioapi/v1/tasks/run` extra endpoint.

---

## Why

`audio.cpp` is a pure-C++ ggml inference engine for audio models (TTS, ASR, VAD,
VC, music, separation), **62 model families / 85+ variants**. It ships two
binaries: `audiocpp_cli` (one-shot) and `audiocpp_server` (HTTP server,
OpenAI-compatible `/v1/audio/speech` + `/v1/models` + `/health`, transcriptions,
and a generic `/v1/tasks/run`).

Today llama-packer serves audio through **two single-purpose backends**:
`whisper-server` (`role: s2t`) and `kokoro-podman` (`role: t2s`, a container).
One native engine covering both — and many more families — collapses that
surface, and it is the **forcing function** for two pending workstreams:

- **B — engine/transport seam.** `audio-cpp` is another *engine* (like
  sd.cpp/whisper.cpp) that may be served by more than one *transport* (host
  process today, podman later). We must not add yet another flat backend class.
- **C — configurable matrix categories.** Audio needs to become a co-load
  category to express `(CHAT) + ((EMB+RERANK) | (TTS&/or STT))`.

## Requirements

- **R1 — New engine backend.** A backend (`audio-cpp`) that runs an
  audio.cpp server process and is managed/proxied by llama-swap the same way
  `sd-server` and `whisper-server` are today (`cmd:` + `proxy` +
  `checkEndpoint`). **Resolved: yes, this works** (Q1).
- **R2 — Roles.** At minimum `t2s` (TTS / voice-clone / voice-conversion) and
  `s2t` (ASR). Keep the door open for `vc`, `vad`, `music`, `separation` as
  later roles without a schema rewrite.
- **R3 — Multi-vendor.** CUDA, HIP/ROCm, Vulkan and CPU; auto-select like
  `kokoro-podman` does (vendor detection → binary/flags), with an override.
  Must work on both the NVIDIA Spark (GB10) and AMD RDNA4 boxes.
- **R4 — Model identity.** audio.cpp models are GGUF, so today's
  GGUF→`llama-server` inference would misroute them. The backend must key off
  audio.cpp's **family + task**, not just the file header, and must not let an
  audio GGUF be classified/served as a chat model.
- **R5 — Sidecar expressiveness.** Sidecars must declare family, task,
  per-family options (guidance/temperature/top-p/…), voice/library refs, and a
  fixed VRAM figure — without free-form `cli_args` for the common path.
- **R6 — VRAM sizing.** Fixed-overhead models (like `s2t`/`image`/`t2s` today).
  Define the estimate (`file size + buffer`, or measured `vram_mb`) across
  families from ~200 MB to ~19 GB.
- **R7 — Matrix participation.** Audio declarable as a matrix category so RAG
  and audio alternate under one chat model (depends on C).
- **R8 — Model discovery.** Locate audio GGUF under the existing models roots,
  infer role from directory, resolve `hf_repo`/snapshot paths like other models.
- **R9 — Multi-model server semantics.** How audio.cpp's `max_loaded_models` /
  `lazy_load` / `idle_unload_ms` LRU interacts with llama-swap's per-entry swap
  model.
- **R10 — Voice library.** `voice_dir` of reference WAVs (+ `prompt_text`) so
  clients address voices by name, and inline base64 refs where needed.
- **R11 — Install story.** How the binary is obtained/built per backend,
  consistent with `extras/update`'s CWD-install convention.
- **R12 — Fallbacks stay.** `whisper-server` and `kokoro-podman` remain valid;
  `audio-cpp` is additive and opt-in.

---

## Open questions — answered

### Q1 — Proxy vs standalone. **Resolved: it CAN be proxied.**

The seed doc's claim that audio.cpp "cannot be hosted by llama-swap and must
run as a separate service" is **false**. `audiocpp_server` is a normal HTTP
server on a port, and llama-swap already lists `audio.cpp` as a supported
server (README) with an extra `/audioapi/v1/tasks/run` endpoint.

Empirical proof — `@dkruyt`'s working config (llama-swap issue #36, 2026-07-03)
drives two models through llama-swap exactly like sd-server/whisper-server:
`cmd:` writes a `server.json` (embedding `${PORT}`) and `exec`s
`audiocpp_server --config …`; `proxy: http://127.0.0.1:${PORT}`;
`checkEndpoint: /health`; `ttl:` for unload. No standalone wrapper needed.

**Decision:** treat `audio-cpp` as a proxied HTTP backend like `sd-server`
(`proxied = True`). The forcing question for the implementation is not *whether*
to proxy, but *what `cmd:` emits* (see Q2).

### Q2 — One entry vs many. **Default: one `audiocpp_server` per entry
(1:1), consistent with the rest of llama-packer.**

`audiocpp_server` *can* hold several models at once
(`max_loaded_models` LRU, `lazy_load`, `idle_unload_ms`,
`/v1/tasks/unload_models(_all)`), which overlaps llama-swap's swap model. But
llama-packer's core invariant is **one entry = one process with a fixed command
line** (docs/llama-swap.md "Design consequences of the entry/process model"):
the server's model set is declared *statically* in `server.json`, so the set of
models a given process serves cannot change at runtime — that is exactly a
llama-swap "entry".

Therefore emit **one `audiocpp_server` process per sidecar entry**, each with a
`server.json` declaring that one model (matching `@dkruyt`'s config and the
whisper-server precedent). llama-swap's matrix then handles resident/evict and
swap; each entry's own `max_loaded_models` is effectively a no-op (one model).

The multi-model LRU is still available as an **opt-in advanced deployment**
(one shared `audiocpp_server` serving several families, llama-swap routed by
`model` id) — document it but do not make it the default. See R9.

### Q3 — Role granularity. **Two primary roles now; vc/vad/music/separation as
later roles — schema-stable because `capabilities` are freeform lists.**

audio.cpp `task` values → llama-packer roles + capabilities:

| audio.cpp tasks | role | `capabilities.in` → `out` |
|---|---|---|
| `tts`, `clon`, `music`, `sfx`, `dialogue`, `edit`, `design`, `ctrl` | `t2s` | text → audio |
| `asr` | `s2t` | audio → text |
| `vad`, `align`, `diar` | `s2t` | audio → text (metadata) |
| `vc`, `s2s` | `vc` (later) | audio → audio |
| `sep` | (later) | audio → audio × tracks |

`vc` (audio→audio) differs from `t2s` (text→audio) in capabilities, so it is a
distinct role — but adding a role is only a `roles` + `dir_role_map` change;
`capabilities.in/out` are freeform, so no schema rewrite. Immediate shipped set
is `t2s` + `s2t` (mirrors the existing `whisper-server`/`kokoro-podman` split);
`vc` added when first requested.

### Q4 — Classification. **Directory-authoritative + GGUF package-spec
fingerprint; never routed to `llama-server`.**

audio.cpp GGUFs are **not** llama.cpp chat GGUFs — they carry an audio.cpp
package spec and family-specific tensor layouts. Two layers keep them off the
chat path:

1. **Directory authority** (the whisper-server precedent): the backend only
   activates under audio-mapped dirs (`t2s/`, `s2t/`, later `vc/`), set via
   `dirs:`. A `.gguf` in `chat/` is never an audio model. This alone prevents
   misrouting.
2. **Package-spec fingerprint** (secondary validation): a valid audio GGUF
   embeds `audiocpp.model_spec.*` metadata (or the model `config.json`
   declares `audiocpp_family` / `audiocpp.model_spec`). The base
   `general.architecture` (e.g. `qwen3`) is *not* a reliable discriminator, so
   don't key off it — key off the `audiocpp_*` spec + family/task, which only
   audio.cpp models carry.

**Formats:** `.gguf` (primary), `.safetensors` (`kokoro_tts`, some community
families), and a **model directory** holding `model.gguf` (+ sidecars) — so the
backend's `formats` should be `{.gguf, .safetensors, hf_repo}`, like `sd-server`.
The role gate (`t2s`/`s2t`/`vc`) plus directory mapping prevents the
`.gguf` format from colliding with `llama-server`'s chat selection.

### Q5 — Sizing. **Fixed-overhead; `model_mib = file size + buffer`,
`ctx_factor = 0`, per-family `compute_mib`.**

Audio models have no llama-style KV cache; VRAM is the model weights + a
session/vocoder/diffusion overhead constant. Estimate like `s2t`/`image`:

```
model_mib  = file_size_mib + buffer
compute_mib = <family category constant>   # e.g. 512 (ASR/VAD) … 3072 (diffusion TTS / separation)
vram_mb    = model_mib + compute_mib
```

Range spans **~200 MB** (pocket_tts 100M Q8, bundled `silero_vad`/`marblenet_vad`)
to **~19 GB** (7B F32 like `personaplex`, `vibevoice`, `supertonic` Q4≈4–5 GB,
F32 7B≈14–15 GB). Provide a per-family `vram_mb` override on the sidecar; base it
on the resolved file size and add a category constant.

The server's own `min_free_memory_mb` guard estimates footprint = weights +
session aux files + runtime-overhead factor + fixed floor — a useful runtime
safety net, but llama-packer still needs its own static estimate for the matrix
solve. `max_loaded_models` multiplies the budget **only** when one server holds
N models (the non-deployed shared-server case); the default 1:1 entry does not
multiply.

### Q6 — Voice library. **`voice_dir` of WAVs + `prompt_text`; precedence +
inline base64 (5 MiB cap) documented upstream.**

`voice_dir` = a directory of `.wav` files plus a `prompt_text` file with one
`<basename>|<transcript>` line per voice. A request `"voice": "<basename>"`
clones `<voice_dir>/<name>.wav` and injects its transcript — no
`voice_ref`/`reference_text` from the caller. `GET /v1/audio/voices?model=<id>`
lists cached ids, preset names, and `voice_dir` basenames.

Voice-field precedence for a TTS request (upstream, authoritative):

1. `voice_ref` — always wins.
2. `voice` matching a configured model preset (`default_voice_preset` /
   `voice_presets`).
3. `voice` matching a `voice_dir` wav basename → voice-library clone.
4. Otherwise → model-native cached voice id.

`voice_ref` accepts a plain path string, `{type: path, path: …}`, or
`{type: base64, data: …}` (a `data:` URI is also accepted); **decoded payload is
capped at 5 MiB**, larger refs must use a path. Security: inline base64 is
decoded server-side under the cap. For cloned voices, stage them in `voice_dir`
(or ship a `default_voice_preset`), since PocketTTS 400s if a request supplies
neither a voice nor a ref. Maps to sidecar keys `voice:` (library name) and
`voice_ref:` (path / inline base64) — R10.

### Q7 — Streaming / latency. **Streaming works through llama-swap; no
`proxy`/`checkEndpoint` change, but note SSE + long-inference.**

- TTS streaming: model `mode: "streaming"` + client `stream_format: "sse"` /
  `response_format: "pcm"` → SSE `speech.audio.delta`/`done`.
- ASR streaming: `stream=true` on `/v1/audio/transcriptions` → SSE;
  `/v1/audio/transcriptions/live` ingests raw PCM chunked and returns deltas
  (reverse proxy needs `proxy_request_buffering off` + `proxy_buffering off`).
- `checkEndpoint: /health` (confirmed by `@dkruyt` — **not** `/health`-only;
  `/health` returns readiness + configured-model count). `healthCheckTimeout` is
  startup-only.
- Per-inference bound is server-side `busy_timeout_ms` (music gen can take
  minutes, default 300 000 ms); a busy model returns 503 (`server_busy`) so the
  client retries. This does not touch llama-swap's proxy/checkEndpoint.
- llama-swap already proxies `/v1/audio/speech` (feature #36) and sets
  `X-Accel-Buffering: no`; streaming is reliable when the reverse proxy is
  configured (docs/llama-swap.md reverse-proxy notes).

One real-world gotcha: `@dkruyt` had to **patch audio.cpp for
multipart/form-data** `/v1/audio/transcriptions` (Open WebUI uploads
multipart). Upstream has since added multipart support (current README shows
`-F file=@…`), so a current build is fine — verify the shipped binary is recent.

### Q8 — Build/install. **Source build per backend; unified llama-swap image is
the easy path.**

- **Source (host binary):** GCC 13+, CMake.
  ```bash
  cmake -S . -B build -DENGINE_ENABLE_CUDA=ON   # or ENGINE_ENABLE_VULKAN=ON
  cmake --build build -j$(nproc) --target audiocpp_server
  ```
  CPU is always-on. See `docs/build/linux.md`.
- **AMD/ROCm (R3, RDNA4 box):** the documented `server.json` backend enum is
  `cpu`/`cuda`/`vulkan`/`metal` and the Linux build doc ships `-DENGINE_ENABLE_VULKAN=ON`.
  **The documented portable AMD path is Vulkan** (build with Vulkan, set
  `backend: vulkan`). Release notes mention "early HIP/ROCm support" — treat any
  `ENGINE_ENABLE_ROCM`/HIP flag as experimental and verify against the shipped
  source before relying on it.
- **NVIDIA (GB10 Spark):** CUDA (`ENGINE_ENABLE_CUDA=ON`, `backend: cuda`).
  Pin the toolkit/compiler (`CUDAToolkit_ROOT` + `CMAKE_CUDA_COMPILER`) to avoid
  a silently mixed old-`libcudart` build.
- **Prebuilt:** the **llama-swap unified Docker image**
  (`unified-cuda13`, `unified-cuda`, `unified-vulkan`) builds `audio.cpp` from
  source — the lowest-effort install. No stable standalone
  `audiocpp_server` release tarball was found; prefer the unified image or a
  source build. (Package *downloads* are via `tools/model_manager_v2.py` or
  `audiocpp_model_manager install <family>`; HF repo
  `audio-cpp/audio.cpp-gguf`.)

**Vendor detection** (mirrors `kokoro-podman`): NVIDIA → CUDA + `backend: cuda`;
AMD → Vulkan (+ HIP/ROCm experimental) + `backend: vulkan`; CPU → default build
+ `backend: cpu`; override via profiles.yaml. Binary precedence mirrors
`whisper-server`: `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
`$AUDIOCPP_BIN_DIR` > `PATH`. `extras/update` builds it to CWD like the other
sources (R11).

### Q9 — Matrix semantics. **Audio = fixed-overhead category** (like `emb`/`rnk`/`image`);
brief-use TTS → low evict cost. Depends on C (`matrix-categories.md`).

Audio is a **fixed-overhead** co-load category: it contributes
`model_mib + compute_mib` at its own (fixed) context, and chat solves the
remainder of the VRAM budget — matching the `solve_matrix_ctx` design in
`matrix-categories.md`. Sets such as

```yaml
matrix:
  categories: {audio: {role: t2s}, emb: {role: embeddings}, rnk: {role: rerank}}
  sets:
    rag-audio: "__CHAT_VARS__ & emb & rnk & audio"
```

keep chat + embed + rerank + a TTS model resident together. Because a TTS model
is invoked briefly and torn down, give it a **low `evict_cost`** so the solver
evicts it in preference to chat context. Residency/unload is still expressed via
`matrix` sets (or `ttl`); the per-server LRU (R9) is the shared-server variant,
not the default.

### Q10 — Coexistence. **Additive, opt-in; pick per capability/maturity.**

- **vs `whisper-server`:** task/API overlap is high (both `role: s2t`, both
  OpenAI `/v1/audio/transcriptions`, both ggml + proxyable) but **model overlap
  is zero** — audio.cpp has **no `whisper` family** and only loads its own
  `audiocpp.model_spec` package specs, so existing whisper `.bin`/GGUF models
  cannot run in it. That is precisely why R12 keeps `whisper-server`. audio.cpp
  offers a *different* ASR architecture mix with broader language coverage and
  per-family accuracy (`qwen3_asr`, `nemotron_asr`, `voxtral_realtime`,
  `vibevoice_asr`, `fun_asr_nano`, `citrinet`, `granite5`, plus
   `sortformer_diar` and `qwen3_forced_aligner` for diarization/word timing).
   Capability gap is wider than models: **whisper is transcribe-only** (audio→
   text) and has **no** voice generation, so audio.cpp adds capabilities whisper
   fundamentally cannot provide — voice conversion (`meanvc2`, `rvc`,
   `seed_vc`, `miocodec`, `vevo2`), zero-shot voice cloning (`clon` on ~20 TTS
   families, e.g. `chatterbox`, `cosyvoice3`, `qwen3_tts`, `voxcpm2`), and
   speech-to-speech (`personaplex`). `whisper-server` stays for whisper
   maturity + proven multilingual/word-timing parity; use `audio-cpp` for family
   breadth or a unified TTS+ASR+VC service.
- **vs `kokoro-podman`:** audio.cpp ships a native `kokoro_tts` family
  (Kokoro 82M, 54 preset voices) — it **can** replace `kokoro-podman`. Keep
  `kokoro-podman` if you rely on remsky voicepacks; switch to `audio-cpp` for a
  single native service.
- Capability gaps to watch: whisper.cpp-only features, remsky voicepacks, and
  any family whose GGUF Q8 still drifts (e.g. `supertonic` Q8 is currently
  unsupported upstream). None block an additive, opt-in rollout (R12).

### Q11 — Naming. **Backend `audio-cpp`; binary `audiocpp_server`/`audiocpp_cli`.**

Matches the repo convention (`sd-server`, `whisper-server`, `kokoro-podman`):
engine `audio.cpp` + `-cpp` suffix. Upstream repo `audio.cpp` (primary
`0xShug0/audio.cpp`, HF mirror `ggml-org`); GGUFs on
`audio-cpp/audio.cpp-gguf`.

### Q12 — Model packaging. **Standard GGUF container with an audio.cpp package
spec; own dirs, not the llama-server tree.**

audio.cpp GGUFs are standard GGUF containers but with an embedded package spec
(`audiocpp.model_spec.*`) and family/tensor layout — they are **not**
llama.cpp chat GGUFs. Resolution is deterministic: `--model-spec-override` →
embedded spec → compiled catalog → external `model_specs/<family>.json`. A
directory with a single `model.gguf` (or sole `*.gguf`) resolves without
renaming.

Discovery: put audio GGUFs under the audio-mapped dirs (`t2s/`, `s2t/`, …),
resolve `hf_repo`/HF snapshot like other models (HF repo
`audio-cpp/audio.cpp-gguf`), and route to the `audio-cpp` backend. No distinct
HF tree is required, but the **directory mapping is mandatory** so the files are
never classified/served as chat models.

---

## Interface (validated against `@dkruyt`'s live config, issue #36)

### Backend / transport

- Engine `audio-cpp`; transport `host` (this plan), optionally `podman` later.
- `proxied = True` (like `sd-server`/`whisper-server`).
- Binary precedence: `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
  `$AUDIOCPP_BIN_DIR` > `PATH` (mirrors `whisper-server`).
- Emitted entry: `cmd:` writes a `server.json` (embedding `${PORT}`) and
  `exec`s `audiocpp_server --config …`; `proxy: http://127.0.0.1:${PORT}`;
  `checkEndpoint: /health`. The `cmd:` is a `sh -c '… heredoc … exec …'` block —
  the templating backend renders `${PORT}` and the resolved model path into it.

### Roles / directories (opt-in via `dirs:` + `backends:`)

- `t2s/` → t2s, `s2t/` → s2t; later `vc/`, `vad/`, `music/`, `sep/`.
- Migration: if `audio-cpp` replaces `whisper-server`/`kokoro-podman`, existing
  sidecars keep working (same `s2t`/`t2s` role, same endpoints); document the
  voice-preset/`voice_ref` differences.

### Sidecar keys (name them properly)

```yaml
role: t2s                      # or s2t (later vc)
audio_cpp:
  family: chatterbox           # chatterbox | chatterbox_turbo | qwen3_asr | qwen3_tts | pocket_tts | …
  task: tts                    # tts | clon | vc | asr | vad | align | music | sep | …
  options: { guidance_scale: 0.5, temperature: 0.8, top_p: 0.8 }  # family request options
  voice: jk                    # voice-library name (server-side voice_dir), or
  voice_ref: voices/jk.wav     # path / {type: base64, data: …} for clone/vc
vram_mb: 4096                  # fixed-overhead pin (existing key); = file size + constant
```

### profiles.yaml (working set)

```yaml
audio_cpp:
  bin: audiocpp_server                 # or absolute path
  backend: auto                        # auto | cuda | vulkan | cpu   (vulkan on AMD)
  device: 0
  # shared server knobs (applied to every emitted server.json):
  # max_loaded_models, idle_unload_ms, busy_timeout_ms, min_free_memory_mb, voice_dir
```

### Emitted entry (rendered from the sidecar above)

```yaml
  "chatterbox-tts":
    name: "Chatterbox TTS (audio.cpp)"
    capabilities: { in: [text], out: [audio] }   # *cap_tts
    ttl: 1800
    checkEndpoint: /health
    cmd: >
      sh -c 'cat > /tmp/audiocpp-chatterbox-${PORT}.json <<JSON
      {"host":"127.0.0.1","port":${PORT},"backend":"${AUDIO_CPP_BACKEND}","device":0,"threads":1,
       "models":[{"id":"chatterbox","family":"chatterbox","path":"/opt/models/audiocpp/chatterbox",
                  "task":"tts","mode":"offline",
                  "default_request_options":{"temperature":0.8,"top_p":0.8}}]}
      JSON
      exec /opt/audio.cpp/build/bin/audiocpp_server --config /tmp/audiocpp-chatterbox-${PORT}.json'
    proxy: "http://127.0.0.1:${PORT}"
```

### Composition with B and C

- **B (engine/transport):** the `audio-cpp` backend renders a `host` command; a
  future `podman` transport renders a `podman run … audiocpp_server …` command
  from the same sidecar keys. One engine, two transports.
- **C (matrix):** declare the `audio` category mapping to `t2s`/`s2t` and add it
  to a `sets` expression (see Q9).

---

## Non-goals (for now)

- Rewriting `whisper-server`/`kokoro-podman` in terms of `audio-cpp`.
- A container/podman transport (host process first; leave the seam for B).
- Model training/conversion; we consume published GGUF packages only.

## Implementation checklist (derived from the above)

1. `backends/audio_cpp.py`: `BaseBackend` subclass, `name: "audio-cpp"`,
   `formats = {.gguf, .safetensors, hf_repo}`, `roles = {t2s, s2t}`,
   `proxied = True`, `handles = {audio_cpp, vram_mb}`. Render the `sh -c` heredoc
   `cmd:` from `audio_cpp:` + resolved path + `${PORT}`; emit `proxy` +
   `checkEndpoint: /health`.
2. Vendor detection → `backend:` (`cuda`/`vulkan`/`cpu`) from profiles/override.
3. Discovery: add `t2s`/`s2t` (+ later) to `dir_role_map`; gate the backend by
   role so audio dirs never fall to `llama-server`.
4. VRAM: `FIXED_OVERHEAD_BACKENDS` + `model_mib = file size + buffer`,
   per-family `compute_mib`; honor sidecar `vram_mb`.
5. Matrix: add the `audio` category (depends on `matrix-categories.md`).
6. Tests: `emit_config` output compares against the rendered entry above; a
   unit asserting `proxy`/`checkEndpoint: /health` and the `server.json` shape.
