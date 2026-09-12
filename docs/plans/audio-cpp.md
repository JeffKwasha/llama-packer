# Plan: `audio-cpp` backend (TTS / ASR / VC via a native ggml engine)

Status: **requirements draft — to be researched and rewritten** (2026-09-11).
This file is a *brief*, not a design: it states what llama-packer needs, the
interface it should expose, and the questions a researcher must answer. Rewrite
freely, but keep the requirement IDs and answer every open question.

Related: [`audio.md`](audio.md) (whisper-server + kokoro-podman — the existing
audio backends this may supersede or complement), [`matrix-categories.md`](matrix-categories.md)
(the `(EMB+RERANK) | (TTS&/or STT)` co-load goal), [`comfyui-sd.md`](comfyui-sd.md)
(the `sd-server` precedent for a non-llama.cpp engine), `SPEC.md` Backend
Selection + vLLM Backend, `docs/architecture.md`.

Seed research (external, provided separately to the researcher — **do not
hardcode its path in repo docs**): a ROCm/RDNA4 build-and-run survey of
`0xShug0/audio.cpp` + Chatterbox, covering binaries, `server.json`, model
sizes, and a TTS-family inventory.

---

## Why

`audio.cpp` is a pure-C++ ggml inference engine for audio models (TTS, ASR, VAD,
VC, music, separation), 60+ model families, all shipped as GGUF. It ships two
binaries: `audiocpp_cli` (one-shot) and `audiocpp_server` (HTTP server with an
OpenAI-compatible `/v1/audio/speech` + `/v1/models` + `/health`, plus
transcriptions and a generic `/v1/tasks/run`).

Today llama-packer serves audio through **two single-purpose backends**:
`whisper-server` (`role: s2t`) and `kokoro-podman` (`role: t2s`, a container).
One native engine covering both — and many more families — would collapse that
surface, and it is the **forcing function** for two pending workstreams:

- **B — engine/transport seam.** `audio-cpp` is another *engine* (like
  sd.cpp/whisper.cpp) that may be served by more than one *transport* (host
  process today, podman later). We should not add yet another flat backend class.
- **C — configurable matrix categories.** Audio needs to become a co-load
  category to express `(CHAT) + ((EMB+RERANK) | (TTS&/or STT))`.

## Requirements

- **R1 — New engine backend.** A backend (working name `audio-cpp`) that runs an
  audio.cpp server process and is managed/proxied by llama-swap the same way
  `sd-server` and `whisper-server` are today (`cmd:` + `proxy` +
  `checkEndpoint`), *or* a documented reason why it cannot be and must run as a
  standalone service instead.
- **R2 — Roles.** At minimum `t2s` (TTS / voice-clone / voice-conversion) and
  `s2t` (ASR). Keep the door open for `vc`, `vad`, `music`, `separation` as
  later roles without a schema rewrite.
- **R3 — Multi-vendor.** CUDA, HIP/ROCm, Vulkan and CPU; auto-select like
  `kokoro-podman` does (vendor detection → binary/flags), with an override. Must
  work on both the NVIDIA Spark (GB10) and AMD RDNA4 boxes.
- **R4 — Model identity.** audio.cpp models are GGUF, so today's
  GGUF→`llama-server` inference would misroute them. The backend must key off
  audio.cpp's **family + task**, not just the file header, and must not let an
  audio GGUF be classified/served as a chat model.
- **R5 — Sidecar expressiveness.** Sidecars must be able to declare family,
  task, per-family options (guidance/temperature/top-p/…), voice/library refs,
  and a fixed VRAM figure — without free-form `cli_args` for the common path.
- **R6 — VRAM sizing.** These are fixed-overhead models (like `s2t`/`image`/
  `t2s` today). Define the estimate (`file size + buffer`, or measured `vram_mb`)
  across families from ~200 MB to ~19 GB.
- **R7 — Matrix participation.** Audio must be declarable as a matrix category
  so RAG and audio can alternate under one chat model (depends on C).
- **R8 — Model discovery.** Locate audio GGUF under the existing models roots,
  infer role from directory, resolve `hf_repo`/snapshot paths like other models.
- **R9 — Multi-model server semantics.** Decide how audio.cpp's own
  `max_loaded_models` / `lazy_load` / `idle_unload_ms` LRU interacts with
  llama-swap's per-entry swap model (one entry per model vs one server with many).
- **R10 — Voice library.** Support a `voice_dir` of reference WAVs (+ prompt
  text) so clients address voices by name, and inline base64 refs where needed.
- **R11 — Install story.** Document how the binary is obtained/built per backend
  (prebuilt release? source build like the seed doc?), consistent with
  `extras/update`’s CWD-install convention.
- **R12 — Fallbacks stay.** `whisper-server` and `kokoro-podman` remain valid;
  `audio-cpp` is additive and opt-in.

## Interface (draft — fill in and justify)

**Backend / transport**
- Engine `audio-cpp`; transport(s) `host` (this plan) and optionally `podman`.
- Binary precedence: `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
  `$AUDIOCPP_BIN_DIR` > `PATH` (mirrors `whisper-server`).
- Emitted entry: `cmd:` runs `audiocpp_server`; `proxy:` points at its port;
  `checkEndpoint:` per the server's health route. Overrides `audiocpp_server`’s
  own `host`/`port`/`backend`/`device` from profiles.yaml + per-model keys.

**Roles / directories** (opt-in via `dirs:` + `backends:`)
- `t2s/` → t2s (TTS/clone), `s2t/` → s2t (ASR); later `vc/`, `vad/`, …
- If audio.cpp replaces a family, document the migration from `whisper-server` /
  `kokoro-podman` without breaking existing sidecars.

**Sidecar keys (working set — name them properly)**
```yaml
role: t2s                      # or s2t
audio_cpp:
  family: chatterbox           # chatterbox | chatterbox_turbo | qwen3_asr | …
  task: clon                   # tts | clon | vc | asr | vad | …
  options: { guidance_scale: 0.5, temperature: 0.8, top_p: 0.8 }
  voice: jk                    # voice-library name (server-side), or
  voice_ref: voices/jk.wav     # path / inline base64 for clone/vc
vram_mb: 4096                  # fixed-overhead pin (existing key)
```

**profiles.yaml (working set)**
```yaml
audio_cpp:
  bin: audiocpp_server
  backend: auto                # auto | cuda | hip | vulkan | cpu
  device: 0
  # server-mode knobs if we adopt server.json: max_loaded_models, idle_unload_ms
```

**Composition with B and C**
- B: show exactly where the engine/transport split lives (backend ABC + a
  transport object), so `audio-cpp host` and `audio-cpp podman` are one engine.
- C: show the category declaration and the resulting matrix `sets` expression.

## Open questions (answer each; cite upstream)

1. **Proxy vs standalone.** llama-swap already proxies arbitrary `cmd:` backends
   (`sd-server`, `whisper-server`, `kokoro-podman`). The seed doc claims
   audio.cpp *cannot* be hosted by llama-swap and must run as a separate service.
   Is that true? If audio.cpp ships as a normal process listening on a port, why
   can’t llama-swap proxy it like sd-server? Resolve this first — it decides the
   whole interface.
2. **One entry vs many.** `audiocpp_server` loads multiple models with LRU
   (`max_loaded_models`, `idle_unload_ms`), which overlaps llama-swap’s swap
   model. Which wins? One llama-swap entry per sidecar (consistent with the rest
   of the config), or one entry per server with internal routing?
3. **Role granularity.** Do `vc`/`vad`/`music`/`separation` become new roles, or
   sub-tasks of `t2s`/`s2t`? What does `capabilities.in/out` look like for each
   (`vc`: audio→audio; `music`: text→audio; etc.)?
4. **Classification.** How do we distinguish an audio GGUF from an LLM GGUF
   reliably? Header `general.architecture` values? A family allowlist? What
   about `.bin`/other formats?
5. **Sizing.** Per-family VRAM model — file size + constant? Any published
   numbers (KV/state, decoder, vocoder)? How does `max_loaded_models` multiply
   the budget, and how does that interact with the matrix solve?
6. **Voice library.** Exact `voice_dir` layout + `prompt_text` format; path vs
   base64 precedence; where cloned voices live on disk; security of inline refs.
7. **Streaming / latency.** Does `/v1/audio/speech` stream, and does that change
   `proxy`/`checkEndpoint` or timeouts (`healthCheckTimeout`)?
8. **Build/install.** Prebuilt releases? If source-only, what does the
   per-vendor build look like, and can `extras/update` (or a sibling script)
   drive it — given `extras/update` now installs to CWD?
9. **Matrix semantics.** For `(CHAT) + ((EMB+RERANK) | (TTS&/or STT))`: is audio
   a *fixed-overhead category* (like emb/rnk today) or does it need its own
   context solve? How do evict costs and residency interplay with a TTS model
   that is only used briefly?
10. **Coexistence.** When should an operator pick `audio-cpp` over
    `whisper-server`/`kokoro-podman`? Any capability gaps (e.g. whisper.cpp
    features, Kokoro voicepacks) that block a full replacement?
11. **Naming.** `audio-cpp` vs `audiocpp` vs `audio.cpp` — match the upstream
    binary names and the repo’s backend-naming convention (`sd-server`,
    `whisper-server`).
12. **Model packaging.** Are audio.cpp GGUFs standard GGUF that our HF snapshot
    resolution + stub logic can discover, or do they need a distinct tree/dir?

## Non-goals (for now)

- Rewriting `whisper-server`/`kokoro-podman` in terms of `audio-cpp`.
- A container/podman transport (host process first; leave the seam for B).
- Model training/conversion; we consume published GGUF packages only.
