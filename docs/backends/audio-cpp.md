# Backend: audio-cpp

Text-to-speech (`t2s`) and speech-to-text (`s2t`) via the native ggml
[audio.cpp](https://github.com/0xShug0/audio.cpp) engine (`audiocpp_server`).

- **Engine:** audio.cpp · **Transports:** host (`audio-cpp`); a container transport is future/optional.
- **Roles:** `t2s`, `s2t`.
- **Formats:** `.gguf`, `.safetensors`, `hf_repo`.
- **Proxied:** yes — `proxy` + `checkEndpoint: /health`.
- **Binary:** `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
  `$AUDIOCPP_BIN_DIR` > `audiocpp_server` on `PATH`.

audio.cpp is a pure-C++ ggml engine for audio models — TTS, ASR, VAD, voice
conversion, music, separation — spanning 80+ families / 120+ variants
(0.8.0, verified 2026-09-16). It
**replaces `kokoro-podman`** (its native `kokoro_tts` family supersedes the
containerized Kokoro-FastAPI backend).

## Roles

audio.cpp `task` values map onto llama-packer roles:

| audio.cpp tasks | role | `capabilities.in` → `out` |
|---|---|---|
| `tts`, `clon`, `music`, `sfx`, `dialogue`, `edit`, `design`, `ctrl` | `t2s` | text → audio |
| `asr` | `s2t` | audio → text |
| `vad`, `align`, `diar` | `s2t` | audio → text (metadata) |
| `vc`, `s2s` | `vc` (later) | audio → audio |
| `sep` | (later) | audio → audio × tracks |

Shipped now: `t2s` + `s2t`. Additional roles are recorded in
[`docs/plans/audio-roles.md`](../plans/audio-roles.md); they are schema-stable
because `capabilities.in/out` are freeform lists.

## Sidecar

```yaml
role: t2s                      # or s2t
audio_cpp:
  family: kokoro_tts           # required — audio.cpp resolves package specs by family
  task: tts                    # default per family (see below), else by role
  options: { guidance_scale: 0.5, temperature: 0.8, top_p: 0.8 }   # → default_request_options
  load_options: { language: english }        # load-time (e.g. pocket_tts language)
  session_options: { language: english }     # session-time pass-through
  voice: af_heart              # preset name / voice_dir wav / model-native voice id
  voice_ref: voices/jk.wav     # path or {type: base64, data} — wins over `voice`
  reference_text: transcript   # only with voice_ref (clone reference transcript)
  voice_presets: {narrator: {voice_id: alba}}
  model_spec_override: /specs  # tensor-only/legacy GGUFs (standalone GGUFs need nothing)
  backend: cpu                 # per-model pin (wins over CLI/profiles/auto)
  device: 0
  threads: 4
vram_mb: 4096                  # fixed-overhead pin (existing key)
```

`voice`/`voice_ref` map onto upstream `default_voice_preset`:
`voice_ref` (path or `{type: base64, data}`) emits the object form (with
`reference_text`); `voice` emits the string form (upstream resolves a
configured preset name, a `voice_dir` wav basename, or the model-native
cached voice id). `voice_ref` wins, matching upstream precedence.

Default `task` per verified family: `kokoro_tts→tts`, `chatterbox→clon`
(base chatterbox is clone-only — no zero-shot `tts`; use
`chatterbox_turbo→tts`), `qwen3_tts→tts`, `pocket_tts→tts`,
`qwen3_asr→asr`, `parakeet_tdt→asr`, `nemotron_asr→asr`; unknown families
fall back to the role default (`t2s→tts`, `s2t→asr`) with a warning.

## profiles.yaml

```yaml
audio_cpp:
  bin: audiocpp_server         # or absolute path
  backend: auto                # auto | cuda | vulkan | cpu | metal | hip (rocm alias); auto: NVIDIA→cuda, AMD→vulkan, else cpu
  device: 0
  tmpdir: /tmp/audiocpp-llama-swap   # optional: shields /tmp/audiocpp-gguf extraction cache from other-uid wedges (emitted as TMPDIR=… in the entry env)
  # applied to every emitted server.json:
  # max_loaded_models, idle_unload_ms, busy_timeout_ms, min_free_memory_mb, voice_dir (absolutized), model_spec_override
```

Precedence for the backend: sidecar `audio_cpp.backend` > `--audio-cpp-backend`
CLI > profiles.yaml `audio_cpp.backend` > vendor auto > cpu. The packer probes
the binary's compiled backends once (`--list-devices`) and warns when an
emitted backend is missing from the probe (the ubuntu-x64 prebuilt reports
`cpu,vulkan` only; `cuda`/`hip` need a matching build).

## Emitted entry

One `audiocpp_server` process per sidecar (1:1) — llama-packer's
"one entry = one process with a fixed command line" invariant; switching
models reloads audio.cpp completely (accepted). The `cmd:` is a `sh -c`
heredoc that writes a `server.json` (the port is llama-swap's `${PORT}`) and
execs the binary:

```yaml
"kokoro-82m":
  capabilities: { in: [text], out: [audio] }
  checkEndpoint: /health
  cmd: |
    sh -c 'mkdir -p /tmp/llama-swap && cat > /tmp/llama-swap/audiocpp-kokoro-${PORT}.json <<JSON
    {"host":"127.0.0.1","port":${PORT},"backend":"vulkan","lazy_load":true,"device":0,"threads":4,
     "models":[{"id":"kokoro-82m","family":"kokoro_tts","path":"…","task":"tts","mode":"offline"}]}
    JSON
    exec /opt/audiocpp_server --config /tmp/llama-swap/audiocpp-kokoro-${PORT}.json'
  proxy: "http://127.0.0.1:${PORT}"
```

`lazy_load: true` is always emitted: the model id registers at server start
and the framework load happens on the first request, matching llama-swap's
load-on-swap semantics. `voice_dir` from profiles.yaml is absolutized
(upstream resolves relative paths against the config file's directory,
`/tmp/llama-swap/`). A profiles.yaml `audio_cpp.tmpdir` is emitted as
`exec env TMPDIR=…` to shield the engine's sidecar-extraction cache
(`/tmp/audiocpp-gguf`) from other-uid ownership wedges on shared boxes.

The heredoc newlines are load-bearing: the writer emits multi-line `cmd`
values as YAML literal blocks (`|`), never folded, so the round-trip is
exact. The per-port `server.json` is rewritten on every model load under
`/tmp/llama-swap/` (flat; `mkdir -p` in the `cmd` creates it).

Standalone GGUFs (the `audio-cpp/audio.cpp-gguf` packages) embed the package
spec and every required sidecar, so no companion layout is needed;
tensor-only/legacy GGUFs are handled with `model_spec_override`. HF snapshot
GGUFs must be **hard links**, not symlinks into extensionless blobs — the
engine resolves the path and rejects extensionless targets with
`unsupported tensor source format` (use `extras/hardlink-audio.py`).

## Classification

Two layers keep audio GGUFs off the chat path and whisper models out of
audio.cpp:

1. **Directory authority** — audio models live under audio-mapped dirs
   (`t2s/`, `s2t/`, …; opt-in via `dirs:`). A `.gguf` in `chat/` is never an
   audio model.
2. **Role + formats** — audio.cpp serves only `t2s`/`s2t` for
   `.gguf`/`.safetensors`/`hf_repo`, so llama.cpp's `.gguf` chat selection
   cannot claim them. whisper `.bin` models are not in its formats, so they
   stay on [`whisper-server`](whisper-server.md).

## VRAM

Fixed overhead: `model_mib = file size + buffer`, `compute_mib` a
family-dependent constant (`_AUDIO_CPP_COMPUTE_MB`, default 1024 MB), so
`vram_mb = model_mib + compute_mib`. The range spans ~200 MB (pocket_tts) to
~19 GB (7B F32). Sidecar `vram_mb:` pins the exact figure.

## Matrix

Declare audio as co-load categories ([`matrix:`](../../profiles.yaml.example)):

```yaml
matrix:
  categories: {tts: {role: t2s}, stt: {role: s2t}}
  evict_costs: {tts: 50, stt: 50}     # brief use: evict before chat context
  sets:
    rag:   "__CHAT_VARS__ & emb & rnk"
    voice: "__CHAT_VARS__ & (tts | stt)"
```

`tts` and `stt` are separate categories because switching between them forces
a resident reload. Declared categories are reserved as fixed-overhead residents
alongside chat and RAG while the chat context stays at/above the co-load floor.

## Build / install

**llama-packer does not build audio.cpp** — obtaining the binary is out of
scope, as with the other engines (`extras/update` only downloads llama.cpp and
llama-swap). Point the backend at an `audiocpp_server` you built or obtained;
resolution is `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
`$AUDIOCPP_BIN_DIR` > `audiocpp_server` on `PATH`.

- **Source build (operator-managed):** upstream `docs/build/linux.md` — GCC 13+,
  CMake; `-DENGINE_ENABLE_CUDA=ON` (NVIDIA), `-DENGINE_ENABLE_VULKAN=ON`
  (portable AMD path), `-DENGINE_ENABLE_HIP=ON` (ROCm, mutually exclusive
  with CUDA), CPU always on. On GB10, pin `CUDAToolkit_ROOT` +
  `CMAKE_CUDA_COMPILER` to avoid a mixed-toolkit build.
- **Prebuilt:** the llama-swap unified image (`unified-cuda13`, `unified-cuda`,
  `unified-vulkan`) bundles it; official ubuntu-x64 packages ship CPU+Vulkan
  (the 0.8.0 one on this box reports `cpu,vulkan`).

A container transport (podman/docker) may be added later; it is not part of the
current host-only support.

## vs whisper.cpp

audio.cpp and whisper.cpp overlap in **task** (`s2t`, OpenAI transcriptions,
ggml, proxyable) but not in **models**: audio.cpp has no whisper family and
cannot load GGML `.bin`. whisper is transcribe-only; audio.cpp adds voice
generation (TTS, zero-shot cloning, voice conversion, speech-to-speech)
whisper fundamentally cannot provide. Keep `whisper-server` for whisper
maturity; use `audio-cpp` for breadth or a unified TTS+ASR service.

## History / sources

The research brief this landed from is preserved in git history
(`docs/plans/audio-cpp.md`, removed after migration): upstream
`app/server/README.md`, `docs/gguf.md`, `docs/build/linux.md`, and the working
config in llama-swap issue #36 (`@dkruyt`, 2026-07-03).

**2026-09-16 rework (0.8.0):** all emitted-key claims verified against the
0.8.0 server README + live binary (AMD R9700 / RADV Vulkan): `hip`/`rocm`
backend, `lazy_load`, `default_voice_preset`/`voice_presets`,
`load_options`/`session_options`, `model_spec_override`, embedded package
specs in standalone GGUFs, symlinked-blob rejection. Live-verified families:
kokoro `tts`, chatterbox_turbo `tts`, chatterbox `clon`, parakeet_tdt `asr`,
nemotron_asr `asr`.
