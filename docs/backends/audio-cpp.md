# Backend: audio-cpp

Text-to-speech (`t2s`) and speech-to-text (`s2t`) via the native ggml
[audio.cpp](https://github.com/0xShug0/audio.cpp) engine (`audiocpp_server`).

- **Engine:** audio.cpp · **Transports:** host (`audio-cpp`); podman later.
- **Roles:** `t2s`, `s2t`.
- **Formats:** `.gguf`, `.safetensors`, `hf_repo`.
- **Proxied:** yes — `proxy` + `checkEndpoint: /health`.
- **Binary:** `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
  `$AUDIOCPP_BIN_DIR` > `audiocpp_server` on `PATH`.

audio.cpp is a pure-C++ ggml engine for audio models — TTS, ASR, VAD, voice
conversion, music, separation — spanning 62 families / 85+ variants. It
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
  family: chatterbox           # chatterbox | qwen3_asr | qwen3_tts | pocket_tts | …
  task: tts                    # tts | clon | vc | asr | vad | align | music | sep | …
  options: { guidance_scale: 0.5, temperature: 0.8, top_p: 0.8 }
  voice: jk                    # voice-library name (server voice_dir), or
  voice_ref: voices/jk.wav     # path / {type: base64, data: …} for clone/vc
vram_mb: 4096                  # fixed-overhead pin (existing key)
```

## profiles.yaml

```yaml
audio_cpp:
  bin: audiocpp_server         # or absolute path
  backend: auto                # auto | cuda | vulkan | cpu   (auto: NVIDIA→cuda, AMD→vulkan, else cpu)
  device: 0
  # applied to every emitted server.json:
  # max_loaded_models, idle_unload_ms, busy_timeout_ms, min_free_memory_mb, voice_dir
```

## Emitted entry

One `audiocpp_server` process per sidecar (1:1) — llama-packer's
"one entry = one process with a fixed command line" invariant. The `cmd:` is a
`sh -c` heredoc that writes a `server.json` (the port is llama-swap's
`${PORT}`) and execs the binary:

```yaml
"chatterbox-tts":
  capabilities: { in: [text], out: [audio] }
  checkEndpoint: /health
  cmd: >
    sh -c 'cat > /tmp/audiocpp-chatterbox-${PORT}.json <<JSON
    {"host":"127.0.0.1","port":${PORT},"backend":"cuda","device":0,"threads":1,
     "models":[{"id":"chatterbox","family":"chatterbox","path":"…","task":"tts","mode":"offline",
                "default_request_options":{"temperature":0.8,"top_p":0.8}}]}
    JSON
    exec /opt/audiocpp_server --config /tmp/audiocpp-chatterbox-${PORT}.json'
  proxy: "http://127.0.0.1:${PORT}"
```

The server's own multi-model LRU (`max_loaded_models`, `/v1/tasks/unload_models`)
is available for a shared-server deployment, but is **not** the default: the
1:1 entry matches the whisper-server/sd-server precedent and leaves residency
to llama-swap's matrix.

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

API-level notes (not yet wired into `extras/update`):

- **Source:** GCC 13+, CMake. `-DENGINE_ENABLE_CUDA=ON` (NVIDIA),
  `-DENGINE_ENABLE_VULKAN=ON` (portable AMD), CPU always on; build the
  `audiocpp_server` target.
- **NVIDIA/Spark:** CUDA, pin `CUDAToolkit_ROOT` + `CMAKE_CUDA_COMPILER`.
- **AMD/RDNA4:** the documented portable path is **Vulkan**; HIP/ROCm is
  experimental.
- **Prebuilt:** the llama-swap unified image (`unified-cuda13`, `unified-cuda`,
  `unified-vulkan`) builds audio.cpp from source; no stable standalone
  `audiocpp_server` tarball was found.

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
