# Plan: Audio backends — whisper-server (Part 1) + kokoro t2s (Part 2)

Status: **both implemented**. Recorded 2026-08-25.
Related: `docs/plans/comfyui-sd.md` (the sd-server precedent this mirrors).

## Decisions

* **Dirs:** `s2t/` (speech→text) and later `t2s/` (text→speech). Symmetric with
  the legacy `t2t`→chat naming. `/mnt/ai/models/whisper` renames to `s2t`.
  Both roles opt-in via `dirs:` like `img: image`.
* **Server:** `whisper-server` (whisper.cpp `examples/server`) — long-lived HTTP
  like `llama-server`; `whisper-cli` is oneshot and cannot be proxied.
  Exposes OpenAI `POST /v1/audio/transcriptions`. `checkEndpoint: /`
  (`/health` pitfall, Discussion #866).
* **No `.bin` stubs:** GGML has no header fingerprint; few models, low churn.
  A `.bin` is served only via an authored same-stem sidecar inside an s2t dir;
  bare orphans log one info line and are skipped. `.bin` outside s2t fails
  backend inference ("no available backend supports format '.bin'").
* **VRAM:** fixed overhead like sd-server — `model_mib = file-size`,
  `ctx_factor = 0`, `compute_mib = 512`; excluded from the chat matrix
  (`FIXED_OVERHEAD_BACKENDS`). `--parallel` maps slots → concurrent workers.

## Part 1 (implemented)

`backends/whisper_server.py`; binary precedence `--whisper-server` >
`whisper.bin` in profiles.yaml > `$WHISPER_BIN_DIR` > PATH. Capabilities:
`in: [audio], out: [text]`. `proxied: ClassVar[bool]` on BaseBackend replaces
the sd-server name-match for proxy/checkEndpoint emission. `utils.NON_CHAT_ROLES`
consolidates the scattered role-exclusion tuple.

## Part 2 — kokoro via rootless podman (implemented)

* Role `t2s`, dirs `{t2s: t2s}`; capabilities `in: [text], out: [audio]`;
  endpoint `POST /v1/audio/speech` (+ `GET /v1/audio/voices`, health `/`,
  container port 8880).
* Backend `kokoro-podman`: rootless **podman**, upstream images from
  remsky/Kokoro-FastAPI. Correction found during research: a **native ROCm
  image exists** (`kokoro-fastapi-rocm`) — no CUDA→ROCm translation needed.
  Vendor detection picks tag + flags:
  * NVIDIA → `kokoro-fastapi-gpu` + `--device nvidia.com/gpu=all` (CDI)
  * AMD → `kokoro-fastapi-rocm` + `--device /dev/kfd --device /dev/dri
    --group-add video --group-add render`
  * CPU → `kokoro-fastapi-cpu`, no flags
* Overrides: CLI `--kokoro-image` > profiles.yaml `t2s.image` > vendor default;
  `t2s.vendor:` overrides detection for tag+flags; `t2s.podman_args` replaces
  auto flags; `t2s.voices_dir` rw-mounts persistent voicepacks at
  `/app/api/src/voices/v1_0` (verified against upstream paths.py).
* Weights are baked into the image — sidecars are typically `hf_repo:`-only;
  formats include `.onnx` for local copies (stem resolution extended).
* VRAM correction: PyTorch runtime floors ~2.4 GiB / peaks ~4 GiB, so fixed
  compute buffer is 3072 MiB (`_KOKORO_COMPUTE_MB`), not the sd/whisper 512.
* No host-binary variant shipped: no standalone binary exists upstream (the
  uv-run path is a source checkout needing espeak-ng); a future variant would
  follow the vLLM two-thin-classes-over-shared-helpers pattern.

## References

* llama-swap audio proxying: `/v1/audio/*` handled natively (playground badges
  `audio_transcriptions` / `audio_speech`)
* whisper.cpp examples/server README (flags: `--model --host --port --parallel
  --language`)
* kokoro-fastapi (remsky): OpenAI-compatible TTS, GPU image, `/v1/audio/speech`
* podman rootless GPU: CDI spec (`nvidia.com/gpu=all`); ROCm requires kfd+dri
  device nodes + video/render groups
