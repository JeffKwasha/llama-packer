# Plan: Audio backends — whisper-server (Part 1) + kokoro t2s (Part 2)

Status: **Part 1 implemented**; Part 2 (t2s) planned. Recorded 2026-08-25.
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

## Part 2 — kokoro via rootless podman (planned)

* Role `t2s`, dirs `{t2s: t2s}`; capabilities `in: [text], out: [audio]`;
  endpoint `POST /v1/audio/speech` (+ `GET /v1/audio/voices`).
* Backend `kokoro-podman`: rootless **podman** (not docker), image
  `ghcr.io/remsky/kokoro-fastapi-gpu` (OpenAI-compatible).
* GPU pass-through by vendor (from existing hardware detection):
  * NVIDIA: CDI — `--device nvidia.com/gpu=all` (podman ≥4; fallback
    `--gpus all` + `-e NVIDIA_VISIBLE_DEVICES=all`).
  * AMD: known-working CUDA→ROCm translation of the CUDA image;
    `--device /dev/kfd --device /dev/dri --group-add video --group-add render`.
    Document as translation, not native; allow per-model image override.
  * Override hatch: profiles.yaml `t2s.podman_args`.
* Reuse `_map_paths_into` (backends/vllm.py) for model/voices bind mounts;
  container port default 8000 mapped to `${PORT}`.
* Formats: `.onnx` (+ voices), `hf_repo`; fixed VRAM overhead (~512 MiB buffer)
  added to `FIXED_OVERHEAD_BACKENDS`.

## References

* llama-swap audio proxying: `/v1/audio/*` handled natively (playground badges
  `audio_transcriptions` / `audio_speech`)
* whisper.cpp examples/server README (flags: `--model --host --port --parallel
  --language`)
* kokoro-fastapi (remsky): OpenAI-compatible TTS, GPU image, `/v1/audio/speech`
* podman rootless GPU: CDI spec (`nvidia.com/gpu=all`); ROCm requires kfd+dri
  device nodes + video/render groups
