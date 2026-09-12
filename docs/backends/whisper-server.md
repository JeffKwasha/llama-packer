# Backend: whisper-server

Speech-to-text via whisper.cpp's `examples/server` (the long-lived analog of
`llama-server`; `whisper-cli` is one-shot and cannot be proxied).

- **Engine:** whisper.cpp · **Transports:** host (`whisper-server`).
- **Roles:** `s2t`.
- **Formats:** `.bin` (GGML). There is no header fingerprint for GGML, so the
  directory is authoritative.
- **Proxied:** yes — `proxy` + `checkEndpoint: "/"`.
- **Binary:** `--whisper-server` > `profiles.yaml whisper.bin` >
  `$WHISPER_BIN_DIR` > `whisper-server` on `PATH`.
- **Opt-in:** `dirs: {s2t: s2t}`; each model needs an authored same-stem
  sidecar (e.g. `ggml-large-v3.md` beside `ggml-large-v3.bin`). Exposes
  OpenAI-compatible `POST /v1/audio/transcriptions`.

## Command shape

```
<whisper_bin> --host 0.0.0.0 --port ${PORT} --model <file.bin> --parallel <N>
  [--language en ...]        # whisper.args; per-model cli_args wins per flag
```

`parallel` maps to concurrent transcription workers.

## profiles.yaml

- `whisper: {bin, args}` — `args` are fleet-wide flags (e.g. `--flash-attn on`).

## VRAM

Fixed overhead — weights + `_WHISPER_COMPUTE_MB` — excluded from the shared
chat matrix. Sidecar `vram_mb:` pin overrides the estimate.

## Relationship to audio-cpp

[`audio-cpp`](audio-cpp.md) is an ASR engine too, but **model overlap is zero**:
audio.cpp has no whisper family and only loads its own `audiocpp.model_spec`
packages, so GGML `.bin` whisper models cannot run on it. `whisper-server`
stays for whisper maturity (multilingual parity, word timing, remsky-era
workflows); use `audio-cpp` for family breadth (`qwen3_asr`, `nemotron_asr`,
`voxtral_realtime`, …) or a unified TTS+ASR service.

## See also

- [`SPEC.md` → Audio Backend (whisper-server)](../../SPEC.md#audio-backend-whisper-server)
- [Backend: audio-cpp](audio-cpp.md)
