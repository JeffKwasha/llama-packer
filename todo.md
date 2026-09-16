# todo.md — audio.cpp rework (2026-09-16, rev 3 — implementation plan)

**Goal:** ship STT + TTS on audio.cpp (kokoro, chatterbox included) via the
existing **1:1 emission** (one `audiocpp_server` per sidecar, one model per
`server.json`). Each capability is its own llama-swap entry; switching models
reloads audio.cpp completely (accepted). No shared-server grouping.

**Branch:** `feat/audio-cpp-rework` (off `dry_backend` @ d57bf87, which carries
the 1:1 support + plan revs 1-2). Box: audio.cpp **0.8.0** prebuilt
(`/mnt/pool/data/tools/audiocpp`, backends `cpu,vulkan`), AMD R9700 RADV
Vulkan, 5 working GGUFs + chatterbox clon verified end-to-end (see §0).

## Architecture shape (verified against the code)

- `BoundBackend` (`backends/__init__.py:61-97`) binds engine+transport; the
  writer/planner read only `build_cmd`, class attrs, and `unsupported_reason`.
- `build_cmd(model, ctx, parallel, cache_type, tvars, …) → (cmd, meta)`
  (`base.py:124-145`); writer `_build_entry` (`writer.py:290-549`) consumes
  cmd + metadata; proxied entries get `proxy:`/`checkEndpoint:` automatically
  (`writer.py:533-535`). audio-cpp needs **no writer changes** — everything
  ships through `build_cmd`.
- The `audio_cpp:` sidecar block is consumed wholesale (`FIELDS` model.py:578,
  `handles` base.py:51) — **inner keys need no registration**; only new
  *top-level* sidecar keys would.
- tvars layering (`__main__.py:707-712`): `audio_cpp_bin`, `audio_cpp_backend`,
  `audio_cpp_<knob>` per `AUDIO_CPP_SERVER_KNOBS` (only when non-empty in
  profiles.yaml).
- Heredoc cmd depends on `_LiteralDumper` literal blocks (`writer.py:2013`).
  Keep newline shape; covered by golden test (`test_backends.py:1063`).
- VRAM: fixed-overhead via `effective_static` (`vram.py:1157-1180`); sidecar
  `vram_mb:` pin is authoritative. Family-aware compute buffer is a consts.py
  lookup (`vram.py:89-92` keyed by backend name).

## 0. Bring-up status (DONE — verified 2026-09-16)

- Hardlink fix applied: 6 HF-snapshot GGUFs are real files (link count 2).
- `/tmp/audiocpp-gguf` extraction cache: honors `$TMPDIR`; stale dir from
  another uid wedges all loads → document + optional `env` knob (below).
- CLI verified: kokoro `tts` ✅, chatterbox_turbo `tts` ✅, chatterbox `clon`
  ✅ (needs `--voice-ref`; base chatterbox has no `tts` task), parakeet `asr`
  ✅, nemotron `asr` ✅. canary: binary lacks `canary_asr` loader — excluded.
- Server verified: `/tmp/claude/audio/server.json` (5 models, `lazy_load`),
  `/v1/audio/speech` + `/v1/audio/transcriptions` round-trip, RTF 0.0145.
  GUI config installed at `/mnt/pool/data/tools/audiocpp/server.json`.

## 1. Backend selection (audio_cpp.py + __main__.py)

- `_AUDIO_CPP_BACKENDS` add `hip`; accept `rocm` alias → normalize to `hip`
  (upstream CLI + HIP.md). Reject `best` (CLI-only upstream value) → warn + cpu.
- **Precedence sidecar > explicit > auto**: `audio_cpp.py:83-84` currently
  tvars-first. Rework: `cfg["backend"]` > tvars `audio_cpp_backend` (CLI >
  profiles, merged in `__main__`) > tvars `audio_cpp_backend_auto` (vendor
  map) > `cpu`. Device/threads likewise sidecar-first; defaults device 0,
  threads 4 (upstream CLI default).
- New CLI flag `--audio-cpp-backend` (`__main__.py` argparse block); `auto`
  resolution stays vendor-based (NVIDIA→cuda, AMD→vulkan, else cpu).
- **Binary-capability probe** in `__main__` (only when `audio_cpp_bin`
  resolves): run `<bin> --list-devices` once, parse device labels
  (`Vulkan:0`, `CPU:0`, …), store tvars `audio_cpp_bin_backends`; `build_cmd`
  warns when the emitted backend isn't in the probed set (non-fatal; probe
  absent in tests → no warning).

## 2. Sidecar schema (all inside the `audio_cpp:` block)

- `family` (required-ish): known-family default task map
  (`kokoro_tts→tts`, `chatterbox→clon`, `chatterbox_turbo→tts`,
  `qwen3_tts→tts`, `pocket_tts→tts`, `qwen3_asr/parakeet_tdt/
  nemotron_asr→asr`), else role default (`t2s→tts`, `s2t→asr`); warn when
  family+task combination is inconsistent with the loader table
  (chatterbox+tts, kokoro_tts+clon).
- **Voice mapping (the real gap):** sidecar `voice_ref` → model entry
  `default_voice_preset: {voice_ref, reference_text}`; sidecar `voice: <name>`
  → `default_voice_preset: "<name>"` (upstream accepts preset name or
  model-native voice id); sidecar `voice_presets` → verbatim.
  **`voice_dir` relative paths resolve against the config file's dir —
  absolutize when emitting from profiles.**
- `load_options` / `session_options` freeform pass-through (PocketTTS
  `language`; probe/verify per family later).
- `options` → `default_request_options` (existing behavior, keep).
- `model_spec_override` per-model pass-through; also add to
  `AUDIO_CPP_SERVER_KNOBS` (top-level upstream key) for tensor-only/legacy
  GGUFs.
- **`env: TMPDIR`** profiles knob `audio_cpp.tmpdir` → entry `env` block
  (`writer.py:519-522` precedent) — kills the shared-box extraction wedge.
- `lazy_load: true` hardcoded in every emitted `server.json` (1:1 + llama-swap
  load-on-swap semantics).
- Standalone GGUFs embed package spec + sidecars → **no `companions:`/
  `files:` key**; tensor-only GGUFs are operator-managed via
  `model_spec_override`.

## 3. Matrix (no code)

`tts`/`stt` declared categories point at separate 1:1 entries — the existing
fixed-overhead machinery already reserves them (`writer.py:1749-1766`,
tested at `test_solve_matrix_reserves_declared_category`). Nothing to build;
document in docs only.

## 4. Model sourcing (existing, document)

`hf_repo: audio-cpp/audio.cpp-gguf` + `model: <snapshot file>` resolves via
`match_snapshot_paths`/`single_snapshot_hit` (utils.py:1059-1104). Keep
`extras/hardlink-audio.py` as the documented pre-pack step (blob rejection
reproduced on 0.8.0). `path` may be a directory upstream; prefer single file.

## 5. VRAM (measure, then refine)

- `effective_static`: `vram_mb = file-size + _AUDIO_CPP_COMPUTE_MB` (1024).
- **Measurement task:** run the 5 verified models, record actual peak VRAM per
  family (kokoro 0.19 GB file, chatterbox 2.1 GB, turbo 0.7 GB, parakeet/
  nemotron 0.9 GB), then replace the flat 1024 with a per-family table in
  `consts.py` (`_AUDIO_CPP_FAMILY_COMPUTE_MB`, fallback 1024) consulted by
  `vram.py` when `backend in AUDIO_CPP_BACKENDS`. Sidecar `vram_mb:` pin
  remains authoritative.

## 6. Docs / examples / tests

- `docs/backends/audio-cpp.md`: 0.8.0 facts, backend table
  (`cuda|cpu|vulkan|metal|hip`, `rocm` alias, no `best`), voice mapping,
  `load_options`, `model_spec_override`, TMPDIR note, hardlink note, drop
  shared-server hints.
- `profiles.yaml.example`: `audio_cpp:` block (bin/backend/device/threads/
  knobs/tmpdir) + `dirs: {t2s: t2s, s2t: s2t}` + matrix `tts/stt`.
- `models_AGENTS.md`: sidecar examples for kokoro (tts), chatterbox (clon +
  voice_ref), chatterbox_turbo (tts), parakeet/nemotron (asr).
- Tests (extend `tests/test_backends.py` audio section): rocm→hip alias,
  `best`→cpu warn, sidecar-beats-tvars precedence, device/threads precedence,
  voice→`default_voice_preset` (string + object form), `load_options`,
  `model_spec_override`, `lazy_load` present, family-task default map +
  chatterbox+tts warning, probe warning (fake tvars). Existing goldens stay
  green (`test_audio_cpp_cmd_shape`, `test_audio_cpp_task_defaults_by_role`).
- Run: `PYTHONDONTWRITEBYTECODE=1 /var/uv/env/bin14/bin/python -m pytest -q
  -p no:cacheprovider`.

## Verification (per AGENTS.md fast-iterate)

Pack one sidecar → `yaml.safe_load` (real newlines, `${PORT}` unquoted) →
llama-swap launch → `pgrep -af audiocpp_server` → `GET /health` →
`POST /v1/audio/speech` (TTS) and `/v1/audio/transcriptions` (ASR). One model
× one cell, <4 min. Full audio.cpp reload on switch is accepted.

**Live gate PASSED (2026-09-16):** packed config (real profiles, real binary
probe) → mini llama-swap :8124 → kokoro-82m entry launched upstream, health
passed, `/v1/audio/speech` through llama-swap returned 24 kHz PCM WAV.
Found + fixed in the live run: audio.cpp model `id` must equal the llama-swap
entry id (`model.template_id`) — the stem default produced "unknown model id"
at request time. Remaining llama-swap noise: it tries to parse audio bodies
for metrics (`invalid JSON in response body`, harmless, records minimal
metrics).

## Risks

- `/tmp/audiocpp-gguf` extraction-dir collision → `audio_cpp.tmpdir` env knob.
- RADV Vulkan non-conformant: watch kokoro/chatterbox pipeline compile;
  cpu fallback exists.
- Upstream key drift: claims verified against 0.8.0 docs + live binary.
- Chatterbox `clon` latency (2.1 GB Q8 clone model) — acceptable; turbo is the
  zero-shot TTS path.
