# todo.md — audio.cpp plan (2026-09-16, rev 2)

**Goal:** ship STT + TTS on audio.cpp — kokoro + chatterbox included — using the
**existing 1:1 emission** (one `audiocpp_server` per sidecar, one model per
`server.json`). Per decision 2026-09-16: it is OK that each capability is its
own entry in llama-swap's model list, and OK that switching models reloads
audio.cpp completely (llama-swap stop/start). No shared-server grouping, no
in-process LRU reliance; `max_loaded_models`/`idle_unload_ms` are not emitted
in 1:1 mode.

**Branch:** audio 1:1 support lives on `dry_backend` (currently checked out,
with uncommitted modifications). Commit/stash that work first; then a small
follow-up branch (or direct commits on `dry_backend`) carries the deltas below.

**Box facts (2026-09-16, verified):** binary is **audio.cpp 0.8.0**
(`/mnt/pool/data/tools/audiocpp`, git 4af1432, prebuilt ubuntu-x64-vulkan →
backends **cpu + vulkan only**); GPU = AMD Radeon AI Pro R9700 (RADV GFX1201,
RDNA4) via Vulkan; ROCm present but binary has no HIP build. Pre-built GGUF
packages already downloaded via HF repo `audio-cpp/audio.cpp-gguf` snapshot
`1b13cd58`: Chatterbox-GGUF, Chatterbox-Turbo-GGUF, Kokoro-82M-GGUF,
Parakeet-TDT-0.6B-v3-GGUF, Nemotron-3.5-ASR-Streaming-0.6B-GGUF,
Canary-180M-Flash-GGUF.

**Upstream source of truth (read 2026-09-16, verified):**
`app/server/README.md` (config keys: `host/port/backend/device/threads/
lazy_load/max_loaded_models/idle_unload_ms/min_free_memory_mb/busy_timeout_ms/
voice_dir/max_request_body_bytes`; per-model: `id/family/path/task/mode/
load_options/session_options/default_request_options/default_voice_preset/
voice_presets/lazy/busy_timeout_ms/model_spec_override`;
endpoints `/health`, `/v1/models`, `/v1/audio/speech`,
`/v1/audio/transcriptions[/details|/live]`, `/v1/audio/alignments`,
`/v1/tasks/run`, `/v1/tasks/unload_models`), `docs/gguf.md` (standalone GGUFs
embed package spec + all sidecars → **no companion files needed**;
`model_spec_override` is the escape hatch for tensor-only/legacy GGUFs;
`path` may be a directory → resolves `model.gguf`), `docs/build/HIP.md`
(`hip`/`rocm` CLI alias; server JSON accepts `"backend": "hip"` — not usable on
this vulkan-only binary).

## 0. Bring-up: make audio.cpp work at all (DONE/in progress)

- **Root cause of "empty UI":** `models/<X>-GGUF/` are symlinks into the HF
  snapshot, whose GGUFs are symlinks to extensionless blobs → engine resolves
  the path, sees no `.gguf` extension → `unsupported tensor source format`
  (reproduced on 0.8.0 CLI; same as `extras/hardlink-audio.py` docstring).
- **Fix (runs as the cache owner; claude cannot write it):** convert snapshot
  GGUF symlinks → hard links (same mergerfs branch as blob, zero duplication,
  `hf cache rm` still frees):
  ```bash
  snap=/mnt/ai/huggingface/hub/models--audio-cpp--audio.cpp-gguf/snapshots/1b13cd58245c74e3ff4ca06925766c5ef7991bd4
  for f in "$snap"/*-GGUF/*.gguf; do
    [ -L "$f" ] || continue
    t=$(readlink -f "$f"); ln "$t" "$f.tmp" && mv -T "$f.tmp" "$f"
  done
  ```
- Then smoke-test CLI, smallest first (<4 min each): kokoro (`task tts`),
  chatterbox (`task clon` — loader registers `clon`+`vc`, not `tts`),
  chatterbox_turbo (`task tts`), parakeet (`task asr`), nemotron (`task asr`).
  `--backend vulkan` (R9700), `--out` to a claude-writable dir.
- Then server: hand-written `server.json` with the same 4-6 models, run
  `audiocpp_server --config`, `curl /health` (returns configured-model count),
  `/v1/models`, `POST /v1/audio/speech`, `POST /v1/audio/transcriptions` with a
  local wav (multipart). This is the config llama-packer must reproduce.

**§0 results (2026-09-16, all verified on the R9700 / RADV Vulkan):**
- Hardlink fix applied by jk — all 6 snapshot GGUFs are link-count-2 real files.
- **Gotcha:** the engine extracts embedded sidecars to `/tmp/audiocpp-gguf/`
  (hardcoded name, honors `$TMPDIR`). A dir owned by another user wedges the
  run (`Permission denied [/tmp/audiocpp-gguf/<hash>]`) — on this box run with
  `TMPDIR` set, or remove the stale `/tmp/audiocpp-gguf` (owned by `hermes`).
- CLI: kokoro `tts` ✅, chatterbox_turbo `tts` ✅, chatterbox `clon` ✅
  (requires `--voice-ref`), parakeet `asr` ✅, nemotron `asr` ✅ (word
  timestamps + clean transcript). Chatterbox base has **no `tts` task** —
  confirmed.
- Server (`/tmp/claude/audio/server.json`: `lazy_load: true`, 3 models):
  `/health` ok → `/v1/audio/speech` (kokoro) 6.9 s incl. lazy load, 24 kHz PCM
  → `/v1/audio/transcriptions` (parakeet, multipart) round-trips the TTS
  output, RTF 0.0145 after load. Working config shape confirmed.

## 1. Backend selection (`vulkan/cpu` now; `cuda/hip/metal` accepted)

- `llama_packer/backends/audio_cpp.py:42` — `_AUDIO_CPP_BACKENDS` already
  `{cuda, vulkan, cpu, metal}`; add `hip` (accept `rocm` alias → normalize);
  reject `best` (CLI-only value, not a server-JSON value).
- `llama_packer/__main__.py:600-603` — `auto` map (NVIDIA→cuda, AMD→vulkan,
  else cpu) is correct for this box (AMD R9700 → vulkan). Add
  `--audio-cpp-backend` CLI override; precedence **sidecar > CLI >
  profiles.yaml `audio_cpp.backend` > auto** (reverses today's tvars-first
  behavior — needs a test).
- Binary-capability gate: probe the binary's backend set (e.g. `--list-devices`
  output or version banner) and warn when an emitted backend isn't in it
  (0.8.0 prebuilt = cpu,vulkan; cuda/hip emission would fail at load).

## 2. Sidecar schema fixes (existing 1:1 path)

Current gaps in `build_cmd` (`audio_cpp.py:64-130`):

- `voice`/`voice_ref` are declared in docs (`audio-cpp.md:42-43`) but **never
  emitted**. Upstream config-side keys are per-model `default_voice_preset`
  (`{voice_ref, reference_text}`) and `voice_presets`, plus server-level
  `voice_dir`. Map: sidecar `voice_ref` → `default_voice_preset`; `voice` →
  preset name. **`voice_dir` relative paths resolve against the config file's
  dir (`/tmp/llama-swap/`) — absolutize.**
- Add `load_options` / `session_options` freeform pass-throughs (PocketTTS
  needs `load_options: {language}`; chatterbox/kokoro don't for preset-voice
  TTS).
- `family` required, warn+skip if unknown to the loader list; per-family
  default task (`kokoro_tts→tts`, `chatterbox→clon`, `chatterbox_turbo→tts`,
  `qwen3_asr/parakeet_tdt/nemotron_asr→asr`).
- `mode` default `offline` (audio_cpp.py:99) — fine; note streaming is
  buffered SSE, not live capture, for these families.
- Add `model_spec_override` pass-through (top-level or per-model) for
  tensor-only/legacy GGUFs; standalone GGUFs from `audio.cpp-gguf` embed spec +
  sidecars, so no `companions:`/`files:` sidecar key — that plan item is
  dropped.

## 3. Model sourcing (simplified)

- `hf_repo: audio-cpp/audio.cpp-gguf` + `model: <snapshot-file>` — resolution
  via `model.py:_resolve_gguf_path` (model.py:823) exists.
- Keep `extras/hardlink-audio.py` as the documented pre-pack step (symlink→
  extensionless blob rejection); optionally wire it into `discover.py` behind
  a flag. It must also handle the "GGUF dir is itself an HF-snapshot symlink"
  layout seen today (snapshot dir containing the *-GGUF dirs).
- Directory `path` is valid upstream (resolves `model.gguf`); prefer
  single-file snapshot GGUF for llama-packer.

## 4. VRAM + health (unchanged, verified)

- `vram.py:92`, `_AUDIO_CPP_COMPUTE_MB=1024` (`consts.py:152`); measured
  reality: kokoro ≈ 190 MB file → ~1.1 GB VRAM, chatterbox q8 2.1 GB file →
  ~3.1 GB, parakeet/nemotron 0.9 GB → ~2 GB (per rdna4 plan table). Fine-tune
  per-family buffer only if measurement disagrees.
- `checkEndpoint: /health` (returns readiness + configured-model count — works
  for 1:1), `proxy: http://127.0.0.1:${PORT}` unchanged.

## 5. Docs / examples / tests

- `docs/backends/audio-cpp.md`: update engine facts (0.8.0, 80+ families),
  backend table (`cuda|cpu|vulkan|metal|hip` in server JSON, `rocm` alias,
  no `best`), voice mapping (`default_voice_preset`/`voice_presets`/`voice_dir`
  absolutized), `load_options`, `model_spec_override`, hardlink note, remove
  shared-server hints (§"one entry = one process" stays the invariant).
- `profiles.yaml.example`: audio_cpp block + `dirs: {t2s: t2s, s2t: s2t}` +
  uncommented matrix `tts/stt` example.
- `llama_packer/templates/models_AGENTS.md`: copy-paste sidecars for kokoro,
  chatterbox (clon), chatterbox_turbo (tts), parakeet/nemotron (asr).
- Tests: backend enum (`hip`/`rocm` alias, `best` rejected, bad→cpu warn),
  backend-capability probe, build_cmd golden with `default_voice_preset` +
  `load_options` + absolutized `voice_dir`, precedence test (sidecar > CLI >
  profiles), port sentinel unquoted / `${PORT}` intact. Run:
  `PYTHONDONTWRITEBYTECODE=1 /var/uv/env/bin14/bin/python -m pytest -q -p no:cacheprovider`.

## Verification (per AGENTS.md fast-iterate)

One model at a time, <4 min: CLI task run → `server.json` 1:1 via llama-packer
→ `yaml.safe_load` (real newlines, `${PORT}` unquoted) → llama-swap launch →
`pgrep -af audiocpp_server` → `GET /health` → real request (`/v1/audio/speech`
for tts/clon, `/v1/audio/transcriptions` for asr — never trust `/v1/models`).
Full audio.cpp reload on switch is accepted (llama-swap stop/start).

## Risks

- `/tmp/audiocpp-gguf` extraction-dir collision (§0): llama-swap runs as a
  different uid than interactive users on shared boxes; either document
  `rm -rf /tmp/audiocpp-gguf` as a setup step or consider emitting
  `TMPDIR=<profiles path>` in the cmd.
- kokoro family upstream is `kokoro_tts` (82M, 54 preset voices) — GGUF
  runtime is "local GGUF BF16/Q8", confirm the packaged kokoro-82m-q8_0 loads
  standalone (it is in the verified GGUF table: Pass).
- chatterbox is clone-only upstream (`clon`, `vc`) — no zero-shot `tts` task;
  TTS requests must go through `chatterbox_turbo` (`tts`) or supply a
  `voice_ref`. Sidecars must set task accordingly.
- Backend enum drift is now resolved against 0.8.0 docs, but pin awareness:
  binary on disk decides what actually runs.
- RADV Vulkan on RDNA4: non-conformant warning is expected; watch for
  pipeline-compile failures on kokoro/chatterbox; cpu fallback exists.
