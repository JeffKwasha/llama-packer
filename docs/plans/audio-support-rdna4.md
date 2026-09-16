# Audio Support in llama-packer — Setup Plan (RDNA4 / R9700)

**Host:** AMD Navi 48 [Radeon AI PRO R9700], gfx1201, RDNA4, **32624 MiB** VRAM.
**Scope:** plan only — no execution. Downloading is allowed; do **not** use shell
hacks that trigger approval prompts. `HF_HOME=/mnt/ai/huggingface` is set.

llama-packer already ships **two** audio backends (see
`docs/backends/whisper-server.md`, `docs/backends/audio-cpp.md`):

| Backend | Engine | Formats | Roles | Endpoint | Compute buffer |
|---------|--------|---------|-------|----------|----------------|
| `whisper-server` | whisper.cpp | GGML `.bin` | `s2t` | `POST /v1/audio/transcriptions` (`/`) | +100 MiB |
| `audio-cpp` | audio.cpp | `.gguf` / `.safetensors` / `hf_repo` | `t2s` / `s2t` | `/health` | +1024 MiB |

VRAM model in this codebase (fixed overhead, excluded from the chat matrix):

```
vram_mb = model_mib(file size) + compute_mib
```

where `compute_mib` is `1024` for audio.cpp and `100` for whisper.cpp
(`llama_packer/consts.py`). Both backends are **host-only** transports today.

---

## 1. AMD RDNA4 compatibility — the key fact

Both engines support **Vulkan**, the portable, no-ROCm-required path. On RDNA4
(gfx1201) **Vulkan is the recommended backend**:

- **audio.cpp**: `backend: auto` in the backend resolves the vendor at launch —
  NVIDIA→`cuda`, **AMD→`vulkan`**, else `cpu` (`audio_cpp.py`). So with `auto`
  the R9700 automatically uses Vulkan. No manual override needed, and no ROCm
  kernel-list issue (RDNA4 is too new/edge for a stable ROCm list).
- **whisper.cpp**: builds with `-DGGML_VULKAN=1` (portable) or `-DGGML_HIP=1 -DAMDGPU_TARGETS=gfx1201`
  (ROCm). Vulkan is the safer choice on this card.
- A `whisper.cpp-rocm` fork (lemonade-sdk) adds HIP for newer cards; optional.

> Note the AMD-SMI metrics endpoint is **not enabled** on this box (see
> `memory_probe.py` / `hardware.py` fallback to `rocminfo`). VRAM is still
> readable from sysfs; no impact on inference.

---

## 2. Model compatibility — small + RDNA4-ready

### 2a. Already on disk (no download needed)

| Path | Format | Backend | Role | On-disk | ≈VRAM |
|------|--------|---------|------|---------|-------|
| `whisper/ggml-large-v3-turbo-q8_0.bin` | GGML `.bin` | whisper | `s2t` | 834 MB | **~0.93 GB** |
| `whisper/ggml-parakeet-tdt-0.6b-v3-q8_0.bin` | GGML `.bin` | whisper | `s2t` | 638 MB | **~0.72 GB** |
| `whisper/ggml-model.bin` | GGML `.bin` | whisper | `s2t` | 466 MB | **~0.57 GB** |
| `whisper/ggml-silero-v6.2.0.bin` | GGML `.bin` (VAD) | whisper | `s2t` | <1 MB | **~0.1 GB** |
| `s2t/Qwen3-TTS-12Hz-1.7B-CustomVoice/` | `model.safetensors` (raw HF ckpt) | audio.cpp | `t2s` | 3.8 GB | **~3.1 GB** |

### 2b. Downloadable from HuggingFace (small, good on 32 GB)

`audio-cpp/audio.cpp-gguf` ships **pre-structured GGUFs** (Apache 2.0 / CC-BY).
`Serveurperso/Qwen3-TTS-GGUF` ships the Qwen3-TTS GGUFs in `Q4_K_M` too.

| Model | Repo | Role | Quant | On-disk | ≈VRAM |
|-------|------|------|-------|---------|-------|
| Qwen3-TTS 12Hz 1.7B | audio.cpp-gguf (`q8_0_v2`) | `t2s` | Q8_0 | **2.57 GB** (verified) | **~3.6 GB** |
| Qwen3-TTS 12Hz 0.6B | audio.cpp-gguf | `t2s` | Q8_0 | **1.90 GB** (verified) | **~2.9 GB** |
| Qwen3-ASR 0.6B | audio.cpp-gguf | `s2t` | Q8_0 | **1.10 GB** (verified) | **~2.1 GB** |
| PocketTTS-100M | audio.cpp-gguf | `t2s` | Q8_0 | ~16 MB | **~1.1 GB** |
| Soprano-TTS-1.1-80M | audio.cpp-gguf | `t2s` | Q8_0 | ~80 MB | **~1.1 GB** |
| Moss-TTS-Nano-100M | audio.cpp-gguf | `t2s` | Q8 | ~100 MB | **~1.1 GB** |
| Magpie-TTS-357M | audio.cpp-gguf | `t2s` | Q8 | ~400 MB | **~1.4 GB** |
| Qwen3-ASR 1.7B | audio.cpp-gguf / `Qwen--Qwen3-ASR-1.7B` | `s2t` | Q8 | ~2.1 GB | **~3.1 GB** |
| Qwen3-ASR 0.6B | audio.cpp-gguf (`Qwen--Qwen3-ASR-0.6B` cached) | `s2t` | Q8 | ~0.7 GB | **~1.7 GB** |
| Nemotron-3.5-ASR-0.6B | audio.cpp-gguf | `s2t` | Q8 | ~0.7 GB | **~1.7 GB** |
| SenseVoice-Small | audio.cpp-gguf | `s2t` | Q8 | ~0.3 GB | **~1.0 GB** |

> Note: audio GGUFs ship in `Q8_0`/`BF16` (not `Q6_K_M`) — that's already the
> memory-efficient sweet spot, matching your Q8 preference. For chat GGUFs your
> usual `Q6_K_M` still applies; this only covers the audio repos.
>
> **Sizing trap — "0.6B" does not mean 0.6 GB.** The size name on Qwen3-TTS
> (and similar audio.cpp GGUFs) refers only to the *talker* LLM. The GGUF is a
> composite that also bundles the 12 Hz speech codec (encoder+decoder VQ
> tokenizer) and a speaker encoder. Verified from the GGUF header of
> `qwen3-tts-12hz-0.6b-base-q8_0.gguf` (2026-09-13): 1,085M total params
> (talker+encoder 804M, speech_tokenizer 69M, VQ codec ~212M) → 1,991 MB on
> disk, not the ~600 MB a naive `0.6B × Q8` estimate implies. Always trust the
> repo file size, not `params × quant bytes`.

---

## 3. VRAM budget (32624 MiB, ~31.7 GiB)

Fixed overheads are small. A realistic co-load on the R9700:

| Resident | ≈VRAM |
|----------|-------|
| Chat model (e.g. Qwen4B Q8 ≈ 2.8 GB + 131k ctx) | 4–6 GB |
| + embed (voyage-4-nano) + rerank (qwen3-0.6b) | ~1.2 GB |
| + Qwen3-TTS 0.6B Q8 (`t2s`) | ~2.0 GB |
| + Qwen3-ASR 0.6B Q8 (`s2t`) | ~1.7 GB |
| + Silero VAD / Parakeet (`s2t`) | ~0.7 GB |
| **Total** | **~11 GB** |

→ **~20 GB headroom.** You can run several small audio models alongside chat,
or a 1.7B TTS + 1.7B ASR + heavy chat with room to spare. Even a 7B chat
stays feasible. `pocket_tts` (≈1.1 GB) is the cheapest to add first.

---

## 4. Setup process

### 4.1 Obtain binaries (out of scope of llama-packer — no build step)

- **audio.cpp** (`audiocpp_server`):
  - Source build: `-DENGINE_ENABLE_VULKAN=ON` (recommended on RDNA4), CPU on;
    optionally `-DENGINE_ENABLE_CUDA=ON` (not used here).
  - Or pull the prebuilt into the **`unified-vulkan`** llama-swap image.
  - Resolution chain: `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
    `$AUDIOCPP_BIN_DIR` > `audiocpp_server` on `PATH`.
- **whisper.cpp** (`whisper-server`, Vulkan build for RDNA4):
  - `cmake -B build -DGGML_VULKAN=1 && cmake --build build -j --config Release`
  - Resolution: `--whisper-server` > `profiles.yaml whisper.bin` >
    `$WHISPER_BIN_DIR` > `whisper-server` on `PATH`.

### 4.2 Stage models into role directories

- **TTS → `t2s/`**, **ASR → `s2t/`** (opt-in dirs; default dir map never routes
  them, so a stray `.gguf` in `chat/` is never treated as audio).
- Each model needs a **same-stem `.md` sidecar** beside its file.
- Staged files already exist in `tts/`, `voice/`, `s2t/` — the `tts/`/`voice/`
  trees hold the *older* piper/kokoro/vosk/V voice-input stack (not audio.cpp);
  the new path is `t2s/` + `s2t/`.

> **Safetensors caveat:** `audio-cpp` lists `.safetensors` as supported, and the
> raw HF checkpoint is at `s2t/Qwen3-TTS-12Hz-1.7B-CustomVoice/model.safetensors`.
> But audio.cpp expects a **specific companion structure** (see upstream
> `docs/gguf.md` — tokenizers, token-ids files). The raw checkpoint may not load
> as-is; prefer the **pre-structured GGUFs** from `audio-cpp/audio.cpp-gguf`
> where possible. If you keep the safetensors, verify audio.cpp's required file
> layout before serving.

### 4.3 Wire profiles.yaml

```yaml
defaults:
  cache_type: q8_0
  parallel: 1

audio_cpp:
  bin: audiocpp_server          # resolved via the chain in 4.1
  backend: auto                 # → vulkan for AMD (R9700), no ROCm needed
  device: 0

whisper:
  bin: whisper-server

dirs: {t2s: t2s, s2t: s2t}       # opt into audio role dirs
backends: [llama-server, vllm, sd-server, whisper-server, audio-cpp]

matrix:
  categories:
    tts: {role: t2s}
    stt: {role: s2t}
  evict_costs: {tts: 50, stt: 50}   # audio: evict before chat context
  sets:
    voice: "__CHAT_VARS__ & (tts | stt)"
    rag:   "__CHAT_VARS__ & emb & rnk"
```

### 4.4 Write sidecars

**audio.cpp** (`t2s/` or `s2t/`, beside the `.gguf`/`.safetensors`):

```yaml
---
role: t2s
audio_cpp:
  family: qwen3_tts        # chatterbox | qwen3_asr | qwen3_tts | pocket_tts | …
  task: tts
  options: { guidance_scale: 0.5, temperature: 0.8, top_p: 0.8 }
  voice: jk
vram_mb: 2048              # pin to the ≈VRAM in §2 (weights + 1024 MiB buffer)
```

**whisper.cpp** (same-stem `.md` beside the `.bin` in `s2t/`):

```yaml
---
role: s2t
vram_mb: 953               # ≈ file MiB + 100 MiB (whisper buffer)
```

### 4.5 Generate + deploy

```sh
# from ~/wrk/llama
uv run llama-packer --models-dir /mnt/ai/models --output config.yaml
```

- Audio entries are written as `audio-cpp` (t2s) / `whisper-server` (s2t) with
  `proxy` + `checkEndpoint`; one process per sidecar (1:1).
- Fix backend with `--audio-cpp-server /path/audiocpp_server` if needed.
- Deploy: `./llama-swap --config config.yaml`.

---

## 5. Deep dive: latency research (your actual goal)

Your goal: **read long articles (Substack) into natural, clear English speech
near real-time at 2x.** That is two low-latency stages — a fast **ASR** (to
tokenize the article into plain text / chunks — usually you just read the text
directly, so ASR is for audio sources, not plain text) and a fast, natural
**TTS**. For *plain-text* articles you skip ASR and need only **TTS at 2×**. The
research below is what to pick and why, on the R9700.

> **Key metric — Real-Time Factor (RTF).** RTF = inference time ÷ audio duration.
> **RTF < 1.0 = faster than real time.** RTF 0.5 = 2× real time. The goal is
> "2×", i.e. **RTF ≤ 0.5** for sustained TTS on long text.

### 5.1 TTS — fastest, natural English (article→speech at 2×)

The decisive numbers come from the Qwen3-TTS technical report (arXiv 2601.15621,
Qwen) and independent GPU benchmarks (gigagpu.com / qwen3-tts.app). Figures are
FP16, single-stream; quantization (Q8/Q6/Q5) raises RTF somewhat but the
*shape* is identical.

| Model | Params | First-packet (concurrency 1) | RTF @1 | VRAM | Notes |
|-------|--------|------------------------------|--------|------|-------|
| **Qwen3-TTS 12Hz 0.6B** | 0.6B (talker) | **97 ms** | **~0.23** (~4× RT) | **~2.9 GB** (1.9 GB file, verified) | 12.5 Hz tokenizer (80 ms/token, 320 ms packets). **Best quality/latency fit.** |
| **Qwen3-TTS 12Hz 1.7B** | 1.7B (talker) | **101 ms** | **~0.25** (~4× RT) | **~3.6 GB** (2.57 GB file, verified) | Best prosody/timbre of family. **Best quality for long-form narration.** |
| PocketTTS-100M | 100M | ~150 ms | **~0.02** (~48× RT) | <1 GB | Fastest streaming TTS (CPU-optimized); quality lowest of the group. |
| Kokoro-82M | 82M | ~40 ms | ~0.03 (~100× RT) | ~0.5 GB | Fastest available, **GGUF now ships** (`Kokoro-82M-GGUF/kokoro-82m-q8_0.gguf`, 180 MB) |
| Soprano-TTS 80M | 80M | low | ~<0.5 | <1 GB | Streaming, minimal footprint; quality light |
| MOSS-TTS-Nano 100M | 100M | low | ~<0.5 | <1 GB | Streaming, minimal footprint; quality light |

**Why 12Hz matters:** Qwen3-TTS's 12.5 Hz tokenizer emits audio in 320 ms
packets — first audio in 97–101 ms and continuous audio output regardless of
how long the article is. That's the latency win that makes narration feel
instant while still rendering well under 2× (this is ~4× real-time).

**Verdict for article→speech at 2× (RTF ≤ 0.5):**
- **Primary pick — Qwen3-TTS 12Hz 0.6B Q8** (`1.9 GB` file, ~2.9 GB on
  GPU, ~4× real-time, first audio ~100 ms). Streaming, best latency/quality
  balance, ships as GGUF. The natural default for narration on the R9700.
  ("0.6B" = talker only; the GGUF bundles the 12 Hz codec — see §2 sizing trap.)
- **Quality pick — Qwen3-TTS 12Hz 1.7B Q8** (`2.57 GB` file, ~3.6 GB on
  GPU, ~4× real-time). Highest prosody/timbre consistency — best for long-form
  narration where clarity over long audio matters. Still well under 2×.
- **Speed pick — PocketTTS-100M Q8** (`~57 MB` weights, <1 GB on GPU, ~48×
  real-time). When raw speed outweighs quality (lower-tier English, weaker on
  proper nouns). Ships as GGUF via audio.cpp.
- **Tiny streaming alternatives** (Soprano-80M, MOSS-TTS-Nano-100M) for minimum
  VRAM/footprint at lower quality.
- Kokoro-82M is fast (RTF ~0.03) but has **no official audio.cpp GGUF**, so it
  is excluded from the audio.cpp path.

Both Qwen3-TTS GGUFs ship via `audio-cpp/audio.cpp-gguf` (`qwen3_tts` family).
Q8 matches your preference; BF16 is available if you want max precision and
have room (0.6B BF16 ~1.2 GB, 1.7B BF16 ~3.3 GB).

**Pipeline for articles:** plain HTML/Markdown → strip to text (a small
preprocessing step) → send to the `t2s` model. TTS latency scales with the
*number of tokens* generated, so cleaner text (no markup, fewer stray tokens)
reduces work. Chunk long articles so the server emits audio as it goes.

### 5.2 STT / ASR — fastest, accurate English

Here the question is subtler: **for plain-text articles you don't need ASR at
all** — you transcribe *audio* sources (a spoken podcast, a recording). But
since you asked, and ASR is the other half of any speech pipeline, here is the
low-latency landscape for local ASR on AMD:

| Model | Params | Format on AMD | RTF / speed | VRAM (working) | English accuracy |
|-------|--------|---------------|-------------|----------------|------------------|
| **Qwen3-ASR 0.6B** | 0.6B | GGUF (audio.cpp `qwen3`) | TTFT ~92 ms; RTF ~0.10 (GPU) | ~1.0–1.5 GB | **~2.3% WER** test-clean; best open accuracy, 52 langs |
| **Qwen3-ASR 1.7B** | 1.7B | GGUF (audio.cpp `qwen3`) | autoregressive, higher TTFT | ~1.9–2.7 GB | **~5.76%** mixed / ~1.3% clean; accuracy ceiling |
| **Nemotron-3.5-ASR-0.6B** | 0.6B | GGUF (audio.cpp `nemotron_asr`) | **streaming p50 ~18–86 ms**, final <100 ms | ~0.7–1.4 GB (weights) | EN-only 2.32% test-clean; 40 locales, cache-aware |
| **Parakeet-TDT-0.6B v3** | 0.6B | GGUF (audio.cpp `parakeet_tdt`) | **~57× real-time** (7.4 s clip → 131 ms) | ~0.9 GB weights / **~5.1 GB working** | **6.32% WER** vs Whisper 7.44%; ~3–5% clean |
| SenseVoice-Small | 0.22B | GGUF (`sense_asr`) | NAR, ~70 ms/10 s (fastest) | ~1 GB | **~14.7% WER English** — poor for English-only; CJK best |
| Whisper.cpp large-v3 | 1.55B | GGML `.bin` | ~8–12× RT (CUDA) | ~1.5–3 GB | solid, mature, 99+ langs |
| Whisper.cpp large-v3-Turbo | 0.81B | GGML `.bin` | ~6.5× RT | ~1.5 GB | slightly lower accuracy, lighter |

**Notes for ASR selection (AMD / R9700):**
- **For plain-text articles you don't need ASR at all** — read the text
  directly. ASR is only for *audio* sources (recordings, podcasts).
- **Qwen3-ASR 0.6B Q8** (`~1.7 GB` GGUF, best accuracy, already cached at
  `Qwen--Qwen3-ASR-0.6B`) is the primary pick. 1.7B if you want the accuracy
  ceiling.
- **Nemotron-3.5-ASR-0.6B** is the lowest-latency streaming choice (p50 ~18 ms,
  chunk size 80 ms–1.12 s, zero accuracy cost); good if streaming latency is
  the priority.
- **Parakeet 0.6B v3** is the fastest single-stream (~57× real-time) but uses
  ~5.1 GB working VRAM (see §5.3 for the AMD/Vulkan caveat).
- **SenseVoice is the wrong pick for English** (~14.7% WER) despite being
  architecturally fastest — keep it only for CJK.
- **VRAM caveat on Parakeet:** the *weights* are tiny (~0.9 GB Q8) but it
  needs ~5.1 GB working at FP16 — the other 0.6B models need far less.
- The R9700's 640 GB/s memory bandwidth is ~2× the L4 / 4070 Ti SUPER most of
  these benchmarks cite, so expect comparable-to-better throughput; the VRAM
  footprint is model-determined and carries across hardware.

**Verdict:**
- **Primary ASR — Qwen3-ASR 0.6B Q8** (`~1.7 GB` GGUF, RTF ~0.1, best open
  accuracy). Already cached at `Qwen--Qwen3-ASR-0.6B`.
- **Speed pick — Parakeet-TDT 0.6B v3** (`~0.7 GB` GGUF). ~4× faster than
  Whisper, matches its accuracy on English. See §5.3 for the AMD caveat.
- **If the source is plain text, skip ASR.** You only need a TTS model. For
  *audio* sources (recordings, podcasts) add the 0.6B Qwen3-ASR.

### 5.3 Parakeet on AMD — usable? Is llama-swap support hard?

- **Shipped as a GGUF by audio.cpp.** `audio.cpp` (0xShug0) carries a
  built-in **`parakeet_tdt`** family — a FastConformer TDT port of
  `nvidia/parakeet-tdt-0.6b-v3` (0.6B, 25 European languages). Default install
  pulls the Transformers `model.safetensors` checkpoint (~2.5 GB, converted at
  session time); standalone GGUFs (`f32`/`f16`/`bf16`/`q8_0`) pass directly via
  `--model`. No Reflection-LM (125M/250M) GGUF ships for audio.cpp.
- **Backend choice on AMD:** audio.cpp supports **`--backend hip/rocm/vulkan`**
  on AMD. `backend: auto` resolves to **`vulkan`** here (`audio_cpp.py`);
  HIP/ROCm (`gfx1201` officially supported from ROCm 7.2, March 2026; earlier
  ROCm via `HSA_OVERRIDE_GFX_VERSION`) is also available. ggml's Vulkan path
  runs on any Vulkan 1.2 GPU, so no ROCm required.
- **Honest caveats:** `parakeet_tdt` is only validated end-to-end on
  CPU + NVIDIA (GTX 1650 Max-Q, RTX 4070 Ti SUPER). **No published R9700/DNA4
  run exists.** audio.cpp itself has asked the community to help test ROCm
  coverage ("thin"); the istft/torch-random CUDA `.cu` paths fall back to CPU,
  and streaming is buffered, not native cache-aware. Expect unvalidated
  perf and *possible* partial CPU fallback. whisper.cpp and Qwen3-ASR are
  more thoroughly exercised on AMD.
- **Accuracy:** Open ASR Leaderboard Parakeet 0.6b-v3 = **6.32% WER vs Whisper
  large-v3 7.44%** (and beats large-v3-turbo 7.83%) at far lower compute —
  ~3–5% WER on clean English (LibriSpeech). Fewer hallucinations than Whisper.
- **Latency:** very low — **~131 ms to transcribe a 7.4 s clip** on a GTX 1650
  Max-Q (~57× real-time). The model is latency-bound on the FastConformer
  encoder. **VRAM is tiny: F32 ~2.4 GB, F16/BF16 ~1.3 GB, Q8_0 ~0.9 GB** —
  comfortably fits even the 32 GB R9700. Equivalents: `ggml-org/parakeet-GGUF`
  q8_0 ~711 MB / q4_k ~467 MB / f16 ~1.26 GB; `mudler/parakeet-cpp-gguf`
  q8_0 ~941 MB / q4_k ~675 MB.
- **How hard to make llama-swap support it: not at all.** llama-swap already
  natively proxies audio.cpp via `/audioapi/v1/tasks/run`, and
  `llama-packer` already emits the `audio-cpp` entry. What you need instead:
  (1) an **audio.cpp build with the HIP (ROCm) or Vulkan ggml backend**
  (or `parakeet.cpp` / `whisper.cpp parakeet-cli`, both Vulkan-capable);
  (2) a llama-swap `models` entry whose cmd launches the audio.cpp server on
  `parakeet_tdt` with `--backend hip`/`--backend vulkan`, plus the bundled
  **Silero VAD** sidecar for chunking; (3) the weights. Real work = obtain the
  AMD engine + validate that `parakeet_tdt`'s ggml ops run on gfx120x. A
  config + sidecar, not a code patch.

## 6. Sidecar authoring for audio — is `AGENTS.md` enough?

`llama_packer/templates/models_AGENTS.md` **covers the audio sidecar keys well**
— `s2t`/`t2s` dir layout, `audio_cpp: {family, task, options, voice}`, and the
`vram_mb` pin all appear (lines ~29–31, 126–132, 128). The template is strong.

**What it is missing for audio (additions made to this doc):**

1. **Latency/real-time guidance.** The template explains *how* to write a
   sidecar, not *which* models to write. There is no mention that TTS/ASR
   models are served at **fixed overhead + a 1024 MiB buffer** (audio.cpp) /
   **100 MiB** (whisper), and that RTF (not "context slots") is the real
   performance number. Added context is in §5.
2. **GGUF companion-structure caveat.** audio.cpp expects **specific companion
   files** (tokenizers, token-id files) alongside the model. The raw HF
   checkpoint (`model.safetensors`) at `s2t/Qwen3-TTS-12Hz-1.7B-CustomVoice/`
   may **not** load as-is; prefer the **pre-structured GGUFs** from
   `audio-cpp/audio.cpp-gguf`. The template's generic "classify before you fill"
   section covers GGUF-vs-safetensors but not this specific requirement.
3. **Backend resolution.** Not documented: `audio-cpp` `backend: auto` →
   **vulkan on AMD**; `whisper-server` builds with `-DGGML_VULKAN=1`. Added in
   §1/§4.1.
4. **`device: 0` + `options` defaults.** The template shows `options`/`device`
   but not that the backend also stamps `max_loaded_models`, `voice_dir`, etc.
   from `profiles.yaml`. Documented in §4.

**A complete TTS sidecar** (`t2s/qwen3_tts_1.7b.md`, beside the GGUF):

```yaml
---
name: "Qwen3-TTS 1.7B English"
model_id: qwen3-tts-1.7b
parameters: 1.7B
quantization: Q8_0
description: "Qwen3-TTS 12Hz custom-voice TTS. Natural English, ~2x real-time. Streaming."
role: t2s                       # audio-cpp
hf_repo: audio-cpp/audio.cpp-gguf   # GGUF source (audio.cpp)
model: Qwen3-TTS-12Hz-1.7B-Base-GGUF   # snapshot filename; verify with ls
audio_cpp:
  family: qwen3_tts             # audio.cpp family (identity + routing)
  task: tts                     # tts | clon | …
  options: {temperature: 0.8, top_p: 0.8}
  voice: jk                     # voice-library name, or voice_ref: voices/jk.wav
vram_mb: 3100                   # ~ weights (2.1 GB) + 1024 MiB buffer; pin if measured
---
```

**A complete ASR sidecar** (`s2t/qwen3_asr_0.6b.md`, beside the GGUF):

```yaml
---
name: "Qwen3-ASR 0.6B"
model_id: qwen3-asr-0.6b
parameters: 0.6B
quantization: Q8_0
description: "Fast, accurate English ASR. RTF ~0.1 on GPU. Streaming."
role: s2t                       # audio-cpp (or whisper-server for .bin)
hf_repo: audio-cpp/audio.cpp-gguf
model: Qwen3-ASR-0.6B-GGUF
audio_cpp:
  family: qwen3_asr
  task: asr
vram_mb: 1700                   # ~ weights (0.7 GB) + 1024 MiB buffer
---
```

**A whisper.cpp sidecar** (`s2t/ggml-large-v3-turbo.md`, beside the `.bin`,
already on disk at `whisper/ggml-large-v3-turbo-q8_0.bin`, 834 MB):

```yaml
---
name: "Whisper Large-V3 Turbo Q8"
model_id: whisper-large-v3-turbo-q8
description: "Whisper.cpp large-v3-turbo Q8_0. ~0.93 GB + 100 MiB buffer."
role: s2t                       # whisper-server (GGML .bin)
vram_mb: 953                    # file MiB (834) + 100 compute buffer
---
```

---

## 7. Recommended starting set (fits easily, matches Q8 preference)

For the **article→speech at 2×** goal, the minimal stack is:

- **TTS:** Qwen3-TTS 1.7B Q8 (`~1.45 GB` weights, ~4.9 GB GPU, ~4× real-time,
  best English quality) — or 0.6B Q8 (`~0.57 GB` weights, ~2.7 GB GPU, ~4×,
  slightly lower prosody) for extra headroom.
- **VAD:** Silero v6.2 (`~0.1 GB`) already staged — drop silence before TTS.

That is ~3.2 GB + chat model + embeds. The R9700 has ~19–20 GB headroom. Add
ASR only for *audio* sources:
- **ASR:** Qwen3-ASR 0.6B Q8 (`~1.7 GB`) or Parakeet 0.6B v3 (`~5.1 GB` working
  VRAM).

Scales up to 1.7B TTS/ASR or more co-loads with room to spare.
