# llama-packer

Because llama-swap doesn't pack itself

[llama-swap](https://github.com/mostlygeek/llama-swap) masterfully orchestrates dozens of models with aliases in a complex matrix defining which to run concurrently. Some have mmproj for vision, MTP or both. Each has options: KV-cache quantization, architecture-specific load parameters, and parallel slots trade context VRAM for concurrent chats.

The interactions of just a few models is complicated busy work... Too hard for humans, and it's kinda silly to load all that data into an AI context and hope it writes the config correctly.

There has to be a better way!

LLAMA-PACKER!

Simple concept - each model has an .md file (sidecar) describing it, and all its options and capabilities.
You point llama-packer at your models directory (and your HF_HOME) and it compiles a list of models and allocates each based on profiles, overrides, per directory models.yaml customizations and individual model sidecars.

Out pops a config.yaml for llama-swap that really packs that LLAMA.


Features:
- want all Qwen3.5+ models to use the sharp chat template? 3 lines in models.yaml
- Add low-temperature `coding` aliases to every model? 2 lines in profiles.yaml
- Want 'text-only' variants that skip mmproj to maximize context or parallel? Automatic.
- budget a couple GB of VRAM so embedding and rerank can be resident for RAG? Automatic — the matrix squeezes chat context to keep them loaded instead of evicted.
- pick some models to use `q8_0` KV cache for longer context? YES


## Quickstart ("It's alive")

prerequisites: git + python 3.10 + uv + linux (and your GPU driver supports vulkan, sorry Intel)

```sh
git clone https://github.com/JeffKwasha/llama-packer.git
cd llama-packer
./extras/update     # fetches llama.cpp + llama-swap binaries
uv run llama-packer --models-dir ~/models --output ~/llama-swap-config.yaml
./llama-swap --config ~/llama-swap-config.yaml
```

## Beyond the smoketest

You're thinking that wasn't impressive, llama-packer guessed poorly, didn't know capabilities like vision or tool calling .. etc 
Sure it might work if you made sidecars for each model... but researching and writing them is TEDIOUS..!

There has to be a better way!

cp llama_packer/templates/models_AGENTS.md ${MODELS_DIR}/AGENTS.md.

Tell your AI harness: 
read models/AGENTS.md for each of my models in models/{chat, embed, rerank} research the model on huggingface and write a sidecar, use a subagent for each model

Now you have capabilities, context windows, MTP support where available.

mv profiles.yaml.example profiles.yaml

- **Aliases.** Give one model several names — `instruct`, `coding`, `thinking` — each with its own sampling parameters, switched per-request with no reload. Declared once in the sidecar's `modes:` (or generated from `profiles.yaml`); emitted as `filters.setParamsByID` overrides.
- **A packed matrix.** Keep small models (embedders, rerankers, a quick draft model) loaded simultaneously alongside your main chat model, with VRAM budgeted across all of them so nothing OOMs. Each gets its own context sized to what actually fits.
- **Auto-parallel.** Unpinned chat models automatically get the (context, slots) pair that scores best under leftover VRAM — more simultaneous chats instead of one stretched window. On by default; `matrix: auto_parallel: false` disables it fleet-wide, a sidecar `parallel:` pin opts one model out.
- **Fleet-wide rewrites in a few lines.** Retarget every Qwen3.5+ model at a new chat template, or put all KV caches on `q8_0` — as override rules in `profiles.yaml`, not per-model edits.
- **Opencode plugin.** Model info flows straight off the running server, so you never hand-edit Opencode config when you add a model (`extras/llamaswap.ts`).
- **mmproj variants.** A vision model can be served two ways: with its mmproj for image input, and as a `-text` alias that drops the projection to reclaim VRAM for a much larger text-only context window (plus a `-vision-Nk` best-effort entry keeping vision available at reduced context).


## What it does

- Measures per-model VRAM via `llama-fit-params` (or safetensors estimation) and calculates the largest context window that fits — resolving companion files (mmproj, MTP drafts) and solving shared embed/rerank/chat budgets when configured
- Assembles llama-swap YAML with per-model metadata, native `capabilities`, and `filters.setParamsByID`
  overrides (aliases like `<model>:<mode>` switch sampling parameters per-request without reload)
- Applies sampling profiles and pattern-scoped override rules (`backend:`, chat templates,
  LoRAs, reasoning flags) from `profiles.yaml`

See [SPEC.md](SPEC.md) for the full schema and [docs/architecture.md](docs/architecture.md) for how it fits together.

## Directional modalities

Each entry advertises `capabilities.in` / `capabilities.out` (llama-swap derives
UI badges from these), driven by `role:` plus declared capabilities:

| Role | Directory | in → out | Backend | Endpoint |
|------|-----------|----------|---------|----------|
| `chat` | `chat/` `vision/` `doc/` | text (+image if image, +video if video, +audio if audio) → text (+audio if speech, +video if omni/video-arch) | llama-server / vLLM | `/v1/chat/completions` |
| `embeddings` | `embed/` | text → vectors | llama-server / vLLM | `/v1/embeddings` |
| `rerank` | `rerank/` | query+docs → scores | llama-server / vLLM | `/v1/rerank` |
| `s2t` | `s2t/` (opt-in) | audio → text | whisper-server | `/v1/audio/transcriptions` |
| `t2s` | `t2s/` (opt-in) | text → audio | audio-cpp | `/v1/audio/speech` |
| `image` | `img/` (opt-in) | text+image → image (or video if video capability / video-arch) | sd-server | `/sdapi/v1/txt2img` |

On a **chat** model, `capabilities: [image]` adds image *input* (`vision`
was removed — llama-swap modalities are text/audio/image/video), `[video]`
adds video *input* (and *output* for omni/video-arch), `[audio]` adds audio
*input* (Transcription badge), `[speech]` adds audio *output* — so a VLM
never advertises "Image Gen", and output stays text unless `speech`/`video`
output is declared. `mmproj:` is a mapping, not a capability: declare
`capabilities: [image]` (or `[image, video]`) inside the block — a bare
filename is an error, and `image`/`video` claimed at top level instead is
advertised by the text-only variant too (warned in the run log). Dedicated `s2t`/`image` roles are for standalone STT /
diffusion micro-services; their modalities are fixed regardless of declared
capabilities (`proxy` + `checkEndpoint: /` are emitted for them too).

## Sidecar example

```yaml
---
name: gemma-4-12B-it-qat-UD-Q4_K_XL
parameters: 12B
context_length: 262144
quantization: Q4_K_XL
mmproj:
  file: gemma-4-12B-it-mmproj-F16.gguf
  capabilities: [image]
capabilities: [tools, reasoning]
freethought: 0.55
strengths: ["strong multimodal", "full 256K context"]
weaknesses: ["MTP adds VRAM"]
default_mode: instruct
modes:
  instruct: { temperature: 0.6, pres_pen: 1.5 }   # llama.cpp param names
  thinking:  { temperature: 1.0, pres_pen: 0.0 }
---
```

Unknown keys pass as `metadata` informing clients with descriptive fields (see [SPEC → Model Metadata](SPEC.md#model-metadata) for the consumed-key list).

### Profiles (`profiles.yaml`)

Sampling and placement live in `profiles.yaml`. Copy [`profiles.yaml.example`](profiles.yaml.example) to `profiles.yaml` (gitignored) — it has one brief commented example per category. The bundled `llama_packer/profiles.yaml` is the fallback when no file is present. All `profiles.yaml` keys are builder-consumed (unknown keys ignored), in contrast to sidecar frontmatter above.

| Category | Brief example |
|---|---|
| `defaults` / `profiles` | `temperature`, `top_p`, `cache_type`, `parallel`, `spare` (+ `base * N` expressions, `description` docs-only) |
| `models_dirs` / `dirs` / `hf_home` | discovery roots & dir→role map (`it2t: chat`) |
| `backends` / `vllm` | enable list (`llama-server`, `vllm-docker`), `image`/`bin`/`docker_args` |
| `llama_server` / `vllm` / `sd` / `whisper` `args:` | fleet-wide server flags (e.g. `llama_server: {args: "--flash-attn on -b 512 -ub 512"}`) |
| `hardware` | `vram`, `baseline_mb`, `unified_system_mb` |
| `overrides` | `when: {base_model: 'qwen3'}` → `backend`/`chat_template`/`loras`/`reasoning-*` |
| `matrix` | shared `emb`/`rnk` co-loading sets via `__CHAT_VARS__` |

Profiles overlay `defaults` and emit `filters.setParamsByID`; sidecar `modes:` / `allow_profiles:` replace or filter them per-model. See [SPEC → profiles.yaml](SPEC.md#profilesyaml) for the full table.

### Models directory guide (`AGENTS.md`)

Run `llama-packer --agents` to write an `AGENTS.md` sidecar guide into each models dir — only when missing, so your edits are never overwritten. The bundled source is [`llama_packer/templates/models_AGENTS.md`](llama_packer/templates/models_AGENTS.md).

### vLLM backend

Serve a model with vLLM instead of llama-server via an override rule in `profiles.yaml` (or a one-off `backend:` line in its sidecar): `backend: vllm` runs the host binary, `backend: vllm-podman` / `backend: vllm-docker` run a container. Memory sizing, image/binary precedence, and budget details are in [docs/backends/vllm.md](docs/backends/vllm.md).

DGX Spark (GB10/Blackwell, unified memory) is a supported vLLM target: VRAM detection falls back to the unified pool, and per-model recipe keys (`vllm_quantization`, `moe_backend`, `mamba:`, `tool_call_parser`, `reasoning_parser`) cover the Blackwell model recipes. Docker entries are self-contained: HF_HOME root mounted read-only at `/root/.cache/huggingface` (offline — models must be pre-staged), `cmdStop`/`unloadTimeout` for container lifecycle, explicit `proxy`.

## See also

- [SPEC.md](SPEC.md) — detailed configuration specification
- [docs/architecture.md](docs/architecture.md) — component map, invariants, extension points
- [profiles.yaml.example](profiles.yaml.example) — commented starter for `profiles.yaml`; [llama_packer/profiles.yaml](llama_packer/profiles.yaml) is the bundled fallback

## Limitations

- vLLM memory sizing needs `vllm-memory-estimator` installed for HF-repo models; without it
  (or a local `.safetensors` file), context falls back to the declared `context_length` and
  vLLM's own startup profiling bounds the allocation
- LoRA adapters are llama-server-only; GGUF MTP draft companions are llama-server-only —
  under vLLM use baked-in MTP (`mtp: true`) or an explicit `speculative_config:` with a
  draft HF repo

## Future Roadmap

- Support multi-image/tensor-parallel vLLM provisioning
- Enrich `throughput_factor` with measured server log data (offline parsing)
- Chip-specific VRAM sizing rules behind the (currently inert) `gpu-family` hook
- Image generation via `sd-server` (stable-diffusion.cpp) — **available** as `role: image` with `dirs: {img: image}` and `backends: [sd-server]` (opt-in; fixed VRAM overhead, `proxy`/`checkEndpoint: /`); see [docs/backends/sd-server.md](docs/backends/sd-server.md) and [docs/plans/comfyui-sd.md](docs/plans/comfyui-sd.md)
- Speech-to-text via `whisper-server` (whisper.cpp) — **available** as `role: s2t` with `dirs: {s2t: s2t}` and `backends: [whisper-server]` (opt-in; GGML `.bin` models with authored same-stem sidecars; fixed VRAM overhead); see [docs/backends/whisper-server.md](docs/backends/whisper-server.md)
- Text-to-speech / speech-to-text via `audio-cpp` (audio.cpp) — **available** as `role: t2s` / `role: s2t` with `dirs: {t2s: t2s, s2t: s2t}` and `backends: [audio-cpp]` (opt-in; sidecar `audio_cpp: {family, task}`; fixed VRAM overhead; replaces the retired kokoro backend); see [docs/backends/audio-cpp.md](docs/backends/audio-cpp.md)
- ComfyUI (`comfyui-boot`) remains future work — see [docs/plans/comfyui-sd.md](docs/plans/comfyui-sd.md) for `comfyui-boot` syntax findings (`/comfyui/` + `compat.ignoreWebsockets`, unified image)
- Configurable matrix categories (e.g. run `stable-diffusion` alongside `VL embedding` and `chat` — not just `emb`/`rnk`) — see [docs/plans/matrix-categories.md](docs/plans/matrix-categories.md)
