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
- budget a couple GB of VRAM so embedding and rerank can stay loaded for RAG? Automatic — the capacity budget charges them to host RAM or a small VRAM slice, so chat keeps its context.
- pick some models to use `q8_0` KV cache for longer context? YES
- ComfyUI or a game took the GPU? Re-pack with `--spare`/`--baseline` — models that no longer fit are dropped (not emitted as load-time OOM landmines), vision models fall back to text-only when only the projection is unaffordable, and squeezed entries say so in their description. The top-level `ttl` additionally lets llama-swap release VRAM after idle.


## Quickstart ("It's alive")

prerequisites: git + python 3.10 + uv + linux (and your GPU driver supports vulkan, sorry Intel)

```sh
git clone https://github.com/JeffKwasha/llama-packer.git
cd llama-packer
./extras/update.py       # llama.cpp + llama-swap; add `all` for every engine
uv run llama-packer --models-dir ~/models --output ~/llama-swap-config.yaml
./llama-swap --config ~/llama-swap-config.yaml
```

Targets are positional: `update.py all` fetches llama.cpp, llama-swap,
audio.cpp, whisper.cpp and stable-diffusion.cpp; a bare run fetches only the
first two. Of the five, only llama-swap publishes a `checksums.txt`, so it is
the only one that gets a real sha256 check — see
[extras/README.md](extras/README.md).

`profiles.yaml` is optional at this stage — the bundled defaults are used and
you get a warning saying so. Copy `profiles.yaml.example` to `profiles.yaml`
next (see [Profiles](#profiles-profilesyaml)) to start tuning.

For development, `uv pip install -e .`. To pin an older tree, clone with
`--branch <release tag>`, or install without cloning via
`uv pip install git+https://github.com/JeffKwasha/llama-packer.git@<release tag>`.
There is no PyPI package — `uvx llama-packer` will not resolve; use the
clone-and-`uv run` path above.

If a run exits with `no llama-b#### directory found`, it needs `llama-server`
and `llama-fit-params`: run `./extras/update.py`, set `LLAMA_BIN_DIR`, or pass
`--llama-server /path/to/llama-server`. A "no models found" pack means the
directory has no `.gguf` files — pass `--models-dir`, or `--create-stubs` if
you also want empty sidecars written beside the ones that have none.

## Running it as a service

The Quickstart leaves you with a foreground process. For a persistent,
systemd-managed deployment rooted at `/opt/llama`:

```sh
sudo ./extras/install.py --user llama --group models --models-dir /mnt/ai/models
```

That creates the service account, fetches the binaries, regenerates the config
under that account, writes a hardened unit, and starts it — then reports what
the API is actually advertising. `--dry-run` prints every change first. The
unit, the account's deliberately empty home, and the group that may re-pack
the config are all explained in [docs/install.md](docs/install.md).

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
- **Capacity-budgeted co-loading.** Keep small models (embedders, rerankers, a quick draft model) loaded alongside your main chat model. Every serving declares its VRAM/host-RAM capacity, llama-swap evicts the cheapest one when a pool is full, and a size-1 role pool keeps one model per role loaded.
- **Auto-parallel.** Chat models without a `parallel:` pin automatically get the (context, slots) pair that scores best under leftover VRAM — more simultaneous chats instead of one stretched window. On by default; `matrix: auto_parallel: false` disables it fleet-wide, a sidecar `parallel:` pin opts one model out.
- **Fleet-wide rewrites in a few lines.** Retarget every Qwen3.5+ model at a new chat template, or put all KV caches on `q8_0` — as override rules in `profiles.yaml`, not per-model edits.
- **Standard discovery.** Model info flows straight off the running server, so you never hand-edit your client config when you add a model. A maintained discovery plugin works against the config unchanged.
- **mmproj variants.** A vision model can be served two ways: with its mmproj for image input, and as a vision-less variant that drops the projection to reclaim VRAM for a much larger context (plus a vision companion entry keeping image input available at reduced context). The `i`/`v` capability flags on the id tell them apart; the exact context is in the display name and metadata.
- **Graceful degradation under reservation.** Re-pack with `--spare`/`--baseline` when another program holds VRAM: models whose weights no longer fit any context are dropped from the config (not emitted as load-time OOM landmines), vision models fall back to text-only when only the projection is unaffordable, and entries squeezed below their design context say so in their description. A reservation that eats the whole card refuses to overwrite the existing config. The top-level `ttl` lets llama-swap unload idle models.


## What it does

- Measures per-model VRAM via `llama-fit-params` (or safetensors estimation) and calculates the largest context window that fits — resolving companion files (mmproj, MTP drafts) and solving shared embed/rerank/chat budgets when configured
- Assembles llama-swap YAML with per-model metadata, native `capabilities`, and `filters.setParamsByID`
  overrides (aliases like `<model>:<mode>` switch sampling parameters per-request without reload)
- Applies sampling profiles and pattern-scoped override rules (`backend:`, chat templates,
  LoRAs, reasoning flags) from `profiles.yaml`

See [SPEC.md](SPEC.md) for the full schema and [docs/architecture.md](docs/architecture.md) for how it fits together.

## Directional modalities

Each entry advertises `capabilities.in` / `capabilities.out` (llama-swap derives
UI badges from these), driven by the task's endpoint plus declared
capabilities:

| Role | Directory | in → out | Backend | Endpoint |
|------|-----------|----------|---------|----------|
| `chat` | `chat/` `vision/` `doc/` | text (+image if image, +video if video, +audio if audio) → text (+audio if speech, +video if omni/video-arch) | llama-server / vLLM | `/v1/chat/completions` |
| `embeddings` | `embed/` | text → vectors | llama-server / vLLM | `/v1/embeddings` |
| `rerank` | `rerank/` | query+docs → scores | llama-server / vLLM | `/v1/rerank` |
| `s2t` | `s2t/` (opt-in) | audio → text | whisper-server, audio-cpp | `/v1/audio/transcriptions` |
| `t2s` | `t2s/` (opt-in) | text → audio | audio-cpp | `/v1/audio/speech` |
| `image` | `img/` (opt-in) | text+image → image (or video if video capability / video-arch) | sd-server | `/sdapi/v1/txt2img` |

Both `s2t` backends are available and the choice is per-model: `whisper-server`
covers the whisper model family, `audio-cpp` covers its own (`parakeet`,
`nemotron_asr`, `qwen3_asr`, `voxtral_realtime`, …) and can also serve `t2s` from
the same process. `extras/update.py all` fetches every one of these engines;
see [docs/backends/](docs/backends/) for the per-engine sidecars and VRAM model.

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

Sampling and placement are in `profiles.yaml`. Copy [`profiles.yaml.example`](profiles.yaml.example) to `profiles.yaml` (gitignored) — it has one brief commented example per category. The bundled `llama_packer/profiles.yaml` is the fallback when no file is present. All `profiles.yaml` keys are builder-consumed (unknown keys ignored), in contrast to sidecar frontmatter above.

| Category | Brief example |
|---|---|
| `defaults` / `profiles` | `temperature`, `top_p`, `cache_type`, `parallel`, `spare` (+ `base * N` expressions, `description` docs-only) |
| `models_dirs` / `roles.<name>.dir` / `hf_home` | discovery roots & tree→role ownership |
| `backends` / `vllm` | enable list (`llama-server`, `vllm-podman`, `vllm-docker`), `image`/`bin`/`container_args` (legacy `docker_args`) |
| `llama_server` / `vllm` / `sd` / `whisper` `args:` | fleet-wide server flags (e.g. `llama_server: {args: "--flash-attn on -b 512 -ub 512"}`) |
| `hardware` | `vram`, `baseline_mb`, `unified_system_mb` |
| `overrides` | `when: {base_model: 'qwen3'}` → `backend`/`chat_template`/`loras`/`reasoning-*` |
| `matrix` | sizing knobs and `sets` (role labels or model ids) |
| `capacity` | `pp_tps` / `drive_mbps` for derived `evict_cost` |

Profiles overlay `defaults` and emit `filters.setParamsByID`; sidecar `modes:` / `allow_profiles:` replace or filter them per-model. See [SPEC → profiles.yaml](SPEC.md#profilesyaml) for the full table.

### Models directory guide (`AGENTS.md`)

Run `llama-packer --agents` to write an `AGENTS.md` sidecar guide into each models dir — only when missing, so your edits are never overwritten. The bundled source is [`llama_packer/templates/models_AGENTS.md`](llama_packer/templates/models_AGENTS.md).

### vLLM backend

Serve a model with vLLM instead of llama-server via an override rule in `profiles.yaml` (or a one-off `backend:` line in its sidecar): `backend: vllm` runs the host binary, `backend: vllm-podman` / `backend: vllm-docker` run a container. Memory sizing, image/binary precedence, and budget details are in [docs/backends/vllm.md](docs/backends/vllm.md).

DGX Spark (GB10/Blackwell, unified memory) is a supported vLLM target: VRAM detection falls back to the unified pool, and per-model recipe keys (`vllm_quantization`, `moe_backend`, `mamba:`, `tool_call_parser`, `reasoning_parser`) cover the Blackwell model recipes. Docker entries are self-contained: HF_HOME root mounted read-only at `/root/.cache/huggingface` (offline — models must be pre-staged), `cmdStop`/`unloadTimeout` for container lifecycle, explicit `proxy`.

## Calibrating VRAM estimates (`--probe-memory`)

Normal packing uses `llama-fit-params` (header-only, ~2 s/model) which is fast but
blind to the MTP draft, recurrent-state cache, and allocator overhead. For tighter
estimates, run a real `llama-server` measurement grid once per architecture family:

```sh
# Probe all arch families found in the models dir
python -m llama_packer --probe-memory --models-dir ~/models

# Probe specific families only (faster, good for resuming)
python -m llama_packer --probe-memory qwen35 gemma4 --models-dir ~/models
```

**Requirements:**
- No running `llama` processes (the probe refuses otherwise)
- GPU visible as device 0 (run from a normal shell, not under the hardened
  systemd unit)
- Enough VRAM for `(design, 2)` points at the cap context

The probe measures `(weights, compute, c, D)` directly from the server's
`-lv 5` buffer lines, fits the affine law, and persists the constants to each
representative's sidecar (`derived:` block, `source: llama-server`). Partial
runs are safe — completed models are written atomically and only the in-flight
model is lost. Re-run with the missing arches listed to resume.

Once the sidecar carries a measured block, packing answers VRAM from it with
one `stat` per model — no GGUF header reads, no fit-params runs. Measuring
near-instantly is the point of the block: it is invalidated only by a file
change (size/mtime) or a serve-flag change (the compute term re-measures via
one `llama-fit-params` run).

For an unattended nightly calibration, wrap it in a script that checks no `llama` process is running, then fires
`--probe-memory`. The probe itself refuses to start beside a running
process.

## Agent skill: authoring `profiles.yaml`

[`docs/profiles-authoring.md`](docs/profiles-authoring.md) is a self-contained
guide (with OpenCode skill frontmatter) covering discovery, `tasks:`,
sampling profiles, CPU/GPU placement, matrix sizing, capacity routing, pools,
TTL, updates, and the `--check`/dry-run/validate loop.

The repo itself does not contain an `.opencode/` tree; the file is a normal
document. To expose it to OpenCode on your machine, copy or symlink it into
the global skills directory under the ID you want:

```sh
mkdir -p ~/.config/opencode/skills
ln -s "$PWD/docs/profiles-authoring.md" \
      ~/.config/opencode/skills/profiles-authoring.md
# or: cp docs/profiles-authoring.md ~/.config/opencode/skills/profiles-authoring.md
```

A flat `~/.config/opencode/skills/<id>.md` is a skill whose ID is the file
name. For a project-local install, symlink it into that project's
`.opencode/skills/` instead:

```sh
mkdir -p .opencode/skills
ln -s "$PWD/docs/profiles-authoring.md" .opencode/skills/profiles-authoring.md
```

For a whole directory of skills, add it to `skills` in `opencode.json(c)`;
relative paths resolve from the active working directory:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "skills": ["./team-skills", "~/shared/opencode-skills"]
}
```

**Installed deployments.** `extras/install.py` creates `<run-dir>/profiles.yaml`
from the bundled default when it is absent (never overwriting an existing
one) and places this guide at
`<run-dir>/.opencode/skills/profiles-authoring/SKILL.md`, owned by the
service account. Pass `--no-agent-guide` to skip the guide. An agent started
in the run directory then discovers it; see
[docs/install.md](docs/install.md#profilesyaml-is-yours) for editing and
re-packing.

For an existing install without the guide, copy it in:

```sh
mkdir -p <run-dir>/.opencode/skills/profiles-authoring
cp docs/profiles-authoring.md \
   <run-dir>/.opencode/skills/profiles-authoring/SKILL.md
```

OpenCode reloads skills on save; no restart is needed. Load it with
`@profiles-authoring` or let the model pick it from the description.

## See also

- [docs/glossary.md](docs/glossary.md) — the vocabulary, if a word in a commit message or a PR does not mean what you think it means
- [SPEC.md](SPEC.md) — detailed configuration specification
- [docs/architecture.md](docs/architecture.md) — component map, invariants, extension points
- [docs/install.md](docs/install.md) — systemd deployment from a run directory ([extras/install.py](extras/install.py))
- [docs/backends/](docs/backends/) — per-engine notes: llama-server, vLLM, sd-server, whisper-server, audio-cpp
- [profiles.yaml.example](profiles.yaml.example) — commented starter for `profiles.yaml`; [llama_packer/profiles.yaml](llama_packer/profiles.yaml) is the bundled fallback
- [docs/profiles-authoring.md](docs/profiles-authoring.md) — agent-facing guide for authoring and validating `profiles.yaml`
- [docs/memory-and-eviction.md](docs/memory-and-eviction.md) — VRAM/RAM budgets, co-residency reservations, eviction cost, TTL
- [docs/llama-swap.md](docs/llama-swap.md) — llama-swap capabilities mapped to the `profiles.yaml`/sidecar input that provides each

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
- Image generation via `sd-server` (stable-diffusion.cpp) — **available** as a role owning `img/` (`role: image`, `dir: img`) with `backends: [sd-server]` (opt-in; fixed VRAM overhead, `proxy`/`checkEndpoint: /`); see [docs/backends/sd-server.md](docs/backends/sd-server.md) and [docs/plans/comfyui-sd.md](docs/plans/comfyui-sd.md)
- Speech-to-text via `whisper-server` (whisper.cpp) — **available** as a role owning `s2t/` (`role: s2t`, `dir: s2t`) with `backends: [whisper-server]` (opt-in; GGML `.bin` models with authored same-stem sidecars; fixed VRAM overhead); see [docs/backends/whisper-server.md](docs/backends/whisper-server.md)
- Text-to-speech / speech-to-text via `audio-cpp` (audio.cpp) — **available** as roles owning `t2s/` and `s2t/` with `backends: [audio-cpp]` (opt-in; sidecar `audio_cpp: {family, task}`; fixed VRAM overhead; replaces the retired kokoro backend); see [docs/backends/audio-cpp.md](docs/backends/audio-cpp.md)
- ComfyUI (`comfyui-boot`) remains future work — see [docs/plans/comfyui-sd.md](docs/plans/comfyui-sd.md) for `comfyui-boot` syntax findings (`/comfyui/` + `compat.ignoreWebsockets`, unified image)
- Capacity routing — **available**: the packer emits llama-swap pools (VRAM, host RAM, one per role), per-model `capacities`, and derived `evict_cost`. `matrix.sets:` switches to the matrix engine. See [SPEC → Capacity routing](SPEC.md#capacity-routing)
