# llama-packer

Generate configs for [llama-swap](https://github.com/mostlygeek/llama-swap) from GGUF (llama.cpp) or HF (vLLM) model metadata. Takes a directory of models and produces ready-to-run server configurations with optimal flags tuned to your hardware.

Scans model directories, reads YAML sidecar files, detects GPU VRAM, budgets memory across models, and writes `config.yaml` — no manual flag wrangling. Serves models with llama-server (GGUF) or vLLM (`backend: vllm` for the host binary, `backend: vllm-docker` for a container). Both backends share the same accurate context-length budgeting and per-request alias support.

## Install

Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or pip):

```sh
uvx llama-packer --dry-run    # one-shot, no install (fetches from PyPI)
```

Or install it as a persistent, editable command (tracks source changes):

```sh
uv pip install -e .          # exposes `llama-packer` on PATH
llama-packer --dry-run
```

Or run straight from source without installing:

```sh
uv run llama-packer --dry-run
```

`llama-packer` resolves `models/`, `profiles.yaml` (falls back to the bundled default), and `llama-b*/` build dirs relative to the current directory. Point it elsewhere with `--models-dir` (accepts several directories, each scanned independently), `--profiles`, and `--llama-server` (or `LLAMA_BIN_DIR`). Pass `--hf-home` to keep Hugging Face cache paths in their own `${HF_HOME}` macro instead of widening `${MODELS_DIR}`.

GGUF/safetensors files without a `.md` sidecar get an empty stub sidecar written automatically, so a directory of bare models works out of the box — fill it in whenever you like (until then identity comes from the filename, context from the default, role from the directory). Skip with `--no-stubs`. Stubs never land in HF `blobs/` trees; they sit beside the human-readable snapshot entry instead.

Output goes to `config.yaml` (`--output` overrides the path) and a sibling `config.env` in the current directory. Print the version with `llama-packer --version`.

## What it does

- Measures per-model VRAM via `llama-fit-params` (or safetensors estimation) and calculates the largest context window that fits — resolving companion files (mmproj, MTP drafts) and solving shared embed/rerank/chat budgets when configured
- Assembles llama-swap YAML with per-model metadata, native `capabilities`, and `filters.setParamsByID`
  overrides (aliases like `<model>:<mode>` switch sampling parameters per-request without reload)
- Applies sampling profiles and pattern-scoped override rules (`backend:`, chat templates,
  LoRAs, reasoning flags) from `profiles.yaml`

See [SPEC.md](SPEC.md) for the full schema and [docs/architecture.md](docs/architecture.md) for how it fits together.

## Sidecar example

```yaml
---
name: gemma-4-12B-it-qat-UD-Q4_K_XL
parameters: 12B
context_length: 262144
quantization: Q4_K_XL
mmproj: gemma-4-12B-it-mmproj-F16.gguf
capabilities: [vision, tools, reasoning]
freethought: 0.55
strengths: ["strong multimodal", "full 256K context"]
weaknesses: ["MTP adds VRAM"]
default_mode: instruct
modes:
  instruct: { temperature: 0.6, pres_pen: 1.5 }   # llama.cpp param names
  thinking:  { temperature: 1.0, pres_pen: 0.0 }
---
```

Any key not consumed by the builder passes through as `metadata` for clients — add descriptive fields without code changes (see [SPEC → Model Metadata](SPEC.md#model-metadata) for the consumed-key list).

### Profiles (`profiles.yaml`)

Sampling and placement live in `profiles.yaml`. Copy [`profiles.yaml.example`](profiles.yaml.example) to `profiles.yaml` (gitignored) — it has one brief commented example per category. The bundled `llama_packer/profiles.yaml` is the fallback when no file is present. All `profiles.yaml` keys are builder-consumed (unknown keys ignored), in contrast to sidecar frontmatter above.

| Category | Brief example |
|---|---|
| `defaults` / `profiles` | `temperature`, `top_p`, `cache_type`, `parallel`, `spare` (+ `base * N` expressions, `description` docs-only) |
| `models_dirs` / `dirs` / `hf_home` | discovery roots & dir→role map (`it2t: chat`) |
| `backends` / `vllm` | enable list (`llama-server`, `vllm-docker`), `image`/`bin`/`docker_args` |
| `hardware` | `vram`, `baseline_mb`, `unified_system_mb` |
| `overrides` | `when: {base_model: 'qwen3'}` → `backend`/`chat_template`/`loras`/`reasoning-*` |
| `matrix` | shared `emb`/`rnk` co-loading sets via `__CHAT_VARS__` |

Profiles overlay `defaults` and emit `filters.setParamsByID`; sidecar `modes:` / `allow_profiles:` replace or filter them per-model. See [SPEC → profiles.yaml](SPEC.md#profilesyaml) for the full table.

### Models directory guide (`AGENTS.md`)

Run `llama-packer --agents` to write an `AGENTS.md` sidecar guide into each models dir — only when missing, so your edits are never overwritten. The bundled source is [`llama_packer/templates/models_AGENTS.md`](llama_packer/templates/models_AGENTS.md).

### vLLM backend

Serve a model with vLLM instead of llama-server via an override rule in `profiles.yaml` (or a one-off `backend:` line in its sidecar): `backend: vllm` runs the host binary, `backend: vllm-docker` runs a container. Memory sizing, image/binary precedence, and budget details are in [SPEC.md → vLLM Backend](SPEC.md#vllm-backend).

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
- Image generation via `sd-server` (stable-diffusion.cpp) — **available** as `role: image` with `dirs: {img: image}` and `backends: [sd-server]` (opt-in; fixed VRAM overhead, `proxy`/`checkEndpoint: /`); see [SPEC.md](SPEC.md#image-backend-sd-server) and [docs/plans/comfyui-sd.md](docs/plans/comfyui-sd.md)
- ComfyUI (`comfyui-boot`) remains future work — see [docs/plans/comfyui-sd.md](docs/plans/comfyui-sd.md) for `comfyui-boot` syntax findings (`/comfyui/` + `compat.ignoreWebsockets`, unified image)
- Configurable matrix categories (e.g. run `stable-diffusion` alongside `VL embedding` and `chat` — not just `emb`/`rnk`) — see [docs/plans/matrix-categories.md](docs/plans/matrix-categories.md)
