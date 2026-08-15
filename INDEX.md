# llama-packer INDEX

Generate llama-swap configs from GGUF/VLLM model metadata. See [README.md](README.md) for the intro and [SPEC.md](SPEC.md) for the schema.

## Entry points

- [`gen-config.py`](gen-config.py) — CLI wrapper → `llama_packer.__main__`
- [`llama_packer/__main__.py`](llama_packer/__main__.py) — CLI, VRAM/env resolution, matrix wiring
- [`llama_packer/profiles.yaml`](llama_packer/profiles.yaml) — sampling profiles + `vllm:` docker defaults (bundled)
- [`llama_packer/templates/models_AGENTS.md`](llama_packer/templates/models_AGENTS.md) — bundled `models/AGENTS.md` guide (written with `--agents`)

## Modules

- [`llama_packer/model.py`](llama_packer/model.py) — `Model` sidecar parsing, field accessors (`hf_repo`, `vllm_image`, `modes`, `role`, ...), companion resolution
- [`llama_packer/writer.py`](llama_packer/writer.py) — `build_config`, `_build_entry`, full-`cmd` template branch, filters.setParamsByID/modes, matrix solver
- [`llama_packer/vram.py`](llama_packer/vram.py) — `VramBudget` fit-params, `solve_matrix_ctx`
- [`llama_packer/hardware.py`](llama_packer/hardware.py) — VRAM detection, `GpuProfile`, family handlers
- [`llama_packer/utils.py`](llama_packer/utils.py) — `_TARGET_TEMPLATES` (+ `vllm-docker`), `VLLM_DEFAULT_*`, sampling keys, discovery/slugify/params

## Backends (templates)

- llama-server — link roles `chat`/`embeddings`/`rerank` → `_TARGET_TEMPLATES`
- vLLM docker — `template: vllm-docker` on a chat sidecar → `docker run ... vllm serve`
- See SPEC.md "vLLM Docker Backend" and [docs/plans/vllm-gb10.md](docs/plans/vllm-gb10.md)

## Docs

- [README.md](README.md) — usage, sidecar example
- [SPEC.md](SPEC.md) — model metadata schema, sampling modes/aliases, vLLM backend, health-check/env/matrix
- [docs/gguf_model_analysis.md](docs/gguf_model_analysis.md) — GGUF sizing notes
- [docs/llama-server_help](docs/llama-server_help) — full llama-server CLI
- [docs/plans/](docs/plans/) — design proposals (vllm-gb10)
- [docs/reference.md](docs/reference.md) — external links

## Data dirs

- `models/` — GGUF + `.md` sidecars (+ `embed/`, `rerank/`); `AGENTS.md` guide auto-written with `--agents` (from bundled `llama_packer/templates/models_AGENTS.md`) if missing
- `llama-b*/` — llama.cpp builds (used via `find_bin_dir`)
- `model_cfg/` — legacy (superseded by llama_packer)