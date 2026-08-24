# llama-packer INDEX

Generate llama-swap configs from GGUF/VLLM model metadata. See [README.md](README.md) for the intro and [SPEC.md](SPEC.md) for the schema.

## Entry points

- [`llama_packer/__main__.py`](llama_packer/__main__.py) — CLI entry point (`llama-packer`), VRAM/env resolution, matrix wiring
- [`llama_packer/profiles.yaml`](llama_packer/profiles.yaml) — sampling profiles + `vllm:` backend defaults (bundled)
- [`llama_packer/templates/models_AGENTS.md`](llama_packer/templates/models_AGENTS.md) — bundled `models/AGENTS.md` guide (written with `--agents`)

## Modules

- [`llama_packer/model.py`](llama_packer/model.py) — `Model` sidecar parsing, field accessors (`backend`, `hf_repo`, `vllm_image`, `modes`, `role`, `chat_template`, ...), companion resolution
- [`llama_packer/profiles.py`](llama_packer/profiles.py) — `Profiles` value object: defaults, spare precedence, `allow_profiles` filtering, per-model variant grouping
- [`llama_packer/writer.py`](llama_packer/writer.py) — `build_config` = filter → `Planner` (variants, mmproj keep/drop + always-on `-text` variant, renamed auto-dropped main, matrix solve, bounded ctx) → `emit_config` (llama-swap entries with bare `<id>` = vision / `<id>-text` = text-only), `_filter_supported`, `write_yaml`
- [`llama_packer/backends/`](llama_packer/backends/) — backend package: `base` (ABC + support matrix + `is_available`), `llama_server`, `vllm` (host + docker); `BACKENDS` registry, `infer_backend`, `VLLM_BACKENDS`, `get_backend`
- [`llama_packer/overrides.py`](llama_packer/overrides.py) — pattern-scoped override rules (global profiles.yaml + directory-scoped `models.yaml`, inner scope wins) → backend/chat-template/lora/hf_repo/cli_args; format-based backend inference
- [`llama_packer/vram.py`](llama_packer/vram.py) — `VramBudget` fit-params, `solve_matrix_ctx`
- [`llama_packer/vllm_estimate.py`](llama_packer/vllm_estimate.py) — vLLM memory estimation via `vllm-memory-estimator` (+ safetensors fallback)
- [`llama_packer/hardware.py`](llama_packer/hardware.py) — VRAM detection, `GpuProfile`, family handlers
- [`llama_packer/utils.py`](llama_packer/utils.py) — `VLLM_DEFAULT_*`, sampling keys, `_KV_CACHE_BYTES`, discovery/slugify/params, role-mapped recursive discovery (`_DEFAULT_DIR_ROLES`, `classify_models`, `dir_role_map`), HF hub snapshot resolution (`hf_hub_cache`, `hf_snapshot_file`), path-macro grouping (`compute_env_prefixes`, `hf_cache_root`)

## Backends

- llama-server — GGUF chat/embeddings/rerank; role flags, MTP, mmproj, chat-template, LoRA
- vLLM — safetensors / `hf_repo`; all roles (`--task embed`/`--task score` for pooling); `vllm serve` (host binary)
- vLLM docker — same, wrapped in `docker run` with bind-mounts for chat-template/lora dirs; per-model `vllm_image:` override
- Backend selection: sidecar/override `backend:` wins (validated against profiles.yaml `backends:` enable list); else inferred from file format + roles, gated by the enable list and configured resources (see SPEC.md "Backend Selection")
- See SPEC.md "vLLM Backend" + "Override Rules" and [docs/plans/vllm-gb10.md](docs/plans/vllm-gb10.md)

## Docs

- [README.md](README.md) — usage, sidecar example
- [SPEC.md](SPEC.md) — model metadata schema, sampling modes/aliases, vLLM backend, health-check/env/matrix
- [docs/architecture.md](docs/architecture.md) — component ownership, plan→emit pipeline, invariants, testing seams
- [docs/gguf_model_analysis.md](docs/gguf_model_analysis.md) — GGUF sizing notes
- [docs/plans/](docs/plans/) — design proposals (vllm-gb10)
- [docs/reference.md](docs/reference.md) — external links

## Data dirs

- `models/` — GGUF + `.md` sidecars (`chat/` `vision/` `doc/` `embed/` `rerank/` by use-case; `img/` etc. ignored; optional per-directory `models.yaml`; see SPEC.md "Model Discovery"); `AGENTS.md` guide auto-written with `--agents` (from bundled `llama_packer/templates/models_AGENTS.md`) if missing
- `/mnt/ai/models` — canonical model root (`t2t/` legacy → `chat/`, `vision/`, `doc`/`ocr/`, `embed/`, `rerank/`; `img/` etc. ignored); configured via profiles.yaml `models_dirs:` + `dirs:` / `hf_home:`
- `profiles.yaml.example` — tracked template; the live `profiles.yaml` is machine-local (gitignored)
- `llama-b*/` — llama.cpp builds (used via `find_bin_dir`)
- `model_cfg/` — legacy (superseded by llama_packer)