# vLLM / DGX Spark backend — design proposal

**Status:** in progress — docker-vLLM scaffold implemented on `main`; recipe keys
+ container correctness on `gb10-support` (see `docs/plans/gb10-support-plan.md`)
**Date:** 2026-08-15 (updated 2026-09-09)
**Branch:** `gb10-support` (supersedes the stale `vllm-gb10` scaffold branch)
**Primary backend:** llama-swap / llama.cpp (unchanged). vLLM is a second backend
*inside llama-swap* for the DGX Spark, not a separate toolchain.

## Decision (2026-08-15)

Earlier research proposed emitting eugr `spark-vllm-docker` recipes as a separate
output. That approach is **dropped**. Instead:

- **llama-swap is the orchestration layer on the Spark too.** It natively manages
  any OpenAI-compatible server, including vLLM (`docs/configuration.md`: *"llama-swap
  supports any OpenAI API compatible server"*; `config.example.yaml` ships a dockerized
  vLLM entry with `evict_costs: v: 50 # vllm backend, slow cold start`).
- This preserves the value we already built: on-demand loading of multiple models,
  per-request aliases/modes via `filters.setParamsByID`, matrix routing, health checks,
  per-model metadata, `/v1/models` listing.
- The two hard problems have concrete solutions, not blocked research:
  1. **Safetensor memory sizing** — `vllm-memory-estimator` (ashishkamra/vllm-memory-estimator,
     CPU-only, imports vLLM's `ModelConfig`/`KVCacheSpec`, reads HF `config.json` +
     safetensors headers). Its `estimate`/`budget --json` acts as a fit-params analog:
     model_mib + ctx factor + concurrency to feed the existing `solve_matrix_ctx`.
  2. **MTP** — vLLM's MTP is native speculative decoding (no companion draft GGUF).
     Supported for Qwen3-Next / Qwen3.5; DeepSeek-style under V1 only. Translate sidecar
     `mtp`/`speculative` to `--num-speculative-tokens` etc.

## Scope so far (implemented on `vllm-gb10`)

### vLLM docker backend scaffold

A chat model opts into vLLM via an `overrides:` rule that sets `backend: vllm-docker`
(see SPEC.md "Override Rules"):

```yaml
# profiles.yaml
overrides:
  - when: {name: 'qwen3-30b'}
    backend: vllm-docker
    hf_repo: Qwen/Qwen3-30B-A3B-Instruct   # optional; derived from hf_url when absent
    vllm_image: vllm/vllm-openai:v0.11.0   # optional per-model image
```

Emitted entry (built by `VllmDockerBackend` in `llama_packer/backends/vllm.py`):

```yaml
models:
  qwen3-30b:
    cmd: |
      docker run --init --rm {{docker_args}} --name ${MODEL_ID}
        -v {{models_dir}}:/models -p ${PORT}:{{container_port}} {{vllm_image}}
        --model <hf_repo> --served-model-name ${MODEL_ID}
        --host 0.0.0.0 --port {{container_port}}
        --max-model-len {{ctx_size}} --gpu-memory-utilization {{gpu_mem_util}}
    filters: {setParamsByID: {qwen3-30b: {...}, "qwen3-30b:coder": {...}}}
```

`cmd` is composed per-backend; backend selection is driven by override rules
(last-match-wins per setting key). Aliases/modes, metadata, capabilities, matrix
all flow through unchanged.

### Image specification

Precedence, highest to lowest:

1. Per-model `vllm_image:` frontmatter (`model.vllm_image`)
2. `--vllm-image` CLI flag
3. `vllm.image` in `profiles.yaml` (`vllm:` section)
4. Built-in default `vllm/vllm-openai:latest` (`utils.VLLM_DEFAULT_IMAGE`)

`profiles.yaml` `vllm:` section also configures `docker_args`, `container_port`,
`gpu_mem_util`:

```yaml
vllm:
  image: vllm/vllm-openai:latest
  docker_args: "--runtime=nvidia --gpus all --shm-size=16g"
  container_port: 8000
  gpu_mem_util: 0.9
```

Model resolution: `hf_repo` frontmatter wins; else parsed from `hf_url`
(`huggingface.co/{owner}/{repo}`); else the local GGUF path. vLLM serves
safetensors, so the GGUF fallback is a last resort.

### Files changed

- `llama_packer/backends/` — backend package (`base`, `llama_server`, `vllm`) composing
  per-engine `cmd`s; `VLLM_DEFAULT_*` constants remain in `utils.py`
- `llama_packer/overrides.py` — pattern-scoped `overrides:` rules select backend/hf_repo
- `llama_packer/writer.py` — `_build_entry` delegates to the selected backend + `_strip_repeat_ws`
- `llama_packer/model.py` — `backend`, `hf_repo`, `vllm_image` properties
- `llama_packer/__main__.py` — `--vllm-image` flag + precedence resolution
- `llama_packer/profiles.yaml` — `vllm:` defaults section
- `README.md`, `SPEC.md` — documented
- `docs/plans/vllm-gb10.md` — this file

## Implemented (since scaffold)

- **vLLM memory estimator** (`llama_packer/vllm_estimate.py`) — `vllm-memory-estimator`
  (optional, Python API) produces `model_mib`/`ctx_factor`/`compute_mib` for vLLM models,
  falling back to the local `.safetensors` header estimate, feeding the existing
  `FitParams`/`calc_ctx`/`solve_matrix_ctx` pipeline unchanged. `--gpu-memory-utilization`
  is derived from the same reserve/spare budget llama.cpp uses.
- **Direct binary mode** — `backend: vllm` emits `vllm serve` (no docker); `vllm-docker`
  stays selectable. Binary resolved via `--vllm-server` > `vllm.bin` > `vllm` on PATH.
- **`hf_repo`-only models** — `Model.gguf_path` is optional for vLLM backends; a sidecar
  with only `hf_repo`/`hf_url` is valid.

## Implemented 2026-09-09 (branch `gb10-support`)

Per `docs/plans/gb10-support-plan.md`:

- **Recipe keys (C1–C5)** — `vllm_quantization` (`--quantization`), `moe_backend`,
  `mamba:` (five `--mamba-*` flags), `tool_call_parser`, `reasoning_parser`; opt-in,
  verbatim, override-rule capable (`SETTING_KEYS` / `Model.FIELDS` / accessors).
  Closes the Nemotron-3.5-Lightning ground-truth flag gap.
- **Docker path mapping + mounts (C6)** — all path-shaped cmd values (model ref,
  speculative draft, chat template) rewritten to container paths: under `vllm.hf_cache`
  → `/root/.cache/huggingface/<rel>` (whole-root mount, symlink-safe), under a
  `models_dir` → `/models…`, else `/extN` parent bind. Fixes the pre-existing
  local-path `--model` host leak. `HF_HUB_OFFLINE=1` + `HF_HOME` on every docker entry
  (no downloads; pre-stage the hub).
- **Container lifecycle (C7)** — `cmdStop: docker stop ${MODEL_ID}` +
  `unloadTimeout: 30` (llama-swap kb ttl-and-unloading), explicit
  `proxy: http://127.0.0.1:${PORT}` (C9, writing-cmd).
- **healthCheckTimeout (C8)** — vLLM floor at `model_size_mb / 100` (100 MB/s load
  assumption; 300 s floor for repo-only models).

Still open from the old list: MTP/spec translation beyond `speculative_config:` (the
explicit JSON key covers dspark/MTP today).

## Planned (not yet implemented)

- **tensor-parallel / multi-GPU** — emit `--tensor-parallel-size` and size the estimator
  accordingly (currently TP is fixed at 1).
- **update design for matrix evict_costs** on vLLM entries (slow cold starts) and
  higher health check timeout where the computed `hct` is too small.

## Open questions / notes

- Embed/rerank stay on llama.cpp even in vLLM mode (llama.cpp is first-class for them).
- eugr recipe export is out of scope for v1.
- Cluster/multi-node (`--discover`) emission is future work — solo Spark first.

## References

- llama-swap: `docs/configuration.md`, `config.example.yaml` (docker vLLM entry,
  `evict_costs` vLLM note).
- vllm-memory-estimator: ashishkamra/vllm-memory-estimator (CPU-only, vLLM-coupled).
- vLLM MTP docs: native speculative decoding (Qwen3-Next/Qwen3.5), V1 for DeepSeek.
- eugr/spark-vllm-docker: recipe system **not used** as the output format.