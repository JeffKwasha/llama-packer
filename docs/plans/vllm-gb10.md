# vLLM / DGX Spark backend — design proposal

**Status:** in progress — docker-vLLM scaffold implemented on `vllm-gb10`
**Date:** 2026-08-15
**Branch:** `vllm-gb10`
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

A chat model opts into vLLM via sidecar `template: vllm-docker`:

```yaml
---
name: qwen3-30b
template: vllm-docker
hf_repo: Qwen/Qwen3-30B-A3B-Instruct   # optional; derived from hf_url when absent
vllm_image: vllm/vllm-openai:v0.11.0   # optional per-model image
context_length: 65536
---
```

Emitted entry (`llama_packer/utils.py` `_TARGET_TEMPLATES["vllm-docker"]`, used by
`_build_entry` in `writer.py` when the template defines a full `cmd`):

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

`cmd` is resolved with `model.template` (explicit sidecar `template:` wins over role).
Aliases/modes, metadata, capabilities, matrix all flow through unchanged.

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

- `llama_packer/utils.py` — `vllm-docker` target template + `VLLM_DEFAULT_*` constants
- `llama_packer/writer.py` — full-`cmd` template branch in `_build_entry` + `_strip_repeat_ws`
- `llama_packer/model.py` — `hf_repo`, `vllm_image` properties
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
- **Direct binary mode** — `template: vllm` emits `vllm serve` (no docker); `vllm-b-docker`
  stays selectable. Binary resolved via `--vllm-server` > `vllm.bin` > `vllm` on PATH.
- **`hf_repo`-only models** — `Model.gguf_path` is optional for vLLM backends; a sidecar
  with only `hf_repo`/`hf_url` is valid.

## Planned (not yet implemented)

- **MTP translation** — vLLM-mode `mtp`/`speculative` → `--num-speculative-tokens`;
  warn-and-skip for unsupported architectures.
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