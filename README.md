# llama-packer

Generate configs for [llama-swap](https://github.com/mostlygeek/llama-swap) from GGUF (llama.cpp) or HF (vLLM) model metadata. Takes a directory of models and produces ready-to-run server configurations with optimal flags tuned to your hardware.

Scans model directories, reads YAML sidecar files, detects GPU VRAM, budgets memory across models, and writes `config.yaml` — no manual flag wrangling. Serves models with llama-server (GGUF) or vLLM (`template: vllm` for the host binary, `template: vllm-docker` for a container). Both backends share the same accurate context-length budgeting and per-request alias support.

## Install

Python 3.10+ and [uv](https://docs.astral.sh/uv/) (or pip):

```sh
uv tool install .        # installs the `gen-config` command
gen-config --dry-run
```

Or run from source without installing:

```sh
uv run gen-config.py
```

`gen-config` resolves `models/`, `profiles.yaml` (falls back to the bundled default), and `llama-b*/` build dirs relative to the current directory. Point it elsewhere with `--models-dir`, `--profiles`, and `--llama-server` (or `LLAMA_BIN_DIR`).

Output goes to `config.yaml` and `config.env` in the current directory.

## What it does

- Discovers models from `.md` sidecar files in `models/`
- Detects GPU VRAM via `amd-smi`, `nvidia-smi`, or `rocminfo`
- Runs `llama-fit-params` to measure per-model VRAM usage
- Calculates optimal context window that fits available VRAM
- Resolves companion files (mmproj, MTP draft models) by fuzzy match
- Applies sampling profiles from `profiles.yaml`
- Assembles llama-swap YAML with per-model metadata, native `capabilities`, and `filters.setParamsByID`
  overrides (aliases like `<model>:<mode>` switch sampling parameters per-request without reload)
- Solves multi-model VRAM budgets for embed/rerank/chat on shared GPUs
- Opts individual models into a vLLM backend with `template: vllm` (host binary) or
  `template: vllm-docker` (container) in the sidecar. vLLM models are memory-estimated
  with `vllm-memory-estimator` (or a local safetensors header) instead of `llama-fit-params`,
  and `--gpu-memory-utilization` is derived from the same reserve/spare budget as llama.cpp.

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

Any key not consumed by the builder passes through to clients — add descriptive fields without code changes.

### Models directory guide (`AGENTS.md`)

Run `gen-config.py --agents` to write a `models/AGENTS.md` guide (from the bundled
template) that explains what llama-packer reads and the sidecar conventions. It is
written only when missing — your edits are never overwritten — and failure to write
is logged without aborting. The bundled source is
[`llama_packer/templates/models_AGENTS.md`](llama_packer/templates/models_AGENTS.md);
edit a copy in your models dir rather than the template to record your own layout.

### vLLM backend

Serve a model with vLLM instead of llama-server by declaring the backend in its sidecar.
`template: vllm` runs the host binary (`vllm serve`); `template: vllm-docker` runs it in a
container. `hf_repo` is optional — when absent it is parsed from `hf_url`:

```yaml
---
name: my-qwen3
template: vllm              # or vllm-docker
hf_repo: Qwen/Qwen3-30B-A3B-Instruct    # optional; derived from hf_url when omitted
vllm_image: vllm/vllm-openai:v0.11.0    # optional per-model image (vllm-docker only)
context_length: 65536
---
```

The emitted entry runs `vllm serve --model <repo> --served-model-name ${MODEL_ID}
--max-model-len <ctx> --gpu-memory-utilization <fraction>`, published to llama-swap's
`${PORT}`. Context is budgeted against the same VRAM pool as llama.cpp, and
`--gpu-memory-utilization` is derived from that budget (override via `vllm.gpu_mem_util`
in profiles.yaml). Image precedence (vllm-docker): per-model `vllm_image:` > `--vllm-image`
CLI > `vllm.image` in profiles.yaml > built-in default. Binary path (vllm): `--vllm-server`
CLI > `vllm.bin` in profiles.yaml > `vllm` on PATH.


## See also

- [SPEC.md](SPEC.md) — detailed configuration specification
- [profiles.yaml](profiles.yaml) — sampling profile definitions

## Limitations

- vLLM memory sizing needs `vllm-memory-estimator` installed for HF-repo models; without it
  (or a local `.safetensors` file), context falls back to the declared `context_length` and
  vLLM's own startup profiling bounds the allocation
- MTP/speculative flags are not translated for the vLLM backend yet
- Runs one vLLM server per model per image/binary; multi-image or cluster/tensor-parallel provisioning is future work
- Multi-GPU setups are not natively configured; use `CUDA_VISIBLE_DEVICES`/`ROCR_VISIBLE_DEVICES` to pin models to specific devices
- Companion VRAM falls back to file-size estimates when `llama-fit-params` measurement fails

## Future Roadmap

- Support multi-image/tensor-parallel vLLM provisioning
- Translate MTP/speculative sidecar fields to vLLM `--speculative-config`
- Enrich `throughput_factor` with measured server log data (offline parsing)
