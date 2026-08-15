# llama-packer

Generate configs for [llama-swap](https://github.com/mostlygeek/llama-swap) from GGUF model metadata. Takes a directory of models and produces ready-to-run server configurations with optimal flags tuned to your hardware.

Scans GGUF model directories, reads YAML sidecar files, detects GPU VRAM, budgets memory across models, and writes `config.yaml` — no manual flag wrangling.

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
- Opts individual models into a vLLM docker backend with `template: vllm-docker` in the sidecar
  (image from `--vllm-image`, the `vllm:` section of `profiles.yaml`, or a per-model `vllm_image:`
  frontmatter override)

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

### vLLM docker backend

Serve a model with vLLM instead of llama-server by declaring the backend in its sidecar
(`hf_repo` is optional; when absent it is parsed from `hf_url`):

```yaml
---
name: my-qwen3
template: vllm-docker
hf_repo: Qwen/Qwen3-30B-A3B-Instruct    # optional; derived from hf_url when omitted
vllm_image: vllm/vllm-openai:v0.11.0    # optional per-model image
context_length: 65536
---
```

The emitted entry is a `docker run` command using `vllm serve` inside the container,
published to llama-swap's `${PORT}`. Image precedence: per-model `vllm_image:` > `--vllm-image`
CLI > `vllm.image` in profiles.yaml > built-in default.


## See also

- [SPEC.md](SPEC.md) — detailed configuration specification
- [profiles.yaml](profiles.yaml) — sampling profile definitions
