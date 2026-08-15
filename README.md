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


## See also

- [SPEC.md](SPEC.md) — detailed configuration specification
- [profiles.yaml](profiles.yaml) — sampling profile definitions
