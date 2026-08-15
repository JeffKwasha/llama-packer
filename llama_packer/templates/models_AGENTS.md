# AGENTS.md — Model Files Reference

This directory holds the model files that `llama-packer` turns into a
`config.yaml` for llama-swap. Write `.md` sidecar files that describe the
models so agents can choose between them intelligently.

> This file was auto-written by `gen-config.py` (bundled template). It is
> never overwritten once it exists — edit it freely to record this
> directory's own layout (which files are here, what each is for). The next
> run skips it if present; pass `--agents` to write it when missing.

## What llama-packer reads here

`gen-config.py` discovers models from three file kinds and classifies them:

| File | Role |
|------|------|
| `<name>.gguf` | A language model served by `llama-server` |
| `<name>.safetensors` | An HF-format model served by the `template: vllm-docker` backend |
| `<name>.md` | YAML-frontmatter sidecar describing the model next to it |

Sidecars drive everything: which models exist, how they are served, what
sample parameters they get, which companions (mmproj/MTP) attach, and what
`metadata` agents see. Any frontmatter key the builder does not consume
passes through to clients, so adding descriptive fields needs no code change.

## DO NOT generate sidecars for these

### `embed/` directory — Text Encoders / Extractors

These are **NOT language models**. They are text encoders/feature extractors
used for embeddings, classification, or RAG pipelines. Do NOT create `.md`
sidecar files for them. Do NOT add them to the `llama-swap` config.

- `embed/<encoder-file>.gguf` — any text encoder / feature extractor

### mmproj files — Vision Projection Models

These are **NOT standalone models**. They are multimodal projection layers
that attach to a parent model. Do NOT create standalone entries for them.

- `<parent>-mmproj-F16.gguf` — projects vision embeddings for `<parent>`

Reference these from the parent model's sidecar via the `mmproj:` frontmatter
field. Discovery treats them as companions (role `mmproj`), never as main
models, and the builder auto-adds the `vision` capability when present.

### fit-params (auto-computed)

Sidecars may contain a `fit-params:` nested block written by `llama-packer`
(`llama-fit-params` VRAM measurements). Not a manual authoring concern — do
not edit; it is invalidated automatically when cache type / parallel change.

### MTP companion files — Speculative Decoding Drafts

These are **NOT standalone models**. They are draft models for MTP
speculative decoding. Do NOT create standalone entries for them.

- `<parent>.mtp.gguf` (or a stem containing `-mtp`) — draft model for `<parent>`

Reference these from the parent model's sidecar via the `speculative:`
frontmatter field, or set `mtp: true` when the draft heads are baked into the
main GGUF.

### Other non-model files

- `.safetensors.index.json`, `tokenizer*`, `*.json`, README stubs, or any file
  that is not a model you want served. Do not sidecar these.

> When in doubt: only write a sidecar for files intended to be directly served
> as chat / embeddings / rerank models.

## Main Models — Safe to generate sidecars

Organize main models by:

- family: gemma (google), qwen (alibaba), mistral, llama (meta), hy (tencent),
  mimo (xiaomi), glm (zai), kimi (moonshot), deepseek, nemotron (nvidia),
  grok (xai), stepfun, minimax, ...
- size
- specialization: coder, reasoning, creative writing, ...
- quantization
- max context window

## Sidecar conventions

Sidecar `.md` files live alongside their model file and use YAML frontmatter
(between `---` lines). **Any field you add flows through to clients
automatically** (pass-through-by-default) — the system is built for
flexibility as new models arrive, so agents can choose models intelligently.

```yaml
---
name: "Model Name"
parameters: 12B              # or "26B-A4B" for MoE (total-active) so throughput uses active params
context_length: 262144
quantization: Q4_K_XL
hf_url: https://huggingface.co/org/model
mmproj: model-mmproj.gguf    # if vision companion exists (auto-adds `vision` capability)
speculative: model.mtp.gguf  # if MTP companion exists
mtp: true                    # if MTP is baked into the main GGUF (vs a companion file)
description: "a decent general model with reasoning."
role: chat                  # chat | embeddings | rerank (default chat; see below)
# --- backend selection ---
template: vllm-docker       # OPTIONAL: serve with vLLM in a container instead of llama-server
hf_repo: org/model          # for vLLM: HF repo id (optional; parsed from hf_url when absent)
vllm_image: vllm/vllm-openai:latest   # optional per-model image override
# --- agent-selection metadata (all optional; exposed to clients) ---
capabilities: [vision, tools, reasoning, audio]  # vision auto-added if mmproj present
freethought: 0.7            # 0.0 = refuses 'distasteful' topics; 1.0 = reasons about anything
license: apache-2.0         # -> general.license (llama-server /v1/models meta)
base_model: llama-3         # -> general.basename
finetune: instruct          # -> general.finetune
type: instruct              # -> general.type (descriptive; NOT the role)
mtp_accuracy: 0.9           # MTP draft acceptance (float); feeds throughput estimate
default_mode: instruct      # default sampling mode (maps to the bare ${MODEL_ID} profile)
modes:                      # full per-mode sampling profiles (replaces global profiles.yaml
  instruct:                 #   overrides for THIS model); llama.cpp names: temperature, top_p,
    temperature: 0.6        #   top_k, min_p, pres_pen, repeat_penalty, freq_pen. Models with
    pres_pen: 1.5           #   1 or 2 modes simply omit the others.
  thinking:
    temperature: 1.0
    pres_pen: 0.0
strengths:                  # concise task phrases for agents to match on
  - "bash tool calling"
  - "low context usage"
weaknesses:
  - "slow on 32GB"
---
```

## Global defaults

`profiles.yaml` (bundled) provides global sampling defaults. A sidecar
`modes:` block replaces those defaults for that model; `default_mode` picks
which mode is the bare `${MODEL_ID}` profile. Sampling values use llama.cpp
parameter names: `temperature`, `top_p`, `top_k`, `min_p`, `pres_pen`,
`repeat_penalty`, `freq_pen`.

## Backends and roles

`role` (chat | embeddings | rerank) selects how the model is served —
embeddings and rerank run under llama-server with `--embedding` / `--rerank`
flags; chat is the default and serves the OpenAI /llm text endpoint. Serve a
chat model with vLLM instead of llama-server by adding `template: vllm-docker`
(plus optional `hf_repo` / `vllm_image`). The emitted entry is a `docker run
... vllm serve` command; vLLM serves safetensors, and image precedence is
per-model `vllm_image:` > `--vllm-image` CLI > `vllm.image` in profiles.yaml >
built-in default.

### Role vs type

`role` and `type` are different things:

- `role` (chat | embeddings | rerank) determines serve behavior. It is
  inferred automatically: a sidecar under `embed/` is treated as `embeddings`,
  under `rerank/` as `rerank`. Declare `role:` explicitly only to override
  that inference.
- `type` is a separate, descriptive label (e.g. `instruct`, `embedding`) that
  flows through to clients as `general.type` — it does NOT determine serving
  behavior. Do not use `type` to mean role.

Do NOT set `mmproj:` or `speculative:` for files in `embed/`.
Do NOT set `mtp: true` for files that don't have MTP layers.
Do NOT set `template: vllm-docker` on a GGUF — vLLM serves safetensors/HF
format; a GGUF's correct backend is the default llama-server one.

## Capabilities and derived fields

`capabilities` drives the derived `modalities` (text + image/audio) shown to
clients. `freethought` is also injected into llama-server's `/v1/models`
`meta` via `--override-kv`. The builder computes `capabilities.context` from
the measured VRAM budget (do not hardcode it) and a relative
`throughput_factor` from active parameters / quantization / MTP.

## Other sidecar controls (llama-packer consumed)

| Key | Meaning |
|-----|---------|
| `device: N` | Pin to GPU N (`CUDA_VISIBLE_DEVICES=N` / `ROCR_VISIBLE_DEVICES=N`) |
| `concurrency: N` | Per-model concurrency limit |
| `spare: 512M` | Extra VRAM reserve for this model (overrides global `--spare`) |
| `allow_profiles: [...]` | Restrict which sampling profiles apply |
| `ignore: true` | Skip this model entirely |
| `reasoning: auto` | Multi-variant generation |

For the full schema reference see the project `SPEC.md`.