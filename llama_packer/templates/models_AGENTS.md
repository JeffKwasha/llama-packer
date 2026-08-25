# AGENTS.md — Model Sidecars

Write one `.md` sidecar per model. `llama-packer` reads them to generate the
llama-swap `config.yaml`; agents read the same fields to choose among models.

## Naming

A sidecar is YAML frontmatter (between `---` lines) in a `.md` file whose stem
matches the model file next to it:

| File | Served as |
|------|-----------|
| `<name>.gguf` | chat model at the root, or under `chat/` `t2t/` `vision/` `doc/` |
| `<name>.safetensors` | chat / embeddings / rerank model (vLLM; role from dir or `role:`) |
| `<name>.md` | sidecar for the file above |
| `chat/<name>.gguf` | chat model (canonical; `t2t/` is legacy alias) |
| `vision/<name>.gguf` | chat model + colocated `*mmproj*.gguf` companion |
| `doc/<name>.gguf` | OCR/extraction model (`ocr/` is legacy alias for `doc/`) |
| `embed/<name>.gguf` | embeddings model (nested dirs like `embed/jina-v5/` keep the role) |
| `rerank/<name>.gguf` | rerank model |
| `img/<name>.gguf` | image model (sd-server; opt-in via `dirs: {img: image}` in `profiles.yaml`) |
| `s2t/<name>.bin` | speech-to-text model (whisper-server; opt-in via `dirs: {s2t: s2t}`; requires an authored same-stem `.md`) |
| `t2s/<name>.md` | text-to-speech model (kokoro-podman; opt-in via `dirs: {t2s: t2s}`; weights are baked into the image — `hf_repo: hexgrad/Kokoro-82M` alone suffices) |

Orphan files under `embed/`/`rerank`/`doc/` get empty stub sidecars automatically
(the role comes from the directory, not the stub). Whisper `.bin` models are
never stubbed — write the sidecar yourself. Other subdirs (`misc/`,
`tmp/`, `hf_hub/`, … — and `img/`, `s2t/`, `t2s/` when not opted in)
are not served — extend via profiles.yaml `dirs:` (skipped dirs are listed in
the run log). A `.modelignore` at a models root excludes files/subtrees in
place (one glob per line, `#` comments).

## Directory-scoped `models.yaml`

Any subdirectory may carry a `models.yaml` that applies only beneath it — the
directory is the filter, so per-vendor settings need no regex:

```yaml
# chat/qwen3/models.yaml
defaults:
  context_length: 16384        # fills gaps (sidecar wins; never name/model/ignore)
overrides:
  - when: true
    chat_template: chat_template.jinja       # colocated; resolves relative to each sidecar
    chat_template_kwargs: {enable_thinking: true}
```

Inner scopes beat outer ones beat global `profiles.yaml` `overrides:`.

Hub-downloaded files need no symlink — do not read `llama_packer` source,
this template is sufficient. List the snapshot dir (`ls <snapshot>/`) to get the
exact `.gguf` filename; then in the sidecar set `model: <that filename>` plus
`hf_repo: org/repo` (or a parseable `hf_url: https://huggingface.co/org/repo`).
It resolves from `$HF_HOME/hub` (no symlink). Companions use the same snapshot
when the local search misses. If the `.md` cannot share the model's stem, `model:`
is also how you point at a differently-named file.

Companions sit next to their parent (or in the same HF snapshot) and are
referenced by filename from the parent sidecar — they are never main models:

| Companion | Field |
|-----------|-------|
| `*mmproj*.gguf` (vision projection) | `mmproj: <file>` |
| `*.mtp.gguf` / `*-mtp.gguf` (MTP draft) | `speculative: <file>` |

Set `mtp: true` when MTP heads are baked into the main GGUF (no companion).

## Classifying model types

Directory intent vs file truth: a served directory (`chat/`, `vision/`, …)
means "text-output models live here"; an image tree (e.g. `img/` with
`checkpoints/ diffusion_models/ text_encoders/`) means "generative-media
pipeline assets". llama-packer classifies each *file* by its headers, never
its filename — before writing a sidecar, verify what a file actually is:

| Check | Command / method |
|-------|------------------|
| GGUF architecture | `gguf-dump --metadata <f>.gguf \| grep general.architecture` (or read the header: diffusion archs are `flux*`, `sdxl`, `sd3*`, `wan*`, `chroma`, …; LLM archs declare `<arch>.context_length`) |
| Safetensors kind | Diffusion weights use DiT/UNet tensor blocks (`double_blocks.*`, `input_blocks.*`, VAE); LLMs use `layers.*.self_attn`/`k_proj`/`lm_head` |
| HF model card | Locally cached snapshots carry `README.md` with a YAML frontmatter `pipeline_tag:` (`text-generation` vs `text-to-image`/`text-to-audio`) — offline and authoritative |
| Not cached locally | `hf models info <org/repo>` or search hf.co (online) |

Diffusion/image-generation weights placed under a served text role are excluded
from the config with an error in the pack log — move them to the image tree
(`img/` with `dirs: {img: image}` and `backends: [sd-server]`) or set `ignore: true`. Filenames prove nothing: classify by headers.

## Sidecar format

Frontmatter must start with `---` on line 1 and end with `---`. `name:` is
required — a sidecar without it is skipped. For an HF-cache GGUF: `parameters`
from the size in the filename (`31B`), `quantization` is the exact suffix after
the last `-` (`...i1-Q4_K_M.gguf` → `Q4_K_M`), `context_length` is the GGUF
header value (`262144` for gemma4 — matches `ls` filename and `README.md`),
`model:` is the snapshot filename, `hf_repo:`/`hf_url:` is the repo id.

```yaml
---
name: "Model Name"           # required — use the sidecar stem
parameters: 12B              # "12B" or MoE "26B-A4B" (total-active); from filename
context_length: 262144       # architectural max (from GGUF header; gemma4 = 262144)
quantization: Q4_K_M         # exact suffix from snapshot filename: Q4_K_M, Q6_K, IQ1_S …
hf_url: https://huggingface.co/org/model  # keep on one line
description: "one-line summary."
# --- serving (only what the model needs) ---
role: chat                  # chat (default) | embeddings | rerank | image (sd-server) | s2t (whisper-server) | t2s (kokoro-podman)
model: model.gguf           # snapshot filename when file lives in HF cache (with hf_repo:)
hf_repo: org/model          # HF cache repo id — required with model: for cache files
# mmproj: model-mmproj.gguf   # only if snapshot actually contains *mmproj*.gguf
# speculative: model.mtp.gguf  # only if snapshot actually contains *mtp*.gguf
# mtp: true                 # only when MTP heads are baked into the main GGUF
# cli_args: "--vae ae.safetensors --lora my.safetensors"  # sd-server flags via cli_args (unstructured); also for any backend's extra flags
# --- agent metadata (optional; passed through) ---
capabilities: [vision, tools, reasoning, audio, speech]  # vision/audio = input, speech = output; image role → in:[text,image] out:[image]
freethought: 0.7            # 0.0 = refuses; 1.0 = reasons about anything
license: apache-2.0
base_model: gemma-4          # family slug (gemma-4, qwen3, llama-3) — not a repo path
finetune: instruct
type: instruct              # descriptive label; type: embedding|rerank|image hints role
mtp_accuracy: 0.9           # MTP draft acceptance; feeds throughput estimate
strengths: ["bash tool calling"]
weaknesses: ["slow on 32GB"]
# --- reasoning (chat + reasoning capability only) ---
# reasoning-format: deepseek  # none | deepseek | deepseek-legacy | auto
# reasoning-preserve: true    # keep thinking trace in history
# --- precision / sampling (usually via profiles.yaml) ---
# cache_type: q8_0
# parallel: 1
# default_mode: instruct
# modes: { instruct: {temperature: 0.6, pres_pen: 1.5} }
---
```

Do not invent keys. `template:` is not valid — use `chat_template:` in
`profiles.yaml`/`models.yaml` overrides (fleet-level). Backend (`backend:`) is
also fleet-level and inferred from file type when absent (`.gguf` → llama-server
or sd-server when `role: image`, `.bin` in `s2t/` → whisper-server,
safetensors/`hf_repo` → vllm-docker); do not set it in sidecars unless pinning. Any field not
listed above still flows through as `metadata`.

Image sidecars (sd-server) are **opt-in**: put the diffusion GGUF under `img/`,
set `dirs: {img: image}` and `backends: [llama-server, sd-server]` in
`profiles.yaml`. Speech-to-text (whisper-server) is opt-in the same way:
`s2t/` dir + `dirs: {s2t: s2t}` + `backends: [..., whisper-server]`, with an
authored same-stem `.md` beside each GGML `.bin` (no stubs are generated).
Text-to-speech (kokoro-podman) is opt-in via `t2s/` + `dirs: {t2s: t2s}` —
weights and voices are baked into the container image, so the sidecar only
needs `hf_repo: hexgrad/Kokoro-82M`.
The `proxy` and
`checkEndpoint: /` fields are emitted automatically for proxied entries —
`sd-server`, `whisper-server`, and `kokoro-podman`
(the default `/health` never returns 200 for sd-server — see `docs/plans/comfyui-sd.md`).

## Roles and directories

`role:` selects how a model is served — `chat` (default), `embeddings`,
`rerank`, `image` (sd-server), `s2t` (whisper-server), or `t2s`
(kokoro-podman). Inferred from the top-level directory (`chat`/`t2t`/`vision`/`doc/` →
chat, `embed/` → embeddings, `rerank/` → rerank, `img/` → image and `s2t/` →
s2t and `t2s/` → t2s when opted in; root defaults to chat) or from
`type:` containing `embedding`/`rerank`/`image`; declare `role:` to override. Roots and
role map are configured in profiles.yaml (`models_dirs:` + `dirs:`); the enable
list `backends:` there also gates which backends are usable. See SPEC.md.

## Other fields

| Key | Meaning |
|-----|---------|
| `model: <file>` | Model file when stem differs; with `hf_repo:` it names the snapshot file |
| `device: N` / `device: cpu` | Pin to GPU N or run on CPU |
| `concurrency: N` | Per-model concurrency limit |
| `allow_profiles: [...]` | Restrict which sampling profiles apply (list, regex, or false) |
| `reasoning-format` / `reasoning-preserve` | Chat + reasoning only (see above) |
| `cache_type` / `parallel` | KV-cache precision / parallel slots + VRAM sizing (chat only; image/s2t fixed 512 MiB, t2s fixed 3072 MiB overhead) |
| `mtp_spec_type` / `mtp_draft_n_max` | Override MTP spec type / max draft tokens (defaults `draft-mtp` / 2) |
| `speculative_config: {...}` | vLLM `--speculative-config` JSON verbatim |
| `ignore: true` | Skip this model entirely |
| `fit-params:` | Auto-written by llama-packer — do not edit |

See `SPEC.md` for the complete schema.
