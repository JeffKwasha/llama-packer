# AGENTS.md — Model Sidecars

Write one `.md` sidecar per model. `llama-packer` reads them to generate the
llama-swap `config.yaml`; agents read the same fields to choose among models.

## Naming and layout

### Sidecar filename

`{FAMILY}{VERSION}-{SIZE}-{ETC}.md` — e.g. `qwen3-30b-a3b-nail.md`,
`gemma4-12b.md`. Allowed characters: `[A-Za-z0-9._-]`. The stem doubles as the
default `model_id`, so write it the way you want the id to read. It must match
the model file next to it; if the name can't be shared (e.g. HF snapshot
names), point at the file with `model:`.

### Directory layout

A sidecar is YAML frontmatter (between `---` lines) in a `.md` file whose stem
matches the model file next to it. The directory sets the role:

| Directory | Role | Notes |
|-----------|------|-------|
| root, `chat/` (`t2t/`) | chat | text-output model |
| `vision/` | chat | + colocated `*mmproj*.gguf` companion |
| `doc/` (`ocr/`) | chat | OCR/extraction |
| `embed/` | embeddings | nested dirs keep the role |
| `rerank/` | rerank | |
| `img/` | image | sd-server; opt-in via profiles.yaml `dirs:` |
| `s2t/` | s2t | whisper-server; opt-in; `.bin` needs an authored sidecar |
| `t2s/` | t2s | kokoro-podman; opt-in; sidecar only needs `hf_repo: hexgrad/Kokoro-82M` |
| (any) `<name>.safetensors` | by dir or `role:` | vLLM |
| (any) `<name>.md` | — | sidecar for the model file above |

Orphan files under `embed/`/`rerank`/`doc/` get empty stub sidecars automatically
(the role comes from the directory, not the stub). Whisper `.bin` models are
never stubbed — write the sidecar yourself. Other subdirs (`misc/`, `tmp/`,
`hf_hub/`, … — and `img/`, `s2t/`, `t2s/` when not opted in) are not served —
extend via profiles.yaml `dirs:` (skipped dirs are listed in the run log). A
`.modelignore` at a models root excludes files/subtrees in place (one glob per
line, `#` comments).

### Companions

Companions sit next to their parent (or in the same HF snapshot) and are
referenced by filename from the parent sidecar — they are never main models:

| Companion | Field |
|-----------|-------|
| `*mmproj*.gguf` (vision projection) | `mmproj: <file>` |
| `*.mtp.gguf` / `*-mtp.gguf` (MTP draft) | `speculative: <file>` |

Set `mtp: true` when MTP heads are baked into the main GGUF (no companion).

## Classify before you fill

Before filling, confirm what the file actually is — filenames lie, headers
don't. A diffusion tree (several subdirs: `checkpoints/`, `diffusion_models/`,
`text_encoders/`) holds generative assets; everything else is a text model.
How to check:

| Check | Command / method |
|-------|------------------|
| GGUF architecture | `gguf-dump --metadata <f>.gguf \| grep general.architecture` (or read the header: diffusion archs are `flux*`, `sdxl`, `sd3*`, `wan*`, `chroma`, …; LLM archs declare `<arch>.context_length`) |
| Safetensors kind | Diffusion weights use DiT/UNet tensor blocks (`double_blocks.*`, `input_blocks.*`, VAE); LLMs use `layers.*.self_attn`/`k_proj`/`lm_head` |
| HF model card | Locally cached snapshots carry `README.md` with a YAML frontmatter `pipeline_tag:` (`text-generation` vs `text-to-image`/`text-to-audio`) — offline and authoritative |
| Not cached locally | `hf models info <org/repo>` or search hf.co (online) |

Diffusion/image-generation weights placed under a served text role are excluded
from the config with an error in the pack log — move them to the image tree
(`img/` with `dirs: {img: image}` and `backends: [sd-server]`) or set
`ignore: true`. Filenames prove nothing: classify by headers.

## Sidecar format

Frontmatter must start with `---` on line 1 and end with `---`. Only `model_id`
is validated; every other key is optional — a sidecar with no `name:` still
serves (identity falls back to the stem). For an HF-cache GGUF: `parameters`
from the size in the filename (`31B`), `quantization` is the exact suffix after
the last `-` (`...i1-Q4_K_M.gguf` → `Q4_K_M`), `context_length` is the GGUF
header value (`262144` for gemma4 — matches `ls` filename and `README.md`),
`model:` is the snapshot filename, `hf_repo:`/`hf_url:` is the repo id.

```yaml
---
# --- identity ---
name: "Model Name"           # display label — keep ≤30 chars, no quant suffix; opencode shows model_id, not this
model_id: qwen3-30b-a3b-nail # optional — canonical id for logs/tools/swap; defaults to the file stem; MUST be unique
parameters: 12B              # "12B" or MoE "26B-A4B" (total-active); from filename
quantization: Q4_K_M         # exact suffix from snapshot filename: Q4_K_M, Q6_K, IQ1_S …
context_length: 262144       # architectural max (from GGUF header; gemma4 = 262144)
description: "one-line summary."
# --- file & huggingface ---
model: model.gguf            # snapshot filename when the file lives in the HF cache (with hf_repo:)
hf_repo: org/model           # HF cache repo id — required with model: for cache files
hf_url: https://huggingface.co/org/model  # alternative to hf_repo; keep on one line
# mmproj: model-mmproj.gguf  # only if the snapshot actually contains *mmproj*.gguf
# speculative: model.mtp.gguf  # only if the snapshot actually contains *mtp*.gguf
# mtp: true                  # only when MTP heads are baked into the main GGUF
# --- serving ---
role: chat                   # chat (default) | embeddings | rerank | image (sd-server) | s2t (whisper-server) | t2s (kokoro-podman)
# cli_args: "--vae ae.safetensors --lora my.safetensors"  # extra backend flags (unstructured)
# vram_mb: 1280              # fixed-overhead backends (s2t/image/t2s): pin total process VRAM
# image_min_tokens: 1024     # image input: min image tokens/image (dynamic-res archs, e.g. Qwen-VL; needs mmproj)
# image_max_tokens: 4096     # image input: cap image tokens/image (bounds KV cost; unset = model default, can be huge)
# --- agent metadata (optional; passed through) ---
capabilities: [image, video, tools, reasoning, audio, speech]  # image=image input, video=video input (and output for omni/video-arch), audio=input, speech=output; image role → in:[text,image] out:[image] or out:[video] if video
freethought: 0.7             # 0.0 = refuses; 1.0 = reasons about anything
license: apache-2.0
base_model: gemma-4-31b-it   # the specific original model a finetune derives from — Dirk → qwen3.8-27b; official releases name themselves. Finetunes share most parameters with their base model and mmproj are usually compatible. A slug, not a repo path
family: gemma4               # architectural generation (see Family registry below) — optional, open field, not validated by llama-packer
architecture: gemma-4        # backend architecture class; informs compatibility (e.g. qwen3, qwen3-vl, flux, sdxl, wan, hunyuan-video, h3, mochi, omni)
finetune: instruct
type: instruct               # descriptive label; type: embedding|rerank|image hints role
mtp_accuracy: 0.9            # MTP draft acceptance; feeds throughput estimate
strengths: ["bash tool calling"]
weaknesses: ["slow on 32GB"]
# --- reasoning (chat + reasoning capability only) ---
# reasoning-format: deepseek  # none | deepseek | deepseek-legacy | auto
# reasoning-preserve: true    # keep thinking trace in history
# --- sampling / precision (usually via profiles.yaml, not here) ---
# cache_type: q8_0
# parallel: 1
# default_mode: instruct
# modes: { instruct: {temperature: 0.6, pres_pen: 1.5} }
---
```

Sampling parameters (`temperature`, `min_p`, `top_p`, `top_k`, `pres_pen`) and
most precision/serving knobs are **fleet-level** — set them in `profiles.yaml`
defaults/profiles or a directory `models.yaml`, not per-sidecar. The sidecar is
for identity, file resolution, and what the model *is*; the fleet config is for
how it *behaves*.

Do not invent keys. `template:` is invalid (use fleet-level `chat_template:`);
`backend:` is fleet-level and inferred from file type — don't set it unless
pinning. `mmproj:` is a companion filename, not a capability: declare
`capabilities: [image]` (or `[image, video]`) explicitly. `vision` is not a
capability — use `image`/`video`. Unknown keys pass through as `metadata` but
log a warning.

## Family

Optional open field — not validated by llama-packer. Tracks architectural
generation, not brand: third-party builds take their architecture's family
(Ornith 1.5, MiniCPM-V 4.6 → `qwen3.5`). Used for override conditions,
client-side browsing, and compatibility grouping (mmproj sharing, fit-matrix
substitution, per-generation defaults).

Standard names (extend as needed): `gemma3`, `gemma4`, `qwen3` (incl. Qwen3-VL),
`qwen3.5` (Qwen3.5/3.6/3.8 — DeltaNet hybrid attention; the 180B offload
variant gets its own name), `glm4`, `laguna2`, `muse-glimmer`, `lfm2`.

Convention: brand + major version; minor only when it changed the architecture
(`qwen3.5`). Skip rather than guess if the architecture is unknown.

## IDs and uniqueness

* `model_id` (or `id`) is the identifier shown in logs, `llama-swap`
  `/v1/models`, and the model picker. Allowed characters: `[A-Za-z0-9._-]`.
  When omitted it defaults to the file stem.
* It must be **unique** fleet-wide — a stem that collides with an explicit
  `model_id` fails the build. Keep it short: `{FAMILY}{VERSION}-{SIZE}-{UNIQUE}`,
  e.g. `qwen3-30b-nail`, `gemma4-12b`, `ornith-15-9b`, `glm47-flash`.
* **No quantization** in the id (`Q4_K_M`, `Q8_0`, `IQ4_K_XS`, `FP8`, `E4M3`,
  `BF16`, …) — it is already in `quantization:`.
* `name:` is an optional display label (≤30 chars); it never affects the id.
  Prefer a pretty variant of `model_id` (`GLM-4.7 Flash` for `glm47-flash`).

## Roles and directories

`role:` selects how a model is served: `chat` (default), `embeddings`, `rerank`,
`image`, `s2t`, `t2s`. It is inferred from the directory (table above) or from
`type:` containing `embedding`/`rerank`/`image`; set `role:` to override.
`image`/`s2t`/`t2s` backends are opt-in via profiles.yaml (`dirs:` +
`backends:`). See SPEC.md for the full map.

## HuggingFace resolution

For a hub-cached model, set `model:` to the exact snapshot filename (get it
with `ls <snapshot>/`) and `hf_repo: org/repo` (or `hf_url:`). No symlink is
needed. If the sidecar can't share the model's stem, `model:` is how you point
at a differently-named file.

## Fleet-level overrides

Per-directory and global overrides — `chat_template`, `chat_template_kwargs`,
`context_length`, and sampling params — live in a directory `models.yaml` and
`profiles.yaml`, not in the sidecar. See SPEC.md.

## Other fields

| Key | Meaning |
|-----|---------|
| `model: <file>` | Model file when stem differs; with `hf_repo:` it names the snapshot file |
| `device: N` / `device: cpu` | Pin to GPU N or run on CPU |
| `concurrency: N` | Per-model concurrency limit |
| `allow_profiles: [...]` | Restrict which sampling profiles apply (list, regex, or false) |
| `reasoning-format` / `reasoning-preserve` | Chat + reasoning only (see above) |
| `cache_type` / `parallel` | KV-cache precision / parallel slots (chat); fixed-overhead roles (image/s2t/t2s) use a fixed budget — `vram_mb` overrides |
| `vram_mb` | Fixed-overhead backends (s2t/image/t2s): pin total process VRAM (e.g. measured via nvidia-smi); wins over the file-size + buffer estimate |
| `mtp_spec_type` / `mtp_draft_n_max` | Override MTP spec type / max draft tokens (defaults `draft-mtp` / 2) |
| `image_min_tokens` / `image_max_tokens` | Image input (mmproj) only, dynamic-resolution archs (Qwen-VL family): floor/cap on image tokens per image, emitted as `--image-min-tokens`/`--image-max-tokens`. Qwen math: 1 token ≈ 28×28 px (2.5-VL) / 32×32 px (3-VL); 1024 tokens ≈ 1 MP — good floor for art/artifact critique. Gemma/SigLIP is fixed ~256 tokens/image: keys are ignored there (warned). The cap also floors the solved context (parallel × max tokens must fit `-c`) |
| `speculative_config: {...}` | vLLM `--speculative-config` JSON verbatim |
| `ignore: true` | Skip this model entirely |
| `fit-params:` | Auto-written by llama-packer — do not edit |

See `SPEC.md` for the complete schema.
