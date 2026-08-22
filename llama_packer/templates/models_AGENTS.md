# AGENTS.md — Model Sidecars

Write one `.md` sidecar per model. `llama-packer` reads them to generate the
llama-swap `config.yaml`; agents read the same fields to choose among models.

## Naming

A sidecar is YAML frontmatter (between `---` lines) in a `.md` file whose stem
matches the model file next to it:

| File | Served as |
|------|-----------|
| `<name>.gguf` | chat model (llama-server) |
| `<name>.safetensors` | chat model (vLLM backend) |
| `<name>.md` | sidecar for the file above |
| `embed/<name>.gguf` | embeddings model |
| `rerank/<name>.gguf` | rerank model |

If the `.md` cannot share the model's stem, point at the file with
`model: <filename>`.

Companions sit next to their parent and are referenced by filename from the
parent sidecar — they are never main models:

| Companion | Field |
|-----------|-------|
| `<parent>-mmproj*.gguf` (vision projection) | `mmproj: <file>` |
| `<parent>.mtp.gguf` / `<parent>-mtp.gguf` (MTP draft) | `speculative: <file>` |

Set `mtp: true` when the MTP heads are baked into the main GGUF (no companion).

## Sidecar format

```yaml
---
name: "Model Name"           # required — a sidecar without it is skipped
parameters: 12B              # or "26B-A4B" for MoE (total-active)
context_length: 262144
quantization: Q4_K_XL
hf_url: https://huggingface.co/org/model
description: "one-line summary."
# --- serving ---
role: chat                  # chat (default) | embeddings | rerank
# Backend is usually NOT declared here: when absent it is inferred from the
# model file (.gguf -> llama-server; safetensors/HF-repo -> vllm-docker).
# Fleet-level choices (backend, chat_template, loras) belong in profiles.yaml
# `overrides:` rules. See SPEC.md "Override Rules".
hf_repo: org/model          # vLLM: HF repo id (parsed from hf_url if absent)
vllm_image: vllm/vllm-openai:latest   # optional per-model vLLM image override
mmproj: model-mmproj.gguf   # vision companion (auto-adds `vision` capability)
speculative: model.mtp.gguf # MTP draft companion file
mtp: true                   # MTP baked into the main GGUF (no companion)
# --- agent metadata (optional; passed through to clients) ---
capabilities: [vision, tools, reasoning, audio]
freethought: 0.7            # 0.0 = refuses; 1.0 = reasons about anything
license: apache-2.0
base_model: llama-3
finetune: instruct
type: instruct              # descriptive label; type: embedding|rerank hints role
mtp_accuracy: 0.9           # MTP draft acceptance; feeds throughput estimate
strengths: ["bash tool calling"]
weaknesses: ["slow on 32GB"]
# --- per-model sampling (replaces global profiles.yaml defaults) ---
default_mode: instruct      # maps to the bare ${MODEL_ID} profile
modes:                      # keys use llama.cpp names: temperature, top_p,
  instruct:                 #   top_k, min_p, pres_pen, repeat_penalty, freq_pen
    temperature: 0.6
    pres_pen: 1.5
  thinking:
    temperature: 1.0
    pres_pen: 0.0
---
```

Any field not listed above still flows through to clients as `metadata`.

## Roles

`role:` selects how a model is served — `chat` (default), `embeddings`
(`--embedding`), or `rerank` (`--rerank`). It is inferred from the `embed/` /
`rerank/` directory or a `type:` of `embedding` / `rerank`; declare `role:`
explicitly to override.

## Other fields

| Key | Meaning |
|-----|---------|
| `model: <file>` | Model file to load when it doesn't match the sidecar stem |
| `device: N` | Pin to GPU N; `device: cpu` runs the model on CPU |
| `concurrency: N` | Per-model concurrency limit |
| `allow_profiles: [...]` | Restrict which sampling profiles apply (list, regex, or false) |
| `reasoning: auto` | Emit one variant per reasoning mode |
| `mtp_spec_type` / `mtp_draft_n_max` | Override MTP spec type / max draft tokens (defaults `draft-mtp` / `2`) |
| `ignore: true` | Skip this model entirely |
| `fit-params:` | Auto-written by llama-packer — do not edit |

See `SPEC.md` for the complete schema.
