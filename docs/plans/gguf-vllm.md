# Plan: GGUF models on vLLM (deferred)

Status: **proposal — not scheduled**. Recorded 2026-08-24 after vLLM pooling
roles landed (chat/embeddings/rerank via `--task`).

## Why deferred

vLLM's GGUF support is experimental, under-optimized, and lives in an
out-of-tree plugin (`vllm-gguf-plugin`). It covers none of the GGUF roles this
fleet actually runs:

| Capability | llama-server (GGUF) | vLLM + GGUF plugin |
|---|---|---|
| vision / mmproj | yes | no |
| MTP / speculative | yes | no |
| embeddings / rerank | yes (`--embedding`, `--rerank`) | no (pooling tasks are safetensors/HF only) |
| chat quants | yes | experimental |

The only payoff would be throughput experiments on plain chat quants.

## Design when picked up

1. **Opt-in only**: add `.gguf` to the vLLM backends' `formats`, but format
   *inference* keeps `.gguf → llama-server`. A model reaches vLLM solely via
   explicit `backend: vllm`/`vllm-docker` in its sidecar or an override rule
   (and must be enabled by profiles.yaml `backends:`).
2. **Tokenizer field**: upstream strongly recommends serving with the base
   model's tokenizer (`--tokenizer org/base`) because GGUF tokenizer conversion
   is slow/buggy. That repo is not derivable from a quant-repo `hf_repo:`, so
   sidecars need a new field, e.g. `tokenizer: Qwen/Qwen3-0.6B`. Warn when a
   GGUF-on-vLLM model lacks it.
3. **Chat-only**: speculative/mmproj paths already warn-and-skip on vLLM; keep
   that. Pooling roles stay safetensors-only.
4. **Prerequisite note**: host vllm binary or container image must have
   `vllm-gguf-plugin` installed — operator-side, documented not managed.
   Per SPEC Assumptions: target newest stable, no compat shims.

Upstream reference: https://docs.vllm.ai/en/latest/features/quantization/gguf/
