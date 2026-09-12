# Backend: vLLM

Serves safetensors / HF-repo models via vLLM — chat, embeddings and rerank.

- **Engine:** vLLM · **Transports:** host, podman, docker.
- **Registry names:** `vllm` (host), `vllm-podman`, `vllm-docker`.
- **Roles:** `chat`, `embeddings` (`--task embed`), `rerank` (`--task score`).
- **Formats:** `.safetensors`, `hf_repo` (serve straight from a repo id,
  resolved offline from the mounted HF hub).
- **Proxied:** no — llama-swap manages the process; container entries add an
  explicit `proxy` + `cmdStop`/`unloadTimeout` (see the transport docs).

## Command shape

```
vllm serve --model <ref> --served-model-name ${MODEL_ID}
  --host 0.0.0.0 --port <${PORT} | container_port>
  --max-model-len <ctx> --gpu-memory-utilization <0..1>
  [--max-num-seqs <parallel>]        # omitted when parallel <= 0 (uncapped)
  [--max-num-batched-tokens <batch>]
  [--quantization <q>] [--moe-backend <b>] [--mamba-* ...]
  [--kv-cache-dtype fp8|nvfp4]       # from cache_type; q8_* -> fp8
  [--task embed|score]
  [--chat-template <ref>]
  [--speculative-config <json>]      # chat only
  [--enable-auto-tool-choice --tool-call-parser <p>] [--reasoning-parser <p>]
```

## Recipe keys (opt-in, verbatim)

Per-model sidecar keys — vLLM auto-detects most from the checkpoint, so these
are only emitted when declared:

| Key | Flag |
|---|---|
| `vllm_quantization` | `--quantization` |
| `moe_backend` | `--moe-backend` |
| `mamba: {backend, ssm_cache_dtype, stochastic_rounding, philox_rounds, cache_mode}` | `--mamba-*` |
| `tool_call_parser` | `--enable-auto-tool-choice --tool-call-parser` |
| `reasoning_parser` | `--reasoning-parser` |
| `speculative_config` | `--speculative-config` (JSON, verbatim) |

`reasoning_parser` is distinct from llama.cpp's `reasoning-format`, which is
stripped from vLLM command lines.

## Container launch (docker/podman)

The engine emits paths; the container transport translates them. Relevant
profiles.yaml keys:

- `vllm: {image, bin, container_port, hf_cache, container_vendor,
  container_args, gpu_mem_util}`
- `hf_cache` is the host **HF_HOME root** (the dir containing `hub/`),
  bind-mounted at `/root/.cache/huggingface`; entries run with
  `HF_HUB_OFFLINE=1`, so models must be pre-staged.
- Per-model `vllm_image:` overrides the image.
- `container_args` (legacy `docker_args`/`podman_args`) replaces the
  device-flag set entirely.

See [docker](../transports/docker.md) and [podman](../transports/podman.md).

## Availability and selection

A model infers vLLM when its format is `.safetensors`/`hf_repo` and the pair's
resources are configured: `vllm_bin` (host) or `vllm_image` + the runtime on
`PATH` (containers). Preference is `vllm` > `vllm-podman` > `vllm-docker` over
the enable list.

## Memory estimation

HF-repo / safetensors models are sized by `vllm-memory-estimator` (with a
safetensors fallback) — see [`vllm_estimate.py`](../../llama_packer/vllm_estimate.py).

## Limitations

- LoRA is not wired into vLLM's module registry; declared `loras` warn and are
  skipped.
- A GGUF `speculative:` companion cannot be loaded by vLLM — use
  `speculative_config:` with a draft HF repo.
- `cache_type` maps only to `fp8`/`nvfp4`/auto; sub-byte block quants warn and
  serve at auto.

## See also

- [`SPEC.md` → vLLM Backend](../../SPEC.md#vllm-backend)
- [docs/plans/vllm-gb10.md](../plans/vllm-gb10.md) (GB10/Blackwell notes)
