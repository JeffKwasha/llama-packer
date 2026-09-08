# FAQ

## Why don't my HF hub models (`$HF_HOME/hub`) show up?

The packer never scans the HF cache. Discovery walks `models_dirs` only
(e.g. `/mnt/ai/models`); the hub cache is a *resolution source*, not a scan
target. A model is served when a sidecar in a served directory names it.

**Sidecar reference (recommended, no symlinks):**

```yaml
# /mnt/ai/models/chat/Ornith-1.5-9B.md
---
name: Ornith 1.5 9B
parameters: 9B
quantization: Q6_K
model: Ornith-1.5-9B-AD-Q8_0-Q6_K.gguf   # exact snapshot filename
hf_repo: AtomicChat/Ornith-1.5-9B-GGUF   # resolved offline from $HF_HOME/hub
# mmproj:                               # optional companion, same snapshot
#   file: mmproj-...gguf
#   capabilities: [image]
---
```

**Symlink into a served dir:** link the snapshot `.gguf` into e.g.
`chat/`; the next run auto-writes a stub sidecar beside it (dedup by realpath
prevents double-serving).

**Not recommended:** mapping the cache itself via `dirs: {hf_hub: chat}` — one
uniform role for mixed content, and stubs get written inside the cache tree.

Never-servable formats regardless of sidecar: CTranslate2 (faster-whisper),
transformers checkpoints (Qwen3-ASR/TTS), diffusion pipelines (ACE-Step) —
they need different backends entirely.

See also `models_AGENTS.md` ("Hub-downloaded files need no symlink") and
SPEC.md "Model Discovery".

## Why is my small model stuck at low context and `parallel: 1`?

Three separate causes — check in order:

1. **Low context is usually the ceiling, not the budget.** Served context
   never exceeds the GGUF architectural max (`capabilities.context` in the
   entry tells you the ceiling). A 4B quant capped at 40960 in its header
   serves 40960 no matter how much VRAM is free — small weights buy *slots*,
   never more context than the file allows.
2. **`parallel: 1` means the solve skipped the model.** Auto-parallel only
   runs for `role: chat` on llama-server/vLLM with no `parallel:` pin in the
   sidecar/block or any profile. It is on by default; `matrix:
   auto_parallel: false` turns it off fleet-wide.
3. **The floor-ceiling trap.** A tool-calling model defaults to a 131072
   floor; if its max context is below that *and* it has no `context_length:`
   pin or explicit `min_context:`, the floor is unreachable and the model
   silently stays at 1 slot. Fix: set `min_context:` (e.g. half the max) or
   pin `context_length:` in the sidecar.

## I deleted `derived:` from sidecars — do I have to run `--probe-memory`?

No. Probes are **opt-in per-arch calibration** and run only when you pass
`--probe-memory` explicitly. Deleting a `derived:` block triggers the
normal pack-time path: the header-only llama-fit-params trio (~0.6 s × 3
per model — a whole fleet re-measures in minutes, no VRAM used, no server).
What you probably saw was the per-arch note

    estimate uncorrected for arch 'qwen35' (mtp=False) — ctx/fit may
    undercount draft and allocator terms; optional: --probe-memory qwen35 calibrates

which is informational: uncalibrated arches estimate uncorrected (the
estimate errs slightly optimistic about allocator overhead). The
corrections live in the durable machine-local `serve-corrections.yaml`
beside `profiles.yaml` — rows carry the witness `shape` and `ts`. Also:
`llama-packer --remeasure` ignores saved blocks for one run without
touching any files.

## How do I tune `-b` / `-ub`?

They are first-class planning keys, resolved sidecar > profile >
`llama_server:` fleet section > role defaults (chat 2048/512, embed/rerank
4096/512 — the llama.cpp builtins):

```yaml
llama_server:
  ubatch: 512      # per-pass tokens: shapes the compute buffer
  # batch: 2048    # logical batch: throughput only, no VRAM effect
```

or per model in the sidecar (`batch:` / `ubatch:`). Both render explicitly
on every llama-server command; `ubatch` also stamps the VRAM measurement
shape, so a change re-measures automatically. Do **not** put `-b`/`-ub` in
`llama_server.args` or sidecar `cli_args:` — the named keys render after
them and win per flag, and `cli_args` values are invisible to the
measurement (a warning points this out).
