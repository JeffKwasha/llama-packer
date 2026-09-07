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
