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
# mmproj: mmproj-...gguf                 # optional companion, same snapshot
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
