# How vision interacts with context in llama.cpp

This document explains the actual mechanism by which image tokens enter the
LLM context, what they cost in VRAM, and how llama-packer's budget maths
accounts for them. It is model-specific because the answer depends on whether
the model uses a separate projector file (mmproj) or has vision baked into its
main weights.

## The two vision architectures

### 1. Separate projector (mmproj) — LLaVA, Qwen2-VL, Gemma 3

The vision encoder is a **separate GGUF file** (the `.mmproj`). At serve time
llama.cpp loads both the main LLM and the projector. The data flow is:

```
raw image
  → img_tool preprocessing (resize, pad, normalize)
  → clip_encode() — the projector network runs a forward pass
  → projected embeddings
  → injected into the LLM's token sequence at the image's position
  → llama_decode() processes them as ordinary tokens
```

The projector has its own weights and compute cost. llama-packer folds these
into the main model's VRAM budget via `effective_static(include_mmproj=True)`,
which adds the projector's affine constants (model_mib, kv_per_token_mib,
slot_mib, compute_mib) to the main model's. When `include_mmproj=False`, the
projector is not folded in — that VRAM is freed for context.

### 2. Native multimodal — Qwen3-VL, Gemma 3 (some variants)

Vision is **baked into the main model weights**. There is no separate mmproj
file. The vision encoder runs as part of the model's own forward pass. Image
tokens are still ordinary tokens in the KV cache — they count against `n_ctx`
exactly like text tokens — but there is no separate projector whose VRAM can
be freed by shedding it.

llama-packer detects this: `model.mmproj` is `None`, so the
`include_mmproj` axis is a **no-op** (the guard in `effective_static` fails and
nothing is folded in regardless of the flag). Shedding "vision" frees no VRAM
because there is no projector to drop. Both rungs afford identical context, the
tie-break keeps the less-shed level, and vision is kept. This is correct: you
cannot shed what is baked into the weights.

## Image tokens in the context

Regardless of architecture, once the projector (separate or native) has run,
its output embeddings are injected into the LLM's token sequence. From that
point on they are **ordinary tokens**:

- They occupy positions in the sequence.
- They consume KV cache memory: `2 × layers × kv_heads × head_dim × seq_len × bytes`.
- They count against `n_ctx` (the context window).
- They are evicted by the sliding window / cache management like any other token.

The number of image tokens is **proportional to pixel area**, not to image
count. A single 1024×1024 image with a ViT patch size of 14×14 produces
roughly `(1024/14)² ≈ 5329` tokens. A 4K image (3840×2160) produces roughly
`(3840/14) × (2160/14) ≈ 42,196` tokens — far exceeding most context windows.
This is why AnyRes / dynamic-resolution tiling exists: it slices high-res
images into smaller patches to keep the token count manageable.

## The `image_max_tokens` floor

Because image tokens count against context, a model that serves vision must
have enough per-slot context to fit at least one maximum-size image. The
sidecar declares this with `image_max_tokens`. llama-packer enforces it as a
**per-slot floor**: when vision is served, the solved context is never allowed
to drop below `image_max_tokens` (when the budget affords it).

The floor is applied in `_image_floor_tokens`:

```python
def _image_floor_tokens(self, include_mmproj: bool) -> int:
    if not include_mmproj:
        return 0
    if not (self.model.mmproj and self.model.mmproj.gguf_path):
        return 0          # ← native vision models get NO floor
    return self.model.image_max_tokens or 0
```

### The gap: native vision models get no floor

The second guard returns 0 when there is no mmproj file. But native vision
models (Qwen3-VL, Gemma 3) still produce image tokens that count against
context. If the budget squeezes their context below `image_max_tokens`, a
maximum-size image will not fit — the same failure the floor exists to prevent.

**This is a bug.** The floor should apply to any model that can process images,
not only those with a separate projector. The fix is to gate the floor on
whether the model *has* vision (mmproj file OR native vision capability), not
on whether it has an mmproj file specifically.

## VRAM budget formula

The full per-slot context solve in `calc_ctx`:

```
available    = (vram_total - reserve - spare) / (1 + margin)
remaining    = available - model_mib - compute_mib
ctx          = remaining / (kv_per_token × parallel + slot_mib)
```

Where:

| Term | Meaning |
|---|---|
| `vram_total` | Total VRAM budget for this model (from `--vram-total` or fleet matrix) |
| `reserve` | `_RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline_mb)` — driver/compositor overhead |
| `spare` | Operator-declared spare (`--spare`) |
| `margin` | Safety margin on measured constants |
| `model_mib` | Weights (main + mmproj if `include_mmproj`) |
| `compute_mib` | Fixed compute overhead (main + mmproj if `include_mmproj`) |
| `kv_per_token` | KV cache bytes per token per layer (main + mmproj if `include_mmproj`) |
| `slot_mib` | Per-slot fixed overhead |
| `parallel` | Number of slots (each slot gets `ctx` tokens; total pool = `parallel × ctx`) |

When `include_mmproj=False`, the projector's contributions to `model_mib`,
`compute_mib`, and `kv_per_token` are all zero — that is the entire mechanism
by which shedding vision buys context.

## Non-causal attention and ubatch

Image tokens use **non-causal attention**: all tokens within a single image
attend to each other bidirectionally (the `scope_non_causal` RAII helper in
libmtmd disables causal masking for the image chunk). This means all image
tokens for one image must be processed in a **single ubatch**. If
`image_max_tokens` exceeds `ubatch`, the prefill will fail. llama.cpp handles
this internally by raising ubatch as needed, but it is a further reason the
context floor matters: a model whose context is squeezed below
`image_max_tokens` cannot process a full image in one ubatch.

## Summary

| Architecture | mmproj file? | `include_mmproj` axis | Image floor applies? |
|---|---|---|---|
| LLaVA, Qwen2-VL, Gemma 3 (external projector) | Yes | Live — shedding frees VRAM | Yes |
| Qwen3-VL, Gemma 3 (native vision) | No | No-op — nothing to shed | **No — this is the bug** |

The fix is to make `_image_floor_tokens` return `image_max_tokens` for any
model that can process images, regardless of whether vision is via a separate
projector or baked into the main weights.
