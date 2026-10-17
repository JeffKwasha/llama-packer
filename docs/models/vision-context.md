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

## The resolution budget: `min_image_megapixels`

Because image tokens count against context, a model that serves vision must
have enough per-slot context to fit an image **and answer about it**. The
operator declares the resolution they intend to serve with
`min_image_megapixels`, and llama-packer converts it to a token floor.

Megapixels rather than tokens because that is the unit an operator thinks in,
and because the token count is not a property of the model alone — it depends
on the resolution of the image actually sent. `min_image_megapixels: 2` means
"this model must be able to handle a 2-megapixel image"; the number of tokens
that is depends on the projector.

### Converting megapixels to tokens

The projector geometry is read from the mmproj GGUF header
(`llama_packer/ggufmeta.py` — a few kilobytes, no tensors, no dependency).
For a pixel-proportional projector each output token covers a square of
`patch_size × spatial_merge_size` pixels:

```
pixels_per_token = (patch_size × spatial_merge_size)²
image_tokens     = min_image_megapixels × 1_000_000 / pixels_per_token
```

Qwen3-VL: `patch_size=16`, `spatial_merge_size=2` → 1024 pixels/token:

| Image | MP | tokens |
|---|---|---|
| 720p | 0.9 | ~900 |
| 1080p | 2.1 | ~2048 |
| 4K | 8.3 | ~8100 |
| Qwen default cap 16384 | 16.8 | 16384 |

That last row is the point of the knob: the model's *default* cap is a 16.8 MP
image, which nobody sends. `min_image_megapixels: 2` cuts the image budget
roughly eightfold and hands the difference back to context.

### The output reserve

The floor is the image budget **plus** room to answer:

```
floor = image_tokens + image_output_reserve
```

Without the reserve an entry could be sized to exactly the image budget: the
model sees the picture and has nothing left to describe it with. The default
is `_DEFAULT_IMAGE_OUTPUT_RESERVE` (1024); a sidecar `image_output_reserve`
overrides it for OCR/transcription, which need far more.

### Warnings

`Model.image_budget_warnings()` reports, once per model, the declarations that
cannot take effect:

| Declaration | Outcome |
|---|---|
| budget on a projector that does not scale with pixels | warned, ignored — falls back to `image_max_tokens` |
| budget that converts to more tokens than `image_max_tokens` | warned, clamped to the cap |
| `min_image_megapixels: 0` on a vision model | warned — vision stops being required and may be shed |
| budget on a model with no projector at all | warned, ignored |

"Does not scale with pixels" is signalled by the absence of
`clip.vision.spatial_merge_size`: resampler/fixed-size projectors (MiniCPM-V,
Gemma 3, LLaVA) produce a roughly constant token count regardless of
resolution, so a megapixel budget means nothing to them.

`image_max_tokens` remains the declared **cap** — the maximum image tokens
llama-server will spend on one image. It is both the clamp target above and,
when no megapixel budget is declared, the fallback image budget.

### Vision requirement

A positive `min_image_megapixels` means vision must actually be served at that
size, so the planner adds `vision` to that model's `keep` set: the projector
stops being shed-able whatever the category's `sacrifice` order says.
`min_image_megapixels: 0` removes that protection — vision may be shed — which
is why it warns.

## The floor in code

```python
def _image_floor_tokens(self, include_mmproj: bool) -> int:
    if not include_mmproj:
        return 0
    image = self.model.min_image_tokens      # from min_image_megapixels
    if image is None:
        image = self.model.image_max_tokens  # legacy fallback / declared cap
    if not image:
        return 0
    cap = self.model.image_max_tokens
    if cap is not None and image > cap:
        image = cap
    return image + self.model.image_output_reserve_tokens
```

The floor applies to both separate-projector and native-vision models: a
declared image budget is itself the indicator that the model processes images.
The pixel-proportional conversion, however, needs a projector header, so a
native-vision model can only use the `image_max_tokens` fallback.

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
| `vram_total` | Total VRAM budget for this model (from `--vram` or fleet matrix) |
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
tokens for one image must be processed in a **single ubatch**. If the image
budget exceeds `ubatch`, the prefill will fail. llama.cpp raises ubatch as
needed, but it is a further reason the context floor matters: a model whose
context is squeezed below its image budget cannot process a full image in one
ubatch.

## Summary

| Architecture | mmproj file? | `include_mmproj` axis | Megapixel knob |
|---|---|---|---|
| LLaVA, MiniCPM-V, Gemma 3 | Yes | Live — shedding frees VRAM | warned + ignored (fixed/resampled tokens) |
| Qwen2-VL / Qwen3-VL | Yes | Live — shedding frees VRAM | honoured (`spatial_merge_size` present) |
| Native multimodal | No | No-op — nothing to shed | warned + ignored (no projector geometry) |

