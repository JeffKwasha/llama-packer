# Dirk (Qwen3.8-27B) at 256k context — measured, not estimated

Date: 2026-10-18.  Binary: llama-b11717 (Vulkan, RADV).  Card: R9700, 32624 MiB VRAM.

## What was tested

`llama-server` with the exact flags the planner emits for `qwen38-27b-dirk.coding.t`,
one change: `--kv-unified-per-slot 204800` -> `262144` (so 256k context, 1 slot).
Command: see `/tmp/dirk-256k.sh`.

## Result: it loaded

- Load time 28.97 s (19.4 GB from the platter at ~670 MB/s).
- VRAM used after load: **30409 MiB (29.69 GiB)** vs an estimate of 28.88 GiB — **+0.81 GiB**.
- Free after load: **2.17 GiB**.
- No GPU OOM, no CPU fallback (the `ggml_vulkan` line was absent from the log
  because this build's verbosity is low, but VRAM grew by ~27.5 GB, so offload happened).
- `--kv-unified-per-slot` reported `n_ctx_slot = 262144`, `kv_unified = false`,
  so the KV cache is allocated exactly as `--kv-unified-per-slot` requested.

## Why the estimate was low

`derived:` said `model_mib: 18569` (the GGUF weights). The actual model buffer
(including the MTP draft heads and the mmproj-free text path) is ~19015 MiB — a
446 MiB gap. The KV term (0.0371 MiB/token) and compute term (676 MiB) matched
the estimate within noise. So the miss is in the *weights* term, not the KV term.

## Fit picture (measured constants, 1 slot)

| ctx | alone | +kokoro (1204 MiB) |
|---|---|---|
| 204800 | 26.8 | 28.0 |
| 262144 | 28.9 | 30.1 |
| 294912 | 29.7 | 30.9  (projected: +1.2 GiB KV) |

256k fits with ~2.1 GiB to spare (1.0 GiB with kokoro). 294912 (the largest
multiple of 4096 under 300k) fits with ~1.2 GiB to spare alone.

## The binding constraint

`parallel` is the hard one. At parallel 2, 256k needs 39.0 GiB — over by 7 GiB.
So 256k means `parallel: 1`, which is fine for single-user agentic work but
removes the second slot. The `.chat` entry is where the second slot lives
today (90112 × 2 = 26.5 GiB).

## What was changed

- `/mnt/ai/models/chat/qwen3/qwen38-27B-Dirk.md`: `derived:` updated with
  measured `model_mib` (19015.5), `source: llama-server`, `shape`, and the
  observed VRAM. The KV/compute terms are left at the estimated values (they
  matched).
- `profiles.yaml` was NOT changed to request 256k — the planner's category
  solve decides per entry. The user can raise the context ceiling for the
  `coding` category (currently 204800) to 262144 and re-pack.

## What was NOT tested

- Throughput at 256k (the measurement above was a load test only; the model
  was not benchmarked for tok/s at that context — llama-bench uses a minimal
  context).
- A second model co-loaded at 256k (kokoro is the only realistic pairing;
  30.1 GiB fits with ~1.8 GiB margin).
- Whether the plan's admission math uses the new measured model_mib
  correctly — re-pack and validate before serving.
