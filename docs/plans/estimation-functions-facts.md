# Estimation functions — facts collected 2026-10-18

Purpose: pin down what the code actually does, what it assumes, and where the
measurements diverge. Analysis and fixes are separate work.

## The affine model

`VRAM(ctx, parallel) = model_mib + compute_mib + kv_per_token_mib * ctx * parallel + slot_mib * parallel`

- `model_mib`: weights + base compute (measured or estimated).
- `kv_per_token_mib`: per-token KV cache cost (measured via trio).
- `slot_mib`: per-slot fixed cost (measured via p=1 vs p=2 difference).
- `compute_mib`: fixed compute buffer (max over the trio; model-dependent).

Source: `llama_packer/vram.py` docstrings and `glossary.md` ("the affine law").

## FitParams sources (in priority order)

1. `saved_for` — persisted in sidecar `derived:` block, per cache_type + shape.
2. In-memory `_static_cache` (process lifetime).
3. `_fit_params_serve` — fast trio (header-only runs, ~0.6s each, no server).
   - Runs 3 `llama-fit-params` subprocesses: (C, p=1), (C, p=2), (C/2, p=1).
   - `c = (pool(C) - pool(C/2)) / (C - C/2)` — per-token KV cost.
   - `D = pool(C, p=2) - pool(C, p=1)` — per-slot cost.
   - `compute_mib = max(compute across the three runs)`.
   - `model_mib = r1["model"]` (from the first run).
   - Source is `"fit-estimate"`.
4. Serve correction (`get_serve_correction`) — arch-level, machine-local
   (`serve-corrections.yaml`), applied as deltas: `delta_c`, `delta_d`,
   `delta_fixed`. Correction shape mismatch warns but still applies.
5. `_fit_params_fast` — transient (no corrections, never persisted).
6. `_estimate_safetensors` — from safetensors header.
7. vLLM: `vllm-memory-estimator` on HF repo, or safetensors header. **UNTESTED — no vLLM installation on this hardware; this is a future roadmap item. No model in the live fleet declares the vllm backend. Nothing in this document is verified for this path.**

## Hardcoded constants (consts.py)

| constant | value | used for |
|---|---|---|
| `_MMPROJ_COMPUTE_MB` | 150 | mmproj fixed compute |
| `_DRAFT_COMPUTE_MB` | 64 | MTP draft compute |
| `_DRAFT_CTX_SAFETY` | 1.6 | safety factor on draft affine terms |
| `_SD_COMPUTE_MB` | 512 | sd-server fixed compute |
| `_WHISPER_COMPUTE_MB` | 100 | whisper-server fixed compute |
| `_AUDIO_CPP_COMPUTE_MB` | 1024 | audio.cpp fixed compute |
| `_VLLM_PER_SEQ_MIB` | 32.0 | vLLM per-sequence growth |
| `_LLAMA_CPP_DEFAULT_BATCH` | 2048 | rendered when nothing configured |
| `_LLAMA_CPP_DEFAULT_UBATCH` | 512 | stamped into measurement shape |
| `_FIXED_COMPUTE_MB` | map | backend name -> fixed compute |

None of these constants have a documented basis (no "measured on X" or
"from upstream docs" comment). They are operator heuristics.

## Companion fitting (_companion_fit)

- mmproj: `(file_size_mb, 0.0, 0.0, _MMPROJ_COMPUTE_MB)` — weight + fixed
  compute only, no KV, no slot scaling. Independent of context and slots.
- MTP draft: `(size_mb, kv_per_token * ratio, slot_mib * ratio,
  _DRAFT_COMPUTE_MB)` where `ratio = (draft_size / main_model_mib) * 1.6`.
  Only if main_fp is valid and measured. Otherwise: `(size_mb, 0, 0, 64)`.
- Draft folded into main only if `main.source != "llama-server"` (serve-shaped
  measurement already includes draft).

## effective_static (main + companions)

- Fixed-overhead backends (sd, whisper, audio-cpp): `(file_size, 0, 0, fixed)`
  or operator-pinned `vram_mb`.
- vLLM: from estimator or safetensors header, no companion folding.
- llama.cpp: main FitParams + companion additions (mmproj, draft).
- MTP draft folded only if `main.source != "llama-server"`.

## calc_ctx (context solving)

1. CPU-resident models: return design context (no VRAM bound).
2. Reserve = `_RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline_mb)`.
3. available = (vram_total - reserve - spare) / (1 + memory_margin).
4. If available <= 0: over_budget_reason set, return min ctx.
5. design = sidecar context_length or GGUF arch max.
6. If design fits (design_cost <= remaining): return min(design, sidecar_ceiling).
7. If design doesn't fit: solve affine equation for max_ctx.
   - If max_ctx < _MIN_CTX_SIZE: over_budget, return 0 (entry disabled).
   - Else: return max_ctx (vram_squeezed = True).
8. Image floor: can raise ctx within cap/affordability.

## Context ceiling vs design context

- `_design_ctx`: GGUF architectural max > sidecar context_length > default.
- `_sidecar_ceiling`: sidecar `context_length` only (operator decision).
- `calc_ctx` uses design for the "does it fit" check, capped by sidecar_ceiling
  for the final value.

## Measurement shape and reuse

- Shape = effective serve flags (global args + resolved -ub).
- `saved_for(cache_type, shape)` rejects blocks where `str(d["shape"]) != str(shape)`.
- Blocks without a `shape` key are rejected (pre-shape legacy).
- `_SHAPE_SOURCES = {"llama-server", "fit-estimate", "fit-params"}`.
- `serve-corrections.yaml`: arch-level, shape mismatch warns but applies anyway.

## Measured vs actual (Dirk, 2026-10-18)

| term | estimated | measured | delta |
|---|---|---|---|
| model_mib | 18569 | ~19015.5 | +446.5 |
| kv_per_token_mib | 0.037109375 | 0.037109375 (matched) | ~0 |
| slot_mib | 598.5 | 598.5 (assumed) | ~0 |
| compute_mib | 676 | 676 (assumed, within 0.81 GiB gap) | ~0 |
| total (256k, par=1) | 28.88 GiB | 29.69 GiB | +0.81 GiB |

- model_mib miss: 446 MiB (2.4% of model size). Source was fit-estimate.
- KV term matched exactly.
- compute term assumed (within the 0.81 GiB total gap).
- sidecar `file.size_mb`: 19909 (file is 19.44 GiB on disk).

## Sidecar derived block sources found

- 11 files under /mnt/ai/models/chat/*/*.md have `shape: --flash-attn on -ub 512`.
- OCR model (NuExtract3) has `source: fit-estimate` with no shape line.
- Sidecars under /mnt/ai/models/chat/*/*.md mostly have `source: fit-estimate`.

## audio.cpp backend constants

- `_AUDIO_CPP_COMPUTE_MB = 1024` — "weights + session/vocoder/diffusion buffer.
  Family-dependent (ASR/VAD ~512 ... diffusion TTS ~3072)."
- Fixed-overhead backend: VRAM = file size + fixed buffer. No KV terms.
- Operators pin exact figure via `vram_mb` sidecar key.

## What was NOT measured

- Whether ubatch 256 vs 512 changes VRAM on the 27B (predicted: no, compute
  buffer is ~0.66 GiB either way and VRAM is flat).
- Whether model_mib miss (446 MiB) is systematic or Dirk-specific.
- Whether the MTP draft scaling (1.6x safety factor) is correct.
- Whether mmproj companion estimate (file size + 150 MiB) is accurate.
- Whether the safetensors header estimate matches llama-fit-params.
- The entire vLLM estimation path (untested — no vLLM installation).


## Probe results (2026-10-18, qwen35 arch, Dirk representative)

The probe ran with real llama-server measurements. Correction row saved to
serve-corrections.json: `qwen35|q8_0|1` (mtp_baked_in=True).

| term | fit-estimate | correction (truth - est) | corrected | measured (262144, par=1) |
|---|---|---|---|---|
| model_mib | 18569 | unchanged (delta_fixed hits compute) | 18569 | ~18615 (derived) |
| kv_per_token_mib | 0.037109375 | +0.00390625 (delta_c) | 0.041015625 | 0.037109375 (matched) |
| slot_mib | 598.5 | +448.87 (delta_d) | 1047.37 | 598.5 (assumed) |
| compute_mib | 676 | -105 (delta_fixed) | 571 | (unverified) |

NOTE: The correction does NOT change model_mib. delta_fixed is applied to
compute_mib, not model_mib. The "measured model_mib" of ~19015.5 from the
earlier notes was my own derivation (total - KV - slot - compute), which is
unreliable because it assumes the KV/slot/compute terms are correct. The
probe's corrected total (30939 MiB) overestimates the measured total (30409
MiB) by 530 MiB, so the correction overshoots. The raw fit-estimate (29572
MiB) underestimates by 837 MiB. The correction moves from -837 to +530, a
1367 MiB swing.

The probe's deltas are measured at 65536/32768/16384 context, not at
262144. Extrapolation to large contexts may not be linear. The
common_fit_params warning ("failed to fit params to free device memory:
n_gpu_layers already set by user to 999, abort") suggests the compute term
measurement was compromised.

Notes:
- The correction makes the estimate MORE conservative (higher), not more accurate.
  Raw fit-estimate: 28.88 GiB vs measured 29.69 GiB (0.81 GiB low).
  Corrected: 30.22 GiB vs measured 29.69 GiB (0.53 GiB high).
  The correction overshoots: 1.34 GiB swing.
- delta_d = +448.87 MiB per slot is the largest correction. This means
  the fit-estimate's slot_mib (598.5) was 449 MiB too low.
- delta_c = +0.00390625 MiB/token. At 262144 context, 1 slot: +1024 MiB.
- delta_fixed = -105 MiB. The fit-estimate overcounted fixed by 105 MiB.
- The net correction: +1024 + 449 - 105 = +1368 MiB.
- The measured model_mib (from the 262144 load) is ~19015.5 MiB.
  The corrected model_mib is 18464 MiB (18569 - 105). Still 551 MiB low.
  So even after correction, model_mib is undercounted.

## Additional facts

- The probe measured at 65536, 32768, 16384 contexts with p=1 and p=2.
- spill_mib = 0.0 for all measurements (no VRAM spill).
- The correction row is keyed by arch "qwen35", cache_type "q8_0", mtp_on=1.
- Dirk has MTP baked in (mtp_baked_in=True), so the correction was filed.
- The correction is applied at pack time via get_serve_correction.
- The fit-estimate block in the sidecar still says source: fit-estimate.
  The correction is applied on top at pack time, not persisted to the sidecar.
- serve-corrections.yaml (arch-level) is separate from per-model derived blocks.
- The correction table's mtp dimension is baked-in drafts only.
  A witness with --spec-draft-model carries companion weights separately.


## VRAM measurement methodology — limitations

### How VRAM was measured

- `mem_info_vram_used` from `/sys/class/drm/card1/device/mem_info_vram_used`
  (AMD sysfs). This is **global GPU VRAM usage** — all processes, including
  the KDE GUI, ComfyUI, and anything else on the card. There is no per-process
  VRAM counter available without root (`amd-smi` exists at `/opt/rocm/bin/amd-smi`
  but was not used).
- The probe measures by: starting a server, running it at specific (ctx, parallel)
  points, reading the global counter, then killing the server. The difference
  between "server running" and "server stopped" is attributed to the model.
- `spill_mib = 0.0` for all measurements — meaning no VRAM spill was detected.
  But this is based on the global counter, which includes other processes' VRAM.
  A spill from one model could be masked by another process freeing VRAM.

### What the probe measured

- Context points: 65536, 32768, 16384 (all with p=1 and p=2).
- These are the probe's internal grid, not user-selected. The probe does not
  measure at 128k, 200k, or 256k.
- The affine law (`VRAM = model + compute + c*ctx + p*D`) is fit to these
  three points and then extrapolated to larger contexts.

### What was actually tested at large context

- One real load at 262144 context (Dirk, 1 slot): 30409 MiB observed.
- This is the only measurement at a context length the fleet actually uses.
- The probe's extrapolation from 65536 -> 262144 overshot by 0.53 GiB.

### What was NOT tested

- 128k context (no measurement at this scale).
- 200k context (no measurement at this scale).
- Multiple models loaded simultaneously (the matrix solver's use case).
- VRAM under the service namespace (all measurements were in a shell, where
  the GPU is visible; the service cannot see the GPU at all).
- Per-process VRAM attribution (global counter only).
- Whether the affine law holds above 65536 context (only one data point).
- GTT/shared memory: the card has 32 GiB VRAM but also system RAM for GTT.
  The measurements don't distinguish between dedicated VRAM and GTT.


## Monitor-based measurement (2026-10-18, root, 262144 context, 1 slot)

Using gpu-vram-monitor.py (root, /proc scanning for per-process attribution).

| metric | value |
|---|---|
| Baseline VRAM (before load) | 1390 MiB |
| VRAM after load (262144, par=1) | 31210 MiB |
| Delta (model's cost) | 29820 MiB (29.12 GiB) |
| Fit-estimate | 28.88 GiB (29572 MiB) |
| Gap (estimate vs measured) | +0.24 GiB (estimate low) |
| Corrected estimate (with serve-correction) | 30.21 GiB (30939 MiB) |
| Gap (corrected vs measured) | +1.09 GiB (correction overshoots) |

Key finding: the correction makes the estimate WORSE, not better. The raw
fit-estimate (28.88 GiB) is closer to reality (29.12 GiB) than the
corrected estimate (30.21 GiB). The correction overshoots by 1.09 GiB.

The correction's deltas:
- delta_c = +0.00390625 (KV per token, +1024 MiB at 262144 context)
- delta_d = +448.87 (per-slot, +449 MiB)
- delta_fixed = -105 (fixed term, -105 MiB)
- Net correction: +1368 MiB

The net correction (+1368 MiB) moves the estimate from -837 MiB (low) to
+530 MiB (high). The correction overshoots the gap.

This is the second time the correction has been measured to overshoot
(the first was the single-load test at 29.69 GiB, where the correction
predicted 30.21 GiB).

The fit-estimate's model_mib (18569) is the term that's off. The correction
doesn't touch model_mib — it applies delta_fixed to compute_mib. So the
weights term remains undercounted regardless of whether the correction is
applied.

### Contamination check

The monitor's process list showed 37-38 GPU processes during the load,
all owned by the desktop session (KDE, browsers, etc.). No other process
allocated or freed VRAM during the load window — the baseline was stable
at 1390 MiB and the post-load was stable at 31210 MiB. The delta is clean.


## Affine law validation (2026-10-18, Dirk, 3 context lengths, par=1)

Three real measurements at 128k, 200k, 256k context. VRAM measured via
gpu-vram-monitor (root, /proc scanning, global counter). Baselines were stable
(1632, 1598, 1644 MiB before each load; no other process allocated during
the window).

| ctx | observed VRAM (MiB) | delta from baseline (MiB) |
|---|---|---|
| 131072 | 26414 | 24782 |
| 204800 | 29253 | 27655 |
| 262144 | 31483 | 29839 |

### Affine law fit (least squares)

- c (KV MiB/token): 0.03860 (sidecar: 0.037109375, error +4.0%)
- fixed (model+compute+slot): 19731 MiB (sidecar: 19245 MiB, error +486 MiB)
- Residuals: -8, +19, -11 MiB (0.07% of total — law is linear)
- Interval slopes: 0.03897 (128k->200k), 0.03809 (200k->256k), spread 2.3%

### What this means

1. The affine law holds at large contexts. Residuals are <0.1% of total.
   The earlier "0.81 GiB gap" was measurement noise, not non-linearity.
2. The KV term is 4.0% off (0.03860 vs 0.037109375). At 262144 context:
   0.00149 x 262144 = 391 MiB. Not huge but not zero.
3. The fixed term is 486 MiB off (19731 vs 19245). This is close to the
   "446 MiB model_mib miss" estimated earlier, suggesting the weights term
   is the primary source of the fixed-term error.
4. The interval slopes differ by 2.3%, which is within measurement noise
   (other processes on the card, thermal throttling, etc.).

### Comparison with probe correction

The probe's correction (qwen35, mtp_on=1):
- delta_c = +0.00390625 (KV per token)
- delta_d = +448.87 (per-slot)
- delta_fixed = -105 (fixed term)

The probe's truth fixed = 19140 (18569 + 676 - 105).
The measured fixed = 19731.
The probe underestimates fixed by 591 MiB.

The probe's truth KV = 0.037109375 + 0.00390625 = 0.041015625.
The measured KV = 0.03860.
The probe overestimates KV by 0.0024 MiB/token (6.2%).

Neither the fit-estimate nor the probe correction matches the measured
constants. The fit-estimate is closer on fixed (486 MiB error vs 591 MiB),
and the probe is closer on KV (6.2% error vs 4.0%... actually the fit-estimate
is closer on KV too: 4.0% vs 6.2%).

### What was NOT tested

- 2 slots at large context (parallel=2). The estimate for 262144x2 is 39.0 GiB
  (over budget). Testing this would confirm or refute the slot term.
- A different model (gemma4, qwen3.6) to see if the miss is arch-specific.
- The effect of ubatch 256 vs 512 on the 27B model's VRAM.
- Whether the affine law holds above 262144 (no test, but 262144 is the
  model's architectural max).


## 6-point affine fit (2026-10-18, Dirk, 3 single-slot + 3 double-slot)

Six real measurements: 128k/200k/256k at par=1, and 40k/60k/80k at par=2.
All loads succeeded (no spill, no OOM). Baselines were stable.

| ctx | par | observed (MiB) | predicted (MiB) | residual |
|---|---|---|---|---|
| 131072 | 1 | 24782 | 24799 | -17 |
| 204800 | 1 | 27655 | 27635 | +20 |
| 262144 | 1 | 29839 | 29842 | -3 |
| 40960 | 2 | 23161 | 23144 | +17 |
| 61440 | 2 | 24712 | 24720 | -8 |
| 81920 | 2 | 26286 | 26296 | -10 |

### Fitted constants

- c (KV MiB/token): 0.03847 (sidecar: 0.037109375, error +3.7%)
- fixed (model+compute): 19520 MiB (sidecar: 19245 MiB, error +275 MiB)
- slot (per parallel slot): 235.9 MiB (sidecar: 598.5 MiB, error -362.6 MiB)
- R-squared: 0.999961, max residual: 20 MiB

### Key findings

1. The affine law holds at large contexts with par=1 and par=2. Residuals
   are <0.1% of total. The law is valid.

2. The KV term (c) is 3.7% too low in the sidecar (0.037109375 vs 0.03847).
   At 262144 context: +364 MiB. At 262144 x2: +728 MiB.

3. The fixed term (model+compute) is 275 MiB too low (19245 vs 19520).
   This is the weights undercount (model_mib 18569 vs ~19015).

4. The slot term is 362.6 MiB too HIGH (598.5 vs 235.9). The sidecar
   overcounts the per-slot cost by 60%. This is a significant error in the
   other direction.

5. Net effect on a 262144x1 serving: +364 + 275 - 363 = +276 MiB.
   Matches the measured +246 MiB gap within noise.

6. The slot term error is the largest single discrepancy. The fit-params
   trio computes slot_mib = pool(C, p=2) - pool(C, p=1), which should be
   the per-slot fixed cost. But it overestimates by 362.6 MiB. This could be
   because the trio's p=1 and p=2 runs have different compute buffers
   (larger ubatch for p=2), and the slot_mib absorbs compute differences.

### What this means for the estimates

The fit-estimate is off in three directions:
- c (KV): undercounted by 3.7% (too optimistic)
- fixed (model+compute): undercounted by 275 MiB (too optimistic)
- slot (per-slot): overcounted by 363 MiB (too pessimistic)

The net error is small (+276 MiB) because the slot overcount partially
cancels the fixed undercount. But the decomposition is wrong: the per-slot
cost is much lower than the code thinks, and the per-token KV cost is
slightly higher.

For a 2-slot serving (the common case), the slot error doubles:
- 2 slots: fixed error +275, KV error +728, slot error -725 = +278 MiB.
  Similar net error but different decomposition.

The slot term should be ~236 MiB, not 598.5 MiB. This affects every
model's VRAM estimate. The slot term is the per-parallel-slot fixed cost
(recurrent state cache + per-slot allocator overhead). The fit-params trio
is overcounting it by 2.5x.


## Fit-params binary output vs sidecar (2026-10-18, Dirk, 131072 ctx, par=1, --flash-attn on -ub 512)

Raw fit-params binary output (Vulkan0): `18569 4501 653` (model, context, compute).
-lv 5 log: KV buffer 4352.00 MiB (131072 cells, 16 layers), recurrent state 149.62 MiB,
compute buffer 653.29 MiB (Vulkan) + 168.30 MiB (Host).

### Raw vs sidecar vs corrected vs measured

| term | fit-params binary | sidecar (derived) | corrected (probe) | measured |
|---|---|---|---|---|
| model_mib | 18569 | 18569 | 18569 | ~19015 |
| kv_per_token | 0.03319 (4352/131072) | 0.037109375 | 0.041015625 | 0.03847 |
| slot_mib | 149.62 (recurrent) | 598.5 | 1047.37 | 235.9 |
| compute_mib | 653 | 676 | 571 | (unverified) |

### Key discrepancies

1. **model_mib**: fit-params reports 18569 (from GGUF header, load_mode=none).
   Measured ~19015 (446 MiB gap). The binary estimates from header metadata,
   not from loaded tensors.

2. **kv_per_token**: fit-params binary gives 0.03319. Sidecar (corrected) gives
   0.037109375. Measured 0.03847. The correction adds 0.00390625 to the raw
   value. But the corrected value (0.041015625) is higher than measured (0.03847).

3. **slot_mib**: fit-params binary gives 149.62 (recurrent state only).
   Sidecar (corrected) gives 598.5. The correction adds 448.87. But measured
   is 235.9 — the sidecar overcounts by 362.6 MiB. The correction made it
   worse (149.62 -> 598.5 -> 1047.37, while reality is 235.9).

4. **compute_mib**: fit-params binary gives 653. Sidecar says 676. Corrected
   says 571. The sidecar's 676 doesn't match any of these. The correction
   subtracts 105 (653 -> 548), but the sidecar says 676. This suggests the
   sidecar's compute_mib was set differently (possibly from a different run
   or manually).

### The correction makes things worse

For a 262144x1 serving:
- Raw fit-estimate: 28.88 GiB (gap: -0.24 GiB, undercounted)
- Corrected: 30.21 GiB (gap: +1.09 GiB, overcounted)
- The correction overshoots by 1.33 GiB.

The correction's deltas (measured at 16k/32k/65k) don't extrapolate
correctly to 262k. The probe's truth fixed (19140) is 525 MiB higher than
the measured total implies (18615). The correction is conservative but
inaccurate.

### Root causes

1. **model_mib undercount**: The fit-params binary estimates from the GGUF
   header (load_mode=none, no tensors loaded). It undercounts by ~446 MiB.
   Fix: use file size as a floor, or measure with tensors loaded.

2. **slot_mib overcount**: The fit-params trio computes slot_mib =
   pool(C, p=2) - pool(C, p=1), which includes the per-slot KV cache at
   the measured context C. This is not a fixed cost — it scales with context.
   The actual per-slot fixed cost (recurrent state) is 149.62 MiB.
   Fix: slot_mib should be r1["rs"] (recurrent state only), not pool2 - pool1.

3. **Correction overshoot**: The correction deltas are measured at small
   contexts and applied to all contexts. The slot delta (+448.87) is
   measured at 65k and applied to 262k, but the per-slot cost should be
   fixed (not context-dependent). The correction is wrong because the
   underlying slot_mib is wrong.

4. **compute_mib discrepancy**: The sidecar's 676 doesn't match the
   corrected 548 or the fit-params binary's 653. The sidecar may have been
   generated with different flags or a different measurement.

## Verification pass (2026-10-18, later) — corrections to the above

Direct re-measurement of `llama-fit-params` on Dirk (q8_0, -ub 512,
--flash-attn on). `-c C --parallel p` holds the **KV pool at C** (split p
ways); only the recurrent-state cache scales with p:

| run | KV pool | RS | report row (model context compute) |
|---|---|---|---|
| -c 131072 -np 1 | 4352.00 (131072 cells) | 149.62 | 18569 4501 653 |
| -c 131072 -np 2 | 4352.00 (65536 cells ×2) | 299.25 | 18569 4651 589 |
| -c 65536 -np 1 | 2176.00 | 149.62 | 18569 2325 589 |
| -c 65536 -np 2 | 2176.00 | 299.25 | 18569 2475 567 |
| -c 16384 -np 1 | 544.00 | 149.62 | 18569 693 541 |
| -c 16384 -np 2 | 544.00 | 299.25 | 18569 843 533 |

- **The trio is exact.** `c = (4352-2176)/65536 = 0.033203125 MiB/token`;
  `D = (4352+299.25)-(4352+149.62) = 149.62 MiB/slot`. The earlier claim that
  `slot_mib` "absorbs per-slot KV and scales with context" is **wrong**: the
  KV pool does not double with `--parallel`, so `pool(2)-pool(1)` is exactly
  the per-slot RS. `slot_mib` needs no fix.
- The sidecar's `slot_mib: 598.5` = raw RS 149.62 + serve-correction
  `delta_d: 299.24` (the baked-in MTP draft's per-slot term). Not a bug.
- **The real `model_mib` gap**: fit-params reports 18569; the real
  llama-server weights line is 19015.5 (−446 MiB). Contrast Gemma-4 Q3_K_M:
  fit 11187 vs file 11202.9 (−16 MiB). So the field is not universally wrong —
  it drops layers fit-params doesn't count (Dirk has `nextn_predict_layers = 1`,
  a baked-in MTP layer).
- **Correction is calibrated far below design context.** `_MEASURE_CTX_CAP =
  65536` (consts.py:83); `_probe_grid` measures the witness at ≤65536/32768/
  16384, but the model serves at design 262144. Any serve-vs-fit gap that
  scales with C (draft KV, allocator rounding) is fitted at 65k and applied
  at 262k — the correction cannot represent it.
- `serve-corrections.yaml` qwen35/q8_0/1 (actual): `delta_c: 0.0`,
  `delta_d: 299.24`, `delta_fixed: 500`, `shape: -ub 512`. The earlier
  "delta_d +448.87 / delta_fixed −105" figures were wrong. delta_fixed +500
  is doing the weights-gap job (fit 18569 + 500 = 19069 ≈ server 19015.5).
- The `derived:` block on qwen38-27B-Dirk.md is `source: llama-server`,
  `measured_at: 262144`, `measured_vram_mib: 30409`, `model_mib: 19015.5`,
  `kv_per_token_mib: 0.037109375`, `slot_mib: 598.5`, `compute_mib: 676` —
  a hand-measured load-test block, **not** produced by the current
  `_fit_params_serve` path (which emits `source: fit-estimate` with no
  `measured_at`).
