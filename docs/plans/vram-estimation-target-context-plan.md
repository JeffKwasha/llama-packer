# VRAM estimation: target-context calibration plan

Status: **C1, C3, C6, C7, C9 implemented (2026-10-18).** C2 (gap-constancy assertion), C4 (err_bound reserve), C5 (per-entry error-factor test) and C8 (`--validate-memory`) remain planned. See the git log for the implementation commit(s).

## The problem, restated

The estimator fits the affine law

    VRAM(C, p) = model + compute + c*C + p*D

with `c`, `D`, `model`, `compute` from `llama-fit-params`, plus a per-arch
correction fitted by `--probe-memory` on a grid capped at
`_MEASURE_CTX_CAP = 65536` (`consts.py:83`). The model is then planned at
contexts up to `design_context` (262144 for Dirk). **The correction is fitted
at ≤64k and applied at ~200k.** The gap it prices is context-independent *if*
it is constant — and the code never checks that.

Measured gap (fit-params vs the real probe run, MTP draft loaded):

| term | fit-params | real server | gap | C-dependent? |
|---|---|---|---|---|
| weights | 18569 | 19899.31 (Vulkan0 18904.68 + CPU_Mapped 994.63) | **+1330** | no |
| RS (p=2) | 299.25 | 897.75 | +598.5 (draft) | no (per-cell) |
| KV | 4352 @131072 | 1088 @32768 (= 0.0332×C) | 0 | matches |
| compute | 589 | 224.59 | −364 | shape, ~constant |

The dominant term is the **weight gap**, and it is a mismeasurement of what
`llama-fit-params` reports: its `model` field is a header-derived logical size,
not the allocated weight buffer. For Gemma-4 Q3_K_M it is fine (11187 vs file
11202.9); for Dirk it drops baked-in layers (`nextn_predict_layers = 1`).
The old "446 MiB" figure was a *derived residual*, not a measurement.

Uncorrected, the estimator is ~8% low at 262144 (27.4 GiB predicted vs 29.69
GiB observed) — a significant error factor.

## What to change, where, and why

### C1 — Measure the correction at the target context, not the cap

- `llama_packer/memory_probe.py::_probe_grid` (line 114): add the design
  context at p=1. Keep the p=2 point at a context that fits.
  Proposed grid: `(design,1)`, `(design//2,1)`, `(c_small,2)`,
  `(c_small,1)`, `(c_small//2,1)` where
  `c_small = min(design, _MEASURE_CTX_CAP)`.
- `llama_packer/memory_probe.py::affine_report` (lines 212-213): `d` is
  currently `pool((big,2)) - pool((big,1))`. Change it to the small-context
  pair, because `(262144,2)` cannot fit (Dirk needs ~39 GB at p=2). `D` is
  context-independent, so this is exact.
- **Why:** `c` is then measured at the context the model actually serves.
- **Effect:** `delta_c`/`delta_d`/`delta_fixed` are fitted at the target.

### C2 — Assert the gap is constant across context

- `memory_probe.py::affine_report`: compute the serve-vs-fit gap at each C
  point and require it flat within tolerance. Store `gap_constancy_max_mib`.
- If it is not flat, the correction cannot be fitted at a smaller context and
  the target point is mandatory.
- **Why:** extrapolation is valid *only* if the gap is constant; this turns an
  assumption into a checked fact.
- **Effect:** a correction that is invalid at design is never persisted.

### C3 — Fix the weight term at the source (highest leverage)

- New helper `llama_packer/ggufmeta.py::gguf_tensor_bytes(path)`: walk the
  tensor-info section (name, n_dims, dims, type, offset) and sum on-disk
  tensor bytes via consecutive offsets (`gap = off[i+1]-off[i]`, plus
  `file_size - off[-1]`). No new dependency; the `_Reader` already exists.
- `llama_packer/vram.py::_measure_affine_trio` (line 836): use
  `model_mib = max(int(r1["model"]), gguf_tensor_bytes(...))` — or replace it
  outright once validated.
- **Why:** removes the −1330 MiB bias at its source instead of burying it in
  `delta_fixed`; `delta_fixed` becomes a genuine per-arch allocator constant.
- **Effect:** the largest single error term is eliminated.

### C4 — Reserve a measured error bound

- `serve-corrections.yaml` schema: add `err_mib` and `err_rel` per row
  (measured at the target context by the probe).
- `llama_packer/vram.py::calc_ctx` (line 1283): fold `err_bound` into the
  reserve:
  `available = (vram_total - reserve - err_bound - spare) / (1 + memory_margin)`.
- **Why:** sizing to the exact budget with a 5-8% error factor guarantees
  over-admission.
- **Effect:** `calc_ctx` returns a context that fits *including* the measured
  error.

### C5 — Per-entry "significant error factor" test

- After solving an entry's context, compute `slack = budget - predicted`.
  If `slack < err_bound`, mark the entry **at risk**: shrink the context or
  emit a warning instead of a config that may not load.
- **Why:** this is the operational meaning of the requested error-factor test.
- **Effect:** no entry is emitted whose estimate is within its own error of
  the limit.

### C6 — Do not persist a correction whose grid fit failed

- `memory_probe.py::affine_report` (lines 238-246): call
  `save_serve_correction` only when the report is `ok`. Today a FAIL is saved.
- **Why:** a failed fit produces a correction that is confidently wrong.

### C7 — Split shape-invariant from shape-dependent constants

- `llama_packer/vram.py::FitParams` (line 403): split into invariant
  `(model_mib, kv_per_token_mib, slot_mib)` and shape-bound `compute_mib`.
  Key and persist them separately.
- **Why:** only `compute_mib` depends on batch/flash flags; today any flag
  change invalidates the whole block and forces a re-measure.

### C8 — `--validate-memory`

- New CLI path: launch one `llama-server` at the *planned* context, read the
  `-lv 5` buffer lines, and report `actual / estimated`. For Dirk at 262144
  this is ~29 s and is the only ground-truth error factor.
- **Why:** the modelled error factor needs an occasional ground-truth check.

### C9 — Docs and dead code

- `vram.py` module header (lines 21-25) still says measurements come from real
  `llama-server` runs; normal runs use fit-params + corrections. Fix.
- `affine_from_pair` (`vram.py:186`) has no callers — remove.

## What NOT to change

- The affine law itself — it holds (R² = 0.999961, max residual ~20 MiB at
  128k-262k).
- The trio's `slot_mib = pool(C,2) - pool(C,1)` — exact, because `-c C
  --parallel p` holds the KV pool at C and only RS scales with p.
- The `-c`/`--parallel` semantics; `--kv-unified-per-slot` is equivalent
  (pool = p*X).

## Test plan

- Unit tests for `gguf_tensor_bytes` against the real GGUFs (Dirk, Gemma-4).
- Unit tests for the new `_probe_grid` and the constancy assertion with a
  mocked `_run_measure_server`.
- Unit tests for the `err_bound` reserve and the at-risk test.
- The existing suite (1044 tests) must stay green; `check_trace` and
  `check_mutation` clean.

## Verification once the GPU is free

1. Re-run `--probe-memory qwen35` with the new target-context grid.
2. Compare `err_mib` against the buffer-line sum for the witness.
3. Re-plan Dirk and confirm the emitted 256k entry is within `err_bound`.
4. Confirm the uncorrected error factor drops from ~8% to within the stated
   margin.
