# VRAM estimation: sound measurement path (plan)

Status: **implemented (Phases 1-3, 5).** 2026-10-18. The user's diagnosis is correct:
the correction table (`serve-corrections.yaml`) exists because the fast path
(fit-params) is an unsound estimator, and `delta_fixed` is the residual that
patches it. This plan replaces the band-aid with a sound measurement path.

## The diagnosis

The affine law `VRAM = model + compute + c*C + p*D` is sound (R² = 0.999961).
The **constants** are unsound because they come from `llama-fit-params`, which is
a header-only estimator, not a measurement:

| constant | fit-params source | why it's wrong |
|---|---|---|
| `model_mib` | GGUF header metadata (`load_mode=none`) | drops baked-in MTP/nextn layers and the CPU-mapped split (~1330 MiB undercount on Dirk) |
| `compute_mib` | max over the trio's three runs | max-over-trio heuristic vs. the server's actual compute buffer (~485 MiB overcount) |
| `c` | `(pool(C) - pool(C/2)) / (C - C/2)` | exact (derived from pool lines) |
| `D` | `pool(C,p=2) - pool(C,p=1)` | exact (derived from pool lines) |

`c` and `D` are sound because they are derived from the `-lv 5` buffer lines,
which are real allocations. `model` and `compute` are unsound because fit-params
estimates them from headers, not measurements.

The correction table (`serve-corrections.yaml`) stores
`delta = server - fit-params` for each arch. `delta_fixed` is the lumped
residual of (weight gap + compute gap + allocator overhead). It is:
- **Large** (~500) because the two errors are large and opposite
- **Context-dependent** (measured at 65k, applied at 262k)
- **Machine-local** (not portable across hardware)
- **Per-arch** (not per-model; a 4B and 27B share the same delta)

The correction is a band-aid. It makes the fast path approximate the sound path,
but it breaks whenever the underlying estimator changes (as C3 demonstrated).

## The sound path already exists: `--probe-memory`

`affine_report` (memory_probe.py) starts a real `llama-server`, reads the
`-lv 5` buffer lines, and measures `(weights, compute, c, D)` directly:

```python
fixed = bufs[(big, 1)]["weights"] + bufs[(big, 1)]["compute"] + output
c     = 2.0 * (pool((big,1)) - pool((big//2,1))) / big
d     = pool((small,2)) - pool((small,1))
```

These are **measurements**, not estimates. The correction table's deltas are
just `measured - estimated`. If we persisted the measured values instead of
the deltas, the correction table would be unnecessary.

## The plan

### Phase 1: Make the probe persist absolute measured values

**File: `memory_probe.py::affine_report`**

Currently persists deltas to `serve-corrections.yaml`. Change it to persist
the measured constants as a complete `FitParams` (source="llama-server") via
`Model.persist_measured()`.

```python
# Instead of:
corr = {"delta_fixed": round(fixed - est_fixed), ...}
save_serve_correction(arch, cache_type, mtp, corr)

# Persist the measured constants directly:
measured = FitParams(
    model_mib=int(fixed_weights),   # server's weight buffer
    kv_per_token_mib=c,             # server's c
    slot_mib=d,                     # server's D
    compute_mib=int(fixed_compute), # server's compute buffer
    source="llama-server",
    cache_type=cache_type,
    shape=shape,
)
model.persist_measured(measured.to_dict())
```

The correction table becomes unnecessary for models that have been probed.
Keep it for backward compatibility (old blocks still read it), but new probes
write absolute values.

**Why:** the probe already measures everything correctly. Persisting the
measured values (not deltas) eliminates the correction table's band-aid role.

### Phase 2: Make normal runs prefer probe-measured values

**File: `vram.py::fit_params_static`**

The priority order is:
1. saved block (if source="llama-server", use it directly — sound)
2. in-memory cache
3. fit-params trio (unsound, fallback)
4. safetensors estimate (fallback)

Add a check: if `saved.source == "llama-server"`, return it without re-measuring
(it's already sound). If `saved.source == "fit-estimate"`, it's unsound —
re-measure (or use it with a warning).

```python
saved = None if self.remeasure else self.saved_for(cache_type, shape)
if saved is not None:
    if saved.source == "llama-server":
        # Sound: measured from a real server. Use directly.
        self._static_cache[(cache_type, shape)] = saved
        return saved
    # Unsound: fit-params estimate. Re-measure if possible.
    ...
```

**Why:** probe-measured blocks are sound; fit-params blocks are not. Prefer
the sound ones.

### Phase 3: Eliminate the correction table

**File: `vram.py::_fit_params_serve`**

Remove the correction application. The fast path becomes:
- fit-params trio for c, D (exact)
- model/compute from fit-params (unsound, fallback)
- No correction (the correction was a band-aid for the unsound model/compute)

When a probe-measured block exists, use it (sound). When it doesn't, use
fit-params (unsound, with a warning).

**File: `memory_probe.py`**

Remove `save_serve_correction` calls. The probe persists absolute values
(Phase 1). The correction table is no longer written.

Keep `get_serve_correction` for backward compatibility (old blocks still
read it), but mark it as deprecated.

**Why:** the correction table's role is replaced by the probe's absolute
measurements. Eliminating it removes the band-aid.

### Phase 4: Make the probe run at target context (already planned as C1)

**File: `memory_probe.py::_probe_grid`** (already changed in C1)

The probe grid now measures at design context (p=1) instead of 65k. The
constants are measured at the context the model actually serves.

**Why:** constants are context-independent by construction, but compute shaping
is context-dependent for some models. Measuring at design ensures accuracy.

### Phase 5: Handle the fallback gracefully

**File: `vram.py::fit_params_static`**

When no probe-measured block exists, use fit-params (unsound). Warn the user:
```
WARNING: no probe measurement for arch 'qwen35'; using fit-params estimate
(unsound). Run --probe-memory to calibrate.
```

This makes the unsoundness explicit rather than hidden behind a correction.

**Why:** the user should know when the estimate is unsound.

### Phase 6: Update tests

- `test_memory_probe.py`: update `affine_report` tests to check that the
  measured constants are persisted (not deltas).
- `test_fit_params.py`: add tests for probe-measured blocks.
- `test_vram_budget.py`: update to reflect the new priority order.

## What this changes

| aspect | before | after |
|---|---|---|
| Fast path constants | fit-params + correction | fit-params (unsound) or probe (sound) |
| Correction table | primary calibration mechanism | eliminated (or allocator-only) |
| Probe output | deltas (serve-corrections.yaml) | absolute FitParams (sidecar derived:) |
| Soundness | unsound (fit-params + band-aid) | sound (probe) or explicitly unsound (fit-params) |
| Speed | fast (~2s/model) | slow (~2min/model) for sound, fast for fallback |

## What this does NOT change

- The affine law (sound, R²=0.999961)
- c and D from fit-params (exact, derived from pool lines)
- The probe's measurement methodology (already sound)
- The sidecar persistence mechanism (already exists)

## Migration

1. Old blocks (source="fit-estimate" with correction table) still read the
   correction table (backward compat).
2. New blocks (source="llama-server") are sound and don't need the table.
3. The correction table is deprecated but not removed (for old blocks).
4. Eventually, all blocks should be re-probed (source="llama-server").

## Open questions

1. Should the probe be run automatically during `--pack`? (Currently opt-in.)
2. Should the fast path warn when uncalibrated? (Currently silent.)
3. Should we remove the correction table entirely, or keep it for old blocks?

## Order of implementation

1. Phase 1 (probe persists absolute values) — unblocks everything
2. Phase 2 (normal runs prefer probe-measured) — makes the sound path primary
3. Phase 3 (eliminate correction table) — removes the band-aid
4. Phase 4 (target context) — already done (C1)
5. Phase 5 (graceful fallback) — makes unsoundness explicit
6. Phase 6 (tests) — validates the new architecture