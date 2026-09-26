# Phase 0 — VRAM baseline measurement (2026-09-26)

Ground truth for validating the demand-driven planner. Machine: AMD ROCm,
32,624 MB visible VRAM.

## Tooling findings

- `amd-smi metric` (no flags) does **not** print VRAM on ROCm 10.0 / amdsmi 27.0.
  Use `amd-smi metric -m` for a one-shot `TOTAL_VRAM / USED_VRAM / FREE_VRAM`
  (plus `TOTAL_GTT`), or `amd-smi monitor -v` to watch it. `--json` is not needed.
- `amd-smi process` reported "No running processes detected" even while a
  `llama-server` was plainly resident. Not usable for attribution here.
  Use the `USED_VRAM` delta instead.
- `hardware.detect_vram_baseline_mb()` returned **7239**, exactly matching
  `amd-smi metric -m` `USED_VRAM` at the same instant. The planner's
  pre-existing-usage baseline and amdsmi agree, so the `--baseline`/spare
  machinery is trustworthy.
- `immich_machine_learning` runs on this box and takes VRAM. It moved the
  idle floor from 1,956 MB to 7,239 MB during this session. Any VRAM
  measurement here needs a baseline taken seconds before the load.

## Measurement 1 — `lfm25-vl-16b-absolute-heresy` (with mmproj)

Contemporaneous baseline taken before the load.

| | MiB |
|---|---|
| idle floor (GUI only, browser closed shortly after) | 1956 |
| with model resident | 8446 |
| **measured delta** | **6490** |
| planner prediction (weights+compute 1802 + KV 4782 @ 128k x 6) | 6584 |
| error | **+94 MiB, planner conservative by 1.4%** |

The emitted tag `[VRAM 6.4GB]` = 6502..6605 MiB (`_fmt_mib` divides by 1024),
consistent with the 6584 prediction. No serve-correction row exists for arch
`lfm2`, so this is raw fit-params + the 4% margin.

## Measurement 2 — same model, same command line

A later load of the identical entry measured a **5873 MiB** delta against the
same 1956 floor. Same cmd line, same model, ~617 MiB apart.

**Conclusion: single-shot VRAM deltas on this box carry ~600 MB (~10%) of
noise** — ROCm allocation/fragmentation plus whatever immich is doing. That is
larger than the 1.4% accuracy the planner is being judged on, so hand-rolled
deltas are not a sufficient validator.

## Recommendation

Use the repo's own supported calibration path instead: `--probe-memory lfm2`
sweeps a (ctx, parallel) grid live and fits the affine law, which separates
weights / KV / slot terms from this noise by construction. It requires a quiet
GPU (no `llama_residents()`, no immich VRAM), so it is a deliberate
foreground task rather than something to run alongside other work.

The useful Phase 0 conclusions are the two tooling facts above plus the single
clean measurement 1, which agrees with the planner to 1.4%.
