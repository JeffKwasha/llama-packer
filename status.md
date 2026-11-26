# Status — 2026-10-18

## Working

- **VRAM estimation validation:** Dirk (Qwen3.8-27B) measured at 262144 context, 1 slot:
  - Observed: 29820 MiB (29.12 GiB), estimate: 28.88 GiB, gap +0.24 GiB
  - Fit-estimate is closer than the corrected estimate (correction overshoots by 1.09 GiB)
  - Sidecar updated with measured constants (`source: llama-server`)
  - Notes: `docs/plans/dirks-256k-context-measurement.md`

- **Audio entry-id fix:** entry key == server.json id for all 7 audio entries.
  Committed (bb60b68). Deployed and verified: TTS 200, ASR 200.

- **Install location resolution:** `update.py` updates each engine where it
  already is, resolved from the run directory. `--print-paths`, `--dir`.
  Committed (6675da2).

- **Tools/ layout:** `/opt/llama/tools/` holds all five engines with stable
  symlinks. `llama-server -> ./llama-b11717/llama-server`,
  `audiocpp_server -> audiocpp_server-v0.9.2/audiocpp_server`, etc.

- **bench.py:** performance history tool. 34 tests. Default = largest served
  model (not a 0.6B proxy). Committed (83a1fcf).

- **build_engine.py:** compile audio.cpp from source for local GPU arch.
  HIP build for gfx1201 installed and working in a shell. Committed (9966e07).

- **gpu_vram_monitor.py:** localhost VRAM API. Global counter + process
  list (contamination check). Committed (a13341f).

- **Estimation facts:** `docs/plans/estimation-functions-facts.md` documents
  the affine model, all constants, measurement paths, and where estimates
  diverge from reality. vLLM marked untested.

- **Tests:** 1044 passing. check_trace and check_mutation clean.

## Serving status

- **Service:** running from `/opt/llama/tools/llama-swap`, config reloaded.
- **Chat:** 200 (b11717 fingerprint, GPU invisible — runs on CPU).
- **TTS (kokoro-82m.s):** 200 audio/wav, 117644 B.
- **ASR (parakeet-tdt-0.6b-v3.a):** 200, transcribed correctly, RTF 0.080.
- **Audio backend:** cpu (Vulkan and HIP both see no device under the service).

## Open issues

1. **GPU invisible under the service — RESOLVED (2026-11-21).** The unit
   uses `DevicePolicy=closed` with explicit `DeviceAllow` for
   `renderD128`/`card1`/`kfd` and no `PrivateDevices`; the service user is
   in `render`. A live check confirmed the service child opens the GPU.
   The old bisect script is gone. A shell user also needs `render` (and
   `video`) for `vulkaninfo`/`--probe-memory` to see the R9700.

2. **model_mib undercount — re-tested, not an estimator error.** Dirk's
   main GGUF is gone. On Qwen3.8-27B-hermes-Q6_K (b12600, `-ub 512`,
   `-c 8192`, 66/66 layers offloaded), fit-params reports `model_mib`
   20054, exactly the serve-shaped `Vulkan0 model` 20054. The GGUF file is
   21391 MiB: the buffer is smaller than the file, not under-measured.
   What *is* off is `compute_mib`: fit-params 781 vs serve 160 at `-ub 512`
   — the per-arch compute correction needs re-derivation, not the weight
   term. KV per token matched (fit 0.033203; serve KV 661 MiB at 8192).

3. **ubatch 512 — the pin is not harmless at 27B.** Same model and shape
   above: serve `Vulkan0 compute` 160 MiB at `-ub 512` vs 640 MiB at
   `-ub 2048`; fit-params fixed term 20835 vs 23626 MiB. A ubatch change
   re-measures (the `derived: shape` stamp includes `-ub`), so the cost is
   a stale estimate, not a wrong served allocation — but the claim that
   ubatch has no VRAM effect at 27B is false.

4. **Audio HIP — no code blocker.** No HIP build is present in
   `/opt/llama/tools`; the old note predates the installer change. This
   is a deployment choice, not a packer defect.

5. **Per-process VRAM — own servers yes, service no (2026-11-21).** DRM
   fdinfo exposes `drm-resident-vram`/`drm-resident-gtt` per GPU fd with no
   root; `extras/gpu_vram_monitor.py` reads it and reports
   `attribution.source`/`attribution.complete` in every sample. Verified on
   a 0.6B server we started: 695.2 MiB VRAM, 11.1 MiB GTT. Failure mode the
   block exists to expose: fdinfo showed a clean figure for that server and
   *no* processes for the `lemonade`-owned llama-swap children — GPU in use,
   attribution empty. /proc access is uid-scoped. `systemd-journal` access
   was granted and works, but llama-swap logs only its own 13 lines/day;
   child `-lv 5` buffers are not forwarded, so the journal cannot substitute
   without a deployment change. For the live service use the global counter
   before/after a load. `amd-smi` is HIP-only; `nvtop -s` shows fdinfo.

## Leftovers (not deleted)

- `/mnt/pool/data/tools/audiocpp` (230 MB) — old audio install
- `/mnt/pool/data/tools/whisper` (69 MB) — old whisper install
- `/mnt/pool/data/tools/llama/stable-diffusion-827` (1.3 GB) — old sd build
- `/mnt/pool/data/tools/llama/llama-b11539/11621/11658` — old llama builds
- `/mnt/pool/data/tools/llama/llama-swap-v262` — old swap binary

## Next steps

1. Persist the qwen35 correction: b12600 serve-shaped runs gave `Vulkan0
   compute` 160 MiB at `-ub 512` vs 640 MiB at `-ub 2048`, while fit-params
   fixed term was 20835 vs 23626. The ~621 MiB overstatement at `-ub 512` is
   the per-arch compute correction to file. `--probe-memory qwen35` refuses
   while the llama-swap service process is running; use a quiet window or the
   direct serve points above.
2. Consider `--probe-memory` for other arch families (gemma-4, qwen3-vl, etc.).
3. Per-process VRAM is live via fdinfo for own measurement servers; the
   service's lemonade-owned children still need journal parsing or the global
   counter.
4. Alias routing was probed live: `:tool` aliases appear in `/v1/models` with
   the same top-level capabilities and context as their base id (the only
   metadata differences are `aliases` on the base and `modelID` on the alias).
   No capability leak was found; the alias-rework follower is not blocked.
