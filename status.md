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
  already is, not in the cwd. `--print-paths`, `--dir`, `extras/engines.ini`.
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

1. **GPU invisible under the service.** Both Vulkan and HIP report
   "registry X has no device 0" under systemd but work in a shell as the
   same user. The unit's hardening (PrivateDevices=true, SystemCallFilter,
   etc.) is the suspect. `/tmp/gpu-bisect.sh` is ready to run as root.
   This blocks GPU chat and GPU audio.

2. **model_mib undercount.** The fit-estimate undercounts by ~446 MiB on
   Dirk (18569 estimated vs ~19015 measured). The correction overshoots
   by +1.09 GiB. The KV term is accurate (0.037109375 matched exactly).

3. **ubatch 512 is pinned** in profiles.yaml based on 4B measurements.
   At 27B scale, ubatch doesn't affect VRAM or throughput. The pin is
   harmless but the reasoning was wrong.

4. **Audio HIP build installed** but can't be used until GPU visibility
   is fixed. `backend: hip` in profiles.yaml, ready to switch.

5. **Per-process VRAM** not available for Vulkan workloads. debugfs
   (`/sys/kernel/debug/dri/`) has per-client directories but needs root
   to read. amd-smi only tracks ROCm/HIP, not Vulkan.

## Leftovers (not deleted)

- `/mnt/pool/data/tools/audiocpp` (230 MB) — old audio install
- `/mnt/pool/data/tools/whisper` (69 MB) — old whisper install
- `/mnt/pool/data/tools/llama/stable-diffusion-827` (1.3 GB) — old sd build
- `/mnt/pool/data/tools/llama/llama-b11539/11621/11658` — old llama builds
- `/mnt/pool/data/tools/llama/llama-swap-v262` — old swap binary

## Next steps

1. Run `sudo /tmp/gpu-bisect.sh` to find the sandbox directive hiding the GPU.
2. Fix the directive in `extras/llama-swap.service.in` and re-render.
3. Switch `audio_cpp.backend` to `hip` and repack.
4. Run `--probe-memory qwen35` to get measured constants for the sidecar.
5. Consider `--probe-memory` for other arch families (gemma-4, qwen3-vl, etc.).
EOF
echo "written"