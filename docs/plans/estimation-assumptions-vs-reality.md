# Measurement assumptions vs reality — facts collected 2026-10-18

## What the code assumes

1. llama_packer/consts.py: `_LLAMA_CPP_DEFAULT_UBATCH = 512`, `_LLAMA_CPP_DEFAULT_BATCH = 2048`.
   Comment: "the compute buffer depends on batch size and flash attention" — but
   only ubatch is stamped into the shape (batch is excluded: "measured to have
   no VRAM effect").

2. llama_packer/vram.py: FitParams block = model_mib, kv_per_token_mib, slot_mib,
   compute_mib, keyed by (model, cache_type, shape). shape is the effective flag
   string (profiles global args + resolved ubatch). A shape mismatch rejects the
   block -> falls to estimates.

3. measurement_args stamps `"{global_args} -ub {ubatch}"` using the *effective*
   ubatch (resolved from the cascade; if None, the constant 512 is used). So
   even a model with no ubatch config is measured with -ub 512 stamped.

4. Model sidecars store `derived: shape: "--flash-attn on -ub 512"` (11 found
   under /mnt/ai/models/chat/*/*.md). Source is `fit-estimate` for those.

5. The docs claim: "the compute buffer depends on batch size and flash attention"
   (vram.py docstring: "-ub shapes the compute buffer").

## What was actually measured

A. llama-bench on 4B model (Vulkan, R9700):
   ubatch 256 -> 675 MiB VRAM peak; 512 -> 828 MiB; 1024 -> 828 MiB.
   So: compute buffer DOES grow with ubatch, ~153 MiB between 256 and 512.
   (Consistent with the assumption.)

B. llama-bench on 27B model (Vulkan, R9700):
   ubatch 256/512/1024 -> 21658/21658/21808 MiB (flat within noise).
   So: at 27B scale, the compute buffer is in the noise vs weights+KV.
   The "compute buffer depends on ubatch" claim is true but the magnitude
   is model-size dependent.

C. Dirk 256k context load (real, measured):
   - ubatch: 512 (fleet default, from config).
   - estimated: 28.88 GiB; observed 29.69 GiB (+0.81).
   - Miss is in model_mib (estimated 18569 vs measured ~19015 MiB), NOT in
     compute or KV. compute_mib 676 matched (within the 0.81 gap). KV term
     0.037109375 MiB/token matched (0.0371 x 262144 = 9728, consistent).
   - So the measurement infrastructure is roughly right; the miss is in the
     weights term.

D. GPU visibility under the service:
   - As user 'lemonade' (sudo -u): llama-server --list-devices shows the
     R9700. As 'claude' shell: shows it. Under the systemd unit: neither
     Vulkan nor HIP enumerate any device; llama.cpp falls back to CPU
     (measured: 7.8 cores busy, VRAM flat at ~2.8 GiB during a chat load).
   - The unit's PrivateDevices=true + DeviceAllow lists are the suspect.
   - ProcSubset=pid was removed (it hid /proc/meminfo); that fixed the
     sys-stats errors but NOT the GPU visibility.
   - RESOLVED 2026-10-19 (directive bisect, transient units, rd1–rd5):
     PrivateDevices=true is the culprit and it is not about access —
     systemd 259's private /dev never materialises the DeviceAllow'd
     /dev/dri/* and /dev/kfd nodes (raw open inside the namespace → ENOENT;
     earlier "nodes present" ls was an instrument bug that ran the probe
     unconfined). Every fix spelling on top of PrivateDevices fails
     (explicit paths, rwm modes, char-dri alias). DevicePolicy=closed +
     the same DeviceAllow list enumerates 1 device with all other
     hardening intact; the meta-alias char-dri does not work under closed.
     Template now: DevicePolicy=closed + @DEVICE_ALLOW@.

E. audio.cpp under the service:
   - backend: hip (source-built for gfx1201) and vulkan both fail with
     "registry X has no device 0 (available: CPU:0)". backend: cpu works
     (TTS 200, ASR 200, RTF 0.096).
   - The build itself enumerates the card when run in a shell.

## Discrepancies between code assumptions and measurements

1. The code assumes ubatch affects VRAM (compute buffer). MEASURED: yes at
   small scale (153 MiB between 256 and 512 on a 4B), no at 27B scale
   (flat). The assumption is true in principle but the magnitude is not
   uniform; the measurement infrastructure stamps -ub into the shape, which
   is correct given the assumption.

2. The estimate of model_mib is off by 446 MiB for Dirk (18569 vs ~19015).
   This is a real error: the sidecar's `file.size_mb` is 19909, but
   model_mib is 18569. The difference is ~1.3 GB of VRAM the estimator
   thinks is free but isn't. Source is `fit-estimate`, not measured.

3. The `compute_mib: 676` measured at -ub 512 matches the measured run
   (within the 0.81 GiB gap, which is dominated by the weights miss). So
   the ubatch->compute assumption is at least internally consistent with the
   measured constants.

4. The VRAM budget math: 32624 MiB total. Dirk@256k uses 30409 MiB measured.
   The planner would have estimated ~28.9 GiB. So the planner is optimistic
   by 0.8 GiB for this model — could cause an over-admission.

5. `serve-corrections.yaml` (arch-level corrections) applies even on shape
   mismatch with a warning. Its shape is separate from the per-model shape.

## What was NOT measured

- Whether ubatch 256 vs 512 changes VRAM on the 27B model (predicted: no,
  because the compute buffer is ~0.66 GiB either way and the VRAM is flat).
- The actual compute buffer size at ubatch 256 on this card.
- Whether the GPU is visible under the service at all (bisect pending).
- Throughput at 256k context (only load was tested, not tok/s).
