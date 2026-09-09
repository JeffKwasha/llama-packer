# GB10 / DGX Spark vLLM support — llama-packer change plan

**Status:** implemented on `gb10-support` (261 tests passing; see "Amendment summary")
**Date:** 2026-09-09 (reviewed & amended, implemented 2026-09-09)
**Branch:** `gb10-support` (new feature branch off `main`; supersedes the stale
`vllm-gb10` scaffold branch — the backend scaffold itself is already on `main`)

**Amendment summary (2026-09-09 review against llama-swap v253 public docs and
the deployed code):** the old C6+C7 are merged and corrected into the new C6
(unified docker path mapping + mounts — which also fixes a *latent bug*: local
model refs were never rewritten into the container, see C6). New C7 (container
lifecycle emission: `cmdStop` + `unloadTimeout`), C8 (healthCheckTimeout vLLM
floor) and C9 (proxy emission) come from llama-swap's documented docker model:

- https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/ttl-and-unloading.md
- https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/writing-cmd.md

**Audience constraint:** the implementer has **NO access to spark2.humaii**. Everything
empirical about the target is captured in [Ground truth](#ground-truth-captured-2026-09-09-from-spark2humaii)
below. Do not attempt to SSH; treat that section as the source of truth and validate
flag names/values against `vllm serve --help` in a local copy of the target image
(see [Verification](#verification-without-spark2-access)).

---

## Why this plan exists

llama-packer already emits a working vLLM backend (`vllm` host + `vllm-docker`) and
already handles GB10 **unified-memory VRAM detection**. What it does *not* emit is the
set of flags that modern Blackwell-targeted models actually need to serve on the Spark.
The live, hand-launched serving command on spark2 (captured below) is the ground truth
for the gap. This plan closes that gap with a small number of explicit, opt-in frontmatter
keys plus four docker-container correctness items (speculative-draft/model path mapping
and HF-cache mounting, container lifecycle emission, health-check budget, proxy emission).

**Non-goals (deferred to the spark2 phase, not this repo):** OS/driver updates, docker
image builds, llama-swap deployment on the host, NAS cache cleanup, service units. Those
are tracked separately; a handoff checklist is at the end.

---

## Ground truth (captured 2026-09-09 from spark2.humaii)

### Hardware / software stack
| Item | Value |
|------|-------|
| Machine | NVIDIA DGX Spark, **GB10** (Blackwell), aarch64 |
| Memory | Unified 128 GB LPDDR5x. `nvidia-smi` reports `[N/A]` for `memory.total` and `Not Supported` in the table — **expected**, not a fault. |
| Driver | **580.173.02** (deliberately held back; recent drivers unstable on this Blackwell variant) |
| CUDA (driver-visible) | 13.0 |
| Host toolkit `nvcc` | 12.0 (irrelevant to runtime; the container carries its own CUDA) |
| OS / kernel | Ubuntu 24.04.4 LTS, kernel `6.17.0-1032-nvidia` |
| Serving vLLM image | `vllm/vllm-openai:v0.27.1` (CUDA 13.0.2 runtime). Also present locally: `nvcr.io/nvidia/vllm:25.12-py3`, `nvcr.io/nvidia/vllm:25.12.post1-py3`, custom `vllm-node-v27` (based on v0.27.1), `eugr/spark-vllm`. |
| Container | Persistent `vllm_node` container (image `vllm-node-v27`, entrypoint `sleep infinity`); vLLM started by hand via `docker exec`. |

### The live GB10 vLLM command (ground truth for required flags)
Serving `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (a **MoE + Mamba hybrid**,
NVFP4, with dSpark speculative decoding). Flag-for-flag:

```
vllm serve /root/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/e8f3c7c4de75ad84fe1bcef95d38eca76214480b \
  --served_model_name nemotron3.5 \
  --trust-remote-code \
  --quantization modelopt_mixed \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.78 \
  --max-num-seqs 4 \
  --moe-backend marlin \
  --kv-cache-dtype fp8 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --speculative_config.method dspark \
  --speculative_config.model /root/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark/snapshots/d10c6ff40d6e69d1f92e407e027de3eafdb77645 \
  --speculative_config.num_speculative_tokens 3
```

Flags llama-packer **already** emits for this model: `--model`, `--served-model-name`,
`--host/--port`, `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs` (from
`parallel`), `--kv-cache-dtype fp8` (from `cache_type: q8_*`).

Flags llama-packer does **not** emit today (the gap this plan closes):
- `--quantization modelopt_mixed`
- `--moe-backend marlin`
- the five `--mamba-*` / mamba-cache flags
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
- `--reasoning-parser nemotron_v3`
- speculative config with a **local-path draft model** (`dspark` + `model <path>`) —
  llama-packer emits the *JSON* form of `--speculative-config`, which is equivalent, but
  it does **not** bind-mount/remap a local draft-model path into the container.

Fleet-wide flags that are fine to leave in `vllm.args` (no code change): `--enable-prefix-caching`,
`--max-num-batched-tokens <N>` (already mapped from `batch:`), `--trust-remote-code`.

### Container mounts (the correct HF layout, done manually today)
```
/nas/humaii/config/models/.hf  -> /root/.cache/huggingface   # HF hub, on the 21T NAS
/home/graphx3d/.cache/vllm     -> /root/.cache/vllm          # vLLM compile cache
/home/graphx3d/.cache/flashinfer -> /root/.cache/flashinfer
/home/graphx3d/.triton         -> /root/.triton
/home/graphx3d/.tilelang       -> /root/.tilelang
```
The HF hub is confirmed on the NAS (`df` → `192.168.7.114:/volume1/Humaii/config/models/.hf`, 21T).

### Host HF env + stray cache (context for the spark2 phase, not this repo)
- Host `HF_HOME=/home/graphx3d/models/nas/.hf` → resolves via symlink to `/nas/humaii/config/models/.hf`. **Already correct** (HF on NAS).
- A **stray ~108 GB local cache** exists at `/home/graphx3d/models/.cache/huggingface` (local disk, not NAS). It largely duplicates models already on the NAS hub but a few entries may be unique. Cleanup/migration is a **spark2-phase** task (audit against NAS first — do not blind-delete).

### What already works (no change needed)
- **Unified-memory VRAM detection** — `llama_packer/hardware.py`: `_detect_vram_nvidia()`
  returns `None` on the `N/A`/`Not Supported` report; `_detect_pool_mb()` falls back to
  total system RAM with `is_unified=True`; `GpuProfile.from_args()` folds a system reserve
  (default 8192 MiB, overridable via `hardware.unified_system_mb`). Verified present in code.

---

## Design principles

1. **Opt-in, verbatim, no guessing.** Every new key emits its flag only when explicitly
   set. No architecture sniffing, no auto-derived values — the operator knows the model's
   recipe (that is how it is launched today). This keeps risk low on experimental
   Blackwell flags.
 2. **Host and docker parity.** `--quantization`, `--moe-backend`, `--mamba-*`,
    `--tool-call-parser`, `--reasoning-parser` are plain `vllm serve` flags, identical in
    both modes → implement once in `VllmHostBackend._serve_flags`. Docker-only concerns —
    path mapping + mounts (C6), lifecycle emission (C7), health budget (C8), proxy (C9) —
    live in `VllmDockerBackend` / the writer.
3. **One source of truth per flag.** A key maps to exactly one flag (or a fixed small
   group, e.g. tool parser ⇒ two flags). No free-form flag injection for these — that
   remains available via the existing `cli_args:` if ever needed.
4. **Do not repurpose existing keys.** `quantization` is already a *metadata* field
   (bits-per-weight, see `Model._quant_bits`, model.py:1467). The vLLM weight-quant
   method gets a **new** key (`vllm_quantization`) to avoid the collision.

---

## Change specifications

All new frontmatter keys are **serving choices**, so they must be settable by override
rules and directory defaults. That means registering them in the shared key sets
(see [Registration points](#registration-points-cross-cutting)) *and* adding accessor
properties on `Model` following the existing `vllm_image` / `reasoning_format` pattern
(model.py:1254-1262, 1191-1205).

### C1 — Weight quantization → `--quantization`
- **Key:** `vllm_quantization:` (string) → `--quantization <value>` verbatim.
- **Why a new key:** `quantization` is taken (metadata). vLLM auto-detects most quants
  from the checkpoint's `config.json`, so this is only needed for cases like
  `modelopt_mixed` where explicit selection matters. Emit only when set; otherwise let
  vLLM auto-detect (do **not** emit a default).
- **Emit site:** `_serve_flags`, after the `--gpu-memory-utilization` block (vllm.py:206-212).
  Order is not semantically significant for vLLM; keep it grouped with the other model
  config flags.
- **Accessors:** `Model.vllm_quantization` property (str | None).
- **No** mapping table from the metadata `quantization` name — deliberately (lossy; e.g.
  "NVFP4" does not imply `modelopt_mixed`).

### C2 — MoE backend → `--moe-backend`
- **Key:** `moe_backend:` (string) → `--moe-backend <value>` verbatim (e.g. `marlin`,
  `cutlass`, `triton`).
- **Emit site:** `_serve_flags`. Emit only when set and role is a generation/embedding
  task that has MoE (in practice: always emit when set; vLLM ignores it for non-MoE
  models).
- **Accessor:** `Model.moe_backend` property.

### C3 — Mamba / hybrid family → five flags
Modern Blackwell-targeted hybrids (Nemotron‑3.5‑Lightning) need a coherent mamba set.
Model it as one structured key:

```yaml
mamba:
  backend: flashinfer            # -> --mamba-backend flashinfer
  ssm_cache_dtype: float16       # -> --mamba-ssm-cache-dtype float16
  stochastic_rounding: true      # -> --enable-mamba-cache-stochastic-rounding   (bool, emit flag when true)
  philox_rounds: 5               # -> --mamba-cache-philox-rounds 5              (int)
  cache_mode: align              # -> --mamba-cache-mode align
```

- **Key:** `mamba:` (mapping). Each present sub-key maps to exactly one flag as above.
  Absent sub-keys emit nothing. A bare `mamba: true` is **not** supported — require the
  mapping (warn-and-skip a boolean, mirroring how other malformed keys are handled).
- **Emit site:** `_serve_flags`. Add a helper `_mamba_flags(model) -> list[str]` next to
  `_kv_cache_dtype_flags` (vllm.py:42) for testability; call it in `_serve_flags`.
- **Accessor:** `Model.mamba` property returning the dict (or None).
- **Caveat (validate in spark2 phase):** these flags are experimental and vLLM-version
  specific. Confirm each name/value against `vllm serve --help` inside the target image
  before relying on it; if a flag is renamed in v0.27.x, adjust the sub-key→flag map here
  (single place to change).

### C4 — Tool-call parser → `--enable-auto-tool-choice --tool-call-parser`
- **Key:** `tool_call_parser:` (string) → emits **both** `--enable-auto-tool-choice` and
  `--tool-call-parser <value>`.
- **Emit site:** `_serve_flags`, chat role only (tool calling is a generation feature).
- **Accessor:** `Model.tool_call_parser` property.
- **Relationship to the existing `tools` capability / matrix demotion:** keep orthogonal.
  This key only emits the parser flags; it does not add/remove the `tools` capability or
  interact with `tools_demoted`. (If a model declares a tool parser, operators should also
  give it the `tools` capability in the sidecar — document that pairing.)

### C5 — Reasoning parser → `--reasoning-parser`
- **Key:** `reasoning_parser:` (string) → `--reasoning-parser <value>` (e.g. `nemotron_v3`,
  `deepseek_r1`).
- **Distinct from** llama.cpp's `reasoning-format` (`Model.reasoning_format`, model.py:1191)
  which emits `--reasoning-format`. Do **not** reuse that key; the vLLM flag is different.
  Document the distinction in SPEC.md and the sidecar template.
- **Emit site:** `_serve_flags`, chat role only.
- **Accessor:** `Model.reasoning_parser` property.

### C6 — Unified docker path mapping + mounts (merges the old C6 + C7)

llama-swap v253 has **no native container abstraction**: a dockerized backend is a normal
entry whose `cmd:` is `docker run … <image> <flags>` — parameters are argv after the image,
container env via `-e` in `cmd` (an entry's `env:` list reaches only the docker *client*
process; `writing-cmd.md`). Everything path-shaped inside `cmd` must therefore be a valid
**in-container** path. Today that contract is broken twice:

- **Latent bug (pre-existing):** `_serve_flags` renders `--model <host path>` verbatim
  (`_model_ref`, vllm.py:186) — the `map_path` callback is applied only to the chat
  template (vllm.py:225-228). A local safetensors under a `models_dirs` root is emitted as
  e.g. `--model /nas/.../snapshot` inside a container where only `/models` exists.
- **Planned C6 (old):** speculative draft local paths were not mapped either.

Fix with **one mapping step** for the docker backend, over all path-valued refs:
model ref, speculative draft paths (`model`, and defensively `draft_model` keys of
`_speculative_config`), and chat template.

**Roots and resolution order** (generalize `_map_paths_into`, vllm.py:137, to take
`hf_cache=None`; resolve each path with `Path.resolve()` first so symlinked layouts map by
*real* location):

1. Under `hf_cache` (host HF hub root) → `/root/.cache/huggingface/<rel>`. The whole root
   is mounted, so HF snapshot blob symlinks (`file → ../../blobs/<hash>`) resolve — this
   kills the snapshots-dir mount bug. Matches the live container's paths exactly
   (ground truth `--speculative_config.model` is `/root/.cache/huggingface/hub/...`).
2. Under any `models_dir` (e.g. `~/models`) → existing `/models`, `/models2`… refs; those
   dirs are already bind-mounted (vllm.py:309-312).
3. Else → existing `/extN` parent-mount fallback (fine for plain files like templates).

**Mounts emitted by `build_cmd`:** the existing `models_dirs` binds, plus
`-v <hf_cache>:/root/.cache/huggingface` when `hf_cache` resolves, plus `/extN` fallbacks.

**Env, every vllm-docker entry:** `-e HF_HOME=/root/.cache/huggingface` and
`-e HF_HUB_OFFLINE=1`. Policy: llama-packer never downloads; an uncached repo-id model
fails fast at startup instead of silently pulling tens of GB into the container. No
`HF_TOKEN` (nothing may download).

**Config surface:** `profiles.yaml → vllm.hf_cache:` (host HF hub root, e.g.
`/nas/humaii/config/models/.hf`), falling back to top-level `hf_home:` when unset.
New template var in `__main__.py` next to the other vLLM vars (~line 645). When
`hf_cache` is empty **and** `_model_ref` is a repo id, warn at pack time (the entry
cannot work offline).

**Host backend:** unchanged (host paths are already correct).

**Implementation notes:**
- `BACKENDS` registry entries are **singletons** — no per-call instance state. Extend
  `_serve_flags` with an optional `model_ref: str | None = None` parameter instead (default
  `None` → `self._model_ref(model)`; docker passes the mapped container ref).
- Scan spec-config dict for existing local file/dir values under keys `model` /
  `draft_model`; if vLLM introduces other path-bearing spec keys, extend the scan list in
  one place. Repo-id spec models (`method: draft_model, model: org/repo`) are left as-is —
  they resolve through the mounted HF cache.
- Keep the JSON emission (`--speculative-config <json>`); rewrite refs *inside* the dict
  before dumping.

### C7 — Container lifecycle emission (vllm-docker only)

llama-swap's documented docker orchestration (`ttl-and-unloading.md`): *"Docker needs one
extra setting: use `cmdStop: docker stop ${MODEL_ID}` so an unload stops the container,
not only the local `docker run` client process"*, and *"It applies to every unload — TTL
expiry, a manual unload, or a swap"*; *"Too low [unloadTimeout] and llama-swap force-kills
a process mid-shutdown, which can leave a container running and its VRAM held."* Our
architecture swaps matrix sets constantly, so this fires on every swap of a docker entry.

- `BaseBackend` gains two ClassVars: `stop_cmd: str | None = None` and
  `unload_timeout: int | None = None` (base.py).
- `VllmDockerBackend`: `stop_cmd = "docker stop ${MODEL_ID}"` (matches the `--name
  ${MODEL_ID}` already in the cmd; `${MODEL_ID}` is valid in any model field per the
  upstream macro table) and `unload_timeout = 30` (upstream's docker example value).
- `writer._build_entry` (writer.py:419): when the backend sets them, emit
  `entry["cmdStop"]` and `entry["unloadTimeout"]`. No other backend sets them
  (kokoro-podman deliberately left as is for now).
- No profiles.yaml knobs — lifecycle correctness is structural, not a recipe choice.

### C8 — healthCheckTimeout vLLM floor

`_health_check_timeout` (`__main__.py:334`) currently raises to a 300 s floor when any
model uses a vLLM backend. The official Spark writeup reports **10–15 min** weight load
for a ~120B NVFP4 model — 300 s would kill a healthy slow start. Per the agreed formula,
assume a conservative **100 MB/s** load rate:

- `hct = max(hct, size_mb // 100)` over vLLM-backended models, using the local
  safetensors size (`gguf_path.stat()`) when available; the existing 300 s floor stays for
  repo-only models with no measurable size.

### C9 — proxy emission for vllm-docker entries

Per upstream (`writing-cmd.md`): *"Running in a container? Set it explicitly —
`proxy: "http://127.0.0.1:${PORT}"` … the single most common configuration error."* We
already publish `-p ${PORT}:{container_port}`, so the default would *usually* work — emit
the explicit form anyway, as the docs recommend:

- In `_build_entry`, alongside C7: `if backend.stop_cmd is not None: entry["proxy"] =
  "http://127.0.0.1:${PORT}"` (docker-lifecycle backends are exactly the container case).
- `checkEndpoint` stays at llama-swap's default `/health` — correct for vLLM.

### Registration points (cross-cutting)
Add each new key (`vllm_quantization`, `moe_backend`, `mamba`, `tool_call_parser`,
`reasoning_parser`) to **all** of these so it is recognized, overridable, and not flagged
unknown/unhandled:

| Location | What | Ref |
|----------|------|-----|
| `backends/base.py` `SETTING_KEYS` | canonical serving-choice key set (feeds `OVERRIDE_KEYS`) | base.py:38 |
| `Model.FIELDS` | keys the Model consumes (not leaked to metadata) | model.py:420-432 |
| `VllmHostBackend.handles` | keys this backend consumes (validation) | vllm.py:181 |
| `Model` accessor properties | one per key, `str`/`dict \| None` | model.py (near 1254) |

Notes:
- `OVERRIDE_KEYS` (overrides.py:33) = `*SETTING_KEYS + {cache_type, parallel, mmproj, speculative}` —
  adding to `SETTING_KEYS` is sufficient for override-rule support; do not add separately.
- These keys are **serving choices**, so they must **not** go into `scope.INTRINSIC_KEYS`
  (scope.py:41) — that set is for model identity and warns when set in fleet defaults.
- `_strip_llama_only_flags` (vllm.py:66) does not list any of the new flags, so they are safe;
  but note `--reasoning-format`/`--reasoning-budget` *are* stripped — `--reasoning-parser` is a
  different string and is unaffected.
- **C6–C9 touch no model-key registration** — they are profile/template-var level
  (`hf_cache`) and backend/writer-level (path mapping, `cmdStop`/`unloadTimeout`,
  health timeout, proxy), not sidecar keys.

---

## profiles.yaml changes (bundled template + docs)

In `llama_packer/profiles.yaml` under the existing `vllm:` section, document (commented or
as safe defaults) the new knobs. Suggested shape:

```yaml
vllm:
  image: vllm/vllm-openai:latest        # pin a GB10-known-good (sm_121-validated) tag
  bin: vllm
  docker_args: "--runtime=nvidia --gpus all --shm-size=16g"
  container_port: 8000
  # gpu_mem_util: 0.9                   # omit to derive from the VRAM budget
  hf_cache: ""                          # host HF hub root mounted into every
                                        # vllm-docker container at
                                        # /root/.cache/huggingface (falls back to
                                        # top-level hf_home:); empty = warn on
                                        # repo-id models (offline policy forbids
                                        # downloads)
```

`docker_args` is the operator's flexibility point for anything container-runtime
specific: GPU device selection (`--gpus device=N` / `-e CUDA_VISIBLE_DEVICES=N`),
`--ipc=host` vs `--shm-size`, and extra read-only perf-caches binds (vLLM/flashinfer/
triton compile caches) — no per-item keys are added for these.

Per-model vLLM recipe keys (sidecar or override rule), GB10 examples:
```yaml
#   backend: vllm-docker
#   vllm_quantization: modelopt_mixed   # -> --quantization
#   moe_backend: marlin                 # -> --moe-backend
#   tool_call_parser: qwen3_coder       # -> --enable-auto-tool-choice --tool-call-parser
#   reasoning_parser: nemotron_v3       # -> --reasoning-parser
#   mamba:                              # hybrid / Mamba models
#     backend: flashinfer
#     ssm_cache_dtype: float16
#     stochastic_rounding: true
#     philox_rounds: 5
#     cache_mode: align
```

`hardware:` needs no change (unified detection already works); optionally document that a
GB10 host may pin `hardware.vram` / `hardware.unified_system_mb` to make the budget explicit.

---

## Documentation updates
- **SPEC.md** "vLLM Backend": add the new recipe keys, their flags, precedence (sidecar >
  override rule > fleet default), and a "Container (vllm-docker)" subsection covering:
  mounts (models_dirs, hf_cache), path rewriting (C6), `HF_HUB_OFFLINE=1` policy (no
  downloads), `cmdStop`/`unloadTimeout` (C7), explicit `proxy` (C9), the
  `reasoning_parser` vs llama.cpp `reasoning-format` distinction, and that `docker_args`
  is the flexibility point for device/shm/ipc tuning.
- **`docs/reference.md`**: add the two upstream llama-swap KB links (ttl-and-unloading,
  writing-cmd) that back the emitted lifecycle fields. (There is no `docs/llama-swap.md`
  feature table in this repo.)
- **README.md**: one line that GB10/DGX Spark is a supported vLLM target with per-model recipe keys.
- **`llama_packer/templates/models_AGENTS.md`** (the sidecar guide written by `--agents`): add the
  new keys to the vLLM section so operators discover them; note that `vllm_quantization` is not
  the same as the metadata `quantization`, and that HF models must be pre-staged (no downloads).
- **`docs/plans/vllm-gb10.md`**: update "Planned (not yet implemented)" — move MTP/spec, and add
  these items as implemented once merged.

---

## Test plan (tests/test_backends.py)
Follow the existing patterns in that file (`_tvars()`, `make_model(...)`, assert on the
returned `cmd` string). Add:

- `test_vllm_quantization_flag`: model with `vllm_quantization="modelopt_mixed"` →
  `"--quantization modelopt_mixed" in cmd`; absent → not in cmd. Host + docker.
- `test_vllm_moe_backend_flag`: `moe_backend="marlin"` → `"--moe-backend marlin" in cmd`.
- `test_vllm_mamba_flags`: full `mamba:` mapping → all five flags present with correct
  values; partial mapping → only the set sub-keys emit; `mamba: true` (bool) → warn + no
  mamba flags (use `caplog`).
- `test_vllm_tool_call_parser`: `tool_call_parser="qwen3_coder"` (chat role) → both
  `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder`; non-chat role → omitted.
- `test_vllm_reasoning_parser`: `reasoning_parser="nemotron_v3"` → `--reasoning-parser nemotron_v3`;
  assert it is NOT stripped (i.e. present in final cmd) and distinct from `--reasoning-format`.
- `test_vllm_spec_draft_local_path_docker` (C6): docker backend, `speculative_config` with a
  local draft path under `hf_cache` → the ref (`/root/.cache/huggingface/...`) appears inside
  the emitted `--speculative-config` JSON and the `-v <hf_cache>:/root/.cache/huggingface`
  mount is present; draft path under a `models_dir` → `/models/...` ref with no extra mount;
  draft path elsewhere → `/extN` mount; host backend keeps the raw host path.
- `test_vllm_docker_model_ref_mapped` (C6): local safetensors under a `models_dir` →
  `--model /models/...` (fixes the latent host-path leak); under `hf_cache` →
  `/root/.cache/huggingface/...`; host backend keeps the raw host path.
- `test_vllm_docker_hf_mount_offline` (C6): docker backend →
  `-v <hf_cache>:/root/.cache/huggingface`, `-e HF_HOME=/root/.cache/huggingface`, and
  `-e HF_HUB_OFFLINE=1` present; empty `hf_cache` → no mount and a pack-time warning for a
  repo-id model.
- `test_vllm_docker_cmdstop` (C7): emitted *entry* contains `cmdStop: docker stop
  ${MODEL_ID}` and `unloadTimeout: 30` (writer-level test); llama-server/host-vllm entries
  do not.
- `test_vllm_docker_proxy` (C9): entry contains `proxy: http://127.0.0.1:${PORT}`.
- `test_health_check_timeout_vllm` (C8): a fleet whose largest vLLM model is N MB gets
  `healthCheckTimeout >= N // 100` (and >= 300 with a repo-only vLLM model).

Extend `_tvars()` (test_backends.py:~29) to include `hf_cache` (default `""`) so existing
tests are unaffected. Run the suite per AGENTS.md:
`PYTHONDONTWRITEBYTECODE=1 /var/uv/env/bin14/bin/python -m pytest -q -p no:cacheprovider`

---

## Verification without spark2 access
1. **Unit tests** above pass (the primary gate — they assert exact emitted strings).
2. **Golden emission:** create a throwaway models dir with one sidecar exercising every new
   key (`backend: vllm-docker`, all recipe keys set, `hf_cache` pointing at a fake HF root),
   run the CLI to emit a config, and eyeball that the `cmd:` block matches the live
   ground-truth command's flag set and the entry carries `cmdStop`/`unloadTimeout`/`proxy`.
   This is the closest proxy to "would it launch correctly" available off-host.
3. **Flag-name sanity (optional, no GPU needed):** pull the target image locally
   (`docker pull vllm/vllm-openai:v0.27.1`) and run `docker run --rm <image> vllm serve --help`
   to confirm every emitted flag name is accepted by that exact version — catches C3 mamba
   renames without touching spark2. (Only if a local docker + GPU-less help works in the
   implementer's env; otherwise defer this check to the spark2 phase.)

### Handoff checklist for the spark2 phase (NOT part of this repo's work)
- [ ] Point `profiles.yaml vllm.image` at a **GB10/sm_121-validated image tag** (pin the exact
      tag or digest — the vLLM Spark writeup uses the `cu130-nightly` track and warns it moves)
      and set `vllm.hf_cache` to the NAS hub (`/nas/humaii/config/models/.hf`).
- [ ] Confirm the emitted docker command launches a real model end-to-end on the Spark;
      tune `vllm.docker_args` on-host if the `--runtime`/`--gpus`/`--ipc`/`--shm-size`
      interplay requires it (NVIDIA's own Spark recipe uses `--gpus all --ipc=host`).
- [ ] Validate C3 mamba flag names/values against the live image's `vllm serve --help`.
- [ ] **Lifecycle check:** load → swap → `/unload` a vllm-docker entry, then verify
      `docker ps` shows no orphaned container and VRAM is released (exercises the emitted
      `cmdStop`/`unloadTimeout`). If orphans appear, raise `unloadTimeout` — documented
      workaround fields, not a code change.
- [ ] **Decommission the manual `vllm_node` persistent container** (`sleep infinity` +
      hand-launched vLLM) once llama-swap serves the fleet — it collides on port 8000 and
      on the GPU.
- [ ] Pre-stage models on the NAS hub ("download once, mount everywhere" — downloads are
      disabled by `HF_HUB_OFFLINE=1`).
- [ ] Audit + migrate/prune the stray ~108 GB local HF cache onto the NAS (do not blind-delete).
- [ ] Decide OS/driver: **recommend no change** — driver 580.173.02/CUDA 13 is coherent and a
      cutting-edge model already runs on it; vLLM container images carry their own CUDA runtime,
      so the host driver only needs to be new enough (it is). Revisit only if a specific target
      model requires a newer CUDA/driver feature.

---

## Risks / open questions
- **Experimental flag drift (C3):** mamba/marlin/dspark flags are vLLM-version-specific and may
  have changed between the version that produced the live command and `v0.27.1`. Mitigation:
  single sub-key→flag map (easy to fix), plus the `--help` sanity check above.
- **`modelopt_mixed` scope:** we treat it as an opaque verbatim value. If more mixed-precision
  checkpoints appear that need different values, that is a per-model key setting, not a code change.
  (The vLLM Spark writeup suggests leaving `--quantization` unset for pre-quantized NVFP4
  checkpoints — vLLM auto-detects — so try without it first on spark2; the key exists because the
  live ground truth sets it.)
- **Spec draft key names (C6):** we remap `model` and `draft_model`. If vLLM introduces other
  path-bearing spec keys, extend the scan list in one place.
- **HF mount target (C6):** assumes `/root/.cache/huggingface` is the container default (true for
  the current images). `HF_HOME` is emitted explicitly, so a changed default is a one-value tweak.
  Note: with `HF_HUB_OFFLINE=1` a repo-id model that is *not* in the mounted hub fails at startup —
  intended (fail fast) but the spark2 agent should recognize it as the cache-miss case, not a bug.
- **Container stop grace (C7):** vLLM teardown can exceed the default 10 s; `unloadTimeout: 30`
  is upstream's own docker example value. If spark2 teardown is slower, raise the constant
  (single place: `VllmDockerBackend.unload_timeout`).
- **Compile caches are ephemeral (C6):** vLLM/flashinfer/triton JIT caches live in the container
  layer, so every start pays JIT cost unless the operator adds cache binds via `vllm.docker_args`
  (documented; the manual `vllm_node` setup mounted four such dirs).
