# Memory-allocation description tags — v1 (shipped) + future plan

Status: **v1 implemented** on branch `description` (tag reports the decided
allocation verbatim in every emitted `config.yaml` description). This doc
records the v1 contract and plans the deferred variations: split
allocations (`+` spill), disk-resident weights (`SSD`), measured host RAM,
and MoE partial offload.

## v1 contract (shipped)

- The tag is a **verbatim report of the decided TOTAL allocation
  attributable to the entry** — weights + folded companions (mmproj, MTP
  draft) + context at the served `(ctx, parallel)`. It never caps,
  re-solves, or second-guesses: a decided `VRAM 73.0GB` on a 32 GB card
  prints as decided.
- Shapes: GPU `[VRAM 9.1GB RAM 1.2GB]` · CPU-only (`device: cpu`)
  `[RAM 2.3GB]` · each variant (main, `-text`, `-vision-Nk`) tagged
  independently.
- Units adapt so small values stay truthful: `≥ 1024 MiB` → `X.XGB`,
  below → integer `MB` (`RAM 64MB`, never `0.0GB`, never blank).
- `RAM` (GPU, v1) = file-size residue still host-side:
  `(main + included companions file size) − quad model_mib`, clamped ≥ 0.
  `RAM` (CPU) = full affine prediction evaluated on host.
- No tag only when nothing was decided (unestimated models —
  `metadata.estimated=false` already covers them).
- Plumbing: `Planner.plan()` computes per-variant numbers
  (`Planner._variant_memory`, `llama_packer/writer.py`) from the combined
  affine quad and carries them on `Variant.mem_*_mib`;
  `format_mem_tag`/`_with_mem_tag` format + append idempotently in
  `_build_entry`; `emit_config` stays math-free. Sidecar `description` is
  never written back.
- `VramBudget.fit_params_static/effective_static` take `allow_cpu`
  (`llama_packer/vram.py`): CPU-resident constants describe host RAM and
  are used for the tag only — never for VRAM sizing (all matrix/ledger
  call sites keep the default `False`).

## Future work (not started)

### 1. Split allocations — activating the `+` grammar

`format_mem_tag` already accepts `spill_mib` (`[VRAM 22.0GB + 1.2GB …]`)
but v1 never passes it: today's allocator never splits (full `-ngl 999`
or CPU). Work: decide *when* a split is the serving decision —
driver-managed overflow (AMD GTT in system RAM), explicit partial
offload (`-ngl N`), MoE expert placement (below) — and have the solver
emit `(device_mib, spill_mib)` instead of deriving the split at display
time. The tag keeps reporting verbatim; only the decider changes.

### 2. Disk-resident weights — the `SSD` token

Qwen3-Next-class models keeping a ~51 GB map on NVME: distinguish
mmap'd file pages (page cache, evictable, disk-backed) from resident
RSS. Work: detection without slowing packs (file size vs `model_mib` vs
`/proc` RSS during the serve-shaped measurement), a persisted
disk-resident constant, and the `ssd_mib` tag argument producing e.g.
`[VRAM 12.0GB + 0.6GB SSD 50.0GB]`. Until then a large
file-minus-resident gap shows up only inside the coarse `RAM` residue.

### 3. Measured host RAM (graduating the `RAM` token)

v1 `RAM` is file residue, not a measurement. The serve-shaped
measurement already parses host-side buffer lines — today only as a
spill-*reject* signal (`parse_spill_mib`, `parse_device_buffers`,
`_WEIGHT_CPU_MAP_TOLERANCE_MIB` in `llama_packer/vram.py`). Work:
persist host KV/RS + compute/output staging as host constants alongside
`FitParams` (own `derived:` keys, own shape binding), and switch the GPU
`RAM` term from residue to measured. The formatter needs no change.

### 4. MoE partial offload

*When* to push experts to CPU and *which* layers: per-layer placement
policy, splitting the affine law into GPU-resident vs host-resident
terms, interaction with the `-ngl 999` default and `device:` pins, and
how the chosen split surfaces as `VRAM cap + spill RAM …`. Builds on
§1 (split representation) and §3 (host measurement).

### 5. Unified memory (rule, already decided)

One physical pool (GB10/Apple Silicon/integrated), but the tag keeps
both tokens with driver-managed = `VRAM`, CPU/NPU = `RAM`. No code
change required — recorded here so a future allocator doesn't
"unify" the display into a single number.

## Sequencing

§3 (measurement) unlocks §1/§4 (both need host numbers to decide
splits); §2 is independent and can land anytime. Each step keeps the
v1 invariant: the tag reports the decider's answer verbatim, and any
model the decider can't size keeps no tag instead of a fabricated one.
