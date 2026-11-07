---
name: Authoring profiles.yaml
description: Build or migrate a llama-packer profiles.yaml — discovery roots, tasks, sampling profiles, CPU/GPU placement, context knobs, capacity routing, pools, TTL, and validation.
---

# Authoring `profiles.yaml`

`profiles.yaml` declares the fleet; llama-packer turns it into `config.yaml`
for llama-swap. Start from the annotated
[`profiles.yaml.example`](https://github.com/JeffKwasha/llama-packer/blob/main/profiles.yaml.example)
and the contract in
[`SPEC.md`](https://github.com/JeffKwasha/llama-packer/blob/main/SPEC.md);
do not invent keys.

## Where the file lives

In an installed deployment it is `<run-dir>/profiles.yaml` (for example
`/opt/llama/profiles.yaml`). The installer creates it from the bundled
default **only when absent**; it never overwrites an existing file. Edit
that copy, not the repo's. It is owned by the service account and
group-writable so the operator's agent can update it.

The installer also places this guide at
`<run-dir>/.opencode/skills/profiles-authoring/SKILL.md`; an agent started
in the run directory can load it.

## Decide the budget

```
reserve        = 1024 + max(1024, baseline_mb)          MiB
pools.default  = VRAM_total - reserve - spare           MiB
```

`spare` is headroom **on top of** the reserve, not the whole headroom. The
reserve covers the driver, compositor and scanout; `spare` covers the
tenants that come and go. Every serving is then sized up to the pool:

```
sum(capacities.default of co-resident servings) ~= pools.default
```

A full pool is the design — a small gap between `pools.default` and the
largest `capacities.default` is not headroom.

- Use `defaults.spare` for ordinary headroom: the desktop, a browser, or
  other tenants that appear later. It is subtracted after the fixed reserve.
- `hardware.baseline_mb` is an advanced override for a **known fixed**
  pre-existing VRAM load. It only matters above the fixed 1 GiB video
  reserve; for normal headroom prefer `spare`.
- Choose `spare` once the tenant envelope is known. With `used(t)` sampled
  over days:

  ```
  base    = floor(used(t))                       driver + compositor
  spare  >= max(tenant_vram(t)) - base           audio, image, browsers
  ```

  Loaded models are excluded: the pool already accounts for them.
  `python -m llama_packer.vram_log --report` gives the floor and the
  per-tenant medians/max. Do not use `--once`: a single sample includes
  whatever models are loaded at that moment and cannot separate the base
  from the tenants. `llama-packer --probe-memory` measures a model family's
  serve-shaped VRAM law — it is not a headroom measurement.
- Check `capacities.default <= pools.default` for every entry. The pool
  already has the headroom deducted, so do not shrink contexts for slack.

## Workflow

1. Read the example and `SPEC.md`. Copy the example's section order.
2. **Discovery** — `models_dirs`, `dirs` (subdirectory → role), optional
   `hf_home`, `backends` (ordered allow/prefer list). Omit `backends` to use
   all registered engines.
3. **Defaults** — `defaults.cache_type`, sampling values, and `spare`.
   Leave `parallel` unset unless you intend to fix the slot count: any
   `parallel:` in `defaults`/a profile pins every model in that group and
   disables auto-parallel. Unset, the packer maximizes

   ```
   score = (ctx / min_context) ** 0.5 * slots ** parallel_power
   ```

   subject to the task floors and the pool.
4. **Tasks** — declare the task ids the fleet uses under `tasks:`; each needs
   `endpoint` and `role`, plus optional `min_context`, `min_parallel`,
   `require`, `keep`, `sacrifice`, `parallel_power`, `ttl`. A sidecar claims
   them with `tasks: [...]`; an undeclared id is not a task. Both floors are
   hard inequalities:

   ```
   ctx   >= min_context
   slots >= min_parallel
   ```

   A serving that cannot meet both produces no entry for that task. Leave a key
   undefined for no floor. `parallel_power` weights slots against context in
   the score above; it never lowers a floor.
5. **Profiles** — at least one key under `profiles:`. A profile overlays
   `defaults` and becomes per-profile sampling overrides (`setParamsByID`).
   Split `(parallel, cache_type, spare)` combinations into separate profiles.
6. **Placement** — `device_roles: [embeddings, rerank, s2t]` (or a
   `{role: cpu}` mapping) moves whole roles to host RAM. Per-model
   `device: cpu` / `device: N` lives in the sidecar, not here. A CPU serving
   charges the `ram` pool (`capacities.ram`) and is evicted under RAM
   pressure.
7. **Context knobs** — `matrix:` holds the context-solve knobs
   (`embed_context`, `rerank_context`, `auto_parallel`, `auto_parallel_max`,
   `parallel_power`, `tools_min_ctx`, `min_chat_ctx`, `coload_min_ctx`,
   `ctx_gain_min`, `estimate_headroom`) and `categories`, which selects the
   model that fills each role (`emb`, `rnk`, `rnk2`, `stt`, `tts`). A category
   may set `selector:` and `ctx:`.
   **Do not author `matrix.sets:` or `matrix.evict_costs:`** — they are
   retired and warn. Routing is capacity-based.
8. **Capacity knobs** — optional `capacity: {pp_tps: 1000, drive_mbps: N}`.
   llama-swap evicts the lowest `evict_cost` first:

   ```
   evict_cost = weights_MB / drive_mbps + ctx * slots / pp_tps   (chat only)
   ```

   `pp_tps` is prompt tokens/s; estimate it from hardware TOPS × active
   parameter count, or measure it with `extras/bench.py` / `llama-bench`. A
   sidecar `pp_tps:` overrides the fleet value. `drive_mbps` defaults to the
   detected read speed.
9. **Pools** — optional `pools:` overrides a pool's `vram`, `spare`,
   `reserve_extra`, or explicit `pins`. The output contains
   `pools.default` (usable VRAM), `pools.ram` (host pool), and one size-1
   pool per role.
10. **Hardware** — optional `hardware.vram`, `baseline_mb`,
    `unified_system_mb`, `gpu_family`, `memory_margin`, `fast_storage`.
11. **TTL** — declare `ttl:` per sidecar or per task:

    ```
    ttl = 0                     never unload
    ttl = min(declared)         otherwise
    ttl = 3600 s                CPU-resident, nothing declared
    ```

    A device entry carries none and inherits `globalTTL`.

## Validate before handing back

Run the packer as `docs/install.md` describes; `llama-packer --help` has the
current flags. A dry run writes YAML to stdout, so redirect it, then:

```sh
llama-swap -config /tmp/config.yaml -validate
```

`-validate` only checks the generated config. Read the packer's `E |`/`W |`
log lines — that is where dropped models and retired keys appear.

Check:

- `capacities.default <= pools.default` for every entry;
- no `matrix.sets:`/`matrix.evict_costs:` warning;
- `ctx_size >= min_context` and `parallel >= min_parallel` per task (a floor
  that cannot be met produces no entry);
- `ttl` is what you intend for keep-warm roles;
- `evict_cost` is high for long-context chat and low for stateless helpers;
- no model was dropped by an `engines:`/`backends:` mismatch;
- the log names the chosen models and the pool sizes.

## Common mistakes

- **Sets are gone.** `rag`/`voice`/`talk` are no longer authored. Add the
  models to `matrix.categories` (if they are role picks) or rely on the
  sidecars; capacity pools decide co-residency.
- **A category is a model pick, not a workload claim.** `emb`/`rnk` name the
  model that fills the embeddings/rerank role. Workloads are `tasks:`.
- **`capacities` are derived, not authored.** They come from the measured
  serving; if you need a model to run alone, leave it unmeasured or give it
  the whole pool. Do not hand-write capacities in `profiles.yaml`.
- **Role pools enforce exclusivity.** `pools.<role> = 1`, so one serving per
  role: requesting another model of the same role evicts the current one
  regardless of `evict_cost`.
- **CPU is not "never evicted".** It charges `ram` and is evicted when the
  host pool is full; `ttl` is the separate idle timer.
- **Aliases are not model ids.** Category vars and `+ref` names resolve to
  entry ids in the config; use a real model id/selector.

## After a llama-packer update

`extras/update.py` moves binaries; it does not migrate `profiles.yaml`.
A new packer can retire a key, and the packer reports alien keys while it
runs. Run the packer as `docs/install.md` describes, read the warnings, fix
the named keys, then re-pack for real and validate. Keep a copy of the
working `profiles.yaml` before editing; the installer only protects the
first creation, not later edits.

Retired spellings:

| old | new |
|---|---|
| `matrix.sets:` / `matrix.evict_costs:` | removed; routing is capacity, `capacities`/`evict_cost` are derived |
| sidecar `role:` / `usage:` | sidecar `tasks: [...]` from `profiles.yaml tasks:` |
| sidecar `backend:` | sidecar `engines: [...]` |
| `pinned` persistence | explicit `ttl:` (sidecar or task) |

Warnings are the migration list. Do not assume an old file is valid because
the binaries updated cleanly.
