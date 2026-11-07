---
name: Authoring profiles.yaml
description: Build or migrate a llama-packer profiles.yaml — discovery roots, tasks, sampling profiles, CPU/GPU placement, matrix sizing, capacity routing, pools, TTL, and validation.
---

# Authoring `profiles.yaml`

`profiles.yaml` is the packer's input. It declares the fleet; the packer
emits `config.yaml` for llama-swap. Start from the annotated
[`profiles.yaml.example`](https://github.com/JeffKwasha/llama-packer/blob/main/profiles.yaml.example) and the contract in
[`SPEC.md`](https://github.com/JeffKwasha/llama-packer/blob/main/SPEC.md); do not invent keys.

## Where the file lives

In an installed deployment it is `<run-dir>/profiles.yaml` (for example
`/opt/llama/profiles.yaml`). The installer creates it from the bundled
`llama_packer/profiles.yaml` **only when absent**; it never overwrites an
existing file. Edit that copy, not the repo's. It is owned by the service
account and group-writable so the operator's agent can update it.

The repo's `docs/profiles-authoring.md` is this guide; the installer also
places it at `<run-dir>/.opencode/skills/profiles-authoring/SKILL.md` so an
agent started in the run directory can load it.

## Workflow

1. Read the example and `SPEC.md`. Copy the example's section order.
2. **Discovery** — `models_dirs`, `dirs` (subdirectory → role), optional
   `hf_home`, `backends` (ordered allow/prefer list). Omit `backends` to use
   all registered engines.
3. **Defaults** — `defaults.cache_type`, `parallel`, sampling values, and an
   optional fleet `spare`.
4. **Tasks** — declare the task ids the fleet uses under `tasks:`; each needs
   `endpoint` and `role`, plus optional `min_context`, `min_parallel`,
   `require`, `keep`, `sacrifice`, `parallel_power`, `ttl`. A sidecar claims
   them with `tasks: [...]`; an undeclared id is not a task.
5. **Profiles** — at least one key under `profiles:`. A profile overlays
   `defaults` and becomes `setParamsByID` aliases. Split
   `(parallel, cache_type, spare)` combinations into separate profiles.
6. **Placement** — `device_roles: [embeddings, rerank, s2t]` (or a
   `{role: cpu}` mapping) moves whole roles to host RAM. Per-model
   `device: cpu` / `device: N` lives in the sidecar, not here.
7. **Matrix sizing** — `matrix:` keeps the context-solve knobs
   (`embed_context`, `rerank_context`, `auto_parallel`, `auto_parallel_max`,
   `parallel_power`, `tools_min_ctx`, `min_chat_ctx`, `coload_min_ctx`,
   `ctx_gain_min`, `estimate_headroom`) and `categories`, which selects the
   model that fills each role (`emb`, `rnk`, `rnk2`, `stt`, `tts`). A category
   may set `selector:` and `ctx:`.
   **Do not author `matrix.sets:` or `matrix.evict_costs:`** — they are
   retired and warn. Routing is capacity-based.
8. **Capacity knobs** — optional `capacity: {pp_tps: 1000, drive_mbps: N}`.
   `pp_tps` is the fleet default prompt-processing rate used to price lost
   prefill in `evict_cost`; a sidecar `pp_tps:` overrides it. Leave
   `drive_mbps` unset to use the detected read speed.
9. **Pools** — optional `pools:` overrides a pool's `vram`, `spare`,
   `reserve_extra`, or explicit `pins`. The packer emits
   `pools.default` (usable VRAM), `pools.ram` (host pool), and one size-1
   pool per role.
10. **Hardware** — optional `hardware.vram`, `baseline_mb`,
    `unified_system_mb`, `gpu_family`, `memory_margin`, `fast_storage`.
11. **TTL** — declare `ttl:` per sidecar or per task. `0` = never unload;
    otherwise the smallest declared lifetime wins. A non-chat CPU helper
    with nothing declared defaults to 3600 s. Chat/device entries inherit
    `globalTTL` (via `--idle-unload`).

## Validate before handing back

```sh
llama-packer --profiles profiles.yaml --dry-run -v > /tmp/config.yaml
llama-swap -config /tmp/config.yaml -validate
```

Check the dry run:

- every emitted entry's `capacities` is ≤ each named pool;
- the sum of a desired co-residency fits `default` and `ram`;
- no `matrix.sets:` warning (remove the key);
- `evict_cost` is high for long-context chat and low for stateless helpers;
- the log names the chosen models and the pool sizes.

## Common mistakes

- **Sets are gone.** `rag`/`voice`/`talk` are no longer authored. Add the
  models to `matrix.categories` (if they are role picks) or rely on the
  sidecars; capacity pools decide co-residency.
- **A category is a model pick, not a workload claim.** `emb`/`rnk` name the
  model that fills the embeddings/rerank role. Workloads are `tasks:`.
- **`capacities` are emitted, not authored.** If you need a model to run
  alone, leave it unmeasured or give it the whole pool; do not hand-write
  capacities in `profiles.yaml`.
- **Role pools enforce exclusivity.** One live serving per role; requesting
  another model of the same role evicts the current one regardless of
  `evict_cost`.
- **CPU is not "never evicted".** It charges `ram` and is evicted when the
  host pool is full; `ttl` is the separate idle timer.
- **Aliases are not model ids.** Category vars and `+ref` names resolve to
  emitted entry ids; use a real model id/selector.

## After a llama-packer update

`extras/update.py` moves binaries; it does not migrate `profiles.yaml`.
A new packer can retire a key, and the packer reports alien keys while it
runs. After updating:

```sh
cd <run-dir>
sudo -u <service-user> llama-packer --profiles profiles.yaml \
  --output /tmp/config.yaml --spare <same-as-install> -v
```

Read the warnings, fix the named keys, then re-pack for real and validate.
Keep a copy of the working `profiles.yaml` before editing; the installer
only protects the first creation, not later edits.

Retired spellings seen so far:

| old | new |
|---|---|
| `matrix.sets:` / `matrix.evict_costs:` | removed; routing is capacity, `capacities`/`evict_cost` are emitted |
| sidecar `role:` / `usage:` | sidecar `tasks: [...]` from `profiles.yaml tasks:` |
| sidecar `backend:` | sidecar `engines: [...]` |
| `pinned` persistence | explicit `ttl:` (sidecar or task) |

Warnings are the migration list. Do not assume an old file is valid because
the binaries updated cleanly.
