---
name: Authoring profiles.yaml
description: Build or migrate a llama-packer profiles.yaml — where models live, what work they do, how much memory they get, how they are placed, and how idle models are unloaded.
---

# Authoring `profiles.yaml`

`profiles.yaml` tells llama-packer which models exist and how they should be
served. llama-packer turns it into `config.yaml`, which llama-swap runs. Start
from the annotated
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

## The words used here

- **Serving** — one loaded model answering requests.
- **Context** — tokens one request may use.
- **Slots** — requests one serving handles at once (`parallel`).
- **Role** — what a model is for: chat, embeddings, rerank, image,
  speech-to-text (`s2t`), text-to-speech (`t2s`).
- **Task** — a named kind of work (for example `coding`), with the smallest
  context and slot count it needs.
- **Pool** — a memory allowance. Models sharing the GPU must fit inside it;
  the router unloads one to make room.

## How much memory models get

Two deductions come off the top of the GPU: a **reserve** for the OS and
display, and **spare** for other programs that come and go.

```
reserve       = 1 GiB + max(1 GiB, fixed baseline)
model memory  = total GPU memory - reserve - spare
```

The reserve is 1 GiB for the OS plus at least 1 GiB for the driver,
compositor and scanout. If something is *always* on the GPU (a VM, a game
left running), set `hardware.baseline_mb`; for ordinary headroom, raise
`spare`.

A 32 GiB card with 1 GiB spare and no fixed baseline:

```
reserve       = 1 + 1      = 2 GiB
model memory  = 32 - 2 - 1 = 29 GiB
```

`spare` is on top of the reserve. That memory is divided among the models
kept loaded, and is meant to be used up: a nearly full pool is normal, not
a sign of no headroom.

Measure `spare` over days, not once:

```
fixed base  = lowest GPU usage over several days   (driver + desktop)
spare      >= peak usage of other programs - fixed base
```

`python -m llama_packer.vram_log --report` gives both. `--once` includes
whatever models happen to be loaded, so it cannot tell the base from
transient programs. `llama-packer --probe-memory` measures a model family's
memory at a given context and slot count; it is not a headroom measurement.

Check: no model gets more memory than that. It already excludes the
headroom, so do not shrink contexts for more.

## How context and slots are chosen

Unless you pin `parallel`, the packer picks the best-scoring pair:

```
score = sqrt(context / required context) * slots ^ power
```

The score rewards context and slots. `parallel_power` (fleet-wide under
`matrix:`, or per task) sets the trade: low favours context, high favours
slots. "Required context" is the task's minimum (`min_context`).

Setting `defaults.parallel` skips this: every model in that group keeps that
slot count.

## Workflow

1. Read the example and `SPEC.md`. Copy the example's section order.
2. **Where models are** — `models_dirs` for the roots; `dirs` maps a
   subdirectory to a role; `hf_home` points at the Hugging Face cache;
   `backends` lists the engines you allow (omit it to allow all installed
   engines).
3. **Defaults** — sampling values, `cache_type`, and `spare`. Leave
   `parallel` unset unless you mean to fix the slot count.
4. **Tasks** — each task names an endpoint, a role, and its minimums:

   ```
   context >= min_context
   slots   >= min_parallel
   ```

   Both are hard: a model that misses either is not offered for that task.
   Omit a minimum for no minimum. One serving can cover several tasks.
5. **Sampling profiles** — a profile overlays the defaults and becomes a
   named set of sampling values that llama-swap applies per request. Split
   `(parallel, cache_type, spare)` combinations into separate profiles.
6. **Placement** — `device_roles: [embeddings, rerank, s2t]` (or a
   `{role: cpu}` mapping) runs whole roles from system RAM. Per-model
   `device: cpu` / `device: N` lives in the model's sidecar, not here. A
   model in RAM uses the RAM pool and can be unloaded when that pool is
   full.
7. **Context limits and role picks** — `matrix:` holds the limits that shape
   context sizing and the choice of which model fills each role.
   `embed_context` and `rerank_context` cap how much context the retrieval
   helpers get (4096 by default), so they take little memory. `categories`
   picks the model for a role (`emb`, `rnk`, `rnk2`, `stt`, `tts`); a pick
   may name a `selector:` and give it its own `ctx:`.
   **Do not write `matrix.sets:` or `matrix.evict_costs:`** — they are
   retired and produce a warning. Memory decides what can be loaded
   together, not named sets.
8. **Eviction** — when a pool is full, llama-swap unloads the serving with
   the lowest eviction cost:

   ```
   eviction cost = weight size / disk read speed
                 + context * slots / prompt speed        (chat only)
   ```

   First term: time to read the model file from disk. Second: chat prefill
   lost when its context is discarded. Units: GiB ÷ GiB/s and tokens ÷
   tokens/s. `capacity.pp_tps` is prompt speed (tokens/s);
   `capacity.drive_mbps` is disk speed (detected if omitted). A sidecar
   `pp_tps` overrides the fleet value. Estimate prompt speed from hardware
   TOPS × active parameter count, or measure it with `extras/bench.py` /
   `llama-bench`.

   A 20 GiB model on a 1 GiB/s disk costs about 20 s to reload; its 32k
   context at 2 slots and 1000 prompt tokens/s adds about 66 s of lost
   prefill.
9. **Pools** — the default config has one GPU pool, one RAM pool, and one
   1-model pool per role. `pools:` changes those limits or reserves memory
   for something the packer does not model.
10. **Hardware** — physical overrides: `hardware.vram`, `baseline_mb`,
    `unified_system_mb`, `gpu_family`, `memory_margin`, `fast_storage`.
11. **Idle unloading (TTL)** — declare `ttl:` per task or per model:

    ```
    0                     never unload
    shortest declared     otherwise
    1 hour                CPU-resident model, nothing declared
    ```

    A GPU model with no TTL of its own follows the global idle setting.

## Validate before handing back

Run the packer as `docs/install.md` describes; `llama-packer --help` has the
current flags. A dry run writes YAML to stdout, so redirect it, then:

```sh
llama-swap -config /tmp/config.yaml -validate
```

`-validate` only checks the generated config. Read the packer's `E |`/`W |`
log lines — that is where dropped models and retired keys appear.

Check:

- no model gets more memory than the pool allows;
- no warning about `matrix.sets:` or `matrix.evict_costs:`;
- every model offered for a task meets its minimum context and slot count;
- idle timeouts are what you intend for keep-warm roles;
- eviction cost is high for long-context chat and low for small helpers;
- no model was dropped because its engine is not enabled;
- the log names the chosen models and the pool sizes.

## Common mistakes

- **Named sets are gone.** `rag`/`voice`/`talk` are no longer authored. Add
  the models to `matrix.categories` if they are role picks, or rely on the
  sidecars; memory decides what can be loaded together.
- **A category is a model choice, not a kind of work.** `emb`/`rnk` name the
  model that fills the embeddings/rerank role. Kinds of work are `tasks:`.
- **Memory is measured, not written.** It describes how much memory the
  model actually takes. If you need a model to run alone, leave it
  unmeasured or give it the whole pool; do not hand-write its allocation.
- **One model per role.** The router unloads the current model of a role
  when you request another model of that role, whatever the eviction cost.
- **RAM models can be unloaded too.** They use the RAM pool; TTL is a
  separate idle timer.
- **A name is not a model id.** Category names and `+ref` names resolve to
  entry ids in the config; use a real model id or selector.

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
| `matrix.sets:` / `matrix.evict_costs:` | removed; memory decides co-residency, allocations and eviction cost are measured |
| sidecar `role:` / `usage:` | sidecar `tasks: [...]`, built from `profiles.yaml tasks:` |
| sidecar `backend:` | sidecar `engines: [...]` |
| `pinned` persistence | explicit `ttl:` (sidecar or task) |

Warnings are the migration list. Do not assume an old file is valid because
the binaries updated cleanly.
