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

- **Serving** — one model loaded into memory and answering requests.
- **Context** — how many tokens one request may use.
- **Slots** — how many requests a serving handles at the same time
  (`parallel`).
- **Role** — what a model is for: chat, embeddings, rerank, image,
  speech-to-text (`s2t`), text-to-speech (`t2s`).
- **Task** — a named kind of work at an endpoint (for example `coding` or
  `chat`), with the least context and the fewest slots it needs.
- **Pool** — a memory budget. The router keeps the servings in a pool inside
  that budget and unloads a serving when the next one would overflow it.

## How much memory models get

Two things come off the top of the GPU:

- a **reserve** for the operating system and the display;
- **spare**, extra memory for other programs that come and go.

```
reserve       = 1 GiB + max(1 GiB, fixed baseline)
model memory  = total GPU memory - reserve - spare
```

The reserve is 1 GiB for the OS plus at least 1 GiB for the driver,
compositor and scanout. If something is *always* on the GPU (a VM, a game
left running), declare it as `hardware.baseline_mb`; for ordinary headroom,
raise `spare`.

`spare` comes on top of the reserve, so it is not the only headroom. The
packer gives each serving it keeps loaded an allocation of that model
memory, and the allocations are meant to fill it. An allocation that nearly
fills the pool is normal — a nearly full pool does not mean "no headroom".

For example, a 32 GiB card, a declared 1 GiB fixed baseline, and 1 GiB of
spare:

```
reserve       = 1 + max(1, 1) = 2 GiB
model memory  = 32 - 2 - 1    = 29 GiB
```

Measure `spare`; do not guess it from one reading:

```
fixed base  = lowest GPU usage over several days   (driver + desktop)
spare      >= peak usage of other programs - fixed base
```

Run `python -m llama_packer.vram_log` for a few days and read
`--report`: the floor is the fixed base, and the per-program peaks above it
are what `spare` has to cover. A single `--once` sample is misleading
because it includes whatever models happen to be loaded at that moment.

`llama-packer --probe-memory` measures how much memory one model family
takes at a given context and slot count. It is not a headroom measurement.

Check the result: every serving's allocation in the generated config must be
no larger than the model memory. The headroom is already inside
`reserve + spare`, so do not shrink contexts to create more.

## How context and slots are chosen

Unless you pin `parallel`, the packer tries combinations of context size and
slot count and keeps the one with the best score:

```
score = sqrt(context / required context) * slots ^ power
```

Bigger context scores higher, and more slots score higher. `parallel_power`
(fleet-wide under `matrix:`, or per task) says how much one extra slot is
worth: a low power favours context, a high power favours slots. "Required
context" is the smallest context a task will accept (`min_context`).

This is why `defaults.parallel` is a big hammer. Set it and every model in
that group keeps that slot count; the packer stops choosing.

## Workflow

1. Read the example and `SPEC.md`. Copy the example's section order.
2. **Where models are** — `models_dirs` for the roots; `dirs` maps a
   subdirectory to a role; `hf_home` points at the Hugging Face cache;
   `backends` is the ordered list of engines you allow. Omit `backends` to
   allow every installed engine.
3. **Defaults** — sampling values, `cache_type`, and `spare`. Leave
   `parallel` unset unless you mean to fix the slot count.
4. **Tasks** — declare the kinds of work the fleet serves. Each task names
   an endpoint and a role, plus whatever minimums you want:

   ```
   context >= min_context
   slots   >= min_parallel
   ```

   Both are hard minimums: a model that cannot meet both is not offered for
   that task. Leave a minimum out to mean "no minimum". One serving can
   satisfy several tasks; it does not need one entry per task.
5. **Sampling profiles** — a profile overlays the defaults and becomes a
   named set of sampling values that llama-swap applies per request. Split
   `(parallel, cache_type, spare)` combinations into separate profiles.
6. **Placement** — `device_roles: [embeddings, rerank, s2t]` (or a
   `{role: cpu}` mapping) runs whole roles from system RAM. Per-model
   `device: cpu` / `device: N` lives in the model's sidecar, not here. A
   model in RAM charges the RAM pool and can be unloaded when that pool is
   full.
7. **Context limits and role picks** — `matrix:` holds the limits that shape
   context sizing and the choice of which model fills each role. The
   `embed_context` and `rerank_context` limits cap how much context the
   retrieval helpers get (4096 by default), so they take little memory.
   `categories` picks the model for a role (`emb`, `rnk`, `rnk2`, `stt`,
   `tts`); a pick may name a `selector:` and give it its own `ctx:`.
   **Do not write `matrix.sets:` or `matrix.evict_costs:`** — they are
   retired and produce a warning. What can be loaded together is decided by
   memory, not by named sets.
8. **Eviction** — when a pool is full, llama-swap unloads the serving whose
   eviction cost is lowest. Reloading costs two things:

   ```
   eviction cost = weight size / disk read speed
                 + context * slots / prompt speed        (chat only)
   ```

   The first term is the time to read the model file back from disk. The
   second is the chat work lost when the loaded conversation is thrown
   away. Both are in seconds: GiB ÷ GiB/s, and tokens ÷ tokens/s.
   `capacity.pp_tps` is the fleet's prompt speed in tokens per second;
   `capacity.drive_mbps` is the disk read speed (detected if you leave it
   out). A model that is much faster or slower can give its own `pp_tps` in
   its sidecar. Estimate prompt speed from hardware TOPS × active parameter
   count, or measure it with `extras/bench.py` / `llama-bench`.

   For example, a 20 GiB model on a 1 GiB/s disk costs about 20 s to
   reload; its 32k context at 2 slots and 1000 prompt tokens/s adds about
   66 s of lost prefill.
9. **Pools** — the config starts with one GPU pool, one system-RAM pool, and
   one pool per role limited to a single serving. `pools:` lets you change
   those limits or reserve extra memory for something the packer does not
   model.
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

- every serving's allocation is no larger than the model memory;
- no warning about `matrix.sets:` or `matrix.evict_costs:`;
- every serving offered for a task meets that task's minimum context and
  slots (a minimum that cannot be met means no entry);
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
- **Allocations are measured, not written.** They describe how much memory
  the model actually takes. If you need a model to run alone, leave it
  unmeasured or give it the whole pool; do not hand-write its allocation.
- **One serving per role.** The router unloads the current model of a role
  when you request another model of that role, whatever the eviction cost.
- **A model in RAM is not "never unloaded".** It uses the RAM pool and can
  be unloaded when that pool is full; TTL is a separate idle timer.
- **Aliases are not model ids.** Category names and `+ref` names resolve to
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
