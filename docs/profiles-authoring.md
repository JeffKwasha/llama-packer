---
name: Authoring profiles.yaml
description: Build or migrate a llama-packer profiles.yaml — where models are, what work they do, how much memory they get, how they are placed, and how idle models are unloaded. Use when writing, editing, or debugging a profiles.yaml for llama-packer.
---

# Authoring `profiles.yaml`

`profiles.yaml` tells llama-packer which models exist and how they should be
served. llama-packer turns it into `config.yaml`, which llama-swap runs.

This guide is the working reference: the file's syntax, what each key means,
and how the keys interact. For the contract behind any key, follow the links
into [`SPEC.md`](../SPEC.md); for a commented starter, copy
[`profiles.yaml.example`](../profiles.yaml.example). Do not invent keys —
unknown top-level keys warn and are ignored.

## Where the file lives

In an installed deployment it is `<run-dir>/profiles.yaml` (for example
`/opt/llama/profiles.yaml`). The installer creates it from the bundled
default **only when absent**; it never overwrites an existing file. Edit that
copy, not the repo's. It is owned by the service account and group-writable
so the operator's agent can update it.

The installer also places this guide at
`<run-dir>/.opencode/skills/profiles-authoring/SKILL.md`; an agent started in
the run directory can load it.

## The words used here

- **Serving** — one loaded model answering requests.
- **Context** — tokens one request may use.
- **Slots** — requests one serving handles at once (`parallel`).
- **Role** — an operator-defined label. Models are assigned to a role by
  the tree it owns; models in the same role evict each other on request.
  The packaged defaults are data; rename or replace them.
- **Task** — a named kind of work bound to an API endpoint.
- **Pool** — a memory allowance with a measured reserve and an operator
  spare. Models sharing it must fit; the router unloads one to make room.

## Minimal file

Only `profiles:` is required. A working start:

```yaml
profiles:
  default:

models_dirs:
  - models
```

That relies on built-in defaults for everything else. The rest of this guide
adds keys as they are needed.

## A fuller example

```yaml
defaults:
  cache_type: q8_0
  spare: 1G                 # headroom for the desktop and other tenants

tasks:
  chat:                     # a named kind of work
    endpoint: openai.chat.completions
    min_context: 64k
    min_parallel: 2
  coding:
    endpoint: openai.chat.completions
    min_context: 130k
    min_parallel: 1
    sacrifice: [vision]
    keep: [draft]
  embeddings:
    endpoint: openai.embeddings
    ttl: 0                  # keep it loaded

models_dirs:
  - models
  - /mnt/ai/models

backends:
  - llama-server
  - sd-server

profiles:
  default:
  coder:
    temperature: "base * 0.7"
    parallel: 2             # a pin: opt this variant out of auto-parallel

matrix:
  parallel_power: 0.75      # low favours context, high favours slots

capacity:
  pp_tps: 1000              # prompt speed, for eviction cost

pools:
  default:
    spare: 2G               # added to the measured reserve

hardware:
  vram: 32G
```

## Key reference

Top-level keys, all optional except `profiles:`. Each has a deeper home in
`SPEC.md`; the purpose column is what an author needs to decide whether to
set it.

| Key | Purpose | Deeper |
|---|---|---|
| `defaults` | Baseline sampling values plus `cache_type`, `parallel`, `spare` | [SPEC §profiles.yaml](../SPEC.md#profilesyaml) |
| `profiles` | Named sampling overlays; **required** | [SPEC §Sampling Modes](../SPEC.md#sampling-modes) |
| `tasks` | Named kinds of work and their floors | [SPEC §profiles.yaml](../SPEC.md#profilesyaml) |
| `models_dirs` | Model root directories | [SPEC §Model Discovery](../SPEC.md#model-discovery-and-stub-sidecars) |
| `hf_home` | Hugging Face cache root | [SPEC §Path Macros](../SPEC.md#path-macros-macros-block-and-configenv) |
| `backends` | Ordered enable/prefer list | [SPEC §Backend Selection](../SPEC.md#backend-selection) |
| `roles` | Operator-defined purposes, owned trees, placement, allocation | [roles-tasks-routing](roles-tasks-routing.md) |
| `matrix` | Sizing knobs and matrix `sets:` | [SPEC §VRAM Allocation](../SPEC.md#vram-allocation) |
| `pools` | Per-pool spare and nominal size | [SPEC §Capacity routing](../SPEC.md#capacity-routing) |
| `capacity` | Eviction-cost inputs (`pp_tps`, `drive_mbps`) | [SPEC §Capacity routing](../SPEC.md#capacity-routing) |
| `hardware` | Physical overrides | [SPEC §Hardware Detection](../SPEC.md#hardware-detection) |
| `overrides` | Pattern-scoped serving rules | [SPEC §Override Rules](../SPEC.md#override-rules) |
| `llama_server`, `vllm`, `sd`, `whisper`, `audio_cpp` | Per-backend fleet-wide args and resources | [SPEC §Global backend args](../SPEC.md#global-backend-args) |
| `macros` | Named flag chunks for `cmd` | [SPEC §Path Macros](../SPEC.md#path-macros-macros-block-and-configenv) |
| `shed` | Default capability-shedding order | [SPEC §profiles.yaml](../SPEC.md#profilesyaml) |
| `ttl` | Top-level idle lifetime, default 3600 | [SPEC §TTL](../SPEC.md#ttl) |
| `min_context` | Default task floor | [SPEC §Tasks](../SPEC.md#tasks-tasks) |

### `defaults`

Values merged under every profile. It also carries planning keys:
`cache_type` (KV precision), `parallel` (a global slot pin — **leave it
unset** to let the packer choose), and `spare` (VRAM headroom).

```yaml
defaults:
  cache_type: q8_0
  spare: 1G
  temperature: 1.0
  top_p: 0.95
```

### `profiles`

A profile overlays `defaults` and becomes llama-swap's per-request sampling
overrides. It may set any sampling key (`temperature`, `top_p`, `top_k`,
`min_p`, `pres_pen`, `repeat_penalty`, `freq_pen`) with a literal or a
`"base * N"` expression, plus the planning keys `cache_type`, `parallel`,
and `spare`. Differing `(parallel, cache_type, spare, batch, ubatch)`
combos split a model into separate config entries; profile names within a
group become request-time overrides. `description` is documentation only.

```yaml
profiles:
  default:
  coder:
    temperature: "base * 0.7"
  server:
    parallel: 2
    spare: 2G
```

### `tasks`

Each task names the endpoint it is served at and its demand. It does not
name a role: the model's role is the model's.

```yaml
tasks:
  coding:
    endpoint: openai.chat.completions
    device: 0                # cpu | GPU index (same as a sidecar)
    min_context: 130k
    max_context: 131072      # optional ceiling
    min_parallel: 1
```

- `device` — `auto`, `cpu`, or a GPU index (SPEC §Device). A task's value is
  a default; a sidecar `device:` wins.
- `min_context`, `min_parallel` — hard floors.
- `max_context` — ceiling on the served context.
- `require` / `keep` — capabilities that must survive.
- `sacrifice` — what to give up first; a preference, not a cost order.
- `parallel_power` — per-task override of `matrix.parallel_power`.
- `ttl` — idle unload for models serving this task.

Tasks that share an endpoint (chat, coding, qa, aux, describe) are aliases
of one serving. Tasks on different endpoints cannot share a serving — the
engine flag differs (`--embeddings` vs chat) — so a model's tasks must name
one endpoint.
`role:` is retired inside a task; the model declares it.

### Model sources

```yaml
models_dirs:
  - models
  - /mnt/ai/models

hf_home: /mnt/data/huggingface

backends:                   # ordered; absent = every registered backend
  - llama-server
  - vllm-docker
```

A role owns a directory with `dir:` (see `roles` below). Every model under
that tree belongs to the role; the directory name itself carries no meaning.
There is no built-in directory table and no default role. `dirs:` is retired.
A model with no owning role and no sidecar `role:` is an error: the run quits.

### `roles`

A role is ownership and allocation. The code ships no role names; the
example below defines the usual set. A model belongs to the role that owns
its directory, or to the role named by its sidecar `role:`.

```yaml
roles:
  chat:
    dir: qwen3             # the tree this role owns
    share: fill            # take what the weighted roles leave
  embeddings:
    dir: embed
    share: 5%              # request weight
    device: cpu            # every model of this role runs from system RAM
```

- `dir` — the models-root-relative directory or directories this role owns;
  `.` is the root. A tree belongs to exactly one role; overlapping trees are
  an error. Optional: a role without one is reachable only by a sidecar
  `role:`.
- A role has no behaviour. The models under it declare `tasks:`, and each
  task names the endpoint it is served at. A role is ownership and
  allocation, nothing else.
- `share` — the role's request in each set: a number (`40`, `40%`, and values
  above 100 are relative weights), `fill` (the remainder, or 0 when none), `min` (0, accept nothing), or `0` (0, accept what remains). Weights are normalised per set; they do not
  sum to a cap.
- `device` — `auto`, `cpu`, or a GPU index, the same vocabulary as a sidecar.
  Unset leaves placement to the sidecar/task. See
  [SPEC §Device](../SPEC.md#device-device) for the precedence and the
  `ram`/`-ngl 0` effect.

`share` is applied per set: the role's budget is the minimum offer across
the sets it belongs to. See [allocation.md](allocation.md).

There is no `min`, `preferred`, `cap`, `weight`, `exclusive`, `persistent`,
`fill`, or per-role count key. A floor is derived per model from its task's
`min_context` and `min_parallel`. A single-leaf set is how a role runs alone.

A task does not name a role. The model's role is the model's.

### `matrix`

Sizing knobs, and matrix sets.

```yaml
matrix:
  tools_min_ctx: 131072
  auto_parallel: false
  auto_parallel_max: 8
  parallel_power: 0.75
  sets:                       # presence switches the config to matrix
    rag: "chat & embeddings & rerank"
    voice: "chat & t2s"
```

- `tools_min_ctx` — below this chat context the entry loses the `tools`
  capability.
- `auto_parallel`, `auto_parallel_max`, `parallel_power` — how leftover
  memory becomes slots vs context.
- `sets` — llama-swap expressions (`&`, `|`, `()`, `+set`) over role labels
  and model ids. A single leaf is a valid set and confines that role to
  running alone. A model in no set runs alone.
- `embed_context` / `rerank_context` are retired; use a task `max_context`.
- The old set-category selector is retired; name the model id directly in
  a set.

`sets:` is not a separate router switch. Writing it makes the config matrix;
leaving it out makes it capacity. Each set is its own budget, sized by the
auction in [allocation.md](allocation.md). Capacity has no set generation
yet; it prices each role directly as `floor + share × (available − floor)`. Matrix never
preloads: it evicts to the cheapest set containing the requested model and
starts only that model.

### `pools`

Capacity mode emits one GPU pool, one RAM pool, and one size-1 pool per
role. Matrix mode emits sets instead. `pools:` adjusts a pool or reserves memory
for something llama-packer does not model. Keys per pool id (`default`,
`ram`, `gpu0`, a role name, …):

```yaml
pools:
  default:
    spare: 2G               # added to the measured reserve
  ram:
    spare: 2G               # keep 2G of host RAM free
```

- `spare:` — added to the measured reserve for that pool. It overrides the
  fleet `defaults.spare` for that pool and reduces the emitted pool.
- `vram:` — the nominal size used to resolve bare-number memory values. It
  does not change the emitted `default` pool.
- `reserve_extra:` and `pins:` are retired and error.

The pool the router sees is `--vram` (or `hardware.vram`) minus the measured
reserve and that pool's spare.

### `capacity`

Inputs to the eviction cost; not sizing knobs.

```yaml
capacity:
  pp_tps: 1000
  drive_mbps: 1000          # detected if omitted
```

A sidecar `pp_tps:` overrides the fleet value.

### `hardware`

Physical overrides for the pool and host budget.

```yaml
hardware:
  vram: 48G
  baseline_mb: 512m
  unified_system_mb: 8192
  gpu_family: rdna4
  memory_margin: 0.04
  fast_storage: /mnt/@/ssd_ai
  host_pool: auto
  system_ram: 64G
  host_pressure: sysctl
  cgroup_dir: /sys/fs/cgroup/system.slice/llama-swap.service
```

`vram`, `baseline_mb`, `unified_system_mb`, `gpu_family`, and
`memory_margin` are the commonly used ones; `host_*`, `system_ram`, and
`cgroup_dir` tune the RAM pool for `device: cpu` models.

### `overrides`

Pattern-scoped serving choices. `when:` is required (`when: true` matches
all); it maps a frontmatter field to a regex. Later matching rules win per
key.

```yaml
overrides:
  - when: {base_model: qwen3}
    chat_template: qwen_chat_template.jinja
    reasoning-format: deepseek
  - when: {stem: 'Qwen3-32B.*safetensors'}
    backend: vllm-docker
    hf_repo: Qwen/Qwen3-32B
```

### `shed`

Default order in which capabilities are given up when a task does not name
its own.

```yaml
shed: [vision, draft]
```

Unlisted capabilities go vision before draft, because an MTP draft roughly
doubles generation speed. Shedding a draft is opt-in per model
(`allow_mtp_shed:`).

### Backend resources and macros

`llama_server`, `vllm`, `sd`, `whisper`, and `audio_cpp` hold fleet-wide
`args:` and backend resources (`bin`, `image`, `docker_args`, …).
`macros:` defines named flag chunks usable as `${NAME}` in `cmd`.

```yaml
llama_server:
  args: "--flash-attn on"
  ubatch: 512

macros:
  COMMON: "--parallel 1 --n-gpu-layers 999"
```

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
kept loaded, and is meant to be used up: a nearly full pool is normal, not a
sign of no headroom.

Measure `spare` over days, not once:

```
fixed base  = lowest GPU usage over several days   (driver + desktop)
spare      >= peak usage of other programs - fixed base
```

`python -m llama_packer.vram_log --report` gives both. `--once` includes
whatever models happen to be loaded, so it cannot tell the base from
transient programs. `llama-packer --probe-memory` measures a model family's
memory at a given context and slot count; it is not a headroom measurement.

Allocations are **derived, not authored**: do not hand-write a model's
memory. The packer measures it; `spare`, `baseline_mb`, and `pools:` are
the knobs that change what is left over.

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

## Placement, eviction, and idle unloading

- **One model per role.** A size-1 role pool means requesting another model
  of the same role unloads the current one, whatever its eviction cost.
  Every role behaves this way.
- **RAM models can be unloaded too.** A `device: cpu` model charges the
  `ram` pool and can be evicted under RAM pressure. TTL is a separate idle
  timer.
- **Eviction** — when a pool is full, llama-swap unloads the serving with
  the lowest eviction cost:

  ```
  eviction cost = weight size / disk read speed
                + context * slots / prompt speed        (KV-cached endpoints)
  ```

  First term: time to read the model file from disk. Second: chat prefill
  lost when its context is discarded. Units: GiB ÷ GiB/s and tokens ÷
  tokens/s. A 20 GiB model on a 1 GiB/s disk costs about 20 s to reload; its
  32k context at 2 slots and 1000 prompt tokens/s adds about 66 s of lost
  prefill.
- **TTL** — declare `ttl:` at the top level, on a role, on a task, or on a
  model:

  ```
  0                     never unload
  most specific value   otherwise
  3600 seconds          top-level default
  ```

  A task or sidecar value overrides the role, and the role overrides the
  top-level `ttl`. TTL never reserves memory.

## Validate before handing back

Run the packer as [`docs/install.md`](install.md) describes; `llama-packer
--help` has the current flags. Validate in three stages:

```sh
# 1. Check the file itself; no models, no server, no writes.
llama-packer --check --profiles profiles.yaml

# 2. Pack a dry run; read every E | and W | line.
llama-packer --dry-run > /tmp/config.yaml

# 3. Ask llama-swap whether the generated config is well formed.
llama-swap -config /tmp/config.yaml -validate
```

`--check` loads the profile and names every problem it knows: unknown
top-level and nested keys (with a did-you-mean hint), missing `profiles:` or
task `endpoint:`, retired `matrix.evict_costs:`/`routing:` and role
keys, a `matrix.sets:` that is not name → expression, wrong types, non-role
retired `device_roles:`, and rules without `when:`. It exits 1 on the first run that finds anything. It does not
discover models, measure, or start a server.

`--dry-run` writes nothing — no config, no stubs, no measurements — and
checks the plan end to end. `-validate` checks the generated config, not
your intent.

Then check the plan itself:

- no model gets more memory than the pool allows;
- no warning about a set expression or `matrix.evict_costs:`;
- every model offered for a task meets its minimum context and slot count;
- idle timeouts are what you intend for keep-warm roles;
- eviction cost is high for long-context chat and low for small helpers;
- no model was dropped because its engine is not enabled;
- the log names the chosen models and the pool sizes.

### Tooling

| Tool | What it answers |
|---|---|
| `llama-packer --check --profiles P` | Is the profile valid, and where exactly is the typo? |
| `llama-packer --dry-run` | Does the file parse, and what did the packer decide? |
| `llama-swap -config FILE -validate` | Is the emitted YAML a valid llama-swap config? |
| `extras/check_sidecar_budget.py FILE --require t2s,embeddings` | Can the required sidecars be loaded beside each chat? [plan](plans/sidecar-residency-eval.md) |
| `extras/sidecar_eval.py --profiles P` | What does each co-residency strategy cost in chat context? [plan](plans/sidecar-residency-eval.md) |
| `python -m llama_packer.vram_log --report` | What is the real fixed baseline and peak tenant usage? |
| `llama-packer --probe-memory` | How much VRAM does a model family take at a context/slot grid? |
| `extras/bench.py` | What prompt speed and disk speed feed the eviction cost? |

## Deeper reading

- [`SPEC.md`](../SPEC.md) — the contract: every top-level key, VRAM
  Allocation, Capacity routing, Host pool.
- [`profiles.yaml.example`](../profiles.yaml.example) — one commented
  example per section.
- [`docs/memory-and-eviction.md`](memory-and-eviction.md) — budgets,
  co-residency, capacities, eviction, and worked hardware examples.
- [`docs/llama-swap.md`](llama-swap.md) — what llama-swap can do and the
  `profiles.yaml`/sidecar input that provides it.
- [`docs/architecture.md`](architecture.md) — the levels (role, task, pool)
  and how they fit.
- [`docs/glossary.md`](glossary.md) — one name per concept.
- [`docs/install.md`](install.md) — install and update workflow.
