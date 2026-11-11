# Roles, tasks, and routing

Status: contract for the refactor. Code follows this page; where they differ,
the code is wrong and gets fixed next.

## Vocabulary

| word | meaning |
|---|---|
| **model** | weights plus sidecar. Declares engines, role, tasks, context ceiling, capabilities, `pp_tps`. |
| **task** | one endpoint contract plus demand. The only thing bound to server API capabilities. |
| **endpoint** | fixed server API surface (chat, embeddings, rerank, images, transcriptions, speech). |
| **role** | operator-defined ownership and allocation policy: a label, the directory it owns, and a `share`. No built-in names. |
| **set** | a named combination of roles or model ids with one budget. Authored sets drive matrix mode; generated sets size capacity mode. |
| **pool** | a memory limit with a measured reserve and an operator spare. `default` is the GPU pool, `ram` is the host pool. |
| **share** | a role's request weight in a set's auction, or `min`, `0`, or `fill`. |
| **floor** | the memory a serving needs at its task's minimum context and slots. Derived per model, never configured. |

A role is not an endpoint, a task, a device, or a slot count. Two models with
different tasks and endpoints can share a role.

## Role

A role is ownership and allocation. Models are assigned by the tree it owns
(or a sidecar `role:`); models in the same role evict each other on request.

```yaml
roles:
  dog:
    dir: qwen3           # the tree this role owns
    share: fill          # take the auction the other weights leave
  cat:
    dir: misc
    share: 20%           # request weight in each set
  always-on:
    dir: embed
    share: min           # floor only
    device: cpu          # every model of this role runs from system RAM
```

The schema:

- `dir` — the models-root-relative directory or directories this role owns;
  `.` is the root. Every model under them belongs to the role. A directory
  name carries no meaning: `qwen3` is a role's tree only because `dir` says
  so. A tree belongs to exactly one role; overlapping trees are an error.
  `dir` is optional; a role without one is reachable only by an explicit
  sidecar `role:`.
- A role has no behaviour. The models under it declare their `tasks:`, and
  each task names the endpoint it is served at. A role is ownership and
  allocation, nothing else.
- `share` — the role's request in each set it belongs to:
  - `N%` (or `N`) — request weight `N`. Weights are relative and may exceed
    100.
  - `fill` — the weight the weighted roles leave below 100, split equally;
    0 when they leave nothing, then accepts a remainder a capped role frees.
  - `min` — request 0, accept nothing; the role stays at its floor.
  - `0` — request 0, accept whatever remains unclaimed.
  A role with no `share` accepts but does not request.
- `device` — `auto`, `cpu`, or a GPU index, exactly the sidecar vocabulary;
  leaves placement to the sidecar/task. See SPEC §Device for the three
  places and their precedence.

There is no `min`, `preferred`, `cap`, `weight`, `exclusive`, or `persistent`
key. `share` carries both the request and the fill. Floors are derived. A role
runs one serving at a time, so exclusivity is a single-leaf set.

Roles are data. The code ships no role names and no directory table. The
default roles are defined in `llama_packer/profiles.yaml`; your `profiles.yaml`
replaces them entirely. `dirs:` is retired: the tree-to-role mapping lives in
the role.

## Task

```yaml
tasks:
  spot:
    endpoint: openai.chat.completions
    device: 0                # auto | cpu | GPU index
    min_context: 32k         # optional; top-level min_context is the default
    max_context: 64k         # optional ceiling
    min_parallel: 2
    require: [vision]
    sacrifice: [draft]
    parallel_power: 0.75
    ttl: 0
```

- `endpoint` (required) — from the fixed endpoint set.
- `device` — `auto`, `cpu`, or a GPU index (SPEC §Device). A task's value is
  a default; a sidecar `device:` wins.
- `min_context` — the serving's context floor. A task value overrides the
  top-level `min_context`. `min_context > max_context` is an error and no
  config is written.
- `min_parallel` — the serving's slot floor. These two produce the memory
  floor.
- `max_context` — ceiling on the served context. Clients chunk to fit.
- `require` / `keep` / `sacrifice` — capability policy.
- `parallel_power` — per-task slot exponent.
- `ttl` — idle lifetime. `0` disables idle unload.

A task does not name a role. The model's role is the model's.

## Model

Sidecar keys that matter here:

```yaml
role: dog
tasks: [spot, hunt, tricks]
engines: [llama-server]
context_length: 131072
capabilities: [vision, tools]
pp_tps: 1200
```

- `role:` is a valid sidecar key. It overrides the role that owns the
  model's tree and must name a role defined in `profiles.yaml`; if it does
  not, the run errors and quits.
- `tasks:` lists task ids. An undeclared id warns and is ignored.
- If `role:` is absent, the role whose `dir` owns the model's tree is used.
- A model with no `role:` and no owning role is an error: the run quits. It
  is never silently assigned a default.

## Memory is a tuple

Every serving has a cost in every pool, including zero, and `cpu` is the
first field:

```python
Memory = namedtuple("Memory", ["cpu", "gpu0", "gpu1", ...])
```

The field list is built from the host: `cpu`, then one field per discrete
GPU, or one shared field for unified memory, plus fake limits such as a role
pool of 1.

| placement | cost tuple |
|---|---|
| GPU model | `{gpu0: n, cpu: 0}` |
| CPU model | `{cpu: n}` |
| projector on the host | `{gpu0: a, cpu: b}` |

Placement is not a special case: it is which fields are non-zero. The planner
is total over pools; the emitted `capacities:` stays sparse, because
llama-swap treats an absent pool as zero. Tensor splits across cards are
future work; the tuple already expresses them.

## Allocation

The full algebra lives in [allocation.md](allocation.md). In short:

- A serving's floor comes from its task's `min_context` and `min_parallel`,
  through the measured memory law.
- A role's floor is the maximum floor of all models in the role.
- A set's floor follows its expression: `&` sums floors, `|` takes the
  maximum, `+set` inlines a set.
- `auction(set) = available − floor(set)`. Members request it by `share`
  weight; a role accepts the minimum offer across the sets it belongs to;
  squeezed sets renormalise their remainder.

A role runs one serving at a time in both routers. In capacity mode every
role is a pool of value 1 and every entry charges 1. In matrix mode a role
label expands to one of its entries.

## Routing

There is no `routing.use`. The mode follows from the input:

- `matrix.sets:` present → matrix.
- absent or empty → capacity.

`group` is not emitted.

### Capacity

The packer emits pools and per-entry `capacities`. llama-swap loads the
requested model if it fits every pool, evicting by `evict_cost` when it does
not. Any combination that fits is allowed.

Set generation is not designed yet. Until it is, every model is its own
single-model set, so capacity sizes each model alone. Serious fleets author
`matrix.sets:`. Once generated, an ephemeral set is identical to an authored
one; only its origin differs.

### Matrix

Sets are the only way to forbid a combination that fits:

```yaml
matrix:
  sets:
    pair: "foo & bar"
    alone: "baz"
```

Expressions use llama-swap syntax: `&`, `|`, `()`, `+set`. Leaves are role
labels or model ids. A single leaf is a valid set and confines that role to
running alone. A model in no set runs alone.

Each set is its own budget. The packer sizes each member by the auction and
gives an entry the tightest allocation among the sets it appears in. There is
no sum over all models.

Matrix does not preload. On a request for X, llama-swap picks the cheapest set
containing X, evicts what is outside it, and starts only X. Subsets are
permitted. A set is a permission, not a prescription.

### Fidelity

| | capacity | matrix |
|---|---|---|
| who decides what may load together | arithmetic: any combination whose capacities fit every pool | the authored sets |
| budget used to size | generated sets | authored sets |
| memory enforced at runtime | yes, by the pools | no, the set is a permission |
| preloads | no | no |
| one serving per role | a role pool of 1 | a role label expands to one entry |
| runs alone | a generated single-role set | a single-leaf set, or no set |
| no idle unload | `ttl: 0` | `ttl: 0` |

Matrix buys the right to forbid a combination that fits. Capacity always
admits any combination whose numbers fit.

## Prefill and eviction cost

Eviction cost is derived, not configured per role:

```
evict_cost = model_size / drive_mbps
           + (context * slots / pp_tps, KV-cached endpoints only)
```

`pp_tps` comes from the sidecar, else `capacity.pp_tps`, else an estimate from
active parameters. Stateless endpoints keep no KV, so they pay no prefill.

## Worked example

```yaml
roles:
  dog:
    dir: qwen3
    share: fill
  cat:
    dir: misc
    share: 20%

tasks:
  spot:
    endpoint: openai.chat.completions
    device: 0
    min_context: 32k
    min_parallel: 2
    require: [vision]
  hunt:
    endpoint: openai.chat.completions
    device: 0
    min_context: 128k
    min_parallel: 1
    require: [vision]
  tricks:
    endpoint: openai.chat.completions
    device: 0
    min_context: 64k
    min_parallel: 1
```

One model under `qwen3/`, `tasks: [spot, hunt, tricks]`, produces up to three
entries. It belongs to `dog` because `dog` owns `qwen3/`; no `role:` line is
needed. All three are in role `dog`, so one is loaded at a time. `dog` fills
and takes what `cat` leaves. A sidecar `role: cat` in the same tree moves it;
`role: wolf` errors because `wolf` is not defined.

## Implementation status

Implemented:

- `roles:` is data. The code loads the default roles and replaces them from
  `profiles.yaml`. Role names are not hardcoded.
- `dir`, `share`, `device` are parsed. Two roles may not own overlapping
  trees.
- A role owns its trees with `dir`; discovery assigns every model from the
  owning tree. `dirs:` is retired and errors, the built-in directory table is
  gone, and `--extra-dirs` warns and is ignored.
- A model that resolves to no role, or a sidecar `role:` naming an undefined
  role, stops the run (`DiscoveryError`).
- `tasks:` no longer takes `role:`. A task's `device:` places every model
  that serves it when the sidecar declares none (conflicting task devices
  warn and leave placement alone); `max_context` is the serving ceiling.
  `role:` inside a task warns and is ignored for old files.
- Sidecar `role:` is valid and overrides the tree's role.
- A role has no behaviour: a model's declared tasks name the endpoint, and
  the endpoint selects engine flags, discovery guards and solve paths.
- Retired role keys (`kind`, `min`, `preferred`, `cap`, `weight`, `exclusive`,
  `persistent`) warn and are ignored.
- The router follows the input: `matrix.sets:` present → matrix, absent →
  capacity. `routing.use` is retired and errors; the group router is gone.
- `matrix.sets:` expands role labels to the entries serving them; a role
  with no entries is removed from the expression and the set keeps the rest.
- The chat solve prices a set by its worst branch.
- `__CHAT_VARS__` / `__COLOAD_VARS__` are retired and rejected; the
  opportunistic co-load pass is gone.

Not implemented yet:

- `share` values and the auction. The solver still uses the global spare.
- Role floor as the maximum model floor; set floor by expression; set
  feasibility.
- The tuple memory model: two fields (`vram`, `ram`), not one per pool.
  `cpu` is not first.
- Generated sets for capacity mode.
- A top-level `ttl` and a role `ttl`; only task and sidecar TTL exist.
- Per-pool `spare` from the CLI; `reserve_extra` and `pins` are still
  accepted.
- The retired set-category selector is still accepted by the shared solve.
- The fallback entry is still priced as a chat task.
- `min_context` as a required task value with a top-level default.
- The old `fill:` key and the old per-role count key are still accepted; the
  parser does not retire them yet.
