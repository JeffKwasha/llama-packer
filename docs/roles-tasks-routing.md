# Roles, tasks, and routing

Status: contract for the refactor. Code follows this page; where they differ,
the code is wrong and gets fixed next.

## Vocabulary

| word | meaning |
|---|---|
| **model** | weights plus sidecar. Declares engines, role, tasks, context ceiling, capabilities, `pp_tps`. |
| **task** | one endpoint contract plus demand. The only thing bound to server API capabilities. |
| **endpoint** | fixed server API surface (chat, embeddings, rerank, images, transcriptions, speech). |
| **role** | operator-defined purpose and allocation policy: a label, the directory it owns, a `share`, and a residency rule. No built-in names. |
| **set** | a named combination of models/roles that llama-swap may keep resident together. Only used in matrix mode. |
| **pool** | a memory limit. `default` is the GPU budget, `ram` is the host budget, and a multi-GPU host has one per card. |
| **share** | the fraction of *slack* a role may claim, as a percentage. |
| **floor** | the memory a serving needs at its task's minimum context and slots. Derived per model, never configured. |

A role is not an endpoint, a task, a device, or a slot count. Two models with
different tasks and endpoints can share a role.

## Role

A role is a purpose. Models are assigned to it by the tree it owns (or a
sidecar `role:`); models in the same role share memory and evict each other
on request.

```yaml
roles:
  dog:
    dir: qwen3           # the tree this role owns
    fill: true           # take the slack no other share claims
    live: 1              # 1 | per-device | capacity
  cat:
    dir: misc
    share: 20%           # fraction of slack
    live: 1
  always-on:
    dir: embed
    share: 5%
    live: 1
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
- `share` — percentage of slack the role may claim. Shares in one pool must
  sum to at most 100%; the remainder is unallocated headroom.
- `fill` — `true` means "take whatever no other share claims". At most one
  role per pool may fill. `share` and `fill` are mutually exclusive.
- `live` — how many servings of the role may be resident:
  - `1` (default): one; requesting another evicts.
  - `per-device`: one per pool-bin (`gpu0`, `gpu1`, `ram`).
  - `capacity`: no role-level limit; the role's share is the only bound.
- `device` — `auto`, `cpu`, or a GPU index, exactly the sidecar vocabulary;
  leaves placement to the sidecar/task. See SPEC §Device for the three
  places and their precedence.

There is no `min`, `preferred`, `cap`, `weight`, `exclusive`, or
`persistent`. Floors are derived (see *Allocation*), a share is already a
cap, and `share: 100%` is what `exclusive` used to mean.

Roles are data. The code ships no role names and no directory table. The
default roles live in `llama_packer/profiles.yaml`; your `profiles.yaml`
replaces them entirely. `dirs:` is retired: the tree-to-role mapping lives in
the role.

## Task

```yaml
tasks:
  spot:
    endpoint: openai.chat.completions
    device: 0                # cpu | GPU index
    min_context: 32k
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
- `min_context`, `min_parallel` — hard floors. These produce the serving's
  memory floor.
- `max_context` — ceiling on the served context. For embeddings/rerank it
  is the RAG window; clients chunk to fit.
- `require` / `keep` / `sacrifice` — capability policy.
- `parallel_power` — per-task slot exponent.
- `ttl` — idle lifetime.

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

Every serving has a cost in every pool, including zero:

| placement | cost tuple |
|---|---|
| GPU model | `{gpu0: n, ram: 0}` |
| CPU model | `{ram: n}` |
| tensor split across two cards | `{gpu0: a, gpu1: b, ram: c}` |
| projector on the host | `{gpu0: a, ram: b}` |

Placement is therefore not a special case: it is which vectors are feasible.
The planner is total over pools; the emitted `capacities:` stays sparse,
because llama-swap treats an absent pool as zero.

## Allocation

A serving's **floor** is the memory it needs at its task's `min_context`,
`min_parallel`, and any capability it must keep. It is derived through the
model's measured memory law, so two models in one role have different floors.

Floors come out first; shares divide only what is left:

```
slack[pool]            = pool − reserve − spare − Σ residents[pool] − Σ floors[pool]
allocation[role][pool] = Σ floors of its servings + share × slack[pool]
```

Residents are the RAG pair, roles placed on the host, `reserve_extra`, pins,
and unmodelled sidecars. A role's floor is not counted twice: a resident that is
also a role contributes its floor to the resident sum and nothing to slack.

A serving is solved inside `floor + role share of slack`, because it may be
alone. What is emitted depends on `live`:

- `live: 1` — the entry charges its actual size to the pool and a size-1
  counting pool enforces one serving. No role memory pool is needed.
- `live: capacity` — the role gets a memory pool equal to its allocation, so
  the runtime enforces the sum when several servings are live.
- `live: per-device` — one counting pool per role and bin.

### Infeasible floors

A model whose floor does not fit its role's allocation in any placement gets
no entry. It is logged with the pool, the need, and the shortfall. Removing it
lowers `Σ floors`, which raises slack, so the process repeats until stable. It
is monotone and terminates.

A role that loses every model keeps its name with a zero floor tuple. It
emits nothing and warns once. That is a legal state: the operator removed the
models, not the policy.

## Routing

There is no `routing.use`. The mode follows from the input:

- `matrix.sets:` present → matrix.
- absent → capacity.

`group` is not emitted. `exclusive` is `share: 100%`; residency is `ttl: 0`.

### Capacity

The packer emits pools and per-entry `capacities`. llama-swap loads the
requested model if it fits every pool, evicting by `evict_cost` when it does
not. Any combination that fits is allowed, so the budget is global:

```
Σ over all models ≤ pool, per pool
```

Shares must sum to at most 100%, because every role may be resident with every
other. That is the honest constraint when every fitting combination is
acceptable.

### Matrix

Sets are the only way to forbid a combination that fits:

```yaml
matrix:
  sets:
    rag: "chat & embeddings & rerank"
    voice: "chat & t2s"
```

Expressions use llama-swap syntax: `&`, `|`, `()`, `+set`, `+auto:orphans`.
Leaves are role labels or model ids. A model in no set runs alone.

Matrix has no global budget. Each set is its own:

```
Σ over each declared set ≤ pool, per pool
```

So `[A&B] [A&C] [A&D]` constrains only the pairs, and all four can take half
the pool. Capacity would demand `A+B+C+D ≤ pool` and force each to a quarter.
Excluding combinations is what lets the remaining ones be larger.

The packer's matrix job: for each set, solve its members to fit, maximising
context; an entry takes the tightest bound among the sets it appears in. No
global sum, no search — the operator's set list *is* the constraint.

Matrix does not preload. On a request for X, llama-swap picks the cheapest set
containing X, evicts models outside it, and starts only X. Subsets are
permitted. A set is a permission, not a prescription.

### Fidelity

| | capacity | matrix |
|---|---|---|
| who decides co-residency | arithmetic: anything that fits | the sets |
| sum range | all models | each declared set |
| memory enforced at runtime | yes | no |
| preloads | no | no |
| one live per role | size-1 counting pool | no set contains two of the role |
| runs alone | `share: 100%` | model in no set |
| never unload | `ttl: 0` + high `evict_cost` | high `evict_costs` |

Matrix buys two things: the right to forbid a combination that fits, and the
right to size each role against its sets rather than against every other role.
When the shares already sum to 100% and no set forbids anything, the two modes
agree and capacity is less to maintain; the packer says so.

## Prefill and eviction cost

Eviction cost is derived, not configured per role:

```
evict_cost = model_size / drive_mbps
           + (context * slots / pp_tps, chat endpoints only)
```

`pp_tps` comes from the sidecar, else `capacity.pp_tps`, else an estimate from
active parameters. Only chat endpoints keep a KV cache, so only they pay
prefill. A resident role can raise the result so it is evicted last.

## Worked example

```yaml
roles:
  dog:
    dir: qwen3
    fill: true
    live: 1
  cat:
    dir: misc
    share: 20%
    live: 1

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
needed. All three are in role `dog`; `live: 1` means one is resident at a
time. `dog` fills, so it takes the slack after `cat` and the residents. A
sidecar `role: cat` in the same tree moves it; `role: wolf` errors because
`wolf` is not defined.

## Implementation status

Implemented:

- `roles:` is data. The code loads the default roles and replaces them from
  `profiles.yaml`. Role names are not hardcoded.
- `dir`, `share`, `fill`, `live`, and `device` are parsed and
  validated: `share` is a percentage in (0, 100], one role may
  fill, shares sum to at most 100%, and two roles may not own overlapping
  trees.
- The default roles and `profiles.yaml.example` use the new schema.
- A role owns its trees with `dir`; discovery assigns every model from the
  owning tree. `dirs:` is retired and errors, the built-in directory table is
  gone, and `--extra-dirs` warns and is ignored.
- A model that resolves to no role, or a sidecar `role:` naming an undefined
  role, stops the run (`DiscoveryError`).
- `tasks:` no longer takes `role:`. A task's `device:` places every model
  that serves it when the sidecar declares none (conflicting task devices
  warn and leave placement alone); `max_context` is the serving ceiling.
  `role:` inside a
  task warns and is ignored for old files.
- Sidecar `role:` is valid and overrides the tree's role.
- A role has no behaviour: a model's declared tasks name the endpoint, and
  the endpoint selects engine flags, discovery guards and solve paths.
- Retired role keys (`kind`, `min`, `preferred`, `cap`, `weight`, `exclusive`,
  `persistent`) warn and are ignored.
- The router follows the input: `matrix.sets:` present → matrix, absent →
  capacity. `routing.use` is retired and errors; the group router is gone.
- `matrix.sets:` expands role labels to the entries serving them; a role
  with no entries is removed from the expression and the set keeps the rest.
- The chat solve prices a set by its worst branch: a role leaf costs the most
  expensive GPU model of that role, a literal leaf costs the model with that
  stem or published id, and the roles being solved cost nothing.
- `__CHAT_VARS__` / `__COLOAD_VARS__` are retired and rejected; the
  opportunistic co-load pass is gone.

Not implemented yet:

- `share` is parsed but not applied: floors, slack, and the role split are not
  computed. The solver still uses the global spare.
- The tuple memory model is two fields (`vram`, `ram`), not one per pool.
- `live: per-device` is not emitted per pool-bin.
- `matrix.categories` is still accepted by the packer's shared solve even
  though the contract retires it.
- The set-feasibility check and the drop fixpoint are not written.
