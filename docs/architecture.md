# Architecture

How llama-packer turns a directory of model sidecars into a llama-swap
`config.yaml`. Behavioral details live in [SPEC.md](../SPEC.md); this document
is about *structure* — which component owns what, and why.

## Data flow

```
main()                                        (__main__.py — thin orchestration)
  ├─ find_bin_dir          → llama_bin, fit_bin        (utils.py)
  ├─ GpuProfile.from_args  → VRAM pool + reserve       (hardware.py)
  ├─ discover.discover     → DFS walk + ScopeStack      (discover.py / scope.py)
  │                          defaults ⊕ sidecar → rules → resolve_companions → finalize
  ├─ _health_check_timeout → healthCheckTimeout        (__main__.py)
  ├─ compute_env_prefixes  → ${VAR} path macros        (utils.py)
  └─ build_config          = _filter_supported         (writer.py — validation boundary)
                           → Planner().plan()          (writer.py — context decisions)
                           → emit_config               (writer.py — pure rendering)
  → write_yaml / config.env
```

## Components and ownership

| Component | Module | Owns |
|-----------|--------|------|
| `Model` | `model.py` | Sidecar aggregate: frontmatter accessors, companion resolution (mmproj/MTP), design context, pass-through metadata. One instance per servable model |
| `ScopeStack` | `scope.py` | The single select-and-set engine for sidecar data: folds `models.yaml` defaults outermost→innermost, applies override rules last-match-wins per key, finalizes backend inference + path refs |
| `Profiles` | `profiles.py` | The profiles.yaml mapping as a typed view: fleet defaults (`cache_type`, `parallel`, spare), `allow_profiles` gating, expression resolution (`base * N`), per-model variant grouping |
| `Planner` | `writer.py` | Every *context decision*: per-category rung search, shared context solve, grouping via `Profiles.groups_for`, bounded-context clamp. Emits `Variant` values |
| `Variant` | `writer.py` | Frozen plan for one llama-swap entry: parallel/cache_type/spare_mb, profile group, ctx_size, include_mmproj, optional vision_ctx, optional `vram_note` (over-budget squeeze tag for the description) |
| `emit_config` | `writer.py` | Pure rendering: plans → entry dicts. Zero VRAM contact, zero I/O |
| `_filter_supported` | `writer.py` | **The** validation boundary: backend format/task compatibility, reasoning-flag value/applicability, cache-type knowability, capability/companion cross-check (`vision` removed → error; mmproj without `image`/`video` → warning). Runs before any VRAM work so rejected models never consume measurements |
| Backends (engines) | `backends/`, `backends/base.py` | The serving engines (llama.cpp, vLLM, sd.cpp, whisper.cpp, audio.cpp): formats and the endpoints they serve, argv construction (`build_cmd`), and the resource requirements needed to launch them (`host_requires`/`container_requires`). Each renders a resolved `Model` into a `cmd` |
| Transports | `backends/transport.py` | How an engine's process is launched: `host`, or a container runtime (`docker`, `podman`). Owns host→container path translation, mounts, container env, GPU device flags and lifecycle (`stop_cmd`/`unload_timeout`). One `ContainerTransport(runtime)` serves both docker and podman |
| Registry binding | `backends/__init__.py` | Materialises one `BoundBackend` per valid (engine, transport) pair; engines declare which transports they support. Names: bare engine for host, `<engine>-<transport>` otherwise. Inference order: engines in declaration order, transports host > podman > docker; a container pair is only inferred when its runtime is on `PATH`. Selection: sidecar/override `backend:` > format/role inference gated by configured resources |
| `VramBudget` | `vram.py` | Per-model VRAM math: fit-params fetch/persist/scaling, companion folding (`effective_static`), `calc_ctx`, matrix solver primitives |
| Rule primitives | `overrides.py` | Rule compilation/validation, regex matching (`when`), path resolution for templates/LoRAs — applied by `ScopeStack` |
| `GpuProfile` | `hardware.py` | VRAM pool detection and reserve semantics (discrete vs unified memory) |

## What we emit, and what llama-swap owns

llama-packer is a config generator. Nothing here controls llama-swap's
runtime: we choose the fields, llama-swap acts on them.

| decision | our output | who acts |
|---|---|---|
| which entries exist | `models:` (`cmd`, `metadata`, `capabilities`, `name`) | llama-swap serves them |
| when an idle entry unloads | `ttl:` per entry, `globalTTL:` top-level (seconds; `0` disables idle unload) | llama-swap's idle timer |
| which entries may be wanted together | `routing.router.settings.capacity` (`pools`, per-entry `capacities`, `evict_cost`) | llama-swap's capacity router |
| what memory each serving claims | `capacities:` on each model | llama-swap, at request time |
| what memory the packer budgets | pack-time arithmetic only | nobody at runtime |

**The packer decides numbers; llama-swap enforces them.** `capacities` are
declarations, so the packer emits the same measurement margin it solved
with. `evict_cost` only orders evictions under pressure; it does not
reserve or preload. A size-1 pool per scheduling role turns "one live
serving per role" into a capacity constraint, so a same-role request evicts
its sibling without any set vocabulary.

Nothing is preloaded: llama-swap starts the requested model and evicts
whatever the pools require. `ttl` is independent — it yields time, not
memory.

Example of the emitted form:

```yaml
routing:
  router:
    use: capacity
    settings:
      capacity:
        pools: {default: 27747, ram: 50000, chat: 1, embeddings: 1, rerank: 1}
models:
  qwen38-27b-dirk.coding.t:
    capacities: {default: 27443, ram: 1300, chat: 1}
    evict_cost: 250
```

## Models, tasks, engines and endpoints

Four nouns, four owners. Keeping them apart is what stops "role" from
becoming a synonym for "engine" or "placement":

| noun | answers | owner |
|---|---|---|
| **model** | *which weights*, what work they can do, and which engines can load them | the sidecar (`engines:`, `tasks:`, `device:`, `parallel:`, `context_length`, …) |
| **task** | *what work* is being done | `tasks.py` — the declared catalog |
| **endpoint** | *where* that work is served | `endpoints.py` — the one registry |
| **engine** | *how* the weights are executed | `backends/<engine>.py` (`formats`, `endpoints`, `host_requires`) |

The engine never names a task or a role. It declares the **formats** it can
load and the **endpoints** it serves. A model declares the engines that can
load it and the tasks it performs; every task maps to exactly one endpoint.
That is the whole compatibility rule:

```
model --declares--> engines                      (can load)
model --declares--> tasks --served at--> endpoints <--served by--> engine
```

So one engine serves many models, one task can be served by several engines,
and a model may perform several tasks. Two models performing the same task
are alternatives, never co-resident.

### A task is a catalog entry

A task is a closed id and carries everything the packer needs to schedule
it: the endpoint it is served at, the scheduling role it implies, its load
semantics (request-time parameter vs one task per process), and the negative
examples that keep near neighbours apart. Adding a task is a contract change
— endpoint + role + policy — never an agent's passing invention; an unknown
id warns and is ignored, and "we don't know" is an empty `tasks:`, not an
invented name.

The task vocabulary is declared in `profiles.yaml tasks:`; the code owns only
the contract it is checked against (the endpoint ids and the scheduling
roles). A sidecar claims its work with `tasks:`; `role:` and `usage:` are
alien keys that warn and are ignored. The six scheduling roles (`chat`,
`embeddings`, `rerank`, `image`, `s2t`, `t2s`) remain the fixed contract.

### An endpoint is standardized

Endpoints are concrete serving surfaces, engine-owned and hardcoded in one
place: `openai.chat.completions`, `openai.embeddings`, `openai.rerank`,
`openai.audio.transcriptions`, `openai.audio.speech`, plus `native.*` for
engine-specific surfaces. Servers and clients choose endpoints; a model
never does. A role's required endpoints are the union of its tasks'
endpoints, and one live serving covers every task sharing an endpoint —
those tasks are aliases of one serving, never one load each.

### A role is a purpose; a category is a model pick

The levels are not interchangeable:

| level | answers | example |
|---|---|---|
| **role** | what a model is *for*; the substitutable slot — one live serving per role | `chat`, `embeddings`, `rerank`, `s2t`, `t2s` |
| **task** | the endpoint contract that implements a role | `openai.chat.completions`, `/v1/embeddings` |
| **category** | a model chosen to fill a role (a matrix var) | `emb`→embeddings, `rnk`/`rnk2`→rerank |
| **capacity pool** | what may be co-resident | `default` VRAM, `ram`, one size-1 pool per role |

A role is not a composition and does not "host" anything. Co-residency is
what the emitted capacities allow: any combination that fits every pool may
coexist, and a size-1 role pool keeps one model per role live. This replaces
matrix sets entirely; there is no `roles:` table and no hand-authored
`sets:`.

`ttl` is the temporal knob: sidecar > task > placement default, `0` = never,
CPU-resident servings default to 3600 s (chat is not special).

### Where the registries live

`roles.py` is the scheduling table: one row per scheduling role with the
chat-pass flag, context knob and `device_roles:` eligibility. Every derived role collection (`SERVED_ROLES`, `NON_CHAT_ROLES`,
`DEVICE_ROLES`) is read off it. The task vocabulary itself comes from
`profiles.yaml tasks:`; each declaration names one of these roles.

`endpoints.py` is the other half: the closed set of endpoint ids and the one
mapping from a task to the endpoint it is served at. An engine declares
`endpoints`; compatibility is `endpoints.serves(engine.endpoints, task)`, so
neither side has to learn the other's vocabulary. The registry's validation
API and the doc/code agreement test live here (`docs_agree`,
`test_registry_docs_agree`).

Validation follows the same split: an alien sidecar key warns at discovery
(`role:`/`usage:` are not keys — the warning points at `tasks:`), an
unsupported `device:` warns (`cpu`/non-negative index), and format/endpoint
compatibility is checked once in `_filter_supported` before any budgeting.

## Invariants (and their single homes)

| Invariant | Home |
|-----------|------|
| Context clamp order: VRAM solve → min(max trained context) → min(`--max-context`) | `Planner._bounded_ctx` — the only package-level `calc_ctx` call site |
| Spare precedence: profile value > CLI `--spare` > 0 | `Profiles.spare_mb` / `global_spare_mb` |
| Cache precision precedence: sidecar > profile > `q8_0` | `Model.cache_type_for` fed only from `Profiles.default_cache_type` / profile values |
| Validation happens exactly once, before budgeting | `build_config` composes filter → plan → emit in that order |
| Plans are values; rendering is pure | `Planner.plan()` returns `dict[stem, list[Variant]]`; `emit_config` has no side effects |
| Precedence rules exist in one place | raw `defaults:`/`profiles:` dicts are only read through `Profiles` |
| Backends are (engine × transport) bindings, not per-combination classes | `BoundBackend` over `ENGINES` × `TRANSPORTS` (`backends/__init__.py`); container mechanics live only in `transport.py` |

The plan/emit split is deliberate: planning depends only on models'
`VramBudget` interfaces (injectable/fakeable), emission is deterministic over
values. That converts "how do I test config generation?" into asserting on
plain data — see `tests/test_planner.py`.

## Testing seams

- Fake a model's budget directly: `model.vram.calc_ctx = lambda *a, **k: ...`
  (see `tests/test_planner.py`) — no subprocess mocking needed.
- Assert emission with literal expected entries: `emit_config` output compares
  equal to hand-written dicts.
- Companion folding math is unit-tested at the `VramBudget` level
  (`tests/test_companions.py`).
- CLI helpers are extracted functions (`_health_check_timeout`,
  `_apply_env_subst`) tested without running `main()`.

## Extension points

- **Add an engine**: create `backends/<name>.py` with a `BaseBackend` subclass
  and add it to `ENGINES` (`backends/__init__.py`). Declare `formats`, `endpoints`,
  `transports` (which of `host`/`podman`/`docker` it runs under), and
  `host_requires`/`container_requires` (the `avail` keys needed to launch).
  Registration order is the inference preference order; the transport seam
  means one engine class serves every supported transport. Nothing else changes.
- **Add a transport**: implement a `Transport` in `backends/transport.py` and
  register it in `TRANSPORTS`; engines opt in via `transports`. Container
  runtimes share one implementation parameterised by the runtime binary.
- **New profiles.yaml key**: read it through `Profiles`; if it affects
  variants, thread it through `groups_for`.
- **New sidecar field**: add to `Model.FIELDS` only if the builder consumes
  it — everything else passes through to client metadata automatically.

## Where to read more

- [backends/](backends/) — one doc per engine: formats/endpoints, command shape, config keys, VRAM
- [transports/](transports/) — host / docker / podman: launch wrapping, path translation, lifecycle
- [SPEC.md](../SPEC.md) — behavioral contract (discovery, fit-params
  estimate, measurement shape, shared context solve, auto-parallel, batch/ubatch
  keys, and what deleting a `derived:` block costs)
  (seconds of header-only work — never a server, never a probe).
