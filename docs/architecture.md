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
| `Planner` | `writer.py` | Every *context decision*: per-category rung search, shared matrix solve, grouping via `Profiles.groups_for`, bounded-context clamp. Emits `Variant` values |
| `Variant` | `writer.py` | Frozen plan for one llama-swap entry: parallel/cache_type/spare_mb, profile group, ctx_size, include_mmproj, optional vision_ctx, optional `vram_note` (over-budget squeeze tag for the description) |
| `emit_config` | `writer.py` | Pure rendering: plans → entry dicts. Zero VRAM contact, zero I/O |
| `_filter_supported` | `writer.py` | **The** validation boundary: backend format/role compatibility, reasoning-flag value/applicability, cache-type knowability, capability/companion cross-check (`vision` removed → error; mmproj without `image`/`video` → warning). Runs before any VRAM work so rejected models never consume measurements |
| Backends (engines) | `backends/`, `backends/base.py` | The serving engines (llama.cpp, vLLM, sd.cpp, whisper.cpp, audio.cpp): operations/formats, argv construction (`build_cmd`), and the resource requirements needed to launch them (`host_requires`/`container_requires`). Each renders a resolved `Model` into a `cmd` |
| Transports | `backends/transport.py` | How an engine's process is launched: `host`, or a container runtime (`docker`, `podman`). Owns host→container path translation, mounts, container env, GPU device flags and lifecycle (`stop_cmd`/`unload_timeout`). One `ContainerTransport(runtime)` serves both docker and podman |
| Registry binding | `backends/__init__.py` | Materialises one `BoundBackend` per valid (engine, transport) pair; engines declare which transports they support. Names: bare engine for host, `<engine>-<transport>` otherwise. Inference order: engines in declaration order, transports host > podman > docker; a container pair is only inferred when its runtime is on `PATH`. Selection: sidecar/override `backend:` > format/role inference gated by configured resources |
| `VramBudget` | `vram.py` | Per-model VRAM math: fit-params fetch/persist/scaling, companion folding (`effective_static`), `calc_ctx`, matrix solver primitives |
| Rule primitives | `overrides.py` | Rule compilation/validation, regex matching (`when`), path resolution for templates/LoRAs — applied by `ScopeStack` |
| `GpuProfile` | `hardware.py` | VRAM pool detection and reserve semantics (discrete vs unified memory) |

## Roles, models and engines

Three nouns, three owners. Keeping them apart is what stops "role" from
becoming a synonym for "engine" or "placement":

| noun | answers | owner |
|---|---|---|
| **role** | *what work* the serving does | `roles.py` (one `Role` per id) |
| **model** | *which weights* do the work, and whether they can | the sidecar (`role:`, `device:`, `parallel:`, `context_length`, …) |
| **engine** | *how* those weights are executed | `backends/<engine>.py` (`operations`, `formats`, `host_requires`) |

The engine never names a role. It implements **operations** (`generate`,
`embed`, `rerank`, `generate_image`, `transcribe`, `synthesize`), and a role
names the operation it needs. That is the whole compatibility rule:

```
model --performs--> role --requires--> operation --implemented by--> engine
```

So one engine serves many models, one role can be served by several engines,
and a model may in principle perform several roles (today the sidecar field
is a single `role:`, so the planner treats it as one). A role may also be
served by several models — a cheap one that takes parallel slots and an
expensive one that cannot are both valid `rerank` models.

### A role definition

A role has one name and only these fields:

- **name** — one canonical id: `chat`, `embeddings`, `rerank`, `image`,
  `s2t`, `t2s`. Directory names (`embed/`, `rerank/`, …) map *to* a role
  (`dirs:`); they are not second spellings of it.
- **work** — the service contract (vectors, scores, transcript, audio,
  completions, image) and the **operation** an engine must implement for it.
  This is what decides "is this a chat role" and therefore which planner
  passes apply.
- **claim** — how a model declares the role: sidecar `role:` or the
  directory mapping.
- **cardinality** — several models may serve one role; a routing set holds
  at most one of them. A second model of a role is an alternative, never a
  co-resident.
- **residency policy** — pinned by default (embeddings, rerank), optional
  (s2t, t2s), or on demand (chat, image).
- **context policy** — the default served context for a resident role
  (`embed_context`/`rerank_context`), overridable per category.

| role | work | operation | residency policy | default context |
|---|---|---|---|---|
| `chat` | completions | `generate` | on demand | solved per usage category |
| `embeddings` | vectors | `embed` | pinned | `embed_context` (4096) |
| `rerank` | relevance scores | `rerank` | pinned | `rerank_context` (4096) |
| `image` | image generation | `generate_image` | on demand | design context |
| `s2t` | transcript | `transcribe` | optional | design context |
| `t2s` | audio | `synthesize` | optional | design context |

**Not in a role definition**, because each belongs to the model or the
engine: the engine that serves it, the estimator that sizes it (affine quad
vs file size + fixed buffer), its file format, its placement (`device:`),
and whether it can take parallel slots. audio.cpp serves some `s2t` models
and whisper-server serves others; adding an STT engine changes no role.

### Where the registry lives

`llama_packer/roles.py` is the one table: a `Role` per id, and the derived
sets read off it — `SERVED_ROLES`, `NON_CHAT_ROLES`, `DEVICE_ROLES` (the
`device_roles:` vocabulary), `OPERATIONS`, and the matrix context knob
(`MatrixKnobs.rag_cap`). Adding a role means adding one row; a role without
an engine operation fails `test_every_role_operation_is_implemented`.

Validation follows the same split: an unsupported *value* of a known sidecar
key warns at discovery (`role:` against the registry with a near-miss hint,
`device:` against `cpu`/non-negative index), and operation/role
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
  and add it to `ENGINES` (`backends/__init__.py`). Declare `formats`, `operations`,
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

- [backends/](backends/) — one doc per engine: operations/formats, command shape, config keys, VRAM
- [transports/](transports/) — host / docker / podman: launch wrapping, path translation, lifecycle
- [SPEC.md](../SPEC.md) — behavioral contract (discovery, fit-params
  estimate, measurement shape, matrix solve, auto-parallel, batch/ubatch
  keys, and what deleting a `derived:` block costs)
  (seconds of header-only work — never a server, never a probe).
