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
| `Planner` | `writer.py` | Every *context decision*: mmproj keep/drop pre-pass, shared matrix solve, grouping via `Profiles.groups_for`, bounded-context clamp. Emits `Variant` values |
| `Variant` | `writer.py` | Frozen plan for one llama-swap entry: parallel/cache_type/spare_mb, profile group, ctx_size, include_mmproj, optional vision_ctx |
| `emit_config` | `writer.py` | Pure rendering: plans → entry dicts. Zero VRAM contact, zero I/O |
| `_filter_supported` | `writer.py` | **The** validation boundary: backend format/role compatibility, reasoning-flag value/applicability, cache-type knowability, capability/companion cross-check (`vision` removed → error; mmproj without `image`/`video` → warning). Runs before any VRAM work so rejected models never consume measurements |
| Backends (engines) | `backends/`, `backends/base.py` | The serving engines (llama.cpp, vLLM, sd.cpp, whisper.cpp, audio.cpp): roles/formats, argv construction (`build_cmd`), and the resource requirements needed to launch them (`host_requires`/`container_requires`). Each renders a resolved `Model` into a `cmd` |
| Transports | `backends/transport.py` | How an engine's process is launched: `host`, or a container runtime (`docker`, `podman`). Owns host→container path translation, mounts, container env, GPU device flags and lifecycle (`stop_cmd`/`unload_timeout`). One `ContainerTransport(runtime)` serves both docker and podman |
| Registry binding | `backends/__init__.py` | Materialises one `BoundBackend` per valid (engine, transport) pair; engines declare which transports they support. Names: bare engine for host, `<engine>-<transport>` otherwise. Inference order: engines in declaration order, transports host > podman > docker; a container pair is only inferred when its runtime is on `PATH`. Selection: sidecar/override `backend:` > format/role inference gated by configured resources |
| `VramBudget` | `vram.py` | Per-model VRAM math: fit-params fetch/persist/scaling, companion folding (`effective_static`), `calc_ctx`, matrix solver primitives |
| Rule primitives | `overrides.py` | Rule compilation/validation, regex matching (`when`), path resolution for templates/LoRAs — applied by `ScopeStack` |
| `GpuProfile` | `hardware.py` | VRAM pool detection and reserve semantics (discrete vs unified memory) |

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
  and add it to `ENGINES` (`backends/__init__.py`). Declare `formats`, `roles`,
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

- [backends/](backends/) — one doc per engine: roles/formats, command shape, config keys, VRAM
- [transports/](transports/) — host / docker / podman: launch wrapping, path translation, lifecycle
- [new-model-pipeline.md](new-model-pipeline.md) — the end-to-end walkthrough
  for "a GGUF appeared": discovery → fit-params estimate → matrix solve →
  auto-parallel → emit, including the measurement-shape contract, the
  batch/ubatch keys, and what deleting a `derived:` block actually costs
  (seconds of header-only work — never a server, never a probe).
