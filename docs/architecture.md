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
| Backends | `backends/` | Registry + ABC; each renders a resolved `Model` into a `cmd`. Selection: sidecar/override `backend:` > format inference gated by configured resources |
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

- **Add a backend**: create `backends/<name>.py` with a `BaseBackend` subclass,
  register it in `BACKENDS` (`backends/__init__.py`). Registration order is the
  inference preference order; declare `formats`, roles, and `is_available`
  resource gating. Nothing else changes.
- **New profiles.yaml key**: read it through `Profiles`; if it affects
  variants, thread it through `groups_for`.
- **New sidecar field**: add to `Model.FIELDS` only if the builder consumes
  it — everything else passes through to client metadata automatically.
