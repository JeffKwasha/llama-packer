# Plan: one rung search — collapse the serial VRAM passes

Status: **superseded** by [category-semantics.md](category-semantics.md) and
`shade_levels()`/`solve_rung()`. Kept for the conflict inventory, which is
still the best inventory of where the six passes disagree with each other.

Note the two names that appear throughout are now dead: `plan_ladder` and
`_shed_order` had no production caller, and were removed. The enumeration
they describe is `shed_levels`, and the rung choice is `solve_rung`.

Recorded 2026-10-15.
Related: [matrix-categories.md](matrix-categories.md), [opportunistic-coload.md](opportunistic-coload.md),
[usage-tag-table.md](usage-tag-table.md), `llama_packer/writer.py`, SPEC.md "Matrix Context Solving",
docs/glossary.md ("the capability ladder").

## Current state

Planning is six passes, run in series, each answering a question that
overlaps its neighbour's. Two of them can disagree, and the disagreement has
no owner.

| # | Pass | Scope | Decides |
|---|---|---|---|
| 1 | `_mmproj_drop_pass` (writer.py:1758) | fleet | keep or drop vision — **before any demand is read** |
| 2 | `_solve_matrix` (writer.py:1975) | fleet | the shared `chat_ctx` |
| 3 | `_coresident_charge` (writer.py:1923) | per group | which sets the stem may join |
| 4 | `_auto_parallel` (writer.py:1670) | per group | `(parallel, ctx)` |
| 5 | `_bounded_ctx` (writer.py:~1595) | per group | squeeze to what fits |
| 6 | `tools_demoted` (writer.py) | per group | drop the `tools` capability |

The conflict inventory — each of these is a case where the passes answer
differently from one question:

1. **Vision is decided fleet-wide, blind to demand.** A category declaring
   `sacrifice: [vision]` gets no shedding from the ladder; only the global
   `_mmproj_drop_pass` acts, and it consults no demand. `plan_ladder` — the
   code that *would* honour it — is unreachable from production:
   `grep -rn "plan_ladder" llama_packer/` returns only its own definition.
   `require:` and `sacrifice:` are currently knobs that do nothing.

2. **A demand may raise a floor the model cannot meet.** `min_context: 1M`
   raises the budget floor (writer.py:~2233), real VRAM is spent trying,
   then `met_demands` drops the category. The id degrades to
   `tpl.<ctx_label>` — a context-derived name that moves on the next repack.

3. **The auto-parallel fallback drops the demand.** The primary call raises
   `floor` to `max(d.min_context)` and passes `min_parallel`; the
   vision→text fallback after an over-budget flip re-derives `floor` from
   `resolve_min_ctx` with neither, so a text-only rung can be solved below
   its own demand's `min_context`.

4. **`sacrifice` is a preference but is merged first-wins.**
   `_shed_order` fills its preference list by iterating demands in order, so
   the first tag in a sidecar's `usage:` list decides the shed order. Two
   demands ranking the same capability differently resolve silently, and
   `DEFAULT_SACRIFICE_ORDER`'s reasoning ("draft is the costly thing to lose,
   so it is last") is overridden by tag order. The docstring calls the merge
   "most protective"; it is not.

5. **`Rung` is not a `Variant`.** The ladder's own output type cannot be
   emitted, so adopting it means either unifying the two or mapping between
   them.

## Goal

One enumeration per `(model, profile group)` that decides, together:
which capabilities are shed, how many slots, and at what context — with
each candidate checked against feasibility *and* demand satisfaction before
it competes.

The ladder already exists and is already tested as a unit (28 call sites in
`tests/test_usage.py`). What is missing is that it produces `Rung`s nothing
consumes.

## Design

### One search, replacing six passes

```python
def solve_rung(view, group, demands, budget, sheddable) -> tuple[Rung, ...]:
    """Rungs worth emitting for one model under one profile group.

    Enumerate (shed subset × slot count), derive ctx from the affine law for
    each, and keep the ones that clear feasibility and the demands. Rank the
    survivors and return them in capability order.
    """
```

For each candidate:

1. **ctx from the budget** — `calc_ctx` at `(shed, parallel)`. One call
   site, not four.
2. **Feasibility** — a candidate whose honest context falls below
   `_MIN_CTX_SIZE` is not a candidate (this is the invariant, now also
   enforced at emit by `_verify_no_overallocation`).
3. **Demand satisfaction** — at least one demand satisfied by
   `(ctx, parallel)` at that shed level. A rung satisfying nothing is
   dropped when demands are non-empty.
4. **Rank** — `parallel_value(ctx, floor, parallel, parallel_power)`, the
   existing score, plus the ladder's gain gate: keep a rung only when it
   gains context over the next-more-capable one.

`afford` is a two-line adapter over `calc_ctx`. Everything else — subset
enumeration, the protective merge, the gain gate — is `plan_ladder`, used as
written.

### What the passes become

| Today | After |
|---|---|
| `_mmproj_drop_pass` | **deleted** — shed is a search dimension, not a fleet-wide pre-decision |
| `tools_demoted` | **deleted** — already per-serving after `tools_demoted` was moved to the solved context; it stays a consequence of the rung, not a separate rule |
| `_auto_parallel` × 4 call sites | **one** — the slot axis of the search |
| `_bounded_ctx` × 4 call sites | **one** — the ctx axis of the search |
| `_solve_matrix` | **narrowed** — keeps exactly one job: how much VRAM do the *non-chat* residents take. It stops deciding anything about a chat model's capabilities. |
| `plan_ladder` | **live** — the enumerator |

The floor stays the dominant lever, unchanged: it is what forces a large
agentic model to `parallel: 1`. `plan_ladder` extends the same logic along
the capability axis instead of duplicating it along a special-cased vision
axis.

### Make `sacrifice` deterministic

`_shed_order` should merge by rank rather than first-wins: average each
capability's position across demands, sort ascending, and log when two
demands rank the same capability first so the disagreement is visible rather
than silent.

### Cap the subset enumeration

`plan_ladder` calls `afford` once per subset, so cost is `2**len(sheddable)`.
`KNOWN_CAPABILITIES` is currently `{vision, draft}` — four calls. Bound
`sheddable` at the search boundary anyway, and short-circuit a rung that
already reaches `max(afford(...))`: no further shedding can buy context.

## Migration

1. **Characterisation tests first.** Freeze current output for a fixture
   fleet: entry ids, `(parallel, ctx)` pairs, shed decisions. Every emitted
   config should be byte-identical at the cutover, or the diff is the bug
   list. Expect differences in `require:`/`sacrifice:` models — that is the
   point — so exclude them from the byte-identical set and assert their new
   behaviour separately.
2. **Shadow mode.** Run the search alongside the existing passes and log
   disagreements without acting on them, over the real model tree. Cheap:
   coefficients are cached per `(cache_type, shape)`.
3. **Cut over per role.** Chat models first (they carry the demand machinery);
   then fixed-overhead and RAG roles, which never enter the ladder.
4. **Delete** the five passes and their tests once shadow mode is clean.

## Risks

- **Cost.** Four subset-slots per model-group, each a `calc_ctx`. With the
  static cache keyed on `(cache_type, shape)` this is arithmetic, not
  measurement — but the first pack after a cache_type change pays a probe
  per variant, and that multiplies.
- **The diff will be large.** Six passes collapse into one, and emitted
  configs change for any model whose demand was previously ignored. That is
  the intent, but it needs the characterisation tests above to be reviewable.
- **Interaction with `_solve_matrix`.** The shared `chat_ctx` currently
  constrains every chat model before per-model solving. In the new shape it
  becomes a *budget input* rather than a target, which is a different
  contract. This is the subtlest part of the cutover and wants its own review.
- **`plan_ladder` was written against a planner that does not exist yet.**
  It has thorough unit tests, but never ran against real models, real
  coefficients or real profiles. Treat its first production run as untested.

## Not in scope

- The affine law and `parallel_power`. Both stay as they are;
  `parallel_power` remains the single knob for the context/concurrency
  trade (SPEC.md, "What `parallel_power` actually controls").
- The matrix's treatment of non-chat residents, which is a separate
  question about sizing unmeasurable models — see
  [opportunistic-coload.md](opportunistic-coload.md).