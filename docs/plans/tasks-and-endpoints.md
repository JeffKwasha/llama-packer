# Plan: tasks and endpoints — the serving-contract redesign

Status: **design accepted in principle** (2026-10-29); **P0–P1 landed**
2026-10-30. Branch `feat/tasks-and-endpoints`, stacked on PR #4
(`feat/cpu-residency-budget`).
Supersedes `docs/plans/audio-roles.md`'s role proposal (cited, absorbed) and
the operations-on-engines half of `llama_packer/roles.py` as landed in the
`6ca96fd` roles refactor. Related: `docs/architecture.md`,
`docs/plans/cpu-residency-budget.md` (portions 1–2, sets), `docs/backends/audio-cpp.md`,
`SPEC.md` §Backend support matrix.

Recorded here so the design lives outside the conversation. Two states of
fact: **D#** decisions, and **O#** open items with the operator's lean
attached.

## D1 — the boundary: llama-packer is a config generator

We do not have arbitrary control over llama-swap's runtime. Every behaviour
this design describes must be expressible in what we *emit*; anything
llama-swap governs is best-effort from our side, and anything else is a
pack-time decision. The mapping is the feasibility check:

| design element | emitted llama-swap surface |
|---|---|
| precedence tiers (pinned / role / task / any) | which routing set an entry joins; llama-swap's subset/swap semantics |
| pinned = must fit | pack-time charge in the mandatory reserve; solved here, not at runtime |
| TTL | per-entry `ttl:` field, resolved from the cascade; global default via `--idle-unload` |
| CPU = cost pool flip | `-ngl 0` + host-pool budget + `metadata.packing` (cmd-level) |
| entry/alias sharing | `aliases:` on one entry when resolved params coincide (already: `:coder` aliases, duplicate-id rule) |
| roles/tasks routing | matrix sets + vars, **generated from the role/task tables** instead of hand-authored |

## D2 — engines serve models; models perform tasks

An engine declares no roles and no operations. The operations-on-engines
model (roles.py `Role.operation`, backends' `operations`,
`BaseBackend.supports_role`, `test_every_role_operation_is_implemented`) is
**rejected**. An engine's contact surface is:

- what weights it can load (`formats`, loader hints — audio.cpp families),
- which **endpoints** it serves.

## D3 — endpoints are concrete, standardized, engine-owned

Endpoints are not arbitrary: standardized ids bound to tasks, e.g.
`openai.chat.completions`, `openai.embeddings`, `openai.rerank`,
`openai.audio.transcriptions`, `openai.audio.speech`, `native.*` for
engine-specific surfaces. One file + one class carries them and the
metadata/validation the code needs (id existence, engine coverage);
hardcoded, in one place. Endpoints are set by servers and clients — the
model never controls them.

## D4 — compatibility is explicit in the sidecar

`engines:` lists engine **names** (`audio-cpp`), not transports/bindings
(`vllm-docker` is a deployment choice, not a model fact). The agent writing
the sidecar reads the engine's documentation and declares which engines can
run the model. No `kind` enumeration, no inference of compatibility.

## D5 — tasks: what a model does, agent-assigned from the catalog

- Sidecar carries top-level `tasks: [asr]` — the model's capability claim.
- Assignment is **judgment, not derivation**: no GGUF-header, README or
  filename algorithm can pick it. Source code (imports/calls) is evidence
  for the agent, never a rule. Hints (family → plausible tasks, known
  archs) exist only to *warn* on a suspicious declaration — never to assign.
- Client-servable tasks must be catalog ids (closed set: a task maps to a
  standardized endpoint; "we don't know" = `tasks:` unset/empty, with
  `unverified` semantics — never an invented name).
- A task entry carries: id (+ aliases), intent (input → output), nearest
  confusions/negative examples (what prevents asr-vs-vad-vs-diar mistakes),
  endpoint(s), the scheduling role it implies, **load semantics** (request-
  time parameter vs one-task-per-process — decides multi-task entries and
  service sharing in the solver), engine hints, evidence sources.
- Governance: **adding a task is a contract change** (endpoint + role +
  policy), never an agent's passing invention. The agent leaves it unset
  rather than inventing.

## D6 — family is the interchangeability unit

Family is defined operationally: *serving one taught you how to serve the
other* — same launch options, template, quantization conventions, engine
behaviors. By that test qwen 3.5/3.6/3.7/3.8 are one family and qwen 3.8
"next" is **a different family** (different architecture). Family
membership is agent-assigned with evidence; the header arch string is a
deterministic hint (many-to-one, confirm-only). `chatterbox` vs
`chatterbox_turbo` are distinct families today (distinct task defaults in
the engine table: `clon` vs `tts`) unless upstream documents turbo as a
parameter flavor — catalog merge then, not inference.

Scope collision resolved: top-level `family:` is *ours* (lineage /
interchangeability); the audio.cpp loader family moves inside the engine
block `audio_cpp.family` (upstream vocabulary, loader-only). Migration: the
~8 audio sidecars currently carry it top-level.

## D7 — role: a composition that hosts elements; does not require them

- A role hosts elements; it does **not** require them — RAG with a rerank
  slot empty is thinner but valid; "requires" is struck from role
  vocabulary. Element presence is a quality signal (warn optionally), never
  a validity gate.
- Elements are **grouped by endpoint**: a role's required endpoints are the
  union of its tasks' endpoints; one live serving per required endpoint.
  Tasks sharing an endpoint within a role are **aliases** (one serving
  covers them; never loaded per-alias). Tasks at different endpoints are
  genuine elements (RAG: `/v1/embeddings` + `/v1/rerank`).
- Two models of the same endpoint in a role are **alternatives** (never
  co-resident) — the same rule portion 1 already enforces for same-role
  pins.
- Within a role, **any non-pinned element may evict any other**, including
  across endpoints (TTS may evict STT or the RAG pair) — precedence by
  tier, not by category. Expressed by which routing sets an entry joins;
  no runtime priority logic.
- Roles are **not pinned by default**; role-switch eviction is emergent
  from set overlap (if two roles share elements, nothing is evicted on
  switch). We never write "evict on role switch".
- Entry sharing when params coincide is **best effort only** and largely
  llama-swap's: we emit shared ids/aliases as a pack-time choice, the
  runtime behaviour is theirs.

## D8 — roles declare their tasks, in role/task language

Roles use the same demand vocabulary as tasks/categories
(`require:` / `keep:` / `sacrifice:` / `min_context:` / `min_parallel:`).
Roles **list their tasks** — that is the connection between the role and
its elements — with a persistence class per element:

```yaml
roles:
  chat:
    tasks:
      - chat:  pinned
      - agent: on_demand        # default
      - qa:    on_demand
  rag:
    min_context: 4k             # reaches every element's own serving
    sacrifice: [concurrency, vision]
    tasks:
      - embed:  pinned
      - rerank: pinned
```

Merge rules when a layer's term meets a task's term:

| case | result |
|---|---|
| task silent, role declares | inherits silently |
| both declare the same op | agree, no change |
| one `require: X`, other `sacrifice: X` | error naming both layers |
| `require:` both | intersect |
| `keep:` both | union |
| `min_*:` both | tightest floor wins |

**Role terms rank earlier in the sacrifice order** (the role degrades
before its tasks do). `concurrency` (slots) is a sheddable dimension at
role level; `parallel_power` remains the value function for where inside a
tier to stop shedding slots. Role `min_context`/`min_parallel` floors reach
every element's serving — retiring `embed_context` / `rerank_context` as
knob special cases. Role `require:` stays capability-level (as today);
element membership is the `tasks:` list.

Old default, restated in this language: "require chat and (rag or voice)" —
chat pinned in every role, rag/voice compositions activated on request;
"rag ⇒ embed+rerank pinned", "voice ⇒ stt+tts pinned".

## D9 — tasks are sacrifice-by-default

Within a role, a listed task is **on_demand unless declared pinned**:
loaded when requested, evicting all but the last requested alias — the
current chat-entry semantics, stated as the declared default instead of
implied by the matrix section. Chat must not squeeze `agent` and `qa`
simultaneously; they are aliases of one endpoint+model serving, evicting
each other by request.

## D10 — CPU placement only flips the cost pool

`device: cpu` switches the charge from VRAM to system RAM and nothing else.
CPU models still give up their memory eventually — by TTL, or by room
competition — with one live serving per slot. Today's `ttl: 0` for every
CPU helper is demoted to a **tunable default** (tests
`test_cpu_entry_ttl_zero` / `test_gpu_entry_has_no_ttl` rework on this
branch, not inherited).

## D11 — TTL is orthogonal to precedence

TTL means *eventually it gets freed* — a recognition that the computer has
duties beyond llama-swap that aren't (and can't be) accounted for here. It
is the **temporal complement of `spare`**: spare reserves *space* for
unaccountable tenants; TTL yields *time* to them. Two answers to one fact.
TTL applies at every precedence tier — even pinned entries may carry one
(and lazily reload on the next request; a pinned entry's guarantee is "it
fits", not "it is warm").

Cascade, most-specific first, seed last: **sidecar > task > role >
CLI/global** (`--idle-unload`). Resolution rule: **smallest wins** — the
tightest lifetime wins wherever two scopes disagree (e.g. a pinned embed
claimed by `rag` and `voice` with different role TTLs), with a warning
naming the two scopes.

## D13 — clean break on config syntax (operator policy)

When a config syntax changes, the old spelling becomes alien: a key that
does not parse under the current schema warns (or errors when it would
silently contradict current semantics) exactly as a typo would — nothing
is grandfathered, and there is no tolerated dual vocabulary. Migration is
a sidecar/profiles *author's edit*, not packer tolerance. Immediate
consequences, recorded so the portions execute them:

- `matrix.categories` specs validate strictly: `role`, `selector`, `ctx`
  are the known keys; anything else warns as an alien key (same class as
  requirement 28/29), never silently ignored.
  **Clarified:** migration doesn't mean tolerated dual syntax — when a
  portion changes the spelling, the fleet's sidecars/profiles are
  migrated as part of that portion; the rule is that the *code* never
  keeps a second, old syntax.
- **O4 resolves: `engines:` is required.** There is no inference fallback
  to keep, because nothing old is grandfathered — D4's "explicit
  compatibility declaration" is the only syntax. `backend:` as a
  compatibility declaration becomes alien when P2 lands: a single-engine
  claim is written `engines: [x]`, and a sidecar still declaring
  `backend:` warns.
- `usage:` tags become alien when P4 lands (`tasks:` is the capability
  claim; an un-migrated sidecar warns that its `usage:` tags are
  unhandled).
- Hand-authored `matrix.sets:` become alien once P5's `roles:` table
  generates the routing sets — dual authorship is the drift this branch
  exists to remove.

Absence of a section is not old syntax: a fleet with no `matrix:` today
keeps working (that is a configuration state, not a tolerated spelling);
the strictness applies to keys that are *present*.

## D12 — the one allocation problem (branch C north star)

Every element in every role is one item in one knapsack; degradation exists
at every level:

```
item = (serving, task demand, sacrifice order, value fn)
       quad (model_mib, kv_per_token, slot_mib, compute_mib) — one currency: MiB, affine
       state (ctx, slots, shed set) — discrete candidates only
```

- Alias tasks on one endpoint collapse into one item (solver-level analog
  of D7).
- Residents stop being "fixed overhead": their ctx and slots are rungs,
  offered through the same ladder language.
- A role is a **constraint schedule**: the ordered ladder of its elements.
  "Role degrades by dropping concurrency for its tasks" = role terms rank
  first in the merged sacrifice order (D8).
- Supporting structure, not two problems: states are discrete and small
  (`shed_levels` prefixes; demand minimums; slot caps); greedy on one
  common shadow price (existing value function extended to per-task
  weights); **tier-0 priority = the pinned set**, so "everything pinned
  fits" is data at the top, not a separate pass.

Invariants the solver may never regress: no pool knowledge ⇒ no gate;
`over_budget` refuses instead of silently shedding below `min_context`;
CPU residents stay go/no-go; the two-tier guarantee ("the resident tier's
VRAM is a number computable up front") is preserved as tier-0 priority
data.

## What this undoes (from `6ca96fd`)

- `llama_packer/roles.py`: drop `Role.operation`, `OPERATIONS`;
  the registry stays as the scheduling table + its derived sets.
- `llama_packer/backends/*.py`: drop `operations`;
  `backends/__init__.py`/`scope.py`: drop `supports_role` and the
  coverage check built on it.
- `tests/test_roles.py::test_every_role_operation_is_implemented`,
  backends tests asserting `operations`: reworked against endpoint/task
  ownership instead.
- `SPEC.md` backend support matrix and `docs/architecture.md`: reworded
  from operations to task/endpoint ownership; `INDEX.md` the same.
- Trace rows 90 (`docs/plans/requirement-trace.md`): replaced by per-
  portion rows as tests land.

## Portions (this branch)

Each portion ships green: trace rows → tests → mutation anchors → suite →
`check_trace` → `check_mutation` → live dry-run → `llama-swap -validate`.
One portion in flight at a time; nothing mixes a behaviour fix with a
refactor.

- **P0 docs** *(landed 2026-10-30; docs only, no trace rows)*: this file;
  `docs/glossary.md` + `docs/architecture.md` vocabulary mapping below;
  `SPEC.md` deltas listed in "What this undoes"; `profiles.yaml.example`
  examples for `engines:`/`tasks:`/`roles:` (marked as new);
  `audio-roles.md` cross-referenced and its role table marked absorbed;
  `cpu-residency-budget.md` pruned to shipped status with its followers and
  open defects moved to `roadmap.md`.
- **P1 undo operations** *(landed 2026-10-30)*: `Role.operation`/`OPERATIONS`,
  backend `operations` and `supports_role` are gone; the scheduling registry
  stays. Compatibility moved one hop to **endpoints**: `endpoints.py` holds
  the closed id set and the task→endpoint map, engines declare `endpoints`,
  and `endpoints.serves()` is the check. P3 therefore owns the registry's
  validation API and doc/code agreement test, not the per-engine lists.
- **P2 `engines:`** *(landed 2026-10-30; fleet migration runs separately)*:
  sidecar key (bare engine names — required, D13's clean break; a
  single-engine claim is `engines: [x]`), resolution (order, allow list,
  format, endpoint, availability), refusal by name when none applies, and a
  near-miss hint on an unknown name. `backend:` is an alien key; only an
  orphan (no authored sidecar) keeps inference. The fleet's sidecars are
  migrated by an operator run of `engines-migrate.py`.
- **P3 endpoint registry**: validation API and the doc/code agreement test
  (the file, the closed id set and the per-engine lists landed in P1);
  endpoint ids checked against the docs.
- **P4 task catalog** *(landed 2026-10-31; fleet migration runs separately)*:
  `profiles.yaml tasks:` declares the ids, their endpoints, scheduling roles
  and demands; `tasks.py` validates each against the hardcoded endpoint and
  role sets. A sidecar
  declares `tasks:`; `role:` and `usage:` are alien keys. An unknown id
  warns and is ignored, and an `audio_cpp.task` that contradicts the model's
  task warns. `templates/models_AGENTS.md` documents the catalog. The fleet
  is migrated by an operator run of `tasks-migrate.py`.
- **P5 roles + persistence + TTL**: profiles `roles:` table (tasks list
  with persistence class, demand layer); matrix sets generated from it;
  per-entry TTL cascade (D11) and tier defaults; pinned reserve charged at
  role floors; CPU-helper TTL default re-tuned.

The unified solver (D12) is **branch C**, after this branch's vocabulary
lands — it consumes tasks/roles/endpoints as inputs.

Scope boundary with PR #4: portion 2 (composition validation — softened to
warn-level cross-endpoint substitution; per-category `ctx:`; host-ledger
set awareness) stays there; portion 3's solver generalisation is deferred
here rather than refactored twice on the way to D12.

## Vocabulary mapping (what docs will call things)

| new | old (current code) |
|---|---|
| role | set (`matrix.sets`) |
| task | role (the six ids; `roles.py`) |
| task alias | usage category (+ profile) |
| model pick for an element | matrix category / var |
| engines | backend inference + `backend:` pin |

## Open decisions

- **O1 — tier names.** `pinned / role / task / any` are working names;
  final wording open (cosmetic).
- **O2 — per-tier TTL defaults.** Pinned: no TTL vs long; role/task/any:
  the `--idle-unload` value; CPU helpers' new default value. Leans exist,
  values are tuning.
- **O3 — sampling ownership.** Demand half (min context/parallel,
  sacrifice order) catalog/role-owned; sampling half stays operator config
  (`profiles:`), the task id naming which profile applies — task ids and
  profile names must then coincide by convention. Confirm.
- **O4 — resolved by D13.** `engines:` is required and is the only
  compatibility spelling. Clarified: **the fleet's sidecars/profiles get
  migrated as part of the portion** — the requirement is that the *code*
  never keeps a second, old syntax, not that old files stay unmigrated.
  After the portion lands, an un-migrated `backend:` warns as alien.
- **O5 — per-role engine lists.** Today: one `engines:` list per model.
  If a model hosts different tasks on different engines, that needs
  `role → engines`. Lean: forbid and document until a real case appears.
- **O6 — engine `family/task → endpoints` table: authoritative in code or
  trusted docs?** Authoritative gives pack-time catches (yue2-as-`s2t`
  passes the endpoint check but fails truthfully); docs-only keeps the
  packer out of the engine's task list. Lean: authoritative, generated
  alongside P3.
- **O7 — multi-task models.** One serving per task for load-time engines
  (audio-cpp) vs one parameterized serving for request-time tasks. The
  catalog carries load semantics per task (D5); emission rules to be
  written when the first real multi-task model appears. Lean: entries.
- **O8 — solver alias merge.** One item per alias group demands the
  strictest floor across aliases (min ctx/parallel, union require/keep) —
  vs independently satisfiable per alias. Lean: strictest-floor merge.
- **O9 — update.py invocation.** Standalone script, stdlib-only, has
  `main()`; not packaged today. Lean: `[project.scripts] llama-updater`
  once `extras` is importable (add `__init__.py`, include `extras*` in
  packaging); keeps `./extras/update.py` working unbundled. Not blocking.
- **O10 — role task-list weights** (core vs enhancer ordering) — set at
  role declaration (D8) or derived from the task catalog. Lean: declared
  order in the roles table is the weight, nothing extra.
