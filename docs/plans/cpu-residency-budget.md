# Plan: CPU-residency budget — a host-RAM pool for `device: cpu` models

Status: **in progress** (branch `feat/cpu-residency-budget`). Recorded
2026-10-23. Related: `docs/plans/memory-allocation-tags.md` (§3 measured host
RAM — this plan is its minimal form), `docs/plans/opportunistic-coload.md`
(CPU-resident coload at 0 GPU cost, today's rule), `docs/plans/gpu-split-ladder.md`
(follower branch — uses the measurement extension this plan builds),
`SPEC.md` §VRAM Allocation, `llama_packer/{vram,hardware,writer,model}.py`.

## Problem

The planner's budget is VRAM-only. A sidecar `model.md` may declare
`device: cpu` (`model.py:1602`), and the current treatment is "not VRAM's
problem": excluded from every VRAM solve, charged 0 by
`_resident_overhead` (`writer.py:2130`), sized to design context. Nothing
checks *whether host RAM can actually hold it*. The operator's existing
statement is only `MemoryMax` on llama-swap.service — a cgroup fuse, not a
planner decision. Goal: the class of sidecars the tool serves — `device: cpu`
— gets a host-RAM budget with the same decision quality as the VRAM solve;
overflow of that pool refuses to emit the model id (~ same `over_budget`
path).

## Decisions (operator-confirmed, 2026-10-22/23)

1. **No automatic CPU anywhere.** `device: cpu` frontmatter is the only
   gate; no profile-level master switch, no solver descent to full CPU, no
   fallback auto-offload. (`model.py`, `vram.py:898` honor the explicit gate
   already; nothing else may set it.)
2. **One co-resident per role** (STT, TTS, embed, rerank): sized variants of
   the same role must never co-reside; the resident winner per role is
   chosen explicitly. For pack-time enforcement: the resident selection
   already picks per role (`SPEC.md` §Residents); this rule is asserted in
   the host ledger — two co-resident quads for one role are a planner bug.
3. **Context: design ctx, no solve.** CPU models serve design/sidecar
   context; the host check is go/no-go, never a solve-down (mirrors today's
   "CPU-resident excluded, sized to own architectural context" rule).
4. **Declarations, not probes.** Pool values are *declared* when operator
   knows better; runtime detection is discovery-with-fallback, never
   assumption. Every detected value is overridable in `profiles.yaml
   hardware:`.
5. **VRAM-resident models never charge host bytes** — a VRAM-only resident
   leaves behind its GGUF's page-cache residue, which `MemAvailable`
   already discounts; nothing is double-counted into the host pool.
6. **Parallel is honored and charged.** Declared `parallel:` on a CPU-side
   chat model is charged its full quad (p·D + KV pool) and may push the
   entry out of budget. Residents (emb/rerank) remain single-slot (SPEC §
   Residents, "a declared parallel: on an embeddings/rerank model is ignored
   with a note"). Speech entries (audio.cpp) have no parallel knob.
7. **Threads: nothing is emitted.** Engine defaults (llama.cpp
   hardware-concurrency, audio.cpp default) are already correct; sidecar
   audio.cpp `threads:` stays a pure per-model passthrough.
8. **VRAM-side placement is also always the standard one** — `device: cpu`
   is a *host-only* statement; a hybrid split is branch 2 and follows
   `sacrifice:`/category vocabulary, not `device:`.
9. **Sibling, not subclass.** A `HostLedger` sits *beside* `PoolLedger`
   (same over_budget machinery), not a refactor to a `memory.py` parent:
   the bulk of `vram.py` is device-specific; the shared atoms are `FitParams`
   and the `-lv 5` log parse. The measurement extension (below) is the only
   DRY surface the two budgets touch.
10. **`device: cpu` is the resident selection.** A role whose candidates
    include a `device: cpu` model takes that model as the resident winner
    (one per role, decision 2) — the tag is the operator's explicit
    "this is the one that stays warm in host RAM" choice, and it must not
    lose to a cheaper device model. Among the eligible candidates the
    winner is the **cheapest measured serving at the role's ctx**
    (`resident_cost_mb` — the same estimator the budget charges with);
    file size is only the fallback when nothing can be sized. A selector
    (`matrix:` `selector:` / CLI `--embed`/`--rerank`) expresses the one
    thing no measurement can know — value — overrides the cost pick, and
    is priced in the log so its cost against the cheapest is visible.
    Duplicate *servings* of a role stay operator catalog/set policy — the
    resident tier holds one, and the host ledger asserts it
    (requirement 14).
11. **Role-wide placement is declared, not inferred.** `profiles.yaml`
    `device_roles:` may name serving roles (`embeddings`, `rerank`,
    `s2t`, `t2s`) whose every model is CPU-placed exactly as if each
    sidecar carried `device: cpu`: same `-ngl 0`, same host charge, same
    VRAM exclusion, same resident eligibility. This is an operator
    declaration under decision 1 — nothing discovers "too big" and moves
    it, the list is explicit, and every application is logged with the
    role and the number of models it matched.
12. **CPU helpers are never evicted.** A `device: cpu` non-chat entry is
    pinned into every *chat* set expression (by its category var when
    it has one, else its entry id) and carries `ttl: 0`: a request names
    the model, loads it once, and no later swap evicts it. Sets permit
    subsets, so pinning never preloads one. Evicting a CPU entry buys no
    device memory, so the matrix's cost model has nothing to trade —
    this is the placement rule, not a new heuristic. **One pin per role**
    (decision 14): the pin is the role's canonical category (`emb`,
    `rnk`, …), and it is skipped for a set that already names a different
    category of the same role — pinning every CPU model made same-role
    alternatives co-resident (two embedders, two rerankers), which is
    exactly the duplicate serving decision 2 forbids.
13. **The set expression is the budget contract.** The routing `sets` are
    not decoration forwarded to llama-swap: they are the operator's
    statement of which models may be resident together, and the solver
    must price them as such. `&` sums (co-resident), `|` takes the max
    (one alternative), `+set` expands the referenced set. A chat model is
    charged the **worst branch of the most expensive chat-capable set it
    belongs to** — never the union of every declared category. A
    category that appears in no set reserves nothing, because llama-swap
    can never hold it beside chat ("a model in no set can only run
    alone"). Reasoning: the old rule summed every declared category, so
    declaring a 3 GB bucket made every chat solve pay 3 GB even when the
    set holding it was an alternative to a cheaper one.
14. **A set is a role composition, not a category list.** One set names a
    use case: RAG is `emb & rnk` (one embedding model *and* one rerank
    model — two roles), 2-way speech is `stt & tts` (one each). Within a
    set a role appears once: two embedders or two rerankers are never
    co-resident, and a second category of the same role is an
    *alternative* (`rnk | rnk2`), not an addition. Roles compose, models
    of one role do not. This is what makes buckets expressible without
    inventing a task vocabulary: `chat & (three | (two & one))` is a
    valid composition of three operator buckets, priced 3 (the `three`
    branch) rather than 6 (their sum).
15. **One canonical category per role is the CPU resident.** Placement is
    per model (`device: cpu`), residency is per role. When several
    categories of one role are CPU-placed, the canonical name (`emb`,
    `rnk`; else the first declared) is the pinned resident; the others
    are catalog entries that load on request and are not forced into
    every set. Same-role alternatives are never co-resident (decision 2),
    so pinning all of them contradicts the budget the sets describe.

## Set composition (the budget contract)

Three different things get confused in the config; keep them apart:

| term | what it is | where it lives | example |
|---|---|---|---|
| **role** | what a model *does*; one resident per role | sidecar `role:` / directory (`dirs:`) | `embeddings`, `rerank`, `s2t`, `t2s` |
| **matrix category** | an operator-named routing var for a served model | `matrix.categories` (or a bucket name) | `emb`, `rnk`, `one`, `two`, `three` |
| **set** | a use case: a composition of roles/categories that must be live together | `matrix.sets` | `rag: __CHAT_VARS__ & emb & rnk` |

A **set is a role composition**. RAG is not "the embedding category" — it is
`emb & rnk`, one model in each of two roles. 2-way speech is `stt & tts`.
A chat set is a composition with chat: `__CHAT_VARS__ & stt & tts`.

### What a role definition holds

A role is not just a name: it is the bundle of facts below. They already
exist in the code, but scattered across constants, backends and the matrix
config — which is how "role" starts to drift into meaning "placement". The
definition is the same for every role; only the values differ.

- **id** — the accepted spelling. There are six, and they are the only
  strings `dirs:`, `role:`, `device_roles:` and `matrix.categories[].role`
  accept.
- **served by** — the engines that can serve it (`backend.roles`).
- **sized by** — the estimator: the measured affine quad when the engine
  exposes KV, or file size + buffer for the fixed-overhead engines.
- **context policy** — solved per usage category (chat), a fixed small
  `embed_context`/`rerank_context` (RAG), or design context.
- **residency** — a pinned resident or an on-demand serving.
- **placeability** — may `device_roles:` move it, and is `device: cpu`
  honoured?
- **cardinality** — one resident per role, one model per role per set; a
  second category of a role is an alternative.
- **parallel** — honoured and charged, or ignored (residents are
  single-slot).

| role | served by | sized by | context | resident | in `device_roles` |
|---|---|---|---|---|---|
| `chat` | llama-server, vLLM | affine quad | solved per usage category | no | no (per-model `device: cpu` only) |
| `embeddings` | llama-server, vLLM | affine quad | `embed_context` (4096) | yes | yes |
| `rerank` | llama-server, vLLM | affine quad | `rerank_context` (4096) | yes | yes |
| `image` | sd-server | file size + buffer | design | no (not a resident role) | no |
| `s2t` | audio-cpp, whisper-server | file size + buffer | design | when declared | yes |
| `t2s` | audio-cpp | file size + buffer | design | when declared | yes |

Two fields a role definition does **not** have: a device, and a
`role: cpu` form. A role says *what* a model does; `device:` says *where*
that particular model runs; `device_roles: {embeddings: cpu}` is only
shorthand for "set `device: cpu` on every model of this role".

Consequences, in the order the solver applies them:

1. **`&` sums, `|` maxes, `+set` expands.** The expression is a boolean
   tree and is priced recursively, so nested branches price correctly:
   `(c1) & (three | (two & one))` = `max(3000, 2000 + 1000)` = 3000, and
   `(c1) & (one | (two & three))` = `max(1000, 2000 + 3000)` = 5000. The
   old depth-0 split priced the second as 1000 — a `(two & three)` group
   became an unknown var worth 0.
2. **The worst chat set wins.** A chat entry can be requested under any set
   that contains it, and llama-swap activates the cheapest set, so the
   entry must survive the heaviest one: `reserve = max over chat-capable
   sets of (that set's branch cost)`. It is not the sum over declared
   categories, and it is not a per-category admission loop.
3. **A category in no set reserves nothing.** It can only run alone, so
   charging it against every chat solve buys nothing and silently shrinks
   context. The planner logs it as unreferenced instead of reserving.
4. **One model per role per set.** A set that names two categories of the
   same role is a *composition error*, not a fatter use case: the second
   one is an alternative. The planner never adds a same-role model to a
   set that already names one (the CPU pin rule, decision 15), and portion
   2 adds the emitted warning/validation for operator sets that do it.
5. **Buckets are just categories with operator names.** `one`, `two`,
   `three` are declared like `emb`/`rnk`; `combo: chat & (three | (two &
   one))` needs no task vocabulary. The packer does not need to know what
   a bucket is *for* — only its cost and whether a set references it.
6. **Each resident is priced at its own served context** — `embed_context`
   / `rerank_context` for the RAG roles, design context for the rest —
   through the same `resident_ctx` / `resident_cost_mb` seam as selection
   and the host ledger. One context policy, one estimator, three
   consumers.
7. **The log names the terms.** A resident costs
   `weights + compute + kv@ctx`, and the selection line prints all three,
   so two residents that both cost 1050 MiB can be seen to be the same
   shape (two Qwen3-0.6B Q6_K models) rather than a repeated number
   hiding a bug.

CPU residency composes with this without a special case: a CPU category is
charged 0 VRAM but its full quad to the host ledger (requirements 11–14),
and it composes into sets exactly like a device category. That is why the
bucket example works on this branch's fleet — all three buckets are CPU
residents, and the GPU side is free either way.

### Portions

**Portion 1 — set pricing and pinning (this branch).** Bugfixes only, no new
config surface.

- recursive set pricing (`&` sum, `|` max, `+set` expand) for both the
  budget and the host-ledger cost view;
- the solve reserves the worst chat set instead of the category union;
- an unreferenced category reserves nothing and says so;
- the CPU pin is one canonical category per role, and a set that already
  names that role keeps its own choice;
- the resident log prints the cost terms.

**Portion 2 — composition validation and per-category context.** A set
naming two categories of one role warns (or refuses) by name; per-category
`ctx:` so a bucket can declare its served context instead of inheriting the
role's; the host ledger's set awareness (a set that cannot fit its CPU
residents is refused as a unit); `matrix.categories` schema documented.

**Portion 3 — refactor / capacity exploration.** Generalize
`solve_matrix_ctx` from `(chat, embed, rerank)` to a category-parameter map
so the solver no longer knows the RAG roles by name; evaluate llama-swap's
native `capacity` routing as the runtime half of the same contract. Out of
scope until portions 1–2 are green and the live config is repacked.

## Host pool equation

Runtime-discovered, all overridable:

| term | source (primary → fallback) | override |
|---|---|---|
| `total` | `/proc/meminfo MemTotal` — sized at pack | `hardware.system_ram` |
| `pressure_zone` | **sysctl**: per-zone high-watermark sum from `/proc/zoneinfo` (+ `vm.min_free_kbytes` sanity floor); if zoneinfo is missing/zeroed/unparseable → the **declared fallback assumption of 25% of total** | `hardware.host_pressure: sysctl \| 25% \| <abs>` |
| `sys_free` | `MemAvailable_pack − pressure_zone` (pack-time read; `MemAvailable` discounts page-cache residue, counts working memory) | — |
| `service_cap` | the serving deployment's cgroup: `memory.high` (soft alpha — pool surface), `memory.max` (hard, refuse) and `memory.current_pack` (in-use at pack). Located from `/proc/<llama-swap pid>/cgroup` → cgroup path. Absent (no service / non-cgroup host) → term skipped with a warning | `hardware.cgroup_dir` |
| `sg_free` | `min(memory.high − memory.current, memory.max − memory.current)` | — |
| **`host_pool`** | `min(sys_free, cgroup_free)` | `hardware.host_pool: auto \| <abs> \| <frac>%` (`auto` = the formula) |

```
charged      = Σ quads(device: cpu models): weights + host-KV(c·C_design) + compute + p·D
over budget  → `over_budget_reason = "host pool"`, entry omitted, same refusal rule as VRAM
```

`C_design` for the host KV term: design/sidecar context (no solve, decision 3).
The `p·D` and compute terms come from the measurement extension (below), which
is per-model and cached in the same `derived:` block family as VRAM terms.

Log every chosen value and its path (declared / sysctl / fallback-25% /
skipped-cgroup) — a silent fallback is a bug, per the repo's budget-notes
doctrine.

## Measurement extension (the DRY seam)

Three rules this seam lands: the serve-device pin (`-ngl`) is a shape
decision, not a constant; host-side buffer lines parse beside the device
sums; a `Host`-named affine report row is legal input. Observed shapes
(Qwen3-Reranker-0.6B Q6_K, `-c 8192`, q8_0):

- A CPU-shaped `llama-server` run prints every buffer line under the
  ``CPU``/``CPU_Mapped`` device names (real weights split across two
  buffers: 121.50 + 344.78 = 466.28 MiB matching the breakdown table's
  model column; KV buffer 476 MiB; compute 36.10 MiB) — and reports a
  per-pass ``memory breakdown`` table whose ``Host`` row is the pass
  total. Both a dry projection pass (`load_mode = none`, sizes 0 except
  compute) and the real load precede the ready marker.
- A CPU-shaped `llama-fit-params` run prints the same ``CPU`` buffer
  lines (0-weight projection), the breakdown table, and the affine
  `Host`-named report row (`Host 466 476 316`) — which
  `parse_fit_log` used to reject, so every CPU-measured value was
  `None`. The KV-pool `size =` lines carry the pool bytes either way, so
  the affine trio math (c from (C, C/2), D from p) is unchanged.

- A Vulkan-less environment (missing device ACL) silently serves the
  full-CPU shape even under `-ngl 999`; the existing spill check
  rejects it as host-KV capacity spill — the pre-existing guardrail
  does the device-refusal job and stays keyed on device shapes.

One persisted block serves both budgets: a model carries a single
`derived:` block describing its *serving placement's* byte terms, and
the `--n-gpu-layers 0` pin rides the `shape` stamp, so a sidecar
`device:` flip invalidates the block by shape (a device flip re-measures
for the other budget) instead of needing a parallel host block family.
Nothing second-guesses the measurement: tag/report rules per
`memory-allocation-tags.md` v1 contract apply verbatim.

`llama-fit-params` acceptance of extra flags (`llama_args` passthrough
in `fit_params`) is already sufficient for both shapes — no new tool;
`measurement_args` stamps the device pin.


### Requirements (citable by tests)

1. §"Measurement extension", *pass window*: a serve log holds the
   projection pass (`load_mode = none`) and the real load before the
   ready marker, and the projection closes with a `| memory breakdown
   [MiB] |` table. The **host** parse reads the buffer lines after that
   table — the final pass — instead of summing the CPU backend's
   pass-repeated real sizes; when a table has no buffer line behind it
   (post-ready summary refresh), or no table exists, the log parses
   whole (fixture/fit-params compatible). The **device** parse keeps its
   whole-log sums: on a real device the projection buffers measure 0.
2. §"Measurement extension", *host bucket*: `parse_host_buffers`
   reports the host-side (non-device) sums — weights (sum over host
   weight lines, zeros excluded by arithmetic), `kv`/`rs` (pool lines,
   summed across contexts), `compute`, `output` — from the same final
   pass as the device sums, without touching the device rules.
3. §"Measurement extension", *device flag*: `device: cpu` measures
   (and shape-stamps) with `--n-gpu-layers 0` instead of `-ngl 999`,
   stamp included by `measurement_args`; a shape that already carries
   the flag is not duplicated on the measurement command.
4. §"Measurement extension", *CPU spill relaxation*: a `device: cpu`
   measurement never runs the device-shaped spill rejection (host
   KV/RS and CPU-mapped weights are its truth); a `-ngl 999` launch
   that lost Vulkan still measures nothing — the spill check rejects
   it exactly as before.
5. §"Measurement extension", *fit-params Host row*: `parse_fit_log`
   falls back to the `Host` report row when no device row exists, so
   the affine trio (`c` from `(C, C/2)`, `D` from the p pair) derives
   CPU quads without measuring anything anew.

### Host-budget requirements (citable by tests)

**Detection** (`hardware.py`, `HostProfile`). Every term logs its path —
`declared` / `sysctl` / `fallback-25%` / `skipped-cgroup`. A silent
fallback is a bug.

6. *total*: `MemTotal` from `/proc/meminfo` (pack-time reference for
   percent overrides); `hardware.system_ram` declares it instead.
7. *pressure zone*: primary is the sysctl arm — per-zone `high`
   high-watermark sum from `/proc/zoneinfo`, raised to the
   `vm.min_free_kbytes` floor when the read is lower; when zoneinfo is
   missing, zeroed, or unparseable the *declared fallback assumption* is
   25% of total, and that is what is logged. `hardware.host_pressure:
   sysctl | 25% | <abs>` chooses the arm.
8. *sys_free*: `MemAvailable − pressure_zone`. A negative result is not
   a bug report — it feeds the pool floor clause (10).
9. *service cgroup*: resolved from the llama-swap process's
   `/proc/<pid>/cgroup` (v2 `0::<path>`, `/sys/fs/cgroup` root) —
   `sg_free = min(present caps − memory.current)` over
   `memory.high`/`memory.max` (`max`/absent caps skip their term). No
   such process, or an unreadable cgroup, skips the term with a
   `skipped-cgroup` warning (then `sg_free` does not bind).
   `hardware.cgroup_dir` declares the path directly.
10. *pool*: `host_pool = min(sys_free, sg_free)` (skipped-cgroup → just
    `sys_free`), clamped at 0. `hardware.host_pool: auto | <abs> |
    <frac>%` overrides the whole result (`auto` = the formula, `<frac>%`
    of total). An infeasible read (term missing *and* no override) is
    not invented: the pool clamps to 0 rather than assuming.

**Budget** (`HostLedger`, sibling of `PoolLedger`).

11. *charge*: a `device: cpu` model is charged its serving quad at the
    served context — weights + compute + KV(c·C_design·p) + p·D, `C_design`
    the design/sidecar context (decision 3: no solve); the quad is the
    same measured block family VRAM uses (the shape-stamped extension).
12. *refusal*: the check is go/no-go against the pool remaining after
    committed residents — over budget logs the shortfall and emits no
    id (no solve-down, ~ the VRAM `over_budget` path, per-category like
    every other refusal).
13. *residents charge host*: a `device: cpu` embed/rerank resident is
    charged to the host ledger at its solved context; its VRAM-ledger
    charge stays 0 (decision 5 — no double counting).
14. *one co-resident per role*: the host ledger asserts two committed
    quads for one role as a planner bug — the resident selection already
    picks one winner per role (decision 2).
15. *declared parallel is charged*: `parallel: N` on a CPU chat model
    multiplies D and the KV pool in the charge and can flip the refusal.
16. *no pool knowledge, no gate*: with detection unavailable (no
    profiles override, unreadable /proc), the ledger logs that fact once
    and does not gate — it never invents a budget it cannot read.
17. *CPU wins the role, cost picks inside it*: for a resident role
    (embeddings, rerank, and the declared fixed categories), a
    `device: cpu` candidate is selected over device candidates regardless
    of footprint (decision 10). Among the eligible candidates the winner
    is the cheapest **measured serving at the role's ctx**
    (`resident_cost_mb`, the same estimator the budget charges with);
    only when nothing can be sized does the smallest-file rule apply, and
    no CPU candidate leaves the selection exactly as it was. A selector
    (`selector:`/`--embed`/`--rerank`) overrides the cost pick and is
    priced in the log.

**metadata.packing** (client-visible facts; emission site shared with
branch 2).

18. *every entry carries `metadata.packing`* — decided facts only:
    `device` (`cpu`/`gpu`), `ctx_size`, `parallel`; `vram_mib` when the
    serving is device-resident and estimated, `ram_mib` for the
    host-side bytes when decided, and `kv_per_token` (per-token KV
    bytes at the served cache_type) when the backend measures an affine
    quad. `device: cpu` entries carry no `vram_mib`.

**Role-wide placement and eviction** (decisions 11–12).

19. *role declaration*: `profiles.yaml device_roles:` (a list of role
    names, or a mapping **role → device** — `{embeddings: cpu}`) places
    every model of each named role on the host — identical in every
    respect to the sidecar tag (requirement 3's shape stamp, the host
    charge, no VRAM charge, resident eligibility). Each role applied is
    logged with the number of models it matched; a role that matches
    nothing warns; leaving the key out changes nothing. A mapping value
    other than `cpu` places nothing and warns: the key is a role and the
    value is a device — there is no `role: cpu` form. Any other shape (a
    bare scalar) warns and places nothing.
20. *CPU helpers are pinned into every set*: every `device: cpu`
    non-chat entry joins every routing set expression — by its category
    var when it has one, else by its entry id — unless already a leaf.
    Set subsets keep this from preloading, so the effect is exactly
    "never evicted by a swap". The entry also carries `ttl: 0`, so idle
    expiry does not unload it. Device entries keep today's behaviour
    (no pin, no `ttl` key).
21. *one answer per role*: a category selector (`matrix.categories.emb`
    / `.rnk`) defines the RAG pair too — the solve, the charge, the
    `emb`/`rnk` vars and the resident entry all name the same model.
    Two independent selections (CLI/role pick vs category spec) must
    never disagree.

**Set composition** (decisions 13–15; portion 1).

22. *recursive pricing*: a set expression prices as a boolean tree —
    `&` sums, `|` maxes, `+set` expands the referenced expression — so
    `(one | (two & three))` is `two + three` when that is larger, and a
    nested group is never mistaken for an unknown var worth 0. A `+set`
    cycle terminates instead of recursing.
23. *worst branch of the worst chat set*: the chat solve reserves the
    most expensive chat-capable set's branch cost. Declared categories
    that only appear as alternatives add nothing, and a category in no
    chat set adds nothing.
24. *unreferenced reserves nothing*: a category no set names is logged
    as unreferenced and charged 0 — it can only run alone.
25. *the CPU pin is per role*: exactly one canonical category per role is
    pinned into the chat sets (`emb`, `rnk`, else the first declared),
    and only when that category is CPU-placed. Pinning never adds a
    second model of a role.
26. *a set keeps its own role choice*: a set that already names a
    category of the pinned role (e.g. `ragz: … & rnk2`) is emitted
    unchanged — the canonical `rnk` is not appended to it.
27. *a priced resident names its terms*: the resident selection log
    shows weights, compute and KV at the served context for a measured
    pick, not only the total.

## `metadata.packing` (client-visible facts; branch-1/2 shared site)

One structured dict emitted by `_build_entry` (`writer.py:440-545`) for every
entry; only facts that were *decided* at pack time:

```yaml
metadata:
  packing:
    device: cpu          # gpu (default) | cpu | gpu+cpu (branch 2, when flags actually emitted)
    vram_mib: …          # device-resident bytes; absent on device: cpu
    ram_mib: …           # host-side bytes (computed/evaluated host affine tag source)
    ctx_size: …          # per-slot served window (duplicate of metadata.ctx_size kept for compat)
    parallel: …          # slot count — today not machine-readable anywhere
    kv_per_token: …      # per-token KV bytes at the served cache_type
# branch 2 adds: `split: cpu-moe` (absent when no split flags emitted to the
# server — decided at the same build step that assembles the cmd).
```

Description stays prose (mem-tag only): the *operator* channel. No new
description grammar; mem-tag `+`/`SSD`/RAM rules unchanged.

## First-wave sidecars

- `embed/jina/jina-v5-small-retrieval` → `device: cpu` (operator preference).
- `rerank/Qwen3-Reranker-0.6B-Q6_K` → `device: cpu` (only GGUF present in
  `rerank/`).
- Everything else untouched; STT/TTS placement stays exactly what the
  deployment's profiles/sidecars say (audio.cpp serving is the operator's
  call; runtime audio backend changes are not this branch's work).

## Tests

- **Mutations (prove discipline):** (a) refusal — a `device: cpu` model whose
  quad exceeds the pool emits no id and logs `over_budget_reason`;
  (b) pool floor — an infeasible `MemAvailable` (fixtures) clamps to 0 and
  omits every CPU model rather than emitting negatives; (c) parallel —
  declared `parallel: 2`'s `p·D`+pool bytes actually decrement the pool and
  can flip the refusal verdict.
- **Detection fixtures:** fake `/proc/meminfo`, `/proc/zoneinfo`,
  `/proc/sys/vm/*`, `/proc/<pid>/cgroup`, cgroup v2 path per fixture — unit
  tests each primary-with-fallback arm and its log line.
- **Matrix invariants:** helper `device: cpu` · no VRAM ledger charge,
  sets keep co-residency (`+auto:orphans` semantics unaffected since the
  expressions are unchanged — llama-swap doc confirms "a model in no set can
  only run alone").
- **Measurement:** `-ngl 0` shape produces a distinct shape stamp and both
  quads; CPU-quads gate the host ledger; device path unchanged (regression).
- **metadata.packing golden** (gpu entry and cpu entry variants).

## SPEC / docs deltas

- `SPEC.md` §VRAM Allocation: sort the role semantics ("residents: one per
  role" (decision 2) + the `device: cpu` host budget summary, mirroring the
  shape of the VRAM two-tier section; refusal/over-budget wording shared.
- `docs/backends/llama-server.md`: `-ngl 0` note now states the budget rule
  off the host pool, not "CPU-resident are free."
- `templates/models_AGENTS.md`: `device: cpu` sidecar example.
- `profiles.yaml.example`: `hardware:` host-pool keys with the resolution
  table as comments.

## Live gate (fast-iterate, packer tests first)

1. Unit suite green (`check_trace`, `check_mutation`).
2. Repack → `/opt/llama/config.yaml`: helper cmds carry `-ngl 0`/`backend:
   cpu`, matrix log lines show `charge 0` for them, VRAM ledger residents
   unchanged, host ledger lines list chosen pool + charges (sysctl path
   named).
3. llama-swap serves emb/rerank CPU (one embed POST, timings into the
   sidecar, AGENTS.md pace — one model per run, log processed before the next).

## Out of scope (followers / roadmap)

- `gpu-split-ladder`: `moe-cpu` rungs, split spill-aware rejection, the
  `+` mem-tag decision, `metadata.packing.split` (the emission site is
  landable here, the split *value* is not).
- `layers N` (`-ngl N`) rungs; resident-partial-offload
  (`docs/plans/resident-partial-offload.md`); measured-host-RAM graduation
  beyond the minimal keys here; alias rework (Option A — pending
  llama-swap alias-routing probe); vLLM host offload; unified-memory pools
  (GB10) — §5 rule stands no code change.
