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

`_run_measure_server` (`vram.py:663`) hardcodes `-ngl 999`; `_buffer_walk`
(`vram.py:193`) skips the Host row. Changes:

1. Parameterize the `-ngl` flag by serving device: a `device: cpu` model
   measures with `-ngl 0` **in a new shape-keyed run** (shape stamp carries
   it, per `vram.py:277-336` — flags that shape compute/RS belong in the
   shape key, and `-ngl 0` is one).
2. `_buffer_walk` returns **host rows** (host KV/RS/weights/compute) beside
   device sums — the same parse, one more output group. This is the §3
   "measured host RAM" landing spot; branch 2 (split rungs) reads the same
   host rows for its spill/host-half quads and reuses the split-aware
   relaxation this must carry for `device: cpu` (a full-offload run's host
   rows are *the* truth, not a spill defect — rejection is currently keyed
   on device spills only and must stay exactly that way for `-ngl 999`;
   host rows are simply reported).
3. `FitParams` gains host-side keys (own `derived:` entries, shape-stamped,
   same persisted cache as device quads). Nothing second-guesses the
   measurement: tag/report rules per `memory-allocation-tags.md` v1
   contract apply verbatim.

`llama-fit-params` acceptance of extra flags (`llama_args` passthrough,
`vram.py:565`) is already sufficient for both shapes — no new tool.

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
