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

**metadata.packing** (client-visible facts; emission site shared with
branch 2).

17. *every entry carries `metadata.packing`* — decided facts only:
    `device` (`cpu`/`gpu`), `ctx_size`, `parallel`; `vram_mib` when the
    serving is device-resident and estimated, `ram_mib` for the
    host-side bytes when decided, and `kv_per_token` (per-token KV
    bytes at the served cache_type) when the backend measures an affine
    quad. `device: cpu` entries carry no `vram_mib`.

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
