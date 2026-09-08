# Plan: Auto-parallel — spend leftover VRAM on concurrent chat slots

Status: **implemented on branch `auto-parallel-requests`** (2026-09-07):
value-function solve (`parallel_value`), `resolve_min_ctx` cascade,
`matrix:` knobs (`auto_parallel`, `auto_parallel_max`, `parallel_power`),
shared `PoolLedger` + `pools:` config + size buckets, vLLM `--max-num-seqs`
emission. Staged for later: per-GPU hardware attribution (ledger is keyed by
pool id; detection still returns one total), exclusive image sets (admission
unchanged — the floor gate already prevents starvation), footprint
clustering for bucket boundaries.
Related: `SPEC.md` Matrix Context Solving + Main-Chat Context Determination, `docs/plans/opportunistic-coload.md`, `llama_packer/{vram,writer,profiles}.py`, `llama_packer/backends/vllm.py`.

## Problem

The matrix solve sizes each chat model for **one** conversation: weights +
compute + `ctx × factor` must fit, and whatever VRAM is left over after a full
context window sits idle while that model is loaded. A 12B on a 24 GB card may
have ~6 GB of headroom at its solved context — enough for several more KV
slots, but today `--parallel` stays at 1 (or the operator-pinned value) and
simultaneous chats queue serially.

## How server parallelism works (the part we'd spend the VRAM on)

**llama.cpp:** `--parallel N` creates N independent KV-cache slots. All slots
**share** the model weights and compute buffers; only the KV cache is per-slot,
and it is **pre-allocated at load time** for all N slots (`N × ctx × f_slot`).
So simultaneous chats = more slots, not a bigger context — each chat still gets
the full solved `ctx`. Unused slots reserve VRAM that nothing else can use
while the model is loaded (fine within a chat set — chat models evict each
other — but it must not eat the fixed co-load budget). `--cont-batching`
(default on) means partially-used slots still batch efficiently; idle slots
cost VRAM only, not compute.

**vLLM is the easy case, and in scope.** vLLM runs one shared paged KV pool —
no per-slot pre-allocation — and caps simultaneous chats with
`--max-num-seqs`, which the code already emits from the parallel number
(`backends/vllm.py`). So the same solve applies to both backends: emit the
solved slot count as `--parallel` for llama.cpp and `--max-num-seqs` for
vLLM, and size the vLLM pool via `gpu-memory-utilization` exactly as today.
Nothing about vLLM needs slot-reservation math.

## Design

**Precedence (operator intent wins, as everywhere):** sidecar `parallel:` or a
profile-pinned `parallel` is final. Auto-parallel only decides for models with
no pin.

**Value-function solve: one knob.** The goal is always to use memory
efficiently, and basic economics says marginal value falls with count — on
both axes. So score every feasible `(context, slots)` pair with:

```
score = (context ÷ floor)^0.5 × slots^B
```

In plain words:

- Divide the candidate context by that model's floor (see `min_context`
  cascade below), so context becomes a plain ratio: 1.0 = sitting on the
  floor, 2.0 = double the floor.
- Take the square root of that ratio. Square root means each doubling is
  worth less than the last — the first doubling past the floor adds ~0.4
  points, the next adds ~0.25, and so on. No knob needed; diminishing returns
  are baked in.
- Multiply by the slot count raised to **B, the "parallel power"** — the one
  and only knob (default 0.75). B = 1 means every simultaneous chat is worth
  full value; lower B means each extra chat counts for less. B below 1 keeps
  diminishing returns on this axis too.
- The score's absolute value means nothing; it only ranks candidates. Try
  every feasible pair on the existing 8k context grid with
  `p ∈ 1…auto_parallel_max`, keep the winner.

Worked examples (floor 128k, B = 0.75):

| Candidate | Context ratio | Score | Verdict |
|---|---|---|---|
| 2 × 128k | 1.0 | 1.0 × 2^0.75 = **1.68** | wins — two responsive chats beat one stretched window |
| 1 × 256k | 2.0 | 1.41 × 1 = 1.41 | |
| 2 × 128k vs 1 × 200k | 1.0 / 1.56 | 1.68 vs 1.25 | the 1→2 bump happens even though context is cut by less than half |
| 2 × 128k vs 1 × 512k | 1.0 / 4.0 | 1.68 vs 2.0 | stretch wins — the math simply decides, per model and per spare-VRAM amount |

That last row is the whole point: the formula exists for the rare case where
cutting context below half buys a slot (or a big stretch beats a slot), and
stays out of the way otherwise.

**Setting the inputs (written for simple AI agents):** `min_context` = "the
smallest context, in tokens, at which this model still does useful agentic
work (multi-step tool-call loops, not single replies)." Rule of thumb: tool
callers get 131072; non-tool models get half their max context. `parallel
power` B = "how much is a second simultaneous chat worth relative to the
first" — 1.0 = full value, 0.5 = half, 0.75 = the default middle. Lower it
when one fat context matters more than several chats; raise it when serving
many users at once.

**`min_context` cascade (first match wins).** The per-model floor is resolved
by one shared helper, `resolve_min_ctx(model)`, used by all three consumers
(the vision keep/drop pre-pass, the chat context determination, and this
solve) so the floor can never disagree with itself:

| # | Condition | floor |
|---|---|---|
| 1 | sidecar declares `min_context:` | that value |
| 2 | ctx is pinned (`context_length:` / `--max-context`) | the pin (a fixed ctx *is* its own floor; the search runs over `p` only) |
| 3 | model has tool calling (`tools` capability) | 131072 (`_MIN_AGENTIC_CTX` — "minimum useful *agentic* context") |
| 4 | otherwise | half the model's max context (0.5 × max is principled, not magic: two slots are only ever considered when nearly two full windows fit) |

Conflict check: if **both** an explicit `min_context:` and a ctx pin are
present and `pin < min_context`, warn — the pin is honoured (it is a hard
fix) and the declared floor is reported as unmet. The existing global
`--min-context` flag keeps its job as the default for every model: it
overrides cascade rows 3–4 but never an explicit sidecar key or a hard pin.
Capability thresholds (`tools_min_ctx` demotion) stay **out of the score** —
they remain a post-hoc metadata label on whatever ctx wins, never a reason to
inflate it; they only *feed* the cascade (row 3).

**Pinned ctx fixes the slot size, it does not disable slots.** If the
operator pins ctx = X and `2×X` fits, every chat still gets the full
requested X — extra slots claim VRAM that would sit idle anyway (chat models
evict each other; co-residents are already subtracted). The search just runs
over `p` alone.

## Memory pools — size buckets, not model types

Every model's footprint = weights + compute buffers + context-KV at its
operating context (+ vision adapter where loaded). That combined number
already exists in code (`effective_static`, variant-aware via
`include_mmproj`) — the plan reuses it as a single input rather than
re-deriving its parts. What the plan adds is **bucketing footprints by size**
and building matrix sets from size compatibility instead of model type:

- **Tiny (always co-resident):** speech-to-text, text-to-speech, embeddings,
  rerank — around a GB or less each. Speaking instead of typing while chatting
  is enormously useful and costs under 0.5 GB for s2t/t2s, so these join every
  set on their GPU for free. Never gate a chat slot on them; subtract them
  first and forget them.
- **Huge (exclusive sets):** image diffusion — tens of GB, roughly chat-sized
  or larger. An image model next to a main chat means one of them starves, so
  huge models get their **own** sets and never share with chat. Do not lump
  them with speech; the size difference is the whole ballgame.
- **Chat (evictive within a set):** one loaded at a time, so each chat model's
  leftover is its own to spend on slots — never pooled across chat models.

Bucket boundaries can start as explicit role-based defaults (the three rows
above) and graduate to clustering actual measured footprints later; the
mechanism doesn't care where the lines come from.

Per-GPU ledger, shared three ways:

```
pool_free(G) = vram(G) − system/driver reserve − display baseline − spare(G)
               − tiny residents on G
kv_budget_i  = pool_free(device_i) − combined_footprint(variant_i)
```

The matrix solve, the vision keep/drop pre-pass, and auto-parallel all read
this **one shared ledger** instead of each doing its own subtraction — one
household budget, not three people guessing the balance. Consequences: a chat
model on GPU 1 is never starved by an image model on GPU 0; tiny residents
can't be OOMed by chat slots (the failure mode
`opportunistic-coload.md` guards against). Open: unpinned models
(tensor-split across all visible GPUs) — v1 charges them to the sum of
per-GPU free pools (tensor split balances load, so the pool is roughly
additive); validate on a real 2-GPU box.

### Configurable pools (operator escape hatch)

The derived math above stays the default, but operators need a declarative
override — e.g. "GPU 0 also runs a game that eats 8 GB", or "trust my
measurement for the sd server more than the estimate". Sketch for
`profiles.yaml`:

```yaml
pools:
  gpu0:
    spare: 2G              # per-GPU override of the global spare
    reserve_extra: 8G      # unmodelled resident (VM, game, other tenant)
    pins:                  # explicit reservations; "auto" = derived math
      embeddings: auto
      s2t: auto
      image: 40G           # operator overrides the estimate
```

Semantics: an explicit value replaces the derivation for that claimant; `auto`
keeps it. All three consumers read the same pool object.

**Emission:** the solved slot count as `--parallel` (llama.cpp) /
`--max-num-seqs` (vLLM) in the entry's `cmd`, with per-slot `-c ctx`;
`metadata.ctx_size: ctx` as today (`parallel` stays cmd-level, out of
metadata, per the existing `test_cache_type_and_parallel_not_in_metadata`
rule). The vision
image floor already scales with parallel (`p × image_max_tokens`, per-slot ctx
unchanged), so no change there beyond using the solved `p`.

**Knobs (new keys in the existing `matrix:` section):**

| key | default | meaning |
|---|---|---|
| `auto_parallel` | `true` | enable the solve step (`false` disables fleet-wide; a sidecar `parallel:` pin opts one model out) |
| `auto_parallel_max` | `8` | hard cap on slots |
| `parallel_power` | `0.75` | B in the score: what each extra simultaneous chat is worth |

`estimate_headroom` pads the analytic compute estimate per candidate as before.

## Answered questions (previously "open")

- **`concurrencyLimit`.** If declared, it caps queued+active requests: `p = min(p*, concurrencyLimit)`. One line, no discussion.
- **Per-slot vs aggregate `ctx_factor`.** *Answered 2026-09-07* — see the
  findings addendum below: the law is affine, not per-slot proportional.
- **Compute-buffer growth with p.** *Answered 2026-09-07* — compute is
  ~constant across p (max over the p=1/p=2 pair, ±32 MiB Vulkan alignment
  quanta; one +128 MiB f16 spike covered by the max).

## Findings: the affine VRAM law (2026-09-07, SEAL record)

Measured with the `--probe-memory` probe across qwen3.8-27B (dense,
arch `qwen35`), gemma-4-12B-QAT (dense+SWA+MTP, arch `gemma4`) and
gemma-4-26B-A4B (MoE+SWA), at p ∈ {1,2,4,8}, C ∈ {64k,128k,256k},
q8_0/f16/bf16/q4_0 (raw data: `parallel.log` in the repo root at the time
of measurement):

```
VRAM(C, p) = model_mib + compute_mib + c*C + p*D
```

- `C` is the **total** KV pool: `-c` in llama.cpp terms, byte-identical to
  `--kv-unified-per-slot X -np p` (verified exactly on both arches).
- Residuals of the two-point fit: ≤ 3 MiB (< 0.1 %) on every measured cell.
- `D` = per-slot cost; SWA archs spend it on ring buffers (gemma4-12B:
  255 MiB q8_0, scaling exactly with KV byte ratios ×1.882 f16 / ×0.529
  q4_0), non-SWA archs on fixed overhead (qwen35: 150 MiB at *every*
  precision). **`D` is therefore not derivable across cache types** —
  blocks are measured per `cache_type` (~1 s per model).
- `c` scales between precisions by exactly the `_KV_CACHE_BYTES` byte ratio.
- The v1 assumption `ctx_factor(p) = p × ctx_factor(1)` over-reserved p=8
  by ~19 GB on qwen3.8-27B (23 GB charged vs ~10 GB real) — auto-parallel
  was choosing p=1 almost everywhere because parallel looked unaffordable.
  Emitted configs are expected to change (more slots / larger contexts);
  that is the fix working, not a regression.
- mmproj: `llama-fit-params` SIGABRTs on projection GGUFs (exit −6, both
  witnesses); the analytic weight + fixed-compute model stands, and
  projection cost is provably independent of context and slots (no KV).
  Live-VRAM validation of the 150 MiB buffer constant was left as an
  optional follow-up (touches the running llama-swap).
- Solve-side safety: `hardware.memory_margin` (default 0.01) inflates every
  measured term; the solve is exact closed-form
  `X = (remaining − p·D)/(c·p)`, so the sweep costs zero binary calls
  beyond the one (p=1, p=2) pair per (model, cache_type).

Fleet validation (`--probe-memory`, all 11 GGUF families): PASS. Constants
of record (`c` MiB/token, `D` MiB/slot, q8_0):

| arch | witness | c | D | max err |
|------|---------|---|---|---------|
| deepseek2 | GLM47-Flash | 0.027428 | 0 | 0.000% |
| gemma3 | gemma-3-12b-it-heretic-v2 | 0.033203 | 255 | 0.000% |
| gemma4 | gemma-4-26B-A4B | 0.010376 | 159 | 0.075% |
| laguna | Laguna-XS-2.1 | 0.020748 | 64 | 0.017% |
| lfm2 | LFM2.5-VL-1.6B | 0.006227 | 0 | 0.87% (16 MiB abs — sub-MiB per-slot cost on a tiny KV; verdict uses an absolute floor) |
| llama | Nous-Hermes-2-Vision | 0.066406 | 0 | 0.000% |
| muse-glimmer | Glimmer-30B | 0.006737 | 52 | 0.077% |
| qwen3 | qwen3-4B | 0.074707 | 0 | 0.000% |
| qwen35 | Qwen3.8-27B | 0.033199 | 150 | 0.020% |
| qwen35moe | Qwen3.6-35B-A3B | 0.010372 | 63 | 0.031% |
| qwen3vl | Qwen3-VL-8B | 0.074707 | 0 | 0.000% |

## Non-goals (v1)

- No runtime slot scaling — slots are fixed at load; changing p is a repack.
- No cross-model slot sharing (one model's leftover never funds another's
  slots; chat models evict, they don't co-run, within a set).
