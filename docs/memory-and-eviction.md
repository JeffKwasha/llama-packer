# Memory allocation and eviction

How llama-packer turns a GPU and a host into budgets, how it spends them on
context and slots, and how llama-swap decides what to unload. The goal is to
use the hardware you have — not to leave VRAM idle out of caution, and not to
advertise co-residency the pool cannot hold.

Two budgets, one rule:

- **VRAM** (`default` pool) is for GPU servings.
- **Host RAM** (`ram` pool) is for `device: cpu` servings.
- An entry is admitted when the summed `capacities` of the resident set fit
  **every** pool. A role with `live: 1` also charges a size-1 counting pool,
  so one serving of that role is live.

Everything below is pack-time arithmetic. llama-swap enforces the emitted
numbers at request time and never re-solves a context.

## The GPU budget

```
reserve    = 1024 + max(1024, hardware.baseline_mb)
global spare = defaults.spare, else --spare, else 0
default pool = vram_total - reserve - global spare
```

- `vram_total` comes from `--vram` > `hardware.vram` > detection.
- `reserve` is the OS plus driver/compositor/scanout. `baseline_mb` replaces
  the fixed video floor for a known always-on load.
- `spare` is headroom for programs that come and go. `defaults.spare` wins
  over `--spare` when both are set.
- On a unified-memory host, `reserve` is `hardware.unified_system_mb`
  (default 8192) instead of the two-term formula.

The emitted `default` pool is that number. It is deliberately close to the
card: allocations are meant to fill it.

## The solve: context vs slots

For each entry the packer finds the best `(ctx, slots)` pair that fits what
is left after residents:

```
score = sqrt(ctx / task_min_context) * slots ** parallel_power
```

- Context below the task floor is rejected outright; `parallel_power` only
  arbitrates between candidates that already clear it.
- `auto_parallel: false` disables the value function fleet-wide; a profile
  or sidecar `parallel:` pins a specific slot count.
- Embeddings and rerank always serve one slot and the context their task
  declares (`max_context`); clients chunk to fit.

Lower `parallel_power` protects context; higher buys concurrency. The
neutral point depends on how expensive a model's slots are, not on a
constant — see SPEC §What `parallel_power` actually controls.

## Co-residency: floors first, then shares

Every serving has a **floor**: the memory it needs at its task's
`min_context` and `min_parallel`, derived through the model's measured memory
law. Floors come out of the pool first, along with residents. What is left is
**slack**, and roles divide it by `share`:

```
slack[pool] = pool − reserve − spare − Σ residents[pool] − Σ floors[pool]
```

This is the mechanism behind "chat was solved to 30293 and a 1252 MiB voice
model could not fit in a 30320 pool". Sizing chat against
`30320 − resident` makes the pair fit while the router still sees 30320.

A role with `share: 100%` leaves no slack for anyone else, so it runs alone.
That replaces the group router's `exclusive`.

Ways to reserve:

| need | knob |
|---|---|
| Keep a sidecar the packer models | give it `ttl: 0`; idle unload is off (capacity pressure can still evict it) |
| Keep a sidecar the packer does not model | `pools.default.reserve_extra: 4G` (chat only) |
| Name an explicit claimant | `pools.default.pins: {image: 6G}` |
| Shrink the solve for one pool | `pools.<pool>.spare: 2G` (does not shrink the emitted `default`) |
| Shrink the emitted GPU pool | `defaults.spare` / `--spare` / `hardware.vram` |
| Keep host RAM free | `pools.ram.spare: 2G` (shrinks the emitted `ram` pool) |

A role placed on the host (`device: cpu`) is the self-describing lever: the
packer charges exactly the variant it will serve against the `ram` pool.
`reserve_extra` is the fallback when the sidecar is outside the model.

## Emission: capacities and margin

Every entry declares the pools it charges:

- GPU entry: `default` = measured VRAM at its served context, `ram` = its
  host footprint, plus `1` in its role pool.
- CPU entry: `ram` only, plus `1` in its role pool.
- Unmeasured entry: the whole `default` pool, so it can only run alone.

Charges carry the solve's measurement margin (`hardware.memory_margin`,
default 4%). A capacity without the margin can sum to just over the pool and
turn a planned co-load into an eviction.

## Eviction

When a request needs a pool that is full, llama-swap unloads the resident
with the **lowest** eviction cost until the set fits:

```
evict_cost = model_size / drive_mbps
           + (ctx * slots / pp_tps, chat only)
```

- First term: time to re-read the weights. Second: prefill lost when chat
  context is discarded.
- `capacity.pp_tps` is the fleet prompt rate; a sidecar `pp_tps:` overrides
  it. `capacity.drive_mbps` is the disk rate (detected if omitted).
- Stateless helpers keep no KV, so prefill is zero for them.
- The cost orders evictions; it never reserves or preloads anything.

Two constraints act before cost:

1. **Role pools are size 1.** Requesting another model of the same role
   evicts the current one regardless of cost. That is how "one model per
   role" becomes a capacity rule.
2. **A set must fit every pool.** No eviction order can admit a combination
   whose capacities exceed a pool.

TTL is independent: `ttl: 0` never unloads; otherwise the smallest declared
lifetime wins; an undeclared CPU-resident serving gets 3600 s; a GPU entry
with no TTL inherits `globalTTL` (only emitted with `--idle-unload`).

## Worked examples

### 24 GiB card: chat + RAG + voice, no guessing

```yaml
defaults:
  spare: 1G                 # default pool = 24G - 2G - 1G = 21G
matrix:
  categories:
    emb: {role: embeddings}
    rnk: {role: rerank}
    tts: {role: t2s}
pools:
  default:
    reserve_extra: 3G       # room for the voice model beside chat
```

Chat is solved against `21G - 3G - RAG`, so a 1.2–3 GiB voice model fits
when requested; the router still sees the full 21G pool.

### 48 GiB card: independent groups with custom roles

```yaml
roles:
  cat: {dir: cats, share: 40%, live: 1}
  dog: {dir: dogs, share: 40%, live: 1}
tasks:
  cat: {endpoint: openai.chat.completions}
  dog: {endpoint: openai.chat.completions}
```

`cat` and `dog` each get a share of slack plus a size-1 counting pool, so one
serving of each may be resident together. Within a role, requesting another
model evicts the current one. Models belong to the role that owns their
directory; a sidecar `role:` overrides it.

### A model that must run alone

Give its role `share: 100%`: no slack is left, so nothing else can be
resident. An unmeasured entry also reserves the whole `default` pool, and
therefore evicts everything else.

### Two GPUs

A sidecar `device: 1` pins the serving to the second GPU and maps it to the
`gpu1` ledger. Today the emitted pool set is the aggregate `default`; treat
multi-GPU attribution as a later extension and size the aggregate honestly.
The target model is a cost tuple per pool (`gpu0`, `gpu1`, `ram`), so a
tensor-split serving charges every card it touches.

## Verify

```sh
llama-packer --check --profiles profiles.yaml     # syntax, roles, keys
llama-packer --dry-run > /tmp/config.yaml         # the plan and its warnings
llama-swap -config /tmp/config.yaml -validate
extras/check_sidecar_budget.py /tmp/config.yaml --require t2s,embeddings,rerank
```

The checker applies the router's own rule to the emitted numbers: it reports
the headroom beside each chat and whether the required set fits. Use it when
you change a reservation; use `--dry-run` when you change the fleet.
