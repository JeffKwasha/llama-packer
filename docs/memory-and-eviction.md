# Memory allocation and eviction

How llama-packer turns hardware into pools, spends them on context and
slots, and how llama-swap decides what to unload. Nothing is assumed loaded:
a set is a permission, and its numbers make the permitted combination fit.

## Pools

```
available(pool) = physical(pool) − (measured reserve + spare)
```

- The measured reserve is the OS plus driver/compositor/scanout floor (about
  2 GiB). `hardware.baseline_mb` replaces the fixed video floor for a known
  load; on a unified-memory host the reserve is
  `hardware.unified_system_mb`.
- `spare` is the operator's addition, one value per pool
  (`pools.<id>.spare`). It is the only reserve knob. There is no `reserve:`
  key: replacing the measured floor is unsafe, and adding to it is what
  `spare` means.
- `default` is the GPU pool; `ram` is the host pool. A serving's cost is a
  tuple built from the host's pools, `cpu` first, then one field per discrete
  GPU or one shared field for unified memory, plus fake limits such as a
  role pool of 1.

The emitted pool is `available`. Allocations are meant to fill it.

## The solve: context vs slots

For each entry the packer finds the best `(ctx, slots)` pair inside its
role's allocation in its pool:

```
score = sqrt(ctx / task_min_context) * slots ** parallel_power
```

- Context below the task floor is rejected. `min_context` comes from the
  task, else the design-derived cascade: the model's explicit
  `min_context`, a pinned `--max-context`, `tools_min_ctx` for tool callers,
  then half the design context. An explicitly set top-level `min_context`
  overrides the last two.
- `auto_parallel: false` disables the value function; `parallel:` pins a
  slot count.
- An entry whose endpoint keeps no KV cache serves one slot at the task's
  `max_context`; clients chunk.

`parallel_power` arbitrates only among candidates that clear the floor.

## Sets: floors first, then the auction

A serving's floor is the memory it needs at its task's `min_context` and
`min_parallel`, through the measured memory law. The context is priced at the
first emitted step (a multiple of 4096) at or above `min_context`. A role's
floor is the maximum floor of all models in the role.

A set's floor follows its expression, component-wise:

```
floor(A & B)       = floor(A) + floor(B)
floor(A | B)       = max(floor(A), floor(B))
floor(A & (B | C)) = floor(A) + max(floor(B), floor(C))
```

If a component of `floor(set)` exceeds `available`, the set is dropped and
logged as an error.

The rest of the set is the auction:

```
auction(set) = available − floor(set)          # one device pool
```

The auction runs on the pool the model's device names. On every other pool
the role is `min`: it pays its static footprint and accepts nothing above
it. A GPU model's host floor is at least 128 MiB. A role whose memory does
not grow with context, such as a fixed RAG window or a whisper backend, uses
`share: min`.

Each member requests a share of it, by weight:

| `share:` | weight | accepts surplus |
|---|---|---|
| `min` | 0 | no |
| `0` | 0 | yes |
| `27` | 27 (relative, uncapped) | yes |
| `27%` | 27 (a 27-point claim; warned if claims sum above 100) | yes |
| `fill` | residual 100-point space, split among fills | yes |

```
weight(fill, set)  = max(0, 100 − Σ other weights in set) / n_fills
request(role, set) = weight / Σ weights in set × auction(set)
offer(role, set)   = floor + request
allocation(role)   = min over its sets of offer
```

Every claimant is normalised by the sum of its weight. A plain number is a
relative weight; `N%` is a claim of N points in the same space. `fill` takes
whatever points are left, claims last, and receives surplus nobody else
accepts.

Each set is priced on its own. A role's limit is the minimum across the sets
it belongs to, so adding a set can only lower it. An offer in one set does
not change its offer in another; each set logs its own unallocated
headroom.

A model then solves inside its role's allocation. One role runs one serving
at a time: in capacity mode every role is a pool of value 1 and every entry
charges 1; in matrix mode a role label expands to one of its entries.

## Live VRAM attribution

`extras/gpu_vram_monitor.py` samples the global VRAM counter and, per GPU fd,
the kernel's DRM fdinfo (`drm-resident-vram`, `drm-resident-gtt`, deduped by
`drm-client-id`). fdinfo is rootless but uid-scoped: it sees servers the
caller owns, not the `lemonade`-owned llama-swap children. Each sample's
`attribution` block names the source and whether it covers the GPU's users;
check `attribution.complete` before treating an empty `processes` list as an
idle GPU. debugfs is a root-only fallback; `nvtop -s` shows the same fdinfo
data as a snapshot. llama-swap's `/logs/stream/upstream` exposes child logs,
but they carry no `-lv 5` buffers unless the serving command asks for them.

## Emission: capacities and margin

Every entry declares the pools it charges:

- GPU entry: `default` = measured VRAM at its served context, `ram` = its
  static host footprint, plus `1` in its role pool.
- CPU entry: `ram` only, plus `1` in its role pool.

Charges carry the solve's measurement margin (`hardware.memory_margin`,
default 4%). A capacity without the margin can sum to just over the pool and
turn a planned co-load into an eviction.

An unmeasured entry is charged a minimum estimate — file size plus
conservative KV/slot assumptions — and flagged unestimated. It may OOM; the
fix is a real measurement or operator `derived:` terms.

## Eviction

When a request needs a pool that is full, llama-swap unloads the loaded
serving with the lowest eviction cost until the request fits:

```
evict_cost = model_size / drive_mbps
           + (ctx * slots / pp_tps, KV-cached endpoints only)
```

- The first term is the time to re-read the weights; the second is the
  prefill lost with the context.
- `capacity.pp_tps` is the fleet prompt rate; a sidecar `pp_tps:` overrides
  it. `capacity.drive_mbps` is the disk rate.
- Stateless endpoints keep no KV, so prefill is zero.

Two constraints act before cost:

1. A role pool is size 1: requesting another model of the role evicts the
   current one.
2. A set must fit every pool. No eviction order admits a combination whose
   capacities exceed a pool.

## TTL

`spare` and `ttl` are the same policy in two dimensions: space and time. Both
exist because the machine has tenants we cannot account for, so both resolve
conflicts conservatively toward them. `spare` keeps room free; `ttl` gives an
idle model back. Neither is a performance knob, and neither is a substitute
for the other.

- Top-level `ttl`, default 3600.
- A role's `ttl` overrides it; a task's or sidecar's `ttl` overrides the role.
- `ttl: 0` disables idle unload. TTL never reserves memory.

## Worked examples

### 24 GiB card: two roles in a set

```yaml
pools:
  default:
    spare: 1G              # available = 24G - 2G - 1G = 21G

roles:
  foo: {dir: foo, share: 60%}
  bar: {dir: bar, share: 40%}

matrix:
  sets:
    pair: "foo & bar"
```

The set floor comes out first; the remaining 21G minus both floors is split
60:40. Requesting a `bar` model while `foo` is loaded keeps both when their
capacities fit; a second `foo` evicts the first.

### 48 GiB card: one role per set

```yaml
roles:
  foo: {dir: foo, share: 40%}
  bar: {dir: bar, share: 40%}

matrix:
  sets:
    foo: "foo"
    bar: "bar"
```

A single-leaf set is how a role is confined to running alone. `foo` may use
the whole auction pool of its set; `bar` does the same in its own.

### A role that must not be evicted

Give its task or sidecar `ttl: 0`. That disables idle unload; capacity
pressure can still evict it.

### Two GPUs

A sidecar `device: 1` pins the serving to the second GPU. The cost tuple
carries one field per pool the host exposes; today the emitted pool set is
the aggregate `default`, so size the aggregate honestly. Per-card emission
and unified memory are future work.

## Verify

```sh
llama-packer --check --profiles profiles.yaml
llama-packer --dry-run > /tmp/config.yaml
llama-swap -config /tmp/config.yaml -validate
```

The checker applies the router's own rule to the emitted numbers.
