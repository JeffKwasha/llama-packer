# Allocation

How llama-packer turns pools, roles and sets into entry sizes. Roles are
arbitrary labels; the examples use `A`, `B`, `C` and `foo`, `bar`.

## Pools

A pool has a physical amount and a reserve:

```
available(pool) = physical(pool) − reserve(pool)
```

- `reserve` is memory not available to allocation: one term per pool, never
  subtracted a second time. It is measured — the system and driver floor
  (about 2 GiB) on a GPU pool, host pressure on a CPU pool.
- The operator adds to the measured reserve with `spare`, one value per pool.
  There is no `reserve:` key: replacing the measured floor is unsafe, and
  adding to it is what `spare` already means.
- `reserve_extra` and `pins` are retired and error. Reserving memory for a
  named role or model is future work and must name the device tuple it
  reserves.
- Pools are independent. Settings and CLI arguments name the pool whose
  spare they change (`gpu0`, `cpu`, …); today only the GPU spare can be set.
- A serving's cost is a tuple over pools, `cpu` first:

  ```python
  Memory = namedtuple("Memory", ["cpu", "gpu0", "gpu1"])
  ```

  `cpu` is emitted as `ram` today, `gpu0` as `default`. Per-card pools and
  unified memory (several devices sharing one field) are future work; the
  tuple already expresses them.
- An entry charges every field its serving touches. A pool absent from an
  entry's capacities is zero to that entry.

## Floor

A model's floor is the memory it needs at its task's `min_context` and
`min_parallel`, through the measured memory law:

```
floor = weights + companions + compute
        + kv_per_token × min_context × min_parallel
        + slot × min_parallel
```

as a `Memory` tuple. A model with several tasks has one floor per task.

**A role's floor is the maximum floor of all the models in the role**, per
pool. A role runs one serving at a time, so its budget is the largest serving
it may have to place.

A set's floor is computed by its expression, component-wise:

```
floor(A & B)        = floor(A) + floor(B)
floor(A | B)        = max(floor(A), floor(B))
floor(A & (B | C))  = floor(A) + max(floor(B), floor(C))
floor(+set)         = floor(set)
```

If any component of `floor(set)` exceeds `available`, the set cannot be
placed. The simple rule: **drop the whole set and log an error**. The older
rule drops the role that brings the pool back under `available`; the simple
rule is preferred until the role-drop order is specified.

## Set members

A set is an expression of role labels and model ids using `&`, `|`, `()`,
`+set`; `+set` inlines a previously defined set.

- A leaf on its own is a valid set: `A` contains only role `A`, which is how
  a role is confined to running alone.
- A role label expands to its entries. A role with no entries is removed from
  the expression; an all-empty set is dropped with a warning.
- A literal model id that emitted no entry is a dangling leaf; it is pruned
  with a warning.
- A set contains at most one serving per role.

## The auction

Every set starts with one budget. Its floors come out first; what remains is
the set's auction pool:

```
auction(set) = available − floor(set)          # component-wise
```

Each role in the set carries a request weight and an accept flag from its
`share:`:

| `share:` | request | accept surplus |
|---|---|---|
| `min` | 0 | no |
| `0` | 0 | yes |
| `27%` | 27 | yes |
| `fill` | 100 | yes |

Weights are relative, not percentages of a whole: 60 and 90 are 60:90. Any
positive value is valid, including above 100 — `share: 123` is 123 parts of
the set's auction.

Requests are normalised inside each set, never capped at an absolute
percentage:

```
request(role, set) = weight(role) / Σ weights in set × auction(set)
offer(role, set)   = floor(role) + request(role, set)
```

A role can appear in several sets. **No role accepts more than the minimum
offer from the sets it is in**, per pool:

```
allocation(role) = min over sets containing role of offer(role, set)
```

When one set gives a role less than another set requested for it, that set
has memory left. Recompute that set: fix the role at its allocation,
subtract it, and renormalise the rest of the auction among the remaining
requesters by the same weights. Repeat until no offer changes.

Example — set `A & B` with auction 15 GiB, `A` at 60 and `B` at 90:

```
A requests 60/150 × 15 = 6
B requests 90/150 × 15 = 9
```

`A`'s other set offers only 5, so `A` accepts 5. Recompute: the remaining
auction is 10 and `B` is the only remaining requester, so `B` is offered 10
and accepts it.

`min` stays at its floor. `0` requests nothing but accepts memory left after
the requesters are placed; when several accepting roles share one set's
leftover, it is split equally among them. `fill` is `100%` by another name.

Each model then solves its context and parallel inside its role's allocation
in its pool. Each set logs its unallocated memory at INFO.

The normalised, iterative rule is the recorded design. A system of equations
would do better; this is the algorithm to implement first.

## One serving per role

A role runs one serving at a time. A request for another model of the role
evicts the current one.

- Every role is a pool in the emitted capacity pools with value 1, and every
  entry charges exactly 1 to its role's pool. A second entry of that role
  cannot be admitted. This is the whole mechanism; there is no per-role
  setting.
- In matrix mode a role label expands to one of its entries, so a set cannot
  hold two servings of one role.

## Routers

| input | router |
|---|---|
| `matrix.sets:` non-empty | matrix: emit the declared sets |
| `matrix.sets:` absent or empty | capacity |

Capacity uses the same algebra with sets generated from the role floors. The
emitted config is pools plus per-entry capacities; the generated sets are
reported in a comment. **Once generated, an ephemeral set is identical to a
declared one**: same floor, same auction, same reallocation. Only its origin
differs. The generation rule is not yet recorded and must not be guessed.

Sets never preload. On a request for X, llama-swap picks a set containing X,
evicts what is outside it, and starts only X. Subsets are permitted.

## TTL

- Top-level `ttl`, default 3600.
- A role's `ttl` overrides the top-level default.
- A task's or sidecar's `ttl` overrides the role.
- `ttl: 0` disables idle unload. TTL never reserves memory.

There is no idle-unload CLI flag.

## Unmeasured models

An unmeasured model is charged a minimum estimate — file size plus conservative
KV/slot assumptions — and flagged unestimated. It may OOM. The fix is a real
measurement or operator-supplied `derived:` terms; the packer does not reserve
the whole pool on a guess.

## Task contract

- A task's `min_context` overrides the top-level `min_context` default.
- `min_context > max_context` is an error: the run stops and no config is
  written.
- Declared tasks on one model that do not share one endpoint are an error.

## Fallback model_id

Every model gets a bare `<stem>` entry in its role, priced at the model's
floor. A model_id that is in no role is an error and emits nothing.

## Gone

These are not part of allocation:

- The fixed helper pair and any notion of a model that is always loaded. An
  embeddings or rerank model is a role like any other and is kept by `ttl` if
  the operator wants it kept.
- The old per-role count setting. One serving per role is the only mode.
- **`matrix.categories`**, `fixed_categories`, `category_ctx`. Their meaning
  must be recovered and either expressed with roles/sets or dropped; the
  names carry no meaning.
- **`routing:`** as an input key, `matrix.evict_costs`.
