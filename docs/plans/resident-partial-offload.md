# Roadmap note: partial offload of resident helper weights to host RAM

Status: **proposal — not scheduled**, recorded 2026-10-22. Related:
`docs/plans/memory-allocation-tags.md` §1 (split representation — the `+`
tag grammar), §3 (measured host RAM), §4 (MoE partial offload — same decider
shape), `docs/plans/opportunistic-coload.md` §2b (the *context*-squeeze
counterpart, which already ships a rung of this idea).

## Idea

Residents (embed, rerank, STT, TTS) sit in the VRAM budget as fixed overhead.
Today the only "shrink a resident" axis is context (`opportunistic-coload`
§2b squeeze: emb/rnk yield KV to chat). Weight *placement* is a second,
orthogonal axis: a resident that would take 3 GB on GPU may take 1 GB on GPU
+ 2 GB host (cpu-moe offload of a 0.6–1.2B embedding model, or simply the
quantized variant swap), freeing the difference back to the chat-context
solve.

Example from the operator (2026-10-22): the `rag` set reserves 1843 MiB while
`voice`/`talk` need less — handing the freed VRAM to the chat-context solve
buys `freed_mib / kv_per_token` tokens per slot; magnitude depends on the
model's own `kv_per_token` (see `docs/plans/estimation-functions-facts.md`),
so no fixed number is claimed here.

## Constraints / rules that gate it

- **Per-role singleton**: one co-resident per role (STT, embed, rerank, TTS).
  Multiple sized variants of the same role must never co-reside; the planner
  picks one per set. A shrink decision applies to the *winner*, not a twin.
- **Chat never shrinks to pay for residents** — the co-load floor
  (`matrix.min_chat_ctx`, tools 128K) still binds; the offload only
  *raises* what's achievable.
- **Voluntary, per-variant**: like `device: cpu`, weight-placement-lean
  residents appear only on models the operator tags (never solver-default),
  while the *choice between variants* stays solver-made via sacrifice-style
  category vocabulary.

## Interlocks

- Needs the **host-pool ledger** (branch `device-cpu-helpers` *before* this —
  a partially-offloaded resident draws host bytes from the same pool the
  fully-CPU residents charge).
- Needs **split placement machinery** (the `gpu-split-ladder` rung decider —
  cpu-moe emission + measured host/device split quad). This note is that
  machinery a second consumer: same `+` mem-tag, same
  `metadata.packing.vram_mib/ram_mib` advertising, one new decision surface
  (per-resident placement) instead of per-chat-category rungs.
- §3 (measured host RAM as a persisted constant, not a residue) is the
  accuracy prerequisite for the accounting to be trusted at these small
  magnitudes; today host bytes are a file-size residue, good enough for
  0.5–1.2 GB residents but should be measured once.

## Sequencing suggestion

`device-cpu-helpers` (host pool) → `gpu-split-ladder` (split machinery) →
this (apply split machinery to resident selection + the one-per-role
placement rule). Not modeled at visit-solve time; realized at repack, as
with everything else.
