# Roadmap

Work that is designed but not scheduled, and defects found but not fixed.
A follower links the plan that carries its design; a defect names the code
it lives in. Nothing here is a commitment — it is where the open items go
when the plan that produced them is pruned.

## Followers (designed, unscheduled)

- **`gpu-split-ladder`** — `moe-cpu` rungs, split spill-aware rejection, the
  `+` mem-tag decision, `metadata.packing.split`. Design lives in
  [resident-partial-offload.md](resident-partial-offload.md) and the split
  notes in [cpu-residency-budget.md](cpu-residency-budget.md); no standalone
  plan doc exists.
- **`layers N` rungs** (`-ngl N`) — partial offload as a ladder dimension.
- **resident-partial-offload** — see
  [resident-partial-offload.md](resident-partial-offload.md).
- **Measured host RAM** beyond the minimal keys in
  [cpu-residency-budget.md](cpu-residency-budget.md) (memory-allocation-tags
  §3).
- **Alias rework** (Option A) — pending the llama-swap alias-routing probe.
- **vLLM host offload.**
- **Unified-memory pools (GB10)** — the §5 rule stands; no code change.

## Open defects

- **`fit-params crashed (exit 1)` on first measurement** of the embed/rerank
  models (`Qwen3-Embedding-0.6B-Q8_0`, `voyage-4-nano-q8_0`,
  `jina-v5-small-retrieval`, `Qwen3-Reranker-0.6B-Q6_K`, …). Pre-existing
  and environmental — the numbers actually used come from the saved sidecar
  `derived:` blocks, so planning is not affected — but the crash hides
  whatever new quads the first measurement would have produced. Diagnose on
  the measurement path (fit-params invocation/flag shape,
  `llama_packer/vram.py`), not the planner.
  **Evidence, 2026-11-02:** reproduced on `llama-b11988` for all four named
  models; the artifacts record `error: invalid argument: --fit-print`.
  Cause: `_detect_matrix` is called with `fit_bin=llama_bin`
  (`__main__.py:1175`), so `vram.fit_params` launches `llama-server`, which
  has `--fit` but not `--fit-print`. The flag belongs to the sibling
  `llama-fit-params` binary, which accepts the same argv and exits 0. The fix
  is the call site (pass `fit_bin`, not `llama_bin`) plus a regression test.
  A failed run now writes
  `~/.cache/llama-packer/fit-params-failures/<ts>-<pid>-<model>-<outcome>.log`
  (argv, model facts, binary identity, full stdout/stderr, exit/signal) and
  indexes it with one `mode: fit` journal row; the warning names the
  artifact and the cause. The next real crash is diagnosable from that file
  alone.
