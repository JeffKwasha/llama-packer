# The new-model pipeline

What happens when a model is added to a scanned directory: how llama-packer
turns "a GGUF file appeared" into a serving entry with a context size and a
slot count. Companion read: [architecture.md](architecture.md) (component
ownership), [../SPEC.md](../SPEC.md) (behavioral contract).

## The one-page version

```
file appears
  → discovery       (discover.py / scope.py)   role, sidecar fold, backend, companions
  → file intrinsics (header-only)              arch, trained ctx → derived: file:
  → VRAM estimate   (vram.py fit_params_static) fit-params trio + corrections → derived:
  → matrix solve    (writer.solve_matrix_ctx)   shared chat budget minus RAG residents
  → per-model plan  (writer.Planner.plan)       mmproj keep/drop, auto-parallel, ctx
  → emit            (writer.emit_config)        llama-swap entries + matrix sets
```

Every stage except the probe is header-only or arithmetic: **a normal pack
run never starts a server**.

## 1. Discovery

The models directories are walked depth-first; the *role* comes from the
first-level directory (`chat/`, `embed/`, `rerank/`, plus the `dirs:` opt-ins
like `vision/`, `s2t/`, `image/`). A GGUF is paired with the same-stem
sidecar `.md` (a stub is generated when missing). HF-hub models resolve
through `hf_repo:` + `model:` to a concrete snapshot file. The sidecar
frontmatter is folded by the ScopeStack: profiles defaults → directory
`models.yaml` → override rules → sidecar block, then the backend is inferred
from the format (`.gguf` → llama-server; safetensors/`hf_repo` → vllm),
gated by the profiles `backends:` allow-list. Companions (mmproj, MTP draft)
resolve from sidecar declarations or same-stem conventions.

## 2. File intrinsics

One header-only GGUF read yields `(arch, trained_context, kind)`, persisted
under `derived: file:` with the file identity (`size_bytes`, `mtime_ns`).
Re-statting happens only when the identity changes — steady state is one
stat per model per run.

## 3. VRAM estimate — the affine constants

`Model.vram.fit_params_static` produces the affine quad
`(model_mib, kv_per_token_mib, slot_mib, compute_mib)`:

1. **Saved block** — the sidecar `derived:` block is reused iff `cache_type`
   **and** `shape` match. `shape` is the exact flag string the numbers were
   measured under (global args + the resolved `-ub`); a profiles or batch-key
   change re-measures instead of serving stale compute terms. `--remeasure`
   skips this path for one run.
2. **llama-fit-params trio** — three header-only runs (~0.6 s each) at the
   model's design context: `(C, p=1)`, `(C, p=2)`, `(C/2, p=1)`. The
   pool-line differences isolate `c` (per-token KV) and `D` (per-slot)
   exactly; `compute` is the trio max.
3. **Serve corrections** — per-arch family deltas measured once by the
   opt-in `--probe-memory` (fit-params estimate vs real llama-server truth),
   stored in the durable machine-local `serve-corrections.yaml` next to
   `profiles.yaml` with the witness `shape` + `ts`. Corrections are
   *optional calibration*: uncalibrated arches estimate uncorrected (one
   note per arch, never an error). Rows measured under a different shape
   still apply, with a one-time note.
4. vLLM models estimate from the HF repo / safetensors header instead —
   no fit-params.

Everything persists back into `derived:` (source `fit-estimate`, cache_type,
shape, ts) so later runs pay nothing.

### Deleting `derived:` — what actually happens

Re-measurement via the trio only: seconds per model, no server, no probes.
`--probe-memory` runs **only** when passed explicitly; it exists to refresh
the optional corrections, never as a pack dependency.

## 4. The shared matrix solve

When a RAG matrix is configured (`emb`/`rnk` vars), the chat budget is the
VRAM pool minus the fixed reserve (2 GiB), the spare (`--spare` /
`defaults.spare`), and the **RAG residents** — the embed and rerank models
at their served contexts, charged into every chat solve (a CPU-resident
embedder is free). The solve produces the largest shared per-slot chat
context (design-capped, rounded to 8192); a squeeze (residents →
`coload_min_ctx`) is adopted only when the chat context would otherwise fall
below `tools_min_ctx`.

## 5. Per-model planning

Per model (and per profile group `(parallel, cache_type, spare, batch,
ubatch)`):

- **mmproj keep/drop pass** — vision stays when the minimum useful context
  is affordable *with* the projection; a dropped main entry is emitted as
  `<id>-text` with a best-effort vision companion entry.
- **Auto-parallel** (chat, unpinned) — a value-function solve over slot
  counts `p = 1..8`: `score = (ctx/floor)^0.5 × slots^0.75`, where `ctx` is
  the affine solve at that `p`. The floor cascades: sidecar `min_context:`
  → serving pin → tools models 131k → half the design context. Ties prefer
  context; the loop stops when a slot count stops affording the floor.
- **Bounded context** — `calc_ctx` solves the affine law for the remaining
  budget (VRAM − reserve − spare − weights − compute), clamps to the design
  context and `--max-context`, rounds down to 8192, never below 4096. A
  model whose weights + compute alone exceed the budget serves at 4096 with
  a warning that says exactly what was needed vs budgeted.

## 6. Batch keys

`batch:` / `ubatch:` are first-class planning keys resolved sidecar >
profile > `llama_server:` fleet section > role defaults (chat 2048/512,
embed/rerank 4096/512 — the llama.cpp builtins). Both render explicitly on
every llama-server command in the per-role flag slot (named flags win over
conflicting fleet args, lose to `cli_args:` — which is why `cli_args` `-b`/
`-ub` is warned about). `ubatch` shapes the compute buffer (~6 MiB per
`-ub` token, dominated by vocab-sized logits rows) and is part of the
measurement shape; `batch` has no VRAM effect and is deliberately *not*
part of the shape (throughput tuning must not invalidate estimates).

## 7. Emit

Each variant becomes one llama-swap entry: command = built-ins (port, `-m`,
`--kv-unified-per-slot`, `--parallel`, cache types, `-ngl`) + feature flags
(MTP, mmproj, vision tokens, chat template, loras, reasoning) + global args
+ per-role/batch flags + `cli_args` (one flag map, later wins). Matrix
`vars`/`sets` tie the chat ids to `emb`/`rnk`; metadata carries the served
context, capabilities, and estimate health (`estimated`, `estimate_error`).
llama-swap (`-watch-config`) picks the file up; evictions are its job.
