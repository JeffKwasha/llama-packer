# Glossary

One name per concept. Where a term has an identifier in the code, that
identifier is given: this table is checked against the source, and the
tests named in the last column are the ones that pin the behaviour.

If a term is not here, either it is new or it has not been agreed on —
either way it should not be used in a commit message until it is.

## The model tree

| Term | Means | Code |
|---|---|---|
| **sidecar** | The `<model>.md` file next to a model file, holding its metadata as YAML frontmatter. The name comes from the printer's word for the separate insert the model is printed on. | any `*.md` beside a `.gguf` |
| **orphan** | A model file with no sidecar. Still discovered and served with built-in defaults — the sidecar is an editing surface, not a requirement. | `discover.py` |
| **stub** | An *empty* sidecar, written only with `--create-stubs`, as something to fill in. | `discover.py:_write_stub` |
| **task** | What a model *does*: a closed catalog id — `chat`, `embeddings`, `rerank`, `image`, `s2t`, `t2s` today. Claimed by the sidecar `tasks:` list, or the directory it sits in (`dirs:`). One task maps to one **endpoint**. It is not a placement, and an unknown id is not a task. `docs/architecture.md` §"Models, tasks, engines and endpoints". | `tasks.py` (P4); the endpoint map is in `endpoints.py` |
| **role** | A composition that *hosts* tasks — a use case, not a kind of work. Declared in `profiles.yaml roles:` with a persistence class per task; it does not require its elements. The matrix `sets:` are generated from it. | profiles `roles:` (P5) |
| **engine** | The serving engine: llama-server, vLLM, sd-server, whisper-server, audio-cpp. It declares the file formats it loads and the endpoints it serves — never a task or a role. (`backend` is the retired spelling, kept only as a code identifier.) | `backends/` |
| **endpoint** | A standardized serving surface: `openai.chat.completions`, `openai.embeddings`, `openai.rerank`, `openai.images.generations`, `openai.audio.transcriptions`, `openai.audio.speech`, `native.*`. Engine-owned; servers and clients choose it, a model never does. | `endpoints.py` |
| **family** | The interchangeability unit: serving one member taught you how to serve the other (same launch options, template, quantization conventions). Agent-assigned with evidence; the arch string is a confirm-only hint. Not the audio.cpp loader `family`, which is `audio_cpp.family`. | sidecar `family:` |
| **device** | Where a serving runs: `cpu` or `gpu`. The sidecar key is `device:`, emitted as `metadata.packing.device`. The only placement word — there is no `role: cpu`. | `Model.on_cpu` |
| **host-resident** | A `device: cpu` model: charged to the host pool, 0 VRAM, `ttl: 0`, pinned into the chat sets. Do not call it "a CPU task" — placement is not identity. | `HostLedger` |
| **`device_roles`** | Profiles declaration that places every model of a task on the host. A list of task ids, or a mapping **task → device**: `{embeddings: cpu}`. | `apply_device_roles` |
| **profiles.yaml** | Fleet-wide sampling defaults, spare values, backend defaults, the blessed `usage:` categories, and the optional `matrix:` section. | `profiles.py` |
| **override** | A rule that beats the profiles default, scoped to a glob or a directory. Per-directory `models.yaml` is the directory-scoped form. | `overrides.py` |
| **derived:** | The frontmatter block holding measured, machine-specific values (fingerprints, `model_mib`, KV coefficients). Deleting it costs a re-measure, never a probe. | `model.py:persist_measured` |
| **`spare`** | VRAM deliberately left free, so another tenant (ComfyUI, a game) can have it. Re-pack with the same `--spare` to reproduce a config. | profiles `spare`, CLI `--spare` |

## VRAM

| Term | Means | Code |
|---|---|---|
| **the affine law** | `VRAM(C, p) = model_mib + compute_mib + c*C + p*D`. Context and slots cost *per-token* and *per-slot* respectively, and the whole tool is built on that. | SPEC "Context Management" |
| **`c`** (`kv_per_token_mib`) | Shared full-attention KV cost per token. | `vram.py` |
| **`D`** (`slot_mib`) | Fixed cost per parallel slot: SWA ring buffers plus slot overhead. | `vram.py` |
| **slot** | One concurrent conversation. `parallel` is how many a model serves at once. | `parallel:` |
| **budget** (`VramBudget`) | The object that answers "what fits", from either measured coefficients or a pin. | `vram.py` |
| **matrix** | The shared-budget solve: one context across all co-resident models, so a resident embedder is paid for out of the chat context. | `solve_matrix_ctx`, SPEC "Matrix Context Solving" |
| **co-resident** | Loaded at the same time as others. Chat models take turns and share the pool; embed/rerank are pinned resident. | SPEC "Matrix Context Solving" |
| **baseline_mb** | The floor the reserve is measured against — what the GPU actually shows used before llama-packer runs. | profiles `hardware:` |
| **`globalTTL`** | Emitted only with `--idle-unload N`: llama-swap unloads anything idle for N seconds. | writer, SPEC 305 |
| **gain-gating** | A rung is kept only if it actually *buys* context; among rungs buying the same context, the least shed wins for the null category and the first qualifier stands. A model already at its design cap therefore keeps its projector rather than getting a pointless text-only twin. | `writer.py:solve_rung` |

## Precedence and lifetime

| Term | Means | Code |
|---|---|---|
| **tier** | Precedence class of an element: `pinned` / `role` / `task` / `any` (working names). It decides which routing sets an entry joins and what may evict it. | profiles `roles:` (P5) |
| **persistence** | A task's declared default inside a role: `pinned` (resident) or `on_demand` (loads on request, evicting its siblings). `on_demand` is the default. | profiles `roles:` (P5) |
| **TTL** | The temporal complement of `spare`: eventually the serving is freed for duties this tool cannot see. Cascade most-specific first — sidecar > task > role > CLI/global (`--idle-unload`); the **smallest** lifetime wins, and a conflict names both scopes. | writer, SPEC §TTL |

## Host RAM

| Term | Means | Code |
|---|---|---|
| **resident** | A task declared `pinned` in its role: served for the role's lifetime at a fixed small context, because losing it loses its KV cache. One live serving per endpoint. | SPEC §Residents |
| **host pool** | The host RAM the ledger may spend: `min(sys_free, cgroup_free)`, each term logged with the arm (`declared`/`sysctl`/`fallback-25%`/`skipped-cgroup`) that chose it. | `hardware.py:HostProfile` |
| **host ledger** | Sibling of `PoolLedger` that charges host-resident quads. Go/no-go, never a solve-down. | `HostLedger` |

## The capability ladder

| Term | Means | Code |
|---|---|---|
| **rung** | One point on the ladder: a set of capabilities deliberately given up, plus the `(ctx, parallel)` the budget affords there. | `writer.py:Rung` |
| **the ladder** (`shed_levels`) | The rungs to try for one category: one per prefix of its `sacrifice:` order, minus anything `require:`/`keep:` protects. | `writer.py:shed_levels` |
| **solve_rung** | Picks the rung: the one buying the most context, tie-broken toward keeping capabilities. `None` when the category cannot be met at all, and no entry is emitted. | `writer.py:solve_rung` |
| **`sacrifice`** | A *preference* order for what to give up first, not a cost order — shedding may buy nothing, which is why every rung is tried rather than one fixed walk. | `UsageDemand.sacrifice` |
| **require** | A capability the category will not be without. Gates existence: a model lacking it does not satisfy the category. | `UsageDemand.require` |
| **keep** | Don't drop what this model has, but a model without it still qualifies. The difference from `require` is why `coding` can keep its draft without excluding every model that has none. | `UsageDemand.keep` |
| **shed: (profiles)** | Order for capabilities a category does not name. Config, not code, so what "unnamed" means is the operator's call. | `Profiles.shed_order` |
| **mmproj** | The vision projector. Shedding it makes a model text-only and frees its VRAM. | llama-server |
| **MTP** | Multi-token prediction draft model. Roughly doubles generation speed and is the costly thing to lose, so it sits late in the sacrifice order. | SPEC "MTP Speculative Decoding" |

## Usage and routing categories

Two different things are called a category. Say which one. Under the
tasks vocabulary a `usage:` tag is a **task alias** (same endpoint, other
demand) and the routing var stays a **matrix category**:

| Term | Means | Code |
|---|---|---|
| **task alias** | A named workload class on one task (`coding`, `chat`, `qa`), today a `usage:` tag: same endpoint, its own minimum context. Aliases of one endpoint share one serving and are never loaded per alias. | profiles `usage:` → task `aliases` (P4) |
| **blessed** | Declared in profiles.yaml. An undeclared tag on a sidecar warns and is ignored — only declared categories exist. | `resolve_usage` |
| **demand** | One usage category's requirement for one model, resolved from its tags. | `UsageDemand` |
| **`min_context`** | The context a usage category requires. A model that cannot serve it at any rung emits nothing for that category. | profiles `usage:` |
| **matrix category** | An operator-named routing var under `matrix.categories` that binds a model to a var a set can name (`emb`, `rnk`, or a bucket such as `one`). A routing name, not a workload claim. | `matrix.categories` |
| **canonical category** | The category whose model is the task's pinned CPU resident: `emb`, `rnk`, else the first declared for that task. | `canonical_cpu_pins` |
| **set** | The emitted form of a role: a composition of tasks/categories that must be live together — `rag: __CHAT_VARS__ & emb & rnk` is embed *and* rerank. `&` sums, `|` maxes, `+set` expands; the chat solve reserves the worst branch of the most expensive chat set. | `matrix.sets`, `set_overhead_mb` |

## Emission

| Term | Means | Code |
|---|---|---|
| **entry id** | The stable identifier of an emitted llama-swap entry, `<template_id>.<category|ctx>.<flags>`. Names the task, not the solve. | `emit_config`, `entry_ids_by_stem` |
| **template_id** | The model identity a variant descends from. | `profiles.py` variant grouping |
| **squeezed** | An entry whose context was reduced below what it asked for, to make room. Says so in its description. | SPEC "Context Management" |
| **auto-parallel** | Choosing slots automatically when not declared. May not trade below a declared demand. | `auto_parallel_max`, SPEC 151 |
| **demotion** | Lowering a tool to a smaller model so a bigger one fits. | `writer.py` |

## Backends and transports

| Term | Means | Code |
|---|---|---|
| **backend** | Retired spelling of **engine**; survives only as a code identifier (`backends/`, `BoundBackend`). A sidecar declaring `backend:` warns as an alien key. | `backends/` |
| **transport** | *How* a backend runs: host, podman, docker. An independent axis from the engine. | `backends/transport.py` |
| **engine resolution** | The sidecar `engines:` list names the engines that can load the model (order, allow list, format, endpoint, availability); refusal by name when none applies. Required on an authored sidecar; only an orphan is inferred. | SPEC "Backend Selection" |
| **coload** | A second entry placed in a slot that is otherwise idle — opportunistic, not reserved. | `coload_stems` |

## Deploy

| Term | Means | Code |
|---|---|---|
| **run directory** | `/opt/llama`: the installed fleet — binaries, `config.yaml`, `config.env`. `root:<group>` mode 2775. | `extras/install.py` |
| **staging dir** | Scratch space *on the destination filesystem*, so an install finishes with an atomic rename. | `update.py:staging` |
| **promote** | The rename that makes a staged build its final name. | `update.py:promote` |
| **versioned entry** | `llama-bNNNNN`, `llama-swap-vNNN`, `<engine>-vX.Y.Z`. Kept newest-N; never pruned while a symlink points at it. | `update.py` |
| **stable primary** | The unversioned name beside a versioned entry — `llama-server` -> `./llama-b12345/llama-server`. What a config names; updating and rolling back both move this one link. | `update.py:_ensure_stable_primary` |
| **tools dir** | `<run dir>/tools`: where a deployment's engines live, so the run directory holds only generated config. Searched before the run directory itself. | `llama_packer/utils.py:binary_search_dirs` |
| **the fleet tag table** | Which upstream release is installed, per engine. | `docs/plans/usage-tag-table.md` |
