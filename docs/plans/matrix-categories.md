# Plan: Configurable matrix categories

Status: **proposal — not scheduled**. Recorded 2026-08-24.
Related: `docs/plans/comfyui-sd.md`, `SPEC.md` Matrix Context Solving, `llama_packer/{__main__,writer,vram}.py`.

## Current state

The swap `matrix` is **hardcoded to three categories**:

* Chat models → vars `c1..cN` (one per emitted entry id, including `-text` vision variants) — `__main__._build_matrix_vars:189`
* Embeddings → var `emb` — `__main__._build_matrix_vars:195` (`vars_["emb"] = embed_model.template_id`)
* Rerank → var `rnk` — `__main__._build_matrix_vars:198` (`vars_["rnk"] = rerank_model.template_id`)

Selection: smallest-VRAM model per role or CLI `--embed`/`--rerank` substring (`__main__._select_model:151`). VRAM solving: `writer._solve_matrix:427` + `writer._solve_matrix_context:501` → `vram.solve_matrix_ctx:642` which budgets

```
reserve = 1024 + max(1024, baseline_mb)
available = vram_total - reserve - spare
chat_ctx solves Σ(chat_weight + chat_factor×chat_ctx) = available − embed_overhead − rerank_overhead
```

with `embed_ctx`/`rerank_ctx` fixed at each model's `design_context` (embed/rerank declare small contexts; chat absorbs the remainder). Sets DSL uses fixed tokens:

```yaml
matrix:
  evict_costs: {emb: 100, rnk: 100}
  sets:
    rag: "__CHAT_VARS__ & emb & rnk"
```

`__CHAT_VARS__` expands to `(c1 | c2 | ...)` (`__main__:463-471`). `matrix` is **all-or-nothing**: if `profiles.yaml` defines it but no `embeddings`/`rerank` model exists, the build warns and skips the matrix (`__main__:426-431`). Vars `emb`/`rnk` are not configurable (see `profiles.yaml.example:87` note).

This covers RAG (`chat + embed + rerank` stay resident together) but cannot express e.g. `stable-diffusion + VL embedding + main chat`, or `image + chat + rerank`, or multiple chat groups with different colocation.

## Goal

Make matrix categories **declarative** in `profiles.yaml`, so operators can define arbitrary co-located groups — e.g. run a `stable-diffusion` (image) model alongside a `vision` embedding and a `chat` model without bespoke code per role.

Examples the design should enable (without exhaustiveness in the shipped example):

```yaml
# Profiles overlay stays as today; matrix categories are separate
matrix:
  evict_costs: {emb: 100, rnk: 100, sdxl: 50, vl: 80}
  categories:
    chat:   { role: chat,         selector: null }        # all chat variants (today's c1..cN)
    emb:    { role: embeddings,   selector: "jina" }      # optional substring selector
    rnk:    { role: rerank }
    sdxl:   { role: image,        dir: img/sdxl }        # future: stable-diffusion.hpp role
    vl:     { role: embeddings,   selector: "vlm-embed" }# fine-grained embedding split
  sets:
    rag:        "__CHAT_VARS__ & emb & rnk"     # backward compat
    creative:   "chat & sdxl & vl"              # new: literal category/var names
    code:       "chat & emb"                    # subset
```

or equivalently, keep the two fixed categories and allow extras:

```yaml
matrix:
  # today's emb/rnk stay as shorthand; new keys extend vars_
  categories:
    image: { role: image, selector: "flux" }
    vl-embed: { role: embeddings, selector: "siglip" }
  # vars become {c1..cN, emb, rnk, image, vl-embed}; sets reference them directly
```

Exact YAML shape is TBD — the sketch above is illustrative. Key is that **var name, role (or dir), and optional selector** are user-declared.

## Design sketch

1. **Profiles schema** — add `matrix.categories: dict[str, {role, selector?, dir?, kind?}]`. When absent, default to today's `{emb: {role: embeddings}, rnk: {role: rerank}}` for backward compat. `matrix.evict_costs` keys must match category var names (validation). `matrix.sets` values are DSL expressions over those var names plus the synthetic `__CHAT_VARS__` (which itself could become `__CATEGORY_VARS__[chat]` in the future, but keep `__CHAT_VARS__` as alias).

2. **Selection** — extend `_select_model(models, role, selector)` to also filter by optional `dir` prefix or `kind` label when roles alone insufficient (e.g. `embeddings` splinters into `vl-embed` vs `text-embed`). Reuse `SERVED_ROLES` validation (`utils.validate_dir_roles`).

3. **Var building** — generalize `_build_matrix_vars` → `vars_: dict[str, str]` where keys are the category names (today `c1..cN` synthetic + `emb`/`rnk`). Chat stays synthetic `c1..cN` (one per entry id); other categories contribute one var each (or `N` if later needed for multi-entry roles like multiple diffusion models). Keep `c1..cN` generation but allow the category name to alias it via `__CHAT_VARS__` or an explicit `chat` var.

4. **VRAM solving** — generalize `solve_matrix_ctx` from `(chat_list, embed_params, rerank_params)` to `(chat_list, category_params: dict[str, params])`. Each non-chat category contributes fixed overhead `weight + factor×context` at its own `design_context`; chat contexts still share the single `chat_ctx` solve. Optionally allow per-category `ctx: auto | <int>` to fix or solve a category's context. CPU-resident categories cost 0 (today's `model.on_cpu` check). Compatibility: when only `emb`/`rnk` present, behavior identical to today.

5. **Sets DSL** — no change to llama-swap's `routing.router.settings.matrix.vars/sets` schema; we just emit the user-declared var names. Keep `__CHAT_VARS__` expansion for backward compat; optionally add `__VARS__[cat]` sugar.

6. **Discovery / roles** — add `image` (and perhaps `audio`) to `SERVED_ROLES` / `dir_role_map` (`img → image`, `comfy → image`) and `backends` role sets (`llama_server` / `stable-diffusion.cpp` / `comfyui-boot` each declare which roles they serve). Until a sizing story exists for diffusion/comfy, those models are fixed-overhead (weight + declared `context_length` or `parameters` heuristic) similar to the `on_cpu` path today.

## Open questions

* Should chat remain the sole *solved* category (all others fixed overhead), or allow solving for multiple categories with priorities? The simple answer is "chat is solved, rest fixed" — matches today's RAG use and avoids multi-variable solving.
* How to size diffusion/ComfyUI models? No `llama-fit-params` analog. Options: honor declared `context_length`/`parameters`, measure via dry-run/load test, or track derived `fit-params` block with `source: estimate`. Initial cut: treat as fixed `model_mib + compute_mib` (file size + constant) and ignore per-token factor.
* DSL naming: keep `emb`/`rnk` as legacy var names or rename to `embed`/`rerank`? Backward-compat suggests keeping `emb`/`rnk` as defaults when `categories` absent.
* Relationship to llama-swap's native `matrix` vs `routing` — llama-swap's `matrix` already supports arbitrary `vars`/`sets`/`evict_costs` (see `config.example.yaml`). We're just driving it declaratively.

## References

* `llama_packer/__main__.py:151-200` `_select_model`, `_build_matrix_vars` (hardcoded `emb`/`rnk`)
* `llama_packer/__main__.py:419-483` matrix detection, `__CHAT_VARS__` expansion, `routing.router.settings.matrix`
* `llama_packer/writer.py:332-570` `Planner` matrix fields, `_solve_matrix` / `_solve_matrix_context`
* `llama_packer/vram.py:642-711` `solve_matrix_ctx` (current 3-category math)
* `llama_packer/utils.py:499-511` `dir_role_map`, `SERVED_ROLES`
* `profiles.yaml.example:82-132` current matrix comment (`emb`/`rnk` are fixed vars)
* `SPEC.md` Matrix Context Solving, `docs/architecture.md` VramBudget/Planner ownership
* Companion plan `docs/plans/comfyui-sd.md` (future image/comfy roles needing categories)
