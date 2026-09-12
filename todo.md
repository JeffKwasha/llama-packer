# todo.md — Plan for B + C

Status: **historical planning snapshot** — the plan as it stood before
implementation; landed on branch `dry_backend` (2026-09-12). See
[Completion status](#completion-status) at the bottom for what shipped, what
was deferred, and what our decisions superseded.

- **B** = backend/transport delineation (engine vs transport seam).
- **C** = configurable matrix categories, to express
  `(CHAT) + ((EMB+RERANK) | (TTS&/or STT))`.
- **Forcing function** = `audio.cpp` (`audiocpp_server`); engine doc now at
  [`docs/backends/audio-cpp.md`](docs/backends/audio-cpp.md).

Companion docs: [`docs/backends/`](docs/backends/),
[`docs/transports/`](docs/transports/),
[`docs/plans/matrix-categories.md`](docs/plans/matrix-categories.md) (implemented),
[`docs/plans/opportunistic-coload.md`](docs/plans/opportunistic-coload.md),
[`docs/architecture.md`](docs/architecture.md).

---

## Where the code is right now (facts, not proposals)

**Backends are flat registry entries** (`llama_packer/backends/__init__.py:34`):
`BACKENDS = {llama-server, vllm, vllm-docker, sd-server, whisper-server,
kokoro-podman}`. Each is a `BaseBackend` (`backends/base.py`) that renders one
resolved `Model` into a `cmd`. The **engine** (what computes) and the
**transport** (host process / docker / podman) are conflated in the name:

| Registry name | Engine | Transport |
|---|---|---|
| `llama-server` | llama.cpp | host |
| `vllm` | vLLM | host |
| `vllm-docker` | vLLM | docker |
| `sd-server` | stable-diffusion.cpp | host |
| `whisper-server` | whisper.cpp | host |
| `kokoro-podman` | Kokoro | podman |

Transport-specific logic already exists but scattered: `vllm.py` carries the
path mapping/mounts/env (`_CONTAINER_HF_HOME`, `_map_paths_into`), lifecycle
(`stop_cmd`, `unload_timeout`), and proxy emission; `kokoro.py` carries podman
vendor/device flags. `BaseBackend` already has ClassVars `proxied`, `stop_cmd`,
`unload_timeout` — a partial seam.

**The matrix is hardcoded to three roles** (`llama_packer/__main__.py`):
`_detect_matrix` (`:250`), `_select_model` (`:225`), `_build_matrix_vars`
(`:288`, emits synthetic `c1..cN` + `emb` + `rnk` + opportunistic co-loads),
`_expand_matrix_sets` (`:381`, `__CHAT_VARS__` / `__COLOAD_VARS__`). Solving
lives in `writer.Planner._solve_matrix` / `_solve_matrix_context` →
`vram.solve_matrix_ctx`. Non-chat roles (`utils.SERVED_ROLES` = chat, embeddings,
rerank, image, s2t, t2s; `NON_CHAT_ROLES` = the last five) are excluded from the
chat solve; `FIXED_OVERHEAD_BACKENDS` = sd ∪ whisper ∪ kokoro get a fixed budget
(`vram.py:938`).

**Consequence:** adding an audio engine multiplies both axes at once (a new
engine × host/podman), and audio can't join the matrix (it's a fixed-overhead
sidecar, not a declarable category).

---

## B — engine / transport delineation

### Goal
One **engine** definition reusable across **transports**, so `audio-cpp` (and
future engines) don't add a flat registry entry per transport, and so
docker/podman path-mapping/lifecycle logic has a single home.

### Proposed seam
Split responsibilities:

- **Engine** (what computes): roles/formats/capabilities; argv builder
  (`serve_flags`); binary/availability + version; VRAM class (measured vs
  fixed-overhead); the set of path-valued refs it emits (model, draft, template).
- **Transport** (how it runs): wrap argv in host/docker/podman; mounts +
  host→container path mapping; container env; lifecycle (`cmdStop`,
  `unloadTimeout`); `proxy` / `checkEndpoint`; port publication.

Registry resolves a name → `(engine, transport)`; keep the existing names as
aliases so nothing else changes:

```
llama-server   = (llama.cpp, host)      vllm       = (vllm, host)
sd-server      = (sd.cpp, host)         vllm-docker= (vllm, docker)
whisper-server = (whisper.cpp, host)    kokoro-podman = (kokoro, podman)
audio-cpp      = (audio.cpp, host)      [later] audio-cpp-podman = (audio.cpp, podman)
```

### Options (decision needed)
1. **Minimal:** keep the flat registry; factor transport helpers into shared
   functions/mixins. Least churn, but the 2-D growth is only half-fixed.
2. **Full:** explicit `Engine` ABC × `Transport` object, registry maps aliases.
   Cleanest; touches every backend + `writer` entry emission.
3. **Middle (recommended):** keep `BaseBackend` subclasses as *engines*, add a
   `transport` collaborator object used for cmd wrapping/mapping/lifecycle.
   Incremental; existing names unchanged; `vllm`/`vllm-docker` collapse to one
   engine + two transports behind the alias.

### Tasks (B)
- [ ] Choose option (1/2/3); write it into `docs/architecture.md` extension points.
- [ ] Extract transport interface: `wrap(argv, model) -> cmd`, `mounts()`,
      `env()`, `lifecycle()`, `proxy()`.
- [ ] Port `vllm-docker` + `kokoro-podman` onto the transport object; prove
      byte-identical emitted entries (golden test).
- [ ] Make backend inference engine-aware, not transport-aware.
- [ ] Keep `FIXED_OVERHEAD_BACKENDS` / `VLLM_BACKENDS` semantics working
      (redefine in terms of engines, not registry names).

### Open questions (B)
- Does the transport ever affect *role/format* inference (e.g. a repo-id model
  only servable offline in a container)? If so, the seam isn't purely mechanical.
- Where does `is_available` live — engine (binary present) vs transport
  (docker/podman present + image pullable)?
- Do we need multi-transport for one model simultaneously, or is transport a
  per-model choice (current behavior)?

---

## C — configurable matrix categories

### Goal
Make co-load categories declarative so `profiles.yaml` can express the target:

```yaml
matrix:
  categories:
    emb:  { role: embeddings }
    rnk:  { role: rerank }
    tts:  { role: t2s }
    stt:  { role: s2t }
  evict_costs: { emb: 100, rnk: 100, tts: 100, stt: 100 }
  sets:
    rag:   "__CHAT_VARS__ & emb & rnk"
    voice: "__CHAT_VARS__ & (tts | stt)"      # <- the new capability
    # target shape: (CHAT) + ((EMB+RERANK) | (TTS&/or STT))
```

### Current pain
Categories are hardcoded to `emb`/`rnk` (`_build_matrix_vars`); vars aren't
configurable; a missing embed/rerank model disables the whole matrix; audio
roles can't participate at all.

### Proposed design (from `matrix-categories.md`, plus audio)
- **Schema:** `matrix.categories: dict[name → {role, selector?, dir?, kind?}]`;
  absent ⇒ today's `{emb: {role: embeddings}, rnk: {role: rerank}}` (back-compat).
  `evict_costs` keys must match category names (validated).
- **Selection:** extend `_select_model(models, role, selector, dir?, kind?)` to
  split a role when needed (e.g. two embedding kinds), reusing
  `utils.validate_dir_roles`.
- **Var building:** generalize `_build_matrix_vars` → `{category: var}`; chat
  stays synthetic `c1..cN` (aliased by `__CHAT_VARS__`); each non-chat category
  contributes one var (fixed-overhead, at its `design_context`).
- **Solver:** generalize `vram.solve_matrix_ctx` from
  `(chat_list, embed_params, rerank_params)` to `(chat_list, category_params)`;
  keep "chat solved, everything else fixed overhead"; optional per-category
  `ctx: auto | <int>`. Behaviour identical when only `emb`/`rnk` are defined.
- **Sets DSL:** unchanged (we only emit user-declared var names); keep
  `__CHAT_VARS__`; `audio` categories are just more fixed-overhead vars.
- **Audio specifics:** audio backends are already `FIXED_OVERHEAD_BACKENDS`;
  C only makes them *declarable* (and gives them `evict_costs`).

### Tasks (C)
- [ ] Add `matrix.categories` to `profiles.yaml.example` + validation + `SPEC.md`.
- [ ] Generalize `_build_matrix_vars` / `_expand_matrix_sets` / `_detect_matrix`.
- [ ] Generalize `solve_matrix_ctx` + `Planner._solve_matrix*` to `category_params`.
- [ ] Decide audio var granularity: `tts`/`stt` separate, or one `audio` category
      with a selector/OR.
- [ ] Matrix disabled-but-defined: today it hard-skips if embed or rerank is
      missing — rework so a category simply contributes nothing when absent
      (needed for "voice-only" fleets).
- [ ] Tests: back-compat (emb/rnk only ⇒ identical output) + new audio set +
      alt-branch (`(EMB+RERANK) | (TTS&/or STT)`).
- [ ] Docs: `SPEC.md` Matrix Context Solving, README "packed matrix" bullet.

### Open questions (C)
- Semantics of the `+` and `|` in the target expression — confirm against
  llama-swap's `settings.matrix` var/set grammar (can a set co-load `chat` with
  *either* branch, and do we emit two sets or one with `|`?).
- Does the matrix need to model *exclusive alternation* (one branch resident at
  a time), or is `evict_costs` enough to steer it?
- If chat is not the only solved category in future, priorities between
  categories — keep "chat solved, rest fixed" for now.
- Interaction with opportunistic co-load (`__COLOAD_VARS__`) once categories are
  arbitrary.

---

## Sequencing

1. **B first, behaviour-preserving.** Refactor transport with golden tests; no
   config output change. This is the safe foundation.
2. **C with existing backends.** Generalize the matrix using today's
   `kokoro-podman`/`whisper-server` as the audio categories — validates the
   design without `audio.cpp` risk.
3. **audio.cpp on the seam.** Add the engine (transport = host) and declare its
   `t2s`/`s2t` categories. Q1 of `audio-cpp.md` may feed back into B.

## Risks
- **B churn** can silently change emitted commands — mitigate with exact-string
  golden tests before/after (the repo already asserts on `cmd` strings).
- **C solver generalization** regresses RAG sizing — mitigate with a frozen
  back-compat fixture asserting identical `ctx`/entry output.
- **audio.cpp unknown (Q1):** if it truly can't be proxied by llama-swap, B's
  transport seam and C's category model both need a standalone-service variant.

## Out of scope
- Rewriting whisper/kokoro in terms of `audio-cpp`.
- ComfyUI / additional engines beyond establishing the seam.
- Any decision already covered by `docs/plans/audio-cpp.md` (that AI owns it).

## Decisions needed before starting
1. B: option 1 / 2 / 3.
2. C: exact `matrix.categories` YAML shape (ratify `matrix-categories.md`).
3. Audio category granularity (`tts`/`stt` vs `audio`).
4. Whether B and C land as separate PRs (recommended) or one.

---

## Completion status (2026-09-12)

**B — done.** Engine and transport are independent axes (`backends/transport.py`,
`BoundBackend`): `vllm`/`vllm-podman`/`vllm-docker` are one engine × three
transports; engines declare `transports`; preference host > podman > docker,
runtime-gated. `docs/architecture.md` updated (components, invariant, extension
points).

**C — done (approach (a)).** `matrix.categories` (default `emb`/`rnk`),
per-category vars, `evict_costs` validation; non-RAG categories are
fixed-overhead residents reserved first. `tts`/`stt` separate. Documented in
`SPEC.md` + README.

**audio-cpp — done (host).** First engine on the seam; roles `t2s`+`s2t`;
replaces kokoro-podman (removed) and complements whisper-server (no model
overlap).

**Superseded / deferred / out of scope:**
- Full `solve_matrix_ctx` → `category_params` generalization — **not done by
  decision** (approach (a) chosen over (b)).
- Missing-category / "voice-only" fleets — **not done**; `emb`+`rnk` are still
  required to enable the matrix.
- `+`/`|` grammar vs llama-swap `settings.matrix` — **unverified**.
- audio-cpp **podman transport** — **future, if ever**.
- audio.cpp **source build** — **out of scope** for llama-packer.
- audio roles `vc`/`vad`/`music`/`sep` — parked in `docs/plans/audio-roles.md`.
