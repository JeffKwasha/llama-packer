# llama-swap capabilities and how to request them

llama-packer's output is a llama-swap `config.yaml`. This page maps what
llama-swap can do to the inputs you write — `profiles.yaml` and model
sidecars — so you can ask for a capability without hand-editing the emitted
config.

Official documentation (the source of truth — this page only indexes it):

- README / feature list: <https://github.com/mostlygeek/llama-swap>
- Knowledge base (focused guides): <https://github.com/mostlygeek/llama-swap/tree/main/docs/kb>
- `config.example.yaml` (every setting, commented): <https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml>
- A running instance also serves its own docs as an MCP endpoint at `/api/mcp`
  and a local "Help" page in the web UI.

The memory model behind routing, capacities and eviction is in
[memory-and-eviction.md](memory-and-eviction.md). The syntax reference is
[profiles-authoring.md](profiles-authoring.md).

## Capability → the input that provides it

| You want | llama-swap does | Write this |
|---|---|---|
| Several models, one loaded at a time | one process per entry, swapped on request | sidecars in `models_dirs`; one entry per model/variant |
| Two models loaded together | capacity pools; any combination that fits may coexist | `roles:` with `share:`; the auction sizes each member |
| One model per role | a size-1 counting pool per role | `roles:` (the pool is emitted automatically) |
| Forbid a combination that fits | matrix sets; only listed combinations may be loaded | `matrix.sets:` |
| Never unload a model | eviction cost, and `ttl: 0` | sidecar `ttl: 0` |
| Pin a model to a GPU | `env` device variable | sidecar `device: N` or `roles.<name>.device: N` |
| Run a model from system RAM | a process with `--n-gpu-layers 0` | sidecar `device: cpu` or `roles.<name>.device: cpu` |
| Unload idle models | `ttl` | top-level, role, task, or sidecar `ttl:` |
| Change sampling per request | `filters.setParamsByID` | `profiles:` and sidecar `modes:` |
| Show modes as models | auto-created aliases in `/v1/models` | `includeAliasesInList: true` (emitted) |
| Advertise context/tools/vision | `capabilities` badges | sidecar `capabilities:`, `context_length`, `tasks:` |
| Expose a static model list | `metadata` passthrough | emitted automatically |
| Tune launch flags fleet-wide | `cmd` rendered from macros | `llama_server.args`, `macros:` |
| Share paths with systemd | path macros | `models_dirs`, `hf_home` → `config.env` |
| Run a proxied service | `proxy:` / `checkEndpoint:` | `backends:` + a role owning the directory (sd, whisper) |
| Cap concurrent requests | `concurrencyLimit` | per-model `parallel:`/`concurrencyLimit:` |
| Budget startup time | `healthCheckTimeout` | auto-calculated; `--health-check-timeout` overrides |

## Composing the capabilities

### One model per role, several roles loaded

Every role is a pool of value 1, and every entry charges 1 to its role's
pool. One serving of that role is loaded at a time; a second request evicts
the first. Different roles coexist when their capacities fit.

```yaml
roles:
  chat:       {dir: chat, share: fill}
  embeddings: {dir: embed, share: 5%, device: cpu}
  rerank:     {dir: rerank, share: 5%, device: cpu}
pools:
  default:
    spare: 3G
```

A set's floors come out first; `share` divides the auction. `chat` fills, so
it takes what the helpers leave.

### Your own roles

Declare a role with its own trees and share.

```yaml
roles:
  cat:     {dir: cats, share: 40%}
  dog:     {dir: dogs, share: 40%}
  hamster: {dir: voices, share: 10%}

tasks:
  cat:   {endpoint: openai.chat.completions}
  dog:   {endpoint: openai.chat.completions}
  speak: {endpoint: openai.audio.speech}
```

Models belong to the role that owns their directory; a sidecar `role:`
overrides it. Tasks do not name a role.

`cat` and `dog` may be loaded together; a second `cat` request evicts the
first. `hamster` is a separate voice role. Shares are weights inside each
set, not a cap on the fleet.

### A model that must run alone

Put its role in a single-leaf set (`alone: "foo"`). It gets the whole auction
of that set, and nothing else may be loaded with it.

### Forbidding a combination that fits

Capacity allows any combination whose costs fit the pools. A set is the only
way to say no:

```yaml
matrix:
  sets:
    rag:   "chat & embeddings & rerank"
    voice: "chat & t2s"
```

Each set is its own budget, sized by the auction. Capacity uses generated
sets with the same algebra and still admits any combination that fits.
Matrix is the one that forbids. `[A&B] [A&C] [A&D]` constrains only the
pairs, so each pair may use the whole pool; a global sum would force every
model to a quarter.

It does not preload. On a request for X, llama-swap picks the cheapest set
containing X, evicts models outside it, and starts only X. Subsets are
permitted, so a set is a permission, not a prescription. A model in no set
runs alone.

### Request-time variants, not reloads

`filters.setParamsByID` changes sampling per request. Anything on the
command line (`-c`, cache type, parallel, mmproj) is fixed for the life of
the process, so a different value is a separate entry.

```yaml
profiles:
  default:
  coder: {temperature: "base * 0.7"}
```

Sidecar `modes:` layer over the same-named profile; `allow_profiles:`
filters what a model may use.

### Idle vs pressure

TTL yields time; capacity yields memory. Use `ttl: 0` on a role, task, or
sidecar to stop idle unload, and `pools.<pool>.spare` when a pool must stay
honest under pressure. The top-level `ttl` defaults to 3600 seconds.

## Sidecar keys that change emitted behaviour

| Sidecar key | Effect in the emitted config |
|---|---|
| `engines:` | Which engine may serve the model (required for an authored sidecar) |
| `tasks:` | Which entries the model is offered for |
| `device: cpu` / `device: N` | Host placement / GPU pin (emits env + `-ngl 0`) |
| `parallel:` | Pins slots for this model (opts out of auto-parallel) |
| `context_length:` | Trained ceiling; the solve may serve below it, never above |
| `cache_type:` | KV precision; a different value is a separate entry |
| `capabilities:` | `/v1/models` badges (`tools`, `image`, `video`, …) |
| `ttl:` | Idle lifetime (`0` = never) |
| `modes:` | Per-request sampling variants layered over profiles |
| `allow_profiles:` | Which named profiles this model may use |
| `cli_args:` | Extra engine flags for this model's command line |
| `vram_mb:` | Operator-pinned VRAM charge; no estimate headroom |
| `pp_tps:` | Prompt rate used for this entry's eviction cost |
| `chat_template:` / `loras:` | Rendered into the command line |
| `mmproj:` | Whether the vision variant exists and what it costs |
| `mtp:` | Speculative-decoding companion flags |
| `hf_repo:` / `model:` / `file:` | Where the weights come from |
| `ignore: true` | Excluded from the emitted config |

## Emitted features → official docs

| llama-packer emits | llama-swap feature | Official doc |
|---|---|---|
| `models.<id>.cmd` with `${PORT}` | One entry = one server process; automatic port assignment | [writing-cmd](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/writing-cmd.md) |
| `filters.setParamsByID` keyed `${MODEL_ID}:<profile>` / `${MODEL_ID}:<mode>` | Per-request parameter rewriting; an alias is auto-created per key — the no-reload variant mechanism (request-body params only) | [filters and request rewriting](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/api-integration/filters-and-request-rewriting.md) |
| `capabilities` (`in`/`out`, `tools`, `reranker`, `context`) | `/v1/models` badges and listing; `context` = served per-slot window (llama-swap's documented meaning), so client compaction and output budgets match what the entry actually serves | [capabilities and model listings](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/capabilities-and-model-listings.md) |
| `metadata` (`ctx_size`, `context_max`, `max_output_tokens`, `modes`, `default_mode`, `image_min/max_tokens`, `chat_template(_kwargs)`, `mmproj_skipped`, `tools_demoted`, `mtp_*`, `estimated`, `estimate_error`) | Arbitrary pass-through in `/v1/models`; used for static client discovery (e.g. mode aliases) without per-request probing. `ctx_size` repeats the served window; `context_max` is the trained maximum when the entry serves below it; `max_output_tokens` is the client-facing generation ceiling consumed by opencode-models-discovery and Hermes — it defaults to the served window and a sidecar may lower it. `estimated: false` + `estimate_error` mark entries whose VRAM could not be measured — they serve at their minimum useful context and declare a file-size floor as their capacity | [config.example.yaml — metadata](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `name`, `description` | Listing display names | [capabilities and model listings](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/capabilities-and-model-listings.md) |
| `env` (e.g. `ROCR_VISIBLE_DEVICES=0`) | Per-model environment variables; used for multi-GPU device pinning | [config.example.yaml — env](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `concurrencyLimit` | Per-model concurrency cap / queueing | [capacity and queues](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/routing/capacity-and-queues.md) |
| `proxy` + `checkEndpoint: /` (sd-server, whisper-server) | Proxied HTTP services instead of managed processes; `/` avoids sd-server's 200-on-/health pitfall (upstream Discussion #866) | [writing-cmd](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/writing-cmd.md) |
| `macros` (`${VAR}` → absolute paths, path + flag macros) | Macro substitution in the config; llama-packer resolves to absolute paths so `-watch-config` reloads pick up new binaries without a service restart, and orders definitions so nested substitution (flag macro referencing a path macro) works | [macros](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/configuration/macros.md) |
| `includeAliasesInList: true` | Presents the auto-created setParamsByID aliases in `/v1/models`; llama-swap's default is `false`, which would hide them from dynamic-list clients (OpenWebUI, OpenClaw, …) | [config.example.yaml — includeAliasesInList](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `routing.router.use: capacity` + `settings.capacity.pools` (`default`, `ram`, one size-1 counting pool per role), per-model `capacities` and `evict_cost` | Resource-pool routing: any combination that fits every pool may coexist; llama-swap evicts the cheapest serving under pressure. llama-packer solves contexts/capacities in Python | [groups and matrix](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/routing/groups-and-matrix.md) |
| `routing.router.use: matrix` + `settings.matrix.sets` | Combination routing: only listed combinations may be loaded; each set is its own budget. Never preloads — it evicts to the cheapest set containing the requested model and starts only that model | [groups and matrix](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/routing/groups-and-matrix.md) |
| `healthCheckTimeout` | Startup health-check budget (auto-calculated from model size or explicit) | [config.example.yaml — healthCheckTimeout](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `ttl` (top-level, default 3600) | Top-level default TTL: unload any model after that many seconds of inactivity (`0` = never). A role, task, or sidecar `ttl:` overrides it | [config.example.yaml — ttl](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |

Not a llama-swap feature: the sibling `config.env` file is llama-packer's own
artifact (systemd `EnvironmentFile=` / docker `--env-file`) that happens to
share the path-macro values with the emitted `macros:` block.

## Design consequences of the entry/process model

- **One entry = one process with a fixed command line.** Anything on the
  server's `cmd` (mmproj on/off, `-c`, cache type, parallel) is fixed for the
  life of that process. llama.cpp has no runtime API to attach/detach an mmproj
  or resize context, so each such difference is a *separate entry*, and moving
  between entries is an unload + relaunch (the "swap"). This is why vision
  on/off variants are distinct ids (the vision-less variant's id simply lacks
  the `i`/`v` capability flags), not one entry with a toggle.
- **No-reload variants exist only for request-body parameters.** llama-swap's
  `setParamsByID` rewrites the outgoing request per model id and auto-creates
  an alias per key (`model:high`, …). That is exactly what llama-packer uses
  for sampling modes and profile parameter groups — zero reload, pure
  filtering. It cannot reach command-line settings; upstream has no cmd-level
  presets (see ggml-org/llama.cpp issue #23704, "Multiple presets for the same
  model").
- **Sizing and enforcement are split by design.** Contexts and capacities are
  computed offline by llama-packer's VRAM solve; the emitted pools tell
  llama-swap what may run together, and it enforces the limits at request
  time. It never re-solves a context.

## Features deliberately not used

| Feature | Why not |
|---|---|
| `profiles` / `selectors` (runtime id pinning) | llama-packer emits concrete per-variant entries; routing is static, decided at pack time from measured VRAM |
| `hooks` (startup preload) | Preload policy belongs to the operator's service unit, not the generated config |
| `peers` (multi-host) | Single-host fleet; multi-host is out of scope |
| `stripParams`, `set-if-undefined`, API keys | No need in the current deployment shape |
| `metrics`, `logLevel`, `web UI` settings | The service unit or a hand-edited layer owns these |

Anything not emitted here can be added to `config.yaml` by hand, but the
next pack overwrites the file. If it must persist, it belongs in a
`profiles.yaml` key or a sidecar field — or upstream as a feature request.
