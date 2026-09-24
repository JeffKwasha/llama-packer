# llama-swap: features used by llama-packer

llama-packer's output is a llama-swap `config.yaml`. This page records which
llama-swap features the generated config relies on, where each is documented
officially, and how llama-packer's design maps onto it.

Official documentation (the source of truth — this page only indexes it):

- README / feature list: <https://github.com/mostlygeek/llama-swap>
- Knowledge base (focused guides): <https://github.com/mostlygeek/llama-swap/tree/main/docs/kb>
- `config.example.yaml` (every setting, commented): <https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml>
- A running instance also serves its own docs as an MCP endpoint at `/api/mcp`
  and a local "Help" page in the web UI.

## Emitted features → official docs

| llama-packer emits | llama-swap feature | Official doc |
|---|---|---|
| `models.<id>.cmd` with `${PORT}` | One entry = one server process; automatic port assignment | [writing-cmd](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/writing-cmd.md) |
| `filters.setParamsByID` keyed `${MODEL_ID}:<profile>` / `${MODEL_ID}:<mode>` | Per-request parameter rewriting; an alias is auto-created per key — the no-reload variant mechanism (request-body params only) | [filters and request rewriting](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/api-integration/filters-and-request-rewriting.md) |
| `capabilities` (`in`/`out`, `tools`, `reranker`, `context`) | `/v1/models` badges and listing; `context` = max trained ctx | [capabilities and model listings](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/capabilities-and-model-listings.md) |
| `metadata` (`ctx_size`, `modes`, `default_mode`, `image_min/max_tokens`, `chat_template(_kwargs)`, `mmproj_skipped`, `tools_demoted`, `mtp_*`, `estimated`, `estimate_error`) | Arbitrary pass-through in `/v1/models`; used for static client discovery (e.g. mode aliases) without per-request probing. `estimated: false` + `estimate_error` mark entries whose VRAM could not be measured — they serve at their minimum useful context and, in a matrix set, ride with no extra reserve (assumed to fit the set's measured allocation) | [config.example.yaml — metadata](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `name`, `description` | Listing display names | [capabilities and model listings](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/capabilities-and-model-listings.md) |
| `env` (e.g. `ROCR_VISIBLE_DEVICES=0`) | Per-model environment variables; used for multi-GPU device pinning | [config.example.yaml — env](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `concurrencyLimit` | Per-model concurrency cap / queueing | [capacity and queues](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/routing/capacity-and-queues.md) |
| `proxy` + `checkEndpoint: /` (sd-server, whisper-server) | Proxied HTTP services instead of managed processes; `/` avoids sd-server's 200-on-/health pitfall (upstream Discussion #866) | [writing-cmd](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/model-runtime/writing-cmd.md) |
| `macros` (`${VAR}` → absolute paths, path + flag macros) | Macro substitution in the config; llama-packer resolves to absolute paths so `-watch-config` reloads pick up new binaries without a service restart, and orders definitions so nested substitution (flag macro referencing a path macro) works | [macros](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/configuration/macros.md) |
| `includeAliasesInList: true` | Presents the auto-created setParamsByID aliases in `/v1/models`; llama-swap's default is `false`, which would hide them from dynamic-list clients (OpenWebUI, OpenClaw, …) | [config.example.yaml — includeAliasesInList](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `routing.router.use: matrix` + `settings.matrix` (`vars`, `evict_costs`, `sets`) | Concurrent-model swap DSL; llama-packer solves contexts/co-loads in Python and emits the *static* sets — llama-swap only does runtime load/evict within them | [groups and matrix](https://github.com/mostlygeek/llama-swap/blob/main/docs/kb/guides/routing/groups-and-matrix.md) |
| `healthCheckTimeout` | Startup health-check budget (auto-calculated from model size or explicit) | [config.example.yaml — healthCheckTimeout](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |
| `globalTTL` (via `--idle-unload SECONDS`) | Top-level default TTL: unload any model after SECONDS of inactivity (`0` = never; key omitted when the flag is absent). Opt-in — for co-residency with ComfyUI/games that need VRAM back without re-packing | [config.example.yaml — globalTTL](https://github.com/mostlygeek/llama-swap/blob/main/docs/config.example.yaml) |

Not a llama-swap feature: the sibling `config.env` file is llama-packer's own
artifact (systemd `EnvironmentFile=` / docker `--env-file`) that happens to
share the path-macro values with the emitted `macros:` block.

## Design consequences of the entry/process model

- **One entry = one process with a fixed command line.** Anything on the
  server's `cmd` (mmproj on/off, `-c`, cache type, parallel) is fixed for the
  life of that process. llama.cpp has no runtime API to attach/detach an mmproj
  or resize context, so each such difference is a *separate entry*, and moving
  between entries is an unload + relaunch (the "swap"). This is why vision
  on/off variants are distinct ids (`<id>` vs `<id>-text` vs
  `<id>-vision-<N>k`), not one entry with a toggle.
- **No-reload variants exist only for request-body parameters.** llama-swap's
  `setParamsByID` rewrites the outgoing request per model id and auto-creates
  an alias per key (`model:high`, …). That is exactly what llama-packer uses
  for sampling modes and profile parameter groups — zero reload, pure
  filtering. It cannot reach command-line settings; upstream has no cmd-level
  presets (see ggml-org/llama.cpp issue #23704, "Multiple presets for the same
  model").
- **The matrix is split-brain by design.** Sizing decisions (which contexts fit,
  what co-loads, tools demotion) are computed offline by llama-packer's VRAM
  solve; the emitted `sets` only tell llama-swap which entries may run
  together. llama-swap never re-solves budgets at runtime.

## Features deliberately not used

| Feature | Why not |
|---|---|
| `profiles` / `selectors` (runtime id pinning) | llama-packer emits concrete per-variant entries; routing is static, decided at pack time from measured VRAM |
| per-model `ttl` | Residency is expressed via matrix sets; the *global* default is optionally emitted as `globalTTL` through `--idle-unload` (see the table above) — per-entry `ttl` is never set |
| `hooks` (startup preload) | Preload policy belongs to the operator's service unit, not the generated config |
| `peers` (multi-host) | Single-host fleet; multi-host is out of scope |
| `stripParams`, `set-if-undefined`, API keys | No need in the current deployment shape |
