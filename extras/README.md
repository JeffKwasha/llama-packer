# extras/

Helper scripts that live next to this repo but aren't part of the
llama-packer Python package.

- [`update`](update) — downloads/installs prebuilt `llama.cpp` + `llama-swap`
  binaries into the repo root. Run `./extras/update --help` for options.
  **Run it from the repo root** (`./extras/update`) — the script detects
  that it is inside `extras/` and installs into the parent directory.
- [`llamaswap.ts`](llamaswap.ts) — opencode plugin: auto-discovers models
  from a running `llama-swap` server and injects them into opencode's
  provider config, plus `llamaswap_models` / `llamaswap_status` /
  `llamaswap_unload` tools. MIT licensed, see `LICENSE.md`.

## llamaswap.ts setup

1. Start `llama-swap` (default `http://localhost:8080`, or set
   `LLAMASWAP_URL`). The plugin reads the base URL from your opencode
   provider section first (`llamaSwap` / `llama.cpp` / `llama-swap`
   `options.baseURL`), then `LLAMASWAP_URL`, then the default.
2. Make the plugin's import resolvable. `llamaswap.ts` imports
   `@opencode-ai/plugin`, and Node/Bun resolve that by walking
   `node_modules` **upward from the plugin file** — there is no global
   dependency store (`npm install -g` only puts binaries on `PATH`;
   `NODE_PATH` is legacy and not honored here). So `extras/` needs a
   `node_modules` providing the package. Easiest, since opencode
   maintains one itself:
   ```sh
   ln -s ~/.config/opencode/node_modules /path/to/extras/node_modules
   ```
   Fallback (self-contained, no reliance on opencode's dir):
   ```sh
   cd /path/to/extras && bun add @opencode-ai/plugin
   ```
3. Register the plugin in `~/.config/opencode/opencode.json`:
   ```json
   { "plugin": ["file:///path/to/extras/llamaswap.ts"] }
   ```
4. Ensure a matching provider section exists (the plugin injects models
   under the first key it finds: `llamaSwap`, then `llama.cpp`):
   ```json
   { "provider": { "llamaSwap": {
     "npm": "@ai-sdk/openai-compatible", "name": "llamaSwap",
     "options": { "baseURL": "http://localhost:8123/v1" } } } }
   ```
5. Restart opencode.

Optional env: `LLAMASWAP_PROVIDER` forces the provider key instead of
auto-detection.

## What it injects

Only tool-capable ("agentic") models are registered — the plugin filters
`GET /v1/models` to entries with `capabilities.function_calling` /
`capabilities.tools` or `tools` in `supported_parameters`. Embedding,
reranker, and diffusion models are skipped. Context limits come from
`meta.n_ctx` → `context_length`/`context_window` → `meta.llamaswap.ctx_size`,
defaulting to 131072; output limit is `min(ctx/2, 65536)`.

## Troubleshooting

- **No models appear**: the plugin failed to load, almost always the
  `node_modules` resolution above. Verify with:
  ```sh
  bun -e 'await import("/path/to/extras/llamaswap.ts").then(m => console.log(Object.keys(m)))'
  ```
  `Cannot find module '@opencode-ai/plugin'` → fix step 2.
  `["LlamaSwapPlugin"]` → import is fine; check the server URL and
  `~/.local/share/opencode/log/opencode.log` for `failed to load plugin`.
- **Models disappear after moving the file**: same cause — resolution is
  relative to the file's location, not `PATH` or CWD. Re-apply step 2
  wherever the file now lives.
- **Wrong provider key**: set `LLAMASWAP_PROVIDER` to pin it.
