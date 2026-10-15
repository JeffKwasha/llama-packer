# extras/

Helper scripts that live next to this repo but aren't part of the
llama-packer Python package.

- [`update.py`](update.py) — downloads/installs prebuilt server binaries into
  the **current working directory**. Run `./extras/update.py --help` for options.
  Pure stdlib Python (no venv/deps), so it runs on a bare box too.
  **Targets are positional selectors** — `llama`/`ll`, `cpp`, `swap`, `audio`/`a`,
  `whisper`/`w`, `sd`/`stable`, and `all` for everything. Globs work (`s*`), and an
  ambiguous prefix is an error rather than a guess. `all` is matched in full only,
  because `a` is an alias for audio.cpp — otherwise the one-letter shorthand would
  be ambiguous. A bare `./extras/update.py` still updates only **llama.cpp +
  llama-swap**, the two that publish `checksums.txt`; `all` adds the other three,
  which print an unverified warning rather than pretending to a sha256 check they
  cannot do.
  Backends are positional too (`vulkan`, `rocm`, `cpu`); llama.cpp takes them as
  part of its legacy version token. Per-project quirks are handled: audio.cpp
  follows llama.cpp's asset naming and ships binaries on its semver tag; whisper.cpp
  publishes a semver tag *and* an asset-less build tag per release (only `bNNNN`
  carries binaries) and ships a **bundle** whose `.so` files travel with the binary
  via `RUNPATH=$ORIGIN`; stable-diffusion.cpp versions as `master-N-<sha>`, ships
  **zips** rather than tarballs, spells the arch `x86_64`, is also a bundle, and
  has both Vulkan and ROCm Linux builds.
  The CPU architecture is auto-detected for all downloads
  (llama.cpp/audio.cpp/whisper `x64`/`arm64`, stable-diffusion `x86_64`/`aarch64`;
  llama-swap `linux_amd64`/`linux_arm64`); `--arch` overrides it. Run it from
  wherever you want the binaries installed. Superseded versions are pruned after
  each install — the newest 3 `llama-b*/` dirs and the newest 2 versioned entries
  per engine (`--keep-swap N`, `--keep-engine-versions N`); anything a symlink still
  points at is kept. Downloads are sha256-verified against the release
  `checksums.txt` before anything is installed (a mismatch quarantines the file
  as `_HASH_FAIL_*` and leaves the current binary alone). Rewritten from bash in
  2026-10: `--help` now exits 0 instead of 1, and `. extras/update.py --help`
  (sourcing) can no longer kill your shell the way the bash version did.
- [`install.py`](install.py) — deploys llama-swap under systemd from a run
  directory (`/opt/llama` by default), running as a chosen system user:
  creates the account, installs the binaries, regenerates `config.yaml` with
  llama-packer, writes the hardened [`llama-swap.service.in`](llama-swap.service.in)
  unit, enables and starts it. See [docs/install.md](../docs/install.md).
  Run `./extras/install.py --help`; `--dry-run` prints every change first.
- [`fit-sweep`](fit-sweep) — llama-fit-params VRAM data points for the affine
  law `VRAM(C, p) = fixed + c·C + D·p`. Header-only (~0.6 s/run, no tensor
  data, no server, no meaningful VRAM use): safe on platter storage and
  beside a busy GPU. Run `./extras/fit-sweep --help` from the repo root.
- [`clear-measured`](clear-measured) — purges the persisted `measured:`
  blocks llama-packer writes into sidecars (they override the live
  estimator and corrections cache until re-measured). Use whenever an
  era's numbers are quarantined; `--before TS` purges by the `ts:` stamp.
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
