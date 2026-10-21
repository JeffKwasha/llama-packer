# extras/

Helper scripts that live next to this repo but aren't part of the
llama-packer Python package.

- [`update.py`](update.py) — downloads/installs prebuilt server binaries.
  **Each engine is updated in the directory it is already installed in**, found
  by searching outwards from the cwd, so the command no longer has to be run
  from the engine's own directory and running it from the wrong place no longer
  installs a second copy. Every run prints the directory it resolved for each
  engine and why; `--print-paths` prints that and downloads nothing, `--dir
  <engine>=<path>` pins one for a single run, and an optional
  [`engines.ini`](engines.ini) (INI, not YAML — this script is pure stdlib) sets
  the root for new installs or pins one engine's directory. `$AUDIOCPP_BIN_DIR`
  and its equivalents work here too: the same variables llama-packer reads.
  Each engine keeps a versioned entry beside a stable symlink
  (`llama-server` -> `./llama-b12345/llama-server`, `llama-swap` ->
  `llama-swap-v266`, `whisper-server` -> `./whisper-server-b4938/whisper-server`,
  `audiocpp_server` -> `audiocpp_server-v0.9.2`), so updating and rolling back
  are both a one-line symlink change. Run `./extras/update.py --help` for options.
  Pure stdlib Python (no venv/deps), so it runs on a bare box too.
  **Targets are positional selectors** — `llama`/`ll`, `cpp`, `swap`, `audio`/`a`,
  `whisper`/`w`, `sd`/`stable`, and `all` for everything. Globs work (`s*`), and an
  ambiguous prefix is an error rather than a guess. `all` is matched in full only,
  because `a` is an alias for audio.cpp — otherwise the one-letter shorthand would
  be ambiguous. A bare `./extras/update.py` still updates only **llama.cpp +
  llama-swap**; `all` adds the other three. Only llama-swap publishes a
  `checksums.txt`, so it is the only one of the five that gets a real sha256
  check.
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
  anywhere: the install directory is resolved per engine, not taken from the
  cwd. Superseded versions are pruned after
  each install — the newest 3 `llama-b*/` dirs and the newest 2 versioned entries
  per engine (`--keep-swap N`, `--keep-engine-versions N`); anything a symlink still
  points at is kept. **llama-swap downloads are sha256-verified against the
  release `checksums.txt`** before anything is installed, and a mismatch
  quarantines the file as `_HASH_FAIL_*` and leaves the current binary alone.
  No other target publishes a manifest — checked against the b11658 llama.cpp
  release, whose 36 assets are all binaries — so the rest are pinned to an
  immutable tag URL over TLS and nothing more. That is a real limit, and it is
  stated rather than papered over. Rewritten from bash in
  2026-10: `--help` now exits 0 instead of 1, and `. extras/update.py --help`
  (sourcing) can no longer kill your shell the way the bash version did.
- [`build_engine.py`](build_engine.py) — compile a serving engine from source into
  the same directory a download would use. Exists because the R9700 is **gfx1201**
  and the stock ROCm packages carry no offload binary for it, so audio.cpp has to
  be built for the local arch: `./extras/build_engine.py audio` delegates to
  upstream's `scripts/build_linux.sh --deployment-build` (which resolves ROCm's own
  clang and defaults HIP builds to a relocatable deployment build) and installs to
  `audiocpp_server-src-v0.9.2-hip-gfx1201` beside the stable `audiocpp_server`
  symlink. `--dry-run` prints the plan (fetch URL, cmake invocation, install path)
  without spending a twenty-minute compile. Two properties worth knowing: the
  installed tree is **verified by running it** — `--list-devices` must enumerate the
  backend it was built for, so a build for the wrong arch never becomes the live
  symlink — and the source edits it applies are announced, logged, and recorded in
  `<versioned>.patched` next to the engine, because a patched tree is not upstream's
  tree. `--from-build DIR` installs an already-compiled `build/bin` without
  recompiling. Pure stdlib, reuses `update.py`'s install-location resolution,
  staging, atomic promotion and symlink machinery.
- [`bench.py`](bench.py) — measure engines and keep the results as history, because
  llama.cpp ships prebuilt binaries and changes often enough that "it used to be
  faster" is not answerable afterwards. Two modes: isolated (`llama-bench` over a
  batch/ubatch grid, which is how this fleet's `ubatch: 512` was chosen) and live
  (timed requests through the running llama-swap, the only way to measure the audio
  engines, which report their own real-time factor). Each run appends one record to
  `~/.cache/llama-packer/bench.jsonl` keyed by engine, version, backend, card, model,
  flags and metric, and prints the change against the previous run of the same
  key; `--report` prints the history. Append-only by design, machine-local, and it
  writes to no config or sidecar. Old builds are still on disk (`--keep-swap N`),
  which is what makes `--versions 2` able to A/B two llama.cpp releases.
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
