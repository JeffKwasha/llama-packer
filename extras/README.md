# extras/

Helper scripts that live next to this repo but aren't part of the
llama-packer Python package.

- [`update.py`](update.py) — updates prebuilt server binaries. One pipeline
  for every engine: resolve a release, download, verify, extract into a
  staging directory and promote by rename, then point one stable symlink and
  prune the tree. **The run directory comes from where the command was run**:
  from the install directory it updates `tools/`, and from `tools/` it updates
  that directory. Nothing about the script's own location matters, and there
  is no config file; llama-packer's `profiles.yaml` holds engine paths.
  Every run prints the directory it resolved for each engine and why;
  `--print-paths` prints that and downloads nothing, `--plan` shows the link
  or release decision, and `--dir <engine>=<path>` pins one for a single run.
  `$AUDIOCPP_BIN_DIR` and its equivalents work here too: the same variables
  llama-packer reads. There is no separate config file; llama-packer's
  `profiles.yaml` is where engine paths live.
  Each engine keeps a versioned entry beside a stable symlink
  (`llama-server` -> `./llama-b12345/llama-server`, `llama-swap` ->
  `llama-swap-v266`, `whisper-server` -> `./whisper-server-b4938/whisper-server`,
  `audiocpp_server` -> `audiocpp_server-v0.9.2`), so updating and rolling back
  are both a one-line symlink change. llama-packer is not this tool's job:
  `install.py` installs it as a package (a snapshot, not an editable link) and
  reaches it through `tools/llama-packer`, and the
  updater is copied to `tools/update` by the installer, so a deployment keeps
  working without the checkout. Run `./extras/update.py
  --help` for options. Pure stdlib Python (no venv/deps), so it runs on a bare
  box too.
  **Targets are positional selectors** — `llama`/`ll`, `cpp`, `swap`, `audio`/`a`,
  `whisper`/`w`, `sd`/`stable`, and `all` for everything. Globs
  work (`s*`), and an ambiguous prefix is an error rather than a guess. `all` is
  matched in full only, because `a` is an alias for audio.cpp — otherwise the
  one-letter shorthand would be ambiguous. A bare `./extras/update.py` only
  links what the service account can execute; `--fetch` downloads and updates
  **llama.cpp and llama-swap**; `all` adds audio.cpp, whisper.cpp and
  stable-diffusion.cpp.
  Only llama-swap publishes a `checksums.txt`; every other target says plainly
  that the download was authenticated by its immutable tag URL over TLS and
  nothing more. `--require-checksums` refuses an archive nobody checksummed.
  Backends are positional too (`vulkan`, `rocm`, `cpu`); llama.cpp takes them as
  part of its legacy version token. Per-project quirks are handled: audio.cpp
  follows llama.cpp's asset naming and ships binaries on its semver tag; whisper.cpp
  publishes a semver tag *and* an asset-less build tag per release (only `bNNNN`
  carries binaries) and ships a **bundle** whose `.so` files travel with the binary
  via `RUNPATH=$ORIGIN`; stable-diffusion.cpp versions as `master-N-<sha>`, ships
  **zips** rather than tarballs, spells the arch `x86_64`, is also a bundle, and
  has both Vulkan and ROCm Linux builds.
  **Release selection is by channel and date.** llama.cpp and whisper.cpp use
  the build channel (their `bNNNN` releases are GitHub prereleases); the rest
  use the stable channel and never take a prerelease. Drafts and releases whose
  name or tag says experimental / do not use / unusable / broken are skipped,
  and candidates are ordered by `published_at`, because the API list is not
  reliably date-ordered. A pinned version is installed as asked, with a warning.
  The CPU architecture is auto-detected for all downloads
  (llama.cpp/audio.cpp/whisper `x64`/`arm64`, stable-diffusion `x86_64`/`aarch64`;
  llama-swap `linux_amd64`/`linux_arm64`); `--arch` overrides it. Run it from
  the install directory (or from `tools/`): the target comes from there, never
  from where the script lives. Superseded versions are pruned after
  each install — the newest 3 `llama-b*/` dirs (`--keep-cpp`) and the newest 2
  versioned entries per engine (`--keep-swap N`, `--keep-engine-versions N`);
  anything a symlink still points at is kept, and so is the migration `.bak`
  for the run that created it. The same pass removes interrupted `.stage-*`
  directories, all but the newest `_HASH_FAIL_*` and a `.bak` whose stable name
  is already a symlink, and it runs on a no-op run too.
  **llama-swap downloads are sha256-verified against the release
  `checksums.txt`** before anything is installed, and a mismatch
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
