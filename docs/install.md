# Installing llama-swap under systemd

`extras/install.py` creates the service account, installs the binaries, 
regenerates `config.yaml`, writes a hardened systemd unit, and starts it.

```sh
sudo ./extras/install.py --user lemonade --group models --models-dir /mnt/ai/models
```

Idempotent: it refreshes the binaries, the config and the unit;
it never recreates the user and never drops an existing config.

## What it does

| Step | Detail |
|---|---|
| Preflight | root, systemd, at least one GPU device node |
| Account | `useradd --system --no-create-home --shell /usr/sbin/nologin` if absent; added to `render`, `video` and `--group` |
| Run directory | `<dir>` as `root:<group>` mode `2775` |
| Binaries | `extras/update.py all`, every engine pinned with `--dir <engine>=<dir>` |
| Config | llama-packer run **as the service user** from `<dir>`, writing `<dir>/config.yaml` and the sibling `config.env` |
| Unit | [`extras/llama-swap.service.in`](../extras/llama-swap.service.in) rendered, `systemd-analyze verify`-ed, written to `/etc/systemd/system/llama-swap.service` |
| Start | `llama-swap -validate` first, then `systemctl enable --now`, then a `/v1/models` probe |

Existing `config.yaml` and the installed unit are copied to `<name>.bak-N`
before being overwritten — both are gitignored and machine-local, so there
is no other way back.

Useful flags: `--dir` (run directory), `--listen` (default
`127.0.0.1:8123`), `--spare`/`--vram` (passed to llama-packer), `--dry-run`,
`--no-start`, `--memory-high/--memory-max/--swap-max`.

## Pass `--hf-home` if your models come from an HF cache

llama-packer resolves the HF cache root itself:

```
--hf-home > profiles.yaml hf_home: > $HF_HOME > ~/.cache/huggingface
```

The installer does **not** pick one for you. Under `sudo` the environment is
stripped by default, so `$HF_HOME` usually does not survive and the fallback
resolves against *root's* home — which quietly orphans every `hf_repo` model
(the kokoro TTS entries and anything else served from a hub snapshot) rather
than failing loudly.

Either pass it explicitly:

```sh
sudo ./extras/install.py --user lemonade --group models \
  --models-dir /mnt/ai/models --hf-home /mnt/ai/huggingface
```

or preserve the environment with `sudo -E`. The installer reads back the
`config.env` it just generated and prints the resolved `HF_HOME`, `LLAMA_DIR`
and `MODELS_DIR`, warning if `HF_HOME` is not a directory. That is deliberate:
a wrong value has no error, it just makes entries stop resolving.

## No per-engine configuration is needed

llama-packer resolves **every** host binary — `llama-server` and
`audiocpp_server`, `whisper-server`, `sd-server` alike — by the same chain,
highest first:

1. `--<engine>-server`
2. `profiles.yaml <engine>.bin`
3. `$SD_BIN_DIR` / `$WHISPER_BIN_DIR` / `$AUDIOCPP_BIN_DIR` / `$VLLM_BIN_DIR`
4. **the run directory** — the directory llama-packer is invoked from
5. `PATH`

Because the installer runs llama-packer *with the run directory as its cwd*,
step 4 resolves everything placed there by `update.py all`. No `profiles.yaml`
edit, no environment variable, no PATH change.

That step exists because llama.cpp has always been found by scanning the cwd
for `llama-b[0-9]*`, while the other engines fell back to `shutil.which()`,
which never looks at the cwd. One chain closes the gap in the direction
llama.cpp already worked, rather than giving each deployment its own
`*_BIN_DIR`. See SPEC.md "Host binary resolution".

## Two orderings that are load-bearing

**Binaries before config.** `find_bin_dir` resolves the llama binary
relative to the working directory, and the emitted `LLAMA_DIR` macro is
derived from wherever that binary was found. A config generated before the
binaries are in place is a config that points nowhere; a config generated
from the repo is a config that points *back at the repo*, leaving
everything in `<dir>` unused. So the updater runs first, and llama-packer
runs with `<dir>` as its cwd — which also makes the result self-referential,
naming the build it will actually run.

**`--spare` decides whether contexts move.** Regenerating re-solves the
whole fleet. Without `--spare` the budget comes from auto-detected VRAM,
which may differ from whatever produced the config being replaced — every
context, id and matrix set can shift. Pass the same `--spare` the current
config was built with to reproduce it.

## The updater is the installer for binaries

`extras/update.py` already owns downloading, sha256 verification and
pruning, so `install.py` runs it rather than keeping a second copy of that
logic in step. `--binaries skip` leaves the binaries alone if you have already
populated the directory. `install.py` passes an explicit `--dir` per engine so
a fresh deployment installs into *its* run directory rather than adopting
engines it finds in a parent — see below for how a hand-run update decides
where to put things.

That does mean **an update needs the GitHub API** (and `GITHUB_TOKEN` for
the rate limit). There is deliberately no systemd timer for it. A new
llama.cpp build changes the binary that VRAM constants were measured
against — `derived: shape` records the flag string, not the build — so
binaries moving without a re-pack would leave the config silently serving
with a stale compute term. `-watch-config` watches the config only, so the
swap would not even be visible as a reload. If you want this automated, the
safe shape is a timer that updates **and** re-packs together, and refuses
the swap when `llama-swap -validate` rejects the result; that is a separate
piece of work from this installer.

## The updater command line

`extras/update.py` takes its targets as positionals and its switches as
flags. The rules below are the ones the tool's tests hold up.

- **The tool's own help lists the targets, the flags and the precedence**, and
  an unknown option or a bad value stops the run rather than being ignored.
- **Targets are positional, first match wins.** `llama`/`ll`, `cpp`,
  `swap`, `audio`/`a`, `whisper`/`w`, `sd`/`stable`, and `all`. `all` must be
  typed in full — a glob works too, and `s*` selects `swap` and `sd`.
- **No target means llama.cpp and llama-swap**: the two that publish
  `checksums.txt`. `all` adds the three that do not.
- **A version positional belongs to the target it follows.** A bare `v<N>` is
  a llama-swap version; a `bNNNN` is a llama.cpp build. `--cpp-only` ignores
  a swap version, and `--swap-only` ignores a build number.
- **`--swap-only` and `--cpp-only` are mutually exclusive**, and a selector
  together with the legacy `--only` flag is refused rather than guessed at.
- **When two version positionals are given, the last one wins**, and a
  positional that is not a target is left to the version parser.
- **`--keep-swap N` / `--keep-engine-versions N`** keep that many old
  versions. A non-positive count is refused; `KEEP_SWAP_VERSIONS` and
  `KEEP_ENGINE_VERSIONS` are the environment equivalents.
- **`--arch`** overrides the detected architecture (`x64`/`amd64`/`linux_amd64`,
  `arm64`/`aarch64`/`linux_arm64`, `s390x`). A bad value is refused at parse
  time, and an architecture the tool does not know is refused rather than
  guessed.
- **`--dir engine=path`** settles a location for one run and overrides every
  configured and discovered one. A bad `--dir` stops the run before anything
  is downloaded.
- **`--print-paths`** prints where each engine would be updated and why, and
  downloads nothing.
- **An asset is chosen by matching a pattern** over the release's download
  URLs — the portable bundle, not the plain tarball, when the backend needs
  one.
- **A release tag the release list does not carry is an error**, not a
  silent skip.
- **`checksums.txt` is verified when the release publishes one.** When it does
  not, the install says plainly that the download was unverified rather than
  pretending to a check it could not do.
- **`--require-checksums` refuses an archive the release did not checksum**,
  where the default install warns that the download was unverified.
- **The newest is kept by number, not by spelling**: `master-1000` outranks
  `master-999`, so a text sort would keep the older build and delete the
  newer one.
- **A versioned file that a symlink still points at is never removed** — that
  would leave a dangling link, and the previous release is what makes a
  rollback possible.

## Where an update installs

An engine is updated **in the directory it is already installed in**, not in
the directory you happened to run the command from. `update.py` searches
outwards from the cwd for an existing install — a `llama-b*/` build, an
`audiocpp_server`, a `whisper-server`, and for the hand-made layouts that
predate it, one of those nested one or two levels down — and updates the one
it finds. Running it from `$HOME` and from the engine's own directory now
produce the same result, and running it from the wrong place no longer
installs a second copy of an engine you already have. Where two directories
in one root both hold the same engine, it says so and refuses to guess.

Each engine keeps its history beside the live one and is reached through a
stable name, so moving forward or rolling back is a single symlink:

```
tools/llama-swap       -> llama-swap-v266
tools/llama-server     -> ./llama-b12345/llama-server
tools/whisper-server   -> ./whisper-server-b4938/whisper-server
tools/audiocpp_server  -> audiocpp_server-v0.9.2
tools/sd-server        -> ./sd-server-master-968-abc/sd-server
```

In a deployment that is the `tools/` subdirectory of the run directory:

```
/opt/llama/
├── config.yaml  config.env          generated, replaced on every pack
└── tools/                            engines; update.py owns this tree
```

`llama-packer` looks in `tools/` before the run directory itself
(`llama_packer/utils.py:binary_search_dirs`), so a deployment can keep several
hundred megabytes of binaries out of the directory that holds its config, and
a run directory installed before this layout — binaries flat, as a repo
checkout still has them — keeps resolving unchanged.

`--print-paths` prints that resolution, with the reason for each choice, and
downloads nothing:

```bash
./extras/update.py --print-paths all
```

To settle a directory explicitly, in order of precedence: `--dir
audio=/path` for one run, an `[engines]` entry in `extras/engines.ini`, or
`$AUDIOCPP_BIN_DIR` and its equivalents — the same variables llama-packer
resolves with, so one export configures both tools. Engines with no existing
install and no configuration go to `[engines] root`, then `$LLAMA_DIR`, then
the cwd.

## Engines not fetched by a bare run

`extras/update.py all` fetches llama.cpp, llama-swap, audio.cpp,
whisper.cpp and stable-diffusion.cpp, so with that run the run directory
holds every engine and nothing needs pointing anywhere else. A bare
`extras/update.py` fetches only llama.cpp + llama-swap, in which case the
other engines are still configured by path — [docs/backends/audio-cpp.md](backends/audio-cpp.md)
has the detail. Point at them however you prefer:

```yaml
# profiles.yaml
audio_cpp:
  bin: /mnt/pool/data/tools/audiocpp/audiocpp_server
```

Resolution order: `--audio-cpp-server` > `profiles.yaml audio_cpp.bin` >
`$AUDIOCPP_BIN_DIR` (a file *or* a directory) > `audiocpp_server` on
`PATH`.

One interaction worth knowing: audio.cpp entries write their generated
`server.json` under `/tmp/llama-swap/`. The unit sets `PrivateTmp=true`,
which gives the service its own writable `/tmp` — correct, but it means the
path is not shared with the host, so a TTS entry cannot be debugged by
looking at that directory from a shell. `profiles.yaml audio_cpp.tmpdir`
overrides it if you need a shared location.

## Why the service user has no home

The unit sets `ProtectHome=true` and
`InaccessiblePaths=/root /home /boot /efi /srv /var/lib /var/spool`. A user
whose home lives under any of those finds it invisible at runtime — which
surfaces as a puzzling permission error inside a model process, not as a
startup failure. `useradd --no-create-home` makes the question moot, and a
nologin shell keeps the account out of interactive use.

This is why the unit does not take a "any existing user" option: supporting
one means rewriting `InaccessiblePaths` per host, and getting that wrong
fails quietly.

## Why the run directory is group-writable

llama-packer writes `config.yaml` and the `derived:` blocks that persist VRAM
measurements into the model sidecars. A fleet that cannot be re-packed rots,
so `<dir>` is setgid and group-writable: members of `--group` re-run
llama-packer and `-watch-config` picks the new config up without a restart.

**That group membership is a trust boundary, not a convenience.** The config
decides which commands llama-swap runs, so anyone in the group can choose
what this box executes as the service account. Keep the membership to
people who already have root.

The service itself never writes to `<dir>` — the unit runs with
`ProtectSystem=strict`.

## Running llama-packer afterwards

The install runs llama-packer as the service user, so re-packing by hand
should match:

```sh
sudo -u lemonade llama-packer --models-dir /mnt/ai/models --output /opt/llama/config.yaml
```

For the re-pack to land, the model tree must be writable by that user — the
installer warns when it is not. A tree that cannot be written is not fatal
(the solve still runs, on the cached estimator) but every measurement taken
during the re-pack is discarded, which shows up later as unexplained context
loss.

## Two things the unit discovers rather than hardcodes

**GPU nodes.** `DeviceAllow` names the nodes that exist on this host. A
pinned `/dev/dri/card0` is wrong the moment the card enumerates as `card1`,
and systemd refuses to start on a `DeviceAllow` that does not resolve. The
connector symlinks beside the real nodes (`card1-DP-1`) are filtered out.
The block uses `DevicePolicy=closed` rather than `PrivateDevices=true`:
verified on systemd 259, the private `/dev` behind `PrivateDevices` never
materialises the allowlisted DRI/kfd nodes (open → ENOENT), so the GPU
silently vanishes from every Vulkan backend; `closed` keeps the cgroup-v2
BPF device blocklist without touching `/dev`.

**Vendor environment.** `/dev/kfd` selects the ROCm variables
(`HIP_VISIBLE_DEVICES`, `ROCBLAS_USE_HIPBLASLT`), `/dev/nvidia*` the CUDA
ones. `GGML_CUDA_DISABLE_GRAPHS=1` is CUDA-path only and is not set on ROCm.

## One systemd trap worth knowing

systemd reads the entire rest of a directive line as its value, so

```ini
MemoryHigh=32G    # soft cap
```

parses as the limit `32G    # soft cap`, is invalid, and the directive is
**dropped without an error**. Prose belongs on its own comment line. The
template keeps this in a comment above the block, and a test asserts no
directive in it carries a trailing `#`.

For the same reason the install's `systemd-analyze verify` step reads its
verdict from the messages rather than the exit status: the command audits
the whole system and exits `0` for a directive it merely "ignored", while
exiting `1` for an unrelated broken timer. Messages are matched by unit name,
so a pre-existing problem elsewhere on the host cannot fail every install.

## After installing

```sh
systemctl status llama-swap
journalctl -u llama-swap -f
curl -s http://127.0.0.1:8123/v1/models | jq '.data[].id'
```

A bad repack is the failure mode to watch: llama-swap rejects a config with
an unknown model reference, and `-watch-config` degrades to the *previous*
config rather than failing. Stale ids, not an error. `llama-swap -config
/opt/llama/config.yaml -validate` is the check that catches it, which is why
the installer runs it before every start.
