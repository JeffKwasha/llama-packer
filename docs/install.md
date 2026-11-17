# Installing llama-swap under systemd

`extras/install.py` creates the service account, installs the binaries,
regenerates `config.yaml`, writes a hardened systemd unit, and starts it.
`<service-user>` and `<service-group>` are yours to choose: the account is
created if it does not exist, the group must already exist.

```sh
sudo ./extras/install.py --user <service-user> --group <service-group> \
  --models-dir /mnt/ai/models
```

Idempotent: it refreshes the binaries, the config and the unit;
it never recreates the user and never drops an existing config.

## What it does

| Step | Detail |
|---|---|
| Preflight | root, systemd, at least one GPU device node |
| Account | `useradd --system --no-create-home --shell /usr/sbin/nologin` if absent; added to `render`, `video` and `--group` |
| Run directory | `<dir>` as `root:<group>` mode `2775` |
| Profiles | `<dir>/profiles.yaml` created from the bundled default (or `--profiles`) **only when absent**; never overwritten |
| Binaries | `extras/update.py all --fetch` with the run directory as cwd; it fills `<dir>/tools` and updates a flat install in place |
| llama-packer | installed as a package from `--packer-source` (default: this checkout) into the environment named by `--python` (default: the installer's), a snapshot rather than an editable link |
| The updater | copied to `<dir>/tools/update`, one of the deployment's tools |
| Config | llama-packer run **as the service user** from `<dir>`, writing `<dir>/config.yaml` and the sibling `config.env` |
| Agent guide | `docs/profiles-authoring.md` copied to `<dir>/.opencode/skills/profiles-authoring/SKILL.md` (default; `--no-agent-guide` skips it) |
| Unit | [`extras/llama-swap.service.in`](../extras/llama-swap.service.in) rendered, `systemd-analyze verify`-ed, written to `/etc/systemd/system/llama-swap.service` |
| Start | `llama-swap -validate` first, then `systemctl enable --now`, then a `/v1/models` probe |

Existing `config.yaml` and the installed unit are copied to `<name>.bak-N`
before being overwritten — both are gitignored and machine-local, so there
is no other way back.

Useful flags: `--dir` (run directory), `--listen` (default
`127.0.0.1:8123`), `--spare`/`--vram` (passed to llama-packer), `--profiles`
(source copied only when `<dir>/profiles.yaml` is absent),
`--no-agent-guide`, `--dry-run`, `--no-start`,
`--memory-high/--memory-max/--swap-max`.

## `profiles.yaml` is yours

The installer creates `<dir>/profiles.yaml` from the bundled
`llama_packer/profiles.yaml` when the run directory has none, owned by the
service account and group-writable. It **never overwrites an existing
file** — your edits survive reinstalls. `--profiles <file>` chooses a
different source for that first copy.

Edit `<dir>/profiles.yaml` for this host, not the repo copy:

- `models_dirs`, `roles.<name>.dir`, `hf_home` — where the models are;
- `backends` — which engines this host enables;
- `defaults.spare` — the headroom for the desktop/other tenants; measure it
  with `python -m llama_packer.vram_log --once` (or `--report` after
  logging). `hardware.vram`/`baseline_mb`/`memory_margin` are the lower-level
  overrides; `baseline_mb` only matters for a known fixed load above the
  1 GiB video floor;
- `defaults.parallel` — leave it unset; a value here is a pin and disables
  auto-parallel for every model;
- `roles.<name>.device: cpu` / `device: cpu` — what runs from host RAM (SPEC §Device);
- `roles` — the purposes the fleet uses, with their allocation policy;
- `tasks` — the task ids the fleet uses; `min_context` and `min_parallel`
  are floors (leave a key undefined for no floor), `max_context` is a
  ceiling, `device` is a placement requirement, and `ttl` sets the idle
  lifetime;
- `matrix.sets` — writing it switches the config to matrix mode; leaving it
  out keeps capacity;
- `capacity` — `pp_tps` / `drive_mbps` for derived eviction cost;
- `pools` — per-pool VRAM/spare overrides.

A CPU-placed serving follows the same TTL rules; the top-level default is
3600 s. Use `ttl: 0` to keep it
warm or an explicit number. A task floor the budget cannot meet emits no
entry rather than a serving below it.

Then re-pack and validate:

```sh
cd <dir>
sudo -u <service-user> llama-packer --profiles profiles.yaml \
  --output config.yaml --spare <same-as-install>
llama-swap -config config.yaml -validate
```

`docs/profiles-authoring.md` is the full authoring guide. The installer
also places it at `<dir>/.opencode/skills/profiles-authoring/SKILL.md`, so
an agent started in `<dir>` can load it; `--no-agent-guide` skips that.

## After a llama-packer update, re-check the profiles syntax

`extras/update.py` updates **binaries**, not `profiles.yaml`. A new
llama-packer can retire a key (for example `dirs:` or `device_roles:`) and
the packer warns about alien keys during a run. After updating:

```sh
cd <dir>
sudo -u <service-user> llama-packer --profiles profiles.yaml --check
sudo -u <service-user> llama-packer --profiles profiles.yaml \
  --output /tmp/config.yaml --spare <same-as-install> -v
```

Read the warnings, migrate the named keys using `docs/profiles-authoring.md`,
then re-pack for real and `-validate`. Do not assume an old profiles file is
still valid just because the binaries updated cleanly.

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
sudo ./extras/install.py --user <service-user> --group <service-group> \
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
2. `profiles.yaml <engine>.bin` (for llama.cpp, `llama_server.bin`)
3. `$SD_BIN_DIR` / `$WHISPER_BIN_DIR` / `$AUDIOCPP_BIN_DIR` / `$VLLM_BIN_DIR`,
   or `$LLAMA_BIN_DIR`
4. **the run directory's `tools/`**, then the directory itself
5. `PATH`

Because the installer runs llama-packer *with the run directory as its cwd*,
step 4 resolves everything placed in `tools/` by `update.py all --fetch`. No `profiles.yaml`
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
context, id and capacity allocation can shift. Pass the same `--spare` the current
config was built with to reproduce it.

## The updater installs the engines

`extras/update.py` owns downloading, sha256 verification, installation and
pruning for the five engines. It does not install llama-packer: that is a
package install, done by `install.py` (or by hand, see "Installing
llama-packer"). `--binaries skip` leaves the binaries alone if you have
already populated the directory. `install.py` runs the updater with the run
directory as its working directory and no `--dir`, so a fresh deployment
fills `<dir>/tools` and an engine already installed flat in `<dir>` is
updated in place rather than duplicated.

Two modes, and the default is the quiet one.

**Link (default).** No network, no download. Each selected target is pointed
at what is already installed in the run directory, and only if the service
account can execute it (`--user`, default the invoking user). Nothing
installed means nothing to do, and the run says so. `tools/update` after an
install is this mode, so a repeated run touches nothing it does not have to.

**Fetch (`--fetch`).** The install/upgrade pipeline, three phases, all five
targets:

1. **Plan.** Every selected target is resolved against GitHub *before*
   anything is downloaded. The plan names each target's current and candidate version,
   the download size, and where it will land. `--plan` stops here.
2. **Install.** Download (with a live progress bar on a terminal), verify
   the hash when the release publishes one, extract into a staging directory
   on the destination filesystem, then promote by rename.
3. **Tidy and report.** The stable symlink moves, superseded versions and
   the leftovers of interrupted runs are removed, and the summary says what
   was installed and what was already current. Every removal is printed as
   it happens; the tool does not total up bytes it reclaimed.

**Release selection is by channel and date, not by list order.** Each target
declares a channel: the upstream *stable* line (llama-swap, audio.cpp,
stable-diffusion.cpp) or the *build* line (llama.cpp's `bNNNNN`, whisper.cpp's
`bNNNN`). Stable targets never take a GitHub prerelease. Build targets do,
because those projects mark every build prerelease while publishing the
binaries only there. Independently of the channel, a release whose name or tag
says **experimental**, **do not use**, **unusable** or **broken** is skipped
(llama-swap ships several such tags), drafts are skipped, and candidates are ordered by
`published_at` — the API list is not reliably date-ordered. An explicitly
named version is installed as asked, with a warning when upstream marks it
experimental or prerelease.

**Pruning is one rule for every target.** The newest *N* versioned entries
per engine are kept (llama.cpp 3, everything else 2), anything a symlink still
points at is never removed, and the entries are ordered by their numbers
rather than their spelling. The same pass removes the files this tool creates
and then abandons: interrupted `.stage-*` directories, all but the newest
`_HASH_FAIL_*` quarantine, and the `.bak` copy once the stable name is a
symlink (the versioned history is the rollback then). Pruning runs on a no-op
run too, so a tree that drifted converges.

The stable names are derived from the tree, not from what the run intended:
each run points `<primary>` at the newest *intact* entry it finds. So a run
killed between the rename and the link is finished by the next one, instead
of leaving "already installed" permanently beside a stale link.

Two runs in one directory do not overlap: the second waits for the first
(`flock` on a per-directory lock file under `/tmp`, so nothing is added to
the tree). Without that, each run would reap the other's `.stage-*` as an
interrupted leftover.

That does mean **an update needs the GitHub API** (and `GITHUB_TOKEN` for
the rate limit). There is deliberately no systemd timer for it. A new
llama.cpp build changes the binary that VRAM constants were measured
against — `derived: shape` records the flag string, not the build — so
binaries moving without a re-pack would leave the config silently serving
with a stale compute term. `-watch-config` watches the config only, so the
swap would not even be visible as a reload. It also does not migrate
`profiles.yaml`; after a llama-packer update, re-run the packer and read its
key warnings (see "After a llama-packer update"). If you want this
automated, the safe shape is a timer that updates **and** re-packs together,
and refuses the swap when `llama-swap -validate` rejects the result; that is
a separate piece of work from this installer.

## The updater command line

`extras/update.py` takes its targets as positionals and its switches as
flags. The rules below are the ones the tool's tests hold up.

- **The tool's own help lists the targets, the flags and the precedence**, and
  an unknown option or a bad value stops the run rather than being ignored.
- **Targets are positional, first match wins.** `llama`/`ll`, `cpp`,
  `swap`, `audio`/`a`, `whisper`/`w`, `sd`/`stable`, and `all`.
  `all` must be typed in full — a glob works too, and `s*` selects `swap`
  and `sd`.
- **No target means llama.cpp and llama-swap**: the serving stack. `all`
  adds audio.cpp, whisper.cpp and stable-diffusion.cpp.
- **By default nothing is downloaded.** The run links what is already
  installed and prunes, and the summary says `linked` rather than
  `installed`. `--fetch` downloads the newest release.
- **`--user ACCOUNT`** is the account whose execute permission decides
  whether a tool may be linked. The check runs as that account, so a binary
  that exists but is root-only is reported and left unlinked. When the check
  cannot be made (no `runuser`, not root, no such account) the run says which
  and links nothing, rather than assuming the answer.
- **A bare `v<N>` is a llama-swap version; a `bNNNN` is a llama.cpp build.**
  The other engines are pinned with `--audio-cpp-version`,
  `--whisper-version` and `--sd-version`, because their tags are not
  distinguishable from a target name. `--cpp-only` ignores a swap version,
  and `--swap-only` ignores a build number.
- **`--swap-only` and `--cpp-only` are mutually exclusive**, and a selector
  together with one of those flags is refused rather than guessed at.
- **When two llama.cpp versions are given, the last one wins**, and a
  positional that is not a target is left to the version parser.
- **`--plan` (also spelled `--dry-run`) shows what the run would do and
  stops**: with `--fetch` the current → candidate versions, size and
  destination; without it, the link decision. `--print-paths` is the narrower question and stops at the
  directory.
- **`--keep-cpp N`, `--keep-swap N` and `--keep-engine-versions N`** keep
  that many old versions. A non-positive count is refused;
  `KEEP_SWAP_VERSIONS` and `KEEP_ENGINE_VERSIONS` are the environment
  equivalents.
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
- **`checksums.txt` is verified when the release publishes one.** llama-swap
  is the only target that does; the rest install from an immutable tag URL
  over TLS, and the install says plainly that the download was unverified
  rather than pretending to a check it could not do.
- **`--require-checksums` refuses an archive the release did not checksum**,
  where the default install warns that the download was unverified.
- **The newest is kept by number, not by spelling**: `master-1000` outranks
  `master-999`, so a text sort would keep the older build and delete the
  newer one.
- **A versioned file that a symlink still points at is never removed** — that
  would leave a dangling link, and the previous release is what makes a
  rollback possible.

## Where an update installs

The location comes from where the command was run, never from where the
script lives. Run `tools/update` from the install directory and it updates
`<install>/tools`; run it from `tools` itself and it updates that directory.
Those are the same place, which is what lets the copied updater be moved or
renamed without breaking anything.

Inside that directory an engine is updated **where it already is**: a
`llama-b*/` build, an `audiocpp_server`, a `whisper-server`, and for the
hand-made layouts that predate `tools/`, one of those nested one or two
levels down. `tools/` first, then the install directory itself. Running it
from the install directory and from `tools/` produce the same result, because
they name the same directory; running it somewhere else installs into *that*
directory's `tools/`, which is why the deployment's copy is the one to run.
Where two directories in one root both hold the same engine, it warns and
updates the directory that holds them rather than picking one of the two.

Each engine keeps its history beside the current one and is reached through a
stable name, so moving forward or rolling back is a single symlink:

```
tools/llama-swap       -> llama-swap-v266
tools/llama-server     -> ./llama-b12345/llama-server
tools/whisper-server   -> ./whisper-server-b4938/whisper-server
tools/audiocpp_server  -> audiocpp_server-v0.9.2/audiocpp_server
tools/sd-server        -> ./sd-server-master-968-abc/sd-server
tools/llama-packer     -> <interpreter>/bin/llama-packer
tools/update                                a copy of extras/update.py
```

In a deployment that is the `tools/` subdirectory of the run directory:

```
/opt/llama/
├── config.yaml  config.env          generated, replaced on every pack
└── tools/                            engines, llama-packer and update.py
```

`llama-packer` and `update` have no versioned entries of their own.
llama-packer is an ordinary package install into the active interpreter — a
snapshot, not an editable link, so the checkout it came from is not needed
afterwards — and the launcher is a symlink to that interpreter's console
script. `update` is a byte-for-byte copy of `extras/update.py`, placed by the
installer as one of the deployment's tools. It is a copy rather than a link
on purpose: the deployment must keep working when the checkout moves or is
not on that host. A run of `tools/update` uses only the directory it was run
from, never the path it was installed from.

## Installing llama-packer

The installer does this for you, and it is also a one-line manual step:

```bash
uv pip install /path/to/llama-packer            # a checkout, as a snapshot
uv pip install git+https://github.com/JeffKwasha/llama-packer@main
```

Run them inside the environment that should own the package (`uv venv` first,
or an existing uv env). A snapshot, never `uv pip install -e`: an editable install leaves a pointer
into a development directory that must then survive forever. `install.py`
accepts `--packer-source PATH|URL` for either form, and links the resulting
console script into `<dir>/tools/llama-packer`. `--python` names the
environment that should own the package; a distro Python that refuses installs
needs an existing uv env or venv.

`extras/install.py` generates the config with that launcher when it runs:
`<dir>/tools/llama-packer --version` exiting zero is what "it runs" means, so
a deployment is packed by the checkout it pinned rather than by whatever the
operator's `PATH` resolves. A launcher that is absent, or that does not run,
is reported and the search continues with this checkout's `.venv`, then
`PATH`; `--packer PATH` skips the search.

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
audio=/path` for one run, or `$AUDIOCPP_BIN_DIR` and its equivalents — the
same variables llama-packer resolves with, so one export configures both
tools. An engine with no existing install goes to `$LLAMA_DIR`, then the run
directory's `tools/`. There is no config file: llama-packer reads
`profiles.yaml` for the same paths, and an engine somebody installed with
their own package manager stays under that tool's management.

## Engines not fetched by a bare run

`extras/update.py all` fetches llama.cpp, llama-swap, audio.cpp,
whisper.cpp and stable-diffusion.cpp, so with that run the run directory holds
every engine and nothing needs pointing anywhere else. A bare
`extras/update.py` has nothing to fetch by default; `--fetch` gets llama.cpp
and llama-swap, in which case the other engines are still configured by
path —
[docs/backends/audio-cpp.md](backends/audio-cpp.md) has the detail. Point at
them however you prefer:

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

The install runs llama-packer as the service user, from the run directory,
with the installed profiles file, so re-packing by hand should match:

```sh
cd /opt/llama
sudo -u <service-user> llama-packer --models-dir /mnt/ai/models \
  --profiles profiles.yaml --output /opt/llama/config.yaml
```

For the re-pack to land, the model tree must be writable by that user — the
installer warns when it is not. A tree that cannot be written is not fatal
(the solve still runs, on the cached estimator) but every measurement taken
during the re-pack is discarded, which shows up later as unexplained context
loss.

## Agent guide in the run directory

`docs/profiles-authoring.md` is a guide for agents editing
`<dir>/profiles.yaml`. The installer places it at
`<dir>/.opencode/skills/profiles-authoring/SKILL.md` by default, owned by
the service account; `--no-agent-guide` skips it.

OpenCode checks `<cwd>/.opencode/skills` and walks up to the project root, so
the agent must be started in `<dir>` (or below it). For a global install
that works from any directory, use
`~/.config/opencode/skills/profiles-authoring.md` instead.

For an existing install without the guide, copy it in:

```sh
mkdir -p <dir>/.opencode/skills/profiles-authoring
cp docs/profiles-authoring.md \
   <dir>/.opencode/skills/profiles-authoring/SKILL.md
```

This is optional: llama-swap itself does not read `.opencode/`.

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
