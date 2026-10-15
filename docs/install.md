# Installing llama-swap under systemd

`extras/install.py` deploys a packed fleet: it creates the service account,
installs the binaries into a run directory, regenerates `config.yaml` with
llama-packer, writes a hardened systemd unit, and starts it.

```sh
sudo ./extras/install.py --user lemonade --group models --models-dir /mnt/ai/models
```

Re-running is safe. It refreshes the binaries, the config and the unit; it
never recreates the user and never drops an existing config.

## What it does

| Step | Detail |
|---|---|
| Preflight | root, systemd, at least one GPU device node |
| Account | `useradd --system --no-create-home --shell /usr/sbin/nologin` if absent; added to `render`, `video` and `--group` |
| Run directory | `<dir>` as `root:<group>` mode `2775` |
| Binaries | `llama-swap` + the highest-numbered `llama-b*` build from the repo (or `--binaries download` / `link` / `skip`) |
| Config | llama-packer run **as the service user**, writing `<dir>/config.yaml` and the sibling `config.env` |
| Unit | [`extras/llama-swap.service.in`](../extras/llama-swap.service.in) rendered, `systemd-analyze verify`-ed, written to `/etc/systemd/system/llama-swap.service` |
| Start | `llama-swap -validate` first, then `systemctl enable --now`, then a `/v1/models` probe |

Useful flags: `--dir` (run directory), `--listen` (default
`127.0.0.1:8123`), `--spare`/`--vram` (passed to llama-packer), `--dry-run`,
`--no-start`, `--memory-high/--memory-max/--swap-max`.

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
