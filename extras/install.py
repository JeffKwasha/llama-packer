#!/usr/bin/env python3
"""install — deploy llama-swap under systemd, rooted at a directory of choice.

    ./extras/install.py --user lemonade --group models
    ./extras/install.py --user lemonade --group models --dry-run

Creates the service user, lays out the run directory, fetches the
binaries (by running ``extras/update.py``, which owns that), regenerates
``config.yaml`` with llama-packer, writes a hardened unit derived from
``llama-swap.service.in``, and starts it.

Two properties of the generated unit drive most of the choices here.

**The service user has no home.** The unit sets ``ProtectHome=true`` and
``InaccessiblePaths=... /home /var/lib ...``, so a user whose home lives
under either would find it invisible at runtime — a class of failure that
shows up as a puzzling permission error inside a model process rather
than as a startup failure. ``useradd --system --no-create-home`` makes
the whole question moot.

**The run directory is group-writable, the service is not.** llama-packer
writes ``config.yaml`` and the ``derived:`` blocks that persist VRAM
measurements, and a fleet that cannot be re-packed is a fleet that rots.
So the directory is ``root:<group>`` mode 2775: members of *group* can
run llama-packer and overwrite the config (the service picks it up
through ``-watch-config`` without a restart), while the unit itself runs
with ``ProtectSystem=strict`` and writes nothing.

    That group membership is a trust boundary, not a convenience:
    anyone in it can rewrite the config, and the config is what
    decides which commands llama-swap runs.

Idempotent: re-running refreshes the binaries, the config and the unit,
and never recreates the user or drops an existing config.
"""

from __future__ import annotations

import argparse
import glob
import grp
import json
import logging
import os
import pwd
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("install")

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = Path(__file__).resolve().parent / "llama-swap.service.in"

#: Groups added to the service user so it can open the GPU. Not every host
#: has all three (a NVIDIA box has no ``render``), so each is conditional.
_GPU_GROUPS = ("render", "video")

#: Connector symlinks live beside the card nodes as ``card0-DP-1`` and
#: would pass a bare glob; only the bare nodes are GPU device handles.
_CARD_RE = re.compile(r"^card\d+$")

#: Template placeholder shape. Scoped to ALL-CAPS so systemd's own
#: ``SystemCallFilter=@system-service`` — an at-set name, not one of ours
#: — does not read as an unsubstituted token.
_TOKEN_RE = re.compile(r"@[A-Z][A-Z_]*@")


class InstallError(RuntimeError):
    """A step could not be completed; the message is the user-facing text."""


class Runner:
    """Runs commands, or just prints them when ``dry_run`` is set."""

    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run

    def run(self, argv: list[str | Path], *, check: bool = True,
            cwd: Path | None = None, env: dict | None = None,
            capture: bool = False) -> subprocess.CompletedProcess:
        shown = " ".join(str(a) for a in argv)
        if self.dry_run:
            print(f"  $ {shown}")
            return subprocess.CompletedProcess(argv, 0, "", "")
        logger.debug("$ %s", shown)
        res = subprocess.run([str(a) for a in argv], cwd=cwd, env=env,
                             capture_output=capture, text=True)
        if check and res.returncode:
            detail = (res.stderr or res.stdout or "").strip() if capture else ""
            raise InstallError(f"command failed ({res.returncode}): {shown}"
                               + (f"\n  {detail}" if detail else ""))
        return res


# ── discovery ───────────────────────────────────────────────────────────


def gpu_nodes(dri: Path = Path("/dev/dri"),
              kfd: Path = Path("/dev/kfd")) -> list[str]:
    """GPU device nodes this host actually has (render + card + kfd).

    Roots are parameters so the scan is testable against a fake ``/dev``
    instead of the real one.
    """
    nodes: list[str] = []
    if dri.is_dir():
        nodes += sorted(f"{dri}/{p.name}" for p in dri.glob("renderD*"))
        nodes += sorted(f"{dri}/{p.name}" for p in dri.glob("card*")
                        if _CARD_RE.match(p.name))
    if kfd.exists():
        nodes.append(str(kfd))
    return [n for n in nodes if Path(n).exists()]


def gpu_vendor() -> str | None:
    """``"amd"``, ``"nvidia"`` or ``None`` — what the device nodes imply."""
    if Path("/dev/kfd").exists():
        return "amd"
    if glob.glob("/dev/nvidia[0-9]*"):
        return "nvidia"
    return None


def newest_build(src: Path, pattern: str = "llama-b*") -> Path | None:
    """Highest-numbered ``llama-bNNNNN`` directory in *src*, or None.

    Numbered by the digits, not by name: ``llama-b9999`` must sort below
    ``llama-b11658``, which a plain lexicographic sort gets backwards.
    """
    best: tuple[int, Path] | None = None
    for d in src.glob(pattern):
        m = re.search(r"(\d+)$", d.name)
        if not (d / "llama-server").is_file():
            continue
        n = int(m.group(1)) if m else -1
        if best is None or n > best[0]:
            best = (n, d)
    return best[1] if best else None


def user_exists(name: str) -> bool:
    try:
        pwd.getpwnam(name)
        return True
    except KeyError:
        return False


def group_exists(name: str) -> bool:
    try:
        grp.getgrnam(name)
        return True
    except KeyError:
        return False


def user_supplementary(user: str) -> set[str]:
    """Names of the groups *user* belongs to, supplementary or primary.

    ``os.getgrouplist`` walks the group database the same way the kernel
    does, which is the only answer that matches what `id -nG` prints.
    Reading ``gr_mem`` off the primary group instead returns that
    group's *members*, not the user's memberships, and reports every
    group as missing.
    """
    try:
        entry = pwd.getpwnam(user)
        gids = os.getgrouplist(user, entry.pw_gid)
        return {grp.getgrgid(g).gr_name for g in gids}
    except KeyError:
        return set()


# ── steps ───────────────────────────────────────────────────────────────


def preflight(r: Runner) -> None:
    if os.geteuid() != 0 and not r.dry_run:
        raise InstallError("must run as root (or under sudo)")
    if not Path("/run/systemd/system").exists():
        raise InstallError("systemd is not running on this host")
    if not gpu_nodes():
        raise InstallError("no GPU device nodes found (/dev/dri, /dev/kfd) — "
                           "the unit's DeviceAllow would allow nothing")
    if not TEMPLATE.is_file():
        raise InstallError(f"unit template missing: {TEMPLATE}")


def ensure_user(r: Runner, user: str, group: str) -> None:
    """Create the service account if absent; add the GPU + config groups.

    ``--no-create-home`` is load-bearing, not tidiness: the unit hides
    ``/home`` and ``/var/lib``, so any home under them is invisible to the
    service. A nologin shell is the other half — nothing logs in as the
    account that serves models.
    """
    if user_exists(user):
        logger.info("user %s already exists (left alone)", user)
    else:
        logger.info("creating system user %s", user)
        r.run(["useradd", "--system", "--no-create-home",
               "--shell", "/usr/sbin/nologin", "--comment",
               "llama-swap service account", user])
    if not group_exists(group):
        # Fatal, not a warning: prepare_dir chowns the run directory to this
        # group, so a typo left the directory created as root:root 2775 --
        # setgid and group-writable, by root -- and then died on KeyError
        # from grp.getgrnam. Refuse before anything is created.
        raise InstallError(
            f"group {group!r} does not exist; create it first "
            f"(groupadd {group}) or pass an existing --group")
    for g in (*_GPU_GROUPS, group):
        if not group_exists(g):
            logger.warning("group %s does not exist; skipping", g)
            continue
        if g in user_supplementary(user):
            continue
        logger.info("adding %s to group %s", user, g)
        r.run(["usermod", "-aG", g, user])


def prepare_dir(r: Runner, dir_: Path, group: str) -> None:
    """Lay out the run directory as ``root:<group>`` mode 2775.

    Setgid so a regenerated ``config.yaml`` keeps the group (and with it
    the next operator's write access); group-writable so members can
    re-pack the fleet. The service itself never writes here — see the
    trust-boundary note in the module docstring.
    """
    if not dir_.exists():
        logger.info("creating %s", dir_)
        r.run(["mkdir", "-p", dir_])
    if not r.dry_run:
        os.chmod(dir_, 0o2775)
        os.chown(dir_, 0, grp.getgrnam(group).gr_gid)
    print(f"  {dir_}  root:{group}  2775 (setgid, group-writable)")


def install_binaries(r: Runner, dir_: Path, mode: str) -> None:
    """Put ``llama-swap`` and a llama.cpp build under *dir_*.

    Delegates to ``extras/update.py``, which already owns downloading,
    checksumming and pruning, and installs into its own cwd. Reimplementing
    that here would give two copies of the same logic to keep in step --
    so this step runs the updater rather than duplicating it.

    It runs *before* the config is generated, and that ordering is load
    bearing rather than cosmetic: ``find_bin_dir`` resolves the llama
    binary relative to the working directory, so a config generated
    before the binaries are in place is a config that points nowhere.
    """
    if mode == "skip":
        print("  binaries: skipped (--binaries skip)")
    else:
        logger.info("fetching binaries into %s via extras/update.py "
                    "(set GITHUB_TOKEN for the API rate limit)", dir_)
        r.run([sys.executable, str(REPO / "extras" / "update.py")], cwd=dir_)
    require_binaries(r, dir_)

def require_binaries(r: Runner, dir_: Path) -> None:
    """Fail early when the config would have nothing to point at.

    Checked here so the error names the missing piece, instead of surfacing
    as llama-packer's ``no llama-b#### directory found here`` from inside the
    config-generation step.
    """
    if newest_build(dir_) is None or not (dir_ / "llama-swap").exists():
        if r.dry_run:
            return
        raise InstallError(
            f"{dir_} has no llama-b* build with a llama-server and no "
            "llama-swap; run ./extras/install.py without --binaries skip, "
            f"or populate it with: (cd {dir_} && {REPO / 'extras' / 'update.py'})")


def backup(path: Path, dry_run: bool) -> Path | None:
    """Copy *path* aside to ``<name>.bak-<n>`` before overwriting it.

    An install replaces a working config and a working unit in one step.
    A regeneration can move every context (see :func:`generate_config`), so
    the previous pair has to be recoverable without a `git checkout` -- they
    are both gitignored, machine-local files with no other copy.
    """
    if dry_run or not path.exists():
        return None
    n = 1
    while (dst := path.with_name(f"{path.name}.bak-{n}")).exists():
        n += 1
    shutil.copy2(path, dst)
    logger.info("backed up %s -> %s", path, dst.name)
    return dst


def check_writable(r: Runner, path: Path, user: str, what: str) -> None:
    """Warn when *path* is not writable by *user*.

    llama-packer runs as the service user, and it persists ``derived:``
    blocks into the model sidecars. A model tree it cannot write is not
    fatal — the solve still runs, on the cached estimator — but every
    measurement it manages to make is thrown away, which is worth saying
    out loud rather than discovering later as mysterious context loss.
    """
    if r.dry_run or not path.exists():
        return
    res = subprocess.run(["runuser", "-u", user, "--", "test", "-w", str(path)],
                         capture_output=True)
    if res.returncode != 0:
        logger.warning("%s: %s is not writable by %s — measurements taken "
                       "during a re-pack will not persist", what, path, user)


def generate_config(r: Runner, user: str, dir_: Path, args: argparse.Namespace,
                    packer: str) -> None:
    """Run llama-packer as the service user to (re)write the config.

    As the service user rather than root, so the ``derived:`` blocks it
    persists into the model tree belong to the same account that serves
    them — and so nothing in the config-generation path needs root.

    Run from *dir_*, not the repo. ``find_bin_dir`` resolves the llama
    binary relative to the working directory, and the emitted ``LLAMA_DIR``
    macro is derived from wherever that binary was found — so generating
    from the repo produces a config that points back at the repo, and
    everything installed under *dir_* is ignored. Generating from *dir_*
    makes the config self-referential: it names the build it will run.
    """
    cmd = [packer, "--models-dir", *args.models_dir,
           "--profiles", str(args.profiles),
           "--output", str(dir_ / "config.yaml")]
    if args.spare:
        cmd += ["--spare", args.spare]
    if args.vram:
        cmd += ["--vram", args.vram]
    if args.hf_home:
        cmd += ["--hf-home", args.hf_home]
    if not args.spare:
        logger.warning("--spare not given: the budget is re-solved from "
                       "auto-detected VRAM, so contexts may differ from the "
                       "config being replaced (pass the previous --spare to "
                       "reproduce it)")
    logger.info("regenerating config as %s, from %s (this sizes every context)",
                user, dir_)
    print(f"  $ cd {dir_} && runuser -u {user} -- {' '.join(cmd)}")
    if r.dry_run:
        return
    backup(dir_ / "config.yaml", r.dry_run)
    env = dict(os.environ)
    # HF_HOME is deliberately NOT synthesized here. llama-packer resolves it
    # itself (`--hf-home` > profiles.yaml `hf_home` > `$HF_HOME` >
    # `~/.cache/huggingface`); inventing a value would be a fourth mechanism
    # over a chain that already exists, and under `sudo` — which strips the
    # environment — the obvious invention (Path.home()/".cache") resolves to
    # *root's* cache, quietly orphaning every hf_repo model. Pass --hf-home,
    # or run with `sudo -E`, and check the value reported below.
    res = r.run(["runuser", "-u", user, "--", *cmd], cwd=dir_, env=env,
                capture=True)
    for path in ("config.yaml", "config.env"):
        if not (dir_ / path).is_file():
            raise InstallError(f"llama-packer did not produce {dir_ / path}")
    report_resolved(dir_)


def report_resolved(dir_: Path) -> None:
    """Print the path macros the emitted config actually resolved to.

    Read back from ``config.env`` rather than predicted, because these are
    the values that decide whether the service can read the weights at all.
    A wrong HF_HOME is silent — the entries simply stop resolving — so it is
    worth showing rather than assuming.
    """
    try:
        text = (dir_ / "config.env").read_text()
    except OSError:
        return
    for line in text.splitlines():
        if line.startswith(("HF_HOME=", "MODELS_DIR=", "LLAMA_DIR=")):
            name, _, value = line.partition("=")
            print(f"    {name}={value}")
            if name == "HF_HOME" and not Path(value).is_dir():
                logger.warning("HF_HOME=%s does not exist — hf_repo models "
                               "will not resolve; pass --hf-home", value)


def render_unit(args: argparse.Namespace, dir_: Path) -> str:
    """Fill the unit template for this host.

    The hardening block comes from the template verbatim; what is
    substituted is only what cannot be known before the install: the
    account, the directory, the listen address, the GPU nodes that exist,
    and the vendor's device-selection variables.
    """
    vendor = gpu_vendor()
    if vendor == "amd":
        gpu_env = ["Environment=HIP_VISIBLE_DEVICES=0",
                   "Environment=ROCBLAS_USE_HIPBLASLT=1"]
    elif vendor == "nvidia":
        # GGML_CUDA_DISABLE_GRAPHS: the CUDA path builds CUDA graphs that
        # a hot-swap unload can trip over. Harmless elsewhere, but it is a
        # CUDA-backend switch, so it is not set on ROCm.
        gpu_env = ["Environment=CUDA_VISIBLE_DEVICES=0",
                   "Environment=GGML_CUDA_DISABLE_GRAPHS=1"]
    else:
        gpu_env = ["# no GPU vendor detected; relying on defaults"]

    nodes = gpu_nodes()
    subs = (
        ("@USER@", args.user),
        ("@GROUP@", args.group),
        ("@DIR@", str(dir_)),
        ("@LISTEN@", args.listen),
        ("@MEM_HIGH@", args.memory_high),
        ("@MEM_MAX@", args.memory_max),
        ("@SWAP_MAX@", args.swap_max),
        ("@GPU_ENV@", "\n".join(gpu_env)),
        ("@DEVICE_ALLOW@", "\n".join(f"DeviceAllow={n} rw" for n in nodes)),
    )
    # Directives only. The header prose names the placeholders it expects,
    # and substituting those mentions would corrupt the documentation
    # into a copy of this host's install.
    lines = []
    for ln in TEMPLATE.read_text().splitlines():
        if ln.lstrip().startswith("#"):
            lines.append(ln)
            continue
        for token, value in subs:
            ln = ln.replace(token, value)
        lines.append(ln)
    text = "\n".join(lines) + "\n"
    leftover = [ln for ln in lines if not ln.lstrip().startswith("#")
                and _TOKEN_RE.search(ln)]
    if leftover:
        raise InstallError(f"unsubstituted unit template tokens: {leftover}")
    return text


def install_unit(r: Runner, args: argparse.Namespace, dir_: Path) -> Path:
    unit = Path(args.unit)
    body = render_unit(args, dir_)
    print(f"  {unit}")
    for ln in body.splitlines():
        if ln.lstrip().startswith("#"):
            continue
        if ln.startswith(("User=", "Group=", "ExecStart=", "DeviceAllow=",
                          "Environment=", "WorkingDirectory=")):
            print(f"    {ln}")
    if r.dry_run:
        return unit
    backup(unit, r.dry_run)
    unit.parent.mkdir(parents=True, exist_ok=True)
    # Verify the candidate before it becomes the unit systemd reads. The
    # previous order wrote in place and then verified, so a rejected unit
    # was left on disk: not running, but re-read on the next daemon-reload
    # or reboot. A tmp file in the same directory makes the swap the rename
    # it should be, and keeps the failure mode "old unit still installed".
    # A valid unit name, not "x.service.tmp": systemd-analyze refuses to
    # even parse anything else ("Failed to prepare filename: Invalid
    # argument") and exits 0, so a broken unit would sail through
    # verification and then be renamed into place.
    tmp = unit.with_name(f"{unit.stem}-candidate{unit.suffix}")
    try:
        tmp.write_text(body)
        os.chmod(tmp, 0o644)
        verify_unit(tmp)
        os.replace(tmp, unit)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    logger.info("installed unit %s", unit)
    return unit


def verify_unit(unit: Path) -> None:
    """Run ``systemd-analyze verify`` and fail on any complaint about
    *this* unit.

    Two traps here, both found against a real host. The command audits
    the whole system and its exit status is the worst finding anywhere,
    so a single broken timer on the box would fail every install for a
    reason unrelated to the unit being written. And it exits **0** for a
    directive it merely "ignored" — which is exactly the shape of the
    bug this check exists to catch (`MemoryHigh=32G  # comment` is read as
    one malformed value and dropped without an error). So the verdict
    comes from the messages, scoped by unit name, and never from the
    exit code.
    """
    res = subprocess.run(["systemd-analyze", "verify", str(unit)],
                         capture_output=True, text=True)
    ours = [ln for ln in (res.stderr or "").splitlines()
            if ln.startswith(str(unit)) or f"{unit.name}:" in ln]
    if ours:
        raise InstallError("systemd rejected the unit:\n  " + "\n  ".join(ours))
    ignored = (res.stderr or "").strip().splitlines()
    if ignored:
        logger.warning("systemd reported %d problem(s) on unrelated units "
                       "(ignored): %s", len(ignored), ignored[0])


def start(r: Runner, unit: Path, dir_: Path, listen: str, user: str) -> None:
    """Validate the config, then enable and start the unit.

    Validation comes first on purpose: llama-swap rejects a config with an
    unknown model reference outright, and ``-watch-config`` degrades to the
    previous config rather than failing — so a bad repack shows up as
    silently stale ids, not as an error. Better to catch it here.
    """
    swap = dir_ / "llama-swap"
    config = dir_ / "config.yaml"
    if not r.dry_run and swap.is_file():
        # As the service user, for the same reason the repack is: the run
        # directory is group-writable by design, so root executing a binary
        # out of it lets any member of --group plant one and have root run
        # it. Unlink+rename needs only directory write permission, not
        # ownership of the file being replaced.
        res = r.run(["runuser", "-u", user, "--", str(swap), "-config",
                     str(config), "-validate"], capture=True, check=False)
        if res.returncode:
            raise InstallError(
                f"llama-swap rejected {config}:\n{(res.stderr or res.stdout).strip()}")
        print("  config validates")
    if not r.dry_run:
        r.run(["systemctl", "daemon-reload"])
        r.run(["systemctl", "enable", "--now", unit.name])
    else:
        print(f"  $ systemctl daemon-reload")
        print(f"  $ systemctl enable --now {unit.name}")
    probe(r, listen)


def probe(r: Runner, listen: str) -> None:
    """Report what the API is actually advertising after the start."""
    if r.dry_run:
        return
    host, _, port = listen.rpartition(":")
    url = f"http://{host or '127.0.0.1'}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as res:
            data = json.load(res)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise InstallError(
            f"service did not answer on {url}: {exc}\n"
            f"  journalctl -u llama-swap -n 50")
    models = data.get("data", [])
    print(f"  {url}: {len(models)} models")
    for m in models[:5]:
        print(f"    {m.get('id')}  ({m.get('name', '')})")
    if len(models) > 5:
        print(f"    ... and {len(models) - 5} more")


# ── entry point ─────────────────────────────────────────────────────────


def find_packer(explicit: str | None) -> str:
    """Locate the ``llama-packer`` entry point.

    Prefers the repo's own venv so the config is generated by the code in
    this checkout rather than by whatever is installed system-wide.
    """
    if explicit:
        return explicit
    venv = REPO / ".venv" / "bin" / "llama-packer"
    if venv.is_file():
        return str(venv)
    found = shutil.which("llama-packer")
    if found:
        return found
    raise InstallError(
        "llama-packer not found — install it (uv sync) or pass --packer PATH")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="extras/install.py",
        description="Install llama-swap under systemd, rooted at a run directory.",
        epilog=(
            "The service account is created --system --no-create-home because the\n"
            "unit hides /home and /var/lib; and the run directory is\n"
            "group-writable so members of --group can re-pack the fleet, which\n"
            "means anyone in that group can choose which commands run.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--user", required=True,
                   help="service account to create/run as (e.g. lemonade)")
    p.add_argument("--group", default=None,
                   help="group owning the run directory and allowed to "
                        "re-pack it (default: the user's primary group)")
    p.add_argument("--dir", type=Path, default=Path("/opt/llama"),
                   help="run directory: binaries, config.yaml, config.env, "
                        "and the unit's WorkingDirectory (default /opt/llama)")
    p.add_argument("--listen", default="127.0.0.1:8123",
                   help="llama-swap listen address (default 127.0.0.1:8123)")
    p.add_argument("--models-dir", nargs="+", required=True,
                   help="model roots passed to llama-packer (repeatable)")
    p.add_argument("--profiles", type=Path, default=REPO / "profiles.yaml",
                   help="profiles.yaml for llama-packer")
    p.add_argument("--spare", default=None,
                   help="VRAM to reserve for other tenants (passed through)")
    p.add_argument("--vram", default=None,
                   help="total VRAM override (passed through)")
    p.add_argument("--hf-home", default=None,
                   help="HF cache root for the config-generation run")
    p.add_argument("--packer", default=None,
                   help="llama-packer executable (default: this repo's venv)")
    p.add_argument("--binaries", choices=("update", "skip"), default="update",
                   help="run extras/update.py in the run directory first "
                        "(default), or leave the binaries already there")
    p.add_argument("--unit", default="/etc/systemd/system/llama-swap.service",
                   help="unit file path to write")
    p.add_argument("--memory-high", default="32G",
                   help="MemoryHigh= soft cap (default 32G)")
    p.add_argument("--memory-max", default="40G",
                   help="MemoryMax= hard cap (default 40G)")
    p.add_argument("--swap-max", default="1G",
                   help="MemorySwapMax= (default 1G)")
    p.add_argument("--no-start", action="store_true",
                   help="install everything but do not enable or start")
    p.add_argument("--dry-run", action="store_true",
                   help="print every change without making it")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Logs and the progress print share stdout: two streams would
    # interleave wrongly once stdout is a pipe and block-buffers.
    logging.basicConfig(level=logging.INFO, format="  %(message)s",
                        stream=sys.stdout)
    if not args.group:
        args.group = grp.getgrgid(pwd.getpwnam(args.user).pw_gid).gr_name \
            if user_exists(args.user) else args.user
    dir_ = args.dir.resolve()
    r = Runner(args.dry_run)

    try:
        preflight(r)
        print(f"\n== account ==\n  user={args.user} group={args.group}")
        ensure_user(r, args.user, args.group)

        print(f"\n== run directory ==\n  {dir_}")
        prepare_dir(r, dir_, args.group)

        print("\n== binaries ==")
        install_binaries(r, dir_, args.binaries)

        print("\n== config ==")
        for md in args.models_dir:
            check_writable(r, Path(md), args.user, "models dir")
        generate_config(r, args.user, dir_, args, find_packer(args.packer))

        print("\n== unit ==")
        unit = install_unit(r, args, dir_)

        print("\n== service ==")
        if args.no_start:
            print("  --no-start: not enabling or starting")
        else:
            start(r, unit, dir_, args.listen, args.user)
    except InstallError as exc:
        logger.error("%s", exc)
        return 1
    print("\ndone." if not args.dry_run else "\ndry run: nothing changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
