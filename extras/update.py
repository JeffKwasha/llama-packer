#!/usr/bin/env python3
"""update — download and install prebuilt server binaries.

Python port of the ``extras/update`` bash script (same CLI, same messages,
same install layout). Unlike the bash version this can never be "sourced"
harmlessly into a shell (the ``. extras/update --help`` habit that used to
close the terminal): a Python script always runs as its own process.

**Where things install.** Every engine has a home directory, and this tool
updates each engine *where it already is* rather than wherever it happens to
be run from — see :func:`resolve_install_dirs`. One flat directory holds all
of them, each as a versioned entry plus a stable symlink:

    tools/llama-swap       -> llama-swap-v266
    tools/llama-server     -> ./llama-b12345/llama-server
    tools/whisper-server   -> ./whisper-server-b4938/whisper-server
    tools/audiocpp_server  -> audiocpp_server-v0.9.2

Old versions stay beside the new one (``--keep-swap`` / ``--keep-engine-versions``),
so rolling forward is a symlink change and rolling back is one too.

Usage: ./extras/update.py [target|backend|version ...] [options]

Targets are positional selectors; `all` must be typed in full.

    ./extras/update.py             # llama.cpp + llama-swap (the verified two)
    ./extras/update.py all         # every target, including the unverified three
    ./extras/update.py llama       # llama.cpp + llama-swap
    ./extras/update.py ll          # same, abbreviated
    ./extras/update.py cpp         # llama.cpp alone
    ./extras/update.py swap        # llama-swap alone
    ./extras/update.py audio       # audio.cpp -> audiocpp_server
    ./extras/update.py whisper     # whisper.cpp -> whisper-server
    ./extras/update.py sd          # stable-diffusion.cpp -> sd-server
    ./extras/update.py "s*"        # globs select a set (here: swap + sd)

Backends are positional, and apply to whichever targets offer one:

    ./extras/update.py sd rocm     # native ROCm build
    ./extras/update.py llama rocm

Versions still work, in the historical shapes:

    ./extras/update.py b8913       # Specific llama.cpp version + latest llama-swap
    ./extras/update.py vulkan      # Latest Vulkan build + latest llama-swap
    ./extras/update.py rocm        # Latest ROCm build + latest llama-swap
    ./extras/update.py --swap-only v252  # Specific llama-swap version
    ./extras/update.py --skip-swap  # Skip llama-swap update

audio.cpp, whisper.cpp and stable-diffusion.cpp publish no `checksums.txt`, so
those installs are unverified and say so; a bare run deliberately leaves them
alone.

Standard library only — no venv, no third-party deps.
exit 1
"""

from __future__ import annotations

import argparse
import configparser
import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
LLAMA_SWAP_REPO = "https://api.github.com/repos/mostlygeek/llama-swap"
KEEP_VERSIONS = 3  # Keep current + N-1 previous llama.cpp build dirs
DEFAULT_KEEP_SWAP_VERSIONS = 2  # Keep current + previous llama-swap binaries
DEFAULT_KEEP_ENGINE_VERSIONS = 2  # Same, for the single-binary engines
DEFAULT_VERSION = "latest-vulkan"

# One prebuilt per backend, and the asset name is the only thing that
# differs between them -- so one table replaces a branch per backend. Taken
# from the b11658 release: the CPU build carries no backend word at all
# ("bin-ubuntu-x64"), which is why "cpu" has to be spelled out here rather
# than derived from the other three. It does not collide with them:
# "ubuntu-vulkan-x64" does not contain "ubuntu-x64".
CPP_BACKENDS = {
    "vulkan": "ubuntu-vulkan-{arch}",
    "rocm": r"ubuntu-rocm-[^-]*-x64",
    "cuda": r"ubuntu-cuda-[^-]*-{arch}",
    "cpu": "ubuntu-{arch}",
}


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


class ApiError(RuntimeError):
    """GitHub API call failed with a message worth showing the user."""


# --- GitHub API ------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    # Optional token for higher API rate limits (60/hr unauth → 5000/hr auth)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get(
        "GITHUB_PERSONAL_ACCESS_TOKEN"
    ) or ""
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _parse_json(body: str):
    try:
        return json.loads(body)
    except ValueError:
        return None


def gh_api(url: str):
    """GET a GitHub API URL and return the parsed JSON body.

    Uses curl rather than urllib: this box reaches the network through a
    SOCKS proxy, which urllib cannot use. Raises ApiError on rate-limit/auth
    failures or an unreachable API; a JSON error body (e.g. 404 "Not Found")
    is returned so callers can detect it.
    """
    cmd = ["curl", "-sSL", "--max-time", "120"]
    for key, value in _auth_headers().items():
        cmd += ["-H", f"{key}: {value}"]
    cmd += ["-w", "\n%{http_code}", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    body, _, code_str = proc.stdout.rpartition("\n")
    status = int(code_str) if code_str.isdigit() else 0

    if status in (401, 403):
        payload = _parse_json(body) or {}
        message = payload.get("message", "") if isinstance(payload, dict) else ""
        if "rate limit" in f"{message} {body}".lower():
            raise ApiError(
                "Error: GitHub API rate limit exceeded (60 req/hr "
                "unauthenticated).\nSet GITHUB_TOKEN for 5000 req/hr: "
                "export GITHUB_TOKEN=ghp_xxxx"
            )
        raise ApiError(
            "Error: GitHub API authentication failed (bad or expired "
            "token).\nCheck GITHUB_TOKEN / GITHUB_PERSONAL_ACCESS_TOKEN."
        )
    if status == 0 or status >= 500:
        detail = f": {proc.stderr.strip()}" if proc.stderr.strip() else ""
        raise ApiError(f"Error: GitHub API request failed for {url}{detail}")

    payload = _parse_json(body)
    if payload is None:
        raise ApiError(f"Error: unexpected non-JSON response from {url}")
    return payload


def _releases(payload) -> list:
    """A GET on /releases gives a list; a tag lookup gives a dict."""
    return payload if isinstance(payload, list) else []


def _is_error_payload(payload) -> bool:
    """True when a tag lookup came back as an API error, not a release."""
    return isinstance(payload, list) or (
        isinstance(payload, dict) and "message" in payload
    )


def find_asset(payload, pattern: str) -> str:
    """First browser_download_url in a release whose URL matches ``pattern``."""
    if not isinstance(payload, dict):
        return ""
    for asset in payload.get("assets", []):
        url = asset.get("browser_download_url", "")
        if re.search(pattern, url):
            return url
    return ""


def latest_llama_cpp_tag(asset_pattern: str = "") -> str:
    """Newest llama.cpp tag that actually carries a prebuilt Linux asset.

    llama.cpp publishes two tags per release: a semver one (``v0.7.0``) holding
    source only, and a build one (``b11669``) holding the tarballs — both on
    the same day. Taking the newest tag outright returns the semver one, and
    the asset lookup then fails with "Could not find Vulkan asset" against a
    perfectly good release sitting directly below it.
    """
    probe = asset_pattern or r"bin-ubuntu-[^-]*\.tar\.gz$"
    for release in _releases(gh_api(RELEASES_URL)):
        if release.get("tag_name") and find_asset(release, probe):
            return release["tag_name"]
    return ""


def latest_llama_swap_tag() -> str:
    for release in _releases(gh_api(f"{LLAMA_SWAP_REPO}/releases")):
        if release.get("tag_name"):
            return release["tag_name"]
    return ""


def release_assets(url: str):
    return gh_api(url)


# --- architecture ----------------------------------------------------------


class ArchError(ValueError):
    """An --arch spelling that is not in the table."""


def _normalize(override: str, mapping: dict[str, str]) -> str:
    """Map an arch alias to its asset spelling, or refuse the input.

    Returning the input unchanged on a miss looked harmless -- the arch is
    operator-supplied, so nothing untrusted reaches it -- but the value goes
    straight into ``re.search``, where ``--arch 'x64.*'`` matches the first
    asset of any name and a pathological pattern can hang the lookup. Worse,
    it failed open: an unrecognised spelling became a *successful* lookup of
    the wrong tarball, which is the one outcome a wrong --arch must never
    have.
    """
    try:
        return mapping[override]
    except KeyError:
        raise ArchError(
            f"unknown arch {override!r}; expected one of: "
            + ", ".join(sorted(mapping))) from None


#: Alias -> asset spelling, one table per upstream naming convention. They
#: genuinely differ (llama.cpp says x64, llama-swap says linux_amd64,
#: stable-diffusion.cpp says x86_64), so they stay separate — but they are
#: named so :func:`arch_arg` can validate against all of them at parse time
#: rather than each resolver rediscovering a bad value later.
ARCH_SWAP = {
    "linux_amd64": "linux_amd64", "x86_64": "linux_amd64",
    "amd64": "linux_amd64", "x64": "linux_amd64",
    "linux_arm64": "linux_arm64", "aarch64": "linux_arm64",
    "arm64": "linux_arm64",
}

#: llama.cpp assets use x64 / arm64 / s390x, e.g.
#: llama-b<int>-bin-ubuntu-vulkan-arm64.tar.gz
ARCH_CPP = {
    "linux_amd64": "x64", "x86_64": "x64", "amd64": "x64", "x64": "x64",
    "linux_arm64": "arm64", "aarch64": "arm64", "arm64": "arm64",
    "s390x": "s390x",
}

#: stable-diffusion.cpp spells it x86_64 / aarch64.
ARCH_SD = {
    "linux_amd64": "x86_64", "x86_64": "x86_64", "x86-64": "x86_64",
    "amd64": "x86_64", "x64": "x86_64", "linux_arm64": "aarch64",
    "aarch64": "aarch64", "arm64": "aarch64",
}


def arch_arg(value: str) -> str:
    """argparse type for --arch: accepted only if some resolver knows it.

    Validating at parse time is the point. A bad --arch used to survive
    every early return and only surface much later -- or, worse, not at all,
    because the version was already current.
    """
    if not value:
        return value          # "" means detect from this machine
    if value in ARCH_CPP or value in ARCH_SWAP or value in ARCH_SD:
        return value
    raise argparse.ArgumentTypeError(
        f"unknown arch {value!r}; expected one of: "
        + ", ".join(sorted(set(ARCH_CPP) | set(ARCH_SWAP) | set(ARCH_SD))))


def get_llama_swap_arch(override: str = "") -> str:
    if override:
        return _normalize(override, ARCH_SWAP)
    machine = platform.machine()
    if machine in ("x86_64", "amd64"):
        return "linux_amd64"
    if machine in ("aarch64", "arm64"):
        return "linux_arm64"
    return "linux_amd64"


def get_llama_cpp_arch(override: str = "") -> str:
    if override:
        return _normalize(override, ARCH_CPP)
    machine = platform.machine()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine == "s390x":
        return "s390x"
    return "x64"


# --- install location ------------------------------------------------------

#: ``extras/engines.ini``, next to this file. Committed, per-deployment, and
#: entirely optional: an absent or empty file means "work it out" (below).
CONFIG_NAME = "engines.ini"

#: Every engine installs the same way once its directory is known: a versioned
#: entry (a file for llama-swap/audio.cpp, a directory for everything that
#: ships shared libraries) plus a stable *primary* name beside it. This table is
#: what makes that discoverable, and it is also the shape the user-facing
#: layout promises:
#:
#:     tools/llama-swap          -> llama-swap-v266
#:     tools/llama-server        -> ./llama-b12345/llama-server
#:     tools/whisper-server      -> ./whisper-server-b4938/whisper-server
#:     tools/audiocpp_server     -> audiocpp_server-v0.9.2
#:
#: ``env`` is the ``*_BIN_DIR`` variable llama-packer already honors for that
#: engine, reused here so one variable configures both tools.
INSTALL_UNITS: dict[str, "InstallUnit"] = {}

#: llama.cpp's stable name, spelled once because both the symlink maintenance
#: and the packer's binary resolution have to agree on it.
CPP_PRIMARY = "llama-server"


@dataclass
class InstallUnit:
    """How one engine's installs are recognized on disk."""

    key: str
    primary: str                  # stable name; always a symlink after install
    versioned: tuple[str, ...]    # globs for this engine's versioned entries
    env: str = ""                 # *_BIN_DIR, shared with llama-packer

    def _entries(self, directory: Path) -> list[Path]:
        """One listing of *directory*, or ``[]`` if it cannot be read.

        Every predicate below reads this rather than globbing: a dozen
        ``glob()`` calls over a tools root that holds thousands of unrelated
        entries is slow, and a predicate that has to be re-derived from the
        filesystem at each step is one whose answer can differ between the
        check and the use.
        """
        try:
            if not directory.is_dir():
                return []
            return list(directory.iterdir())
        except OSError:
            return []

    def _holds_directly(self, names: list[str]) -> bool:
        """Whether *names* shows an install in this directory itself.

        The strict half of :meth:`_holds`: no looking inside subdirectories.
        It is what separates "this root *is* the engine's home" from "this root
        merely *contains* the engine's home".
        """
        return any(fnmatch(name, pattern) for name in names
                   for pattern in self.versioned) or self.primary in names

    def _holds(self, names: list[str], directory: Path) -> bool:
        """Whether a listing called *names* shows an install of this engine.

        Three shapes, in decreasing order of how this tool writes them:

        * a versioned entry beside the primary name (its own layout),
        * the primary name itself,
        * the primary name one or two levels down — the hand-made layouts
          that predate this tool, such as ``whisper/bin -> whisper-v1.9.2/bin/``
          or a source-build tree.

        The looseness is deliberate and one-directional: a false positive
        costs an update that lands in a directory that already looks like this
        engine's, while a false negative costs a *second copy of the engine*
        installed beside the first — which is the whole problem this layer
        exists to remove.
        """
        if any(fnmatch(name, pattern) for name in names
               for pattern in self.versioned):
            return True
        if self.primary in names:
            return True
        for name in names:
            child = directory / name
            if not child.is_dir():
                continue
            if (child / self.primary).exists():
                return True
            try:
                if any((grandchild / self.primary).exists()
                       for grandchild in child.iterdir()
                       if grandchild.is_dir()):
                    return True
            except OSError:
                continue
        return False

    def holding_children(self, directory: Path) -> list[Path]:
        """Subdirectories of *directory* that each hold an install of this
        engine — the ``tools/audiocpp`` + ``tools/old_audiocpp`` case."""
        return [e for e in self._entries(directory)
                if e.is_dir()
                and not any(e.match(p) for p in self.versioned)
                and self._holds([c.name for c in self._entries(e)], e)]

    def descends_into(self, directory: Path) -> Path | None:
        """The engine's own subdirectory of *directory*, when it has one.

        The fleet shape is one directory per engine (``tools/whisper``,
        ``tools/audiocpp``) sitting inside a tools root. Matching only the
        root would install a second copy *beside* the first one — the very
        duplication this layer exists to prevent — so when the root holds no
        install of its own but exactly one of its children does, that child is
        the home.

        Never descends into a versioned entry: ``llama-b11658/llama-server``
        would otherwise read as "the cpp home is the build directory", and the
        next update would install build 12345 *inside* build 11658.
        """
        entries = self._entries(directory)
        names = [e.name for e in entries]
        if not names or self._holds_directly(names):
            return None
        children = self.holding_children(directory)
        return children[0] if len(children) == 1 else None

    def lives_in(self, directory: Path) -> bool:
        """Whether *directory* already holds an install of this engine.

        See :meth:`_holds` for why "already holds" is read loosely, and
        :meth:`descends_into` for what happens when the install is one level
        down.
        """
        entries = self._entries(directory)
        if not entries:
            return False
        return self._holds([e.name for e in entries], directory)


for _key, _primary, _versioned, _env in (
    ("cpp", CPP_PRIMARY, ("llama-b[0-9]*",), "LLAMA_BIN_DIR"),
    ("swap", "llama-swap", ("llama-swap-v[0-9]*",), "LLAMA_SWAP_BIN_DIR"),
    ("audio", "audiocpp_server", ("audiocpp_server-v[0-9]*",), "AUDIOCPP_BIN_DIR"),
    ("whisper", "whisper-server", ("whisper-server-b[0-9]*",), "WHISPER_BIN_DIR"),
    ("sd", "sd-server", ("sd-server-master-*", "sd-server-v[0-9]*"),
     "SD_BIN_DIR"),
):
    INSTALL_UNITS[_key] = InstallUnit(
        key=_key, primary=_primary, versioned=_versioned, env=_env
    )


def config_path() -> Path:
    """Where ``engines.ini`` is looked for: beside this script."""
    return Path(__file__).resolve().parent / CONFIG_NAME


def load_engines_config(path: Path | None = None) -> tuple[Path | None, dict[str, str]]:
    """``(root, per-engine overrides)`` from ``engines.ini``; never raises.

    INI via :mod:`configparser`, not YAML: this script is pure stdlib so it
    runs on a box with no venv, and hand-rolling a YAML subset for five keys
    is a parser bug waiting to happen. A malformed file is reported and
    treated as absent -- a typo in the config must not stop someone from
    updating a binary.

    ``[engines] root`` is where new installs go; each other key overrides one
    engine, so an engine that already lives somewhere else keeps living
    there. An empty value means "unset" (the template ships that way).
    """
    path = path or config_path()
    parser = configparser.ConfigParser()
    try:
        if not path.is_file():
            return None, {}
        parser.read(path, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError) as exc:
        eprint(f"Warning: ignoring unreadable {path}: {exc}")
        return None, {}
    if not parser.has_section("engines"):
        return None, {}
    entries = {
        key.strip(): value.strip()
        for key, value in parser.items("engines")
    }
    root = entries.pop("root", "")
    for key in sorted(entries):
        if key not in INSTALL_UNITS:
            eprint(f"Warning: {path}: unknown engine {key!r} (known: "
                   f"{', '.join(ALL_TARGETS)})")
    engines = {k: Path(v).expanduser() for k, v in entries.items()
               if v and k in INSTALL_UNITS}
    return (Path(root).expanduser() if root else None), engines


def _env_dir(unit: InstallUnit) -> Path | None:
    """``$ENGINE_BIN_DIR`` as a directory, or None.

    The packer lets these name either a binary or a directory, so both are
    accepted here: a file contributes its parent, which is what an operator
    who exported the full binary path means by it. ``audiocpp_server`` has no
    suffix to go on, so "is it a file" decides rather than the spelling.
    """
    raw = os.environ.get(unit.env, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.is_dir():
        return path
    if path.is_file() or path.suffix:
        return path.parent
    return path


def _discovery_candidates() -> list[Path]:
    """Directories that plausibly already hold an install, in order.

    Ordered so the nearest answer wins: the cwd (where this tool used to
    install, and where anyone who learned the old habit is standing), then
    the repo directory itself, then its parent — which on this fleet is where
    every engine actually lives — then ``$LLAMA_DIR``. Duplicates are dropped
    so the summary does not print the same directory under two names.
    """
    here = Path.cwd()
    repo = Path(__file__).resolve().parent.parent
    candidates = [here, repo, repo.parent]
    if raw := os.environ.get("LLAMA_DIR", "").strip():
        candidates.append(Path(raw).expanduser())
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def resolve_install_dirs(
    overrides: dict[str, str] | None = None,
    config: tuple[Path | None, dict[str, str]] | None = None,
) -> dict[str, tuple[Path, str]]:
    """``{engine key: (directory, why)}`` for every engine.

    The whole point of this function: **update what is already installed,
    where it already is**. An engine found in a candidate directory is
    updated in that directory; only an engine with no existing install is
    placed under the configured root. That is the difference between running
    this from anywhere on the box and having to stand in five directories in
    turn.

    Precedence, highest first:

    1. ``--dir <key>=<path>`` — for one run, no config touched.
    2. ``[engines] <key>`` in ``engines.ini`` — the operator's declaration.
    3. ``$<KEY>_BIN_DIR`` — the same variable llama-packer resolves with, so
       one export configures both tools.
    4. **an existing install**, searched from the cwd outwards (§
       :func:`_discovery_candidates`).
    5. ``[engines] root``, then ``$LLAMA_DIR``, then the cwd (the historical
       behaviour, kept so a fresh install with no configuration at all still
       works).
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    root, per_engine = config if config is not None else load_engines_config()
    candidates = _discovery_candidates()
    resolved: dict[str, tuple[Path, str]] = {}
    for key in ALL_TARGETS:
        unit = INSTALL_UNITS[key]
        if key in overrides:
            resolved[key] = (Path(overrides[key]).expanduser(), "--dir")
            continue
        if key in per_engine:
            resolved[key] = (per_engine[key], "engines.ini")
            continue
        if env_dir := _env_dir(unit):
            resolved[key] = (env_dir, f"${unit.env}")
            continue
        for candidate in candidates:
            # Descend first: a root that merely *contains* the engine's own
            # directory (tools/ holding tools/audiocpp/) would otherwise be
            # read as the home, and the update would land beside the install
            # it was supposed to be updating.
            if inner := unit.descends_into(candidate):
                resolved[key] = (inner, "existing install")
                break
            rivals = unit.holding_children(candidate)
            if len(rivals) > 1:
                # Two directories each holding a copy: guessing which one is
                # live would put the update beside the engine it was meant to
                # replace. Say so and let the operator name it.
                eprint(f"Warning: {candidate} holds {len(rivals)} "
                       f"{key} installs ({', '.join(sorted(p.name for p in rivals))}); "
                       f"not guessing — set it with --dir {key}=<path> or "
                       f"[engines] {key} in {CONFIG_NAME}")
            if unit.lives_in(candidate):
                resolved[key] = (candidate, "existing install")
                break
        else:
            if root is not None:
                resolved[key] = (root, "engines.ini root")
            elif raw := os.environ.get("LLAMA_DIR", "").strip():
                resolved[key] = (Path(raw).expanduser(), "$LLAMA_DIR")
            else:
                resolved[key] = (Path.cwd(), "current directory")
    return resolved


def parse_dir_overrides(values: list[str] | None) -> dict[str, str]:
    """``["audio=/x", "bad"]`` -> ``{"audio": "/x"}``; a bad one is an error.

    Silent acceptance of a malformed override would install somewhere the
    operator did not name, so an unusable ``--dir`` stops the run before any
    download happens.
    """
    out: dict[str, str] = {}
    for raw in values or []:
        key, sep, path = raw.partition("=")
        key = key.strip()
        if not sep or not key or not path.strip():
            raise TargetError(
                f"--dir expects <engine>=<path>, got {raw!r} "
                f"(engines: {', '.join(ALL_TARGETS)})")
        if key not in INSTALL_UNITS:
            raise TargetError(
                f"--dir names unknown engine {key!r} "
                f"(engines: {', '.join(ALL_TARGETS)})")
        out[key] = path.strip()
    return out


def report_install_dirs(dirs: dict[str, tuple[Path, str]]) -> None:
    """Print where each engine will be updated, and how that was decided.

    Always printed, not only with ``--print-paths``: the reader's question
    about an installer is "where did it put it", and a tool that answers that
    only on request has already lost the argument.
    """
    print("\nInstall locations:")
    width = max((len(k) for k in dirs), default=4)
    for key in ALL_TARGETS:
        if key not in dirs:
            continue
        directory, why = dirs[key]
        primary = INSTALL_UNITS[key].primary
        print(f"  {key:<{width}}  {directory}  "
              f"[{primary}] ({why})")


# --- filesystem helpers ----------------------------------------------------


def _owned_by_current_user(path: Path) -> bool:
    """Safety gate: INSTALL_DIR must be owned by the current user."""
    try:
        return path.stat().st_uid == os.getuid()
    except OSError:
        return False


def highest_installed(install_dir: Path) -> int:
    """Highest installed llama.cpp build number among llama-b<NNNN> dirs."""
    highest = 0
    for entry in install_dir.glob("llama-b[0-9]*"):
        match = re.fullmatch(r"llama-b([0-9]+)", entry.name)
        if match and entry.is_dir():
            highest = max(highest, int(match.group(1)))
    return highest


def cpp_build_number(tag: str) -> int | None:
    """11507 from 'b11507'; None when the tag has no numeric build part."""
    tag = tag.removeprefix("b")
    return int(tag) if tag.isdigit() else None


@contextlib.contextmanager
def staging(install_dir: Path, label: str):
    """Scratch space *on the destination filesystem*, removed on exit.

    Every install builds in one of these and is then promoted by a rename.
    Two reasons, both load-bearing:

    * Atomicity. ``os.rename`` within one filesystem is atomic; a copy
      across filesystems is not. Staging under install_dir is what makes
      promotion a rename rather than a tree-copy, so a versioned name is
      never observed half-populated -- which is what let an interrupted
      extraction be mistaken for a finished install.
    * Ownership. The pruners only delete what the current uid owns, so a
      /tmp leftover from an interrupted run is not ours to reclaim.
    """
    path = Path(tempfile.mkdtemp(dir=install_dir, prefix=f".stage-{label}-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def promote(src: Path, target: Path, trash: Path) -> None:
    """Move a fully-built path into its final name, atomically.

    All three paths come from a :func:`staging` block under the same
    install_dir, so they share one filesystem. An existing destination is
    renamed aside rather than deleted here: the final name must not be
    absent-and-unpopulated across a window anyone can observe, and the
    staging block's cleanup disposes of it.
    """
    if target.exists() or target.is_symlink():
        # A distinct name: ``src`` is itself inside ``trash`` under its own
        # final name, and renaming a directory onto a non-empty one fails.
        aside, n = trash / f".superseded-{target.name}", 0
        while aside.exists():
            n += 1
            aside = trash / f".superseded-{target.name}-{n}"
        os.rename(target, aside)
    os.rename(src, target)


def swap_symlink(target: Path, link_target: str, backup: Path) -> bool:
    """Back up whatever ``target`` currently is, then point it somewhere new.

    Handles the three shapes found in the wild -- a symlink, a regular
    file left by the bash version, or nothing yet -- so the callers do not
    each reimplement it. Refuses to destroy the old binary when the backup
    cannot be written: a rollback path that does not exist is not a
    rollback path.
    """
    if target.is_symlink():
        real = target.resolve()
        source, follow = (real, True) if real.is_file() else (target, False)
        print(f"Backing up {target.name} -> {backup.name}")
        try:
            shutil.copyfile(source, backup, follow_symlinks=follow)
        except OSError as exc:
            eprint(f"Error: could not back up {target.name} to {backup}: {exc}")
            return False
    elif target.is_file():
        print(f"Backing up old {target.name} to {backup.name}")
        try:
            shutil.copyfile(target, backup)
        except OSError as exc:
            eprint(f"Error: could not back up {target.name} to {backup}: {exc}")
            return False
        target.unlink()
        print(f"Migrated {target.name} to a symlink")
    replace_symlink(target, link_target)
    print(f"{target.name} -> {link_target}")
    return True


def is_real_file(path: Path) -> bool:
    """True for a regular file, false for a symlink -- even a valid one.

    tarfile's ``data`` filter vets a member's *name* and drops setuid bits,
    but it does not inspect a symlink's linkname. So ``llama-swap ->
    /etc/shadow`` extracts cleanly, and the code that then calls
    ``Path.is_file()`` (which follows links), chmods and execs the result
    would touch and run the target. Refusing links here closes that first.
    """
    return path.is_file() and not path.is_symlink()


def extract_tar_gz(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        # No try/except fallback: on an interpreter without `filter` the
        # unfiltered call is full CVE-2007-4559 traversal, and silently
        # dropping a security control is worse than refusing to run.
        tf.extractall(dest, filter="data")  # Python >= 3.11.4, 3.12+


def extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip release (stable-diffusion.cpp ships these, not tarballs).

    zipfile has no ``filter=`` argument -- it was proposed and never
    landed -- so the ``try/except TypeError`` this used to carry always
    fell through to the unfiltered call while its comment claimed
    otherwise. Check the names instead, rather than leaning on zipfile's
    incidental mangling of them.
    """
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if os.path.isabs(name) or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in archive: {name}")
        zf.extractall(dest)


def extract_archive(archive: Path, dest: Path, fmt: str = "tar.gz") -> None:
    if fmt == "zip":
        extract_zip(archive, dest)
    else:
        extract_tar_gz(archive, dest)


def download(url: str, dest: Path, *, quiet: bool = True) -> None:
    """Download url to dest with curl (--fail so HTTP errors don't land on disk)."""
    cmd = ["curl", "-fSL", "--retry", "3", "--retry-delay", "2"]
    if quiet:
        cmd.append("-s")
    cmd += ["-o", str(dest), url]
    subprocess.run(cmd, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_symlink(link: Path, target: str) -> None:
    """ln -snf target link"""
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(target, link)


def _no_exec(path: Path) -> None:
    os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o111)


# --- llama.cpp -------------------------------------------------------------


def install_llama_cpp(url: str, tag: str, install_dir: Path) -> tuple[bool, bool]:
    """Download + extract a llama.cpp release tarball into install_dir.

    Returns (ok, installed): already-present and not-an-error cases give
    (True, False); a real install gives (True, True).
    """
    target_dir = install_dir / f"llama-{tag}"
    # Gate on content, never on the name. The gate used to be is_dir(), so
    # an extraction cut short by a full disk or a Ctrl-C left a half-tree
    # that every later run reported as "already installed" -- and that
    # install.py then wrote into config.yaml as LLAMA_DIR.
    if is_real_file(target_dir / "llama-server"):
        print(f"llama.cpp {tag} already installed at {target_dir}")
        return True, False

    print(f"Downloading llama.cpp {tag}...")
    with staging(install_dir, tag) as stage:
        archive = stage / url.rsplit("/", 1)[-1]
        try:
            download(url, archive, quiet=False)
        except subprocess.CalledProcessError:
            eprint(f"Error: download failed for {url}")
            return False, False
        print("Extracting...")
        try:
            extract_tar_gz(archive, stage)
        except (tarfile.TarError, OSError) as exc:
            eprint(f"Error: extraction failed for {url}: {exc}")
            return False, False

        staged = stage / target_dir.name
        if not staged.is_dir():
            # The tarball names its top-level directory after the release,
            # but that is an assumption about a third-party artifact.
            found = [d for d in stage.iterdir() if d.is_dir()]
            if len(found) != 1:
                eprint(f"Error: {url} did not contain {target_dir.name}")
                return False, False
            staged = found[0]
        if not is_real_file(staged / "llama-server"):
            eprint(f"Error: {url} contains no llama-server binary")
            return False, False
        promote(staged, target_dir, stage)

    print(f"Installed to: {target_dir}")
    print(f"Run: {target_dir}/llama-server --help")
    return True, True


def cleanup_old_versions(install_dir: Path, keep: int = KEEP_VERSIONS) -> None:
    """Remove old llama-b<NNNN> dirs, keeping only the newest ``keep``."""
    if not _owned_by_current_user(install_dir):
        return

    # Only match directories with 4+ digit version numbers (e.g. llama-b8913)
    dirs = []
    for entry in install_dir.glob("llama-b[0-9][0-9][0-9][0-9]*"):
        match = re.match(r"llama-b([0-9]+)", entry.name)
        if entry.is_dir() and match:
            dirs.append((int(match.group(1)), entry))
    if len(dirs) <= keep:
        return

    for _, entry in sorted(dirs, key=lambda item: item[0], reverse=True)[keep:]:
        print(f"Removing old version: {entry}")
        shutil.rmtree(entry, ignore_errors=True)


def update_llama_cpp_symlinks(install_dir: Path, new_target: str) -> None:
    """Repoint top-level symlinks that target an old llama-b* dir."""
    if not new_target.startswith("llama-"):
        new_target = f"llama-{new_target}"
    if not (install_dir / new_target).is_dir():
        return
    for link in install_dir.iterdir():
        if not link.is_symlink():
            continue
        # The stable name points *into* a build (`./llama-b123/llama-server`),
        # so it must not be rewritten to the build directory the way a
        # directory-level alias is. _ensure_stable_primary owns it.
        if link.name == CPP_PRIMARY:
            continue
        target = os.readlink(link)
        if re.search(r"llama-b[0-9]", target):
            print(f"Updating symlink {link.name} -> {new_target} (was {target})")
            replace_symlink(link, new_target)
    _ensure_stable_primary(install_dir, CPP_PRIMARY,
                           f"./{new_target}/{CPP_PRIMARY}")


def _ensure_stable_primary(install_dir: Path, primary: str,
                           link_target: str) -> None:
    """Keep ``<primary>`` a symlink to *link_target*, creating it if absent.

    Every engine's stable name is a symlink beside its versioned entries, so
    a config can name ``tools/llama-server`` and a later update moves what it
    points at without anything referencing it changing. Rollback is the same
    one-line change in the other direction.

    A real file of that name is left alone and reported: something put it
    there deliberately, and replacing a working binary with a link into a
    directory a later prune could remove is not a call this makes silently.
    """
    path = install_dir / primary
    if path.exists() and not path.is_symlink():
        eprint(f"Note: {path} is a real file, not a symlink; leaving it "
               "alone (point your config at it, or remove it to get the "
               "stable-name link)")
        return
    if path.is_symlink():
        if os.readlink(path) == link_target:
            return
        print(f"Updating symlink {primary} -> {link_target} "
              f"(was {os.readlink(path)})")
    replace_symlink(path, link_target)
    if not path.is_symlink():      # created rather than repointed
        print(f"{primary} -> {link_target}")


def update_llama_cpp(
    install_dir: Path, version: str, arch_override: str
) -> tuple[bool, str | None]:
    """llama.cpp half of the run.

    Returns (ok, installed_tag): installed_tag is the release tag that was
    actually unpacked, so the caller knows to prune old builds and re-point
    symlinks. None when nothing new landed -- including when the latest
    build is already present, which used to report the tag anyway and so
    made a no-op run prune and rewire as a side effect.
    """
    arch = get_llama_cpp_arch(arch_override)
    backend = next((b for b in CPP_BACKENDS if b in version), "vulkan")
    pattern = CPP_BACKENDS[backend]
    if "{arch}" in pattern:
        pattern = pattern.format(arch=arch)
    elif arch != "x64":
        # Upstream ships these backends x64-only.
        eprint(f"Error: {backend} builds are Linux x64 only (detected {arch})")
        return False, None

    latest = "latest" in version
    if latest:
        tag = latest_llama_cpp_tag(pattern)
        if not tag:
            eprint("Error: Could not fetch latest llama.cpp release tag")
            return False, None
    else:
        tag = version

    payload = release_assets(f"{RELEASES_URL}/tags/{tag}")
    if _is_error_payload(payload):
        eprint(f"Error: Version {tag} not found")
        return False, None
    if latest and (build := cpp_build_number(tag)) is not None \
            and build < highest_installed(install_dir):
        print(f"Already have version b{highest_installed(install_dir)}, "
              f"which is newer than {tag}")
        return True, None

    url = find_asset(payload, pattern)
    if not url:
        eprint(f"Error: Could not find {backend} ({arch}) asset for {tag}")
        return False, None
    ok, installed = install_llama_cpp(url, tag, install_dir)
    if not ok:
        return False, None
    return True, tag if installed else None


# --- llama-swap ------------------------------------------------------------


def installed_llama_swap_version(binary: Path) -> int:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0
    match = re.search(r"v([0-9]+)", result.stdout)
    return int(match.group(1)) if match else 0


def expected_sha256(checksums_path: Path, archive_name: str) -> str:
    lines = checksums_path.read_text(errors="replace").splitlines()
    for line in lines:
        if f"  {archive_name}" in line:
            return line.split()[0]
    for line in lines:  # fallback: some versions use a single space
        if archive_name in line:
            return line.split()[0]
    return ""


def update_llama_swap_symlinks(install_dir: Path, new_name: str) -> None:
    """Repoint top-level symlinks that target an old llama-swap-* binary."""
    for link in install_dir.iterdir():
        if not link.is_symlink() or link.name == "llama-swap":
            continue
        target = os.readlink(link)
        if "llama-swap" in target:
            print(f"Updating symlink {link.name} -> {new_name} (was {target})")
            replace_symlink(link, new_name)


def cleanup_old_llama_swap_versions(install_dir: Path, keep: int) -> None:
    """Remove old llama-swap-vNNN binaries, keeping the newest ``keep``.

    A versioned file that any symlink in install_dir still points at is never
    removed (that would leave a dangling link), so the previous release also
    survives rollback via llama-swap.bak.
    """
    if not _owned_by_current_user(install_dir):
        return
    if not isinstance(keep, int) or keep < 1:
        keep = 1

    # Only exact versioned binaries (llama-swap-vNNN), never the primary
    # symlink, .bak, or _HASH_FAIL_* files
    files = [
        entry.name
        for entry in install_dir.glob("llama-swap-v[0-9]*")
        if entry.is_file()
        and not entry.is_symlink()
        and re.fullmatch(r"llama-swap-v[0-9]+", entry.name)
    ]
    if len(files) <= keep:
        return

    referenced = {
        os.readlink(entry) for entry in install_dir.iterdir() if entry.is_symlink()
    }
    for name in sorted(files, key=lambda n: int(n.rsplit("v", 1)[-1]), reverse=True)[
        keep:
    ]:
        if name in referenced:
            continue
        print(f"Removing old llama-swap: {name}")
        (install_dir / name).unlink()


def update_llama_swap(
    install_dir: Path, keep_swap: int, requested_tag: str = "",
    arch_override: str = "",
) -> bool:
    """Check latest (or requested) release, verify sha256, atomically replace."""
    if requested_tag and requested_tag != "latest":
        tag = requested_tag if requested_tag.startswith("v") else f"v{requested_tag}"
    else:
        tag = latest_llama_swap_tag()

    if not tag:
        eprint("Error: Could not fetch latest llama-swap tag")
        return False
    try:
        tag_num = int(tag.lstrip("v"))
    except ValueError:
        eprint(f"Error: could not parse llama-swap version from tag {tag}")
        return False

    target = install_dir / "llama-swap"
    installed = installed_llama_swap_version(target) if target.exists() else 0

    if not requested_tag or requested_tag == "latest":
        if 0 < installed and tag_num <= installed:
            print(f"llama-swap v{installed} already up-to-date (latest {tag})")
            return True
        # also handle case where installed > tag_num (local build ahead)
        if installed > tag_num:
            print(f"Already have llama-swap v{installed}, which is newer than {tag}")
            return True
    elif installed == tag_num:
        print(f"llama-swap {tag} already installed")
        return True

    arch = get_llama_swap_arch(arch_override)
    payload = release_assets(f"{LLAMA_SWAP_REPO}/releases/tags/{tag}")
    if _is_error_payload(payload):
        eprint(f"Error: llama-swap version {tag} not found")
        return False

    url = find_asset(payload, f"{arch}.tar.gz")
    checksums_url = find_asset(payload, "checksums.txt")
    if not url:
        eprint(f"Error: Could not find llama-swap asset for {tag} ({arch})")
        return False
    if not checksums_url:
        eprint(f"Error: Could not find checksums.txt for {tag}")
        return False

    with staging(install_dir, "llama-swap") as tmpdir:
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = tmpdir / archive_name

        print(f"Downloading llama-swap {tag} ({arch})...")
        try:
            download(url, archive_path)
        except subprocess.CalledProcessError:
            eprint(f"Error: Failed to download {url}")
            return False
        if not verify_archive(archive_path, archive_name, payload, "required",
                              install_dir):
            return False

        # PGP signature check: not implemented — upstream (mostlygeek/llama-swap)
        # publishes no .asc/.sig files and the goreleaser config has no
        # signing; gpg/minisign are available but there is no secure signature
        # to verify. If signatures become available at a stable URL and
        # verification is <10 lines, add: download the .sig next to
        # checksums.txt and gpgv --verify it before the hash check.

        print("Extracting...")
        try:
            extract_tar_gz(archive_path, tmpdir)
        except (tarfile.TarError, OSError) as exc:
            eprint(f"Error: Failed to extract {archive_path}: {exc}")
            return False
        extracted = tmpdir / "llama-swap"
        if not is_real_file(extracted):
            eprint("Error: Extracted archive does not contain a "
                   "llama-swap binary")
            return False
        # hash verified above; only set exec after successful verification
        os.chmod(extracted, 0o755)

        versioned = install_dir / f"llama-swap-{tag}"  # e.g. llama-swap-v252
        promote(extracted, versioned, tmpdir)

    if not swap_symlink(target, versioned.name,
                        install_dir / "llama-swap.bak"):   # single, overwritten
        return False

    # Update any other symlinks in install_dir that point to old
    # llama-swap-* binaries, then drop superseded versioned binaries.
    update_llama_swap_symlinks(install_dir, versioned.name)
    cleanup_old_llama_swap_versions(install_dir, keep_swap)

    try:
        result = subprocess.run(
            [str(target), "--version"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        new_ver = result.stdout.splitlines()[0] if result.stdout.splitlines() else ""
    except (OSError, subprocess.TimeoutExpired):
        new_ver = ""
    if not new_ver:  # wrong arch or fails to execute
        new_ver = f"v{tag_num}"
    print(f"Installed llama-swap {tag} to {versioned} (symlink {target}) ({new_ver})")
    return True


# --- single-binary engines -------------------------------------------------


@dataclass(frozen=True)
class Engine:
    """A prebuilt *server* released as one archive.

    llama.cpp keeps its own code path (its release unpacks a whole build
    directory and is versioned by build number), so it is not an Engine.

    Fields
      tag_re    which tags carry assets. Not all of them do: whisper.cpp
                publishes a semver tag *and* a build tag on the same day,
                and only the build tag has binaries. stable-diffusion.cpp
                versions as ``master-<n>-<sha>``. Matching the tag shape
                rather than taking "latest" is what stops this picking a
                release with zero assets.
      backends  backend name -> regex over the download URL, with
                ``{arch}`` substituted. A single-entry mapping is an engine
                that ships one Linux build and has nothing to choose.
      names     candidate paths for the binary inside the archive.
      fmt       "tar.gz" or "zip" — stable-diffusion.cpp ships zips.
      arch      which architecture vocabulary the asset names use:
                "cpp" (x64/arm64) or "sd" (x86_64/aarch64).
      bundle    True when the archive's directory must be installed whole.
                whisper.cpp's and stable-diffusion's prebuilt servers link
                against sibling ``.so`` files and find them via
                ``RUNPATH=$ORIGIN``, so copying out the binary alone yields
                something that dies with "libwhisper.so.1: cannot open
                shared object file".
      checksum  "required" or "warn" — see ENGINES.
    """
    key: str
    label: str
    repo: str
    primary: str
    versioned: str
    backup: str
    backends: dict[str, str]
    tag_re: str
    names: tuple[str, ...]
    checksum: str
    fmt: str = "tar.gz"
    arch: str = "cpp"
    keep: int = DEFAULT_KEEP_ENGINE_VERSIONS
    bundle: bool = False

    @property
    def api(self) -> str:
        return f"https://api.github.com/repos/{self.repo}"

    @property
    def single_backend(self) -> str:
        """The one backend name, when there is only one."""
        return next(iter(self.backends))

    def asset_pattern(self, backend: str) -> str:
        """Pattern for *backend*, falling back to the only one there is."""
        if backend in self.backends:
            return self.backends[backend]
        if len(self.backends) == 1:
            # Only one build exists, so a request for another is being
            # ignored rather than satisfied -- say so, because "sd rocm"
            # quietly installing Vulkan is not what was asked for.
            if backend and backend != self.single_backend:
                print(f"  Note: no {backend} build published; using "
                      f"{self.single_backend}")
            return self.single_backend_pattern()
        return ""

    def single_backend_pattern(self) -> str:
        return next(iter(self.backends.values()))


#: audio.cpp follows llama.cpp's asset naming almost exactly
#: (``audio-v0.9.2-bin-ubuntu-x64-vulkan-portable.tar.gz``) and ships binaries on its
#: semver tag, so it needs nothing special. There is no ROCm/HIP build — the
#: AMD path is the Vulkan one, which is what this fleet already runs.
#:
#: The **portable** build is the one to install, and this was measured rather
#: than assumed: on the R9700/radv fleet the plain ``-vulkan`` build of the
#: same tag answers /health and then dies with SIGILL on the first synthesis
#: request, while ``-vulkan-portable`` returns a valid 24 kHz WAV. The portable
#: builds target a baseline CPU instruction set, so they cannot be the thing
#: that dies there. An earlier version of this comment anchored the pattern
#: specifically to *exclude* ``-portable``, which is why the deployment ran a
#: hand-copied portable build for a year while this installed the other one.
#:
#: ``bundle=True`` for the same reason it is true elsewhere: the portable
#: tarball's ``audiocpp_server`` loads ``libggml-vulkan.so`` and its CPU
#: backend variants from its own directory, so the directory has to travel with
#: it. The non-portable build needs no libraries and was a solo binary; that is
#: no longer what gets installed. The source tree the tarball also carries is
#: not worth installing.
AUDIO_CPP = Engine(
    key="audio",
    label="audio.cpp",
    repo="0xShug0/audio.cpp",
    primary="audiocpp_server",
    versioned="audiocpp_server-{tag}",
    backup="audiocpp_server.bak",
    backends={"vulkan": r"ubuntu-{arch}-vulkan-portable\.tar\.gz$"},
    tag_re=r"v[0-9]+(\.[0-9]+)*",
    names=("audiocpp_server", "bin/audiocpp_server", "build/bin/audiocpp_server"),
    checksum="warn",
    bundle=True,
)

#: whisper.cpp publishes two tags per release: a semver one with no assets and
#: a ``bNNNN`` build one that has them, both on the same day. tag_re selects
#: the build line, and the asset scan then skips any build that is missing
#: our platform. bundle=True because of the $ORIGIN libraries.
WHISPER = Engine(
    key="whisper",
    label="whisper.cpp",
    repo="ggml-org/whisper.cpp",
    primary="whisper-server",
    versioned="whisper-server-{tag}",
    backup="whisper-server.bak",
    backends={"default": r"whisper-bin-ubuntu-{arch}\.tar\.gz$"},
    tag_re=r"b[0-9]+",
    names=("whisper-server", "main", "bin/whisper-server", "build/bin/whisper-server"),
    checksum="warn",
    bundle=True,
)

#: stable-diffusion.cpp versions as ``master-<build>-<sha>``, ships zips, uses
#: the ``x86_64`` spelling, and — like the other two — publishes no checksums.
#: Both Linux builds are bundles; verified against master-968 that sd-server
#: carries RUNPATH=$ORIGIN and needs ~35 sibling .so files beside it. The ROCm
#: build is 285 MB against Vulkan's 37 MB, so vulkan is the default.
STABLE_DIFFUSION = Engine(
    key="sd",
    label="stable-diffusion.cpp",
    repo="leejet/stable-diffusion.cpp",
    primary="sd-server",
    versioned="sd-server-{tag}",
    backup="sd-server.bak",
    backends={
        "vulkan": r"Linux-Ubuntu-[0-9.]+-{arch}-vulkan\.zip$",
        "rocm": r"Linux-Ubuntu-[0-9.]+-{arch}-rocm-[0-9.]+\.zip$",
        "cpu": r"Linux-Ubuntu-[0-9.]+-{arch}\.zip$",
    },
    tag_re=r"master-[0-9]+-[0-9a-f]+",
    names=("sd-server", "bin/sd-server", "build/bin/sd-server"),
    checksum="warn",
    fmt="zip",
    arch="sd",
    bundle=True,
)

ENGINES = {e.key: e for e in (AUDIO_CPP, WHISPER, STABLE_DIFFUSION)}

#: Backends the CLI accepts as a bare positional. Engines that ship only one
#: Linux build ignore this and take what they have.
KNOWN_BACKENDS = ("vulkan", "rocm", "hip", "cuda", "cpu")


def get_sd_arch(override: str = "") -> str:
    """stable-diffusion.cpp spells the architecture ``x86_64`` / ``aarch64``."""
    if override:
        return _normalize(override, ARCH_SD)
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return "x86_64"


ARCH_RESOLVERS = {"cpp": get_llama_cpp_arch, "sd": get_sd_arch}


def engine_arch(engine: Engine, override: str = "") -> str:
    return ARCH_RESOLVERS[engine.arch](override)


# --- target selection ------------------------------------------------------

#: What a target name resolves to. The llama.cpp pair is one entry because
#: they are the two halves of the same serving stack — ``ll`` should mean
#: "the llama bits", not "whisper, oddly".
TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "llama": ("cpp", "swap"),
    "cpp": ("cpp",),
    "swap": ("swap",),
    "audio": ("audio",),
    "whisper": ("whisper",),
    "sd": ("sd",),
}

#: Accepted spellings per group. The first is canonical.
TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "llama": ("llama", "ll"),
    "cpp": ("cpp", "llama.cpp"),
    "swap": ("swap", "llama-swap"),
    "audio": ("audio", "a", "audio.cpp"),
    "whisper": ("whisper", "w", "whisper.cpp"),
    "sd": ("sd", "stable", "stable-diffusion", "sams", "stable-diffusion.cpp"),
}

#: ``all`` is deliberately NOT in TARGET_ALIASES. It is matched in full or
#: not at all: ``a`` is an alias for audio.cpp, so any prefix or glob
#: expansion that could reach "all" would make that one-letter shorthand
#: ambiguous. Its members are already covered by the individual groups, so
#: excluding it costs nothing.
ALL_ALIASES = ("all",)

#: Every updatable target, in the order they are installed.
ALL_TARGETS = ("cpp", "swap", "audio", "whisper", "sd")

#: What a bare invocation updates: the two verified ones. ``all`` is a
#: separate, explicit word because the other three install binaries that no
#: upstream publishes a sha256 for.
DEFAULT_TARGETS = ("cpp", "swap")


class TargetError(ValueError):
    """A target selector matched nothing, or matched too much."""


def _alias_index() -> dict[str, str]:
    """Every accepted spelling -> canonical group name."""
    index: dict[str, str] = {}
    for group, names in TARGET_ALIASES.items():
        for name in names:
            index[name] = group
    return index


def resolve_target(token: str, index: dict[str, str]) -> tuple[str, ...] | None:
    """Target names a selector token, or None if it is not a selector.

    Three tiers, in order:

      1. exact alias (this is the only tier ``all`` answers to)
      2. glob, so ``s*`` or ``*w*`` select a set
      3. unique prefix, so ``sd`` and ``sd``-prefixes stay short

    Prefix matching is last precisely because it is the loosest: an
    ambiguous prefix is an error rather than a guess, since guessing wrong
    here means installing the wrong engine's binary.
    """
    low = token.lower()
    if low in index:
        return TARGET_GROUPS[index[low]]
    if low in ALL_ALIASES:
        return ALL_TARGETS
    if any(ch in low for ch in "*?["):
        hits = sorted({index[name] for name in index if fnmatch(name, low)})
        if hits:
            return tuple(t for t in ALL_TARGETS if t in hits)
        return None
    prefix = sorted({index[name] for name in index if name.startswith(low)})
    if len(prefix) == 1:
        return TARGET_GROUPS[prefix[0]]
    if len(prefix) > 1:
        raise TargetError(
            f"{token!r} is ambiguous: matches {', '.join(prefix)} "
            "(be more specific, or use a glob like 'sd*')")
    return None


def resolve_targets(tokens: list[str]) -> tuple[str, ...]:
    """Turn positional target selectors into a deduplicated target list."""
    index = _alias_index()
    chosen: list[str] = []
    for token in tokens:
        targets = resolve_target(token, index)
        if targets is None:
            continue  # not a selector: caller treats it as a version
        for target in targets:
            if target not in chosen:
                chosen.append(target)
    return tuple(t for t in ALL_TARGETS if t in chosen)


def resolve_engine_tag(engine: Engine, arch: str, requested: str = "",
                       backend: str = "") -> str:
    """Newest tag matching ``tag_re`` that actually carries our asset.

    Reading the *release list* rather than ``/releases/latest`` is the point:
    ``latest`` on whisper.cpp is the asset-less semver tag, so "latest" is
    exactly the wrong answer. The list payload carries each release's
    ``assets``, so this costs one API call.
    """
    if requested:
        return requested if requested.startswith(("v", "b")) else f"v{requested}"
    for release in _releases(gh_api(f"{engine.api}/releases")):
        tag = release.get("tag_name") or ""
        if not re.fullmatch(engine.tag_re, tag):
            continue
        if find_asset(release, engine.asset_pattern(backend).format(arch=arch)):
            return tag
    return ""


def verify_archive(archive: Path, archive_name: str, payload, policy: str,
                   install_dir: Path) -> bool:
    """Check the archive against ``checksums.txt`` when one is published.

    ``required`` — the one llama.cpp and llama-swap use. A missing manifest
    is an error, because the install would otherwise be a silent downgrade
    of the guarantee.

    ``warn`` — for projects that publish no manifest at all. The download is
    still pinned to an immutable tag URL over TLS, but there is nothing to
    verify it against, so that limit is stated out loud rather than implied
    away. Quarantining still happens on a real mismatch: if a manifest turns
    up one day, the failure path is already there.
    """
    checksums_url = find_asset(payload, "checksums.txt")
    if not checksums_url:
        if policy == "required":
            eprint("Error: Could not find checksums.txt (this project is "
                   "expected to publish one; refusing to install unverified)")
            return False
        eprint(f"  !! {archive_name}: UNVERIFIED — upstream publishes no "
               "checksums.txt, so this binary is authenticated by TLS and "
               "its tag URL only.")
        eprint("     Anything less than a pinned tag download. Pin the "
               "version with --*-version if that matters to you.")
        return True
    try:
        checksums_path = archive.parent / "checksums.txt"
        download(checksums_url, checksums_path)
    except subprocess.CalledProcessError:
        eprint(f"Error: Failed to download checksums {checksums_url}")
        return False
    expected = expected_sha256(checksums_path, archive_name)
    if not expected:
        eprint(f"Error: Could not find hash for {archive_name} in checksums.txt")
        return False
    print("Verifying sha256...")
    actual = sha256_file(archive)
    if actual != expected:
        eprint(f"Error: sha256 mismatch for {archive_name}")
        eprint(f"  expected: {expected}")
        eprint(f"  actual:   {actual}")
        fail_path = install_dir / f"_HASH_FAIL_{archive_name}"
        eprint(f"Hash verification failed — saving downloaded file as {fail_path} "
               "(no execute bit, symlink not updated)")
        shutil.copyfile(archive, fail_path)
        _no_exec(fail_path)
        return False
    print(f"Checksum verified: {actual}")
    return True


def find_extracted(root: Path, names: tuple[str, ...]) -> Path | None:
    """Locate the server binary in an extracted tree.

    These projects disagree on layout, so look for each candidate and,
    failing that, for the primary name anywhere in the tree.
    """
    for name in names:
        candidate = root / name
        if is_real_file(candidate):
            return candidate
    primary = Path(names[0]).name
    matches = sorted(p for p in root.rglob(primary) if is_real_file(p))
    return matches[0] if matches else None


def bundle_root(root: Path) -> Path:
    """The directory to install whole for a ``bundle`` engine.

    Release tarballs wrap their contents in a single directory named after
    the build; unwrap it so the versioned directory holds the binary and
    its libraries rather than one more level of nesting.
    """
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


def _engine_is_intact(engine: Engine, path: Path) -> bool:
    """Whether ``path`` looks like a finished install of ``engine``.

    Promotion is atomic, so anything installed from here on is whole. This
    check exists for directories the pre-atomic code left behind, where a
    name alone was enough to be believed -- and believing one wired a
    truncated engine into the serving config.

    A bundle is checked for content rather than for its binary by name:
    where the binary sits inside a bundle is whatever the release author
    chose, and is not known until after extraction.
    """
    if engine.bundle:
        if not path.is_dir() or path.is_symlink():
            return False
        return any(is_real_file(p) for p in path.rglob("*"))
    return is_real_file(path)


def version_sort_key(name: str) -> tuple[tuple[int, ...], str]:
    """Order versioned names by their numbers, not their letters.

    One rule for every engine, because the names disagree: llama.cpp sorts
    on ``b5454``, stable-diffusion.cpp on ``master-1000-bbb``, whisper.cpp
    on a semver. Sorting those as text puts ``master-999`` *above*
    ``master-1000``, which makes the pruner drop the newest build and keep
    an older one. Taking every digit run, in order, compares all three
    correctly -- ``(5454,) > (1000,) > (999,)``, and ``v0.9.2`` becomes
    ``(0, 9, 2)`` so semvers order among themselves too.
    """
    digits = tuple(int(d) for d in re.findall(r"[0-9]+", name))
    return digits, name


def cleanup_engine_versions(engine: Engine, install_dir: Path,
                            keep: int, referenced: set[str]) -> None:
    """Drop superseded ``<primary>-<tag>`` entries, newest ``keep`` kept.

    Handles both shapes: a versioned file for a solo binary, a versioned
    directory for a bundle. An entry any symlink still points at is never
    removed — deleting it would leave a dangling link, and the ``.bak``
    depends on the previous release staying readable for rollback.

    ``referenced`` holds whole symlink targets, which for a bundle are
    ``<dir>/<binary>``. Comparing those against a bare directory name never
    matches, so the guard silently did nothing for exactly the engines with
    the most to lose. Compare the first component instead.
    """
    if not _owned_by_current_user(install_dir):
        return
    keep = keep if isinstance(keep, int) and keep >= 1 else 1
    pattern = re.escape(engine.primary) + r"-[vb][0-9][0-9A-Za-z.+-]*"
    entries = sorted(
        (e for e in install_dir.iterdir()
         if not e.is_symlink() and re.fullmatch(pattern, e.name)
         and (e.is_file() or e.is_dir())),
        key=lambda e: version_sort_key(e.name),
        reverse=True,
    )
    if len(entries) <= keep:
        return
    roots = {t.split("/", 1)[0] for t in referenced}
    for entry in entries[keep:]:
        if entry.name in roots:
            continue
        print(f"Removing old {engine.primary}: {entry.name}")
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def update_engine(engine: Engine, install_dir: Path, arch: str,
                  keep: int, requested_tag: str = "",
                  backend: str = "") -> bool:
    """Fetch, verify and install one engine."""
    pattern = engine.asset_pattern(backend)
    if not pattern:
        eprint(f"Error: {engine.label} has no {backend!r} build "
               f"(offers: {', '.join(engine.backends)})")
        return False
    tag = resolve_engine_tag(engine, arch, requested_tag, backend)
    if not tag:
        eprint(f"Error: Could not fetch a {engine.label} tag carrying a "
               f"{arch} build (tags matching {engine.tag_re})")
        return False

    installed_path = install_dir / engine.versioned.format(tag=tag)
    if _engine_is_intact(engine, installed_path):
        print(f"{engine.label} {tag} already installed at {installed_path.name}")
        return True

    payload = release_assets(f"{engine.api}/releases/tags/{tag}")
    if _is_error_payload(payload):
        eprint(f"Error: {engine.label} version {tag} not found")
        return False
    url = find_asset(payload, pattern.format(arch=arch))
    if not url:
        eprint(f"Error: Could not find a {engine.label} {arch} "
               f"{backend} asset for {tag}")
        return False

    with staging(install_dir, engine.primary) as tmp:
        archive_name = url.rsplit("/", 1)[-1]
        archive = tmp / archive_name
        print(f"Downloading {engine.label} {tag} ({arch})...")
        try:
            download(url, archive)
        except subprocess.CalledProcessError:
            eprint(f"Error: Failed to download {url}")
            return False
        if not verify_archive(archive, archive_name, payload, engine.checksum,
                              install_dir):
            return False

        print("Extracting...")
        tree = tmp / "tree"
        try:
            extract_archive(archive, tree, engine.fmt)
        except (tarfile.TarError, zipfile.BadZipFile, ValueError,
                OSError) as exc:
            eprint(f"Error: Failed to extract {archive_name}: {exc}")
            return False

        if engine.bundle:
            # Install the whole directory: the server resolves its shared
            # libraries through RUNPATH=$ORIGIN, so they have to travel
            # with it. Copying out the binary alone yields a file that dies
            # with "libwhisper.so.1: cannot open shared object file".
            bundle = bundle_root(tree)
            binary = find_extracted(bundle, engine.names)
            if binary is None:
                eprint(f"Error: {archive_name} contains no {engine.primary} "
                       f"binary (looked for {', '.join(engine.names)})")
                return False
            os.chmod(binary, 0o755)
            install_as = bundle
            # Point at wherever the binary actually landed inside the bundle,
            # rather than assuming it sits at the top — the libraries it
            # needs must come along either way.
            link_target = installed_path.name / binary.relative_to(bundle)
        else:
            binary = find_extracted(tree, engine.names)
            if binary is None:
                eprint(f"Error: {archive_name} contains no {engine.primary} "
                       f"binary (looked for {', '.join(engine.names)})")
                return False
            os.chmod(binary, 0o755)      # execute bit only after verification
            install_as = binary
            link_target = installed_path.name

        promote(install_as, installed_path, tmp)

    if not swap_symlink(install_dir / engine.primary, str(link_target),
                        install_dir / engine.backup):
        return False

    referenced = {os.readlink(e) for e in install_dir.iterdir() if e.is_symlink()}
    cleanup_engine_versions(engine, install_dir, keep, referenced)
    print(f"Installed {engine.label} {tag} to {installed_path}")
    return True


# --- argument parsing ------------------------------------------------------


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {number})")
    return number


def resolve_keeps(args: argparse.Namespace) -> int:
    """Engine-version keep count: flag, then env, then the default.

    argparse defaults to ``None`` rather than the constant so that the
    environment variable can still win; a non-zero default made the
    ``if not keep_engines`` test below permanently false.
    """
    if args.keep_engine_versions is not None:
        return args.keep_engine_versions
    env = os.environ.get("KEEP_ENGINE_VERSIONS", "")
    return int(env) if env.isdigit() else DEFAULT_KEEP_ENGINE_VERSIONS


def keep_swap_default() -> int:
    raw = os.environ.get("KEEP_SWAP_VERSIONS", "")
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return DEFAULT_KEEP_SWAP_VERSIONS


def resolve_positionals(
    args: list[str], swap_version: str, cpp_only: bool, swap_only: bool
) -> tuple[list[str], str, str, str, list[str]]:
    """Split positionals into (target tokens, backend, version, swap tag, notes).

    A positional is a target selector first, then a backend name, and only
    then a version — so ``update.py sd rocm`` reads the way it looks. The
    legacy ``v<NNN>`` heuristic for llama-swap is unchanged.

    Raises :class:`TargetError` on a selector that matches nothing or
    several things. That used to be appended to ``notes`` and printed, which
    left ``selected`` empty and fell through to the default targets — so
    ``update.py s`` printed "ambiguous: matches sd, swap" and then went
    ahead and installed several gigabytes, exiting 0. A selector the user
    typed and we could not resolve must stop the run.
    """
    version = DEFAULT_VERSION
    backend = ""
    notes: list[str] = []
    index = _alias_index()
    rest: list[str] = []
    for arg in args:
        try:
            targets = resolve_target(arg, index)
        except TargetError:
            raise
        if targets is not None:
            rest.append(arg)          # collected below, order-independent
            continue
        low = arg.lower()
        if low in ("vulkan", "rocm", "hip", "cuda", "cpu"):
            if backend and backend != low:
                notes.append(f"Note: {backend} given after {low}; using {backend}")
            else:
                backend = backend or low
            continue
        if re.fullmatch(r"v[0-9]+", arg) and not swap_version and not cpp_only:
            swap_version = arg
            if not swap_only:
                notes.append(
                    f"Note: Interpreting {arg} as llama-swap version "
                    "(use --swap-version to be explicit)"
                )
        else:
            version = arg
    return rest, backend, version, swap_version, notes


def build_parser() -> argparse.ArgumentParser:
    alias_lines = []
    for group, names in TARGET_ALIASES.items():
        alias_lines.append(f"  {group:<9} {' '.join(names)}")
    parser = argparse.ArgumentParser(
        description=(
            "Download and install prebuilt server binaries. Each engine is "
            "updated in the directory it is already installed in; a new "
            "engine goes under the configured root (extras/engines.ini, "
            "$LLAMA_DIR, else the current directory). Run with --print-paths "
            "to see the resolution without downloading anything."
        ),
        epilog=(
            "Targets (positional, first match wins; 'all' must be typed in full):\n"
            + "\n".join(alias_lines)
            + f"\n  {'all':<9} {' '.join(ALL_ALIASES)}   # everything\n"
            "  (a glob works too: 's*' selects swap and sd)\n"
            "With no target, llama.cpp and llama-swap are updated -- the two\n"
            "that publish checksums.txt. 'all' adds the three that do not.\n"
            "\n"
            "Backends (positional): " + ", ".join(KNOWN_BACKENDS)
            + "\n\n"
            "Install locations, highest precedence first:\n"
            "  --dir <engine>=<path>       one run only\n"
            f"  [engines] <engine> in {CONFIG_NAME}\n"
            "  $<ENGINE>_BIN_DIR           the same variable llama-packer reads\n"
            "  an existing install found from the cwd outwards\n"
            f"  [engines] root, $LLAMA_DIR, else the cwd\n"
            "\n"
            "Environment:\n"
            "  GITHUB_TOKEN            Optional. GitHub personal access token for\n"
            "                          higher API rate limits (5000 req/hr vs 60).\n"
            "                          Also reads GITHUB_PERSONAL_ACCESS_TOKEN.\n"
            "  KEEP_SWAP_VERSIONS      Optional. Same as --keep-swap.\n"
            "  KEEP_ENGINE_VERSIONS    Optional. Same as --keep-engine-versions.\n"
            "  LLAMA_DIR               Optional. Install root when an engine has\n"
            "                          no install yet and no config entry.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "args",
        nargs="*",
        metavar="target|version",
        help="Targets (llama/ll, cpp, swap, audio/a, whisper/w, sd/stable, "
        "all), a backend (vulkan, rocm, cpu), or a version tag (b8913).",
    )
    parser.add_argument(
        "--swap-only",
        action="store_true",
        help="Only update llama-swap (skip llama.cpp)",
    )
    parser.add_argument(
        "--cpp-only",
        action="store_true",
        help="Only update llama.cpp (skip llama-swap)",
    )
    parser.add_argument(
        "--skip-swap",
        action="store_true",
        help="Skip llama-swap update",
    )
    parser.add_argument(
        "--swap-version",
        metavar="TAG",
        default="",
        help="Specific llama-swap version (e.g., v252)",
    )
    parser.add_argument(
        "--keep-swap",
        type=positive_int,
        metavar="N",
        default=None,
        help="How many llama-swap-vNNN binaries to keep (default 2: the current "
        "one plus one previous). Older ones are removed after each install; a "
        "versioned binary that a symlink still points at is never removed.",
    )
    parser.add_argument(
        "--arch",
        metavar="ARCH",
        type=arch_arg,        # refuse a bad one here, not three layers down
        default="",
        help="Override the detected CPU architecture for both downloads "
        "(x64/amd64/linux_amd64, arm64/aarch64/linux_arm64, s390x)",
    )
    parser.add_argument(
        "--audio-cpp-version",
        metavar="TAG",
        default="",
        help="Specific audio.cpp release tag (e.g. v0.9.2)",
    )
    parser.add_argument(
        "--whisper-version",
        metavar="TAG",
        default="",
        help="Specific whisper.cpp release tag (build tags look like b5454)",
    )
    parser.add_argument(
        "--sd-version",
        metavar="TAG",
        default="",
        help="Specific stable-diffusion.cpp release tag (e.g. master-968-8cd731e)",
    )
    parser.add_argument(
        "--dir",
        action="append",
        metavar="ENGINE=PATH",
        help="Install this engine into PATH for this run (repeatable). "
             "Overrides every configured and discovered location.",
    )
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="Print where each engine would be updated and why, then exit "
             "without downloading anything.",
    )
    parser.add_argument(
        "--keep-engine-versions",
        metavar="N",
        type=positive_int,
        default=None,   # None so KEEP_ENGINE_VERSIONS below can still apply
        help="How many old versions of each audio engine to keep "
        f"(default {DEFAULT_KEEP_ENGINE_VERSIONS}). Or KEEP_ENGINE_VERSIONS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.swap_only and args.cpp_only:
        eprint("Error: --swap-only and --cpp-only are mutually exclusive")
        return 1

    try:
        dir_overrides = parse_dir_overrides(args.dir)
    except TargetError as exc:
        eprint(f"Error: {exc}")
        return 1

    try:
        target_tokens, backend, version, swap_version, notes = \
            resolve_positionals(args.args, args.swap_version, args.cpp_only,
                                args.swap_only)
    except TargetError as exc:
        eprint(f"Error: {exc}")
        eprint("Nothing was installed. See --help for the target list.")
        return 1
    targets = resolve_targets(target_tokens)

    # Explicit selectors win; the legacy --*-only flags narrow the default;
    # nothing selected at all means the two verified targets.
    if targets:
        if args.cpp_only or args.swap_only:
            eprint("Error: target selectors and --cpp-only/--swap-only are "
                   "alternatives; pass one or the other")
            return 1
        selected = targets
    elif args.swap_only:
        selected = ("swap",)
    elif args.cpp_only:
        selected = ("cpp",)
    elif args.skip_swap:
        selected = ("cpp",)
    else:
        selected = DEFAULT_TARGETS

    for note in notes:
        eprint(note)
    print("Targets: " + ", ".join(selected))

    # Every target's directory is settled before anything is downloaded, and
    # printed, so a run against an unexpected location is visible immediately
    # rather than after a 200 MB download. --print-paths stops here: the
    # answer to "where would this go?" is worth having without the download.
    install_dirs = resolve_install_dirs(dir_overrides)
    report_install_dirs(install_dirs)
    if args.print_paths:
        return 0

    keep_swap = (
        args.keep_swap
        if args.keep_swap is not None
        else keep_swap_default()
    )
    keep_engines = resolve_keeps(args)
    pinned = {
        "audio": args.audio_cpp_version,
        "whisper": args.whisper_version,
        "sd": args.sd_version,
    }

    # Each target is independent: one failing must not abandon the rest, and
    # llama.cpp/llama-swap keep their historical non-zero exit because that
    # is the serving stack this tool exists to keep working.
    status = 0
    if "cpp" in selected:
        cpp_version = _backend_version(version, backend)
        cpp_dir, _why = install_dirs["cpp"]
        ok, installed_tag = update_llama_cpp(cpp_dir, cpp_version,
                                             args.arch)
        if not ok:
            status = 1
        elif installed_tag is not None:
            # Cleanup old versions after successful install and re-point any
            # symlinks that target an old llama-b* dir at the new one.
            cleanup_old_versions(cpp_dir)
            update_llama_cpp_symlinks(cpp_dir, installed_tag)

    if "swap" in selected:
        swap_dir, _why = install_dirs["swap"]
        if swap_version:
            ok = update_llama_swap(swap_dir, keep_swap, swap_version,
                                   args.arch)
        else:
            ok = update_llama_swap(swap_dir, keep_swap, "", args.arch)
        if not ok:
            eprint("Warning: llama-swap update failed")
            status = 1

    for key in ("audio", "whisper", "sd"):
        if key not in selected:
            continue
        engine = ENGINES[key]
        engine_dir, _why = install_dirs[key]
        print(f"\n=== {engine.label} ===")
        try:
            ok = update_engine(engine, engine_dir, engine_arch(engine,
                                                               args.arch),
                               keep_engines, pinned[key],
                               backend or engine.single_backend)
        except ApiError as exc:
            eprint(f"Warning: {engine.label} update failed: {exc}")
            ok = False
        if not ok:
            eprint(f"Warning: {engine.label} update failed")
            status = 1

    return status


def _backend_version(version: str, backend: str) -> str:
    """Fold a bare backend positional into llama.cpp's version token.

    llama.cpp predates the shared backend vocabulary and takes it as part of
    the version string ("latest-vulkan"), so keep that shape rather than
    changing its signature.

    A backend already named in ``version`` wins. That is what used to make
    ``update.py cpu`` install Vulkan: DEFAULT_VERSION is "latest-vulkan",
    the function returned it unchanged because it contained a backend word,
    and the Vulkan branch claimed it.
    """
    if not backend:
        return version
    if backend == "hip":
        backend = "rocm"          # the two names for the same upstream build
    if version == DEFAULT_VERSION:
        # The Vulkan in the default is a default, not a choice. Treating it
        # as an explicit backend is what made `update.py cpu` install
        # Vulkan: the token already contained a backend word, so the
        # requested one was discarded before anyone looked at it.
        version = "latest"
    if any(b in version for b in CPP_BACKENDS):
        return version            # already explicit: it wins
    return f"{version}-{backend}"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ApiError as exc:
        eprint(str(exc))
        sys.exit(1)
    except ArchError as exc:
        eprint(f"Error: {exc}")
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
