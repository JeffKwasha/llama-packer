#!/usr/bin/env python3
"""Update prebuilt engines. See docs/install.md §The updater command line."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
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
KEEP_VERSIONS = 3             # llama.cpp build dirs: current + two previous
DEFAULT_KEEP_SWAP_VERSIONS = 2
DEFAULT_KEEP_ENGINE_VERSIONS = 2
DEFAULT_VERSION = "latest-vulkan"

#: Words upstream puts in a release name when it should not be installed.
#: llama-swap publishes tags such as ``v227 (experimental)``.
SKIP_WORDS = ("experimental", "do not use", "unusable", "broken")

#: One prebuilt per backend. Anchored on the base name so the ``cudart-*``
#: support tarballs cannot answer a "cuda" request, and the CPU pattern is a
#: negative lookahead rather than "no backend word": ubuntu-26.04-s390x has
#: two hyphens in the platform, which the old "ubuntu-{arch}" could not match.
CPP_BACKENDS = {
    "vulkan": r"/llama-[^/]*-bin-ubuntu-vulkan-{arch}\.tar\.gz$",
    "rocm": r"/llama-[^/]*-bin-ubuntu-rocm-[^-/]*-x64\.tar\.gz$",
    "cuda": r"/llama-[^/]*-bin-ubuntu-cuda-[^-/]*-{arch}\.tar\.gz$",
    "cpu": r"/llama-[^/]*-bin-ubuntu-(?!vulkan|rocm|cuda)(?:[0-9.]+-)?{arch}\.tar\.gz$",
}

ALL_TARGETS = ("cpp", "swap", "audio", "whisper", "sd")


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


def _flagged_words(payload) -> list[str]:
    if not isinstance(payload, dict):
        return []
    text = " ".join(str(payload.get(key, ""))
                    for key in ("name", "tag_name")).lower()
    return [word for word in SKIP_WORDS if word in text]


def _release_usable(release: dict, channel: str) -> bool:
    """Whether a release may be installed by *channel*.

    ``build`` is for projects that mark every build release a GitHub
    prerelease while publishing the binaries only there (llama.cpp,
    whisper.cpp). ``stable`` never takes a prerelease. Independently, a
    release upstream labelled experimental/unusable is skipped, and so is a
    draft.
    """
    if release.get("draft"):
        return False
    if channel == "stable" and release.get("prerelease"):
        return False
    return not _flagged_words(release)


def _release_date(release: dict) -> str:
    for key in ("published_at", "created_at"):
        value = release.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _release_rank(release: dict):
    """Newest first: date, then the numeric version, then the tag text.

    The API list is not reliably date-ordered, and two releases on one day
    are common (a semver tag beside a build tag), so the numeric fallback
    matters.
    """
    tag = release.get("tag_name") or ""
    return (_release_date(release), version_sort_key(tag))


def select_release(releases: list, channel: str, tag_re: str,
                   asset_pattern: str = "") -> dict | None:
    """Newest usable release whose tag matches and whose assets match."""
    candidates = [
        release for release in releases
        if isinstance(release, dict)
        and re.fullmatch(tag_re, release.get("tag_name") or "")
        and _release_usable(release, channel)
    ]
    candidates.sort(key=_release_rank, reverse=True)
    for release in candidates:
        if not asset_pattern or find_asset(release, asset_pattern):
            return release
    return None


def find_asset_entry(payload, pattern: str) -> dict:
    if not isinstance(payload, dict):
        return {}
    for asset in payload.get("assets", []) or []:
        if re.search(pattern, asset.get("browser_download_url", "")):
            return asset
    return {}


def find_asset(payload, pattern: str) -> str:
    """First browser_download_url in a release whose URL matches ``pattern``."""
    return find_asset_entry(payload, pattern).get("browser_download_url", "")


def latest_llama_cpp_tag(asset_pattern: str = "") -> str:
    """Newest llama.cpp build tag that carries a prebuilt Linux asset.

    llama.cpp publishes a source-only semver tag and a build tag on the same
    day; taking the newest tag outright picks the one with no binaries.
    """
    probe = asset_pattern or r"bin-ubuntu-[^-]*\.tar\.gz$"
    release = select_release(
        _releases(gh_api(RELEASES_URL)), "build", r"b[0-9]+", probe)
    return release.get("tag_name", "") if release else ""


def latest_llama_swap_tag() -> str:
    release = select_release(
        _releases(gh_api(f"{LLAMA_SWAP_REPO}/releases")),
        "stable", r"v[0-9]+")
    return release.get("tag_name", "") if release else ""


def release_assets(url: str):
    return gh_api(url)


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


ARCH_SWAP = {
    "linux_amd64": "linux_amd64", "x86_64": "linux_amd64",
    "amd64": "linux_amd64", "x64": "linux_amd64",
    "linux_arm64": "linux_arm64", "aarch64": "linux_arm64",
    "arm64": "linux_arm64",
}

ARCH_CPP = {
    "linux_amd64": "x64", "x86_64": "x64", "amd64": "x64", "x64": "x64",
    "linux_arm64": "arm64", "aarch64": "arm64", "arm64": "arm64",
    "s390x": "s390x",
}

ARCH_SD = {
    "linux_amd64": "x86_64", "x86_64": "x86_64", "x86-64": "x86_64",
    "amd64": "x86_64", "x64": "x86_64", "linux_arm64": "aarch64",
    "aarch64": "aarch64", "arm64": "aarch64",
}


def arch_arg(value: str) -> str:
    """argparse type for --arch: accepted only if some resolver knows it."""
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


def get_sd_arch(override: str = "") -> str:
    """stable-diffusion.cpp spells the architecture ``x86_64`` / ``aarch64``."""
    if override:
        return _normalize(override, ARCH_SD)
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return "x86_64"


# --- install location ------------------------------------------------------

CPP_PRIMARY = "llama-server"

INSTALL_UNITS: dict[str, "InstallUnit"] = {}


def _is_venv(path: Path) -> bool:
    """A managed environment. Updating inside one is another tool's job."""
    return (path / "pyvenv.cfg").is_file()


@dataclass
class InstallUnit:
    """How one target's installs are recognized on disk."""

    key: str
    primary: str                  # stable name; always a symlink after install
    versioned: tuple[str, ...]    # globs for this engine's versioned entries
    env: str = ""                 # *_BIN_DIR, shared with llama-packer

    def _entries(self, directory: Path) -> list[Path]:
        try:
            if not directory.is_dir():
                return []
            return list(directory.iterdir())
        except OSError:
            return []

    def _holds_directly(self, names: list[str]) -> bool:
        return any(fnmatch(name, pattern) for name in names
                   for pattern in self.versioned) or self.primary in names

    def _holds(self, names: list[str], directory: Path) -> bool:
        if any(fnmatch(name, pattern) for name in names
               for pattern in self.versioned):
            return True
        if self.primary in names:
            return True
        for name in names:
            child = directory / name
            if not child.is_dir() or _is_venv(child):
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
                if e.is_dir() and not _is_venv(e)
                and not any(e.match(p) for p in self.versioned)
                and self._holds([c.name for c in self._entries(e)], e)]

    def descends_into(self, directory: Path) -> Path | None:
        """The engine's own subdirectory of *directory*, when it has one."""
        entries = self._entries(directory)
        names = [e.name for e in entries]
        if not names or self._holds_directly(names):
            return None
        children = self.holding_children(directory)
        return children[0] if len(children) == 1 else None

    def lives_in(self, directory: Path) -> bool:
        if _is_venv(directory):
            return False
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


def tools_root() -> Path:
    """The tools directory this run updates, from where it was run.

    Run from the install directory, it is that directory's ``tools/``; run
    from ``tools/`` itself (or from anywhere without one) it is the cwd. The
    script's own location never enters into it.
    """
    cwd = Path.cwd()
    if cwd.name == "tools":
        return cwd          # already there; never tools/tools
    return cwd / "tools"    # the install directory's tools/


def install_root() -> Path:
    """The directory that holds ``tools/`` for this run."""
    tools = tools_root()
    return tools.parent if tools.name == "tools" else tools


def _env_dir(unit: InstallUnit) -> Path | None:
    """``$ENGINE_BIN_DIR`` as a directory, or None."""
    if not unit.env:
        return None
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
    """The tools directory, then the install directory (the flat legacy layout)."""
    tools = tools_root()
    here = install_root()
    return [tools] if here == tools else [tools, here]


def resolve_install_dirs(
    overrides: dict[str, str] | None = None,
) -> dict[str, tuple[Path, str]]:
    """Directory and reason per engine; docs/install.md §Where an update installs."""
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    candidates = _discovery_candidates()
    resolved: dict[str, tuple[Path, str]] = {}
    for key in ALL_TARGETS:
        unit = INSTALL_UNITS[key]
        if key in overrides:
            resolved[key] = (Path(overrides[key]).expanduser(), "--dir")
            continue
        if env_dir := _env_dir(unit):
            resolved[key] = (env_dir, f"${unit.env}")
            continue
        for candidate in candidates:
            if inner := unit.descends_into(candidate):
                resolved[key] = (inner, "existing install")
                break
            rivals = unit.holding_children(candidate)
            if len(rivals) > 1:
                eprint(f"Warning: {candidate} holds {len(rivals)} "
                       f"{key} installs ({', '.join(sorted(p.name for p in rivals))}); "
                       f"not guessing — set it with --dir {key}=<path>")
            if unit.lives_in(candidate):
                resolved[key] = (candidate, "existing install")
                break
        else:
            if raw := os.environ.get("LLAMA_DIR", "").strip():
                resolved[key] = (Path(raw).expanduser(), "$LLAMA_DIR")
            else:
                resolved[key] = (tools_root(), "run directory")
    return resolved


def parse_dir_overrides(values: list[str] | None) -> dict[str, str]:
    """``["audio=/x", "bad"]`` -> ``{"audio": "/x"}``; a bad one is an error."""
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
    """Print where each target will be updated, and how that was decided."""
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


def _can_prune(path: Path) -> bool:
    """Safety gate: only prune a directory this run may write to.

    Ownership was the old test, but a group-writable tools tree owned by the
    service account is exactly the deployment this tool installs, and the
    process running an update may not own it. What actually matters is that
    removing an entry will succeed.
    """
    try:
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


def highest_installed(install_dir: Path) -> int:
    """Highest installed llama.cpp build number among llama-b<NNNN> dirs."""
    highest = 0
    for entry in install_dir.glob("llama-b[0-9]*"):
        match = re.fullmatch(r"llama-b([0-9]+)", entry.name)
        if match and _engine_is_intact(CPP, entry):
            highest = max(highest, int(match.group(1)))
    return highest


def cpp_build_number(tag: str) -> int | None:
    """11507 from 'b11507'; None when the tag has no numeric build part."""
    tag = tag.removeprefix("b")
    return int(tag) if tag.isdigit() else None


@contextlib.contextmanager
def dir_lock(install_dir: Path):
    """One run at a time per directory.

    Without it, two runs sharing a tree delete each other's ``.stage-*`` as
    "interrupted leftover" and one of them fails its rename for no stated
    reason. The key is a lock file in the temporary directory named after the
    resolved path, so the tree itself gains nothing to go stale, and locking
    a directory that does not exist yet does not create it.
    """
    key = hashlib.sha256(str(install_dir.resolve()).encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"llama-update-{key}.lock"
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        # Another account's umask made the file read-only to us; flock does
        # not need write access.
        fd = os.open(lock_path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Waiting for another update in {install_dir}...")
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def staging(install_dir: Path, label: str):
    """Scratch space *on the destination filesystem*, removed on exit.

    Promotion is an ``os.rename`` within one filesystem, so a versioned name
    is never observed half-populated — which is what let an interrupted
    extraction be mistaken for a finished install.
    """
    path = Path(tempfile.mkdtemp(dir=install_dir, prefix=f".stage-{label}-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def promote(src: Path, target: Path, trash: Path) -> None:
    """Move a fully-built path into its final name, atomically."""
    if target.exists() or target.is_symlink():
        aside, n = trash / f".superseded-{target.name}", 0
        while aside.exists():
            n += 1
            aside = trash / f".superseded-{target.name}-{n}"
        os.rename(target, aside)
    os.rename(src, target)


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

    ``extractall`` writes every member as a regular file, symlink members
    included (their target lands as content), so the name check is the whole
    guarantee: nothing can point outside *dest* the way a tar member can.
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


def fetch(url: str, dest: Path, label: str) -> bool:
    """Download with a visible progress line, or report the failure."""
    print(f"Downloading {label}...")
    try:
        download(url, dest, quiet=not sys.stdout.isatty())
    except subprocess.CalledProcessError:
        eprint(f"Error: download failed for {url}")
        return False
    return True


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


def expected_sha256(checksums_path: Path, archive_name: str) -> str:
    lines = checksums_path.read_text(errors="replace").splitlines()
    for line in lines:
        if f"  {archive_name}" in line:
            return line.split()[0]
    for line in lines:  # fallback: some versions use a single space
        if archive_name in line:
            return line.split()[0]
    return ""


def verify_archive(archive: Path, archive_name: str, payload, policy: str,
                   install_dir: Path) -> bool:
    """Check the archive against ``checksums.txt`` when one is published.

    ``required`` — llama-swap only. A missing manifest is an error, because
    the install would otherwise be a silent downgrade of the guarantee.

    ``warn`` — the engines that publish no manifest. The download is still
    pinned to an immutable tag URL over TLS, but there is nothing to verify
    it against, so that limit is stated out loud. Quarantining on a real
    mismatch still happens if a manifest appears.
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
        return True
    checksums_path = archive.parent / "checksums.txt"
    if not fetch(checksums_url, checksums_path, "checksums.txt"):
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


# --- engines ---------------------------------------------------------------


@dataclass
class Engine:
    """One target installed as a versioned entry plus a stable name.

    ``kind`` is the archive shape: ``single`` promotes one executable file,
    ``bundle`` installs the extracted directory whole (shared libraries load
    through ``RUNPATH=$ORIGIN``).

    ``channel`` says which releases may be installed: ``build`` accepts the
    GitHub prereleases that llama.cpp and whisper.cpp mark their binary
    releases as, ``stable`` never does.
    """
    key: str
    label: str
    repo: str
    primary: str
    versioned: str
    backup: str
    tag_re: str
    channel: str
    names: tuple[str, ...]
    checksum: str
    backends: dict[str, str]
    fmt: str = "tar.gz"
    arch: str = "cpp"
    kind: str = "bundle"
    link_prefix: str = ""

    @property
    def api(self) -> str:
        return f"https://api.github.com/repos/{self.repo}"

    @property
    def version_prefix(self) -> str:
        return self.versioned.split("{tag}")[0]

    @property
    def version_pattern(self) -> str:
        if not self.versioned:
            return r"(?!)"
        return (re.escape(self.version_prefix)
                + f"(?:{self.tag_re}|v[0-9][0-9A-Za-z.+-]*)$")

    @property
    def single_backend(self) -> str:
        return next(iter(self.backends))

    def tag_from_name(self, name: str) -> str:
        return name[len(self.version_prefix):]

    def asset_pattern(self, backend: str) -> str:
        """Pattern for *backend*, falling back to the only one there is."""
        if backend in self.backends:
            return self.backends[backend]
        if len(self.backends) == 1:
            if backend and backend != self.single_backend:
                print(f"  Note: no {backend} build published; using "
                      f"{self.single_backend}")
            return self.single_backend_pattern()
        return ""

    def single_backend_pattern(self) -> str:
        return next(iter(self.backends.values()))


#: llama.cpp unpacked whole: the directory carries its shared libraries and
#: the server is resolved inside it. The "./" prefix is the spelling the
#: deployment layout documents (``tools/llama-server -> ./llama-b12345/...``).
CPP = Engine(
    key="cpp",
    label="llama.cpp",
    repo="ggml-org/llama.cpp",
    primary=CPP_PRIMARY,
    versioned="llama-{tag}",
    backup="llama-server.bak",
    tag_re=r"b[0-9]+",
    channel="build",
    names=(CPP_PRIMARY,),
    checksum="warn",
    backends=CPP_BACKENDS,
    arch="cpp",
    link_prefix="./",
)

#: llama-swap is one executable, verified against the release checksums.
SWAP = Engine(
    key="swap",
    label="llama-swap",
    repo="mostlygeek/llama-swap",
    primary="llama-swap",
    versioned="llama-swap-{tag}",
    backup="llama-swap.bak",
    tag_re=r"v[0-9]+",
    channel="stable",
    names=("llama-swap",),
    checksum="required",
    backends={"default": r"{arch}\.tar\.gz$"},
    kind="single",
)

#: audio.cpp follows llama.cpp's asset naming and ships binaries on its
#: semver tag. The **portable** build is the one that works here: the plain
#: ``-vulkan`` build of the same tag dies with SIGILL on the first synthesis
#: request on the R9700/radv fleet, while ``-vulkan-portable`` returns a
#: valid WAV. It is a bundle because the server loads its backend libraries
#: from its own directory.
AUDIO_CPP = Engine(
    key="audio",
    label="audio.cpp",
    repo="0xShug0/audio.cpp",
    primary="audiocpp_server",
    versioned="audiocpp_server-{tag}",
    backup="audiocpp_server.bak",
    tag_re=r"v[0-9]+(\.[0-9]+)*",
    channel="stable",
    names=("audiocpp_server", "bin/audiocpp_server", "build/bin/audiocpp_server"),
    checksum="warn",
    backends={"vulkan": r"ubuntu-{arch}-vulkan-portable\.tar\.gz$"},
)

#: whisper.cpp publishes a semver tag with no assets and a ``bNNNN`` build
#: tag with them on the same day; tag_re selects the build line.
WHISPER = Engine(
    key="whisper",
    label="whisper.cpp",
    repo="ggml-org/whisper.cpp",
    primary="whisper-server",
    versioned="whisper-server-{tag}",
    backup="whisper-server.bak",
    tag_re=r"b[0-9]+",
    channel="build",
    names=("whisper-server", "main", "bin/whisper-server", "build/bin/whisper-server"),
    checksum="warn",
    backends={"default": r"whisper-bin-ubuntu-{arch}\.tar\.gz$"},
)

#: stable-diffusion.cpp versions as ``master-<build>-<sha>``, ships zips,
#: spells the architecture ``x86_64``, and is a bundle (sd-server needs ~35
#: sibling .so files found through RUNPATH=$ORIGIN).
STABLE_DIFFUSION = Engine(
    key="sd",
    label="stable-diffusion.cpp",
    repo="leejet/stable-diffusion.cpp",
    primary="sd-server",
    versioned="sd-server-{tag}",
    backup="sd-server.bak",
    tag_re=r"master-[0-9]+-[0-9a-f]+",
    channel="stable",
    names=("sd-server", "bin/sd-server", "build/bin/sd-server"),
    checksum="warn",
    backends={
        "vulkan": r"Linux-Ubuntu-[0-9.]+-{arch}-vulkan\.zip$",
        "rocm": r"Linux-Ubuntu-[0-9.]+-{arch}-rocm-[0-9.]+\.zip$",
        "cpu": r"Linux-Ubuntu-[0-9.]+-{arch}\.zip$",
    },
    fmt="zip",
    arch="sd",
)

ENGINES = {e.key: e for e in
           (CPP, SWAP, AUDIO_CPP, WHISPER, STABLE_DIFFUSION)}

#: Backends the CLI accepts as a bare positional. Engines that ship only one
#: Linux build ignore this and take what they have.
KNOWN_BACKENDS = ("vulkan", "rocm", "hip", "cuda", "cpu")

ARCH_RESOLVERS = {"cpp": get_llama_cpp_arch, "sd": get_sd_arch}


def engine_arch(engine: Engine, override: str = "") -> str:
    return ARCH_RESOLVERS[engine.arch](override)


# --- target selection ------------------------------------------------------

TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "llama": ("cpp", "swap"),
    "cpp": ("cpp",),
    "swap": ("swap",),
    "audio": ("audio",),
    "whisper": ("whisper",),
    "sd": ("sd",),
}

TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "llama": ("llama", "ll"),
    "cpp": ("cpp", "llama.cpp"),
    "swap": ("swap", "llama-swap"),
    "audio": ("audio", "a", "audio.cpp"),
    "whisper": ("whisper", "w", "whisper.cpp"),
    "sd": ("sd", "stable", "stable-diffusion", "sams", "stable-diffusion.cpp"),
}

#: ``all`` is deliberately NOT in TARGET_ALIASES: ``a`` is an alias for
#: audio.cpp, so any prefix or glob expansion that could reach "all" would
#: make that one-letter shorthand ambiguous.
ALL_ALIASES = ("all",)

class TargetError(ValueError):
    """A target selector matched nothing, or matched too much."""


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for group, names in TARGET_ALIASES.items():
        for name in names:
            index[name] = group
    return index


def resolve_target(token: str, index: dict[str, str]) -> tuple[str, ...] | None:
    """Target names a selector token, or None if it is not a selector.

    Exact alias first (the only tier ``all`` answers to), then globs, then a
    unique prefix. An ambiguous prefix is an error rather than a guess.
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


def version_sort_key(name: str) -> tuple[tuple[int, ...], str]:
    """Order versioned names by their numbers, not their letters.

    ``master-999`` must not outrank ``master-1000``, and ``v0.10.0`` must
    outrank ``v0.9.2``. Every digit run, in order, does both.
    """
    digits = tuple(int(d) for d in re.findall(r"[0-9]+", name))
    return digits, name


def versioned_entries(engine: Engine, install_dir: Path) -> list[Path]:
    """This engine's versioned entries, newest first."""
    if not engine.versioned or not install_dir.is_dir():
        return []
    entries = [
        entry for entry in install_dir.iterdir()
        if not entry.is_symlink()
        and re.fullmatch(engine.version_pattern, entry.name)
        and (entry.is_file() or entry.is_dir())
    ]
    entries.sort(key=lambda e: version_sort_key(e.name), reverse=True)
    return entries


def find_extracted(root: Path, names: tuple[str, ...]) -> Path | None:
    """Locate the server binary in an extracted tree."""
    for name in names:
        candidate = root / name
        if is_real_file(candidate):
            return candidate
    primary = Path(names[0]).name
    matches = sorted(p for p in root.rglob(primary) if is_real_file(p))
    return matches[0] if matches else None


def bundle_root(root: Path) -> Path:
    """The directory to install whole for a bundle.

    Release tarballs wrap their contents in a single directory named after
    the build; unwrap it so the versioned directory holds the binary and its
    libraries rather than one more level of nesting.
    """
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


def _engine_is_intact(engine: Engine, path: Path) -> bool:
    """Whether ``path`` looks like a finished install of ``engine``.

    Content, never the name: an extraction cut short by a full disk must not
    be reported as installed forever after.
    """
    if engine.kind == "single":
        return is_real_file(path)
    if not path.is_dir() or path.is_symlink():
        return False
    return find_extracted(path, engine.names) is not None


def current_engine_tag(engine: Engine, install_dir: Path) -> str:
    for entry in versioned_entries(engine, install_dir):
        if _engine_is_intact(engine, entry):
            return engine.tag_from_name(entry.name)
    return ""


def _tag_number(tag: str) -> int | None:
    match = re.search(r"[0-9]+", tag)
    return int(match.group()) if match else None


def _normalize_tag(engine: Engine, requested: str) -> str:
    tag = requested.strip()
    if not tag:
        return ""
    if re.fullmatch(engine.tag_re, tag):
        return tag
    if engine.key in ("swap", "audio") and re.fullmatch(r"[0-9][0-9.]*", tag):
        return f"v{tag}"
    if engine.key in ("cpp", "whisper") and tag.isdigit():
        return f"b{tag}"
    return tag


def split_cpp_version(version: str) -> tuple[str, str]:
    """``(backend, tag)`` from llama.cpp's legacy version token.

    llama.cpp predates the shared backend vocabulary and takes it as part of
    the version string ("latest-vulkan", "b11658-rocm"). A bare backend token
    is a backend request with no pinned build.
    """
    backend = ""
    parts: list[str] = []
    for part in version.split("-"):
        low = part.lower()
        if low == "hip":
            low = "rocm"
        if low in CPP_BACKENDS:
            backend = backend or low
            continue
        if part:
            parts.append(part)
    tag = "-".join(parts)
    return backend, "" if tag == "latest" else tag


def resolve_engine_tag(engine: Engine, arch: str, requested: str = "",
                       backend: str = "") -> str:
    """Newest tag matching ``tag_re`` that carries our asset.

    Reading the *release list* rather than ``/releases/latest`` is the point:
    ``latest`` on whisper.cpp is the asset-less semver tag, so "latest" is
    exactly the wrong answer.
    """
    if requested:
        return _normalize_tag(engine, requested)
    pattern = engine.asset_pattern(backend)
    if not pattern:
        return ""
    release = select_release(
        _releases(gh_api(f"{engine.api}/releases")),
        engine.channel, engine.tag_re, pattern.format(arch=arch))
    return release.get("tag_name", "") if release else ""


# --- plan and install ------------------------------------------------------


@dataclass
class Plan:
    """One target's resolved decision: current state, candidate, destination."""

    engine: Engine
    directory: Path
    tag: str = ""
    current: str = ""
    url: str = ""
    size: int = 0
    payload: dict | None = None
    source: str = ""
    policy: str = ""
    noop: bool = False
    note: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def target(self) -> Path:
        return self.directory / self.engine.versioned.format(tag=self.tag)


def _missing_asset_error(engine: Engine, tag: str, arch: str,
                         backend: str) -> str:
    if engine.key == "cpp":
        return f"Error: Could not find {backend} ({arch}) asset for {tag}"
    if engine.key == "swap":
        return f"Error: Could not find llama-swap asset for {tag} ({arch})"
    return (f"Error: Could not find a {engine.label} {arch} "
            f"{backend} asset for {tag}")


def _not_found_error(engine: Engine, tag: str) -> str:
    if engine.key == "cpp":
        return f"Error: Version {tag} not found"
    return f"Error: {engine.label} version {tag} not found"


def plan_engine(engine: Engine, install_dir: Path, arch: str = "",
                requested_tag: str = "", user: str = "",
                backend: str = "", policy: str = "") -> Plan:
    """Resolve one engine to a :class:`Plan` without downloading anything."""
    plan = Plan(engine=engine, directory=install_dir,
                policy=policy or engine.checksum)
    plan.current = current_engine_tag(engine, install_dir)
    pattern = engine.asset_pattern(backend)
    if not pattern:
        plan.error = (f"Error: {engine.label} has no {backend!r} build "
                      f"(offers: {', '.join(engine.backends)})")
        return plan
    if engine.key == "cpp" and backend == "rocm" and arch != "x64":
        plan.error = f"Error: rocm builds are Linux x64 only (detected {arch})"
        return plan
    pattern = pattern.format(arch=arch)

    requested = _normalize_tag(engine, requested_tag)
    if requested:
        plan.tag = requested
        if _engine_is_intact(engine, plan.target):
            plan.noop = True
            plan.note = f"{engine.label} {requested} already installed"
            return plan
        if engine.kind == "single":
            number = _tag_number(requested)
            primary = install_dir / engine.primary
            installed = (installed_llama_swap_version(primary, user)
                         if primary.exists() else 0)
            if number is not None and installed == number:
                plan.noop = True
                plan.note = f"{engine.label} {requested} already installed"
                return plan
        payload = release_assets(f"{engine.api}/releases/tags/{requested}")
        if _is_error_payload(payload):
            plan.error = _not_found_error(engine, requested)
            return plan
        _warn_if_flagged(engine, requested, payload)
        plan.payload = payload
    else:
        if engine.key == "cpp":
            tag = latest_llama_cpp_tag(pattern)
            if not tag:
                plan.error = "Error: Could not fetch latest llama.cpp release tag"
                return plan
            build = cpp_build_number(tag)
            highest = highest_installed(install_dir)
            if build is not None and highest and build < highest:
                plan.current = f"b{highest}"
                plan.noop = True
                plan.note = (f"Already have version b{highest}, "
                             f"which is newer than {tag}")
                return plan
        elif engine.key == "swap":
            tag = latest_llama_swap_tag()
            if not tag:
                plan.error = "Error: Could not fetch latest llama-swap tag"
                return plan
            primary = install_dir / engine.primary
            installed = (installed_llama_swap_version(primary, user)
                         if primary.exists() else 0)
            number = _tag_number(tag)
            if installed and number is not None and installed >= number:
                plan.current = f"v{installed}"
                plan.noop = True
                plan.note = (f"llama-swap v{installed} already up-to-date "
                             f"(latest {tag})")
                return plan
        else:
            tag = resolve_engine_tag(engine, arch, "", backend)
            if not tag:
                plan.error = (f"Error: Could not fetch a {engine.label} tag "
                              f"carrying a {arch} build (tags matching "
                              f"{engine.tag_re})")
                return plan
        plan.tag = tag
        if _engine_is_intact(engine, plan.target):
            plan.noop = True
            plan.note = f"{engine.label} {tag} already installed at {plan.target.name}"
            return plan
        payload = release_assets(f"{engine.api}/releases/tags/{tag}")
        if _is_error_payload(payload):
            plan.error = _not_found_error(engine, tag)
            return plan
        plan.payload = payload

    entry = find_asset_entry(plan.payload, pattern)
    if not entry:
        plan.error = _missing_asset_error(engine, plan.tag, arch, backend)
        return plan
    plan.url = entry.get("browser_download_url", "")
    plan.size = int(entry.get("size") or 0)
    return plan


def _warn_if_flagged(engine: Engine, tag: str, payload) -> None:
    words = _flagged_words(payload)
    prerelease = isinstance(payload, dict) and payload.get("prerelease")
    if words:
        eprint(f"Warning: {engine.label} {tag} is marked "
               f"{', '.join(words)}; installing as requested")
    elif prerelease:
        eprint(f"Warning: {engine.label} {tag} is a prerelease; "
               "installing as requested")


def apply_plan(plan: Plan) -> bool:
    if plan.error:
        eprint(plan.error)
        return False
    if plan.noop:
        print(plan.note)
        return True
    installed = install_archive(
        plan.engine, plan.url, plan.tag, plan.directory,
        plan.payload or {}, policy=plan.policy)
    return installed is not None


def install_archive(engine: Engine, url: str, tag: str, install_dir: Path,
                    payload, *, policy: str = "") -> Path | None:
    """Download, verify, extract and promote one archive."""
    try:
        install_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        eprint(f"Error: could not create {install_dir}: {exc}")
        return None
    versioned = install_dir / engine.versioned.format(tag=tag)
    try:
        with staging(install_dir, engine.primary) as stage:
            archive_name = url.rsplit("/", 1)[-1]
            archive = stage / archive_name
            if not fetch(url, archive, f"{engine.label} {tag}"):
                return None
            if not verify_archive(archive, archive_name, payload,
                                  policy or engine.checksum, install_dir):
                return None
            print("Extracting...")
            tree = stage / "tree"
            try:
                extract_archive(archive, tree, engine.fmt)
            except (tarfile.TarError, zipfile.BadZipFile, ValueError,
                    OSError) as exc:
                eprint(f"Error: Failed to extract {archive_name}: {exc}")
                return None

            if engine.kind == "single":
                binary = find_extracted(tree, engine.names)
                if binary is None:
                    eprint(f"Error: {archive_name} contains no {engine.primary} "
                           f"binary (looked for {', '.join(engine.names)})")
                    return None
                os.chmod(binary, 0o755)
                install_as = binary
            else:
                root = bundle_root(tree)
                binary = find_extracted(root, engine.names)
                if binary is None:
                    eprint(f"Error: {archive_name} contains no {engine.primary} "
                           f"binary (looked for {', '.join(engine.names)})")
                    return None
                os.chmod(binary, 0o755)   # execute bit only after verification
                install_as = root
            promote(install_as, versioned, stage)
    except OSError as exc:
        eprint(f"Error: could not install {engine.label} into "
               f"{install_dir}: {exc}")
        return None

    print(f"Installed {engine.label} {tag} to {versioned}")
    return versioned


def swap_symlink(target: Path, link_target: str, backup: Path) -> tuple[bool, bool]:
    """Back up a real file being migrated, then point ``target`` somewhere new.

    Returns ``(ok, backed_up)``. A symlink repoint copies nothing: the
    versioned history beside it is the rollback, and copying a 550 MB engine
    binary to ``.bak`` on every update was how this tree grew. Only a real
    file — the shape the old bash version left behind — is copied aside.
    """
    if target.is_symlink():
        if os.readlink(target) == link_target:
            return True, False
        replace_symlink(target, link_target)
        print(f"{target.name} -> {link_target}")
        return True, False
    if target.is_file():
        print(f"Backing up old {target.name} to {backup.name}")
        try:
            shutil.copyfile(target, backup)
        except OSError as exc:
            eprint(f"Error: could not back up {target.name} to {backup}: {exc}")
            return False, False
        target.unlink()
        print(f"Migrated {target.name} to a symlink")
        replace_symlink(target, link_target)
        print(f"{target.name} -> {link_target}")
        return True, True
    replace_symlink(target, link_target)
    print(f"{target.name} -> {link_target}")
    return True, False


def _binary_version_line(binary: Path, user: str = "") -> str:
    cmd = [str(binary), "--version"]
    if user:
        cmd = ["runuser", "-u", user, "--", *cmd]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    lines = result.stdout.splitlines()
    return lines[0] if lines else ""


def installed_llama_swap_version(binary: Path, user: str = "") -> int:
    match = re.search(r"v([0-9]+)", _binary_version_line(binary, user))
    return int(match.group(1)) if match else 0


def link_target_for(engine: Engine, entry: Path) -> str:
    """Where the stable name should point for one installed versioned entry."""
    if engine.kind == "single":
        return entry.name
    binary = find_extracted(entry, engine.names)
    if binary is None:
        return ""
    return f"{engine.link_prefix}{entry.name}/{binary.relative_to(entry)}"


def entry_binary(engine: Engine, entry: Path) -> Path | None:
    """The executable inside one installed entry."""
    if engine.kind == "single":
        return entry
    return find_extracted(entry, engine.names)


def executable_by(path: Path | None, user: str = "") -> tuple[bool, str]:
    """``(ok, why_not)``: can *user* run this? Ownership is not the question."""
    who = user or "this user"
    if path is None or not path.is_file():
        return False, f"{path} is not a file"
    if not user:
        return os.access(path, os.X_OK), f"not executable by {who}"
    runuser = shutil.which("runuser")
    if runuser is None:
        return False, f"cannot check as {user}: runuser not found"
    res = subprocess.run([runuser, "-u", user, "--", "test", "-x", str(path)],
                         capture_output=True, text=True)
    if res.returncode == 0:
        return True, ""
    message = (getattr(res, "stderr", "") or getattr(res, "stdout", "")
               or "").strip().splitlines()
    detail = message[0] if message else "refused"
    if "does not exist" in detail or "no such" in detail.lower():
        return False, f"no such account: {user}"
    if "may not be used" in detail or "permission" in detail.lower():
        return False, f"cannot check as {user} (run as root): {detail}"
    return False, f"not executable by {user}"


def ensure_links(engine: Engine, install_dir: Path,
                 preferred: str = "", user: str = "") -> tuple[bool, bool]:
    """Point the stable name at an intact entry, from what is on disk.

    *preferred* is the tag this run just installed, so an explicit pin moves
    the link even when it is older than the newest entry beside it (that is
    what makes a rollback by tag work). Without it, the newest intact entry
    wins, so a run that was killed after the rename converges on the next one
    instead of reporting "already installed" forever with a stale link.
    """
    installed = versioned_entries(engine, install_dir)
    if preferred:
        pinned = install_dir / engine.versioned.format(tag=preferred)
        installed = [pinned, *(e for e in installed if e != pinned)]
    for entry in installed:
        if not _engine_is_intact(engine, entry):
            continue
        target = link_target_for(engine, entry)
        if not target:
            continue
        ok, why = executable_by(entry_binary(engine, entry), user)
        if not ok:
            eprint(f"Note: {entry.name}: {why}; not linking it")
            continue
        if engine.key == "cpp":
            _ensure_stable_primary(install_dir, engine.primary, target)
            return True, False
        return swap_symlink(install_dir / engine.primary, target,
                            install_dir / engine.backup)
    return True, False


def _ensure_stable_primary(install_dir: Path, primary: str,
                           link_target: str) -> None:
    """Keep ``<primary>`` a symlink to *link_target*, creating it if absent.

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


# --- pruning ---------------------------------------------------------------


def normalize_link_target(target: str) -> str:
    """The versioned entry a symlink target starts in, without ``./``."""
    text = target
    while text.startswith("./"):
        text = text[2:]
    return text.split("/", 1)[0]


def cleanup_engine_versions(engine: Engine, install_dir: Path,
                            keep: int, referenced: set[str]) -> None:
    """Drop superseded versioned entries, newest ``keep`` kept.

    An entry any symlink still points at is never removed — deleting it
    would leave a dangling link, and the previous release staying readable is
    what makes rollback possible.
    """
    if not _can_prune(install_dir):
        return
    keep = keep if isinstance(keep, int) and keep >= 1 else 1
    entries = versioned_entries(engine, install_dir)
    if len(entries) <= keep:
        return
    roots = {normalize_link_target(target) for target in referenced}
    for entry in entries[keep:]:
        if entry.name in roots:
            continue
        print(f"Removing old {engine.primary}: {entry.name}")
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def _remove_leftovers(engine: Engine, install_dir: Path,
                      keep_backup: bool) -> None:
    for entry in install_dir.glob(".stage-*"):
        print(f"Removing interrupted staging: {entry.name}")
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    quarantined = sorted(
        install_dir.glob("_HASH_FAIL_*"),
        key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in quarantined[1:]:
        print(f"Removing old quarantine: {stale.name}")
        stale.unlink(missing_ok=True)
    if not engine.backup or keep_backup:
        return
    backup = install_dir / engine.backup
    primary = install_dir / engine.primary
    if backup.exists() and primary.is_symlink() and len(
            versioned_entries(engine, install_dir)) > 1:
        print(f"Removing stale backup: {backup.name}")
        backup.unlink(missing_ok=True)


def prune_engine(engine: Engine, install_dir: Path, keep: int,
                 keep_backup: bool = False) -> None:
    """Keep the newest entries, then drop this tool's abandoned leftovers."""
    if not _can_prune(install_dir):
        return
    referenced = {os.readlink(entry) for entry in install_dir.iterdir()
                  if entry.is_symlink()}
    cleanup_engine_versions(engine, install_dir, keep, referenced)
    _remove_leftovers(engine, install_dir, keep_backup)


# --- reporting -------------------------------------------------------------


def report_plan(plans: list[Plan]) -> None:
    print("\nPlan:")
    for plan in plans:
        if plan.error:
            print(f"  {plan.engine.key:<8} {plan.error}")
        elif plan.noop:
            print(f"  {plan.engine.key:<8} {plan.note}")
        else:
            current = plan.current or "not installed"
            size = f"{plan.size / 1e6:.1f} MB" if plan.size else "size unknown"
            print(f"  {plan.engine.key:<8} {current} -> {plan.tag}  "
                  f"{size}  {plan.directory}")


def report_summary(plans: list[Plan], results: list[bool]) -> None:
    failed = [ok for ok in results if not ok]
    print("\nDone:" if not failed else "\nDone (with failures):")
    for plan, ok in zip(plans, results):
        if not ok:
            reason = plan.error or "see the plan above"
            print(f"  {plan.engine.label:<22} not updated: {reason}")
        elif plan.noop:
            print(f"  {plan.engine.label:<22} up to date")
        else:
            print(f"  {plan.engine.label:<22} installed {plan.tag}")


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
    """Engine-version keep count: flag, then env, then the default."""
    if args.keep_engine_versions is not None:
        return args.keep_engine_versions
    env = os.environ.get("KEEP_ENGINE_VERSIONS", "")
    return int(env) if env.isdigit() else DEFAULT_KEEP_ENGINE_VERSIONS


def keep_swap_default() -> int:
    raw = os.environ.get("KEEP_SWAP_VERSIONS", "")
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return DEFAULT_KEEP_SWAP_VERSIONS


def keep_cpp_default() -> int:
    raw = os.environ.get("KEEP_CPP_VERSIONS", "")
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return KEEP_VERSIONS


def resolve_positionals(
    args: list[str], swap_version: str, cpp_only: bool, swap_only: bool
) -> tuple[list[str], str, str, str, list[str]]:
    """Split positionals into (target tokens, backend, version, swap tag, notes).

    A positional is a target selector first, then a backend name, and only
    then a version — so ``update.py sd rocm`` reads the way it looks.
    """
    version = DEFAULT_VERSION
    backend = ""
    notes: list[str] = []
    index = _alias_index()
    rest: list[str] = []
    for arg in args:
        targets = resolve_target(arg, index)
        if targets is not None:
            rest.append(arg)
            continue
        low = arg.lower()
        if low in KNOWN_BACKENDS:
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
            "Download and install prebuilt engine binaries. Run it from the "
            "install directory (it updates tools/) or from tools itself. Each "
            "engine is updated where it already is; a new one goes under the "
            "configured root. Run with --plan to see the resolved versions "
            "without downloading anything."
        ),
        epilog=(
            "Targets (positional, first match wins; 'all' must be typed in full):\n"
            + "\n".join(alias_lines)
            + f"\n  {'all':<9} {' '.join(ALL_ALIASES)}   # everything\n"
            "  (a glob works too: 's*' selects swap and sd)\n"
            "With no target, all five engines are updated; 'all' spells it out.\n"
            "A named version pins only that target.\n"
            "\n"
            "Backends (positional): " + ", ".join(KNOWN_BACKENDS)
            + "\n\n"
            "Install locations, highest precedence first:\n"
            "  --dir <engine>=<path>       one run only\n"
            "  $<ENGINE>_BIN_DIR           the same variable llama-packer reads\n"
            "  an existing install in tools/ or the install directory\n"
            "  $LLAMA_DIR, else the run directory's tools/\n"
            "\n"
            "Environment:\n"
            "  GITHUB_TOKEN            Optional. GitHub personal access token for\n"
            "                          higher API rate limits (5000 req/hr vs 60).\n"
            "                          Also reads GITHUB_PERSONAL_ACCESS_TOKEN.\n"
            "  KEEP_CPP_VERSIONS       Optional. Same as --keep-cpp.\n"
            "  KEEP_SWAP_VERSIONS      Optional. Same as --keep-swap.\n"
            "  KEEP_ENGINE_VERSIONS    Optional. Same as --keep-engine-versions.\n"
            "  LLAMA_DIR               Optional. Install root when a target has\n"
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
        "--keep-cpp",
        type=positive_int,
        metavar="N",
        default=None,
        help=f"How many llama.cpp build dirs to keep (default {KEEP_VERSIONS}). "
        "Or KEEP_CPP_VERSIONS.",
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
        "--fetch",
        action="store_true",
        help="Accepted alias for the default: a bare run already downloads "
             "and installs the newest release.",
    )
    parser.add_argument(
        "--user",
        default="",
        metavar="ACCOUNT",
        help="Service account that must be able to execute a tool before it "
             "is linked (default: the invoking user).",
    )
    parser.add_argument(
        "--dir",
        action="append",
        metavar="ENGINE=PATH",
        help="Install this engine into PATH for this run (repeatable). "
             "Overrides every configured and discovered location.",
    )
    parser.add_argument(
        "--require-checksums",
        action="store_true",
        help="Refuse any archive whose release publishes no checksums.txt. "
             "The default installs it with a warning instead.",
    )
    parser.add_argument(
        "--plan",
        "--dry-run",
        dest="plan",
        action="store_true",
        help="Resolve every selected target and print current -> candidate, "
             "size and destination, then exit without downloading.",
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


def _select_targets(args: argparse.Namespace, targets: tuple[str, ...]
                    ) -> tuple[str, ...] | None:
    if targets:
        if args.cpp_only or args.swap_only:
            eprint("Error: target selectors and --cpp-only/--swap-only are "
                   "alternatives; pass one or the other")
            return None
        return targets
    if args.swap_only:
        return ("swap",)
    if args.cpp_only:
        return ("cpp",)
    if args.skip_swap:
        return ("cpp",)
    return ALL_TARGETS


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
    selected = _select_targets(args, resolve_targets(target_tokens))
    if selected is None:
        return 1

    for note in notes:
        eprint(note)
    print("Targets: " + ", ".join(selected))

    install_dirs = resolve_install_dirs(dir_overrides)
    report_install_dirs(install_dirs)
    if args.print_paths:
        return 0

    keep_swap = args.keep_swap if args.keep_swap is not None else keep_swap_default()
    keep_cpp = args.keep_cpp if args.keep_cpp is not None else keep_cpp_default()
    keep_engines = resolve_keeps(args)
    policy = "required" if args.require_checksums else ""

    if version == DEFAULT_VERSION:
        cpp_backend, cpp_tag = "", ""
    else:
        cpp_backend, cpp_tag = split_cpp_version(version)
    cpp_backend = (cpp_backend
                   or ("rocm" if backend == "hip" else backend)
                   or "vulkan")

    pinned = {
        "cpp": cpp_tag,
        "swap": swap_version,
        "audio": args.audio_cpp_version,
        "whisper": args.whisper_version,
        "sd": args.sd_version,
    }

    status = 0
    plans: list[Plan] = []
    for key in selected:
        directory = install_dirs[key][0]
        try:
            if key == "cpp":
                plan = plan_engine(
                    CPP, directory, arch=get_llama_cpp_arch(args.arch),
                    requested_tag=cpp_tag, user=args.user,
                    backend=cpp_backend, policy=policy)
            elif key == "swap":
                plan = plan_engine(
                    SWAP, directory, arch=get_llama_swap_arch(args.arch),
                    requested_tag=swap_version, user=args.user,
                    policy=policy)
            else:
                engine = ENGINES[key]
                plan = plan_engine(
                    engine, directory, arch=engine_arch(engine, args.arch),
                    requested_tag=pinned[key], user=args.user,
                    backend=backend or engine.single_backend, policy=policy)
        except (ApiError, ArchError) as exc:
            eprint(f"Warning: {key} plan failed: {exc}")
            status = 1
            plan = Plan(engine=ENGINES[key], directory=directory,
                        error=str(exc))
        plans.append(plan)

    report_plan(plans)
    if args.plan:
        return 0 if all(plan.ok for plan in plans) else 1

    keeps = {"cpp": keep_cpp, "swap": keep_swap}
    results: list[bool] = []
    directories = sorted({plan.directory for plan in plans}, key=str)
    with contextlib.ExitStack() as locks:
        for directory in directories:
            locks.enter_context(dir_lock(directory))
        for plan in plans:
            ok = apply_one(plan, keeps.get(plan.engine.key, keep_engines),
                           args.user)
            results.append(ok)
            if not ok:
                status = 1
    report_summary(plans, results)
    return status


def apply_one(plan: Plan, keep: int, user: str = "") -> bool:
    """Apply one plan: install if needed, then always converge the links.

    Links and pruning are derived from the tree rather than from whether this
    run installed something, so a run that died between the rename and the
    link is finished by the next one instead of being reported as current
    forever.
    """
    if not apply_plan(plan):
        return False
    if plan.tag:
        ok, backed_up = ensure_links(plan.engine, plan.directory, plan.tag,
                                     user)
    else:
        ok, backed_up = True, False
    if ok:
        prune_engine(plan.engine, plan.directory, keep,
                     keep_backup=backed_up)
    return ok


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
