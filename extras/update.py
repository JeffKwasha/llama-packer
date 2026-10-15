#!/usr/bin/env python3
"""update — download and install prebuilt llama.cpp + llama-swap binaries.

Python port of the ``extras/update`` bash script (same CLI, same messages,
same install layout). Binaries install into the **current working directory**,
not the script's location — run it from wherever you want ``llama-b*/`` and
``llama-swap`` to live.

Unlike the bash version this can never be "sourced" harmlessly into a shell
(the ``. extras/update --help`` habit that used to close the terminal): a
Python script always runs as its own process.

Usage: ./extras/update.py [version|vulkan|rocm] [options]
Examples:
    ./extras/update.py             # Latest llama.cpp Vulkan + latest llama-swap
    ./extras/update.py b8913       # Specific llama.cpp version + latest llama-swap
    ./extras/update.py vulkan      # Latest Vulkan build + latest llama-swap
    ./extras/update.py rocm         # Latest ROCm build + latest llama-swap
    ./extras/update.py --swap-only  # Only update llama-swap
    ./extras/update.py --cpp-only   # Only update llama.cpp
    ./extras/update.py --swap-only v252  # Specific llama-swap version
    ./extras/update.py --skip-swap  # Skip llama-swap update

Standard library only — no venv, no third-party deps.
exit 1
"""

from __future__ import annotations

import argparse
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
from pathlib import Path

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
LLAMA_SWAP_REPO = "https://api.github.com/repos/mostlygeek/llama-swap"
KEEP_VERSIONS = 3  # Keep current + N-1 previous llama.cpp build dirs
DEFAULT_KEEP_SWAP_VERSIONS = 2  # Keep current + previous llama-swap binaries
DEFAULT_VERSION = "latest-vulkan"


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


def latest_llama_cpp_tag() -> str:
    for release in _releases(gh_api(RELEASES_URL)):
        if release.get("tag_name"):
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


def _normalize(override: str, mapping: dict[str, str]) -> str | None:
    return mapping.get(override)


def get_llama_swap_arch(override: str = "") -> str:
    if override:
        return _normalize(
            override,
            {
                "linux_amd64": "linux_amd64",
                "x86_64": "linux_amd64",
                "amd64": "linux_amd64",
                "x64": "linux_amd64",
                "linux_arm64": "linux_arm64",
                "aarch64": "linux_arm64",
                "arm64": "linux_arm64",
            },
        ) or override
    machine = platform.machine()
    if machine in ("x86_64", "amd64"):
        return "linux_amd64"
    if machine in ("aarch64", "arm64"):
        return "linux_arm64"
    return "linux_amd64"


def get_llama_cpp_arch(override: str = "") -> str:
    if override:
        # llama.cpp assets use x64 / arm64 / s390x (e.g. llama-b<int>-bin-ubuntu-vulkan-arm64.tar.gz)
        return _normalize(
            override,
            {
                "linux_amd64": "x64",
                "x86_64": "x64",
                "amd64": "x64",
                "x64": "x64",
                "linux_arm64": "arm64",
                "aarch64": "arm64",
                "arm64": "arm64",
                "s390x": "s390x",
            },
        ) or override
    machine = platform.machine()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine == "s390x":
        return "s390x"
    return "x64"


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


def extract_tar_gz(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        try:
            tf.extractall(dest, filter="data")  # Python >= 3.11.4, 3.12+
        except TypeError:  # pragma: no cover - older interpreters
            tf.extractall(dest)


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
    if target_dir.is_dir():
        print(f"llama.cpp {tag} already installed at {target_dir}")
        return True, False

    print(f"Downloading llama.cpp {tag}...")
    with tempfile.TemporaryDirectory(dir=install_dir) as tmp:
        archive = Path(tmp) / url.rsplit("/", 1)[-1]
        try:
            download(url, archive, quiet=False)
        except subprocess.CalledProcessError:
            eprint(f"Error: download failed for {url}")
            return False, False
        print("Extracting...")
        try:
            extract_tar_gz(archive, install_dir)
        except (tarfile.TarError, OSError) as exc:
            eprint(f"Error: extraction failed for {url}: {exc}")
            return False, False

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
        target = os.readlink(link)
        if re.search(r"llama-b[0-9]", target):
            print(f"Updating symlink {link.name} -> {new_target} (was {target})")
            replace_symlink(link, new_target)


def update_llama_cpp(
    install_dir: Path, version: str, arch_override: str
) -> tuple[bool, str | None]:
    """llama.cpp half of the run.

    Returns (ok, installed_tag): installed_tag is the release tag that was
    actually unpacked (so the caller can prune old builds and re-point
    symlinks), or None when nothing new landed.
    """
    arch = get_llama_cpp_arch(arch_override)
    highest = highest_installed(install_dir)

    tag = None
    if "vulkan" in version or version == "latest":
        tag = latest_llama_cpp_tag()
        if not tag:
            eprint("Error: Could not fetch latest llama.cpp release tag")
            return False, None
        if (build := cpp_build_number(tag)) is not None and highest > build:
            print(f"Already have version b{highest}, which is newer than {tag}")
            return True, None
        payload = release_assets(f"{RELEASES_URL}/tags/{tag}")
        url = find_asset(payload, f"ubuntu-vulkan-{arch}")
        if not url:
            eprint(f"Error: Could not find Vulkan ({arch}) asset for {tag}")
            return False, None
        ok, _installed = install_llama_cpp(url, tag, install_dir)
        return (ok, tag) if ok else (False, None)

    if "rocm" in version:
        # Note: the Linux ROCm builds upstream ships are x64-only.
        tag = latest_llama_cpp_tag()
        if not tag:
            eprint("Error: Could not fetch latest llama.cpp release tag")
            return False, None
        if (build := cpp_build_number(tag)) is not None and highest > build:
            print(f"Already have version b{highest}, which is newer than {tag}")
            return True, None
        if arch != "x64":
            eprint(f"Error: ROCm builds are Linux x64 only (detected {arch})")
            return False, None
        payload = release_assets(f"{RELEASES_URL}/tags/{tag}")
        url = find_asset(payload, r"ubuntu-rocm-[^-]*-x64")
        if not url:
            eprint(f"Error: Could not find ROCm asset for {tag}")
            return False, None
        ok, _installed = install_llama_cpp(url, tag, install_dir)
        return (ok, tag) if ok else (False, None)

    # Specific version tag
    payload = release_assets(f"{RELEASES_URL}/tags/{version}")
    if _is_error_payload(payload):
        eprint(f"Error: Version {version} not found")
        return False, None
    url = find_asset(payload, f"ubuntu-vulkan-{arch}")
    if not url and arch == "x64":
        url = find_asset(payload, r"ubuntu-rocm-[^-]*-x64")
    if not url:
        eprint(f"Error: Could not find Linux ({arch}) asset for {version}")
        return False, None
    ok, installed = install_llama_cpp(url, version, install_dir)
    return (ok, version if installed else None)


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
    install_dir: Path, keep_swap: int, requested_tag: str = ""
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

    arch = get_llama_swap_arch()
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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        archive_name = url.rsplit("/", 1)[-1]
        archive_path = tmpdir / archive_name
        checksums_path = tmpdir / "checksums.txt"

        print(f"Downloading llama-swap {tag} ({arch})...")
        try:
            download(url, archive_path)
        except subprocess.CalledProcessError:
            eprint(f"Error: Failed to download {url}")
            return False
        try:
            download(checksums_url, checksums_path)
        except subprocess.CalledProcessError:
            eprint(f"Error: Failed to download checksums {checksums_url}")
            return False

        expected = expected_sha256(checksums_path, archive_name)
        if not expected:
            eprint(f"Error: Could not find hash for {archive_name} in checksums.txt")
            return False

        print("Verifying sha256...")
        actual = sha256_file(archive_path)
        if actual != expected:
            eprint(f"Error: sha256 mismatch for {archive_name}")
            eprint(f"  expected: {expected}")
            eprint(f"  actual:   {actual}")
            fail_path = install_dir / f"_HASH_FAIL_{archive_name}"
            eprint(
                f"Hash verification failed — saving downloaded file as {fail_path} "
                "(no execute bit, symlink not updated)"
            )
            shutil.copyfile(archive_path, fail_path)
            _no_exec(fail_path)
            return False
        print(f"Checksum verified: {actual}")

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
        if not extracted.is_file():
            eprint("Error: Extracted archive does not contain llama-swap binary")
            return False
        # hash verified above; only set exec after successful verification
        os.chmod(extracted, 0o755)

        versioned = install_dir / f"llama-swap-{tag}"  # e.g. llama-swap-v252
        backup = install_dir / "llama-swap.bak"  # single backup file (overwrite)
        try:
            shutil.move(str(extracted), str(versioned))
        except OSError as exc:
            eprint(f"Error: Failed to move verified binary to {versioned}: {exc}")
            return False
        os.chmod(versioned, 0o755)

        # Ownership check before touching symlink/file
        if (target.exists() or target.is_symlink()) and not os.access(
            install_dir, os.W_OK
        ):
            eprint(f"Warning: not owner/writable of {target}, skipping replace")
            return False

        if target.is_symlink():
            current_target = os.readlink(target)
            real_path = target.resolve()  # handles relative links
            if real_path.is_file():
                print(f"Backing up old symlink target {current_target} to {backup}")
                try:
                    shutil.copyfile(real_path, backup)
                except OSError:
                    eprint(f"Warning: could not back up {real_path} to {backup}")
                    return False
            else:
                print(f"Backing up old symlink {target} -> {current_target} to {backup}")
                try:
                    shutil.copyfile(target, backup, follow_symlinks=False)
                except OSError:
                    eprint(f"Warning: could not back up {target} to {backup}")
                    return False
            replace_symlink(target, versioned.name)
            print(f"Updated symlink {target} -> {versioned.name}")
        elif target.is_file():
            print(f"Backing up old binary to {backup} (single file, overwriting)")
            try:
                shutil.copyfile(target, backup)
            except OSError as exc:
                # Unlike the bash version, refuse to destroy the old binary
                # when the backup could not be written.
                eprint(f"Error: could not back up {target} to {backup}: {exc}")
                return False
            target.unlink()
            replace_symlink(target, versioned.name)
            print(f"Migrated {target} to symlink -> {versioned.name} (was regular file)")
        else:
            replace_symlink(target, versioned.name)
            print(f"Created symlink {target} -> {versioned.name}")

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


# --- argument parsing ------------------------------------------------------


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer (got {number})")
    return number


def keep_swap_default() -> int:
    raw = os.environ.get("KEEP_SWAP_VERSIONS", "")
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return DEFAULT_KEEP_SWAP_VERSIONS


def resolve_positionals(
    args: list[str], swap_version: str, cpp_only: bool, swap_only: bool
) -> tuple[str, str, list[str]]:
    """Split positional args into llama.cpp version and (heuristic) swap tag.

    A bare ``v<NNN>`` positional is taken as a llama-swap version when no
    --swap-version was given and --cpp-only is not in play.
    """
    version = DEFAULT_VERSION
    notes: list[str] = []
    for arg in args:
        if re.fullmatch(r"v[0-9]+", arg) and not swap_version and not cpp_only:
            swap_version = arg
            if not swap_only:
                notes.append(
                    f"Note: Interpreting {arg} as llama-swap version "
                    "(use --swap-version to be explicit)"
                )
        else:
            version = arg
    return version, swap_version, notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and install llama.cpp and llama-swap prebuilt binaries. "
            "Install location is the current working directory."
        ),
        epilog=(
            "Environment:\n"
            "  GITHUB_TOKEN            Optional. GitHub personal access token for\n"
            "                          higher API rate limits (5000 req/hr vs 60).\n"
            "                          Also reads GITHUB_PERSONAL_ACCESS_TOKEN.\n"
            "  KEEP_SWAP_VERSIONS      Optional. Same as --keep-swap.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "args",
        nargs="*",
        metavar="version",
        help="Specific llama.cpp version tag (e.g., b8913, b8902), or "
        "vulkan (default) / rocm",
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
        default="",
        help="Override the detected CPU architecture for both downloads "
        "(x64/amd64/linux_amd64 or arm64/aarch64/linux_arm64)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    install_dir = Path.cwd()

    if args.swap_only and args.cpp_only:
        eprint("Error: --swap-only and --cpp-only are mutually exclusive")
        return 1

    version, swap_version, notes = resolve_positionals(
        args.args, args.swap_version, args.cpp_only, args.swap_only
    )
    for note in notes:
        eprint(note)

    keep_swap = (
        args.keep_swap
        if args.keep_swap is not None
        else keep_swap_default()
    )

    # Main: llama.cpp
    if not args.swap_only:
        ok, installed_tag = update_llama_cpp(install_dir, version, args.arch)
        if not ok:
            return 1
        if installed_tag is not None:
            # Cleanup old versions after successful install and re-point any
            # symlinks that target an old llama-b* dir at the new one.
            cleanup_old_versions(install_dir)
            update_llama_cpp_symlinks(install_dir, installed_tag)

    # Main: llama-swap (unless skipped)
    if args.swap_only:
        skip_swap = False
    elif args.cpp_only:
        skip_swap = True
    else:
        skip_swap = args.skip_swap

    if not skip_swap:
        if swap_version:
            if not update_llama_swap(install_dir, keep_swap, swap_version):
                return 1
        elif not update_llama_swap(install_dir, keep_swap):
            eprint("Warning: llama-swap update failed")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ApiError as exc:
        eprint(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
