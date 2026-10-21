"""Tests for extras/update.py — the prebuilt server-binary installer.

The installer reaches only two seams (``gh_api`` and ``download``), so every
network path is monkeypatched here; nothing in this file touches the network
or the real install directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

EXTRAS = Path(__file__).resolve().parents[1] / "extras"

_spec = importlib.util.spec_from_file_location("update", EXTRAS / "update.py")
upd = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = upd  # dataclass resolves annotations via sys.modules
_spec.loader.exec_module(upd)

SWAP_URL = "https://example.com/llama-swap-v262-linux_amd64.tar.gz"
CHECKSUMS_URL = "https://example.com/checksums.txt"
CPP_URL = "https://example.com/llama-b11539-bin-ubuntu-vulkan-x64.tar.gz"


# --- helpers ---------------------------------------------------------------


def make_swap_tarball(text: str = "version: v262 (fake), built at 2026-10-03\n") -> bytes:
    """A gzip tarball containing a single executable llama-swap stub."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        payload = f"#!/bin/sh\necho '{text}'\n".encode()
        info = tarfile.TarInfo("llama-swap")
        info.size = len(payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def make_cpp_tarball(tag: str = "b11539") -> bytes:
    """A gzip tarball that extracts to llama-<tag>/ like upstream's does."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(f"llama-{tag}/llama-server")
        payload = b"#!/bin/sh\necho server\n"
        info.size = len(payload)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def checksums_line(name: str, blob: bytes, prefix: str = "  ") -> str:
    return f"{hashlib.sha256(blob).hexdigest()}{prefix}{name}\n"


def fake_api(swap_tag: str = "v262", cpp_tag: str = "b11539", urls=None):
    """A gh_api stand-in: latest tag list, or a release with assets."""
    assets = [
        {
            "browser_download_url": (
                urls[0] if urls else SWAP_URL
            )
        },
        {"browser_download_url": urls[1] if urls else CHECKSUMS_URL},
    ]

    def _api(url: str):
        if url.endswith("/releases"):
            if "llama-swap" in url:
                return [{"tag_name": swap_tag}]
            return [{"tag_name": cpp_tag}]
        if "/tags/" in url:
            return {"assets": assets, "tag_name": url.rsplit("/", 1)[-1]}
        return []

    return _api


def stub_download(blob: bytes, checksum: str):
    """A download stand-in that hands out blob (archive) or checksum text."""

    def _download(url: str, dest: Path, *, quiet: bool = True) -> None:
        if url.endswith(".txt"):
            dest.write_text(checksum)
        else:
            dest.write_bytes(blob)

    return _download


def make_binary(path: Path, text: str) -> Path:
    path.write_text(f"#!/bin/sh\necho '{text}'\n")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _no_deployment_config(monkeypatch, tmp_path):
    """Keep every test off this box's committed ``extras/engines.ini``.

    Two separate hazards, one fixture. The shipped file names a real
    deployment's directories, so a test that resolves install locations
    without saying where they come from would assert against whichever box the
    suite runs on -- and, worse, *install into it*: resolution feeds the
    installers, and a stubbed download plus a real deployment directory is a
    real deployment directory with a fake binary in it. That happened once
    (a 22-byte llama-server in /opt/llama/tools) before this fixture existed.

    Pinning the discovery candidates to ``tmp_path`` as well closes the second
    door: with the shipped config unreadable, discovery walks outwards from
    the cwd and would otherwise reach the real repo and the real tools root.
    Tests that care about configuration pass it in explicitly.
    """
    monkeypatch.setattr(upd, "config_path",
                        lambda: tmp_path / "no-such-engines.ini")
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])


# --- version discovery / bookkeeping ---------------------------------------


def test_highest_installed_ignores_non_numeric_suffix(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Highest installed llama.cpp build number among llama-b<NNNN> dirs.
    (tmp_path / "llama-b11507").mkdir()
    (tmp_path / "llama-b8913").mkdir()
    (tmp_path / "llama-b1234-extra").mkdir()  # upstream never ships this
    (tmp_path / "llama-bnope").mkdir()
    assert upd.highest_installed(tmp_path) == 11507


def test_highest_installed_empty(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Highest installed llama.cpp build number among llama-b<NNNN> dirs.
    assert upd.highest_installed(tmp_path) == 0


def test_cpp_build_number():
    # Validates: docs/install.md, 'Where an update installs': 11507 from 'b11507'; None when the tag has no numeric build part.
    assert upd.cpp_build_number("b11507") == 11507
    assert upd.cpp_build_number("11507") == 11507
    assert upd.cpp_build_number("v262") is None


def test_cleanup_old_versions_keeps_newest_n(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Remove old llama-b<NNNN> dirs, keeping only the newest ``keep``.
    for build in (10000, 11000, 11500, 11501, 11502):
        (tmp_path / f"llama-b{build}").mkdir()
    upd.cleanup_old_versions(tmp_path, keep=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "llama-b11500",
        "llama-b11501",
        "llama-b11502",
    ]


def test_cleanup_old_versions_noop_when_not_owner(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': cleanup only removes what this tool installed.
    for build in (10000, 11000, 11500, 11501):
        (tmp_path / f"llama-b{build}").mkdir()
    monkeypatch.setattr(upd, "_owned_by_current_user", lambda _p: False)
    upd.cleanup_old_versions(tmp_path, keep=1)
    assert len(list(tmp_path.iterdir())) == 4


def test_prune_swap_keeps_newest_but_never_a_referenced_file(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': the newest is kept and a file something still points at is never removed.
    for version in (260, 261, 262):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    os.symlink("llama-swap-v260", tmp_path / "rollback-link")

    upd.cleanup_old_llama_swap_versions(tmp_path, keep=2)

    # newest two kept; v260 survives only because a symlink points at it
    assert (tmp_path / "llama-swap-v262").is_file()
    assert (tmp_path / "llama-swap-v261").is_file()
    assert (tmp_path / "llama-swap-v260").is_file()


def test_prune_swap_removes_unreferenced(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Remove old llama-swap-vNNN binaries.
    for version in (260, 261, 262):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    upd.cleanup_old_llama_swap_versions(tmp_path, keep=2)
    assert not (tmp_path / "llama-swap-v260").exists()


def test_prune_swap_ignores_bak_and_foreign_files(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Remove old llama-swap-vNNN binaries.
    for version in (260, 261):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    (tmp_path / "llama-swap.bak").write_bytes(b"x")
    (tmp_path / "_HASH_FAIL_llama-swap-v262-linux_amd64.tar.gz").write_bytes(b"y")
    make_binary(tmp_path / "llama-swap", "stub")  # primary, possibly a symlink

    upd.cleanup_old_llama_swap_versions(tmp_path, keep=1)
    assert not (tmp_path / "llama-swap-v260").exists()
    assert (tmp_path / "llama-swap").exists()


def test_prune_swap_bogus_keep_defaults_to_one(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Remove old llama-swap-vNNN binaries.
    for version in (260, 261, 262):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    upd.cleanup_old_llama_swap_versions(tmp_path, keep=0)
    assert (tmp_path / "llama-swap-v262").is_file()
    assert not (tmp_path / "llama-swap-v261").exists()


# --- symlinks --------------------------------------------------------------


def test_update_cpp_symlinks_repoints_only_llama_b_links(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Repoint top-level symlinks that target an old llama-b* dir.
    (tmp_path / "llama-b12000").mkdir()
    os.symlink("llama-b11507", tmp_path / "server")
    os.symlink("llama-b11507", tmp_path / "server2")
    os.symlink("llama-swap-v262", tmp_path / "swap")  # not ours to touch
    (tmp_path / "llama-b11999").mkdir()

    upd.update_llama_cpp_symlinks(tmp_path, "b12000")

    assert os.readlink(tmp_path / "server") == "llama-b12000"
    assert os.readlink(tmp_path / "server2") == "llama-b12000"
    assert os.readlink(tmp_path / "swap") == "llama-swap-v262"


def test_update_cpp_symlinks_absent_target_is_noop(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': a symlink whose target is gone is left alone.
    os.symlink("llama-b11507", tmp_path / "server")
    upd.update_llama_cpp_symlinks(tmp_path, "b12000")
    assert os.readlink(tmp_path / "server") == "llama-b11507"


def test_update_cpp_symlinks_accepts_plain_build(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': Repoint top-level symlinks that target an old llama-b* dir.
    (tmp_path / "llama-b12000").mkdir()
    os.symlink("llama-b11507", tmp_path / "server")
    upd.update_llama_cpp_symlinks(tmp_path, "b12000")
    assert os.readlink(tmp_path / "server") == "llama-b12000"


def test_update_swap_symlinks_repoints_but_skips_primary(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': each engine keeps its history beside the live one and is reached through a stable name
    make_binary(tmp_path / "llama-swap-v262", "stub")
    os.symlink("llama-swap-v261", tmp_path / "llama-swap")
    os.symlink("llama-swap-v261", tmp_path / "rollback")
    os.symlink("llama-b11507", tmp_path / "server")  # not ours to touch

    upd.update_llama_swap_symlinks(tmp_path, "llama-swap-v262")

    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v261"
    assert os.readlink(tmp_path / "rollback") == "llama-swap-v262"
    assert os.readlink(tmp_path / "server") == "llama-b11507"


# --- argument parsing ------------------------------------------------------


def test_help_exits_zero(capsys):
    # Validates: docs/install.md, 'The updater command line': the tool's own help lists the targets, the flags and the precedence
    with pytest.raises(SystemExit) as excinfo:
        upd.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_unknown_option_exits_two(capsys):
    # Validates: docs/install.md, 'The updater command line': an unknown option or a bad value stops the run rather than being ignored
    with pytest.raises(SystemExit) as excinfo:
        upd.build_parser().parse_args(["--bogus"])
    assert excinfo.value.code == 2
    assert "--bogus" in capsys.readouterr().err


def test_keep_swap_rejects_non_positive():
    # Validates: docs/install.md, 'The updater command line': a non-positive keep count is refused
    with pytest.raises(SystemExit):
        upd.build_parser().parse_args(["--keep-swap", "0"])
    with pytest.raises(SystemExit):
        upd.build_parser().parse_args(["--keep-swap", "abc"])


def _positional(args, **kw):
    """Unpack resolve_positionals' five outputs."""
    tokens, backend, version, swap, notes = upd.resolve_positionals(
        args, kw.get("swap_version", ""),
        cpp_only=kw.get("cpp_only", False),
        swap_only=kw.get("swap_only", False))
    return tokens, backend, version, swap, notes


def test_positional_v_number_becomes_swap_version():
    # Validates: docs/install.md, 'The updater command line': a bare v<N> is a llama-swap version and a bNNNN is a llama.cpp build
    _, _, version, swap, notes = _positional(["v252"])
    assert (version, swap) == (upd.DEFAULT_VERSION, "v252")
    assert len(notes) == 1  # ambiguity note


def test_positional_v_number_with_swap_only_is_silent():
    # Validates: docs/install.md, 'The updater command line': --swap-only ignores a build number
    _, _, _, swap, notes = _positional(["v252"], swap_only=True)
    assert swap == "v252"
    assert notes == []


def test_positional_cpp_only_ignores_v_number():
    # Validates: docs/install.md, 'The updater command line': --cpp-only ignores a swap version
    _, _, version, swap, _ = _positional(["v252"], cpp_only=True)
    assert (version, swap) == ("v252", "")


def test_positional_last_build_wins():
    # Validates: docs/install.md, 'The updater command line': when two version positionals are given, the last one wins
    _, _, version, swap, _ = _positional(["b8913", "v252", "b8902"])
    assert (version, swap) == ("b8902", "v252")


def test_positional_backend_is_separate_from_version():
    # `sd rocm` has to read as "the sd target, the rocm backend" rather than
    # swallowing rocm as a version string.
    # Validates: docs/install.md, 'Where an update installs': Turn positional target selectors into a deduplicated target list.
    tokens, backend, version, _, _ = _positional(["sd", "rocm"])
    assert upd.resolve_targets(tokens) == ("sd",)
    assert backend == "rocm"
    assert version == upd.DEFAULT_VERSION


def test_main_rejects_mutually_exclusive(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater command line': --swap-only and --cpp-only are mutually exclusive
    monkeypatch.chdir(tmp_path)
    assert upd.main(["--swap-only", "--cpp-only"]) == 1
    assert "mutually exclusive" in capsys.readouterr().err


# --- GitHub API ------------------------------------------------------------


def test_gh_api_parses_status_line(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': GET a GitHub API URL and return the parsed JSON body. Uses curl rather than urllib: this box reaches the network through a.
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '[{"tag_name": "b1"}]\n200'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    assert upd.gh_api("https://example.com/") == [{"tag_name": "b1"}]


def test_gh_api_401_is_auth_failure(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': GET a GitHub API URL and return the parsed JSON body. Uses curl rather than urllib: this box reaches the network through a.
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '{"message": "Bad credentials"}\n401'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    with pytest.raises(upd.ApiError, match="authentication failed"):
        upd.gh_api("https://example.com/")


def test_gh_api_403_rate_limit(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': GET a GitHub API URL and return the parsed JSON body. Uses curl rather than urllib: this box reaches the network through a.
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '{"message": "API rate limit exceeded"}\n403'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    with pytest.raises(upd.ApiError, match="rate limit"):
        upd.gh_api("https://example.com/")


def test_gh_api_transport_failure(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': GET a GitHub API URL and return the parsed JSON body. Uses curl rather than urllib: this box reaches the network through a.
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = "\n000"
            stderr = "curl: (7) Could not connect"

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    with pytest.raises(upd.ApiError, match="request failed"):
        upd.gh_api("https://example.com/")


def test_gh_api_404_returns_body(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': GET a GitHub API URL and return the parsed JSON body. Uses curl rather than urllib: this box reaches the network through a.
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '{"message": "Not Found"}\n404'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    assert upd.gh_api("https://example.com/") == {"message": "Not Found"}


def test_find_asset_matches_regex():
    # Validates: docs/install.md, 'The updater command line': an asset is chosen by matching a pattern over the release's download URLs
    payload = {
        "assets": [
            {"browser_download_url": "https://example.com/llama-b1-bin-ubuntu-rocm-6.4-x64.tar.gz"},
            {"browser_download_url": "https://example.com/llama-b1-bin-ubuntu-vulkan-x64.tar.gz"},
        ]
    }
    assert upd.find_asset(payload, "ubuntu-vulkan-x64").endswith("vulkan-x64.tar.gz")
    assert upd.find_asset(payload, r"ubuntu-rocm-[^-]*-x64").endswith("rocm-6.4-x64.tar.gz")
    assert upd.find_asset(payload, "windows") == ""
    assert upd.find_asset([], "x64") == ""  # error payload


# --- llama-swap install ----------------------------------------------------


def install_swap(tmp_path, monkeypatch, *, tag="v262", blob=None, checksum=None,
                 api=None, existing=None):
    blob = make_swap_tarball() if blob is None else blob
    checksum = checksums_line("llama-swap-v262-linux_amd64.tar.gz", blob) if checksum is None else checksum
    if existing is not None:
        make_binary(tmp_path / "llama-swap", existing)
    monkeypatch.setattr(upd, "gh_api", api or fake_api())
    monkeypatch.setattr(upd, "download", stub_download(blob, checksum))
    return upd.update_llama_swap(tmp_path, keep_swap=2, requested_tag=tag)


def test_swap_install_creates_symlink_and_verifies(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    assert install_swap(tmp_path, monkeypatch) is True
    out = capsys.readouterr().out
    assert "Checksum verified:" in out
    assert "llama-swap -> llama-swap-v262" in out
    assert (tmp_path / "llama-swap-v262").is_file()
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"
    assert not list(tmp_path.glob("_HASH_FAIL_*"))
    assert not list(tmp_path.glob("**/*tmp*"))


def test_swap_install_hash_mismatch_quarantines(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    blob = make_swap_tarball()
    wrong = "0" * 64 + "  llama-swap-v262-linux_amd64.tar.gz\n"
    assert install_swap(tmp_path, monkeypatch, blob=blob, checksum=wrong) is False
    err = capsys.readouterr().err
    assert "sha256 mismatch" in err
    quarantined = next(tmp_path.glob("_HASH_FAIL_*"))
    assert quarantined.is_file()
    assert not (stat.S_IMODE(quarantined.stat().st_mode) & 0o111)  # no exec bit
    assert not (tmp_path / "llama-swap").exists()  # symlink untouched


def test_swap_install_hash_lookup_single_space_works(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    blob = make_swap_tarball()
    checksum = checksums_line("llama-swap-v262-linux_amd64.tar.gz", blob, prefix=" ")
    assert install_swap(tmp_path, monkeypatch, blob=blob, checksum=checksum) is True
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"


def test_swap_install_backs_up_regular_file_then_symlinks(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    assert install_swap(tmp_path, monkeypatch, existing="old") is True
    out = capsys.readouterr().out
    assert "Backing up old llama-swap to llama-swap.bak" in out
    assert "Migrated llama-swap to a symlink" in out
    assert "echo 'old'" in (tmp_path / "llama-swap.bak").read_text()
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"


def test_swap_install_symlink_branch_backs_up_target(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    make_binary(tmp_path / "llama-swap-v261", "version: v261 (fake), built at 2026-01-01\n")
    os.symlink("llama-swap-v261", tmp_path / "llama-swap")

    assert install_swap(tmp_path, monkeypatch) is True
    out = capsys.readouterr().out
    # One wording for every shape now that swap_symlink handles all three.
    assert "Backing up llama-swap -> llama-swap.bak" in out
    assert "llama-swap -> llama-swap-v262" in out
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"
    assert "version: v261" in (tmp_path / "llama-swap.bak").read_text()


def test_swap_install_backup_failure_keeps_old_binary(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    def failing_copy(*args, **kwargs):
        raise OSError("no space left on device")

    monkeypatch.setattr(upd.shutil, "copyfile", failing_copy)
    assert install_swap(tmp_path, monkeypatch, existing="old") is False
    err = capsys.readouterr().err
    assert "could not back up" in err
    # the old binary must survive, symlink must not be touched
    assert make_binary(tmp_path / "llama-swap", "old") is not None
    assert not (tmp_path / "llama-swap").is_symlink()


def test_swap_install_missing_asset(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    api = fake_api(urls=[CHECKSUMS_URL, CHECKSUMS_URL])  # no tar.gz asset
    assert install_swap(tmp_path, monkeypatch, api=api) is False
    assert "Could not find llama-swap asset" in capsys.readouterr().err


def test_swap_install_missing_checksums(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    api = fake_api(urls=[SWAP_URL, SWAP_URL])  # no checksums.txt asset
    assert install_swap(tmp_path, monkeypatch, api=api) is False
    assert "Could not find checksums.txt" in capsys.readouterr().err


def test_swap_install_unknown_tag(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater command line': a release tag the release list does not carry is an error, not a silent skip
    def api(_url):
        return {"message": "Not Found"}

    assert install_swap(tmp_path, monkeypatch, tag="v9999", api=api) is False
    assert "not found" in capsys.readouterr().err


def test_swap_install_already_up_to_date(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Check latest (or requested) release, verify sha256, atomically replace.
    make_binary(tmp_path / "llama-swap", "version: v262 (fake), built at 2026-10-03\n")
    monkeypatch.setattr(upd, "gh_api", fake_api())
    monkeypatch.setattr(
        upd, "download", lambda *a, **k: pytest.fail("must not download anything")
    )
    assert upd.update_llama_swap(tmp_path, keep_swap=2) is True
    assert "already up-to-date" in capsys.readouterr().out


def test_swap_install_local_build_ahead_of_latest(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Check latest (or requested) release, verify sha256, atomically replace.
    make_binary(tmp_path / "llama-swap", "version: v300 (local build), built at 2027-01-01\n")
    monkeypatch.setattr(upd, "gh_api", fake_api())
    monkeypatch.setattr(
        upd, "download", lambda *a, **k: pytest.fail("must not download anything")
    )
    assert upd.update_llama_swap(tmp_path, keep_swap=2) is True
    assert "already up-to-date" in capsys.readouterr().out


def test_swap_install_explicit_tag_already_present(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Check latest (or requested) release, verify sha256, atomically replace.
    make_binary(tmp_path / "llama-swap", "version: v262 (fake), built at 2026-10-03\n")
    monkeypatch.setattr(upd, "gh_api", fake_api())
    assert upd.update_llama_swap(tmp_path, keep_swap=2, requested_tag="262") is True
    assert "already installed" in capsys.readouterr().out


def test_swap_install_accepts_bare_number_tag(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    assert install_swap(tmp_path, monkeypatch, tag="262") is True
    assert (tmp_path / "llama-swap-v262").is_file()


def test_swap_install_prunes_old_unreferenced_binaries(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    for version in (260, 261):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    assert install_swap(tmp_path, monkeypatch) is True
    assert (tmp_path / "llama-swap-v262").is_file()
    assert (tmp_path / "llama-swap-v261").is_file()
    assert not (tmp_path / "llama-swap-v260").exists()


def test_installed_swap_version_parses_v_number(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': each engine keeps its history beside the live one and is reached through a stable name
    binary = make_binary(
        tmp_path / "llama-swap", "version: v262 (079c35a), built at 2026-10-03\n"
    )
    assert upd.installed_llama_swap_version(binary) == 262
    assert upd.installed_llama_swap_version(tmp_path / "missing") == 0


def test_installed_swap_version_zero_from_unparseable(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': each engine keeps its history beside the live one and is reached through a stable name
    binary = make_binary(tmp_path / "llama-swap", "no version here\n")
    assert upd.installed_llama_swap_version(binary) == 0


def test_expected_sha256_double_and_single_space(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': the updater owns downloading, sha256 verification and pruning
    blob = b"payload"
    digest = hashlib.sha256(blob).hexdigest()
    (tmp_path / "checksums.txt").write_text(
        f"deadbeef  other.tar.gz\n{digest}  llama-swap-v262-linux_amd64.tar.gz\n"
    )
    assert upd.expected_sha256(tmp_path / "checksums.txt", "llama-swap-v262-linux_amd64.tar.gz") == digest
    (tmp_path / "checksums.txt").write_text(f"{digest} llama-swap-v262-linux_amd64.tar.gz\n")
    assert upd.expected_sha256(tmp_path / "checksums.txt", "llama-swap-v262-linux_amd64.tar.gz") == digest
    assert upd.expected_sha256(tmp_path / "checksums.txt", "absent.tar.gz") == ""


def test_sha256_file(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': the updater owns downloading, sha256 verification and pruning
    blob = b"payload"
    path = tmp_path / "blob"
    path.write_bytes(blob)
    assert upd.sha256_file(path) == hashlib.sha256(blob).hexdigest()


# --- llama.cpp install -----------------------------------------------------


def install_cpp(tmp_path, monkeypatch, *, version=None, blob=None, url=CPP_URL,
                tag="b11539"):
    blob = make_cpp_tarball(tag) if blob is None else blob
    monkeypatch.setattr(
        upd, "gh_api", lambda _url: {"assets": [{"browser_download_url": url}]}
    )
    monkeypatch.setattr(upd, "download", stub_download(blob, ""))
    return upd.update_llama_cpp(
        tmp_path, version=tag if version is None else version, arch_override=""
    )


def test_cpp_install_extracts_into_version_dir(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    ok, installed = install_cpp(tmp_path, monkeypatch)
    assert (ok, installed) == (True, "b11539")
    out = capsys.readouterr().out
    assert "Installed to: " in out
    assert (tmp_path / "llama-b11539" / "llama-server").is_file()


def test_cpp_install_prunes_and_rewires_links(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': an engine is updated in the directory it is already installed in
    for build in (10000, 11000, 11500, 11501):
        (tmp_path / f"llama-b{build}").mkdir()
    os.symlink("llama-b11500", tmp_path / "server")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "update_llama_swap", lambda *_a, **_k: True)
    monkeypatch.setattr(
        upd, "gh_api", lambda _url: {"assets": [{"browser_download_url": CPP_URL}]}
    )
    monkeypatch.setattr(upd, "download", stub_download(make_cpp_tarball(), ""))

    # pruning + symlink rewiring happen in main() after a real install
    assert upd.main(["b11539"]) == 0
    assert (tmp_path / "llama-b11539").is_dir()
    assert not (tmp_path / "llama-b10000").exists()
    assert not (tmp_path / "llama-b11000").exists()
    # three newest dirs kept: b11539, b11501, b11500 (the one the link pointed at)
    assert (tmp_path / "llama-b11500").is_dir()
    assert os.readlink(tmp_path / "server") == "llama-b11539"


def test_cpp_already_present_is_noop_for_pruning(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    (d := tmp_path / "llama-b11539").mkdir()
    (d / "llama-server").write_bytes(b"bin")   # a real install, not a stub dir
    (tmp_path / "llama-b10000").mkdir()
    os.symlink("llama-b10000", tmp_path / "server")

    ok, installed = install_cpp(tmp_path, monkeypatch)
    assert (ok, installed) == (True, None)  # nothing new landed
    assert (tmp_path / "llama-b10000").is_dir()
    assert os.readlink(tmp_path / "server") == "llama-b10000"


def test_cpp_truncated_dir_is_not_certified_as_installed(tmp_path,
                                                        monkeypatch, capsys):
    """An interrupted extraction must not read as a good install.

    The gate used to be ``is_dir()``, so a half-extracted tree -- the
    90% that landed before the disk filled or the user hit Ctrl-C -- was
    reported as "already installed" forever after, and install.py wrote
    its path into config.yaml as LLAMA_DIR. Requiring the binary is what
    makes a partial install invisible to both tools.

    Validates: docs/plans/requirement-trace.md #37: installs are atomic; a truncated
    tree is never certified as a good install.
    """
    (d := tmp_path / "llama-b11539").mkdir()          # directory exists...
    (d / "libggml.so").write_bytes(b"partial")        # ...but no llama-server

    ok, installed = install_cpp(tmp_path, monkeypatch)
    assert (ok, installed) == (True, "b11539"), "must reinstall over the stump"
    assert (d / "llama-server").is_file()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".stage-")]


def test_cpp_tarball_without_llama_server_is_rejected(tmp_path, monkeypatch,
                                                      capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': True when a tag lookup came back as an API error, not a release.
    """A tarball that is not a llama.cpp release must not be installed."""
    import io as _io
    import tarfile as _tf
    buf = _io.BytesIO()
    with _tf.open(fileobj=buf, mode="w:gz") as tf:
        info = _tf.TarInfo("llama-b11539/README.md")
        info.size = 3
        tf.addfile(info, _io.BytesIO(b"hi\n"))
    ok, installed = install_cpp(tmp_path, monkeypatch, blob=buf.getvalue())
    assert (ok, installed) == (False, None)
    assert "contains no llama-server" in capsys.readouterr().err
    assert not (tmp_path / "llama-b11539").exists()


def test_cpp_unknown_version(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': llama.cpp half of the run. Returns (ok.
    monkeypatch.setattr(upd, "gh_api", lambda _url: {"message": "Not Found"})
    ok, installed = upd.update_llama_cpp(tmp_path, "b9999", "")
    assert (ok, installed) == (False, None)
    assert "Version b9999 not found" in capsys.readouterr().err


def test_cpp_no_matching_asset(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': llama.cpp half of the run. Returns (ok.
    monkeypatch.setattr(
        upd,
        "gh_api",
        lambda _url: {"assets": [{"browser_download_url": "https://example.com/readme.txt"}]},
    )
    ok, _ = upd.update_llama_cpp(tmp_path, "b11539", "")
    assert ok is False
    assert "Could not find vulkan (x64) asset for b11539" in capsys.readouterr().err


def test_cpp_rocm_non_x64_arch(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': llama.cpp half of the run. Returns (ok.
    monkeypatch.setattr(upd, "gh_api", fake_api(cpp_tag="b11539"))
    ok, _ = upd.update_llama_cpp(tmp_path, "rocm", "arm64")
    assert ok is False
    assert "rocm builds are Linux x64 only" in capsys.readouterr().err


def test_cpp_download_failure_is_an_error(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater is the installer for binaries': llama.cpp half of the run. Returns (ok.
    def failing_download(*_args, **_kwargs):
        raise subprocess.CalledProcessError(22, ["curl"])

    monkeypatch.setattr(
        upd,
        "gh_api",
        lambda _url: {"assets": [{"browser_download_url": CPP_URL}]},
    )
    monkeypatch.setattr(upd, "download", failing_download)
    ok, _ = upd.update_llama_cpp(tmp_path, "b11539", "")
    assert ok is False
    assert "download failed" in capsys.readouterr().err
    assert not (tmp_path / "llama-b11539").exists()


def test_extract_tar_gz_processes_plain_tarball(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': the updater owns downloading, sha256 verification and pruning
    archive = tmp_path / "blob.tar.gz"
    archive.write_bytes(make_cpp_tarball())
    upd.extract_tar_gz(archive, tmp_path)
    assert (tmp_path / "llama-b11539" / "llama-server").is_file()


# --- architecture detection ------------------------------------------------


def test_cpp_arch_detection(monkeypatch):
    # Validates: docs/install.md, 'The updater command line': --arch overrides the detected architecture for both downloads
    for machine, expected in [
        ("x86_64", "x64"),
        ("amd64", "x64"),
        ("aarch64", "arm64"),
        ("arm64", "arm64"),
        ("s390x", "s390x"),
        ("riscv64", "x64"),
    ]:
        monkeypatch.setattr(upd.platform, "machine", lambda m=machine: m)
        assert upd.get_llama_cpp_arch() == expected


def test_swap_arch_detection(monkeypatch):
    # Validates: docs/install.md, 'The updater command line': --arch overrides the detected architecture for both downloads
    for machine, expected in [("x86_64", "linux_amd64"), ("aarch64", "linux_arm64")]:
        monkeypatch.setattr(upd.platform, "machine", lambda m=machine: m)
        assert upd.get_llama_swap_arch() == expected


def test_cpp_arch_override_aliases():
    # Validates: docs/install.md, 'The updater command line': --arch accepts x64/amd64/linux_amd64, arm64/aarch64/linux_arm64 and s390x
    for override, expected in [
        ("linux_amd64", "x64"),
        ("x64", "x64"),
        ("linux_arm64", "arm64"),
        ("arm64", "arm64"),
        ("s390x", "s390x"),
    ]:
        assert upd.get_llama_cpp_arch(override) == expected


def test_swap_arch_override_aliases():
    # Validates: docs/install.md, 'The updater command line': --arch accepts x64/amd64/linux_amd64, arm64/aarch64/linux_arm64 and s390x
    for override, expected in [
        ("linux_amd64", "linux_amd64"),
        ("x64", "linux_amd64"),
        ("linux_arm64", "linux_arm64"),
        ("aarch64", "linux_arm64"),
    ]:
        assert upd.get_llama_swap_arch(override) == expected



# --- single-binary engines (audio.cpp, whisper.cpp) -------------------------


AUDIO_URL = "https://example.com/audio-v0.9.2-bin-ubuntu-x64-vulkan-portable.tar.gz"
AUDIO_PLAIN_URL = "https://example.com/audio-v0.9.2-bin-ubuntu-x64-vulkan.tar.gz"
WHISPER_URL = "https://example.com/whisper-bin-ubuntu-x64.tar.gz"


def make_audio_tarball() -> bytes:
    """The real audio.cpp portable tarball: a directory, not a lone binary.

    It ships libggml-vulkan.so and the CPU backend variants beside the server,
    which loads them from its own directory — so copying out the binary alone
    yields something that starts, answers /health, and then cannot find a
    backend.
    """
    return make_engine_tarball(
        "audiocpp_server",
        wrap="audio-v0.9.2-bin-ubuntu-x64-vulkan-portable",
        extra=(("libggml-vulkan.so", b"\x7fELF fake"),
               ("libggml-cpu-x64.so", b"\x7fELF fake")))


def make_engine_tarball(name: str = "audiocpp_server", subdir: str = "",
                        wrap: str = "",
                        extra: tuple[tuple[str, bytes], ...] = ()) -> bytes:
    """A gzip tarball holding one executable.

    *wrap* reproduces the real whisper.cpp layout (everything under one
    directory named after the build); *subdir* nests the binary deeper;
    *extra* adds sibling files such as the .so files a $ORIGIN binary needs.
    """
    buf = io.BytesIO()
    payload = b"#!/bin/sh\necho engine\n"
    members: list[tuple[str, bytes]] = [
        (f"{subdir}/{name}" if subdir else name, payload)]
    members.extend(extra)
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path, data in members:
            info = tarfile.TarInfo(f"{wrap}/{path}" if wrap else path)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def engine_api(releases, assets_by_tag=None):
    """gh_api stand-in taking the *release list* shape used by the new path."""
    def _api(url: str):
        if url.endswith("/releases"):
            return releases
        if "/tags/" in url:
            tag = url.rsplit("/", 1)[-1]
            return {"assets": (assets_by_tag or {}).get(tag, []),
                    "tag_name": tag}
        return []
    return _api


def test_resolve_engine_tag_skips_assetless_semver_tag(monkeypatch):
    # whisper.cpp publishes v1.9.5 (no assets) and b5454 (assets) on the
    # same day. Taking "latest" would find a release with nothing to
    # download; matching the tag shape is what makes this correct.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Newest tag matching ``tag_re`` that actually carries our asset. Reading the *release list* rather than ``/releases/latest`` is the point:.
    releases = [
        {"tag_name": "v1.9.5", "assets": []},
        {"tag_name": "b5454",
         "assets": [{"browser_download_url": WHISPER_URL}]},
        {"tag_name": "v1.9.4", "assets": []},
    ]
    monkeypatch.setattr(upd, "gh_api", engine_api(releases))
    assert upd.resolve_engine_tag(upd.WHISPER, "x64") == "b5454"


def test_resolve_engine_tag_skips_build_without_our_arch(monkeypatch):
    # b5130 exists but only carries arm64 assets, so it cannot serve here.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Newest tag matching ``tag_re`` that actually carries our asset. Reading the *release list* rather than ``/releases/latest`` is the point:.
    releases = [
        {"tag_name": "b5454",
         "assets": [{"browser_download_url": WHISPER_URL}]},
        {"tag_name": "b5130",
         "assets": [{"browser_download_url":
                     "https://example.com/whisper-bin-ubuntu-arm64.tar.gz"}]},
    ]
    monkeypatch.setattr(upd, "gh_api", engine_api(releases))
    assert upd.resolve_engine_tag(upd.WHISPER, "x64") == "b5454"


def test_resolve_engine_tag_audio_cpp_uses_semver(monkeypatch):
    # audio.cpp ships binaries on its semver tag, unlike whisper.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Newest tag matching ``tag_re`` that actually carries our asset. Reading the *release list* rather than ``/releases/latest`` is the point:.
    releases = [
        {"tag_name": "v0.9.2",
         "assets": [{"browser_download_url": AUDIO_URL}]},
    ]
    monkeypatch.setattr(upd, "gh_api", engine_api(releases))
    assert upd.resolve_engine_tag(upd.AUDIO_CPP, "x64") == "v0.9.2"


def test_resolve_engine_tag_no_match_returns_empty(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Newest tag matching ``tag_re`` that actually carries our asset. Reading the *release list* rather than ``/releases/latest`` is the point:.
    monkeypatch.setattr(upd, "gh_api", engine_api([{"tag_name": "v1.9.5",
                                                    "assets": []}]))
    assert upd.resolve_engine_tag(upd.WHISPER, "x64") == ""


def test_audio_cpp_asset_pattern_picks_the_portable_build():
    # Validates: docs/backends/audio-cpp.md, 'Build / install': Pattern for *backend*, falling back to the only one there is.
    """The portable build is the one that works on this fleet.

    Measured on the R9700/radv host: the plain ``-vulkan`` build of the same
    tag answers /health and then dies with SIGILL on the first synthesis
    request; ``-vulkan-portable`` returns a valid WAV. The pattern used to be
    anchored to exclude ``-portable``, so this tool installed the build that
    crashes while the deployment ran a hand-copied portable one.
    """
    payload = {"assets": [
        {"browser_download_url": AUDIO_URL},
        {"browser_download_url": AUDIO_PLAIN_URL},
    ]}
    pattern = upd.AUDIO_CPP.asset_pattern("").format(arch="x64")
    assert upd.find_asset(payload, pattern) == AUDIO_URL
    assert not upd.find_asset({"assets": [
        {"browser_download_url": AUDIO_PLAIN_URL}]}, pattern)


def test_audio_cpp_is_a_bundle_because_its_libs_must_travel():
    # The portable server loads libggml-vulkan.so from its own directory.
    # Validates: docs/backends/audio-cpp.md, 'Build / install': the audio-cpp asset is a bundle, because its libraries have to travel with it.
    assert upd.AUDIO_CPP.bundle, "a solo binary would not find its backend"


def test_verify_archive_warns_when_upstream_has_no_checksums(tmp_path,
                                                             capsys):
    # The chosen policy: install anyway, but say plainly that there was
    # nothing to verify against.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Check the archive against ``checksums.txt`` when one is published.
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"x")
    assert upd.verify_archive(archive, "a.tar.gz", {"assets": []},
                              "warn", tmp_path) is True
    err = capsys.readouterr().err
    assert "UNVERIFIED" in err
    assert "no checksums.txt" in err


def test_verify_archive_required_refuses_without_checksums(tmp_path):
    # Validates: docs/install.md, 'The updater command line': --require-checksums refuses an archive the release did not checksum
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"x")
    assert upd.verify_archive(archive, "a.tar.gz", {"assets": []},
                              "required", tmp_path) is False


def test_verify_archive_quarantines_mismatch_in_install_dir(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    # The quarantined file must land where it survives the temp dir that
    # holds the download, and without the execute bit.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Check the archive against ``checksums.txt`` when one is published.
    blob = b"not what checksums.txt says"
    archive = tmp_path / "dl" / "a.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(blob)
    payload = {"assets": [
        {"browser_download_url": "https://example.com/a.tar.gz"},
        {"browser_download_url": CHECKSUMS_URL},
    ]}
    monkeypatch.setattr(
        upd, "download",
        lambda url, dest, **kw: dest.write_text(checksums_line("a.tar.gz",
                                                              b"different")))
    assert upd.verify_archive(archive, "a.tar.gz", payload, "required",
                              tmp_path) is False
    assert "sha256 mismatch" in capsys.readouterr().err
    quarantined = tmp_path / "_HASH_FAIL_a.tar.gz"
    assert quarantined.read_bytes() == blob
    assert not quarantined.stat().st_mode & 0o111


def test_update_engine_installs_and_symlinks(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Fetch, verify and install one engine.
    blob = make_audio_tarball()
    releases = [{"tag_name": "v0.9.2",
                 "assets": [{"browser_download_url": AUDIO_URL}]}]
    monkeypatch.setattr(upd, "gh_api",
                        engine_api(releases, {"v0.9.2": releases[0]["assets"]}))
    monkeypatch.setattr(upd, "download",
                        lambda url, dest, **kw: dest.write_bytes(blob))
    assert upd.update_engine(upd.AUDIO_CPP, tmp_path, "x64", 2) is True
    primary = tmp_path / "audiocpp_server"
    assert primary.is_symlink()
    assert os.readlink(primary) == "audiocpp_server-v0.9.2/audiocpp_server"
    bundle = tmp_path / "audiocpp_server-v0.9.2"
    assert (bundle / "libggml-vulkan.so").is_file(), "libraries must travel"


def test_update_engine_installs_bundle_with_its_libraries(tmp_path, monkeypatch):
    # whisper.cpp's prebuilt server links against sibling .so files and
    # finds them via RUNPATH=$ORIGIN, so a copy of the binary alone is a
    # broken install. The whole directory has to land, and the symlink has
    # to point into it.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Fetch, verify and install one engine.
    blob = make_engine_tarball(
        "whisper-server", wrap="whisper-bin-ubuntu-x64",
        extra=(("libwhisper.so.1", b"\x7fELF fake"),))
    releases = [{"tag_name": "b5454",
                 "assets": [{"browser_download_url": WHISPER_URL}]}]
    monkeypatch.setattr(upd, "gh_api",
                        engine_api(releases, {"b5454": releases[0]["assets"]}))
    monkeypatch.setattr(upd, "download",
                        lambda url, dest, **kw: dest.write_bytes(blob))
    assert upd.update_engine(upd.WHISPER, tmp_path, "x64", 2) is True
    bundle = tmp_path / "whisper-server-b5454"
    assert bundle.is_dir(), "the bundle directory must be installed whole"
    assert (bundle / "libwhisper.so.1").is_file(), "libraries must travel with it"
    primary = tmp_path / "whisper-server"
    assert primary.is_symlink()
    assert os.readlink(primary) == "whisper-server-b5454/whisper-server"
    assert primary.resolve().is_file(), "symlink must not dangle"
    # The wrapping build directory must not survive as extra nesting.
    assert not (bundle / "whisper-bin-ubuntu-x64").exists()


def test_update_engine_backs_up_the_replaced_binary(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Fetch, verify and install one engine.
    make_binary(tmp_path / "audiocpp_server", "old build")
    blob = make_audio_tarball()
    releases = [{"tag_name": "v0.9.2",
                 "assets": [{"browser_download_url": AUDIO_URL}]}]
    monkeypatch.setattr(upd, "gh_api",
                        engine_api(releases, {"v0.9.2": releases[0]["assets"]}))
    monkeypatch.setattr(upd, "download",
                        lambda url, dest, **kw: dest.write_bytes(blob))
    assert upd.update_engine(upd.AUDIO_CPP, tmp_path, "x64", 2) is True
    assert "old build" in (tmp_path / "audiocpp_server.bak").read_text()


def test_update_engine_is_a_noop_when_already_present(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Fetch, verify and install one engine.
    bundle = tmp_path / "audiocpp_server-v0.9.2"
    bundle.mkdir()
    (bundle / "audiocpp_server").write_bytes(b"bin")
    called = []
    monkeypatch.setattr(upd, "gh_api",
                        lambda url: called.append(url) or [])
    assert upd.update_engine(upd.AUDIO_CPP, tmp_path, "x64", 2, "v0.9.2") is True
    assert called == [], "must not hit the API when the version is already there"


def test_cleanup_engine_versions_keeps_referenced(tmp_path):
    # Validates: docs/install.md, 'The updater command line': a versioned file that a symlink still points at is never removed
    for tag in ("v2", "v3"):
        make_binary(tmp_path / f"audiocpp_server-{tag}", tag)
    # A stale symlink still pointing at v3: pruning must not dangle it.
    (tmp_path / "audiocpp_server-v1").symlink_to("audiocpp_server-v3")
    referenced = {"audiocpp_server-v3", "audiocpp_server-v1"}
    upd.cleanup_engine_versions(upd.AUDIO_CPP, tmp_path, 1, referenced)
    assert (tmp_path / "audiocpp_server-v3").exists()
    assert (tmp_path / "audiocpp_server-v1").is_symlink()
    assert not (tmp_path / "audiocpp_server-v2").exists()


def test_engines_are_opt_in(monkeypatch, tmp_path):
    # A plain run must not touch the audio engines: they install unverified.
    # Validates: docs/install.md, 'Where an update installs': main.
    monkeypatch.setattr(upd.sys, "argv", ["update.py"])
    seen = []
    monkeypatch.setattr(upd, "update_engine",
                        lambda *a, **k: seen.append(a[0].key))
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a: (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a: True)
    assert upd.main([]) == 0
    assert seen == []


# --- target selection ------------------------------------------------------


def _sel(token):
    return upd.resolve_target(token, upd._alias_index())


def test_target_prefix_shorthands():
    # The one-letter forms the CLI advertises.
    # Validates: docs/install.md, 'Where an update installs': Every accepted spelling -> canonical group name.
    assert _sel("ll") == ("cpp", "swap")
    assert _sel("llama") == ("cpp", "swap")
    assert _sel("a") == ("audio",)
    assert _sel("w") == ("whisper",)
    assert _sel("sd") == ("sd",)
    assert _sel("stable") == ("sd",)


def test_all_needs_a_full_match():
    # `a` is an alias for audio.cpp, so `all` must not be reachable by
    # prefix or glob -- otherwise the one-letter shorthand is ambiguous.
    # Validates: docs/install.md, 'Where an update installs': Every accepted spelling -> canonical group name.
    assert _sel("all") == upd.ALL_TARGETS
    for token in ("al", "alls", "a*", "*l*l*"):
        assert _sel(token) != upd.ALL_TARGETS, token
    # and 'a' still means audio, unambiguously
    assert _sel("a") == ("audio",)


def test_target_glob_selects_a_set():
    # Validates: docs/install.md, 'Where an update installs': Every accepted spelling -> canonical group name.
    assert _sel("s*") == ("swap", "sd")
    assert _sel("*") == upd.ALL_TARGETS


def test_ambiguous_prefix_is_an_error_not_a_guess():
    """
    Validates: docs/plans/requirement-trace.md #39: an ambiguous selector exits
    non-zero instead of falling through to the default targets.
    """
    # Guessing wrong here installs the wrong engine's binary.
    with pytest.raises(upd.TargetError, match="ambiguous"):
        _sel("s")


def test_non_target_positional_is_left_for_the_version_parser():
    # Validates: docs/install.md, 'The updater command line': a positional that is not a target is left to the version parser
    for token in ("b8913", "v252", "rocm", "vulkan", "latest"):
        assert _sel(token) is None, token


def test_resolve_targets_dedupes_and_keeps_install_order():
    # Validates: docs/install.md, 'The updater command line': the targets are de-duplicated and keep the install order
    assert upd.resolve_targets(["whisper", "ll", "w"]) == (
        "cpp", "swap", "whisper")


def test_every_target_is_selectable_and_named():
    # No orphan engine: each one must be reachable, or it is unreachable.
    # Validates: docs/install.md, 'Where an update installs': Target names a selector token.
    reachable = set()
    for token in list(upd.TARGET_ALIASES) + list(upd.ALL_ALIASES):
        reachable.update(_sel(token) or ())
    assert reachable == set(upd.ALL_TARGETS)


def test_main_selects_only_named_targets(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater command line': only the named targets are selected
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a, **k: calls.append("cpp") or (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a, **k: calls.append("swap") or True)
    monkeypatch.setattr(upd, "update_engine",
                        lambda eng, *a, **k: calls.append(eng.key) or True)
    assert upd.main(["audio"]) == 0
    assert calls == ["audio"]
    assert "audio.cpp" in capsys.readouterr().out


def test_bare_run_still_defaults_to_the_verified_pair(tmp_path, monkeypatch,
                                                      capsys):
    # The other three publish no checksums.txt, so a routine bare run must
    # not start installing them.
    # Validates: docs/install.md, 'Where an update installs': main.
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a, **k: calls.append("cpp") or (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a, **k: calls.append("swap") or True)
    monkeypatch.setattr(upd, "update_engine",
                        lambda eng, *a, **k: calls.append(eng.key) or True)
    assert upd.main([]) == 0
    assert calls == ["cpp", "swap"]


def test_main_all_selects_every_target(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater command line': all selects every engine
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a, **k: calls.append("cpp") or (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a, **k: calls.append("swap") or True)
    monkeypatch.setattr(upd, "update_engine",
                        lambda eng, *a, **k: calls.append(eng.key) or True)
    assert upd.main(["all"]) == 0
    assert set(calls) == set(upd.ALL_TARGETS)


def test_main_rejects_selector_plus_legacy_only_flag(tmp_path, monkeypatch,
                                                    capsys):
    # Validates: docs/install.md, 'The updater command line': a selector together with the legacy --only flag is refused rather than guessed at
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a, **k: (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a, **k: True)
    assert upd.main(["audio", "--cpp-only"]) == 1
    assert "alternatives" in capsys.readouterr().err


# --- stable-diffusion specifics --------------------------------------------


def test_sd_asset_patterns_are_backend_specific():
    # Validates: docs/backends/audio-cpp.md, 'Build / install': Pattern for *backend*, falling back to the only one there is.
    sd = upd.STABLE_DIFFUSION
    vulkan = sd.asset_pattern("vulkan").format(arch="x86_64")
    rocm = sd.asset_pattern("rocm").format(arch="x86_64")
    assert "vulkan" in vulkan and "rocm" not in vulkan
    assert "rocm" in rocm
    # an unknown backend must not silently fall through to some other one
    assert sd.asset_pattern("metal") == ""


def test_sd_uses_x86_64_not_x64():
    # stable-diffusion.cpp spells it differently from every other project.
    # Validates: docs/install.md, 'Engines not fetched by a bare run': engine_arch.
    assert upd.engine_arch(upd.STABLE_DIFFUSION, "x64") == "x86_64"
    assert upd.engine_arch(upd.WHISPER, "x64") == "x64"


def test_sd_is_a_zip_bundle():
    # Validates: docs/backends/audio-cpp.md, 'Build / install': the sd-server asset is a zip bundle.
    sd = upd.STABLE_DIFFUSION
    assert sd.fmt == "zip"
    assert sd.bundle, "sd-server links against sibling .so via RUNPATH=$ORIGIN"


def test_zip_extraction_roundtrip(tmp_path):
    # Validates: docs/install.md, 'The updater is the installer for binaries': the updater owns downloading, sha256 verification and pruning
    import zipfile as _zip
    archive = tmp_path / "rel.zip"
    with _zip.ZipFile(archive, "w") as zf:
        zf.writestr("sd-server", "#!/bin/sh\n")
    dest = tmp_path / "out"
    upd.extract_archive(archive, dest, "zip")
    assert (dest / "sd-server").is_file()


def test_sd_tag_pattern_matches_and_rejects():
    # Validates: docs/backends/audio-cpp.md, 'Build / install': the tag pattern accepts the sd-server tags and rejects the others.
    import re as _re
    assert _re.fullmatch(upd.STABLE_DIFFUSION.tag_re, "master-968-8cd731e")
    assert not _re.fullmatch(upd.STABLE_DIFFUSION.tag_re, "v1.9.5")
    assert not _re.fullmatch(upd.STABLE_DIFFUSION.tag_re, "b5454")


def test_llama_swap_honours_arch_override(tmp_path, monkeypatch):
    # --arch used to be silently dropped here, while every other target
    # honoured it -- so `update.py all --arch arm64` installed an x86
    # llama-swap next to arm64 everything else.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Check latest (or requested) release, verify sha256, atomically replace.
    seen = {}
    monkeypatch.setattr(upd, "get_llama_swap_arch",
                        lambda override="": seen.setdefault("arch", override) or "linux_arm64")
    monkeypatch.setattr(upd, "installed_llama_swap_version", lambda p: 0)
    monkeypatch.setattr(upd, "latest_llama_swap_tag", lambda: "v999")
    monkeypatch.setattr(upd, "release_assets", lambda url: {"assets": []})
    assert upd.update_llama_swap(tmp_path, 2, "v999", "arm64") is False
    assert seen["arch"] == "arm64"


def test_single_backend_engine_says_it_ignored_the_request(capsys):
    # whisper.cpp publishes one Linux build; asking for rocm must not look
    # like it was honoured.
    # Validates: docs/backends/audio-cpp.md, 'Build / install': Pattern for *backend*, falling back to the only one there is.
    pattern = upd.WHISPER.asset_pattern("rocm")
    assert pattern, "there is exactly one build, so one is used"
    assert "Note:" in capsys.readouterr().out


def test_latest_cpp_tag_skips_the_assetless_semver_tag(monkeypatch):
    # llama.cpp publishes v0.7.0 (source only) and b11669 (36 assets) on the
    # same day. Taking the newest tag outright returned v0.7.0 and then
    # failed with "Could not find Vulkan asset" — so `update.py` was unable
    # to update llama.cpp at all.
    # Validates: docs/install.md, 'The updater is the installer for binaries': Newest llama.cpp tag that actually carries a prebuilt Linux asset.
    releases = [
        {"tag_name": "v0.7.0",
         "assets": [{"browser_download_url": "https://x/source.tar.gz"}]},
        {"tag_name": "b11669",
         "assets": [{"browser_download_url":
                     "https://x/llama-b11669-bin-ubuntu-vulkan-x64.tar.gz"}]},
        {"tag_name": "v0.6.9", "assets": []},
    ]
    monkeypatch.setattr(upd, "gh_api", lambda url: releases)
    assert upd.latest_llama_cpp_tag("ubuntu-vulkan-x64") == "b11669"


def test_latest_cpp_tag_returns_empty_when_nothing_matches(monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': Newest llama.cpp tag that actually carries a prebuilt Linux asset.
    monkeypatch.setattr(upd, "gh_api",
                        lambda url: [{"tag_name": "v0.7.0", "assets": []}])
    assert upd.latest_llama_cpp_tag("ubuntu-vulkan-x64") == ""


# --- pruning: the guard that silently did nothing for bundles --------------


def _bundle(install_dir, name):
    d = install_dir / name
    d.mkdir()
    (d / "server").write_bytes(b"bin")
    return d


def test_prune_never_removes_the_bundle_a_symlink_points_at(tmp_path):
    """A bundle symlink names ``<dir>/<binary>``, not ``<dir>``.

    Comparing that against a bare directory name never matched, so the
    "never delete what a symlink points at" guarantee was a no-op for
    whisper.cpp and stable-diffusion.cpp -- and the engine just installed
    was the one deleted, leaving a dangling link.

    Validates: docs/plans/requirement-trace.md #38: the pruner must not delete a
    bundle a symlink points at.
    """
    keep = _bundle(tmp_path, "sd-server-v1000")
    _bundle(tmp_path, "sd-server-v999")
    # A newer bundle, so the referenced one is NOT the newest: with keep=1
    # it survives only because the guard protects it, not because it is
    # inside the keep count. Without the third bundle the test stayed green
    # with the guard removed.
    _bundle(tmp_path, "sd-server-v1001")
    os.symlink("sd-server-v1000/sd-server", tmp_path / "sd-server")

    upd.cleanup_engine_versions(upd.STABLE_DIFFUSION, tmp_path, 1,
                                {"sd-server-v1000/sd-server"})
    assert keep.is_dir(), "the referenced bundle was deleted"
    assert os.readlink(tmp_path / "sd-server") == "sd-server-v1000/sd-server"


def test_prune_keeps_the_newest_by_number_not_by_spelling(tmp_path):
    # Validates: docs/install.md, 'The updater command line': the newest is kept by number, not by spelling
    """master-999 must not outrank master-1000.

    Sorted as text, '9' > '1', so the pruner kept the older build and
    deleted the newer one.
    """
    for tag in ("sd-server-v9", "sd-server-v10", "sd-server-v11"):
        _bundle(tmp_path, tag)
    os.symlink("sd-server-v11/sd-server", tmp_path / "sd-server")

    upd.cleanup_engine_versions(upd.STABLE_DIFFUSION, tmp_path, 1, {"sd-server-v11/sd-server"})
    assert (tmp_path / "sd-server-v11").is_dir(), "newest build was pruned"
    assert not (tmp_path / "sd-server-v9").exists()


def test_version_sort_key_orders_mixed_naming_schemes():
    # Validates: docs/install.md, 'Where an update installs': Order versioned names by their numbers.
    names = ["whisper-server-b999", "sd-server-master-1000-bbb",
             "audiocpp_server-v0.9.2", "audiocpp_server-v0.10.0",
             "llama-b5454"]
    ordered = sorted(names, key=upd.version_sort_key, reverse=True)
    assert ordered.index("llama-b5454") == 0
    assert ordered.index("audiocpp_server-v0.10.0") < ordered.index("audiocpp_server-v0.9.2")


# --- one backend table, and no silent fallback to Vulkan -------------------


def test_cpu_backend_does_not_install_vulkan():
    """`update.py cpu` used to fetch the Vulkan tarball and say nothing.

    DEFAULT_VERSION is "latest-vulkan", and the old folding saw a backend
    word already present and returned it untouched, so the Vulkan branch
    claimed the request. An operator on a box without Vulkan silently got
    a build they did not ask for.

    Validates: docs/plans/requirement-trace.md #40: cpu/cuda must not silently install
    Vulkan.
    """
    assert upd._backend_version(upd.DEFAULT_VERSION, "cpu") == "latest-cpu"
    assert upd._backend_version(upd.DEFAULT_VERSION, "cuda") == "latest-cuda"
    assert upd._backend_version(upd.DEFAULT_VERSION, "rocm") == "latest-rocm"
    # An explicit backend already in the token still wins.
    assert upd._backend_version("latest-cuda", "cpu") == "latest-cuda"
    assert upd._backend_version("b11658", "rocm") == "b11658-rocm"


@pytest.mark.parametrize("backend,expected", [
    ("latest-cpu", "ubuntu-x64"),
    ("latest-cuda", r"ubuntu-cuda-[^-]*-x64"),
    ("latest-vulkan", "ubuntu-vulkan-x64"),
    ("latest-rocm", r"ubuntu-rocm-[^-]*-x64"),
])
def test_each_backend_matches_its_own_asset(backend, expected):
    # Validates: docs/backends/audio-cpp.md, 'Build / install': each backend matches its own asset, never another's.
    """The CPU build carries no backend word; the others do.

    Patterns come from the b11658 release listing, so this is the one place
    a rename upstream shows up as a failing test rather than a wrong build.
    """
    pattern = upd.CPP_BACKENDS[
        next(b for b in upd.CPP_BACKENDS if b in backend)]
    pattern = pattern.format(arch="x64")
    assert pattern == expected


def test_cpu_pattern_does_not_match_a_vulkan_asset():
    # Validates: docs/backends/audio-cpp.md, 'Build / install': a cpu build does not match a vulkan asset.
    """The whole point of spelling cpu out: no overlap with vulkan/rocm."""
    vulkan = "https://x/llama-b11658-bin-ubuntu-vulkan-x64.tar.gz"
    rocm = "https://x/llama-b11658-bin-ubuntu-rocm-10.0-x64.tar.gz"
    plain = "https://x/llama-b11658-bin-ubuntu-x64.tar.gz"
    import re as _re
    cpu = upd.CPP_BACKENDS["cpu"].format(arch="x64")
    assert not _re.search(cpu, vulkan)
    assert not _re.search(cpu, rocm)
    assert _re.search(cpu, plain)


def test_already_latest_does_not_claim_an_install(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'The updater is the installer for binaries': llama.cpp half of the run. Returns (ok.
    """A no-op run must not report a tag, or main prunes and rewires anyway.

    The latest-build path used to discard install_llama_cpp's "installed"
    flag and return the tag regardless, so a run that found b11539 already
    present still counted as an install.
    """
    d = tmp_path / "llama-b11539"
    d.mkdir()
    (d / "llama-server").write_bytes(b"bin")
    monkeypatch.setattr(upd, "latest_llama_cpp_tag", lambda _p: "b11539")
    monkeypatch.setattr(upd, "release_assets", lambda _u: {"assets": [
        {"browser_download_url": CPP_URL}]})
    ok, installed = upd.update_llama_cpp(tmp_path, "latest-vulkan", "")
    assert ok is True
    assert installed is None, "claimed an install it did not perform"


# --- a bad --arch is refused, not interpolated into a regex ----------------


@pytest.mark.parametrize("bad", ["x64.*", "(a+)+$", "nonsense"])
def test_unknown_arch_is_refused(bad):
    # Validates: docs/install.md, 'The updater command line': an architecture the tool does not know is refused rather than guessed
    """It used to flow verbatim into re.search.

    Not a trust boundary -- the operator supplies it -- but it failed open:
    an unrecognised spelling became a successful lookup of a different
    tarball, which is the one outcome a wrong --arch must never have.
    """
    for fn in (upd.get_llama_cpp_arch, upd.get_llama_swap_arch):
        with pytest.raises(upd.ArchError) as exc:
            fn(bad)
        assert "expected one of" in str(exc.value)


def test_known_arch_aliases_still_resolve():
    # Validates: docs/install.md, 'The updater command line': --arch accepts x64/amd64/linux_amd64, arm64/aarch64/linux_arm64 and s390x
    assert upd.get_llama_cpp_arch("amd64") == "x64"
    assert upd.get_llama_cpp_arch("aarch64") == "arm64"
    assert upd.get_llama_swap_arch("x86_64") == "linux_amd64"
    # empty means "detect from this machine", never an error
    assert upd.get_llama_cpp_arch("") in ("x64", "arm64", "s390x")


def test_keep_engine_versions_env_var_is_reachable(monkeypatch):
    # Validates: docs/install.md, 'The updater command line': KEEP_SWAP_VERSIONS and KEEP_ENGINE_VERSIONS are the environment equivalents
    """default=2 made `if not keep_engines` permanently False.

    So KEEP_ENGINE_VERSIONS was documented in --help and never read.
    """
    p = upd.build_parser()
    assert p.parse_args([]).keep_engine_versions is None
    assert p.parse_args(["--keep-engine-versions", "4"]).keep_engine_versions == 4
    monkeypatch.setenv("KEEP_ENGINE_VERSIONS", "7")
    resolved = upd.resolve_keeps(upd.build_parser().parse_args([]))
    assert resolved == 7


def test_bad_arch_is_rejected_at_parse_time():
    # Validates: docs/install.md, 'The updater command line': a bad value is refused at parse time
    """It used to survive every early return.

    `update.py swap --arch 'x64.*'` against a current version printed
    "already up-to-date" and exited 0 — the value was never looked at,
    because the download it would have poisoned had been skipped.
    """
    with pytest.raises(SystemExit):
        upd.build_parser().parse_args(["--arch", "x64.*"])
    assert upd.build_parser().parse_args(["--arch", "aarch64"]).arch == "aarch64"
    assert upd.build_parser().parse_args([]).arch == ""   # detect


def test_arch_tables_are_consistent():
    # Validates: docs/install.md, 'The updater command line': --arch accepts x64/amd64/linux_amd64, arm64/aarch64/linux_arm64 and s390x
    """Each upstream spells the arch its own way; the union validates all."""
    assert upd.get_llama_cpp_arch("amd64") == "x64"
    assert upd.get_llama_swap_arch("amd64") == "linux_amd64"
    assert upd.get_sd_arch("amd64") == "x86_64"
    assert upd.get_sd_arch("arm64") == "aarch64"


def test_ambiguous_target_selector_is_refused(monkeypatch, capsys):
    """An unresolvable selector stops the run; it does not fall through.

    `resolve_positionals` used to append the error to `notes`, which left
    `selected` empty and fell through to the default targets — so
    `update.py s` printed "ambiguous: matches sd, swap" and then went ahead
    and installed several gigabytes, exiting 0.

    Validates: docs/plans/requirement-trace.md #39 -- an ambiguous target
    selector exits non-zero.
    """
    monkeypatch.setattr(upd, "gh_api", lambda _u: {"assets": []})
    rc = upd.main(["s"])
    assert rc == 1, "an ambiguous selector must not install anything"
    assert "ambiguous" in capsys.readouterr().err


# --- install-location resolution -------------------------------------------
#
# The rule these tests defend: running `update.py` from anywhere on the box
# updates the engine that is already installed, in the directory it is
# already in. Before this, the install directory was `Path.cwd()`, so the only
# way to update audio.cpp was to cd into its directory first -- and running it
# from the wrong place installed a second copy of an engine that was already
# there.


def _clean_env(monkeypatch):
    """No inherited configuration, so resolution sees only what a test sets."""
    for var in ("LLAMA_DIR", "LLAMA_BIN_DIR", "LLAMA_SWAP_BIN_DIR",
                "AUDIOCPP_BIN_DIR", "WHISPER_BIN_DIR", "SD_BIN_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_existing_install_is_updated_in_place_from_anywhere(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    elsewhere = tmp_path / "somewhere-else"
    elsewhere.mkdir()
    home = tmp_path / "tools"
    (home / "audiocpp").mkdir(parents=True)
    make_binary(home / "audiocpp" / "audiocpp_server", "audio")

    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(upd, "_discovery_candidates",
                        lambda: [elsewhere, home])
    dirs = upd.resolve_install_dirs()

    assert dirs["audio"] == (home / "audiocpp", "existing install")


def test_the_stable_name_wins_over_a_nested_binary(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    """A root holding the primary name *is* the home; no descending."""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tools"
    root.mkdir()
    make_binary(root / "llama-swap", "v265")          # the stable name
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [root])

    assert upd.resolve_install_dirs()["swap"][0] == root


def test_descends_into_a_per_engine_directory(tmp_path, monkeypatch):
    # The fleet shape: tools/ holds tools/audiocpp/, and tools/audiocpp/ is
    # where the engine lives. Installing into tools/ instead would put a
    # second audiocpp_server beside the first.
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tools"
    (root / "audiocpp").mkdir(parents=True)
    make_binary(root / "audiocpp" / "audiocpp_server", "audio")
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [root])

    assert upd.resolve_install_dirs()["audio"][0] == root / "audiocpp"


def test_never_descends_into_a_versioned_build_dir(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    """llama-b11658/ holds llama-server; that is not the cpp home."""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tools"
    build = root / "llama-b11658"
    build.mkdir(parents=True)
    make_binary(build / "llama-server", "server")
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [root])

    assert upd.resolve_install_dirs()["cpp"][0] == root


def test_two_installs_in_one_root_are_not_a_guess(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    """audiocpp/ and old_audiocpp/ both hold an engine: ask, do not pick."""
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "tools"
    for name in ("audiocpp", "old_audiocpp"):
        (root / name).mkdir(parents=True)
        make_binary(root / name / "audiocpp_server", "audio")
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [root])

    dirs = upd.resolve_install_dirs()

    err = capsys.readouterr().err
    assert "not guessing" in err and "old_audiocpp" in err
    assert "--dir audio=" in err, "the warning must say how to settle it"
    assert dirs["audio"][0] == root   # the root, not one of the two


def test_dir_override_beats_everything(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "load_engines_config",
                        lambda: (tmp_path, {"audio": str(tmp_path / "from-cfg")}))

    dirs = upd.resolve_install_dirs({"audio": str(tmp_path / "explicit")})

    assert dirs["audio"] == (tmp_path / "explicit", "--dir")


def test_config_beats_discovery(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    make_binary(home / "audiocpp_server", "audio")
    monkeypatch.setattr(upd, "_discovery_candidates",
                        lambda: [home])
    monkeypatch.setattr(upd, "load_engines_config",
                        lambda: (None, {"audio": tmp_path / "declared"}))

    assert upd.resolve_install_dirs()["audio"] == (
        tmp_path / "declared", "engines.ini")


def test_bin_dir_env_is_shared_with_llama_packer(tmp_path, monkeypatch):
    # The packer already resolves $AUDIOCPP_BIN_DIR; one variable, both tools.
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDIOCPP_BIN_DIR", str(tmp_path / "engines"))
    monkeypatch.setattr(upd, "load_engines_config", lambda: (None, {}))
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])

    assert upd.resolve_install_dirs()["audio"] == (
        tmp_path / "engines", "$AUDIOCPP_BIN_DIR")


def test_bin_dir_env_accepts_a_binary_path(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    binary = tmp_path / "engines" / "whisper-server"
    binary.parent.mkdir()
    make_binary(binary, "server")
    monkeypatch.setenv("WHISPER_BIN_DIR", str(binary))
    monkeypatch.setattr(upd, "load_engines_config", lambda: (None, {}))
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])

    assert upd.resolve_install_dirs()["whisper"][0] == tmp_path / "engines"


def test_root_takes_over_when_nothing_is_installed(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "load_engines_config",
                        lambda: (tmp_path / "tools", {}))
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])

    dirs = upd.resolve_install_dirs()
    assert dirs["audio"] == (tmp_path / "tools", "engines.ini root")


def test_fallback_without_config_or_env_is_the_cwd(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': ``{engine key: (directory.
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "load_engines_config", lambda: (None, {}))
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])

    assert upd.resolve_install_dirs()["cpp"] == (tmp_path, "current directory")


# --- engines.ini ------------------------------------------------------------


def test_engines_config_reads_root_and_per_engine(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': ``(root.
    cfg = tmp_path / "engines.ini"
    cfg.write_text(
        "[engines]\n"
        "# a comment\n"
        "root = /tools\n"
        "audio = /tools/audiocpp\n"
        "\n"
        "[other]\n"
        "root = ignored\n")

    root, engines = upd.load_engines_config(cfg)

    assert root == Path("/tools")
    assert engines == {"audio": Path("/tools/audiocpp")}


def test_engines_config_empty_values_mean_unset(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': ``(root.
    cfg = tmp_path / "engines.ini"
    cfg.write_text("[engines]\nroot =\naudio =\n")
    assert upd.load_engines_config(cfg) == (None, {})


def test_engines_config_absent_file_is_not_an_error(tmp_path):
    # Validates: docs/install.md, 'Where an update installs': ``(root.
    assert upd.load_engines_config(tmp_path / "nope.ini") == (None, {})


def test_engines_config_malformed_is_reported_not_fatal(tmp_path, capsys):
    # Validates: docs/install.md, 'Where an update installs': ``(root.
    cfg = tmp_path / "engines.ini"
    cfg.write_text("this is not ini [[[\n")
    root, engines = upd.load_engines_config(cfg)
    assert (root, engines) == (None, {})
    assert "ignoring unreadable" in capsys.readouterr().err


def test_engines_config_unknown_engine_is_reported(tmp_path, capsys):
    # Validates: docs/install.md, 'Where an update installs': ``(root.
    cfg = tmp_path / "engines.ini"
    cfg.write_text("[engines]\nnope = /x\n")
    _, engines = upd.load_engines_config(cfg)
    assert engines == {}
    assert "unknown engine" in capsys.readouterr().err


def test_shipped_engines_ini_names_only_known_engines(capsys):
    # Validates: docs/install.md, 'Where an update installs': ``(root.
    """The committed file is a real deployment's, not a template.

    It must parse cleanly and name nothing the resolver does not know — a
    typo here silently sends an update to the wrong place, which is exactly
    the failure the config exists to prevent.
    """
    shipped = EXTRAS / "engines.ini"
    assert shipped.is_file()
    root, engines = upd.load_engines_config(shipped)
    assert set(engines) <= set(upd.INSTALL_UNITS)
    assert capsys.readouterr().err == ""
    # Whatever it declares, every engine it covers resolves to that directory
    # rather than falling back to a guess.
    for key, declared in engines.items():
        dirs = upd.resolve_install_dirs(config=(root, engines))
        assert dirs[key][0] == declared, key


# --- CLI surface ------------------------------------------------------------


def test_dir_override_must_be_engine_equals_path():
    # Validates: docs/install.md, 'Where an update installs': ``["audio=/x".
    with pytest.raises(upd.TargetError, match=r"expects <engine>=<path>"):
        upd.parse_dir_overrides(["audio"])
    with pytest.raises(upd.TargetError, match="unknown engine"):
        upd.parse_dir_overrides(["nope=/x"])


def test_print_paths_downloads_nothing(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater command line': --print-paths prints where each engine would be updated and why, and downloads nothing
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])
    monkeypatch.setattr(upd, "update_llama_cpp",
                        lambda *a, **k: pytest.fail("must not install"))
    monkeypatch.setattr(upd, "update_llama_swap",
                        lambda *a, **k: pytest.fail("must not install"))
    monkeypatch.setattr(upd, "update_engine",
                        lambda *a, **k: pytest.fail("must not install"))

    assert upd.main(["all", "--print-paths"]) == 0

    out = capsys.readouterr().out
    assert "Install locations:" in out
    assert "audiocpp_server" in out


def test_main_passes_each_engine_its_own_directory(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': an engine is updated in the directory it is already installed in
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])
    seen = {}
    monkeypatch.setattr(
        upd, "update_llama_cpp",
        lambda d, *_a, **_k: (seen.__setitem__("cpp", d), (True, None))[1])
    monkeypatch.setattr(
        upd, "update_llama_swap",
        lambda d, *_a, **_k: (seen.__setitem__("swap", d), True)[1])
    monkeypatch.setattr(
        upd, "update_engine",
        lambda eng, d, *_a, **_k: (seen.__setitem__(eng.key, d), True)[1])

    assert upd.main(["all", "--dir", "audio=/srv/audio"]) == 0
    assert seen["audio"] == Path("/srv/audio")


def test_bad_dir_stops_before_any_download(tmp_path, monkeypatch, capsys):
    # Validates: docs/install.md, 'The updater command line': a bad --dir stops the run before anything is downloaded
    _clean_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "update_llama_cpp",
                        lambda *a, **k: pytest.fail("must not install"))
    assert upd.main(["cpp", "--dir", "audio"]) == 1
    assert "--dir expects" in capsys.readouterr().err


# --- the stable llama-server name ------------------------------------------


def test_cpp_install_creates_the_stable_server_link(tmp_path, monkeypatch):
    # tools/llama-server -> ./llama-bNNNN/llama-server: the name a config can
    # pin, so moving to a new build is a symlink change and nothing that
    # references it has to be edited.
    # Validates: docs/install.md, 'Where an update installs': main.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "_discovery_candidates", lambda: [tmp_path])
    monkeypatch.setattr(upd, "update_llama_swap", lambda *_a, **_k: True)
    monkeypatch.setattr(
        upd, "gh_api", lambda _url: {"assets": [{"browser_download_url": CPP_URL}]}
    )
    monkeypatch.setattr(upd, "download", stub_download(make_cpp_tarball(), ""))

    assert upd.main(["b11539"]) == 0

    server = tmp_path / "llama-server"
    assert server.is_symlink()
    assert os.readlink(server) == "./llama-b11539/llama-server"
    assert server.resolve().is_file(), "the stable name must not dangle"


def test_stable_server_link_follows_a_new_build(tmp_path, monkeypatch):
    # Validates: docs/install.md, 'Where an update installs': Repoint top-level symlinks that target an old llama-b* dir.
    (build := tmp_path / "llama-b11539").mkdir()
    make_binary(build / "llama-server", "server")
    (build2 := tmp_path / "llama-b11600").mkdir()
    make_binary(build2 / "llama-server", "server")
    os.symlink("./llama-b11539/llama-server", tmp_path / "llama-server")

    upd.update_llama_cpp_symlinks(tmp_path, "b11600")

    assert os.readlink(tmp_path / "llama-server") == "./llama-b11600/llama-server"


def test_a_real_llama_server_file_is_never_replaced(tmp_path, capsys):
    # Validates: docs/install.md, 'Where an update installs': Repoint top-level symlinks that target an old llama-b* dir.
    """A hand-placed binary is somebody's deliberate choice."""
    make_binary(tmp_path / "llama-server", "hand-built")
    build = tmp_path / "llama-b11600"
    build.mkdir()
    make_binary(build / "llama-server", "server")

    upd.update_llama_cpp_symlinks(tmp_path, "b11600")

    assert not (tmp_path / "llama-server").is_symlink()
    assert "is a real file" in capsys.readouterr().err
