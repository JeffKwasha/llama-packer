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


# --- version discovery / bookkeeping ---------------------------------------


def test_highest_installed_ignores_non_numeric_suffix(tmp_path):
    (tmp_path / "llama-b11507").mkdir()
    (tmp_path / "llama-b8913").mkdir()
    (tmp_path / "llama-b1234-extra").mkdir()  # upstream never ships this
    (tmp_path / "llama-bnope").mkdir()
    assert upd.highest_installed(tmp_path) == 11507


def test_highest_installed_empty(tmp_path):
    assert upd.highest_installed(tmp_path) == 0


def test_cpp_build_number():
    assert upd.cpp_build_number("b11507") == 11507
    assert upd.cpp_build_number("11507") == 11507
    assert upd.cpp_build_number("v262") is None


def test_cleanup_old_versions_keeps_newest_n(tmp_path):
    for build in (10000, 11000, 11500, 11501, 11502):
        (tmp_path / f"llama-b{build}").mkdir()
    upd.cleanup_old_versions(tmp_path, keep=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "llama-b11500",
        "llama-b11501",
        "llama-b11502",
    ]


def test_cleanup_old_versions_noop_when_not_owner(tmp_path, monkeypatch):
    for build in (10000, 11000, 11500, 11501):
        (tmp_path / f"llama-b{build}").mkdir()
    monkeypatch.setattr(upd, "_owned_by_current_user", lambda _p: False)
    upd.cleanup_old_versions(tmp_path, keep=1)
    assert len(list(tmp_path.iterdir())) == 4


def test_prune_swap_keeps_newest_but_never_a_referenced_file(tmp_path):
    for version in (260, 261, 262):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    os.symlink("llama-swap-v260", tmp_path / "rollback-link")

    upd.cleanup_old_llama_swap_versions(tmp_path, keep=2)

    # newest two kept; v260 survives only because a symlink points at it
    assert (tmp_path / "llama-swap-v262").is_file()
    assert (tmp_path / "llama-swap-v261").is_file()
    assert (tmp_path / "llama-swap-v260").is_file()


def test_prune_swap_removes_unreferenced(tmp_path):
    for version in (260, 261, 262):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    upd.cleanup_old_llama_swap_versions(tmp_path, keep=2)
    assert not (tmp_path / "llama-swap-v260").exists()


def test_prune_swap_ignores_bak_and_foreign_files(tmp_path):
    for version in (260, 261):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    (tmp_path / "llama-swap.bak").write_bytes(b"x")
    (tmp_path / "_HASH_FAIL_llama-swap-v262-linux_amd64.tar.gz").write_bytes(b"y")
    make_binary(tmp_path / "llama-swap", "stub")  # primary, possibly a symlink

    upd.cleanup_old_llama_swap_versions(tmp_path, keep=1)
    assert not (tmp_path / "llama-swap-v260").exists()
    assert (tmp_path / "llama-swap").exists()


def test_prune_swap_bogus_keep_defaults_to_one(tmp_path):
    for version in (260, 261, 262):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    upd.cleanup_old_llama_swap_versions(tmp_path, keep=0)
    assert (tmp_path / "llama-swap-v262").is_file()
    assert not (tmp_path / "llama-swap-v261").exists()


# --- symlinks --------------------------------------------------------------


def test_update_cpp_symlinks_repoints_only_llama_b_links(tmp_path):
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
    os.symlink("llama-b11507", tmp_path / "server")
    upd.update_llama_cpp_symlinks(tmp_path, "b12000")
    assert os.readlink(tmp_path / "server") == "llama-b11507"


def test_update_cpp_symlinks_accepts_plain_build(tmp_path):
    (tmp_path / "llama-b12000").mkdir()
    os.symlink("llama-b11507", tmp_path / "server")
    upd.update_llama_cpp_symlinks(tmp_path, "b12000")
    assert os.readlink(tmp_path / "server") == "llama-b12000"


def test_update_swap_symlinks_repoints_but_skips_primary(tmp_path):
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
    with pytest.raises(SystemExit) as excinfo:
        upd.build_parser().parse_args(["--help"])
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_unknown_option_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        upd.build_parser().parse_args(["--bogus"])
    assert excinfo.value.code == 2
    assert "--bogus" in capsys.readouterr().err


def test_keep_swap_rejects_non_positive():
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
    _, _, version, swap, notes = _positional(["v252"])
    assert (version, swap) == (upd.DEFAULT_VERSION, "v252")
    assert len(notes) == 1  # ambiguity note


def test_positional_v_number_with_swap_only_is_silent():
    _, _, _, swap, notes = _positional(["v252"], swap_only=True)
    assert swap == "v252"
    assert notes == []


def test_positional_cpp_only_ignores_v_number():
    _, _, version, swap, _ = _positional(["v252"], cpp_only=True)
    assert (version, swap) == ("v252", "")


def test_positional_last_build_wins():
    _, _, version, swap, _ = _positional(["b8913", "v252", "b8902"])
    assert (version, swap) == ("b8902", "v252")


def test_positional_backend_is_separate_from_version():
    # `sd rocm` has to read as "the sd target, the rocm backend" rather than
    # swallowing rocm as a version string.
    tokens, backend, version, _, _ = _positional(["sd", "rocm"])
    assert upd.resolve_targets(tokens) == ("sd",)
    assert backend == "rocm"
    assert version == upd.DEFAULT_VERSION


def test_main_rejects_mutually_exclusive(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert upd.main(["--swap-only", "--cpp-only"]) == 1
    assert "mutually exclusive" in capsys.readouterr().err


# --- GitHub API ------------------------------------------------------------


def test_gh_api_parses_status_line(monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '[{"tag_name": "b1"}]\n200'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    assert upd.gh_api("https://example.com/") == [{"tag_name": "b1"}]


def test_gh_api_401_is_auth_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '{"message": "Bad credentials"}\n401'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    with pytest.raises(upd.ApiError, match="authentication failed"):
        upd.gh_api("https://example.com/")


def test_gh_api_403_rate_limit(monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '{"message": "API rate limit exceeded"}\n403'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    with pytest.raises(upd.ApiError, match="rate limit"):
        upd.gh_api("https://example.com/")


def test_gh_api_transport_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = "\n000"
            stderr = "curl: (7) Could not connect"

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    with pytest.raises(upd.ApiError, match="request failed"):
        upd.gh_api("https://example.com/")


def test_gh_api_404_returns_body(monkeypatch):
    def fake_run(cmd, **kwargs):
        class Result:
            stdout = '{"message": "Not Found"}\n404'
            stderr = ""

        return Result()

    monkeypatch.setattr(upd.subprocess, "run", fake_run)
    assert upd.gh_api("https://example.com/") == {"message": "Not Found"}


def test_find_asset_matches_regex():
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
    assert install_swap(tmp_path, monkeypatch) is True
    out = capsys.readouterr().out
    assert "Checksum verified:" in out
    assert "Created symlink" in out
    assert (tmp_path / "llama-swap-v262").is_file()
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"
    assert not list(tmp_path.glob("_HASH_FAIL_*"))
    assert not list(tmp_path.glob("**/*tmp*"))


def test_swap_install_hash_mismatch_quarantines(tmp_path, monkeypatch, capsys):
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
    blob = make_swap_tarball()
    checksum = checksums_line("llama-swap-v262-linux_amd64.tar.gz", blob, prefix=" ")
    assert install_swap(tmp_path, monkeypatch, blob=blob, checksum=checksum) is True
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"


def test_swap_install_backs_up_regular_file_then_symlinks(tmp_path, monkeypatch, capsys):
    assert install_swap(tmp_path, monkeypatch, existing="old") is True
    out = capsys.readouterr().out
    assert "Backing up old binary" in out
    assert "Migrated" in out
    assert "echo 'old'" in (tmp_path / "llama-swap.bak").read_text()
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"


def test_swap_install_symlink_branch_backs_up_target(tmp_path, monkeypatch, capsys):
    make_binary(tmp_path / "llama-swap-v261", "version: v261 (fake), built at 2026-01-01\n")
    os.symlink("llama-swap-v261", tmp_path / "llama-swap")

    assert install_swap(tmp_path, monkeypatch) is True
    out = capsys.readouterr().out
    assert "Backing up old symlink target" in out
    assert "Updated symlink" in out
    assert os.readlink(tmp_path / "llama-swap") == "llama-swap-v262"
    assert "version: v261" in (tmp_path / "llama-swap.bak").read_text()


def test_swap_install_backup_failure_keeps_old_binary(tmp_path, monkeypatch, capsys):
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
    api = fake_api(urls=[CHECKSUMS_URL, CHECKSUMS_URL])  # no tar.gz asset
    assert install_swap(tmp_path, monkeypatch, api=api) is False
    assert "Could not find llama-swap asset" in capsys.readouterr().err


def test_swap_install_missing_checksums(tmp_path, monkeypatch, capsys):
    api = fake_api(urls=[SWAP_URL, SWAP_URL])  # no checksums.txt asset
    assert install_swap(tmp_path, monkeypatch, api=api) is False
    assert "Could not find checksums.txt" in capsys.readouterr().err


def test_swap_install_unknown_tag(tmp_path, monkeypatch, capsys):
    def api(_url):
        return {"message": "Not Found"}

    assert install_swap(tmp_path, monkeypatch, tag="v9999", api=api) is False
    assert "not found" in capsys.readouterr().err


def test_swap_install_already_up_to_date(tmp_path, monkeypatch, capsys):
    make_binary(tmp_path / "llama-swap", "version: v262 (fake), built at 2026-10-03\n")
    monkeypatch.setattr(upd, "gh_api", fake_api())
    monkeypatch.setattr(
        upd, "download", lambda *a, **k: pytest.fail("must not download anything")
    )
    assert upd.update_llama_swap(tmp_path, keep_swap=2) is True
    assert "already up-to-date" in capsys.readouterr().out


def test_swap_install_local_build_ahead_of_latest(tmp_path, monkeypatch, capsys):
    make_binary(tmp_path / "llama-swap", "version: v300 (local build), built at 2027-01-01\n")
    monkeypatch.setattr(upd, "gh_api", fake_api())
    monkeypatch.setattr(
        upd, "download", lambda *a, **k: pytest.fail("must not download anything")
    )
    assert upd.update_llama_swap(tmp_path, keep_swap=2) is True
    assert "already up-to-date" in capsys.readouterr().out


def test_swap_install_explicit_tag_already_present(tmp_path, monkeypatch, capsys):
    make_binary(tmp_path / "llama-swap", "version: v262 (fake), built at 2026-10-03\n")
    monkeypatch.setattr(upd, "gh_api", fake_api())
    assert upd.update_llama_swap(tmp_path, keep_swap=2, requested_tag="262") is True
    assert "already installed" in capsys.readouterr().out


def test_swap_install_accepts_bare_number_tag(tmp_path, monkeypatch):
    assert install_swap(tmp_path, monkeypatch, tag="262") is True
    assert (tmp_path / "llama-swap-v262").is_file()


def test_swap_install_prunes_old_unreferenced_binaries(tmp_path, monkeypatch):
    for version in (260, 261):
        make_binary(tmp_path / f"llama-swap-v{version}", "stub")
    assert install_swap(tmp_path, monkeypatch) is True
    assert (tmp_path / "llama-swap-v262").is_file()
    assert (tmp_path / "llama-swap-v261").is_file()
    assert not (tmp_path / "llama-swap-v260").exists()


def test_installed_swap_version_parses_v_number(tmp_path):
    binary = make_binary(
        tmp_path / "llama-swap", "version: v262 (079c35a), built at 2026-10-03\n"
    )
    assert upd.installed_llama_swap_version(binary) == 262
    assert upd.installed_llama_swap_version(tmp_path / "missing") == 0


def test_installed_swap_version_zero_from_unparseable(tmp_path):
    binary = make_binary(tmp_path / "llama-swap", "no version here\n")
    assert upd.installed_llama_swap_version(binary) == 0


def test_expected_sha256_double_and_single_space(tmp_path):
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
    ok, installed = install_cpp(tmp_path, monkeypatch)
    assert (ok, installed) == (True, "b11539")
    out = capsys.readouterr().out
    assert "Installed to: " in out
    assert (tmp_path / "llama-b11539" / "llama-server").is_file()


def test_cpp_install_prunes_and_rewires_links(tmp_path, monkeypatch):
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
    (tmp_path / "llama-b11539").mkdir()
    (tmp_path / "llama-b10000").mkdir()
    os.symlink("llama-b10000", tmp_path / "server")

    ok, installed = install_cpp(tmp_path, monkeypatch)
    assert (ok, installed) == (True, None)  # nothing new landed
    assert (tmp_path / "llama-b10000").is_dir()
    assert os.readlink(tmp_path / "server") == "llama-b10000"


def test_cpp_unknown_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(upd, "gh_api", lambda _url: {"message": "Not Found"})
    ok, installed = upd.update_llama_cpp(tmp_path, "b9999", "")
    assert (ok, installed) == (False, None)
    assert "Version b9999 not found" in capsys.readouterr().err


def test_cpp_no_matching_asset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        upd,
        "gh_api",
        lambda _url: {"assets": [{"browser_download_url": "https://example.com/readme.txt"}]},
    )
    ok, _ = upd.update_llama_cpp(tmp_path, "b11539", "")
    assert ok is False
    assert "Could not find Linux (x64) asset for b11539" in capsys.readouterr().err


def test_cpp_rocm_non_x64_arch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(upd, "gh_api", fake_api(cpp_tag="b11539"))
    ok, _ = upd.update_llama_cpp(tmp_path, "rocm", "arm64")
    assert ok is False
    assert "ROCm builds are Linux x64 only" in capsys.readouterr().err


def test_cpp_download_failure_is_an_error(tmp_path, monkeypatch, capsys):
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
    archive = tmp_path / "blob.tar.gz"
    archive.write_bytes(make_cpp_tarball())
    upd.extract_tar_gz(archive, tmp_path)
    assert (tmp_path / "llama-b11539" / "llama-server").is_file()


# --- architecture detection ------------------------------------------------


def test_cpp_arch_detection(monkeypatch):
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
    for machine, expected in [("x86_64", "linux_amd64"), ("aarch64", "linux_arm64")]:
        monkeypatch.setattr(upd.platform, "machine", lambda m=machine: m)
        assert upd.get_llama_swap_arch() == expected


def test_cpp_arch_override_aliases():
    for override, expected in [
        ("linux_amd64", "x64"),
        ("x64", "x64"),
        ("linux_arm64", "arm64"),
        ("arm64", "arm64"),
        ("s390x", "s390x"),
    ]:
        assert upd.get_llama_cpp_arch(override) == expected


def test_swap_arch_override_aliases():
    for override, expected in [
        ("linux_amd64", "linux_amd64"),
        ("x64", "linux_amd64"),
        ("linux_arm64", "linux_arm64"),
        ("aarch64", "linux_arm64"),
    ]:
        assert upd.get_llama_swap_arch(override) == expected



# --- single-binary engines (audio.cpp, whisper.cpp) -------------------------


AUDIO_URL = "https://example.com/audio-v0.9.2-bin-ubuntu-x64-vulkan.tar.gz"
WHISPER_URL = "https://example.com/whisper-bin-ubuntu-x64.tar.gz"


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
    releases = [
        {"tag_name": "v0.9.2",
         "assets": [{"browser_download_url": AUDIO_URL}]},
    ]
    monkeypatch.setattr(upd, "gh_api", engine_api(releases))
    assert upd.resolve_engine_tag(upd.AUDIO_CPP, "x64") == "v0.9.2"


def test_resolve_engine_tag_no_match_returns_empty(monkeypatch):
    monkeypatch.setattr(upd, "gh_api", engine_api([{"tag_name": "v1.9.5",
                                                    "assets": []}]))
    assert upd.resolve_engine_tag(upd.WHISPER, "x64") == ""


def test_audio_cpp_asset_pattern_skips_portable():
    # audio.cpp publishes both a plain and a "-portable" Vulkan build; the
    # pattern is anchored so the wrong one cannot win by sort order.
    payload = {"assets": [
        {"browser_download_url":
         "https://x/audio-v0.9.2-bin-ubuntu-x64-vulkan-portable.tar.gz"},
        {"browser_download_url": AUDIO_URL},
    ]}
    assert upd.find_asset(payload, upd.AUDIO_CPP.asset_pattern("").format(
        arch="x64")) == AUDIO_URL


def test_verify_archive_warns_when_upstream_has_no_checksums(tmp_path,
                                                             capsys):
    # The chosen policy: install anyway, but say plainly that there was
    # nothing to verify against.
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"x")
    assert upd.verify_archive(archive, "a.tar.gz", {"assets": []},
                              "warn", tmp_path) is True
    err = capsys.readouterr().err
    assert "UNVERIFIED" in err
    assert "no checksums.txt" in err


def test_verify_archive_required_refuses_without_checksums(tmp_path):
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(b"x")
    assert upd.verify_archive(archive, "a.tar.gz", {"assets": []},
                              "required", tmp_path) is False


def test_verify_archive_quarantines_mismatch_in_install_dir(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    # The quarantined file must land where it survives the temp dir that
    # holds the download, and without the execute bit.
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
    blob = make_engine_tarball("audiocpp_server")
    releases = [{"tag_name": "v0.9.2",
                 "assets": [{"browser_download_url": AUDIO_URL}]}]
    monkeypatch.setattr(upd, "gh_api",
                        engine_api(releases, {"v0.9.2": releases[0]["assets"]}))
    monkeypatch.setattr(upd, "download",
                        lambda url, dest, **kw: dest.write_bytes(blob))
    assert upd.update_engine(upd.AUDIO_CPP, tmp_path, "x64", 2) is True
    primary = tmp_path / "audiocpp_server"
    assert primary.is_symlink()
    assert os.readlink(primary) == "audiocpp_server-v0.9.2"
    assert (tmp_path / "audiocpp_server-v0.9.2").is_file()


def test_update_engine_installs_bundle_with_its_libraries(tmp_path, monkeypatch):
    # whisper.cpp's prebuilt server links against sibling .so files and
    # finds them via RUNPATH=$ORIGIN, so a copy of the binary alone is a
    # broken install. The whole directory has to land, and the symlink has
    # to point into it.
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
    make_binary(tmp_path / "audiocpp_server", "old build")
    blob = make_engine_tarball("audiocpp_server")
    releases = [{"tag_name": "v0.9.2",
                 "assets": [{"browser_download_url": AUDIO_URL}]}]
    monkeypatch.setattr(upd, "gh_api",
                        engine_api(releases, {"v0.9.2": releases[0]["assets"]}))
    monkeypatch.setattr(upd, "download",
                        lambda url, dest, **kw: dest.write_bytes(blob))
    assert upd.update_engine(upd.AUDIO_CPP, tmp_path, "x64", 2) is True
    assert "old build" in (tmp_path / "audiocpp_server.bak").read_text()


def test_update_engine_is_a_noop_when_already_present(tmp_path, monkeypatch):
    make_binary(tmp_path / "audiocpp_server-v0.9.2", "engine")
    called = []
    monkeypatch.setattr(upd, "gh_api",
                        lambda url: called.append(url) or [])
    assert upd.update_engine(upd.AUDIO_CPP, tmp_path, "x64", 2, "v0.9.2") is True
    assert called == [], "must not hit the API when the version is already there"


def test_cleanup_engine_versions_keeps_referenced(tmp_path):
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
    assert _sel("ll") == ("cpp", "swap")
    assert _sel("llama") == ("cpp", "swap")
    assert _sel("a") == ("audio",)
    assert _sel("w") == ("whisper",)
    assert _sel("sd") == ("sd",)
    assert _sel("stable") == ("sd",)


def test_all_needs_a_full_match():
    # `a` is an alias for audio.cpp, so `all` must not be reachable by
    # prefix or glob -- otherwise the one-letter shorthand is ambiguous.
    assert _sel("all") == upd.ALL_TARGETS
    for token in ("al", "alls", "a*", "*l*l*"):
        assert _sel(token) != upd.ALL_TARGETS, token
    # and 'a' still means audio, unambiguously
    assert _sel("a") == ("audio",)


def test_target_glob_selects_a_set():
    assert _sel("s*") == ("swap", "sd")
    assert _sel("*") == upd.ALL_TARGETS


def test_ambiguous_prefix_is_an_error_not_a_guess():
    # Guessing wrong here installs the wrong engine's binary.
    with pytest.raises(upd.TargetError, match="ambiguous"):
        _sel("s")


def test_non_target_positional_is_left_for_the_version_parser():
    for token in ("b8913", "v252", "rocm", "vulkan", "latest"):
        assert _sel(token) is None, token


def test_resolve_targets_dedupes_and_keeps_install_order():
    assert upd.resolve_targets(["whisper", "ll", "w"]) == (
        "cpp", "swap", "whisper")


def test_every_target_is_selectable_and_named():
    # No orphan engine: each one must be reachable, or it is unreachable.
    reachable = set()
    for token in list(upd.TARGET_ALIASES) + list(upd.ALL_ALIASES):
        reachable.update(_sel(token) or ())
    assert reachable == set(upd.ALL_TARGETS)


def test_main_selects_only_named_targets(tmp_path, monkeypatch, capsys):
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
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a, **k: calls.append("cpp") or (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a, **k: calls.append("swap") or True)
    monkeypatch.setattr(upd, "update_engine",
                        lambda eng, *a, **k: calls.append(eng.key) or True)
    assert upd.main([]) == 0
    assert calls == ["cpp", "swap"]


def test_main_all_selects_every_target(tmp_path, monkeypatch):
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
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(upd, "update_llama_cpp", lambda *a, **k: (True, None))
    monkeypatch.setattr(upd, "update_llama_swap", lambda *a, **k: True)
    assert upd.main(["audio", "--cpp-only"]) == 1
    assert "alternatives" in capsys.readouterr().err


# --- stable-diffusion specifics --------------------------------------------


def test_sd_asset_patterns_are_backend_specific():
    sd = upd.STABLE_DIFFUSION
    vulkan = sd.asset_pattern("vulkan").format(arch="x86_64")
    rocm = sd.asset_pattern("rocm").format(arch="x86_64")
    assert "vulkan" in vulkan and "rocm" not in vulkan
    assert "rocm" in rocm
    # an unknown backend must not silently fall through to some other one
    assert sd.asset_pattern("metal") == ""


def test_sd_uses_x86_64_not_x64():
    # stable-diffusion.cpp spells it differently from every other project.
    assert upd.engine_arch(upd.STABLE_DIFFUSION, "x64") == "x86_64"
    assert upd.engine_arch(upd.WHISPER, "x64") == "x64"


def test_sd_is_a_zip_bundle():
    sd = upd.STABLE_DIFFUSION
    assert sd.fmt == "zip"
    assert sd.bundle, "sd-server links against sibling .so via RUNPATH=$ORIGIN"


def test_zip_extraction_roundtrip(tmp_path):
    import zipfile as _zip
    archive = tmp_path / "rel.zip"
    with _zip.ZipFile(archive, "w") as zf:
        zf.writestr("sd-server", "#!/bin/sh\n")
    dest = tmp_path / "out"
    upd.extract_archive(archive, dest, "zip")
    assert (dest / "sd-server").is_file()


def test_sd_tag_pattern_matches_and_rejects():
    import re as _re
    assert _re.fullmatch(upd.STABLE_DIFFUSION.tag_re, "master-968-8cd731e")
    assert not _re.fullmatch(upd.STABLE_DIFFUSION.tag_re, "v1.9.5")
    assert not _re.fullmatch(upd.STABLE_DIFFUSION.tag_re, "b5454")


def test_llama_swap_honours_arch_override(tmp_path, monkeypatch):
    # --arch used to be silently dropped here, while every other target
    # honoured it -- so `update.py all --arch arm64` installed an x86
    # llama-swap next to arm64 everything else.
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
    pattern = upd.WHISPER.asset_pattern("rocm")
    assert pattern, "there is exactly one build, so one is used"
    assert "Note:" in capsys.readouterr().out


def test_latest_cpp_tag_skips_the_assetless_semver_tag(monkeypatch):
    # llama.cpp publishes v0.7.0 (source only) and b11669 (36 assets) on the
    # same day. Taking the newest tag outright returned v0.7.0 and then
    # failed with "Could not find Vulkan asset" — so `update.py` was unable
    # to update llama.cpp at all.
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
    monkeypatch.setattr(upd, "gh_api",
                        lambda url: [{"tag_name": "v0.7.0", "assets": []}])
    assert upd.latest_llama_cpp_tag("ubuntu-vulkan-x64") == ""
