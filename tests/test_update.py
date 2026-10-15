"""Tests for extras/update.py — the llama.cpp + llama-swap installer.

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
import tarfile
from pathlib import Path

import pytest

EXTRAS = Path(__file__).resolve().parents[1] / "extras"

_spec = importlib.util.spec_from_file_location("update", EXTRAS / "update.py")
upd = importlib.util.module_from_spec(_spec)
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


def test_positional_v_number_becomes_swap_version():
    version, swap, notes = upd.resolve_positionals(
        ["v252"], "", cpp_only=False, swap_only=False
    )
    assert (version, swap) == (upd.DEFAULT_VERSION, "v252")
    assert len(notes) == 1  # ambiguity note


def test_positional_v_number_with_swap_only_is_silent():
    _, swap, notes = upd.resolve_positionals(
        ["v252"], "", cpp_only=False, swap_only=True
    )
    assert swap == "v252"
    assert notes == []


def test_positional_cpp_only_ignores_v_number():
    version, swap, _ = upd.resolve_positionals(
        ["v252"], "", cpp_only=True, swap_only=False
    )
    assert (version, swap) == ("v252", "")


def test_positional_last_build_wins():
    version, swap, _ = upd.resolve_positionals(
        ["b8913", "v252", "b8902"], "", cpp_only=False, swap_only=False
    )
    assert (version, swap) == ("b8902", "v252")


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

