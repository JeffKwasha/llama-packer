# tests/test_utils_paths.py
"""Contract tests for filesystem/version helpers in utils."""

from __future__ import annotations

from pathlib import Path

import pytest

from llama_packer import utils
from llama_packer.consts import (
    _HDD_READ_MBPS,
    _NVME_READ_MBPS,
    _SSD_READ_MBPS,
    _UNKNOWN_READ_MBPS,
)


class Completed:
    def __init__(self, rc: int = 0, out: str = ""):
        self.returncode = rc
        self.stdout = out
        self.stderr = ""


# ── get_available_versions / find_bin_dir ─────────────────────────────────

def test_get_available_versions_lists_sorted_dirs_only(tmp_path):
    for n in (100, 99, 101):
        (tmp_path / f"llama-b{n}").mkdir()
    (tmp_path / "llama-b102").write_text("not a dir")
    (tmp_path / "other").mkdir()
    assert utils.get_available_versions(tmp_path) == [99, 100, 101]


def test_find_bin_dir_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_BIN_DIR", "/custom/bin")
    assert utils.find_bin_dir("latest", tmp_path) == "/custom/bin"


def test_find_bin_dir_latest_and_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    (tmp_path / "llama-b7").mkdir()
    (tmp_path / "llama-b12").mkdir()
    assert utils.find_bin_dir("latest", tmp_path) == "llama-b12"
    assert utils.find_bin_dir("7", tmp_path) == "llama-b7"


def test_find_bin_dir_unknown_version_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    (tmp_path / "llama-b7").mkdir()
    with pytest.raises(SystemExit) as exc:
        utils.find_bin_dir("9", tmp_path)
    assert "9" in str(exc.value) and "7" in str(exc.value)


# ── the tools/ layout ──────────────────────────────────────────────────────
#
# A deployment keeps its engines in <run dir>/tools so the directory holding
# config.yaml is not also a 400 MB binary dump. Everything that resolved a
# binary from the run directory has to know that, and every layout installed
# before it has to keep resolving.


def _build(base: Path, name: str = "llama-b11714") -> Path:
    d = base / name
    d.mkdir(parents=True)
    (d / "llama-server").write_text("x")
    return d


def test_find_bin_dir_resolves_inside_tools(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    _build(tmp_path / utils.TOOLS_SUBDIR)
    assert utils.find_bin_dir("latest", tmp_path) == "tools/llama-b11714"


def test_tools_wins_over_a_stale_flat_build(tmp_path, monkeypatch):
    """Both layouts present: tools/ is the deployment, the flat copy is old."""
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    _build(tmp_path, "llama-b11539")
    _build(tmp_path / utils.TOOLS_SUBDIR, "llama-b11714")
    assert utils.find_bin_dir("latest", tmp_path) == "tools/llama-b11714"


def test_available_versions_span_both_layouts(tmp_path):
    _build(tmp_path, "llama-b11539")
    _build(tmp_path / utils.TOOLS_SUBDIR, "llama-b11714")
    assert utils.get_available_versions(tmp_path) == [11539, 11714]


def test_no_builds_anywhere_names_both_places_it_looked(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        utils.find_bin_dir("latest", tmp_path)
    message = str(exc.value)
    assert str(tmp_path) in message
    assert utils.TOOLS_SUBDIR in message


def test_binary_search_dirs_puts_tools_first(tmp_path):
    dirs = utils.binary_search_dirs(tmp_path)
    assert dirs[0] == tmp_path / utils.TOOLS_SUBDIR
    assert dirs[-1] == tmp_path


def test_find_bin_dir_latest_with_no_builds_exits(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    with pytest.raises(SystemExit):
        utils.find_bin_dir("latest", tmp_path)


# ── get_model_size_mb ─────────────────────────────────────────────────────

def test_get_model_size_mb_rounds_down(tmp_path):
    f = tmp_path / "m.bin"
    f.write_bytes(b"x" * (2 * 1024 * 1024 + 5))
    assert utils.get_model_size_mb(str(f)) == 2


# ── _detect_drive_speed ───────────────────────────────────────────────────

def _lsblk(mount_to_line: dict[str, str]):
    def run(cmd, **kw):
        mount = cmd[-1]
        line = mount_to_line.get(mount)
        if line is None:
            return Completed(1, "")
        return Completed(0, line)
    return run


def test_drive_speed_classifies_and_returns_slowest(monkeypatch):
    monkeypatch.setattr(utils, "mount_root", lambda p: p)
    monkeypatch.setattr(utils.subprocess, "run", _lsblk({
        "nvme": "nvme0n1     0\n",
        "hdd": "sda         1\n",
        "ssd": "sdb         0\n",
    }))
    # Slowest drive bounds the estimate.
    assert utils._detect_drive_speed(
        [Path("nvme"), Path("hdd"), Path("ssd")]) == _HDD_READ_MBPS


def test_drive_speed_nvme_and_ssd(monkeypatch):
    monkeypatch.setattr(utils, "mount_root", lambda p: p)
    monkeypatch.setattr(utils.subprocess, "run",
                        _lsblk({"nvme": "nvme0n1 0\n", "ssd": "sdb 0\n"}))
    assert utils._detect_drive_speed([Path("nvme")]) == _NVME_READ_MBPS
    assert utils._detect_drive_speed([Path("ssd")]) == _SSD_READ_MBPS


def test_drive_speed_unknown_on_tool_failure(monkeypatch):
    monkeypatch.setattr(utils, "mount_root", lambda p: "/x")
    monkeypatch.setattr(utils.subprocess, "run",
                        lambda *a, **k: Completed(1, ""))
    assert utils._detect_drive_speed([Path("/x/m.gguf")]) == _UNKNOWN_READ_MBPS


def test_drive_speed_no_paths_is_unknown():
    assert utils._detect_drive_speed([]) == _UNKNOWN_READ_MBPS
