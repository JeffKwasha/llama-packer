"""Per-process VRAM from DRM fdinfo, without root."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

EXTRAS = Path(__file__).resolve().parents[1] / "extras"
_spec = importlib.util.spec_from_file_location(
    "gpu_vram_monitor", EXTRAS / "gpu_vram_monitor.py")
mon = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mon
_spec.loader.exec_module(mon)


def _proc(tmp_path, pid, *, client="100", vram="711892 KiB",
          gtt="11384 KiB", fds=2):
    root = tmp_path / str(pid)
    fd_dir = root / "fd"
    fd_dir.mkdir(parents=True)
    info_dir = root / "fdinfo"
    info_dir.mkdir()
    node = tmp_path / "renderD128"
    node.write_text("")
    for n in range(3, 3 + fds):
        os.symlink(node, fd_dir / str(n))
        (info_dir / str(n)).write_text(
            f"pos:\t0\nflags:\t02100002\ndrm-driver:\tamdgpu\n"
            f"drm-client-id:\t{client}\npasid:\t{int(client) - 1}\n"
            f"drm-total-vram:\t{vram}\ndrm-resident-vram:\t{vram}\n"
            f"drm-total-gtt:\t{gtt}\ndrm-resident-gtt:\t{gtt}\n")
    (root / "comm").write_text("llama-server\n")
    return root


def test_fdinfo_reports_per_process_vram(tmp_path):
    # Validates: extras/gpu_vram_monitor.py, module docstring: fdinfo gives
    # per-process VRAM without root or debugfs.
    _proc(tmp_path, 4242)
    procs = mon.sample_gpu_processes_fdinfo(tmp_path)
    assert len(procs) == 1
    assert procs[0]["pid"] == 4242
    assert procs[0]["name"] == "llama-server"
    assert procs[0]["vram_mib"] == 695.2
    assert procs[0]["gtt_mib"] == 11.1


def test_fdinfo_counts_a_client_once_across_fds(tmp_path):
    # Validates: extras/gpu_vram_monitor.py, 'drm-client-id' dedupe: several
    # render-node fds on one client are one allocation.
    _proc(tmp_path, 4242, fds=3)
    assert len(mon.sample_gpu_processes_fdinfo(tmp_path)) == 1
    assert mon.sample_gpu_processes_fdinfo(tmp_path)[0]["vram_mib"] == 695.2


def test_fdinfo_sums_distinct_clients_in_one_process(tmp_path):
    # Validates: extras/gpu_vram_monitor.py, module docstring: distinct GPU
    # clients in one process are separate allocations and are summed.
    root = _proc(tmp_path, 4242, client="100", vram="1024 KiB", fds=1)
    extra = root / "fdinfo" / "9"
    extra.write_text("drm-client-id:\t101\ndrm-resident-vram:\t2048 KiB\n"
                     "drm-resident-gtt:\t0 KiB\n")
    os.symlink(tmp_path / "renderD128", root / "fd" / "9")
    procs = mon.sample_gpu_processes_fdinfo(tmp_path)
    assert procs[0]["vram_mib"] == 3.0


def test_attribution_reports_incomplete_when_gpu_is_busy_but_unreadable(
        monkeypatch, tmp_path):
    # Validates: extras/gpu_vram_monitor.py, module docstring: no processes
    # Validates: DEPLOY_FDINFO_ATTRIBUTION_REPORTS_SCOPE — tests/test_gpu_vram_monitor.py
    # with VRAM figures while the GPU is in use is an explicit warning, not
    # an empty list a caller can mistake for an idle GPU.
    monkeypatch.setattr(mon, "sample_gpu_processes_with_source",
                        lambda: ([], "none"))
    monkeypatch.setattr(mon, "read_sysfs", lambda p: 8 * 1024**3
                        if p == mon.VRAM_USED else 32 * 1024**3)
    sample = mon.take_sample()
    assert sample["attribution"]["complete"] is False
    assert "other users" in sample["attribution"]["note"]
    assert sample["attribution"]["source"] == "none"


def test_attribution_names_fdinfo_when_it_reports_vram(monkeypatch):
    # Validates: extras/gpu_vram_monitor.py, module docstring: the source is
    # named so a consumer knows the scope of the figures.
    monkeypatch.setattr(mon, "sample_gpu_processes_with_source",
                        lambda: ([{"pid": 1, "vram_mib": 695.2}], "fdinfo"))
    monkeypatch.setattr(mon, "read_sysfs", lambda p: None)
    sample = mon.take_sample()
    assert sample["attribution"]["source"] == "fdinfo"
    assert sample["attribution"]["complete"] is True


def test_fdinfo_ignores_unreadable_and_non_gpu_fds(tmp_path):
    # Validates: extras/gpu_vram_monitor.py, module docstring: a process the
    # caller cannot inspect is skipped, not reported with a bogus zero.
    root = _proc(tmp_path, 4242, vram="no such value", fds=1)
    (root / "fd" / "3").unlink()
    os.symlink(tmp_path / "renderD128", root / "fd" / "3")
    (root / "fdinfo" / "3").unlink()
    assert mon.sample_gpu_processes_fdinfo(tmp_path) == []
