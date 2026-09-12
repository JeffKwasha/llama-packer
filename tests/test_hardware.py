# tests/test_hardware.py
"""Contract tests for hardware detection and GpuProfile construction.

These pin observable behavior: tool-output parsing (including unified-memory
sentinels), detection precedence, and the ``GpuProfile.from_args`` precedence
contract (explicit args > YAML > auto-detect).  External tools are stubbed at
``subprocess.run``; the "check=True raises on failure" semantics are emulated so
detectors are exercised the way the real subprocess would.
"""

from __future__ import annotations

import subprocess

import pytest

from llama_packer import hardware
from llama_packer.consts import _RESERVE_SYSTEM
from llama_packer.hardware import (
    _UNIFIED_SYSTEM_RESERVE_DEFAULT,
    GpuProfile,
)


class Completed:
    def __init__(self, rc: int = 0, out: str = ""):
        self.returncode = rc
        self.stdout = out
        self.stderr = ""


def stub_run(dispatch):
    """subprocess.run replacement; *dispatch* maps argv -> Completed."""
    def run(cmd, **kw):
        resp = dispatch(list(cmd))
        if kw.get("check") and resp.returncode != 0:
            raise subprocess.CalledProcessError(resp.returncode, cmd)
        return resp
    return run


def tool_dispatch(amd=None, rocm=None, nvidia=None, free=None):
    fallback = Completed(1)
    table = {"amd-smi": amd, "rocminfo": rocm, "nvidia-smi": nvidia, "free": free}
    return lambda cmd: table.get(cmd[0]) or fallback


def boom(*_a, **_k):
    raise AssertionError("auto-detection must not run when vram is explicit")


# ── detect_gpu_vendor ─────────────────────────────────────────────────────

def test_vendor_prefers_amd_smi(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, "{}"))))
    assert hardware.detect_gpu_vendor() == "amd"


def test_vendor_falls_back_to_rocminfo(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               rocm=Completed(0))))
    assert hardware.detect_gpu_vendor() == "amd"


def test_vendor_falls_back_to_nvidia(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               rocm=Completed(1),
                                               nvidia=Completed(0, "8192"))))
    assert hardware.detect_gpu_vendor() == "nvidia"


def test_vendor_nvidia_without_output_is_cpu(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               rocm=Completed(1),
                                               nvidia=Completed(0, ""))))
    assert hardware.detect_gpu_vendor() == "cpu"


def test_vendor_nothing_available_is_cpu(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware.detect_gpu_vendor() == "cpu"


# ── detect_gpu_env_var ────────────────────────────────────────────────────

def test_env_var_amd(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, "{}"))))
    assert hardware.detect_gpu_env_var() == "ROCR_VISIBLE_DEVICES"


def test_env_var_nvidia(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               nvidia=Completed(0, "8192"))))
    assert hardware.detect_gpu_env_var() == "CUDA_VISIBLE_DEVICES"


def test_env_var_inconclusive_defaults_to_rocm(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware.detect_gpu_env_var() == "ROCR_VISIBLE_DEVICES"


# ── _detect_vram_nvidia sentinels ─────────────────────────────────────────
@pytest.mark.parametrize("out,expected", [
    ("8192", 8192),
    ("N/A", None),
    ("[N/A]", None),
    ("Not Supported", None),   # unified-memory host: caller falls back to RAM
    ("", None),
])
def test_nvidia_vram_sentinels(monkeypatch, out, expected):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(nvidia=Completed(0, out))))
    assert hardware._detect_vram_nvidia() == expected


# ── _detect_vram_amd ──────────────────────────────────────────────────────

def test_amd_vram_from_amd_smi_json(monkeypatch):
    payload = '{"gpu_data":[{"mem_usage":{"total_vram":{"value":24576}}}]}'
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, payload))))
    assert hardware._detect_vram_amd() == 24576


# ── detect_vram_baseline_mb ───────────────────────────────────────────────

def test_baseline_from_amd_smi(monkeypatch):
    payload = '{"gpu_data":[{"mem_usage":{"used_vram":{"value":1234}}}]}'
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, payload))))
    assert hardware.detect_vram_baseline_mb() == 1234


def test_baseline_from_nvidia(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               nvidia=Completed(0, "512"))))
    assert hardware.detect_vram_baseline_mb() == 512


def test_baseline_nvidia_not_supported_is_zero(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               nvidia=Completed(0, "Not Supported"))))
    assert hardware.detect_vram_baseline_mb() == 0


def test_baseline_no_tools_is_zero(monkeypatch):
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware.detect_vram_baseline_mb() == 0


# ── _detect_system_ram_mb / _detect_pool_mb ───────────────────────────────

def test_system_ram_parsed_from_free(monkeypatch):
    out = ("              total        used        free\n"
           "Mem:          131072       1000       90000\n")
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(free=Completed(0, out))))
    assert hardware._detect_system_ram_mb() == 131072


def test_pool_prefers_discrete_amd(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: 24576)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: 8192)
    assert hardware._detect_pool_mb() == (24576, False)


def test_pool_uses_discrete_nvidia(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: 8192)
    assert hardware._detect_pool_mb() == (8192, False)


def test_pool_unified_when_nvidia_reports_na(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "_detect_system_ram_mb", lambda: 122880)
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(nvidia=Completed(0, "NVIDIA GB10"))))
    assert hardware._detect_pool_mb() == (122880, True)


def test_pool_falls_back_to_system_ram_without_tools(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "_detect_system_ram_mb", lambda: 65536)
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware._detect_pool_mb() == (65536, True)


def test_pool_raises_clean_systemexit_when_undetectable(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: None)
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    monkeypatch.setattr(hardware, "_detect_system_ram_mb",
                        lambda: (_ for _ in ()).throw(RuntimeError("no ram")))
    with pytest.raises(SystemExit):
        hardware._detect_pool_mb()


def test_detect_vram_mb_returns_pool_total(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (12345, True))
    assert hardware.detect_vram_mb() == 12345


# ── GpuProfile.from_args precedence ───────────────────────────────────────

def test_from_args_explicit_vram_skips_detection(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", boom)
    assert GpuProfile.from_args(vram="32G").vram_mb == 32 * 1024


def test_from_args_yaml_vram_used_when_no_cli(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", boom)
    assert GpuProfile.from_args(yaml_hw={"vram": "16G"}).vram_mb == 16 * 1024


def test_from_args_autodetects_when_unset(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (24576, False))
    p = GpuProfile.from_args()
    assert p.vram_mb == 24576
    assert p.baseline_mb == 0


def test_from_args_family_precedence():
    assert GpuProfile.from_args(vram="1G", gpu_family="rocm7",
                                yaml_hw={"gpu_family": "cuda12"}).family == "rocm7"
    assert GpuProfile.from_args(vram="1G",
                                yaml_hw={"gpu_family": "cuda12"}).family == "cuda12"
    assert GpuProfile.from_args(vram="1G").family == "default"


def test_from_args_yaml_baseline_when_no_cli():
    p = GpuProfile.from_args(vram="1G", yaml_hw={"baseline_mb": "4G"})
    assert p.baseline_mb == 4 * 1024


def test_from_args_cli_baseline_beats_yaml():
    # Documented contract: explicit args > YAML > auto-detect.
    p = GpuProfile.from_args(vram="1G", baseline="2G",
                             yaml_hw={"baseline_mb": "4G"})
    assert p.baseline_mb == 2 * 1024


def test_from_args_unified_folds_default_system_reserve(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    p = GpuProfile.from_args()
    assert p.baseline_mb == max(0, _UNIFIED_SYSTEM_RESERVE_DEFAULT - _RESERVE_SYSTEM)


def test_from_args_unified_system_knob_cli_beats_yaml(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    p = GpuProfile.from_args(unified_system_mb="16G",
                             yaml_hw={"unified_system_mb": "32G"})
    assert p.baseline_mb == 16 * 1024 - _RESERVE_SYSTEM


def test_from_args_explicit_baseline_disables_unified_folding(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    p = GpuProfile.from_args(baseline="2G")
    assert p.baseline_mb == 2 * 1024


# ── GpuProfile.detect ─────────────────────────────────────────────────────

def test_detect_discrete_keeps_measured_baseline(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (24576, False))
    monkeypatch.setattr(hardware, "detect_vram_baseline_mb", lambda: 777)
    p = GpuProfile.detect()
    assert (p.vram_mb, p.baseline_mb) == (24576, 777)


def test_detect_unified_folds_system_reserve(monkeypatch):
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    monkeypatch.setattr(hardware, "detect_vram_baseline_mb", lambda: 0)
    p = GpuProfile.detect()
    assert p.baseline_mb == max(0, _UNIFIED_SYSTEM_RESERVE_DEFAULT - _RESERVE_SYSTEM)
