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
    HostProfile,
)


# ── Host RAM pool (`device: cpu` budget) ──────────────────────────────────

_MEMINFO = ("MemTotal:       65536000 kB\n"   # 64000 MiB
            "MemFree:       10485760 kB\n"
            "MemAvailable:  41943040 kB\n")  # 40960 MiB
_ZONEINFO = ("  DMA:\n"
             "      high  100\n"
             "  Normal:\n"
             "      high  200\n")             # 300 kB total, below the floor


def _proc_fixture(monkeypatch, tmp_path, *, zoneinfo=_ZONEINFO,
                  meminfo=_MEMINFO, min_free="1024", cgroup_span_pid=77):
    """Point the host-detection readers at fixture files."""
    meminfo_path = tmp_path / "meminfo"
    meminfo_path.write_text(meminfo)
    monkeypatch.setattr(hardware, "_PROC_MEMINFO", meminfo_path)
    zi = tmp_path / "zoneinfo"
    zi.write_text(zoneinfo if zoneinfo is not None else "")
    monkeypatch.setattr(hardware, "_PROC_ZONEINFO", zi)
    vm = tmp_path / "min_free_kbytes"
    vm.write_text(min_free)
    monkeypatch.setattr(hardware, "_VM_MIN_FREE", vm)
    monkeypatch.setattr(hardware, "_find_llama_swap_pid", lambda: None)


def test_pressure_zone_from_zoneinfo(monkeypatch, tmp_path, caplog):
    # Validates: docs/plans/cpu-residency-budget.md #7 -- sysctl arm: per-zone high sum, min_free_kbytes floor.
    """The zone `high` sum feeds the sysctl arm, raised to the
    min_free_kbytes floor when the watermark sum is lower."""
    # 300 kB of high-watermarks < the 1024 kB floor → the floor wins.
    _proc_fixture(monkeypatch, tmp_path)
    hp = HostProfile.detect(stable=False)
    assert hp.provenance["pressure"] == "sysctl"
    assert hp.pressure_zone_mb == 1                 # 1024 kB Floored
    assert hp.sys_free_mb == 40960 - 1
    assert hp.pool_mb == 40960 - 1

    # Zone sum above the floor: the watermark sum is the arm.
    _proc_fixture(monkeypatch, tmp_path, zoneinfo="  Normal:\n    high 4000000\n",
                  min_free="1024")
    hp = HostProfile.detect(stable=False)
    assert hp.pressure_zone_mb == 4000000 // 1024


def test_pressure_zone_fallback_is_logged(monkeypatch, tmp_path, caplog):
    # Validates: docs/plans/cpu-residency-budget.md #7 -- missing/zeroed zoneinfo falls back to 25% of total, logged.
    """unreadable zoneinfo is the declared fallback trigger: 25% of total,
    and the log names the arm — a silent fallback is a bug."""
    _proc_fixture(monkeypatch, tmp_path, zoneinfo="no zones here\n")
    with caplog.at_level("INFO"):
        hp = HostProfile.detect(stable=False)
    assert hp.provenance["pressure"] == "fallback-25%"
    assert hp.pressure_zone_mb == 64000 // 4
    assert any("fallback-25" in r.message for r in caplog.records)


def test_pressure_zone_override_asserted(monkeypatch, tmp_path):
    # Validates: docs/plans/cpu-residency-budget.md #7 -- hardware.host_pressure: sysctl | 25% | <abs>.
    _proc_fixture(monkeypatch, tmp_path)
    hp = HostProfile.detect(yaml_hw={"host_pressure": "4G"})
    assert hp.pressure_zone_mb == 4096
    assert hp.provenance["pressure"] == "declared"

    hp = HostProfile.detect(yaml_hw={"host_pressure": "25%"})
    assert hp.pressure_zone_mb == 64000 // 4
    assert hp.provenance["pressure"] == "fallback-25%"


def test_host_pool_mixed_terms(monkeypatch, tmp_path):
    # Validates: docs/plans/cpu-residency-budget.md #10 -- host_pool = min(sys_free, cgroup_free).
    """Both terms bind: the pool is the smaller of sys_free and the
    service cgroup headroom."""
    _proc_fixture(monkeypatch, tmp_path)
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.high").write_text(f"{32 * 1024**3}")
    (cgroup / "memory.max").write_text(f"{40 * 1024**3}")
    (cgroup / "memory.current").write_text(f"{4 * 1024**3}")
    monkeypatch.setattr(hardware, "_find_llama_swap_pid", lambda: 77)
    monkeypatch.setattr(hardware, "_service_cgroup_path", lambda pid: cgroup)
    hp = HostProfile.detect(stable=False)
    # cgroup free: min(32G−4G, 40G−4G) = 28G < sys_free 40959 → binding term
    assert hp.cgroup_free_mb == 28 * 1024
    assert hp.pool_mb == 28 * 1024
    assert hp.provenance["cgroup"] == "detected"


def test_host_pool_overrides_arm(monkeypatch, tmp_path):
    # Validates: docs/plans/cpu-residency-budget.md #10 -- hardware.host_pool: auto | <abs> | <frac>%.
    _proc_fixture(monkeypatch, tmp_path)
    hp = HostProfile.detect(yaml_hw={"host_pool": "16G"})
    assert hp.pool_mb == 16 * 1024
    assert hp.provenance["pool"] == "declared"
    hp = HostProfile.detect(yaml_hw={"host_pool": "50%"})
    assert hp.pool_mb == 32000           # frac of the detected total
    hp = HostProfile.detect(yaml_hw={})
    assert hp.provenance["pool"] == "sys_free"


def test_cgroup_missing_service_skips_term(monkeypatch, tmp_path, caplog):
    # Validates: docs/plans/cpu-residency-budget.md #9 -- no service cgroup skips the term with a warning.
    _proc_fixture(monkeypatch, tmp_path)
    with caplog.at_level("WARNING"):
        hp = HostProfile.detect(stable=False)
    assert hp.cgroup_free_mb is None
    assert hp.pool_mb == hp.sys_free_mb
    assert hp.provenance["cgroup"] == "skipped-cgroup"
    assert any("skipped-cgroup" in r.message for r in caplog.records)


def test_host_pool_clamps_to_zero(monkeypatch, tmp_path):
    # Validates: docs/plans/cpu-residency-budget.md #10 -- an infeasible read clamps the pool to 0, never invents.
    """sys_free may go negative (requirement 8); the POOL is where the
    floor lives — a squeezed box plans no CPU models rather than a
    negative budget."""
    _proc_fixture(monkeypatch, tmp_path, meminfo="MemTotal: 65536000 kB\n"
                                                "MemAvailable: 100 kB\n")
    monkeypatch.setattr(hardware, "_find_llama_swap_pid", lambda: None)
    hp = HostProfile.detect(yaml_hw={"host_pressure": "4G"}, stable=False)
    assert hp.sys_free_mb == 100 // 1024 - 4096        # negative is fine
    assert hp.pool_mb == 0
    assert hp.provenance["pool"] == "sys_free"


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
    # Validates: SPEC.md, 'Hardware Detection': Detect the GPU vendor: ``"amd"``.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, "{}"))))
    assert hardware.detect_gpu_vendor() == "amd"


def test_vendor_falls_back_to_rocminfo(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the GPU vendor: ``"amd"``.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               rocm=Completed(0))))
    assert hardware.detect_gpu_vendor() == "amd"


def test_vendor_falls_back_to_nvidia(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the GPU vendor: ``"amd"``.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               rocm=Completed(1),
                                               nvidia=Completed(0, "8192"))))
    assert hardware.detect_gpu_vendor() == "nvidia"


def test_vendor_nvidia_without_output_is_cpu(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the GPU vendor: ``"amd"``.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               rocm=Completed(1),
                                               nvidia=Completed(0, ""))))
    assert hardware.detect_gpu_vendor() == "cpu"


def test_vendor_nothing_available_is_cpu(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the GPU vendor: ``"amd"``.
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware.detect_gpu_vendor() == "cpu"


# ── detect_gpu_env_var ────────────────────────────────────────────────────

def test_env_var_amd(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Return the vendor env var used to pin a process to a GPU device. ROCm uses ROCR_VISIBLE_DEVICES; NVIDIA uses CUDA_VISIBLE_DEVICES.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, "{}"))))
    assert hardware.detect_gpu_env_var() == "ROCR_VISIBLE_DEVICES"


def test_env_var_nvidia(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Return the vendor env var used to pin a process to a GPU device. ROCm uses ROCR_VISIBLE_DEVICES; NVIDIA uses CUDA_VISIBLE_DEVICES.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               nvidia=Completed(0, "8192"))))
    assert hardware.detect_gpu_env_var() == "CUDA_VISIBLE_DEVICES"


def test_env_var_inconclusive_defaults_to_rocm(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Return the vendor env var used to pin a process to a GPU device. ROCm uses ROCR_VISIBLE_DEVICES; NVIDIA uses CUDA_VISIBLE_DEVICES.
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
    # Validates: SPEC.md, 'Hardware Detection': Detect VRAM via AMD tools. Returns MiB or None.
    payload = '{"gpu_data":[{"mem_usage":{"total_vram":{"value":24576}}}]}'
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, payload))))
    assert hardware._detect_vram_amd() == 24576


# ── detect_vram_baseline_mb ───────────────────────────────────────────────

def test_baseline_from_amd_smi(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect baseline VRAM in MiB already consumed by the driver/compositor. This is the VRAM *used* before any model is loaded (kernel driver,.
    payload = '{"gpu_data":[{"mem_usage":{"used_vram":{"value":1234}}}]}'
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(0, payload))))
    assert hardware.detect_vram_baseline_mb() == 1234


def test_baseline_from_nvidia(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect baseline VRAM in MiB already consumed by the driver/compositor. This is the VRAM *used* before any model is loaded (kernel driver,.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               nvidia=Completed(0, "512"))))
    assert hardware.detect_vram_baseline_mb() == 512


def test_baseline_nvidia_not_supported_is_zero(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect baseline VRAM in MiB already consumed by the driver/compositor. This is the VRAM *used* before any model is loaded (kernel driver,.
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(amd=Completed(1),
                                               nvidia=Completed(0, "Not Supported"))))
    assert hardware.detect_vram_baseline_mb() == 0


def test_baseline_no_tools_is_zero(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect baseline VRAM in MiB already consumed by the driver/compositor. This is the VRAM *used* before any model is loaded (kernel driver,.
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware.detect_vram_baseline_mb() == 0


# ── _detect_system_ram_mb / _detect_pool_mb ───────────────────────────────

def test_system_ram_parsed_from_free(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Get total system RAM in MiB (for unified memory fallback).
    out = ("              total        used        free\n"
           "Mem:          131072       1000       90000\n")
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(free=Completed(0, out))))
    assert hardware._detect_system_ram_mb() == 131072


def test_pool_prefers_discrete_amd(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the memory pool available to models. Returns ``(pool_mb.
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: 24576)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: 8192)
    assert hardware._detect_pool_mb() == (24576, False)


def test_pool_uses_discrete_nvidia(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the memory pool available to models. Returns ``(pool_mb.
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: 8192)
    assert hardware._detect_pool_mb() == (8192, False)


def test_pool_unified_when_nvidia_reports_na(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the memory pool available to models. Returns ``(pool_mb.
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "_detect_system_ram_mb", lambda: 122880)
    monkeypatch.setattr(hardware.subprocess, "run",
                        stub_run(tool_dispatch(nvidia=Completed(0, "NVIDIA GB10"))))
    assert hardware._detect_pool_mb() == (122880, True)


def test_pool_falls_back_to_system_ram_without_tools(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the memory pool available to models. Returns ``(pool_mb.
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: None)
    monkeypatch.setattr(hardware, "_detect_system_ram_mb", lambda: 65536)
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    assert hardware._detect_pool_mb() == (65536, True)


def test_pool_raises_clean_systemexit_when_undetectable(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect the memory pool available to models. Returns ``(pool_mb.
    monkeypatch.setattr(hardware, "_detect_vram_amd", lambda: None)
    monkeypatch.setattr(hardware, "_detect_vram_nvidia", lambda: None)
    monkeypatch.setattr(hardware.subprocess, "run", stub_run(tool_dispatch()))
    monkeypatch.setattr(hardware, "_detect_system_ram_mb",
                        lambda: (_ for _ in ()).throw(RuntimeError("no ram")))
    with pytest.raises(SystemExit):
        hardware._detect_pool_mb()


def test_detect_vram_mb_returns_pool_total(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Detect VRAM budget in MiB (pool total, reserve folded by caller).
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (12345, True))
    assert hardware.detect_vram_mb() == 12345


# ── GpuProfile.from_args precedence ───────────────────────────────────────

def test_from_args_explicit_vram_skips_detection(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    monkeypatch.setattr(hardware, "_detect_pool_mb", boom)
    assert GpuProfile.from_args(vram="32G").vram_mb == 32 * 1024


def test_from_args_yaml_vram_used_when_no_cli(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    monkeypatch.setattr(hardware, "_detect_pool_mb", boom)
    assert GpuProfile.from_args(yaml_hw={"vram": "16G"}).vram_mb == 16 * 1024


def test_from_args_autodetects_when_unset(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (24576, False))
    p = GpuProfile.from_args()
    assert p.vram_mb == 24576
    assert p.baseline_mb == 0


def test_from_args_family_precedence():
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    assert GpuProfile.from_args(vram="1G", gpu_family="rocm7",
                                yaml_hw={"gpu_family": "cuda12"}).family == "rocm7"
    assert GpuProfile.from_args(vram="1G",
                                yaml_hw={"gpu_family": "cuda12"}).family == "cuda12"
    assert GpuProfile.from_args(vram="1G").family == "default"


def test_from_args_yaml_baseline_when_no_cli():
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    p = GpuProfile.from_args(vram="1G", yaml_hw={"baseline_mb": "4G"})
    assert p.baseline_mb == 4 * 1024


def test_from_args_cli_baseline_beats_yaml():
    # Documented contract: explicit args > YAML > auto-detect.
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    p = GpuProfile.from_args(vram="1G", baseline="2G",
                             yaml_hw={"baseline_mb": "4G"})
    assert p.baseline_mb == 2 * 1024


def test_from_args_unified_folds_default_system_reserve(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    p = GpuProfile.from_args()
    assert p.baseline_mb == max(0, _UNIFIED_SYSTEM_RESERVE_DEFAULT - _RESERVE_SYSTEM)


def test_from_args_unified_system_knob_cli_beats_yaml(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    p = GpuProfile.from_args(unified_system_mb="16G",
                             yaml_hw={"unified_system_mb": "32G"})
    assert p.baseline_mb == 16 * 1024 - _RESERVE_SYSTEM


def test_from_args_explicit_baseline_disables_unified_folding(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Construct profile with precedence: explicit args > YAML > auto-detect.
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    p = GpuProfile.from_args(baseline="2G")
    assert p.baseline_mb == 2 * 1024


# ── GpuProfile.detect ─────────────────────────────────────────────────────

def test_detect_discrete_keeps_measured_baseline(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Auto-detect local GPU hardware.
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (24576, False))
    monkeypatch.setattr(hardware, "detect_vram_baseline_mb", lambda: 777)
    p = GpuProfile.detect()
    assert (p.vram_mb, p.baseline_mb) == (24576, 777)


def test_detect_unified_folds_system_reserve(monkeypatch):
    # Validates: SPEC.md, 'Hardware Detection': Auto-detect local GPU hardware.
    monkeypatch.setattr(hardware, "_detect_pool_mb", lambda: (122880, True))
    monkeypatch.setattr(hardware, "detect_vram_baseline_mb", lambda: 0)
    p = GpuProfile.detect()
    assert p.baseline_mb == max(0, _UNIFIED_SYSTEM_RESERVE_DEFAULT - _RESERVE_SYSTEM)
