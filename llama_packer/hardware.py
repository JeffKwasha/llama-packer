"""GPU hardware detection and target profiling.

Auto-detection lives here so that callers can alternatively construct a
``GpuProfile`` from CLI arguments or a ``profiles.yaml`` ``hardware:`` section,
making it possible to generate configs for remote or unavailable hardware.

Per-library calculation rules (cuda11/cuda12/rocm6/rocm7, flash-attention/FP8
support, unified vs dedicated VRAM) are deferred: no formula currently diverges
by library, so ``GpuProfile.family`` is an inert annotation.  When a rule does
diverge, reintroduce a small handler hierarchy keyed by ``family`` rather than
scattering conditionals through the budget math.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from llama_packer import utils

logger = logging.getLogger(__name__)

# ── VRAM detection (moved from utils.py) ──────────────────────────────────


def _detect_vram_amd() -> int | None:
    """Detect VRAM via AMD tools. Returns MiB or None."""
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "-m", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            data = json.loads(out.stdout)
            total = data["gpu_data"][0]["mem_usage"]["total_vram"]["value"]
            return int(total)
    except Exception:
        pass

    try:
        for p in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
            return int(p.read_text().strip()) // (1024 ** 2)
    except Exception:
        pass

    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            if "Global Memory" in line or "VRAM" in line:
                m = re.search(r"(\d+)\s*([GM])B", line)
                if m:
                    val = int(m.group(1))
                    return val * 1024 if m.group(2) == "G" else val
    except Exception:
        pass

    return None


def _detect_vram_nvidia() -> int | None:
    """Detect VRAM via nvidia-smi. Returns MiB or None.

    On unified memory systems (e.g. NVIDIA GB10/Grace), nvidia-smi reports
    'Not Supported' for memory — returns None so caller falls back to system RAM.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            val = out.stdout.strip().splitlines()[0].strip()
            if val in ("N/A", "[N/A]", "Not Supported", ""):
                return None
            return int(val)
    except Exception:
        pass
    return None


def _detect_system_ram_mb() -> int:
    """Get total system RAM in MiB (for unified memory fallback)."""
    try:
        out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5)
        for line in out.stdout.splitlines():
            if line.startswith("Mem:"):
                return int(line.split()[1])
    except Exception:
        pass
    try:
        return int(Path("/proc/meminfo").read_text().split()[1]) // 1024
    except Exception:
        pass
    raise SystemExit("error: could not determine system RAM")


def _detect_pool_mb() -> tuple[int, bool]:
    """Detect the memory pool available to models.

    Returns ``(pool_mb, is_unified)``.  Discrete GPUs (AMD/NVIDIA tools) report
    full card VRAM; unified-memory systems (NVIDIA GB10/Grace, Apple Silicon,
    Intel integrated) and boxes with no GPU tools fall back to total system RAM.
    The ``is_unified`` flag lets callers fold the fixed reserve into a single
    system reservation instead of adding it on top (see ``GpuProfile.detect``).
    """
    vram = _detect_vram_amd()
    if vram is not None:
        logger.info("vram: %d MiB (discrete GPU)", vram)
        return vram, False

    vram = _detect_vram_nvidia()
    if vram is not None:
        logger.info("vram: %d MiB (nvidia-smi)", vram)
        return vram, False

    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and "NVIDIA" in out.stdout:
            ram = _detect_system_ram_mb()
            logger.warning("unified memory detected (nvidia-smi reports N/A) — "
                           "system RAM %d MiB", ram)
            return ram, True
    except Exception:
        pass

    try:
        ram = _detect_system_ram_mb()
        logger.warning("no GPU detection tool available — system RAM %d MiB", ram)
        return ram, True
    except Exception:
        pass

    raise SystemExit(
        "error: could not detect VRAM — no GPU tools found (amd-smi, nvidia-smi, rocminfo)\n"
        "use --vram to specify (e.g. --vram 32G)"
    )


# Default reservation (MiB) for the OS/kernel/driver on unified-memory hosts.
# These machines exist to run models, so the pool is total system RAM minus a
# modest system slice rather than 50% of RAM.  Override per-host via
# ``hardware.unified_system_mb`` in profiles.yaml or ``--unified-system-mb``.
# The value is a guesstimate: in-use memory on a unified host already includes
# the slices we reserve, so it is not measured, only reserved.
_UNIFIED_SYSTEM_RESERVE_DEFAULT = 8192


def detect_vram_mb() -> int:
    """Detect VRAM budget in MiB (pool total, reserve folded by caller)."""
    return _detect_pool_mb()[0]


def detect_vram_baseline_mb() -> int:
    """Detect baseline VRAM in MiB already consumed by the driver/compositor.

    This is the VRAM *used* before any model is loaded (kernel driver,
    display compositor, other processes). It is reserved on top of the fixed
    reserves so that VRAM budgets do not assume a blank GPU. Returns 0 when no
    usable tool is available (treated as no baseline).
    """
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "-m", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            baseline = int(json.loads(out.stdout)["gpu_data"][0]["mem_usage"]["used_vram"]["value"])
            logger.info("vram baseline: %d MiB (amd-smi used_vram)", baseline)
            return baseline
    except Exception:
        pass

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            val = out.stdout.strip().splitlines()[0].strip()
            if val not in ("N/A", "[N/A]", "Not Supported", ""):
                baseline = int(val)
                logger.info("vram baseline: %d MiB (nvidia-smi memory.used)", baseline)
                return baseline
    except Exception:
        pass

    return 0


def detect_gpu_env_var() -> str:
    """Return the vendor env var used to pin a process to a GPU device.

    ROCm uses ROCR_VISIBLE_DEVICES; NVIDIA uses CUDA_VISIBLE_DEVICES.
    Defaults to ROCR_VISIBLE_DEVICES when detection is inconclusive.
    """
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "-m", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return "ROCR_VISIBLE_DEVICES"
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return "CUDA_VISIBLE_DEVICES"
    except Exception:
        pass
    return "ROCR_VISIBLE_DEVICES"


# ── GpuProfile ────────────────────────────────────────────────────────────
#
# Per-library memory rules (cuda11/cuda12/rocm6/rocm7, flash-attention/FP8
# support, unified vs dedicated VRAM, server) are not implemented yet: today no
# formula actually diverges by library.  When one does, reintroduce a small
# handler hierarchy keyed by `family` (see docs/plans/vllm-gb10.md) instead of
# scattering conditionals through the budget math.


@dataclass
class GpuProfile:
    """Target GPU hardware description.

    ``vram_mb`` feeds the VRAM budget equation; ``family`` is an inert
    annotation reserved for future library-specific rules.  Profiles can be
    constructed from auto-detection, CLI args, or a YAML section, so configs
    can be generated for hardware that is not present locally.
    """

    vram_mb: int
    family: str = "default"
    baseline_mb: int = 0

    @classmethod
    def detect(cls) -> GpuProfile:
        """Auto-detect local GPU hardware."""
        pool_mb, is_unified = _detect_pool_mb()
        baseline_mb = detect_vram_baseline_mb()
        if is_unified:
            reserve = _UNIFIED_SYSTEM_RESERVE_DEFAULT
            logger.info("unified memory: reserving %d MiB for the system", reserve)
            baseline_mb = max(baseline_mb, reserve - utils._RESERVE_SYSTEM)
        return cls(vram_mb=pool_mb, family="default", baseline_mb=baseline_mb)

    @classmethod
    def from_args(
        cls,
        *,
        vram: str | None = None,
        gpu_family: str | None = None,
        yaml_hw: dict | None = None,
        baseline: str | None = None,
        unified_system_mb: str | None = None,
    ) -> GpuProfile:
        """Construct profile with precedence: explicit args > YAML > auto-detect."""
        yaml_hw = yaml_hw or {}

        # System reservation on unified-memory hosts: CLI > YAML > default.
        # Folded into the reserve only when the pool is auto-detected as
        # unified (explicit --vram/hardware.vram owns the number instead).
        system_mb = _UNIFIED_SYSTEM_RESERVE_DEFAULT
        if yaml_hw.get("unified_system_mb"):
            system_mb = utils.parse_mem_mb(str(yaml_hw["unified_system_mb"]))
            logger.info("unified system reserve: %d MiB (from profiles.yaml hardware.unified_system_mb)",
                        system_mb)
        if unified_system_mb is not None:
            system_mb = utils.parse_mem_mb(str(unified_system_mb))
            logger.info("unified system reserve: %d MiB (from --unified-system-mb)", system_mb)

        is_unified = False
        if vram is not None:
            vram_mb = utils.parse_mem_mb(vram)
            logger.info("vram: %d MiB (from --vram)", vram_mb)
        elif yaml_hw.get("vram"):
            vram_mb = utils.parse_mem_mb(str(yaml_hw["vram"]))
            logger.info("vram: %d MiB (from profiles.yaml hardware.vram)", vram_mb)
        else:
            vram_mb, is_unified = _detect_pool_mb()

        family = gpu_family or yaml_hw.get("gpu_family", "default")

        # Baseline reserve: explicit (profiles.yaml hardware.baseline_mb or
        # --baseline) > 0. Auto-detection of the live ``used_vram`` is
        # intentionally NOT used by default: it includes the packer's own
        # resident model servers (llama-swap keeps models loaded), which would
        # make the budget assume a blank GPU and collapse every context to the
        # minimum. The fixed _RESERVE_VIDEO (1024 MiB) already covers driver/
        # compositor overhead. Set hardware.baseline_mb or --baseline to opt in
        # only when other non-model processes occupy VRAM.
        baseline_mb = 0
        explicit_baseline = False
        if yaml_hw.get("baseline_mb"):
            baseline_mb = utils.parse_mem_mb(str(yaml_hw["baseline_mb"]))
            explicit_baseline = True
            logger.info("vram baseline: %d MiB (from profiles.yaml hardware.baseline_mb)", baseline_mb)
        elif baseline is not None:
            baseline_mb = utils.parse_mem_mb(str(baseline))
            explicit_baseline = True
            logger.info("vram baseline: %d MiB (from --baseline)", baseline_mb)

        # Fold the fixed reserve into the unified system reservation so the
        # knob reads as the total system cost: reserve = _RESERVE_SYSTEM +
        # max(_RESERVE_VIDEO, baseline).  Setting baseline_mb = knob -
        # _RESERVE_SYSTEM makes the reserve exactly the knob.  Explicit
        # --baseline/hardware.baseline_mb (live spike detection) wins instead.
        if is_unified and not explicit_baseline:
            baseline_mb = max(baseline_mb, system_mb - utils._RESERVE_SYSTEM)
            logger.info("unified memory: reserving %d MiB for the system (knob %d MiB)",
                        system_mb, system_mb)

        return cls(vram_mb=vram_mb, family=family, baseline_mb=baseline_mb)