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
from llama_packer.consts import _RESERVE_SYSTEM

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


def detect_gpu_vendor() -> str:
    """Detect the GPU vendor: ``"amd"``, ``"nvidia"``, or ``"cpu"``.

    Same probes as :func:`detect_gpu_env_var` (amd-smi / rocminfo, then
    nvidia-smi) but returning a vendor label — used to pick container
    default images and device pass-through flags.
    """
    try:
        out = subprocess.run(
            ["amd-smi", "metric", "-m", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return "amd"
    except Exception:
        pass
    try:
        subprocess.run(["rocminfo"], capture_output=True, timeout=10,
                       check=True)
        return "amd"
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return "nvidia"
    except Exception:
        pass
    return "cpu"


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
            baseline_mb = max(baseline_mb, reserve - _RESERVE_SYSTEM)
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
        if baseline is not None:
            baseline_mb = utils.parse_mem_mb(str(baseline))
            explicit_baseline = True
            logger.info("vram baseline: %d MiB (from --baseline)", baseline_mb)
        elif yaml_hw.get("baseline_mb"):
            baseline_mb = utils.parse_mem_mb(str(yaml_hw["baseline_mb"]))
            explicit_baseline = True
            logger.info("vram baseline: %d MiB (from profiles.yaml hardware.baseline_mb)", baseline_mb)

        # Fold the fixed reserve into the unified system reservation so the
        # knob reads as the total system cost: reserve = _RESERVE_SYSTEM +
        # max(_RESERVE_VIDEO, baseline).  Setting baseline_mb = knob -
        # _RESERVE_SYSTEM makes the reserve exactly the knob.  Explicit
        # --baseline/hardware.baseline_mb (live spike detection) wins instead.
        if is_unified and not explicit_baseline:
            baseline_mb = max(baseline_mb, system_mb - _RESERVE_SYSTEM)
            logger.info("unified memory: reserving %d MiB for the system (knob %d MiB)",
                        system_mb, system_mb)

        return cls(vram_mb=vram_mb, family=family, baseline_mb=baseline_mb)


# ── Host RAM detection (the `device: cpu` budget) ─────────────────────────
#
# SPEC.md §Host pool.  Every
# term logs its path (declared / sysctl / fallback-25% / skipped-cgroup) —
# a silent fallback is a bug.  Declared overrides in `profiles.yaml
# hardware:` always win over detection; detection is discovery with
# fallback, never assumption.

_PRESSURE_FALLBACK_FRACTION = 0.25

_PROC_MEMINFO = Path("/proc/meminfo")
_PROC_ZONEINFO = Path("/proc/zoneinfo")
_VM_MIN_FREE = Path("/proc/sys/vm/min_free_kbytes")
_CGROUP_ROOT = Path("/sys/fs/cgroup")


def _parse_meminfo_kb(text: str) -> dict[str, int]:
    """`MemAvailable:  4786243 kB` → ``{"MemAvailable": 4786243}``."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+) kB", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def _zone_high_sum_kb(text: str) -> int | None:
    """Sum of every zone's `high` watermark, kB.

    Returns None when the text holds no plausibly-synced zone table —
    absent, zeroed, or unparseable zoneinfo is the declared fallback
    trigger (requirement 7), not a 0 that reads like success.
    """
    total = 0
    for line in text.splitlines():
        m = re.match(r"\s+high\s+(\d+)", line)
        if m:
            total += int(m.group(1))
    return total if total > 0 else None


def _cgroup_free_kb(caps: dict[str, str], current_kb: int) -> int | None:
    """`min(present caps − current)` over memory.high/memory.max.

    cgroup v2 cap files count bytes; they arrive converted to kB.
    `max`/absent caps skip their term; no readable cap at all → None
    (the cgroup does not bind).
    """
    frees = []
    for key in ("high", "max"):
        raw = caps.get(key)
        if raw is None or raw == "max":
            continue
        try:
            cap_kb = int(raw) // 1024
        except ValueError:
            continue
        if cap_kb == 0:
            continue
        frees.append(max(0, cap_kb - current_kb))
    return min(frees) if frees else None


def _service_cgroup_path(pid: int) -> Path | None:
    """The cgroup dir of *pid* (v2 `0::<path>`, v1 memory controller).

    Returns None when no cgroup entry can be resolved — skipped-cgroup
    is the caller's logged outcome.
    """
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, name = fields
        if controllers == "":           # cgroup v2 root entry
            return _CGROUP_ROOT / name.lstrip("/")
        if "memory" in controllers.split(","):
            return _CGROUP_ROOT / name.lstrip("/")
    return None


@dataclass
class HostProfile:
    """The host-RAM budget pool for `device: cpu` models.

    ``pool_mb`` is the chargeable surface: ``min(sys_free, cgroup_free)``
    clamped at 0 by the pool floor clause.  ``cgroup_free_mb`` is None
    when the service cgroup could not be located or binds nothing
    (requirement 9 — skipped-cgroup); ``provenance`` names the arm each
    term chose from.
    """

    pool_mb: int
    total_mb: int
    pressure_zone_mb: int
    sys_free_mb: int
    cgroup_free_mb: int | None = None
    provenance: dict[str, str] | None = None

    @classmethod
    def detect(cls, yaml_hw: dict | None = None,
               service_pid: int | None = None,
               *, stable: bool = False) -> HostProfile:
        """Detect the host pool, honored precedence: declared > detected.

        ``stable`` forces the deterministic fixture arms only: no pid scan
        and no live sysctl, so unit tests (and dry runs without a service)
        exercise the declared/fallback paths instead of box state.
        """
        hw = yaml_hw or {}
        prov: dict[str, str] = {}

        total_mb: int | None
        if hw.get("system_ram"):
            total_mb = utils.parse_mem_mb(str(hw["system_ram"]), 0)
            prov["total"] = "declared"
            logger.info("host pool: total %d MiB (declared hardware.system_ram)",
                        total_mb)
        else:
            try:
                meminfo = _parse_meminfo_kb(_PROC_MEMINFO.read_text())
                total_kb = meminfo.get("MemTotal")
                total_mb = total_kb // 1024 if total_kb is not None else None
            except OSError:
                total_mb = None
            if total_mb is None:
                prov["total"] = "unreadable"
                logger.warning("host pool: /proc/meminfo unreadable — "
                               "totals unknown; declare hardware.system_ram")
                total_mb = 0

        # Pressure zone: sysctl arm first (requirement 7).
        zone_kb: int | None = None
        min_free_kb = 0
        try:
            min_free_kb = int(_VM_MIN_FREE.read_text().strip())
        except (OSError, ValueError):
            min_free_kb = 0
        if not stable:
            try:
                zone_kb = _zone_high_sum_kb(_PROC_ZONEINFO.read_text())
            except OSError:
                zone_kb = None
        if zone_kb is not None:
            zone_kb = max(zone_kb, min_free_kb)
            pressure_mb = max(0, zone_kb // 1024)
            prov["pressure"] = "sysctl"
            logger.info("host pool: pressure zone %d MiB (sysctl: zone "
                        "high-watermarks, floor min_free_kbytes=%d kB)",
                        pressure_mb, min_free_kb)
        else:
            frac = _PRESSURE_FALLBACK_FRACTION
            pressure_mb = int((total_mb or 0) * frac)
            prov["pressure"] = "fallback-25%"
            logger.info("host pool: pressure zone %d MiB "
                        "(fallback-25pct of total: zoneinfo unreadable)",
                        pressure_mb)

        # Declared pressure arm wins over both.
        hp = hw.get("host_pressure")
        if hp is not None:
            s = str(hp).strip().lower()
            if s == "sysctl" and zone_kb is not None and not stable:
                pressure_mb = max(0, zone_kb // 1024)
                prov["pressure"] = "sysctl"
            elif s == "25%":
                pressure_mb = int((total_mb or 0)
                                  * _PRESSURE_FALLBACK_FRACTION)
                prov["pressure"] = "fallback-25%"
            else:
                try:
                    pressure_mb = int(utils.parse_mem_mb(str(hp), total_mb or 0))
                    prov["pressure"] = "declared"
                except (TypeError, ValueError, AssertionError):
                    logger.warning("host pool: hardware.host_pressure %r "
                                   "unreadable; keeping the detected arm",
                                   hp)
            logger.info("host pool: pressure zone %d MiB (arm %s)",
                        pressure_mb, prov["pressure"])

        mem_available_mb: int | None
        if hw.get("system_ram"):
            # A declared total replaces the runtime read wholesale — the
            # operator owns the number, so no MemAvailable probe races it
            # with unknown semantics.
            sys_free_mb = max(0, (total_mb or 0) - pressure_mb)
            prov["sys_free"] = "declared"
        else:
            try:
                info = _parse_meminfo_kb(_PROC_MEMINFO.read_text())
                avail = info.get("MemAvailable")
                mem_available_mb = None if avail is None else avail // 1024
            except OSError:
                mem_available_mb = None
            if mem_available_mb is None:
                prov["sys_free"] = "unreadable"
                sys_free_mb = 0
            else:
                sys_free_mb = mem_available_mb - pressure_mb
                prov["sys_free"] = "detected"

        # Service cgroup cap (requirement 9).
        cgroup_free_mb: int | None = None
        cgroup_dir = hw.get("cgroup_dir")
        if cgroup_dir:
            path = Path(str(cgroup_dir))
            prov["cgroup"] = "declared"
        elif not stable and service_pid is None:
            service_pid = _find_llama_swap_pid()
            path = _service_cgroup_path(service_pid) if service_pid else None
            prov["cgroup"] = ("detected" if path is not None
                              else "skipped-cgroup")
        else:
            path = None
            prov["cgroup"] = "skipped-cgroup"
        if path is not None:
            try:
                caps = {}
                for key in ("high", "max", "current"):
                    try:
                        caps[key] = (path / f"memory.{key}").read_text().strip()
                    except OSError:
                        pass
                if "current" in caps:
                    cgroup_free_kb = _cgroup_free_kb(
                        caps, int(caps["current"]) // 1024)
                    cgroup_free_mb = (cgroup_free_kb // 1024
                                      if cgroup_free_kb is not None else None)
                if cgroup_free_mb is None:
                    logger.warning("host pool: service cgroup at %s binds "
                                   "nothing (skipped-cgroup)", path)
                    prov["cgroup"] = "skipped-cgroup"
                else:
                    logger.info("host pool: cgroup headroom %d MiB (%s%s)",
                                cgroup_free_mb,
                                path,
                                ", memory.high" if "high" in caps else "")
            except (TypeError, ValueError) as e:
                logger.warning("host pool: cgroup caps at %s unreadable "
                               "(%s) — skipped-cgroup", path, e)
                cgroup_free_mb = None
                prov["cgroup"] = "skipped-cgroup"
        elif not cgroup_dir:
            logger.warning("host pool: no llama-swap cgroup found — "
                           "skipped-cgroup (declare hardware.cgroup_dir)")

        if hw.get("host_pool") is not None:
            raw = str(hw["host_pool"]).strip()
            if raw.lower() == "auto":
                pool_mb = min(sys_free_mb, cgroup_free_mb) \
                    if cgroup_free_mb is not None else sys_free_mb
                prov["pool"] = "auto"
            elif raw.endswith("%"):
                try:
                    frac = float(raw[:-1]) / 100.0
                except ValueError:
                    frac = None
                if frac is None:
                    logger.warning("host pool: hardware.host_pool %r "
                                   "unreadable; keeping the formula", raw)
                    pool_mb = min(sys_free_mb, cgroup_free_mb) \
                        if cgroup_free_mb is not None else sys_free_mb
                else:
                    pool_mb = int((total_mb or 0) * frac)
                    prov["pool"] = "declared"
                    logger.info("host pool: %d MiB (declared hardware."
                                "host_pool)", pool_mb)
            else:
                pool_mb = utils.parse_mem_mb(raw, total_mb or 0)
                prov["pool"] = "declared"
                logger.info("host pool: %d MiB (declared hardware.host_pool)",
                            pool_mb)
        else:
            pool_mb = min(sys_free_mb, cgroup_free_mb) \
                if cgroup_free_mb is not None else sys_free_mb
            prov["pool"] = ("min(sys_free, cgroup_free)"
                            if cgroup_free_mb is not None else "sys_free")
            logger.info("host pool: %d MiB (%s: sys_free %d%s, pressure zone "
                        "%d)", pool_mb, prov["pool"], sys_free_mb,
                        f", cgroup_free {cgroup_free_mb}" if cgroup_free_mb is not None else "",
                        pressure_mb)

        return cls(pool_mb=max(0, pool_mb),
                   total_mb=total_mb or 0,
                   pressure_zone_mb=pressure_mb,
                   sys_free_mb=sys_free_mb,
                   cgroup_free_mb=cgroup_free_mb,
                   provenance=prov)


def _find_llama_swap_pid() -> int | None:
    """A running llama-swap pid, or None.

    Scans /proc comm files; the first match wins (llama-swap is a single
    service per host in every deployment the packer serves).
    """
    try:
        for cgroup_file in Path("/proc").glob("[0-9]*/comm"):
            try:
                if cgroup_file.read_text().strip() == "llama-swap":
                    return int(cgroup_file.parent.name)
            except OSError:
                continue
    except OSError:
        pass
    return None