"""VRAM budget calculation: affine memory measurement, context sizing, and
matrix solving.

The VRAM law (validated across families; see :mod:`llama_packer.memory_probe`
and docs/plans/auto-parallel.md):

    VRAM(C, p) = model_mib + compute_mib + c*C + p*D

``C`` is the *total* shared KV pool (llama.cpp ``-c``; byte-identical to
``--kv-unified-per-slot X -np p`` with pool ``p*X``), ``c`` the shared
per-token cost, and ``D`` the fixed per-slot cost.  Every serve-shaped
extra — MTP draft weights and KV, hybrid-arch recurrent-state (RS) caches,
batch-size compute, slot overhead — is affine in ``(C, p)``, so a
three-run measurement trio under the exact flags the server will run
with pins the law from device totals alone:

    D = t(C, 2) - t(C, 1)          # the fixed term cancels
    c = 2 * (t(C, 1) - t(C/2, 1)) / C   # the pool difference cancels it
    fixed = t(C, 1) - c*C - D

Measurements come from real ``llama-server`` runs (device buffer lines in
the ``-lv 5`` log), not ``llama-fit-params`` — fit-params reports the KV
pool only, blind to the RS cache, the draft, and the batch-dependent
compute, which is exactly the undercount that made Dirk spill 4.8 GB into
GTT (2026-09-07; see docs/plans/auto-parallel.md).

``FitParams`` values persist in the sidecar ``derived:`` block — per
``cache_type`` and per measured flag **shape** (``derived: shape``: the
effective serve args — profiles global args + role batch flags; the
compute buffer depends on batch size and flash attention, so blocks are
never reused across shapes; a flags change re-measures).  Blocks are keyed
by ``source``: blocks from the retired fit-params measurement are rejected
on load and re-measured.  ``Model.persist_measured`` is the single writer.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from llama_packer import gpu_state
from llama_packer import utils
from llama_packer import vllm_estimate
from llama_packer.consts import (
    _CTX_ROUND_TO,
    _DRAFT_COMPUTE_MB,
    _DRAFT_CTX_SAFETY,
    _KV_SPILL_TOLERANCE_MIB,
    _MIN_CTX_SIZE,
    _MMPROJ_COMPUTE_MB,
    _MEASURE_STALL_S,
    _MEASURE_TIMEOUT_S,
    _MTP_SPEC_TYPE,
    _RESERVE_SYSTEM,
    _RESERVE_VIDEO,
    _SD_COMPUTE_MB,
    _VLLM_PER_SEQ_MIB,
    _WHISPER_COMPUTE_MB,
    _WEIGHT_CPU_MAP_TOLERANCE_MIB,
    _AUDIO_CPP_COMPUTE_MB,
    ESTIMATE_ERROR_REASON,
)
from llama_packer.backends import (AUDIO_CPP_BACKENDS, FIXED_OVERHEAD_BACKENDS,
                                   SD_BACKENDS, VLLM_BACKENDS,
                                   WHISPER_BACKENDS, get_backend)

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# ── FitParams: persisted categorized VRAM measurements ──────────────────

# Required keys in the measured frontmatter block
_FIT_PARAMS_REQUIRED = frozenset(
    {"model_mib", "kv_per_token_mib", "slot_mib", "compute_mib"})

# Per-backend fixed compute map, assembled from the backend name sets.
_FIXED_COMPUTE_MB = {**{n: _SD_COMPUTE_MB for n in SD_BACKENDS},
                     **{n: _WHISPER_COMPUTE_MB for n in WHISPER_BACKENDS},
                     **{n: _AUDIO_CPP_COMPUTE_MB for n in AUDIO_CPP_BACKENDS}}

# Persisted blocks acceptable without re-measurement.  The retired plain
# fit-params measurement (KV-pool only, no corrections) is deliberately
# absent.
_MEASURED_SOURCES = frozenset(
    {"llama-server", "fit-estimate", "vllm-estimate",
     "safetensors-estimate"})

# Device buffer lines in a llama.cpp ``-lv 5`` log: weights, per-context KV
# and recurrent-state pools, compute reserves, output buffers.
_BUFFER_RE = re.compile(
    r"(?:load_tensors|llama_kv_cache|llama_memory_recurrent|sched_reserve"
    r"|llama_context):\s+(\S+)\s+(model|KV|RS|compute|output)"
    r"\s+buffer size\s*=\s*([\d.]+)\s*MiB")


def _is_device_buffer(dev: str) -> bool:
    """True when *dev* names device-resident memory (not host-visible)."""
    return not (dev.endswith("_Host") or dev.startswith("CPU")
                or "Mapped" in dev)


def parse_device_buffers(log_text: str) -> dict[str, float]:
    """Sum device-resident buffer lines of a llama-server ``-lv 5`` log.

    Returns MiB per component: ``weights`` (max across load passes — the
    final pass reports real numbers, earlier fit-projection passes may
    print 0 under on-demand loading), ``kv``, ``rs``, ``compute`` and
    ``output`` (sums across every context the process created: the main
    context plus the MTP draft context when one is built).  Host-visible
    buffers (``Vulkan_Host``, ``CPU_Mapped``) are excluded: they live in
    GTT/system RAM, not VRAM — see :func:`parse_spill_mib` for when that
    exclusion makes a measurement invalid.
    """
    device, _, _ = _buffer_walk(log_text)
    return device


def parse_spill_mib(log_text: str) -> float:
    """MiB of measurement buffers that landed outside device memory.

    The per-run validity check for *capacity* spills: host-visible
    KV/RS pools (the pool did not fit beside whatever else holds VRAM —
    the 2026-09-07 Dirk GTT spill) plus CPU-mapped weights beyond the
    structural placement llama.cpp always uses on Vulkan (~1 GiB
    observed in healthy runs).  Host compute/output staging is a normal
    Vulkan cost and is not counted.
    """
    _, host_kv_rs, cpu_weights = _buffer_walk(log_text)
    return (host_kv_rs
            + max(0.0, cpu_weights - _WEIGHT_CPU_MAP_TOLERANCE_MIB))


_COMP_BY_TOKEN = {"model": "weights", "KV": "kv", "RS": "rs",
                  "compute": "compute", "output": "output"}


def _buffer_walk(log_text: str) -> tuple[dict[str, float], float, float]:
    """One pass over the buffer lines → (device sums, host KV/RS, CPU weights).

    Device sums: ``weights`` as the max per device across load passes,
    every other component summed across contexts.  Host KV/RS lines are
    the capacity-spill signature; CPU-mapped weight lines are tracked
    separately (structural placement up to a tolerance, offload failure
    beyond it).
    """
    weights: dict[str, float] = {}
    totals = {"kv": 0.0, "rs": 0.0, "compute": 0.0, "output": 0.0}
    host_kv_rs = 0.0
    cpu_weights = 0.0
    for m in _BUFFER_RE.finditer(log_text):
        dev, token, mib = m.group(1), m.group(2), float(m.group(3))
        comp = _COMP_BY_TOKEN[token]
        if _is_device_buffer(dev):
            if comp == "weights":
                weights[dev] = max(weights.get(dev, 0.0), mib)
            else:
                totals[comp] += mib
        elif comp in ("kv", "rs"):
            host_kv_rs += mib
        elif comp == "weights" and dev.startswith("CPU"):
            cpu_weights += mib
    return {"weights": sum(weights.values()), **totals}, host_kv_rs, cpu_weights


def _free_port() -> int:
    """A bindable localhost port for a throwaway measurement server."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def affine_from_pair(
    ctx_mib_1: float, ctx_mib_2: float, total_ctx: float,
) -> tuple[float, float]:
    """Derive ``(c, D)`` from two measurements at the same total context.

    ``D`` is the per-slot cost (``ctx(2) − ctx(1)``); ``c`` the remaining
    per-token KV share.  ``total_ctx`` must be the full ``-c`` pool the two
    measurements were taken at.  Negative measurement noise clamps ``D``
    to zero (err toward reserving more).
    """
    slot_mib = max(float(ctx_mib_2 - ctx_mib_1), 0.0)
    kv_per_token = (ctx_mib_1 - slot_mib) / total_ctx if total_ctx > 0 else 0.0
    return kv_per_token, slot_mib


# llama-fit-params ``-lv 5`` log lines (0.6 s per run, no tensor data read):
# the device report row plus the exact KV-pool and recurrent-state sizes.
_FIT_REPORT_RE = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.M)
_FIT_KV_RE = re.compile(r"llama_kv_cache:\s+size =\s*([\d.]+) MiB")
_FIT_RS_RE = re.compile(
    r"llama_memory_recurrent:\s+size =\s*([\d.]+) MiB \( *(\d+) cells")


def parse_fit_log(log_text: str) -> dict[str, float] | None:
    """Parse a llama-fit-params ``--fit-print on --fit off -lv 5`` run.

    Returns ``{model, context, compute, kv, rs, rs_cells}`` (MiB; the
    device report row only — the ``Host`` row is skipped), or None when
    the report row is missing.  ``kv`` sums *every* KV pool the context
    creates — for SWA architectures that is the C-scaling full-attention
    pool plus the fixed-size per-slot sliding ring — and ``rs`` sums the
    recurrent-state caches (``rs_cells`` their cells), so the
    ``(C, C/2)`` pool difference isolates the per-token term exactly.
    """
    for m in _FIT_REPORT_RE.finditer(log_text):
        dev = m.group(1)
        if dev == "Host" or dev.endswith("_Host"):
            continue
        kv = sum(float(x.group(1)) for x in _FIT_KV_RE.finditer(log_text))
        rs = 0.0
        rs_cells = 0
        for x in _FIT_RS_RE.finditer(log_text):
            rs += float(x.group(1))
            rs_cells += int(x.group(2))
        return {
            "model": int(m.group(2)),
            "context": int(m.group(3)),
            "compute": int(m.group(4)),
            "kv": kv,
            "rs": rs,
            "rs_cells": rs_cells,
        }
    return None


# ── per-arch serve corrections (measured once by --probe-memory) ─────────

_LEGACY_CORRECTIONS_NAME = "serve-corrections.json"
_CORRECTIONS_NAME = "serve-corrections.yaml"
_CORR_WARNED: set[str] = set()


def _corrections_path() -> Path:
    """Durable machine-local corrections file beside ``profiles.yaml``.

    Corrections are hardware/backend-specific (allocator overhead, RS
    geometry differences between estimate and serve): machine-local, hence
    gitignored — but durable and inspectable, unlike a cache directory.
    ``LLAMA_PACKER_CORRECTIONS`` relocates the file (tests; multi-host
    checkouts).  The retired ``~/.cache`` JSON stays readable as a
    fallback so existing calibrations survive the move without a re-probe.
    """
    override = os.environ.get("LLAMA_PACKER_CORRECTIONS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / _CORRECTIONS_NAME


def _load_corrections() -> dict:
    """Parsed corrections table: the YAML file, else the legacy cache JSON.

    Returns ``{}`` when neither exists.  A malformed YAML file is warned
    about and treated as empty (the estimate then runs uncorrected).
    """
    path = _corrections_path()
    for candidate, is_yaml in ((path, True),):
        try:
            with open(candidate) as f:
                table = yaml.safe_load(f)
            return table if isinstance(table, dict) else {}
        except FileNotFoundError:
            continue
        except (OSError, ValueError, yaml.YAMLError) as e:
            logger.warning("serve corrections unreadable (%s): %s — "
                           "estimates run uncorrected", candidate, e)
            return {}
    # Legacy fallback: the retired machine-local cache JSON.
    legacy = gpu_state.cache_dir() / _LEGACY_CORRECTIONS_NAME
    try:
        with open(legacy) as f:
            table = json.load(f)
        return table if isinstance(table, dict) else {}
    except (OSError, ValueError):
        return {}


def get_serve_correction(
    arch: str, cache_type: str, mtp_on: bool, shape: str = "",
) -> dict | None:
    """The measured ``(delta_fixed, delta_c, delta_d)`` row for an arch.

    Rows are written by ``--probe-memory`` (serve-shaped llama-server grid
    on the family representative minus the fast fit-params estimate) and
    keyed by ``arch | cache_type | mtp``; each row records the flag shape
    the witness ran under.  A row measured under a *different* shape is
    still applied — the deltas are mostly shape-independent allocator
    bias — with a one-time per-arch note that a re-probe would refresh it.
    Returns None when uncalibrated (the estimate runs uncorrected; that is
    never an error).
    """
    table = _load_corrections()
    row = table.get(f"{arch}|{cache_type}|{int(mtp_on)}")
    if row is None or not isinstance(row, dict):
        return None
    row_shape = str(row.get("shape", ""))
    if row_shape and shape and row_shape != shape:
        key = f"{arch}|{cache_type}|{int(mtp_on)}"
        if key not in _CORR_WARNED:
            _CORR_WARNED.add(key)
            logger.warning(
                "serve correction for arch %r was measured under %r, now "
                "estimating under %r — applying anyway (allocator-level "
                "deltas); --probe-memory %s refreshes it",
                arch, row_shape, shape, arch)
    return row


def save_serve_correction(
    arch: str, cache_type: str, mtp_on: bool, row: dict,
) -> None:
    """Persist one correction row (merges into the YAML table)."""
    path = _corrections_path()
    table = _load_corrections()
    row = dict(row)
    row.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    table[f"{arch}|{cache_type}|{int(mtp_on)}"] = row
    try:
        with open(path, "w") as f:
            yaml.safe_dump(table, f, default_flow_style=False, sort_keys=True)
    except OSError as e:
        logger.warning("cannot write serve corrections %s: %s", path, e)


# The llama-server readiness marker: every buffer line precedes it.
_MEASURE_READY = "listening on http"


def _proc_io_read(pid: int) -> int:
    """Bytes the child read from storage (``/proc/<pid>/io``; own child).

    The activity signal for the stall window: the multi-minute mmap load
    of a platter model is *silent* in the log (nothing is printed
    between "loading tensors" and completion), but its reads are
    visible here.  Returns -1 when unreadable.
    """
    try:
        with open(f"/proc/{pid}/io", "rb") as f:
            for line in f:
                if line.startswith(b"read_bytes:"):
                    return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return -1


def _wait_ready(
    proc: subprocess.Popen,
    log_file,
    ready_token: bytes,
    timeout_s: float,
    stall_s: float,
) -> str:
    """Poll the log for the ready marker.

    Returns ``"ready"``, or the failure shape: ``"exited"`` (the process
    died first), ``"stall"`` (no progress — log size *and* storage reads
    unchanged — for *stall_s* while alive; abandoned at the stall window
    instead of burning the full timeout) or ``"timeout"``.  Every buffer
    line precedes the ready marker.
    """
    deadline = time.monotonic() + timeout_s
    last_size, last_io, last_progress = -1, -1, time.monotonic()
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return "exited"
        log_file.seek(0, os.SEEK_END)
        size = log_file.tell()
        io_read = _proc_io_read(proc.pid)
        if size != last_size or io_read != last_io:
            last_size, last_io = size, io_read
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress > stall_s:
            return "stall"
        log_file.seek(0)
        if ready_token in log_file.read():
            return "ready"
        time.sleep(0.5)
    return "timeout"

# Flag-shape sources: sources whose constants depend on the measurement
# flag string (llama.cpp batch/attention flags).  The measured ``llama_args``
# is stamped into every ``derived:`` block verbatim (`derived: shape`), so
# the estimate and the serve can never drift apart on batch flags.
# vllm/safetensors estimates are flag-independent.
_SHAPE_SOURCES = frozenset({"llama-server", "fit-estimate", "fit-params"})


@dataclass
class FitParams:
    """Affine VRAM constants for a model at one cache precision.

    Constant for a given (model, cache_type) pair — parallel-independent by
    construction — so they persist to sidecar frontmatter and are reused
    across runs.

    Attributes:
        model_mib:         Weight-loading cost (constant regardless of
                           context or slots)
        kv_per_token_mib:  ``c`` — per-token cost of everything that scales
                           with the total pool (full-attention KV + MTP
                           draft KV when the server runs one)
        slot_mib:          ``D`` — fixed VRAM per parallel slot (recurrent
                           state cells + SWA ring buffers + slot overhead)
        compute_mib:       Compute/workspace buffers and any other constant
                           term (max over the p=1/p=2 measurement pair)
        source:            How values were obtained ("llama-server",
                            "vllm-estimate", "safetensors-estimate")
        cache_type:        KV cache quantization these values were measured
                            with (blocks are never derived across types)
        shape:             The effective serve-flag string the values were
                            measured under (:func:`measure_args`) — the
                            compute buffer depends on batch size and flash
                            attention, so blocks are shape-bound
    """

    model_mib: int
    kv_per_token_mib: float
    slot_mib: float
    compute_mib: int
    source: str
    cache_type: str
    shape: str = ""

    @classmethod
    def from_dict(cls, d: object, cache_type: str,
                  shape: str = "") -> FitParams | None:
        """Validate and construct from a frontmatter dict.

        Returns None if the block is missing, incomplete, has non-numeric
        values, if cache_type doesn't match the current request, if the
        block predates the serve-shaped measurement (``source`` "fit-params"
        or absent: those numbers undercount the RS cache, the MTP draft and
        the batch-dependent compute, so they are re-measured), or if the
        block predates shape stamping or was measured under a different
        flag shape (batch size / flash attention change the compute term —
        stale numbers are re-measured).
        """
        if not isinstance(d, dict):
            return None
        if not _FIT_PARAMS_REQUIRED.issubset(d):
            return None
        try:
            model_mib = int(d["model_mib"])
            kv_per_token_mib = float(d["kv_per_token_mib"])
            slot_mib = float(d["slot_mib"])
            compute_mib = int(d["compute_mib"])
        except (TypeError, ValueError):
            return None
        if model_mib <= 0 or kv_per_token_mib <= 0 or slot_mib < 0 \
                or compute_mib < 0:
            return None
        saved_cache = str(d.get("cache_type", ""))
        if saved_cache != str(cache_type):
            return None
        source = str(d.get("source", ""))
        if source not in _MEASURED_SOURCES:
            return None
        if "shape" not in d:
            # Pre-shape block: the flags it was measured under are unknown,
            # so its compute term cannot be trusted for any current shape.
            return None
        if source in _SHAPE_SOURCES and str(d["shape"]) != str(shape):
            return None
        return cls(
            model_mib=model_mib,
            kv_per_token_mib=kv_per_token_mib,
            slot_mib=slot_mib,
            compute_mib=compute_mib,
            source=source,
            cache_type=saved_cache,
            shape=str(d["shape"]),
        )

    def to_dict(self) -> dict:
        """Serialize to a frontmatter nested dict.

        ``ts`` stamps the measurement (ISO 8601): when an era's numbers
        are ever quarantined again, stale blocks are identifiable by
        inspection instead of by archaeology.  ``shape`` records the flag
        string the values were measured under.
        """
        return {
            "model_mib": self.model_mib,
            "kv_per_token_mib": self.kv_per_token_mib,
            "slot_mib": self.slot_mib,
            "compute_mib": self.compute_mib,
            "source": self.source,
            "cache_type": self.cache_type,
            "shape": self.shape,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

    def vram_mib(self, ctx_per_slot: int, parallel: int = 1) -> int:
        """Predicted VRAM at (per-slot context, slots) — the affine law."""
        pool = ctx_per_slot * parallel
        return int(self.model_mib + self.compute_mib
                   + self.kv_per_token_mib * pool + self.slot_mib * parallel)


# ── VramBudget: per-model VRAM calculator ────────────────────────────────


class VramBudget:
    """VRAM budget calculator for a single model.

    Holds a reference to the ``Model`` for gguf_path, design context, and
    companion sizes.  Fit-params values are checked in this order:

    1. ``saved_for`` (persisted in sidecar frontmatter, per cache_type)
    2. in-memory ``_static_cache`` (within process lifetime)
    3. ``llama-fit-params`` binary subprocess (one p=1/p=2 pair)
    4. safetensors header estimation (fallback)

    When values are newly computed, they are persisted to the sidecar.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self._cache: dict[tuple, dict[str, float]] = {}
        self._serve_cache: dict[tuple, dict[str, float]] = {}
        self._static_cache: dict[tuple[str, str], FitParams] = {}
        self._effective_cache: dict[tuple, tuple[int, float, float, int]] = {}
        self._companion_cache: dict[tuple, tuple[int, float, float, int]] = {}
        self._logged: set[str] = set()
        #: CLI ``--remeasure``: ignore the saved block and re-run the trio
        #: for every model (never a server; blocks re-persist with fresh
        #: ts/shape).
        self.remeasure: bool = False
        #: Set when no estimate source worked for this model — the planner
        #: surfaces it as metadata.estimated/estimate_error on every entry.
        self.unestimated_reason: str | None = None

    # ── saved fit-params from frontmatter ──

    def saved_for(self, cache_type: str, shape: str = "") -> FitParams | None:
        """Return the persisted measured block when it matches the *requested*
        cache type and flag shape.

        Validating against the requested values (not the sidecar-declared
        ones) is what makes a cache-type or serve-flags change invalidate a
        stale block and force a re-measurement instead of reusing mismatched
        numbers.  Any legacy block (pre-affine: no ``slot_mib``, or
        pre-shape: no ``shape``) also fails validation and is re-measured +
        rewritten.
        """
        raw = self.model.measured_block()
        if raw is None:
            return None
        return FitParams.from_dict(raw, cache_type, shape)

    # ── raw fit-params binary call ──

    #: Cache value of :meth:`fit_params`: the report triple plus the
    #: KV-pool and recurrent-state lines of the ``-lv 5`` log.
    def fit_params(
        self,
        fit_bin: str,
        fit_ctx: int | None = None,
        cache_type: str = "q8_0",
        parallel: int = 1,
        model_path: str | None = None,
        label: str | None = None,
        llama_args: str = "",
    ) -> dict[str, float] | None:
        """Run llama-fit-params and parse its buffer report.

        Returns a dict with ``model``/``context``/``compute`` (the report
        triple, MiB), ``kv`` (the KV pool size from the ``-lv 5`` log) and
        ``rs``/``rs_cells`` (the recurrent-state cache total and cell
        count — hybrid models only, zeros otherwise), or None on failure
        (binary missing, timeout, parse error).  ``llama_args`` are the
        profiles global flags (flash attention, batch sizes) the eventual
        serve will run with — compute depends on them.  ``label`` is used
        in log messages instead of the model stem (useful when measuring a
        companion GGUF).
        """
        if model_path is None:
            if self.model.gguf_path is None:
                return None
            model_path = str(self.model.gguf_path)
        label = label or self.model.stem
        cache_key = (model_path, fit_ctx, cache_type, parallel, llama_args)
        if cache_key in self._cache:
            return self._cache[cache_key]

        cmd = [
            fit_bin,
            "--fit-print", "on",
            "--fit", "off",
            "-lv", "5",
            "-m", str(model_path),
            "--cache-type-k", cache_type,
            "--cache-type-v", cache_type,
        ]
        if fit_ctx is not None:
            cmd += ["-c", str(fit_ctx)]
        if parallel > 1:
            cmd += ["--parallel", str(parallel)]
        cmd += llama_args.split()

        logger.info("measuring VRAM: %s via llama-fit-params", label)
        try:
            # errors="replace": fit-params logs can carry non-UTF-8 bytes
            # (tokenizer dumps, control chars); strict decoding would raise
            # UnicodeDecodeError inside subprocess itself.
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 errors="replace", timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("fit-params timeout for %s", label)
            return None
        except FileNotFoundError:
            logger.warning("fit-params binary not found: %s", fit_bin)
            return None
        except Exception as e:
            msg = f"fit-params failed for {label}: {e}"
            if msg not in self._logged:
                self._logged.add(msg)
                logger.warning(msg)
            return None

        parsed = parse_fit_log(out.stdout + "\n" + out.stderr)
        if parsed is not None:
            self._cache[cache_key] = parsed
            return parsed

        if out.returncode != 0:
            msg = f"fit-params crashed for {label} (exit {out.returncode})"
            if msg not in self._logged:
                self._logged.add(msg)
                logger.warning(msg)
        else:
            preview = "\n".join(out.stdout.splitlines()[:3])
            msg = f"could not parse fit-params output for {label}: {preview}"
            if msg not in self._logged:
                self._logged.add(msg)
                logger.warning(msg)
        return None

    # ── serve-shaped llama-server measurement ──

    def _mtp_measure_flags(self) -> list[str]:
        """Speculative-decode flags mirroring the serve emission exactly."""
        mtp_on, n_max = self.model._mtp_info()
        if not mtp_on:
            return []
        companion = self.model.mtp
        if companion is None and self.model.frontmatter.get("speculative"):
            return []  # declared companion missing: serve skips MTP too
        flags = ["--spec-type",
                 str(self.model.frontmatter.get("mtp_spec_type", _MTP_SPEC_TYPE)),
                 "--spec-draft-n-max", str(n_max)]
        if companion is not None and companion.gguf_path:
            flags += ["--spec-draft-model", str(companion.gguf_path)]
        return flags

    def _run_measure_server(
        self,
        server_bin: str,
        cache_type: str,
        ctx: int,
        parallel: int,
        llama_args: str,
    ) -> dict[str, float] | None:
        """One serve-shaped llama-server run → device buffer MiB, or None.

        Flags mirror what the emitted command will run with (cache types,
        all layers on GPU, MTP draft, the profiles global args — flash
        attention and the role batch sizes — plus the embed/rerank batch
        override), because compute and recurrent-state allocations depend
        on them.

        Guardrails (2026-09-07 contamination post-mortem): the pre-flight
        refuses to measure beside a resident llama process; the child
        runs in its own process group and dies on every exit path; a hung
        load is abandoned at the stall window, not the full timeout; and
        a run whose weights/KV/RS spilled to host-visible memory is
        rejected (its device sums undercount the truth).  Every run is
        journaled.
        """
        shape = (llama_args or "").strip()
        cache_key = ("serve", cache_type, ctx, parallel, shape)
        if cache_key in self._serve_cache:
            return self._serve_cache[cache_key]

        if self.model.gguf_path is None:
            return None
        residents = gpu_state.llama_residents()
        if residents:
            self._warn_once(
                "VRAM measurement refused for %s — llama processes "
                "resident: %s", self.model.stem, "; ".join(residents[:3]))
            gpu_state.journal({"mode": "serve", "model": self.model.stem,
                               "outcome": "refused-residents"})
            return None

        cmd = [
            str(server_bin), "-m", str(self.model.gguf_path),
            "-c", str(ctx), "--parallel", str(parallel),
            "--cache-type-k", cache_type, "--cache-type-v", cache_type,
            "-ngl", "999",
            "--port", str(_free_port()), "-lv", "5",
        ]
        cmd += self._mtp_measure_flags()
        cmd += shape.split()
        # Line-buffered stdout even redirected to a file: the stall
        # window reads log growth, and block buffering hides it during
        # the multi-minute load (false stalls on platter models).
        if shutil.which("stdbuf"):
            cmd = ["stdbuf", "-oL", "-eL"] + cmd

        logger.info("measuring VRAM: %s via llama-server (ctx=%d, p=%d)",
                    self.model.stem, ctx, parallel)
        started = time.monotonic()
        proc: subprocess.Popen | None = None
        outcome, spill, text = "launch-failed", 0.0, ""
        buffers: dict[str, float] | None = None
        with tempfile.NamedTemporaryFile(
                "w+b", suffix=".log", prefix="lp-measure-") as tf:
            try:
                proc = subprocess.Popen(cmd, stdout=tf,
                                        stderr=subprocess.STDOUT,
                                        start_new_session=True)
            except (OSError, ValueError) as e:
                self._warn_once("llama-server measurement: cannot launch %s: %s",
                                server_bin, e)
            else:
                try:
                    outcome = _wait_ready(
                        proc, tf, _MEASURE_READY.encode("ascii", "ignore"),
                        _MEASURE_TIMEOUT_S, _MEASURE_STALL_S)
                    if outcome == "ready":
                        tf.seek(0)
                        text = tf.read().decode("utf-8", errors="replace")
                finally:
                    # Our child, our kill — on every exit path, always.
                    if proc.poll() is None:
                        gpu_state.kill_process_group(proc)
                    else:
                        proc.wait()
        if outcome == "ready":
            spill = parse_spill_mib(text)
            if spill <= _KV_SPILL_TOLERANCE_MIB:
                buffers = parse_device_buffers(text)
                self._serve_cache[cache_key] = buffers
                outcome = "ok"
            else:
                outcome = "spill"
        gpu_state.journal({
            "mode": "serve", "model": self.model.stem, "ctx": ctx,
            "parallel": parallel,
            "pid": proc.pid if proc is not None else None,
            "outcome": outcome, "spill_mib": round(spill, 1),
            "dur_s": round(time.monotonic() - started, 1),
        })
        if buffers is None:
            self._warn_once(
                "llama-server measurement for %s (ctx=%d, p=%d) failed: %s",
                self.model.stem, ctx, parallel, outcome)
        return buffers

    def _measure_affine_trio(
        self,
        design: int,
        cache_type: str,
        llama_args: str,
        source: str,
        fit_bin: str,
    ) -> FitParams | None:
        """The (C, p) measurement trio from llama-fit-params — exact.

        Three header-only runs (~0.6 s each: no tensor data, no server,
        no meaningful VRAM use) with the exact serve flags pin the affine
        constants from the KV/recurrent-state pool lines alone:

            c = (pool(C) - pool(C/2)) / (C - C/2)   # ring/cells cancel
            D = pool(C, p=2) - pool(C, p=1)         # per-slot fixed cost

        where *pool* sums every KV pool and recurrent-state cache the
        ``-lv 5`` log reports (for SWA models that includes the
        fixed-size per-slot sliding ring, which the difference cancels).
        The divisor is the *actual* ctx delta between the two runs, so
        any design context works (the former ``max(C/2, 4096)`` floor
        silently mis-derived ``c`` for designs below 8192 — and sub-4k
        text designs are an error or a testcase anyway).
        The compute row is *not* part of the fit — under serve batch
        flags it is not affine in (C, p) (deepseek2: −64 MiB per ctx
        halving, −168 per extra slot) — so its max over the trio is the
        constant term, and the per-arch serve correction measured by
        ``--probe-memory`` closes the remaining fit-params-vs-serve gap
        (allocator overhead, MTP draft, compute shaping).
        """
        r1 = self.fit_params(fit_bin=fit_bin, fit_ctx=design,
                             cache_type=cache_type, parallel=1,
                             llama_args=llama_args)
        r2 = self.fit_params(fit_bin=fit_bin, fit_ctx=design,
                             cache_type=cache_type, parallel=2,
                             llama_args=llama_args)
        r3_ctx = max(design // 2, 1)
        r3 = self.fit_params(fit_bin=fit_bin,
                             fit_ctx=r3_ctx,
                             cache_type=cache_type, parallel=1,
                             llama_args=llama_args)
        if r1 is None or r2 is None or r3 is None \
                or design - r3_ctx <= 0 or r1["kv"] <= 0 or r3["kv"] <= 0:
            return None
        kv_per_token = (r1["kv"] - r3["kv"]) / (design - r3_ctx)
        if kv_per_token <= 0:
            return None
        pool1 = r1["kv"] + r1["rs"]
        slot_mib = max((r2["kv"] + r2["rs"]) - pool1, 0.0)
        compute_mib = int(max(r1["compute"], r2["compute"], r3["compute"]))
        return FitParams(
            model_mib=int(r1["model"]),
            kv_per_token_mib=kv_per_token,
            slot_mib=slot_mib,
            compute_mib=compute_mib,
            source=source,
            cache_type=cache_type,
            shape=llama_args,
        )

    def _fit_params_serve(
        self,
        fit_bin: str,
        cache_type: str,
        llama_args: str = "",
    ) -> FitParams | None:
        """Serve-shaped estimate from the fast fit-params trio (+ corrections).

        The trio (:meth:`_measure_affine_trio`) is exact for ``c`` and
        ``D``; the MTP draft and the allocator-level terms are not
        observable this way, so the measured per-arch correction
        (``--probe-memory``, machine-local cache) closes the gap:

            c   = c_kv                       (+ delta_c)
            D   = D_fit + per_cell * (mtp ? 2 : 1)   (+ delta_d)
            fix = model + compute + per_cell * (mtp ? 1 : 0)  (+ delta_fixed)

        Normal packer runs never start a server; this is the only
        measurement they do.  Not persisted — persistence belongs to
        :meth:`fit_params_static`.
        """
        shape = (llama_args or "").strip()
        params = self._measure_affine_trio(self._design_ctx(), cache_type,
                                           shape, source="fit-estimate",
                                           fit_bin=fit_bin)
        if params is None:
            return None
        # The correction's mtp dimension means "a draft is baked into the
        # main GGUF" — that is what the witness measured and what delta_d
        # prices.  A separate draft companion does not qualify (its cost
        # is folded in by effective_static instead).
        mtp_on = self.model.mtp is None and self.model._mtp_info()[0]
        corr = get_serve_correction(self.model.arch or "unknown", cache_type,
                                    mtp_on, shape)
        if corr is not None:
            # Corrections are measured at witness scale; clamp so a much
            # smaller model of the same arch can't go negative (from_dict
            # would reject the block and re-measure on every run).
            params = FitParams(
                model_mib=params.model_mib,
                kv_per_token_mib=max(1e-6, params.kv_per_token_mib
                                     + corr["delta_c"]),
                slot_mib=max(0.0, params.slot_mib + corr["delta_d"]),
                compute_mib=max(0, params.compute_mib
                                + int(corr["delta_fixed"])),
                source="fit-estimate",
                cache_type=cache_type,
                shape=shape,
            )
        else:
            # Uncorrected — never an error, so one note per arch, not per
            # model.  Probes are opt-in calibration, not a requirement.
            key = f"{self.model.arch}|{cache_type}|{int(mtp_on)}"
            if key not in _CORR_WARNED:
                _CORR_WARNED.add(key)
                logger.warning(
                    "estimate uncorrected for arch %r (mtp=%s) — ctx/fit "
                    "may undercount draft and allocator terms; optional: "
                    "--probe-memory %s calibrates",
                    self.model.arch, mtp_on, self.model.arch)
        return params

    def _fit_params_fast(self, cache_type: str, fit_bin: str,
                         llama_args: str = "") -> FitParams | None:
        """Transient fit-params trio — the uncalibrated approximation.

        Used only when the serve-shaped measurement is impossible (no
        usable GPU at pack time).  The numbers miss the MTP draft and
        the allocator-level terms, so they are never persisted: the next
        run re-attempts the calibrated measurement.
        """
        return self._measure_affine_trio(self._design_ctx(), cache_type,
                                         (llama_args or "").strip(),
                                         source="fit-params",
                                         fit_bin=fit_bin)

    # ── static params (model_mib, kv_per_token_mib, slot_mib, compute_mib) ──

    def fit_params_static(
        self,
        fit_bin: str,
        cache_type: str = "q8_0",
        llama_args: str = "",
    ) -> FitParams | None:
        """Get affine VRAM constants for this model at *cache_type*.

        Checks: saved frontmatter → in-memory cache → the fast fit-params
        estimate (never a server; normal runs pay ~1-2 s once per model)
        → plain fit-params pair (transient) → safetensors estimate.  New
        estimates are persisted to the sidecar so later runs pay nothing.
        Every path runs under the model's measured flag shape (the
        ``llama_args`` the caller composes: global args + the resolved
        ``-ub``) — the in-memory cache and the persisted block are both
        shape-bound, so a profiles/batch-key change re-measures instead of
        reusing stale compute terms.  ``remeasure`` (CLI ``--remeasure``)
        skips the saved-block path entirely.
        """
        shape = (llama_args or "").strip()
        if (cache_type, shape) in self._static_cache:
            return self._static_cache[(cache_type, shape)]

        if self.model.backend in FIXED_OVERHEAD_BACKENDS \
                or getattr(self.model, "on_cpu", False):
            # Fixed-overhead backends are sized from file size + a fixed
            # buffer (effective_static); CPU-resident models are not
            # VRAM-bound.  Neither has meaningful FitParams.
            return None

        # 1. Saved values from frontmatter (legacy fit-params blocks and
        #    wrong-shape blocks are rejected by FitParams.from_dict and
        #    re-measured).  A valid block living under a legacy key is
        #    rewritten under ``derived:`` — the label is the documentation.
        #    ``--remeasure`` skips this path entirely.
        saved = None if self.remeasure else self.saved_for(cache_type, shape)
        if saved is not None:
            from llama_packer.model import MEASURED_KEY
            if MEASURED_KEY not in self.model.frontmatter:
                self._persist(saved)
            self._static_cache[(cache_type, shape)] = saved
            return saved

        # 2. vLLM backends: estimate from the HF repo (vllm-memory-estimator)
        #    or a local safetensors header — the server measures GGUF only.
        if self.model.backend in VLLM_BACKENDS:
            params = self._fit_params_vllm(cache_type)
            if params is not None:
                self._static_cache[(cache_type, shape)] = params
                self._persist(params)
            return params

        # 3. Fast serve-shaped estimate (fit-params + arch corrections) —
        #    normal runs never start a server.
        params = self._fit_params_serve(fit_bin, cache_type, llama_args)
        if params is not None:
            self._static_cache[(cache_type, shape)] = params
            self._persist(params)
            return params

        # 4. Plain fit-params pair as a transient fallback (no corrections;
        #    never persisted — the next run retries the calibrated path).
        params = self._fit_params_fast(cache_type, fit_bin, llama_args)
        if params is not None:
            self._warn_once(
                "%s: llama-server measurement unavailable; using the "
                "fit-params approximation (not persisted)", self.model.stem)
            self._static_cache[(cache_type, shape)] = params
            return params

        # 5. Try safetensors estimation fallback
        params = self._estimate_safetensors(cache_type, self._design_ctx())
        if params is not None:
            self._static_cache[(cache_type, shape)] = params
            self._persist(params)
        return params

    def _fit_params_vllm(
        self, cache_type: str,
    ) -> FitParams | None:
        """Estimate VRAM constants for a vLLM-served model.

        Sources, in order:
        1. ``vllm-memory-estimator`` on the HF repo (accurate; reuses vLLM's
           own config/KV-cache logic) — requires ``hf_repo`` and the package.
           vLLM's paged KV pool is shared, so there is no llama.cpp-style
           per-slot recurrent term; the fixed ``_VLLM_PER_SEQ_MIB`` prices
           CUDA-graph/scheduler growth per active sequence (the estimator's
           own overhead is measured at one active sequence).
        2. local ``.safetensors`` header estimate (``utils.estimate_safetensors``).

        Returns None when neither is available (no estimator, no local file):
        the caller then sizes the model to its declared context and lets vLLM's
        own startup profiling bound the actual allocation.
        """
        design = self._design_ctx()
        if self.model.hf_repo:
            est = vllm_estimate.estimate_vllm(
                self.model.hf_repo, design,
                max_active_seqs=1, cache_type=cache_type,
            )
            if est is not None:
                model_mib, kv_per_token, compute_mib = est
                return FitParams(
                    model_mib=model_mib,
                    kv_per_token_mib=kv_per_token,
                    slot_mib=_VLLM_PER_SEQ_MIB,
                    compute_mib=compute_mib,
                    source="vllm-estimate",
                    cache_type=cache_type,
                )
        if self.model.gguf_path and str(self.model.gguf_path).endswith(".safetensors"):
            return self._estimate_safetensors(cache_type, design)
        return None

    def _estimate_safetensors(
        self, cache_type: str, design: int,
    ) -> FitParams | None:
        """Estimate FitParams from safetensors header (fallback for non-GGUF).

        Header numbers come from the canonical file instance (parsed once
        per process); the per-cache-type derivation below stays local.
        The header gives only a per-token KV number — the per-slot term is
        unknown and conservatively zero.
        """
        assert self.model.gguf_path is not None
        if not str(self.model.gguf_path).endswith(".safetensors"):
            return None

        try:
            nums = self.model.safetensors_numbers(cache_type)
        except Exception as e:
            self._warn_once(
                "fit-params failed and safetensors estimate unavailable "
                "for %s: %s", self.model.stem, e,
            )
            return None
        if nums is None:
            self._warn_once(
                "fit-params failed and safetensors estimate unavailable "
                "for %s", self.model.stem,
            )
            return None
        self._warn_once(
            "fit-params failed for %s; estimating VRAM from safetensors header",
            self.model.stem,
        )

        est_model_mib, est_kv_per_token_mib = nums
        compute_mib = int(0.02 * est_model_mib) + 128
        return FitParams(
            model_mib=est_model_mib,
            kv_per_token_mib=est_kv_per_token_mib,
            slot_mib=0.0,
            compute_mib=compute_mib,
            source="safetensors-estimate",
            cache_type=cache_type,
        )

    # ── companion measurement / estimation ──

    def _companion_fit(
        self,
        companion: "Model",
        main_fp: FitParams | None,
        cache_type: str = "q8_0",
        is_mmproj: bool = False,
    ) -> tuple[int, float, float, int] | None:
        """Estimate a companion (mmproj or MTP draft) affine VRAM constants.

        Returns (model_mib, kv_per_token_mib, slot_mib, compute_mib).
        Companion GGUFs cannot be measured by llama-fit-params — mmproj fails
        to load as a standalone model and MTP draft heads abort on a missing
        ``ctx_other`` — so the binary is not even attempted; instead VRAM is
        estimated directly from the file size, with the MTP draft's affine
        terms scaled from the main model.  This avoids launching a subprocess
        that would only crash (SIGABRT).  Results are cached per companion.
        """
        cache_key = ("companion", companion.stem, cache_type, is_mmproj)
        if cache_key in self._companion_cache:
            return self._companion_cache[cache_key]

        size_mb = utils.get_model_size_mb(str(companion.gguf_path))
        if is_mmproj:
            # The projection has no KV cache: weight + fixed compute only,
            # independent of context and slots.
            params = (size_mb, 0.0, 0.0, _MMPROJ_COMPUTE_MB)
        elif main_fp is not None and main_fp.kv_per_token_mib > 0 \
                and main_fp.model_mib > 0:
            # The draft holds its own KV cache that scales with context, so
            # both affine terms scale from the main model by relative size,
            # padded by a safety factor so the estimate errs on reserving more.
            ratio = (size_mb / main_fp.model_mib) * _DRAFT_CTX_SAFETY
            params = (size_mb,
                      main_fp.kv_per_token_mib * ratio,
                      main_fp.slot_mib * ratio,
                      _DRAFT_COMPUTE_MB)
        else:
            params = (size_mb, 0.0, 0.0, _DRAFT_COMPUTE_MB)
        msg = f"companion {companion.stem} VRAM estimated from file size (fit-params cannot measure mmproj/MTP)"
        if msg not in self._logged:
            self._logged.add(msg)
            logger.info(msg)

        self._companion_cache[cache_key] = params
        return params

    def effective_static(
        self,
        fit_bin: str,
        cache_type: str = "q8_0",
        design_ctx: int | None = None,
        include_mmproj: bool = True,
        llama_args: str = "",
    ) -> tuple[int, float, float, int] | None:
        """Combined affine VRAM constants for main model plus its companions.

        Returns (model_mib, kv_per_token_mib, slot_mib, compute_mib) where
        the mmproj's weight/compute is folded into the main model's
        numbers, so downstream context math sees a single budget.  The MTP
        draft is folded only when the main block did *not* come from the
        serve-shaped measurement — those blocks are measured with the
        draft running and already carry it.
        """
        cache_key = ("effective", cache_type, include_mmproj, llama_args)
        if cache_key in self._effective_cache:
            return self._effective_cache[cache_key]

        # Fixed-overhead backends (sd-server diffusion, whisper-server s2t):
        # VRAM = weights (file size, 0 when baked into a container image) + a
        # fixed runtime buffer, no KV terms.  These are excluded
        # from the shared chat matrix, so precise factors are irrelevant;
        # calc_ctx returns design_ctx when kv_per_token_mib==0.
        if self.model.backend in FIXED_OVERHEAD_BACKENDS:
            # Operator-pinned total VRAM (`vram_mb` in the sidecar) is the
            # sizing: it *is* the fixed overhead, no measurement or estimate.
            pin = self.model.frontmatter.get("vram_mb")
            if pin is not None:
                try:
                    pinned = int(pin)
                except (TypeError, ValueError):
                    pinned = 0
                    self._warn_once(
                        "vram_mb: %s: %r is not an integer; ignoring",
                        self.model.stem, pin)
                if pinned > 0:
                    return (pinned, 0.0, 0.0, 0)
            main_mb = 0
            try:
                if self.model.gguf_path and self.model.gguf_path.is_file():
                    main_mb = utils.get_model_size_mb(str(self.model.gguf_path))
            except OSError:
                main_mb = 0
            params = (main_mb, 0.0, 0.0,
                      _FIXED_COMPUTE_MB.get(self.model.backend, _SD_COMPUTE_MB))
            self._effective_cache[cache_key] = params
            return params

        main = self.fit_params_static(fit_bin, cache_type=cache_type,
                                      llama_args=llama_args)
        if main is None:
            return None

        # vLLM serves safetensors from an HF repo — vision/draft companions are
        # baked into the repo, not separate GGUF files, so nothing to fold in.
        if self.model.backend in VLLM_BACKENDS:
            params = (main.model_mib, main.kv_per_token_mib, main.slot_mib,
                      main.compute_mib)
            self._effective_cache[cache_key] = params
            return params

        model_mib = main.model_mib
        kv_per_token = main.kv_per_token_mib
        slot_mib = main.slot_mib
        compute_mib = main.compute_mib

        if main.source != "llama-server" and self.model.mtp \
                and self.model.mtp.gguf_path:
            draft = self._companion_fit(self.model.mtp, main,
                                        cache_type=cache_type)
            if draft:
                model_mib += draft[0]
                kv_per_token += draft[1]
                slot_mib += draft[2]
                compute_mib += draft[3]

        if include_mmproj and self.model.mmproj and self.model.mmproj.gguf_path:
            proj = self._companion_fit(self.model.mmproj, main,
                                       cache_type=cache_type,
                                       is_mmproj=True)
            if proj:
                model_mib += proj[0]
                kv_per_token += proj[1]
                slot_mib += proj[2]
                compute_mib += proj[3]

        params = (model_mib, kv_per_token, slot_mib, compute_mib)
        self._effective_cache[cache_key] = params
        return params

    # ── context calculation ──

    def calc_ctx(
        self,
        vram_total_mb: int,
        fit_bin: str,
        parallel: int = 1,
        spare_mb: int = 0,
        include_mmproj: bool = True,
        baseline_mb: int = 0,
        cache_type: str = "q8_0",
        design_ctx: int | None = None,
        memory_margin: float = 0.0,
        llama_args: str = "",
    ) -> int:
        """Calculate max per-slot context size for given VRAM.

        The solved value is the context available to *each* parallel slot;
        the emitted server flag is ``--kv-unified-per-slot X`` (shared pool
        ``parallel * X``).  Uses saved affine constants when available,
        otherwise measures the (p=1, p=2) pair and persists them.

        ``include_mmproj=False`` drops the vision projection from the budget
        (used when skipping mmproj to reach the minimum useful context).
        ``baseline_mb`` is the driver/compositor VRAM already in use; the
        effective reserve is ``_RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline)``.
        ``memory_margin`` inflates every measured term (safety against
        measurement residual).  ``llama_args`` are the profiles global
        server flags the measurement mirrors (flash attention, batch).
        """
        # CPU-resident models (--n-gpu-layers 0) are not VRAM-bound, so size
        # them to their own architectural/sidecar context limit rather than the
        # (possibly tiny) GPU budget, which would otherwise shrink them.
        if self.model.on_cpu:
            return self._design_ctx()

        reserve = _RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline_mb)
        available = (vram_total_mb - reserve - spare_mb) / (1.0 + memory_margin)

        if available <= 0:
            logger.warning("available VRAM <= 0 for %s (spare=%d)",
                           self.model.stem, spare_mb)
            return _MIN_CTX_SIZE

        design = design_ctx if design_ctx is not None else self._design_ctx()

        # Try saved or compute combined static params (main + companions)
        static = self.effective_static(
            fit_bin, cache_type=cache_type,
            design_ctx=design, include_mmproj=include_mmproj,
            llama_args=llama_args,
        )

        if static is None:
            # No estimate source worked (fit-params broken, no safetensors
            # header, no vLLM estimator). Never crash the pack: size to the
            # design context and flag the model unestimated — the planner
            # serves it at its minimum useful context and matrix
            # participants get a conservative largest-measured reserve.
            if self.unestimated_reason is None:
                self.unestimated_reason = ESTIMATE_ERROR_REASON
            self._warn_once(
                "VRAM measurement failed for %s; sizing to design context "
                "%d (unverified — may OOM)", self.model.stem, design)
            return design

        model_mib, kv_per_token, slot_mib, compute_mib = static
        remaining = available - model_mib - compute_mib
        if remaining <= 0:
            logger.warning(
                "%s: weights + compute need %d MiB, only %d MiB budgeted "
                "(VRAM %d - reserve %d - spare %d) — serving at minimum "
                "context %d",
                self.model.stem, int(model_mib + compute_mib),
                int(available), vram_total_mb, reserve, spare_mb,
                _MIN_CTX_SIZE)
            return _MIN_CTX_SIZE

        # Image token budget: image tokens are ordinary tokens inside the
        # slot's context (no VRAM beyond it), but a max-size image must *fit*
        # in each slot — the per-slot context is never allowed to drop below
        # image_max_tokens when the budget affords it.
        img_floor = self._image_floor_tokens(include_mmproj)

        # If design context fits, use it (capped by sidecar context_length)
        design_cost = (kv_per_token * design + slot_mib) * parallel
        if design_cost <= remaining:
            sidecar_ctx = self.model.frontmatter.get("context_length")
            if sidecar_ctx is not None:
                ctx = min(design, sidecar_ctx)
            else:
                ctx = design
            return self._raise_to_image_floor(ctx, img_floor, cap=ctx,
                                              affordable=ctx)

        # Solve the affine equation for the per-slot context
        if kv_per_token <= 0:
            return _MIN_CTX_SIZE
        max_ctx = int((remaining - slot_mib * parallel)
                      / (kv_per_token * parallel))
        ctx = (max_ctx // _CTX_ROUND_TO) * _CTX_ROUND_TO
        ctx = max(ctx, _MIN_CTX_SIZE)
        return self._raise_to_image_floor(ctx, img_floor,
                                          cap=max_ctx, affordable=max_ctx)

    def _image_floor_tokens(self, include_mmproj: bool) -> int:
        """Per-slot token floor that must stay reservable for image tokens.

        ``image_max_tokens`` when the vision projection is served and the
        sidecar declares a cap; 0 otherwise.
        """
        if not include_mmproj:
            return 0
        if not (self.model.mmproj and self.model.mmproj.gguf_path):
            return 0
        return self.model.image_max_tokens or 0

    def _raise_to_image_floor(
        self, ctx: int, floor: int, cap: int, affordable: int,
    ) -> int:
        """Raise ``ctx`` to the image token floor within cap/affordability.

        ``cap`` bounds the raise by the declared/architectural context;
        ``affordable`` by what the VRAM budget supports. When the floor
        exceeds what is available the request is logged (once) and the
        context is left unchanged — an oversized image then overflows and
        fails at request time instead of the server failing at load time.
        """
        if floor <= 0 or ctx >= floor:
            return ctx
        target = min(floor, cap, affordable)
        if target > ctx:
            if target < floor:
                self._warn_once(
                    "image tokens: %s: image_max_tokens budget %d per slot "
                    "exceeds the affordable context %d; raised ctx to %d, "
                    "large images may still not fit",
                    self.model.stem, floor, affordable, target)
            return target
        self._warn_once(
            "image tokens: %s: image_max_tokens budget %d per slot exceeds "
            "the solved context %d (affordable %d); large images may not fit",
            self.model.stem, floor, ctx, affordable)
        return ctx

    # ── helpers ──

    def _warn_once(self, msg: str, *args: object) -> None:
        """Log a warning once per process (deduplicated by formatted message)."""
        text = msg % args if args else msg
        if text not in self._logged:
            self._logged.add(text)
            logger.warning(text)

    def _design_ctx(self) -> int:
        """Design context: GGUF architectural max > sidecar > default."""
        return self.model.design_context

    def _persist(self, params: FitParams) -> None:
        """Persist measured VRAM constants via the single sidecar writer.

        Delegates to :meth:`Model.persist_measured` — the only place that
        writes the dynamic ``derived:`` branch.
        """
        self.model.persist_measured(params.to_dict())


# ── Module-level: matrix context solver ──────────────────────────────────


def solve_matrix_ctx(
    vram_total_mb: int,
    spare_mb: int,
    chat_models: list[tuple[Model, int, float, float, int, int, int]],
    embed_params: tuple[int, float, float, int] | None,
    rerank_params: tuple[int, float, float, int] | None,
    embed_ctx: int = 0,
    rerank_ctx: int = 0,
    baseline_mb: int = 0,
    fixed_overhead_mb: int = 0,
    memory_margin: float = 0.0,
) -> int:
    """Solve the shared per-slot chat context under the affine VRAM law.

    The VRAM budget for a co-resident group sharing one per-slot context X:

        available = Σ_i (model_i + compute_i + c_i*(p_i*X) + p_i*D_i)
                  + (embed_weight + embed_factor*embed_ctx + embed_slots)
                  + (rerank_weight + rerank_factor*rerank_ctx + rerank_slots)

    Args:
        vram_total_mb: Total VRAM in MB
        spare_mb: Reserved VRAM in MB
        chat_models: List of (model, model_mib, kv_per_token_mib, slot_mib,
            compute_mib, parallel, image_floor_tokens) — the floor being
            ``image_max_tokens`` per slot for served vision, 0 otherwise
        embed_params: (model_mib, kv_per_token_mib, slot_mib, compute_mib)
            for embedder (runs at its own parallel count)
        rerank_params: same, for reranker
        embed_ctx: Requested per-slot context for embedding model
        rerank_ctx: Requested per-slot context for reranking model
        baseline_mb: Driver/compositor VRAM already in use (added to the reserve)
        fixed_overhead_mb: Extra fixed VRAM held by opportunistic co-loads
            (subtracted from the chat budget before solving)
        memory_margin: Fraction inflated against every measured term

    Returns:
        Maximum shared per-slot chat context in tokens (rounded to
        _CTX_ROUND_TO)
    """
    reserve = _RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline_mb)
    available = (vram_total_mb - reserve - spare_mb) / (1.0 + memory_margin)

    embed_overhead = 0
    if embed_params and embed_ctx > 0:
        e_mib, e_factor, e_slots, e_compute = embed_params
        embed_overhead = e_mib + e_compute \
            + int(e_factor * embed_ctx + e_slots)

    rerank_overhead = 0
    if rerank_params and rerank_ctx > 0:
        r_mib, r_factor, r_slots, r_compute = rerank_params
        rerank_overhead = r_mib + r_compute \
            + int(r_factor * rerank_ctx + r_slots)

    remaining_for_chat = available - embed_overhead - rerank_overhead \
        - fixed_overhead_mb
    if remaining_for_chat <= 0:
        return _MIN_CTX_SIZE

    best_ctx = 0
    for model, model_mib, kv_factor, slot_mib, compute_mib, parallel, \
            img_floor in chat_models:
        if kv_factor <= 0:
            # Unestimable rider (zero-cost placeholder): it rides whatever
            # the measurable models afford and must not set the shared bar
            # — its native max would inflate chat_ctx for everyone (and
            # suppress tools demotion). The planner serves it at its type
            # floor and flags the entry unverified.
            continue
        chat_budget = remaining_for_chat - model_mib - compute_mib
        if chat_budget <= 0:
            continue
        ctx = int((chat_budget - slot_mib * parallel)
                  / (kv_factor * parallel))
        # The image token floor is raise-to-fit only when it was already
        # affordable; a larger floor cannot buy VRAM it doesn't have.
        if img_floor > ctx:
            logger.warning(
                "matrix: %s image_max_tokens budget %d per slot exceeds "
                "the solved chat ctx %d; large images may not fit",
                model.stem, img_floor, ctx)
        ctx = (ctx // _CTX_ROUND_TO) * _CTX_ROUND_TO
        ctx = max(ctx, _MIN_CTX_SIZE)
        arch_max = model.design_context
        ctx = min(ctx, arch_max)
        best_ctx = max(best_ctx, ctx)

    return best_ctx if best_ctx > 0 else _MIN_CTX_SIZE
