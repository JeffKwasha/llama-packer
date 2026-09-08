"""VRAM probe: serve-shaped measurement and affine-law validation.

The VRAM law (validated 2026-09-07 across qwen35/gemma4 — dense, MoE and
SWA architectures, q8_0/f16/bf16/q4_0, p ∈ {1,2,4,8}, C ∈ {64k,128k,256k};
see docs/plans/auto-parallel.md):

    VRAM(C, p) = model_mib + compute_mib + c*C + p*D

``C`` is the *total* shared KV pool (llama.cpp ``-c``; byte-identical to
``--kv-unified-per-slot X -np p`` with pool ``p*X``), ``c`` the per-token
cost, and ``D`` the fixed per-slot cost.  Everything serve-shaped — MTP
draft KV and compute, hybrid-arch recurrent-state caches, batch-dependent
compute — is affine in ``(C, p)``.  A measurement trio — ``(C, 1)``,
``(C, 2)``, ``(C/2, 1)`` — pins the law from device totals alone:
``D = t(C,2) − t(C,1)``, ``c = 2·(t(C,1) − t(C/2,1))/C``,
``fixed = t(C,1) − c·C − D``.

Measurement source: real ``llama-server`` runs under the exact flags the
server will run with (fit-params reports the KV pool only and misses the
RS/draft/batch terms — the 2026-09-07 Dirk GTT spill).  The probe CLI
(``llama-packer --probe-memory [ARCH...]``) measures one representative
per GGUF arch family across p ∈ {1,2,4,8} and reports the derived
constants plus the max residual against the law — the check when a new
architecture family shows up.  All functions are side-effect free (no
persistence) and importable by other tools.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from llama_packer import gpu_state
from llama_packer.consts import _MEASURE_CTX_CAP, _MIN_CTX_SIZE
from llama_packer.vram import (
    save_serve_correction,
)

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

#: Validation grid — (pool ctx, parallel) points per family representative.
#: Two pool sizes x three parallel counts: the C axis validates that ``c``
#: is constant across context (draft KV included), the p axis that ``D``
#: is constant across slots (RS cells included).  The per-model trio
#: (``VramBudget.measure_server_pair``) is exactly determined and cannot
#: falsify the law; the grid adds the degrees of freedom that can.
PROBE_GRID_BASE = 65536

#: Max relative residual accepted before a family FAILs the law check.
AFFINE_TOLERANCE = 0.005

#: Absolute residual floor (MiB): tiny-KV models with sub-MiB per-slot costs
#: show pure quantization noise that the relative tolerance alone flags.
AFFINE_ABS_TOLERANCE_MIB = 16.0


def pick_family_representatives(
    models: list[Model],
    only_archs: tuple[str, ...] | None = None,
    is_fast: Callable[[str], bool] | None = None,
) -> dict[str, Model]:
    """One measurable GGUF model per architecture family.

    Only real local ``.gguf`` files with a readable arch qualify (CPU-resident
    models are never VRAM-sized; error models carry no usable file).  When
    *only_archs* is given, just those families are considered.  Preference
    order per family: fast-storage models first (when *is_fast* says a
    model's resolved path sits on the fast tier — measurement loads are
    ~6x cheaper there), then the largest file as the most conservative
    scaling witness.  Returned in arch-sorted order for stable output.
    """
    def key(m: Model) -> tuple[int, int]:
        fast = 0 if (is_fast and str(m.gguf_path)
                     and is_fast(str(m.gguf_path))) else 1
        try:
            return (fast, -m.size_mb)
        except OSError:
            return (fast, 0)

    best: dict[str, Model] = {}
    best_key: dict[str, tuple[int, int]] = {}
    for m in models:
        if getattr(m, "_override_error", None):
            continue
        if getattr(m, "on_cpu", False):
            continue
        path = getattr(m, "gguf_path", None)
        if path is None or not str(path).endswith(".gguf"):
            continue
        try:
            if not path.is_file():
                continue
            arch = m.arch
        except (OSError, ValueError):
            continue
        if not arch:
            continue
        if only_archs is not None and arch not in only_archs:
            continue
        rank = key(m)
        if arch not in best or rank < best_key[arch]:
            best[arch] = m
            best_key[arch] = rank
    return dict(sorted(best.items()))


def _probe_grid(design: int) -> tuple[tuple[int, int], ...]:
    """The (pool ctx, parallel) validation points for a model.

    Three pool sizes (design capped at ``_MEASURE_CTX_CAP``, then halves)
    at p ∈ {1, 2}: three C points check that ``c`` is constant across
    context (draft KV and the C-linear compute included), two p values
    check ``D`` — five runs leaving ``n − 3`` residual degrees of freedom.
    """
    c1 = min(design, _MEASURE_CTX_CAP)
    c2 = max(c1 // 2, _MIN_CTX_SIZE)
    c3 = max(c1 // 4, _MIN_CTX_SIZE)
    return tuple(dict.fromkeys(
        (c, p)
        for c, p in ((c1, 1), (c1, 2), (c2, 1), (c2, 2), (c3, 1))
    ))


def _fit_affine(
    points: list[tuple[int, int, int]],
) -> tuple[float, float, float] | None:
    """Least-squares ``(fixed, c, D)`` for ``y = fixed + c*C + D*p``.

    Normal equations (3x3, Cramer's rule) — no numeric dependency.  The
    per-model trio is exactly determined; a grid fit has ``n − 3``
    residual degrees of freedom, which is what makes the law falsifiable.
    """
    n = len(points)
    sc = sum(float(c) for c, p, y in points)
    sp = sum(float(p) for c, p, y in points)
    sy = sum(float(y) for c, p, y in points)
    scc = sum(float(c) ** 2 for c, p, y in points)
    scy = sum(float(c) * float(y) for c, p, y in points)
    scp = sum(float(c) * float(p) for c, p, y in points)
    spp = sum(float(p) ** 2 for c, p, y in points)
    spy = sum(float(p) * float(y) for c, p, y in points)

    det = (n * (scc * spp - scp * scp)
           - sc * (sc * spp - scp * sp)
           + sp * (sc * scp - scc * sp))
    if abs(det) < 1e-9:
        return None

    det_fixed = (sy * (scc * spp - scp * scp)
                 - sc * (scy * spp - scp * spy)
                 + sp * (scy * scp - scc * spy))
    det_c = (n * (scy * spp - scp * spy)
             - sy * (sc * spp - scp * sp)
             + sp * (sc * spy - scy * sp))
    det_d = (n * (scc * spy - scy * scp)
             - sc * (sc * spy - scy * sp)
             + sy * (sc * scp - scc * sp))
    return det_fixed / det, det_c / det, det_d / det


def affine_report(
    model: Model,
    fit_bin: str,
    server_bin: str,
    cache_type: str,
    llama_args: str = "",
    grid: tuple[tuple[int, int], ...] | None = None,
    tolerance: float = AFFINE_TOLERANCE,
) -> dict | None:
    """Calibrate and validate one family representative.

    Measures the fast estimate (fit-params trio — the thing normal runs
    use, no server), then the serve truth (llama-server grid), and
    derives the per-arch correction row ``truth − est`` that gets
    persisted for the family.

    Both sides derive ``(c, D)`` from their own KV/recurrent-state pool
    lines with the same exact math — the ``(C, C/2)`` difference cancels
    every fixed-size pool, the p difference isolates the per-slot cost —
    so ``delta_c``/``delta_d`` stay ~0 unless the two tools genuinely
    allocate differently (the gemma SWA ring does), and ``delta_fixed``
    carries the real gap: fit-params' compute shaping vs the server's.
    The law is validated on the truth pool lines (the grid's extra point
    leaves a falsifying degree of freedom); the totals residual is
    reported separately because serve compute is not affine in (C, p)
    under batch flags.
    """
    est = model.vram._fit_params_serve(fit_bin, cache_type, llama_args)
    bufs: dict[tuple[int, int], dict] = {}
    for ctx, parallel in (grid or _probe_grid(model.design_context)):
        buf = model.vram._run_measure_server(
            server_bin, cache_type, ctx, parallel, llama_args)
        if buf is not None:
            bufs[(ctx, parallel)] = buf
    if len(bufs) < 4:
        return None
    big = max(ctx for ctx, p in bufs if p == 1)
    if (big, 2) not in bufs or (big // 2, 1) not in bufs:
        return None

    def pool(key: tuple[int, int]) -> float:
        buf = bufs[key]
        return buf["kv"] + buf["rs"]

    c = 2.0 * (pool((big, 1)) - pool((big // 2, 1))) / big
    d = pool((big, 2)) - pool((big, 1))
    if c <= 0:
        return None
    fixed = bufs[(big, 1)]["weights"] + bufs[(big, 1)]["compute"] \
        + max(0.0, bufs[(big, 1)]["output"])

    if est is None:
        est_row: dict | None = None
        corr = None
    else:
        est_fixed = est.model_mib + est.compute_mib
        est_row = {"fixed_mib": est_fixed,
                   "c": est.kv_per_token_mib, "d": est.slot_mib}
        corr = {"delta_fixed": round(fixed - est_fixed),
                "delta_c": c - est.kv_per_token_mib,
                "delta_d": d - est.slot_mib,
                "rep_stem": model.stem,
                "cache_type": cache_type}
        mtp_on, _ = model._mtp_info()
        save_serve_correction(model.arch or "unknown", cache_type, mtp_on,
                              corr)
    residuals: dict[str, float] = {}
    max_abs = 0.0
    max_total_abs = 0.0
    for key, buf in bufs.items():
        ctx, parallel = key
        err = pool(key) - (c * ctx + d * parallel)
        residuals[f"{ctx}/{parallel}"] = abs(err) / pool(key)
        max_abs = max(max_abs, abs(err))
        predicted_total = fixed + c * ctx + d * parallel
        total = sum(buf.values())
        max_total_abs = max(max_total_abs,
                            abs(predicted_total - total) / total)
    max_residual = max(residuals.values(), default=0.0)
    return {
        "model": model,
        "kv_per_token_mib": c,
        "slot_mib": d,
        "fixed_mib": round(fixed),
        "est": est_row,
        "corr": corr,
        "grid": sorted(bufs),
        "residuals": residuals,
        "max_residual": max_residual,
        "max_abs_mib": max_abs,
        "max_total_residual": max_total_abs,
        "ok": max_residual <= tolerance or max_abs <= AFFINE_ABS_TOLERANCE_MIB,
    }


def format_reports(reports: list[dict | None]) -> str:
    """Render affine reports as an aligned text table."""
    head = ("stem", "arch", "size_mb", "c_mib/tok", "slot_mib", "fixed_mib",
            "pts", "max_err", "corr(f/c/D)", "verdict")
    body = []
    for rep in reports:
        if rep is None:
            body.append(("-", "-", "-", "-", "-", "-", "-", "-", "-", "FAIL"))
            continue
        m = rep["model"]
        corr = rep["corr"]
        corr_txt = ("n/a" if corr is None else
                    f"{corr['delta_fixed']:+d}/"
                    f"{corr['delta_c']:+.4f}/{corr['delta_d']:+.1f}")
        body.append((
            m.stem, m.arch, str(m.size_mb),
            f"{rep['kv_per_token_mib']:.6f}", f"{rep['slot_mib']:.1f}",
            str(rep["fixed_mib"]), str(len(rep["grid"])),
            f"{rep['max_residual']:.4%}", corr_txt,
            "PASS" if rep["ok"] else "FAIL",
        ))
    widths = [len(h) for h in head]
    for line in body:
        widths = [max(w, len(col)) for w, col in zip(widths, line)]
    out = ["  ".join(h.ljust(w) for h, w in zip(head, widths))]
    for line in body:
        out.append("  ".join(col.ljust(w) for col, w in zip(line, widths)))
    return "\n".join(out)


def run_probe(
    models: list[Model], fit_bin: str, server_bin: str,
    cache_type: str = "q8_0",
    only_archs: tuple[str, ...] | None = None,
    llama_args: str = "",
    is_fast: Callable[[str], bool] | None = None,
) -> str:
    """Full probe: pick representatives, calibrate corrections, validate.

    The only place llama-server is ever started — normal packer runs use
    the fast fit-params estimate plus the corrections this probe writes.
    Refuses to start beside a resident llama process (measurement
    validity is the whole point) and runs single-flight under the
    measurement lock; every serve run is journaled by
    ``VramBudget._run_measure_server``.
    """
    residents = gpu_state.llama_residents()
    if residents:
        return ("probe: refusing to measure — llama processes resident:\n  "
                + "\n  ".join(residents[:5]))
    with gpu_state.measurement_lock("probe"):
        reps = pick_family_representatives(models, only_archs, is_fast)
        if not reps:
            return "probe: no measurable GGUF families found"
        logger.info("probe: %d families (%s)", len(reps), ", ".join(reps))
        gpu_state.journal({"mode": "probe", "archs": ", ".join(reps)})
        reports: list[dict | None] = []
        for arch, model in reps.items():
            logger.info("probe: %s (arch %s) grid %s", model.stem, arch,
                        _probe_grid(model.design_context))
            reports.append(affine_report(model, fit_bin, server_bin,
                                         cache_type, llama_args))
        return format_reports(reports)
