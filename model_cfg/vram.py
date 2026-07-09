"""VRAM budget calculation: fit-params measurement, context sizing, and
matrix solving.

Extracted from ``model.py`` to separate VRAM calculation concerns from the
Model class, and to provide a natural home for the persisted ``fit-params``
block (``FitParams`` dataclass stored in sidecar frontmatter).
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from model_cfg import utils

if TYPE_CHECKING:
    from model_cfg.model import Model

logger = logging.getLogger(__name__)

# ── FitParams: persisted categorized VRAM measurements ──────────────────

# Required keys in the fit-params frontmatter block
_FIT_PARAMS_REQUIRED = frozenset({"model_mib", "ctx_factor", "compute_mib"})


@dataclass
class FitParams:
    """Categorized VRAM measurements for a model.

    These are constant for a given (model, cache_type, parallel) combination,
    so they can be persisted to sidecar frontmatter and reused across runs.

    Attributes:
        model_mib:   Weight-loading cost (constant regardless of context)
        ctx_factor:  MiB per token (linear KV-cache factor)
        compute_mib: Fixed compute overhead
        source:      How values were obtained ("fit-params" or "safetensors-estimate")
        cache_type:  KV cache quantization these values were measured with
        parallel:    Parallel slots these values were measured with
    """

    model_mib: int
    ctx_factor: float
    compute_mib: int
    source: str
    cache_type: str
    parallel: int

    @classmethod
    def from_dict(cls, d: dict, cache_type: str, parallel: int) -> FitParams | None:
        """Validate and construct from a frontmatter dict.

        Returns None if the block is missing, incomplete, has non-numeric
        values, or if cache_type/parallel don't match the current request
        (stale values trigger re-computation).
        """
        if not isinstance(d, dict):
            return None
        if not _FIT_PARAMS_REQUIRED.issubset(d):
            return None
        try:
            model_mib = int(d["model_mib"])
            ctx_factor = float(d["ctx_factor"])
            compute_mib = int(d["compute_mib"])
        except (TypeError, ValueError):
            return None
        if model_mib <= 0 or ctx_factor <= 0 or compute_mib < 0:
            return None
        saved_cache = str(d.get("cache_type", ""))
        saved_parallel = int(d.get("parallel", 1))
        if saved_cache != str(cache_type) or saved_parallel != int(parallel):
            return None
        return cls(
            model_mib=model_mib,
            ctx_factor=ctx_factor,
            compute_mib=compute_mib,
            source=str(d.get("source", "fit-params")),
            cache_type=saved_cache,
            parallel=saved_parallel,
        )

    def to_dict(self) -> dict:
        """Serialize to a frontmatter nested dict."""
        return {
            "model_mib": self.model_mib,
            "ctx_factor": self.ctx_factor,
            "compute_mib": self.compute_mib,
            "source": self.source,
            "cache_type": self.cache_type,
            "parallel": self.parallel,
        }


# ── VramBudget: per-model VRAM calculator ────────────────────────────────


class VramBudget:
    """VRAM budget calculator for a single model.

    Holds a reference to the ``Model`` for gguf_path, design context, and
    companion sizes.  Fit-params values are checked in this order:

    1. ``saved`` property (persisted in sidecar frontmatter)
    2. in-memory ``_static_cache`` (within process lifetime)
    3. ``llama-fit-params`` binary subprocess
    4. safetensors header estimation (fallback)

    When values are newly computed, they are persisted to the sidecar.
    """

    def __init__(self, model: Model) -> None:
        self.model = model
        self._cache: dict[tuple, tuple[int, int, int]] = {}
        self._static_cache: dict[tuple, FitParams] = {}
        self._sf_estimate: tuple[int, float] | None = None
        self._logged: set[str] = set()

    # ── saved fit-params from frontmatter ──

    @property
    def saved(self) -> FitParams | None:
        """Read validated fit-params block from model frontmatter."""
        raw = self.model.frontmatter.get("fit-params")
        if raw is None:
            return None
        cache_type = str(self.model.frontmatter.get("cache_type", "q8_0"))
        parallel = int(self.model.frontmatter.get("parallel", 1))
        return FitParams.from_dict(raw, cache_type, parallel)

    # ── raw fit-params binary call ──

    def fit_params(
        self,
        fit_bin: str,
        fit_ctx: int | None = None,
        fit_target_mib: int | None = None,
        cache_type: str = "q8_0",
        parallel: int = 1,
    ) -> tuple[int, int, int] | None:
        """Run llama-fit-params and return (model_mib, context_mib, compute_mib).

        Returns None on failure (binary missing, timeout, parse error).
        """
        cache_key = (fit_ctx, fit_target_mib, cache_type, parallel)
        if cache_key in self._cache:
            return self._cache[cache_key]

        assert self.model.gguf_path is not None
        cmd = [
            fit_bin,
            "--fit-print", "on",
            "-m", str(self.model.gguf_path),
            "--cache-type-k", cache_type,
            "--cache-type-v", cache_type,
        ]
        if fit_target_mib is not None:
            cmd += ["--fit-target", str(fit_target_mib)]
        if fit_ctx is not None:
            cmd += ["-c", str(fit_ctx)]
        if parallel > 1:
            cmd += ["--parallel", str(parallel)]

        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("fit-params timeout for %s", self.model.stem)
            return None
        except FileNotFoundError:
            logger.warning("fit-params binary not found: %s", fit_bin)
            return None
        except Exception as e:
            logger.warning("fit-params failed for %s: %s", self.model.stem, e)
            return None

        for line in out.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].startswith("Vulkan"):
                try:
                    model_mib = int(parts[1])
                    context_mib = int(parts[2])
                    compute_mib = int(parts[3])
                    self._cache[cache_key] = (model_mib, context_mib, compute_mib)
                    return model_mib, context_mib, compute_mib
                except ValueError:
                    pass

        if out.returncode != 0:
            msg = f"fit-params crashed for {self.model.stem} (exit {out.returncode})"
            if msg not in self._logged:
                self._logged.add(msg)
                logger.warning(msg)
        else:
            preview = "\n".join(out.stdout.splitlines()[:3])
            msg = f"could not parse fit-params output for {self.model.stem}: {preview}"
            if msg not in self._logged:
                self._logged.add(msg)
                logger.warning(msg)
        return None

    # ── static params (model_mib, ctx_factor, compute_mib) ──

    def fit_params_static(
        self,
        fit_bin: str,
        cache_type: str = "q8_0",
        parallel: int = 1,
    ) -> FitParams | None:
        """Get static VRAM parameters for this model.

        Checks: saved frontmatter → in-memory cache → binary → safetensors.
        Persists new values to sidecar on computation.
        """
        cache_key = (cache_type, parallel)
        if cache_key in self._static_cache:
            return self._static_cache[cache_key]

        # 1. Check saved values from frontmatter
        saved = self.saved
        if saved is not None:
            self._static_cache[cache_key] = saved
            return saved

        # 2. Compute via binary or safetensors
        design = self._design_ctx()
        result = self.fit_params(
            fit_bin=fit_bin,
            fit_ctx=design,
            fit_target_mib=None,
            cache_type=cache_type,
            parallel=parallel,
        )

        if result is not None:
            model_mib, ctx_at_design_mib, compute_mib = result
            context_factor = ctx_at_design_mib / design if design > 0 else 0.0
            params = FitParams(
                model_mib=model_mib,
                ctx_factor=context_factor,
                compute_mib=compute_mib,
                source="fit-params",
                cache_type=cache_type,
                parallel=parallel,
            )
        else:
            # 3. Try safetensors estimation fallback
            params = self._estimate_safetensors(cache_type, parallel, design)

        if params is not None:
            self._static_cache[cache_key] = params
            self._persist(params)

        return params

    def _estimate_safetensors(
        self, cache_type: str, parallel: int, design: int,
    ) -> FitParams | None:
        """Estimate FitParams from safetensors header (fallback for non-GGUF)."""
        assert self.model.gguf_path is not None
        if not str(self.model.gguf_path).endswith(".safetensors"):
            return None

        if self._sf_estimate is None:
            try:
                self._sf_estimate = utils.estimate_safetensors(
                    self.model.gguf_path, cache_type
                )
            except Exception as e:
                logger.warning(
                    "fit-params failed and safetensors estimate unavailable "
                    "for %s: %s", self.model.stem, e,
                )
                return None
            logger.warning(
                "fit-params failed for %s; estimating VRAM from safetensors header",
                self.model.stem,
            )

        est_model_mib, est_kv_per_token_mib = self._sf_estimate
        compute_mib = int(0.02 * est_model_mib) + 128
        ctx_at_design_mib = int(est_kv_per_token_mib * design)
        context_factor = ctx_at_design_mib / design if design > 0 else 0.0
        return FitParams(
            model_mib=est_model_mib,
            ctx_factor=context_factor,
            compute_mib=compute_mib,
            source="safetensors-estimate",
            cache_type=cache_type,
            parallel=parallel,
        )

    # ── context calculation ──

    def calc_ctx(
        self,
        vram_total_mb: int,
        fit_bin: str,
        parallel: int = 1,
        spare_mb: int = 0,
        mmproj_mb: int = 0,
        cache_type: str = "q8_0",
        design_ctx: int | None = None,
    ) -> int:
        """Calculate max context size for given VRAM.

        Uses saved fit-params if available; otherwise runs the binary (or
        safetensors estimation) and persists the results.
        """
        available = (
            vram_total_mb
            - utils._RESERVE_SYSTEM
            - utils._RESERVE_VIDEO
            - spare_mb
            - mmproj_mb
        )

        # Subtract MTP companion size
        if self.model.mtp and self.model.mtp.gguf_path:
            mtp_mb = utils.get_model_size_mb(str(self.model.mtp.gguf_path))
            available -= mtp_mb
            logger.debug("mtp companion %s: %d MB subtracted from VRAM",
                         self.model.mtp.stem, mtp_mb)

        if available <= 0:
            logger.warning("available VRAM <= 0 for %s (spare=%d, mmproj=%d)",
                           self.model.stem, spare_mb, mmproj_mb)
            return utils._MIN_CTX_SIZE

        design = design_ctx if design_ctx is not None else self._design_ctx()

        # Try saved or compute static params
        static = self.fit_params_static(fit_bin, cache_type=cache_type, parallel=parallel)

        if static is None:
            # Fatal: no way to estimate VRAM for this model
            raise RuntimeError(
                f"fit-params failed to measure VRAM for {self.model.stem}; "
                f"cannot estimate a safe context size. Ensure llama-fit-params "
                f"is available/built and the model format is supported "
                f"(safetensors may be unsupported)."
            )

        remaining = available - static.model_mib - static.compute_mib
        if remaining <= 0:
            logger.warning("model + compute exceeds available VRAM for %s", self.model.stem)
            return utils._MIN_CTX_SIZE

        # If design context fits, use it (capped by sidecar context_length)
        ctx_at_design_mib = int(static.ctx_factor * design)
        if ctx_at_design_mib <= remaining:
            sidecar_ctx = self.model.frontmatter.get("context_length")
            if sidecar_ctx is not None:
                return min(design, sidecar_ctx)
            return design

        # Scale down linearly
        max_ctx = int(remaining / static.ctx_factor)
        max_ctx = (max_ctx // utils._CTX_ROUND_TO) * utils._CTX_ROUND_TO
        return max(max_ctx, utils._MIN_CTX_SIZE)

    # ── helpers ──

    def _design_ctx(self) -> int:
        """Design context: GGUF architectural max > sidecar > default."""
        arch = self.model.gguf_context_length
        if arch is not None and arch > 0:
            return arch
        return int(self.model.frontmatter.get(
            "context_length", utils._DEFAULT_CONTEXT_LENGTH
        ))

    def _persist(self, params: FitParams) -> None:
        """Update only the fit-params block in the sidecar, preserving everything else.

        Reads the raw .md file, re-parses the frontmatter, injects the
        fit-params block, and writes back.  All other frontmatter keys and
        the markdown body are preserved unchanged.
        """
        md_path = self.model.md_path
        try:
            content = md_path.read_text(encoding="utf-8")
        except (OSError, PermissionError) as e:
            logger.debug("cannot read sidecar for fit-params persist (%s): %s", md_path, e)
            return

        if not content.startswith("---"):
            return
        parts = content.split("---", 2)
        if len(parts) < 3:
            return

        try:
            fm = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError:
            logger.debug("cannot parse sidecar frontmatter for persist: %s", md_path)
            return

        fm["fit-params"] = params.to_dict()
        new_content = "---\n" + yaml.dump(fm, sort_keys=False).rstrip() + "\n---" + parts[2]

        try:
            md_path.write_text(new_content, encoding="utf-8")
            logger.debug("persisted fit-params for %s", self.model.stem)
        except (OSError, PermissionError) as e:
            logger.debug("cannot write sidecar for fit-params persist (%s): %s", md_path, e)


# ── Module-level: matrix context solver ──────────────────────────────────


def solve_matrix_ctx(
    vram_total_mb: int,
    spare_mb: int,
    chat_models: list[tuple[Model, int, float, int]],
    embed_params: tuple[int, float, int] | None,
    rerank_params: tuple[int, float, int] | None,
    embed_ctx: int = 0,
    rerank_ctx: int = 0,
) -> int:
    """Solve the VRAM budget equation for chat context.

    The VRAM budget is:
        available = Σ(chat_weight + chat_factor*chat_ctx)
                   + (embed_weight + embed_factor*embed_ctx)
                   + (rerank_weight + rerank_factor*rerank_ctx)

    Since all chat models share the same VRAM pool (llama-swap evicts),
    we solve for the chat context that the LARGEST chat model needs.
    The fit-params for each chat model already accounts for its own weight.

    Args:
        vram_total_mb: Total VRAM in MB
        spare_mb: Reserved VRAM in MB
        chat_models: List of (model, model_mib, context_factor, compute_mib)
        embed_params: (model_mib, context_factor, compute_mib) for embedder
        rerank_params: (model_mib, context_factor, compute_mib) for reranker
        embed_ctx: Requested context for embedding model
        rerank_ctx: Requested context for reranking model

    Returns:
        Maximum chat context in tokens (rounded to _CTX_ROUND_TO)
    """
    available = vram_total_mb - utils._RESERVE_SYSTEM - utils._RESERVE_VIDEO - spare_mb

    embed_overhead = 0
    if embed_params and embed_ctx > 0:
        e_mib, e_factor, e_compute = embed_params
        embed_overhead = e_mib + e_compute + int(e_factor * embed_ctx)

    rerank_overhead = 0
    if rerank_params and rerank_ctx > 0:
        r_mib, r_factor, r_compute = rerank_params
        rerank_overhead = r_mib + r_compute + int(r_factor * rerank_ctx)

    remaining_for_chat = available - embed_overhead - rerank_overhead
    if remaining_for_chat <= 0:
        return utils._MIN_CTX_SIZE

    best_ctx = 0
    for model, model_mib, context_factor, compute_mib in chat_models:
        chat_budget = remaining_for_chat - model_mib - compute_mib
        if chat_budget <= 0:
            continue
        if context_factor > 0:
            ctx = int(chat_budget / context_factor)
        else:
            ctx = model.gguf_context_length or utils._DEFAULT_CONTEXT_LENGTH
        ctx = (ctx // utils._CTX_ROUND_TO) * utils._CTX_ROUND_TO
        ctx = max(ctx, utils._MIN_CTX_SIZE)
        arch_max = model.gguf_context_length or model.frontmatter.get(
            "context_length", utils._DEFAULT_CONTEXT_LENGTH
        )
        ctx = min(ctx, arch_max)
        best_ctx = max(best_ctx, ctx)

    return best_ctx if best_ctx > 0 else utils._MIN_CTX_SIZE