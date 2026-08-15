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

from llama_packer import utils
from llama_packer import vllm_estimate

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# ── FitParams: persisted categorized VRAM measurements ──────────────────

# Required keys in the fit-params frontmatter block
_FIT_PARAMS_REQUIRED = frozenset({"model_mib", "ctx_factor", "compute_mib"})

# Companion VRAM fallback constants (see _companion_fit docstring).
# mmproj: fixed compute buffer on top of its weights (vision projection buffers).
_MMPROJ_COMPUTE_MB = 150
# MTP draft: fixed compute overhead + per-token KV factor estimate safety margin.
_DRAFT_COMPUTE_MB = 64
_DRAFT_CTX_SAFETY = 1.6


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
        self._effective_cache: dict[tuple, tuple[int, float, int]] = {}
        self._companion_cache: dict[tuple, tuple[int, float, int]] = {}
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
        model_path: str | None = None,
        label: str | None = None,
    ) -> tuple[int, int, int] | None:
        """Run llama-fit-params and return (model_mib, context_mib, compute_mib).

        Returns None on failure (binary missing, timeout, parse error).
        ``label`` is used in log messages instead of the model stem (useful
        when measuring a companion GGUF).
        """
        if model_path is None:
            if self.model.gguf_path is None:
                return None
            model_path = str(self.model.gguf_path)
        label = label or self.model.stem
        cache_key = (model_path, fit_ctx, fit_target_mib, cache_type, parallel)
        if cache_key in self._cache:
            return self._cache[cache_key]

        cmd = [
            fit_bin,
            "--fit-print", "on",
            "--fit", "off",
            "-m", str(model_path),
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
            logger.warning("fit-params timeout for %s", label)
            return None
        except FileNotFoundError:
            logger.warning("fit-params binary not found: %s", fit_bin)
            return None
        except Exception as e:
            logger.warning("fit-params failed for %s: %s", label, e)
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

        # 2. vLLM backends: estimate from the HF repo (vllm-memory-estimator)
        #    or a local safetensors header — llama-fit-params only measures GGUF.
        if self.model.is_vllm:
            params = self._fit_params_vllm(cache_type, parallel)
            if params is not None:
                self._static_cache[cache_key] = params
                self._persist(params)
            return params

        # 3. Compute via binary or safetensors
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
            # 4. Try safetensors estimation fallback
            params = self._estimate_safetensors(cache_type, parallel, design)

        if params is not None:
            self._static_cache[cache_key] = params
            self._persist(params)

        return params

    def _fit_params_vllm(
        self, cache_type: str, parallel: int,
    ) -> FitParams | None:
        """Estimate VRAM params for a vLLM-served model.

        Sources, in order:
        1. ``vllm-memory-estimator`` on the HF repo (accurate; reuses vLLM's
           own config/KV-cache logic) — requires ``hf_repo`` and the package.
        2. local ``.safetensors`` header estimate (``utils.estimate_safetensors``).

        Returns None when neither is available (no estimator, no local file):
        the caller then sizes the model to its declared context and lets vLLM's
        own startup profiling bound the actual allocation.
        """
        design = self._design_ctx()
        if self.model.hf_repo:
            est = vllm_estimate.estimate_vllm(
                self.model.hf_repo, design,
                max_active_seqs=parallel,
            )
            if est is not None:
                model_mib, ctx_factor, compute_mib = est
                return FitParams(
                    model_mib=model_mib,
                    ctx_factor=ctx_factor,
                    compute_mib=compute_mib,
                    source="vllm-estimate",
                    cache_type=cache_type,
                    parallel=parallel,
                )
        if self.model.gguf_path and str(self.model.gguf_path).endswith(".safetensors"):
            return self._estimate_safetensors(cache_type, parallel, design)
        return None

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

    # ── companion measurement / estimation ──

    def _companion_fit(
        self,
        companion: "Model",
        fit_bin: str,
        main_fp: FitParams | None,
        design_ctx: int,
        cache_type: str = "q8_0",
        parallel: int = 1,
        is_mmproj: bool = False,
    ) -> tuple[int, float, int] | None:
        """Estimate a companion (mmproj or MTP draft) static VRAM params.

        Returns (model_mib, ctx_factor, compute_mib). Companion GGUFs cannot be
        measured by llama-fit-params — mmproj fails to load as a standalone
        model and MTP draft heads abort on a missing ``ctx_other`` — so the
        binary is not even attempted; instead VRAM is estimated directly from
        the file size (with the MTP draft's per-token KV factor scaled from the
        main model). This avoids launching a subprocess that would only crash
        (SIGABRT), which on a GPU that already has a model resident is both
        noisy and risky. Results are cached per companion.
        """
        cache_key = ("companion", companion.stem, cache_type, parallel)
        if cache_key in self._companion_cache:
            return self._companion_cache[cache_key]

        # Conservative file-size estimate. The MTP draft holds its own KV cache
        # that scales with context, so we estimate its per-token factor from
        # the main model scaled by relative size, padded by a safety factor so
        # the estimate errs on the side of reserving more.
        size_mb = utils.get_model_size_mb(str(companion.gguf_path))
        if is_mmproj:
            params = (size_mb, 0.0, _MMPROJ_COMPUTE_MB)
        elif main_fp is not None and main_fp.ctx_factor > 0 and main_fp.model_mib > 0:
            draft_factor = main_fp.ctx_factor * (size_mb / main_fp.model_mib) * _DRAFT_CTX_SAFETY
            params = (size_mb, draft_factor, _DRAFT_COMPUTE_MB)
        else:
            params = (size_mb, 0.0, _DRAFT_COMPUTE_MB)
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
        parallel: int = 1,
        design_ctx: int | None = None,
        include_mmproj: bool = True,
    ) -> tuple[int, float, int] | None:
        """Combined static VRAM params for main model plus its companions.

        Returns (model_mib, ctx_factor, compute_mib) where the MTP draft's
        weight and per-token KV factor and the mmproj's weight/compute are
        folded into the main model's numbers, so downstream context math sees a
        single budget.  This is what fixes companion VRAM being under-budgeted
        (previously charged by raw file size only).
        """
        cache_key = ("effective", cache_type, parallel, include_mmproj)
        if cache_key in self._effective_cache:
            return self._effective_cache[cache_key]

        main = self.fit_params_static(fit_bin, cache_type=cache_type, parallel=parallel)
        if main is None:
            return None

        # vLLM serves safetensors from an HF repo — vision/draft companions are
        # baked into the repo, not separate GGUF files, so nothing to fold in.
        if self.model.is_vllm:
            params = (main.model_mib, main.ctx_factor, main.compute_mib)
            self._effective_cache[cache_key] = params
            return params

        design = design_ctx if design_ctx is not None else self._design_ctx()
        model_mib = main.model_mib
        ctx_factor = main.ctx_factor
        compute_mib = main.compute_mib

        if self.model.mtp and self.model.mtp.gguf_path:
            draft = self._companion_fit(self.model.mtp, fit_bin, main, design,
                                        cache_type=cache_type, parallel=parallel)
            if draft:
                model_mib += draft[0]
                ctx_factor += draft[1]
                compute_mib += draft[2]

        if include_mmproj and self.model.mmproj and self.model.mmproj.gguf_path:
            proj = self._companion_fit(self.model.mmproj, fit_bin, main, design,
                                       cache_type=cache_type, parallel=parallel,
                                       is_mmproj=True)
            if proj:
                model_mib += proj[0]
                ctx_factor += proj[1]
                compute_mib += proj[2]

        params = (model_mib, ctx_factor, compute_mib)
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
    ) -> int:
        """Calculate max context size for given VRAM.

        Uses saved fit-params if available; otherwise runs the binary (or
        safetensors estimation) and persists the results.  Companion (MTP draft,
        mmproj) VRAM is folded in via :meth:`effective_static`.

        ``include_mmproj=False`` drops the vision projection from the budget
        (used when skipping mmproj to reach the minimum useful context).
        ``baseline_mb`` is the driver/compositor VRAM already in use; the
        effective reserve is ``_RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline)``.
        """
        # CPU-resident models (--n-gpu-layers 0) are not VRAM-bound, so size
        # them to their own architectural/sidecar context limit rather than the
        # (possibly tiny) GPU budget, which would otherwise shrink them.
        if self.model.on_cpu:
            return self._design_ctx()

        reserve = utils._RESERVE_SYSTEM + max(utils._RESERVE_VIDEO, baseline_mb)
        available = vram_total_mb - reserve - spare_mb

        if available <= 0:
            logger.warning("available VRAM <= 0 for %s (spare=%d)",
                           self.model.stem, spare_mb)
            return utils._MIN_CTX_SIZE

        design = design_ctx if design_ctx is not None else self._design_ctx()

        # Try saved or compute combined static params (main + companions)
        static = self.effective_static(
            fit_bin, cache_type=cache_type, parallel=parallel,
            design_ctx=design, include_mmproj=include_mmproj,
        )

        if static is None:
            if self.model.is_vllm:
                # No memory estimate available (no estimator, no local
                # safetensors): size to the declared context and let vLLM's own
                # startup profiling bound the actual allocation.
                return self._design_ctx()
            # Fatal: no way to estimate VRAM for this model
            raise RuntimeError(
                f"fit-params failed to measure VRAM for {self.model.stem}; "
                f"cannot estimate a safe context size. Ensure llama-fit-params "
                f"is available/built and the model format is supported "
                f"(safetensors may be unsupported)."
            )

        model_mib, ctx_factor, compute_mib = static
        remaining = available - model_mib - compute_mib
        if remaining <= 0:
            logger.warning("model + compute exceeds available VRAM for %s", self.model.stem)
            return utils._MIN_CTX_SIZE

        # If design context fits, use it (capped by sidecar context_length)
        ctx_at_design_mib = int(ctx_factor * design)
        if ctx_at_design_mib <= remaining:
            sidecar_ctx = self.model.frontmatter.get("context_length")
            if sidecar_ctx is not None:
                return min(design, sidecar_ctx)
            return design

        # Scale down linearly
        if ctx_factor <= 0:
            return utils._MIN_CTX_SIZE
        max_ctx = int(remaining / ctx_factor)
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
    baseline_mb: int = 0,
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
        baseline_mb: Driver/compositor VRAM already in use (added to the reserve)

    Returns:
        Maximum chat context in tokens (rounded to _CTX_ROUND_TO)
    """
    reserve = utils._RESERVE_SYSTEM + max(utils._RESERVE_VIDEO, baseline_mb)
    available = vram_total_mb - reserve - spare_mb

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