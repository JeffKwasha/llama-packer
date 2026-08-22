# llama_packer/vllm_estimate.py
"""vLLM memory estimation for the ``vllm`` / ``vllm-docker`` backends.

llama.cpp models are measured by ``llama-fit-params``; vLLM has no equivalent
binary.  Instead we use the optional ``vllm-memory-estimator`` package, which
reuses vLLM's own ``ModelConfig`` / ``KVCacheSpec`` logic (accurate for MLA,
sliding-window, hybrid-Mamba, and tensor-parallel models) to produce the same
``(model_mib, ctx_factor, compute_mib)`` triple that feeds the existing
``FitParams`` pipeline.  When the package is absent, callers fall back to the
local safetensors-header estimate (``utils.estimate_safetensors``).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def estimate_vllm(
    hf_repo: str,
    design_ctx: int,
    tensor_parallel_size: int = 1,
    max_active_seqs: int = 1,
) -> tuple[int, float, int] | None:
    """Estimate ``(model_mib, ctx_factor, compute_mib)`` for an HF model.

    Maps the estimator's components onto llama.cpp's fit-params categories:

    - ``model_mib``   = parameter (weight) bytes
    - ``compute_mib`` = activations + workspace + vLLM runtime overhead
    - ``ctx_factor``  = KV-cache bytes per token (at ``max_active_seqs``)

    ``ctx_factor`` is KV-cache per token so it matches the semantics of
    ``llama-fit-params`` (per-token KV, folded with ``parallel`` via
    ``max_active_seqs``).

    Returns None when the estimator package is not installed or the estimate
    fails (caller then falls back to a local safetensors estimate).
    """
    try:
        from memory_estimator import EstimatorInputs, estimate_from_inputs  # type: ignore[reportMissingImports]
    except ImportError:
        logger.debug("vllm-memory-estimator not installed; skipping")
        return None

    try:
        _summary, est = estimate_from_inputs(EstimatorInputs(
            model_id=hf_repo,
            max_seq_len=design_ctx,
            max_active_seqs=max_active_seqs,
            tensor_parallel_size=tensor_parallel_size,
        ))
    except Exception as e:
        logger.warning("vllm-memory-estimator failed for %s: %s", hf_repo, e)
        return None

    model_mib = int(est.parameters.nominal_gib * 1024)
    compute_mib = int(
        (est.activations.nominal_gib
         + est.workspace.nominal_gib
         + est.vllm_overhead.nominal_gib) * 1024
    )
    kv_cache_mib = est.kv_cache.nominal_gib * 1024
    if model_mib <= 0:
        return None
    ctx_factor = kv_cache_mib / design_ctx if design_ctx > 0 else 0.0
    return model_mib, ctx_factor, compute_mib
