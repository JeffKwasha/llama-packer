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

from llama_packer.consts import _KV_CACHE_BYTES

logger = logging.getLogger(__name__)


def estimate_vllm(
    hf_repo: str,
    design_ctx: int,
    tensor_parallel_size: int = 1,
    max_active_seqs: int = 1,
    cache_type: str = "f16",
) -> tuple[int, float, int] | None:
    """Estimate ``(model_mib, ctx_factor, compute_mib)`` for an HF model.

    Maps the estimator's components onto llama.cpp's fit-params categories:

    - ``model_mib``   = parameter (weight) bytes
    - ``compute_mib`` = activations + workspace + vLLM runtime overhead
    - ``ctx_factor``  = KV-cache bytes per token (at ``max_active_seqs``)

    ``ctx_factor`` is KV-cache per token so it matches the semantics of
    ``llama-fit-params`` (per-token KV, folded with ``parallel`` via
    ``max_active_seqs``).

    The estimator prices KV at the model's native (auto) dtype — bf16/f16,
    2 bytes/elem.  ``cache_type`` rescales it to the configured cache
    precision (``q8_0`` → vLLM ``--kv-cache-dtype fp8`` = 1.0625 B/elem,
    etc.) so the estimate matches what the emitted command will serve.

    Returns None when the estimator package is not installed or the estimate
    fails (caller then falls back to a local safetensors estimate).
    """
    try:
        from memory_estimator import EstimatorInputs, estimate_from_inputs  # type: ignore[reportMissingImports]
    except ImportError:
        logger.debug("vllm-memory-estimator not installed; skipping")
        return None

    logger.info("estimating vLLM memory: %s", hf_repo)
    import os
    # Never download model weights: the estimator only needs config/tokenizer
    # metadata, and any weight fetch would be far over the 5MB budget. Force
    # the HF hub client offline for the duration of the call; the caller falls
    # back to the local safetensors-header estimate when this returns None.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
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
    finally:
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline

    model_mib = int(est.parameters.nominal_gib * 1024)
    compute_mib = int(
        (est.activations.nominal_gib
         + est.workspace.nominal_gib
         + est.vllm_overhead.nominal_gib) * 1024
    )
    kv_cache_mib = est.kv_cache.nominal_gib * 1024
    if model_mib <= 0:
        return None
    # The estimator assumes the native (auto) KV dtype (~2 B/elem); rescale
    # to the configured cache precision so ctx_factor matches the emission.
    kv_cache_mib *= _KV_CACHE_BYTES.get(cache_type, 2.0) / 2.0
    ctx_factor = kv_cache_mib / design_ctx if design_ctx > 0 else 0.0
    return model_mib, ctx_factor, compute_mib
