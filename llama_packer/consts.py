"""Fundamental constants for llama-packer.

These are the irreducible numbers the system runs on - VRAM reserves,
context defaults, KV-cache byte factors, backend defaults, and MTP
parameters. They are imported directly by whoever needs them; no
stub module remains.
"""

from __future__ import annotations

import re

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Context defaults
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_DEFAULT_CONTEXT_LENGTH = 32768
_CTX_ROUND_TO = 8192
_MIN_CTX_SIZE = 4096

# Minimum context for a chat model to be useful for agentic work
# (tool-call loops, long sessions). 128k.
_MIN_AGENTIC_CTX = 131072

# Reason string emitted as metadata.estimate_error on entries for models
# with no usable VRAM estimate (metadata.estimated=false): every estimate
# source failed (fit-params measurement, safetensors header, vLLM
# estimator). Such models are served at their minimum useful context and,
# in a matrix set, ride with no extra reserve — assumed to fit within the
# set's measured allocation.
ESTIMATE_ERROR_REASON = (
    "no VRAM estimate available (measurement failed); served at minimum "
    "useful context with no extra VRAM reserve"
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# VRAM reservation (MB)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_RESERVE_SYSTEM = 1024
_RESERVE_VIDEO = 1024

# Fraction inflated against every measured VRAM term when solving, so the
# law errs toward reserving more.  Beyond measurement residual this covers
# the VRAM the backend allocator consumes outside llama.cpp's own buffers
# (driver overhead, fragmentation — ~4% observed on Vulkan/radv) plus the
# desktop/compositor.  profiles.yaml `hardware.memory_margin` overrides.
_MEMORY_MARGIN = 0.04

# Serve-shaped measurement (VramBudget llama-server pair): the pair runs at
# min(design context, this cap) — small enough that both members fit VRAM,
# large enough to sit far above any SWA window so the per-token KV term is
# in its linear regime.
_MEASURE_CTX_CAP = 65536

# Per-run timeout (s) — generous enough for a ~30 GB model on a platter
# drive (~4-6 min load) — and the stall window (s): a run whose log stops
# growing for this long is hung and gets killed.
_MEASURE_TIMEOUT_S = 900
_MEASURE_STALL_S = 120

# Host-resident KV/RS (Vulkan_Host pool) above this (MiB) rejects a
# measurement point: the pool did not fit device memory (GPU busy or
# model too large) — the 2026-09-07 Dirk GTT spill signature.
_KV_SPILL_TOLERANCE_MIB = 8.0

# CPU-mapped weights beyond this (MiB) reject a point too (layers that
# never made it to the device), but llama.cpp structurally places some
# tensors host-side on Vulkan even in healthy runs (~1 GiB observed), so
# the tolerance must exceed that placement.
_WEIGHT_CPU_MAP_TOLERANCE_MIB = 2048.0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# MTP defaults
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_MTP_SPEC_TYPE = "draft-mtp"
_MTP_DRAFT_N_MAX = 2
_MTP_DRAFT_P_MIN = 0.75

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# vLLM backend defaults
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

VLLM_DEFAULT_IMAGE = "vllm/vllm-openai:latest"
VLLM_DEFAULT_BIN = "vllm"
VLLM_DEFAULT_CONTAINER_PORT = 8000
VLLM_DEFAULT_DOCKER_ARGS = "--runtime=nvidia --gpus all --shm-size=16g"
VLLM_DEFAULT_GPU_MEM_UTIL = 0.9

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# KV-cache bytes per element (incl. block overhead)
# Rounded up so derived memory estimates err toward reserving more.
# This is also the set of cache types llama-packer can size.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_KV_CACHE_BYTES = {
    "q8_0": 1.0625, "q8_1": 1.0625, "q8_k": 1.0625,
    "f16": 2.0, "bf16": 2.0, "f32": 4.0,
    "q4_0": 0.5625, "q4_1": 0.625, "q4_k": 0.5625,
    "q5_0": 0.6875, "q5_1": 0.75, "q5_k": 0.6875,
    "q6_0": 0.8125, "q6_k": 0.8125,
    "iq4_nl": 0.5625,
    # 4-bit E2M1 + FP8 E4M3 block scales per 16 elements ~= 0.5625 B/elem.
    "nvfp4": 0.5625,
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Safetensors dtype bytes
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_SAFETENSORS_DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8": 1, "F6E4M3FN": 1, "F6E5M2": 1, "F4": 1,
    "F6E2M1FN": 0.5, "F3": 0.375, "F2": 0.25, "F1": 0.125,
    # Real safetensors header dtype names (the safetensors spec, not torch
    # enum spellings): fp8 and the sub-byte MX family.  FP8 checkpoints
    # store ~1 B/elem — the same density as a Q8_0 GGUF.
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "F6_E2M3": 0.75, "F6_E3M2": 0.75, "F4_E2M1": 0.5, "F4_BNN": 0.125,
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Drive read speeds (MB/s, conservative real-world estimates)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_NVME_READ_MBPS = 1500
_SSD_READ_MBPS = 300
_HDD_READ_MBPS = 100
_UNKNOWN_READ_MBPS = 100

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# VRAM compute constants (MB)
# Backend-specific fixed compute overheads used by VramBudget.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_MMPROJ_COMPUTE_MB = 150
_DRAFT_COMPUTE_MB = 64
_DRAFT_CTX_SAFETY = 1.6
_SD_COMPUTE_MB = 512

# vLLM per-sequence runtime growth (MiB): CUDA-graph capture and scheduler
# workspace scale with the largest captured batch (``--max-num-seqs``).  The
# memory-estimator prices its overhead at one active sequence, so the affine
# fold carries the batch-dim growth through ``slot_mib``.  Conservative —
# errs toward reserving more, like the other fixed heuristics here.
_VLLM_PER_SEQ_MIB = 32.0
_WHISPER_COMPUTE_MB = 100
_KOKORO_COMPUTE_MB = 3072

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Diffusion architecture regex patterns
# Prefixes of stable-diffusion.cpp / ComfyUI general.architecture
# values.  Architecture names are a controlled vocabulary set by the
# gguf conversion scripts, so prefix matching is reliable.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_DIFFUSION_ARCH_RES = tuple(
    re.compile(p, re.I) for p in (
        r"^flux", r"^sd\d", r"^sdxl$", r"^ssd1", r"^stable-diffusion",
        r"^chroma", r"^wan\d?", r"^hidream", r"^ltxv?$", r"^hunyuan",
        r"^mochi", r"^cosmos", r"^auraflow", r"^pixart", r"^kandinsky",
        r"^sana", r"^ace-?step", r"^omnigen", r"^qwen[-_]?image",
        r"^z-?image", r"^ernie[-_]?image", r"^lumina", r"^sdx?-?l",
    )
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Default directory roles
# of a model file (relative to a models-dir root) selects the role;
# files at the root itself are chat.  Subdirectories absent from this
# map are not served at all.  Extend or override via the
# profiles.yaml dirs: mapping.  Keys are matched case-insensitively.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_DEFAULT_DIR_ROLES = {
    "chat": "chat", "t2t": "chat", "vision": "chat",
    "doc": "chat", "ocr": "chat",
    "embed": "embeddings", "rerank": "rerank",
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Non-chat roles
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

_NON_CHAT_ROLES = frozenset({"embeddings", "rerank", "image", "s2t", "t2s"})
