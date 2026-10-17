# llama_packer/backends/vllm.py
"""vLLM engine: safetensors / HF-repo serving.

Serves chat, embeddings (``--task embed``) and rerank (``--task score``)
models, either as a host binary or inside a container.  The *engine* owns the
serve command line; the *transport* (``transport.py``) owns how that command
is launched — host process, docker or podman — and translates the model /
chat-template / speculative-draft paths for containers.

Speculative decoding applies to generation only.  LoRA is not yet wired into
vLLM's module registry, so a declared ``loras`` setting is warned about and
skipped.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from llama_packer import utils
from llama_packer.consts import (
    _MTP_DRAFT_N_MAX,
    VLLM_DEFAULT_GPU_MEM_UTIL,
    VLLM_DEFAULT_BIN,
    VLLM_DEFAULT_CONTAINER_PORT,
    VLLM_DEFAULT_IMAGE,
)
from llama_packer.backends.base import BaseBackend
from llama_packer.backends.transport import (
    CONTAINER_HF_HOME,
    HostTransport,
    Launch,
    Transport,
    container_device_flags,
)

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# Our cache_type values that map onto vLLM's --kv-cache-dtype. vLLM supports
# only auto (f16/bf16/f32) and fp8 KV caches; our q8_* precisions are ~8-bit
# and map to "fp8" (e4m3). Sub-byte block quants have no vLLM equivalent.
_KV_DTYPE_FP8 = frozenset({"q8_0", "q8_1", "q8_k"})
_KV_DTYPE_AUTO = frozenset({"f16", "bf16", "f32"})


def _kv_cache_dtype_flags(cache_type: str) -> list[str]:
    """Translate our cache_type into vLLM ``--kv-cache-dtype`` flags.

    Single configuration means one cache precision decision drives both
    backends wherever a valid flag exists — including experimental values;
    whether the serving build supports them is the operator's call.  Only a
    cache_type with *no* valid upstream flag is warned about and skipped.
    """
    if cache_type in _KV_DTYPE_FP8:
        return ["--kv-cache-dtype", "fp8"]
    if cache_type in _KV_DTYPE_AUTO:
        return []  # vLLM "auto" already serves at half/full precision
    if cache_type == "nvfp4":
        return ["--kv-cache-dtype", "nvfp4"]
    logger.warning("vllm: cache_type %r has no --kv-cache-dtype equivalent "
                   "(valid values: auto/fp8/nvfp4); serving at auto instead",
                   cache_type)
    return []


# llama.cpp-only flags that must never reach a vLLM command line.  The
# render layers (``vllm args``, sidecar ``cli_args``) and llama-swap dir
# flag macros are backend-agnostic, so a llama-shaped flag can leak in;
# strip it (flag and value) before the cmd ships.
_LLAMA_ONLY_FLAGS = frozenset({
    "-b", "-ub", "-c", "--ctx-size", "--ctx",
    "--batch-size", "--ubatch-size", "-ngl", "--n-gpu-layers",
    "--flash-attn", "--no-flash-attn", "-fa",
    "--cache-type-k", "--cache-type-v", "--ctk", "--ctv",
    "--kv-unified-per-slot", "--parallel",
    "--mmproj", "--no-mmproj", "--embd-normalize", "--pooling",
    "--lora", "--lora-scaled", "--reasoning-format", "--reasoning-budget",
    "--spec-type", "--spec-draft-n-max", "--no-mmap", "--mlock",
    "--threads", "-t", "--main-gpu", "--tensor-split",
})


def _strip_llama_only_flags(args: str) -> str:
    """Drop llama.cpp-only flags (and their values) from a free-form layer.

    Applied to the backend-agnostic render layers (``vllm args``, sidecar
    ``cli_args``) — never to the builtin flag list, whose values (JSON
    speculative configs, quoted templates) must round-trip untouched.
    """
    flags = utils._pair_flags(shlex.split(args))
    return " ".join(
        flag if not value else f"{flag} {shlex.quote(value)}"
        for flag, value in flags.items()
        if flag not in _LLAMA_ONLY_FLAGS)


# The ``mamba:`` recipe mapping (one sub-key → one flag; absent sub-keys emit
# nothing).  Experimental vLLM flags: validate names/values against the served
# image's `vllm serve --help` before relying on them — rename/remap here only.
_MAMBA_FLAG_MAP: dict[str, tuple[str, type]] = {
    "backend": ("--mamba-backend", str),
    "ssm_cache_dtype": ("--mamba-ssm-cache-dtype", str),
    "philox_rounds": ("--mamba-cache-philox-rounds", int),
    "cache_mode": ("--mamba-cache-mode", str),
    "stochastic_rounding": ("--enable-mamba-cache-stochastic-rounding", bool),
}


def _mamba_flags(model: "Model") -> list[str]:
    """Render the ``mamba:`` recipe (hybrid/Mamba models) into vLLM flags."""
    raw = model.frontmatter.get("mamba")
    if raw is None or raw is False or raw == "":
        return []
    if not isinstance(raw, dict):
        logger.warning("vllm: %s: mamba: must be a mapping of sub-keys "
                       "(backend:, ssm_cache_dtype:, stochastic_rounding:, "
                       "philox_rounds:, cache_mode:); got %r — ignoring",
                       model.stem, raw)
        return []
    m = raw
    flags: list[str] = []
    for key, (flag, cast) in _MAMBA_FLAG_MAP.items():
        v = m.get(key)
        if v is None or v is False:
            continue
        if cast is bool:
            if v is True:
                flags.append(flag)
            else:
                logger.warning("vllm: %s: mamba.%s=%r is not boolean (ignored)",
                               model.stem, key, v)
        elif cast is int:
            try:
                flags += [flag, str(int(v))]
            except (TypeError, ValueError):
                logger.warning("vllm: %s: mamba.%s=%r is not an int (ignored)",
                               model.stem, key, v)
        else:
            flags += [flag, str(v)]
    for k in m:
        if k not in _MAMBA_FLAG_MAP:
            logger.warning("vllm: %s: unknown mamba sub-key %r (known: %s)",
                           model.stem, k, ", ".join(sorted(_MAMBA_FLAG_MAP)))
    return flags


def _speculative_config(model: "Model") -> dict | None:
    """The ``--speculative-config`` JSON dict for *model*, or None.

    Precedence:

    1. Explicit ``speculative_config:`` frontmatter (a raw mapping — full
       control over any vLLM method: eagle3, ngram, draft_model, ...).
    2. Baked-in MTP (``mtp: true``) → ``{"method": "mtp",
       "num_speculative_tokens": N}`` with N from the same
       ``mtp_draft_n_max`` key and default the llama-server path uses —
       one configuration, identical semantics on every backend.

    A GGUF ``speculative:`` companion cannot be loaded by vLLM (it needs an
    HF repo); that case is warned about and skipped — use
    ``speculative_config: {method: draft_model, model: <hf-repo>, ...}``
    instead.  See https://docs.vllm.ai/en/latest/features/speculative_decoding/
    """
    fm = model.frontmatter
    cfg = fm.get("speculative_config")
    if isinstance(cfg, dict) and cfg:
        return cfg
    if fm.get("mtp"):
        n = int(fm.get("mtp_draft_n_max", _MTP_DRAFT_N_MAX))
        return {"method": "mtp", "num_speculative_tokens": n}
    if fm.get("speculative"):
        logger.warning("vllm: %s: GGUF speculative companion %r cannot be loaded "
                       "by vLLM; skipping speculative decoding (use "
                       "`speculative_config:` with a draft HF repo instead)",
                       model.stem, fm["speculative"])
    return None


def _spec_meta(model: "Model") -> dict:
    """Backend metadata for the writer's mtp_* metadata keys."""
    if model.role != "chat":
        return {"mtp_enabled": False}
    spec = _speculative_config(model)
    if not spec:
        return {"mtp_enabled": False}
    meta = {"mtp_enabled": True,
            "mtp_draft_max": spec.get("num_speculative_tokens", 0)}
    return meta


def _container_args(tvars: dict, runtime: str) -> str:
    """Extra runtime flags for a container launch.

    Precedence: explicit ``container_args`` > the legacy per-runtime key
    (``docker_args``/``podman_args``) > a vendor-detected default.
    """
    explicit = tvars.get("container_args")
    if explicit:
        return str(explicit)
    legacy = tvars.get(f"{runtime}_args")
    if legacy:
        return str(legacy)
    vendor = str(tvars.get("container_vendor") or "cpu")
    device = container_device_flags(vendor, runtime)
    return " ".join(filter(None, (device, "--shm-size=16g")))


class VllmBackend(BaseBackend):
    """The vLLM engine (serve command); launcher supplied by the transport."""

    name = "vllm"
    formats = frozenset({".safetensors", "hf_repo"})
    roles = frozenset({"chat", "embeddings", "rerank"})
    handles = frozenset({
        "cli_args", "chat_template", "hf_repo",
        "vllm_quantization", "moe_backend", "mamba",
        "tool_call_parser", "reasoning_parser",
    })
    #: One engine, three launchers.
    transports = frozenset({"host", "podman", "docker"})
    host_requires = frozenset({"vllm_bin"})
    container_requires = frozenset({"vllm_image"})
    # Overridden at bind time; host is the standalone default.
    transport: Transport = HostTransport()

    # Per-role serving task (mirrors llama-server's _ROLE_FLAGS symmetry).
    _ROLE_TASK = {
        "embeddings": ["--task", "embed"],
        "rerank": ["--task", "score"],
    }

    def _serve_flags(
        self,
        model: "Model",
        ctx_size: int,
        port: str,
        gpu_mem_util: str,
        cache_type: str = "q8_0",
        parallel: int = 1,
        map_path: Callable[[Path], str] | None = None,
        batch: int | None = None,
        model_ref: str | None = None,
        spec_override: dict | None = None,
    ) -> list[str]:
        flags = [
            "--model", model_ref,
            "--served-model-name", "${MODEL_ID}",
            "--host", "0.0.0.0", "--port", str(port),
            "--max-model-len", str(ctx_size),
            "--gpu-memory-utilization", str(gpu_mem_util),
        ]
        if parallel > 0:
            # Aligned with llama-server's --parallel: same sidecar/profile
            # key, same slot-count meaning on every backend.  parallel <= 0
            # is uncapped: omit the admission limit entirely and let vLLM
            # fill its paged KV pool elastically (it queues, never OOMs).
            flags += ["--max-num-seqs", str(parallel)]
        if batch:
            # Chunked-prefill batch: tokens per scheduler step (the -b
            # analog).  ubatch has no vLLM equivalent — handled internally.
            flags += ["--max-num-batched-tokens", str(batch)]
        # Model-config recipe keys: emitted only when declared (opt-in,
        # verbatim — vLLM auto-detects most of this from the checkpoint).
        vq = model.vllm_quantization
        if vq:
            flags += ["--quantization", vq]
        mb = model.moe_backend
        if mb:
            flags += ["--moe-backend", mb]
        flags += _mamba_flags(model)
        flags += _kv_cache_dtype_flags(cache_type)
        flags += self._ROLE_TASK.get(model.role, [])
        ct = model.resolved_chat_template
        if ct is not None:
            ref = map_path(ct) if map_path else str(ct)
            flags += ["--chat-template", ref]
        if model.role == "chat":
            # Speculative decoding is a generation-only feature.  A container
            # transport passes its rewritten copy (container refs); host
            # derives from frontmatter.
            spec = spec_override if spec_override is not None \
                else _speculative_config(model)
            if spec:
                flags += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
            # Generation features: tool calling and reasoning traces are
            # parsed per model recipe.  Distinct from llama.cpp's
            # --reasoning-format (Model.reasoning_format).
            tcp = model.tool_call_parser
            if tcp:
                flags += ["--enable-auto-tool-choice", "--tool-call-parser", tcp]
            rp = model.reasoning_parser
            if rp:
                flags += ["--reasoning-parser", rp]
        return flags

    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
        batch: int | None = None,
        ubatch: int | None = None,
    ) -> tuple[str, dict]:
        transport = self.transport
        is_container = transport.container
        gpu_mem_util = str(tvars.get("gpu_mem_util", VLLM_DEFAULT_GPU_MEM_UTIL))
        vllm_bin = tvars.get("vllm_bin", VLLM_DEFAULT_BIN)
        container_port = int(tvars.get("container_port", VLLM_DEFAULT_CONTAINER_PORT))

        model_ref = model.hf_repo or (
            str(model.gguf_path) if model.gguf_path is not None else None)
        # Invariant: vLLM only serves models with a repo id or a local file.
        assert model_ref is not None
        spec = _speculative_config(model) if model.role == "chat" else None
        if spec is not None:
            spec = dict(spec)  # rewrite our copy — never mutate the frontmatter

        # Every path-shaped value inside cmd must be a launch-side path.  A
        # repo-id ref stays verbatim (resolved offline through the mounted
        # hub); the local file is mapped only when it IS the ref.
        extra_paths: list[Path] = []
        ct = model.resolved_chat_template
        if ct is not None:
            extra_paths.append(ct)
        if model.hf_repo is None and model.gguf_path is not None:
            extra_paths.append(model.gguf_path)
        spec_path_keys: list[str] = []
        if spec:
            for key in ("model", "draft_model"):
                v = spec.get(key)
                if isinstance(v, str) and v and Path(v).is_absolute() and Path(v).exists():
                    extra_paths.append(Path(v))
                    spec_path_keys.append(key)

        paths = transport.path_map(tvars, extra_paths)
        ref = paths.ref

        if is_container and model.hf_repo and not str(tvars.get("hf_cache") or ""):
            logger.warning("vllm-docker: %s: hf_cache is not configured and the "
                           "model ref is a repo id — with HF_HUB_OFFLINE=1 the "
                           "entry cannot resolve (set profiles.yaml vllm.hf_cache)",
                           model.stem)
        if is_container and spec:
            for key in spec_path_keys:
                spec[key] = ref(Path(spec[key]))

        resolved_ref = ref(Path(model_ref)) if model_ref is not None else None
        serve_flags = self._serve_flags(
            model, ctx_size, str(container_port) if is_container else "${PORT}",
            gpu_mem_util, cache_type=cache_type, parallel=parallel,
            map_path=ref, batch=batch,
            model_ref=resolved_ref if is_container else model_ref,
            spec_override=spec,
        )
        serve = utils.render_command(
            [vllm_bin, "serve"], serve_flags,
            global_args=_strip_llama_only_flags(tvars.get("vllm_args") or ""),
            cli_args=_strip_llama_only_flags(
                (model.frontmatter.get("cli_args") or "").strip()),
        )
        if not is_container:
            return serve, _spec_meta(model)

        # Container: hand the engine-specific bits to the transport, which
        # owns the `docker run` / `podman run` wrapper, mounts and lifecycle.
        image = model.vllm_image or tvars.get("vllm_image", VLLM_DEFAULT_IMAGE)
        launch = Launch(
            image=image,
            port=container_port,
            args=_container_args(tvars, transport.name),
            env=(f"HF_HOME={CONTAINER_HF_HOME}", "HF_HUB_OFFLINE=1"),
        )
        cmd = transport.wrap(serve, model, launch, paths)
        return cmd, _spec_meta(model)
