# llama_packer/backends/vllm.py
"""vLLM backends: host-binary and containerized serving.

Both serve safetensors / HF-repo models.  Roles map onto vLLM's serving
tasks: chat (generation), embeddings (``--task embed``) and rerank
(``--task score``, exposing /v1/rerank and /v1/score).  Speculative
decoding applies to generation only.  LoRA is not yet wired into vLLM's
module registry, so a declared ``loras`` setting is warned about and skipped.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from llama_packer import utils
from llama_packer.consts import (
    _MTP_DRAFT_N_MAX,
    VLLM_DEFAULT_GPU_MEM_UTIL,
    VLLM_DEFAULT_BIN,
    VLLM_DEFAULT_CONTAINER_PORT,
    VLLM_DEFAULT_DOCKER_ARGS,
    VLLM_DEFAULT_IMAGE,
)
from llama_packer.backends.base import BaseBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# In-container HF cache root (the images' default HOME cache).  The profiles
# ``vllm.hf_cache`` (host HF_HOME root — the dir containing ``hub/``) is
# bind-mounted here; HF_HUB_OFFLINE keeps every lookup inside that mounted hub
# — llama-packer never downloads.
_CONTAINER_HF_HOME = "/root/.cache/huggingface"

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


def _map_paths_into(paths: list[Path], models_dirs,
                    hf_cache: str | None = None) -> tuple[list[str], list[str]]:
    """Map host paths for use inside the vLLM container.

    Resolution order per path (``Path.resolve()`` first, so symlinked layouts
    map by their *real* location — an HF snapshot symlinked under ``~/models``
    maps into the HF cache branch):

    1. under ``hf_cache`` (host HF_HOME root — the dir containing ``hub/``) →
       ``/root/.cache/huggingface/<rel>``;
       the whole root is bind-mounted separately, so HF snapshot blob symlinks
       (``file → ../../blobs/<hash>``) resolve
    2. under any of ``models_dirs`` → that dir's container target (``/models``,
       ``/models2``, ...)
    3. else → dedicated read-only parent bind (``-v <parent>:/extN``) and an
       ``/extN/<name>`` ref

    Returns ``(container_refs, docker_mount_flags)``.  The hf_cache *mount* is
    not part of the returned mounts — emit ``-v <hf_cache>:/root/.cache/huggingface``
    once in build_cmd when hf_cache is configured.
    """
    if isinstance(models_dirs, (str, Path)):
        models_dirs = [str(models_dirs)]
    hf_root = Path(hf_cache).resolve() if hf_cache else None
    roots = [
        (Path(d).resolve(), "/models" if i == 0 else f"/models{i + 1}")
        for i, d in enumerate(models_dirs) if d
    ]
    refs: list[str] = []
    mounts: list[str] = []
    parent_targets: dict[Path, str] = {}
    for p in sorted(paths, key=str):
        rp = p.resolve()
        mapped = False
        if hf_root is not None:
            try:
                rel = rp.relative_to(hf_root)
            except ValueError:
                pass
            else:
                refs.append(f"{_CONTAINER_HF_HOME}/{rel}")
                mapped = True
        if not mapped:
            for root, target in roots:
                try:
                    rel = rp.relative_to(root)
                except ValueError:
                    continue
                refs.append(f"{target}/{rel}")
                mapped = True
                break
        if mapped:
            continue
        parent = rp.parent
        if parent not in parent_targets:
            idx = len(parent_targets)
            parent_targets[parent] = f"/ext{idx}"
            mounts.append(f"-v {parent}:{parent_targets[parent]}")
        refs.append(f"{parent_targets[parent]}/{rp.name}")
    return refs, mounts


class VllmHostBackend(BaseBackend):
    name = "vllm"
    formats = frozenset({".safetensors", "hf_repo"})
    roles = frozenset({"chat", "embeddings", "rerank"})
    handles = frozenset({
        "cli_args", "chat_template", "hf_repo",
        "vllm_quantization", "moe_backend", "mamba",
        "tool_call_parser", "reasoning_parser",
    })

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("vllm_bin"))

    def _model_ref(self, model: "Model") -> str:
        return model.hf_repo or str(model.gguf_path)

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
            "--model", model_ref or self._model_ref(model),
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
            # Speculative decoding is a generation-only feature.  Docker passes
            # its rewritten copy (container refs); host derives from frontmatter.
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
        gpu_mem_util = tvars.get("gpu_mem_util", VLLM_DEFAULT_GPU_MEM_UTIL)
        vllm_bin = tvars.get("vllm_bin", VLLM_DEFAULT_BIN)
        flags = self._serve_flags(model, ctx_size, "${PORT}", str(gpu_mem_util),
                                  cache_type=cache_type, parallel=parallel,
                                  batch=batch)
        cmd = utils.render_command(
            [vllm_bin, "serve"], flags,
            global_args=_strip_llama_only_flags(tvars.get("vllm_args") or ""),
            cli_args=_strip_llama_only_flags(
                (model.frontmatter.get("cli_args") or "").strip()),
        )
        return cmd, _spec_meta(model)


class VllmDockerBackend(VllmHostBackend):
    name = "vllm-docker"
    # Container lifecycle (see BaseBackend): llama-swap stops the container
    # itself on swap/unload — without cmdStop it can only kill the docker run
    # client, leaving the container running with its VRAM held.  unloadTimeout
    # must exceed the stop grace (docker stop is slow).
    stop_cmd = "docker stop ${MODEL_ID}"
    unload_timeout = 30

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("vllm_image"))

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
        # Per-model sidecar `vllm_image:` overrides the global default.
        gpu_mem_util = tvars.get("gpu_mem_util", VLLM_DEFAULT_GPU_MEM_UTIL)
        container_port = tvars.get("container_port", VLLM_DEFAULT_CONTAINER_PORT)
        docker_args = tvars.get("docker_args", VLLM_DEFAULT_DOCKER_ARGS)
        image = model.vllm_image or tvars.get("vllm_image", VLLM_DEFAULT_IMAGE)
        models_dirs = tvars.get("models_dirs") or [tvars.get("models_dir", "")]
        models_dirs = [d for d in models_dirs if d]
        hf_cache = str(tvars.get("hf_cache") or "")

        # Every path-shaped value inside cmd must be a container path.  Model
        # refs that are repo ids stay verbatim (resolved offline through the
        # mounted hub); the local file is mapped only when it IS the ref (a
        # repo id makes the local file irrelevant — mapping it would add a
        # pointless bind).
        model_ref = model.hf_repo or None
        if model_ref is None and model.gguf_path is not None:
            model_ref = str(model.gguf_path)
        spec = _speculative_config(model) if model.role == "chat" else None
        if spec is not None:
            spec = dict(spec)  # rewrite our copy — never mutate the frontmatter
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

        refs, ext_mounts = _map_paths_into(extra_paths, models_dirs, hf_cache or None)

        def ref_for(path: Path) -> str:
            """Container ref of the Nth mapped path (paths are sorted by str)."""
            for p, ref in zip(sorted(extra_paths, key=str), refs):
                if p == path:
                    return ref
            return str(path)

        container_model_ref = None
        if model_ref is not None:
            container_model_ref = ref_for(Path(model_ref))
        if model.hf_repo and not hf_cache:
            logger.warning("vllm-docker: %s: hf_cache is not configured and the "
                           "model ref is a repo id — with HF_HUB_OFFLINE=1 the "
                           "entry cannot resolve (set profiles.yaml vllm.hf_cache)",
                           model.stem)
        if spec:
            for key in spec_path_keys:
                spec[key] = ref_for(Path(spec[key]))

        def _map(p: Path) -> str:
            return ref_for(p)

        vllm_bin = tvars.get("vllm_bin", VLLM_DEFAULT_BIN)
        serve_flags = self._serve_flags(
            model, ctx_size, str(container_port), gpu_mem_util,
            cache_type=cache_type, parallel=parallel, map_path=_map,
            batch=batch, model_ref=container_model_ref, spec_override=spec,
        )
        serve = utils.render_command(
            [vllm_bin, "serve"], serve_flags,
            global_args=_strip_llama_only_flags(tvars.get("vllm_args") or ""),
            cli_args=_strip_llama_only_flags(
                (model.frontmatter.get("cli_args") or "").strip()),
        )
        bind = [
            f"-v {d}:{'/models' if i == 0 else f'/models{i + 1}'}"
            for i, d in enumerate(models_dirs)
        ]
        docker_parts = [
            "docker run --init --rm",
            docker_args,
            "--name ${MODEL_ID}",
            *bind,
            # Shared HF hub: repo-id refs resolve from the mounted cache and
            # HF_HUB_OFFLINE forbids downloads (llama-packer never fetches).
            *( [f"-v {hf_cache}:{_CONTAINER_HF_HOME}"] if hf_cache else [] ),
            f"-e HF_HOME={_CONTAINER_HF_HOME}",
            "-e HF_HUB_OFFLINE=1",
            *ext_mounts,
            f"-p ${{PORT}}:{container_port}",
            image,
        ]
        cmd = " ".join(docker_parts) + " " + serve
        return cmd, _spec_meta(model)
