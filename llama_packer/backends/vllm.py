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


def _map_paths_into(paths: list[Path], models_dirs) -> tuple[list[str], list[str]]:
    """Map host paths for use inside the vLLM container.

    Paths under any of ``models_dirs`` are rewritten under that dir's container
    target (``/models``, ``/models2``, ...).  Other paths each get a dedicated
    read-only bind mount (``-v <parent>:/extN``) and an ``/extN/<name>`` ref.

    Returns ``(container_refs, docker_mount_flags)``.
    """
    if isinstance(models_dirs, (str, Path)):
        models_dirs = [str(models_dirs)]
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
    handles = frozenset({"cli_args", "chat_template", "hf_repo"})

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
    ) -> list[str]:
        flags = [
            "--model", self._model_ref(model),
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
        flags += _kv_cache_dtype_flags(cache_type)
        flags += self._ROLE_TASK.get(model.role, [])
        ct = model.resolved_chat_template
        if ct is not None:
            ref = map_path(ct) if map_path else str(ct)
            flags += ["--chat-template", ref]
        if model.role == "chat":
            # Speculative decoding is a generation-only feature.
            spec = _speculative_config(model)
            if spec:
                flags += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
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

        extra_paths: list[Path] = []
        ct = model.resolved_chat_template
        if ct is not None:
            extra_paths.append(ct)

        refs, mounts = _map_paths_into(extra_paths, models_dirs)
        container_ct_ref = refs[0] if (ct is not None and refs) else None

        def _map(p: Path) -> str:
            return container_ct_ref if (ct is not None and p == ct) else str(p)

        vllm_bin = tvars.get("vllm_bin", VLLM_DEFAULT_BIN)
        serve_flags = self._serve_flags(
            model, ctx_size, str(container_port), gpu_mem_util,
            cache_type=cache_type, parallel=parallel, map_path=_map,
            batch=batch
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
            *mounts,
            f"-p ${{PORT}}:{container_port}",
            image,
        ]
        cmd = " ".join(docker_parts) + " " + serve
        return cmd, _spec_meta(model)
