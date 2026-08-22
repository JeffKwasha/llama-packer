# llama_packer/backends/vllm.py
"""vLLM backends: host-binary and containerized serving.

Both serve safetensors / HF-repo models.  vLLM renders chat models only in
this version; embeddings/rerank selection for vLLM is left for a later
release, so the support matrix rejects those roles (the framework then logs
the error and skips that combo).  LoRA is not yet wired into vLLM's module
registry, so a declared ``loras`` setting is warned about and skipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from llama_packer import utils
from llama_packer.backends.base import BaseBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)


def _speculative_config(model: "Model") -> dict | None:
    """The ``--speculative-config`` JSON dict for *model*, or None.

    Precedence:

    1. Explicit ``speculative_config:`` frontmatter (a raw mapping — full
       control over any vLLM method: eagle3, ngram, draft_model, ...).
    2. Baked-in MTP (``mtp: true``) → ``{"method": "mtp",
       "num_speculative_tokens": N}`` (vLLM docs recommend N=1 to start).

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
        n = int(fm.get("mtp_draft_n_max", utils.VLLM_DEFAULT_MTP_TOKENS))
        return {"method": "mtp", "num_speculative_tokens": n}
    if fm.get("speculative"):
        logger.warning("vllm: %s: GGUF speculative companion %r cannot be loaded "
                       "by vLLM; skipping speculative decoding (use "
                       "`speculative_config:` with a draft HF repo instead)",
                       model.stem, fm["speculative"])
    return None


def _spec_meta(model: "Model") -> dict:
    """Backend metadata for the writer's mtp_* metadata keys."""
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
    roles = frozenset({"chat"})
    handles = frozenset({"cli_args", "chat_template", "hf_repo"})

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("vllm_bin"))

    def _model_ref(self, model: "Model") -> str:
        return model.hf_repo or str(model.gguf_path)

    def _serve_flags(
        self,
        model: "Model",
        ctx_size: int,
        port: str,
        gpu_mem_util: str,
        map_path: Callable[[Path], str] | None = None,
    ) -> list[str]:
        flags = [
            "--model", self._model_ref(model),
            "--served-model-name", "${MODEL_ID}",
            "--host", "0.0.0.0", "--port", str(port),
            "--max-model-len", str(ctx_size),
            "--gpu-memory-utilization", str(gpu_mem_util),
        ]
        ct = model.resolved_chat_template
        if ct is not None:
            ref = map_path(ct) if map_path else str(ct)
            flags += ["--chat-template", ref]
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
    ) -> tuple[str, dict]:
        gpu_mem_util = tvars.get("gpu_mem_util", utils.VLLM_DEFAULT_GPU_MEM_UTIL)
        vllm_bin = tvars.get("vllm_bin", utils.VLLM_DEFAULT_BIN)
        flags = self._serve_flags(model, ctx_size, "${PORT}", str(gpu_mem_util))
        cli_args = (model.frontmatter.get("cli_args") or "").strip()
        cmd = utils.render_command([vllm_bin, "serve"], flags, cli_args)
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
    ) -> tuple[str, dict]:
        # Per-model sidecar `vllm_image:` overrides the global default.
        gpu_mem_util = tvars.get("gpu_mem_util", utils.VLLM_DEFAULT_GPU_MEM_UTIL)
        container_port = tvars.get("container_port", utils.VLLM_DEFAULT_CONTAINER_PORT)
        docker_args = tvars.get("docker_args", utils.VLLM_DEFAULT_DOCKER_ARGS)
        image = model.vllm_image or tvars.get("vllm_image", utils.VLLM_DEFAULT_IMAGE)
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

        vllm_bin = tvars.get("vllm_bin", utils.VLLM_DEFAULT_BIN)
        serve_flags = self._serve_flags(
            model, ctx_size, str(container_port), gpu_mem_util, map_path=_map
        )
        serve = utils.render_command(
            [vllm_bin, "serve"], serve_flags,
            (model.frontmatter.get("cli_args") or "").strip(),
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
