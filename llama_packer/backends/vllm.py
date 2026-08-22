# llama_packer/backends/vllm.py
"""vLLM backends: host-binary and containerized serving.

Both serve safetensors / HF-repo models.  vLLM renders chat models only in
this version; embeddings/rerank selection for vLLM is left for a later
release, so the support matrix rejects those roles (the framework then logs
the error and skips that combo).  LoRA is not yet wired into vLLM's module
registry, so a declared ``loras`` setting is warned about and skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from llama_packer import utils
from llama_packer.backends.base import BaseBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)


def _map_paths_into(paths: list[Path], models_dir: str) -> tuple[list[str], list[str]]:
    """Map host paths for use inside the vLLM container.

    Paths under ``models_dir`` are rewritten under ``/models`` (which the
    docker invocation already binds).  Other paths each get a dedicated
    read-only bind mount (``-v <parent>:/extN``) and an ``/extN/<name>`` ref.

    Returns ``(container_refs, docker_mount_flags)``.
    """
    models_root = Path(models_dir).resolve()
    refs: list[str] = []
    mounts: list[str] = []
    parent_targets: dict[Path, str] = {}
    for p in sorted(paths, key=str):
        rp = p.resolve()
        try:
            rel = rp.relative_to(models_root)
            refs.append(f"/models/{rel}")
            continue
        except ValueError:
            pass
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

    def _serve_parts(
        self,
        model: "Model",
        ctx_size: int,
        port: str,
        gpu_mem_util: str,
        tvars: dict,
        map_path: Callable[[Path], str] | None = None,
    ) -> list[str]:
        parts = [
            tvars.get("vllm_bin", utils.VLLM_DEFAULT_BIN), "serve",
            "--model", self._model_ref(model),
            "--served-model-name", "${MODEL_ID}",
            "--host", "0.0.0.0", "--port", str(port),
            "--max-model-len", str(ctx_size),
            "--gpu-memory-utilization", str(gpu_mem_util),
        ]
        ct = model.resolved_chat_template
        if ct is not None:
            ref = map_path(ct) if map_path else str(ct)
            parts += ["--chat-template", ref]
        cli_args = (model.frontmatter.get("cli_args") or "").strip()
        if cli_args:
            parts.append(cli_args)
        return parts

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
        parts = self._serve_parts(model, ctx_size, "${PORT}", str(gpu_mem_util), tvars)
        return " ".join(parts), {"mtp_enabled": False}


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
        models_dir = tvars.get("models_dir", "")

        extra_paths: list[Path] = []
        ct = model.resolved_chat_template
        if ct is not None:
            extra_paths.append(ct)

        refs, mounts = _map_paths_into(extra_paths, models_dir)
        container_ct_ref = refs[0] if (ct is not None and refs) else None

        def _map(p: Path) -> str:
            return container_ct_ref if (ct is not None and p == ct) else str(p)

        serve_parts = self._serve_parts(
            model, ctx_size, str(container_port), gpu_mem_util, tvars, map_path=_map
        )
        docker_parts = [
            "docker run --init --rm",
            docker_args,
            "--name ${MODEL_ID}",
            f"-v {models_dir}:/models",
            *mounts,
            f"-p ${{PORT}}:{container_port}",
            image,
        ]
        cmd = " ".join(docker_parts) + " " + " ".join(serve_parts)
        return cmd, {"mtp_enabled": False}
