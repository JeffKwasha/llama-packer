# llama_packer/backends/llama_server.py
"""llama-server backend: GGUF chat / embeddings / rerank serving."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from llama_packer.backends.base import BaseBackend
from llama_packer import utils

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)


class LlamaServerBackend(BaseBackend):
    name = "llama-server"
    formats = frozenset({".gguf"})
    roles = frozenset({"chat", "embeddings", "rerank"})
    handles = frozenset({"cli_args", "chat_template", "loras"})

    # Per-role server-mode flags, appended after the shared core arguments.
    _ROLE_FLAGS = {
        "embeddings": "--embedding --embd-normalize 2 -b 4096 -ub 4096",
        "rerank": "--rerank --pooling rank -b 4096 -ub 4096",
    }

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("llama_bin"))

    def _mtp_args(self, model: "Model") -> tuple[list[str], dict]:
        """Speculative-decoding flags plus metadata contributions."""
        mtp_on, n_max = model._mtp_info()
        if not mtp_on:
            return [], {"mtp_enabled": False}
        spec_type = model.frontmatter.get("mtp_spec_type", utils._MTP_SPEC_TYPE)
        args = ["--spec-type", spec_type, "--spec-draft-n-max", str(n_max)]
        if model.mtp and model.mtp.gguf_path:
            args += ["--spec-draft-model", str(model.mtp.gguf_path)]
        elif model.frontmatter.get("speculative"):
            logger.warning("mtp: companion %s missing for %s",
                           model.frontmatter["speculative"], model.stem)
            return [], {"mtp_enabled": False}
        return args, {"mtp_enabled": True, "mtp_draft_max": n_max}

    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
    ) -> tuple[str, dict]:
        parts = [
            tvars.get("llama_bin", ""),
            "--port ${PORT}",
            "-m", str(model.gguf_path),
            "-c", str(ctx_size),
            "--parallel", str(parallel),
            "--cache-type-k", cache_type,
            "--cache-type-v", cache_type,
        ]
        n_gpu_layers = "--n-gpu-layers 0" if model.on_cpu else "--n-gpu-layers 999"
        parts.append(n_gpu_layers)

        role_flags = self._ROLE_FLAGS.get(model.role)
        if role_flags:
            parts.append(role_flags)

        mtp_args, meta = self._mtp_args(model)
        parts += mtp_args

        if include_mmproj and model.mmproj and model.mmproj.gguf_path:
            parts += ["--mmproj", str(model.mmproj.gguf_path)]

        ct = model.resolved_chat_template
        if ct is not None:
            parts += ["--jinja", "--chat-template-file", str(ct)]

        for lora in model.resolved_loras:
            parts += ["--lora", str(lora)]

        cli_args = (model.frontmatter.get("cli_args") or "").strip()
        if cli_args:
            parts.append(cli_args)

        return " ".join(p for p in parts if p), meta
