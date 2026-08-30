# llama_packer/backends/sd_server.py
"""sd-server backend: stable-diffusion.cpp image generation."""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING, ClassVar

from llama_packer.backends.base import BaseBackend
from llama_packer import utils

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)


class SdServerBackend(BaseBackend):
    name = "sd-server"
    formats = frozenset({".gguf", ".safetensors", "hf_repo"})
    roles = frozenset({"image"})
    handles = frozenset({"cli_args"})
    proxied = True

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("sd_bin"))

    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
    ) -> tuple[str, dict]:
        # Main diffusion model is the resolved gguf/safetensors.
        assert model.gguf_path is not None or model.hf_repo is not None
        diffusion = str(model.gguf_path) if model.gguf_path else str(model.hf_repo)

        flags: list[str] = [
            "--listen-port", "${PORT}",
            "--listen-ip", "0.0.0.0",
            "--diffusion-model", diffusion,
        ]

        # Global fleet-wide args (profiles.yaml `sd.args`) precede the
        # per-model cli_args; render_command dedups conflicting flags so
        # per-model values win.
        flags += shlex.split(tvars.get("sd_args") or "")

        cli_args = (model.frontmatter.get("cli_args") or "").strip()
        cmd = utils.render_command(
            [tvars.get("sd_bin", "sd-server")], flags, cli_args,
        )
        return cmd, {}
