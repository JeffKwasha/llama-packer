# llama_packer/backends/whisper_server.py
"""whisper-server backend: whisper.cpp speech-to-text serving.

Runs the whisper.cpp HTTP server (examples/server — the long-lived analog of
llama-server; `whisper-cli` is the oneshot transcriber and cannot be proxied
by llama-swap).  Models are GGML ``.bin`` files discovered only under an
``s2t``-mapped directory (no header fingerprint exists for GGML, so the
directory is authoritative).  Exposes OpenAI-compatible
``POST /v1/audio/transcriptions``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llama_packer.backends.base import BaseBackend
from llama_packer import utils

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)


class WhisperServerBackend(BaseBackend):
    name = "whisper-server"
    formats = frozenset({".bin"})
    roles = frozenset({"s2t"})
    handles = frozenset({"cli_args"})  # --language, --threads, etc.
    proxied = True

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("whisper_bin"))

    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
    ) -> tuple[str, dict]:
        assert model.gguf_path is not None
        flags = [
            "--host", "0.0.0.0",
            "--port", "${PORT}",
            "--model", str(model.gguf_path),
            # Convert (parallel slots) → concurrent transcription workers.
            "--parallel", str(parallel),
        ]
        cmd = utils.render_command(
            [tvars.get("whisper_bin", "whisper-server")], flags,
            global_args=tvars.get("whisper_args") or "",
            cli_args=(model.frontmatter.get("cli_args") or "").strip(),
        )
        return cmd, {}
