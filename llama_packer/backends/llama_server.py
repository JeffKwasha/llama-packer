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

# Architectures with a static-resolution vision encoder (fixed resize to a
# single tile). llama-server ignores --image-min/max-tokens for these, so
# declaring them in a sidecar is flagged instead of silently doing nothing.
_STATIC_IMAGE_ARCHES = frozenset({"gemma3", "gemma4"})
_warned_static_arch: set[str] = set()


class LlamaServerBackend(BaseBackend):
    name = "llama-server"
    formats = frozenset({".gguf"})
    roles = frozenset({"chat", "embeddings", "rerank"})
    handles = frozenset({
        "cli_args", "chat_template", "loras",
        "reasoning-format", "reasoning-preserve",
    })

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

    def _image_token_args(self, model: "Model", include_mmproj: bool) -> list[str]:
        """--image-min/max-tokens flags from sidecar declarations.

        Only emitted when the vision projection is actually served: the flags
        are meaningless for a text-only variant. Declared on a model without
        an mmproj — or on a static-resolution arch (Gemma/SigLIP, fixed ~256
        tokens/image) — is warned about and skipped.
        """
        imin, imax = model.image_min_tokens, model.image_max_tokens
        if imin is None and imax is None:
            return []
        if not (model.mmproj and model.mmproj.gguf_path):
            logger.warning(
                "image tokens: %s declares image_min/max_tokens but has no "
                "mmproj companion; flags skipped", model.stem)
            return []
        if not include_mmproj:
            return []  # text-only variant: vision not served, silently skip
        arch = None
        if model.gguf_path:
            arch, _ = utils.gguf_header_probe(model.gguf_path)
        if arch in _STATIC_IMAGE_ARCHES:
            msg = (f"image tokens: {model.stem}: arch {arch!r} has a "
                   f"static-resolution vision encoder (fixed ~256 tokens per "
                   f"image); --image-min/max-tokens are no-ops and skipped")
            if msg not in _warned_static_arch:
                _warned_static_arch.add(msg)
                logger.warning(msg)
            return []
        args: list[str] = []
        if imin is not None:
            args += ["--image-min-tokens", str(imin)]
        if imax is not None:
            args += ["--image-max-tokens", str(imax)]
        return args

    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
    ) -> tuple[str, dict]:
        flags = [
            "--port", "${PORT}",
            "-m", str(model.gguf_path),
            "-c", str(ctx_size),
            "--parallel", str(parallel),
            "--cache-type-k", cache_type,
            "--cache-type-v", cache_type,
            "--n-gpu-layers", ("0" if model.on_cpu else "999"),
        ]

        mtp_args, meta = self._mtp_args(model)
        flags += mtp_args

        if include_mmproj and model.mmproj and model.mmproj.gguf_path:
            flags += ["--mmproj", str(model.mmproj.gguf_path)]
        flags += self._image_token_args(model, include_mmproj)

        ct = model.resolved_chat_template
        if ct is not None:
            flags += ["--jinja", "--chat-template-file", str(ct)]

        loras = model.resolved_loras
        if loras:
            # llama-server accepts a single --lora with comma-separated adapters.
            flags += ["--lora", ",".join(str(l) for l in loras)]

        # Reasoning flags are only meaningful for chat models.
        if model.role == "chat":
            rf = model.reasoning_format
            if rf is not None:
                flags += ["--reasoning-format", rf]
            if model.reasoning_preserve:
                flags += ["--reasoning-preserve"]

        cmd = utils.render_command(
            [tvars.get("llama_bin", "")], flags,
            global_args=tvars.get("llama_args") or "",
            role_flags=self._ROLE_FLAGS.get(model.role, ""),
            cli_args=(model.frontmatter.get("cli_args") or "").strip(),
        )
        return cmd, meta
