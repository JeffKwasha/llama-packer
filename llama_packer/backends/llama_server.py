# llama_packer/backends/llama_server.py
"""llama-server backend: GGUF chat / embeddings / rerank serving."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from llama_packer.backends.base import BaseBackend
from llama_packer import utils
from llama_packer.consts import _MTP_SPEC_TYPE

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# Architectures with a static-resolution vision encoder (fixed resize to a
# single tile). llama-server ignores --image-min/max-tokens for these, so
# declaring them in a sidecar is flagged instead of silently doing nothing.
_STATIC_IMAGE_ARCHES = frozenset({"gemma3", "gemma4"})
_warned_static_arch: set[str] = set()
_warned_cli_batch: set[str] = set()


class LlamaServerBackend(BaseBackend):
    name = "llama-server"
    formats = frozenset({".gguf"})
    roles = frozenset({"chat", "embeddings", "rerank"})
    handles = frozenset({
        "cli_args", "chat_template", "loras",
        "reasoning-format", "reasoning-preserve",
        "batch", "ubatch",
    })
    host_requires = frozenset({"llama_bin"})

    # Per-role server-mode flags, appended after the shared core arguments.
    # The batch half moved to first-class ``batch:``/``ubatch:`` planning
    # keys (:meth:`default_batch_ubatch` holds the role defaults): ``-ub``
    # sets the per-pass activation/logits buffer (~6 MiB per token on
    # Vulkan, dominated by the vocab-sized logits rows) — -ub 4096 cost
    # ~2.9 GiB of compute buffer per model where 512 (the llama.cpp default)
    # costs ~0.3 GiB with no measurable throughput difference (2026-09-08).
    _ROLE_FLAGS = {
        "embeddings": "--embedding --embd-normalize 2",
        "rerank": "--rerank --pooling rank",
    }

    # (batch, ubatch) per role — the llama.cpp builtins for chat, tuned
    # depth for embed/rerank.  Consulted by the Planner's resolution
    # cascade (sidecar > profile > fleet > role).
    _ROLE_BATCH = {
        "chat": (2048, 512),
        "embeddings": (4096, 512),
        "rerank": (4096, 512),
    }

    def default_batch_ubatch(self, role: str) -> tuple[int, int]:
        return self._ROLE_BATCH.get(role, (2048, 512))

    def _mtp_args(self, model: "Model") -> tuple[list[str], dict]:
        """Speculative-decoding flags plus metadata contributions."""
        mtp_on, n_max = model._mtp_info()
        if not mtp_on:
            return [], {"mtp_enabled": False}
        spec_type = model.frontmatter.get("mtp_spec_type", _MTP_SPEC_TYPE)
        p_min = model.mtp_draft_p_min
        args = ["--spec-type", spec_type, "--spec-draft-n-max", str(n_max),
                "--draft-p-min", str(p_min)]
        if model.mtp and model.mtp.gguf_path:
            args += ["--spec-draft-model", str(model.mtp.gguf_path)]
        elif model.frontmatter.get("speculative"):
            logger.warning("mtp: companion %s missing for %s",
                           model.frontmatter["speculative"], model.stem)
            return [], {"mtp_enabled": False}
        return args, {"mtp_enabled": True, "mtp_draft_max": n_max,
                      "mtp_draft_p_min": p_min}

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
        arch = model.arch if model.gguf_path else None
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
        batch: int | None = None,
        ubatch: int | None = None,
    ) -> tuple[str, dict]:
        # First-class batch keys render explicitly in the per-role slot
        # (named flags win over conflicting fleet args, lose to cli_args —
        # which is why cli_args -b/-ub is warned about).
        role_flags = self._ROLE_FLAGS.get(model.role, "")
        if batch is not None or ubatch is not None:
            default_batch, default_ubatch = self.default_batch_ubatch(model.role)
            role_flags += f" -b {batch or default_batch} -ub {ubatch or default_ubatch}"
        cli = (model.frontmatter.get("cli_args") or "").strip()
        if any(tok in ("-b", "-ub") for tok in cli.split()):
            msg = (f"{model.stem}: cli_args carries -b/-ub — shadowed by the "
                   f"rendered batch/ubatch and invisible to the VRAM "
                   f"measurement; use the batch:/ubatch: keys instead")
            if msg not in _warned_cli_batch:
                _warned_cli_batch.add(msg)
                logger.warning(msg)
        flags = [
            "--port", "${PORT}",
            "-m", str(model.gguf_path),
            # Per-slot context: the shared KV pool is sized to parallel*X,
            # byte-identical to a -c pool of parallel*X tokens (validated
            # against llama-fit-params).  ctx_size is what each slot serves
            # and what metadata advertises.
            "--kv-unified-per-slot", str(ctx_size),
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
            role_flags=role_flags,
            cli_args=cli,
        )
        return cmd, meta
