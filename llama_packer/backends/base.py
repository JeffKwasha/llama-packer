# llama_packer/backends/base.py
"""Backend abstraction: launch-command composition per serving engine.

A *backend* turns a resolved ``Model`` (plus its effective settings and a
context size) into the llama-swap ``cmd`` string.  The class attributes below
form the support matrix consulted by ``build_config``:

    name      registry key — the sidecar / override ``backend:`` value
    formats   model file formats the engine can load: ``.gguf``,
              ``.safetensors`` and/or ``hf_repo`` (serving from an HF repo id)
    roles     model roles it can serve: chat / embeddings / rerank
    handles   SETTING_KEYS it renders into the command; anything a user
              declares that a backend does not handle is warned about instead
              of silently dropped
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# Settings keys that override rules / sidecars may declare.  Each backend
# declares which it renders via ``handles``.  ``FRAMEWORK_CONSUMED`` keys are
# consumed by the framework, not rendered per-backend: ``backend`` is the
# selection itself; ``hf_repo`` drives hub-cache resolution of the model file
# and companions (Model layer), so declaring it alongside any backend is fine.
# ``METADATA_ONLY`` keys are client-facing metadata only (no server flag).
SETTING_KEYS = frozenset({
    "backend", "hf_repo", "chat_template", "chat_template_kwargs",
    "loras", "cli_args", "reasoning-format", "reasoning-preserve",
})
FRAMEWORK_CONSUMED = frozenset({"backend", "hf_repo"})
METADATA_ONLY = frozenset({"chat_template_kwargs"})


class BaseBackend(ABC):
    """A serving engine that renders a Model into a llama-swap ``cmd``."""

    name: ClassVar[str]
    formats: ClassVar[frozenset[str]]
    roles: ClassVar[frozenset[str]]
    handles: ClassVar[frozenset[str]]
    # True when the server is a proxied HTTP service (llama-swap needs the
    # `proxy:` + `checkEndpoint:` fields instead of managing inference).
    proxied: ClassVar[bool] = False

    def unsupported_reason(self, model: "Model") -> str | None:
        """Return why this backend cannot serve *model*, or None if it can."""
        if model.role not in self.roles:
            return (f"role {model.role!r} not supported "
                    f"(supports: {', '.join(sorted(self.roles))})")
        if "hf_repo" in self.formats and model.hf_repo:
            return None
        suffix = model.gguf_path.suffix.lower() if model.gguf_path else None
        if suffix not in self.formats:
            got = (suffix
                   or ("no local model file" if not model.hf_repo
                       else model.hf_repo))
            return (f"format {got!r} not supported "
                    f"(supports: {', '.join(sorted(self.formats))})")
        return None

    def warn_unhandled(self, declared: set[str]) -> None:
        """Warn about declared settings this backend does not render."""
        for key in sorted(declared - self.handles):
            logger.warning("backend %s cannot handle setting %r (ignored)",
                           self.name, key)

    def is_available(self, avail: dict) -> bool:
        """True when the resources this backend needs to launch are configured.

        ``avail`` maps resource names to their configured values (e.g.
        ``llama_bin``, ``vllm_image``, ``vllm_bin``).  Backends override this
        to gate format-based inference: a format is only auto-assigned to a
        backend that can actually run with the current configuration.
        """
        return True

    @abstractmethod
    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
    ) -> tuple[str, dict]:
        """Compose the launch command.

        Returns ``(cmd, metadata_contributions)``.  ``metadata_contributions``
        is merged into the entry's ``metadata`` block by the writer.
        """
