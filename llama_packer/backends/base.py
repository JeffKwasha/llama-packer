# llama_packer/backends/base.py
"""Backend abstraction: launch-command composition per serving engine.

A *backend* turns a resolved ``Model`` (plus its effective settings and a
context size) into the llama-swap ``cmd`` string.  The class attributes below
form the support matrix consulted by ``build_config``:

    name      registry key — the sidecar / override ``backend:`` value
    formats   model file formats the engine can load: ``.gguf``,
              ``.safetensors`` and/or ``hf_repo`` (legacy: serving directly
              from an HF repo id when no local file is resolved; ``hf_repo``
              itself is *not* a file – it is the *place* where a file lives,
              resolved by ``Model._resolve_gguf_path`` to a concrete snapshot
              file, ``gguf_path``)
    roles     model roles it can serve: chat / embeddings / rerank
    handles   SETTING_KEYS it renders into the command; anything a user
              declares that a backend does not handle is warned about instead
              of silently dropped
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from llama_packer.backends.transport import HostTransport, Transport

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
    # First-class batch keys (llama-server renders them explicitly; other
    # backends warn the declaration as unhandled).
    "batch", "ubatch",
    # vLLM recipe keys (rendered by the vllm / vllm-docker backends)
    "vllm_quantization", "moe_backend", "mamba", "tool_call_parser",
    "reasoning_parser",
})
FRAMEWORK_CONSUMED = frozenset({"backend", "hf_repo"})
METADATA_ONLY = frozenset({"chat_template_kwargs"})


class BaseBackend(ABC):
    """A serving engine that renders a Model into a llama-swap ``cmd``."""

    name: str
    formats: frozenset[str]
    roles: frozenset[str]
    handles: frozenset[str]
    #: Transports this engine can run under (see ``transport.py``).  The
    #: registry materialises one bound backend per supported pair.
    transports: ClassVar[frozenset[str]] = frozenset({"host"})
    #: ``avail`` keys the engine needs to launch, split by transport kind.
    host_requires: ClassVar[frozenset[str]] = frozenset()
    container_requires: ClassVar[frozenset[str]] = frozenset()
    #: The launcher this engine is bound to (set by the registry binding;
    #: host is the standalone default so engines remain directly testable).
    transport: Transport = HostTransport()
    # True when the server is a proxied HTTP service (llama-swap needs the
    # `proxy:` + `checkEndpoint:` fields instead of managing inference).
    proxied: bool = False
    # Health path llama-swap polls for a proxied server.  "/" suits sd-server;
    # audio.cpp exposes /health; whisper-server accepts "/".
    check_endpoint: str = "/"
    # Container lifecycle (llama-swap docker orchestration, docs/kb
    # guides/model-runtime/ttl-and-unloading.md): `cmdStop` stops the container
    # itself — without it llama-swap can only stop the `docker run` client
    # process, leaving the container running and its VRAM held.  `unloadTimeout`
    # must exceed the stop grace (docker stop is slow).  Only container backends
    # set these; None keeps llama-server entries free of both fields.
    stop_cmd: str | None = None
    unload_timeout: int | None = None

    def unsupported_reason(self, model: "Model") -> str | None:
        """Return why this backend cannot serve *model*, or None if it can."""
        if model.role not in self.roles:
            return (f"role {model.role!r} not supported "
                    f"(supports: {', '.join(sorted(self.roles))})")
        if "hf_repo" in self.formats and model.hf_repo:
            return None
        suffix = model.gguf_path.suffix.lower() if model.gguf_path else None
        if suffix not in self.formats:
            got = suffix or "no file"
            if model.gguf_path:
                got = model.gguf_path.name
            elif model.hf_repo:
                got = f"hf_repo {model.hf_repo!r} (no local file resolved)"
            return (f"format {got!r} not supported "
                    f"(supports: {', '.join(sorted(self.formats))})")
        return None

    def warn_unhandled(self, declared: set[str]) -> None:
        """Warn about declared settings this backend does not render."""
        for key in sorted(declared - self.handles):
            logger.warning("backend %s cannot handle setting %r (ignored)",
                           self.name, key)

    def supports(self, avail: dict, transport: Transport) -> bool:
        """Whether this engine can launch under *transport* with *avail*."""
        required = (self.container_requires if transport.container
                    else self.host_requires)
        return all(avail.get(key) for key in required)

    def default_batch_ubatch(self, role: str) -> tuple[int, int]:
        """``(batch, ubatch)`` defaults for *role* when nothing is
        configured — the llama.cpp builtins.  llama-server overrides per
        role (embed/rerank keep their tuned batch depth)."""
        return (2048, 512)

    @abstractmethod
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
        """Compose the launch command.

        ``batch``/``ubatch`` are the resolved first-class batch keys
        (sidecar > profile > fleet > role); llama-server renders them
        explicitly (and stamps ``ubatch`` into the VRAM measurement shape);
        other backends ignore them.

        Returns ``(cmd, metadata_contributions)``.  ``metadata_contributions``
        is merged into the entry's ``metadata`` block by the writer.
        """
