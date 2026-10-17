# llama_packer/backends/__init__.py
"""Backend registry: engines × transports.

A *backend* (engine) is a serving engine that renders a resolved ``Model``
into a llama-swap ``cmd`` (see ``base.py``).  A *transport* is how that
process is launched — bare host process or a container runtime (see
``transport.py``).  The two are independent axes: each engine declares the
transports it supports (``BaseBackend.transports``) and this module
materialises one :class:`BoundBackend` per valid pair.

Registry names are the bound names: the bare engine name for the host
transport (``vllm``, ``llama-server``, …), suffixed for others
(``vllm-podman``, ``vllm-docker``).  Registration order is the inference
preference order: engines in declaration order, and within an engine the
``TRANSPORT_PREFERENCE`` (host > podman > docker).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_packer.backends.base import (
    FRAMEWORK_CONSUMED,
    METADATA_ONLY,
    SETTING_KEYS,
    BaseBackend,
)
from llama_packer.backends.audio_cpp import AudioCppBackend
from llama_packer.backends.llama_server import LlamaServerBackend
from llama_packer.backends.sd_server import SdServerBackend
from llama_packer.backends.transport import (
    TRANSPORT_PREFERENCE,
    ContainerTransport,
    HostTransport,
    Transport,
)
from llama_packer.backends.vllm import VllmBackend
from llama_packer.backends.whisper_server import WhisperServerBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

#: Engines in inference preference order.
ENGINES: tuple[type[BaseBackend], ...] = (
    LlamaServerBackend,
    VllmBackend,
    SdServerBackend,
    WhisperServerBackend,
    AudioCppBackend,
)

#: Transport instances by name.  Container runtimes share one implementation;
#: only the runtime binary/tag differs.
TRANSPORTS: dict[str, Transport] = {
    "host": HostTransport(),
    "podman": ContainerTransport("podman"),
    "docker": ContainerTransport("docker"),
}


class BoundBackend(BaseBackend):
    """An engine paired with a transport — one launchable backend.

    Delegates the engine's support matrix, validation and command building,
    and layers on the transport's launch wrapper, lifecycle fields and
    resource gating.  This is what ``get_backend()`` returns and what the
    writer sees.
    """

    def __init__(self, engine: BaseBackend, transport: Transport):
        self.engine = engine
        self.transport = transport
        engine.transport = transport
        self.name = transport.bind_name(engine.name)
        self.formats = engine.formats
        self.roles = engine.roles
        self.handles = engine.handles
        self.proxied = engine.proxied
        self.check_endpoint = engine.check_endpoint
        self.stop_cmd = transport.stop_cmd
        self.unload_timeout = transport.unload_timeout

    def unsupported_reason(self, model: "Model") -> str | None:
        return self.engine.unsupported_reason(model)

    def warn_unhandled(self, declared: set[str]) -> None:
        return self.engine.warn_unhandled(declared)

    def default_batch_ubatch(self, role: str) -> tuple[int, int]:
        return self.engine.default_batch_ubatch(role)

    def is_available(self, avail: dict) -> bool:
        return (self.engine.supports(avail, self.transport)
                and self.transport.is_available(avail))

    def build_cmd(self, *args, **kwargs) -> tuple[str, dict]:
        return self.engine.build_cmd(*args, **kwargs)


def _iter_bound_backends():
    """Yield one bound backend per valid (engine, transport) pair."""
    for engine_cls in ENGINES:
        for tname in TRANSPORT_PREFERENCE:
            if tname in engine_cls.transports:
                yield BoundBackend(engine_cls(), TRANSPORTS[tname])


BACKENDS: dict[str, BoundBackend] = {
    b.name: b for b in _iter_bound_backends()
}


def _names_for(engine_cls: type[BaseBackend]) -> frozenset[str]:
    """Bound names whose engine is *engine_cls* (any transport)."""
    return frozenset(n for n, b in BACKENDS.items()
                     if isinstance(b.engine, engine_cls))


# Backends that serve from an HF repo / safetensors rather than a local GGUF.
VLLM_BACKENDS = _names_for(VllmBackend)
SD_BACKENDS = _names_for(SdServerBackend)
WHISPER_BACKENDS = _names_for(WhisperServerBackend)
AUDIO_CPP_BACKENDS = _names_for(AudioCppBackend)

# Backends with fixed VRAM overhead (weights + buffer, no per-token KV factor):
# excluded from the shared chat matrix solve.
FIXED_OVERHEAD_BACKENDS = SD_BACKENDS | WHISPER_BACKENDS | AUDIO_CPP_BACKENDS

# Fallback backend when nothing is declared and inference cannot run
# (e.g. a bare Model constructed outside the normal pipeline).
DEFAULT_BACKEND = "llama-server"


def get_backend(name: str) -> BoundBackend:
    """Return the bound backend for *name* (raises KeyError if unknown)."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r} (available: {', '.join(sorted(BACKENDS))})"
        ) from None


def infer_backend(model: "Model", avail: dict | None = None,
                  allowed: list[str] | None = None) -> str | None:
    """Infer a backend name from the model's file format and configuration.

    Backends register the formats they can load (``formats``); the registry
    walks them in preference order and returns the first whose format covers
    the model AND whose resources are configured (``is_available``).
    A locally resolved model file's extension wins over ``hf_repo``.

    *allowed* (from profiles.yaml ``backends:``) both reorders inference and
    disables unlisted backends; None keeps every registered backend in
    registration order.  Entries are bound names (``vllm`` is the host pair).

    Returns None when no backend can serve the model with the current
    configuration — the caller logs and skips the model.
    """
    if model.gguf_path is not None:
        fmt = model.gguf_path.suffix.lower()
    elif model.hf_repo:
        fmt = "hf_repo"
    else:
        return None
    avail = avail or {}
    if allowed is None:
        candidates = list(BACKENDS.values())
    else:
        candidates = [BACKENDS[n] for n in allowed if n in BACKENDS]
    for backend in candidates:
        if fmt in backend.formats and backend.is_available(avail):
            # Role must be supported — otherwise a .gguf diffusion model under
            # role=image would incorrectly infer llama-server and then be skipped
            # as unsupported.  Infer the only backend that can actually serve it.
            if model.role not in backend.roles:
                continue
            return backend.name
    return None


def validate_backend_names(names) -> str | None:
    """Validate a profiles.yaml ``backends:`` list; error message or None."""
    for name in names:
        if name not in BACKENDS:
            return (f"backends: unknown backend {name!r} "
                    f"(available: {', '.join(sorted(BACKENDS))})")
    return None


__all__ = [
    "BACKENDS",
    "ENGINES",
    "TRANSPORTS",
    "VLLM_BACKENDS",
    "SD_BACKENDS",
    "WHISPER_BACKENDS",
    "AUDIO_CPP_BACKENDS",
    "FIXED_OVERHEAD_BACKENDS",
    "DEFAULT_BACKEND",
    "SETTING_KEYS",
    "FRAMEWORK_CONSUMED",
    "METADATA_ONLY",
    "BaseBackend",
    "BoundBackend",
    "Transport",
    "get_backend",
    "infer_backend",
    "validate_backend_names",
]
