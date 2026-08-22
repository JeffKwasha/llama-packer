# llama_packer/backends/__init__.py
"""Backend registry and shared constants.

A *backend* is a serving engine that renders a resolved ``Model`` into a
llama-swap ``cmd`` string (see ``backends/base.py``).  Add a backend by
creating ``backends/<name>.py`` with a ``BaseBackend`` subclass and registering
it in ``BACKENDS`` below — no other file needs to change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llama_packer.backends.base import (
    FRAMEWORK_CONSUMED,
    METADATA_ONLY,
    SETTING_KEYS,
    BaseBackend,
)
from llama_packer.backends.llama_server import LlamaServerBackend
from llama_packer.backends.vllm import VllmDockerBackend, VllmHostBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

# Backend instances keyed by their ``name`` (the value of a sidecar /
# override ``backend:`` field).  Registration order is also the preference
# order for format-based backend inference (see ``infer_backend``): for
# safetensors/HF-repo models the containerized vLLM is preferred over the
# host binary when both are available.
BACKENDS: dict[str, BaseBackend] = {
    b.name: b
    for b in (LlamaServerBackend(), VllmDockerBackend(), VllmHostBackend())
}

# Backends that serve from an HF repo / safetensors rather than a local GGUF.
VLLM_BACKENDS = frozenset({"vllm", "vllm-docker"})

# Fallback backend when nothing is declared and inference cannot run
# (e.g. a bare Model constructed outside the normal pipeline).
DEFAULT_BACKEND = "llama-server"


def get_backend(name: str) -> BaseBackend:
    """Return the backend instance for *name* (raises KeyError if unknown)."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r} (available: {', '.join(sorted(BACKENDS))})"
        ) from None


def infer_backend(model: "Model", avail: dict | None = None) -> str | None:
    """Infer a backend name from the model's file format and configuration.

    Backends register the formats they can load (``formats``); the registry
    walks them in preference order and returns the first whose format covers
    the model AND whose required resources are configured (``is_available``).
    A locally resolved model file's extension wins over ``hf_repo``.

    Returns None when no registered backend can serve the model with the
    current configuration — the caller logs and skips the model.
    """
    if model.gguf_path is not None:
        fmt = model.gguf_path.suffix.lower()
    elif model.hf_repo:
        fmt = "hf_repo"
    else:
        return None
    avail = avail or {}
    for backend in BACKENDS.values():
        if fmt in backend.formats and backend.is_available(avail):
            return backend.name
    return None


__all__ = [
    "BACKENDS",
    "VLLM_BACKENDS",
    "DEFAULT_BACKEND",
    "SETTING_KEYS",
    "FRAMEWORK_CONSUMED",
    "METADATA_ONLY",
    "BaseBackend",
    "get_backend",
    "infer_backend",
]
