# llama_packer/backends/kokoro.py
"""kokoro-podman backend: Kokoro-82M text-to-speech in a rootless container.

Runs the OpenAI-compatible Kokoro-FastAPI server (remsky) via podman.
Weights and ~50 voicepacks are baked into the upstream image, so a t2s
sidecar typically carries only ``hf_repo: hexgrad/Kokoro-82M`` for identity
— no local model file is required.  Exposes ``POST /v1/audio/speech``,
``GET /v1/audio/voices``, health on ``/``.  Container port is 8880.

Vendor handling (image tag + device flags): auto-detected, overridable per
run via profiles.yaml ``t2s: {vendor:, image:, podman_args:, voices_dir:}``
and CLI --kokoro-image.  A native ROCm image exists upstream, so AMD needs
no CUDA translation — only different device nodes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llama_packer.backends.base import BaseBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

# Upstream prebuilt images (models baked in; pin to a release tag for stability).
KOKORO_DEFAULT_IMAGES = {
    "nvidia": "ghcr.io/remsky/kokoro-fastapi-gpu:latest",   # cu126; -cu128 for RTX 50-series
    "amd": "ghcr.io/remsky/kokoro-fastapi-rocm:latest",     # native ROCm (experimental, amd64)
    "cpu": "ghcr.io/remsky/kokoro-fastapi-cpu:latest",
}
KOKORO_CONTAINER_PORT = 8880

# Device pass-through per vendor.  NVIDIA uses CDI (podman >= 4); ROCm needs
# the compute + render device nodes and media groups (see upstream
# docs/troubleshooting.md#linux-gpu-permissions).
KOKORO_DEVICE_FLAGS = {
    "nvidia": "--device nvidia.com/gpu=all",
    "amd": "--device /dev/kfd --device /dev/dri "
           "--group-add video --group-add render",
    "cpu": "",
}


def _kokoro_image(vendor: str) -> str:
    """Default image for *vendor* (falls back to CPU when unknown)."""
    return KOKORO_DEFAULT_IMAGES.get(vendor, KOKORO_DEFAULT_IMAGES["cpu"])


class KokoroPodmanBackend(BaseBackend):
    name = "kokoro-podman"
    formats = frozenset({".onnx", "hf_repo"})
    roles = frozenset({"t2s"})
    handles = frozenset({"hf_repo"})
    proxied = True

    def is_available(self, avail: dict) -> bool:
        return bool(avail.get("kokoro_image"))

    def build_cmd(
        self,
        model: "Model",
        ctx_size: int,
        parallel: int,
        cache_type: str,
        tvars: dict,
        include_mmproj: bool = True,
    ) -> tuple[str, dict]:
        image = tvars.get("kokoro_image") or _kokoro_image("cpu")
        vendor = tvars.get("kokoro_vendor", "cpu")
        device_flags = str(tvars.get("podman_args")
                           or KOKORO_DEVICE_FLAGS.get(vendor,
                                                      KOKORO_DEVICE_FLAGS["cpu"]))
        parts = [
            "podman run --init --rm",
            "--name ${MODEL_ID}",
            "-p ${PORT}:" + str(tvars.get("kokoro_container_port",
                                          KOKORO_CONTAINER_PORT)),
        ]
        if device_flags:
            parts.append(device_flags)
        voices_dir = tvars.get("voices_dir")
        if voices_dir:
            # Read-write: voicepacks load per request (.pt) and combined
            # voices are saved back into this directory by the server.
            parts.append(f"-v {voices_dir}:/app/api/src/voices/v1_0")
        parts.append(image)
        return " ".join(parts), {}
