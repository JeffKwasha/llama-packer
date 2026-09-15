# llama_packer/backends/audio_cpp.py
"""audio-cpp backend: text-to-speech / speech-to-text via audio.cpp.

Runs the native ggml ``audiocpp_server`` (0xShug0/audio.cpp) as a proxied HTTP
service, exactly like ``sd-server``/``whisper-server``: llama-swap owns the
entry, writes a ``server.json`` under ``/tmp/llama-swap/`` and execs the binary.  One entry per sidecar
model (1:1) keeps llama-packer's "one entry = one process with a fixed command
line" invariant; audio.cpp's multi-model LRU is a shared-server deployment we
do not emit.

Roles: ``t2s`` (TTS / voice-clone) and ``s2t`` (ASR).  Model identity is
directory-authoritative (``t2s/`` / ``s2t/``) plus the audio.cpp family/task
declared in the sidecar ``audio_cpp:`` block — audio GGUFs are never routed to
``llama-server``, and whisper ``.bin`` models are never routed here (audio.cpp
has no whisper family).  VRAM is fixed-overhead (weights + a family buffer).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from llama_packer import utils
from llama_packer.backends.base import BaseBackend

if TYPE_CHECKING:
    from llama_packer.model import Model

logger = logging.getLogger(__name__)

AUDIO_CPP_DEFAULT_BIN = "audiocpp_server"

# Server-level ``server.json`` keys sourced from profiles.yaml ``audio_cpp:``
# and stamped into every emitted config.
AUDIO_CPP_SERVER_KNOBS = (
    "max_loaded_models", "idle_unload_ms", "busy_timeout_ms",
    "min_free_memory_mb", "voice_dir",
)

# The audio.cpp backend enum.  `auto` is resolved by the CLI (vendor probe).
_AUDIO_CPP_BACKENDS = frozenset({"cuda", "vulkan", "cpu", "metal"})

# JSON placeholder for the port; llama-swap substitutes ${PORT} in the cmd
# string, so the number must survive into the heredoc unquoted.
_PORT_SENTINEL = "__AUDIOCPP_PORT__"


class AudioCppBackend(BaseBackend):
    """The audio.cpp engine (TTS/ASR); host process for now."""

    name = "audio-cpp"
    formats = frozenset({".gguf", ".safetensors", "hf_repo"})
    roles = frozenset({"t2s", "s2t"})
    handles = frozenset({"audio_cpp"})
    # Host first; a container transport is a later deployment (the engine
    # would reuse transport.py unchanged).
    transports = frozenset({"host"})
    host_requires = frozenset({"audio_cpp_bin"})
    proxied = True
    # audio.cpp answers readiness on /health (returns configured-model count).
    check_endpoint = "/health"

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
        cfg = model.frontmatter.get("audio_cpp") or {}
        if not isinstance(cfg, dict):
            logger.warning("audio-cpp: %s: audio_cpp: must be a mapping "
                           "(family/task/options/voice); ignoring",
                           model.stem)
            cfg = {}

        binary = str(tvars.get("audio_cpp_bin") or AUDIO_CPP_DEFAULT_BIN)
        backend = str(tvars.get("audio_cpp_backend")
                      or cfg.get("backend") or "cpu")
        if backend not in _AUDIO_CPP_BACKENDS:
            logger.warning("audio-cpp: %s: backend %r is not one of %s; "
                           "falling back to cpu", model.stem, backend,
                           ", ".join(sorted(_AUDIO_CPP_BACKENDS)))
            backend = "cpu"
        task = str(cfg.get("task") or ("asr" if model.role == "s2t" else "tts"))
        family = str(cfg.get("family") or getattr(model, "arch", "") or model.stem)
        path = str(model.gguf_path) if model.gguf_path else (model.hf_repo or "")

        model_entry: dict = {
            "id": model.stem,
            "family": family,
            "path": path,
            "task": task,
            "mode": str(cfg.get("mode") or "offline"),
        }
        options = cfg.get("options")
        if isinstance(options, dict) and options:
            model_entry["default_request_options"] = options

        server: dict = {
            "host": "127.0.0.1",
            "port": _PORT_SENTINEL,
            "backend": backend,
            "device": int(tvars.get("audio_cpp_device") or cfg.get("device") or 0),
            "threads": int(tvars.get("audio_cpp_threads") or cfg.get("threads") or 1),
        }
        for knob in AUDIO_CPP_SERVER_KNOBS:
            value = tvars.get(f"audio_cpp_{knob}")
            if value not in (None, ""):
                server[knob] = value
        server["models"] = [model_entry]

        # json.dumps then unquote the port placeholder so it lands as a JSON
        # number after llama-swap's ${PORT} substitution.
        payload = json.dumps(server, separators=(",", ":"))
        payload = payload.replace(f'"{_PORT_SENTINEL}"', "${PORT}")

        config_path = f"/tmp/llama-swap/audiocpp-{utils.slugify(model.stem)}-${{PORT}}.json"
        cmd = (
            f"sh -c 'mkdir -p /tmp/llama-swap && cat > {config_path} <<JSON\n"
            f"{payload}\n"
            f"JSON\n"
            f"exec {binary} --config {config_path}'"
        )
        return cmd, {}
