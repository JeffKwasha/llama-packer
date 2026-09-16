# llama_packer/backends/audio_cpp.py
"""audio-cpp backend: text-to-speech / speech-to-text via audio.cpp.

Runs the native ggml ``audiocpp_server`` (0xShug0/audio.cpp) as a proxied HTTP
service, exactly like ``sd-server``/``whisper-server``: llama-swap owns the
entry, writes a ``server.json`` under ``/tmp/llama-swap/`` and execs the
binary.  One entry per sidecar model (1:1) keeps llama-packer's "one entry =
one process with a fixed command line" invariant; switching models reloads
audio.cpp completely (llama-swap stop/start), which is the accepted behavior.

Roles: ``t2s`` (TTS / voice-clone / music) and ``s2t`` (ASR / VAD / align /
diar).  Model identity is directory-authoritative (``t2s/`` / ``s2t/``) plus
the audio.cpp family/task declared in the sidecar ``audio_cpp:`` block —
audio GGUFs are never routed to ``llama-server``, and whisper ``.bin`` models
are never routed here (audio.cpp has no whisper family).  VRAM is
fixed-overhead (weights + a family buffer).

Verified against audio.cpp 0.8.0 (upstream docs + live binary, 2026-09-16):
server.json keys, ``hip`` backend + ``rocm`` CLI alias, ``lazy_load``,
``default_voice_preset``/``voice_presets``, ``load_options``/
``session_options``, ``model_spec_override``, per-model ``busy_timeout_ms``.
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
    "min_free_memory_mb", "voice_dir", "model_spec_override",
)

# The audio.cpp server-JSON backend enum (upstream app/server/README.md).
# ``rocm`` is a CLI alias for ``hip`` and is normalized here; ``best`` exists
# only in the CLI enum, not as a server-JSON value.
_AUDIO_CPP_BACKENDS = frozenset({"cuda", "vulkan", "cpu", "metal", "hip"})
_BACKEND_ALIASES = {"rocm": "hip"}

# audio.cpp family → default ``task`` for the families shipped/verified here
# (audio.cpp 0.8.0 loader table).  Unknown families fall back to the role
# default (t2s→tts, s2t→asr).
_FAMILY_TASK_DEFAULTS = {
    "kokoro_tts": "tts", "chatterbox": "clon", "chatterbox_turbo": "tts",
    "qwen3_tts": "tts", "pocket_tts": "tts",
    "qwen3_asr": "asr", "parakeet_tdt": "asr", "nemotron_asr": "asr",
}
# text→audio vs audio→text task families, for role-consistency warnings.
_T2S_TASKS = frozenset({"tts", "clon", "music", "gen", "sfx", "dialogue",
                        "edit", "design", "ctrl"})
_S2T_TASKS = frozenset({"asr", "vad", "align", "diar"})

# JSON placeholder for the port; llama-swap substitutes ${PORT} in the cmd
# string, so the number must survive into the heredoc unquoted.
_PORT_SENTINEL = "__AUDIOCPP_PORT__"


def _first_int(model: "Model", default: int, *candidates) -> int:
    """First coercible candidate (sidecar layering) else *default*."""
    for value in candidates:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning("audio-cpp: %s: ignoring non-integer value %r",
                           model.stem, value)
    return default


class AudioCppBackend(BaseBackend):
    """The audio.cpp engine (TTS/ASR/music); host process."""

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

    def _resolve_backend(self, model: "Model", tvars: dict, cfg: dict) -> str:
        """audio.cpp backend enum: sidecar > explicit (CLI > profiles) > auto
        > cpu, with ``rocm`` normalized to ``hip`` and ``best`` rejected."""
        raw = str(cfg.get("backend")
                  or tvars.get("audio_cpp_backend")
                  or tvars.get("audio_cpp_backend_auto")
                  or "cpu").lower()
        backend = _BACKEND_ALIASES.get(raw, raw)
        if backend not in _AUDIO_CPP_BACKENDS:
            logger.warning("audio-cpp: %s: backend %r is not one of %s "
                           "(rocm is accepted as an alias for hip; best is "
                           "not a server-JSON value); falling back to cpu",
                           model.stem, raw,
                           ", ".join(sorted(_AUDIO_CPP_BACKENDS)))
            backend = "cpu"
            return backend
        probed = str(tvars.get("audio_cpp_bin_backends") or "")
        if probed:
            available = {p.strip().lower() for p in probed.split(",") if p.strip()}
            if backend not in available:
                logger.warning(
                    "audio-cpp: %s: probed binary does not report backend "
                    "%r (reports: %s) — the load will fail unless the "
                    "binary was built for it", model.stem, backend, probed)
        return backend

    def _resolve_task(self, model: "Model", cfg: dict, family: str) -> str:
        """Task: sidecar > family default > role default, with warnings for
        sidecar/role inconsistency and clone-only families."""
        task = str(cfg.get("task") or _FAMILY_TASK_DEFAULTS.get(family)
                   or ("asr" if model.role == "s2t" else "tts"))
        expected = _FAMILY_TASK_DEFAULTS.get(family)
        if family == "chatterbox" and task == "tts":
            logger.warning("audio-cpp: %s: base chatterbox has no zero-shot "
                           "'tts' task (clon/vc only); use "
                           "chatterbox_turbo for tts", model.stem)
        if expected and task != expected:
            logger.warning("audio-cpp: %s: family %r normally runs task "
                           "%r (sidecar says %r)", model.stem, family,
                           expected, task)
        if model.role == "t2s" and task in _S2T_TASKS:
            logger.warning("audio-cpp: %s: task %r is audio→text but the "
                           "role is t2s (text→audio); capabilities will "
                           "mismatch the model", model.stem, task)
        if model.role == "s2t" and task in _T2S_TASKS:
            logger.warning("audio-cpp: %s: task %r is text→audio but the "
                           "role is s2t (audio→text); capabilities will "
                           "mismatch the model", model.stem, task)
        return task

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
        backend = self._resolve_backend(model, tvars, cfg)
        family = str(cfg.get("family") or getattr(model, "arch", "") or model.stem)
        if not cfg.get("family"):
            logger.warning("audio-cpp: %s: no audio_cpp.family declared; "
                           "using %r (audio.cpp resolves package specs by "
                           "family — set it explicitly)",
                           model.stem, family)
        task = self._resolve_task(model, cfg, family)
        path = str(model.gguf_path) if model.gguf_path else (model.hf_repo or "")

        # audio.cpp routes requests by its model `id`; llama-swap routes by
        # the llama-swap entry id (model.template_id). They must match or the
        # proxied request dies with "unknown model id".
        model_entry: dict = {
            "id": model.template_id,
            "family": family,
            "path": path,
            "task": task,
            "mode": str(cfg.get("mode") or "offline"),
        }
        options = cfg.get("options")
        if isinstance(options, dict) and options:
            model_entry["default_request_options"] = options
        for freeform in ("load_options", "session_options"):
            value = cfg.get(freeform)
            if isinstance(value, dict) and value:
                model_entry[freeform] = value
        if cfg.get("model_spec_override"):
            model_entry["model_spec_override"] = cfg["model_spec_override"]

        # Voice handling (upstream resolution precedence: voice_ref wins;
        # `voice` names a configured preset, a voice_dir wav, or a model-
        # native cached voice id).
        voice_ref = cfg.get("voice_ref")
        voice = cfg.get("voice")
        if voice_ref is not None:
            preset: dict = {"voice_ref": voice_ref}
            if cfg.get("reference_text"):
                preset["reference_text"] = str(cfg["reference_text"])
            model_entry["default_voice_preset"] = preset
        elif voice is not None:
            model_entry["default_voice_preset"] = str(voice)
        if isinstance(cfg.get("voice_presets"), dict) and cfg["voice_presets"]:
            model_entry["voice_presets"] = cfg["voice_presets"]

        server: dict = {
            "host": "127.0.0.1",
            "port": _PORT_SENTINEL,
            "backend": backend,
            # llama-swap swaps the whole process; defer the framework load
            # until the first request instead of loading at server start.
            "lazy_load": True,
            "device": _first_int(model, 0, cfg.get("device"),
                                 tvars.get("audio_cpp_device")),
            "threads": _first_int(model, 4, cfg.get("threads"),
                                  tvars.get("audio_cpp_threads")),
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
        # `env TMPDIR=…` shields the engine's sidecar-extraction cache
        # (/tmp/audiocpp-gguf) from other-uid ownership wedges on shared
        # boxes; profiles.yaml audio_cpp.tmpdir sets it.
        tmpdir = str(tvars.get("audio_cpp_tmpdir") or "")
        launch = f"exec {binary} --config {config_path}"
        if tmpdir:
            launch = f"exec env TMPDIR={tmpdir} {binary} --config {config_path}"
        cmd = (
            f"sh -c 'mkdir -p /tmp/llama-swap && cat > {config_path} <<JSON\n"
            f"{payload}\n"
            f"JSON\n"
            f"{launch}'"
        )
        return cmd, {}
