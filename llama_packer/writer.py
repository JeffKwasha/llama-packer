# llama_packer/writer.py
"""Generate llama-swap config entries from Model objects.

Responsibilities are split along a plan → emit seam:

- :class:`Planner` turns models + VRAM budget into per-model
  :class:`Variant` plans (context sizes, mmproj keep/drop, profile groups).
- :func:`emit_config` renders those plans into llama-swap entry dicts —
  no VRAM math, trivially testable.
- :func:`build_config` composes: filter → plan → emit.
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llama_packer.model import Model
from llama_packer.vram import solve_matrix_ctx
from llama_packer.hardware import detect_gpu_env_var
from llama_packer import utils
from llama_packer.consts import (
    _DIFFUSION_ARCH_RES,
    _KV_CACHE_BYTES,
    _MEMORY_MARGIN,
    _MIN_AGENTIC_CTX,
    _MIN_CTX_SIZE,
    ESTIMATE_ERROR_REASON,
)
from llama_packer.profiles import Profiles, parse_spare_mb
from llama_packer.backends import (
    SETTING_KEYS,
    FRAMEWORK_CONSUMED,
    METADATA_ONLY,
    VLLM_BACKENDS,
    get_backend,
)

logger = logging.getLogger(__name__)

# llama-server ``--reasoning-format`` modes (see its CLI help).
_REASONING_FORMATS = frozenset({"none", "deepseek", "deepseek-legacy", "auto"})
_REASONING_FLAG_KEYS = ("reasoning-format", "reasoning-preserve")


def _model_can_reason(model: Model) -> bool:
    """True when the model is a chat model that advertises reasoning support."""
    if model.role != "chat":
        return False
    return "reasoning" in [c.lower() for c in model.capabilities]


def _strip_repeat_ws(text: str) -> str:
    """Collapse runs of whitespace to single spaces (templates with | blocks)."""
    return " ".join(text.split())


def _fmt_mib(mib: float) -> str:
    """Human memory size: ``X.XGB`` at/above 1 GiB, integer ``MB`` below.

    Small values stay truthful (``RAM 64MB``, never ``0.0GB``) so a tiny
    host residue is distinguishable from a missing number.
    """
    if mib >= 1024:
        return f"{mib / 1024:.1f}GB"
    return f"{int(round(mib))}MB"


def format_mem_tag(
    vram_mib: float | None,
    ram_mib: float,
    *,
    spill_mib: float | None = None,
    ssd_mib: float | None = None,
) -> str:
    """Memory-allocation tag appended to emitted entry descriptions.

    A verbatim report of the decided TOTAL allocation attributable to the
    entry (weights + companions + context at the served ``(ctx, slots)``)
    — never capped, never re-solved: an impossible-looking ``VRAM 73.0GB``
    on a 32 GB card is printed as decided.  ``vram_mib=None`` is a
    CPU-resident serving (``RAM``-only tag).  ``spill_mib`` (driver-managed
    overflow, ``VRAM cap + spill``) and ``ssd_mib`` (disk-resident weights)
    are reserved grammar for a future allocator that decides a split —
    today's allocator never splits, so callers leave them None.
    """
    if vram_mib is None:
        tag = f"RAM {_fmt_mib(ram_mib)}"
    else:
        vram_part = f"VRAM {_fmt_mib(vram_mib)}"
        if spill_mib is not None:
            vram_part += f" + {_fmt_mib(spill_mib)}"
        tag = f"{vram_part} RAM {_fmt_mib(ram_mib)}"
    if ssd_mib is not None:
        tag += f" SSD {_fmt_mib(ssd_mib)}"
    return f"[{tag}]"


# A previously emitted memory tag at the end of a description (stripped
# before appending a fresh one, so copying a generated description back
# into a sidecar never stacks tags on re-pack).
_MEM_TAG_RE = re.compile(r"\s*\[(?:VRAM|RAM)\b[^\]]*\]\s*$")


def _with_mem_tag(description: str | None, tag: str | None) -> str | None:
    """Description with a fresh memory tag appended (idempotent).

    Returns None only when there is neither a description nor a tag.
    """
    if tag is None:
        return description
    base = _MEM_TAG_RE.sub("", description or "").rstrip()
    return f"{base} {tag}" if base else tag


def _filter_supported(models: list[Model], default_cache_type: str = "q8_0") -> list[Model]:
    """Final validation before any backend renders a command.

    Single place where a resolved model is validated (and, when an *option* is
    wrong, cleaned): backend format/role compatibility, reasoning-flag value
    and applicability, and cache-type knowability.  A rejected model is logged
    as an error and skipped; a rejected option is dropped.  Returns the
    supported subset.
    """
    supported: list[Model] = []
    for model in models:
        if getattr(model, "_override_error", None):
            # Already logged (and the model flagged) during scope finalization.
            continue

        backend = get_backend(model.backend)
        reason = backend.unsupported_reason(model)
        if reason:
            model_file = str(model.gguf_path) if model.gguf_path else (model.hf_repo or "no file")
            logger.error("No backend supports %s in %s (backend %s, role %s): %s",
                         model_file, model.md_path, backend.name, model.role, reason)
            continue

        # Diffusion/image-generation GGUF under a non-image role would
        # incorrectly emit a llama-server entry. Classify by header, not filename.
        if model.gguf_path and model.gguf_path.is_file():
            arch = model.arch
            if arch and any(rx.search(arch) for rx in _DIFFUSION_ARCH_RES):
                if model.role != "image":
                    logger.error("skipping %s: diffusion arch %r requires role: image (sd-server); "
                                 "move to img/ with dirs:{img:image} or set ignore:true (found role=%r)",
                                 model.stem, arch, model.role)
                    continue
                # explicit architecture hint for video diffusion (H3 etc.)
                fm_arch = str(model.frontmatter.get("architecture") or "").lower()
                if not fm_arch:
                    logger.warning("sidecar %s: diffusion arch %r but no architecture: set (e.g. architecture: wan/hunyuan-video/flux) for backend routing",
                                   model.stem, arch)

        # Capability / companion cross-check: a companion file is not a
        # capability — what it enables must be declared where it is served.
        caps_l = [str(c).lower() for c in model.capabilities]
        has_mmproj = bool(model.mmproj and model.mmproj.gguf_path)
        if "vision" in caps_l:
            logger.error("skipping capability 'vision' on %s: removed, use 'image' "
                         "(llama-swap modalities are text/audio/image/video)",
                         model.stem)
        if has_mmproj and model.role not in ("s2t", "t2s", "image", "embeddings", "rerank"):
            block = model.mmproj_overlay
            if not block:
                # Legacy companion (no declared purpose): keep the
                # historical warning.
                if "image" not in caps_l and "video" not in caps_l:
                    logger.warning(
                        "sidecar %s: mmproj companion present but neither 'image' nor 'video' "
                        "declared — projection costs VRAM but is not advertised; "
                        "declare capabilities: [image] (or [image, video])",
                        model.stem)
            else:
                block_caps = [str(c).lower()
                              for c in (block.get("capabilities") or [])]
                if not block_caps:
                    logger.warning(
                        "sidecar %s: mmproj block declares no capabilities — "
                        "the companion's serving difference is not advertised; "
                        "declare what it adds (e.g. capabilities: [image])",
                        model.stem)
                for c in ("image", "video"):
                    if c in caps_l and c not in block_caps:
                        logger.warning(
                            "sidecar %s: capability %r claimed at top level but "
                            "served only with the companion — off-variants "
                            "advertise it without the file; move it into the "
                            "mmproj block", model.stem, c)

        # Sanitize the serving views: the companion-on view (block merged)
        # plus the base frontmatter (the companion-off serving).  Pops apply
        # to every dict so a bad key dies on all variants.
        eff = model.view_for(True)
        fms = [model.frontmatter]
        if eff is not model:
            fms.append(eff.frontmatter)
        fm = eff.frontmatter

        # Reasoning flags must name a known mode and apply to a reasoning model.
        rf = fm.get("reasoning-format")
        if rf is not None and str(rf).lower() not in _REASONING_FORMATS:
            logger.error("skipping %s: unknown reasoning-format %r (allowed: %s); ignored",
                         model.stem, rf, ", ".join(sorted(_REASONING_FORMATS)))
            for d in fms:
                d.pop("reasoning-format", None)
        if not _model_can_reason(eff):
            for k in _REASONING_FLAG_KEYS:
                if k in fm:
                    logger.error("skipping %s: %s declared on a non-reasoning model "
                                 "(role=%r, capabilities=%s); ignored",
                                 model.stem, k, eff.role, eff.capabilities)
                    for d in fms:
                        d.pop(k, None)

        # cache_type must be a precision we can size memory for.
        cache_type = eff.cache_type_for(default_cache_type)
        if cache_type not in _KV_CACHE_BYTES:
            logger.error("skipping %s: unknown cache_type %r (known: %s)",
                         model.stem, cache_type, ", ".join(sorted(_KV_CACHE_BYTES)))
            continue

        declared = {k for k in SETTING_KEYS if k in fm}
        backend.warn_unhandled(declared - FRAMEWORK_CONSUMED - METADATA_ONLY)
        supported.append(model)
    return supported


def _build_mode_params(model: Model, profiles_group: list[tuple[str, dict]],
                        profiles_defaults: dict) -> dict[str, dict]:
    """Build ``setParamsByID`` from a model's sidecar-declared ``modes``.

    Each declared mode layers over the same-named resolved profile (or the
    fleet defaults when no profile shares the name) with the single merge
    rule — unspecified keys inherit from below instead of being dropped.
    Every merged numeric param is emitted (full-profile definition, no
    diff-vs-defaults suppression). The model's ``default_mode`` maps to the
    bare ``${MODEL_ID}`` key; every other mode to ``${MODEL_ID}:<mode>``.

    Returns an empty dict when the model declares no modes (caller then keeps
    the global-profile behavior).
    """
    modes = model.modes
    if not modes:
        return {}

    prof_by_name = {pname: resolved for pname, resolved in profiles_group}
    set_params: dict[str, dict] = {}
    default_mode = model.default_mode
    for name, params in modes.items():
        base = prof_by_name.get(name, profiles_defaults)
        merged = utils.merge_layer(base, params,
                                   origin=f"modes:{model.stem}:{name}")
        overrides: dict = {}
        for k, v in merged.items():
            if k not in utils.SAMPLING_KEYS:
                # Inherited non-sampling keys (cache_type, parallel, …) are
                # expected from the layer below — only warn about keys the
                # sidecar itself authored.
                if k in params:
                    logger.warning("modes: %s: unknown sampling key %r (ignored)", model.stem, k)
                continue
            if isinstance(v, bool):
                logger.warning("modes: %s: %s.%s=%r is not numeric (ignored)", model.stem, name, k, v)
                continue
            if isinstance(v, (int, float)):
                val = v
            else:
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    logger.warning("modes: %s: %s.%s=%r is not numeric (ignored)", model.stem, name, k, v)
                    continue
            overrides[utils.request_sampling_key(k)] = round(val, 6) if isinstance(val, float) else val
        key = "${MODEL_ID}" if name == default_mode else f"${{MODEL_ID}}:{name}"
        if overrides:
            set_params[key] = overrides
    return set_params


def _build_entry(
    model: Model,
    parallel: int,
    cache_type: str,
    profiles_group: list[tuple[str, dict]],
    profiles_defaults: dict,
    template_vars: dict,
    context_length: int,
    ctx_size: int,
    include_mmproj: bool = True,
    name_suffix: str = "",
    tools_demoted: bool = False,
    estimate_error: str | None = None,
    batch: int | None = None,
    ubatch: int | None = None,
    mem_vram_mib: float | None = None,
    mem_ram_mib: float | None = None,
) -> tuple[str, dict]:
    """Build a single llama-swap config entry for a model+profile group.

    ``include_mmproj=False`` omits the vision projection from the command,
    removes the ``image``/``video`` input capabilities, and flags
    ``metadata.mmproj_skipped``.
    ``name_suffix`` is appended to the model display name (e.g. the vision
    variant's `` [vision 92k]``).
    ``tools_demoted=True`` drops the ``tools`` capability: the matrix solve
    served the model below ``tools_min_ctx``, so advertising tool calling
    would mislead clients — ``metadata.tools_demoted`` records why.

    The first step resolves the serving view: with a companion block, the
    companion-on variant serves the block merged over the frontmatter while
    the     companion-off variant serves the base frontmatter (strip-by-recompute
    — purpose is emergent from the block, never hardcoded).  Pass the base
    model here, not a view: views return themselves from ``view_for``.

    ``mem_vram_mib``/``mem_ram_mib`` are the decided TOTAL allocation for
    this serving (reported verbatim as a trailing description tag via
    :func:`format_mem_tag`; ``None``/``None`` or a non-None
    ``estimate_error`` emits no tag).
    """
    model = model.view_for(include_mmproj)
    base_id = model.template_id

    # Build the launch command via the model's backend.  The backend is chosen
    # by override rules (model.backend), and validation/skip of unsupported
    # format/role combos happens in build_config's pre-pass.
    backend = get_backend(model.backend)
    # Verbose trace: what backend was picked and why (visible with -V/-VV)
    if logger.isEnabledFor(logging.INFO):
        arch = str(model.frontmatter.get("architecture") or model.frontmatter.get("base_model") or "")
        fmt = model.gguf_path.suffix if model.gguf_path else ("hf_repo" if model.hf_repo else "no-file")
        logger.info("backend %s -> %s (role=%s fmt=%s arch=%s caps=%s%s)",
                    model.stem, backend.name, model.role, fmt, arch or "-",
                    ",".join(model.capabilities) or "-",
                    " mmproj" if (model.mmproj and model.mmproj.gguf_path) else "")
        logger.debug("backend %s handles=%s", backend.name, sorted(backend.handles))
    cmd_str, backend_meta = backend.build_cmd(
        model, ctx_size, parallel, cache_type, template_vars,
        include_mmproj=include_mmproj,
        batch=batch, ubatch=ubatch,
    )
    cmd_str = _strip_repeat_ws(cmd_str)

    # Profile params → setParamsByID
    set_params: dict[str, dict] = {}
    for pname, resolved_prof in profiles_group:
        overrides = {}
        for k in utils.SAMPLING_KEYS:
            val, dval = resolved_prof.get(k), profiles_defaults.get(k)
            if val is not None and dval is not None and val != dval:
                overrides[utils.request_sampling_key(k)] = round(val, 6) if isinstance(val, float) else val
        if overrides:
            key = "${MODEL_ID}" if pname == "default" else f"${{MODEL_ID}}:{pname}"
            set_params[key] = overrides

    # Sidecar-declared modes layer over the same-named profile (or the
    # fleet defaults) and replace the global profile sampling overrides
    # for this model.
    mode_params = _build_mode_params(model, profiles_group, profiles_defaults)
    if mode_params:
        set_params = mode_params

    names = [p[0] for p in profiles_group]
    has_default = "default" in names
    entry_id = base_id if (has_default or len(profiles_group) > 1) else f"{base_id}.{names[0]}"

    # Metadata: pass-through frontmatter + computed selection signals.
    # Pass-through-by-default: any new sidecar field an agent writes flows
    # through automatically; only builder-consumed keys are excluded.
    metadata = model.pass_through_metadata()

    # Directional modalities (llama-swap derives badges from these):
    # image → image INPUT; video → video INPUT (and OUTPUT for omni/video-arch);
    # audio → audio INPUT (Transcription); speech → audio OUTPUT.
    # Output stays text unless `speech`/`video` output is declared.
    # role=image (sd-server) → diffusion outputs image or video by architecture;
    #   in:[text,image] out:[image] vs out:[video] (see architecture/video token).
    # role=s2t (whisper-server) → audio in, text out (Transcription badge).
    caps_l = [c.lower() for c in model.capabilities]
    arch = str(model.frontmatter.get("architecture") or "").lower()
    is_video_arch = "video" in arch or arch in {"wan", "hunyuan-video", "h3", "mochi", "cosmos"}
    if model.role == "image":
        in_mods = ["text", "image"]
        # architecture or explicit video token decides output modality
        if "video" in caps_l or is_video_arch:
            out_mods = ["video"]
        else:
            out_mods = ["image"]
    elif model.role == "s2t":
        in_mods = ["audio"]
        out_mods = ["text"]
    elif model.role == "t2s":
        in_mods = ["text"]
        out_mods = ["audio"]
    else:
        in_mods = ["text"]
        out_mods = ["text"]
        if "image" in caps_l:
            in_mods.append("image")
        if "video" in caps_l:
            in_mods.append("video")
        if "audio" in caps_l:
            in_mods.append("audio")
        if "speech" in caps_l:
            out_mods.append("audio")
        # omni models with video capability can also generate video
        if "video" in caps_l and (is_video_arch or arch == "omni" or "omni" in arch):
            if "video" not in out_mods:
                out_mods.append("video")

    # When mmproj is dropped, remove its associated input modalities.
    # mmproj does not imply image/video - only explicit tokens are removed,
    # and only when a companion exists (baked-in video stays).  With a
    # companion block, ``model`` is already the base (off) view so block
    # capabilities are absent; this stays as a safety net for unconditional
    # top-level image/video claims.
    if not include_mmproj and model.role != "image" and model.mmproj and model.mmproj.gguf_path:
        caps = [c for c in model.capabilities if str(c).lower() not in ("image", "video")]
        if "image" in caps_l and "image" in in_mods:
            in_mods.remove("image")
        if "video" in caps_l and "video" in in_mods:
            in_mods.remove("video")
        metadata["mmproj_skipped"] = True
    else:
        caps = list(model.capabilities)

    # tools demotion: served context below the tools threshold → stop
    # advertising tool calling (per-variant; the sidecar declaration is
    # untouched, so a re-pack at a higher context restores it).
    if tools_demoted and "tools" in [c.lower() for c in caps]:
        caps = [c for c in caps if str(c).lower() != "tools"]
        metadata["tools_demoted"] = True
        logger.info("capabilities: %s: tools demoted (served ctx %d below "
                    "tools_min_ctx)", model.stem, ctx_size)

    tf = model.throughput_factor()
    if tf is not None:
        metadata["throughput_factor"] = tf

    metadata["mtp_enabled"] = backend_meta.get("mtp_enabled", False)
    if backend_meta.get("mtp_enabled"):
        metadata["mtp_draft_max"] = backend_meta["mtp_draft_max"]

    # Image token budget (client-facing so callers can size requests): only
    # advertised on variants that actually serve the vision projection.
    # ``model`` is the serving view, so block-declared token keys are
    # present exactly on companion-on variants.
    if include_mmproj and model.mmproj and model.mmproj.gguf_path:
        if model.image_min_tokens is not None:
            metadata["image_min_tokens"] = model.image_min_tokens
        if model.image_max_tokens is not None:
            metadata["image_max_tokens"] = model.image_max_tokens

    # Expose the resolved chat template so clients know which Jinja template
    # drives the model, and which kwargs they may pass per-request
    # (e.g. Qwen's enable_thinking).  These are client-facing only — no
    # server-side flag exists for the kwargs.
    ct = model.resolved_chat_template
    if ct is not None:
        metadata["chat_template"] = ct.stem
    kwargs = model.chat_template_kwargs
    if kwargs:
        metadata["chat_template_kwargs"] = copy.deepcopy(kwargs)

    # Expose declared sampling modes so clients (hermes, opencode, UIs) can
    # discover the per-request aliases ("<id>:<mode>") without hitting the
    # model list. Static discovery: metadata is passed through in /v1/models.
    mode_params_keys = model.modes
    if mode_params_keys:
        metadata["modes"] = sorted(mode_params_keys)
        metadata["default_mode"] = model.default_mode

    entry: dict = {"cmd": cmd_str}
    if set_params:
        # setParamsByID is a llama-swap *filter* and must be nested under
        # `filters:` — a top-level key is silently ignored.
        entry["filters"] = {"setParamsByID": set_params}
    if model.name:
        entry["name"] = model.name + name_suffix
    mem_tag = None
    if estimate_error is None and mem_ram_mib is not None:
        mem_tag = format_mem_tag(mem_vram_mib, mem_ram_mib)
    if model.description or mem_tag is not None:
        entry["description"] = _with_mem_tag(model.description, mem_tag)
    # The VRAM-served -c limit (vs. capabilities.context = max trained).
    metadata["ctx_size"] = ctx_size
    # Client-facing estimate health: when no VRAM estimate source worked,
    # the entry is served at its minimum useful context with a conservative
    # matrix reserve — tell clients so they can treat it differently.
    if estimate_error is not None:
        metadata["estimated"] = False
        metadata["estimate_error"] = estimate_error
    if metadata:
        entry["metadata"] = metadata

    # Native llama-swap capabilities block (shown in /v1/models).
    # `context` is the model's maximum trained context (GGUF architectural max
    # > sidecar context_length > default); the VRAM-served -c limit is exposed
    # separately as metadata.ctx_size.
    entry["capabilities"] = {
        "in": in_mods,
        "out": out_mods,
        "tools": "tools" in [c.lower() for c in caps],
        "reranker": "reranker" in caps_l or model.role == "rerank",
        "context": context_length,
    }

    # Per-model GPU device pinning (multi-GPU). Emits the appropriate
    # vendor env var so the server only sees that device.
    dev = model.device
    if dev is not None:
        env_var = detect_gpu_env_var()
        entry["env"] = [f"{env_var}={dev}"]

    # Per-model concurrency limit.
    conc = model.concurrency
    if conc is not None:
        entry["concurrencyLimit"] = conc

    # Proxied backends (sd-server, whisper-server, audio-cpp) are proxied HTTP
    # services, not llama-swap managed inference — expose the standard proxy
    # fields so llama-swap can health-check and route.  Each backend names its
    # own health path (sd-server answers "/"; audio.cpp exposes /health).
    if backend.proxied:
        entry["proxy"] = "http://127.0.0.1:${PORT}"
        entry["checkEndpoint"] = getattr(backend, "check_endpoint", "/")

    # Container lifecycle (llama-swap docker orchestration): cmdStop stops the
    # container itself — without it a swap/unload kills only the `docker run`
    # client process, leaving the container running with its VRAM held
    # (llama-swap kb guides/model-runtime/ttl-and-unloading.md).  unloadTimeout
    # must exceed the stop grace (docker stop is slow).  Container backends are
    # also the documented case for the explicit proxy field ("the single most
    # common configuration error" — kb guides/model-runtime/writing-cmd.md).
    if backend.stop_cmd is not None:
        entry["cmdStop"] = backend.stop_cmd
        entry["unloadTimeout"] = backend.unload_timeout
        entry["proxy"] = "http://127.0.0.1:${PORT}"

    return entry_id, entry


# ── Planning ──────────────────────────────────────────────────────────────


# Entry-id suffix of every no-mmproj variant.  Invariant: the bare ``<id>``
# always serves vision when the model has an mmproj; the text-only serving is
# always ``<id>-text`` (see :func:`emit_config`).
TEXT_SUFFIX = "-text"


@dataclass(frozen=True)
class MatrixKnobs:
    """Tunables of the matrix solve, from the ``matrix:`` config section.

    Context tiers, smallest to largest: ``coload_min_ctx`` (emb/rnk squeeze
    floor) → ``min_chat_ctx`` (co-load decision floor) → ``tools_min_ctx``
    (tools advertisement threshold).  ``embed_context`` / ``rerank_context``
    are RAG caps — the maximum context those residents serve at (default
    4096, capped by each model's design context); the served value is
    ``min(rag_cap(role), design_context)`` everywhere, including for
    unestimable RAG models.  ``ctx_gain_min`` gates the squeeze adoption;
    ``estimate_headroom`` pads estimated co-load overheads.
    ``auto_parallel`` enables the value-function (ctx, slots) solve for
    unpinned chat models (see :func:`parallel_value`); ``auto_parallel_max``
    caps the slots and ``parallel_power`` is the score's slot exponent.
    Auto-parallel is on by default; ``matrix: auto_parallel: false``
    disables it fleet-wide and a sidecar/block ``parallel:`` pin opts a
    single model out.  ``vllm_auto_parallel_max`` is the cap for vLLM
    backends: continuous batching makes a slot cost KV only (no per-slot
    workspace), so vLLM tolerates far higher concurrency — and a sidecar
    ``parallel: 0`` on a vLLM model means *uncapped* (emit no
    ``--max-num-seqs``; vLLM admits elastically up to the pool, queueing
    the rest).  On llama.cpp backends ``parallel: 0`` is invalid and falls
    back to the fleet default with a warning.
    """
    min_chat_ctx: int = 65536
    tools_min_ctx: int = 131072
    coload_min_ctx: int = 20480
    embed_context: int = 4096
    rerank_context: int = 4096
    ctx_gain_min: int = 4096
    estimate_headroom: float = 1.25
    auto_parallel: bool = True
    auto_parallel_max: int = 8
    vllm_auto_parallel_max: int = 16
    parallel_power: float = 0.75

    @classmethod
    def from_cfg(cls, matrix_cfg: dict | None) -> "MatrixKnobs":
        """Parse knobs from the matrix section, warning and defaulting on
        invalid values (a bad knob must not silently break the solve)."""
        cfg = matrix_cfg or {}
        knobs: dict = {}
        for key in ("min_chat_ctx", "tools_min_ctx", "coload_min_ctx",
                    "embed_context", "rerank_context", "ctx_gain_min",
                    "auto_parallel_max", "vllm_auto_parallel_max"):
            v = cfg.get(key)
            if v is None:
                continue
            try:
                iv = int(v)
                assert iv > 0
            except (TypeError, ValueError, AssertionError):
                logger.warning("matrix: %s=%r is not a positive integer; "
                               "using default", key, v)
                continue
            knobs[key] = iv
        v = cfg.get("estimate_headroom")
        if v is not None:
            try:
                fv = float(v)
                assert fv >= 1.0
            except (TypeError, ValueError, AssertionError):
                logger.warning("matrix: estimate_headroom=%r is not a float "
                               ">= 1.0; using default", v)
            else:
                knobs["estimate_headroom"] = fv
        v = cfg.get("auto_parallel")
        if v is not None:
            knobs["auto_parallel"] = _parse_bool_knob("auto_parallel", v)
        v = cfg.get("parallel_power")
        if v is not None:
            try:
                fv = float(v)
                assert fv > 0
            except (TypeError, ValueError, AssertionError):
                logger.warning("matrix: parallel_power=%r is not a float "
                               "> 0; using default", v)
            else:
                knobs["parallel_power"] = fv
        return cls(**knobs)

    def rag_cap(self, role: str) -> int:
        """Configured context cap for a RAG role (embeddings | rerank).

        The served RAG context is ``min(rag_cap(role), design_context)``
        everywhere; unknown roles fall back to ``coload_min_ctx`` (legacy
        behavior).
        """
        if role == "embeddings":
            return self.embed_context
        if role == "rerank":
            return self.rerank_context
        return self.coload_min_ctx


def _parse_bool_knob(key: str, v: object) -> bool:
    """Parse an opt-in knob; warn and return False on garbage."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in (
            "true", "yes", "on", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in (
            "false", "no", "off", "0"):
        return False
    logger.warning("matrix: %s=%r is not a boolean; using default", key, v)
    return False


def parallel_value(ctx: int, floor: int, slots: int, power: float) -> float:
    """Rank a (context, slots) candidate: ``(ctx/floor)^0.5 × slots^power``.

    Context is the ratio over the model's floor, square-rooted so each
    doubling is worth less than the last; slots carry the operator-set
    parallel power (B).  Absolute values are meaningless — only the ranking
    of feasible candidates matters.  ``floor`` must be positive.
    """
    return (ctx / floor) ** 0.5 * slots ** power


def resolve_min_ctx(view, *, pin_ctx: int | None = None,
                    tools_min_ctx: int = 131072,
                    fallback_min_ctx: int = _MIN_AGENTIC_CTX,
                    fallback_explicit: bool = False) -> int:
    """Per-model minimum useful context: the hard floor of the (ctx, slots)
    search and the context-ratio normalizer in :func:`parallel_value`.

    Cascade, first match wins: explicit sidecar ``min_context:`` → pinned
    serving ctx (a fixed ctx is its own floor) → tool callers get the
    agentic floor → everyone else gets half their max context (two slots are
    then only considered when nearly two full windows fit).  The global
    ``--min-context`` flag overrides the last two rows when explicitly set,
    never an explicit sidecar key or a hard pin.  A pin below an explicit
    floor warns (the pin is honoured; the floor is reported unmet).
    """
    fm = view.frontmatter or {}
    explicit = fm.get("min_context")
    if explicit is not None:
        try:
            floor = int(explicit)
            assert floor > 0
        except (TypeError, ValueError, AssertionError):
            logger.warning("min_context=%r on %s is not a positive integer; "
                           "ignoring", explicit, view.stem)
        else:
            if pin_ctx is not None and pin_ctx < floor:
                logger.warning(
                    "min_context=%d on %s is above the pinned context %d; "
                    "honouring the pin, floor unmet", floor, view.stem,
                    pin_ctx)
            return floor
    if pin_ctx is not None:
        return pin_ctx
    if "tools" in [c.lower() for c in (view.capabilities or [])]:
        return fallback_min_ctx if fallback_explicit else tools_min_ctx
    if fallback_explicit:
        return fallback_min_ctx
    return max(int(view.design_context // 2), _MIN_CTX_SIZE)


# Size buckets for resident accounting: footprints bucket by size class, not
# model type.  Tiny residents (speech, RAG) join every set on their GPU for
# pocket change; huge residents (diffusion) are chat-scale and tracked as
# exclusive candidates; chat models evict each other within a set.
BUCKET_TINY_ROLES = frozenset({"s2t", "t2s", "embeddings", "rerank"})
BUCKET_HUGE_BACKENDS = frozenset({"sd-server"})


def footprint_bucket(model) -> str:
    """Size bucket of a model for ledger accounting: tiny/huge/chat."""
    if model.role in BUCKET_TINY_ROLES:
        return "tiny"
    if model.role == "image" or model.backend in BUCKET_HUGE_BACKENDS:
        return "huge"
    return "chat"


class PoolLedger:
    """One shared VRAM ledger: every claimant charged to a named pool.

    Single-pool math today (the detected total), keyed by pool id so
    per-GPU attribution slots in later: ``pool_id_for`` maps a model's
    ``device:`` pin to ``gpuN``, unpinned models land on ``default``.  An
    operator-declared ``pools: <id>: vram:`` overrides the pool size;
    per-pool ``spare:`` overrides the global spare; ``reserve_extra:`` and
    explicit ``pins:`` (``"auto"`` = derived math stays authoritative)
    accumulate into :meth:`extra_reserve_mb`, which callers add to the
    solve's spare.  Residents (RAG at solved ctx, adopted co-loads) are
    registered by the planner after the matrix solve.
    """

    def __init__(self, vram_total_mb: int, pools_cfg: dict | None = None):
        self.vram_total_mb = vram_total_mb
        self.pools_cfg = pools_cfg or {}
        self.residents: list[tuple[str, str, int]] = []  # (label, pool, mb)

    def pool_id_for(self, model) -> str:
        dev = (model.frontmatter or {}).get("device")
        if dev is None:
            return "default"
        try:
            return f"gpu{int(dev)}"
        except (TypeError, ValueError):
            return str(dev)

    def pool_vram_mb(self, pool_id: str) -> int:
        raw = self.pools_cfg.get(pool_id, {}).get("vram")
        if raw is not None:
            try:
                return int(utils.parse_mem_mb(str(raw), self.vram_total_mb))
            except (TypeError, ValueError, AssertionError):
                logger.warning("pools: %s vram=%r unreadable; using detected "
                               "total", pool_id, raw)
        return self.vram_total_mb

    def add_resident(self, label: str, mb: int, pool_id: str = "default"
                     ) -> None:
        if mb > 0:
            self.residents.append((label, pool_id, int(mb)))

    def _explicit_pin_mb(self, pool_id: str, raw: object,
                         claimant: str) -> int:
        if raw is None:
            return 0
        if isinstance(raw, str) and raw.strip().lower() == "auto":
            return 0  # derived math stays authoritative
        try:
            return int(utils.parse_mem_mb(str(raw),
                                          self.pool_vram_mb(pool_id)))
        except (TypeError, ValueError, AssertionError):
            logger.warning("pools: %s pins %s=%r unreadable; ignoring",
                           pool_id, claimant, raw)
            return 0

    def spare_for(self, pool_id: str, global_spare_mb: int) -> int:
        raw = self.pools_cfg.get(pool_id, {}).get("spare")
        if raw is None:
            return global_spare_mb
        try:
            return int(utils.parse_mem_mb(str(raw),
                                          self.pool_vram_mb(pool_id)))
        except (TypeError, ValueError, AssertionError):
            logger.warning("pools: %s spare=%r unreadable; using global",
                           pool_id, raw)
            return global_spare_mb

    def extra_reserve_mb(self, pool_id: str = "default") -> int:
        """MB to add to the solve's spare for this pool: unmodelled
        residents, explicit pins, and registered co-residents."""
        spec = self.pools_cfg.get(pool_id, {})
        total = 0
        raw_extra = spec.get("reserve_extra")
        if raw_extra is not None:
            try:
                total += int(utils.parse_mem_mb(str(raw_extra),
                                                self.pool_vram_mb(pool_id)))
            except (TypeError, ValueError, AssertionError):
                logger.warning("pools: %s reserve_extra=%r unreadable; "
                               "ignoring", pool_id, raw_extra)
        for claimant, raw in (spec.get("pins") or {}).items():
            total += self._explicit_pin_mb(pool_id, raw, claimant)
        total += sum(mb for _, pool, mb in self.residents if pool == pool_id)
        return total


@dataclass(frozen=True)
class MatrixSolve:
    """Result of the shared matrix solve.

    ``chat_ctx`` is the solved shared chat context.  ``embed_ctx`` /
    ``rerank_ctx`` are the contexts the RAG models should be served at
    (design context, or the squeezed value when the squeeze was adopted).
    ``coloads`` lists opportunistically included non-chat models as
    (stem, fixed overhead MB) pairs.
    """
    chat_ctx: int
    embed_ctx: int
    rerank_ctx: int
    coloads: tuple[tuple[str, int], ...] = ()
    squeeze: bool = False


@dataclass(frozen=True)
class Variant:
    """One planned llama-swap entry: resolved serving params + contexts.

    ``vision_ctx`` is set only when mmproj was dropped from the main variant
    but a companion exists — the emitter then adds a best-effort vision entry.
    A variant with ``include_mmproj=False`` is emitted as the ``<id>-text``
    entry (this is both the on-demand text-only variant of a vision-keeping
    model and the renamed main entry of an auto-dropped model).
    """
    parallel: int
    cache_type: str
    spare_mb: int
    profiles_group: list[tuple[str, dict]] = field(compare=False)
    ctx_size: int
    include_mmproj: bool
    #: Resolved ``-b``/``-ub`` (sidecar > profile > fleet > role); rendered
    #: explicitly on every llama-server command; ``ubatch`` also stamps the
    #: VRAM measurement shape.
    batch: int | None = None
    ubatch: int | None = None
    vision_ctx: int | None = None
    coload: bool = False
    tools_demoted: bool = False
    #: Non-None when the model has no usable VRAM estimate — surfaced to
    #: clients as metadata.estimated=false / metadata.estimate_error.
    estimate_error: str | None = None
    #: Decided TOTAL memory attributable to this serving (weights +
    #: companions + context at the served ctx/slots), in MiB — the numbers
    #: the emitted description tag reports verbatim.  ``mem_vram_mib=None``
    #: is a CPU-resident serving (RAM-only tag).  Both None when undecided
    #: (no estimate) — then no tag is emitted.
    mem_vram_mib: float | None = None
    mem_ram_mib: float | None = None
    #: Same pair for the best-effort vision companion entry (served at
    #: ``vision_ctx`` with the projection); None when no vision entry.
    mem_vision_vram_mib: float | None = None
    mem_vision_ram_mib: float | None = None


def resolve_batch_ubatch(
    profiles: Profiles, model: Model, profile: dict | None = None,
) -> tuple[int, int]:
    """Resolved ``(batch, ubatch)`` for *model* — sidecar keys > profile
    keys > fleet ``llama_server: batch:/ubatch:`` > backend role defaults
    (the llama.cpp builtins).  One resolution site: the rendered command
    and the VRAM measurement shape both consume these values, so the serve
    and the estimate can never drift apart on batch flags."""
    try:
        backend = get_backend(model.backend)
        role_defaults = backend.default_batch_ubatch(model.role)
    except KeyError:
        role_defaults = (2048, 512)
    return model.batch_ubatch_for(
        profile, profiles.llama_server_cfg if profiles else {}, role_defaults)


def measurement_args(
    profiles: Profiles, model: Model, global_args: str = "",
    profile: dict | None = None,
) -> str:
    """The exact flag string VRAM constants are measured (and shape-stamped)
    under: global args + the model's resolved ``-ub``.  ``batch`` is
    excluded — measured to have no VRAM effect, so throughput tuning must
    not invalidate estimates."""
    _, ubatch = resolve_batch_ubatch(profiles, model, profile)
    base = (global_args or "").strip()
    return f"{base} -ub {ubatch}".strip()


class Planner:
    """Turn models + VRAM budget into per-model serving :class:`Variant`s.

    Owns every context decision: the mmproj keep/drop pre-pass, the shared
    matrix context solve, profile grouping, and the bounded-context clamp.
    Depends only on models' ``VramBudget`` interfaces, so tests substitute
    fake budgets instead of monkeypatching subprocesses.  Emission of
    llama-swap dicts is a separate concern (:func:`emit_config`).
    """

    def __init__(
        self,
        models: list[Model],
        profiles: Profiles,
        fit_bin: str,
        vram_total: int,
        *,
        spare: str | None = None,
        max_context: int | None = None,
        matrix_cfg: dict | None = None,
        embed_model: Model | None = None,
        rerank_model: Model | None = None,
        fixed_categories: list[tuple[str, Model]] | None = None,
        baseline_mb: int = 0,
        min_context: int = _MIN_AGENTIC_CTX,
        min_context_explicit: bool = False,
        memory_margin: float = _MEMORY_MARGIN,
        llama_args: str = "",
        progress_cb=None,
    ):
        self.models = models
        self.profiles = profiles
        self.fit_bin = fit_bin
        self.llama_args = llama_args
        self.vram_total = vram_total
        self.spare = spare
        self.max_context = max_context
        self.matrix_cfg = matrix_cfg
        self.knobs = MatrixKnobs.from_cfg(matrix_cfg)
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        # Declared matrix categories beyond the RAG pair (e.g. tts/stt):
        # fixed-overhead residents reserved alongside chat and RAG.
        self.fixed_categories = fixed_categories or []
        self.baseline_mb = baseline_mb
        self.min_context = min_context
        self.min_context_explicit = min_context_explicit
        self.memory_margin = memory_margin
        self.ledger = PoolLedger(vram_total, self.profiles.pools_cfg)
        self.chat_ctx: int | None = None  # matrix-solved shared context, if any
        self.matrix_result: MatrixSolve | None = None
        # Optional per-model tick (stem) for the progress bar; None in tests.
        self.progress_cb = progress_cb
        # Quads synthesized for unestimable matrix participants (stem →
        # (model_mib, kv_per_token, slot_mib, compute_mib)); see
        # _solve_matrix_context. Kept for ledger resident charging.
        self.synthetic_quads: dict[str, tuple[int, float, float, int]] = {}

    # ── bounded ctx: the single home of the clamp invariant ──

    def _bounded_ctx(
        self,
        model: Model,
        *,
        parallel: int,
        cache_type: str,
        spare_mb: int,
        include_mmproj: bool,
        design_ctx: int | None = None,
        context_length: int | None = None,
        profile: dict | None = None,
    ) -> int:
        """VRAM-solved context clamped to the model's max trained context
        (*context_length*) and the CLI ``--max-context`` cap.  Measured
        under the model's resolved ``-ub`` shape (:func:`measurement_args`);
        *profile* adds the per-profile tier when the solve is group-scoped."""
        ctx = model.vram.calc_ctx(
            self.vram_total,
            fit_bin=self.fit_bin,
            parallel=parallel,
            spare_mb=spare_mb,
            include_mmproj=include_mmproj,
            baseline_mb=self.baseline_mb,
            cache_type=cache_type,
            design_ctx=design_ctx,
            memory_margin=self.memory_margin,
            llama_args=measurement_args(self.profiles, model,
                                        self.llama_args, profile),
        )
        if context_length is not None:
            ctx = min(ctx, context_length)
        if self.max_context is not None:
            ctx = min(ctx, self.max_context)
        return ctx

    def _unestimated_ctx(self, view) -> int:
        """Minimum useful context for a model with no VRAM estimate.

        Type-based floor: a matrix chat model follows the shared solved
        context like every chat (the solve charged exactly that for it);
        chat without a matrix gets the per-model floor cascade; embeddings
        and rerank serve at their configured ``embed_context`` /
        ``rerank_context`` (capped by the trained design context and
        ``--max-context``).
        """
        if view.role in ("embeddings", "rerank"):
            ctx = min(view.design_context, self.knobs.rag_cap(view.role))
        elif view.role == "chat" and self.chat_ctx is not None:
            ctx = self.chat_ctx
        else:
            ctx = resolve_min_ctx(
                view, pin_ctx=self._serving_pin(view),
                tools_min_ctx=self.knobs.tools_min_ctx,
                fallback_min_ctx=self.min_context,
                fallback_explicit=self.min_context_explicit)
        cap = view.design_context
        if self.max_context is not None:
            cap = min(cap, self.max_context)
        return max(_MIN_CTX_SIZE, min(ctx, cap))

    # ── auto-parallel: value-function (ctx, slots) solve ──

    def _serving_pin(self, view) -> int | None:
        """Explicitly pinned serving ctx, or None.

        A sidecar/block ``context_length:`` or the CLI ``--max-context``
        fixes the per-slot size (cascade row 2: a fixed ctx is its own
        floor).  Returns the tighter of the two when both are present.
        """
        pins = []
        sidecar_ctx = (view.frontmatter or {}).get("context_length")
        if sidecar_ctx is not None:
            try:
                pin = int(sidecar_ctx)
                assert pin > 0
            except (TypeError, ValueError, AssertionError):
                logger.warning("context_length=%r on %s is not a positive "
                               "integer; ignoring pin", sidecar_ctx, view.stem)
            else:
                pins.append(pin)
        if self.max_context is not None:
            if self.max_context > 0:
                pins.append(self.max_context)
            else:
                logger.warning("--max-context %d is not positive; ignoring",
                               self.max_context)
        return min(pins) if pins else None

    def _auto_parallel(
        self,
        view,
        *,
        cache_type: str,
        spare_mb: int,
        include_mmproj: bool,
        cap_ctx: int,
        floor: int,
        pin_ctx: int | None,
        group_parallel: int,
        profile: dict | None = None,
    ) -> tuple[int, int]:
        """Solve (parallel, ctx) for one unpinned chat variant.

        Scores ``(ctx/floor)^0.5 × slots^B`` over feasible candidates and
        returns the winner.  Feasibility for a slot count comes from the
        existing budget path (:meth:`_bounded_ctx`); the score rises with
        ctx, so only the max affordable ctx per slot count is scored.  The
        loop breaks at the first unaffordable slot count (affordability
        falls as slots rise).  Falls back to today's outcome (group
        parallel, max affordable ctx) when the floor is unreachable.
        """
        power = self.knobs.parallel_power
        # vLLM's continuous batching makes a slot cost KV only (no per-slot
        # workspace), so concurrency is cheap — cap it higher there.
        pmax = (self.knobs.vllm_auto_parallel_max
                if view.backend in VLLM_BACKENDS
                else self.knobs.auto_parallel_max)

        solved: dict[int, int] = {}

        def maxctx(p: int) -> int:
            # Memoized: the eager fallback and the loop share one solve per
            # slot count, so an unaffordable p=1 warns once, not twice.
            if p not in solved:
                solved[p] = self._bounded_ctx(
                    view, parallel=p, cache_type=cache_type,
                    spare_mb=spare_mb, include_mmproj=include_mmproj,
                    design_ctx=self.chat_ctx, context_length=cap_ctx,
                    profile=profile)
            return solved[p]

        fallback = (group_parallel,
                    min(maxctx(group_parallel), cap_ctx))
        best: tuple[float, int, int] | None = None  # (score, p, ctx)
        for p in range(1, pmax + 1):
            m = maxctx(p)
            if m < floor:
                break
            ctx = min(m, cap_ctx)
            if pin_ctx is not None:
                if m < pin_ctx:
                    continue
                ctx = min(pin_ctx, cap_ctx)
            if ctx < floor:
                continue
            score = parallel_value(ctx, floor, p, power)
            if best is None or score > best[0]:
                best = (score, p, ctx)
        if best is None:
            logger.info("auto-parallel: %s floor %d unreachable; keeping "
                        "parallel=%d", view.stem, floor, group_parallel)
            return fallback
        _, p_star, ctx_star = best
        if (p_star, ctx_star) != fallback:
            logger.info("auto-parallel: %s parallel=%d ctx=%d (floor %d)",
                        view.stem, p_star, ctx_star, floor)
        return p_star, ctx_star

    # ── planning passes ──

    def _mmproj_drop_pass(self) -> dict[str, bool]:
        """Decide per chat model whether the main entry keeps its mmproj.

        A model keeps vision when it reaches the minimum useful context WITH
        the projection loaded; otherwise the main entry drops it (and is
        emitted as ``<id>-text``; a best-effort vision variant is emitted
        alongside) — but only when dropping actually helps: a model whose
        design context is below the minimum even text-only keeps its vision,
        since sacrificing it buys nothing.  Uses the global spare and fleet
        defaults; per-profile spare still bounds ctx per group.
        """
        drop: dict[str, bool] = {}
        global_spare_mb = self.profiles.global_spare_mb(self.spare, self.vram_total)
        for model in self.models:
            if model.role in utils.NON_CHAT_ROLES:
                continue
            if not (model.mmproj and model.mmproj.gguf_path):
                continue
            on_view = model.view_for(True)
            cache_type = on_view.cache_type_for(self.profiles.default_cache_type)
            parallel = on_view.parallel_for(self.profiles.default_parallel)
            ctx_with = self._bounded_ctx(
                on_view, parallel=parallel, cache_type=cache_type,
                spare_mb=global_spare_mb, include_mmproj=True)
            if ctx_with >= self.min_context:
                drop[model.stem] = False
                logger.info("mmproj: keep for %s (ctx %d >= %d)",
                            model.stem, ctx_with, self.min_context)
                continue
            ctx_without = self._bounded_ctx(
                model, parallel=parallel, cache_type=cache_type,
                spare_mb=global_spare_mb, include_mmproj=False)
            if ctx_without < self.min_context:
                # Below the minimum either way — no configuration fixes this;
                # dropping vision would only degrade the model. Keep it.
                logger.info("mmproj: %s design ctx %d is below min-context %d "
                            "with or without vision; keeping vision",
                            model.stem, ctx_with, self.min_context)
                continue
            drop[model.stem] = True
            logger.info("mmproj: drop for %s (vision ctx %d < %d; text ctx %d)",
                        model.stem, ctx_with, self.min_context, ctx_without)
        return drop

    def _solve_matrix(self, drop_stems: set[str]) -> MatrixSolve | None:
        """Shared chat context when a matrix section is configured."""
        if not (self.matrix_cfg and self.embed_model and self.rerank_model):
            return None
        synthetic: dict[str, tuple[int, float, float, int]] = {}
        result = _solve_matrix_context(
            self.models, self.embed_model, self.rerank_model,
            self.fit_bin, self.vram_total, self.spare, self.profiles,
            baseline_mb=self.baseline_mb, drop_stems=drop_stems,
            knobs=self.knobs, memory_margin=self.memory_margin,
            llama_args=self.llama_args, synthetic=synthetic,
            fixed_categories=self.fixed_categories,
        )
        self.synthetic_quads = synthetic
        # Flag the synthesized models so plan() serves them at their
        # minimum useful context and emission marks them unestimated.
        for stem in synthetic:
            for m in self.models:
                if m.stem == stem:
                    m.vram.unestimated_reason = ESTIMATE_ERROR_REASON
                    break
        if result is not None:
            logger.info("matrix: solved chat_ctx=%d (squeeze=%s, coloads=%s)",
                        result.chat_ctx, result.squeeze,
                        [s for s, _ in result.coloads])
        return result

    def _build_ledger(self) -> PoolLedger:
        """Shared VRAM ledger from the matrix result + ``pools:`` overrides.

        Residents = RAG models at their solved contexts plus adopted
        co-loads (the models actually served alongside chat), each charged
        to its device pool with its size bucket logged.  Per-model chat
        solves add this pool's extra reserve to their spare so slots can't
        OOM co-residents.
        """
        ledger = PoolLedger(self.vram_total,
                            self.profiles.pools_cfg)
        res = self.matrix_result
        if res is not None:
            for model, ctx, label in ((self.embed_model, res.embed_ctx,
                                       "embed"),
                                      (self.rerank_model, res.rerank_ctx,
                                       "rerank")):
                if model is None:
                    continue
                mb = self._resident_overhead(model, ctx)
                pool = ledger.pool_id_for(model)
                ledger.add_resident(f"{label}:{model.stem}", mb, pool)
                logger.info("ledger: %s resident %s (%s bucket, pool %s%s)",
                            label, model.stem, footprint_bucket(model), pool,
                            " — no reserve (unestimated)"
                            if model.stem in self.synthetic_quads else "")
            by_stem = {m.stem: m for m in self.models}
            for stem, mb in res.coloads:
                m = by_stem.get(stem)
                pool = ledger.pool_id_for(m) if m is not None else "default"
                ledger.add_resident(f"coload:{stem}", mb, pool)
                logger.info("ledger: co-load resident %s (%s bucket, pool "
                            "%s)", stem,
                            footprint_bucket(m) if m is not None else "?",
                            pool)
        return ledger

    def _resident_overhead(self, model, ctx: int) -> int:
        """VRAM (MB) of a resident model at its served context, 0 on error."""
        if model is None or model.on_cpu:
            return 0
        try:
            quad = _static_params(
                model, self.fit_bin, self.profiles.default_cache_type,
                measurement_args(self.profiles, model, self.llama_args))
            if quad is None:
                # Unestimable resident: charged 0 — the no-expansion rider
                # policy (see _solve_matrix_context).
                quad = self.synthetic_quads.get(model.stem)
            if quad is None:
                return 0
            mib, kv_factor, slot_mib, compute = quad
            parallel = model.parallel_for(self.profiles.default_parallel)
            return mib + compute + int(
                kv_factor * ctx * parallel + slot_mib * parallel)
        except Exception as e:  # keep planning alive; matrix already solved
            logger.warning("ledger: cannot size resident %s (%s)",
                           model.stem, e)
            return 0

    def _variant_memory(
        self,
        view,
        *,
        cache_type: str,
        include_mmproj: bool,
        ctx_size: int,
        parallel: int,
        profile: dict | None,
    ) -> tuple[float | None, float | None]:
        """Decided TOTAL memory (MiB) attributable to one serving variant.

        ``(vram_mib, ram_mib)`` evaluated from the combined affine quad at
        the variant's served ``(ctx_size, parallel)`` — weights + folded
        companions + context — exactly what the emitted description tag
        reports verbatim (no capping, no re-solving).  CPU-resident
        servings return ``(None, ram_mib)`` (host-RAM constants via
        ``allow_cpu`` — never used for VRAM sizing).  ``(None, None)``
        when undecidable (no estimate source worked).  ``parallel <= 0``
        (uncapped vLLM) is priced single-seq, matching its ctx solve.
        """
        quad = view.vram.effective_static(
            self.fit_bin, cache_type=cache_type,
            include_mmproj=include_mmproj,
            llama_args=measurement_args(self.profiles, view,
                                        self.llama_args, profile),
            allow_cpu=True,
        )
        if quad is None:
            return (None, None)
        model_mib, kv_factor, slot_mib, compute_mib = quad
        p = parallel if parallel > 0 else 1
        total = (model_mib + compute_mib
                 + kv_factor * ctx_size * p + slot_mib * p)
        if view.on_cpu:
            return (None, float(total))
        files_mb = view.size_mb
        if include_mmproj and view.mmproj is not None \
                and view.mmproj.gguf_path is not None:
            files_mb += view.mmproj.size_mb
        if view.mtp is not None and view.mtp.gguf_path is not None:
            files_mb += view.mtp.size_mb
        return (float(total), max(0.0, float(files_mb - model_mib)))

    def plan(self) -> dict[str, list[Variant]]:
        """Plan serving variants for every model, keyed by stem.

        Order matters: the mmproj drop decision runs first (its result feeds
        the matrix solve), then per-model variants are grouped by profile.
        The matrix result threads through everywhere: the solved chat context
        clamps chat entries, an adopted emb/rnk squeeze clamps the RAG
        entries' contexts, tools are demoted on chat entries solved below
        ``tools_min_ctx``, and included co-loads are flagged so the emitter
        can expose them for matrix-var construction.
        """
        drop_mmproj = self._mmproj_drop_pass()
        self.matrix_result = self._solve_matrix(
            {s for s, d in drop_mmproj.items() if d})
        if self.matrix_result is not None:
            self.chat_ctx = self.matrix_result.chat_ctx
        coload_stems = ({s for s, _ in self.matrix_result.coloads}
                        if self.matrix_result else set())
        # Shared ledger: matrix residents + pools: overrides.  Per-model
        # chat solves charge this pool's extra reserve to their spare.
        self.ledger = self._build_ledger()

        plan: dict[str, list[Variant]] = {}
        for model in self.models:
            include_mmproj = not drop_mmproj.get(model.stem, False)
            view = model.view_for(include_mmproj)
            on_view = model.view_for(True)
            context_length = view.design_context
            # Unestimable model: no measurement source worked. Serve at the
            # minimum useful context (type-based floor) — the matrix solve
            # carries it with no extra reserve — and mark every entry for
            # client-facing metadata.  The flag may sit on either budget:
            # view_for() rebuilds VramBudget per view (mmproj models carry
            # a separate companion-on budget that runs its own solves).
            est_error = getattr(model.vram, "unestimated_reason", None) \
                or getattr(view.vram, "unestimated_reason", None)
            # Squeeze: an adopted emb/rnk squeeze is realized by clamping the
            # RAG entry's served context (the emit is what frees the VRAM).
            if view.role == "embeddings" and self.matrix_result:
                context_length = min(context_length,
                                     self.matrix_result.embed_ctx)
            elif view.role == "rerank" and self.matrix_result:
                context_length = min(context_length,
                                     self.matrix_result.rerank_ctx)
            tools_demoted = (
                view.role == "chat"
                and self.chat_ctx is not None
                and self.chat_ctx < self.knobs.tools_min_ctx
                and "tools" in [c.lower() for c in view.capabilities])
            is_coload = model.stem in coload_stems

            groups = self.profiles.groups_for(view, self.vram_total, self.spare)
            variants: list[Variant] = []
            for (parallel, cache_type, spare_mb, batch, ubatch), group in groups.items():
                # Ledger charge: this pool's extra reserve (pools: overrides
                # + co-residents) joins the spare for chat solves only — RAG
                # and fixed-overhead roles ARE residents; charging them
                # themselves would double-count.
                pool_id = self.ledger.pool_id_for(view)
                spare_mb = self.ledger.spare_for(pool_id, spare_mb)
                spare_eff = spare_mb
                if view.role == "chat":
                    spare_eff += self.ledger.extra_reserve_mb(pool_id)
                # Auto-parallel: unpinned chat models on slot-based backends
                # get a value-function (parallel, ctx) solve instead of the
                # group parallel.  Pinned = sidecar/block `parallel:` or any
                # profile in the group declaring `parallel`.
                auto = (
                    self.knobs.auto_parallel
                    and view.role == "chat"
                    and view.backend in ("llama-server", "vllm", "vllm-docker")
                    and "parallel" not in (view.frontmatter or {})
                    and not any("parallel" in resolved
                                for _, resolved in group))
                # parallel 0: uncapped vLLM serving (no --max-num-seqs; the
                # shared pool admits elastically, queueing the rest).  The
                # ctx solve stays single-seq — vLLM validates that one
                # max-length sequence fits its pool at startup.  On
                # llama.cpp backends 0 is invalid: warn, use the default.
                uncapped = False
                if parallel <= 0:
                    if view.backend in VLLM_BACKENDS:
                        uncapped = True
                    else:
                        logger.warning(
                            "%s: parallel %d is invalid; using fleet "
                            "default %d", view.stem, parallel,
                            self.profiles.default_parallel)
                        parallel = self.profiles.default_parallel
                group_parallel = parallel
                if uncapped:
                    parallel = 1   # single-seq ctx solve; emission drops it
                if est_error is not None:
                    ctx_size = self._unestimated_ctx(view)
                elif auto:
                    pin_ctx = self._serving_pin(view)
                    floor = resolve_min_ctx(
                        view, pin_ctx=pin_ctx,
                        tools_min_ctx=self.knobs.tools_min_ctx,
                        fallback_min_ctx=self.min_context,
                        fallback_explicit=self.min_context_explicit)
                    cap_ctx = context_length
                    if self.max_context is not None:
                        cap_ctx = min(cap_ctx, self.max_context)
                    parallel, ctx_size = self._auto_parallel(
                        view, cache_type=cache_type, spare_mb=spare_eff,
                        include_mmproj=include_mmproj, cap_ctx=cap_ctx,
                        floor=floor, pin_ctx=pin_ctx,
                        group_parallel=parallel, profile=group[0][1])
                else:
                    ctx_size = self._bounded_ctx(
                        view, parallel=parallel, cache_type=cache_type,
                        spare_mb=spare_eff, include_mmproj=include_mmproj,
                        design_ctx=self.chat_ctx,
                        context_length=context_length,
                        profile=group[0][1])
                if est_error is None:
                    # Discovery during this model's own solve: calc_ctx
                    # flagged the model unestimable (no measurement source
                    # worked) — re-plan it at the type floor.  The flag may
                    # sit on the *view's* budget: view_for() rebuilds
                    # VramBudget per view, and the solve ran on `view`.
                    est_error = getattr(model.vram, "unestimated_reason",
                                        None) \
                        or getattr(view.vram, "unestimated_reason", None)
                    if est_error is not None:
                        parallel = group_parallel
                        ctx_size = self._unestimated_ctx(view)

                vision_ctx: int | None = None
                if not include_mmproj and model.mmproj and model.mmproj.gguf_path:
                    vision_ctx = ctx_size if est_error is not None else (
                        self._bounded_ctx(
                            on_view, parallel=parallel, cache_type=cache_type,
                            spare_mb=spare_eff, include_mmproj=True,
                            design_ctx=self.chat_ctx,
                            context_length=context_length,
                            profile=group[0][1]))
                if uncapped:
                    parallel = 0   # emit uncapped: no --max-num-seqs

                if est_error is not None:
                    mem: tuple[float | None, float | None] = (None, None)
                    vmem: tuple[float | None, float | None] = (None, None)
                else:
                    mem = self._variant_memory(
                        view, cache_type=cache_type,
                        include_mmproj=include_mmproj, ctx_size=ctx_size,
                        parallel=parallel, profile=group[0][1])
                    vmem = self._variant_memory(
                        on_view, cache_type=cache_type,
                        include_mmproj=True, ctx_size=vision_ctx,
                        parallel=parallel, profile=group[0][1]) \
                        if vision_ctx is not None else (None, None)

                variants.append(Variant(
                    parallel=parallel, cache_type=cache_type, spare_mb=spare_mb,
                    profiles_group=group, ctx_size=ctx_size,
                    include_mmproj=include_mmproj, batch=batch, ubatch=ubatch,
                    vision_ctx=vision_ctx,
                    coload=is_coload, tools_demoted=tools_demoted,
                    estimate_error=est_error,
                    mem_vram_mib=mem[0], mem_ram_mib=mem[1],
                    mem_vision_vram_mib=vmem[0],
                    mem_vision_ram_mib=vmem[1]))

                # On-demand text-only variant: when the main entry keeps its
                # mmproj, also plan a no-vision entry (``<id>-text``) so
                # clients can pick the lower-memory serving.  When the main
                # entry was auto-dropped it IS the ``-text`` entry, so no
                # separate variant is needed.
                if (include_mmproj and view.role == "chat"
                        and model.mmproj and model.mmproj.gguf_path):
                    text_ctx = ctx_size if est_error is not None else (
                        self._bounded_ctx(
                            model, parallel=parallel, cache_type=cache_type,
                            spare_mb=spare_eff, include_mmproj=False,
                            design_ctx=self.chat_ctx,
                            context_length=context_length,
                            profile=group[0][1]))
                    tmem: tuple[float | None, float | None] = \
                        (None, None) if est_error is not None else \
                        self._variant_memory(
                            model, cache_type=cache_type,
                            include_mmproj=False, ctx_size=text_ctx,
                            parallel=parallel, profile=group[0][1])
                    variants.append(Variant(
                        parallel=parallel, cache_type=cache_type, spare_mb=spare_mb,
                        profiles_group=group, ctx_size=text_ctx,
                        include_mmproj=False, batch=batch, ubatch=ubatch,
                        coload=is_coload,
                        tools_demoted=tools_demoted,
                        estimate_error=est_error,
                        mem_vram_mib=tmem[0], mem_ram_mib=tmem[1]))
            plan[model.stem] = variants
            if self.progress_cb is not None:
                self.progress_cb(model.stem)
        return plan


def _static_params(model: Model, fit_bin: str, cache_type: str,
                   llama_args: str = "",
                   ) -> tuple[int, float, float, int] | None:
    """(model_mib, kv_per_token_mib, slot_mib, compute_mib) quad, or None
    when unmeasurable."""
    fp = model.vram.fit_params_static(fit_bin, cache_type=cache_type,
                                      llama_args=llama_args)
    if fp is None:
        return None
    return (fp.model_mib, fp.kv_per_token_mib, fp.slot_mib, fp.compute_mib)


def _solve_matrix_context(
    chat_models: list[Model],
    embed_model: Model,
    rerank_model: Model,
    fit_bin: str,
    vram_total: int,
    spare: str | None,
    profiles: Profiles,
    baseline_mb: int = 0,
    drop_stems: set[str] | None = None,
    knobs: MatrixKnobs | None = None,
    memory_margin: float = _MEMORY_MARGIN,
    llama_args: str = "",
    synthetic: dict[str, tuple[int, float, float, int]] | None = None,
    fixed_categories: list[tuple[str, Model]] | None = None,
) -> MatrixSolve | None:
    """Solve the shared VRAM budget for chat context plus co-loads.

    Runs the affine measurement once per model to get static constants, then
    solves:
        available = Σ(chat_weight + chat_kv*slots*chat_ctx + slots*chat_slot)
                  + (embed_weight + embed_kv*embed_ctx + embed_slot)
                  + (rerank_weight + rerank_kv*rerank_ctx + rerank_slot)

    Three passes, in order:

    1. *Baseline*: embed/rerank at their configured context (``rag_cap``
       capped by design context) → ``chat_ctx₀``.
    2. *Squeeze* (§2b): when ``chat_ctx₀`` is below ``tools_min_ctx``, re-solve
       with embed/rerank contexts clamped to ``coload_min_ctx``; adopt only
       when the gain reaches ``ctx_gain_min``.
    3. *Opportunistic co-loads* (§2): enabled ``s2t``/``image`` models not on
       the GPU pool's excluded list, smallest fixed overhead first, are
       included while the chat solve stays at or above the floor
        (``tools_min_ctx`` when a tools chat model can keep it, else
        ``min_chat_ctx``).  Estimated candidates carry ``estimate_headroom``.

    Unestimable participants (every estimate source failed) stay in the
    solve with a zero-cost placeholder — charged nothing extra — served at
    their minimum useful context, assumed to fit within the set's measured
    allocation. Synthesized stems are reported through ``synthetic`` (stem
    → quad) so the planner can flag the models and charge the ledger the
    same numbers.

    Returns the :class:`MatrixSolve` (chat context, adopted RAG contexts,
    included co-loads) or None on failure.
    """
    knobs = knobs or MatrixKnobs()
    spare_mb = parse_spare_mb(spare, vram_total)

    drop_stems = drop_stems or set()
    fixed_categories = fixed_categories or []

    # Get static params for chat models (companion VRAM folded in).
    # The drop decision (mmproj skipped to reach the min useful context) is
    # decided in Planner._mmproj_drop_pass and threaded in via drop_stems.
    # Two-phase: measure everything first, then give models with no
    # estimate a zero-cost placeholder — the rider stays in the set at its
    # minimum useful context and is assumed to fit the measured allocation
    # without expanding it (no extra reserve).
    chat_params: list[tuple[Model, int, float, float, int, int, int]] = []
    chat_meta: list[tuple[Model, int, int, tuple[int, float, float, int] | None]] = []
    real_quads: list[tuple[int, float, float, int]] = []
    for m in chat_models:
        # Embed/rerank/image/s2t models are handled outside the shared chat
        # budget (fixed overhead / separate pool). Including a 40 GB
        # diffusion model would collapse the chat budget, so exclude it.
        if m.role in utils.NON_CHAT_ROLES or m.on_cpu:
            continue
        # Overlay keys (cache_type/parallel/context_length/image_max_tokens)
        # live in the block and are only visible on the companion-on view.
        on = m.view_for(m.stem not in drop_stems)
        cache_type = on.cache_type_for(profiles.default_cache_type)
        parallel = on.parallel_for(profiles.default_parallel)
        if parallel <= 0:
            # Uncapped vLLM: the shared solve is single-seq safe (its own
            # emission solves ctx for one max-length sequence).  On
            # llama.cpp backends 0 is invalid — warn, use the default.
            if on.backend in VLLM_BACKENDS:
                parallel = 1
            else:
                logger.warning("%s: parallel %d is invalid; using default",
                               on.stem, profiles.default_parallel)
                parallel = profiles.default_parallel
        img_floor = 0
        if (m.stem not in drop_stems and m.mmproj and m.mmproj.gguf_path
                and on.image_max_tokens):
            img_floor = on.image_max_tokens
        fp = on.vram.effective_static(fit_bin, cache_type=cache_type,
                                      include_mmproj=m.stem not in drop_stems,
                                      llama_args=measurement_args(profiles, m,
                                                                  llama_args))
        if fp is not None:
            real_quads.append(fp)
        chat_meta.append((m, parallel, img_floor, fp))

    # Get static params for embed/rerank. CPU-resident models cost 0 VRAM.
    embed_params = None
    if not embed_model.on_cpu:
        embed_params = _static_params(embed_model, fit_bin,
                                      profiles.default_cache_type,
                                      measurement_args(profiles, embed_model,
                                                       llama_args))
    rerank_params = None
    if not rerank_model.on_cpu:
        rerank_params = _static_params(rerank_model, fit_bin,
                                       profiles.default_cache_type,
                                       measurement_args(profiles, rerank_model,
                                                        llama_args))
    for quad in (embed_params, rerank_params):
        if quad is not None:
            real_quads.append(quad)

    if not chat_meta:
        # No chat participants: the shared solve has nothing to solve.
        return None
    if not real_quads:
        logger.warning("matrix: no measurable model to bound unestimated "
                       "participants; aborting shared solve")
        return None

    def _with_bound(quad: tuple[int, float, float, int] | None,
                    stem: str) -> tuple[int, float, float, int]:
        """Measured quad, or a zero-cost placeholder for an unestimable one.

        Policy: the rider stays in the set at its minimum useful context and
        is assumed to fit within the set's measured allocation — it does not
        expand it (no extra reserve). Runtime OOM on such a rider is the
        accepted risk; the entry is flagged ``estimated: false``.
        """
        if quad is not None:
            return quad
        if synthetic is not None:
            synthetic[stem] = (0, 0.0, 0.0, 0)
        logger.warning(
            "matrix: %s has no VRAM estimate; serving it in the set at its "
            "minimum useful context with no extra reserve (assumed to fit "
            "the measured allocation)", stem)
        return (0, 0.0, 0.0, 0)

    for m, parallel, img_floor, fp in chat_meta:
        quad = _with_bound(fp, m.stem)
        chat_params.append((m, quad[0], quad[1], quad[2], quad[3],
                            parallel, img_floor))

    if embed_params is None and not embed_model.on_cpu:
        embed_params = _with_bound(None, embed_model.stem)
    if rerank_params is None and not rerank_model.on_cpu:
        rerank_params = _with_bound(None, rerank_model.stem)

    # Reserve for embed/rerank at their configured context (the
    # ``embed_context`` / ``rerank_context`` knobs), capped by the model's
    # declared context (sidecar context_length > GGUF architectural max) —
    # a 32k reranker costs ~4x the KV of an 8k one and must be budgeted as
    # declared.  Unestimable RAG residents ride free (zero-cost placeholder,
    # like unestimable chat riders) and are served at the same configured
    # context — the operator's contract, accepted unverified.
    embed_ctx = min(knobs.rag_cap("embeddings"),
                    embed_model.design_context)
    rerank_ctx = min(knobs.rag_cap("rerank"), rerank_model.design_context)

    def _solve(embed_ctx_: int, rerank_ctx_: int,
               fixed_overhead_mb: int = 0) -> int:
        return solve_matrix_ctx(
            vram_total_mb=vram_total,
            spare_mb=spare_mb,
            chat_models=chat_params,
            embed_params=embed_params,
            rerank_params=rerank_params,
            embed_ctx=embed_ctx_,
            rerank_ctx=rerank_ctx_,
            baseline_mb=baseline_mb,
            fixed_overhead_mb=fixed_overhead_mb,
            memory_margin=memory_margin,
        )

    # 1. Baseline.
    chat_ctx = _solve(embed_ctx, rerank_ctx)

    # 2. emb/rerank squeeze: when chat falls below the tools threshold, the
    #    RAG models yield context (down to coload_min_ctx) to buy it back.
    squeeze = False
    if chat_ctx < knobs.tools_min_ctx:
        sq_embed = min(embed_ctx, knobs.coload_min_ctx)
        sq_rerank = min(rerank_ctx, knobs.coload_min_ctx)
        if (sq_embed, sq_rerank) != (embed_ctx, rerank_ctx):
            sq_ctx = _solve(sq_embed, sq_rerank)
            gain = sq_ctx - chat_ctx
            if gain >= knobs.ctx_gain_min:
                logger.info(
                    "matrix: squeeze adopted (embed/rerank -> %d/%d, "
                    "chat %d -> %d, gain %d)",
                    sq_embed, sq_rerank, chat_ctx, sq_ctx, gain)
                chat_ctx, embed_ctx, rerank_ctx = sq_ctx, sq_embed, sq_rerank
                squeeze = True
            else:
                logger.info(
                    "matrix: squeeze rejected (gain %d < ctx_gain_min %d)",
                    gain, knobs.ctx_gain_min)

    # 3. Opportunistic co-loads: enabled s2t/image models, smallest first.
    #    Floor: keep tools_min_ctx for a tools chat model that still has it;
    #    otherwise min_chat_ctx.  (When the baseline is already below the
    #    tools threshold, tools are demoted downstream and the floor is the
    #    co-load floor.)
    any_tools = any(
        "tools" in [c.lower() for c in m.capabilities]
        for m, *_ in chat_params)
    floor = knobs.tools_min_ctx if (any_tools and chat_ctx >= knobs.tools_min_ctx) \
        else knobs.min_chat_ctx
    coloads: list[tuple[str, int]] = []
    if chat_ctx >= floor:
        declared_stems = {m.stem for _, m in fixed_categories}
        used = 0

        def _reserve(oh: int, stem: str, declared: bool) -> int | None:
            """Reserve *oh* MB if chat stays at/above the floor; else None."""
            nonlocal used
            ctx = _solve(embed_ctx, rerank_ctx, fixed_overhead_mb=used + oh)
            if ctx < floor:
                return None
            used += oh
            if not declared:
                coloads.append((stem, oh))
            return ctx

        # 3a. Declared categories (explicit operator intent, e.g. tts/stt)
        #     are reserved first; one that does not fit is reported and left
        #     unreserved.
        for name, m in fixed_categories:
            oh = _coload_overhead(m, fit_bin, profiles, knobs, llama_args)
            if oh is None:
                logger.warning("matrix: category %r (%s) skipped: cannot size it",
                               name, m.stem)
                continue
            ctx = _reserve(oh, m.stem, declared=True)
            if ctx is None:
                logger.warning(
                    "matrix: category %r (%s) does not fit below the chat "
                    "floor %d; not reserved", name, m.stem, floor)
            else:
                logger.info(
                    "matrix: category %r (%s) reserved (%d MB, chat_ctx=%d)",
                    name, m.stem, oh, ctx)

        # 3b. Opportunistic co-loads: s2t/image models that are *not* already
        #     a declared category, smallest fixed overhead first.
        overheads: list[tuple[int, str, Model]] = []
        for m in chat_models:
            if m.role not in ("s2t", "image") or m.stem in declared_stems:
                continue
            oh = _coload_overhead(m, fit_bin, profiles, knobs,
                                  llama_args)
            if oh is None:
                logger.warning("matrix: co-load %s skipped: cannot size it",
                               m.stem)
                continue
            overheads.append((oh, m.stem, m))
        for oh, stem, m in sorted(overheads, key=lambda t: t[0]):
            ctx = _reserve(oh, stem, declared=False)
            if ctx is None:
                logger.warning(
                    "matrix: co-load %s skipped: would drop chat ctx below "
                    "the floor %d", stem, floor)
            else:
                logger.info("matrix: co-load %s included (%d MB, chat_ctx=%d)",
                            stem, oh, ctx)
    return MatrixSolve(
        chat_ctx=chat_ctx, embed_ctx=embed_ctx, rerank_ctx=rerank_ctx,
        coloads=tuple(coloads), squeeze=squeeze,
    )


def _coload_overhead(
    m: Model, fit_bin: str, profiles: Profiles, knobs: MatrixKnobs,
    llama_args: str = "",
) -> int | None:
    """Fixed VRAM overhead (MB) of an opportunistic co-load candidate.

    Uses the backend's effective static params (weights + fixed compute; the
    KV terms are 0 for these backends).  Overheads that are *estimated*
    rather than measured or pinned carry ``estimate_headroom`` so a bad
    guess errs toward reserving more.
    """
    if m.on_cpu:
        return 0
    # An operator-pinned vram_mb is authoritative — no headroom.
    pinned = m.frontmatter.get("vram_mb") is not None
    cache_type = m.cache_type_for(profiles.default_cache_type)
    fp = m.vram.fit_params_static(fit_bin, cache_type=cache_type,
                                  llama_args=measurement_args(profiles, m,
                                                              llama_args))
    measured = pinned or (fp is not None
                          and fp.source in ("llama-server", "vllm-estimate"))
    quad = m.vram.effective_static(fit_bin, cache_type=cache_type,
                                   llama_args=measurement_args(profiles, m,
                                                               llama_args))
    if quad is None:
        return None
    model_mib, kv_factor, slot_mib, compute_mib = quad
    # These backends have zero KV terms; nonzero factors would mean a
    # context-driven model wrongly landed in the pool — charge design ctx.
    overhead = model_mib + compute_mib + int(
        kv_factor * m.design_context + slot_mib)
    if not measured:
        overhead = int(overhead * knobs.estimate_headroom)
    return overhead


# ── Emission ──────────────────────────────────────────────────────────────


def emit_config(models: list[Model], plan: dict[str, list[Variant]],
                profiles: Profiles, template_vars: dict) -> EmittedConfig:
    """Render planned :class:`Variant`s into the llama-swap config dict.

    Pure transformation — no VRAM math, no I/O. Each variant becomes one
    entry.  Id invariant: the bare ``<id>`` always serves vision when the
    model has an mmproj — every no-mmproj variant is emitted as
    ``<id>`` + :data:`TEXT_SUFFIX` (name suffix ``[text]``), whether it is
    the on-demand text-only variant or the main entry of an auto-dropped
    model.  A variant with ``vision_ctx`` additionally emits a best-effort
    vision companion entry, id-suffixed ``-vision-<N>k`` where
    ``N = vision_ctx // 1000`` (e.g. 92567 → ``-vision-92k``).

    Raises ValueError on duplicate entry ids (two models slugging to the same
    id); callers surface it as a fatal configuration error.  Returns an
    :class:`EmittedConfig` whose ``entry_ids_by_stem`` maps each model stem to
    the ids it produced.
    """
    entries: dict[str, dict] = {}
    owner: dict[str, str] = {}  # entry id → model stem (collision detection)
    ids_by_stem: dict[str, list[str]] = {}
    coload_stems: set[str] = set()
    for model in models:
        for v in plan.get(model.stem, []):
            view = model.view_for(v.include_mmproj)
            context_length = view.design_context
            text_only = not v.include_mmproj
            if v.coload:
                coload_stems.add(model.stem)
            entry_id, entry = _build_entry(
                view, v.parallel, v.cache_type, v.profiles_group,
                profiles.defaults, template_vars, context_length, v.ctx_size,
                include_mmproj=v.include_mmproj,
                name_suffix=" [text]" if text_only else "",
                tools_demoted=v.tools_demoted,
                estimate_error=v.estimate_error,
                batch=v.batch, ubatch=v.ubatch,
                mem_vram_mib=v.mem_vram_mib, mem_ram_mib=v.mem_ram_mib,
            )
            if text_only:
                entry_id += TEXT_SUFFIX
            if entry_id in entries:
                raise ValueError(
                    f"duplicate entry id {entry_id!r}: model {model.stem!r} "
                    f"collides with {owner[entry_id]!r} — rename one of them")
            entries[entry_id] = entry
            owner[entry_id] = model.stem
            ids_by_stem.setdefault(model.stem, []).append(entry_id)

            if v.vision_ctx is None:
                continue
            n_k = v.vision_ctx // 1000
            on_view = model.view_for(True)
            vision_id, vision_entry = _build_entry(
                on_view, v.parallel, v.cache_type, v.profiles_group,
                profiles.defaults, template_vars, on_view.design_context,
                v.vision_ctx,
                include_mmproj=True,
                name_suffix=f" [vision {n_k}k]",
                tools_demoted=v.tools_demoted,
                estimate_error=v.estimate_error,
                batch=v.batch, ubatch=v.ubatch,
                mem_vram_mib=v.mem_vision_vram_mib,
                mem_ram_mib=v.mem_vision_ram_mib,
            )
            vision_id += f"-vision-{n_k}k"
            if vision_id in entries:
                raise ValueError(
                    f"duplicate entry id {vision_id!r}: model {model.stem!r} "
                    f"collides with {owner[vision_id]!r} — rename one of them")
            entries[vision_id] = vision_entry
            owner[vision_id] = model.stem

    config = EmittedConfig(entry_ids_by_stem=ids_by_stem,
                           coload_stems=sorted(coload_stems))
    config["models"] = {
        eid: entries[eid]
        for eid in sorted(entries, key=lambda e: (e.count("."), e))
    }
    # Present setParamsByID aliases (e.g. "<id>:<mode>") in the /v1/models
    # listing so dynamic-list clients (OpenWebUI, OpenClaw, ...) can select
    # them. Default in llama-swap is false and aliases would be invisible.
    config["includeAliasesInList"] = True
    return config


class EmittedConfig(dict):
    """Emitted llama-swap config plus the stem → emitted entry-ids mapping.

    A plain ``dict`` everywhere YAML/serialization is concerned (convert with
    ``EmittedConfig.plain()`` before dumping — a dict subclass would otherwise
    emit a ``!!python/object`` tag); carries ``entry_ids_by_stem`` so callers
    (e.g. matrix-var construction) can map a model to the entry ids it
    actually produced instead of re-deriving id naming conventions, and
    ``coload_stems`` — the opportunistically included non-chat models (see
    ``MatrixSolve``) — for the same purpose.
    """

    def __init__(self, *args, entry_ids_by_stem: dict[str, list[str]] | None = None,
                 coload_stems: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_ids_by_stem: dict[str, list[str]] = entry_ids_by_stem or {}
        self.coload_stems: list[str] = coload_stems or []

    def plain(self) -> dict:
        return dict(self)


def build_config(
    models: list[Model],
    profiles_cfg: dict | Profiles,
    template_vars: dict,
    fit_bin: str,
    vram_total: int,
    spare: str | None = None,
    max_context: int | None = None,
    matrix_cfg: dict | None = None,
    embed_model: Model | None = None,
    rerank_model: Model | None = None,
    fixed_categories: list[tuple[str, Model]] | None = None,
    baseline_mb: int = 0,
    min_context: int = _MIN_AGENTIC_CTX,
    min_context_explicit: bool = False,
    memory_margin: float = _MEMORY_MARGIN,
    progress_cb=None,
) -> EmittedConfig:
    """Build llama-swap config from list of Model objects.

    Composes the pipeline: validate/filter models, plan serving variants
    (:class:`Planner`), render entries (:func:`emit_config`).

    Args:
        models: List of Model instances
        profiles_cfg: Full profiles.yaml config (or a prepared Profiles object)
        template_vars: Template variables (llama_bin, models_dirs)
        fit_bin: Path to the llama-server binary (measurement runs)
        vram_total: Total VRAM in MB
        spare: Global spare VRAM string (overridden by profile.spare if present)
        max_context: Hard cap on context length
        matrix_cfg: Matrix configuration for embed/rerank context solving
        embed_model: Embedding model (if matrix configured)
        rerank_model: Reranking model (if matrix configured)
        fixed_categories: Declared matrix categories beyond the RAG pair
            (name, model), e.g. tts/stt — reserved as fixed-overhead
            residents alongside chat and RAG (matrix categories).
        baseline_mb: Driver/compositor VRAM already in use (added to reserve)
        min_context: Minimum useful context for chat models. When a chat model
            with an mmproj companion cannot reach this WITH vision, the vision
            projection is dropped from the main entry, which is renamed
            ``<id>-text`` (a ``vision-<N>k`` variant is emitted alongside,
            still exposing vision at best-effort context).  Pass
            ``min_context_explicit=True`` when the value came from an
            explicit ``--min-context`` flag (as opposed to the default): the
            explicit flag overrides the per-model floor cascade's default
            rows, the default does not.
        memory_margin: Fraction inflated against every measured VRAM term
            when solving (accuracy safety against measurement residual).
    """
    profiles = profiles_cfg if isinstance(profiles_cfg, Profiles) else Profiles(profiles_cfg)

    # Validate BEFORE any VRAM work: rejected models must not consume
    # fit-params runs or pollute the shared matrix context solve.
    supported = _filter_supported(models, profiles.default_cache_type)

    planner = Planner(
        supported, profiles, fit_bin, vram_total,
        spare=spare, max_context=max_context,
        matrix_cfg=matrix_cfg, embed_model=embed_model,
        rerank_model=rerank_model, fixed_categories=fixed_categories,
        baseline_mb=baseline_mb,
        min_context=min_context, min_context_explicit=min_context_explicit,
        memory_margin=memory_margin,
        llama_args=template_vars.get("llama_args", ""),
        progress_cb=progress_cb,
    )
    return emit_config(supported, planner.plan(), profiles, template_vars)


def write_yaml(config: dict, path: Path | str) -> None:
    """Write config to YAML file."""
    payload = config.plain() if isinstance(config, EmittedConfig) else config
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
