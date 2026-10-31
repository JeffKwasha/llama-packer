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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Container, Iterable, Mapping

import yaml

from llama_packer.model import Model
from llama_packer.vram import solve_matrix_ctx
from llama_packer.hardware import detect_gpu_env_var
from llama_packer import utils
from llama_packer.endpoints import CHAT
from llama_packer.tasks import Task, TaskCatalogue, resolve_tasks
from llama_packer.consts import (
    DEFAULT_SACRIFICE_ORDER,
    KNOWN_CAPABILITIES,
    _DIFFUSION_ARCH_RES,
    _KV_CACHE_BYTES,
    _LLAMA_CPP_DEFAULT_UBATCH,
    _MEMORY_MARGIN,
    _MIN_AGENTIC_CTX,
    _MIN_CTX_SIZE,
    _RESERVE_VIDEO,
    ESTIMATE_ERROR_REASON,
)
from llama_packer import roles
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
    """Collapse runs of whitespace to single spaces (templates with | blocks).

    Newlines are preserved: multi-line values (e.g. audio-cpp's heredoc
    ``cmd``) depend on them.
    """
    return "\n".join(" ".join(line.split()) for line in text.split("\n"))


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
        backend.warn_unhandled(declared - FRAMEWORK_CONSUMED - METADATA_ONLY,
                               stem=model.stem)
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
    entry_id_tail: str = "",
    tools_demoted: bool = False,
    task_demands: tuple["Task", ...] = (),
    estimate_error: str | None = None,
    batch: int | None = None,
    ubatch: int | None = None,
    mem_vram_mib: float | None = None,
    mem_ram_mib: float | None = None,
    kv_per_token_mib: float | None = None,
    vram_note: str | None = None,
) -> tuple[str, dict]:
    """Build a single llama-swap config entry for a model+profile group.

    ``include_mmproj=False`` omits the vision projection from the command,
    removes the ``image``/``video`` input capabilities, and flags
    ``name_suffix`` is appended to the model display name.
    ``entry_id_tail`` is the already-disambiguated ``.<category>.<flags>``
    segment; it is folded into the returned entry id **here** rather than by
    the caller so the id is final before the launch command is composed —
    backends that route on the requested model id (audio.cpp) embed it in
    their own config payload and must see exactly the id the entry is
    published under.

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
    ``vram_note`` (when set) is appended after the mem tag: the entry
    serves below its design context because ``--spare``/``--baseline``
    constrained the budget.
    """
    model = model.view_for(include_mmproj)
    base_id = model.template_id

    # The entry id is settled before the command is composed: the category/
    # flags tail is passed in already disambiguated, so the backend gets the
    # exact string this entry will be published under (audio.cpp validates
    # the requested model id against its own config and 500s on a mismatch).
    names = [p[0] for p in profiles_group]
    has_default = "default" in names
    entry_id = base_id if (has_default or len(profiles_group) > 1) else f"{base_id}.{names[0]}"
    if entry_id_tail:
        entry_id = f"{entry_id}.{entry_id_tail}"

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
        entry_id=entry_id,
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
    desc = _with_mem_tag(model.description, mem_tag)
    if vram_note:
        desc = f"{desc} {vram_note}".strip() if desc else vram_note
    if desc is not None:
        entry["description"] = desc
    # capabilities.context carries the served window (what clients size
    # against); ctx_size is the long-standing metadata spelling of the same
    # number, kept for clients that read it. context_max preserves the
    # trained maximum when the entry is folded below it.
    metadata["ctx_size"] = ctx_size
    if model.role == "chat":
        # The served window is the true generation ceiling: llama-server
        # allows a reply up to n_ctx minus the prompt, and clients clamp to
        # the room a prompt leaves. A sidecar may lower it for a model whose
        # long replies degrade, but it can never exceed the window.
        ceiling = metadata.get("max_output_tokens")
        if ceiling is not None and (
                not isinstance(ceiling, int) or isinstance(ceiling, bool)
                or ceiling <= 0):
            logger.warning(
                "%s: max_output_tokens=%r is not a positive integer; using "
                "the served window", model.label, ceiling)
            ceiling = None
        metadata["max_output_tokens"] = min(ceiling or ctx_size, ctx_size)
    if context_length != ctx_size:
        metadata["context_max"] = context_length
    # Client-visible packing facts (docs/plans/cpu-residency-budget.md
    # §metadata.packing): only what was decided at pack time. memory.packing
    # dicts go inside metadata so clients can filter by placement.
    packing: dict = {"device": "cpu" if model.on_cpu else "gpu"}
    if model.on_cpu:
        if mem_ram_mib is not None:
            packing["ram_mib"] = int(round(mem_ram_mib))
    elif mem_vram_mib is not None:
        packing["vram_mib"] = int(round(mem_vram_mib))
        if mem_ram_mib is not None:
            packing["ram_mib"] = int(round(mem_ram_mib))
    packing["ctx_size"] = ctx_size
    packing["parallel"] = parallel
    if kv_per_token_mib is not None:
        packing["kv_per_token"] = int(round(kv_per_token_mib * 1024 * 1024))
    metadata["packing"] = packing
    # The tasks this serving actually satisfies — the same set the id's
    # task segment names.  Client-facing so a metadata-aware client can
    # filter without parsing the id.
    if task_demands:
        metadata["tasks"] = [d.id for d in task_demands]
    # Client-facing estimate health: when no VRAM estimate source worked,
    # the entry is served at its minimum useful context with a conservative
    # matrix reserve — tell clients so they can treat it differently.
    if estimate_error is not None:
        metadata["estimated"] = False
        metadata["estimate_error"] = estimate_error
    if metadata:
        entry["metadata"] = metadata

    # Native llama-swap capabilities block (shown in /v1/models).
    # `context` is the served per-slot window: llama-swap documents it as the
    # window one request can use, and clients derive their compaction and
    # output limits from it. The trained maximum is metadata.context_max.
    entry["capabilities"] = {
        "in": in_mods,
        "out": out_mods,
        "tools": "tools" in [c.lower() for c in caps],
        "reranker": "reranker" in caps_l or model.role == "rerank",
        "context": ctx_size,
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

    # CPU entries never idle-expire (design decision 12): a host-resident
    # helper is cheap to keep and expensive to reload (process start +
    # page-cache warmup), so ttl 0 keeps it up between requests. Device
    # entries keep the inherited/global ttl.
    if model.on_cpu:
        entry["ttl"] = 0

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


#: Reserved key under which :meth:`Planner.plan` carries the per-set
#: admission map for the emitter.  ``emit_config`` is a pure function of the
#: plan dict, so planner state that the emitter needs rides along under a
#: key no model stem can collide with (NUL is not valid in a stem).
SET_ADMISSION_KEY = "\x00set_admission"
HOST_REFUSED_KEY = "\x00host_refused"


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
        """Configured context cap for a role with a resident context policy.

        The served RAG context is ``min(rag_cap(role), design_context)``
        everywhere; a role without a knob (and an unknown role) falls back
        to ``coload_min_ctx``.
        """
        knobs = roles.get(role)
        if knobs is None or knobs.ctx_knob is None:
            return self.coload_min_ctx
        return getattr(self, knobs.ctx_knob)


def _parse_ctx_size(v: object) -> int | None:
    """Parse a context minimum: a positive int, or a ``128k``/``1M`` string."""
    if isinstance(v, str):
        s = v.strip().lower().rstrip("k").rstrip("m")
        mult = 1024 if v.strip().lower().endswith("k") else (
            1024 * 1024 if v.strip().lower().endswith("m") else 1)
        try:
            n = int(s)
        except ValueError:
            return None
        return n * mult
    if isinstance(v, bool):
        return None
    if not isinstance(v, (int, float)):
        return None
    n = int(v)
    return n if n > 0 else None


def _parse_capability_list(v: object) -> tuple[str, ...] | None:
    """Parse a capability name list into a de-duplicated ordered tuple.

    Order is preserved: ``sacrifice`` is a preference order and the ladder
    walks it in the order written.
    """
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple, set)):
        return None
    out: list[str] = []
    for item in v:
        if not isinstance(item, str):
            return None
        name = item.strip().lower()
        if name and name not in out:
            out.append(name)
    return tuple(out)


def shed_levels(sacrifice: tuple[str, ...] | None,
                default_order: tuple[str, ...] = DEFAULT_SACRIFICE_ORDER,
                required: frozenset[str] = frozenset(),
                kept: frozenset[str] = frozenset(),
                ) -> tuple[frozenset[str], ...]:
    """Degradation levels to try, least-shed first.

    One level per prefix of the declared order: ``[]``, ``{vision}``,
    ``{vision, draft}``. Cumulative because each step is a further
    concession — but bounded by the prefix, so a capability the category
    never named is only reached once the named ones are gone. Anything in
    ``required`` (must have it) or ``keep`` (do not drop what this model has)
    is never shed at any level.

    ``None`` means *never degrade* and yields the single empty rung: that is
    the null category, which has no goals to protect and so keeps mmproj and
    MTP even when that costs it all of its context. An empty tuple is
    different — a category that expressed no preference is still willing to
    concede in the default order.

    This replaces the old fleet-wide ``_mmproj_drop_pass``, which decided
    vision from a global context threshold before any category was read.
    """
    if sacrifice is None:
        return (frozenset(),)
    protected = required | kept
    order = [c for c in sacrifice
             if c in KNOWN_CAPABILITIES and c not in protected]
    order += [c for c in default_order
              if c in KNOWN_CAPABILITIES and c not in protected and c not in order]
    levels = [frozenset()]
    for cap in order:
        levels.append(levels[-1] | {cap})
    return tuple(levels)


#: The null category: every model's baseline serving. Bare minimums only and
#: no category name in the id. It keeps mmproj and MTP whenever it can — with
#: no goals to protect, capability is the only thing left to give up — and
#: concedes a capability only when holding it would leave no legal serving at
#: all. An empty ``sacrifice`` says exactly that: default-order shedding,
#: gated by the minimum, and the first qualifying rung wins rather than the
#: one buying the most context (a category wants context; this wants
#: capability).
#:
#: It joins no set. Routing is what categories are for, and this has no name
#: to route on.
FALLBACK: "Task" = Task("", CHAT, "chat", min_context=_MIN_CTX_SIZE,
                        min_parallel=1, require=frozenset(), sacrifice=())


@dataclass(frozen=True)
class Rung:
    """One rung of the capability/context ladder: a set of shed capabilities.

    ``shed`` is the set of capabilities deliberately given up at this rung
    (empty = the model runs at full capability). ``ctx``/``parallel`` are
    what the budget affords *at this rung*, and ``demand`` the category being
    served -- singular, because a rung exists to answer one category.
    """

    shed: frozenset[str]
    ctx: int
    parallel: int
    demand: Task





CHAT_VARS_PLACEHOLDER = "__CHAT_VARS__"
COLOAD_VARS_PLACEHOLDER = "__COLOAD_VARS__"

#: Var-name prefix for an opportunistically co-loaded model, by role.  Shared
#: with the emitter so the planner can cost a set and the router can expand
#: it from one naming rule.
COLOAD_VAR_PREFIX = {"s2t": "s2t", "image": "img", "t2s": "t2s"}


def coload_var_name(role: str, taken: Container[str]) -> str:
    """The co-load var name for *role*, numbered off *taken* to avoid a clash."""
    prefix = COLOAD_VAR_PREFIX.get(role, "coload")
    name, n = prefix, 1
    while name in taken:
        n += 1
        name = f"{prefix}{n}"
    return name


def _strip_parens(expr: str) -> str:
    """Drop one layer of balanced outer parentheses."""
    s = expr.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    return s  # the trailing ')' closes something earlier
        s = s[1:-1].strip()
    return s


def _split_top(expr: str, op: str) -> list[str]:
    """Split *expr* on *op* at parenthesis depth 0, dropping outer parens.

    ``"(emb | rnk2)"`` splits to ``["emb", "rnk2"]`` rather than the literal
    ``["(emb", "rnk2)"]`` — every consumer here wants var names.
    """
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == op and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [s for s in (_strip_parens(p) for p in parts) if s]


def _expr_leaves(expr: str, sets_cfg: Mapping[str, str] | None = None,
                 _seen: frozenset[str] = frozenset()) -> set[str]:
    """Every leaf in a set expression, with ``+set`` references expanded.

    The leaves are var names (or model IDs — llama-swap accepts both) plus
    the two placeholders.  An unknown reference stays as its literal token so
    callers can report it.
    """
    expr = _strip_parens(expr)
    if not expr:
        return set()
    parts = _split_top(expr, "|")
    if len(parts) == 1:
        parts = _split_top(expr, "&")
    if len(parts) > 1:
        out: set[str] = set()
        for part in parts:
            out |= _expr_leaves(part, sets_cfg, _seen)
        return out
    atom = expr.strip()
    if atom.startswith("+"):
        name = atom[1:]
        if sets_cfg and name in sets_cfg and name not in _seen:
            return _expr_leaves(str(sets_cfg[name]), sets_cfg,
                                _seen | {name})
        return {atom}
    return {atom}


def _prune_leaves(expr: str, drops: frozenset[str],
                  sets_cfg: Mapping[str, str] | None = None,
                  _seen: frozenset[str] = frozenset()) -> str:
    """Remove the leaf var(s) in *drops* from a set expression.

    The host-unit refusal pass (requirement 33) deletes CPU categories
    that cannot fit; the sets naming them must not dangle.  Sets keep
    their remaining members, and a group whose every member was refused
    collapses, so `a & (b & c)` prunes `b` into `a & c`, never `a & ()`.
    An expression that prunes to nothing returns "" — the caller drops
    the set.  ``+set`` references prune recursively (cycle-guarded),
    matching :func:`_expr_leaves`; drops are category var names, so a
    leaf spelled as an entry id is never matched.
    """
    expr = _strip_parens(expr)
    if not expr:
        return ""
    for op in ("|", "&"):
        parts = _split_top(expr, op)
        if len(parts) > 1:
            other = "&" if op == "|" else "|"
            kept = []
            for part in parts:
                pruned = _prune_leaves(part, drops, sets_cfg, _seen)
                if pruned:
                    # Re-group a part carrying the sibling operator: the
                    # original parens may have enclosed a refused member
                    # (`(a | b) & c` pruning `b` must stay `(a) & c`).
                    kept.append(pruned if other not in pruned
                                else f"({pruned})")
            return f" {op} ".join(kept) if kept else ""
    atom = expr
    if atom.startswith("+"):
        name = atom[1:]
        if sets_cfg and name in sets_cfg and name not in _seen:
            return _prune_leaves(str(sets_cfg[name]), drops, sets_cfg,
                                 _seen | {name})
        return atom
    return "" if atom in drops else atom


def _cat_stem(fixed_categories: list[tuple[str, "Model"]],
              name: str) -> str:
    """The stem a declared category var (or "-") binds, for log lines."""
    for var, model in fixed_categories:
        if var == name:
            return model.stem if model is not None else "-"
    return "-"


def _resident_options(expr: str, sets_cfg: Mapping[str, str] | None = None,
                      _seen: frozenset[str] = frozenset(),
                      ) -> list[frozenset[str]]:
    """Every and-set the expression may hold simultaneously.

    `&` merges members (all live together); `|` branches (one of the
    alternatives lives) — so an expression yields one option per
    substitution path, and a role pair co-resides only when an option
    holds both.  ``three | (two & one)`` yields ``{three}`` and
    ``{two, one}``: `one` and `three` never ride together.  ``+set``
    references expand (cycle-guarded), like :func:`_expr_leaves`.
    """
    expr = _strip_parens(expr)
    if not expr:
        return []
    for op in ("|", "&"):
        parts = _split_top(expr, op)
        if len(parts) > 1:
            options: list[frozenset[str]] = []
            for part in parts:
                sub = _resident_options(part, sets_cfg, _seen)
                if not sub:
                    continue
                if not options:
                    options = list(sub)
                elif op == "&":
                    options = [a | b for a in options for b in sub]
                else:
                    options.extend(sub)
            return options or []
    atom = expr
    if atom.startswith("+"):
        name = atom[1:]
        if sets_cfg and name in sets_cfg and name not in _seen:
            return _resident_options(str(sets_cfg[name]), sets_cfg,
                                     _seen | {name})
        return [frozenset({atom})]
    return [frozenset({atom})]


def _has_chat_slot(expr: str, sets_cfg: Mapping[str, str] | None = None) -> bool:
    """True when the expression allows a chat entry to be resident."""
    if CHAT_VARS_PLACEHOLDER in expr:
        return True
    return any(re.fullmatch(r"c\d+", leaf)
               for leaf in _expr_leaves(expr, sets_cfg))


def set_overhead_mb(expr: str, var_mb: Mapping[str, int],
                    chat_vars: frozenset[str],
                    coload_vars: frozenset[str],
                    sets_cfg: Mapping[str, str] | None = None,
                    coload_mb: int | None = None,
                    _seen: frozenset[str] = frozenset()) -> int:
    """Worst-case co-resident VRAM (MB) a set allows beside chat.

    The expression is a boolean tree: ``&`` means the terms are resident
    together (sum), ``|`` means one alternative is chosen (max).  Nesting is
    priced recursively, so ``(3 | (2 & 1))`` costs ``max(3, 2 + 1)`` and
    ``(1 | (2 & 3))`` costs 5 — the old depth-0 split priced the second as 1.

    ``+set`` references expand against *sets_cfg*.  ``coload_mb`` overrides
    the cost of ``__COLOAD_VARS__`` when the caller knows the included
    co-loads (the matrix solve does); otherwise their var costs are summed,
    because co-loads genuinely stack.
    """
    expr = _strip_parens(expr)
    if not expr:
        return 0
    alternatives = _split_top(expr, "|")
    if len(alternatives) > 1:
        return max(set_overhead_mb(a, var_mb, chat_vars, coload_vars,
                                   sets_cfg, coload_mb, _seen)
                   for a in alternatives)
    terms = _split_top(expr, "&")
    if len(terms) > 1:
        return sum(set_overhead_mb(t, var_mb, chat_vars, coload_vars,
                                   sets_cfg, coload_mb, _seen)
                   for t in terms)
    atom = expr.strip()
    if atom == CHAT_VARS_PLACEHOLDER or atom in chat_vars \
            or re.fullmatch(r"c\d+", atom):
        return 0
    if atom == COLOAD_VARS_PLACEHOLDER:
        if coload_mb is not None:
            return coload_mb
        return sum(var_mb.get(v, 0) for v in coload_vars)
    if atom.startswith("+"):
        name = atom[1:]
        if sets_cfg and name in sets_cfg and name not in _seen:
            return set_overhead_mb(str(sets_cfg[name]), var_mb, chat_vars,
                                   coload_vars, sets_cfg, coload_mb,
                                   _seen | {name})
        return 0
    return var_mb.get(atom, 0)


def _probe_at_overhead(planner: "Planner", model: Model,
                       overhead_mb: int) -> tuple[int, int] | None:
    """What ``model`` could be served at if *overhead_mb* is spoken for.

    A side probe: the real solve pins a group parallel and honours
    auto-parallel, neither of which is known yet at admission time, so this
    asks the plain question "does a single-sequence entry still clear its
    demand minimums" and lets the real solve refine it afterwards.
    """
    view = model.view_for(model.mmproj is not None
                          and model.mmproj.gguf_path is not None)
    cache_type = view.cache_type_for(planner.profiles.default_cache_type)
    try:
        ctx = planner._bounded_ctx(
            view, parallel=1, cache_type=cache_type,
            spare_mb=overhead_mb, include_mmproj=True)
    except Exception:  # a probe must never break planning
        return None
    if ctx <= 0:
        return None
    return ctx, 1


def ctx_label(ctx: int) -> str:
    """Compact served-context label, at most 4 characters.

    Used in the display name (always) and as the entry-id tail for an
    *untagged* serving — one satisfying no blessed ``usage:`` category, so
    there is no stable task name to use.  ``262144 -> '256k'``,
    ``139264 -> '136k'``, ``1048762 -> '1M'``.

    The unit threshold is 1000k, not 1024k, so the label is never wider than
    four characters; above it the value switches to whole megas.  Exact
    contexts below the threshold cannot collide, and at or above it the
    rounding is under 0.1% — the exact number is always available in the
    entry's ``metadata.ctx_size``.
    """
    k = max(int(ctx), 0) // 1024
    return f"{k}k" if k < 1000 else f"{max(1, round(ctx / (1024 * 1024)))}M"


#: Flag letters for the id capability segment, in fixed emission order.
#: ``t`` tools, ``i`` image input, ``v`` video input, ``a`` audio input,
#: ``s`` audio (speech) output.  A serving with none of these (plain text in,
#: text out, no tools) emits no segment — the absence is the statement.
FLAG_LETTERS: tuple[tuple[str, str], ...] = (
    ("tools", "t"),
    ("image", "i"),
    ("video", "v"),
    ("audio", "a"),
    ("speech", "s"),
)


def flags_label(capabilities: Iterable[str] | None) -> str:
    """The id flag segment for a *served* capability set.

    Each letter is present iff the serving actually carries it, so a rung
    whose tools were demoted (served below ``tools_min_ctx``) does not claim
    ``t``.  Returns ``""`` when nothing qualifies — a plain text-input/text-
    output serving has nothing to flag.  The letters are joined in
    :data:`FLAG_LETTERS` order (deterministic, so the id is stable).
    """
    caps_l = {str(c).lower() for c in (capabilities or ())}
    return "".join(letter for cap, letter in FLAG_LETTERS if cap in caps_l)


def served_flags(model: "Model", include_mmproj: bool,
                 tools_demoted: bool) -> str:
    """The flag segment for *model* as actually served by this variant.

    Mirrors :func:`_build_entry`'s capability handling: a no-mmproj serving
    drops the ``image``/``video`` inputs (when a companion exists), and a
    demoted serving drops ``tools`` — so the id's flags cannot claim a
    capability the entry does not advertise.  Keeping this next to the
    emitter's own cap derivation is deliberate: both read the same
    ``model.capabilities`` vocabulary.
    """
    caps = [str(c).lower() for c in model.capabilities]
    if (not include_mmproj and model.role != "image"
            and model.mmproj and model.mmproj.gguf_path):
        caps = [c for c in caps if c not in ("image", "video")]
    if tools_demoted:
        caps = [c for c in caps if c != "tools"]
    return flags_label(caps)




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
                    fallback_explicit: bool = False,
                    category_floor: int | None = None) -> int:
    """Per-model minimum useful context: the hard floor of the (ctx, slots)
    search and the context-ratio normalizer in :func:`parallel_value`.

    Cascade, first match wins: explicit sidecar ``min_context:`` → a pinned
    serving ctx (``--max-context``; a fixed ctx is its own floor) →
    ``category_floor`` → tool callers get the agentic floor → everyone else
    gets half their design context (two slots are then only considered when
    nearly two full windows fit).  The global ``--min-context`` flag
    overrides the last two rows when explicitly set, never an explicit
    sidecar key or a hard pin.  A pin below an explicit floor warns (the pin
    is honoured; the floor is reported unmet).

    ``category_floor`` is the strongest demand the model declares with
    ``usage:``, and it outranks the two heuristics below it because they are
    guesses about undeclared models while a category is the operator saying
    what the model is *for*. Without that, a model tagged ``aux`` (80k) that
    also lists the ``tools`` capability was still floored at 131072 by the
    tools heuristic -- an unreachable floor, so it fell back to one slot and
    then failed its own category's ``min_parallel``. Declaring the work was
    making the model *less* satisfiable.

    ``context_length:`` appears nowhere in this cascade. It is the model's
    design maximum -- a ceiling applied when serving (see
    :meth:`VramBudget._sidecar_ceiling`), never a floor.
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
    if category_floor is not None:
        return category_floor
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


class HostLedger:
    """Sibling of PoolLedger: host-RAM claims by ``device: cpu`` models.

    Committed residents (embed/rerank winners, one per role) are charged
    at build time; every other CPU-resident entry is judged go/no-go
    against the pool remaining after them — never solved down, mirror of
    the VRAM `over_budget` refusal (SPEC.md §Host pool).  ``pool_mb=None``
    (detection unavailable) means the ledger cannot gate: it logs that
    once and reports an unbounded pool rather than inventing one.
    """

    def __init__(self, pool_mb: int | None):
        self.pool_mb = pool_mb
        self.residents: list[tuple[str, str, int]] = []  # (label, role, mb)
        self._roles: set[str] = set()

    def add_resident(self, label: str, role: str, mb: int) -> None:
        # One co-resident per role (design decision 2): the resident
        # selection already picks one winner — two quads for one role is
        # a planner bug, not a pool this ledger would silently book.
        if role in self._roles:
            raise AssertionError(
                f"host ledger: two committed residents for role {role!r} "
                f"({label!r}, already have one)")
        self._roles.add(role)
        if mb > 0:
            self.residents.append((label, role, int(mb)))

    def resident_mb(self) -> int:
        return sum(mb for _, _, mb in self.residents)

    def available(self) -> int | None:
        """Pool remaining after committed residents; None = cannot gate."""
        if self.pool_mb is None:
            return None
        return self.pool_mb - self.resident_mb()


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

    A variant with ``include_mmproj=False`` is the vision-less serving a
    category chose by shedding the projector; a variant with it true serves
    vision. There is no companion entry and no text-only twin — vision is a
    rung each category decides for itself.
    """
    parallel: int
    cache_type: str
    spare_mb: int
    profiles_group: list[tuple[str, dict]] = field(compare=False)
    ctx_size: int
    include_mmproj: bool
    #: Resolved ``-b``/``-ub`` (sidecar > profile > fleet > role).  ``None``
    #: renders no flag — llama.cpp's own builtin then applies.  ``ubatch``
    #: also feeds the VRAM measurement shape (see :func:`measurement_args`).
    batch: int | None = None
    ubatch: int | None = None
    coload: bool = False
    tools_demoted: bool = False
    #: The categories this rung is named for. A rung answers exactly one, so
    #: this is normally a single demand; a tuple only so the emitter can
    #: express "nothing declared" without a second field. Empty for the
    #: fallback entry, which carries no category name in its id.
    task_demands: tuple[Task, ...] = ()
    #: The category this variant answers, or :data:`FALLBACK` for the null
    #: category. Singular, because a rung exists to serve one category -- which
    #: is what lets one model emit several entries and stay unambiguous.
    category: Task | None = None
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
    #: The affine per-token KV constant (MiB/token) of this serving, or
    #: None for fixed-overhead backends — metadata.packing.kv_per_token.
    kv_per_token_mib: float | None = None
    #: Non-None when the VRAM budget could not afford the design context
    #: (``--spare``/``--baseline`` active): appended to the description so
    #: the entry visibly carries its over-budget status.
    vram_note: str | None = None


def resolve_batch_ubatch(
    profiles: Profiles, model: Model, profile: dict | None = None,
) -> tuple[int | None, int | None]:
    """Resolved ``(batch, ubatch)`` for *model* — sidecar keys > profile
    keys > fleet ``llama_server: batch:/ubatch:`` > backend role defaults.
    ``None`` means "nothing configured": the backend renders no flag and the
    engine's builtin applies.  One resolution site: the rendered command
    and the VRAM measurement shape both consume these values, so the serve
    and the estimate can never drift apart on batch flags."""
    try:
        backend = get_backend(model.backend)
        role_defaults = backend.default_batch_ubatch(model.role)
    except KeyError:
        role_defaults = (None, None)
    return model.batch_ubatch_for(
        profile, profiles.llama_server_cfg if profiles else {}, role_defaults)


def measurement_args(
    profiles: Profiles, model: Model, global_args: str = "",
    profile: dict | None = None,
) -> str:
    """The exact flag string VRAM constants are measured (and shape-stamped)
    under: global args + the CPU device pin when the model is
    CPU-resident (`--n-gpu-layers 0`, the measurement of the host-side
    quads — docs/plans/cpu-residency-budget.md) + the model's effective
    ``-ub``.  ``batch`` is excluded — measured to have no VRAM effect, so
    throughput tuning must not invalidate estimates.

    The *effective* ubatch is stamped, not just an explicitly configured one:
    an unset key still serves at llama.cpp's builtin, so the compute buffer
    the constants were measured for is the one the server will actually
    allocate.  Omitting it would change every ``derived: shape`` key and
    force a fleet-wide re-measurement for a flag that has not moved.  The
    same holds for the device pin: a sidecar ``device: cpu`` edit must
    re-measure (the block describes the other budget's byte terms), so the
    pin rides the shape stamp.
    """
    _, ubatch = resolve_batch_ubatch(profiles, model, profile)
    if ubatch is None:
        ubatch = _LLAMA_CPP_DEFAULT_UBATCH
    base = (global_args or "").strip()
    if model.on_cpu:
        base = f"{base} --n-gpu-layers 0".strip()
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
        category_ctx: dict[str, int] | None = None,
        baseline_mb: int = 0,
        min_context: int = _MIN_AGENTIC_CTX,
        min_context_explicit: bool = False,
        memory_margin: float = _MEMORY_MARGIN,
        llama_args: str = "",
        host_pool_mb: int | None = None,
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
        # Blessed usage categories.  A model may only *tag* itself with one
        # of these; an unblessed tag is warned about and ignored.
        self.tasks = TaskCatalogue.from_cfg(
            getattr(profiles, "tasks_cfg", None))
        # Per-set co-resident overhead (MB) and the sets a chat model was
        # admitted to, filled in by _solve_usage_sets.
        self.set_overheads: dict[str, int] = {}
        # (model stem, category name) -> the sets that entry may join. Keyed
        # per category, not per stem: one model's `.coding` and `.aux` entries
        # are different servings, and whichever was solved first used to speak
        # for both.
        self.set_admission: dict[tuple[str, str | None], frozenset[str]] = {}
        # stem -> the sets this model was charged for, per profile group.
        # Kept only so the charge and the admission it feeds cannot drift.
        self._charge_sets: dict[str, list[tuple[frozenset[str], int]]] = {}
        # Declared matrix categories beyond the RAG pair (e.g. tts/stt):
        # fixed-overhead residents reserved alongside chat and RAG.
        self.fixed_categories = fixed_categories or []
        # Per-category served context (requirement 32): category var name ->
        # the ctx the operator declared for that model. A cap applied at
        # one seam (resident_ctx / the category seam in the planner loop),
        # never a floor.
        self.category_ctx = category_ctx or {}
        self._cat_vars_by_stem: dict[str, list[str]] = {}
        for var, m in self.fixed_categories:
            if m is not None:
                self._cat_vars_by_stem.setdefault(m.stem, []).append(var)
        for var, m in (("emb", embed_model), ("rnk", rerank_model)):
            if m is not None:
                self._cat_vars_by_stem.setdefault(m.stem, []).append(var)
        # Categories refused by the host-unit pass (requirement 33): filled
        # in by plan(), before _build_ledger judges the fixed categories.
        self.host_unit_refused_vars: frozenset[str] = frozenset()
        self.host_unit_refused_stems: frozenset[str] = frozenset()
        self.baseline_mb = baseline_mb
        self.min_context = min_context
        self.min_context_explicit = min_context_explicit
        self.memory_margin = memory_margin
        self.ledger = PoolLedger(vram_total, self.profiles.pools_cfg)
        self.host_ledger = HostLedger(host_pool_mb)
        # Stems committed to the host ledger as residents: their serving is
        # charged once (the unit pass), never re-checked at emission.
        self.host_ledger_stems: set[str] = set()
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
        include_draft: bool = True,
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
            include_draft=include_draft,
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

    def can_meet(self, model, demand: "Task") -> bool:
        """Whether *model* has what the category ``require``s.

        ``require`` gates existence, which is a different thing from
        ``keep``: a model lacking a required capability does not satisfy the
        category at all, so no entry may be named for it. ``keep`` only
        forbids shedding.

        This gate used to live in ``met_demands`` and went with it, which left
        a text-only model free to claim ``require: [vision]`` -- an id that
        promised a capability the entry did not advertise.
        """
        have_vision = (model.mmproj is not None
                       and model.mmproj.gguf_path is not None)
        have_draft = model.mtp is not None or model.mtp_baked_in
        for cap in demand.require:
            if cap == "vision" and not have_vision:
                return False
            if cap == "draft" and not have_draft:
                return False
        return True

    def solve_rung(self, model, demand: "Task", *,
                   parallel: int, cache_type: str, spare_mb: int,
                   design_ctx: int | None = None,
                   context_length: int | None = None,
                   profile: dict | None = None,
                   prefer_capability: bool = False) -> "Rung | None":
        """The serving this category should get: most context that fits.

        Walks the category's own shed levels (:func:`shed_levels`), asking the
        budget what each one affords, and returns the level that buys the most
        context — or ``None`` when the category's ``min_context`` is out of
        reach at every level, which is what "this category cannot be served"
        means, and what suppresses the entry.

        Among levels that buy the *same* context, the least-shed wins. That
        is the whole of "coding drops vision to maximise context, and only
        keeps vision when it is free": shedding vision usually buys context,
        so the model goes text-only; when VRAM is plentiful enough to hold
        design context either way, nothing is bought and the projector stays
        attached rather than being thrown away for no gain.

        Only the shed axis is decided here. Slot count stays with the
        parallel solve, which runs afterwards and may squeeze further.

        Two rules for which qualifying rung wins, and they differ because
        the categories differ. A category wants the **most context** — coding
        would rather have 262144 tokens text-only than 131072 with a
        projector. The null category wants the **least shed**, taking the
        first rung that qualifies: it has no goals, so handing up a capability
        to buy context nobody asked for is a pointless trade.
        """
        best: Rung | None = None
        for shed in shed_levels(demand.sacrifice,
                                default_order=self.profiles.shed_order,
                                required=demand.require,
                                kept=demand.keep):
            # The view follows the shed level. view_for() gives an mmproj
            # model a *separate* companion-on budget, so probing a text-only
            # rung against the vision view would measure the vision budget
            # again and never see what dropping the projector buys.
            view = model.view_for("vision" not in shed)
            ctx = self._bounded_ctx(
                view, parallel=max(1, parallel), cache_type=cache_type,
                spare_mb=spare_mb,
                include_mmproj="vision" not in shed,
                include_draft="draft" not in shed,
                design_ctx=design_ctx, context_length=context_length,
                profile=profile)
            if ctx <= 0 or ctx < demand.min_context:
                continue
            over = getattr(view.vram, "over_budget_reason", None)
            if over is not None:
                # Weights plus this level's companions do not fit. A lower
                # rung may still shed down to something that does -- but if
                # none does, the category emits nothing rather than a config
                # that cannot load. This is the per-category form of the
                # fleet-wide over-budget gate, and was lost with it.
                logger.info("%s: %s (%s)", view.stem,
                            demand.id or "fallback", over)
                continue
            rung = Rung(shed=shed, ctx=ctx, parallel=max(1, parallel),
                        demand=demand)
            if best is None:
                best = rung
            elif prefer_capability:
                # shed_levels is least-shed first, so the first rung that
                # qualifies is the answer and nothing later can improve it.
                return best
            elif ctx > best.ctx:
                best = rung
            # Equal context: the earlier (less-shed) level stands.
        return best

    def _tools_demoted(self, view, ctx: int) -> bool:
        """Whether this serving must drop its ``tools`` capability.

        Tool calling needs a real window; below ``matrix.tools_min_ctx`` the
        entry cannot honour it, and advertising the capability anyway
        misleads clients about what they will get. Decided per serving
        because each entry of a model serves its own context -- a `.coding`
        entry and a `.aux` entry of the same model rarely match.
        """
        return (view.role == "chat"
                and ctx < self.knobs.tools_min_ctx
                and "tools" in [c.lower() for c in view.capabilities])

    def _serving_pin(self, view) -> int | None:
        """Explicitly pinned serving ctx, or None.

        Only ``--max-context`` pins. A sidecar ``context_length:`` does
        *not*: it declares what the model was **designed** for -- the ceiling
        on what may be served -- while the served size is worked out from the
        VRAM budget, the usage category and the parallel/context bias.

        Treating it as a pin instead made the two indistinguishable, and the
        cost was concrete: a 262144 design max became an unreachable floor,
        so the solve fell back to parallel 1 and a model tagged for two
        slots could never satisfy its own category. Worse, the only way to
        fix it was to edit the sidecar -- rewriting a capability claim about
        the model in order to express an operating decision. `min_context:`
        is the key that means "serve at least this", and it was already
        separate.
        """
        pins = []
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
        power: float | None = None,
        min_parallel: int = 1,
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

        *min_parallel* is a soft floor from a usage demand: candidates that
        meet it win outright, but meeting the context floor takes precedence,
        because the context is the property the entry id advertises and
        concurrency is only a throughput preference.
        """
        power = self.knobs.parallel_power if power is None else power
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
        cands: list[tuple[int, int]] = []
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
            cands.append((p, ctx))
        # A usage demand's min_parallel is preferred, but never at the cost
        # of the context floor: the id advertises the context, and a client
        # that cannot fit its prompt has nothing to gain from a spare slot.
        if min_parallel > 1:
            preferred = [c for c in cands if c[0] >= min_parallel]
            if preferred:
                cands = preferred
        # Ties go to *more* slots. With power > 0 a model holding its cap
        # scores strictly higher at every slot count, so it never ties; the
        # flat case is power 0, where slots are worth nothing and the choice
        # is purely "can it hold the context at more of them". Keeping the
        # first maximum instead gave the parallel-1 entry, which is the
        # opposite of what a plentiful-VRAM model should do.
        best: tuple[float, int, int] | None = None  # (score, p, ctx)
        for p, ctx in cands:
            score = parallel_value(ctx, floor, p, power)
            if best is None or score >= best[0]:
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


    def demands_for(self, model: Model) -> tuple[Task, ...]:
        """The usage demands in force for *model*.

        A tag is only a claim: it is honoured when an entry can actually
        meet the category's minimums, and silently ignored otherwise, so a
        model is never held to a demand the VRAM cannot satisfy.
        """
        if not self.tasks:
            return ()
        resolved = resolve_tasks(model.task_ids, self.tasks, logger,
                                 stem=model.stem)
        demands = tuple(task for task in resolved if task.role == "chat")
        # A declared resolution budget means vision must actually be served at
        # that size, so it stops being shed-able for this model whatever the
        # category's sacrifice order says. ``keep`` rather than ``require``:
        # a text-only model is unaffected, and the image floor already fails
        # the rung when the budget cannot accommodate the image.
        mp = model.min_image_megapixels
        if mp is not None and mp > 0 and model.mmproj is not None:
            demands = tuple(replace(d, keep=d.keep | {"vision"}) for d in demands)
        return demands


    def _price_sets(self) -> None:
        """Cost every chat set by the residents it can hold alongside chat.

        Replaces the single fleet-wide ledger charge: an entry has to
        survive the heaviest set it lands in, and which set that is depends
        on what the model can actually afford.
        """
        if not self.matrix_cfg:
            return
        sets_cfg = self.matrix_cfg.get("sets") or {}
        if not sets_cfg:
            return
        by_stem = {m.stem: m for m in self.models}
        taken: set[str] = set()
        var_mb: dict[str, int] = {}

        def overhead(m: Model, name: str | None = None) -> int:
            # Same pricing the matrix solve used to reserve these, so the
            # per-set figures and the shared solve cannot disagree.
            oh = _coload_overhead(m, self.fit_bin, self.profiles, self.knobs,
                                  self.llama_args,
                                  ctx=self.category_ctx.get(name))
            return int(oh or 0)

        def rag_overhead(m: Model, name: str | None = None) -> int:
            """emb/rnk cost at their *solved* RAG context, as the matrix does.

            The matrix prices these from the same quad formula it feeds
            solve_matrix_ctx with, at the small solved context — not at
            design context, and without the co-load headroom.  A
            per-category ``ctx:`` caps the context (requirement 32).
            """
            quad = _static_params(m, self.fit_bin,
                                  self.profiles.default_cache_type,
                                  measurement_args(self.profiles, m,
                                                   self.llama_args))
            if quad is None:
                return 0
            mib, factor, slots, compute = quad
            return int(mib + compute + factor * self._rag_ctx(m, name)
                       + slots)

        # Declared categories: `emb`/`rnk` plus any extra (stt/tts/…).
        for name, model in (self.fixed_categories or []):
            if model is None or name in self.host_unit_refused_vars:
                continue
            taken.add(name)
            var_mb[name] = overhead(model, name)
        for name, model in (("emb", self.embed_model),
                            ("rnk", self.rerank_model)):
            if model is not None and name not in taken:
                taken.add(name)
                var_mb[name] = rag_overhead(model, name)
        # Co-loads: the matrix solve already priced them the same way.
        coload_vars: set[str] = set()
        for stem, mb in (getattr(self.matrix_result, "coloads", ()) or ()):
            m = by_stem.get(stem)
            if m is None:
                continue
            name = coload_var_name(m.role, taken)
            taken.add(name)
            var_mb[name] = int(mb)
            coload_vars.add(name)

        for sname, expr in sets_cfg.items():
            expr = str(expr)
            if not _has_chat_slot(expr, sets_cfg):
                continue  # not a chat set
            self.set_overheads[sname] = set_overhead_mb(
                expr, var_mb, frozenset(), frozenset(coload_vars),
                sets_cfg=sets_cfg)
        if self.set_overheads:
            logger.info("tasks: per-set co-resident overhead (MB): %s",
                        ", ".join(f"{k}={v}" for k, v in
                                  sorted(self.set_overheads.items())))

    def _category_ctx_cap(self, model: Model) -> int | None:
        """The per-category ``ctx:`` cap for *model*, min over its vars."""
        caps = [self.category_ctx[v]
                for v in self._cat_vars_by_stem.get(model.stem, ())
                if self.category_ctx.get(v) is not None]
        return min(caps) if caps else None

    def _rag_ctx(self, model: Model, name: str | None = None) -> int:
        """Served context for a RAG/fixed-overhead category.

        The role keeps its policy (RAG cap, design context); a per-category
        ``ctx:`` (requirement 32) caps the result for the named category,
        never floors it.
        """
        res = self.matrix_result
        if res is None:
            ctx = model.design_context
        elif model.role == "embeddings":
            ctx = min(model.design_context, res.embed_ctx)
        elif model.role == "rerank":
            ctx = min(model.design_context, res.rerank_ctx)
        else:
            ctx = model.design_context
        if name is not None:
            cap = self.category_ctx.get(name)
            if cap is not None:
                ctx = min(ctx, cap)
        return ctx

    def _host_unit_refusals(self) -> frozenset[str]:
        """Judge each chat-capable composition's CPU residents as one unit.

        Requirement 33: the CPU residents a composition holds together (the
        committed embed/rerank pair plus the fixed CPU categories it names)
        must fit the pool remaining after the pair — the host analog of the
        VRAM worst-set contract.  A composition that cannot fit refuses its
        fixed CPU categories together (all-or-nothing per category: one
        that another, fitting composition admits survives); the committed
        pair is never refused (requirement 13 — the ledger exists to carry
        it), and refusals never invent a budget (requirement 16 — no pool,
        no gate; a category in no set runs alone, judged solo).
        """
        if self.host_ledger.pool_mb is None or self.host_ledger.pool_mb <= 0:
            return frozenset()
        res = self.matrix_result
        if res is None:
            return frozenset()
        pair_rent = 0
        pair_stems: set[str] = set()
        for model, ctx in ((self.embed_model, res.embed_ctx),
                           (self.rerank_model, res.rerank_ctx)):
            if model is None or not model.on_cpu:
                continue
            pair_rent += self._host_resident_mb(model, ctx)
            pair_stems.add(model.stem)
        available = self.host_ledger.pool_mb - pair_rent
        cats: dict[str, int] = {}
        for name, m in self.fixed_categories:
            if not m.on_cpu or name in pair_stems:
                continue
            cats[name] = self._host_resident_mb(m, self._rag_ctx(m, name))
        raw_sets = {str(n): str(e)
                    for n, e in (self.matrix_cfg or {}).get("sets",
                                                            {}).items()}
        if not raw_sets:
            # No routing sets: every category runs alone (the section-24
            # solo rule) — its own quad must fit the pool by itself.
            refused = {n for n, mb in cats.items() if mb > available}
            if refused:
                logger.warning(
                    "host ledger: no sets — %s exceed(s) the %d MiB free "
                    "after residents; refused",
                    ", ".join(f"{n} ({cats[n]} MiB)"
                              for n in sorted(refused)), available)
            return frozenset(refused)
        per_set: dict[str, list[str]] = {}
        fails: dict[str, tuple[int, int]] = {}
        for sname, expr in raw_sets.items():
            if not _has_chat_slot(expr, raw_sets):
                continue
            members = sorted(n for n in _expr_leaves(expr, raw_sets)
                             if n in cats)
            per_set[sname] = members
            total = sum(cats[n] for n in members)
            if available < total:
                fails[sname] = (total, available - total)
        if not fails:
            return frozenset()
        for sname in sorted(fails):
            total, over = fails[sname]
            logger.warning(
                "host ledger: composition %r cannot fit as a unit: %d MiB "
                "of CPU residents (%s) but %d MiB free after the committed "
                "pair — %d MiB short",
                sname, total, ", ".join(per_set[sname]) or "-",
                available, -over)
        refused: set[str] = set()
        for name, mb in sorted(cats.items()):
            named = [s for s, ms in per_set.items() if name in ms]
            if named:
                if all(s in fails for s in named):
                    refused.add(name)
                    logger.warning(
                        "host ledger: %s (%s) refused as a unit — every "
                        "composition naming it exceeds the pool",
                        name, _cat_stem(self.fixed_categories, name))
            elif mb > available:
                refused.add(name)
                logger.warning(
                    "host ledger: %s (%s) refused — in no set, larger than "
                    "the %d MiB free after residents (it can only run "
                    "alone)", name, _cat_stem(self.fixed_categories, name),
                    available)
        return frozenset(refused)

    def _coresident_charge(self, model: Model, pool_id: str) -> int:
        """Co-resident MB to charge *model*'s solve, and record its sets.

        Per the matrix contract an entry must survive the heaviest set it
        belongs to, so this charges the priciest set the model is admitted
        to.  When no sets are configured it falls back to the ledger's
        fleet-wide reserve, so an operator without a ``matrix:`` section
        sees no change.

        Admission: a model joins a set when, at that set's overhead, it can
        still meet one of its usage demands.  With no declared demands the
        model is admitted everywhere it fits, which preserves today's
        behaviour for untagged models.
        """
        base = self.ledger.extra_reserve_mb(pool_id)
        if not self.set_overheads or model.role != "chat":
            return base
        demands = self.demands_for(model)
        if not demands:
            # Untagged: keep the previous fleet-wide charge rather than
            # inventing a per-set answer the operator never asked for.
            return base

        worst = 0
        admitted: set[str] = set()
        for sname, oh in sorted(self.set_overheads.items()):
            probe = _probe_at_overhead(self, model, oh)
            if probe is None:
                continue
            ctx, parallel = probe
            if any(d.satisfied_by(ctx, parallel) for d in demands):
                admitted.add(sname)
                worst = max(worst, oh)
        if not admitted:
            # Meets nothing anywhere: charge the heaviest set so the entry
            # is sized for the worst thing it could be asked to share with,
            # and let the caller drop it from every set.
            worst = max(self.set_overheads.values())
            admitted = set(self.set_overheads)
        # This function only *charges* the solve now. Which sets an entry
        # may join is decided later, per category, on the serving that was
        # actually resolved -- see record_admission.
        self._charge_sets.setdefault(model.stem, []).append(
            (frozenset(admitted), max(base, worst)))
        logger.info(
            "tasks: %s co-resident charge %d MB (sets: %s)",
            model.stem, max(base, worst),
            ", ".join(sorted(admitted)) or "none")
        return max(base, worst)

    def record_admission(self, model, demand, ctx: int, parallel: int) -> None:
        """Which sets this one *category's entry* may join.

        Keyed by (stem, category) and judged on the resolved (ctx, parallel),
        because that is the serving llama-swap would hold beside the set's
        residents. Keying by stem alone let one category's entry ride on
        another's verdict -- a `.coding` entry sized for 204800 and a `.aux`
        entry sized for 81920 were treated as the same model, so whichever
        was solved first spoke for both.

        The fallback entry joins no set: routing is what categories are for,
        and it has no name to route on.
        """
        if demand is FALLBACK:
            self.set_admission[(model.stem, None)] = frozenset()
            return
        admitted = {sname
                    for sname, oh in self.set_overheads.items()
                    if self._serves_beside(model, demand, oh, parallel)}
        if not admitted:
            # Meets the category nowhere beside a resident. It still serves
            # standalone -- as its own entry, outside every set.
            logger.info("tasks: %s.%s joins no set: it cannot hold its "
                        "%d-token minimum beside any resident",
                        model.stem, demand.id, demand.min_context)
        else:
            logger.info("tasks: %s.%s admitted to %s",
                        model.stem, demand.id, ", ".join(sorted(admitted)))
        self.set_admission[(model.stem, demand.id)] = frozenset(admitted)

    def _serves_beside(self, model, demand, overhead_mb: int,
                       parallel: int) -> bool:
        """Whether this entry still clears its demand beside ``overhead_mb``.

        The probe is single-sequence by design -- it answers "does one
        sequence still fit with that much spoken for" -- so its slot count is
        not the entry's. Testing ``satisfied_by`` against the probe's 1 would
        refuse every category asking for 2 slots, so the slot half of the
        requirement is checked against the serving that was actually
        resolved, and only the context half against the probe.
        """
        probe = _probe_at_overhead(self, model, overhead_mb)
        if probe is None:
            return False
        ctx, _ = probe
        return (ctx >= demand.min_context
                and max(1, parallel) >= demand.min_parallel)

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
            category_ctx=self.category_ctx,
            coload_sets=self.matrix_cfg.get("sets"),
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
        if self.host_ledger.pool_mb is None:
            # Cannot gate what cannot be read — say so once, loudly.
            logger.info("host ledger: no pool detected and none declared — "
                        "device: cpu models are ungated "
                        "(profiles.yaml hardware: host keys)")
        res = self.matrix_result
        if res is not None:
            for model, ctx, label in ((self.embed_model, res.embed_ctx,
                                       "embed"),
                                      (self.rerank_model, res.rerank_ctx,
                                       "rerank")):
                if model is None:
                    continue
                if model.on_cpu:
                    # A CPU resident charges host RAM, not VRAM (decision 5).
                    mb = self._host_resident_mb(model, ctx)
                    # The co-residency role is the matrix slot, not the
                    # frontmatter role: the resident selection picks one
                    # winner per slot.
                    self.host_ledger.add_resident(
                        f"{label}:{model.stem}", label, mb)
                    self.host_ledger_stems.add(model.stem)
                    logger.info("host ledger: %s resident %s (%d MiB of "
                                "%s pool, ctx %d)", label, model.stem, mb,
                                self.host_ledger.pool_mb, ctx)
                    continue
                mb = self._resident_overhead(model, ctx)
                pool = ledger.pool_id_for(model)
                ledger.add_resident(f"{label}:{model.stem}", mb, pool)
                logger.info("ledger: %s resident %s (%s bucket, pool %s%s)",
                            label, model.stem, footprint_bucket(model), pool,
                            " — no reserve (unestimated)"
                            if model.stem in self.synthetic_quads else "")
            # Declared fixed categories (stt/tts/rnk2/…) are residents too:
            # a CPU one charges the host pool, one per category, at the
            # same served context the planner prices them at.  Without
            # this, a role-wide device_roles switch moves a 3 GiB rerank
            # variant off VRAM and nothing accounts for it in RAM.
            for name, model in (self.fixed_categories or []):
                if model is None or not model.on_cpu:
                    continue
                if name in self.host_unit_refused_vars:
                    logger.info(
                        "host ledger: %s (%s) refused as a unit; nothing "
                        "committed and no id is emitted", name, model.stem)
                    continue
                ctx = self._rag_ctx(model, name)
                mb = self._host_resident_mb(model, ctx)
                self.host_ledger.add_resident(f"{name}:{model.stem}", name, mb)
                self.host_ledger_stems.add(model.stem)
                logger.info("host ledger: %s resident %s (%d MiB of %s pool, "
                            "ctx %d)", name, model.stem, mb,
                            self.host_ledger.pool_mb, ctx)
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
            parallel = model.parallel_for(self.profiles.default_parallel)
            mb = resident_cost_mb(
                model, fit_bin=self.fit_bin,
                cache_type=self.profiles.default_cache_type,
                profiles=self.profiles, llama_args=self.llama_args,
                ctx=ctx, parallel=parallel, allow_cpu=True)
            if mb is None:
                # Unestimable resident: charged 0 — the no-expansion rider
                # policy (see _solve_matrix_context).
                quad = self.synthetic_quads.get(model.stem)
                if quad is None:
                    return 0
                mb = _quad_cost_mb(quad, ctx, parallel)
            return mb
        except Exception as e:  # keep planning alive; matrix already solved
            logger.warning("ledger: cannot size resident %s (%s)",
                           model.stem, e)
            return 0

    def _host_resident_mb(self, model, ctx: int) -> int:
        """Host-RAM (MiB) of a CPU-resident resident at its served ctx.

        The same estimator the VRAM charge uses (requirement 11), single-
        slot (residents never take declared parallel), 0 on shader — a
        resident that cannot be measured is a rider charged nothing (the
        matrix solve's no-expansion rule).
        """
        if getattr(self.host_ledger, "pool_mb", None) is None \
                or self.host_ledger.pool_mb <= 0:
            return 0
        try:
            mb = resident_cost_mb(
                model, fit_bin=self.fit_bin,
                cache_type=self.profiles.default_cache_type,
                profiles=self.profiles, llama_args=self.llama_args,
                ctx=ctx, allow_cpu=True)
            if mb is None:
                quad = self.synthetic_quads.get(model.stem)
                if quad is None:
                    return 0
                mb = _quad_cost_mb(quad, ctx)
            return mb
        except Exception as e:  # keep planning alive; matrix already solved
            logger.warning("host ledger: cannot size resident %s (%s)",
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
    ) -> tuple[float | None, float | None, float | None]:
        """Decided TOTAL memory (MiB) attributable to one serving variant.

        ``(vram_mib, ram_mib, kv_per_token_mib)`` evaluated from the combined
        affine quad at the variant's served ``(ctx_size, parallel)`` —
        weights + folded companions + context — exactly what the emitted
        description tag reports verbatim (no capping, no re-solving).
        ``kv_per_token_mib`` is the affine ``c`` — None for fixed-overhead
        backends.  CPU-resident servings return ``(None, ram_mib, …)``
        (host-RAM constants via ``allow_cpu`` — never used for VRAM sizing).
        ``(None, None, None)`` when undecidable (no estimate source worked).
        ``parallel <= 0`` (uncapped vLLM) is priced single-seq, matching its
        ctx solve.
        """
        quad = view.vram.effective_static(
            self.fit_bin, cache_type=cache_type,
            include_mmproj=include_mmproj,
            llama_args=measurement_args(self.profiles, view,
                                        self.llama_args, profile),
            allow_cpu=True,
        )
        if quad is None:
            return (None, None, None)
        model_mib, kv_factor, slot_mib, compute_mib = quad
        p = parallel if parallel > 0 else 1
        total = (model_mib + compute_mib
                 + kv_factor * ctx_size * p + slot_mib * p)
        if view.on_cpu:
            return (None, float(total), float(kv_factor))
        files_mb = view.size_mb
        if include_mmproj and view.mmproj is not None \
                and view.mmproj.gguf_path is not None:
            files_mb += view.mmproj.size_mb
        if view.mtp is not None and view.mtp.gguf_path is not None:
            files_mb += view.mtp.size_mb
        return (float(total), max(0.0, float(files_mb - model_mib)),
                float(kv_factor))

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
        # The shared budget assumes every projector is shed, which holds for
        # the categories that drop vision and leaves one that keeps its
        # projector to decide in its own rung. `_mmproj_drop_pass` used to
        # pick that per model from a global context threshold, before any
        # category was read -- which is how it came to contradict the
        # category a model actually declared.
        self.matrix_result = self._solve_matrix(
            {m.stem for m in self.models
             if m.mmproj is not None and m.mmproj.gguf_path})
        if self.matrix_result is not None:
            self.chat_ctx = self.matrix_result.chat_ctx
        coload_stems = ({s for s, _ in self.matrix_result.coloads}
                        if self.matrix_result else set())
        # Host compositions are judged as units before anything commits
        # (requirement 33): the fixed CPU categories of a composition that
        # does not fit refuse together, before the ledger sees them.
        self.host_unit_refused_vars = self._host_unit_refusals()
        self.host_unit_refused_stems = frozenset(
            m.stem for name, m in self.fixed_categories
            if name in self.host_unit_refused_vars and m is not None)
        # Shared ledger: matrix residents + pools: overrides.  Per-model
        # chat solves charge this pool's extra reserve to their spare.
        self.ledger = self._build_ledger()
        # Per-set co-resident costs, so a chat entry can be sized for the
        # heaviest set it actually lands in rather than one fleet-wide
        # figure.  Needs the matrix result, so it runs after the solve.
        self._price_sets()

        # values are Variant lists, except the reserved admission key
        plan: dict = {}
        n_models = len(self.models)
        for i, model in enumerate(self.models, 1):
            logger.info("planning %d/%d: %s", i, n_models, model.stem)
            model.image_budget_warnings()
            if model.stem in self.host_unit_refused_stems:
                # Refused with its composition (requirement 33): no id may
                # claim it, so no variants are built at all.
                logger.info("host ledger: %s belongs to a refused unit; "
                            "entry omitted", model.stem)
                if self.progress_cb is not None:
                    self.progress_cb(model.stem)
                continue
            include_mmproj = True   # per-category now; see solve_rung
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
            # Per-serving, decided from the context this entry will actually
            # serve -- not from the fleet's matrix target. The old test keyed
            # off self.chat_ctx, which is (a) None whenever no matrix: section
            # exists (the default), so demotion could never fire at all, and
            # (b) the fleet-wide goal rather than this entry's own ctx, so a
            # model that squeezed below the threshold kept advertising tools.
            is_coload = model.stem in coload_stems

            groups = self.profiles.groups_for(view, self.vram_total, self.spare)
            variants: list[Variant] = []
            for (parallel, cache_type, spare_mb, batch, ubatch), group in groups.items():
                # Reset the per-group serving view: a prior group's
                # vision→text fallback must not leak into this group.
                include_mmproj = True   # per-category now; see solve_rung
                view = model.view_for(include_mmproj)
                context_length = view.design_context
                if view.role == "embeddings" and self.matrix_result:
                    context_length = min(context_length,
                                         self.matrix_result.embed_ctx)
                elif view.role == "rerank" and self.matrix_result:
                    context_length = min(context_length,
                                         self.matrix_result.rerank_ctx)
                cap = self._category_ctx_cap(model)
                if cap is not None and cap < context_length:
                    context_length = cap
                # Ledger charge: this pool's extra reserve (pools: overrides
                # + co-residents) joins the spare for chat solves only — RAG
                # and fixed-overhead roles ARE residents; charging them
                # themselves would double-count.
                pool_id = self.ledger.pool_id_for(view)
                spare_mb = self.ledger.spare_for(pool_id, spare_mb)
                spare_eff = spare_mb
                if view.role == "chat":
                    # Co-residents are charged per set, not fleet-wide: the
                    # ledger's single figure under-reserves every model that
                    # can also land in a heavier set (ragz's Zerank alone is
                    # 5764 MB more than rag's co-residents).
                    spare_eff += self._coresident_charge(model, pool_id)
                # Auto-parallel: unpinned chat models on slot-based backends
                # get a value-function (parallel, ctx) solve instead of the
                # group parallel.  Pinned = sidecar/block `parallel:` or any
                # profile in the group declaring `parallel`.
                auto = (
                    self.knobs.auto_parallel
                    and view.role == "chat"
                    and view.backend in ("llama-server", "vllm", "vllm-docker")
                    # A CPU-resident model's KV pool lives in host RAM, which
                    # no VRAM ledger sees or budgets, and calc_ctx returns
                    # the design context regardless of the slot count. So
                    # every candidate looks equally affordable and the score
                    # reduces to p**power, which walks straight to the cap:
                    # 8 slots on a model declared for one is an 8x KV
                    # over-commit in an arena nothing accounts for.
                    and not getattr(view, "on_cpu", False)
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
                # One solve per category, plus the fallback. Each declares
                # its own floor and its own minimum slots, and each resolves
                # its own (shed, ctx, parallel) against the same budget -- so
                # a `.coding` entry and a `.aux` entry for one model are
                # entitled to different sizes, which is the point.
                if est_error is not None:
                    ctx_size = self._unestimated_ctx(view)
                    serves = [(FALLBACK, view, parallel, ctx_size)]
                else:
                    serves = []
                    pin_ctx = self._serving_pin(view)
                    cap_ctx = context_length
                    if self.max_context is not None:
                        cap_ctx = min(cap_ctx, self.max_context)
                    for demand in (*self.demands_for(model), FALLBACK):
                        is_null = demand is FALLBACK
                        if not self.can_meet(model, demand):
                            # require gates existence: a model without the
                            # capability does not satisfy the category, so
                            # nothing may be named for it.
                            logger.info("%s: %s requires %s, which this "
                                        "model lacks; not emitted",
                                        model.stem, demand.id or "fallback",
                                        ", ".join(sorted(demand.require)))
                            continue
                        floor = resolve_min_ctx(
                            view, pin_ctx=pin_ctx,
                            tools_min_ctx=self.knobs.tools_min_ctx,
                            fallback_min_ctx=self.min_context,
                            fallback_explicit=self.min_context_explicit,
                            category_floor=demand.min_context)
                        rung = self.solve_rung(
                            model, demand, parallel=parallel,
                            cache_type=cache_type, spare_mb=spare_eff,
                            design_ctx=self.chat_ctx,
                            context_length=cap_ctx, profile=group[0][1],
                            prefer_capability=is_null)
                        if rung is None:
                            # Cannot be served at any rung: no id may claim it.
                            # The fallback is the floor on that.
                            continue
                        shed_vision = "vision" in rung.shed
                        serve_view = model.view_for(not shed_vision)
                        if auto:
                            par, ctx = self._auto_parallel(
                                serve_view, cache_type=cache_type,
                                spare_mb=spare_eff,
                                include_mmproj=not shed_vision,
                                cap_ctx=cap_ctx, floor=floor,
                                pin_ctx=pin_ctx,
                                power=0.0 if is_null else None,
                                min_parallel=demand.min_parallel,
                                group_parallel=parallel,
                                profile=group[0][1])
                        else:
                            par = parallel
                            ctx = self._bounded_ctx(
                                serve_view, parallel=par,
                                cache_type=cache_type, spare_mb=spare_eff,
                                include_mmproj=not shed_vision,
                                design_ctx=self.chat_ctx,
                                context_length=cap_ctx,
                                profile=group[0][1])
                            if ctx < demand.min_context:
                                continue
                        # Admission is judged on the serving just resolved,
                        # per category -- not on a probe taken before the
                        # solve, and not shared across the model's other
                        # categories.
                        if view.role == "chat":
                            self.record_admission(model, demand, ctx, par)
                        serves.append((demand, serve_view, par, ctx))
                    if not serves:
                        continue
                ctx_size = serves[0][3]
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
                        # `serves` came from the solve this check just
                        # invalidated; rebuild it so the emitted entry does
                        # not carry an estimated context while claiming to be
                        # unestimated.
                        serves = [(FALLBACK, view, group_parallel, ctx_size)]

                # Squeeze note: the budget could not afford the design
                # context and an explicit reservation (--spare/--baseline)
                # is what pushed it over — stamp the entry description.
                # Gate on the user-declared spare (pool-adjusted, before
                # co-resident charges) or a baseline above the fixed floor;
                # and only when the served ctx is actually below design
                # (the matrix chat_ctx target can squeeze without landing
                # under the model's own context_length).
                vram_note: str | None = None
                if (est_error is None
                        and getattr(view.vram, "vram_squeezed", False)
                        and ctx_size < context_length
                        and (spare_mb > 0 or self.baseline_mb > _RESERVE_VIDEO)):
                    vram_note = (
                        f"(over configured VRAM limits — serving at "
                        f"{ctx_size:,} of {context_length:,} design tokens; "
                        f"re-run without --spare/--baseline for full "
                        f"context)")

                # One Variant per served category. There is no
                # main/companion pairing and no vision-less twin: two entries
                # for one model is two categories in sidecars, and vision is a
                # rung each category decides for itself.
                for demand, serve_view, serve_par, serve_ctx in serves:
                    shed_vision = serve_view is not view
                    demoted = self._tools_demoted(serve_view, serve_ctx)
                    task_demands = () if demand is FALLBACK else (demand,)
                    if est_error is not None:
                        mem = (None, None, None)
                    else:
                        mem = self._variant_memory(
                            serve_view, cache_type=cache_type,
                            include_mmproj=not shed_vision,
                            ctx_size=serve_ctx, parallel=serve_par,
                            profile=group[0][1])
                    host_free = self.host_ledger.available() \
                        if serve_view.on_cpu else None
                    # A committed resident's serving is not re-charged: the
                    # unit pass already admitted (or refused) that stem's
                    # whole quad — counting it again here would always
                    # refuse a resident on a pool sized to hold only it.
                    host_extra = mem[1] if (host_free is not None
                                            and serve_view.stem
                                            not in self.host_ledger_stems
                                            ) else None
                    if (host_extra is not None and host_extra > host_free):
                        # Go/no-go on the host pool: the CPU-resident quad
                        # does not fit after committed residents. No id may
                        # claim it and there is no solve-down (requirement 12) —
                        # the operator picks a smaller sidecar or a bigger box.
                        logger.info(
                            "%s: %s over_budget_reason: host pool — %d MiB "
                            "needed, %d MiB free after residents; entry "
                            "omitted", serve_view.stem,
                            demand.id or "fallback", int(mem[1]), host_free)
                        continue
                    variants.append(Variant(
                        parallel=(0 if uncapped else serve_par),
                        cache_type=cache_type, spare_mb=spare_mb,
                        profiles_group=group, ctx_size=serve_ctx,
                        include_mmproj=not shed_vision,
                        batch=batch, ubatch=ubatch,
                        coload=is_coload,
                        tools_demoted=demoted, task_demands=task_demands,
                        category=demand,
                        estimate_error=est_error,
                        mem_vram_mib=mem[0], mem_ram_mib=mem[1],
                        kv_per_token_mib=mem[2],
                        vram_note=(vram_note if serve_ctx < context_length
                                   else None)))
            if variants:
                plan[model.stem] = variants
            if self.progress_cb is not None:
                self.progress_cb(model.stem)
        # The admission map is planner state, not a variant, but the emitter
        # is a pure function of `plan` — so it rides along under a key no
        # stem can collide with, and is popped by the caller.
        plan[SET_ADMISSION_KEY] = dict(self.set_admission)
        plan[HOST_REFUSED_KEY] = sorted(self.host_unit_refused_vars)
        return plan


def _static_params(model: Model, fit_bin: str, cache_type: str,
                   llama_args: str = "",
                   allow_cpu: bool = False,
                   ) -> tuple[int, float, float, int] | None:
    """(model_mib, kv_per_token_mib, slot_mib, compute_mib) quad, or None
    when unmeasurable.  ``allow_cpu`` lifts the CPU-resident gate — the
    host ledger sizes `device: cpu` quads through the same gate."""
    fp = model.vram.fit_params_static(fit_bin, cache_type=cache_type,
                                      llama_args=llama_args,
                                      allow_cpu=allow_cpu)
    if fp is None:
        return None
    return (fp.model_mib, fp.kv_per_token_mib, fp.slot_mib, fp.compute_mib)


def _quad_cost_mb(quad: tuple[int, float, float, int], ctx: int,
                  parallel: int = 1) -> int:
    """Affine serving cost (MiB) of one measured quad at (ctx, parallel)."""
    mib, kv_factor, slot_mib, compute = quad
    return int(mib + compute + kv_factor * ctx * parallel
               + slot_mib * parallel)


def resident_quad(model: Model, *, fit_bin: str, cache_type: str,
                  profiles, llama_args: str, allow_cpu: bool = False,
                  ) -> tuple[int, float, float, int] | None:
    """The measured quad behind :func:`resident_cost_mb`, or None.

    Exposed separately so the resident-selection log can name the terms —
    two models with the same total cost are not a bug if their weights, KV
    and compute terms match.
    """
    return _static_params(model, fit_bin, cache_type,
                          measurement_args(profiles, model, llama_args),
                          allow_cpu=allow_cpu)


def resident_cost_mb(model: Model, *, fit_bin: str, cache_type: str,
                     profiles, llama_args: str, ctx: int,
                     parallel: int = 1, allow_cpu: bool = False) -> int | None:
    """Measured serving cost (MiB) of one resident at its served context.

    The one estimator behind resident *selection*, the VRAM resident
    charge and the host resident charge: the same affine quad family
    (shape-stamped, so a `device: cpu` model measures under `-ngl 0`),
    priced at the same context.  None when nothing can size it — callers
    treat that as "not a candidate for a cost comparison", never as 0.
    """
    quad = resident_quad(model, fit_bin=fit_bin, cache_type=cache_type,
                         profiles=profiles, llama_args=llama_args,
                         allow_cpu=allow_cpu)
    if quad is None:
        return None
    return _quad_cost_mb(quad, ctx, parallel)


def resident_cost_detail(model: Model, *, fit_bin: str, cache_type: str,
                         profiles, llama_args: str, ctx: int,
                         allow_cpu: bool = False) -> tuple[int, str] | None:
    """(total MiB, term breakdown) for one resident, or None.

    The breakdown exists so two residents that both cost 1050 MiB can be
    seen to be the same *shape* — same weights, same compute buffer, same
    KV/token — rather than a repeated number masking a bug.
    """
    quad = resident_quad(model, fit_bin=fit_bin, cache_type=cache_type,
                         profiles=profiles, llama_args=llama_args,
                         allow_cpu=allow_cpu)
    if quad is None:
        return None
    mib, kv_factor, slot_mib, compute = quad
    total = _quad_cost_mb(quad, ctx)
    kv_mb = int(kv_factor * ctx + slot_mib)
    return total, (f"{mib} weights + {compute} compute + {kv_mb} kv "
                   f"@{ctx}")


def resident_ctx(model: Model, knobs: MatrixKnobs,
                 ctx_override: int | None = None) -> int:
    """The context a resident role is served at — the use-case size.

    Embedding and rerank residents serve their configured RAG cap (a client
    chunks to fit); everything else serves its declared design context.
    A per-category ``ctx:`` (portion 2) caps that further: the operator's
    declaration is the served context for the model, every consumer prices
    and serves it identically, and it is still a cap — never a floor — so
    a squeezed or design-limited serving cannot exceed it.
    """
    ctx = (model.design_context if model.role not in (
        "embeddings", "rerank") else min(model.design_context,
                                         knobs.rag_cap(model.role)))
    if ctx_override is not None:
        ctx = min(ctx, ctx_override)
    return ctx


def _sets_reference_coloads(sets_cfg: dict | None) -> bool:
    """True when any routing set can actually co-locate a co-load with chat.

    The opportunistic co-load pass reserves each candidate's VRAM against
    *every* chat solve, on the premise that llama-swap will hold it
    alongside the chat model.  That only happens if some set expression
    references the ``__COLOAD_VARS__`` placeholder — the sole way a
    co-load's var reaches a set.  A co-load with no set referencing it is
    evicted independently, so its reservation buys nothing and only
    shrinks context (observed 2026-09-26: two unreferenced ASR co-loads
    charged 4725 MB and collapsed a 27B's context from 188416 to 32768).
    Such models belong in ``__CHAT_VARS__`` instead — alternatives that
    evict chat rather than co-residents that shrink it.
    """
    return any("__COLOAD_VARS__" in str(expr)
               for expr in (sets_cfg or {}).values())


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
    category_ctx: dict[str, int] | None = None,
    coload_sets: dict | None = None,
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
       Skipped entirely when no routing set references
       ``__COLOAD_VARS__`` (*coload_sets*): a co-load nothing can
       co-locate is never resident with chat, so reserving for it would
       only cost context.

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
    ctx_by_cat = category_ctx or {}
    embed_ctx = resident_ctx(embed_model, knobs, ctx_by_cat.get("emb"))
    rerank_ctx = resident_ctx(rerank_model, knobs, ctx_by_cat.get("rnk"))

    # The routing sets are the budget contract (design decision 13): a set
    # expression says which models are allowed to be resident together, so
    # the chat solve reserves the most expensive *branch* any chat-capable
    # set allows — not the sum of every category the operator declared.
    # ``(3 | (2 & 1))`` costs 3; ``(1 | (2 & 3))`` costs 5.  A category no
    # chat set names reserves nothing, because llama-swap can never hold it
    # beside chat ("a model in no set can only run alone").
    raw_sets = {str(n): str(e) for n, e in (coload_sets or {}).items()}
    chat_sets = {n: e for n, e in raw_sets.items()
                 if _has_chat_slot(e, raw_sets)}
    fixed_mb: dict[str, int] = {}
    for name, m in fixed_categories:
        if m.on_cpu:
            mb = 0
        else:
            mb = _coload_overhead(m, fit_bin, profiles, knobs, llama_args,
                                  ctx=ctx_by_cat.get(name))
            if mb is None:
                logger.warning("matrix: category %r (%s) cannot be sized; "
                               "charged 0", name, m.stem)
                mb = 0
        fixed_mb[name] = mb
        fixed_mb.setdefault(m.stem, mb)
    if raw_sets:
        leaves: set[str] = set()
        for expr in raw_sets.values():
            leaves |= _expr_leaves(expr, raw_sets)
        for name, m in fixed_categories:
            if name not in leaves and m.stem not in leaves:
                logger.info(
                    "matrix: category %r (%s) is in no set; it can only "
                    "run alone, so no VRAM is reserved for it",
                    name, m.stem)

    def _set_costs(embed_ctx_: int, rerank_ctx_: int,
                   coload_mb: int = 0) -> dict[str, int]:
        """Per-set worst-branch cost at the given RAG contexts."""
        var_mb = dict(fixed_mb)
        emb_mb = (_quad_cost_mb(embed_params, embed_ctx_)
                  if embed_params is not None else 0)
        rnk_mb = (_quad_cost_mb(rerank_params, rerank_ctx_)
                  if rerank_params is not None else 0)
        var_mb["emb"] = emb_mb
        var_mb["rnk"] = rnk_mb
        # llama-swap accepts a model id as a leaf just as it accepts a var.
        if embed_model is not None:
            var_mb[embed_model.stem] = emb_mb
        if rerank_model is not None:
            var_mb[rerank_model.stem] = rnk_mb
        return {name: set_overhead_mb(expr, var_mb, frozenset(), frozenset(),
                                      sets_cfg=raw_sets, coload_mb=coload_mb)
                for name, expr in chat_sets.items()}

    def _overhead(embed_ctx_: int, rerank_ctx_: int,
                  extra_mb: int = 0) -> int:
        """VRAM held by non-chat residents at these RAG contexts.

        In set mode this is the whole non-chat side of the worst branch
        (RAG pair included), so the solver is given no separate embed/rerank
        params.  In legacy mode the solver still prices the RAG pair from
        its params; *extra_mb* carries the declared categories and co-loads
        the caller admits.
        """
        if not raw_sets:
            return extra_mb
        return max(_set_costs(embed_ctx_, rerank_ctx_, extra_mb).values(),
                   default=0)

    def _solve(embed_ctx_: int, rerank_ctx_: int,
               extra_mb: int = 0) -> int:
        # With sets, the expression tree owns the RAG pair too: passing
        # their params as well would charge them twice.
        set_mode = bool(raw_sets)
        return solve_matrix_ctx(
            vram_total_mb=vram_total,
            spare_mb=spare_mb,
            chat_models=chat_params,
            embed_params=None if set_mode else embed_params,
            rerank_params=None if set_mode else rerank_params,
            embed_ctx=0 if set_mode else embed_ctx_,
            rerank_ctx=0 if set_mode else rerank_ctx_,
            baseline_mb=baseline_mb,
            fixed_overhead_mb=_overhead(embed_ctx_, rerank_ctx_, extra_mb),
            memory_margin=memory_margin,
        )

    # 1. Baseline.
    chat_ctx = _solve(embed_ctx, rerank_ctx)
    if chat_sets:
        costs = _set_costs(embed_ctx, rerank_ctx)
        worst = max(costs, key=costs.get)
        logger.info(
            "matrix: budget follows sets — worst %r reserves %d MB "
            "(chat_ctx=%d, %d set(s))",
            worst, costs[worst], chat_ctx, len(chat_sets))

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

        def _admit(oh: int, stem: str, declared: bool) -> int | None:
            """Admit *oh* MB more if chat stays at/above the floor."""
            nonlocal used
            ctx = _solve(embed_ctx, rerank_ctx, used + oh)
            if ctx < floor:
                return None
            used += oh
            if not declared:
                coloads.append((stem, oh))
            return ctx

        # 3a. Legacy mode (no routing sets): declared categories are
        #     reserved first, one at a time; one that does not fit is
        #     reported and left unreserved.  With sets, categories are
        #     already inside the set cost tree above — adding them again
        #     would double-charge and hide the alternatives.
        if not raw_sets:
            for name, m in fixed_categories:
                oh = _coload_overhead(m, fit_bin, profiles, knobs,
                                      llama_args,
                                      ctx=ctx_by_cat.get(name))
                if oh is None:
                    logger.warning(
                        "matrix: category %r (%s) skipped: cannot size it",
                        name, m.stem)
                    continue
                ctx = _admit(oh, m.stem, declared=True)
                if ctx is None:
                    logger.warning(
                        "matrix: category %r (%s) does not fit below the "
                        "chat floor %d; not reserved", name, m.stem, floor)
                else:
                    logger.info(
                        "matrix: category %r (%s) reserved (%d MB, "
                        "chat_ctx=%d)", name, m.stem, oh, ctx)

        # 3b. Opportunistic co-loads: s2t/image models that are *not* already
        #     a declared category, smallest fixed overhead first.
        #     Gated on some set referencing __COLOAD_VARS__: reserving a
        #     co-load nothing can co-locate charges every chat solve for
        #     residency that never happens.  Models too big to live beside
        #     chat belong in __CHAT_VARS__ (they evict chat instead).
        if not _sets_reference_coloads(coload_sets):
            logger.info(
                "matrix: no routing set references __COLOAD_VARS__; "
                "skipping opportunistic co-loads (they cannot be "
                "co-located with chat, so reserving their VRAM would only "
                "shrink context — put them in __CHAT_VARS__ to have them "
                "evict chat instead)")
        else:
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
                ctx = _admit(oh, stem, declared=False)
                if ctx is None:
                    logger.warning(
                        "matrix: co-load %s skipped: would drop chat ctx below "
                        "the floor %d", stem, floor)
                else:
                    logger.info(
                        "matrix: co-load %s included (%d MB, chat_ctx=%d)",
                        stem, oh, ctx)
    return MatrixSolve(
        chat_ctx=chat_ctx, embed_ctx=embed_ctx, rerank_ctx=rerank_ctx,
        coloads=tuple(coloads), squeeze=squeeze,
    )


def _coload_overhead(
    m: Model, fit_bin: str, profiles: Profiles, knobs: MatrixKnobs,
    llama_args: str = "", ctx: int | None = None,
) -> int | None:
    """Fixed VRAM overhead (MB) of a declared category / co-load candidate.

    Uses the backend's effective static params (weights + fixed compute; the
    KV terms are 0 for these backends) at *ctx* (default: the model's design
    context, or the role's RAG cap — see :func:`resident_ctx`).  Overheads
    that are *estimated* rather than measured or pinned carry
    ``estimate_headroom`` so a bad guess errs toward reserving more.
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
    service_ctx = resident_ctx(m, knobs) if ctx is None else ctx
    overhead = model_mib + compute_mib + int(
        kv_factor * service_ctx + slot_mib)
    if not measured:
        overhead = int(overhead * knobs.estimate_headroom)
    return overhead


# ── Emission ──────────────────────────────────────────────────────────────


def emit_config(models: list[Model], plan: dict,
                profiles: Profiles, template_vars: dict) -> EmittedConfig:
    """Render planned :class:`Variant`s into the llama-swap config dict.

    Pure transformation — no VRAM math, no I/O. Each variant becomes one
    entry.  Entry ids are **stable across repacks**: they name the task, not
    the solved context.  Grammar ``<template_id>.<category>.<flags>`` where

    - ``<category>`` is the ``usage:`` category this entry serves; omitted
      for an untagged model or one that meets no category, which then gets
      the bare stem;
    - ``<flags>`` is the served capability segment (:func:`flags_label`),
      omitted entirely for a plain text-in/text-out serving.

    Capability/value detail lives in the display ``name`` and in
    ``metadata`` (``ctx_size``, ``usage``), never in the id. The id used to
    carry the solved context, and opencode/hermes cache it at startup -- so
    182k -> 180k renamed the model out from under every client that had
    cached it. The context is not part of a serving's identity.

    Raises ValueError on duplicate entry ids (two models slugging to the same
    id); callers surface it as a fatal configuration error.  Returns an
    :class:`EmittedConfig` whose ``entry_ids_by_stem`` maps each model stem to
    the ids it produced.
    """
    catalogue = TaskCatalogue.from_cfg(getattr(profiles, "tasks_cfg", None))
    entries: dict[str, dict] = {}
    owner: dict[str, str] = {}  # entry id → model stem (collision detection)
    ids_by_stem: dict[str, list[str]] = {}
    category_of: dict[str, str | None] = {}
    coload_stems: set[str] = set()
    # Per-model tails already used: two profile groups (e.g. different
    # cache_type) can land on the same category+flags, and they still need
    # distinct ids.  The first keeps the stable name; later duplicates append
    # their context label (deterministic given the plan's iteration order).
    used_tails: dict[str, set[str]] = {}

    def unique_tail(stem: str, tail: str, key: str) -> str:
        """Disambiguate two servings of one stem that share a tail.

        *key* must describe the serving, never the solve. It used to be the
        context label, which this module's own docstring rules out: a
        context-derived suffix changes whenever VRAM changes, and
        opencode/hermes cache entry ids at startup, so a repack silently
        repoints a cached id at a different physical serving -- and where
        two entries of a stem can swap emission order between packs, the
        bare tail changes which serving it names. Cache type and whether the
        mmproj is attached are operator-declared and do not move with the
        solve, which is exactly the property an id needs.
        """
        taken = used_tails.setdefault(stem, set())
        cand = tail
        n = 1
        while cand in taken:
            n += 1
            cand = f"{tail}-{key}" if n == 2 else f"{tail}-{key}-{n}"
        taken.add(cand)
        return cand

    def id_and_suffix(v: Variant, ctx: int, flags: str) -> tuple[str, str]:
        """The ``.<category>.<flags>`` id tail and the display suffix.

        *flags* is the served capability segment (:func:`served_flags`).
        """
        # The category is the variant's own, read straight off it: a rung
        # answers exactly one category, so there is no "heaviest satisfied"
        # search to run.
        cats = v.task_demands
        cat = cats[0].id if cats and cats[0].id else None
        # The category is operator-chosen; a name that does not survive id
        # slugging (or slugs empty) falls back to the context label so the id
        # stays within the model_id charset (opencode/llama-swap/shell-safe).
        cat_id = utils.slugify(cat) if cat is not None else ""
        if not cat_id:
            cat = None
        # The id never carries the solved context. A client that cached
        # `...256k` sees a *different model* after any repack that moved the
        # context by one rounding step — 182k -> 180k is not a change of
        # serving, it is a change of name, and clients treat the name as
        # identity. The context belongs in the display name and in
        # `metadata.ctx_size`, both of which a client re-reads.
        tail = ".".join(x for x in (cat_id, flags) if x)
        disp = f" · {cat}" if cat is not None else ""
        disp += f" {ctx_label(ctx)}"          # display only, never the id
        if flags:
            disp += f" [{flags}]"
        return tail, disp

    for model in models:
        for v in plan.get(model.stem, []):
            view = model.view_for(v.include_mmproj)
            context_length = view.design_context
            if v.coload:
                coload_stems.add(model.stem)
            flags = served_flags(view, v.include_mmproj, v.tools_demoted)
            tail, disp = id_and_suffix(v, v.ctx_size, flags)
            tail = unique_tail(model.stem, tail,
                                f"{v.cache_type}"
                                f"{'-vis' if v.include_mmproj else ''}")
            entry_id, entry = _build_entry(
                view, v.parallel, v.cache_type, v.profiles_group,
                profiles.defaults, template_vars, context_length, v.ctx_size,
                include_mmproj=v.include_mmproj,
                name_suffix=disp,
                entry_id_tail=tail,
                tools_demoted=v.tools_demoted,
                task_demands=v.task_demands,
                estimate_error=v.estimate_error,
                batch=v.batch, ubatch=v.ubatch,
                mem_vram_mib=v.mem_vram_mib, mem_ram_mib=v.mem_ram_mib,
                kv_per_token_mib=v.kv_per_token_mib,
                vram_note=v.vram_note,
            )
            if entry_id in entries:
                raise ValueError(
                    f"duplicate entry id {entry_id!r}: model {model.stem!r} "
                    f"collides with {owner[entry_id]!r} — rename one of them")
            entries[entry_id] = entry
            owner[entry_id] = model.stem
            ids_by_stem.setdefault(model.stem, []).append(entry_id)
            category_of[entry_id] = (v.category.id if v.category else None)

    set_admission = plan.pop(SET_ADMISSION_KEY, {}) or {}
    host_refused = tuple(sorted(plan.pop(HOST_REFUSED_KEY, ()) or ()))
    config = EmittedConfig(entry_ids_by_stem=ids_by_stem,
                           coload_stems=sorted(coload_stems),
                           set_admission=set_admission,
                           entry_category=category_of,
                           host_unit_refused=host_refused)
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
                 coload_stems: list[str] | None = None,
                 set_admission: dict | None = None,
                 entry_category: dict[str, str | None] | None = None,
                 set_overheads: dict[str, int] | None = None,
                 host_unit_refused: tuple[str, ...] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_ids_by_stem: dict[str, list[str]] = entry_ids_by_stem or {}
        self.coload_stems: list[str] = coload_stems or []
        # entry id -> the category it serves (None for the fallback). Carried
        # because the id names the category, so it is the emitter that knows.
        self.entry_category: dict[str, str | None] = entry_category or {}
        # (model stem, category) -> the chat sets that entry may join.
        self.set_admission: dict[tuple[str, str | None], frozenset[str]] = (
            set_admission or {})
        # category var names refused by the host-unit pass (requirement 33).
        # The emitter dropped their stems; the caller prunes the vectors
        # and set leaves that named them.
        self.host_unit_refused: tuple[str, ...] = host_unit_refused
        # set name -> MiB allocated to that set's co-resident models (emb/rnk
        # and any fixed categories).  Emitted as a comment above each `sets:`
        # entry so the config documents its own memory split.
        self.set_overheads: dict[str, int] = set_overheads or {}

    def plain(self) -> dict:
        return dict(self)


def _verify_no_overallocation(config: EmittedConfig, plan: dict,
                              models: list[Model], planner: "Planner") -> None:
    """Re-derive every emitted entry's VRAM; drop any that cannot fit.

    The invariant this tool exists to keep: no emitted entry asks for more
    VRAM than the card has. Every solve upstream honours it, but there are
    several passes and several clamps, and one edit to any of them could
    reopen the hole without anything noticing. So the last step recomputes
    each entry's footprint from the numbers actually written out -- its own
    ctx_size and parallel, under the same affine law -- and compares that
    against the budget the planner was given.

    A violation here is a bug, not an operator condition: a model that simply
    does not fit is handled upstream by omission, which is the contract. So
    it is logged at ERROR, the entry is removed, and the pack still emits
    everything that does fit -- impossible to miss, not fatal.
    """
    from llama_packer.vram import _RESERVE_SYSTEM, _RESERVE_VIDEO

    reserve = _RESERVE_SYSTEM + max(_RESERVE_VIDEO, planner.baseline_mb)
    spare_mb = planner.profiles.global_spare_mb(planner.spare, planner.vram_total)
    available = int((planner.vram_total - reserve - spare_mb)
                    / (1.0 + planner.memory_margin))

    by_stem = {m.stem: m for m in models}
    # entry_ids_by_stem is the emitter's own answer to "which model produced
    # this id" -- the same mapping __main__ uses to build matrix vars.
    owner = {eid: stem for stem, ids in config.entry_ids_by_stem.items()
             for eid in ids}
    removed: list[str] = []
    for entry_id, entry in list(config.get("models", {}).items()):
        model = by_stem.get(owner.get(entry_id, ""))
        if model is None:
            continue
        served = entry.get("metadata", {}).get("ctx_size")
        for variant in plan.get(model.stem, ()):
            if variant.ctx_size != served:
                continue
            profile = (variant.profiles_group[0][1]
                       if variant.profiles_group else None)
            quad = _static_params(
                model, planner.fit_bin, variant.cache_type,
                measurement_args(planner.profiles, model, planner.llama_args,
                                 profile))
            if quad is None:
                break
            model_mib, kv, slot_mib, compute_mib = quad
            parallel = max(1, variant.parallel)
            need = int(model_mib + compute_mib
                       + kv * variant.ctx_size * parallel
                       + slot_mib * parallel)
            if need > available:
                logger.error(
                    "invariant violated: %s needs %d MiB at ctx=%d p=%d but "
                    "only %d MiB is available — dropped. This is a planner "
                    "bug, not an operator condition.", entry_id, need,
                    variant.ctx_size, parallel, available)
                removed.append(entry_id)
            break

    if not removed:
        return
    for entry_id in removed:
        config["models"].pop(entry_id, None)
    config.entry_ids_by_stem = {
        stem: [i for i in ids if i not in removed]
        for stem, ids in config.entry_ids_by_stem.items()}


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
    category_ctx: dict[str, int] | None = None,
    baseline_mb: int = 0,
    min_context: int = _MIN_AGENTIC_CTX,
    min_context_explicit: bool = False,
    memory_margin: float = _MEMORY_MARGIN,
    host_pool_mb: int | None = None,
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
            projection is shed from the entry, which then carries no
            ``i``/``v`` id flag.  Pass
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
        category_ctx=category_ctx,
        baseline_mb=baseline_mb,
        min_context=min_context, min_context_explicit=min_context_explicit,
        memory_margin=memory_margin,
        llama_args=template_vars.get("llama_args", ""),
        host_pool_mb=host_pool_mb,
        progress_cb=progress_cb,
    )
    plan = planner.plan()
    config = emit_config(supported, plan, profiles, template_vars)
    _verify_no_overallocation(config, plan, supported, planner)
    config.set_overheads = dict(planner.set_overheads)
    return config


class _LiteralDumper(yaml.Dumper):
    """YAML dumper that keeps newlines intact via literal blocks.

    PyYAML's default renders multi-line strings as folded (single-quoted)
    scalars, which collapse load-bearing newlines (e.g. audio-cpp's
    heredoc ``cmd``) into spaces.  Forcing ``|`` style on strings that
    contain a newline makes the round-trip exact.  Scoped to this subclass
    so no global ``yaml.add_representer`` side effects leak elsewhere.
    """


def _literal_str_representer(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _literal_str_representer)


def annotate_set_memory(yaml_text: str, set_overheads: Mapping[str, int],
                        ) -> str:
    """Insert a ``# <set>: N MiB`` comment above each matrix set.

    PyYAML cannot emit comments, so the already-serialized text is walked:
    only direct children of the ``sets:`` mapping are annotated (matched by
    indentation), so a model id or var that happens to share a set's name is
    never touched.  No-op when there are no priced sets.
    """
    if not set_overheads:
        return yaml_text
    lines = yaml_text.splitlines()
    out: list[str] = []
    in_sets = False
    sets_indent = -1
    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped == "sets:":
            in_sets = True
            sets_indent = indent
            out.append(line)
            continue
        if in_sets:
            if stripped and indent <= sets_indent:
                in_sets = False
            elif stripped and ":" in stripped:
                key = stripped.split(":", 1)[0].strip().strip("\"'")
                mb = set_overheads.get(key)
                if mb is not None:
                    pad = " " * indent
                    out.append(f"{pad}# {key}: {mb} MiB allocated to its "
                               f"co-resident models ({mb / 1024:.2f} GiB)")
        out.append(line)
    text = "\n".join(out)
    return text + "\n" if yaml_text.endswith("\n") else text


def dump_yaml(payload: dict, set_overheads: Mapping[str, int] | None = None,
              ) -> str:
    """Serialize *payload* to a YAML string, preserving embedded newlines.

    When *set_overheads* is given, each matrix set gains a comment recording
    the VRAM allocated to its co-resident models.
    """
    text = yaml.dump(payload, default_flow_style=False, sort_keys=False,
                     allow_unicode=True, Dumper=_LiteralDumper)
    return annotate_set_memory(text, set_overheads or {})


def write_yaml(config: dict, path: Path | str,
               set_overheads: Mapping[str, int] | None = None) -> None:
    """Write config to YAML file.

    *set_overheads* defaults to ``config.set_overheads`` when *config* is an
    :class:`EmittedConfig`, so callers get the matrix-set memory comments
    without threading the planner result through by hand.
    """
    if set_overheads is None:
        set_overheads = getattr(config, "set_overheads", None)
    payload = config.plain() if isinstance(config, EmittedConfig) else config
    with open(path, "w") as f:
        f.write(dump_yaml(payload, set_overheads))
