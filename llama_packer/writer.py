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
from typing import Callable, Iterable

import yaml

from llama_packer.model import Model
from llama_packer.hardware import detect_gpu_env_var
from llama_packer import utils
from llama_packer.endpoints import (CHAT, EMBEDDINGS, IMAGES, RERANK,
                                   SPEECH, TRANSCRIPTIONS)
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
    _RESERVE_SYSTEM,
    _RESERVE_VIDEO,
)
from llama_packer import allocation, roles
from llama_packer.profiles import Profiles
from llama_packer.backends import (
    SETTING_KEYS,
    FRAMEWORK_CONSUMED,
    METADATA_ONLY,
    VLLM_BACKENDS,
    get_backend,
)

logger = logging.getLogger(__name__)

#: A server process always has a small host footprint.
HOST_PROCESS_MIB = 128

# llama-server ``--reasoning-format`` modes (see its CLI help).
_REASONING_FORMATS = frozenset({"none", "deepseek", "deepseek-legacy", "auto"})
_REASONING_FLAG_KEYS = ("reasoning-format", "reasoning-preserve")

def _model_can_reason(model: Model) -> bool:
    """True when the model is a chat model that advertises reasoning support."""
    if not model.serves_endpoint(CHAT):
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
    host-RAM serving (``RAM``-only tag).  ``spill_mib`` (driver-managed
    overflow, ``VRAM cap + spill``) and ``ssd_mib`` (on-disk weights)
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
                if not model.serves_endpoint(IMAGES):
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


#: Default idle lifetime for a host-RAM helper with no declared value.
#: Capacity pools bound RAM; TTL additionally yields to duties llama-swap
#: cannot see. A chat serving on CPU inherits the global default instead.
def resolve_entry_ttl(model: Model, demand: "Task | None",
                      task_ttl: int | None = None,
                      role_ttl: int | None = None) -> int | None:
    """The idle-unload lifetime for one entry, in seconds.

    Most specific wins: sidecar > task > role > top-level ``ttl``. A device
    entry with nothing declared returns None, so llama-swap's ``globalTTL``
    applies.
    """
    if "ttl" in getattr(model, "sidecar_keys", ()):
        raw = model.frontmatter.get("ttl")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            logger.warning("%s: ttl=%r is not a non-negative integer; "
                           "ignored", model.label, raw)
        else:
            return raw
    if demand is not None and demand.ttl is not None:
        return demand.ttl
    if task_ttl is not None:
        return task_ttl
    if role_ttl is not None:
        return role_ttl
    return None


def _task_ttl(profiles: Profiles) -> dict[str, int]:
    """Per-task TTL defaults, from the task catalogue. ``0`` wins."""
    return {task.id: task.ttl
            for task in TaskCatalogue.from_cfg(profiles.tasks_cfg).by_id.values()
            if task.ttl is not None}


def _entry_task_ttl(model: Model, demand: "Task | None",
                    task_ttl: dict[str, int] | None) -> int | None:
    """The task-level TTL for one entry: its demand, or the model's tasks."""
    if not task_ttl or demand is None:
        return None
    return task_ttl.get(demand.id)


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
    ttl: int | None = None,
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
    if model.serves_endpoint(IMAGES):
        in_mods = ["text", "image"]
        # architecture or explicit video token decides output modality
        if "video" in caps_l or is_video_arch:
            out_mods = ["video"]
        else:
            out_mods = ["image"]
    elif model.serves_endpoint(TRANSCRIPTIONS):
        in_mods = ["audio"]
        out_mods = ["text"]
    elif model.serves_endpoint(SPEECH):
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
    if (not include_mmproj and not model.serves_endpoint(IMAGES)
            and model.mmproj and model.mmproj.gguf_path):
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
    if model.serves_endpoint(CHAT):
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
        "reranker": "reranker" in caps_l
                    or model.serves_endpoint(RERANK),
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

    # Idle unload is a resolved lifetime, not a placement rule.  The most
    # specific declared value wins; otherwise the entry inherits the
    # top-level globalTTL.
    if ttl is not None:
        entry["ttl"] = ttl

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


@dataclass(frozen=True)
class MatrixKnobs:
    """Tunables of the matrix solve, from the ``matrix:`` config section.

    Context tiers, smallest to largest: ``coload_min_ctx`` (emb/rnk squeeze
    floor) → ``min_chat_ctx`` (co-load decision floor) → ``tools_min_ctx``
    (tools advertisement threshold).  ``embed_context`` / ``rerank_context``
    are RAG caps — the maximum context those fixed-overhead roles serve at (default
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
        for key in ("embed_context", "rerank_context"):
            if cfg.get(key) is not None:
                logger.warning("matrix: %s is retired; set max_context on the "
                               "embeddings/rerank task instead", key)
        for key in ("min_chat_ctx", "tools_min_ctx", "coload_min_ctx",
                    "ctx_gain_min", "auto_parallel_max",
                    "vllm_auto_parallel_max"):
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
def fallback_task(model: Model,
                  empty_min: int = _MIN_CTX_SIZE) -> Task:
    """The bare ``<stem>`` entry: no endpoint, the cheapest declared task.

    A client may call the model by its bare id. The entry serves it at the
    smallest floor among the model's tasks, so the alias never claims a
    larger serving than the model already declares. A task floor above the
    design context is unreachable and is not a candidate.
    """
    design = model.design_context
    tasks = [task for task in (model.tasks_resolved or ())
             if (task.min_context or 0) <= design]
    if tasks:
        task = min(tasks, key=lambda t: (t.min_context or empty_min))
        return Task("", "", min_context=task.min_context or empty_min,
                    min_parallel=1, require=frozenset(), sacrifice=())
    return Task("", "", min_context=empty_min, min_parallel=1,
                require=frozenset(), sacrifice=())


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
    if (not include_mmproj and not model.serves_endpoint(IMAGES)
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
        return max(pin_ctx, _MIN_CTX_SIZE)
    if category_floor is not None:
        return max(category_floor, _MIN_CTX_SIZE)
    if "tools" in [c.lower() for c in (view.capabilities or [])]:
        return max(fallback_min_ctx if fallback_explicit
                   else tools_min_ctx, _MIN_CTX_SIZE)
    if fallback_explicit:
        return max(fallback_min_ctx, _MIN_CTX_SIZE)
    return max(int(view.design_context // 2), _MIN_CTX_SIZE)


# Size buckets for footprint accounting: footprints bucket by size class, not
# model type.  Tiny servings (speech, RAG) ride along beside chat for pocket
# change; huge servings (diffusion) are chat-scale and tracked as exclusive
# candidates; chat models evict each other.
BUCKET_TINY_ENDPOINTS = frozenset({EMBEDDINGS, RERANK,
                                   TRANSCRIPTIONS, SPEECH})
BUCKET_HUGE_BACKENDS = frozenset({"sd-server"})


def footprint_bucket(model) -> str:
    """Size bucket of a model for ledger accounting: tiny/huge/chat."""
    needed = {t.endpoint for t in model.tasks_resolved}
    if needed and needed <= BUCKET_TINY_ENDPOINTS:
        return "tiny"
    if model.serves_endpoint(IMAGES) or model.backend in BUCKET_HUGE_BACKENDS:
        return "huge"
    return "chat"


class PoolLedger:
    """One shared VRAM ledger: every pool's declared size and spare.

    Single-pool math today (the detected total), keyed by pool id so
    per-GPU attribution slots in later: ``pool_id_for`` maps a model's
    ``device:`` pin to ``gpuN``, unpinned models land on ``default``.  An
    operator-declared ``pools: <id>: vram:`` overrides the pool size and a
    per-pool ``spare:`` overrides the global spare.
    """

    def __init__(self, vram_total_mb: int, pools_cfg: dict | None = None):
        self.vram_total_mb = vram_total_mb
        self.pools_cfg = pools_cfg or {}

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


class HostLedger:
    """Sibling of PoolLedger: the host-RAM pool for ``device: cpu`` models.

    A CPU-served entry is judged go/no-go against the pool — never solved
    down, mirror of the VRAM `over_budget` refusal (SPEC.md §Host pool).
    ``pool_mb=None`` (detection unavailable) means the ledger cannot gate:
    it logs that once and reports an unbounded pool rather than inventing
    one.
    """

    def __init__(self, pool_mb: int | None):
        self.pool_mb = pool_mb

    def available(self) -> int | None:
        """Host pool size; None = cannot gate."""
        return self.pool_mb


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
    tools_demoted: bool = False
    #: The categories this rung is named for. A rung answers exactly one, so
    #: this is normally a single demand; a tuple only so the emitter can
    #: express "nothing declared" without a second field. Empty for the
    #: fallback entry, which carries no category name in its id.
    task_demands: tuple[Task, ...] = ()
    #: The category this variant answers, or the bare fallback task
    #: category. Singular, because a rung exists to serve one category -- which
    #: is what lets one model emit several entries and stay unambiguous.
    category: Task | None = None
    #: Non-None when the model has no usable VRAM estimate — surfaced to
    #: clients as metadata.estimated=false / metadata.estimate_error.
    estimate_error: str | None = None
    #: Decided TOTAL memory attributable to this serving (weights +
    #: companions + context at the served ctx/slots), in MiB — the numbers
    #: the emitted description tag reports verbatim.  ``mem_vram_mib=None``
    #: is a host-RAM serving (RAM-only tag).  Both None when undecided
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
    keys > fleet ``llama_server: batch:/ubatch:``.  ``None`` means "nothing
    configured": the backend renders no flag and the engine's builtin
    applies.  One resolution site: the rendered command and the VRAM
    measurement shape both consume these values, so the serve and the
    estimate can never drift apart on batch flags."""
    return model.batch_ubatch_for(
        profile, profiles.llama_server_cfg if profiles else {})


def measurement_args(
    profiles: Profiles, model: Model, global_args: str = "",
    profile: dict | None = None,
) -> str:
    """The exact flag string VRAM constants are measured (and shape-stamped)
    under: global args + the CPU device pin when the model is
    CPU-served (`--n-gpu-layers 0`, the measurement of the host-side
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


def _inline_set_refs(raw: dict[str, str]) -> dict[str, str]:
    """Inline +set references; reject cycles and any other + form."""
    def resolve(name: str, stack: tuple[str, ...]) -> str:
        def repl(match):
            ref = match.group(1)
            if ref.startswith("auto"):
                raise ValueError(f"+{ref} cannot be priced; name the roles")
            if ref not in raw:
                raise ValueError(f"+{ref} names no set")
            if ref in stack:
                raise ValueError(
                    "set reference cycle: " + " -> ".join(stack + (ref,)))
            return "(" + resolve(ref, stack + (ref,)) + ")"
        return re.sub(r"\+([A-Za-z0-9._:-]+)", repl, raw[name])

    return {name: resolve(name, (name,)) for name in raw}


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
        # A bare int is MiB (already resolved); anything declared is a string
        # that resolve_spare re-parses. Without this, 64 MiB becomes 64 GiB.
        self.spare = (spare if isinstance(spare, str) or spare is None
                      else f"{int(spare)}m")
        self.max_context = max_context
        self.matrix_cfg = matrix_cfg
        # `matrix.sets` belongs to the emitted router; the solve reads knobs.
        self.knobs = MatrixKnobs.from_cfg(matrix_cfg)
        # Blessed usage categories.  A model may only *tag* itself with one
        # of these; an unblessed tag is warned about and ignored.
        self.tasks = TaskCatalogue.from_cfg(
            getattr(profiles, "tasks_cfg", None))
        self.baseline_mb = baseline_mb
        self.min_context = min_context
        self.min_context_explicit = min_context_explicit
        self.memory_margin = memory_margin
        self.ledger = PoolLedger(vram_total, self.profiles.pools_cfg)
        # ``pools.ram.spare`` is the host-pool equivalent of the GPU spare:
        # keep that much out of both the ledger and the emitted pool.
        ram_spare = self.profiles.pools_cfg.get("ram", {}).get("spare")
        if ram_spare is not None and host_pool_mb is not None:
            host_pool_mb = max(0, host_pool_mb - utils.parse_mem_mb(
                str(ram_spare), host_pool_mb))
        self.host_ledger = HostLedger(host_pool_mb)
        # Capacity-router pools.  The GPU limit is the same budget the solve
        # used (total minus reserve and spare); the host limit is the
        # detected/declared host pool.  One size-1 pool per scheduling role
        # makes "one served entry per role" a capacity constraint.
        reserve = _RESERVE_SYSTEM + max(_RESERVE_VIDEO, baseline_mb)
        spare_mb = profiles.global_spare_mb(self.spare, vram_total)
        default_spare = self.ledger.spare_for("default", spare_mb)
        pools: dict[str, int] = {}
        gpu_pool = int(vram_total - reserve - default_spare)
        if gpu_pool > 0:
            pools["default"] = gpu_pool
        if host_pool_mb is not None and host_pool_mb > 0:
            pools["ram"] = int(host_pool_mb)
        for role in roles.names():
            pools[role] = 1
        self.capacity_pools = pools
        # Auction budgets for the authored sets, keyed by (role, pool).
        self.role_budget: dict[tuple[str, str], int] = {}
        self.matrix_sets_approved: dict[str, str] = {}
        self.matrix_sets_rejected: dict[str, str] = {}
        # Optional per-model tick (stem) for the progress bar; None in tests.
        self.progress_cb = progress_cb

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
        ctx = model.vram.solve_context(
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

        Type-based floor: a fixed-overhead serving uses its task's
        ``max_context`` (capped by the trained design context and
        ``--max-context``); chat gets the per-model floor cascade.
        """
        if view.serves_endpoint(EMBEDDINGS) or view.serves_endpoint(RERANK):
            ctx = served_ctx(view, self.tasks)
        else:
            ctx = resolve_min_ctx(
                view, pin_ctx=self._serving_pin(view),
                tools_min_ctx=self.knobs.tools_min_ctx,
                fallback_min_ctx=self.min_context,
                fallback_explicit=self.min_context_explicit)
        cap = view.design_context
        task_cap = self.tasks.max_context_for(view.task_ids)
        if task_cap is not None:
            cap = min(cap, task_cap)
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

        This gate used to be in ``met_demands``, which left a text-only
        model free to claim ``require: [vision]`` -- an id that promised a
        capability the entry did not advertise.
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
            if ctx <= 0 or ctx < (demand.min_context or self.min_context):
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
        return (view.serves_endpoint(CHAT)
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
    ) -> tuple[int, int] | None:
        """Solve (parallel, ctx) for one unpinned chat variant.

        Scores ``(ctx/floor)^0.5 × slots^B`` over feasible candidates and
        returns the winner.  Feasibility for a slot count comes from the
        existing budget path (:meth:`_bounded_ctx`); the score rises with
        ctx, so only the max affordable ctx per slot count is scored.  The
        loop breaks at the first unaffordable slot count (affordability
        falls as slots rise).

        ``floor`` (the task's ``min_context``) and ``min_parallel`` are both
        floors.  No candidate that meets them means no entry for this task —
        the same rule as an unreachable context.  ``parallel_power`` is the
        knob that makes the solver try harder for slots; it never lowers a
        floor.
        """
        power = self.knobs.parallel_power if power is None else power
        # vLLM's continuous batching makes a slot cost KV only (no per-slot
        # workspace), so concurrency is cheap — cap it higher there.
        pmax = (self.knobs.vllm_auto_parallel_max
                if view.backend in VLLM_BACKENDS
                else self.knobs.auto_parallel_max)

        solved: dict[int, int] = {}

        def maxctx(p: int) -> int:
            # Memoized: the loop shares one solve per slot count.
            if p not in solved:
                solved[p] = self._bounded_ctx(
                    view, parallel=p, cache_type=cache_type,
                    spare_mb=spare_mb, include_mmproj=include_mmproj,
                    context_length=cap_ctx, profile=profile)
            return solved[p]

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
        if min_parallel > 1:
            cands = [c for c in cands if c[0] >= min_parallel]
        if not cands:
            logger.info("auto-parallel: %s has no serving at ctx>=%d and "
                        "slots>=%d; no entry", view.stem, floor,
                        min_parallel)
            return None
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
        _, p_star, ctx_star = best
        if p_star != group_parallel:
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
        demands = tuple(task for task in resolved
                        if task.endpoint == CHAT)
        # A declared resolution budget means vision must actually be served at
        # that size, so it stops being shed-able for this model whatever the
        # category's sacrifice order says. ``keep`` rather than ``require``:
        # a text-only model is unaffected, and the image floor already fails
        # the rung when the budget cannot accommodate the image.
        mp = model.min_image_megapixels
        if mp is not None and mp > 0 and model.mmproj is not None:
            demands = tuple(replace(d, keep=d.keep | {"vision"}) for d in demands)
        return demands


    def _role_budgets(self) -> dict[tuple[str, str], int]:
        """Price the authored sets, dropping any set that cannot fit."""
        raw = {str(n): str(e) for n, e in
               ((self.matrix_cfg or {}).get("sets") or {}).items()}
        if not raw:
            return {}
        inlined = _inline_set_refs(raw)
        role_floor: dict[str, dict[str, int]] = {"default": {}, "ram": {}}
        role_source: dict[tuple[str, str], tuple[int, str, str]] = {}
        role_pools: dict[str, set[str]] = {}
        model_floor: dict[str, tuple[str, int]] = {}
        leaf_role: dict[str, str] = {}

        def model_keys(m):
            keys = {m.stem, m.template_id,
                    str(m.frontmatter.get("model_id", "")),
                    utils.slugify(str(m.frontmatter.get("name", "")))}
            keys.discard("")
            return keys

        for m in self.models:
            pool = "ram" if m.on_cpu else "default"
            role_pools.setdefault(m.role, set()).add(pool)
            view = m.view_for(True)
            quad = view.vram.effective_static(
                self.fit_bin, cache_type=self.profiles.default_cache_type,
                include_mmproj=True,
                llama_args=measurement_args(self.profiles, view,
                                            self.llama_args),
                allow_cpu=True)
            if quad is None:
                est = max(HOST_PROCESS_MIB,
                          int(m.size_mb or 0) + HOST_PROCESS_MIB)
                previous = role_floor[pool].get(m.role, 0)
                if est > previous:
                    role_source[(m.role, pool)] = (est, m.stem, "unmeasured")
                role_floor[pool][m.role] = max(previous, est)
                if not m.on_cpu:
                    role_floor["ram"][m.role] = max(
                        role_floor["ram"].get(m.role, 0), HOST_PROCESS_MIB)
                for key in model_keys(m):
                    model_floor[key] = (pool, est)
                    model_floor[key.lower()] = (pool, est)
                    if not m.on_cpu:
                        model_floor[key + "#ram"] = ("ram", HOST_PROCESS_MIB)
                        model_floor[key.lower() + "#ram"] = (
                            "ram", HOST_PROCESS_MIB)
                    leaf_role[key] = m.role
                    leaf_role[key.lower()] = m.role
                continue
            demands = m.tasks_resolved or ()
            best = 0
            best_task = None
            for task in demands:
                need = allocation.floor_mb(
                    quad, task.min_context or self.min_context,
                    task.min_parallel or 1, self.memory_margin)
                if need > best:
                    best, best_task = need, task
            if best == 0:
                best = allocation.floor_mb(
                    quad, _MIN_CTX_SIZE, 1, self.memory_margin)
            previous = role_floor[pool].get(m.role, 0)
            if best > previous:
                role_source[(m.role, pool)] = (
                    best, m.stem, best_task.id if best_task else "")
            role_floor[pool][m.role] = max(previous, best)
            if not m.on_cpu:
                files = m.size_mb or 0
                if m.mmproj is not None and m.mmproj.gguf_path is not None:
                    files += m.mmproj.size_mb
                if m.mtp is not None and m.mtp.gguf_path is not None:
                    files += m.mtp.size_mb
                host = max(HOST_PROCESS_MIB, int(files - quad[0]))
                role_floor["ram"][m.role] = max(
                    role_floor["ram"].get(m.role, 0), host)
                for key in model_keys(m):
                    model_floor[key + "#ram"] = ("ram", host)
                    model_floor[key.lower() + "#ram"] = ("ram", host)
            for key in model_keys(m):
                model_floor[key] = (pool, best)
                model_floor[key.lower()] = (pool, best)
                leaf_role[key] = m.role
                leaf_role[key.lower()] = m.role

        def role_of_leaf(leaf: str) -> str | None:
            low = leaf.lower()
            if low in roles.names():
                return low
            if low in leaf_role:
                return leaf_role[low]
            for key, role in leaf_role.items():
                if low.startswith(key + "."):
                    return role
            return None

        def floor_of_leaf(pool: str, leaf: str) -> int | None:
            leaf = leaf.lower()
            if leaf in roles.names():
                return role_floor[pool].get(leaf, 0)
            found = model_floor.get(leaf)
            if found is not None and pool == "ram":
                host = model_floor.get(leaf + "#ram")
                if host is not None and found[0] == "default":
                    return host[1]
            if found is not None and found[0] == pool:
                return found[1]
            for key, (key_pool, mb) in model_floor.items():
                if key_pool == pool and leaf.startswith(key + "."):
                    return mb
            return None

        approved: dict[str, str] = {}
        rejected: dict[str, str] = {}
        for name, text in inlined.items():
            try:
                expr = allocation.parse(text)
            except ValueError as exc:
                raise ValueError(f"set {name!r}: {exc}") from None
            leaves = allocation.leaves(expr)
            seen: dict[str, str] = {}
            for leaf in leaves:
                role = role_of_leaf(leaf)
                if role is None:
                    raise ValueError(
                        f"set {name!r}: {leaf!r} is not a role, model stem, "
                        "or entry id")
                if role in seen:
                    raise ValueError(
                        f"set {name!r}: role {role!r} appears twice")
                seen[role] = leaf
            bad = False
            for pool in ("default", "ram"):
                table = {}
                for leaf in leaves:
                    mb = floor_of_leaf(pool, leaf)
                    if mb is None:
                        raise ValueError(
                            f"set {name!r}: {leaf!r} has no {pool} floor")
                    table[leaf] = mb
                need = allocation.floor_of(expr, table)
                cap = self.capacity_pools.get(pool, 0)
                if cap and need > cap:
                    logger.error(
                        "matrix: set %r floor %d MiB exceeds %s %d; dropped",
                        name, need, pool, cap)
                    bad = True
            if bad:
                rejected[name] = inlined[name]
                continue
            approved[name] = inlined[name]
        if not approved:
            raise ValueError("matrix.sets: no set fits the available pools")
        self.matrix_sets_approved = dict(approved)
        self.matrix_sets_rejected = dict(rejected)

        by_pool: dict[str, dict[str, allocation.Expr]] = {}
        floors: dict[str, dict[str, int]] = {"default": {}, "ram": {}}
        leaves_by_pool: dict[str, dict[str, str]] = {"default": {}, "ram": {}}
        for name, text in approved.items():
            expr = allocation.parse(text)
            for pool in ("default", "ram"):
                table = {leaf: floor_of_leaf(pool, leaf)
                         for leaf in allocation.leaves(expr)}
                by_pool.setdefault(pool, {})[name] = expr
                floors[pool].update(table)
                for leaf in allocation.leaves(expr):
                    leaves_by_pool[pool][leaf] = leaf_role.get(
                        leaf, leaf if leaf in roles.names() else "")

        for (role, pool), (mb, stem, task) in sorted(role_source.items()):
            share = roles.get(role).share if roles.get(role) else None
            if share is None:
                continue
            low = share.kind in ("min", "zero") or (
                share.kind == "weight" and share.weight < 25)
            cap = self.capacity_pools.get(pool, 0)
            if low and cap and mb > cap / 100:
                logger.warning(
                    "matrix: role %r floor %d MiB on %s (share %s) is set by "
                    "%s/%s; that role cannot grow, so split it if the floor "
                    "is not intended", role, mb, pool, share.kind, stem,
                    task or "fallback")
        logger.info("matrix: role floors %s",
                    {p: dict(sorted(v.items())) for p, v in role_floor.items()
                     if v})
        budget: dict[tuple[str, str], int] = {}
        for pool, setmap in by_pool.items():
            available = self.capacity_pools.get(pool)
            if not available:
                continue
            weights, fills, accepts = {}, set(), set()
            for leaf, role in leaves_by_pool[pool].items():
                share = roles.get(role).share if role else None
                if share is None or pool not in role_pools.get(role, set()):
                    weights[leaf] = 0.0
                    continue
                weights[leaf] = share.weight if share.requests else 0.0
                if share.kind == "fill":
                    fills.add(leaf)
                if share.accepts:
                    accepts.add(leaf)
            try:
                alloc, left = allocation.auction(
                    available, setmap, floors[pool], weights,
                    frozenset(fills), frozenset(accepts))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"set auction failed: {exc}") from None
            logger.info("matrix: %s auction %s -> %s (floor %d)",
                        pool, sorted(setmap), alloc, available)
            for leaf, mb in alloc.items():
                role = role_of_leaf(leaf)
                if role is None:
                    continue
                budget[(role, pool)] = min(
                    budget.get((role, pool), mb), mb)
            if any(mb > 0 for mb in left.values()):
                logger.info("matrix: auction leaves %s", left)
        if budget:
            logger.info("matrix: auction budgets %s",
                        ", ".join(f"{r}/{p}={mb}"
                                  for (r, p), mb in sorted(budget.items())))
        return budget

    def _host_charge_mb(self, model, ctx: int) -> int:
        """Host-RAM (MiB) of a CPU-served model at its served ctx.

        The same estimator the VRAM charge uses (requirement 11), single-
        slot (a fixed-overhead role never takes declared parallel), 0 when
        nothing can size it.
        """
        if getattr(self.host_ledger, "pool_mb", None) is None \
                or self.host_ledger.pool_mb <= 0:
            return 0
        try:
            mb = fixed_cost_mb(
                model, fit_bin=self.fit_bin,
                cache_type=self.profiles.default_cache_type,
                profiles=self.profiles, llama_args=self.llama_args,
                ctx=ctx, allow_cpu=True)
            return 0 if mb is None else mb
        except Exception as e:  # keep planning alive
            logger.warning("host ledger: cannot size %s (%s)",
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
        backends.  CPU-served servings return ``(None, ram_mib, …)``
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
        return (float(total),
                max(float(HOST_PROCESS_MIB), float(files_mb - model_mib)),
                float(kv_factor))

    def plan(self) -> dict[str, list[Variant]]:
        """Plan serving variants for every model, keyed by stem.

        Order matters: the authored sets are priced first (their role
        budgets feed the per-model solve), then variants are grouped by
        profile. Tools are demoted on chat entries solved below
        ``tools_min_ctx``.
        """
        if not ((self.matrix_cfg or {}).get("sets")):
            logger.warning(
                "matrix: no sets; capacity sizes each model alone. Author "
                "matrix.sets for co-residency.")
        self.role_budget = self._role_budgets()
        # Shared ledger: pools: overrides.  Per-model chat solves charge
        # this pool's extra reserve to their spare.
        self.ledger = PoolLedger(self.vram_total, self.profiles.pools_cfg)

        plan: dict = {}
        n_models = len(self.models)
        for i, model in enumerate(self.models, 1):
            logger.info("planning %d/%d: %s", i, n_models, model.stem)
            model.image_budget_warnings()
            include_mmproj = True   # per-category now; see solve_rung
            view = model.view_for(include_mmproj)
            context_length = view.design_context
            # Unestimable model: no measurement source worked. Serve at the
            # minimum useful context (type-based floor) and mark every entry
            # for client-facing metadata.  The flag may sit on either
            # budget: view_for() rebuilds VramBudget per view (mmproj models
            # carry a separate companion-on budget that runs its own solves).
            est_error = getattr(model.vram, "unestimated_reason", None) \
                or getattr(view.vram, "unestimated_reason", None)
            groups = self.profiles.groups_for(view, self.vram_total, self.spare)
            variants: list[Variant] = []
            for (parallel, cache_type, spare_mb, batch, ubatch), group in groups.items():
                # Reset the per-group serving view: a prior group's
                # vision→text fallback must not leak into this group.
                include_mmproj = True   # per-category now; see solve_rung
                view = model.view_for(include_mmproj)
                context_length = view.design_context
                # A role with an auction budget solves inside it; otherwise
                # the pool's declared spare is the reserve.
                pool_id = self.ledger.pool_id_for(view)
                budget_pool = "ram" if view.on_cpu else "default"
                budget = self.role_budget.get((model.role, budget_pool))
                if budget is not None:
                    reserve = _RESERVE_SYSTEM + max(_RESERVE_VIDEO,
                                                    self.baseline_mb)
                    spare_eff = max(0, self.vram_total - reserve - budget
                                    + spare_mb)
                else:
                    spare_mb = self.ledger.spare_for(pool_id, spare_mb)
                    spare_eff = spare_mb
                # Auto-parallel: unpinned chat models on slot-based backends
                # get a value-function (parallel, ctx) solve instead of the
                # group parallel.  Pinned = sidecar/block `parallel:` or any
                # profile in the group declaring `parallel`.
                auto = (
                    self.knobs.auto_parallel
                    and view.serves_endpoint(CHAT)
                    and view.backend in ("llama-server", "vllm", "vllm-docker")
                    # A CPU-served model's KV pool lives in host RAM, which
                    # no VRAM ledger sees or budgets, and solve_context returns
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
                fallback = fallback_task(model)
                if est_error is not None:
                    ctx_size = self._unestimated_ctx(view)
                    serves = [(fallback, view, parallel, ctx_size)]
                else:
                    serves = []
                    pin_ctx = self._serving_pin(view)
                    cap_ctx = context_length
                    if self.max_context is not None:
                        cap_ctx = min(cap_ctx, self.max_context)
                    task_cap = self.tasks.max_context_for(model.task_ids)
                    for demand in (*self.demands_for(model), fallback):
                        is_null = demand is fallback
                        demand_cap = cap_ctx
                        if is_null:
                            if task_cap is not None:
                                demand_cap = min(demand_cap, task_cap)
                        elif demand.max_context is not None:
                            demand_cap = min(demand_cap, demand.max_context)
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
                            category_floor=(demand.min_context
                                            or self.min_context))
                        rung = self.solve_rung(
                            model, demand, parallel=parallel,
                            cache_type=cache_type, spare_mb=spare_eff,
                            context_length=demand_cap, profile=group[0][1],
                            prefer_capability=is_null)
                        if rung is None:
                            # Cannot be served at any rung: no id may claim it.
                            # The fallback is the floor on that.
                            continue
                        shed_vision = "vision" in rung.shed
                        serve_view = model.view_for(not shed_vision)
                        if auto:
                            solved = self._auto_parallel(
                                serve_view, cache_type=cache_type,
                                spare_mb=spare_eff,
                                include_mmproj=not shed_vision,
                                cap_ctx=demand_cap, floor=floor,
                                pin_ctx=pin_ctx,
                                power=0.0 if is_null else demand.parallel_power,
                                min_parallel=demand.min_parallel,
                                group_parallel=parallel,
                                profile=group[0][1])
                            if solved is None:
                                continue
                            par, ctx = solved
                        else:
                            par = parallel
                            ctx = self._bounded_ctx(
                                serve_view, parallel=par,
                                cache_type=cache_type, spare_mb=spare_eff,
                                include_mmproj=not shed_vision,
                                context_length=demand_cap,
                                profile=group[0][1])
                            if ctx < demand.min_context:
                                continue
                            if par < demand.min_parallel and not uncapped:
                                logger.info(
                                    "%s: pinned parallel=%d is below the %s "
                                    "min_parallel=%d; no entry", model.stem,
                                    par, demand.id or "fallback",
                                    demand.min_parallel)
                                continue
                        serves.append((demand, serve_view, par, ctx))
                    if not serves:
                        continue
                ctx_size = serves[0][3]
                if est_error is None:
                    # Discovery during this model's own solve: solve_context
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
                        serves = [(fallback, view, group_parallel, ctx_size)]

                # Squeeze note: the budget could not afford the design
                # context and an explicit reservation (--spare/--baseline)
                # is what pushed it over — stamp the entry description.
                # Gate on the user-declared spare (pool-adjusted, before the
                # pool's extra reserve) or a baseline above the fixed floor;
                # and only when the served ctx is actually below design.
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
                    task_demands = () if demand is fallback else (demand,)
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
                    host_charge = mem[1] if host_free is not None else None
                    if (host_charge is not None and host_charge > host_free):
                        # Go/no-go on the host pool: the CPU-served quad does
                        # not fit. No id may claim it and there is no
                        # solve-down (requirement 12) — the operator picks a
                        # smaller sidecar or a bigger box.
                        logger.info(
                            "%s: %s over_budget_reason: host pool — %d MiB "
                            "needed, %d MiB in the pool; entry omitted",
                            serve_view.stem, demand.id or "fallback",
                            int(mem[1]), host_free)
                        continue
                    variants.append(Variant(
                        parallel=(0 if uncapped else serve_par),
                        cache_type=cache_type, spare_mb=spare_mb,
                        profiles_group=group, ctx_size=serve_ctx,
                        include_mmproj=not shed_vision,
                        batch=batch, ubatch=ubatch,
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
        return plan


def _static_params(model: Model, fit_bin: str, cache_type: str,
                   llama_args: str = "",
                   allow_cpu: bool = False,
                   ) -> tuple[int, float, float, int] | None:
    """(model_mib, kv_per_token_mib, slot_mib, compute_mib) quad, or None
    when unmeasurable.  ``allow_cpu`` lifts the CPU gate — the host ledger
    sizes `device: cpu` quads through the same gate."""
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


def fixed_quad(model: Model, *, fit_bin: str, cache_type: str,
               profiles, llama_args: str, allow_cpu: bool = False,
               ) -> tuple[int, float, float, int] | None:
    """The measured quad behind :func:`fixed_cost_mb`, or None.

    Exposed separately so a cost comparison can name the terms — two
    servings with the same total cost are not a bug if their weights, KV
    and compute terms match.
    """
    return _static_params(model, fit_bin, cache_type,
                          measurement_args(profiles, model, llama_args),
                          allow_cpu=allow_cpu)


def fixed_cost_mb(model: Model, *, fit_bin: str, cache_type: str,
                  profiles, llama_args: str, ctx: int,
                  parallel: int = 1, allow_cpu: bool = False) -> int | None:
    """Measured serving cost (MiB) of one fixed-overhead serving at its ctx.

    The one estimator behind the VRAM and host charges: the same affine
    quad family (shape-stamped, so a `device: cpu` model measures under
    `-ngl 0`), priced at the same context.  None when nothing can size it
    — callers treat that as "not a candidate for a cost comparison", never
    as 0.
    """
    quad = fixed_quad(model, fit_bin=fit_bin, cache_type=cache_type,
                      profiles=profiles, llama_args=llama_args,
                      allow_cpu=allow_cpu)
    if quad is None:
        return None
    return _quad_cost_mb(quad, ctx, parallel)


def fixed_cost_detail(model: Model, *, fit_bin: str, cache_type: str,
                      profiles, llama_args: str, ctx: int,
                      allow_cpu: bool = False) -> tuple[int, str] | None:
    """(total MiB, term breakdown) for one fixed-overhead serving, or None.

    The breakdown exists so two servings that both cost 1050 MiB can be
    seen to be the same *shape* — same weights, same compute buffer, same
    KV/token — rather than a repeated number masking a bug.
    """
    quad = fixed_quad(model, fit_bin=fit_bin, cache_type=cache_type,
                      profiles=profiles, llama_args=llama_args,
                      allow_cpu=allow_cpu)
    if quad is None:
        return None
    mib, kv_factor, slot_mib, compute = quad
    total = _quad_cost_mb(quad, ctx)
    kv_mb = int(kv_factor * ctx + slot_mib)
    return total, (f"{mib} weights + {compute} compute + {kv_mb} kv "
                   f"@{ctx}")


def served_ctx(model: Model, tasks: TaskCatalogue | None = None,
               ctx_override: int | None = None) -> int:
    """The context a fixed-overhead serving is served at — the use-case size.

    The cap is the tightest ``max_context`` among the model's tasks (for
    embeddings/rerank that is the RAG window); a model whose tasks declare
    none serves its design context.  ``ctx_override`` caps it further.  It
    is always a cap, never a floor.
    """
    cap = tasks.max_context_for(model.task_ids) if tasks else None
    ctx = (model.design_context if cap is None
           else min(model.design_context, cap))
    if ctx_override is not None:
        ctx = min(ctx, ctx_override)
    return ctx


# ── Emission ──────────────────────────────────────────────────────────────


def _capacity_fields(model: Model, variant: "Variant", pools: dict[str, int],
                     drive_mbps: int, pp_tps: float,
                     memory_margin: float) -> dict:
    """Capacity-router fields for one entry: pools it charges, eviction cost.

    ``capacities`` is what llama-swap enforces; it must carry the same
    measurement margin the solve used or the router can admit an overcommit.
    A size-1 pool per scheduling role makes "one served entry per role" a
    capacity constraint, so the requested model evicts its siblings without
    any set vocabulary.
    """
    caps: dict[str, int] = {}
    estimate = max(1, int(model.size_mb or 0) + HOST_PROCESS_MIB)
    if model.on_cpu:
        if variant.mem_ram_mib is not None and "ram" in pools:
            caps["ram"] = max(1, int(round(
                variant.mem_ram_mib * (1.0 + memory_margin))))
        elif "ram" in pools:
            caps["ram"] = min(pools["ram"], estimate)
    else:
        vram = variant.mem_vram_mib
        if vram is None:
            # Unmeasured: the file size plus the process floor. Flagged and
            # admitted on a guess; the operator supplies derived: terms.
            if "default" in pools:
                caps["default"] = min(pools["default"], estimate)
        elif vram and "default" in pools:
            caps["default"] = max(1, int(round(
                vram * (1.0 + memory_margin))))
        if variant.mem_ram_mib and variant.mem_ram_mib > 0 \
                and "ram" in pools:
            caps["ram"] = max(1, int(round(
                variant.mem_ram_mib * (1.0 + memory_margin))))
        elif "ram" in pools:
            caps["ram"] = min(pools["ram"], HOST_PROCESS_MIB)
    role = model.role
    if role in pools:
        caps[role] = 1

    cost = 0.0
    size_mb = float(model.size_mb or 0)
    if size_mb > 0 and drive_mbps > 0:
        cost += size_mb / drive_mbps
    if model.serves_endpoint(CHAT):
        pp = None
        if "pp_tps" in getattr(model, "sidecar_keys", ()):
            try:
                pp = float(model.frontmatter.get("pp_tps"))
            except (TypeError, ValueError):
                pp = None
        if pp is None:
            pp = pp_tps
        if pp and pp > 0:
            cost += variant.ctx_size * max(1, variant.parallel) / pp
    fields = {"evict_cost": max(0, int(round(cost)))}
    if caps:
        fields["capacities"] = caps
    return fields


def emit_config(models: list[Model], plan: dict,
                profiles: Profiles, template_vars: dict, *,
                capacity_pools: dict[str, int] | None = None,
                matrix_sets: dict[str, str] | None = None,
                drive_mbps: int = 100,
                pp_tps: float = 1000.0,
                memory_margin: float = _MEMORY_MARGIN,
                task_ttl: dict[str, int] | None = None) -> EmittedConfig:
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
        tail = "" if not cats else ".".join(
            x for x in (cat_id, flags) if x)
        disp = f" · {cat}" if cat is not None else ""
        disp += f" {ctx_label(ctx)}"          # display only, never the id
        if flags:
            disp += f" [{flags}]"
        return tail, disp

    for model in models:
        for v in plan.get(model.stem, []):
            view = model.view_for(v.include_mmproj)
            context_length = view.design_context
            flags = served_flags(view, v.include_mmproj, v.tools_demoted)
            tail, disp = id_and_suffix(v, v.ctx_size, flags)
            tail = unique_tail(model.stem, tail,
                                f"{v.cache_type}"
                                f"{'-vis' if v.include_mmproj else ''}")
            demand = v.task_demands[0] if v.task_demands else None
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
                ttl=resolve_entry_ttl(
                    view, demand,
                    _entry_task_ttl(view, demand, task_ttl),
                    roles.get(view.role).ttl if roles.get(view.role) else None),
            )
            entry.update(_capacity_fields(
                view, v, capacity_pools or {}, drive_mbps, pp_tps,
                memory_margin))
            if entry_id in entries:
                raise ValueError(
                    f"duplicate entry id {entry_id!r}: model {model.stem!r} "
                    f"collides with {owner[entry_id]!r} — rename one of them")
            entries[entry_id] = entry
            owner[entry_id] = model.stem
            ids_by_stem.setdefault(model.stem, []).append(entry_id)

    config = EmittedConfig(entry_ids_by_stem=ids_by_stem,
                           capacity_pools=dict(capacity_pools or {}),
                           matrix_sets=matrix_sets)
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
    actually produced instead of re-deriving id naming conventions.
    """

    def __init__(self, *args, entry_ids_by_stem: dict[str, list[str]] | None = None,
                 capacity_pools: dict[str, int] | None = None,
                 matrix_sets: dict[str, str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_ids_by_stem: dict[str, list[str]] = entry_ids_by_stem or {}
        self.matrix_sets: dict[str, str] = matrix_sets or {}
        # Capacity-router pool limits (gpu default, host ram, one size-1
        # pool per scheduling role). Emitted under routing.router.settings.
        self.capacity_pools: dict[str, int] = capacity_pools or {}

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
    baseline_mb: int = 0,
    min_context: int = _MIN_AGENTIC_CTX,
    min_context_explicit: bool = False,
    memory_margin: float = _MEMORY_MARGIN,
    host_pool_mb: int | None = None,
    drive_mbps: int = 100,
    pp_tps: float = 1000.0,
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
        matrix_cfg: Matrix configuration: the authored sets price the roles
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
        # Keep the raw value: Planner resolves ``defaults.spare``/``--spare``
        # itself. Passing the resolved MiB would be re-parsed as GB.
        spare=spare,
        max_context=max_context,
        matrix_cfg=matrix_cfg,
        baseline_mb=baseline_mb,
        min_context=min_context, min_context_explicit=min_context_explicit,
        memory_margin=memory_margin,
        llama_args=template_vars.get("llama_args", ""),
        host_pool_mb=host_pool_mb,
        progress_cb=progress_cb,
    )
    plan = planner.plan()
    matrix_sets = planner.matrix_sets_approved
    config = emit_config(supported, plan, profiles, template_vars,
                         capacity_pools=(None if matrix_sets
                                         else planner.capacity_pools),
                         matrix_sets=matrix_sets,
                         drive_mbps=drive_mbps, pp_tps=pp_tps,
                         memory_margin=memory_margin,
                         task_ttl=_task_ttl(profiles))
    _verify_no_overallocation(config, plan, supported, planner)
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


def dump_yaml(payload: dict) -> str:
    """Serialize *payload* to a YAML string, preserving embedded newlines."""
    return yaml.dump(payload, default_flow_style=False, sort_keys=False,
                     allow_unicode=True, Dumper=_LiteralDumper)


def write_yaml(config: dict, path: Path | str) -> None:
    """Write config to YAML file."""
    payload = config.plain() if isinstance(config, EmittedConfig) else config
    with open(path, "w") as f:
        f.write(dump_yaml(payload))
