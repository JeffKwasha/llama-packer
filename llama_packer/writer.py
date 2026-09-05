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
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from llama_packer.model import Model
from llama_packer.vram import solve_matrix_ctx
from llama_packer.hardware import detect_gpu_env_var
from llama_packer import utils
from llama_packer.profiles import Profiles, parse_spare_mb
from llama_packer.backends import (
    SETTING_KEYS,
    FRAMEWORK_CONSUMED,
    METADATA_ONLY,
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
            arch, _ = utils.gguf_header_probe(model.gguf_path)
            if arch and any(rx.search(arch) for rx in utils._DIFFUSION_ARCH_RES):
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

        # s2t/t2s/image must not be served as chat via llama-server
        if model.role in ("s2t", "t2s", "image") and backend.name == "llama-server":
            logger.error("skipping %s: role %r must not use backend %r (use %s)",
                         model.stem, model.role, backend.name,
                         {"s2t":"whisper-server","t2s":"kokoro-podman","image":"sd-server"}[model.role])
            continue

        # Capability / companion cross-check: mmproj is a file, not a
        # capability — what it enables must be declared explicitly.
        caps_l = [str(c).lower() for c in model.capabilities]
        has_mmproj = bool(model.mmproj and model.mmproj.gguf_path)
        if "vision" in caps_l:
            logger.error("skipping capability 'vision' on %s: removed, use 'image' "
                         "(llama-swap modalities are text/audio/image/video)",
                         model.stem)
        if has_mmproj and model.role not in ("s2t", "t2s", "image", "embeddings", "rerank") \
                and "image" not in caps_l and "video" not in caps_l:
            logger.warning("sidecar %s: mmproj companion present but neither 'image' nor 'video' "
                           "declared — projection costs VRAM but is not advertised; "
                           "declare capabilities: [image] (or [image, video])",
                           model.stem)

        fm = model.frontmatter

        # Reasoning flags must name a known mode and apply to a reasoning model.
        rf = fm.get("reasoning-format")
        if rf is not None and str(rf).lower() not in _REASONING_FORMATS:
            logger.error("skipping %s: unknown reasoning-format %r (allowed: %s); ignored",
                         model.stem, rf, ", ".join(sorted(_REASONING_FORMATS)))
            fm.pop("reasoning-format")
        if not _model_can_reason(model):
            for k in _REASONING_FLAG_KEYS:
                if k in fm:
                    logger.error("skipping %s: %s declared on a non-reasoning model "
                                 "(role=%r, capabilities=%s); ignored",
                                 model.stem, k, model.role, model.capabilities)
                    fm.pop(k)

        # cache_type must be a precision we can size memory for.
        cache_type = model.cache_type_for(default_cache_type)
        if cache_type not in utils._KV_CACHE_BYTES:
            logger.error("skipping %s: unknown cache_type %r (known: %s)",
                         model.stem, cache_type, ", ".join(sorted(utils._KV_CACHE_BYTES)))
            continue

        declared = {k for k in SETTING_KEYS if k in fm}
        backend.warn_unhandled(declared - FRAMEWORK_CONSUMED - METADATA_ONLY)
        supported.append(model)
    return supported


def _build_mode_params(model: Model) -> dict[str, dict]:
    """Build ``setParamsByID`` from a model's sidecar-declared ``modes``.

    Full-profile definition: every declared numeric param is emitted — no
    diff-vs-defaults suppression. The model's ``default_mode`` maps to the
    bare ``${MODEL_ID}`` key; every other mode to ``${MODEL_ID}:<mode>``.

    Returns an empty dict when the model declares no modes (caller then keeps
    the global-profile behavior).
    """
    modes = model.modes
    if not modes:
        return {}

    set_params: dict[str, dict] = {}
    default_mode = model.default_mode
    for name, params in modes.items():
        overrides: dict = {}
        for k, v in params.items():
            if k not in utils.SAMPLING_KEYS:
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
    """
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

    # Sidecar-declared modes fully replace the global profile sampling
    # overrides for this model.
    mode_params = _build_mode_params(model)
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
    # and only when a companion exists (baked-in video stays).
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
    if model.description:
        entry["description"] = model.description
    # The VRAM-served -c limit (vs. capabilities.context = max trained).
    metadata["ctx_size"] = ctx_size
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

    # Proxied backends (sd-server, whisper-server) are proxied HTTP services,
    # not llama-swap managed inference — expose the standard proxy fields so
    # llama-swap can health-check and route.  checkEndpoint "/" avoids the
    # /health pitfall (Discussion #866: sd-server returns 200 on / only).
    if backend.proxied:
        entry["proxy"] = "http://127.0.0.1:${PORT}"
        entry["checkEndpoint"] = "/"

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
    (tools advertisement threshold).  ``ctx_gain_min`` gates the squeeze
    adoption; ``estimate_headroom`` pads estimated co-load overheads.
    """
    min_chat_ctx: int = 65536
    tools_min_ctx: int = 131072
    coload_min_ctx: int = 20480
    ctx_gain_min: int = 4096
    estimate_headroom: float = 1.25

    @classmethod
    def from_cfg(cls, matrix_cfg: dict | None) -> "MatrixKnobs":
        """Parse knobs from the matrix section, warning and defaulting on
        invalid values (a bad knob must not silently break the solve)."""
        cfg = matrix_cfg or {}
        knobs: dict = {}
        for key in ("min_chat_ctx", "tools_min_ctx", "coload_min_ctx",
                    "ctx_gain_min"):
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
        return cls(**knobs)


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
    vision_ctx: int | None = None
    coload: bool = False
    tools_demoted: bool = False


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
        baseline_mb: int = 0,
        min_context: int = utils._MIN_USEFUL_CTX,
    ):
        self.models = models
        self.profiles = profiles
        self.fit_bin = fit_bin
        self.vram_total = vram_total
        self.spare = spare
        self.max_context = max_context
        self.matrix_cfg = matrix_cfg
        self.knobs = MatrixKnobs.from_cfg(matrix_cfg)
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.baseline_mb = baseline_mb
        self.min_context = min_context
        self.chat_ctx: int | None = None  # matrix-solved shared context, if any
        self.matrix_result: MatrixSolve | None = None

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
    ) -> int:
        """VRAM-solved context clamped to the model's max trained context
        (*context_length*) and the CLI ``--max-context`` cap."""
        ctx = model.vram.calc_ctx(
            self.vram_total,
            fit_bin=self.fit_bin,
            parallel=parallel,
            spare_mb=spare_mb,
            include_mmproj=include_mmproj,
            baseline_mb=self.baseline_mb,
            cache_type=cache_type,
            design_ctx=design_ctx,
        )
        if context_length is not None:
            ctx = min(ctx, context_length)
        if self.max_context is not None:
            ctx = min(ctx, self.max_context)
        return ctx

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
            cache_type = model.cache_type_for(self.profiles.default_cache_type)
            parallel = model.parallel_for(self.profiles.default_parallel)
            ctx_with = self._bounded_ctx(
                model, parallel=parallel, cache_type=cache_type,
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
        result = _solve_matrix_context(
            self.models, self.embed_model, self.rerank_model,
            self.fit_bin, self.vram_total, self.spare, self.profiles,
            baseline_mb=self.baseline_mb, drop_stems=drop_stems,
            knobs=self.knobs,
        )
        if result is not None:
            logger.info("matrix: solved chat_ctx=%d (squeeze=%s, coloads=%s)",
                        result.chat_ctx, result.squeeze,
                        [s for s, _ in result.coloads])
        return result

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

        plan: dict[str, list[Variant]] = {}
        for model in self.models:
            context_length = model.design_context
            # Squeeze: an adopted emb/rnk squeeze is realized by clamping the
            # RAG entry's served context (the emit is what frees the VRAM).
            if model.role == "embeddings" and self.matrix_result:
                context_length = min(context_length,
                                     self.matrix_result.embed_ctx)
            elif model.role == "rerank" and self.matrix_result:
                context_length = min(context_length,
                                     self.matrix_result.rerank_ctx)
            tools_demoted = (
                model.role == "chat"
                and self.chat_ctx is not None
                and self.chat_ctx < self.knobs.tools_min_ctx
                and "tools" in [c.lower() for c in model.capabilities])
            is_coload = model.stem in coload_stems

            include_mmproj = not drop_mmproj.get(model.stem, False)

            groups = self.profiles.groups_for(model, self.vram_total, self.spare)
            variants: list[Variant] = []
            for (parallel, cache_type, spare_mb), group in groups.items():
                ctx_size = self._bounded_ctx(
                    model, parallel=parallel, cache_type=cache_type,
                    spare_mb=spare_mb, include_mmproj=include_mmproj,
                    design_ctx=self.chat_ctx, context_length=context_length)

                vision_ctx: int | None = None
                if not include_mmproj and model.mmproj and model.mmproj.gguf_path:
                    vision_ctx = self._bounded_ctx(
                        model, parallel=parallel, cache_type=cache_type,
                        spare_mb=spare_mb, include_mmproj=True,
                        design_ctx=self.chat_ctx, context_length=context_length)

                variants.append(Variant(
                    parallel=parallel, cache_type=cache_type, spare_mb=spare_mb,
                    profiles_group=group, ctx_size=ctx_size,
                    include_mmproj=include_mmproj, vision_ctx=vision_ctx,
                    coload=is_coload, tools_demoted=tools_demoted))

                # On-demand text-only variant: when the main entry keeps its
                # mmproj, also plan a no-vision entry (``<id>-text``) so
                # clients can pick the lower-memory serving.  When the main
                # entry was auto-dropped it IS the ``-text`` entry, so no
                # separate variant is needed.
                if (include_mmproj and model.role == "chat"
                        and model.mmproj and model.mmproj.gguf_path):
                    text_ctx = self._bounded_ctx(
                        model, parallel=parallel, cache_type=cache_type,
                        spare_mb=spare_mb, include_mmproj=False,
                        design_ctx=self.chat_ctx, context_length=context_length)
                    variants.append(Variant(
                        parallel=parallel, cache_type=cache_type, spare_mb=spare_mb,
                        profiles_group=group, ctx_size=text_ctx,
                        include_mmproj=False, coload=is_coload,
                        tools_demoted=tools_demoted))
            plan[model.stem] = variants
        return plan


def _static_params(model: Model, fit_bin: str, cache_type: str,
                   parallel: int) -> tuple[int, float, int] | None:
    """(model_mib, ctx_factor, compute_mib) triple, or None when unmeasurable."""
    fp = model.vram.fit_params_static(fit_bin, cache_type=cache_type,
                                      parallel=parallel)
    return (fp.model_mib, fp.ctx_factor, fp.compute_mib) if fp else None


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
) -> MatrixSolve | None:
    """Solve the shared VRAM budget for chat context plus co-loads.

    Runs fit-params once per model to get static parameters, then solves:
        available = Σ(chat_weight + chat_factor*chat_ctx)
                   + (embed_weight + embed_factor*embed_ctx)
                   + (rerank_weight + rerank_factor*rerank_ctx)

    Three passes, in order:

    1. *Baseline*: embed/rerank at their design context → ``chat_ctx₀``.
    2. *Squeeze* (§2b): when ``chat_ctx₀`` is below ``tools_min_ctx``, re-solve
       with embed/rerank contexts clamped to ``coload_min_ctx``; adopt only
       when the gain reaches ``ctx_gain_min``.
    3. *Opportunistic co-loads* (§2): enabled ``s2t``/``image`` models not on
       the GPU pool's excluded list, smallest fixed overhead first, are
       included while the chat solve stays at or above the floor
       (``tools_min_ctx`` when a tools chat model can keep it, else
       ``min_chat_ctx``).  Estimated candidates carry ``estimate_headroom``.

    Returns the :class:`MatrixSolve` (chat context, adopted RAG contexts,
    included co-loads) or None on failure.
    """
    knobs = knobs or MatrixKnobs()
    spare_mb = parse_spare_mb(spare, vram_total)

    drop_stems = drop_stems or set()

    # Get static params for chat models (companion VRAM folded in).
    # The drop decision (mmproj skipped to reach the min useful context) is
    # decided in Planner._mmproj_drop_pass and threaded in via drop_stems.
    chat_params = []
    for m in chat_models:
        # Embed/rerank/image/s2t models are handled outside the shared chat
        # budget (fixed overhead / separate pool). Including a 40 GB
        # diffusion model would collapse the chat budget, so exclude it.
        if m.role in utils.NON_CHAT_ROLES or m.on_cpu:
            continue
        cache_type = m.cache_type_for(profiles.default_cache_type)
        parallel = m.parallel_for(profiles.default_parallel)
        fp = m.vram.effective_static(fit_bin, cache_type=cache_type, parallel=parallel,
                                     include_mmproj=m.stem not in drop_stems)
        if fp is None:
            logger.warning("matrix: could not get fit params for %s", m.stem)
            continue
        img_floor = 0
        if (m.stem not in drop_stems and m.mmproj and m.mmproj.gguf_path
                and m.image_max_tokens):
            img_floor = parallel * m.image_max_tokens
        chat_params.append((m, fp[0], fp[1], fp[2], img_floor))

    if not chat_params:
        return None

    # Get static params for embed/rerank. CPU-resident models cost 0 VRAM.
    embed_params = None
    if not embed_model.on_cpu:
        embed_params = _static_params(embed_model, fit_bin,
                                      profiles.default_cache_type,
                                      profiles.default_parallel)
    rerank_params = None
    if not rerank_model.on_cpu:
        rerank_params = _static_params(rerank_model, fit_bin,
                                       profiles.default_cache_type,
                                       profiles.default_parallel)

    # Reserve for embed/rerank at their own declared context (sidecar
    # context_length > GGUF architectural max), not an arbitrary constant —
    # a 32k reranker costs ~4x the KV of an 8k one and must be budgeted as
    # declared.
    embed_ctx = embed_model.design_context
    rerank_ctx = rerank_model.design_context

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
        overheads: list[tuple[int, str, Model]] = []
        for m in chat_models:
            if m.role not in ("s2t", "image"):
                continue
            oh = _coload_overhead(m, fit_bin, profiles, knobs)
            if oh is None:
                logger.warning("matrix: co-load %s skipped: cannot size it",
                               m.stem)
                continue
            overheads.append((oh, m.stem, m))
        used = 0
        for oh, stem, m in sorted(overheads, key=lambda t: t[0]):
            ctx = _solve(embed_ctx, rerank_ctx, fixed_overhead_mb=used + oh)
            if ctx >= floor:
                used += oh
                coloads.append((stem, oh))
                logger.info("matrix: co-load %s included (%d MB, chat_ctx=%d)",
                            stem, oh, ctx)
            else:
                logger.warning(
                    "matrix: co-load %s skipped: would drop chat ctx to %d "
                    "(floor %d)", stem, ctx, floor)
    return MatrixSolve(
        chat_ctx=chat_ctx, embed_ctx=embed_ctx, rerank_ctx=rerank_ctx,
        coloads=tuple(coloads), squeeze=squeeze,
    )


def _coload_overhead(
    m: Model, fit_bin: str, profiles: Profiles, knobs: MatrixKnobs,
) -> int | None:
    """Fixed VRAM overhead (MB) of an opportunistic co-load candidate.

    Uses the backend's effective static params (weights + fixed compute;
    ctx_factor is 0 for these backends).  Overheads that are *estimated*
    rather than measured or pinned carry ``estimate_headroom`` so a bad
    guess errs toward reserving more.
    """
    if m.on_cpu:
        return 0
    # An operator-pinned vram_mb is authoritative — no headroom.
    pinned = m.frontmatter.get("vram_mb") is not None
    cache_type = m.cache_type_for(profiles.default_cache_type)
    parallel = m.parallel_for(profiles.default_parallel)
    fp = m.vram.fit_params_static(fit_bin, cache_type=cache_type,
                                  parallel=parallel)
    measured = pinned or (fp is not None and fp.source == "fit-params")
    triple = m.vram.effective_static(fit_bin, cache_type=cache_type,
                                     parallel=parallel)
    if triple is None:
        return None
    model_mib, ctx_factor, compute_mib = triple
    # These backends have ctx_factor 0; a nonzero factor would mean a
    # context-driven model wrongly landed in the pool — charge design ctx.
    overhead = model_mib + compute_mib + int(ctx_factor * m.design_context)
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
        context_length = model.design_context
        for v in plan.get(model.stem, []):
            text_only = not v.include_mmproj
            if v.coload:
                coload_stems.add(model.stem)
            entry_id, entry = _build_entry(
                model, v.parallel, v.cache_type, v.profiles_group,
                profiles.defaults, template_vars, context_length, v.ctx_size,
                include_mmproj=v.include_mmproj,
                name_suffix=" [text]" if text_only else "",
                tools_demoted=v.tools_demoted,
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
            vision_id, vision_entry = _build_entry(
                model, v.parallel, v.cache_type, v.profiles_group,
                profiles.defaults, template_vars, context_length, v.vision_ctx,
                include_mmproj=True,
                name_suffix=f" [vision {n_k}k]",
                tools_demoted=v.tools_demoted,
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
    baseline_mb: int = 0,
    min_context: int = utils._MIN_USEFUL_CTX,
) -> EmittedConfig:
    """Build llama-swap config from list of Model objects.

    Composes the pipeline: validate/filter models, plan serving variants
    (:class:`Planner`), render entries (:func:`emit_config`).

    Args:
        models: List of Model instances
        profiles_cfg: Full profiles.yaml config (or a prepared Profiles object)
        template_vars: Template variables (llama_bin, models_dirs)
        fit_bin: Path to llama-fit-params binary
        vram_total: Total VRAM in MB
        spare: Global spare VRAM string (overridden by profile.spare if present)
        max_context: Hard cap on context length
        matrix_cfg: Matrix configuration for embed/rerank context solving
        embed_model: Embedding model (if matrix configured)
        rerank_model: Reranking model (if matrix configured)
        baseline_mb: Driver/compositor VRAM already in use (added to reserve)
        min_context: Minimum useful context for chat models. When a chat model
            with an mmproj companion cannot reach this WITH vision, the vision
            projection is dropped from the main entry, which is renamed
            ``<id>-text`` (a ``vision-<N>k`` variant is emitted alongside,
            still exposing vision at best-effort context).
    """
    profiles = profiles_cfg if isinstance(profiles_cfg, Profiles) else Profiles(profiles_cfg)

    # Validate BEFORE any VRAM work: rejected models must not consume
    # fit-params runs or pollute the shared matrix context solve.
    supported = _filter_supported(models, profiles.default_cache_type)

    planner = Planner(
        supported, profiles, fit_bin, vram_total,
        spare=spare, max_context=max_context,
        matrix_cfg=matrix_cfg, embed_model=embed_model,
        rerank_model=rerank_model, baseline_mb=baseline_mb,
        min_context=min_context,
    )
    return emit_config(supported, planner.plan(), profiles, template_vars)


def write_yaml(config: dict, path: Path | str) -> None:
    """Write config to YAML file."""
    payload = config.plain() if isinstance(config, EmittedConfig) else config
    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
