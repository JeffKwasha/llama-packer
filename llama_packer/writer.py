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
            # Already logged (and the model flagged) by apply_overrides.
            continue

        backend = get_backend(model.backend)
        reason = backend.unsupported_reason(model)
        if reason:
            logger.error("skipping %s: %s backend cannot serve it: %s",
                         model.stem, backend.name, reason)
            continue

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
) -> tuple[str, dict]:
    """Build a single llama-swap config entry for a model+profile group.

    ``include_mmproj=False`` omits the vision projection from the command,
    removes the ``vision`` capability, and flags ``metadata.mmproj_skipped``.
    ``name_suffix`` is appended to the model display name (e.g. the vision
    variant's `` [vision 92k]``).
    """
    base_id = utils.slugify(model.name)

    # Build the launch command via the model's backend.  The backend is chosen
    # by override rules (model.backend), and validation/skip of unsupported
    # format/role combos happens in build_config's pre-pass.
    backend = get_backend(model.backend)
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

    caps = list(model.capabilities)
    modalities = ["text"]
    if "vision" in caps:
        modalities.append("image")
    if "audio" in caps:
        modalities.append("audio")

    # When mmproj is dropped, remove the (auto-added) vision capability so the
    # main entry no longer advertises image input.
    if not include_mmproj and model.mmproj and model.mmproj.gguf_path:
        caps = [c for c in caps if c.lower() != "vision"]
        if "image" in modalities:
            modalities.remove("image")
        metadata["mmproj_skipped"] = True

    tf = model.throughput_factor()
    if tf is not None:
        metadata["throughput_factor"] = tf

    metadata["mtp_enabled"] = backend_meta.get("mtp_enabled", False)
    if backend_meta.get("mtp_enabled"):
        metadata["mtp_draft_max"] = backend_meta["mtp_draft_max"]

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
        "in": modalities,
        "out": modalities,
        "tools": "tools" in caps,
        "reranker": "reranker" in caps or model.role == "rerank",
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

    return entry_id, entry


# ── Planning ──────────────────────────────────────────────────────────────


# Entry-id suffix of every no-mmproj variant.  Invariant: the bare ``<id>``
# always serves vision when the model has an mmproj; the text-only serving is
# always ``<id>-text`` (see :func:`emit_config`).
TEXT_SUFFIX = "-text"


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
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.baseline_mb = baseline_mb
        self.min_context = min_context
        self.chat_ctx: int | None = None  # matrix-solved shared context, if any

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
        alongside).  Uses the global spare and fleet defaults; per-profile
        spare still bounds ctx per group.
        """
        drop: dict[str, bool] = {}
        global_spare_mb = self.profiles.global_spare_mb(self.spare, self.vram_total)
        for model in self.models:
            if model.role in ("embeddings", "rerank"):
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
            drop[model.stem] = True
            logger.info("mmproj: drop for %s (vision ctx %d < %d; text ctx %d)",
                        model.stem, ctx_with, self.min_context, ctx_without)
            if ctx_without < self.min_context:
                logger.warning("mmproj: %s cannot reach %d context even without "
                               "vision (text ctx %d)",
                               model.stem, self.min_context, ctx_without)
        return drop

    def _solve_matrix(self, drop_stems: set[str]) -> int | None:
        """Shared chat context when a matrix section is configured."""
        if not (self.matrix_cfg and self.embed_model and self.rerank_model):
            return None
        chat_ctx = _solve_matrix_context(
            self.models, self.embed_model, self.rerank_model,
            self.fit_bin, self.vram_total, self.spare, self.profiles,
            baseline_mb=self.baseline_mb, drop_stems=drop_stems,
        )
        if chat_ctx is not None:
            logger.info("matrix: solved chat_ctx=%d", chat_ctx)
        return chat_ctx

    def plan(self) -> dict[str, list[Variant]]:
        """Plan serving variants for every model, keyed by stem.

        Order matters: the mmproj drop decision runs first (its result feeds
        the matrix solve), then per-model variants are grouped by profile.
        """
        drop_mmproj = self._mmproj_drop_pass()
        self.chat_ctx = self._solve_matrix({s for s, d in drop_mmproj.items() if d})

        plan: dict[str, list[Variant]] = {}
        for model in self.models:
            context_length = model.design_context
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
                    include_mmproj=include_mmproj, vision_ctx=vision_ctx))

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
                        include_mmproj=False))
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
) -> int | None:
    """Solve VRAM budget equation for chat context given embed/rerank allocations.

    Runs fit-params once per model to get static parameters, then solves:
        available = Σ(chat_weight + chat_factor*chat_ctx)
                   + (embed_weight + embed_factor*embed_ctx)
                   + (rerank_weight + rerank_factor*rerank_ctx)

    Returns the maximum chat context that fits, or None on failure.

    ``drop_stems`` is the set of chat-model stems whose mmproj is being skipped
    to reach the minimum useful context (their combined budget omits mmproj).
    """
    spare_mb = parse_spare_mb(spare, vram_total)

    drop_stems = drop_stems or set()

    # Get static params for chat models (companion VRAM folded in).
    # The drop decision (mmproj skipped to reach the min useful context) is
    # decided in Planner._mmproj_drop_pass and threaded in via drop_stems.
    chat_params = []
    for m in chat_models:
        # Embed/rerank models are handled via the separate overhead terms below
        # (and CPU-resident models cost no VRAM), so they must not enter the
        # shared chat budget — otherwise a small resident model can dictate the
        # chat context for the whole fleet.
        if m.role in ("embeddings", "rerank") or m.on_cpu:
            continue
        cache_type = m.cache_type_for(profiles.default_cache_type)
        parallel = m.parallel_for(profiles.default_parallel)
        fp = m.vram.effective_static(fit_bin, cache_type=cache_type, parallel=parallel,
                                     include_mmproj=m.stem not in drop_stems)
        if fp is None:
            logger.warning("matrix: could not get fit params for %s", m.stem)
            continue
        chat_params.append((m, fp[0], fp[1], fp[2]))

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

    return solve_matrix_ctx(
        vram_total_mb=vram_total,
        spare_mb=spare_mb,
        chat_models=chat_params,
        embed_params=embed_params,
        rerank_params=rerank_params,
        embed_ctx=embed_ctx,
        rerank_ctx=rerank_ctx,
        baseline_mb=baseline_mb,
    )


# ── Emission ──────────────────────────────────────────────────────────────


def emit_config(models: list[Model], plan: dict[str, list[Variant]],
                profiles: Profiles, template_vars: dict) -> dict:
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
    for model in models:
        context_length = model.design_context
        for v in plan.get(model.stem, []):
            text_only = not v.include_mmproj
            entry_id, entry = _build_entry(
                model, v.parallel, v.cache_type, v.profiles_group,
                profiles.defaults, template_vars, context_length, v.ctx_size,
                include_mmproj=v.include_mmproj,
                name_suffix=" [text]" if text_only else "",
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
            )
            vision_id += f"-vision-{n_k}k"
            if vision_id in entries:
                raise ValueError(
                    f"duplicate entry id {vision_id!r}: model {model.stem!r} "
                    f"collides with {owner[vision_id]!r} — rename one of them")
            entries[vision_id] = vision_entry
            owner[vision_id] = model.stem

    config = EmittedConfig(entry_ids_by_stem=ids_by_stem)
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

    def __init__(self, *args, entry_ids_by_stem: dict[str, list[str]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_ids_by_stem: dict[str, list[str]] = entry_ids_by_stem or {}

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
) -> dict:
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
