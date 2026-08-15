# llama_packer/writer.py
"""Generate llama-swap config entries from Model objects."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path

import yaml

from llama_packer.model import Model, _CL_RE
from llama_packer.vram import solve_matrix_ctx
from llama_packer.hardware import detect_gpu_env_var
from llama_packer import utils

logger = logging.getLogger(__name__)

# Profile key that controls spare VRAM (bytes)
_SPARE_KEY = "spare"


def _strip_repeat_ws(text: str) -> str:
    """Collapse runs of whitespace to single spaces (templates with | blocks)."""
    return " ".join(text.split())


def _filter_profiles(profile_list: dict, allow_profiles) -> list[tuple[str, dict]]:
    """Filter profiles according to allow_profiles frontmatter value."""
    if allow_profiles is False:
        return []
    if allow_profiles is None or allow_profiles is True:
        return list(profile_list.items())
    if isinstance(allow_profiles, str):
        try:
            pattern = re.compile(allow_profiles)
        except re.error:
            logger.warning("invalid allow_profiles regex %r, returning all profiles", allow_profiles)
            return list(profile_list.items())
        return [(pname, pover) for pname, pover in profile_list.items() if pattern.search(pname)]
    return list(profile_list.items())


def _resolve_reasoning_cli_args(cli_args: str, reasoning: str | None) -> list[tuple[str, str]]:
    """Resolve reasoning variants from frontmatter reasoning field.

    Returns list of (cli_args_variant, variant_suffix).
    """
    if not reasoning:
        return [(cli_args, "")]

    _RE_REASONING = re.compile(r"--reasoning\s+\S+")

    if reasoning is True:
        return [(cli_args, "")]

    if reasoning == "auto":
        variants = []
        for mode in ["none", "native", "openai"]:
            suffix = f".reasoning.{mode}"
            if mode == "none":
                args = _RE_REASONING.sub("", cli_args).strip()
            else:
                args = _RE_REASONING.sub(f"--reasoning {mode}", cli_args).strip()
                if "--reasoning" not in args:
                    args = (args + f" --reasoning {mode}").strip()
            variants.append((args, suffix))
        return variants

    args = _RE_REASONING.sub(f"--reasoning {reasoning}", cli_args).strip()
    if "--reasoning" not in args:
        args = (args + f" --reasoning {reasoning}").strip()
    return [(args, "")]


def _build_mtp_args(
    model: Model,
) -> tuple[str, bool, int, str | None]:
    """Build MTP CLI arguments for a model.

    Returns:
        (cli_arg_str, mtp_enabled, mtp_n_max, draft_path)
    """
    has_mtp = model.frontmatter.get("mtp")
    speculative = model.frontmatter.get("speculative")

    # If no MTP at all
    if not has_mtp and not speculative:
        return "", False, utils._MTP_DRAFT_N_MAX, None

    # If baked-in MTP (no companion file)
    if has_mtp and not speculative:
        n_max = model.frontmatter.get("mtp_draft_n_max", utils._MTP_DRAFT_N_MAX)
        spec_type = model.frontmatter.get("mtp_spec_type", utils._MTP_SPEC_TYPE)
        args = f"--spec-type {spec_type} --spec-draft-n-max {n_max}"
        return args, True, n_max, None

    # If companion MTP
    if speculative and "mtp" in Path(speculative).stem.lower():
        if not model.gguf_path:
            return "", False, utils._MTP_DRAFT_N_MAX, None
        # Search in both .gguf parent and .md parent
        companion = None
        for d in [model.gguf_path.parent, model.md_path.parent]:
            candidate = d / speculative
            if candidate.is_file():
                companion = candidate
                break
        if companion:
            n_max = model.frontmatter.get("mtp_draft_n_max", utils._MTP_DRAFT_N_MAX)
            spec_type = model.frontmatter.get("mtp_spec_type", utils._MTP_SPEC_TYPE)
            draft_path = str(companion)
            args = f"--spec-type {spec_type} --spec-draft-n-max {n_max} --spec-draft-model {draft_path}"
            return args, True, n_max, draft_path
        else:
            logger.warning("mtp: companion %s missing for %s", speculative, model.stem)
            return "", False, utils._MTP_DRAFT_N_MAX, None

    # MTP with non-MTP companion name (unusual)
    return "", False, utils._MTP_DRAFT_N_MAX, None


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

    # Build the launch command from the model's backend template (a full `cmd`
    # string).  The sidecar may declare an explicit `template:` (e.g. vllm);
    # otherwise the template follows the role.
    target = model.template
    t_conf = utils._TARGET_TEMPLATES.get(target, utils._TARGET_TEMPLATES["chat"])

    # MTP args (llama-server only; vLLM MTP is not translated yet).
    mtp_arg_str, mtp_enabled, mtp_n_max, _ = _build_mtp_args(model)
    if model.is_vllm:
        mtp_arg_str = ""
        mtp_enabled = False
        mtp_n_max = utils._MTP_DRAFT_N_MAX

    # mmproj (llama-server only, omitted when the vision projection is dropped).
    mmproj_args = ""
    if not model.is_vllm and include_mmproj and model.mmproj and model.mmproj.gguf_path:
        mmproj_args = f"--mmproj {model.mmproj.gguf_path}"

    # Model source: HF repo for vLLM backends, local file for llama-server.
    model_ref = (model.hf_repo or str(model.gguf_path)) if model.is_vllm else str(model.gguf_path)

    vllm_image = model.vllm_image or template_vars.get("vllm_image", utils.VLLM_DEFAULT_IMAGE)
    # GPU-resident models pin all layers to VRAM so the runtime matches the
    # fit-params measurement; CPU-resident models (embed/rerank) stay on CPU.
    n_gpu_layers = "--n-gpu-layers 0" if model.on_cpu else "--n-gpu-layers 999"

    cmd_str = utils.resolve_template(t_conf["cmd"], {
        "llama_bin": template_vars.get("llama_bin", ""),
        "vllm_bin": template_vars.get("vllm_bin", utils.VLLM_DEFAULT_BIN),
        "vllm_image": vllm_image,
        "model_path": model_ref,
        "ctx_size": str(ctx_size),
        "parallel": str(parallel),
        "cache_type": cache_type,
        "n_gpu_layers": n_gpu_layers,
        "mtp_args": mtp_arg_str,
        "mmproj_args": mmproj_args,
        "extra_args": model.cli_args.strip(),
        "gpu_mem_util": str(template_vars.get("gpu_mem_util", utils.VLLM_DEFAULT_GPU_MEM_UTIL)),
        "container_port": str(template_vars.get("container_port", utils.VLLM_DEFAULT_CONTAINER_PORT)),
        "docker_args": template_vars.get("docker_args", utils.VLLM_DEFAULT_DOCKER_ARGS),
        "models_dir": template_vars.get("models_dir", ""),
    })
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

    metadata["mtp_enabled"] = mtp_enabled
    if mtp_enabled:
        metadata["mtp_draft_max"] = mtp_n_max

    # Expose declared sampling modes so clients (hermes, opencode, UIs) can
    # discover the per-request aliases ("<id>:<mode>") without hitting the
    # model list. Static discovery: metadata is passed through in /v1/models.
    mode_params = model.modes
    if mode_params:
        metadata["modes"] = sorted(mode_params)
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
    if metadata:
        entry["metadata"] = metadata

    # Native llama-swap capabilities block (shown in /v1/models).
    entry["capabilities"] = {
        "in": modalities,
        "out": modalities,
        "tools": "tools" in caps,
        "reranker": "reranker" in caps or model.role == "rerank",
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

    return entry_id, entry


def _solve_matrix_context(
    chat_models: list[Model],
    embed_model: Model,
    rerank_model: Model,
    fit_bin: str,
    vram_total: int,
    spare: str | None,
    defaults: dict,
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
    # Resolve global spare
    spare_mb = 0
    if spare:
        spare_mb = utils.parse_mem_mb(str(spare), vram_total)

    drop_stems = drop_stems or set()
    cache_type = str(defaults.get("cache_type", "q8_0"))
    parallel = int(defaults.get("parallel", 1))

    # Get static params for chat models (companion VRAM folded in).
    # The drop decision (mmproj skipped to reach the min useful context) is
    # decided in build_config and threaded in via drop_stems.
    chat_params = []
    for m in chat_models:
        # Embed/rerank models are handled via the separate overhead terms below
        # (and CPU-resident models cost no VRAM), so they must not enter the
        # shared chat budget — otherwise a small resident model can dictate the
        # chat context for the whole fleet.
        if m.role in ("embeddings", "rerank") or m.on_cpu:
            continue
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
        embed_fp = embed_model.vram.fit_params_static(fit_bin, cache_type=cache_type, parallel=parallel)
        embed_params = (embed_fp.model_mib, embed_fp.ctx_factor, embed_fp.compute_mib) if embed_fp else None
    rerank_params = None
    if not rerank_model.on_cpu:
        rerank_fp = rerank_model.vram.fit_params_static(fit_bin, cache_type=cache_type, parallel=parallel)
        rerank_params = (rerank_fp.model_mib, rerank_fp.ctx_factor, rerank_fp.compute_mib) if rerank_fp else None

    # Use conservative context defaults for embed/rerank if not specified
    embed_ctx = 8192  # typical embed context
    rerank_ctx = 8192  # typical rerank context

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


def build_config(
    models: list[Model],
    profiles_cfg: dict,
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

    Args:
        models: List of Model instances
        profiles_cfg: Full profiles.yaml config
        template_vars: Template variables (llama_bin, models_dir)
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
            projection is dropped from the main entry (a ``vision-<N>k`` variant
            is emitted instead, still exposing vision at best-effort context).
    """
    defaults = profiles_cfg.get("defaults", {})
    profile_list = profiles_cfg.get("profiles", {})

    entries: dict[str, dict] = {}

    # ── Pre-pass: decide mmproj drop per chat model ──
    # A chat model keeps mmproj if it can reach the minimum useful context WITH
    # vision; otherwise the main entry drops it and a best-effort vision variant
    # is emitted.  The decision uses the global spare and default cache/parallel;
    # per-profile spare overrides still bound ctx via calc_ctx per group.
    global_spare_mb = 0
    global_spare_str = defaults.get("spare") or spare
    if global_spare_str:
        global_spare_mb = utils.parse_mem_mb(str(global_spare_str), vram_total)
    default_cache_type = str(defaults.get("cache_type", "q8_0"))
    default_parallel = int(defaults.get("parallel", 1))

    # stem -> True when the main entry should omit mmproj
    drop_mmproj: dict[str, bool] = {}
    for model in models:
        if model.role in ("embeddings", "rerank"):
            continue
        if not (model.mmproj and model.mmproj.gguf_path):
            continue
        ctx_with = model.vram.calc_ctx(
            vram_total, fit_bin=fit_bin, parallel=default_parallel,
            spare_mb=global_spare_mb, include_mmproj=True,
            baseline_mb=baseline_mb, cache_type=default_cache_type,
        )
        if max_context is not None:
            ctx_with = min(ctx_with, max_context)
        if ctx_with >= min_context:
            drop_mmproj[model.stem] = False
            logger.info("mmproj: keep for %s (ctx %d >= %d)", model.stem, ctx_with, min_context)
            continue
        ctx_without = model.vram.calc_ctx(
            vram_total, fit_bin=fit_bin, parallel=default_parallel,
            spare_mb=global_spare_mb, include_mmproj=False,
            baseline_mb=baseline_mb, cache_type=default_cache_type,
        )
        if max_context is not None:
            ctx_without = min(ctx_without, max_context)
        drop_mmproj[model.stem] = True
        logger.info("mmproj: drop for %s (vision ctx %d < %d; text ctx %d)",
                    model.stem, ctx_with, min_context, ctx_without)
        if ctx_without < min_context:
            logger.warning("mmproj: %s cannot reach %d context even without vision "
                           "(text ctx %d)", model.stem, min_context, ctx_without)

    # ── Pre-pass: solve matrix context if configured ──
    matrix_chat_ctx: int | None = None
    if matrix_cfg and embed_model and rerank_model:
        matrix_chat_ctx = _solve_matrix_context(
            models, embed_model, rerank_model,
            fit_bin, vram_total, spare, defaults,
            baseline_mb=baseline_mb, drop_stems={s for s, d in drop_mmproj.items() if d},
        )
        if matrix_chat_ctx is not None:
            logger.info("matrix: solved chat_ctx=%d", matrix_chat_ctx)

    for model in models:
        # Use GGUF architectural max as the effective context limit.
        # Falls back to sidecar context_length, then default.
        context_length = model.gguf_context_length or model.context_length

        # Whether this model's main entry keeps or drops mmproj
        include_mmproj = not drop_mmproj.get(model.stem, False)

        # Filter profiles by model's allow_profiles
        filtered_profiles = _filter_profiles(profile_list, model.allow_profiles)

        # Resolve reasoning variants
        reasoning_variants = _resolve_reasoning_cli_args(model.cli_args, model.reasoning)

        for variant_cli_args, variant_suffix in reasoning_variants:
            # Deep-copy frontmatter so each variant has independent cli_args
            variant_fm = copy.deepcopy(model.frontmatter)
            variant_fm["cli_args"] = variant_cli_args

            # Group by (parallel, cache_type, spare_mb)
            groups: dict[tuple, list] = {}
            for pname, pover in filtered_profiles:
                resolved_profile = utils.resolve_params(pover, defaults)
                parallel_val = variant_fm.get("parallel", resolved_profile.get("parallel", 1))
                cache_type_val = str(resolved_profile.get("cache_type", "q8_0"))

                # Resolve spare for this profile
                profile_spare_str = resolved_profile.get("spare") or spare
                if profile_spare_str:
                    profile_spare_mb = utils.parse_mem_mb(str(profile_spare_str), vram_total)
                else:
                    profile_spare_mb = 0

                gkey = (int(parallel_val), cache_type_val, profile_spare_mb)
                groups.setdefault(gkey, []).append((pname, resolved_profile))

            # If no profiles matched, generate a single entry with defaults
            if not groups:
                default_spare_str = defaults.get("spare") or spare
                default_spare_mb = utils.parse_mem_mb(str(default_spare_str), vram_total) if default_spare_str else 0
                groups = {
                    (int(variant_fm.get("parallel", 1)), str(defaults.get("cache_type", "q8_0")), default_spare_mb): [
                        ("default", dict(defaults)),
                    ],
                }

            for (parallel, cache_type, spare_mb), profiles_group in groups.items():
                # Use matrix-solved context as design context if available
                design_ctx = matrix_chat_ctx if matrix_chat_ctx else None

                # Main entry (possibly without mmproj)
                ctx_size = model.vram.calc_ctx(
                    vram_total,
                    fit_bin=fit_bin,
                    parallel=parallel,
                    spare_mb=spare_mb,
                    include_mmproj=include_mmproj,
                    baseline_mb=baseline_mb,
                    cache_type=cache_type,
                    design_ctx=design_ctx,
                )
                ctx_size = min(ctx_size, context_length)
                if max_context is not None:
                    ctx_size = min(ctx_size, max_context)

                entry_id, entry = _build_entry(
                    model, parallel, cache_type, profiles_group,
                    defaults, template_vars, context_length, ctx_size,
                    include_mmproj=include_mmproj,
                )
                if variant_suffix:
                    entry_id = entry_id + variant_suffix
                entries[entry_id] = entry

                # Vision variant: emitted when mmproj was dropped from the main
                # entry. Best-effort context WITH vision; id ``-vision-<N>k``
                # where N = context // 1000 (e.g. 92567 -> vision-92k).
                if not include_mmproj and model.mmproj and model.mmproj.gguf_path:
                    vision_ctx = model.vram.calc_ctx(
                        vram_total,
                        fit_bin=fit_bin,
                        parallel=parallel,
                        spare_mb=spare_mb,
                        include_mmproj=True,
                        baseline_mb=baseline_mb,
                        cache_type=cache_type,
                        design_ctx=design_ctx,
                    )
                    vision_ctx = min(vision_ctx, context_length)
                    if max_context is not None:
                        vision_ctx = min(vision_ctx, max_context)
                    n_k = vision_ctx // 1000
                    vision_id, vision_entry = _build_entry(
                        model, parallel, cache_type, profiles_group,
                        defaults, template_vars, context_length, vision_ctx,
                        include_mmproj=True,
                        name_suffix=f" [vision {n_k}k]",
                    )
                    vision_id = f"{vision_id}-vision-{n_k}k"
                    if variant_suffix:
                        vision_id = vision_id + variant_suffix
                    entries[vision_id] = vision_entry

    config: dict = {}
    config["models"] = {
        eid: entries[eid]
        for eid in sorted(entries, key=lambda e: (e.count("."), e))
    }

    # Present setParamsByID aliases (e.g. "<id>:<mode>") in the /v1/models
    # listing so dynamic-list clients (OpenWebUI, OpenClaw, ...) can select
    # them. Default in llama-swap is false and aliases would be invisible.
    config["includeAliasesInList"] = True
    return config


def write_yaml(config: dict, path: Path | str) -> None:
    """Write config to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)