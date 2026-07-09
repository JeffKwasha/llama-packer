# model_cfg/output/llama_swap.py
"""Generate llama-swap config entries from Model objects."""

from __future__ import annotations

import copy
import logging
import re
from pathlib import Path

import yaml

from model_cfg.model import Model
from model_cfg import utils

logger = logging.getLogger(__name__)

# Profile key that controls spare VRAM (bytes)
_SPARE_KEY = "spare"

# Regex to match context_limit_*G keys in frontmatter
_CL_RE = re.compile(r"^context_limit_\d+G$")


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


def _build_entry(
    model: Model,
    parallel: int,
    cache_type: str,
    profiles_group: list[tuple[str, dict]],
    profiles_defaults: dict,
    template_vars: dict,
    context_length: int,
    ctx_size: int,
) -> tuple[str, dict]:
    """Build a single llama-swap config entry for a model+profile group."""
    base_id = utils.slugify(model.name)

    # Build command string
    target = model.targets[0] if model.targets else "llama-server"
    t_conf = utils._TARGET_TEMPLATES.get(target, utils._TARGET_TEMPLATES["llama-server"])

    parts = []
    parts.append(utils.resolve_template(t_conf["bin"], {"llama_bin": template_vars.get("llama_bin", "")}))
    parts.append("--port ${PORT}")
    parts.append(utils.resolve_template(t_conf["model"], {"model_path": str(model.gguf_path)}))
    if ctx_size:
        parts.append(utils.resolve_template(t_conf["ctx"], {"ctx_size": str(ctx_size)}))
    if parallel > 1:
        parts.append(utils.resolve_template(t_conf["parallel"], {"parallel": str(parallel)}))
    parts.append(utils.resolve_template(t_conf["cache_type"], {"cache_type": cache_type}))

    # MTP args
    mtp_arg_str, mtp_enabled, mtp_n_max, _ = _build_mtp_args(model)
    if mtp_enabled and mtp_arg_str:
        parts.append(mtp_arg_str)

    # mmproj
    if model.mmproj and model.mmproj.gguf_path:
        parts.append(utils.resolve_template(t_conf["mmproj"], {"mmproj_path": str(model.mmproj.gguf_path)}))

    # Extra CLI args
    extra = model.cli_args.strip()
    if extra:
        parts.append(extra)

    cmd_str = " ".join(parts).strip()

    # Profile params → setParamsByID
    set_params: dict[str, dict] = {}
    for pname, resolved_prof in profiles_group:
        overrides = {}
        for k in utils.SAMPLING_KEYS:
            val, dval = resolved_prof.get(k), profiles_defaults.get(k)
            if val is not None and dval is not None and val != dval:
                overrides[k] = round(val, 6) if isinstance(val, float) else val
        if overrides:
            key = "${MODEL_ID}" if pname == "default" else f"${{MODEL_ID}}:{pname}"
            set_params[key] = overrides

    names = [p[0] for p in profiles_group]
    has_default = "default" in names
    entry_id = base_id if (has_default or len(profiles_group) > 1) else f"{base_id}.{names[0]}"

    # Metadata: remaining frontmatter keys not consumed
    metadata = {k: copy.deepcopy(v) for k, v in model.frontmatter.items()
                if k not in Model.FIELDS
                and k not in ("parameters", "quantization", "template", "description")
                and not _CL_RE.match(k)
                and v}
    metadata["mtp_enabled"] = mtp_enabled
    if mtp_enabled:
        metadata["mtp_draft_max"] = mtp_n_max

    entry: dict = {"cmd": cmd_str}
    if set_params:
        entry["setParamsByID"] = set_params
    if model.name:
        entry["name"] = model.name
    if model.description:
        entry["description"] = model.description
    if metadata:
        entry["metadata"] = metadata

    return entry_id, entry


def build_config(
    models: list[Model],
    profiles_cfg: dict,
    template_vars: dict,
    fit_bin: str,
    vram_total: int,
    spare: str | None = None,
    max_context: int | None = None,
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
    """
    defaults = profiles_cfg.get("defaults", {})
    profile_list = profiles_cfg.get("profiles", {})

    entries: dict[str, dict] = {}

    for model in models:
        context_length = model.context_length

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
                    profile_spare_mb = utils.resolve_spare_mb(str(profile_spare_str), vram_total)
                else:
                    profile_spare_mb = 0

                gkey = (int(parallel_val), cache_type_val, profile_spare_mb)
                groups.setdefault(gkey, []).append((pname, resolved_profile))

            # If no profiles matched, generate a single entry with defaults
            if not groups:
                default_spare_str = defaults.get("spare") or spare
                default_spare_mb = utils.resolve_spare_mb(str(default_spare_str), vram_total) if default_spare_str else 0
                groups = {
                    (int(variant_fm.get("parallel", 1)), str(defaults.get("cache_type", "q8_0")), default_spare_mb): [
                        ("default", dict(defaults)),
                    ],
                }

            for (parallel, cache_type, spare_mb), profiles_group in groups.items():
                ctx_size = model.calc_ctx(
                    vram_total,
                    fit_bin=fit_bin,
                    parallel=parallel,
                    spare_mb=spare_mb,
                    cache_type=cache_type,
                )
                ctx_size = min(ctx_size, context_length)
                if max_context is not None:
                    ctx_size = min(ctx_size, max_context)

                entry_id, entry = _build_entry(
                    model, parallel, cache_type, profiles_group,
                    defaults, template_vars, context_length, ctx_size,
                )
                if variant_suffix:
                    entry_id = entry_id + variant_suffix
                entries[entry_id] = entry

    config: dict = {}
    config["models"] = {
        eid: entries[eid]
        for eid in sorted(entries, key=lambda e: (e.count("."), e))
    }
    return config


def write_yaml(config: dict, path: Path | str) -> None:
    """Write config to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def write_ini(config: dict, path: Path | str) -> None:
    """Write config to INI file."""
    try:
        import configparser
    except ImportError:
        raise ImportError("configparser is required for INI output")

    parser = configparser.ConfigParser()
    parser.add_section("models")

    for model_id, model_cfg in config.get("models", {}).items():
        parser.set("models", model_id, model_cfg.get("cmd", ""))

        for key, value in model_cfg.items():
            if key == "cmd":
                continue
            section = f"models.{model_id}"
            if not parser.has_section(section):
                parser.add_section(section)
            if isinstance(value, dict):
                for k, v in value.items():
                    parser.set(section, str(k), str(v))
            else:
                parser.set(section, key, str(value))

    with open(path, "w") as f:
        parser.write(f)