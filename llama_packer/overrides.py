# llama_packer/overrides.py
"""Pattern-scoped settings applied to discovered models.

Override rules live under ``overrides:`` in profiles.yaml (global) and in
``models.yaml`` files inside the models tree (directory-scoped — they apply
only to models beneath their directory, and a closer directory beats an
outer one beats global).  Each rule matches a model by one or more field→regex
pairs (``when``) and sets settings keys (last match wins per key).  This is
the single mechanism for selecting a backend, a chat template, LoRA adapters,
``hf_repo`` and ``cli_args``.

``when`` is required — a rule without one stops the run (the intended
configuration would otherwise be silently ignored).  Use the regexes against
frontmatter fields plus the synthetic ``stem``/``name``, or ``when: true`` to
match every model.

Models that end up with no ``backend`` after sidecar + rules get one inferred
from their file format (see ``backends.infer_backend``): GGUF → llama-server,
safetensors / HF-repo → vLLM docker, gated by each backend's configured
resources.  Models whose format no available backend covers are logged and
skipped.

Regex literals are easiest in YAML single-quoted or unquoted scalars — only
double quotes interpret backslashes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from llama_packer import utils
from llama_packer.backends import BACKENDS, SETTING_KEYS, infer_backend

logger = logging.getLogger(__name__)


def _field_value(model, field: str) -> str | None:
    if field == "stem":
        return model.stem
    if field == "name":
        return model.frontmatter.get("name")
    val = model.frontmatter.get(field)
    if val is None:
        return None
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def rule_matches(when, model) -> bool:
    """True if *when* matches *model* (``True`` matches everything)."""
    if when is True:
        return True
    for field, pattern in when.items():
        value = _field_value(model, str(field))
        try:
            hit = value is not None and re.search(str(pattern), value) is not None
        except re.error as e:
            logger.warning("override: invalid regex %r for %r: %s",
                           pattern, field, e)
            return False
        if not hit:
            return False
    return True


def _compile_rule_list(raw_rules, origin: str) -> list[tuple]:
    """Validate and compile override *raw_rules*; stop the run on malformed ones."""
    compiled: list[tuple] = []
    for i, rule in enumerate(raw_rules):
        if not isinstance(rule, dict):
            raise SystemExit(
                f"error: {origin} overrides[{i}]: must be a mapping "
                f"(got {type(rule).__name__}) — fix the rule")
        if "when" not in rule or rule["when"] is None or rule["when"] == {} or rule["when"] == "":
            raise SystemExit(
                f"error: {origin} overrides[{i}]: missing 'when' — every "
                f"rule must declare what it matches (use 'when: true' for all "
                f"models)")
        when = rule["when"]
        if when is not True and not isinstance(when, dict):
            raise SystemExit(
                f"error: {origin} overrides[{i}]: 'when' must be a mapping "
                f"of field→regex pairs, or true")
        settings = {k: v for k, v in rule.items() if k != "when"}
        unknown = set(settings) - SETTING_KEYS
        for k in unknown:
            logger.warning("%s overrides[%d]: unknown setting %r (ignored)", origin, i, k)
        settings = {k: v for k, v in settings.items() if k in SETTING_KEYS}
        if not settings:
            raise SystemExit(
                f"error: {origin} overrides[{i}]: no known settings — "
                f"valid keys: {', '.join(sorted(SETTING_KEYS))}")
        compiled.append((when, settings))
    return compiled


def _compile_rules(profiles_cfg) -> list[tuple]:
    """Compile global ``overrides`` from profiles.yaml."""
    return _compile_rule_list((profiles_cfg or {}).get("overrides") or [], "profiles.yaml")


def compile_scoped_rules(dir_cfgs: dict) -> list[tuple]:
    """Compile directory-scoped ``models.yaml`` rules.

    *dir_cfgs* maps a scope directory to its parsed ``models.yaml``; returns
    ``(scope_dir, when, settings)`` tuples for :func:`apply_overrides`.
    """
    out: list[tuple] = []
    for scope, cfg in (dir_cfgs or {}).items():
        origin = str(Path(scope) / utils.DIR_CONFIG_NAME)
        for when, settings in _compile_rule_list(cfg.get("overrides") or [], origin):
            out.append((Path(scope).resolve(), when, settings))
    return out


def resolve_setting_paths(model) -> list[str]:
    """Resolve chat_template / loras refs to absolute files.

    Paths are resolved relative to the sidecar's own directory (the natural
    "file next to the model" convention).  Absolute refs pass through.  Returns
    a list of human-readable error strings (empty when all resolve); resolved
    paths are stored on ``model`` attributes the backends read.
    """
    errors: list[str] = []
    base = model.md_path.parent

    ct = model.frontmatter.get("chat_template")
    if isinstance(ct, str) and ct:
        p = Path(ct) if Path(ct).is_absolute() else base / ct
        if p.is_file():
            model._resolved_chat_template = p.absolute()
        else:
            errors.append(f"chat_template file not found: {ct}")

    loras = model.frontmatter.get("loras") or []
    if isinstance(loras, str):
        loras = [loras]
    resolved: list[Path] = []
    for ref in loras:
        ref_path = Path(str(ref))
        p = ref_path if ref_path.is_absolute() else base / ref_path
        if p.is_file():
            resolved.append(p.absolute())
        else:
            errors.append(f"lora file not found: {ref}")
    if resolved:
        model._resolved_loras = resolved

    return errors


def apply_overrides(models, profiles_cfg, avail: dict | None = None,
                    scoped_rules: list | None = None) -> None:
    """Apply profile override rules to *models* in place.

    Seeds each model from its own sidecar settings, layers matching rules
    (last match wins per key), infers a backend from the file format when none
    was declared (``avail`` describes the configured resources) and resolves
    external file references.  Models with unresolved errors are flagged with
    ``model._override_error`` (already logged) and skipped by the writer.

    *scoped_rules* carries directory-scoped rules from ``models.yaml`` files
    as ``(scope_dir, when, settings)`` tuples; a rule applies only to models
    inside *scope_dir*.  Application order is global rules first, then scoped
    outermost → innermost, so the closest scope wins per key.
    """
    compiled = _compile_rules(profiles_cfg)
    scoped = sorted(scoped_rules or [],
                    key=lambda sr: len(sr[0].parts))  # outermost first
    # profiles.yaml `backends:` — ordered enable/prefer list; absent = all.
    allowed = (profiles_cfg or {}).get("backends") or None
    if allowed is not None:
        allowed = [str(b) for b in allowed]

    for model in models:
        merged = {k: model.frontmatter[k] for k in SETTING_KEYS
                  if k in model.frontmatter}
        for when, settings in compiled:
            if rule_matches(when, model):
                for k, v in settings.items():
                    prev = merged.get(k)
                    if prev is not None and prev != v:
                        logger.debug("override: %s: %s %r -> %r",
                                     model.stem, k, prev, v)
                    merged[k] = v
        parent = model.md_path.parent.resolve()
        for scope, when, settings in scoped:
            if parent != scope and scope not in parent.parents:
                continue  # model outside this models.yaml's subtree
            if rule_matches(when, model):
                for k, v in settings.items():
                    prev = merged.get(k)
                    if prev is not None and prev != v:
                        logger.debug("override: %s: %s %r -> %r (%s)",
                                     model.stem, k, prev, v, scope)
                    merged[k] = v

        errors: list[str] = []
        backend_name = merged.get("backend")
        if backend_name is not None and backend_name not in BACKENDS:
            errors.append(f"unknown backend {backend_name!r}")
        elif backend_name is not None and allowed is not None and backend_name not in allowed:
            errors.append(f"backend {backend_name!r} is disabled by "
                          f"profiles.yaml backends: {allowed}")
        elif backend_name is None:
            # Nothing declared (sidecar or rule): infer from the file format,
            # gated by which backends are enabled (profiles.yaml backends:)
            # and whose resources are actually configured.
            inferred = infer_backend(model, avail, allowed=allowed)
            if inferred is not None:
                logger.debug("override: %s: inferred backend %r", model.stem, inferred)
                merged["backend"] = inferred
            else:
                fmt = (model.gguf_path.suffix.lower() if model.gguf_path
                       else "hf_repo" if model.hf_repo else "none")
                errors.append(f"no available backend supports format {fmt!r}")

        # Commit merged settings to frontmatter *before* resolving external file
        # references, so chat_template/loras values are present for resolution.
        if merged:
            model.frontmatter.update(merged)

        errors += resolve_setting_paths(model)

        if errors:
            for e in errors:
                logger.error("override: skipping %s: %s", model.stem, e)
            model._override_error = errors[0]
            continue
