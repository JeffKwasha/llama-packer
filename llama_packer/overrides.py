# llama_packer/overrides.py
"""Pattern-scoped override rules: matching, compilation, path resolution.

A *rule* matches a model by one or more field→regex pairs (``when``) and sets
settings keys.  Rules come from ``profiles.yaml`` (global) and ``models.yaml``
files inside the models tree; :class:`llama_packer.scope.ScopeStack` applies
them during discovery's walk (last match wins per key).  This module owns the
rule primitives only — selection/merging lives in the scope stack.

``when`` is required — a rule without one stops the run (the intended
configuration would otherwise be silently ignored).  Use the regexes against
frontmatter fields plus the synthetic ``stem``/``name``, or ``when: true`` to
match every model.

Regex literals are easiest in YAML single-quoted or unquoted scalars — only
double quotes interpret backslashes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from llama_packer.backends import SETTING_KEYS

logger = logging.getLogger(__name__)

# Settings keys an override *rule* may set: everything a sidecar/backend can
# declare plus the serving/companion choices that are pure frontmatter data.
# (Deliberately separate from backends.SETTING_KEYS, which drives the
# backends' own "unhandled setting" warnings.)
OVERRIDE_KEYS = frozenset({
    *SETTING_KEYS,
    "cache_type", "parallel", "mmproj", "speculative",
})

# Frontmatter keys a rule may not set (identity/skip semantics stay per-model):
# same forbidden set as directory defaults.
FORBIDDEN_RULE_KEYS = frozenset({"name", "model", "ignore"})

# Frontmatter keys whose change invalidates resolved companion models —
# ScopeStack.apply_rules callers re-run Model.resolve_companions() when any of
# these were touched by a rule.
COMPANION_KEYS = frozenset({"mmproj", "speculative", "mtp", "hf_repo"})


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


def compile_rule_list(raw_rules, origin: str) -> list[tuple]:
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
        unknown = set(settings) - OVERRIDE_KEYS
        for k in unknown:
            logger.warning("%s overrides[%d]: unknown setting %r (ignored)", origin, i, k)
        settings = {k: v for k, v in settings.items() if k in OVERRIDE_KEYS}
        bad = [k for k in settings if k in FORBIDDEN_RULE_KEYS]
        if bad:
            raise SystemExit(
                f"error: {origin} overrides[{i}]: may not set {', '.join(sorted(bad))} "
                f"(per-model keys)")
        if not settings:
            raise SystemExit(
                f"error: {origin} overrides[{i}]: no known settings — "
                f"valid keys: {', '.join(sorted(OVERRIDE_KEYS))}")
        compiled.append((when, settings))
    return compiled


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
