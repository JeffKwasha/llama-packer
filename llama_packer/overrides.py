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

from llama_packer import utils
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

    Each ref is any file ref: a string (absolute path, or relative to the
    sidecar's own directory — the natural "file next to the model"
    convention — plus ``hub:org/repo:file`` hub refs and snapshot
    filename/globs when the model declares ``hf_repo``) or a mapping
    (``{file:}`` ≡ the bare string; ``{hf_repo:, file:[, pick:]}`` names a
    hub file, ``hf_repo`` defaulting to the model's own). Absolute refs
    pass through. Returns a list of human-readable error strings (empty
    when all resolve); resolved paths are stored on ``model`` attributes
    the backends read.
    """
    errors: list[str] = []
    anchors = [model.md_path.parent]
    repo_default = model.frontmatter.get("hf_repo")

    def _resolve(ref) -> Path | None:
        return model._finder.resolve_path(ref, anchors=anchors,
                                          repo=repo_default,
                                          hf_home=model._hf_home)

    def _describe(ref) -> str:
        return utils.describe_ref(ref, repo_default)

    ct = model.frontmatter.get("chat_template")
    if ct is None or ct is False or ct == "":
        pass  # unset (False/empty also disables an inherited template)
    elif isinstance(ct, (str, dict)):
        hit = _resolve(ct)
        if hit is not None and hit.is_file():
            model._resolved_chat_template = hit.absolute()
        else:
            errors.append(f"chat_template file not found: {_describe(ct)}")
    else:
        errors.append(f"chat_template file not found: {ct!r}")

    loras = model.frontmatter.get("loras") or []
    if isinstance(loras, (str, dict)):
        loras = [loras]
    resolved: list[Path] = []
    if isinstance(loras, list):
        for ref in loras:
            if not isinstance(ref, (str, dict)):
                errors.append(f"lora file not found: {ref!r}")
                continue
            hit = _resolve(ref)
            if hit is not None and hit.is_file():
                resolved.append(hit.absolute())
            else:
                errors.append(f"lora file not found: {_describe(ref)}")
    elif loras:
        errors.append(f"lora file not found: {loras!r}")
    if resolved:
        model._resolved_loras = resolved

    return errors
