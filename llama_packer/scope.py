# llama_packer/scope.py
"""ScopeStack: the single select-and-set engine for sidecar data.

Every way llama-packer sets key/value pairs on a model's frontmatter is one
dumb algorithm applied at one of two precedence levels:

1. **defaults** — directory-scoped ``models.yaml`` mappings, folded
   outermost → innermost; the sidecar's own values win over all of them.
2. **rules** — pattern-scoped override rules (``when`` → settings), from
   ``profiles.yaml`` (global) and per-directory ``models.yaml``; last match
   wins per key, so inner scopes beat outer ones beat global.

A :class:`ScopeStack` is pushed once per scope during discovery's depth-first
walk (global rules first, then each ancestor directory), so both levels share
the same merge code and the same ordering.  After defaults and rules have had
their say, :meth:`ScopeStack.finalize` resolves backend inference and external
file references — the only remaining per-model step before planning.
"""

from __future__ import annotations

import logging

from llama_packer.backends import BACKENDS, infer_backend
from llama_packer.overrides import (
    compile_rule_list,
    resolve_setting_paths,
    rule_matches,
)

logger = logging.getLogger(__name__)


class ScopeStack:
    """A depth-first stack of configuration scopes.

    Push one scope per level of the walk: the global ``overrides`` from
    profiles.yaml at the bottom, then each directory's ``models.yaml`` as
    discovery descends.  Defaults fold outermost → innermost with the sidecar
    on top; rules accumulate in application order so per-key last-match-wins
    naturally gives inner > outer > global.
    """

    def __init__(self, avail: dict | None = None, allowed: list[str] | None = None):
        self._defaults: list[dict] = []          # one dict per open scope
        self._rule_counts: list[int] = []        # rules added by each scope
        self._rules: list[tuple] = []            # flat (when, settings), outer→inner
        # Backend-inference resources and enable/prefer list (profiles.yaml
        # ``backends:``); used by finalize().
        self.avail = avail or {}
        self.allowed = allowed

    # ── scoping ──

    def push(self, cfg: dict | None, origin: str = "scope") -> None:
        """Enter a scope: layer its ``defaults`` and append its ``overrides``."""
        cfg = cfg or {}
        rules = compile_rule_list(cfg.get("overrides") or [], origin)
        self._defaults.append(cfg.get("defaults") or {})
        self._rule_counts.append(len(rules))
        self._rules.extend(rules)

    def pop(self) -> None:
        """Leave the most recent scope."""
        if not self._rule_counts:
            raise RuntimeError("ScopeStack.pop() with no pushed scope")
        n = self._rule_counts.pop()
        if n:
            del self._rules[-n:]
        self._defaults.pop()

    # ── layer 1: defaults ──

    @property
    def defaults(self) -> dict:
        """Folded defaults of every open scope, outermost → innermost."""
        merged: dict = {}
        for d in self._defaults:
            merged.update(d)
        return merged

    def merge_defaults(self, frontmatter: dict) -> dict:
        """Sidecar frontmatter layered over folded scope defaults.

        The sidecar always wins over defaults; defaults win over built-in
        property fallbacks by virtue of being present in the dict at all.
        """
        return {**self.defaults, **frontmatter}

    # ── layer 2: rules ──

    def apply_rules(self, model) -> set[str]:
        """Apply every open scope's override rules to *model* in place.

        Last match wins per key.  Returns the set of frontmatter keys that
        changed, so the caller can invalidate derived state (see
        :data:`COMPANION_KEYS`).
        """
        updates: dict = {}
        for when, settings in self._rules:
            if rule_matches(when, model):
                for k, v in settings.items():
                    prev = model.frontmatter.get(k)
                    if prev is not None and prev != v:
                        logger.debug("override: %s: %s %r -> %r",
                                     model.stem, k, prev, v)
                    updates[k] = v
        model.frontmatter.update(updates)
        return set(updates)

    # ── post-merge finalization ──

    def finalize(self, model) -> None:
        """Backend selection/inference + external file resolution for *model*.

        Runs after defaults and rules are fully applied, so inference sees the
        final ``backend``/``hf_repo`` and path references resolve against the
        final values.  Models with unresolved errors are flagged with
        ``model._override_error`` (already logged) for the writer to skip.
        """
        errors: list[str] = []
        backend_name = model.frontmatter.get("backend")
        if backend_name is not None and str(backend_name) not in BACKENDS:
            errors.append(f"unknown backend {backend_name!r}")
        elif backend_name is not None and self.allowed is not None \
                and str(backend_name) not in self.allowed:
            errors.append(f"backend {backend_name!r} is disabled by "
                          f"profiles.yaml backends: {self.allowed}")
        elif backend_name is None:
            inferred = infer_backend(model, self.avail, allowed=self.allowed)
            if inferred is not None:
                logger.debug("override: %s: inferred backend %r",
                             model.stem, inferred)
                model.frontmatter["backend"] = inferred
            else:
                fmt = (model.gguf_path.suffix.lower() if model.gguf_path
                       else "hf_repo" if model.hf_repo else "none")
                errors.append(f"no available backend supports format {fmt!r}")

        errors += resolve_setting_paths(model)

        if errors:
            for e in errors:
                logger.error("override: skipping %s: %s", model.stem, e)
            model._override_error = errors[0]
