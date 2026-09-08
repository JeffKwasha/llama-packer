# llama_packer/profiles.py
"""profiles.yaml as a value object.

Single home for reading the input config: baseline defaults, sampling-key
expression resolution, ``allow_profiles`` filtering, per-model variant
grouping, and spare-VRAM parsing.  Nothing else should reach into the raw
``defaults`` / ``profiles`` dicts — go through this class so precedence rules
(sidecar > profile > built-in) exist in exactly one place.
"""

from __future__ import annotations

import logging
import re

from llama_packer import utils

logger = logging.getLogger(__name__)


def parse_spare_mb(spare_str: str | None, vram_total: int) -> int:
    """Parse a spare string ("4G", "512m", bare MB); empty/None -> 0."""
    return utils.parse_mem_mb(str(spare_str), vram_total) if spare_str else 0


class Profiles:
    """Typed view over the profiles.yaml mapping."""

    def __init__(self, cfg: dict | None):
        self._cfg = cfg or {}
        self.defaults: dict = self._cfg.get("defaults", {}) or {}
        self.profile_list: dict = self._cfg.get("profiles", {}) or {}

    # ── fleet-wide defaults ──

    @property
    def default_cache_type(self) -> str:
        return str(self.defaults.get("cache_type", "q8_0"))

    @property
    def default_parallel(self) -> int:
        return int(self.defaults.get("parallel", 1))

    def global_spare_mb(self, cli_override: str | None = None,
                        vram_total: int = 0) -> int:
        """Fleet-wide spare VRAM in MB: ``defaults.spare`` > ``--spare`` > 0."""
        return parse_spare_mb(self.defaults.get("spare") or cli_override, vram_total)

    def spare_mb(self, preferred: str | None = None, cli_override: str | None = None,
                 vram_total: int = 0) -> int:
        """Spare VRAM in MB for one resolution: *preferred* (profile value)
        > *cli_override* (global ``--spare``) > 0."""
        return parse_spare_mb(preferred or cli_override, vram_total)

    @property
    def pools_cfg(self) -> dict:
        """Raw ``pools:`` section: ``{pool_id: {vram, spare, reserve_extra,
        pins}}`` with non-dict garbage dropped (warned, never fatal).

        Pools are VRAM ledgers keyed by id (``gpu0`` … or ``default``).
        ``vram:`` declares a pool's size (otherwise the detected total);
        ``spare:`` overrides the global spare for that pool; ``reserve_extra:``
        holds unmodelled residents (VM, game, other tenant); ``pins:`` maps a
        claimant name to an explicit reservation or ``"auto"`` (derived math
        stays authoritative).  Resolution against pool sizes happens in the
        ledger (:class:`writer.PoolLedger`), not here.
        """
        raw = self._cfg.get("pools")
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            logger.warning("pools: section is not a mapping; ignoring")
            return {}
        pools = {}
        for pid, spec in raw.items():
            if not isinstance(spec, dict):
                logger.warning("pools: %r is not a mapping; ignoring", pid)
                continue
            pins = spec.get("pins") or {}
            if not isinstance(pins, dict):
                logger.warning("pools: %r pins is not a mapping; ignoring",
                               pid)
                pins = {}
            pools[str(pid)] = {
                "vram": spec.get("vram"),
                "spare": spec.get("spare"),
                "reserve_extra": spec.get("reserve_extra"),
                "pins": {str(k): v for k, v in pins.items()},
            }
        return pools

    @property
    def llama_server_cfg(self) -> dict:
        """Raw ``llama_server:`` section (``args``, ``batch``, ``ubatch``,
        ``env``).  ``batch``/``ubatch`` are the fleet tier of the named
        batch-key cascade (sidecar > profile > fleet > role)."""
        raw = self._cfg.get("llama_server")
        return dict(raw) if isinstance(raw, dict) else {}

    # ── per-model selection ──

    def matched_for(self, model) -> list[tuple[str, dict]]:
        """(name, resolved-profile) pairs this model allows.

        Honors the model's ``allow_profiles`` regex/list/false gate; each
        profile is layered over ``defaults`` with ``base * N`` expressions
        resolved.
        """
        return [
            (pname, utils.resolve_params(pover, self.defaults))
            for pname, pover in _filter_profiles(self.profile_list, model.allow_profiles)
        ]

    def groups_for(self, model, vram_total: int,
                   spare_override: str | None = None) -> dict[tuple, list[tuple[str, dict]]]:
        """Group the model's allowed profiles by
        (parallel, cache_type, spare_mb, batch, ubatch).

        Each group shares one VRAM solve and one llama-swap entry; the profile
        names within a group become ``setParamsByID`` keys.  ``batch``/``ubatch``
        are part of the key so per-profile values cannot collide inside one
        entry's command (they render per entry and stamp the measurement
        shape).  When no profile matches, a single group derived from
        ``defaults`` is returned.

        Embed/rerank always serve single-slot: their parallel is pinned to 1
        (declared values are ignored with a note) — a resident's parallelism
        must never buy context away from the main chat it serves.
        """
        try:
            from llama_packer.backends import get_backend
            role_defaults = get_backend(model.backend).default_batch_ubatch(model.role)
        except Exception:
            role_defaults = (2048, 512)
        rag_role = model.role in ("embeddings", "rerank")
        groups: dict[tuple, list] = {}
        for pname, resolved in self.matched_for(model):
            parallel = model.parallel_for(resolved.get("parallel", 1))
            if rag_role:
                parallel = _rag_parallel(model, parallel)
            cache_type = model.cache_type_for(
                str(resolved.get("cache_type", self.default_cache_type)))
            spare = self.spare_mb(resolved.get("spare"), spare_override, vram_total)
            batch, ubatch = model.batch_ubatch_for(
                resolved, self.llama_server_cfg, role_defaults)
            groups.setdefault(
                (int(parallel), cache_type, spare, batch, ubatch), []
            ).append((pname, resolved))

        if not groups:
            batch, ubatch = model.batch_ubatch_for(
                None, self.llama_server_cfg, role_defaults)
            groups = {
                (1 if rag_role else model.parallel_for(1),
                 model.cache_type_for(self.default_cache_type),
                 self.global_spare_mb(spare_override, vram_total), batch, ubatch): [
                    ("default", dict(self.defaults)),
                ],
            }
        return groups


# Embed/rerank serve single-slot: their parallel is a chat knob, never a
# resident one — a note once per model when a declaration is ignored.
_rag_parallel_warned: set[str] = set()


def _rag_parallel(model, declared: int) -> int:
    """Pin an embeddings/rerank model to parallel 1.

    Resident parallelism must never buy context away from the main chat it
    serves ("main chat slightly better" beats "embed/rerank parallel > 1").
    A declared parallel > 1 is ignored with a once-per-model note.
    """
    if declared == 1:
        return 1
    if model.stem not in _rag_parallel_warned:
        _rag_parallel_warned.add(model.stem)
        logger.warning(
            "%s: embed/rerank serve single-slot — parallel %d ignored "
            "(resident parallelism must not shrink the main chat context "
            "it serves)", model.stem, declared)
    return 1


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
            logger.warning("invalid allow_profiles regex %r, returning all profiles",
                           allow_profiles)
            return list(profile_list.items())
        return [(pname, pover) for pname, pover in profile_list.items()
                if pattern.search(pname)]
    return list(profile_list.items())
