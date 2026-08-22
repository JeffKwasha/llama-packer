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
        """Group the model's allowed profiles by (parallel, cache_type, spare_mb).

        Each group shares one VRAM solve and one llama-swap entry; the profile
        names within a group become ``setParamsByID`` keys.  When no profile
        matches, a single group derived from ``defaults`` is returned.
        """
        groups: dict[tuple, list] = {}
        for pname, resolved in self.matched_for(model):
            parallel = model.parallel_for(resolved.get("parallel", 1))
            cache_type = model.cache_type_for(
                str(resolved.get("cache_type", self.default_cache_type)))
            spare = self.spare_mb(resolved.get("spare"), spare_override, vram_total)
            groups.setdefault((int(parallel), cache_type, spare), []).append((pname, resolved))

        if not groups:
            groups = {
                (model.parallel_for(1), model.cache_type_for(self.default_cache_type),
                 self.global_spare_mb(spare_override, vram_total)): [
                    ("default", dict(self.defaults)),
                ],
            }
        return groups


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
