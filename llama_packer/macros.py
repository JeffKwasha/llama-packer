# llama_packer/macros.py
"""User-derived command-line macros for llama-swap.

Each macro captures a *named* flag chunk derived from a user-visible
configuration source (profiles.yaml defaults/profiles, per-directory
models.yaml overrides, explicit macros) so that the name itself
documents provenance (``PROFILE_DEFAULTS``, ``MODELS_CHAT_QWEN3``, …).

The class :class:`Macro` owns the global registry (``Macro[name]``), while
:class:`Macros` is the builder that self-registers :class:`Macro` objects
from the various sources.  ``Macro.apply(cmd)`` then opportunistically
rewrites any emitted ``cmd`` string longest-macro-first, preserving
semantics (exact flag/value equality).
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Callable

import yaml

from llama_packer import utils

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_packer.profiles import Profiles

logger = logging.getLogger(__name__)


def _render_flags(flags: dict[str, str]) -> str:
    """Render an ordered flag→value map to a shell token string."""
    parts: list[str] = []
    for k, v in flags.items():
        parts.append(k)
        if v != "":
            parts.append(v)
    return " ".join(parts)


def _flags_from_cache_parallel(settings: dict | None) -> dict[str, str]:
    """Extract cache/parallel flags from a settings mapping."""
    out: dict[str, str] = {}
    if not isinstance(settings, dict):
        return out
    if "cache_type" in settings and settings["cache_type"] is not None:
        ct = str(settings["cache_type"])
        out["--cache-type-k"] = ct
        out["--cache-type-v"] = ct
    if "parallel" in settings and settings["parallel"] is not None:
        try:
            out["--parallel"] = str(int(settings["parallel"]))
        except (TypeError, ValueError):
            out["--parallel"] = str(settings["parallel"])
    return out


def _flags_for_settings(
    settings: dict, base_dir: Path, sub: Callable[[str], str] | None
) -> dict[str, str]:
    """Map builder settings to flag chunks.

    Handles ``cache_type``, ``parallel``, ``chat_template``,
    ``loras``, ``reasoning-format``, ``reasoning-preserve``, ``cli_args``.
    Paths are resolved against *base_dir* and then passed through *sub*
    (placeholder substitution) when provided.
    """
    flags: dict[str, str] = {}
    flags.update(_flags_from_cache_parallel(settings))

    if "chat_template" in settings and settings["chat_template"]:
        raw = str(settings["chat_template"])
        p = Path(raw)
        if not p.is_absolute():
            p = base_dir / raw if base_dir else p
        # Use smart_resolve to preserve symlinks on same mount (mirrors model layer)
        try:
            rp = utils.smart_resolve(p)
        except Exception:
            rp = p.resolve()
        val = sub(str(rp)) if sub else str(rp)
        # presence of a chat template implies --jinja as well
        flags["--jinja"] = ""
        flags["--chat-template-file"] = val

    if "loras" in settings and settings["loras"]:
        loras = settings["loras"]
        if isinstance(loras, str):
            loras = [loras]
        resolved = []
        for lo in loras:
            lp = Path(str(lo))
            if not lp.is_absolute():
                lp = base_dir / str(lo) if base_dir else lp
            try:
                rp = utils.smart_resolve(lp)
            except Exception:
                rp = lp.resolve()
            resolved.append(sub(str(rp)) if sub else str(rp))
        flags["--lora"] = ",".join(resolved)

    if "reasoning-format" in settings and settings["reasoning-format"]:
        flags["--reasoning-format"] = str(settings["reasoning-format"])

    if settings.get("reasoning-preserve"):
        flags["--reasoning-preserve"] = ""

    if "cli_args" in settings and settings["cli_args"]:
        cli = str(settings["cli_args"]).strip()
        if cli:
            try:
                cli_flags = utils._pair_flags(shlex.split(cli))
            except ValueError as e:
                logger.warning("macros: failed to parse cli_args %r: %s", cli, e)
                cli_flags = {}
            # cli_args last-write-wins: merge on top
            flags.update(cli_flags)

    return flags


class Macro:
    """A single named flag macro.

    ``Macro[name]`` returns the registered instance (via ``__class_getitem__``).
    The global registry lives on this class.
    """

    _registry: dict[str, "Macro"] = {}

    def __init__(self, name: str, source: str, flags: dict[str, str]) -> None:
        self.name = name
        self.source = source
        # preserve order
        self.flags: dict[str, str] = dict(flags)
        self.rendered: str = _render_flags(self.flags)
        self.weight: int = len(self.flags)
        if name in Macro._registry:
            logger.warning(
                "macros: collision for %r (%s replaces %s)",
                name,
                source,
                Macro._registry[name].source,
            )
        Macro._registry[name] = self

    # --- registry access -------------------------------------------------

    def __class_getitem__(cls, name: str) -> "Macro":
        return cls._registry[name]

    @classmethod
    def get(cls, name: str) -> "Macro | None":
        return cls._registry.get(name)

    @classmethod
    def all(cls) -> list["Macro"]:
        """All registered macros sorted longest-first (weight desc, then name)."""
        return sorted(cls._registry.values(), key=lambda m: (-m.weight, m.name))

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def definitions(cls) -> dict[str, str]:
        """{name: rendered} for inclusion in config ``macros:`` block."""
        return {m.name: m.rendered for m in cls._registry.values()}

    # --- application -----------------------------------------------------

    @classmethod
    def apply(cls, cmd: str) -> str:
        """Return *cmd* with longest matching macro flag subsets replaced.

        Parses *cmd* into ``(head, flag_dict)`` via :func:`utils._pair_flags`,
        checks each registered macro (longest first) for exact subset equality,
        removes matched flags and emits ``${NAME}`` refs in place of the first
        occurrence of each macro's flags (preserving original flag order).
        Idempotent.
        """
        if not cmd or not cls._registry:
            return cmd
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            # Fallback: don't try to be clever on unparsable commands
            logger.warning("macros: cannot parse cmd for reduction: %r", cmd)
            return cmd
        if not tokens:
            return cmd
        head = [tokens[0]]
        flag_tokens = tokens[1:]
        try:
            entry_flags = utils._pair_flags(flag_tokens)
        except Exception:
            return cmd

        # Greedy longest-first subset match
        remaining = dict(entry_flags)  # ordered copy
        matched: list[str] = []
        matched_macros: list[Macro] = []
        for macro in cls.all():
            if not macro.flags:
                continue
            # need exact subset: every k/v in macro.flags present with same value
            if all(k in remaining and remaining[k] == v for k, v in macro.flags.items()):
                matched.append(macro.name)
                matched_macros.append(macro)
                for k in macro.flags:
                    remaining.pop(k, None)

        if not matched:
            return cmd

        # Ordered emission: walk original entry_flags order, emitting a macro
        # ref at the first occurrence of any of its flags, then skipping the
        # rest of that macro's flags.
        owner_map: dict[str, str] = {}
        for name, macro in zip(matched, matched_macros):
            for k in macro.flags:
                # first macro to claim a key wins (already ensured no overlap)
                if k not in owner_map:
                    owner_map[k] = name

        emitted: set[str] = set()
        out = list(head)
        for k, v in entry_flags.items():
            if k in owner_map:
                mname = owner_map[k]
                if mname not in emitted:
                    out.append(f"${{{mname}}}")
                    emitted.add(mname)
                # skip the flag itself (covered by macro)
                continue
            out.append(k)
            if v != "":
                out.append(v)
        return " ".join(out)

    @classmethod
    def apply_to_dict(cls, flags: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        """Dict-level variant (for testing / writer integration).

        Returns ``(remaining_flags, matched_names)``.
        """
        remaining = dict(flags)
        matched: list[str] = []
        for macro in cls.all():
            if all(k in remaining and remaining[k] == v for k, v in macro.flags.items()):
                matched.append(macro.name)
                for k in macro.flags:
                    remaining.pop(k, None)
        return remaining, matched


class Macros:
    """Builder that self-registers :class:`Macro` objects from user config.

    Sources (all optional):
      * ``profiles.yaml:defaults``
      * each ``profiles.yaml:profiles.<name>``
      * explicit ``profiles.yaml:macros`` (string values)
      * per-directory ``models.yaml`` ``defaults``/``overrides``
      * builtin ``COMMON_GPU``

    *sub* is the placeholder substitution callable (``utils.make_subst``)
    used to turn absolute template/lora paths into ``${MODELS_DIR}`` form
    so that macros match post-env commands (placeholder domain).
    """

    def __init__(
        self,
        profiles_cfg: dict | None,
        profiles: "Profiles | None" = None,
        models_dirs: list[Path | str] | None = None,
        sub: Callable[[str], str] | None = None,
    ) -> None:
        self.profiles_cfg = profiles_cfg or {}
        self.profiles = profiles
        self.models_dirs = [Path(d) for d in (models_dirs or [])]
        self.sub = sub

        # Start clean for this build
        # Note: caller may have cleared already; we clear to ensure no stale state
        # but do not clear if this is a secondary builder — the last builder wins
        # which matches the "new replaces old" collision policy.
        self._build()

    def _build(self) -> None:
        # 1. PROFILE_DEFAULTS
        if self.profiles is not None:
            dflags = _flags_from_cache_parallel(self.profiles.defaults)
            if dflags:
                Macro("PROFILE_DEFAULTS", "profiles.yaml:defaults", dflags)

            # 2. Each profile's own flags (delta or full — if it defines cache/parallel)
            for pname, pover in (self.profiles.profile_list or {}).items():
                pflags = _flags_from_cache_parallel(pover)
                if pflags:
                    mname = f"PROFILE_{pname.upper().replace('-', '_')}"
                    Macro(mname, f"profiles.yaml:profiles.{pname}", pflags)

        # 3. Explicit macros from profiles.yaml:macros
        explicit = self.profiles_cfg.get("macros") if isinstance(self.profiles_cfg.get("macros"), dict) else None
        if isinstance(explicit, dict):
            for mname, mval in explicit.items():
                if not isinstance(mname, str):
                    logger.warning("macros: invalid macro name %r (ignored)", mname)
                    continue
                if not isinstance(mval, str):
                    logger.warning(
                        "macros: macro %r value must be a string (got %T), ignored",
                        mname,
                        type(mval),
                    )
                    continue
                try:
                    flags = utils._pair_flags(shlex.split(mval))
                except ValueError as e:
                    logger.warning("macros: failed to parse %r=%r: %s", mname, mval, e)
                    continue
                if flags:
                    Macro(mname, "profiles.yaml:macros", flags)

        # 4. Per-directory models.yaml
        for models_dir in self.models_dirs:
            if not models_dir.is_dir():
                continue
            for cfg_path in models_dir.rglob("models.yaml"):
                try:
                    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                except Exception as e:
                    logger.warning("macros: failed to read %s: %s", cfg_path, e)
                    continue
                if not isinstance(data, dict):
                    continue
                base_dir = cfg_path.parent
                # Relative name for macro
                try:
                    rel = base_dir.relative_to(models_dir)
                    if rel == Path("."):
                        rel_name = base_dir.name.upper()
                    else:
                        rel_name = "_".join(p.upper().replace("-", "_") for p in rel.parts)
                    mname = f"MODELS_{rel_name}"
                except ValueError:
                    mname = f"MODELS_{base_dir.name.upper().replace('-', '_')}"

                # defaults
                defaults = data.get("defaults") or {}
                agg_flags: dict[str, str] = {}
                if isinstance(defaults, dict):
                    agg_flags.update(_flags_for_settings(defaults, base_dir, self.sub))

                # overrides — merge all settings (last wins) to capture common flags
                overrides = data.get("overrides") or []
                merged_settings: dict = {}
                if isinstance(overrides, list):
                    for rule in overrides:
                        if not isinstance(rule, dict):
                            continue
                        # settings are all keys except "when"
                        for k, v in rule.items():
                            if k == "when":
                                continue
                            merged_settings[k] = v
                if isinstance(merged_settings, dict) and merged_settings:
                    agg_flags.update(_flags_for_settings(merged_settings, base_dir, self.sub))

                if agg_flags:
                    Macro(mname, f"{cfg_path}:defaults+overrides", agg_flags)

        # 5. Builtin common GPU flag
        Macro("COMMON_GPU", "builtin:gpu", {"--n-gpu-layers": "999"})
